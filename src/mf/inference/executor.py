"""Exact incremental execution for chunk-causal text blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from mf.contracts.geometry import GeometryContract
from mf.modeling.attention import SharedQKVOAttention
from mf.modeling.block import MFBlock
from mf.modeling.chunk_causal_layout import (
    build_image_chunk_positions,
    build_text_chunk_positions,
)
from mf.modeling.mrope import build_vision_segment_mrope_positions


@dataclass(slots=True)
class LayerKVCache:
    key: Tensor
    value: Tensor

    def validate(self) -> LayerKVCache:
        if self.key.ndim != 4 or self.value.shape != self.key.shape:
            raise ValueError(
                "layer key/value cache must have matching [B, S, H, D] shapes"
            )
        if self.key.device != self.value.device or self.key.dtype != self.value.dtype:
            raise ValueError("layer key/value cache must share device and dtype")
        return self


class BlockCache:
    """Preallocated per-layer K/V with an explicitly committed prefix."""

    def __init__(self, layers: tuple[LayerKVCache, ...], cache_seqlens: Tensor) -> None:
        if not layers:
            raise ValueError("BlockCache requires at least one layer")
        self.layers = tuple(layer.validate() for layer in layers)
        first = self.layers[0].key
        if (
            cache_seqlens.shape != (first.shape[0],)
            or cache_seqlens.dtype is not torch.int32
        ):
            raise ValueError("cache_seqlens must be int32 with shape [B]")
        if cache_seqlens.device != first.device:
            raise ValueError("cache_seqlens and cache tensors must share a device")
        if any(layer.key.shape != first.shape for layer in self.layers):
            raise ValueError("all cache layers must have the same shape")
        self.cache_seqlens = cache_seqlens
        self._max_committed_len = int(cache_seqlens.max().item())
        self._cache_revision = 0
        self._row_group_cache: dict[
            tuple[int, tuple[int, ...]], tuple[tuple[Tensor, int], ...]
        ] = {}
        if self._max_committed_len > first.shape[1]:
            raise ValueError("initial cache length exceeds max_cache_len")

    @classmethod
    def allocate(
        cls,
        *,
        num_layers: int,
        batch_size: int,
        max_cache_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BlockCache:
        for name, value in (
            ("num_layers", num_layers),
            ("batch_size", batch_size),
            ("max_cache_len", max_cache_len),
            ("num_heads", num_heads),
            ("head_dim", head_dim),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        shape = (batch_size, max_cache_len, num_heads, head_dim)
        layers = tuple(
            LayerKVCache(
                key=torch.empty(shape, device=device, dtype=dtype),
                value=torch.empty(shape, device=device, dtype=dtype),
            )
            for _ in range(num_layers)
        )
        return cls(
            layers,
            torch.zeros(batch_size, device=device, dtype=torch.int32),
        )

    @property
    def max_cache_len(self) -> int:
        return self.layers[0].key.shape[1]

    @property
    def batch_size(self) -> int:
        return self.layers[0].key.shape[0]

    @property
    def cache_position(self) -> Tensor:
        return self.cache_seqlens

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        layer = self.layers[layer_idx]
        return layer.key, layer.value

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not 0 <= layer_idx < len(self.layers):
            raise IndexError("layer_idx is outside the cache")
        return self._max_committed_len

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        if not 0 <= layer_idx < len(self.layers):
            raise IndexError("layer_idx is outside the cache")
        return self.max_cache_len

    def reset(self) -> None:
        self.cache_seqlens.zero_()
        self._max_committed_len = 0
        self._cache_revision += 1
        self._row_group_cache.clear()

    def row_groups(self, append_count: int | Tensor) -> tuple[tuple[Tensor, int], ...]:
        """Group batch rows by the valid cache length after an append."""

        if isinstance(append_count, Tensor):
            if append_count.shape != self.cache_seqlens.shape:
                raise ValueError("per-row append_count must have shape [B]")
            append_counts = tuple(int(value) for value in append_count.tolist())
            if any(value <= 0 for value in append_counts):
                raise ValueError("per-row append_count values must be positive")
        else:
            if type(append_count) is not int or append_count <= 0:
                raise ValueError("append_count must be a positive integer")
            append_counts = (append_count,) * self.batch_size
        cache_key = (self._cache_revision, append_counts)
        cached = self._row_group_cache.get(cache_key)
        if cached is not None:
            return cached
        buckets: dict[int, list[int]] = {}
        for row, (start, count) in enumerate(
            zip(self.cache_seqlens.tolist(), append_counts, strict=True)
        ):
            buckets.setdefault(int(start) + count, []).append(row)
        groups = tuple(
            (
                torch.tensor(rows, device=self.cache_seqlens.device, dtype=torch.long),
                valid_length,
            )
            for valid_length, rows in sorted(buckets.items())
        )
        self._row_group_cache.clear()
        self._row_group_cache[cache_key] = groups
        return groups

    def commit(self, token_count: int | Tensor) -> None:
        if isinstance(token_count, Tensor):
            if token_count.shape != self.cache_seqlens.shape:
                raise ValueError("per-row token_count must have shape [B]")
            counts = token_count.to(device=self.cache_seqlens.device, dtype=torch.int32)
            if bool(counts.le(0).any()):
                raise ValueError("per-row token_count values must be positive")
        else:
            if type(token_count) is not int or token_count <= 0:
                raise ValueError("token_count must be a positive integer")
            next_max = self._max_committed_len + token_count
            if next_max > self.max_cache_len:
                raise ValueError("committed cache length exceeds max_cache_len")
            self.cache_seqlens.add_(token_count)
            self._max_committed_len = next_max
            self._cache_revision += 1
            self._row_group_cache.clear()
            return
        next_lengths = self.cache_seqlens + counts
        if bool((next_lengths > self.max_cache_len).any()):
            raise ValueError("committed cache length exceeds max_cache_len")
        self.cache_seqlens.copy_(next_lengths)
        self._max_committed_len = int(next_lengths.max().item())
        self._cache_revision += 1
        self._row_group_cache.clear()

    def batch_select_indices(self, batch_indices: Tensor) -> None:
        if batch_indices.ndim != 1 or batch_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("batch_indices must be a one-dimensional integer tensor")
        indices = batch_indices.to(device=self.cache_seqlens.device, dtype=torch.long)
        self.layers = tuple(
            LayerKVCache(
                key=layer.key.index_select(0, indices),
                value=layer.value.index_select(0, indices),
            )
            for layer in self.layers
        )
        self.cache_seqlens = self.cache_seqlens.index_select(0, indices)
        self._max_committed_len = (
            int(self.cache_seqlens.max().item()) if self.cache_seqlens.numel() else 0
        )
        self._cache_revision += 1
        self._row_group_cache.clear()

    def reorder_cache(self, batch_indices: Tensor) -> None:
        self.batch_select_indices(batch_indices)


class BlockGenerationSession:
    """Own one request batch's committed-prefix cache and block execution state."""

    def __init__(
        self,
        model: nn.Module,
        *,
        batch_size: int,
        max_cache_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if getattr(model, "sequence_layout", None) != "chunk_causal":
            raise ValueError(
                "cached block generation requires chunk_causal sequence layout"
            )
        backbone = getattr(model, "backbone", None)
        if backbone is None or not backbone.blocks:
            raise ValueError("cached block generation requires a MF backbone")
        blocks = tuple(backbone.blocks)
        if any(type(block) is not MFBlock for block in blocks):
            raise TypeError("cached block generation requires standard MFBlock layers")
        if any(type(block.attention) is not SharedQKVOAttention for block in blocks):
            raise TypeError(
                "cached block generation currently requires shared attention"
            )
        attention = blocks[0].attention
        self.model = model
        self.backbone = backbone
        self.blocks = blocks
        self.device = torch.device(device)
        self.dtype = dtype
        self._modality_indices: dict[tuple[str, int, int], tuple[Tensor, Tensor]] = {}
        self.cache = BlockCache.allocate(
            num_layers=len(blocks),
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            num_heads=attention.num_heads,
            head_dim=attention.head_dim,
            device=self.device,
            dtype=dtype,
        )

    @property
    def past_key_values(self) -> BlockCache:
        return self.cache

    @property
    def cache_position(self) -> Tensor:
        return self.cache.cache_position

    def reset(self) -> None:
        self.cache.reset()

    def _forward_hidden(
        self,
        tokens: Tensor,
        position_ids: Tensor,
        *,
        apply_final_norm: bool = True,
        modality: Literal["vision", "text"] = "text",
        cache_append_lengths: Tensor | None = None,
    ) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[0] != self.cache.batch_size:
            raise ValueError("tokens must have shape [B, T, H] matching the cache")
        batch_size, token_count, hidden_size = tokens.shape
        if hidden_size != self.backbone.hidden_size:
            raise ValueError("token hidden size does not match the backbone")
        if tokens.device != self.device:
            raise ValueError("tokens and cache must share a device")
        if self.cache.get_seq_length() + token_count > self.cache.max_cache_len:
            raise ValueError("current block exceeds cache capacity")
        routing_key = (modality, batch_size, token_count)
        modality_indices = self._modality_indices.get(routing_key)
        if modality_indices is None:
            active = torch.arange(
                batch_size * token_count,
                device=tokens.device,
                dtype=torch.long,
            )
            empty = active[:0]
            modality_indices = (
                (active, empty) if modality == "vision" else (empty, active)
            )
            self._modality_indices[routing_key] = modality_indices
        flat_vision, flat_text = modality_indices
        cache_groups = self.cache.row_groups(
            token_count if cache_append_lengths is None else cache_append_lengths
        )
        hidden = tokens
        for layer_idx, block in enumerate(self.blocks):
            flat_hidden = hidden.flatten(0, 1)
            attention_input = block.attention_norm(
                flat_hidden,
                flat_vision,
                flat_text,
            ).view_as(hidden)
            layer_cache = self.cache.layers[layer_idx]
            attention_output = block.attention.forward_cached(
                attention_input,
                position_ids,
                key_cache=layer_cache.key,
                value_cache=layer_cache.value,
                cache_seqlens=self.cache.cache_seqlens,
                cache_groups=cache_groups,
                force_sdpa_cache=cache_append_lengths is not None,
            )
            hidden = hidden + attention_output
            flat_hidden = hidden.flatten(0, 1)
            ffn_input = block.ffn_norm(
                flat_hidden,
                flat_vision,
                flat_text,
            )
            ffn_output = block.ffn(
                ffn_input,
                flat_vision,
                flat_text,
            ).view_as(hidden)
            hidden = hidden + ffn_output
        if not apply_final_norm:
            return hidden
        if self.backbone.fp32_boundaries:
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                return self.backbone.final_norm(hidden.float())
        return self.backbone.final_norm(hidden)

    def prefill_vision_condition(
        self,
        latents_norm: Tensor,
        *,
        token_timestep: Tensor,
    ) -> None:
        geometry = getattr(self.model, "geometry", None) or GeometryContract()
        geometry.validate()
        if getattr(self.model, "image_chunk_conditioning", "token_additive") == (
            "legacy_active_prefix"
        ):
            tokens = self.model.embeddings.embed_vision(latents_norm, token_timestep)
            position_ids = build_vision_segment_mrope_positions(
                torch.zeros(
                    latents_norm.shape[0],
                    dtype=torch.long,
                    device=latents_norm.device,
                ),
                vision_tokens=geometry.vision_tokens,
                grid_size=geometry.vision_grid_size,
            )
        else:
            tokens = self.model.embeddings.embed_chunk_vision(
                latents_norm,
                token_timestep,
            )
            position_ids = (
                build_image_chunk_positions(
                    0,
                    height=geometry.vision_grid_size[0],
                    width=geometry.vision_grid_size[1],
                    device=latents_norm.device,
                )
                .unsqueeze(1)
                .expand(-1, latents_norm.shape[0], -1)
            )
        self._forward_hidden(
            tokens,
            position_ids,
            apply_final_norm=False,
            modality="vision",
        )
        self.cache.commit(tokens.shape[1])

    def prefill_text_condition(
        self,
        latents_norm: Tensor,
        *,
        position_base: int,
        content_mask: Tensor | None = None,
    ) -> None:
        if content_mask is not None:
            if (
                content_mask.dtype != torch.bool
                or content_mask.shape != latents_norm.shape[:2]
            ):
                raise ValueError("content_mask must be bool with shape [B, T]")
            lengths = content_mask.sum(dim=1, dtype=torch.long)
            expected = (
                torch.arange(content_mask.shape[1], device=content_mask.device)[None, :]
                < lengths[:, None]
            )
            if not torch.equal(content_mask, expected):
                raise ValueError("content_mask must be prefix-contiguous")
        else:
            lengths = None
        tokens = self.model.embeddings.embed_chunk_text(
            latents_norm,
            previous_x0_norm=torch.zeros_like(latents_norm),
            token_timestep=torch.ones(
                latents_norm.shape[:2],
                device=latents_norm.device,
                dtype=latents_norm.dtype,
            ),
        )
        position_ids = (
            build_text_chunk_positions(
                position_base,
                latents_norm.shape[1],
                device=latents_norm.device,
            )
            .unsqueeze(1)
            .expand(-1, latents_norm.shape[0], -1)
        )
        self._forward_hidden(
            tokens,
            position_ids,
            apply_final_norm=False,
            cache_append_lengths=lengths,
        )
        self.cache.commit(tokens.shape[1] if lengths is None else lengths)

    def predict_text_block(
        self,
        latents_norm: Tensor,
        *,
        previous_x0_norm: Tensor,
        token_timestep: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        tokens = self.model.embeddings.embed_chunk_text(
            latents_norm,
            previous_x0_norm=previous_x0_norm,
            token_timestep=token_timestep,
        )
        hidden = self._forward_hidden(tokens, position_ids)
        head = self.model.heads.text_output_head
        if self.model.heads.fp32_boundaries:
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                return head(hidden.to(dtype=head.weight.dtype))
        return head(hidden)

    def commit_text_block(self, latents_norm: Tensor, *, position_ids: Tensor) -> None:
        clean_tokens = self.model.embeddings.embed_chunk_text(
            latents_norm,
            previous_x0_norm=torch.zeros_like(latents_norm),
            token_timestep=torch.ones(
                latents_norm.shape[:2],
                device=latents_norm.device,
                dtype=latents_norm.dtype,
            ),
        )
        self._forward_hidden(clean_tokens, position_ids, apply_final_norm=False)
        self.cache.commit(latents_norm.shape[1])


__all__ = ["BlockCache", "BlockGenerationSession", "LayerKVCache"]
