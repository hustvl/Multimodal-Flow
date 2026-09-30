from __future__ import annotations

import os
import time
import warnings
from importlib import import_module
from collections.abc import Callable, Sequence
from functools import lru_cache, partial

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from torch.nn.attention.flex_attention import FlexKernelOptions
except ImportError:
    _SUPPORTS_FLEX_BACKEND_SELECTION = False
else:
    _SUPPORTS_FLEX_BACKEND_SELECTION = "BACKEND" in FlexKernelOptions.__annotations__

from mf.modeling.attention_backend import FlexBackend, validate_flex_backend
from mf.modeling.layers import (
    RMSNorm,
    make_linear,
    route_by_modality_map,
)
from mf.modeling.mrope import (
    MROPE_SECTION,
    MRoPERotaryEmbedding,
    apply_rotary_pos_emb,
)
from mf.modeling.packing import PackedLayout


def _build_flex_kernel_options(
    forward_num_stages: str | None,
    backward_num_stages: str | None,
    rows_guaranteed_safe: str | None = None,
    *,
    backend: str | None = None,
) -> dict[str, int | str]:
    if backend == "FLASH":
        if forward_num_stages is not None or backward_num_stages is not None:
            raise RuntimeError("FA4 cannot use MF's Triton pipeline stage overrides")
        return {"BACKEND": "FLASH"}
    forward = "1" if forward_num_stages is None else forward_num_stages
    if forward not in {"1", "2", "3"}:
        raise RuntimeError("MF_FLEX_FWD_NUM_STAGES must be one of 1, 2, or 3")
    backward = "1" if backward_num_stages is None else backward_num_stages
    if backward not in {"1", "2", "3"}:
        raise RuntimeError("MF_FLEX_BWD_NUM_STAGES must be one of 1, 2, or 3")
    safe_rows = "0" if rows_guaranteed_safe is None else rows_guaranteed_safe
    if safe_rows not in {"0", "1"}:
        raise RuntimeError("MF_FLEX_ROWS_GUARANTEED_SAFE must be either 0 or 1")
    options: dict[str, int | str] = {
        "fwd_num_stages": int(forward),
        "bwd_num_stages": int(backward),
    }
    if safe_rows == "1":
        options["ROWS_GUARANTEED_SAFE"] = True
    if backend is not None:
        options["BACKEND"] = backend
    return options


FlashAttentionWithKVCache = Callable[..., Tensor]


def _flex_kernel_options(backend: FlexBackend) -> dict[str, int | str]:
    if backend == "fa4":
        if not _SUPPORTS_FLEX_BACKEND_SELECTION:
            raise RuntimeError(
                "FA4 FlexAttention requires a PyTorch build with BACKEND support"
            )
        return {"BACKEND": "FLASH"}
    return _build_flex_kernel_options(
        os.getenv("MF_FLEX_FWD_NUM_STAGES"),
        os.getenv("MF_FLEX_BWD_NUM_STAGES"),
        os.getenv("MF_FLEX_ROWS_GUARANTEED_SAFE"),
        backend="TRITON" if _SUPPORTS_FLEX_BACKEND_SELECTION else None,
    )


def _sdpa_with_kvcache(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    *,
    k: Tensor,
    v: Tensor,
    cache_seqlens: Tensor,
    causal: bool,
    cache_groups: tuple[tuple[Tensor, int], ...] | None = None,
) -> Tensor:
    """Portable cached attention backed by PyTorch's fused SDPA dispatcher."""

    if causal:
        raise ValueError("cached MF attention expects causal=False")
    if k.shape != v.shape or k.shape[:1] != query.shape[:1]:
        raise ValueError("cached K/V must match the query batch")
    block_length = k.shape[1]
    write_positions = cache_seqlens.to(torch.long).unsqueeze(1) + torch.arange(
        block_length,
        device=query.device,
    )
    batch_indices = torch.arange(query.shape[0], device=query.device).unsqueeze(1)
    key_cache[batch_indices, write_positions] = k
    value_cache[batch_indices, write_positions] = v

    if cache_groups is None:
        buckets: dict[int, list[int]] = {}
        for row, start in enumerate(cache_seqlens.tolist()):
            buckets.setdefault(int(start) + block_length, []).append(row)
        cache_groups = tuple(
            (
                torch.tensor(rows, device=query.device, dtype=torch.long),
                valid_length,
            )
            for valid_length, rows in sorted(buckets.items())
        )

    output = torch.empty_like(query)
    for row_indices, valid_length in cache_groups:
        group_query = query.index_select(0, row_indices)
        group_key = key_cache[:, :valid_length].index_select(0, row_indices)
        group_value = value_cache[:, :valid_length].index_select(0, row_indices)
        attended = F.scaled_dot_product_attention(
            group_query.transpose(1, 2),
            group_key.transpose(1, 2),
            group_value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        output.index_copy_(0, row_indices, attended)
    return output


@lru_cache(maxsize=1)
def _load_flash_attn_with_kvcache() -> FlashAttentionWithKVCache:
    try:
        from flash_attn import flash_attn_with_kvcache
    except (ImportError, OSError):
        warnings.warn(
            "flash_attn_with_kvcache is unavailable; cached inference is using PyTorch fused SDPA",
            RuntimeWarning,
            stacklevel=2,
        )
        return _sdpa_with_kvcache
    return flash_attn_with_kvcache


@lru_cache(maxsize=2)
def _load_compiled_flex_attention(backend: FlexBackend) -> Callable[..., Tensor]:
    """Compile Flex Attention once instead of taking its eager fallback."""

    validate_flex_backend(backend)
    try:
        from torch.nn.attention.flex_attention import flex_attention
    except ImportError as exc:
        raise RuntimeError("MF Flex Attention requires torch>=2.6") from exc
    if not hasattr(torch, "compile"):
        raise RuntimeError("MF Flex Attention requires torch.compile")
    if backend == "fa4":
        if not _SUPPORTS_FLEX_BACKEND_SELECTION:
            raise RuntimeError(
                "FA4 FlexAttention requires a PyTorch build with BACKEND support"
            )
        if not torch.compiler.is_compiling():
            try:
                import_module("flash_attn.cute")
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "FA4 FlexAttention requires a compatible flash-attn-4 installation "
                    "providing flash_attn.cute"
                ) from exc
        return torch.compile(
            partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
            dynamic=False,
            fullgraph=True,
        )
    return torch.compile(flex_attention, dynamic=True, fullgraph=True)


def _run_flex_attention(
    backend: FlexBackend,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: object,
) -> Tensor:
    compiled = _load_compiled_flex_attention(backend)
    if backend == "fa4":
        return compiled(query, key, value, block_mask=block_mask)
    return compiled(
        query,
        key,
        value,
        block_mask=block_mask,
        kernel_options=_flex_kernel_options(backend),
    )


def _pad_flex_qkv(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    sequence_length: int,
) -> tuple[Tensor, Tensor, Tensor, int]:
    real_token_count = query.shape[0]
    if key.shape[0] != real_token_count or value.shape[0] != real_token_count:
        raise ValueError("Flex query, key, and value token counts must match")
    if type(sequence_length) is not int or sequence_length < real_token_count:
        raise ValueError("Flex sequence length must cover every real token")
    if sequence_length == real_token_count:
        return query, key, value, real_token_count

    token_padding = sequence_length - real_token_count
    padding = (0, 0, 0, 0, 0, token_padding)
    return (
        F.pad(query, padding),
        F.pad(key, padding),
        F.pad(value, padding),
        real_token_count,
    )


def _to_flex_bhld(tensor: Tensor) -> Tensor:
    """Present contiguous LHD storage with the exact production Flex strides."""

    if tensor.ndim != 3:
        raise ValueError("Flex input must have shape [tokens, heads, head_dim]")
    return tensor.permute(1, 0, 2).unsqueeze(0)


def prewarm_compiled_flex_attention(
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int,
    head_dim: int,
    sequence_lengths: tuple[int, ...],
    kernel_block_size: int,
    text_block_size: int,
    flex_backend: FlexBackend = "triton",
) -> tuple[float, ...]:
    """Compile the bounded production Flex shapes before the first training update."""

    from mf.modeling.chunk_adapter import build_chunk_flex_prewarm_mask

    if device.type != "cuda":
        raise RuntimeError("MF Flex prewarm requires a CUDA device")
    if dtype not in {torch.bfloat16, torch.float16}:
        raise ValueError("MF Flex prewarm requires a 16-bit floating dtype")
    if type(num_heads) is not int or num_heads <= 0:
        raise ValueError("MF Flex prewarm num_heads must be positive")
    if type(head_dim) is not int or head_dim <= 0:
        raise ValueError("MF Flex prewarm head_dim must be positive")
    if not sequence_lengths:
        return ()

    generator = torch.Generator(device=device).manual_seed(20260817)
    elapsed_seconds: list[float] = []
    for sequence_length in sequence_lengths:
        block_mask = build_chunk_flex_prewarm_mask(
            device,
            sequence_length=sequence_length,
            kernel_block_size=kernel_block_size,
            text_block_size=text_block_size,
            flex_backend=flex_backend,
        )
        shape = (sequence_length, num_heads, head_dim)
        query_lhd = torch.randn(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
            requires_grad=True,
        )
        key_lhd = torch.randn(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
            requires_grad=True,
        )
        value_lhd = torch.randn(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
            requires_grad=True,
        )
        query = _to_flex_bhld(query_lhd)
        key = _to_flex_bhld(key_lhd)
        value = _to_flex_bhld(value_lhd)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = _run_flex_attention(
            flex_backend,
            query,
            key,
            value,
            block_mask,
        )
        loss = output.float().square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed_seconds.append(time.perf_counter() - started)

        if not torch.isfinite(output).all():
            raise RuntimeError("MF Flex prewarm output is non-finite")
        for name, tensor in (
            ("query", query_lhd),
            ("key", key_lhd),
            ("value", value_lhd),
        ):
            if tensor.grad is None or not torch.isfinite(tensor.grad).all():
                raise RuntimeError(f"MF Flex prewarm {name} gradient is non-finite")
            if torch.count_nonzero(tensor.grad) == 0:
                raise RuntimeError(f"MF Flex prewarm {name} gradient is zero")
        del block_mask, key, key_lhd, loss, output, query, query_lhd, value, value_lhd

    torch.cuda.empty_cache()
    return tuple(elapsed_seconds)


class _QKVOAttentionBase(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mrope_section: tuple[int, int, int],
        rope_theta: float,
    ) -> None:
        super().__init__()
        if hidden_size != num_heads * head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rotary = (
            MRoPERotaryEmbedding(
                head_dim=head_dim,
                mrope_section=mrope_section,
                rope_theta=rope_theta,
            )
            if sum(mrope_section) == head_dim // 2
            else None
        )
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def _project_qkv(
        self, qkv: Tensor, position_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if qkv.ndim != 3 or qkv.shape[-1] != 3 * self.hidden_size:
            raise ValueError("cached qkv must have shape [B, T, 3 * hidden_size]")
        if position_ids.shape != (3, qkv.shape[0], qkv.shape[1]):
            raise ValueError("cached position_ids must have shape [3, B, T]")
        query, key, value = qkv.view(
            qkv.shape[0],
            qkv.shape[1],
            3,
            self.num_heads,
            self.head_dim,
        ).unbind(dim=2)
        query = self.q_norm(query)
        key = self.k_norm(key)
        if self.rotary is not None:
            flat_query = query.flatten(0, 1)
            flat_key = key.flatten(0, 1)
            flat_positions = position_ids.flatten(1, 2)
            cos, sin = self.rotary(flat_positions, dtype=query.dtype)
            flat_query, flat_key = apply_rotary_pos_emb(
                flat_query,
                flat_key,
                cos,
                sin,
            )
            query = flat_query.view_as(query)
            key = flat_key.view_as(key)
        return query, key, value

    def _normalize_packed_qk(
        self,
        query: Tensor,
        key: Tensor,
        packed_layout: PackedLayout,
    ) -> tuple[Tensor, Tensor]:
        return self.q_norm(query), self.k_norm(key)

    def _attend(
        self,
        qkv: Tensor,
        packed_layout: PackedLayout,
        *,
        implementation: str = "flex",
        flex_backend: FlexBackend = "triton",
    ) -> Tensor:
        if packed_layout.hidden_size != self.hidden_size:
            raise ValueError(
                f"packed hidden size must be {self.hidden_size}; got {packed_layout.hidden_size}"
            )

        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=1)
        query, key = self._normalize_packed_qk(query, key, packed_layout)
        if self.rotary is not None:
            cos, sin = self.rotary(packed_layout.positions, dtype=query.dtype)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if implementation == "flex":
            if packed_layout.flex_block_mask is None:
                raise ValueError("Flex Attention requires a compiled block mask")
            if packed_layout.flex_sequence_length is None:
                raise ValueError("Flex Attention requires a padded sequence length")
            if packed_layout.tokens.device.type != "cuda":
                raise RuntimeError("MF Flex Attention requires CUDA tensors")
            query, key, value, real_token_count = _pad_flex_qkv(
                query,
                key,
                value,
                sequence_length=packed_layout.flex_sequence_length,
            )
            attended = _run_flex_attention(
                flex_backend,
                _to_flex_bhld(query),
                _to_flex_bhld(key),
                _to_flex_bhld(value),
                packed_layout.flex_block_mask,
            )
            attended = attended.squeeze(0).permute(1, 0, 2)
            return attended[:real_token_count].reshape(
                real_token_count, self.hidden_size
            )

        raise ValueError("MF training requires FlexAttention")


class SharedQKVOAttention(_QKVOAttentionBase):
    """One pre-projection and output projection shared by both modalities."""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        mrope_section: tuple[int, int, int] = MROPE_SECTION,
        rope_theta: float = 10_000.0,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, head_dim, mrope_section, rope_theta)
        self.qkv_proj = make_linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = make_linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        packed_layout: PackedLayout,
        *,
        implementation: str = "flex",
        flex_backend: FlexBackend = "triton",
    ) -> Tensor:
        attended = self._attend(
            self.qkv_proj(packed_layout.tokens),
            packed_layout,
            implementation=implementation,
            flex_backend=flex_backend,
        )
        return self.out_proj(attended)

    def forward_cached(
        self,
        tokens: Tensor,
        position_ids: Tensor,
        *,
        key_cache: Tensor,
        value_cache: Tensor,
        cache_seqlens: Tensor,
        cache_groups: tuple[tuple[Tensor, int], ...] | None = None,
        force_sdpa_cache: bool = False,
    ) -> Tensor:
        """Attend one dense block to a committed prefix plus the current block."""

        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size:
            raise ValueError("cached tokens must have shape [B, T, hidden_size]")
        expected_cache = (
            tokens.shape[0],
            key_cache.shape[1],
            self.num_heads,
            self.head_dim,
        )
        if key_cache.shape != expected_cache or value_cache.shape != expected_cache:
            raise ValueError("key_cache and value_cache have incompatible shapes")
        if (
            cache_seqlens.shape != (tokens.shape[0],)
            or cache_seqlens.dtype is not torch.int32
        ):
            raise ValueError("cache_seqlens must be int32 with shape [B]")
        if (
            key_cache.device != tokens.device
            or value_cache.device != tokens.device
            or cache_seqlens.device != tokens.device
        ):
            raise ValueError("tokens and cache tensors must share a device")
        query, key, value = self._project_qkv(self.qkv_proj(tokens), position_ids)
        if key_cache.dtype != key.dtype or value_cache.dtype != value.dtype:
            raise ValueError("projected K/V and cache tensors must share a dtype")
        cache_backend = _load_flash_attn_with_kvcache()
        if force_sdpa_cache:
            cache_backend = _sdpa_with_kvcache
        cache_kwargs = {
            "k": key,
            "v": value,
            "cache_seqlens": cache_seqlens,
            "causal": False,
        }
        if cache_backend is _sdpa_with_kvcache:
            cache_kwargs["cache_groups"] = cache_groups
        attended = cache_backend(query, key_cache, value_cache, **cache_kwargs)
        return self.out_proj(attended.reshape_as(tokens))


class ModalitySpecificQKVOAttention(_QKVOAttentionBase):
    """Independent vision/text QKVO projections with one global attention map."""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        mrope_section: tuple[int, int, int] = MROPE_SECTION,
        rope_theta: float = 10_000.0,
        *,
        bias: bool = True,
        modality_specific_qk_norms: bool = False,
        modality_ids: Sequence[int] = (0, 1),
    ) -> None:
        super().__init__(hidden_size, num_heads, head_dim, mrope_section, rope_theta)
        self.vision_q_norm = (
            RMSNorm(head_dim, eps=1e-6) if modality_specific_qk_norms else None
        )
        self.vision_k_norm = (
            RMSNorm(head_dim, eps=1e-6) if modality_specific_qk_norms else None
        )
        self.vision_qkv_proj = make_linear(hidden_size, 3 * hidden_size, bias=bias)
        self.text_qkv_proj = make_linear(hidden_size, 3 * hidden_size, bias=bias)
        self.vision_out_proj = make_linear(hidden_size, hidden_size, bias=bias)
        self.text_out_proj = make_linear(hidden_size, hidden_size, bias=bias)
        self.extra_qkv_proj = nn.ModuleDict(
            {
                str(modality_id): make_linear(
                    hidden_size, 3 * hidden_size, bias=bias
                )
                for modality_id in modality_ids
                if modality_id not in (0, 1)
            }
        )
        self.extra_out_proj = nn.ModuleDict(
            {
                str(modality_id): make_linear(hidden_size, hidden_size, bias=bias)
                for modality_id in modality_ids
                if modality_id not in (0, 1)
            }
        )
        self.extra_q_norm = nn.ModuleDict(
            {
                str(modality_id): RMSNorm(head_dim, eps=1e-6)
                for modality_id in modality_ids
                if modality_id not in (0, 1) and modality_specific_qk_norms
            }
        )
        self.extra_k_norm = nn.ModuleDict(
            {
                str(modality_id): RMSNorm(head_dim, eps=1e-6)
                for modality_id in modality_ids
                if modality_id not in (0, 1) and modality_specific_qk_norms
            }
        )

    def _normalize_packed_qk(
        self,
        query: Tensor,
        key: Tensor,
        packed_layout: PackedLayout,
    ) -> tuple[Tensor, Tensor]:
        text_query, text_key = self.q_norm(query), self.k_norm(key)
        if self.vision_q_norm is None or self.vision_k_norm is None:
            return text_query, text_key
        q_modules = {"0": self.vision_q_norm, **self.extra_q_norm}
        k_modules = {"0": self.vision_k_norm, **self.extra_k_norm}
        return (
            route_by_modality_map(
                query,
                packed_layout.modality_indices,
                q_modules,
                fallback=self.q_norm,
            ),
            route_by_modality_map(
                key,
                packed_layout.modality_indices,
                k_modules,
                fallback=self.k_norm,
            ),
        )

    def forward(
        self,
        packed_layout: PackedLayout,
        *,
        implementation: str = "flex",
        flex_backend: FlexBackend = "triton",
    ) -> Tensor:
        qkv = route_by_modality_map(
            packed_layout.tokens,
            packed_layout.modality_indices,
            {"0": self.vision_qkv_proj, "1": self.text_qkv_proj, **self.extra_qkv_proj},
            fallback=self.text_qkv_proj,
        )
        attended = self._attend(
            qkv,
            packed_layout,
            implementation=implementation,
            flex_backend=flex_backend,
        )
        return route_by_modality_map(
            attended,
            packed_layout.modality_indices,
            {"0": self.vision_out_proj, "1": self.text_out_proj, **self.extra_out_proj},
            fallback=self.text_out_proj,
        )
