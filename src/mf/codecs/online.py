from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor, nn

from mf.contracts.batch import (
    TEXT_LATENT_DIM,
    TEXT_TOKENS,
    VISION_LATENT_DIM,
    VISION_TOKENS,
    EncodedTaskBatch,
    RawBatchPreparation,
    RawTaskBatch,
    task_branch_roles,
    tensor_mutation_version,
)
from mf.contracts.chunks import (
    ChunkRoutingMetadata,
    build_text_segment_ids,
)
from mf.contracts.geometry import GeometryContract
from mf.contracts.task_registry import BranchRole, task_definitions

TEXT_STATS_NORMAL = 0
TEXT_STATS_EOS = 1
TEXT_STATS_PAD_IGNORE = 2


class VisionEncoder(Protocol):
    def encode(self, images: Tensor) -> Tensor: ...


class TextEncoder(Protocol):
    def encode(self, token_ids: Tensor, attention_mask: Tensor) -> Tensor: ...


def _present_indices(present: Tensor) -> Tensor:
    if present.device.type != "cpu":
        raise ValueError("RawTaskBatch routing tensors must remain on CPU")
    return present.nonzero(as_tuple=False).flatten()


def _require_cpu_batch(batch: RawTaskBatch) -> None:
    for name in (
        "task_type",
        "vision_present",
        "text_present",
        "images",
        "text_token_ids",
        "text_content_mask",
        "text_prompt_token_ids",
        "text_prompt_content_mask",
    ):
        tensor = getattr(batch, name)
        if isinstance(tensor, Tensor) and tensor.device.type != "cpu":
            raise ValueError(
                f"RawTaskBatch.{name} must remain on CPU; got {tensor.device}"
            )
    if batch.physical_layout is not None:
        physical = batch.physical_layout.validate()
        for name in (
            "token_embeddings",
            "active_token_mask",
            "position_ids",
            "sequence_ids",
            "chunk_indices",
            "modality_ids",
            "view_ids",
            "block_indices",
            "target_mask",
        ):
            tensor = getattr(physical, name)
            if tensor.device.type != "cpu":
                raise ValueError(
                    f"RawTaskBatch.physical_layout.{name} must remain on CPU"
                )


def _codec_dimension(codec: nn.Module, name: str, default: int) -> int:
    value = getattr(codec, name, default)
    if type(value) is not int or value <= 0:
        raise ValueError(f"vision codec {name} must be a positive integer")
    return value


def _codec_device(codec: nn.Module, *, fallback: torch.device) -> torch.device:
    anchor = next(codec.parameters(), None)
    if anchor is None:
        anchor = next(codec.buffers(), None)
    return fallback if anchor is None else anchor.device


def _scatter_latents(
    compact: Tensor,
    indices: Tensor,
    *,
    present_count: int,
    batch_size: int,
    tokens: int,
    latent_dim: int,
    name: str,
) -> Tensor:
    expected = (present_count, tokens, latent_dim)
    if tuple(compact.shape) != expected:
        raise ValueError(
            f"{name} must have shape {list(expected)}; got {list(compact.shape)}"
        )
    if compact.device != indices.device:
        raise ValueError(
            f"{name} must be on the routing device; got {compact.device} and {indices.device}"
        )
    full = compact.new_zeros((batch_size, tokens, latent_dim))
    return full.index_copy(0, indices, compact)


def prepare_raw_batch_cpu(
    batch: RawTaskBatch,
    *,
    eos_token_id: int,
    geometry: GeometryContract | None = None,
) -> RawBatchPreparation:
    """Prepare deterministic metadata and compact images without codecs or device work."""
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("eos_token_id must be a non-negative integer")
    _require_cpu_batch(batch)
    batch.validate()
    size = batch.task_type.shape[0]
    mask = batch.text_content_mask
    if mask is None:
        mask = torch.zeros((size, TEXT_TOKENS), dtype=torch.bool)
    ids = batch.text_token_ids
    if ids is None:
        ids = torch.zeros_like(mask, dtype=torch.long)
    prompt_mask = batch.text_prompt_content_mask
    if prompt_mask is None:
        prompt_mask = torch.zeros((size, 0), dtype=torch.bool)
    vision_role, text_role = task_branch_roles(batch.task_type)
    routing = ChunkRoutingMetadata(
        vision_role=vision_role.to(torch.long),
        text_role=text_role.to(torch.long),
        text_content_mask=mask,
        text_prompt_content_mask=prompt_mask,
        text_segment_ids=build_text_segment_ids(
            ids, mask, eos_token_id=eos_token_id
        ),
        sequence_contracts=batch.sequence_contracts,
        compiled_sequences=batch.compiled_sequences,
        geometry=geometry,
    ).validate()
    vision_indices = _present_indices(batch.vision_present)
    inputs = RawBatchPreparation.inputs(batch)
    return RawBatchPreparation(
        source_tensors=inputs,
        source_versions=tuple(
            None if value is None else tensor_mutation_version(value)
            for value in inputs
        ),
        chunk_routing=routing,
        vision_indices=vision_indices,
        text_indices=_present_indices(batch.text_present),
        prompt_indices=_present_indices(prompt_mask.any(dim=1)),
        compact_images=(
            batch.images.index_select(0, vision_indices)
            if batch.images is not None and vision_indices.numel()
            else None
        ),
        sequence_signatures=(
            None
            if batch.sequence_contracts is None
            else tuple(sequence.routing_signature() for sequence in batch.sequence_contracts)
        ),
    )


class OnlineBatchEncoder(nn.Module):
    """Encode only present modalities and restore their original batch rows."""

    def __init__(
        self,
        *,
        vision: nn.Module,
        text: nn.Module,
        target_block_size: int | None = None,
        eos_token_id: int,
        geometry: GeometryContract | None = None,
    ) -> None:
        super().__init__()
        self.vision: VisionEncoder = vision
        self.text: TextEncoder = text
        self.vision_tokens = _codec_dimension(vision, "latent_tokens", VISION_TOKENS)
        self.vision_latent_dim = _codec_dimension(
            vision, "latent_dim", VISION_LATENT_DIM
        )
        self.text_latent_dim = _codec_dimension(text, "latent_dim", TEXT_LATENT_DIM)
        self.geometry = geometry or GeometryContract(
            vision_tokens=self.vision_tokens,
            vision_latent_dim=self.vision_latent_dim,
            text_latent_dim=self.text_latent_dim,
        )
        self.geometry.validate()
        if target_block_size is not None and (
            type(target_block_size) is not int or target_block_size <= 0
        ):
            raise ValueError("target_block_size must be a positive integer")
        self.target_block_size = target_block_size
        if type(eos_token_id) is not int or eos_token_id < 0:
            raise ValueError("eos_token_id must be a non-negative integer")
        self.eos_token_id = eos_token_id

    def _encode_target_rows(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
        active_indices_cpu: Tensor,
    ) -> Tensor:
        block_size = self.target_block_size
        if block_size is None:
            return self.text.encode(token_ids, attention_mask)
        rows, tokens = token_ids.shape
        if tokens % block_size != 0:
            raise ValueError(
                "target text length must be divisible by target_block_size"
            )
        flat_ids = token_ids.reshape(-1, block_size)
        flat_mask = attention_mask.reshape(-1, block_size)
        if (
            active_indices_cpu.device.type != "cpu"
            or active_indices_cpu.dtype is not torch.long
        ):
            raise ValueError("active target block indices must be CPU int64")
        if active_indices_cpu.numel() == 0:
            raise ValueError("target text rows must contain at least one active block")
        active_indices = active_indices_cpu.to(
            device=token_ids.device, non_blocking=True
        )
        encoded = self.text.encode(
            flat_ids.index_select(0, active_indices),
            flat_mask.index_select(0, active_indices),
        )
        flat_output = encoded.new_zeros(
            (flat_ids.shape[0], block_size, self.text_latent_dim)
        )
        flat_output.index_copy_(0, active_indices, encoded)
        return flat_output.view(rows, tokens, self.text_latent_dim)

    def _encode_compact_text(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
        task_type_cpu: Tensor,
        attention_mask_cpu: Tensor,
    ) -> Tensor:
        if self.target_block_size is None:
            return self.text.encode(token_ids, attention_mask)
        if (
            task_type_cpu.device.type != "cpu"
            or attention_mask_cpu.device.type != "cpu"
        ):
            raise ValueError("compact text routing inputs must remain on CPU")
        if tuple(attention_mask_cpu.shape) != tuple(attention_mask.shape):
            raise ValueError("CPU and device text masks must have identical shapes")
        target_rows = torch.zeros_like(task_type_cpu, dtype=torch.bool)
        for definition in task_definitions():
            if definition.role("text") is BranchRole.TARGET:
                target_rows |= task_type_cpu == definition.task_id
        target_indices_cpu = target_rows.nonzero(as_tuple=False).flatten()
        condition_indices_cpu = (~target_rows).nonzero(as_tuple=False).flatten()
        parts: list[tuple[Tensor, Tensor]] = []
        if target_indices_cpu.numel() > 0:
            target_indices = target_indices_cpu.to(
                device=token_ids.device, non_blocking=True
            )
            target_mask_cpu = attention_mask_cpu.index_select(0, target_indices_cpu)
            block_size = self.target_block_size
            active_indices_cpu = (
                target_mask_cpu.reshape(-1, block_size)
                .any(dim=1)
                .nonzero(as_tuple=False)
                .flatten()
            )
            target_encoded = self._encode_target_rows(
                token_ids.index_select(0, target_indices),
                attention_mask.index_select(0, target_indices),
                active_indices_cpu,
            )
            parts.append((target_indices, target_encoded))
        if condition_indices_cpu.numel() > 0:
            condition_indices = condition_indices_cpu.to(
                device=token_ids.device,
                non_blocking=True,
            )
            condition_encoded = self.text.encode(
                token_ids.index_select(0, condition_indices),
                attention_mask.index_select(0, condition_indices),
            )
            parts.append((condition_indices, condition_encoded))
        if not parts:
            raise ValueError("compact text batch must contain at least one row")
        compact = parts[0][1].new_zeros((*token_ids.shape, self.text_latent_dim))
        for indices, encoded in parts:
            compact.index_copy_(0, indices, encoded)
        return compact

    @torch.no_grad()
    def encode(self, batch: RawTaskBatch) -> EncodedTaskBatch:
        _require_cpu_batch(batch)
        if batch.physical_layout is not None:
            batch.validate()
            physical = batch.physical_layout
            assert physical is not None
            codec_device = _codec_device(self.text, fallback=physical.token_embeddings.device)
            physical = physical.to(codec_device, non_blocking=True)
            device = physical.token_embeddings.device
            text_tokens = (
                batch.text_content_mask.shape[1]
                if batch.text_content_mask is not None
                else TEXT_TOKENS
            )
            empty_text = torch.zeros(
                (batch.task_type.shape[0], text_tokens),
                dtype=torch.bool,
                device=device,
            )
            return EncodedTaskBatch(
                task_type=batch.task_type.to(device=device),
                vision_present=torch.zeros_like(
                    batch.vision_present, device=device
                ),
                text_present=torch.zeros_like(batch.text_present, device=device),
                text_content_mask=empty_text,
                chunk_routing_cpu=None,
                sequence_contracts=batch.sequence_contracts,
                compiled_sequences=batch.compiled_sequences,
                vision_tokens=self.vision_tokens,
                vision_latent_dim=self.vision_latent_dim,
                text_latent_dim=self.text_latent_dim,
                geometry=self.geometry,
                physical_layout=physical,
            ).validate()
        prepared = batch.cpu_preparation
        if prepared is None or not prepared.matches(batch):
            prepared = prepare_raw_batch_cpu(
                batch,
                eos_token_id=self.eos_token_id,
                geometry=self.geometry,
            )
        else:
            batch.validate()
        batch_size = batch.task_type.shape[0]
        chunk_routing_cpu = prepared.chunk_routing
        vision_indices_cpu = prepared.vision_indices
        text_indices_cpu = prepared.text_indices
        prompt_indices_cpu = prepared.prompt_indices
        vision_count = vision_indices_cpu.shape[0]
        text_count = text_indices_cpu.shape[0]
        prompt_count = prompt_indices_cpu.shape[0]

        cpu = batch.task_type.device
        vision_device = (
            _codec_device(self.vision, fallback=cpu) if vision_count > 0 else None
        )
        text_device = _codec_device(self.text, fallback=cpu) if text_count > 0 else None
        active_devices = {
            device for device in (vision_device, text_device) if device is not None
        }
        if len(active_devices) > 1:
            raise ValueError(
                "vision and text codecs must share a device when both modalities are present"
            )
        output_device = next(iter(active_devices), cpu)

        task_type = batch.task_type.to(device=output_device, non_blocking=True)
        vision_present = batch.vision_present.to(
            device=output_device, non_blocking=True
        )
        text_present = batch.text_present.to(device=output_device, non_blocking=True)

        vision_latents: Tensor | None = None
        if vision_count > 0:
            if batch.images is None:
                raise ValueError("images are required when vision is present")
            assert vision_device is not None
            vision_indices = vision_indices_cpu.to(
                device=vision_device, non_blocking=True
            )
            assert prepared.compact_images is not None
            compact_images = prepared.compact_images.to(
                device=vision_device,
                non_blocking=True,
            )
            compact_vision = self.vision.encode(compact_images)
            vision_latents = _scatter_latents(
                compact_vision,
                vision_indices,
                present_count=vision_count,
                batch_size=batch_size,
                tokens=self.vision_tokens,
                latent_dim=self.vision_latent_dim,
                name="vision codec output",
            )

        text_latents: Tensor | None = None
        text_token_ids: Tensor | None = None
        text_stats_type: Tensor | None = None
        text_tokens = (
            TEXT_TOKENS
            if batch.text_content_mask is None
            else batch.text_content_mask.shape[1]
        )
        text_content_mask = torch.zeros(
            (batch_size, text_tokens),
            dtype=torch.bool,
            device=output_device,
        )

        if text_count > 0:
            if batch.text_token_ids is None or batch.text_content_mask is None:
                raise ValueError("text inputs are required when text is present")
            assert text_device is not None
            compact_task_type_cpu = batch.task_type.index_select(0, text_indices_cpu)
            compact_mask_cpu = batch.text_content_mask.index_select(0, text_indices_cpu)
            text_indices = text_indices_cpu.to(device=text_device, non_blocking=True)
            compact_ids = batch.text_token_ids.index_select(0, text_indices_cpu).to(
                device=text_device,
                non_blocking=True,
            )
            compact_mask = compact_mask_cpu.to(
                device=text_device,
                non_blocking=True,
            )
            compact_text = self._encode_compact_text(
                compact_ids,
                compact_mask,
                compact_task_type_cpu,
                compact_mask_cpu,
            )
            text_latents = _scatter_latents(
                compact_text,
                text_indices,
                present_count=text_count,
                batch_size=batch_size,
                tokens=text_tokens,
                latent_dim=self.text_latent_dim,
                name="text codec output",
            )
            text_content_mask.index_copy_(0, text_indices, compact_mask)
            text_token_ids = compact_ids.new_zeros((batch_size, text_tokens))
            text_token_ids.index_copy_(0, text_indices, compact_ids)
            text_stats_type = torch.full_like(text_token_ids, TEXT_STATS_NORMAL)
            eos = (text_token_ids == self.eos_token_id) & text_content_mask
            text_stats_type.masked_fill_(eos, TEXT_STATS_EOS)

        prompt_latents: Tensor | None = None
        prompt_stats_type: Tensor | None = None
        prompt_token_ids: Tensor | None = None
        prompt_tokens = (
            0
            if batch.text_prompt_content_mask is None
            else batch.text_prompt_content_mask.shape[1]
        )
        prompt_content_mask = torch.zeros(
            (batch_size, prompt_tokens),
            dtype=torch.bool,
            device=output_device,
        )
        if prompt_count > 0:
            if (
                batch.text_prompt_token_ids is None
                or batch.text_prompt_content_mask is None
            ):
                raise ValueError("prompt inputs are required for image_to_text rows")
            assert text_device is not None
            prompt_indices = prompt_indices_cpu.to(
                device=text_device, non_blocking=True
            )
            compact_prompt_ids = batch.text_prompt_token_ids.index_select(
                0,
                prompt_indices_cpu,
            ).to(device=text_device, non_blocking=True)
            compact_prompt_mask = batch.text_prompt_content_mask.index_select(
                0,
                prompt_indices_cpu,
            ).to(device=text_device, non_blocking=True)
            compact_prompt = self.text.encode(compact_prompt_ids, compact_prompt_mask)
            prompt_latents = _scatter_latents(
                compact_prompt,
                prompt_indices,
                present_count=prompt_count,
                batch_size=batch_size,
                tokens=prompt_tokens,
                latent_dim=self.text_latent_dim,
                name="text prompt codec output",
            )
            prompt_content_mask.index_copy_(0, prompt_indices, compact_prompt_mask)
            prompt_token_ids = compact_prompt_ids.new_zeros((batch_size, prompt_tokens))
            prompt_token_ids.index_copy_(0, prompt_indices, compact_prompt_ids)
            prompt_stats_type = torch.full(
                (batch_size, prompt_tokens),
                TEXT_STATS_PAD_IGNORE,
                dtype=torch.long,
                device=output_device,
            )
            prompt_stats_type.masked_fill_(prompt_content_mask, TEXT_STATS_NORMAL)

        return EncodedTaskBatch(
            task_type=task_type,
            vision_present=vision_present,
            text_present=text_present,
            vision_latents_raw=vision_latents,
            text_latents_raw=text_latents,
            text_token_ids=text_token_ids,
            text_content_mask=text_content_mask,
            text_latent_stats_type=text_stats_type,
            text_prompt_latents_raw=prompt_latents,
            text_prompt_token_ids=prompt_token_ids,
            text_prompt_content_mask=prompt_content_mask if prompt_tokens > 0 else None,
            text_prompt_latent_stats_type=prompt_stats_type,
            chunk_routing_cpu=chunk_routing_cpu,
            sequence_contracts=batch.sequence_contracts,
            compiled_sequences=batch.compiled_sequences,
            vision_tokens=self.vision_tokens,
            vision_latent_dim=self.vision_latent_dim,
            text_latent_dim=self.text_latent_dim,
            geometry=self.geometry,
        )
