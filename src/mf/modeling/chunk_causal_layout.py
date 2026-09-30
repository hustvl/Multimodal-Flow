from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor

from mf._compat import Self
from mf.contracts.chunks import ChunkModality, ChunkTokenView
from mf.contracts.physical import PhysicalSequenceLayout
from mf.contracts.sequence import MODALITY_REGISTRY
from mf.contracts.sequence import CompiledSequence
from mf.contracts.task_registry import BranchRole
from mf.modeling.packing import (
    PackedRoutingLayout,
    build_ordered_packed_routing_layout,
)
from mf.registries import register_physical_layout

_INTEGER_DTYPES = frozenset(
    {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)
_INDEX_DTYPES = frozenset({torch.int32, torch.int64})


@dataclass(frozen=True)
class ChunkCausalBlockSpec:
    """One compiler block with an explicit role from CompiledSequence."""

    sequence_id: int
    chunk_index: int
    modality: int | ChunkModality
    view: ChunkTokenView
    block_index: int
    active_mask: Tensor
    position_ids: Tensor
    source_indices: Tensor
    role: BranchRole
    output_slot: str | None = None
    frame_index: int | None = None
    timestamp: float | None = None
    source_id: str | None = None

    def validate(self) -> Self:
        for name, value in (
            ("sequence_id", self.sequence_id),
            ("chunk_index", self.chunk_index),
            ("block_index", self.block_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.modality, int) or self.modality < 0:
            raise ValueError("modality must be a non-negative integer")
        if not isinstance(self.view, ChunkTokenView):
            raise ValueError("view must be an ChunkTokenView")
        if self.active_mask.ndim != 1 or self.active_mask.dtype is not torch.bool:
            raise ValueError("active_mask must be bool [S]")
        if not bool(self.active_mask.any()):
            raise ValueError("chunk blocks must contain at least one active token")
        if self.position_ids.shape != (3, self.active_mask.shape[0]):
            raise ValueError("position_ids must have shape [3, S]")
        if self.position_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("position_ids must have an integer dtype")
        if self.source_indices.shape != self.active_mask.shape:
            raise ValueError("source_indices must have shape [S]")
        if self.source_indices.dtype not in _INDEX_DTYPES:
            raise ValueError("source_indices must have an integer index dtype")
        if self.position_ids.device != self.active_mask.device:
            raise ValueError("position_ids and active_mask must share one device")
        if self.source_indices.device != self.active_mask.device:
            raise ValueError("source_indices and active_mask must share one device")
        if self.output_slot is not None and (
            not isinstance(self.output_slot, str) or not self.output_slot.strip()
        ):
            raise ValueError("output_slot must be a non-empty string when provided")
        if not isinstance(self.role, BranchRole) or self.role is BranchRole.ABSENT:
            raise ValueError("role must be CONDITION or TARGET when provided")
        if self.frame_index is not None and (
            type(self.frame_index) is not int or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer when provided")
        if self.timestamp is not None and not isinstance(self.timestamp, (int, float)):
            raise ValueError("timestamp must be numeric when provided")
        if self.source_id is not None and (
            not isinstance(self.source_id, str) or not self.source_id.strip()
        ):
            raise ValueError("source_id must be a non-empty string when provided")
        return self


@dataclass(frozen=True)
class ChunkCausalSequencePlan:
    active_token_mask: Tensor
    dense_sequence_ids: Tensor
    dense_chunk_indices: Tensor
    dense_modality_ids: Tensor
    dense_view_ids: Tensor
    dense_block_indices: Tensor
    dense_target_mask: Tensor
    position_ids: Tensor
    packed_routing_layout: PackedRoutingLayout
    compiled_sequences: tuple[CompiledSequence, ...] | None = None


@dataclass(frozen=True)
class ChunkCausalRoutingLayout:
    sequence_plan: ChunkCausalSequencePlan
    cache_identity: tuple[Tensor, ...]

    def validate_cache_identity(self, current: tuple[Tensor, ...]) -> None:
        if len(current) != len(self.cache_identity) or any(
            actual is not expected
            for actual, expected in zip(current, self.cache_identity, strict=True)
        ):
            raise ValueError("chunk-causal routing cache identity changed")


def build_text_chunk_positions(
    position_base: int,
    token_count: int,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    if (
        isinstance(position_base, bool)
        or not isinstance(position_base, int)
        or position_base < 0
    ):
        raise ValueError("position_base must be a non-negative integer")
    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or token_count <= 0
    ):
        raise ValueError("token_count must be a positive integer")
    token = torch.arange(token_count, dtype=torch.long, device=device) + position_base
    return torch.stack((token, token, token))


def build_image_chunk_positions(
    position_base: int,
    *,
    height: int,
    width: int,
    device: torch.device | str = "cpu",
) -> Tensor:
    if (
        isinstance(position_base, bool)
        or not isinstance(position_base, int)
        or position_base < 0
    ):
        raise ValueError("position_base must be a non-negative integer")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (height, width)
    ):
        raise ValueError("height and width must be positive integers")
    y = torch.arange(height, dtype=torch.long, device=device).repeat_interleave(width)
    x = torch.arange(width, dtype=torch.long, device=device).repeat(height)
    base = torch.full_like(y, position_base)
    return torch.stack((base, y + position_base, x + position_base))


def _validate_sequence(
    specs: tuple[ChunkCausalBlockSpec, ...],
    *,
    allow_omitted_condition_chunks: bool = False,
) -> tuple[int, ChunkModality]:
    chunk_indices = sorted({spec.chunk_index for spec in specs})
    first_chunk = 0 if not allow_omitted_condition_chunks else chunk_indices[0]
    if chunk_indices != list(range(first_chunk, chunk_indices[-1] + 1)):
        raise ValueError(
            "chunk indices must be contiguous from zero within each sequence"
        )
    roles_by_chunk: dict[int, BranchRole] = {}
    for spec in specs:
        previous_role = roles_by_chunk.setdefault(spec.chunk_index, spec.role)
        if previous_role is not spec.role:
            raise ValueError("all physical blocks of one chunk must share its role")
    target_indices = tuple(
        chunk_index
        for chunk_index in chunk_indices
        if roles_by_chunk[chunk_index] is BranchRole.TARGET
    )
    if not target_indices:
        raise ValueError("a compiled sequence must contain a target chunk")
    target_index = target_indices[0]
    if target_indices != tuple(range(target_index, chunk_indices[-1] + 1)):
        raise ValueError("compiled target chunks must form the final ordered suffix")
    target = tuple(spec for spec in specs if spec.role is BranchRole.TARGET)
    metadata_by_chunk: dict[int, tuple[object, ...]] = {}
    for spec in specs:
        metadata = (
            spec.role,
            spec.frame_index,
            spec.timestamp,
            spec.source_id,
        )
        previous = metadata_by_chunk.setdefault(spec.chunk_index, metadata)
        if previous != metadata:
            raise ValueError(
                "all physical blocks of one semantic chunk must share sequence metadata"
            )
    for spec in specs:
        if spec.role is BranchRole.CONDITION and (
            spec.view is not ChunkTokenView.CONTENT or spec.output_slot is not None
        ):
            raise ValueError("context chunks require CONTENT view and no output slot")
    for chunk_index in target_indices:
        chunk_target = tuple(spec for spec in target if spec.chunk_index == chunk_index)
        modalities = {int(spec.modality) for spec in chunk_target}
        if len(modalities) != 1:
            raise ValueError("one target chunk cannot mix modalities")
        modality = next(iter(modalities))
        if modality == int(ChunkModality.TEXT):
            clean = {
                spec.block_index
                for spec in chunk_target
                if spec.view is ChunkTokenView.CONTENT
            }
            noisy = {
                spec.block_index
                for spec in chunk_target
                if spec.view is ChunkTokenView.NOISY_TEXT
            }
            if clean != noisy or not clean or clean != set(range(max(clean) + 1)):
                raise ValueError(
                    "text targets require paired contiguous clean/noisy blocks"
                )
            for spec in chunk_target:
                expected = "text" if spec.view is ChunkTokenView.NOISY_TEXT else None
                if spec.output_slot != expected:
                    raise ValueError(
                        "only noisy text target blocks may write output_slot='text'"
                    )
        elif any(spec.view is not ChunkTokenView.CONTENT for spec in chunk_target):
            raise ValueError("non-text target chunks require CONTENT view")
    return target_index, int(target[0].modality)


def _materialize(
    rows: tuple[tuple[ChunkCausalBlockSpec, ...], ...],
    active: Tensor,
    *,
    allow_omitted_condition_chunks: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch_size, sequence_length = active.shape
    device = active.device
    sequence = torch.full(
        (batch_size, sequence_length), -1, dtype=torch.long, device=device
    )
    chunk = torch.full_like(sequence, -1)
    modality = torch.full_like(sequence, -1)
    view = torch.full_like(sequence, -1)
    block = torch.full_like(sequence, -1)
    target = torch.zeros_like(active)
    positions = torch.zeros(
        (3, batch_size, sequence_length), dtype=torch.long, device=device
    )
    occupied = torch.zeros_like(active)
    for row_index, row in enumerate(rows):
        grouped: dict[int, list[ChunkCausalBlockSpec]] = defaultdict(list)
        for spec in row:
            grouped[spec.sequence_id].append(spec)
        for value in grouped.values():
            _validate_sequence(
                tuple(value),
                allow_omitted_condition_chunks=allow_omitted_condition_chunks,
            )
        for spec in row:
            source = spec.source_indices[spec.active_mask]
            source_rows = torch.div(source, sequence_length, rounding_mode="floor")
            if bool(source_rows.ne(row_index).any()):
                raise ValueError("source indices must belong to their physical row")
            local = source - row_index * sequence_length
            if bool(occupied[row_index].index_select(0, local).any()):
                raise ValueError("chunk block source indices must be disjoint")
            occupied[row_index, local] = True
            sequence[row_index, local] = spec.sequence_id
            chunk[row_index, local] = spec.chunk_index
            modality[row_index, local] = int(spec.modality)
            view[row_index, local] = int(spec.view)
            block[row_index, local] = spec.block_index
            target[row_index, local] = spec.role is BranchRole.TARGET
            positions[:, row_index, local] = spec.position_ids[:, spec.active_mask]
    if not torch.equal(occupied, active):
        raise ValueError("chunk blocks must partition active_token_mask exactly")
    return sequence, chunk, modality, view, block, target, positions


def compile_chunk_causal_sequence(
    rows: tuple[tuple[ChunkCausalBlockSpec, ...], ...],
    *,
    active_token_mask: Tensor,
    hidden_size: int,
    cache_identity: tuple[Tensor, ...] = (),
    compiled_sequences: tuple[CompiledSequence, ...] | None = None,
    allow_omitted_condition_chunks: bool = False,
) -> ChunkCausalRoutingLayout:
    """Compile chunk-causal visibility from an authoritative sequence contract."""

    if active_token_mask.ndim != 2 or active_token_mask.dtype is not torch.bool:
        raise ValueError("active_token_mask must be bool [B, L]")
    if (
        not rows
        or len(rows) != active_token_mask.shape[0]
        or any(not row for row in rows)
    ):
        raise ValueError("chunk rows must match a non-empty active_token_mask batch")
    checked = tuple(tuple(spec.validate() for spec in row) for row in rows)
    if compiled_sequences is not None:
        if len(compiled_sequences) != len(checked):
            raise ValueError("compiled_sequences must match the chunk row batch")
        for row, compiled in zip(checked, compiled_sequences, strict=True):
            compiled.validate()
            chunks = {chunk.chunk_index: chunk for chunk in compiled.chunks}
            seen: set[int] = set()
            for spec in row:
                chunk = chunks.get(spec.chunk_index)
                if chunk is None:
                    raise ValueError(
                        "physical layout references a chunk absent from CompiledSequence"
                    )
                expected_modality = MODALITY_REGISTRY.resolve(chunk.modality).stable_id
                if int(spec.modality) != expected_modality or spec.role is not chunk.role:
                    raise ValueError(
                        "physical chunk role/modality disagrees with CompiledSequence"
                    )
                seen.add(spec.chunk_index)
            missing = set(chunks) - seen
            if missing and (
                not allow_omitted_condition_chunks
                or any(chunks[index].target_mask for index in missing)
            ):
                raise ValueError(
                    "physical layout must materialize every CompiledSequence chunk"
                )
    sequence, chunk, modality, view, block, target, positions = _materialize(
        checked,
        active_token_mask,
        allow_omitted_condition_chunks=allow_omitted_condition_chunks,
    )
    ordered_sources: list[Tensor] = []
    for row in checked:
        grouped: dict[int, list[ChunkCausalBlockSpec]] = defaultdict(list)
        for spec in row:
            grouped[spec.sequence_id].append(spec)
        for sequence_id in sorted(grouped):
            specs = tuple(
                sorted(
                    grouped[sequence_id],
                    key=lambda spec: (
                        spec.chunk_index,
                        spec.view is ChunkTokenView.NOISY_TEXT,
                        spec.block_index,
                    ),
                )
            )
            for spec in specs:
                ordered_sources.append(spec.source_indices[spec.active_mask])
    ordered = torch.cat(tuple(ordered_sources))
    packed = build_ordered_packed_routing_layout(
        active_token_mask, positions, modality, ordered, hidden_size=hidden_size
    )
    plan = ChunkCausalSequencePlan(
        active_token_mask,
        sequence,
        chunk,
        modality,
        view,
        block,
        target,
        positions,
        packed,
        compiled_sequences,
    )
    return ChunkCausalRoutingLayout(plan, cache_identity)


def compile_physical_sequence_layout(
    physical: PhysicalSequenceLayout,
    *,
    hidden_size: int,
    cache_identity: tuple[Tensor, ...] = (),
) -> ChunkCausalRoutingLayout:
    """Compile a codec-provided physical sequence without image/text branches."""

    physical.validate()
    if physical.token_embeddings.shape[-1] != hidden_size:
        raise ValueError(
            "physical token embeddings must already match the backbone hidden size"
        )
    active = physical.active_token_mask
    if physical.ordered_active_indices is None:
        flat = torch.nonzero(active.reshape(-1), as_tuple=False).flatten()
        if flat.numel():
            keys = (
                physical.sequence_ids.reshape(-1).index_select(0, flat),
                physical.chunk_indices.reshape(-1).index_select(0, flat),
                physical.view_ids.reshape(-1).index_select(0, flat),
                physical.block_indices.reshape(-1).index_select(0, flat),
                flat,
            )
            order = torch.arange(flat.numel(), device=flat.device)
            for key in reversed(keys):
                order = order[torch.argsort(key.index_select(0, order), stable=True)]
            ordered = flat.index_select(0, order)
        else:
            ordered = flat
    else:
        ordered = physical.ordered_active_indices
    packed = build_ordered_packed_routing_layout(
        active,
        physical.position_ids,
        physical.modality_ids,
        ordered,
        hidden_size=hidden_size,
        target_mask=physical.target_mask,
    )
    plan = ChunkCausalSequencePlan(
        active,
        physical.sequence_ids,
        physical.chunk_indices,
        physical.modality_ids,
        physical.view_ids,
        physical.block_indices,
        physical.target_mask,
        physical.position_ids,
        packed,
        physical.compiled_sequences,
    )
    return ChunkCausalRoutingLayout(plan, cache_identity)


register_physical_layout(
    "chunk_causal",
    compile_physical_sequence_layout,
    replace=True,
)
