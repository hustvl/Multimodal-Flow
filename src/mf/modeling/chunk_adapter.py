from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor

from mf.contracts.batch import (
    TEXT_PREFIX_TOKENS,
    BranchRole,
    tensor_mutation_version,
)
from mf.contracts.chunks import ChunkModality, ChunkRoutingMetadata, ChunkTokenView
from mf.contracts.geometry import GeometryContract
from mf.contracts.model import MFModelInput
from mf.modeling.chunk_causal_layout import (
    ChunkCausalBlockSpec,
    ChunkCausalRoutingLayout,
    build_image_chunk_positions,
    compile_chunk_causal_sequence,
)
from mf.registries import PHYSICAL_LAYOUT_REGISTRY
from mf.modeling.attention_backend import (
    flex_mask_block_size,
    mark_flex_mask_metadata_static,
)
from mf.modeling.mrope import (
    VISION_PREFIX_POSITION_GROUPS,
    build_text_segment_mrope_positions,
    build_vision_segment_mrope_positions,
)

ImageChunkConditioning = Literal["token_additive", "legacy_active_prefix"]
T2IChunkSemantics = Literal["chunk_native", "legacy_block_exact"]


def chunk_causal_physical_token_counts(
    routing: ChunkRoutingMetadata,
    *,
    image_chunk_conditioning: ImageChunkConditioning = "token_additive",
    t2i_chunk_semantics: T2IChunkSemantics = "chunk_native",
) -> Tensor:
    """Count tokens retained by the MF chunk compiler for each logical row."""

    routing.validate()
    geometry = routing.geometry or GeometryContract()
    geometry.validate()
    if image_chunk_conditioning not in {"token_additive", "legacy_active_prefix"}:
        raise ValueError("unknown image chunk conditioning mode")
    if t2i_chunk_semantics not in {"chunk_native", "legacy_block_exact"}:
        raise ValueError("unknown T2I chunk semantics")
    if (
        t2i_chunk_semantics == "legacy_block_exact"
        and image_chunk_conditioning != "legacy_active_prefix"
    ):
        raise ValueError(
            "legacy_block_exact T2I semantics require legacy_active_prefix image conditioning"
        )
    vision_active = routing.vision_role != int(BranchRole.ABSENT)
    text_active = routing.text_role != int(BranchRole.ABSENT)
    text_target = routing.text_role == int(BranchRole.TARGET)
    text_tokens = routing.text_content_mask.sum(dim=1, dtype=torch.long)
    prompt_tokens = routing.text_prompt_content_mask.sum(dim=1, dtype=torch.long)
    legacy_t2i_prefix = torch.zeros_like(vision_active)
    if t2i_chunk_semantics == "legacy_block_exact":
        legacy_t2i_prefix = (routing.vision_role == int(BranchRole.TARGET)) & (
            routing.text_role == int(BranchRole.CONDITION)
        )
    counts = (
        vision_active.to(dtype=torch.long)
        * (
            geometry.vision_layout_tokens
            if image_chunk_conditioning == "legacy_active_prefix"
            else geometry.vision_tokens
        )
        + prompt_tokens
        + text_tokens
        * (text_active.to(dtype=torch.long) + text_target.to(dtype=torch.long))
        + legacy_t2i_prefix.to(dtype=torch.long) * TEXT_PREFIX_TOKENS
    )
    if bool(counts.le(0).any()):
        raise ValueError("MF chunk rows must retain at least one physical token")
    return counts


def _source(row: int, sequence_length: int, start: int, indices: Tensor) -> Tensor:
    return row * sequence_length + start + indices


def _spec(
    *,
    row: int,
    sequence_length: int,
    physical_start: int,
    token_indices: Tensor,
    sequence_id: int,
    chunk_index: int,
    modality: ChunkModality,
    position_base: int,
    view: ChunkTokenView = ChunkTokenView.CONTENT,
    block_index: int = 0,
    output_slot: str | None = None,
    role: BranchRole = BranchRole.CONDITION,
    local_positions: Tensor | None = None,
    position_ids: Tensor | None = None,
    grid_size: tuple[int, int] | None = None,
) -> ChunkCausalBlockSpec:
    if local_positions is None:
        local_positions = torch.arange(
            token_indices.numel(), dtype=torch.long, device=token_indices.device
        )
    if position_ids is not None:
        positions = position_ids
    elif modality is ChunkModality.IMAGE:
        if grid_size is None:
            raise ValueError("image chunk specs require grid_size or position_ids")
        positions = build_image_chunk_positions(
            position_base,
            height=grid_size[0],
            width=grid_size[1],
            device=token_indices.device,
        ).index_select(1, token_indices)
    else:
        absolute_positions = local_positions + position_base
        positions = torch.stack(
            (absolute_positions, absolute_positions, absolute_positions)
        )
    active = torch.ones(
        token_indices.numel(), dtype=torch.bool, device=token_indices.device
    )
    return ChunkCausalBlockSpec(
        sequence_id=sequence_id,
        chunk_index=chunk_index,
        modality=modality,
        view=view,
        block_index=block_index,
        active_mask=active,
        position_ids=positions,
        source_indices=_source(row, sequence_length, physical_start, token_indices),
        output_slot=output_slot,
        role=role,
    )


def _image_spec(
    *,
    row: int,
    sequence_length: int,
    sequence_id: int,
    chunk_index: int,
    position_base: int,
    image_chunk_conditioning: ImageChunkConditioning,
    device: torch.device,
    geometry: GeometryContract | None = None,
    output_slot: str | None = None,
    role: BranchRole = BranchRole.CONDITION,
) -> ChunkCausalBlockSpec:
    geometry = geometry or GeometryContract()
    geometry.validate()
    if image_chunk_conditioning == "legacy_active_prefix":
        token_indices = torch.arange(
            geometry.vision_layout_tokens, dtype=torch.long, device=device
        )
        positions = build_vision_segment_mrope_positions(
            torch.tensor([position_base], dtype=torch.long, device=device),
            vision_tokens=geometry.vision_tokens,
            grid_size=geometry.vision_grid_size,
        )[:, 0]
        physical_start = 0
    else:
        token_indices = torch.arange(
            geometry.vision_tokens, dtype=torch.long, device=device
        )
        positions = build_image_chunk_positions(
            position_base,
            height=geometry.vision_grid_size[0],
            width=geometry.vision_grid_size[1],
            device=device,
        )
        physical_start = geometry.vision_prefix_tokens
    return _spec(
        row=row,
        sequence_length=sequence_length,
        physical_start=physical_start,
        token_indices=token_indices,
        sequence_id=sequence_id,
        chunk_index=chunk_index,
        modality=ChunkModality.IMAGE,
        position_base=position_base,
        output_slot=output_slot,
        role=role,
        position_ids=positions,
        grid_size=geometry.vision_grid_size,
    )


def _target_text_specs(
    *,
    row: int,
    sequence_length: int,
    clean_start: int,
    noisy_start: int,
    token_indices: Tensor,
    local_positions: Tensor,
    sequence_id: int,
    chunk_index: int,
    block_size: int,
    position_base: int,
) -> tuple[ChunkCausalBlockSpec, ...]:
    blocks = torch.div(local_positions, block_size, rounding_mode="floor")
    specs: list[ChunkCausalBlockSpec] = []
    for block_index in torch.unique(blocks, sorted=True).tolist():
        select = blocks == block_index
        block_tokens = token_indices[select]
        block_positions = local_positions[select]
        specs.append(
            _spec(
                row=row,
                sequence_length=sequence_length,
                physical_start=clean_start,
                token_indices=block_tokens,
                local_positions=block_positions,
                sequence_id=sequence_id,
                chunk_index=chunk_index,
                modality=ChunkModality.TEXT,
                position_base=position_base,
                block_index=block_index,
                role=BranchRole.TARGET,
            )
        )
        specs.append(
            _spec(
                row=row,
                sequence_length=sequence_length,
                physical_start=noisy_start,
                token_indices=block_tokens,
                local_positions=block_positions,
                sequence_id=sequence_id,
                chunk_index=chunk_index,
                modality=ChunkModality.TEXT,
                position_base=position_base,
                view=ChunkTokenView.NOISY_TEXT,
                block_index=block_index,
                output_slot="text",
                role=BranchRole.TARGET,
            )
        )
    return tuple(specs)


def _row_specs(
    routing: ChunkRoutingMetadata | MFModelInput,
    row: int,
    *,
    sequence_length: int,
    clean_start: int,
    noisy_start: int,
    prompt_content_start: int,
    text_prefix_start: int,
    text_block_size: int,
    image_chunk_conditioning: ImageChunkConditioning,
    t2i_chunk_semantics: T2IChunkSemantics,
) -> tuple[ChunkCausalBlockSpec, ...]:
    assert routing.text_segment_ids is not None
    device = routing.text_content_mask.device
    geometry = routing.geometry or GeometryContract()
    geometry.validate()
    vision_position_span = max(geometry.vision_grid_size)
    compiled = (
        None
        if routing.compiled_sequences is None
        else routing.compiled_sequences[row]
    )
    vision_role = (
        compiled.primary_role("image")
        if compiled is not None
        else BranchRole(int(routing.vision_role[row].item()))
    )
    text_role = (
        compiled.primary_role("text")
        if compiled is not None
        else BranchRole(int(routing.text_role[row].item()))
    )
    if vision_role is BranchRole.TARGET and text_role is BranchRole.TARGET:
        raise ValueError("MF requires at most one target modality per chunk")

    text_indices = torch.nonzero(
        routing.text_content_mask[row], as_tuple=False
    ).flatten()
    prompt_indices = torch.nonzero(
        routing.text_prompt_content_mask[row], as_tuple=False
    ).flatten()
    segment_ids = routing.text_segment_ids[row]
    specs: list[ChunkCausalBlockSpec] = []

    if vision_role is BranchRole.ABSENT and text_role is BranchRole.TARGET:
        target_chunk = 0
        if prompt_indices.numel():
            unique_segments = torch.unique(segment_ids[text_indices], sorted=True)
            if unique_segments.numel() != 1 or int(unique_segments[0].item()) != 0:
                raise ValueError(
                    "prompt-conditioned text targets require exactly one sequence with id 0"
                )
            prompt_local = torch.arange(
                prompt_indices.numel(),
                dtype=torch.long,
                device=prompt_indices.device,
            )
            specs.append(
                _spec(
                    row=row,
                    sequence_length=sequence_length,
                    physical_start=prompt_content_start,
                    token_indices=prompt_indices,
                    local_positions=prompt_local,
                    sequence_id=0,
                    chunk_index=0,
                    modality=ChunkModality.TEXT,
                    position_base=0,
                )
            )
            target_chunk = 1
        for sequence_id in torch.unique(
            segment_ids[text_indices], sorted=True
        ).tolist():
            segment_tokens = text_indices[segment_ids[text_indices] == sequence_id]
            local = torch.arange(
                segment_tokens.numel(), dtype=torch.long, device=segment_tokens.device
            )
            specs.extend(
                _target_text_specs(
                    row=row,
                    sequence_length=sequence_length,
                    clean_start=clean_start,
                    noisy_start=noisy_start,
                    token_indices=segment_tokens,
                    local_positions=local,
                    sequence_id=sequence_id,
                    chunk_index=target_chunk,
                    block_size=text_block_size,
                    position_base=int(prompt_indices.numel()),
                )
            )
        return tuple(specs)

    if vision_role is BranchRole.TARGET and text_role is BranchRole.ABSENT:
        return (
            _image_spec(
                row=row,
                sequence_length=sequence_length,
                sequence_id=0,
                chunk_index=0,
                position_base=0,
                image_chunk_conditioning=image_chunk_conditioning,
                device=device,
                geometry=geometry,
                output_slot="image",
                role=BranchRole.TARGET,
            ),
        )

    if text_indices.numel() == 0:
        raise ValueError("conditional MF samples require active text tokens")
    unique_segments = torch.unique(segment_ids[text_indices], sorted=True)
    if unique_segments.numel() != 1:
        raise ValueError("conditional MF samples require exactly one text content")

    if vision_role is BranchRole.TARGET and text_role is BranchRole.CONDITION:
        if t2i_chunk_semantics == "legacy_block_exact":
            text_positions = build_text_segment_mrope_positions(
                torch.tensor(
                    [VISION_PREFIX_POSITION_GROUPS + vision_position_span],
                    dtype=torch.long,
                    device=device,
                ),
                text_tokens=int(text_indices.numel()),
            )[:, 0]
            prefix_indices = torch.arange(
                TEXT_PREFIX_TOKENS,
                dtype=torch.long,
                device=device,
            )
            specs.append(
                _spec(
                    row=row,
                    sequence_length=sequence_length,
                    physical_start=text_prefix_start,
                    token_indices=prefix_indices,
                    sequence_id=0,
                    chunk_index=0,
                    modality=ChunkModality.TEXT,
                    position_base=0,
                    position_ids=text_positions[:, :TEXT_PREFIX_TOKENS],
                )
            )
            specs.append(
                _spec(
                    row=row,
                    sequence_length=sequence_length,
                    physical_start=clean_start,
                    token_indices=text_indices,
                    sequence_id=0,
                    chunk_index=0,
                    modality=ChunkModality.TEXT,
                    position_base=0,
                    position_ids=text_positions[:, TEXT_PREFIX_TOKENS:],
                )
            )
            specs.append(
                _image_spec(
                    row=row,
                    sequence_length=sequence_length,
                    sequence_id=0,
                    chunk_index=1,
                    position_base=0,
                    image_chunk_conditioning=image_chunk_conditioning,
                    device=device,
                    geometry=geometry,
                    output_slot="image",
                    role=BranchRole.TARGET,
                )
            )
            return tuple(specs)
        local = torch.arange(text_indices.numel(), dtype=torch.long, device=device)
        specs.append(
            _spec(
                row=row,
                sequence_length=sequence_length,
                physical_start=clean_start,
                token_indices=text_indices,
                local_positions=local,
                sequence_id=0,
                chunk_index=0,
                modality=ChunkModality.TEXT,
                position_base=0,
            )
        )
        specs.append(
            _image_spec(
                row=row,
                sequence_length=sequence_length,
                sequence_id=0,
                chunk_index=1,
                position_base=int(text_indices.numel()),
                image_chunk_conditioning=image_chunk_conditioning,
                device=device,
                geometry=geometry,
                output_slot="image",
                role=BranchRole.TARGET,
            )
        )
        return tuple(specs)

    if vision_role is BranchRole.CONDITION and text_role is BranchRole.TARGET:
        specs.append(
            _image_spec(
                row=row,
                sequence_length=sequence_length,
                sequence_id=0,
                chunk_index=0,
                position_base=0,
                image_chunk_conditioning=image_chunk_conditioning,
                device=device,
                geometry=geometry,
            )
        )
        target_chunk = 1
        if prompt_indices.numel():
            specs.append(
                _spec(
                    row=row,
                    sequence_length=sequence_length,
                    physical_start=prompt_content_start,
                    token_indices=prompt_indices,
                    sequence_id=0,
                    chunk_index=1,
                    modality=ChunkModality.TEXT,
                    position_base=vision_position_span,
                )
            )
            target_chunk = 2
        local = torch.arange(text_indices.numel(), dtype=torch.long, device=device)
        specs.extend(
            _target_text_specs(
                row=row,
                sequence_length=sequence_length,
                clean_start=clean_start,
                noisy_start=noisy_start,
                token_indices=text_indices,
                local_positions=local,
                sequence_id=0,
                chunk_index=target_chunk,
                block_size=text_block_size,
                position_base=vision_position_span + int(prompt_indices.numel()),
            )
        )
        return tuple(specs)
    raise ValueError("unsupported MF branch-role combination")


def _chunk_visibility(
    sequence: Tensor,
    chunk: Tensor,
    modality: Tensor,
    view: Tensor,
    block: Tensor,
    target: Tensor,
    query: Tensor,
    key: Tensor,
) -> Tensor:
    same_sequence = sequence[query] == sequence[key]
    qt, kt = target[query], target[key]
    qe, ke = chunk[query], chunk[key]
    qv, kv = view[query], view[key]
    qb, kb = block[query], block[key]
    context = ~qt & ~kt & (ke <= qe)
    clean = qt & (qv == int(ChunkTokenView.CONTENT)) & (
        ~kt
        | (
            kt
            & (ke == qe)
            & (kv == int(ChunkTokenView.CONTENT))
            & (kb <= qb)
        )
    )
    noisy = (
        qt
        & (qv == int(ChunkTokenView.NOISY_TEXT))
        & (
            ~kt
            | (
                kt
                & (
                    (ke == qe)
                    & (
                        ((kv == int(ChunkTokenView.CONTENT)) & (kb < qb))
                        | ((kv == int(ChunkTokenView.NOISY_TEXT)) & (kb == qb))
                    )
                )
            )
        )
    )
    return same_sequence & (context | clean | noisy)


def build_chunk_flex_prewarm_mask(
    device: torch.device,
    *,
    sequence_length: int,
    kernel_block_size: int,
    text_block_size: int,
    flex_backend: str = "triton",
) -> object:
    """Prewarm Flex with a production-shaped noisy text chunk."""

    from torch.nn.attention.flex_attention import create_block_mask

    if (
        type(sequence_length) is not int
        or sequence_length <= 0
        or sequence_length % kernel_block_size
    ):
        raise ValueError("prewarm length must be a positive kernel-block multiple")
    if type(text_block_size) is not int or text_block_size <= 0:
        raise ValueError("text block size must be positive")

    offsets = torch.arange(sequence_length, device=device)
    row = torch.zeros_like(offsets)
    sequence = torch.zeros_like(offsets)
    chunk = torch.zeros_like(offsets)
    modality = torch.full_like(offsets, int(ChunkModality.TEXT))
    view = torch.full_like(offsets, int(ChunkTokenView.NOISY_TEXT))
    block = torch.div(offsets, text_block_size, rounding_mode="floor")
    target = torch.ones_like(offsets, dtype=torch.bool)
    mark_flex_mask_metadata_static(
        flex_backend, row, sequence, chunk, modality, view, block, target
    )

    def mask_mod(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
        del batch, head
        return (row[query] == row[key]) & _chunk_visibility(
            sequence, chunk, modality, view, block, target, query, key
        )

    return create_block_mask(
        mask_mod,
        1,
        None,
        sequence_length,
        sequence_length,
        device,
        flex_mask_block_size(kernel_block_size, flex_backend),
    )


def _attach_flex(
    layout: ChunkCausalRoutingLayout,
    *,
    kernel_block_size: int,
    sequence_bucket_size: int,
    fixed_sequence_length: bool,
    flex_backend: str,
) -> ChunkCausalRoutingLayout:
    from torch.nn.attention.flex_attention import create_block_mask

    plan = layout.sequence_plan
    packed = plan.packed_routing_layout
    active = packed.active_indices
    row = torch.div(active, packed.sequence_length, rounding_mode="floor")
    metadata = [
        dense.reshape(-1).index_select(0, active)
        for dense in (
            plan.dense_sequence_ids,
            plan.dense_chunk_indices,
            plan.dense_modality_ids,
            plan.dense_view_ids,
            plan.dense_block_indices,
            plan.dense_target_mask,
        )
    ]
    token_count = active.numel()
    if fixed_sequence_length:
        if token_count > sequence_bucket_size:
            raise ValueError(
                "packed chunk sequence exceeds fixed Flex length "
                f"({token_count} > {sequence_bucket_size})"
            )
        padded = sequence_bucket_size
    else:
        padded = ((token_count + sequence_bucket_size - 1) // sequence_bucket_size) * (
            sequence_bucket_size
        )
    dummy_count = padded - token_count
    if dummy_count:
        offsets = torch.arange(dummy_count, device=active.device)
        row = torch.cat(
            (row, -(torch.div(offsets, kernel_block_size, rounding_mode="floor") + 1))
        )
        defaults = (
            0,
            0,
            int(ChunkModality.IMAGE),
            int(ChunkTokenView.CONTENT),
            0,
            True,
        )
        metadata = [
            torch.cat(
                (
                    values,
                    torch.full(
                        (dummy_count,), fill, dtype=values.dtype, device=values.device
                    ),
                )
            )
            for values, fill in zip(metadata, defaults, strict=True)
        ]
    sequence, chunk, modality, view, block, target = metadata
    mark_flex_mask_metadata_static(
        flex_backend, row, sequence, chunk, modality, view, block, target
    )

    def mask_mod(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
        del batch, head
        return (row[query] == row[key]) & _chunk_visibility(
            sequence, chunk, modality, view, block, target, query, key
        )

    mask = create_block_mask(
        mask_mod,
        1,
        None,
        padded,
        padded,
        active.device,
        flex_mask_block_size(kernel_block_size, flex_backend),
    )
    packed = replace(
        packed,
        flex_block_mask=mask,
        flex_sequence_length=padded,
    )
    return replace(layout, sequence_plan=replace(plan, packed_routing_layout=packed))


def _move_layout_to_device(
    layout: ChunkCausalRoutingLayout,
    *,
    device: torch.device,
    cache_identity: tuple[Tensor, ...],
) -> ChunkCausalRoutingLayout:
    moved: dict[int, Tensor] = {}

    def move(tensor: Tensor) -> Tensor:
        cached = moved.get(id(tensor))
        if cached is None:
            cached = tensor.to(device=device, non_blocking=True)
            moved[id(tensor)] = cached
        return cached

    plan = layout.sequence_plan
    packed = plan.packed_routing_layout
    packed = replace(
        packed,
        positions=move(packed.positions),
        modality_ids=move(packed.modality_ids),
        cu_seqlens=move(packed.cu_seqlens),
        active_indices=move(packed.active_indices),
        active_mask=move(packed.active_mask),
        dense_modality_ids=move(packed.dense_modality_ids),
    )
    plan = replace(
        plan,
        active_token_mask=move(plan.active_token_mask),
        dense_sequence_ids=move(plan.dense_sequence_ids),
        dense_chunk_indices=move(plan.dense_chunk_indices),
        dense_modality_ids=move(plan.dense_modality_ids),
        dense_view_ids=move(plan.dense_view_ids),
        dense_block_indices=move(plan.dense_block_indices),
        dense_target_mask=move(plan.dense_target_mask),
        position_ids=move(plan.position_ids),
        packed_routing_layout=packed,
    )
    return ChunkCausalRoutingLayout(plan, cache_identity)


def _build_chunk_active_mask(
    rows: tuple[tuple[ChunkCausalBlockSpec, ...], ...],
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> Tensor:
    active = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)
    for row in rows:
        if not row:
            continue
        # Filter per spec to preserve indexing errors, then batch the scatter by row.
        source = torch.cat(tuple(spec.source_indices[spec.active_mask] for spec in row))
        local_row = torch.div(source, sequence_length, rounding_mode="floor")
        local = source - local_row * sequence_length
        active[local_row, local] = True
    return active


def _compile_chunk_metadata(
    routing: ChunkRoutingMetadata | MFModelInput,
    *,
    sequence_length: int,
    text_block_size: int,
    hidden_size: int,
    attention_backend: str,
    image_chunk_conditioning: ImageChunkConditioning,
    t2i_chunk_semantics: T2IChunkSemantics,
    cache_identity: tuple[Tensor, ...] = (),
) -> ChunkCausalRoutingLayout:
    geometry = routing.geometry or GeometryContract()
    geometry.validate()
    prompt_tokens = routing.text_prompt_content_mask.shape[1]
    prompt_layout_tokens = TEXT_PREFIX_TOKENS + prompt_tokens if prompt_tokens else 0
    text_prefix_start = geometry.vision_layout_tokens + prompt_layout_tokens
    clean_start = text_prefix_start + TEXT_PREFIX_TOKENS
    rows = tuple(
        _row_specs(
            routing,
            row,
            sequence_length=sequence_length,
            clean_start=clean_start,
            noisy_start=clean_start + routing.text_content_mask.shape[1],
            prompt_content_start=geometry.vision_layout_tokens + TEXT_PREFIX_TOKENS,
            text_prefix_start=text_prefix_start,
            text_block_size=text_block_size,
            image_chunk_conditioning=image_chunk_conditioning,
            t2i_chunk_semantics=t2i_chunk_semantics,
        )
        for row in range(routing.vision_role.shape[0])
    )
    null_conditioning = getattr(routing, "null_conditioning", None)
    if null_conditioning is not None:
        if null_conditioning.shape != (routing.vision_role.shape[0],):
            raise ValueError("null_conditioning must have shape [B]")
        rows = tuple(
            tuple(
                spec
                for spec in row_specs
                if not (
                    bool(null_conditioning[row_index].item())
                    and spec.role is BranchRole.CONDITION
                )
            )
            for row_index, row_specs in enumerate(rows)
        )
    active = _build_chunk_active_mask(
        rows,
        batch_size=routing.vision_role.shape[0],
        sequence_length=sequence_length,
        device=routing.vision_role.device,
    )
    compiled_sequences = routing.compiled_sequences
    return compile_chunk_causal_sequence(
        rows,
        active_token_mask=active,
        hidden_size=hidden_size,
        cache_identity=cache_identity,
        compiled_sequences=compiled_sequences,
        allow_omitted_condition_chunks=(
            null_conditioning is not None and bool(null_conditioning.any())
        ),
    )


def _routing_inputs(routing: ChunkRoutingMetadata) -> tuple[Tensor, ...]:
    return (
        routing.vision_role,
        routing.text_role,
        routing.text_content_mask,
        routing.text_prompt_content_mask,
        routing.text_segment_ids,
        routing.null_conditioning,
    )


def _geometry_signature(geometry: GeometryContract) -> tuple[int, int, int, int, int]:
    return (
        geometry.vision_tokens,
        geometry.vision_latent_dim,
        geometry.text_latent_dim,
        geometry.vision_grid_size[0],
        geometry.vision_grid_size[1],
    )


@dataclass(frozen=True, kw_only=True)
class PreparedChunkRouting(ChunkRoutingMetadata):
    """One CPU-only compiled plan, rebound to current device tensors at consumption."""

    layout: ChunkCausalRoutingLayout
    policy: tuple[object, ...]
    source_tensors: tuple[Tensor | None, ...]
    source_versions: tuple[int | None, ...]

    def matches(self, policy: tuple[object, ...]) -> bool:
        return self.policy == policy and all(
            actual is expected
            and version is not None
            and tensor_mutation_version(actual) == version
            for actual, expected, version in zip(
                _routing_inputs(self),
                self.source_tensors,
                self.source_versions,
                strict=True,
            )
        )


def prepare_chunk_routing_cpu(
    routing: ChunkRoutingMetadata,
    *,
    text_block_size: int,
    hidden_size: int,
    attention_backend: str,
    image_chunk_conditioning: ImageChunkConditioning = "token_additive",
    t2i_chunk_semantics: T2IChunkSemantics = "chunk_native",
) -> PreparedChunkRouting:
    routing.validate()
    geometry = routing.geometry or GeometryContract()
    geometry.validate()
    prompt_tokens = routing.text_prompt_content_mask.shape[1]
    length = (
        geometry.vision_layout_tokens
        + TEXT_PREFIX_TOKENS
        + (TEXT_PREFIX_TOKENS + prompt_tokens if prompt_tokens else 0)
        + 2 * routing.text_content_mask.shape[1]
    )
    layout = _compile_chunk_metadata(
        routing,
        sequence_length=length,
        text_block_size=text_block_size,
        hidden_size=hidden_size,
        attention_backend=attention_backend,
        image_chunk_conditioning=image_chunk_conditioning,
        t2i_chunk_semantics=t2i_chunk_semantics,
    )
    inputs = _routing_inputs(routing)
    return PreparedChunkRouting(
        vision_role=routing.vision_role,
        text_role=routing.text_role,
        text_content_mask=routing.text_content_mask,
        text_prompt_content_mask=routing.text_prompt_content_mask,
        text_segment_ids=routing.text_segment_ids,
        sequence_contracts=routing.sequence_contracts,
        compiled_sequences=layout.sequence_plan.compiled_sequences,
        geometry=geometry,
        null_conditioning=routing.null_conditioning,
        layout=layout,
        policy=(
            length,
            text_block_size,
            hidden_size,
            attention_backend,
            image_chunk_conditioning,
            t2i_chunk_semantics,
            _geometry_signature(geometry),
        ),
        source_tensors=inputs,
        source_versions=tuple(tensor_mutation_version(value) for value in inputs),
    )


def build_chunk_causal_layout(
    model_input: MFModelInput,
    *,
    hidden_size: int,
    attention_backend: str,
    flex_kernel_block_size: int,
    flex_sequence_bucket_size: int,
    flex_backend: str = "triton",
    flex_fixed_sequence_length: bool = False,
    image_chunk_conditioning: ImageChunkConditioning = "token_additive",
    t2i_chunk_semantics: T2IChunkSemantics = "chunk_native",
) -> ChunkCausalRoutingLayout:
    """Adapt the current mixed pretrain batch to the role-free MF compiler."""

    if model_input.physical_layout is not None:
        model_input.validate_metadata()
        physical = model_input.physical_layout
        cache_identity = (
            physical.active_token_mask,
            physical.position_ids,
            physical.sequence_ids,
            physical.chunk_indices,
            physical.modality_ids,
            physical.view_ids,
            physical.block_indices,
            physical.target_mask,
        )
        compiler = PHYSICAL_LAYOUT_REGISTRY.resolve(physical.layout_name)
        layout = compiler(
            physical,
            hidden_size=hidden_size,
            cache_identity=cache_identity,
        )
        if attention_backend != "flex":
            raise ValueError("unknown MF attention backend")
        return _attach_flex(
            layout,
            kernel_block_size=flex_kernel_block_size,
            sequence_bucket_size=flex_sequence_bucket_size,
            fixed_sequence_length=flex_fixed_sequence_length,
            flex_backend=flex_backend,
        )
    if model_input.chunk_routing_cpu is None:
        model_input.validate_metadata()
    if not model_input.block_causal_text:
        raise ValueError("MF requires block-causal text state")
    if image_chunk_conditioning not in {"token_additive", "legacy_active_prefix"}:
        raise ValueError("unknown image chunk conditioning mode")
    if t2i_chunk_semantics not in {"chunk_native", "legacy_block_exact"}:
        raise ValueError("unknown T2I chunk semantics")
    if (
        t2i_chunk_semantics == "legacy_block_exact"
        and image_chunk_conditioning != "legacy_active_prefix"
    ):
        raise ValueError(
            "legacy_block_exact T2I semantics require legacy_active_prefix image conditioning"
        )
    assert model_input.text_block_size is not None
    routing: ChunkRoutingMetadata | MFModelInput = model_input
    if model_input.chunk_routing_cpu is not None:
        routing = model_input.chunk_routing_cpu.validate()
        if model_input.null_conditioning is not None:
            routing = replace(
                routing,
                null_conditioning=model_input.null_conditioning.detach().cpu(),
            )
    geometry = routing.geometry or GeometryContract()
    geometry.validate()
    sequence_length = model_input.active_token_mask.shape[1]
    cache_identity = (
        model_input.active_token_mask,
        model_input.vision_role,
        model_input.text_role,
        model_input.text_content_mask,
        model_input.text_prompt_content_mask,
        model_input.text_segment_ids,
        model_input.null_conditioning,
    )
    policy = (
        sequence_length,
        model_input.text_block_size,
        hidden_size,
        attention_backend,
        image_chunk_conditioning,
        t2i_chunk_semantics,
        _geometry_signature(geometry),
    )
    if isinstance(routing, PreparedChunkRouting) and routing.matches(policy):
        layout = routing.layout
    else:
        layout = _compile_chunk_metadata(
            routing,
            sequence_length=sequence_length,
            text_block_size=model_input.text_block_size,
            hidden_size=hidden_size,
            attention_backend=attention_backend,
            image_chunk_conditioning=image_chunk_conditioning,
            t2i_chunk_semantics=t2i_chunk_semantics,
            cache_identity=cache_identity if routing is model_input else (),
        )
    if routing is not model_input:
        layout = _move_layout_to_device(
            layout,
            device=model_input.active_token_mask.device,
            cache_identity=cache_identity,
        )
    if attention_backend != "flex":
        raise ValueError("unknown MF attention backend")
    return _attach_flex(
        layout,
        kernel_block_size=flex_kernel_block_size,
        sequence_bucket_size=flex_sequence_bucket_size,
        fixed_sequence_length=flex_fixed_sequence_length,
        flex_backend=flex_backend,
    )
