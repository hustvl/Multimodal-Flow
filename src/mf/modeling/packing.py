from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)

_VISION_MODALITY_ID = 0
_TEXT_MODALITY_ID = 1


@dataclass(frozen=True)
class PackedRoutingLayout:
    """Reusable GPU-resident routing metadata with no hidden-state payload."""

    positions: Tensor
    modality_ids: Tensor
    cu_seqlens: Tensor
    active_indices: Tensor
    active_mask: Tensor
    dense_modality_ids: Tensor
    batch_size: int
    sequence_length: int
    hidden_size: int
    max_seqlen: int
    flex_block_mask: object | None
    flex_sequence_length: int | None
    target_mask: Tensor | None

    @property
    def modality_indices(self) -> dict[int, Tensor]:
        """Return packed token indices grouped by canonical modality id."""

        return {
            int(modality_id): torch.nonzero(
                self.modality_ids == modality_id,
                as_tuple=False,
            ).flatten()
            for modality_id in torch.unique(self.modality_ids, sorted=True).tolist()
        }

    @property
    def vision_indices(self) -> Tensor:
        """Compatibility view for legacy image/text kernels."""

        return torch.nonzero(
            self.modality_ids == _VISION_MODALITY_ID,
            as_tuple=False,
        ).flatten()

    @property
    def text_indices(self) -> Tensor:
        """Compatibility view for legacy image/text kernels."""

        return torch.nonzero(
            self.modality_ids == _TEXT_MODALITY_ID,
            as_tuple=False,
        ).flatten()

    def gather(self, x: Tensor) -> Tensor:
        expected_shape = (self.batch_size, self.sequence_length, self.hidden_size)
        if x.shape != expected_shape:
            raise ValueError(f"x must have shape {expected_shape}")
        if x.device != self.active_indices.device:
            raise ValueError("x and packed layout must be on the same device")
        return x.reshape(-1, self.hidden_size).index_select(0, self.active_indices)

    def with_tokens(self, x: Tensor) -> PackedLayout:
        """Bind current hidden states without changing reusable routing metadata."""

        tokens = self.gather(x)
        positions = self.positions
        modality_ids = self.modality_ids
        text_indices = self.text_indices
        flex_length = self.flex_sequence_length
        if flex_length is not None and tokens.shape[0] < flex_length:
            real_token_count = tokens.shape[0]
            dummy_count = flex_length - real_token_count
            tokens = torch.cat(
                (tokens, tokens.new_zeros(dummy_count, self.hidden_size))
            )
            if positions.shape[1] != flex_length:
                positions = torch.cat(
                    (
                        positions,
                        positions.new_zeros(positions.shape[0], dummy_count),
                    ),
                    dim=1,
                )
            if modality_ids.shape[0] != flex_length:
                modality_ids = torch.cat(
                    (
                        modality_ids,
                        modality_ids.new_full((dummy_count,), _TEXT_MODALITY_ID),
                    )
                )
            if text_indices.shape[0] + self.vision_indices.shape[0] != flex_length:
                text_indices = torch.cat(
                    (
                        text_indices,
                        torch.arange(
                            real_token_count,
                            flex_length,
                            device=text_indices.device,
                            dtype=text_indices.dtype,
                        ),
                    )
                )

        return PackedLayout(
            positions=positions,
            modality_ids=modality_ids,
            cu_seqlens=self.cu_seqlens,
            active_indices=self.active_indices,
            active_mask=self.active_mask,
            dense_modality_ids=self.dense_modality_ids,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            hidden_size=self.hidden_size,
            max_seqlen=self.max_seqlen,
            flex_block_mask=self.flex_block_mask,
            flex_sequence_length=self.flex_sequence_length,
            target_mask=self.target_mask,
            tokens=tokens,
        )

    def unpack(self, packed_tokens: Tensor) -> Tensor:
        active_count = self.active_indices.shape[0]
        expected_counts = {active_count}
        if self.flex_sequence_length is not None:
            expected_counts.add(self.flex_sequence_length)
        expected_shapes = {(count, self.hidden_size) for count in expected_counts}
        if packed_tokens.ndim != 2 or packed_tokens.shape not in expected_shapes:
            expected = ", ".join(str(shape) for shape in sorted(expected_shapes))
            raise ValueError(f"packed_tokens must have shape {expected}")
        if packed_tokens.device != self.active_indices.device:
            raise ValueError(
                "packed_tokens and packed layout must be on the same device"
            )

        packed_tokens = packed_tokens[:active_count]

        flat = packed_tokens.new_zeros(
            self.batch_size * self.sequence_length,
            self.hidden_size,
        )
        flat = torch.index_copy(flat, 0, self.active_indices, packed_tokens)
        return flat.view(self.batch_size, self.sequence_length, self.hidden_size)


@dataclass(frozen=True)
class PackedLayout(PackedRoutingLayout):
    """Current packed hidden states bound to reusable routing metadata."""

    tokens: Tensor

    def with_tokens(self, x: Tensor) -> PackedLayout:
        return replace(self, tokens=self.gather(x))


def _validate_routing_inputs(
    mask: Tensor,
    positions: Tensor,
    modality_ids: Tensor,
    hidden_size: int,
) -> tuple[int, int]:
    if mask.ndim != 2:
        raise ValueError("mask must have shape [B, L]")
    batch_size, sequence_length = mask.shape
    if mask.dtype is not torch.bool:
        raise ValueError("mask must have dtype torch.bool")
    if positions.shape != (3, batch_size, sequence_length):
        raise ValueError("positions must have shape [3, B, L]")
    if positions.dtype not in _INTEGER_DTYPES:
        raise ValueError("positions must have an integer dtype")
    if modality_ids.shape != (batch_size, sequence_length):
        raise ValueError("modality_ids must have shape [B, L]")
    if modality_ids.dtype not in _INTEGER_DTYPES:
        raise ValueError("modality_ids must have an integer dtype")
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size <= 0
    ):
        raise ValueError("hidden_size must be a positive integer")
    for name, tensor in (
        ("positions", positions),
        ("modality_ids", modality_ids),
    ):
        if tensor.device != mask.device:
            raise ValueError(f"{name} must be on the same device as mask")
    return batch_size, sequence_length


def _build_packed_routing_layout_from_indices(
    mask: Tensor,
    positions: Tensor,
    modality_ids: Tensor,
    active_indices: Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    target_mask: Tensor | None = None,
) -> PackedRoutingLayout:
    packed_positions = positions.reshape(3, -1).index_select(1, active_indices)
    packed_modality_ids = modality_ids.reshape(-1).index_select(0, active_indices)
    packed_target_mask = (
        None
        if target_mask is None
        else target_mask.reshape(-1).index_select(0, active_indices)
    )
    sequence_lengths = mask.sum(dim=1, dtype=torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device=mask.device)
    cu_seqlens = torch.cat(
        (zero, torch.cumsum(sequence_lengths, dim=0, dtype=torch.int32)),
        dim=0,
    )

    return PackedRoutingLayout(
        positions=packed_positions,
        modality_ids=packed_modality_ids,
        cu_seqlens=cu_seqlens,
        active_indices=active_indices,
        active_mask=mask,
        dense_modality_ids=modality_ids,
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        max_seqlen=sequence_length,
        flex_block_mask=None,
        flex_sequence_length=None,
        target_mask=packed_target_mask,
    )


def build_ordered_packed_routing_layout(
    mask: Tensor,
    positions: Tensor,
    modality_ids: Tensor,
    ordered_active_indices: Tensor,
    *,
    hidden_size: int,
    target_mask: Tensor | None = None,
) -> PackedRoutingLayout:
    """Build packed routing from an explicit within-sample token order."""

    batch_size, sequence_length = _validate_routing_inputs(
        mask,
        positions,
        modality_ids,
        hidden_size,
    )
    if ordered_active_indices.ndim != 1:
        raise ValueError("ordered_active_indices must have shape [N]")
    if ordered_active_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("ordered_active_indices must have an integer index dtype")
    if ordered_active_indices.device != mask.device:
        raise ValueError("ordered_active_indices must be on the same device as mask")

    expected_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    if ordered_active_indices.numel() != expected_indices.numel() or not torch.equal(
        torch.sort(ordered_active_indices).values, expected_indices
    ):
        raise ValueError(
            "ordered_active_indices must partition active tokens exactly once"
        )

    if ordered_active_indices.numel() > 1:
        sample_ids = torch.div(
            ordered_active_indices,
            sequence_length,
            rounding_mode="floor",
        )
        if bool(torch.any(sample_ids[1:] < sample_ids[:-1])):
            raise ValueError("ordered_active_indices must remain grouped by sample")

    return _build_packed_routing_layout_from_indices(
        mask,
        positions,
        modality_ids,
        ordered_active_indices,
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        target_mask=target_mask,
    )
