from __future__ import annotations

import torch
from torch import Tensor, nn

from mf.contracts.batch import (
    MODALITY_TOKENS,
    TEXT_PREFIX_TOKENS,
    TIME_TOKENS,
    VISION_PREFIX_TOKENS,
    VISION_TOKENS,
)

MROPE_SECTION = (8, 12, 12)
ROPE_THETA = 10_000.0
VISION_GRID_SIZE = 16
VISION_PREFIX_POSITION_GROUPS = 2
TEXT_PREFIX_POSITION_GROUPS = 3

_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


def _build_interleaved_frequency_axes(
    mrope_section: tuple[int, int, int],
) -> tuple[int, ...]:
    """Generalize Qwen3-VL's round-robin M/Y/X frequency routing.

    Qwen3-VL v4.57.1 cycles M/Y/X until Y and X exhaust, then assigns its
    larger M remainder. MF applies the same skip-exhausted rule to (8,12,12),
    yielding M=[0,3,6,9,12,15,18,21],
    Y=[1,4,7,10,13,16,19,22,24,26,28,30], and
    X=[2,5,8,11,14,17,20,23,25,27,29,31].
    """

    remaining = list(mrope_section)
    frequency_axes: list[int] = []
    while any(remaining):
        for axis in range(3):
            if remaining[axis] > 0:
                frequency_axes.append(axis)
                remaining[axis] -= 1
    return tuple(frequency_axes)


def _validate_segment_bases(base_positions: Tensor) -> None:
    if base_positions.ndim != 1 or base_positions.shape[0] <= 0:
        raise ValueError("base_positions must have shape [B] with B > 0")
    if base_positions.dtype not in _INTEGER_DTYPES:
        raise ValueError("base_positions must have an integer dtype")
    if bool((base_positions < 0).any()):
        raise ValueError("base_positions must be non-negative")


def build_vision_segment_mrope_positions(
    base_positions: Tensor,
    *,
    vision_tokens: int = VISION_TOKENS,
    grid_size: tuple[int, int] = (VISION_GRID_SIZE, VISION_GRID_SIZE),
) -> Tensor:
    """Build one vision segment with grouped prefix and shifted 2D-grid positions."""

    _validate_segment_bases(base_positions)
    if type(vision_tokens) is not int or vision_tokens <= 0:
        raise ValueError("vision_tokens must be a positive integer")
    if (
        type(grid_size) is not tuple
        or len(grid_size) != 2
        or any(type(value) is not int or value <= 0 for value in grid_size)
        or grid_size[0] * grid_size[1] != vision_tokens
    ):
        raise ValueError("grid_size must multiply to vision_tokens")
    batch_size = base_positions.shape[0]
    positions = torch.empty(
        (3, batch_size, VISION_PREFIX_TOKENS + vision_tokens),
        dtype=torch.long,
        device=base_positions.device,
    )
    bases = base_positions.to(dtype=torch.long)
    positions[:, :, :TIME_TOKENS] = bases[None, :, None]
    positions[:, :, TIME_TOKENS:VISION_PREFIX_TOKENS] = (bases + 1)[None, :, None]

    vision_index = torch.arange(
        vision_tokens,
        dtype=torch.long,
        device=base_positions.device,
    )
    content_base = bases + VISION_PREFIX_POSITION_GROUPS
    positions[0, :, VISION_PREFIX_TOKENS:] = content_base[:, None]
    positions[1, :, VISION_PREFIX_TOKENS:] = (
        content_base[:, None] + vision_index[None, :] // grid_size[1]
    )
    positions[2, :, VISION_PREFIX_TOKENS:] = (
        content_base[:, None] + vision_index[None, :] % grid_size[1]
    )
    return positions.contiguous()


def build_text_segment_mrope_positions(
    base_positions: Tensor,
    *,
    text_tokens: int,
) -> Tensor:
    """Build one text segment with three grouped prefix positions and 1D content."""

    _validate_segment_bases(base_positions)
    if type(text_tokens) is not int or text_tokens <= 0:
        raise ValueError("text_tokens must be a positive integer")
    batch_size = base_positions.shape[0]
    positions = torch.empty(
        (3, batch_size, TEXT_PREFIX_TOKENS + text_tokens),
        dtype=torch.long,
        device=base_positions.device,
    )
    bases = base_positions.to(dtype=torch.long)
    modality_start = TIME_TOKENS
    guidance_start = modality_start + MODALITY_TOKENS
    positions[:, :, :modality_start] = bases[None, :, None]
    positions[:, :, modality_start:guidance_start] = (bases + 1)[None, :, None]
    positions[:, :, guidance_start:TEXT_PREFIX_TOKENS] = (bases + 2)[None, :, None]

    content_positions = (
        bases[:, None]
        + TEXT_PREFIX_POSITION_GROUPS
        + torch.arange(text_tokens, dtype=torch.long, device=base_positions.device)[
            None, :
        ]
    )
    positions[:, :, TEXT_PREFIX_TOKENS:] = content_positions[None, :, :]
    return positions.contiguous()


class MRoPERotaryEmbedding(nn.Module):
    """Three-axis rotary embedding with interleaved M/Y/X frequency routing."""

    def __init__(
        self,
        head_dim: int = 64,
        mrope_section: tuple[int, int, int] = MROPE_SECTION,
        rope_theta: float = ROPE_THETA,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        if len(mrope_section) != 3 or any(section <= 0 for section in mrope_section):
            raise ValueError("mrope_section must contain three positive integers")
        if sum(mrope_section) != head_dim // 2:
            raise ValueError("sum(mrope_section) must equal head_dim / 2")

        self.head_dim = head_dim
        self.mrope_section = tuple(mrope_section)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer(
            "frequency_axes",
            torch.tensor(_build_interleaved_frequency_axes(self.mrope_section)),
            persistent=False,
        )

    def forward(
        self,
        position_ids: Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        if position_ids.ndim < 2 or position_ids.shape[0] != 3:
            raise ValueError("position_ids must have shape [3, ...]")
        if position_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("position_ids must have an integer dtype")
        if position_ids.device != self.inv_freq.device:
            raise ValueError(
                "position_ids and rotary frequencies must be on the same device"
            )

        selected_positions = (
            position_ids.to(torch.float32)
            .movedim(0, -1)
            .index_select(-1, self.frequency_axes)
        )
        selected_freqs = selected_positions * self.inv_freq
        embedding = torch.cat((selected_freqs, selected_freqs), dim=-1)
        return embedding.cos().to(dtype=dtype), embedding.sin().to(dtype=dtype)


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply precomputed MRoPE to packed query and key heads."""

    if query.shape != key.shape:
        raise ValueError("query and key must have identical shapes")
    if query.ndim != 3:
        raise ValueError("query and key must have shape [tokens, heads, head_dim]")
    if cos.shape != (query.shape[0], query.shape[-1]) or sin.shape != cos.shape:
        raise ValueError("cos and sin must have shape [tokens, head_dim]")

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + _rotate_half(query) * sin,
        key * cos + _rotate_half(key) * sin,
    )
