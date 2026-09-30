"""Attention backend selection and optional flash-attention adapters."""

from __future__ import annotations

from typing import Literal

from torch import Tensor


FlexBackend = Literal["triton", "fa4"]


def validate_flex_backend(backend: str) -> None:
    if backend not in ("triton", "fa4"):
        raise ValueError("flex_backend must be triton or fa4")


def mark_flex_mask_metadata_static(backend: FlexBackend, *metadata: Tensor) -> None:
    """Specialize bucketed mask lengths without specializing ragged FFN routes."""
    if backend != "fa4":
        return
    import torch

    # Dynamic metadata shapes otherwise become captured SymInts in mask_mod,
    # which FLASH's CuTeDSL lowering rejects even when the scalar is unused.
    for tensor in metadata:
        torch._dynamo.mark_static(tensor, 0)


def flex_mask_block_size(
    kernel_block_size: int, backend: FlexBackend
) -> int | tuple[int, int]:
    if backend == "fa4":
        if kernel_block_size != 128:
            raise ValueError("FA4 requires a 128-token KV kernel block")
        # FA4's two query tiles are physical tiling, not the semantic text block.
        return (256, 128)
    return kernel_block_size
