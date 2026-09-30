from __future__ import annotations

import torch
from torch import Tensor

from mf.codecs.frozen import FrozenCodec

RAW_PIXEL_IMAGE_RESOLUTION = 256
RAW_PIXEL_PATCH_SIZE = 16
RAW_PIXEL_GRID_SIZE = 16
RAW_PIXEL_TOKENS = 256
RAW_PIXEL_DIM = 768


def _validate_images(images: Tensor) -> None:
    expected = (3, RAW_PIXEL_IMAGE_RESOLUTION, RAW_PIXEL_IMAGE_RESOLUTION)
    if images.ndim != 4 or tuple(images.shape[1:]) != expected:
        raise ValueError(
            f"raw-pixel images must have shape [B, 3, 256, 256]; got {list(images.shape)}"
        )
    if not torch.is_floating_point(images):
        raise ValueError("raw-pixel images must have a floating-point dtype")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("raw-pixel images must be finite")
    if bool(((images < 0) | (images > 1)).any()):
        raise ValueError("raw-pixel images must be in [0, 1]")


class RawPixelEncoder(FrozenCodec):
    """Deterministic 16x16 RGB patchification in the [-1, 1] pixel space."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.freeze()

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        _validate_images(images)
        pixels = images.mul(2.0).sub(1.0)
        patches = pixels.unfold(2, RAW_PIXEL_PATCH_SIZE, RAW_PIXEL_PATCH_SIZE).unfold(
            3, RAW_PIXEL_PATCH_SIZE, RAW_PIXEL_PATCH_SIZE
        )
        tokens = patches.permute(0, 2, 3, 4, 5, 1).reshape(
            images.shape[0], RAW_PIXEL_TOKENS, RAW_PIXEL_DIM
        )
        return tokens.contiguous()


class RawPixelDecoder(FrozenCodec):
    """Inverse raw-RGB patchification used for sampling and evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.freeze()

    @torch.no_grad()
    def decode(self, tokens: Tensor) -> Tensor:
        expected = (RAW_PIXEL_TOKENS, RAW_PIXEL_DIM)
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(
                f"raw-pixel tokens must have shape [B, 256, 768]; got {list(tokens.shape)}"
            )
        if not torch.is_floating_point(tokens):
            raise ValueError("raw-pixel tokens must have a floating-point dtype")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("raw-pixel tokens must be finite")
        patches = tokens.reshape(
            tokens.shape[0],
            RAW_PIXEL_GRID_SIZE,
            RAW_PIXEL_GRID_SIZE,
            RAW_PIXEL_PATCH_SIZE,
            RAW_PIXEL_PATCH_SIZE,
            3,
        )
        pixels = patches.permute(0, 5, 1, 3, 2, 4).reshape(
            tokens.shape[0], 3, RAW_PIXEL_IMAGE_RESOLUTION, RAW_PIXEL_IMAGE_RESOLUTION
        )
        return pixels.add(1.0).mul(0.5).clamp_(0.0, 1.0).contiguous()
