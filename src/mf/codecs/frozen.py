from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch import nn

from mf._compat import Self


def preprocessing_buffer(values: Sequence[float] | Tensor, *, name: str) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tuple(tensor.shape) != (3,):
        raise ValueError(f"{name} must contain exactly three channel values")
    if name == "image_std" and bool((tensor <= 0).any()):
        raise ValueError("image_std values must be positive")
    return tensor.view(1, 3, 1, 1)


def require_local_encoder(
    path: Path, *, name: str, require_preprocessor: bool = True
) -> None:
    required = (
        ("config.json", "preprocessor_config.json")
        if require_preprocessor
        else ("config.json",)
    )
    has_weights = (path / "model.safetensors").is_file() or (
        path / "pytorch_model.bin"
    ).is_file()
    if (
        not path.is_dir()
        or not all((path / item).is_file() for item in required)
        or not has_weights
    ):
        raise FileNotFoundError(f"{name} encoder assets are incomplete at {path}")


class FrozenCodec(nn.Module):
    """An inference-only module that cannot be switched back to training mode."""

    def freeze(self) -> None:
        self.requires_grad_(False)
        nn.Module.train(self, False)

    def train(self, mode: bool = True) -> Self:
        nn.Module.train(self, False)
        return self
