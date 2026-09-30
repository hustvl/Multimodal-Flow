from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from mf.config.schema import MFConfig, ScalarTextStatsConfig, TensorStatsConfig
from mf.latents.stats import LatentStatsRegistry


def _load_stats_file(
    path: str, cache: dict[Path, Mapping[str, object]]
) -> Mapping[str, object]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved not in cache:
        payload = torch.load(resolved, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(payload, Mapping):
            raise TypeError(f"latent stats file must contain a mapping: {resolved}")
        cache[resolved] = payload
    return cache[resolved]


def _stats_tensor(
    config: TensorStatsConfig,
    key: str,
    *,
    cache: dict[Path, Mapping[str, object]],
) -> Tensor:
    payload = _load_stats_file(config.path, cache)
    if key not in payload:
        raise KeyError(f"latent stats key {key!r} is missing from {config.path}")
    value = payload[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"latent stats key {key!r} must contain a tensor")
    if tuple(value.shape) != config.expected_shape:
        raise ValueError(
            f"latent stats key {key!r} must have shape {list(config.expected_shape)}; "
            f"got {list(value.shape)}"
        )
    return value.detach().to(dtype=torch.float32, device="cpu")


def _text_stats_tensors(
    config: TensorStatsConfig | ScalarTextStatsConfig,
    *,
    latent_dim: int,
    cache: dict[Path, Mapping[str, object]],
) -> tuple[Tensor, Tensor]:
    if isinstance(config, TensorStatsConfig):
        return (
            _stats_tensor(config, config.mean_key, cache=cache),
            _stats_tensor(config, config.std_key, cache=cache),
        )
    return (
        torch.full((latent_dim,), config.mean, dtype=torch.float32),
        torch.full((latent_dim,), config.std, dtype=torch.float32),
    )


def load_latent_stats_registry(config: MFConfig) -> LatentStatsRegistry:
    """Load the configured vision and shared text statistics."""

    if not isinstance(config, MFConfig):
        raise TypeError("config must be a MFConfig")
    cache: dict[Path, Mapping[str, object]] = {}
    vision = config.latent_stats.vision
    normal = config.latent_stats.text_normal
    text_mean, text_std = _text_stats_tensors(
        normal,
        latent_dim=config.codecs.text.latent_dim,
        cache=cache,
    )
    vision_mean = _stats_tensor(vision, vision.mean_key, cache=cache)
    vision_std = _stats_tensor(vision, vision.std_key, cache=cache)
    expected_vision_shape = (
        config.codecs.vision.latent_tokens,
        config.codecs.vision.latent_dim,
    )
    if tuple(vision_mean.shape) != expected_vision_shape or tuple(vision_std.shape) != expected_vision_shape:
        raise ValueError(
            "vision latent stats must match the configured codec geometry; "
            f"expected {list(expected_vision_shape)}"
        )
    if tuple(text_mean.shape) != (config.codecs.text.latent_dim,):
        raise ValueError("text latent stats must match the configured codec dimension")
    return LatentStatsRegistry(
        vision_mean=vision_mean,
        vision_std=vision_std,
        text_normal_mean=text_mean,
        text_normal_std=text_std,
    )


def build_train_runtime(*, config: MFConfig, args: argparse.Namespace):
    from mf.runtime import build_train_runtime as build

    return build(config=config, args=args)
