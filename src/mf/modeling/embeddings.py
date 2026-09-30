from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mf.contracts.batch import (
    MODALITY_TOKENS,
    TEXT_LATENT_DIM,
    TEXT_TOKENS,
    TIME_TOKENS,
    VISION_TOKENS,
)
from mf.modeling.layers import make_linear, normal_002_

VISION_GRID_SIZE = 16


def _sincos_1d(embedding_size: int, positions: Tensor) -> Tensor:
    half_size = embedding_size // 2
    frequencies = torch.arange(half_size, dtype=torch.float32, device=positions.device)
    frequencies = 1.0 / (10_000.0 ** (frequencies / half_size))
    phases = positions.to(torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat((phases.sin(), phases.cos()), dim=1)


def build_vision_sincos_2d(
    hidden_size: int = 1024,
    grid_size: tuple[int, int] = (VISION_GRID_SIZE, VISION_GRID_SIZE),
) -> Tensor:
    """Build a row/column sin-cos table for a resolved vision grid."""

    if hidden_size % 4 != 0:
        raise ValueError("hidden_size must be divisible by 4 for 2D sincos")
    height, width = grid_size
    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise ValueError("grid_size must contain positive integers")
    coordinates_h = torch.arange(height, dtype=torch.float32)
    coordinates_w = torch.arange(width, dtype=torch.float32)
    rows, columns = torch.meshgrid(coordinates_h, coordinates_w, indexing="ij")
    axis_size = hidden_size // 2
    return torch.cat(
        (
            _sincos_1d(axis_size, rows.reshape(-1)),
            _sincos_1d(axis_size, columns.reshape(-1)),
        ),
        dim=1,
    )


class TimestepEmbedder(nn.Module):
    """Embed one scalar with a shared sinusoidal MLP."""

    def __init__(
        self,
        hidden_size: int = 1024,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        if frequency_embedding_size <= 0:
            raise ValueError("frequency_embedding_size must be positive")
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_0 = make_linear(
            frequency_embedding_size,
            hidden_size,
            weight_initializer=normal_002_,
        )
        self.mlp_2 = make_linear(
            hidden_size,
            hidden_size,
            weight_initializer=normal_002_,
        )

    @staticmethod
    def timestep_embedding(
        clean_t: Tensor,
        dim: int,
        max_period: int = 10_000,
    ) -> Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=clean_t.device)
            / half
        )
        phases = clean_t[:, None].to(torch.float32) * frequencies[None]
        embedding = torch.cat((phases.cos(), phases.sin()), dim=-1)
        if dim % 2:
            embedding = torch.cat(
                (embedding, torch.zeros_like(embedding[:, :1])),
                dim=-1,
            )
        return embedding

    def forward(self, clean_t: Tensor) -> Tensor:
        if clean_t.ndim != 1 or not torch.is_floating_point(clean_t):
            raise ValueError("clean_t must be a floating-point tensor with shape [B]")
        embedding = self.timestep_embedding(
            clean_t,
            self.frequency_embedding_size,
        )
        embedding = embedding.to(dtype=self.mlp_0.weight.dtype)
        return self.mlp_2(F.silu(self.mlp_0(embedding)))


class TransfusionVisionEmbeddings(nn.Module):
    """Shared Wave 6 image and BOI/EOI embedding contract."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vision_tokens: int,
        vision_latent_dim: int,
        include_boundary_embedding: bool = True,
    ) -> None:
        super().__init__()
        if vision_tokens <= 0 or vision_latent_dim <= 0:
            raise ValueError("vision shape must be positive")
        self.hidden_size = hidden_size
        self.vision_tokens = vision_tokens
        self.vision_latent_dim = vision_latent_dim
        self.vision_input_proj = make_linear(vision_latent_dim, hidden_size)
        self.vision_timestep_embedder = TimestepEmbedder(hidden_size)
        self.boundary_embedding = (
            nn.Embedding(2, hidden_size) if include_boundary_embedding else None
        )
        if self.boundary_embedding is not None:
            nn.init.normal_(self.boundary_embedding.weight, mean=0.0, std=0.02)
        if hidden_size % 4 == 0:
            side = int(vision_tokens**0.5)
            if side * side == vision_tokens:
                position = build_vision_sincos_2d(hidden_size, (side, side))
            else:
                position = torch.zeros(vision_tokens, hidden_size)
        else:
            position = torch.zeros(vision_tokens, hidden_size)
        self.register_buffer("vision_position", position, persistent=False)

    def embed_vision(self, latents: Tensor, clean_t: Tensor) -> Tensor:
        if latents.ndim != 3 or latents.shape[1:] != (
            self.vision_tokens,
            self.vision_latent_dim,
        ):
            raise ValueError(
                "vision latents must have shape "
                f"[B, {self.vision_tokens}, {self.vision_latent_dim}]"
            )
        if clean_t.shape != latents.shape[:1]:
            raise ValueError("vision timestep must have shape [B]")
        projected = self.vision_input_proj(latents)
        projected = projected + self.vision_position.to(dtype=projected.dtype)
        return projected + self.vision_timestep_embedder(clean_t).unsqueeze(1)


class MFInputEmbeddings(nn.Module):
    """MF latent projection with modality-specific clean prefixes."""

    def __init__(
        self,
        hidden_size: int = 1024,
        *,
        vision_latent_dim: int = 768,
        vision_tokens: int = VISION_TOKENS,
        vision_grid_size: tuple[int, int] = (VISION_GRID_SIZE, VISION_GRID_SIZE),
        text_latent_dim: int = TEXT_LATENT_DIM,
        text_tokens: int = TEXT_TOKENS,
        text_input_bottleneck_dim: int = 128,
        text_input_projection_mode: Literal["bottleneck", "linear"] = "bottleneck",
        fp32_boundaries: bool = False,
    ) -> None:
        super().__init__()
        if type(vision_latent_dim) is not int or vision_latent_dim <= 0:
            raise ValueError("vision_latent_dim must be a positive integer")
        if type(vision_tokens) is not int or vision_tokens <= 0:
            raise ValueError("vision_tokens must be a positive integer")
        if vision_tokens != vision_grid_size[0] * vision_grid_size[1]:
            raise ValueError("vision_tokens must match vision_grid_size")
        if type(text_latent_dim) is not int or text_latent_dim <= 0:
            raise ValueError("text_latent_dim must be a positive integer")
        if type(text_tokens) is not int or text_tokens <= 0:
            raise ValueError("text_tokens must be a positive integer")
        if type(text_input_bottleneck_dim) is not int or text_input_bottleneck_dim <= 0:
            raise ValueError("text_input_bottleneck_dim must be a positive integer")
        if text_input_projection_mode not in ("bottleneck", "linear"):
            raise ValueError(
                "text_input_projection_mode must be 'bottleneck' or 'linear'"
            )
        if type(fp32_boundaries) is not bool:
            raise ValueError("fp32_boundaries must be a Python bool")

        self.hidden_size = hidden_size
        self.vision_latent_dim = vision_latent_dim
        self.vision_tokens = vision_tokens
        self.vision_grid_size = tuple(vision_grid_size)
        self.text_latent_dim = text_latent_dim
        self.text_tokens = text_tokens
        self.text_input_bottleneck_dim = text_input_bottleneck_dim
        self.text_input_projection_mode = text_input_projection_mode
        self.fp32_boundaries = fp32_boundaries
        self.vision_input_proj = make_linear(vision_latent_dim, hidden_size)
        if text_input_projection_mode == "bottleneck":
            self.text_input_proj = nn.Sequential(
                make_linear(text_latent_dim, text_input_bottleneck_dim, bias=False),
                make_linear(text_input_bottleneck_dim, hidden_size),
            )
        else:
            self.text_input_proj = make_linear(text_latent_dim, hidden_size)

        self.text_state_proj = make_linear(2 * text_latent_dim, text_latent_dim)
        self.timestep_embedder = TimestepEmbedder(hidden_size)
        self.learned_time = nn.Parameter(torch.empty(1, TIME_TOKENS, hidden_size))
        self.vision_modality_tokens = nn.Parameter(
            torch.empty(1, MODALITY_TOKENS, hidden_size)
        )
        self.text_modality_tokens = nn.Parameter(
            torch.empty(1, MODALITY_TOKENS, hidden_size)
        )
        self.register_buffer(
            "vision_sincos_2d",
            build_vision_sincos_2d(hidden_size, self.vision_grid_size),
        )
        nn.init.normal_(self.learned_time, std=0.02)
        nn.init.normal_(self.vision_modality_tokens, std=0.02)
        nn.init.normal_(self.text_modality_tokens, std=0.02)

    def _time_tokens(self, clean_t: Tensor) -> Tensor:
        time_embedding = self.timestep_embedder(clean_t)
        learned = self.learned_time.expand(clean_t.shape[0], -1, -1)
        return learned + time_embedding.unsqueeze(1)

    def _vision_prefix(self, clean_t: Tensor) -> Tensor:
        return torch.cat(
            (
                self._time_tokens(clean_t),
                self.vision_modality_tokens.expand(clean_t.shape[0], -1, -1),
            ),
            dim=1,
        )

    def _text_prefix(self, clean_t: Tensor) -> Tensor:
        guidance = self.learned_text_guidance.expand(clean_t.shape[0], -1, -1)
        return torch.cat(
            (
                self._time_tokens(clean_t),
                self.text_modality_tokens.expand(clean_t.shape[0], -1, -1),
                guidance,
            ),
            dim=1,
        )

    def embed_text_prefix(self, clean_t: Tensor) -> Tensor:
        """Embed the structural text prefix without guidance injection."""

        return self._text_prefix(clean_t)

    def embed_text_content(
        self,
        text_latents_norm: Tensor,
        *,
        previous_x0_norm: Tensor,
    ) -> Tensor:
        """Project current/previous text state without adding prefix tokens."""

        if (
            text_latents_norm.ndim != 3
            or text_latents_norm.shape[-1] != self.text_latent_dim
        ):
            raise ValueError(
                f"text_latents_norm must have shape [B, T, {self.text_latent_dim}]"
            )
        if previous_x0_norm.shape != text_latents_norm.shape:
            raise ValueError("previous_x0_norm must match text_latents_norm")
        if previous_x0_norm.device != text_latents_norm.device:
            raise ValueError(
                "previous_x0_norm and text_latents_norm must share a device"
            )

        state_input = torch.cat((text_latents_norm, previous_x0_norm), dim=-1).to(
            dtype=self.text_state_proj.weight.dtype
        )
        if self.fp32_boundaries:
            with torch.autocast(
                device_type=text_latents_norm.device.type, enabled=False
            ):
                text_state = self.text_state_proj(state_input.float())
                return self.text_input_proj(text_state)
        return self.text_input_proj(self.text_state_proj(state_input))

    def _chunk_modality(
        self,
        modality: Literal["image", "text"],
        *,
        batch_size: int,
        token_count: int,
    ) -> Tensor:
        """Reuse the checkpoint-compatible modality parameters as token features."""

        learned = (
            self.vision_modality_tokens
            if modality == "image"
            else self.text_modality_tokens
        )
        return learned.mean(dim=1, keepdim=True).expand(batch_size, token_count, -1)

    def embed_chunk_vision(
        self,
        vision_latents_norm: Tensor,
        token_timestep: Tensor,
    ) -> Tensor:
        """Embed image content without legacy prefix or absolute input positions."""

        if vision_latents_norm.ndim != 3 or vision_latents_norm.shape[1:] != (
            self.vision_tokens,
            self.vision_latent_dim,
        ):
            raise ValueError(
                f"vision_latents_norm must have shape [B, {self.vision_tokens}, {self.vision_latent_dim}]"
            )
        if token_timestep.shape != vision_latents_norm.shape[:1]:
            raise ValueError("image chunk timestep must have shape [B]")
        projected = self.vision_input_proj(vision_latents_norm)
        projected = projected + self.vision_sincos_2d.to(
            device=projected.device, dtype=projected.dtype
        )
        time = self.timestep_embedder(token_timestep).unsqueeze(1)
        modality = self._chunk_modality(
            "image",
            batch_size=projected.shape[0],
            token_count=projected.shape[1],
        )
        return projected + time + modality

    def embed_chunk_text(
        self,
        text_latents_norm: Tensor,
        *,
        previous_x0_norm: Tensor,
        token_timestep: Tensor,
    ) -> Tensor:
        """Embed text content with one continuous time value per token."""

        if token_timestep.shape != text_latents_norm.shape[:2]:
            raise ValueError("text chunk timestep must have shape [B, T]")
        projected = self.embed_text_content(
            text_latents_norm,
            previous_x0_norm=previous_x0_norm,
        )
        time = self.timestep_embedder(token_timestep.reshape(-1)).view_as(projected)
        modality = self._chunk_modality(
            "text",
            batch_size=projected.shape[0],
            token_count=projected.shape[1],
        )
        embedded = projected + time + modality
        return embedded

    def embed_vision(
        self,
        vision_latents_norm: Tensor,
        clean_t: Tensor,
    ) -> Tensor:
        if vision_latents_norm.ndim != 3 or vision_latents_norm.shape[1:] != (
            self.vision_tokens,
            self.vision_latent_dim,
        ):
            raise ValueError(
                f"vision_latents_norm must have shape [B, {self.vision_tokens}, {self.vision_latent_dim}]"
            )
        projected = self.vision_input_proj(vision_latents_norm)
        projected = projected + self.vision_sincos_2d.to(dtype=projected.dtype)
        return torch.cat((self._vision_prefix(clean_t), projected), dim=1)

    def embed_text(
        self,
        text_latents_norm: Tensor,
        clean_t: Tensor,
        *,
        previous_x0_norm: Tensor,
    ) -> Tensor:
        if text_latents_norm.ndim != 3 or text_latents_norm.shape[1:] != (
            self.text_tokens,
            self.text_latent_dim,
        ):
            raise ValueError(
                f"text_latents_norm must have shape [B, {self.text_tokens}, {self.text_latent_dim}]"
            )
        projected = self.embed_text_content(
            text_latents_norm,
            previous_x0_norm=previous_x0_norm,
        )
        prefix = self.embed_text_prefix(clean_t)
        return torch.cat((prefix, projected), dim=1)
