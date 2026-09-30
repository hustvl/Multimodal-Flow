from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from numbers import Real
from typing import Literal, Protocol

import torch
from torch import Tensor

from mf.contracts.batch import (
    TEXT_LATENT_DIM,
    TEXT_TOKENS,
    VISION_LATENT_DIM,
    VISION_TOKENS,
    BranchRole,
)
from mf.contracts.geometry import GeometryContract
from mf.contracts.model import MFModelInput
from mf.latents.time import sample_shifted_clean_t

_PAD_STATS_TYPE = 2


class NormalizedX0Model(Protocol):
    """Inference surface that exposes normalized predictions without decoding them."""

    text_tokens: int
    vision_tokens: int
    vision_latent_dim: int
    text_latent_dim: int

    def prepare_routing_layout(self, model_input: MFModelInput) -> object: ...

    def predict_normalized(
        self,
        model_input: MFModelInput,
        routing_layout: object,
    ) -> object: ...

    def denormalize_vision_prediction(
        self, vision_prediction_norm: Tensor
    ) -> Tensor: ...

    def denormalize_text_prediction(
        self,
        text_prediction_norm: Tensor,
        text_latent_stats_type: Tensor,
        text_content_mask: Tensor,
    ) -> Tensor: ...


class VisionDecoder(Protocol):
    def decode(self, raw_latents: Tensor) -> Tensor: ...


class TextDecoder(Protocol):
    def __call__(
        self, raw_latents: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor: ...


class TextBlockSession(Protocol):
    def prefill_vision_condition(
        self,
        latents_norm: Tensor,
        *,
        token_timestep: Tensor,
    ) -> None: ...

    def prefill_text_condition(
        self,
        latents_norm: Tensor,
        *,
        position_base: int,
        content_mask: Tensor | None = None,
    ) -> None: ...

    def predict_text_block(
        self,
        latents_norm: Tensor,
        *,
        previous_x0_norm: Tensor,
        token_timestep: Tensor,
        position_ids: Tensor,
    ) -> Tensor: ...

    def commit_text_block(
        self, latents_norm: Tensor, *, position_ids: Tensor
    ) -> None: ...


class TextBlockSessionFactory(Protocol):
    def __call__(
        self,
        *,
        batch_size: int,
        max_cache_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TextBlockSession: ...


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _model_vision_latent_dim(model: object) -> int:
    return _positive_int(
        "model.vision_latent_dim", getattr(model, "vision_latent_dim", 768)
    )


def _model_geometry(model: object) -> GeometryContract:
    geometry = getattr(model, "geometry", None)
    if isinstance(geometry, GeometryContract):
        return geometry.validate()
    vision_tokens = _positive_int(
        "model.vision_tokens", getattr(model, "vision_tokens", VISION_TOKENS)
    )
    vision_latent_dim = _model_vision_latent_dim(model)
    text_latent_dim = _positive_int(
        "model.text_latent_dim", getattr(model, "text_latent_dim", TEXT_LATENT_DIM)
    )
    grid = getattr(model, "vision_grid_size", None)
    if grid is None:
        side = int(vision_tokens**0.5)
        grid = (side, side) if side * side == vision_tokens else (1, vision_tokens)
    return GeometryContract(
        vision_tokens=vision_tokens,
        vision_latent_dim=vision_latent_dim,
        text_latent_dim=text_latent_dim,
        vision_grid_size=tuple(grid),
    ).validate()


def _model_vision_tokens(model: object) -> int:
    return _model_geometry(model).vision_tokens


def _model_text_latent_dim(model: object) -> int:
    return _model_geometry(model).text_latent_dim


def _model_text_block_size(model: object) -> int | None:
    if getattr(model, "sequence_layout", None) != "chunk_causal":
        return None
    return _positive_int(
        "model.text_block_size", getattr(model, "text_block_size", None)
    )


def _finite_non_negative(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    result = float(value)
    if result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive(name: str, value: object) -> float:
    result = _finite_non_negative(name, value)
    if result == 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _require_device(device: object) -> torch.device:
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    return device


@dataclass(frozen=True)
class SamplerConfig:
    """MF sampler controls with standard classifier-free guidance."""

    num_inference_steps: int = 100
    vision_latent_dim: int = VISION_LATENT_DIM
    cfg_scale: float = 2.0
    image_alpha: float = 6.0
    text_alpha: float = 6.0
    amp_dtype: torch.dtype | None = torch.bfloat16
    method: Literal["ode", "sde"] = "ode"
    sde_gamma: float = 1.0
    t_lognorm_mu: float = 0.0
    t_lognorm_sigma: float = 1.0
    velocity_t_eps: float = 0.05
    vision_noise_scale: float = 1.0
    text_noise_scale: float = 1.0
    text_decoder_input_space: Literal["raw", "normalized"] = "raw"

    def __post_init__(self) -> None:
        _positive_int("num_inference_steps", self.num_inference_steps)
        _finite_positive("image_alpha", self.image_alpha)
        _positive_int("vision_latent_dim", self.vision_latent_dim)
        _finite_positive("text_alpha", self.text_alpha)
        _finite_positive("vision_noise_scale", self.vision_noise_scale)
        _finite_positive("text_noise_scale", self.text_noise_scale)
        _finite_non_negative("cfg_scale", self.cfg_scale)
        _finite_non_negative("sde_gamma", self.sde_gamma)
        if not math.isfinite(float(self.t_lognorm_mu)):
            raise ValueError("t_lognorm_mu must be finite")
        if not math.isfinite(float(self.t_lognorm_sigma)) or self.t_lognorm_sigma <= 0:
            raise ValueError("t_lognorm_sigma must be finite and positive")
        if not math.isfinite(float(self.velocity_t_eps)) or self.velocity_t_eps <= 0:
            raise ValueError("velocity_t_eps must be finite and positive")
        if self.method not in ("ode", "sde"):
            raise ValueError("method must be 'ode' or 'sde'")
        if self.amp_dtype not in (None, torch.bfloat16, torch.float16):
            raise ValueError("amp_dtype must be torch.bfloat16, torch.float16, or None")
        if self.text_decoder_input_space not in ("raw", "normalized"):
            raise ValueError("text_decoder_input_space must be 'raw' or 'normalized'")


def _require_latents(name: str, value: object, shape: tuple[int, int]) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 3
        or tuple(value.shape[1:]) != shape
    ):
        actual = (
            list(value.shape) if isinstance(value, Tensor) else type(value).__name__
        )
        raise ValueError(
            f"{name} must have shape [B, {shape[0]}, {shape[1]}]; got {actual}"
        )
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must have a floating-point dtype")
    return value


def _require_text_metadata(
    batch_size: int,
    device: torch.device,
    content_mask: object,
    latent_stats_type: object,
) -> tuple[Tensor, Tensor]:
    if (
        not isinstance(content_mask, Tensor)
        or content_mask.ndim != 2
        or content_mask.shape[0] != batch_size
        or content_mask.shape[1] <= 0
    ):
        raise ValueError("content_mask must have shape [B, T] with T > 0")
    if content_mask.dtype is not torch.bool:
        raise ValueError("content_mask must have dtype torch.bool")
    expected = tuple(content_mask.shape)
    if (
        not isinstance(latent_stats_type, Tensor)
        or tuple(latent_stats_type.shape) != expected
    ):
        raise ValueError("latent_stats_type must match content_mask shape [B, T]")
    if latent_stats_type.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("latent_stats_type must have an integer dtype")
    if content_mask.device != device or latent_stats_type.device != device:
        raise ValueError("text metadata must be on the same device as its condition")
    return content_mask, latent_stats_type


@dataclass(frozen=True)
class TextCondition:
    latents_norm: Tensor
    content_mask: Tensor
    latent_stats_type: Tensor

    def __post_init__(self) -> None:
        latents = self.latents_norm
        if (
            not isinstance(latents, Tensor)
            or latents.ndim != 3
            or latents.shape[1] <= 0
            or latents.shape[2] <= 0
        ):
            actual = (
                list(latents.shape)
                if isinstance(latents, Tensor)
                else type(latents).__name__
            )
            raise ValueError(f"latents_norm must have shape [B, T, D] with D > 0; got {actual}")
        if not torch.is_floating_point(latents):
            raise ValueError("latents_norm must have a floating-point dtype")
        content_mask, _ = _require_text_metadata(
            latents.shape[0], latents.device, self.content_mask, self.latent_stats_type
        )
        if content_mask.shape[1] != latents.shape[1]:
            raise ValueError("text condition latents and metadata must use the same T")


@dataclass(frozen=True)
class VisionCondition:
    latents_norm: Tensor

    def __post_init__(self) -> None:
        latents = self.latents_norm
        if (
            not isinstance(latents, Tensor)
            or latents.ndim != 3
            or latents.shape[1] <= 0
            or latents.shape[2] <= 0
        ):
            actual = (
                list(latents.shape)
                if isinstance(latents, Tensor)
                else type(latents).__name__
            )
            raise ValueError(
                f"latents_norm must have shape [B, V, D] with V, D > 0; got {actual}"
            )
        if not torch.is_floating_point(latents):
            raise ValueError("latents_norm must have a floating-point dtype")


@dataclass(frozen=True)
class TextTarget:
    content_mask: Tensor
    latent_stats_type: Tensor

    def validate(self, *, batch_size: int, device: torch.device) -> None:
        _require_text_metadata(
            batch_size, device, self.content_mask, self.latent_stats_type
        )


@dataclass(frozen=True)
class VisionSample:
    normalized_latents: Tensor
    raw_latents: Tensor
    images: Tensor


@dataclass(frozen=True)
class TextSample:
    normalized_latents: Tensor
    raw_latents: Tensor
    logits: Tensor
    token_ids: Tensor


def shifted_clean_time_grid(
    *,
    num_inference_steps: int = 100,
    alpha: float,
    t_lognorm_mu: float,
    t_lognorm_sigma: float,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    """Build N+1 time points from the exact training distribution for N updates."""
    num_inference_steps = _positive_int("num_inference_steps", num_inference_steps)
    alpha = _finite_positive("alpha", alpha)
    device = _require_device(device)

    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    internal = (
        sample_shifted_clean_t(
            num_inference_steps - 1,
            alpha,
            t_lognorm_mu,
            t_lognorm_sigma,
            generator,
            device,
            torch.float32,
        )
        .sort()
        .values
        if num_inference_steps > 1
        else torch.empty(0, dtype=torch.float32, device=device)
    )
    clean_t = torch.cat(
        (
            torch.zeros(1, dtype=torch.float32, device=device),
            internal,
            torch.ones(1, dtype=torch.float32, device=device),
        )
    )
    if not bool(torch.all(clean_t[1:] > clean_t[:-1])):
        raise ValueError("float32 shifted clean-t grid is not strictly increasing")
    return clean_t


def prepare_sde_backstep(
    x_t: Tensor,
    *,
    clean_t: Tensor,
    next_clean_t: Tensor,
    gamma: float,
    noise: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply the stochastic backstep before the deterministic x0 update."""
    if x_t.shape != noise.shape or x_t.device != noise.device:
        raise ValueError("noise must match x_t shape and device")
    gamma = _finite_non_negative("gamma", gamma)
    h = next_clean_t.to(dtype=torch.float32) - clean_t.to(dtype=torch.float32)
    alpha = torch.clamp(1.0 - gamma * h, min=0.0, max=1.0)
    return (
        alpha * x_t.float() + (1.0 - alpha) * noise.float(),
        alpha * clean_t.to(dtype=torch.float32),
    )


def advance_linear_path_x0(
    x_t: Tensor,
    x0_prediction: Tensor,
    clean_t: Tensor,
    next_clean_t: Tensor,
    *,
    t_eps: float = 0.05,
) -> Tensor:
    """Advance the linear path using adjacent entries from a validated clean-t grid."""
    if not isinstance(x_t, Tensor) or not isinstance(x0_prediction, Tensor):
        raise TypeError("x_t and x0_prediction must be tensors")
    if x_t.shape != x0_prediction.shape or x_t.device != x0_prediction.device:
        raise ValueError("x_t and x0_prediction must have matching shape and device")
    for name, value in (("clean_t", clean_t), ("next_clean_t", next_clean_t)):
        if (
            not isinstance(value, Tensor)
            or value.numel() != 1
            or value.device != x_t.device
            or not torch.is_floating_point(value)
        ):
            raise ValueError(f"{name} must be a scalar tensor on the latent device")
    t = clean_t.to(dtype=torch.float32)
    if not isinstance(t_eps, (int, float)) or not math.isfinite(t_eps) or t_eps <= 0:
        raise ValueError("t_eps must be finite and positive")
    next_t = next_clean_t.to(dtype=torch.float32)
    denominator = torch.clamp(1.0 - t, min=float(t_eps))
    velocity = (x0_prediction.float() - x_t.float()) / denominator
    return x_t.float() + (next_t - t) * velocity


def sample_seed(*, run_seed: int, global_sample_index: int) -> int:
    """Derive a stable torch seed from run identity and global sample identity."""
    for name, value in (
        ("run_seed", run_seed),
        ("global_sample_index", global_sample_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    payload = f"{run_seed}:{global_sample_index}".encode("ascii")
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"mf-noise-v1",
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _solver_seed(*, run_seed: int, global_sample_index: int, step: int) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    payload = f"{run_seed}:{global_sample_index}:{step}".encode("ascii")
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"mf-sde-v1",
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _sample_solver_noise(
    *,
    global_sample_indices: Sequence[int],
    latent_shape: tuple[int, int],
    run_seed: int,
    step: int,
    device: torch.device,
    noise_scale: float = 1.0,
) -> Tensor:
    samples = []
    for global_sample_index in global_sample_indices:
        generator = torch.Generator(device=device)
        generator.manual_seed(
            _solver_seed(
                run_seed=run_seed,
                global_sample_index=global_sample_index,
                step=step,
            )
        )
        samples.append(
            torch.randn(
                latent_shape,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        )
    return torch.stack(samples) * float(noise_scale)


def _time_grid(
    config: SamplerConfig,
    *,
    alpha: float,
    seed: int,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return shifted_clean_time_grid(
        num_inference_steps=config.num_inference_steps,
        alpha=alpha,
        t_lognorm_mu=config.t_lognorm_mu,
        t_lognorm_sigma=config.t_lognorm_sigma,
        generator=generator,
        device=device,
    )


def _solver_input(
    x_t: Tensor,
    clean_t: Tensor,
    next_clean_t: Tensor,
    *,
    config: SamplerConfig,
    global_sample_indices: Sequence[int],
    seed: int,
    step: int,
    final_step: bool,
    noise_scale: float,
) -> tuple[Tensor, Tensor]:
    if type(final_step) is not bool:
        raise TypeError("final_step must be a Python bool")
    if config.method == "ode" or final_step:
        return x_t, clean_t
    noise = _sample_solver_noise(
        global_sample_indices=global_sample_indices,
        latent_shape=(x_t.shape[1], x_t.shape[2]),
        run_seed=seed,
        step=step,
        device=x_t.device,
        noise_scale=noise_scale,
    )
    return prepare_sde_backstep(
        x_t,
        clean_t=clean_t,
        next_clean_t=next_clean_t,
        gamma=config.sde_gamma,
        noise=noise,
    )


def sample_initial_noise(
    *,
    global_sample_indices: Sequence[int],
    latent_shape: tuple[int, int],
    run_seed: int,
    device: torch.device,
    noise_scale: float = 1.0,
) -> Tensor:
    """Sample Gaussian latents keyed by global identity, independent of rank grouping."""
    indices = tuple(global_sample_indices)
    if not indices:
        raise ValueError("global_sample_indices must not be empty")
    if not isinstance(latent_shape, tuple) or len(latent_shape) != 2:
        raise ValueError("latent_shape must be a two-integer tuple")
    shape = tuple(_positive_int("latent_shape", value) for value in latent_shape)
    device = _require_device(device)
    samples = []
    for global_sample_index in indices:
        seed = sample_seed(
            run_seed=run_seed,
            global_sample_index=global_sample_index,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        samples.append(
            torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        )
    return torch.stack(samples) * float(noise_scale)


def _branch_roles(
    batch_size: int,
    *,
    vision: BranchRole,
    text: BranchRole,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    return (
        torch.full((batch_size,), int(vision), dtype=torch.long, device=device),
        torch.full((batch_size,), int(text), dtype=torch.long, device=device),
    )


def _active_token_mask(
    vision_role: Tensor,
    text_role: Tensor,
    text_content_mask: Tensor,
    text_prompt_content_mask: Tensor,
    *,
    text_block_size: int | None = None,
    geometry: GeometryContract | None = None,
) -> Tensor:
    geometry = geometry or GeometryContract()
    geometry.validate()
    batch_size = vision_role.shape[0]
    prompt_tokens = text_prompt_content_mask.shape[1]
    prompt_layout_tokens = (
        geometry.text_prefix_tokens + prompt_tokens if prompt_tokens > 0 else 0
    )
    active = torch.zeros(
        (
            batch_size,
            geometry.vision_layout_tokens
            + prompt_layout_tokens
            + geometry.text_prefix_tokens
            + text_content_mask.shape[1] * (2 if text_block_size is not None else 1),
        ),
        dtype=torch.bool,
        device=vision_role.device,
    )
    vision_present = vision_role != int(BranchRole.ABSENT)
    text_present = text_role != int(BranchRole.ABSENT)
    active[:, : geometry.vision_layout_tokens] = vision_present[:, None]
    prompt_prefix_start = geometry.vision_layout_tokens
    prompt_content_start = prompt_prefix_start
    if prompt_tokens > 0:
        prompt_content_start += geometry.text_prefix_tokens
        active[:, prompt_prefix_start:prompt_content_start] = (
            text_prompt_content_mask.any(dim=1)[:, None]
        )
        active[:, prompt_content_start : prompt_content_start + prompt_tokens] = (
            text_prompt_content_mask
        )
    text_start = prompt_content_start + prompt_tokens
    latent_start = text_start + geometry.text_prefix_tokens
    active[:, text_start:latent_start] = text_present[:, None]
    if text_block_size is None:
        active[:, latent_start:] = text_content_mask & text_present[:, None]
    else:
        clean_end = latent_start + text_content_mask.shape[1]
        active[:, latent_start:clean_end] = text_content_mask & text_present[:, None]
        active[:, clean_end:] = (
            text_content_mask & (text_role == int(BranchRole.TARGET))[:, None]
        )
    return active


def _model_input(
    model: object | None = None,
    *,
    vision_latents_norm: Tensor | None,
    text_latents_norm: Tensor | None,
    vision_timestep: Tensor,
    text_timestep: Tensor,
    vision_role_value: BranchRole,
    text_role_value: BranchRole,
    text_content_mask: Tensor,
    text_latent_stats_type: Tensor,
    text_prompt_latents_norm: Tensor | None = None,
    text_prompt_content_mask: Tensor | None = None,
    text_previous_x0_norm: Tensor | None = None,
    null_conditioning: Tensor | None = None,
    active_token_mask: Tensor | None = None,
    text_clean_latents_norm: Tensor | None = None,
    text_token_timestep: Tensor | None = None,
    text_block_size: int | None = None,
    text_segment_ids: Tensor | None = None,
) -> MFModelInput:
    if vision_timestep.ndim != 1 or text_timestep.shape != vision_timestep.shape:
        raise ValueError("vision_timestep and text_timestep must have shape [B]")
    batch_size = vision_timestep.shape[0]
    device = vision_timestep.device
    if model is not None:
        geometry = _model_geometry(model)
    else:
        vision_tokens = (
            vision_latents_norm.shape[1]
            if vision_latents_norm is not None
            else VISION_TOKENS
        )
        vision_latent_dim = (
            vision_latents_norm.shape[2]
            if vision_latents_norm is not None
            else VISION_LATENT_DIM
        )
        text_latent_dim = (
            text_latents_norm.shape[2]
            if text_latents_norm is not None
            else (
                text_prompt_latents_norm.shape[2]
                if text_prompt_latents_norm is not None
                else TEXT_LATENT_DIM
            )
        )
        side = int(vision_tokens**0.5)
        geometry = GeometryContract(
            vision_tokens=vision_tokens,
            vision_latent_dim=vision_latent_dim,
            text_latent_dim=text_latent_dim,
            vision_grid_size=(
                (side, side) if side * side == vision_tokens else (1, vision_tokens)
            ),
        ).validate()
    vision_role, text_role = _branch_roles(
        batch_size,
        vision=vision_role_value,
        text=text_role_value,
        device=device,
    )
    if text_role_value is BranchRole.ABSENT:
        if text_latents_norm is not None or text_previous_x0_norm is not None:
            raise ValueError(
                "ABSENT text must not materialize current or previous latents"
            )
    elif text_latents_norm is None:
        raise ValueError("an active text role requires text latents")
    elif text_previous_x0_norm is None:
        text_previous_x0_norm = torch.zeros_like(text_latents_norm)
    if vision_role_value is BranchRole.ABSENT and vision_latents_norm is not None:
        raise ValueError("ABSENT vision must not materialize latents")
    if text_prompt_content_mask is None:
        text_prompt_content_mask = torch.zeros(
            (batch_size, 0), dtype=torch.bool, device=device
        )
    if text_prompt_latents_norm is None:
        if bool(text_prompt_content_mask.any()):
            raise ValueError("an active text prompt mask requires prompt latents")
    elif (
        text_prompt_latents_norm.shape[:2] != text_prompt_content_mask.shape
        or text_prompt_latents_norm.shape[2] != geometry.text_latent_dim
    ):
        raise ValueError(
            "text prompt latents and mask must have matching [B, P] dimensions"
        )
    if null_conditioning is None:
        null_conditioning = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if text_block_size is not None:
        if text_clean_latents_norm is None:
            if text_role_value is BranchRole.CONDITION:
                if text_latents_norm is None:
                    raise RuntimeError(
                        "block-causal text condition is missing clean latents"
                    )
                text_clean_latents_norm = text_latents_norm
            else:
                text_clean_latents_norm = torch.zeros(
                    (batch_size, text_content_mask.shape[1], geometry.text_latent_dim),
                    dtype=vision_timestep.dtype,
                    device=device,
                )
        if text_token_timestep is None:
            text_token_timestep = torch.where(
                (text_role == int(BranchRole.TARGET))[:, None] & text_content_mask,
                text_timestep[:, None],
                torch.zeros_like(text_timestep[:, None]),
            )
        if text_segment_ids is None:
            text_segment_ids = torch.where(
                text_content_mask,
                torch.zeros_like(text_content_mask, dtype=torch.long),
                torch.full_like(text_content_mask, -1, dtype=torch.long),
            )
    return MFModelInput(
        vision_latents_norm=vision_latents_norm,
        text_latents_norm=text_latents_norm,
        vision_timestep=vision_timestep,
        text_timestep=text_timestep,
        text_prompt_latents_norm=text_prompt_latents_norm,
        text_prompt_content_mask=text_prompt_content_mask,
        vision_role=vision_role,
        text_role=text_role,
        text_content_mask=text_content_mask,
        text_latent_stats_type=text_latent_stats_type,
        active_token_mask=(
            _active_token_mask(
                vision_role,
                text_role,
                text_content_mask,
                text_prompt_content_mask,
                text_block_size=text_block_size,
                geometry=geometry,
            )
            if active_token_mask is None
            else active_token_mask
        ),
        text_previous_x0_norm=text_previous_x0_norm,
        null_conditioning=null_conditioning,
        text_clean_latents_norm=text_clean_latents_norm,
        text_token_timestep=text_token_timestep,
        text_block_size=text_block_size,
        text_segment_ids=text_segment_ids,
        vision_latent_dim=(
            geometry.vision_latent_dim
        ),
        geometry=geometry,
    ).validate()


def _autocast(
    device: torch.device, dtype: torch.dtype | None
) -> AbstractContextManager[object]:
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _prediction(output: object, name: str, expected: Tensor) -> Tensor:
    prediction = getattr(output, name, None)
    if not isinstance(prediction, Tensor):
        raise TypeError(f"model output must provide tensor {name}")
    if prediction.shape != expected.shape or prediction.device != expected.device:
        raise ValueError(f"{name} must match the target latent shape and device")
    return prediction


@torch.inference_mode()
def sample_vision(
    model: NormalizedX0Model,
    text_condition: TextCondition,
    config: SamplerConfig,
    seed: int,
    *,
    global_sample_indices: Sequence[int],
    decoder: VisionDecoder,
) -> VisionSample:
    """Sample T2I with standard CFG between T2I and Image-only routes."""

    if not isinstance(config, SamplerConfig):
        raise TypeError("config must be a SamplerConfig")
    if not isinstance(text_condition, TextCondition):
        raise TypeError("text_condition must be a TextCondition")
    text = text_condition.latents_norm
    batch_size = text.shape[0]
    device = text.device
    clean_text = torch.where(
        text_condition.content_mask.unsqueeze(-1),
        text,
        torch.zeros_like(text),
    )
    indices = tuple(global_sample_indices)
    if len(indices) != batch_size:
        raise ValueError("global_sample_indices must have one entry per condition")
    x_t = sample_initial_noise(
        global_sample_indices=indices,
        latent_shape=(_model_vision_tokens(model), _model_vision_latent_dim(model)),
        run_seed=seed,
        device=device,
        noise_scale=config.vision_noise_scale,
    )
    grid = _time_grid(config, alpha=config.image_alpha, seed=seed, device=device)
    initial_timestep = grid[0].expand(batch_size)
    conditional_template = _model_input(
        model=model,
        vision_latents_norm=x_t,
        text_latents_norm=clean_text,
        vision_timestep=initial_timestep,
        text_timestep=torch.ones_like(initial_timestep),
        vision_role_value=BranchRole.TARGET,
        text_role_value=BranchRole.CONDITION,
        text_content_mask=text_condition.content_mask,
        text_latent_stats_type=text_condition.latent_stats_type,
        text_block_size=_model_text_block_size(model),
    )
    conditional_layout = model.prepare_routing_layout(conditional_template)
    unconditional_template = _model_input(
        model=model,
        vision_latents_norm=x_t,
        text_latents_norm=torch.zeros_like(clean_text),
        vision_timestep=initial_timestep,
        text_timestep=torch.ones_like(initial_timestep),
        vision_role_value=BranchRole.TARGET,
        text_role_value=BranchRole.CONDITION,
        text_content_mask=text_condition.content_mask,
        text_latent_stats_type=text_condition.latent_stats_type,
        null_conditioning=torch.ones(
            batch_size,
            dtype=torch.bool,
            device=device,
        ),
        text_block_size=_model_text_block_size(model),
    )
    unconditional_layout = model.prepare_routing_layout(unconditional_template)

    for step, (clean_t, next_clean_t) in enumerate(
        zip(grid[:-1], grid[1:], strict=True)
    ):
        solver_x, solver_t = _solver_input(
            x_t,
            clean_t,
            next_clean_t,
            config=config,
            global_sample_indices=indices,
            seed=seed,
            step=step,
            final_step=step == config.num_inference_steps - 1,
            noise_scale=config.vision_noise_scale,
        )
        vision_timestep = solver_t.expand(batch_size)
        conditional = replace(
            conditional_template,
            vision_latents_norm=solver_x,
            vision_timestep=vision_timestep,
        ).validate()
        unconditional = replace(
            unconditional_template,
            vision_latents_norm=solver_x,
            vision_timestep=vision_timestep,
        ).validate()
        with _autocast(device, config.amp_dtype):
            conditional_output = model.predict_normalized(
                conditional, conditional_layout
            )
            unconditional_output = model.predict_normalized(
                unconditional,
                unconditional_layout,
            )
        x0_conditional = _prediction(
            conditional_output,
            "vision_pred_norm",
            solver_x,
        ).float()
        x0_unconditional = _prediction(
            unconditional_output,
            "vision_pred_norm",
            solver_x,
        ).float()
        x0_guided = x0_unconditional + config.cfg_scale * (
            x0_conditional - x0_unconditional
        )
        x_t = advance_linear_path_x0(
            solver_x,
            x0_guided,
            solver_t,
            next_clean_t,
            t_eps=config.velocity_t_eps,
        )

    normalized = x_t
    raw = model.denormalize_vision_prediction(normalized)
    images = decoder.decode(raw)
    return VisionSample(normalized_latents=normalized, raw_latents=raw, images=images)


@torch.inference_mode()
def sample_vision_unconditional(
    model: NormalizedX0Model,
    config: SamplerConfig,
    seed: int,
    *,
    global_sample_indices: Sequence[int],
    device: torch.device,
    decoder: VisionDecoder,
) -> VisionSample:
    """Sample Image-only with an absent text branch and no guidance state."""

    if not isinstance(config, SamplerConfig):
        raise TypeError("config must be a SamplerConfig")
    indices = tuple(global_sample_indices)
    if not indices:
        raise ValueError("global_sample_indices must not be empty")
    device = _require_device(device)
    batch_size = len(indices)
    text_tokens = _positive_int(
        "model.text_tokens",
        getattr(model, "text_tokens", TEXT_TOKENS),
    )
    x_t = sample_initial_noise(
        global_sample_indices=indices,
        latent_shape=(_model_vision_tokens(model), _model_vision_latent_dim(model)),
        run_seed=seed,
        device=device,
        noise_scale=config.vision_noise_scale,
    )
    grid = _time_grid(config, alpha=config.image_alpha, seed=seed, device=device)
    text_content_mask = torch.zeros(
        (batch_size, text_tokens),
        dtype=torch.bool,
        device=device,
    )
    text_stats_type = torch.full(
        (batch_size, text_tokens),
        _PAD_STATS_TYPE,
        dtype=torch.long,
        device=device,
    )
    initial_timestep = grid[0].expand(batch_size)
    route_template = _model_input(
        model=model,
        vision_latents_norm=x_t,
        text_latents_norm=None,
        vision_timestep=initial_timestep,
        text_timestep=torch.zeros_like(initial_timestep),
        vision_role_value=BranchRole.TARGET,
        text_role_value=BranchRole.ABSENT,
        text_content_mask=text_content_mask,
        text_latent_stats_type=text_stats_type,
        text_block_size=_model_text_block_size(model),
    )
    routing_layout = model.prepare_routing_layout(route_template)

    for step, (clean_t, next_clean_t) in enumerate(
        zip(grid[:-1], grid[1:], strict=True)
    ):
        solver_x, solver_t = _solver_input(
            x_t,
            clean_t,
            next_clean_t,
            config=config,
            global_sample_indices=indices,
            seed=seed,
            step=step,
            final_step=step == config.num_inference_steps - 1,
            noise_scale=config.vision_noise_scale,
        )
        vision_timestep = solver_t.expand(batch_size)
        model_input = replace(
            route_template,
            vision_latents_norm=solver_x,
            vision_timestep=vision_timestep,
        ).validate()
        with _autocast(device, config.amp_dtype):
            output = model.predict_normalized(model_input, routing_layout)
        x0_prediction = _prediction(output, "vision_pred_norm", solver_x).float()
        x_t = advance_linear_path_x0(
            solver_x,
            x0_prediction,
            solver_t,
            next_clean_t,
            t_eps=config.velocity_t_eps,
        )

    normalized = x_t
    raw = model.denormalize_vision_prediction(normalized)
    images = decoder.decode(raw)
    return VisionSample(normalized_latents=normalized, raw_latents=raw, images=images)


def _decode_text_latents(
    decoder: TextDecoder,
    decoder_latents: Tensor,
    content_mask: Tensor,
    *,
    block_size: int | None,
) -> Tensor:
    batch_size, text_tokens, latent_dim = decoder_latents.shape
    if block_size is None:
        return decoder(decoder_latents, attention_mask=content_mask)
    if text_tokens % block_size != 0:
        raise ValueError("decoder text length must be divisible by block_size")
    block_count = text_tokens // block_size
    flat_latents = decoder_latents.reshape(
        batch_size * block_count,
        block_size,
        latent_dim,
    )
    flat_mask = content_mask.reshape(batch_size * block_count, block_size)
    flat_logits = decoder(flat_latents, attention_mask=flat_mask)
    if not isinstance(flat_logits, Tensor) or tuple(flat_logits.shape[:2]) != (
        batch_size * block_count,
        block_size,
    ):
        raise ValueError(
            "block text decoder must return logits with shape [B*blocks, K, vocab]"
        )
    return flat_logits.view(batch_size, text_tokens, -1)


def _text_sample_from_normalized(
    model: NormalizedX0Model,
    normalized: Tensor,
    text_target: TextTarget,
    config: SamplerConfig,
    decoder: TextDecoder,
    *,
    block_size: int | None,
) -> TextSample:
    batch_size, text_tokens = normalized.shape[:2]
    raw = model.denormalize_text_prediction(
        normalized,
        text_target.latent_stats_type,
        text_target.content_mask,
    )
    decoder_latents = (
        normalized if config.text_decoder_input_space == "normalized" else raw
    )
    logits = _decode_text_latents(
        decoder,
        decoder_latents,
        text_target.content_mask,
        block_size=block_size,
    )
    if not isinstance(logits, Tensor) or tuple(logits.shape[:2]) != (
        batch_size,
        text_tokens,
    ):
        raise ValueError("text decoder must return logits with shape [B, T, vocab]")
    return TextSample(
        normalized_latents=normalized,
        raw_latents=raw,
        logits=logits,
        token_ids=logits.argmax(dim=-1),
    )


def _block_token_timesteps(
    content_mask: Tensor,
    *,
    start: int,
    end: int,
    current_timestep: Tensor,
) -> Tensor:
    token_timestep = content_mask.to(dtype=current_timestep.dtype)
    token_timestep[:, start:end] = torch.where(
        content_mask[:, start:end],
        current_timestep[:, None],
        torch.zeros_like(token_timestep[:, start:end]),
    )
    return token_timestep


def _shared_contiguous_prefix_length(name: str, mask: Tensor) -> int:
    lengths = _contiguous_prefix_lengths(name, mask)
    if not bool(lengths.eq(lengths[0]).all()):
        raise ValueError(f"cached block generation requires equal {name} lengths")
    return int(lengths[0].item())


def _contiguous_prefix_lengths(name: str, mask: Tensor) -> Tensor:
    if mask.ndim != 2 or mask.dtype is not torch.bool:
        raise ValueError(f"{name} must be a bool [B, T] tensor")
    lengths = mask.sum(dim=1, dtype=torch.long)
    expected = (
        torch.arange(mask.shape[1], device=mask.device).unsqueeze(0) < lengths[:, None]
    )
    if not bool(mask.eq(expected).all()):
        raise ValueError(f"cached block generation requires contiguous {name} prefixes")
    return lengths


def _sample_text_block_causal(
    model: NormalizedX0Model,
    *,
    vision: Tensor,
    clean_prompt: Tensor,
    prompt_mask: Tensor,
    text_target: TextTarget,
    config: SamplerConfig,
    seed: int,
    indices: tuple[int, ...],
    decoder: TextDecoder,
    block_size: int,
    stop_at_eos_token_id: int | None,
    block_session_factory: TextBlockSessionFactory | None,
) -> TextSample:
    batch_size = vision.shape[0]
    device = vision.device
    text_tokens = text_target.content_mask.shape[1]
    token_mask = text_target.content_mask.unsqueeze(-1)
    initial_noise = sample_initial_noise(
        global_sample_indices=indices,
        latent_shape=(text_tokens, _model_text_latent_dim(model)),
        run_seed=seed,
        device=device,
        noise_scale=config.text_noise_scale,
    )
    initial_noise = torch.where(
        token_mask, initial_noise, torch.zeros_like(initial_noise)
    )
    clean_state = torch.zeros_like(initial_noise)
    initial_noisy = torch.zeros_like(initial_noise)
    first_end = min(block_size, text_tokens)
    initial_noisy[:, :first_end] = initial_noise[:, :first_end]
    grid = _time_grid(config, alpha=config.text_alpha, seed=seed, device=device)
    text_timestep = torch.ones(batch_size, dtype=grid.dtype, device=device)
    token_timestep = text_target.content_mask.to(dtype=grid.dtype)
    conditional_template = _model_input(
        model=model,
        vision_latents_norm=vision,
        text_latents_norm=initial_noisy,
        text_prompt_latents_norm=clean_prompt,
        text_prompt_content_mask=prompt_mask,
        vision_timestep=torch.ones_like(text_timestep),
        text_timestep=text_timestep,
        vision_role_value=BranchRole.CONDITION,
        text_role_value=BranchRole.TARGET,
        text_content_mask=text_target.content_mask,
        text_latent_stats_type=text_target.latent_stats_type,
        text_previous_x0_norm=torch.zeros_like(initial_noise),
        text_clean_latents_norm=clean_state,
        text_token_timestep=token_timestep,
        text_block_size=block_size,
    )
    unconditional_template = _model_input(
        model=model,
        vision_latents_norm=torch.zeros_like(vision),
        text_latents_norm=initial_noisy,
        text_prompt_latents_norm=torch.zeros_like(clean_prompt),
        text_prompt_content_mask=prompt_mask,
        vision_timestep=torch.ones_like(text_timestep),
        text_timestep=text_timestep,
        vision_role_value=BranchRole.CONDITION,
        text_role_value=BranchRole.TARGET,
        text_content_mask=text_target.content_mask,
        text_latent_stats_type=text_target.latent_stats_type,
        text_previous_x0_norm=torch.zeros_like(initial_noise),
        null_conditioning=torch.ones(
            batch_size,
            dtype=torch.bool,
            device=device,
        ),
        text_clean_latents_norm=clean_state,
        text_token_timestep=token_timestep,
        text_block_size=block_size,
    )
    conditional_session = None
    unconditional_session = None
    prompt_tokens = 0
    prompt_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    generated_tokens = text_tokens
    vision_position_base = max(_model_geometry(model).vision_grid_size)
    if block_session_factory is None:
        conditional_layout = model.prepare_routing_layout(conditional_template)
        unconditional_layout = model.prepare_routing_layout(unconditional_template)
    else:
        prompt_lengths = _contiguous_prefix_lengths("prompt", prompt_mask)
        prompt_tokens = int(prompt_lengths.max().item())
        generated_tokens = _shared_contiguous_prefix_length(
            "text target",
            text_target.content_mask,
        )
        image_tokens = (
            _model_geometry(model).vision_layout_tokens
            if getattr(model, "image_chunk_conditioning", "token_additive")
            == "legacy_active_prefix"
            else _model_geometry(model).vision_tokens
        )
        session_dtype = config.amp_dtype or torch.float32
        conditional_session = block_session_factory(
            batch_size=batch_size,
            max_cache_len=image_tokens + prompt_tokens + generated_tokens,
            device=device,
            dtype=session_dtype,
        )
        unconditional_session = block_session_factory(
            batch_size=batch_size,
            max_cache_len=image_tokens + prompt_tokens + generated_tokens,
            device=device,
            dtype=session_dtype,
        )
        with _autocast(device, config.amp_dtype):
            conditional_session.prefill_vision_condition(
                vision,
                token_timestep=torch.ones_like(text_timestep),
            )
            if prompt_tokens:
                conditional_session.prefill_text_condition(
                    clean_prompt[:, :prompt_tokens],
                    position_base=vision_position_base,
                    content_mask=prompt_mask[:, :prompt_tokens],
                )
            unconditional_session.prefill_vision_condition(
                torch.zeros_like(vision),
                token_timestep=torch.ones_like(text_timestep),
            )
            if prompt_tokens:
                unconditional_session.prefill_text_condition(
                    torch.zeros_like(clean_prompt[:, :prompt_tokens]),
                    position_base=vision_position_base,
                    content_mask=prompt_mask[:, :prompt_tokens],
                )
        conditional_layout = None
        unconditional_layout = None
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for block_index, start in enumerate(range(0, generated_tokens, block_size)):
        end = min(start + block_size, generated_tokens)
        local_positions = torch.arange(start, end, device=device, dtype=torch.long)
        conditional_position_ids = (
            local_positions[None, None, :]
            + vision_position_base
            + prompt_lengths[None, :, None]
        ).expand(3, -1, -1)
        unconditional_position_ids = conditional_position_ids
        block_mask = token_mask[:, start:end]
        if stop_at_eos_token_id is not None:
            block_mask = block_mask & ~finished[:, None, None]
        x_block = initial_noise[:, start:end]
        for step, (clean_t, next_clean_t) in enumerate(
            zip(grid[:-1], grid[1:], strict=True)
        ):
            solver_block, solver_t = _solver_input(
                x_block,
                clean_t,
                next_clean_t,
                config=config,
                global_sample_indices=indices,
                seed=seed,
                step=block_index * config.num_inference_steps + step,
                final_step=step == config.num_inference_steps - 1,
                noise_scale=config.text_noise_scale,
            )
            solver_block = torch.where(
                block_mask,
                solver_block,
                torch.zeros_like(solver_block),
            )
            current_timestep = solver_t.expand(batch_size)
            with _autocast(device, config.amp_dtype):
                if conditional_session is None or unconditional_session is None:
                    if conditional_layout is None or unconditional_layout is None:
                        raise RuntimeError(
                            "uncached I2T generation is missing routing layouts"
                        )
                    noisy_state = torch.zeros_like(clean_state)
                    noisy_state[:, start:end] = solver_block
                    token_timestep = _block_token_timesteps(
                        text_target.content_mask,
                        start=start,
                        end=end,
                        current_timestep=current_timestep,
                    )
                    conditional = replace(
                        conditional_template,
                        text_latents_norm=noisy_state,
                        text_clean_latents_norm=clean_state,
                        text_token_timestep=token_timestep,
                        text_previous_x0_norm=torch.zeros_like(noisy_state),
                    ).validate()
                    unconditional = replace(
                        unconditional_template,
                        text_latents_norm=noisy_state,
                        text_clean_latents_norm=clean_state,
                        text_token_timestep=token_timestep,
                        text_previous_x0_norm=torch.zeros_like(noisy_state),
                    ).validate()
                    conditional_output = model.predict_normalized(
                        conditional,
                        conditional_layout,
                    )
                    unconditional_output = model.predict_normalized(
                        unconditional,
                        unconditional_layout,
                    )
                    x0_conditional = _prediction(
                        conditional_output,
                        "text_pred_norm",
                        noisy_state,
                    )[:, start:end]
                    x0_unconditional = _prediction(
                        unconditional_output,
                        "text_pred_norm",
                        noisy_state,
                    )[:, start:end]
                else:
                    zeros = torch.zeros_like(solver_block)
                    block_timestep = current_timestep[:, None].expand(-1, end - start)
                    x0_conditional = conditional_session.predict_text_block(
                        solver_block,
                        previous_x0_norm=zeros,
                        token_timestep=block_timestep,
                        position_ids=conditional_position_ids,
                    )
                    x0_unconditional = unconditional_session.predict_text_block(
                        solver_block,
                        previous_x0_norm=zeros,
                        token_timestep=block_timestep,
                        position_ids=unconditional_position_ids,
                    )
            x0_conditional = x0_conditional.float()
            x0_unconditional = x0_unconditional.float()
            x0_block = x0_unconditional + config.cfg_scale * (
                x0_conditional - x0_unconditional
            )
            x0_block = torch.where(block_mask, x0_block, torch.zeros_like(x0_block))
            x_block = advance_linear_path_x0(
                solver_block,
                x0_block,
                solver_t,
                next_clean_t,
                t_eps=config.velocity_t_eps,
            )
            x_block = torch.where(block_mask, x_block, torch.zeros_like(x_block))
        next_clean_state = clean_state.clone()
        next_clean_state[:, start:end] = x_block
        clean_state = next_clean_state
        if conditional_session is not None and unconditional_session is not None:
            with _autocast(device, config.amp_dtype):
                conditional_session.commit_text_block(
                    x_block,
                    position_ids=conditional_position_ids,
                )
                unconditional_session.commit_text_block(
                    x_block,
                    position_ids=unconditional_position_ids,
                )
        if stop_at_eos_token_id is not None:
            block_stats = text_target.latent_stats_type[:, start:end]
            block_content_mask = text_target.content_mask[:, start:end]
            block_raw = model.denormalize_text_prediction(
                x_block,
                block_stats,
                block_content_mask,
            )
            block_decoder_input = (
                x_block
                if config.text_decoder_input_space == "normalized"
                else block_raw
            )
            block_logits = decoder(
                block_decoder_input, attention_mask=block_content_mask
            )
            if not isinstance(block_logits, Tensor) or tuple(
                block_logits.shape[:2]
            ) != (
                batch_size,
                end - start,
            ):
                raise ValueError(
                    "block text decoder must return logits with shape [B, K, vocab]"
                )
            block_tokens = block_logits.argmax(dim=-1)
            finished |= block_tokens.eq(stop_at_eos_token_id).any(dim=1)
            if bool(finished.all()):
                break

    result = _text_sample_from_normalized(
        model,
        clean_state,
        text_target,
        config,
        decoder,
        block_size=block_size,
    )
    if stop_at_eos_token_id is None:
        return result
    eos = result.token_ids.eq(stop_at_eos_token_id)
    positions = torch.arange(text_tokens, device=device).expand(batch_size, -1)
    first_eos = torch.where(eos, positions, text_tokens).amin(dim=1)
    stopped_tokens = torch.where(
        positions >= first_eos[:, None],
        torch.full_like(result.token_ids, stop_at_eos_token_id),
        result.token_ids,
    )
    return replace(result, token_ids=stopped_tokens)


def _sample_text_unconditional_block_causal(
    model: NormalizedX0Model,
    *,
    text_target: TextTarget,
    config: SamplerConfig,
    seed: int,
    indices: tuple[int, ...],
    decoder: TextDecoder,
    block_size: int,
    text_prompt: TextCondition | None = None,
    stop_at_eos_token_id: int | None = None,
    block_session: TextBlockSession | None = None,
) -> TextSample:
    batch_size = len(indices)
    device = text_target.content_mask.device
    text_tokens = text_target.content_mask.shape[1]
    token_mask = text_target.content_mask.unsqueeze(-1)
    initial_noise = sample_initial_noise(
        global_sample_indices=indices,
        latent_shape=(text_tokens, _model_text_latent_dim(model)),
        run_seed=seed,
        device=device,
        noise_scale=config.text_noise_scale,
    )
    initial_noise = torch.where(
        token_mask, initial_noise, torch.zeros_like(initial_noise)
    )
    clean_state = torch.zeros_like(initial_noise)
    initial_noisy = torch.zeros_like(initial_noise)
    first_end = min(block_size, text_tokens)
    initial_noisy[:, :first_end] = initial_noise[:, :first_end]
    grid = _time_grid(config, alpha=config.text_alpha, seed=seed, device=device)
    text_timestep = torch.ones(batch_size, dtype=grid.dtype, device=device)
    clean_prompt = None
    prompt_mask = None
    prompt_position_offsets = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=device,
    )
    if text_prompt is not None:
        clean_prompt = torch.where(
            text_prompt.content_mask.unsqueeze(-1),
            text_prompt.latents_norm,
            torch.zeros_like(text_prompt.latents_norm),
        )
        prompt_mask = text_prompt.content_mask
        if block_session is not None:
            prompt_position_offsets = text_prompt.content_mask.sum(
                dim=1, dtype=torch.long
            )
    route_template = _model_input(
        model=model,
        vision_latents_norm=None,
        text_latents_norm=initial_noisy,
        text_prompt_latents_norm=clean_prompt,
        text_prompt_content_mask=prompt_mask,
        vision_timestep=torch.zeros_like(text_timestep),
        text_timestep=text_timestep,
        vision_role_value=BranchRole.ABSENT,
        text_role_value=BranchRole.TARGET,
        text_content_mask=text_target.content_mask,
        text_latent_stats_type=text_target.latent_stats_type,
        text_previous_x0_norm=torch.zeros_like(initial_noise),
        text_clean_latents_norm=clean_state,
        text_token_timestep=text_target.content_mask.to(dtype=grid.dtype),
        text_block_size=block_size,
    )
    routing_layout = (
        None
        if block_session is not None
        else model.prepare_routing_layout(route_template)
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for block_index, start in enumerate(range(0, text_tokens, block_size)):
        end = min(start + block_size, text_tokens)
        local_positions = torch.arange(start, end, device=device, dtype=torch.long)
        position_ids = (
            local_positions[None, None, :] + prompt_position_offsets[None, :, None]
        ).expand(3, -1, -1)
        block_mask = token_mask[:, start:end]
        if stop_at_eos_token_id is not None:
            block_mask = block_mask & ~finished[:, None, None]
        x_block = initial_noise[:, start:end]
        for step, (clean_t, next_clean_t) in enumerate(
            zip(grid[:-1], grid[1:], strict=True)
        ):
            solver_block, solver_t = _solver_input(
                x_block,
                clean_t,
                next_clean_t,
                config=config,
                global_sample_indices=indices,
                seed=seed,
                step=block_index * config.num_inference_steps + step,
                final_step=step == config.num_inference_steps - 1,
                noise_scale=config.text_noise_scale,
            )
            solver_block = torch.where(
                block_mask,
                solver_block,
                torch.zeros_like(solver_block),
            )
            noisy_state = torch.zeros_like(clean_state)
            noisy_state[:, start:end] = solver_block
            previous_state = torch.zeros_like(clean_state)
            current_timestep = solver_t.expand(batch_size)
            token_timestep = _block_token_timesteps(
                text_target.content_mask,
                start=start,
                end=end,
                current_timestep=current_timestep,
            )
            with _autocast(device, config.amp_dtype):
                if block_session is None:
                    model_input = replace(
                        route_template,
                        text_latents_norm=noisy_state,
                        text_clean_latents_norm=clean_state,
                        text_token_timestep=token_timestep,
                        text_previous_x0_norm=previous_state,
                    ).validate()
                    if routing_layout is None:
                        raise RuntimeError(
                            "uncached text generation is missing its routing layout"
                        )
                    output = model.predict_normalized(model_input, routing_layout)
                    x0_prediction = _prediction(output, "text_pred_norm", noisy_state)
                    x0_block = x0_prediction[:, start:end]
                else:
                    x0_block = block_session.predict_text_block(
                        solver_block,
                        previous_x0_norm=torch.zeros_like(solver_block),
                        token_timestep=current_timestep[:, None].expand(
                            -1, end - start
                        ),
                        position_ids=position_ids,
                    )
            x0_block = x0_block.float()
            x0_block = torch.where(block_mask, x0_block, torch.zeros_like(x0_block))
            x_block = advance_linear_path_x0(
                solver_block,
                x0_block,
                solver_t,
                next_clean_t,
                t_eps=config.velocity_t_eps,
            )
            x_block = torch.where(block_mask, x_block, torch.zeros_like(x_block))
        next_clean_state = clean_state.clone()
        next_clean_state[:, start:end] = x_block
        clean_state = next_clean_state
        if block_session is not None:
            with _autocast(device, config.amp_dtype):
                block_session.commit_text_block(x_block, position_ids=position_ids)

        if stop_at_eos_token_id is not None:
            block_stats = text_target.latent_stats_type[:, start:end]
            block_content_mask = text_target.content_mask[:, start:end]
            block_raw = model.denormalize_text_prediction(
                x_block,
                block_stats,
                block_content_mask,
            )
            block_decoder_input = (
                x_block
                if config.text_decoder_input_space == "normalized"
                else block_raw
            )
            block_logits = decoder(
                block_decoder_input, attention_mask=block_content_mask
            )
            if not isinstance(block_logits, Tensor) or tuple(
                block_logits.shape[:2]
            ) != (
                batch_size,
                end - start,
            ):
                raise ValueError(
                    "block text decoder must return logits with shape [B, K, vocab]"
                )
            block_tokens = block_logits.argmax(dim=-1)
            finished |= block_tokens.eq(stop_at_eos_token_id).any(dim=1)
            if bool(finished.all()):
                break

    result = _text_sample_from_normalized(
        model,
        clean_state,
        text_target,
        config,
        decoder,
        block_size=block_size,
    )
    if stop_at_eos_token_id is None:
        return result
    eos = result.token_ids.eq(stop_at_eos_token_id)
    positions = torch.arange(text_tokens, device=device).expand(batch_size, -1)
    first_eos = torch.where(eos, positions, text_tokens).amin(dim=1)
    stopped_tokens = torch.where(
        positions >= first_eos[:, None],
        torch.full_like(result.token_ids, stop_at_eos_token_id),
        result.token_ids,
    )
    return replace(result, token_ids=stopped_tokens)


@torch.inference_mode()
def sample_text(
    model: NormalizedX0Model,
    vision_condition: VisionCondition,
    text_prompt: TextCondition,
    text_target: TextTarget,
    config: SamplerConfig,
    seed: int,
    *,
    global_sample_indices: Sequence[int],
    decoder: TextDecoder,
    stop_at_eos_token_id: int | None = None,
    block_session_factory: TextBlockSessionFactory | None = None,
) -> TextSample:
    """Sample I2T with standard CFG between I2T and Text-only routes."""

    if not isinstance(config, SamplerConfig):
        raise TypeError("config must be a SamplerConfig")
    if not isinstance(vision_condition, VisionCondition):
        raise TypeError("vision_condition must be a VisionCondition")
    vision = vision_condition.latents_norm
    batch_size = vision.shape[0]
    if not isinstance(text_prompt, TextCondition):
        raise TypeError("text_prompt must be a TextCondition")
    device = vision.device
    text_target.validate(batch_size=batch_size, device=device)
    text_tokens = text_target.content_mask.shape[1]
    indices = tuple(global_sample_indices)
    if len(indices) != batch_size:
        raise ValueError("global_sample_indices must have one entry per condition")
    if text_prompt.latents_norm.shape[0] != batch_size:
        raise ValueError("text_prompt must have one row per vision condition")
    if text_prompt.latents_norm.device != device:
        raise ValueError("text_prompt and vision_condition must share a device")
    clean_prompt = torch.where(
        text_prompt.content_mask.unsqueeze(-1),
        text_prompt.latents_norm,
        torch.zeros_like(text_prompt.latents_norm),
    )
    block_size = _model_text_block_size(model)
    if block_size is not None:
        return _sample_text_block_causal(
            model,
            vision=vision,
            clean_prompt=clean_prompt,
            prompt_mask=text_prompt.content_mask,
            text_target=text_target,
            config=config,
            seed=seed,
            indices=indices,
            decoder=decoder,
            block_size=block_size,
            stop_at_eos_token_id=stop_at_eos_token_id,
            block_session_factory=block_session_factory,
        )
    token_mask = text_target.content_mask.unsqueeze(-1)
    x_t = sample_initial_noise(
        global_sample_indices=indices,
        latent_shape=(text_tokens, _model_text_latent_dim(model)),
        run_seed=seed,
        device=device,
        noise_scale=config.text_noise_scale,
    )
    x_t = torch.where(token_mask, x_t, torch.zeros_like(x_t))
    grid = _time_grid(config, alpha=config.text_alpha, seed=seed, device=device)
    initial_timestep = grid[0].expand(batch_size)
    conditional_template = _model_input(
        model=model,
        vision_latents_norm=vision,
        text_latents_norm=x_t,
        text_prompt_latents_norm=clean_prompt,
        text_prompt_content_mask=text_prompt.content_mask,
        vision_timestep=torch.ones_like(initial_timestep),
        text_timestep=initial_timestep,
        vision_role_value=BranchRole.CONDITION,
        text_role_value=BranchRole.TARGET,
        text_content_mask=text_target.content_mask,
        text_latent_stats_type=text_target.latent_stats_type,
        text_block_size=_model_text_block_size(model),
    )
    conditional_layout = model.prepare_routing_layout(conditional_template)
    unconditional_template = _model_input(
        model=model,
        vision_latents_norm=torch.zeros_like(vision),
        text_latents_norm=x_t,
        text_prompt_latents_norm=torch.zeros_like(clean_prompt),
        text_prompt_content_mask=text_prompt.content_mask,
        vision_timestep=torch.ones_like(initial_timestep),
        text_timestep=initial_timestep,
        vision_role_value=BranchRole.CONDITION,
        text_role_value=BranchRole.TARGET,
        text_content_mask=text_target.content_mask,
        text_latent_stats_type=text_target.latent_stats_type,
        null_conditioning=torch.ones(
            batch_size,
            dtype=torch.bool,
            device=device,
        ),
        text_block_size=_model_text_block_size(model),
    )
    unconditional_layout = model.prepare_routing_layout(unconditional_template)

    for step, (clean_t, next_clean_t) in enumerate(
        zip(grid[:-1], grid[1:], strict=True)
    ):
        solver_x, solver_t = _solver_input(
            x_t,
            clean_t,
            next_clean_t,
            config=config,
            global_sample_indices=indices,
            seed=seed,
            step=step,
            final_step=step == config.num_inference_steps - 1,
            noise_scale=config.text_noise_scale,
        )
        solver_x = torch.where(token_mask, solver_x, torch.zeros_like(solver_x))
        text_timestep = solver_t.expand(batch_size)
        previous_state = torch.zeros_like(solver_x)
        conditional = replace(
            conditional_template,
            text_latents_norm=solver_x,
            text_timestep=text_timestep,
            text_previous_x0_norm=previous_state,
        ).validate()
        unconditional = replace(
            unconditional_template,
            text_latents_norm=solver_x,
            text_timestep=text_timestep,
            text_previous_x0_norm=previous_state,
        ).validate()
        with _autocast(device, config.amp_dtype):
            conditional_output = model.predict_normalized(
                conditional, conditional_layout
            )
            unconditional_output = model.predict_normalized(
                unconditional,
                unconditional_layout,
            )
        x0_conditional = _prediction(
            conditional_output,
            "text_pred_norm",
            solver_x,
        ).float()
        x0_unconditional = _prediction(
            unconditional_output,
            "text_pred_norm",
            solver_x,
        ).float()
        x0_guided = x0_unconditional + config.cfg_scale * (
            x0_conditional - x0_unconditional
        )
        x_t = advance_linear_path_x0(
            solver_x,
            x0_guided,
            solver_t,
            next_clean_t,
            t_eps=config.velocity_t_eps,
        )
        x_t = torch.where(token_mask, x_t, torch.zeros_like(x_t))

    normalized = x_t
    raw = model.denormalize_text_prediction(
        normalized,
        text_target.latent_stats_type,
        text_target.content_mask,
    )
    decoder_latents = (
        normalized if config.text_decoder_input_space == "normalized" else raw
    )
    logits = decoder(decoder_latents, attention_mask=text_target.content_mask)
    if not isinstance(logits, Tensor) or tuple(logits.shape[:2]) != (
        batch_size,
        text_tokens,
    ):
        raise ValueError("text decoder must return logits with shape [B, T, vocab]")
    return TextSample(
        normalized_latents=normalized,
        raw_latents=raw,
        logits=logits,
        token_ids=logits.argmax(dim=-1),
    )


@torch.inference_mode()
def sample_text_continuation(
    model: NormalizedX0Model,
    text_prompt: TextCondition,
    text_target: TextTarget,
    config: SamplerConfig,
    seed: int,
    *,
    global_sample_indices: Sequence[int],
    decoder: TextDecoder,
    stop_at_eos_token_id: int | None = None,
    block_session_factory: TextBlockSessionFactory | None = None,
) -> TextSample:
    """Generate a text suffix from a clean prompt with the block-causal solver."""

    if not isinstance(config, SamplerConfig):
        raise TypeError("config must be a SamplerConfig")
    if not isinstance(text_prompt, TextCondition):
        raise TypeError("text_prompt must be a TextCondition")
    if getattr(model, "sequence_layout", None) != "chunk_causal":
        raise ValueError(
            "prompt-conditioned text continuation requires chunk_causal layout"
        )
    block_size = _model_text_block_size(model)
    if block_size is None:
        raise ValueError(
            "prompt-conditioned text continuation requires block-causal text"
        )
    batch_size = text_prompt.latents_norm.shape[0]
    device = text_prompt.latents_norm.device
    if not bool(text_prompt.content_mask.any(dim=1).all()):
        raise ValueError("every continuation request requires a non-empty prompt")
    text_target.validate(batch_size=batch_size, device=device)
    indices = tuple(global_sample_indices)
    if len(indices) != batch_size:
        raise ValueError("global_sample_indices must have one entry per prompt")
    if stop_at_eos_token_id is not None and (
        isinstance(stop_at_eos_token_id, bool)
        or not isinstance(stop_at_eos_token_id, int)
        or stop_at_eos_token_id < 0
    ):
        raise ValueError("stop_at_eos_token_id must be a non-negative integer or None")
    target_tokens = text_target.content_mask.shape[1]
    if block_session_factory is not None and not bool(text_target.content_mask.all()):
        raise ValueError("cached continuation requires a dense fixed-length target")
    block_session = (
        None
        if block_session_factory is None
        else block_session_factory(
            batch_size=batch_size,
            max_cache_len=text_prompt.content_mask.shape[1] + target_tokens,
            device=device,
            dtype=config.amp_dtype or torch.float32,
        )
    )
    if block_session is not None:
        with _autocast(device, config.amp_dtype):
            block_session.prefill_text_condition(
                text_prompt.latents_norm,
                position_base=0,
                content_mask=text_prompt.content_mask,
            )
    return _sample_text_unconditional_block_causal(
        model,
        text_target=text_target,
        config=config,
        seed=seed,
        indices=indices,
        decoder=decoder,
        block_size=block_size,
        text_prompt=text_prompt,
        stop_at_eos_token_id=stop_at_eos_token_id,
        block_session=block_session,
    )


@torch.inference_mode()
def sample_text_unconditional(
    model: NormalizedX0Model,
    text_target: TextTarget,
    config: SamplerConfig,
    seed: int,
    *,
    global_sample_indices: Sequence[int],
    decoder: TextDecoder,
    block_session_factory: TextBlockSessionFactory | None = None,
) -> TextSample:
    """Sample unconditional text with the token-native guidance path."""

    if not isinstance(config, SamplerConfig):
        raise TypeError("config must be a SamplerConfig")
    indices = tuple(global_sample_indices)
    if not indices:
        raise ValueError("global_sample_indices must not be empty")
    batch_size = len(indices)
    device = text_target.content_mask.device
    text_target.validate(batch_size=batch_size, device=device)
    text_tokens = text_target.content_mask.shape[1]
    block_size = _model_text_block_size(model)
    if block_size is not None:
        if block_session_factory is not None and not bool(
            text_target.content_mask.all()
        ):
            raise ValueError(
                "cached block generation currently requires a dense fixed-length target"
            )
        block_session = (
            None
            if block_session_factory is None
            else block_session_factory(
                batch_size=batch_size,
                max_cache_len=text_tokens,
                device=device,
                dtype=config.amp_dtype or torch.float32,
            )
        )
        return _sample_text_unconditional_block_causal(
            model,
            text_target=text_target,
            config=config,
            seed=seed,
            indices=indices,
            decoder=decoder,
            block_size=block_size,
            block_session=block_session,
        )
    token_mask = text_target.content_mask.unsqueeze(-1)
    x_t = sample_initial_noise(
        global_sample_indices=indices,
        latent_shape=(text_tokens, _model_text_latent_dim(model)),
        run_seed=seed,
        device=device,
        noise_scale=config.text_noise_scale,
    )
    x_t = torch.where(token_mask, x_t, torch.zeros_like(x_t))
    grid = _time_grid(config, alpha=config.text_alpha, seed=seed, device=device)
    initial_timestep = grid[0].expand(batch_size)
    route_template = _model_input(
        model=model,
        vision_latents_norm=None,
        text_latents_norm=x_t,
        vision_timestep=torch.zeros_like(initial_timestep),
        text_timestep=initial_timestep,
        vision_role_value=BranchRole.ABSENT,
        text_role_value=BranchRole.TARGET,
        text_content_mask=text_target.content_mask,
        text_latent_stats_type=text_target.latent_stats_type,
        text_previous_x0_norm=torch.zeros_like(x_t),
        text_block_size=_model_text_block_size(model),
    )
    routing_layout = model.prepare_routing_layout(route_template)
    for step, (clean_t, next_clean_t) in enumerate(
        zip(grid[:-1], grid[1:], strict=True)
    ):
        solver_x, solver_t = _solver_input(
            x_t,
            clean_t,
            next_clean_t,
            config=config,
            global_sample_indices=indices,
            seed=seed,
            step=step,
            final_step=step == config.num_inference_steps - 1,
            noise_scale=config.text_noise_scale,
        )
        solver_x = torch.where(token_mask, solver_x, torch.zeros_like(solver_x))
        text_timestep = solver_t.expand(batch_size)
        previous_state = torch.zeros_like(solver_x)
        model_input = replace(
            route_template,
            text_latents_norm=solver_x,
            text_timestep=text_timestep,
            text_previous_x0_norm=previous_state,
        ).validate()
        with _autocast(device, config.amp_dtype):
            output = model.predict_normalized(model_input, routing_layout)
        x0_prediction = _prediction(output, "text_pred_norm", solver_x).float()
        x_t = advance_linear_path_x0(
            solver_x,
            x0_prediction,
            solver_t,
            next_clean_t,
            t_eps=config.velocity_t_eps,
        )
        x_t = torch.where(token_mask, x_t, torch.zeros_like(x_t))

    normalized = x_t
    raw = model.denormalize_text_prediction(
        normalized,
        text_target.latent_stats_type,
        text_target.content_mask,
    )
    decoder_latents = (
        normalized if config.text_decoder_input_space == "normalized" else raw
    )
    logits = decoder(decoder_latents, attention_mask=text_target.content_mask)
    if not isinstance(logits, Tensor) or tuple(logits.shape[:2]) != (
        batch_size,
        text_tokens,
    ):
        raise ValueError("text decoder must return logits with shape [B, T, vocab]")
    return TextSample(
        normalized_latents=normalized,
        raw_latents=raw,
        logits=logits,
        token_ids=logits.argmax(dim=-1),
    )
