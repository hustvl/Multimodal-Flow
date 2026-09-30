"""Typed requests and results for the public multimodal inference API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import ClassVar, Literal, TypeAlias

import torch
from torch import Tensor

from mf.evaluation.sampling import SamplerConfig
from mf.contracts.physical import PhysicalSequenceLayout
from mf.inference.loading import CheckpointConfig, HFCheckpointConfig
from mf.instructions import IMAGE_CAPTION_PROMPT


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling parameters a caller may reasonably vary.

    Checkpoint-dependent settings such as ``velocity_t_eps`` are read from the
    checkpoint rather than changed independently during generation.
    """

    num_inference_steps: int | None = None
    cfg_scale: float | None = None
    method: Literal["ode", "sde"] | None = None
    sde_gamma: float | None = None
    image_alpha: float | None = None
    text_alpha: float | None = None
    t_lognorm_mu: float | None = None
    t_lognorm_sigma: float | None = None
    vision_noise_scale: float | None = None
    text_noise_scale: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_inference_steps is not None and (
            type(self.num_inference_steps) is not int or self.num_inference_steps <= 0
        ):
            raise ValueError("num_inference_steps must be a positive integer")
        if self.cfg_scale is not None and self.cfg_scale < 0.0:
            raise ValueError("cfg_scale must be non-negative")
        if self.method is not None and self.method not in ("ode", "sde"):
            raise ValueError("method must be 'ode' or 'sde'")
        if self.sde_gamma is not None and self.sde_gamma < 0.0:
            raise ValueError("sde_gamma must be non-negative")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class PhysicalGenerationRequest:
    """A semantic physical sequence handed to a registered generation loop."""

    physical_layout: PhysicalSequenceLayout
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.physical_layout, PhysicalSequenceLayout):
            raise TypeError("physical_layout must be a PhysicalSequenceLayout")
        self.physical_layout.validate()
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))


def checkpoint_sampler_config(config: CheckpointConfig) -> SamplerConfig:
    """Rebuild the sampler settings the checkpoint's own evaluation would use."""

    if isinstance(config, HFCheckpointConfig):
        sampling = config.generation
        num_inference_steps = sampling.image_num_inference_steps
    else:
        sampling = config.evaluation.sampling
        num_inference_steps = sampling.num_inference_steps
    shift = config.flow.timestep_shift
    return SamplerConfig(
        num_inference_steps=num_inference_steps,
        vision_latent_dim=config.codecs.vision.latent_dim,
        cfg_scale=sampling.cfg_scale,
        method=sampling.method,
        sde_gamma=sampling.sde_gamma,
        image_alpha=shift.image_alpha,
        text_alpha=shift.text_alpha,
        t_lognorm_mu=shift.t_lognorm_mu,
        t_lognorm_sigma=shift.t_lognorm_sigma,
        vision_noise_scale=config.flow.vision_noise_scale,
        text_noise_scale=config.flow.text_noise_scale,
        velocity_t_eps=config.flow.velocity_t_eps,
        amp_dtype=torch.bfloat16,
        text_decoder_input_space=config.model.text_decoder.input_space,
    )


def apply_overrides(
    base: SamplerConfig, overrides: GenerationConfig | None
) -> SamplerConfig:
    """Overlay the caller's non-``None`` choices onto the checkpoint defaults."""

    if overrides is None:
        return base
    updates = {
        name: value
        for name, value in (
            ("num_inference_steps", overrides.num_inference_steps),
            ("cfg_scale", overrides.cfg_scale),
            ("method", overrides.method),
            ("sde_gamma", overrides.sde_gamma),
            ("image_alpha", overrides.image_alpha),
            ("text_alpha", overrides.text_alpha),
            ("t_lognorm_mu", overrides.t_lognorm_mu),
            ("t_lognorm_sigma", overrides.t_lognorm_sigma),
            ("vision_noise_scale", overrides.vision_noise_scale),
            ("text_noise_scale", overrides.text_noise_scale),
        )
        if value is not None
    }
    return replace(base, **updates) if updates else base


class InferenceTask(str, Enum):
    """Stable task identifiers suitable for local and serving dispatch."""

    TEXT = "text"
    TEXT_TO_TEXT = "text-to-text"
    IMAGE_TO_TEXT = "image-to-text"
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE = "image"


def _require_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_start_index(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("start_index must be a non-negative integer")


def _normalize_strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    return normalized


def _validate_common(start_index: int, max_batch_size: int | None = None) -> None:
    _require_start_index(start_index)
    if max_batch_size is not None:
        _require_positive_integer(max_batch_size, "max_batch_size")


def _normalize_request_indices(
    values: Sequence[int] | None,
    *,
    start_index: int,
    count: int,
) -> tuple[int, ...] | None:
    if values is None:
        return None
    if start_index != 0:
        raise ValueError(
            "sample_indices cannot be combined with a non-zero start_index"
        )
    if isinstance(values, (str, bytes)):
        raise ValueError("sample_indices must be a sequence of non-negative integers")
    indices = tuple(values)
    if len(indices) != count:
        raise ValueError("sample_indices must contain one entry per request sample")
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("sample_indices must contain non-negative integers")
    return indices


def _resolved_indices(
    values: Sequence[int] | None,
    *,
    start_index: int,
    count: int,
) -> tuple[int, ...]:
    return (
        tuple(range(start_index, start_index + count))
        if values is None
        else tuple(values)
    )


@dataclass(frozen=True, slots=True)
class TextGenerationRequest:
    """Generate text without a user-provided prefix."""

    num_samples: int = 1
    config: GenerationConfig | None = None
    start_index: int = 0
    sample_indices: Sequence[int] | None = None
    max_batch_size: int = 8
    use_cache: bool | None = None

    task: ClassVar[InferenceTask] = InferenceTask.TEXT

    def __post_init__(self) -> None:
        _require_positive_integer(self.num_samples, "num_samples")
        _validate_common(self.start_index, self.max_batch_size)
        object.__setattr__(
            self,
            "sample_indices",
            _normalize_request_indices(
                self.sample_indices,
                start_index=self.start_index,
                count=self.num_samples,
            ),
        )

    @property
    def resolved_sample_indices(self) -> tuple[int, ...]:
        return _resolved_indices(
            self.sample_indices,
            start_index=self.start_index,
            count=self.num_samples,
        )


@dataclass(frozen=True, slots=True)
class TextToTextRequest:
    """Continue one or more text prefixes."""

    prompts: Sequence[str]
    config: GenerationConfig | None = None
    start_index: int = 0
    sample_indices: Sequence[int] | None = None
    target_length: int | None = None
    max_prompt_tokens: int | None = None
    prompt_truncation: Literal["error", "left_keep_suffix"] = "left_keep_suffix"
    stop: Sequence[str] = ()
    max_batch_size: int = 8
    use_cache: bool | None = None

    task: ClassVar[InferenceTask] = InferenceTask.TEXT_TO_TEXT

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompts", _normalize_strings(self.prompts, "prompts"))
        if isinstance(self.stop, (str, bytes)):
            raise TypeError("stop must be a sequence of strings")
        normalized_stop = tuple(self.stop)
        if any(not isinstance(value, str) or not value for value in normalized_stop):
            raise ValueError("stop must contain non-empty strings")
        object.__setattr__(self, "stop", normalized_stop)
        _validate_common(self.start_index, self.max_batch_size)
        object.__setattr__(
            self,
            "sample_indices",
            _normalize_request_indices(
                self.sample_indices,
                start_index=self.start_index,
                count=len(self.prompts),
            ),
        )
        if self.target_length is not None:
            _require_positive_integer(self.target_length, "target_length")
        if self.max_prompt_tokens is not None:
            _require_positive_integer(self.max_prompt_tokens, "max_prompt_tokens")
        if self.prompt_truncation not in ("error", "left_keep_suffix"):
            raise ValueError("prompt_truncation must be 'error' or 'left_keep_suffix'")

    @property
    def resolved_sample_indices(self) -> tuple[int, ...]:
        return _resolved_indices(
            self.sample_indices,
            start_index=self.start_index,
            count=len(self.prompts),
        )


@dataclass(frozen=True, slots=True)
class ImageToTextRequest:
    """Generate text conditioned on one or more images."""

    images: Sequence[Tensor]
    config: GenerationConfig | None = None
    start_index: int = 0
    sample_indices: Sequence[int] | None = None
    target_length: int | None = None
    prompt: str | Sequence[str] = IMAGE_CAPTION_PROMPT
    max_prompt_tokens: int | None = None
    stop_at_eos: bool | None = None
    skip_special_tokens: bool = True
    use_cache: bool | None = None

    task: ClassVar[InferenceTask] = InferenceTask.IMAGE_TO_TEXT

    def __post_init__(self) -> None:
        if isinstance(self.images, Tensor):
            raise TypeError("images must be a sequence of unbatched tensors")
        images = tuple(self.images)
        if not images:
            raise ValueError("images must not be empty")
        if any(not isinstance(image, Tensor) for image in images):
            raise TypeError("images must contain tensors")
        object.__setattr__(self, "images", images)
        _validate_common(self.start_index)
        object.__setattr__(
            self,
            "sample_indices",
            _normalize_request_indices(
                self.sample_indices,
                start_index=self.start_index,
                count=len(images),
            ),
        )
        if self.target_length is not None:
            _require_positive_integer(self.target_length, "target_length")
        if isinstance(self.prompt, str):
            if not self.prompt:
                raise ValueError("prompt must not be empty")
        else:
            prompts = _normalize_strings(self.prompt, "prompt")
            if len(prompts) != len(images):
                raise ValueError("prompt must contain one string per image")
            object.__setattr__(self, "prompt", prompts)
        if self.max_prompt_tokens is not None:
            _require_positive_integer(self.max_prompt_tokens, "max_prompt_tokens")

    @property
    def resolved_prompts(self) -> tuple[str, ...]:
        if isinstance(self.prompt, str):
            return (self.prompt,) * len(self.images)
        return tuple(self.prompt)

    @property
    def resolved_sample_indices(self) -> tuple[int, ...]:
        return _resolved_indices(
            self.sample_indices,
            start_index=self.start_index,
            count=len(self.images),
        )


@dataclass(frozen=True, slots=True)
class TextToImageRequest:
    """Generate images conditioned on text prompts."""

    prompts: Sequence[str]
    config: GenerationConfig | None = None
    start_index: int = 0
    sample_indices: Sequence[int] | None = None

    task: ClassVar[InferenceTask] = InferenceTask.TEXT_TO_IMAGE

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompts", _normalize_strings(self.prompts, "prompts"))
        _validate_common(self.start_index)
        object.__setattr__(
            self,
            "sample_indices",
            _normalize_request_indices(
                self.sample_indices,
                start_index=self.start_index,
                count=len(self.prompts),
            ),
        )

    @property
    def resolved_sample_indices(self) -> tuple[int, ...]:
        return _resolved_indices(
            self.sample_indices,
            start_index=self.start_index,
            count=len(self.prompts),
        )


@dataclass(frozen=True, slots=True)
class ImageGenerationRequest:
    """Generate images without a text condition."""

    num_samples: int = 1
    config: GenerationConfig | None = None
    start_index: int = 0
    sample_indices: Sequence[int] | None = None

    task: ClassVar[InferenceTask] = InferenceTask.IMAGE

    def __post_init__(self) -> None:
        _require_positive_integer(self.num_samples, "num_samples")
        _validate_common(self.start_index)
        object.__setattr__(
            self,
            "sample_indices",
            _normalize_request_indices(
                self.sample_indices,
                start_index=self.start_index,
                count=self.num_samples,
            ),
        )

    @property
    def resolved_sample_indices(self) -> tuple[int, ...]:
        return _resolved_indices(
            self.sample_indices,
            start_index=self.start_index,
            count=self.num_samples,
        )


@dataclass(frozen=True, slots=True)
class TextGenerationResult:
    """Text outputs with deterministic global sample indices."""

    task: InferenceTask
    texts: Sequence[str]
    sample_indices: Sequence[int]

    def __post_init__(self) -> None:
        task = InferenceTask(self.task)
        if task not in {
            InferenceTask.TEXT,
            InferenceTask.TEXT_TO_TEXT,
            InferenceTask.IMAGE_TO_TEXT,
        }:
            raise ValueError(f"{task.value} is not a text task")
        texts = tuple(self.texts)
        sample_indices = tuple(self.sample_indices)
        if len(texts) != len(sample_indices):
            raise ValueError("sample_indices must align one-to-one with texts")
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "texts", texts)
        object.__setattr__(self, "sample_indices", sample_indices)


@dataclass(frozen=True, slots=True)
class ImageGenerationResult:
    """Batched image outputs with deterministic global sample indices."""

    task: InferenceTask
    images: Tensor
    sample_indices: Sequence[int]

    def __post_init__(self) -> None:
        task = InferenceTask(self.task)
        if task not in {InferenceTask.TEXT_TO_IMAGE, InferenceTask.IMAGE}:
            raise ValueError(f"{task.value} is not an image task")
        if not isinstance(self.images, Tensor) or self.images.ndim < 1:
            raise TypeError("images must be a batched tensor")
        sample_indices = tuple(self.sample_indices)
        if self.images.shape[0] != len(sample_indices):
            raise ValueError("sample_indices must align one-to-one with images")
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "sample_indices", sample_indices)


InferenceRequest: TypeAlias = (
    TextGenerationRequest
    | TextToTextRequest
    | ImageToTextRequest
    | TextToImageRequest
    | ImageGenerationRequest
)
InferenceResult: TypeAlias = TextGenerationResult | ImageGenerationResult


__all__ = [
    "GenerationConfig",
    "apply_overrides",
    "checkpoint_sampler_config",
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "ImageToTextRequest",
    "PhysicalGenerationRequest",
    "InferenceRequest",
    "InferenceResult",
    "InferenceTask",
    "TextGenerationRequest",
    "TextGenerationResult",
    "TextToImageRequest",
    "TextToTextRequest",
]
