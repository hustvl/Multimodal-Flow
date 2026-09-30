from enum import IntEnum

import torch
from torch import Tensor, nn

DEFAULT_VISION_STATS_TOKENS = 256
DEFAULT_TEXT_LATENT_DIM = 512
_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)
_STATS_BUFFER_NAMES = frozenset(
    {
        "vision_mean",
        "vision_std",
        "text_normal_mean",
        "text_normal_std",
    }
)


class TextLatentStatsType(IntEnum):
    """Stable routing identifiers stored in text_latent_stats_type tensors."""

    NORMAL_TEXT = 0
    EOS = 1
    PAD_IGNORE = 2


def _validated_stat(
    name: str,
    value: object,
    expected_shape: tuple[int, ...],
    *,
    is_std: bool,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {list(expected_shape)}; got {list(value.shape)}"
        )
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must have a floating-point dtype; got {value.dtype}")
    if value.device.type == "meta":
        raise ValueError(f"{name} must be materialized so its values can be validated")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    if is_std and not bool((value > 0).all()):
        raise ValueError(f"{name} must contain strictly positive values")
    return value.detach().clone()


def _require_floating_tensor(name: str, value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must have a floating-point dtype; got {value.dtype}")
    return value


def _require_tensor_condition(condition: Tensor, message: str) -> None:
    try:
        torch._assert_async(condition, message)
    except RuntimeError as error:
        raise ValueError(message) from error


def _promoted_compute_dtype(*tensors: Tensor) -> torch.dtype:
    compute_dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        compute_dtype = torch.promote_types(compute_dtype, tensor.dtype)
    if compute_dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return compute_dtype


class LatentStatsRegistry(nn.Module):
    """Validated latent statistics stored as detached, non-parameter buffers."""

    def __init__(
        self,
        vision_mean: Tensor,
        vision_std: Tensor,
        text_normal_mean: Tensor,
        text_normal_std: Tensor,
    ) -> None:
        super().__init__()
        if (
            not isinstance(vision_mean, Tensor)
            or vision_mean.ndim != 2
            or vision_mean.shape[0] <= 0
            or vision_mean.shape[1] <= 0
        ):
            vision_shape_description = (
                list(vision_mean.shape)
                if isinstance(vision_mean, Tensor)
                else type(vision_mean).__name__
            )
            raise ValueError(
                "vision_mean must have shape [N, D] with N, D > 0; "
                f"got {vision_shape_description}"
            )
        vision_shape = tuple(vision_mean.shape)
        if (
            not isinstance(text_normal_mean, Tensor)
            or not isinstance(text_normal_std, Tensor)
            or text_normal_mean.ndim != 1
            or text_normal_std.shape != text_normal_mean.shape
            or text_normal_mean.shape[0] <= 0
        ):
            raise ValueError(
                "text statistics must be matching one-dimensional tensors with "
                f"a positive latent dimension; got "
                f"{list(text_normal_mean.shape)} and {list(text_normal_std.shape)}"
            )
        text_shape = tuple(text_normal_mean.shape)
        self.vision_latent_dim = vision_shape[1]
        self.vision_tokens = vision_shape[0]
        self.text_latent_dim = text_shape[0]
        stats = {
            "vision_mean": _validated_stat(
                "vision_mean", vision_mean, vision_shape, is_std=False
            ),
            "vision_std": _validated_stat(
                "vision_std", vision_std, vision_shape, is_std=True
            ),
            "text_normal_mean": _validated_stat(
                "text_normal_mean", text_normal_mean, text_shape, is_std=False
            ),
            "text_normal_std": _validated_stat(
                "text_normal_std", text_normal_std, text_shape, is_std=True
            ),
        }
        devices = {tensor.device for tensor in stats.values()}
        if len(devices) != 1:
            raise ValueError("all latent stats must be on the same device")
        for name, tensor in stats.items():
            self.register_buffer(name, tensor)

    def __setattr__(self, name: str, value: object) -> None:
        buffers = self.__dict__.get("_buffers", {})
        if name in _STATS_BUFFER_NAMES and name in buffers:
            raise AttributeError(f"{name} is an immutable stats buffer")
        super().__setattr__(name, value)

    def _require_registry_device(self, name: str, tensor: Tensor) -> None:
        if tensor.device != self.vision_mean.device:
            raise ValueError(
                f"{name} must be on the registry device; "
                f"got {tensor.device} and {self.vision_mean.device}"
            )

    def _validate_vision(self, value: object) -> Tensor:
        tensor = _require_floating_tensor("vision_latents", value)
        expected = tuple(self.vision_mean.shape)
        if tensor.ndim != 3 or tuple(tensor.shape[1:]) != expected:
            raise ValueError(
                f"vision_latents must have shape [B, {expected[0]}, {expected[1]}]; "
                f"got {list(tensor.shape)}"
            )
        self._require_registry_device("vision_latents", tensor)
        return tensor

    def normalize_vision(self, vision_latents_raw: Tensor) -> Tensor:
        raw = self._validate_vision(vision_latents_raw)
        compute_dtype = _promoted_compute_dtype(raw, self.vision_mean, self.vision_std)
        raw_compute = raw.to(dtype=compute_dtype)
        mean = self.vision_mean.to(dtype=compute_dtype)
        std = self.vision_std.to(dtype=compute_dtype)
        return ((raw_compute - mean) / std).to(dtype=raw.dtype)

    def denormalize_vision(self, vision_latents_norm: Tensor) -> Tensor:
        normalized = self._validate_vision(vision_latents_norm)
        compute_dtype = _promoted_compute_dtype(
            normalized, self.vision_mean, self.vision_std
        )
        normalized_compute = normalized.to(dtype=compute_dtype)
        mean = self.vision_mean.to(dtype=compute_dtype)
        std = self.vision_std.to(dtype=compute_dtype)
        return (normalized_compute * std + mean).to(dtype=normalized.dtype)

    def _validate_text(
        self,
        value: object,
        stats_type: object,
        content_mask: object,
    ) -> tuple[Tensor, Tensor]:
        tensor = _require_floating_tensor("text_latents", value)
        if (
            tensor.ndim != 3
            or tensor.shape[1] <= 0
            or tensor.shape[2] != self.text_latent_dim
        ):
            raise ValueError(
                "text_latents must have shape "
                f"[B, T, {self.text_latent_dim}] with T > 0; "
                f"got {list(tensor.shape)}"
            )
        text_tokens = tensor.shape[1]
        if not isinstance(stats_type, Tensor):
            raise ValueError("text_latent_stats_type must be a torch.Tensor")
        if tuple(stats_type.shape) != tuple(tensor.shape[:2]):
            raise ValueError(
                f"text_latent_stats_type must have shape [B, {text_tokens}]; "
                f"got {list(stats_type.shape)}"
            )
        if stats_type.dtype not in _INTEGER_DTYPES:
            raise ValueError(
                f"text_latent_stats_type must have an integer dtype; got {stats_type.dtype}"
            )
        if not isinstance(content_mask, Tensor):
            raise ValueError("text_content_mask must be a torch.Tensor")
        if tuple(content_mask.shape) != tuple(tensor.shape[:2]):
            raise ValueError(
                f"text_content_mask must have shape [B, {text_tokens}]; "
                f"got {list(content_mask.shape)}"
            )
        if content_mask.dtype is not torch.bool:
            raise ValueError(
                f"text_content_mask must have dtype torch.bool; got {content_mask.dtype}"
            )
        self._require_registry_device("text_latents", tensor)
        for name, routing_tensor in (
            ("text_latent_stats_type", stats_type),
            ("text_content_mask", content_mask),
        ):
            if routing_tensor.device != tensor.device:
                raise ValueError(
                    f"{name} must be on the same device as text_latents; "
                    f"got {routing_tensor.device} and {tensor.device}"
                )

        normal_mask = stats_type == int(TextLatentStatsType.NORMAL_TEXT)
        eos_mask = stats_type == int(TextLatentStatsType.EOS)
        pad_mask = stats_type == int(TextLatentStatsType.PAD_IGNORE)
        unknown_mask = ~(normal_mask | eos_mask | pad_mask)
        _require_tensor_condition(
            ~unknown_mask.any(),
            "text_latent_stats_type has an unknown value; expected 0, 1, or 2",
        )
        _require_tensor_condition(
            ~(content_mask & pad_mask).any(),
            "active text tokens cannot use PAD_IGNORE stats type",
        )
        _require_tensor_condition(
            ~(~content_mask & ~pad_mask).any(),
            "inactive text tokens must use PAD_IGNORE stats type",
        )
        return tensor, content_mask

    def _transform_text(
        self,
        value: Tensor,
        stats_type: Tensor,
        content_mask: Tensor,
        *,
        denormalize: bool,
    ) -> Tensor:
        tensor, content_mask = self._validate_text(value, stats_type, content_mask)
        compute_dtype = _promoted_compute_dtype(
            tensor,
            self.text_normal_mean,
            self.text_normal_std,
        )
        tensor_compute = tensor.to(dtype=compute_dtype)
        mean = self.text_normal_mean.to(dtype=compute_dtype)
        std = self.text_normal_std.to(dtype=compute_dtype)
        transformed = (
            tensor_compute * std + mean
            if denormalize
            else (tensor_compute - mean) / std
        )
        result = torch.where(
            content_mask.unsqueeze(-1), transformed, torch.zeros_like(transformed)
        )
        return result.to(dtype=tensor.dtype)

    def normalize_text(
        self,
        text_latents_raw: Tensor,
        text_latent_stats_type: Tensor,
        text_content_mask: Tensor,
    ) -> Tensor:
        return self._transform_text(
            text_latents_raw,
            text_latent_stats_type,
            text_content_mask,
            denormalize=False,
        )

    def denormalize_text(
        self,
        text_latents_norm: Tensor,
        text_latent_stats_type: Tensor,
        text_content_mask: Tensor,
    ) -> Tensor:
        return self._transform_text(
            text_latents_norm,
            text_latent_stats_type,
            text_content_mask,
            denormalize=True,
        )
