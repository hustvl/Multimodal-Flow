from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mf.codecs.frozen import FrozenCodec, preprocessing_buffer, require_local_encoder
from mf.codecs.rae_decoder import RAEDecoder
from mf.contracts.batch import VISION_LATENT_DIM, VISION_TOKENS

_IMAGE_SIZE = 256
_DEFAULT_IMAGE_MEAN = (0.5, 0.5, 0.5)
_DEFAULT_IMAGE_STD = (0.5, 0.5, 0.5)
_MISSING = object()
_LOADED_CONFIG = {
    "hidden_size": VISION_LATENT_DIM,
    "image_size": _IMAGE_SIZE,
    "patch_size": 16,
}


def _load_siglip2_encoder(
    model_path: Path,
) -> tuple[nn.Module, Sequence[float], Sequence[float]]:
    require_local_encoder(model_path, name="SigLIP2 RAE")
    from transformers import AutoImageProcessor, SiglipModel

    processor = AutoImageProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=False,
    )
    model = SiglipModel.from_pretrained(model_path, local_files_only=True).vision_model
    return model, processor.image_mean, processor.image_std


def _check_encoder_config(model: nn.Module) -> None:
    config = getattr(model, "config", _MISSING)
    for name, value in _LOADED_CONFIG.items():
        actual = _MISSING if config is _MISSING else getattr(config, name, _MISSING)
        if actual is _MISSING:
            raise ValueError(
                f"SigLIP2 RAE encoder config is missing required metadata {name}"
            )
        if actual != value:
            raise ValueError(
                f"SigLIP2 RAE encoder config {name} must be {value}; got {actual}"
            )


def _remove_output_layernorm_affine(model: nn.Module) -> None:
    layernorm = getattr(model, "post_layernorm", _MISSING)
    if not isinstance(layernorm, nn.LayerNorm):
        raise ValueError(
            "SigLIP2 RAE encoder must expose its final LayerNorm as model.post_layernorm"
        )
    layernorm.elementwise_affine = False
    layernorm.weight = None
    layernorm.bias = None


class SigLIP2RAEEncoder(FrozenCodec):
    """Frozen no-affine SigLIP2-base encoder used by the matching RAE."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        model: nn.Module | None = None,
        image_mean: Sequence[float] | Tensor | None = None,
        image_std: Sequence[float] | Tensor | None = None,
    ) -> None:
        super().__init__()
        if model is None:
            model, loaded_mean, loaded_std = _load_siglip2_encoder(Path(model_path))
            image_mean = loaded_mean if image_mean is None else image_mean
            image_std = loaded_std if image_std is None else image_std
        self.model = model
        _check_encoder_config(self.model)
        _remove_output_layernorm_affine(self.model)
        self.image_mean = preprocessing_buffer(
            _DEFAULT_IMAGE_MEAN if image_mean is None else image_mean,
            name="image_mean",
        )
        self.image_std = preprocessing_buffer(
            _DEFAULT_IMAGE_STD if image_std is None else image_std,
            name="image_std",
        )
        self.freeze()

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"images must have shape [N, 3, H, W]; got {list(images.shape)}"
            )
        if images.shape[-2:] != (_IMAGE_SIZE, _IMAGE_SIZE):
            images = F.interpolate(
                images,
                size=(_IMAGE_SIZE, _IMAGE_SIZE),
                mode="bicubic",
                align_corners=False,
            )
        mean = self.image_mean.to(device=images.device, dtype=torch.float32)
        std = self.image_std.to(device=images.device, dtype=torch.float32)
        pixel_values = (images - mean) / std
        parameter = next(self.model.parameters(), None)
        if parameter is not None and parameter.is_floating_point():
            pixel_values = pixel_values.to(dtype=parameter.dtype)
        self.model.eval()
        amp = (
            torch.autocast(device_type="cuda", dtype=pixel_values.dtype)
            if images.is_cuda and pixel_values.dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with amp:
            output = self.model(
                pixel_values,
                output_hidden_states=True,
                interpolate_pos_encoding=True,
            )
        latents = getattr(output, "last_hidden_state", None)
        expected = (images.shape[0], VISION_TOKENS, VISION_LATENT_DIM)
        if not isinstance(latents, Tensor) or tuple(latents.shape) != expected:
            shape = None if not isinstance(latents, Tensor) else list(latents.shape)
            raise ValueError(
                f"SigLIP2 RAE encoder output must have shape [N, 256, 768]; got {shape}"
            )
        if not bool(torch.isfinite(latents).all()):
            raise ValueError("SigLIP2 RAE encoder output must be finite")
        return latents


class SigLIP2RAEDecoder(RAEDecoder):
    """Matching frozen SigLIP2 raw-latent RAE decoder."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        *,
        decoder: nn.Module | None = None,
        image_mean: Sequence[float] | Tensor = _DEFAULT_IMAGE_MEAN,
        image_std: Sequence[float] | Tensor = _DEFAULT_IMAGE_STD,
    ) -> None:
        super().__init__(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            decoder=decoder,
            image_mean=image_mean,
            image_std=image_std,
        )
