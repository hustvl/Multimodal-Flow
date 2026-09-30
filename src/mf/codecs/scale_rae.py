from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mf.codecs.frozen import FrozenCodec, preprocessing_buffer, require_local_encoder
from mf.contracts.batch import VISION_TOKENS

_IMAGE_SIZE = 224
_PATCH_SIZE = 14
_SIGLIP_DIM = 1152
_WEBSSL_DIM = 1024
_DEFAULT_SIGLIP_MEAN = (0.5, 0.5, 0.5)
_DEFAULT_SIGLIP_STD = (0.5, 0.5, 0.5)
_DEFAULT_WEBSSL_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_WEBSSL_STD = (0.229, 0.224, 0.225)


def _load_siglip(
    model_path: str | Path,
) -> tuple[nn.Module, Sequence[float], Sequence[float]]:
    from transformers import AutoImageProcessor, AutoModel

    path = Path(model_path).expanduser()
    local = path.exists()
    if local:
        require_local_encoder(path, name="Scale-RAE", require_preprocessor=False)
    source = path if local else str(model_path)
    processor = AutoImageProcessor.from_pretrained(
        source, local_files_only=local, use_fast=False
    )
    model = AutoModel.from_pretrained(
        source,
        local_files_only=local,
    ).vision_model
    return model, processor.image_mean, processor.image_std


def _load_webssl(
    path: Path,
) -> tuple[nn.Module, Sequence[float], Sequence[float]]:
    require_local_encoder(path, name="Scale-RAE", require_preprocessor=False)
    from transformers import AutoImageProcessor, Dinov2Model

    processor = AutoImageProcessor.from_pretrained(
        path, local_files_only=True, use_fast=False
    )
    model = Dinov2Model.from_pretrained(path, local_files_only=True)
    return model, processor.image_mean, processor.image_std


class _ScaleRAEEncoder(FrozenCodec):
    latent_tokens: int = VISION_TOKENS
    latent_dim: int

    def __init__(
        self,
        *,
        model: nn.Module,
        latent_dim: int,
        image_mean: Sequence[float] | Tensor,
        image_std: Sequence[float] | Tensor,
    ) -> None:
        super().__init__()
        config = getattr(model, "config", None)
        expected = {
            "hidden_size": latent_dim,
            "image_size": _IMAGE_SIZE,
            "patch_size": _PATCH_SIZE,
        }
        for name, value in expected.items():
            actual = None if config is None else getattr(config, name, None)
            if actual != value:
                raise ValueError(
                    f"Scale-RAE encoder config {name} must be {value}; got {actual}"
                )
        self.model = model
        self.latent_dim = latent_dim
        self.register_buffer(
            "image_mean",
            preprocessing_buffer(image_mean, name="image_mean"),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            preprocessing_buffer(image_std, name="image_std"),
            persistent=False,
        )
        self.freeze()

    def _pixels(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"images must have shape [N, 3, H, W]; got {list(images.shape)}"
            )
        if images.shape[-2:] != (_IMAGE_SIZE, _IMAGE_SIZE):
            raise ValueError(
                "images must be preprocessed to 224x224 before Scale-RAE encoding"
            )
        mean = self.image_mean.to(device=images.device, dtype=torch.float32)
        std = self.image_std.to(device=images.device, dtype=torch.float32)
        pixels = (images.float() - mean) / std
        parameter = next(self.model.parameters(), None)
        if parameter is not None and parameter.is_floating_point():
            pixels = pixels.to(dtype=parameter.dtype)
        return pixels

    def _validate(self, latents: object, *, batch_size: int) -> Tensor:
        expected = (batch_size, VISION_TOKENS, self.latent_dim)
        if not isinstance(latents, Tensor) or tuple(latents.shape) != expected:
            actual = None if not isinstance(latents, Tensor) else list(latents.shape)
            raise ValueError(
                f"Scale-RAE encoder output must have shape "
                f"[N, {VISION_TOKENS}, {self.latent_dim}]; got {actual}"
            )
        if not bool(torch.isfinite(latents).all()):
            raise ValueError("Scale-RAE encoder output must be finite")
        return latents


class ScaleRAESigLIP2Encoder(_ScaleRAEEncoder):
    """Frozen Scale-RAE SigLIP2-so400m encoder using final-layer affine-free LN."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        model: nn.Module | None = None,
        image_mean: Sequence[float] | Tensor | None = None,
        image_std: Sequence[float] | Tensor | None = None,
    ) -> None:
        if model is None:
            model, loaded_mean, loaded_std = _load_siglip(model_path)
            image_mean = loaded_mean if image_mean is None else image_mean
            image_std = loaded_std if image_std is None else image_std
        super().__init__(
            model=model,
            latent_dim=_SIGLIP_DIM,
            image_mean=_DEFAULT_SIGLIP_MEAN if image_mean is None else image_mean,
            image_std=_DEFAULT_SIGLIP_STD if image_std is None else image_std,
        )

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        pixels = self._pixels(images)
        self.model.eval()
        embeddings = getattr(self.model, "embeddings", None)
        encoder = getattr(self.model, "encoder", None)
        if callable(embeddings) and callable(encoder):
            embedded = embeddings(pixels, interpolate_pos_encoding=True)
            encoded = encoder(inputs_embeds=embedded)
            last_hidden_state = getattr(encoded, "last_hidden_state", None)
        else:
            output = self.model(
                pixels,
                output_hidden_states=True,
                interpolate_pos_encoding=True,
            )
            hidden_states = getattr(output, "hidden_states", None)
            last_hidden_state = (
                hidden_states[-1]
                if isinstance(hidden_states, (tuple, list)) and hidden_states
                else None
            )
        if not isinstance(last_hidden_state, Tensor):
            raise TypeError(
                "Scale-RAE SigLIP2 encoder must return final encoder states"
            )
        latents = F.layer_norm(
            last_hidden_state,
            (_SIGLIP_DIM,),
            weight=None,
            bias=None,
            eps=1e-6,
        )
        return self._validate(latents, batch_size=images.shape[0])


class ScaleRAEWebSSLEncoder(_ScaleRAEEncoder):
    """Frozen Scale-RAE WebSSL-DINO encoder using patch tokens after CLS."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        model: nn.Module | None = None,
        image_mean: Sequence[float] | Tensor | None = None,
        image_std: Sequence[float] | Tensor | None = None,
    ) -> None:
        if model is None:
            model, loaded_mean, loaded_std = _load_webssl(Path(model_path))
            image_mean = loaded_mean if image_mean is None else image_mean
            image_std = loaded_std if image_std is None else image_std
        super().__init__(
            model=model,
            latent_dim=_WEBSSL_DIM,
            image_mean=_DEFAULT_WEBSSL_MEAN if image_mean is None else image_mean,
            image_std=_DEFAULT_WEBSSL_STD if image_std is None else image_std,
        )

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        pixels = self._pixels(images)
        self.model.eval()
        output = self.model(pixels)
        sequence = getattr(output, "last_hidden_state", None)
        latents = None if not isinstance(sequence, Tensor) else sequence[:, 1:]
        return self._validate(latents, batch_size=images.shape[0])
