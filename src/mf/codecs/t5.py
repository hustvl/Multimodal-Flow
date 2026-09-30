from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from mf.codecs.frozen import FrozenCodec
from mf.contracts.batch import TEXT_LATENT_DIM

_MISSING = object()


def _require_t5_assets(model_path: Path) -> None:
    has_weights = (model_path / "model.safetensors").is_file() or (
        model_path / "pytorch_model.bin"
    ).is_file()
    if (
        not model_path.is_dir()
        or not (model_path / "config.json").is_file()
        or not has_weights
    ):
        raise FileNotFoundError(
            f"T5-small assets are missing at {model_path}; local config and weights are required"
        )


def _load_t5(model_path: str | Path) -> nn.Module:
    from transformers import T5EncoderModel

    path = Path(model_path).expanduser()
    if path.exists():
        _require_t5_assets(path)
        return T5EncoderModel.from_pretrained(path, local_files_only=True)
    return T5EncoderModel.from_pretrained(str(model_path))


def _check_t5_config(model: nn.Module, *, expected_width: int) -> None:
    config = getattr(model, "config", _MISSING)
    width = _MISSING if config is _MISSING else getattr(config, "d_model", _MISSING)
    if width != expected_width:
        raise ValueError(
            f"T5 encoder output width must be {expected_width}; got {width}"
        )


class T5TextEncoder(FrozenCodec):
    """Frozen local-only T5-small encoder."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        model: nn.Module | None = None,
        latent_dim: int = TEXT_LATENT_DIM,
    ) -> None:
        super().__init__()
        if type(latent_dim) is not int or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer")
        self.latent_dim = latent_dim
        self.model = _load_t5(model_path) if model is None else model
        _check_t5_config(self.model, expected_width=latent_dim)
        self.freeze()

    @torch.no_grad()
    def encode(self, token_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] <= 0:
            raise ValueError("token_ids must have shape [N, T] with T > 0")
        expected_inputs = tuple(token_ids.shape)
        if tuple(attention_mask.shape) != expected_inputs:
            raise ValueError("attention_mask must match token_ids shape [N, T]")
        self.model.eval()
        output = self.model(input_ids=token_ids, attention_mask=attention_mask)
        latents = getattr(output, "last_hidden_state", None)
        if not isinstance(latents, Tensor):
            raise TypeError("T5-small encoder must return a tensor last_hidden_state")
        expected_output = (*token_ids.shape, self.latent_dim)
        if tuple(latents.shape) != expected_output:
            raise ValueError(
                "T5-small encoder output must have shape "
                f"[N, {token_ids.shape[1]}, {self.latent_dim}]; got {list(latents.shape)}"
            )
        return latents
