from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import ViTMAELayer

from mf.codecs.frozen import FrozenCodec, preprocessing_buffer
from mf.contracts.batch import VISION_LATENT_DIM, VISION_TOKENS

OUTPUT_IMAGE_SIZE = 256
DECODER_PATCH_SIZE = 16


class RAEGeneralDecoder(nn.Module):
    """Minimal ViT-MAE decoder matching the released RAE checkpoint keys."""

    def __init__(self, config: ViTMAEConfig, *, num_patches: int) -> None:
        super().__init__()
        self.decoder_embed = nn.Linear(
            config.hidden_size, config.decoder_hidden_size, bias=True
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size),
            requires_grad=False,
        )
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        decoder_config._attn_implementation = "eager"
        self.decoder_layers = nn.ModuleList(
            ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)
        )
        self.decoder_norm = nn.LayerNorm(
            config.decoder_hidden_size, eps=config.layer_norm_eps
        )
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size,
            config.patch_size**2 * config.num_channels,
            bias=True,
        )
        self.trainable_cls_token = nn.Parameter(
            torch.zeros(1, 1, decoder_config.hidden_size)
        )
        self.config = config
        self.num_patches = num_patches

    def forward(self, raw_latents: Tensor) -> Tensor:
        hidden = self.decoder_embed(raw_latents)
        cls_token = self.trainable_cls_token.expand(hidden.shape[0], -1, -1)
        hidden = torch.cat((cls_token, hidden), dim=1) + self.decoder_pos_embed
        for layer in self.decoder_layers:
            layer_output = layer(hidden, head_mask=None)
            hidden = (
                layer_output if isinstance(layer_output, Tensor) else layer_output[0]
            )
        hidden = self.decoder_norm(hidden)
        return self.decoder_pred(hidden)[:, 1:]

    def unpatchify(self, logits: Tensor) -> Tensor:
        grid = int(self.num_patches**0.5)
        patch = int(self.config.patch_size)
        channels = int(self.config.num_channels)
        expected = (logits.shape[0], self.num_patches, patch * patch * channels)
        if tuple(logits.shape) != expected:
            raise ValueError(
                f"decoder logits must have shape {list(expected)}; got {list(logits.shape)}"
            )
        images = logits.reshape(logits.shape[0], grid, grid, patch, patch, channels)
        return images.permute(0, 5, 1, 3, 2, 4).reshape(
            logits.shape[0], channels, grid * patch, grid * patch
        )


def load_rae_decoder(
    config_path: Path,
    checkpoint_path: Path,
    *,
    latent_dim: int = VISION_LATENT_DIM,
    image_size: int = OUTPUT_IMAGE_SIZE,
    patch_size: int = DECODER_PATCH_SIZE,
) -> nn.Module:
    config_file = config_path / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"RAE decoder config is missing: {config_file}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RAE decoder checkpoint is missing: {checkpoint_path}")
    config = ViTMAEConfig.from_pretrained(config_path, local_files_only=True)
    # Released RAE configs are architecture templates; codec-specific geometry is reloaded here.
    config.hidden_size = latent_dim
    config.patch_size = patch_size
    config.image_size = image_size
    decoder = RAEGeneralDecoder(config, num_patches=VISION_TOKENS)
    state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(state, dict):
        raise TypeError(
            f"RAE decoder checkpoint must contain a state dict: {checkpoint_path}"
        )
    decoder.load_state_dict(state, strict=True, assign=True)
    return decoder


class RAEDecoder(FrozenCodec):
    """Frozen raw-codec-latent decoder with no MF latent transforms."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        image_mean: Sequence[float] | Tensor,
        image_std: Sequence[float] | Tensor,
        decoder: nn.Module | None = None,
        latent_dim: int = VISION_LATENT_DIM,
        image_size: int = OUTPUT_IMAGE_SIZE,
        patch_size: int = DECODER_PATCH_SIZE,
    ) -> None:
        super().__init__()
        if type(latent_dim) is not int or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer")
        if type(image_size) is not int or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        if type(patch_size) is not int or patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        self.decoder = (
            load_rae_decoder(
                Path(config_path),
                Path(checkpoint_path),
                latent_dim=latent_dim,
                image_size=image_size,
                patch_size=patch_size,
            )
            if decoder is None
            else decoder
        )
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.patch_size = patch_size
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

    @torch.no_grad()
    def decode(self, raw_latents: Tensor) -> Tensor:
        expected = (raw_latents.shape[0], VISION_TOKENS, self.latent_dim)
        if tuple(raw_latents.shape) != expected:
            raise ValueError(
                f"raw_latents must have shape [N, 256, {self.latent_dim}]; "
                f"got {list(raw_latents.shape)}"
            )
        self.decoder.eval()
        output: Any = self.decoder(raw_latents)
        logits = getattr(output, "logits", output)
        if not isinstance(logits, Tensor):
            raise TypeError("RAE decoder must return tensor logits")
        images = self.decoder.unpatchify(logits)
        expected_images = (raw_latents.shape[0], 3, self.image_size, self.image_size)
        if tuple(images.shape) != expected_images:
            raise ValueError(
                f"RAE decoder image output must have shape [N, 3, {self.image_size}, "
                f"{self.image_size}]; "
                f"got {list(images.shape)}"
            )
        mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        std = self.image_std.to(device=images.device, dtype=images.dtype)
        return images * std + mean
