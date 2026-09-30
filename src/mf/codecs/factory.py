from __future__ import annotations

from torch import nn

from mf.codecs.dino_rae import DinoRAEDecoder, DinoRAEEncoder
from mf.codecs.rae_decoder import RAEDecoder
from mf.codecs.raw_pixel import RawPixelDecoder, RawPixelEncoder
from mf.codecs.scale_rae import ScaleRAESigLIP2Encoder, ScaleRAEWebSSLEncoder
from mf.codecs.siglip2_rae import SigLIP2RAEDecoder, SigLIP2RAEEncoder
from mf.codecs.t5 import T5TextEncoder
from mf.config.schema import VisionCodecConfig
from mf.codecs.registry import (
    TEXT_CODEC_REGISTRY,
    VISION_CODEC_REGISTRY,
    register_text_codec,
    register_vision_codec,
)


def _build_dino_rae_decoder(config: VisionCodecConfig) -> nn.Module:
    return DinoRAEDecoder(
        config_path=config.decoder_config_path,
        checkpoint_path=config.decoder_checkpoint_path,
    )


def _build_siglip2_rae_decoder(config: VisionCodecConfig) -> nn.Module:
    return SigLIP2RAEDecoder(
        config_path=config.decoder_config_path,
        checkpoint_path=config.decoder_checkpoint_path,
    )


def _build_scale_rae_siglip2_decoder(config: VisionCodecConfig) -> nn.Module:
    return RAEDecoder(
        config_path=config.decoder_config_path,
        checkpoint_path=config.decoder_checkpoint_path,
        latent_dim=config.latent_dim,
        image_size=config.encoder_input_resolution,
        patch_size=config.patch_size,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )


def _build_scale_rae_webssl_decoder(config: VisionCodecConfig) -> nn.Module:
    return RAEDecoder(
        config_path=config.decoder_config_path,
        checkpoint_path=config.decoder_checkpoint_path,
        latent_dim=config.latent_dim,
        image_size=config.encoder_input_resolution,
        patch_size=config.patch_size,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    )


def _register_builtin_codecs() -> None:
    register_vision_codec(
        "DINO RAE",
        encoder=lambda config: DinoRAEEncoder(config.model_path),
        decoder=_build_dino_rae_decoder,
        capabilities=("image_encoding", "image_decoding"),
        replace=True,
    )
    register_text_codec(
        "T5-small",
        encoder=lambda config: T5TextEncoder(
            config.model_path,
            latent_dim=config.latent_dim,
        ),
        capabilities=("text_encoding",),
        replace=True,
    )
    register_vision_codec(
        "SigLIP2 RAE",
        encoder=lambda config: SigLIP2RAEEncoder(config.model_path),
        decoder=_build_siglip2_rae_decoder,
        capabilities=("image_encoding", "image_decoding"),
        replace=True,
    )
    register_vision_codec(
        "Scale RAE SigLIP2",
        encoder=lambda config: ScaleRAESigLIP2Encoder(config.model_path),
        decoder=_build_scale_rae_siglip2_decoder,
        capabilities=("image_encoding", "image_decoding"),
        replace=True,
    )
    register_vision_codec(
        "Scale RAE WebSSL",
        encoder=lambda config: ScaleRAEWebSSLEncoder(config.model_path),
        decoder=_build_scale_rae_webssl_decoder,
        capabilities=("image_encoding", "image_decoding"),
        replace=True,
    )
    register_vision_codec(
        "Raw Pixel",
        encoder=lambda _config: RawPixelEncoder(),
        decoder=lambda _config: RawPixelDecoder(),
        capabilities=("image_encoding", "image_decoding"),
        replace=True,
    )


_register_builtin_codecs()


def build_vision_encoder(config: VisionCodecConfig) -> nn.Module:
    return VISION_CODEC_REGISTRY.build_encoder(config)


def build_vision_decoder(config: VisionCodecConfig) -> nn.Module:
    return VISION_CODEC_REGISTRY.build_decoder(config)


def build_text_encoder(config: object) -> nn.Module:
    return TEXT_CODEC_REGISTRY.build_encoder(config)
