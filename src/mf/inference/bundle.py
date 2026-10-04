"""Assemble the modules inference needs, and nothing else."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path

import torch
from torch import nn

from mf.codecs.text_decoder import LatentTextDecoder
from mf.config.schema import MFConfig
from mf.contracts.batch import TEXT_PREFIX_TOKENS
from mf.contracts.text import ResolvedTextContract, resolve_text_contract
from mf.inference.loading import (
    CheckpointConfig,
    HFCheckpointConfig,
    WeightSource,
    load_module_states,
    placeholder_stats,
    read_checkpoint_config,
)

_LATENT_STATS_PREFIX = "latent_stats_registry."
_LEGACY_REMOVED_MODEL_KEYS = frozenset(
    {
        "embeddings.learned_text_guidance",
        "embeddings.text_guidance_embedder.mlp_0.bias",
        "embeddings.text_guidance_embedder.mlp_0.weight",
        "embeddings.text_guidance_embedder.mlp_2.bias",
        "embeddings.text_guidance_embedder.mlp_2.weight",
    }
)


def _load_hf_model_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    stats = {
        name: tensor
        for name, tensor in state.items()
        if name.startswith(_LATENT_STATS_PREFIX)
    }
    parameters = {
        name: tensor
        for name, tensor in state.items()
        if not name.startswith(_LATENT_STATS_PREFIX)
        and name not in _LEGACY_REMOVED_MODEL_KEYS
    }
    assigned = model.load_state_dict(parameters, strict=False, assign=True)
    if set(assigned.missing_keys) != set(stats) or assigned.unexpected_keys:
        raise RuntimeError("HF model weights do not match the configured architecture")
    copied = model.load_state_dict(stats, strict=False)
    if set(copied.missing_keys) != set(parameters) or copied.unexpected_keys:
        raise RuntimeError("HF latent statistics do not match the model")


def _restore_fp32_boundaries(model: nn.Module) -> None:
    """Restore modules that explicitly run outside autocast to FP32."""

    model.embeddings.text_state_proj.float()
    model.embeddings.text_input_proj.float()
    model.backbone.final_norm.float()


def _inference_config(
    config: CheckpointConfig,
    *,
    compile_blocks: bool,
    sequence_bucket_size: int | None,
) -> MFConfig:
    """Retune the training config for single-process generation.

    Three things change. Compiling costs tens of seconds to minutes before the
    first sample and only repays it over a long run, so it is off by default.

    Packed training fixes the Flex sequence bucket at the pack width. Generation
    needs only the vision tokens, prompt, text prefix, and clean/noisy text pair,
    so inheriting the training bucket needlessly pads each forward.

    Shrinking it means dropping ``tasks.chunk_pack``, because the schema ties the
    bucket to the pack width and ties the pack to the packing planner. Nothing at
    inference reads either: the samplers build their own routing layout from the
    model, and the task planner only ever runs while training.
    """

    text_block = config.flow.text_block_causal
    bucket = (
        _generation_bucket_size(config)
        if sequence_bucket_size is None
        else _require_bucket(sequence_bucket_size, text_block.flex_kernel_block_size)
    )
    model = config.model.model_copy(
        update={
            "compile_packed_blocks": compile_blocks,
            "text_decoder": config.model.text_decoder.model_copy(
                update={"compile_forward": False}
            ),
        }
    )
    flow = config.flow.model_copy(
        update={
            "text_block_causal": text_block.model_copy(
                update={
                    "flex_sequence_bucket_size": bucket,
                    "flex_fixed_sequence_length": False,
                    "flex_prewarm_sequence_lengths": (bucket,),
                }
            )
        }
    )
    updates: dict[str, object] = {"model": model, "flow": flow}
    if isinstance(config, MFConfig):
        updates["tasks"] = config.tasks.model_copy(
            update={"planner": "deterministic_global", "chunk_pack": None}
        )
    return config.model_copy(update=updates)


def _resolve_package_paths(
    config: CheckpointConfig,
    root: Path,
) -> CheckpointConfig:
    """Resolve portable asset paths relative to a downloaded model package."""

    asset_root = Path(os.environ.get("MF_ASSETS_ROOT", root)).expanduser()

    def resolve(value: str | Path | None) -> str | None:
        if value is None:
            return None
        source = str(value)
        if source.startswith("hf-model://"):
            return source.removeprefix("hf-model://")
        if source.startswith("hf-file://"):
            reference = source.removeprefix("hf-file://")
            parts = reference.split("/", 2)
            if len(parts) != 3:
                raise ValueError(f"invalid Hugging Face file reference: {source}")
            from huggingface_hub import hf_hub_download

            return hf_hub_download(
                repo_id=f"{parts[0]}/{parts[1]}",
                filename=parts[2],
            )
        path = Path(source).expanduser()
        return str(path if path.is_absolute() else (asset_root / path).resolve())

    vision_updates = {
        name: resolve(getattr(config.codecs.vision, name))
        for name in (
            "model_path",
            "decoder_config_path",
            "decoder_checkpoint_path",
        )
        if hasattr(config.codecs.vision, name)
    }
    vision = config.codecs.vision.model_copy(update=vision_updates)
    text = config.codecs.text.model_copy(
        update={
            "model_path": resolve(config.codecs.text.model_path),
            "tokenizer_path": resolve(config.codecs.text.tokenizer_path),
        }
    )
    codecs = config.codecs.model_copy(update={"vision": vision, "text": text})
    return config.model_copy(update={"codecs": codecs})


def _require_bucket(value: int, kernel_block_size: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("sequence_bucket_size must be a positive integer")
    if value % kernel_block_size:
        raise ValueError(
            f"sequence_bucket_size must be a multiple of {kernel_block_size}"
        )
    return value


def _generation_bucket_size(config: CheckpointConfig) -> int:
    """The longest packed sequence any single-sample generation task can need."""

    text_tokens = (
        config.text.max_length
        if isinstance(config, HFCheckpointConfig)
        else config.data.text_max_length
    )
    vision_tokens = config.codecs.vision.latent_tokens
    prompt_tokens = (
        config.text.image_to_text_prompt_max_length
        if isinstance(config, HFCheckpointConfig)
        else config.data.image_to_text_prompt_max_length
    )
    # Captioning is the widest task: vision condition, instruction prompt, the
    # text prefix, and the clean and noisy copies of the text field.
    longest = vision_tokens + prompt_tokens + TEXT_PREFIX_TOKENS + 2 * text_tokens
    kernel_block_size = config.flow.text_block_causal.flex_kernel_block_size
    return -(-longest // kernel_block_size) * kernel_block_size


@dataclass
class InferenceBundle:
    """Everything the samplers need, with the heavy codecs built on demand.

    The vision encoder, vision decoder and text encoder each cost a separate
    model load, and no single task needs all three: text-only generation needs
    none of them, captioning needs the vision encoder and the text encoder, and
    text-to-image needs the text encoder and the vision decoder.
    """

    config: CheckpointConfig
    device: torch.device
    model: nn.Module
    text_decoder: nn.Module
    tokenizer: object
    checkpoint_dir: Path
    text_contract: ResolvedTextContract | None = None
    load_counts: dict[str, int] = field(default_factory=dict)
    _text_encoder: nn.Module | None = field(default=None, repr=False)
    _vision_encoder: nn.Module | None = field(default=None, repr=False)
    _vision_decoder: nn.Module | None = field(default=None, repr=False)

    @property
    def text_encoder(self) -> nn.Module:
        if self._text_encoder is None:
            from mf.codecs.factory import build_text_encoder

            encoder = build_text_encoder(self.config.codecs.text)
            self._text_encoder = encoder.to(self.device, dtype=torch.bfloat16).eval()
        return self._text_encoder

    @property
    def vision_encoder(self) -> nn.Module:
        if self._vision_encoder is None:
            from mf.codecs.factory import build_vision_encoder

            encoder = build_vision_encoder(self.config.codecs.vision)
            self._vision_encoder = encoder.to(self.device, dtype=torch.bfloat16).eval()
        return self._vision_encoder

    @property
    def vision_decoder(self) -> nn.Module:
        if self._vision_decoder is None:
            from mf.codecs.factory import build_vision_decoder

            decoder = build_vision_decoder(self.config.codecs.vision)
            self._vision_decoder = decoder.to(self.device).eval()
        return self._vision_decoder

    def close(self) -> None:
        """Release task-specific codecs while leaving shared model modules untouched."""

        self._text_encoder = None
        self._vision_encoder = None
        self._vision_decoder = None


def load_bundle(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    weights: WeightSource = "ema",
    compile_blocks: bool = False,
    sequence_bucket_size: int | None = None,
    extensions: Sequence[str] = (),
) -> InferenceBundle:
    """Build the model and text decoder from a checkpoint directory."""

    from mf.extensions import freeze_extensions, load_extensions
    from mf.runtime import RuntimeTokenizer, new_model

    load_extensions(extensions)
    root = Path(checkpoint_dir).expanduser().resolve(strict=True)
    resolved_device = torch.device(device)
    config = _inference_config(
        _resolve_package_paths(read_checkpoint_config(root), root),
        compile_blocks=compile_blocks,
        sequence_bucket_size=sequence_bucket_size,
    )
    shared_decoder_path: Path | None = None
    if isinstance(config, HFCheckpointConfig):
        shared_decoder_path = Path(config.text_decoder_path).expanduser()
        if not shared_decoder_path.is_absolute():
            shared_decoder_path = (root / shared_decoder_path).resolve()
        if not shared_decoder_path.is_file():
            raise FileNotFoundError(
                f"Hugging Face text decoder is missing: {shared_decoder_path}"
            )
        if config.vision_statistics_path is not None:
            stats_path = Path(config.vision_statistics_path).expanduser()
            if not stats_path.is_absolute():
                stats_path = (root / stats_path).resolve()
            if not stats_path.is_file():
                raise FileNotFoundError(
                    f"Hugging Face vision statistics are missing: {stats_path}"
                )
    freeze_extensions()

    model_state, decoder_state, counts = load_module_states(
        root,
        weights=weights,
        text_decoder_path=shared_decoder_path,
    )

    stats = placeholder_stats(
        config.codecs.vision.latent_dim,
        config.codecs.text.latent_dim,
        config.codecs.vision.latent_tokens,
    )
    model = new_model(config, stats)
    is_hf_package = (root / "model.safetensors").is_file()
    if is_hf_package:
        _load_hf_model_state(model, model_state)
        if config.model.fp32_boundaries:
            _restore_fp32_boundaries(model)
    else:
        extra = set(model_state) - set(model.state_dict())
        if extra:
            if not extra.issubset(_LEGACY_REMOVED_MODEL_KEYS):
                raise RuntimeError(
                    "native checkpoint has unexpected model weights: "
                    f"{sorted(extra)}"
                )
            model_state = {
                name: tensor
                for name, tensor in model_state.items()
                if name not in _LEGACY_REMOVED_MODEL_KEYS
            }
        model.load_state_dict(model_state, strict=True)
    model = model.to(resolved_device).eval()

    text_decoder = LatentTextDecoder(
        input_dim=config.codecs.text.latent_dim,
        max_length=config.model.text_decoder.max_length,
        fp32_boundaries=config.model.fp32_boundaries,
        compile_forward=False,
    )
    text_decoder.load_state_dict(
        decoder_state,
        strict=True,
        assign=is_hf_package,
    )
    if is_hf_package and config.model.fp32_boundaries:
        text_decoder.float()
    text_decoder = text_decoder.to(resolved_device).eval()

    tokenizer = RuntimeTokenizer(config.codecs.text.tokenizer_path)
    configured_eos = (
        config.objective.eos_token_id
        if isinstance(config, MFConfig)
        else config.eos_token_id
    )
    configured_pad = (
        config.objective.pad_token_id
        if isinstance(config, MFConfig)
        else config.pad_token_id
    )
    text_contract = resolve_text_contract(
        tokenizer=tokenizer,
        eos_token_id=configured_eos,
        pad_token_id=configured_pad,
        latent_dim=config.codecs.text.latent_dim,
        max_length=(
            config.codecs.text.max_length
            if isinstance(config, MFConfig)
            else config.text.max_length
        ),
        tokenizer_revision=getattr(config.codecs.text, "tokenizer_revision", None),
    )
    return InferenceBundle(
        config=config,
        device=resolved_device,
        model=model,
        text_decoder=text_decoder,
        tokenizer=tokenizer,
        checkpoint_dir=root,
        text_contract=text_contract,
        load_counts=counts,
    )


__all__ = ["InferenceBundle", "load_bundle"]
