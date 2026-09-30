"""Load a MF checkpoint for inference without assembling the training stack."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal, TypeAlias

import torch
from pydantic import Field, ValidationError
from torch import Tensor, nn

from mf.config.schema import (
    MFConfig,
    NonNegativeFloat,
    NonNegativeInt,
    NonEmptyStr,
    PositiveFloat,
    PositiveInt,
    PositiveIntTuple,
    StrictModel,
)
from mf.latents.stats import LatentStatsRegistry

WeightSource = Literal["ema", "raw"]


class InferenceTextDecoderConfig(StrictModel):
    hidden_size: PositiveInt
    depth: PositiveInt
    num_heads: PositiveInt
    head_dim: PositiveInt
    mlp_ratio: PositiveFloat
    bottleneck_dim: PositiveInt
    max_length: PositiveInt
    vocab_size: PositiveInt
    input_space: Literal["raw", "normalized"]
    compile_forward: bool = False


class InferenceModelConfig(StrictModel):
    name: str
    sequence_layout: Literal["chunk_causal"]
    image_chunk_conditioning: Literal["token_additive", "legacy_active_prefix"]
    t2i_chunk_semantics: Literal["chunk_native", "legacy_block_exact"]
    hidden_size: PositiveInt
    depth: PositiveInt
    num_heads: PositiveInt
    head_dim: PositiveInt
    ffn_hidden_size: PositiveInt
    attention_mode: Literal["shared", "modality_specific"]
    ffn_mode: Literal["shared", "modality_specific"]
    fp32_boundaries: bool
    compile_packed_blocks: bool = False
    text_input_bottleneck_dim: PositiveInt
    text_input_projection_mode: Literal["bottleneck", "linear"]
    mrope_section: PositiveIntTuple
    text_decoder: InferenceTextDecoderConfig


class InferenceVisionCodecConfig(StrictModel):
    name: str
    model_path: str
    decoder_config_path: str
    decoder_checkpoint_path: str
    image_preprocessing: Literal[
        "legacy_center_crop_bicubic_v1",
        "siglip2_resize_bicubic_v1",
    ]
    encoder_input_resolution: PositiveInt
    patch_size: PositiveInt
    grid_size: PositiveIntTuple
    latent_tokens: PositiveInt
    latent_dim: PositiveInt


class InferenceTextCodecConfig(StrictModel):
    name: str
    model_path: str
    tokenizer_path: str
    max_length: PositiveInt
    latent_dim: PositiveInt
    tokenizer_revision: NonEmptyStr | None = None


class InferenceCodecsConfig(StrictModel):
    vision: InferenceVisionCodecConfig
    text: InferenceTextCodecConfig


class InferenceAttentionConfig(StrictModel):
    block_size: Literal[8, 16]
    attention_backend: Literal["flex"]
    flex_backend: Literal["triton", "fa4"]
    flex_kernel_block_size: Literal[64, 128]
    flex_sequence_bucket_size: PositiveInt
    flex_fixed_sequence_length: bool
    flex_prewarm_sequence_lengths: PositiveIntTuple
    image_to_text_eos_stop: bool


class InferenceTimestepShiftConfig(StrictModel):
    t_lognorm_mu: float
    t_lognorm_sigma: PositiveFloat
    image_alpha: PositiveFloat
    text_alpha: PositiveFloat


class InferenceFlowConfig(StrictModel):
    velocity_t_eps: PositiveFloat
    vision_noise_scale: PositiveFloat
    text_noise_scale: PositiveFloat
    text_block_causal: InferenceAttentionConfig
    timestep_shift: InferenceTimestepShiftConfig


class InferenceTextConfig(StrictModel):
    max_length: PositiveInt
    image_to_text_prompt_max_length: PositiveInt
    image_to_text_target_length: PositiveInt


class InferenceGenerationConfig(StrictModel):
    text_num_inference_steps: PositiveInt = 16
    image_num_inference_steps: PositiveInt
    cfg_scale: NonNegativeFloat
    method: Literal["ode", "sde"] = "ode"
    sde_gamma: NonNegativeFloat = 1.0

class HFCheckpointConfig(StrictModel):
    model_type: Literal["multimodal_flow"]
    format_version: Literal[1]
    eos_token_id: NonNegativeInt = 1
    pad_token_id: NonNegativeInt | None = None
    seed: int = Field(ge=0)
    text_decoder_path: NonEmptyStr = "text_decoder.safetensors"
    vision_statistics_path: NonEmptyStr | None = None
    model: InferenceModelConfig
    codecs: InferenceCodecsConfig
    flow: InferenceFlowConfig
    text: InferenceTextConfig
    generation: InferenceGenerationConfig


CheckpointConfig: TypeAlias = MFConfig | HFCheckpointConfig

_EMA_KEY_PARTS = 5
_BACKBONE_PREFIX = "backbone."
_TEXT_DECODER_PREFIX = "text_decoder."
_DEFAULT_VISION_STATS_TOKENS = 256
_DEFAULT_TEXT_STATS_DIM = 512
_CHECKPOINT_VERSION = 1
_STEP_PATTERN = re.compile(r"step_(\d{6})")
_HF_MODEL_FILE = "model.safetensors"
_HF_TEXT_DECODER_FILE = "text_decoder.safetensors"
_EVALUATION_PAYLOADS = (
    "model.pt",
    "text_decoder.pt",
    "ema.pt",
    "resolved_config.json",
    "run_manifest.json",
)


def validate_evaluation_checkpoint(
    expected_step: int,
    checkpoint_dir: str | Path,
) -> Path:
    """Validate the portable subset required by standalone evaluation.

    Training restore additionally validates optimizer, scheduler, RNG, data-stream,
    and per-rank state against its current distributed world size. Standalone
    evaluation intentionally validates only the immutable checkpoint identity and
    inference payloads because its data-parallel world size may differ.
    """

    if type(expected_step) is not int or expected_step < 0:
        raise ValueError("expected_step must be a non-negative integer")
    root = Path(checkpoint_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"checkpoint path must be a directory: {root}")
    match = _STEP_PATTERN.fullmatch(root.name)
    if match is None or int(match.group(1)) != expected_step:
        raise ValueError(
            f"checkpoint directory does not match step {expected_step}: {root}"
        )

    marker_path = root / "COMPLETED"
    if not marker_path.is_file() or marker_path.is_symlink():
        raise ValueError(f"checkpoint is not marked COMPLETED: {root}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"checkpoint COMPLETED marker is invalid: {root}") from error
    if not isinstance(marker, dict) or set(marker) != {
        "format",
        "version",
        "step",
        "manifest_sha256",
    }:
        raise ValueError("checkpoint COMPLETED marker fields are invalid")
    if (
        not isinstance(marker["format"], str)
        or not marker["format"]
        or type(marker["version"]) is not int
        or marker["version"] != _CHECKPOINT_VERSION
        or type(marker["step"]) is not int
        or marker["step"] != expected_step
        or not isinstance(marker["manifest_sha256"], str)
    ):
        raise ValueError("checkpoint COMPLETED marker identity is invalid")

    for name in _EVALUATION_PAYLOADS:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"checkpoint evaluation payload is missing: {name}")
    manifest_sha256 = hashlib.sha256(
        (root / "run_manifest.json").read_bytes()
    ).hexdigest()
    if marker["manifest_sha256"] != manifest_sha256:
        raise ValueError("checkpoint manifest hash is corrupt")
    return root


def _legacy_training_config_to_inference(payload: dict[str, object]) -> dict[str, object]:
    """Project a legacy training config onto the public inference contract.

    Older training checkpoints contain optimizer, data, and evaluation-suite fields
    that are intentionally absent from the inference schema. Only architecture,
    codec, flow, and generation semantics cross this compatibility boundary.
    """

    model = payload["model"]
    codecs = payload["codecs"]
    flow = payload["flow"]
    data = payload["data"]
    evaluation = payload["evaluation"]
    objective = payload["objective"]
    assert isinstance(model, dict)
    assert isinstance(codecs, dict)
    assert isinstance(flow, dict)
    assert isinstance(data, dict)
    assert isinstance(evaluation, dict)
    assert isinstance(objective, dict)
    old_vision = codecs["vision"]
    old_text = codecs["text"]
    old_decoder = model["text_decoder"]
    old_attention = flow["text_block_causal"]
    old_shift = flow["timestep_shift"]
    old_sampling = evaluation["sampling"]
    assert isinstance(old_vision, dict)
    assert isinstance(old_text, dict)
    assert isinstance(old_decoder, dict)
    assert isinstance(old_attention, dict)
    assert isinstance(old_shift, dict)
    assert isinstance(old_sampling, dict)
    image_text = data.get("image_text", {})
    assert isinstance(image_text, dict)
    instruction = image_text.get("image_to_text_instruction", {})
    assert isinstance(instruction, dict)
    return {
        "model_type": "multimodal_flow",
        "format_version": 1,
        "eos_token_id": objective.get("flow_ar_eos_token_id", 1),
        "pad_token_id": objective.get("flow_ar_pad_token_id", 0),
        "seed": payload.get("run", {}).get("seed", 42),
        "model": {
            "name": model["name"],
            "sequence_layout": "chunk_causal",
            "image_chunk_conditioning": model.get(
                "image_event_conditioning", "token_additive"
            ),
            "t2i_chunk_semantics": "chunk_native",
            "hidden_size": model["hidden_size"],
            "depth": model["depth"],
            "num_heads": model["num_heads"],
            "head_dim": model["head_dim"],
            "ffn_hidden_size": model["ffn_hidden_size"],
            "attention_mode": model["attention_mode"],
            "ffn_mode": model["ffn_mode"],
            "fp32_boundaries": model["fp32_boundaries"],
            "compile_packed_blocks": False,
            "text_input_bottleneck_dim": model["text_input_bottleneck_dim"],
            "text_input_projection_mode": model["text_input_projection_mode"],
            "mrope_section": model["mrope_section"],
            "text_decoder": {
                "hidden_size": old_decoder["hidden_size"],
                "depth": old_decoder["depth"],
                "num_heads": old_decoder["num_heads"],
                "head_dim": old_decoder["head_dim"],
                "mlp_ratio": old_decoder["mlp_ratio"],
                "bottleneck_dim": old_decoder["bottleneck_dim"],
                "max_length": old_decoder["max_length"],
                "vocab_size": old_decoder["vocab_size"],
                "input_space": old_decoder["input_space"],
                "compile_forward": False,
            },
        },
        "codecs": {
            "vision": {
                "name": old_vision["name"],
                "model_path": old_vision["model_path"],
                "decoder_config_path": old_vision["decoder_config_path"],
                "decoder_checkpoint_path": old_vision["decoder_checkpoint_path"],
                "image_preprocessing": old_vision["image_preprocessing"],
                "encoder_input_resolution": old_vision["encoder_input_resolution"],
                "patch_size": old_vision["patch_size"],
                "grid_size": old_vision["grid_size"],
                "latent_tokens": old_vision["latent_tokens"],
                "latent_dim": old_vision["latent_dim"],
            },
            "text": {
                "name": old_text["name"],
                "model_path": old_text["model_path"],
                "tokenizer_path": old_text["tokenizer_path"],
                "max_length": old_text["max_length"],
                "latent_dim": old_text["latent_dim"],
                "tokenizer_revision": None,
            },
        },
        "flow": {
            "velocity_t_eps": flow["velocity_t_eps"],
            "vision_noise_scale": flow["vision_noise_scale"],
            "text_noise_scale": flow["text_noise_scale"],
            "text_block_causal": {
                "block_size": old_attention["block_size"],
                "attention_backend": old_attention["attention_backend"],
                "flex_backend": "triton",
                "flex_kernel_block_size": old_attention["flex_kernel_block_size"],
                "flex_sequence_bucket_size": old_attention["flex_sequence_bucket_size"],
                "flex_fixed_sequence_length": False,
                "flex_prewarm_sequence_lengths": old_attention["flex_prewarm_sequence_lengths"],
                "image_to_text_eos_stop": old_attention["image_to_text_eos_stop"],
            },
            "timestep_shift": {
                "t_lognorm_mu": old_shift["t_lognorm_mu"],
                "t_lognorm_sigma": old_shift["t_lognorm_sigma"],
                "image_alpha": old_shift["image_alpha"],
                "text_alpha": old_shift["text_alpha"],
            },
        },
        "text": {
            "max_length": old_text["max_length"],
            "image_to_text_prompt_max_length": instruction.get("max_length", 32),
            "image_to_text_target_length": evaluation.get(
                "image_to_text", {}
            ).get("target_noise_length", 64),
        },
        "generation": {
            "text_num_inference_steps": 16,
            "image_num_inference_steps": 64,
            "cfg_scale": old_sampling.get("cfg_scale", 1.0),
            "method": old_sampling.get("method", "ode"),
            "sde_gamma": old_sampling.get("sde_gamma", 1.0),
        },
    }


def read_checkpoint_config(checkpoint_dir: str | Path) -> CheckpointConfig:
    """Read the fully resolved config a checkpoint was written with."""

    root = Path(checkpoint_dir).expanduser().resolve(strict=True)
    path = next(
        (
            root / name
            for name in ("config.json", "resolved_config.json")
            if (root / name).is_file()
        ),
        None,
    )
    if path is None:
        raise FileNotFoundError(
            f"checkpoint is missing config.json or resolved_config.json: {root}"
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("model_type") == "multimodal_flow":
        return HFCheckpointConfig.model_validate(payload)
    try:
        return MFConfig.model_validate(payload)
    except ValidationError as error:
        try:
            return HFCheckpointConfig.model_validate(
                _legacy_training_config_to_inference(payload)
            )
        except (KeyError, TypeError, AssertionError, ValidationError):
            raise error


def placeholder_stats(
    vision_latent_dim: int,
    text_latent_dim: int = _DEFAULT_TEXT_STATS_DIM,
    vision_tokens: int = _DEFAULT_VISION_STATS_TOKENS,
) -> LatentStatsRegistry:
    """Build a registry whose values the checkpoint will overwrite.

    The four statistics are persistent buffers on the model, so they arrive with
    ``model.pt``. The registry still validates them at construction, which is why
    the standard deviations are ones rather than zeros.
    """

    if type(vision_latent_dim) is not int or vision_latent_dim <= 0:
        raise ValueError("vision_latent_dim must be a positive integer")
    if type(text_latent_dim) is not int or text_latent_dim <= 0:
        raise ValueError("text_latent_dim must be a positive integer")
    if type(vision_tokens) is not int or vision_tokens <= 0:
        raise ValueError("vision_tokens must be a positive integer")
    return LatentStatsRegistry(
        vision_mean=torch.zeros(vision_tokens, vision_latent_dim),
        vision_std=torch.ones(vision_tokens, vision_latent_dim),
        text_normal_mean=torch.zeros(text_latent_dim),
        text_normal_std=torch.ones(text_latent_dim),
    )


def _module_state(payload: object) -> dict[str, Tensor]:
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError(
            "checkpoint payload must be a mapping carrying a 'state' entry"
        )
    state = payload["state"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint 'state' must be a mapping")
    return dict(state)


def ema_shadow_overrides(
    ema_state: dict[str, object], prefix: str
) -> dict[str, Tensor]:
    """Map one module's EMA shadows onto its ``state_dict`` keys.

    ``ema.pt`` stores ``ExponentialMovingAverage.state_dict()``, not a state dict:
    ``shadows`` is a positional tuple aligned with ``parameter_keys``, whose entries
    read ``"{optimizer}:{group}:{group_name}:{index}:{parameter_name}"``. Only
    trainable parameters are tracked, so buffers and frozen parameters have no
    shadow and must keep the values that came with ``model.pt``.
    """

    keys = ema_state.get("parameter_keys")
    shadows = ema_state.get("shadows")
    shapes = ema_state.get("parameter_shapes")
    if not isinstance(keys, tuple) or not isinstance(shadows, tuple):
        raise ValueError("EMA state must carry parameter_keys and shadows tuples")
    if len(keys) != len(shadows):
        raise ValueError(
            f"EMA parameter_keys and shadows disagree: {len(keys)} against {len(shadows)}"
        )
    if shapes is not None and len(shapes) != len(keys):
        raise ValueError("EMA parameter_shapes must align with parameter_keys")

    overrides: dict[str, Tensor] = {}
    for index, key in enumerate(keys):
        if not isinstance(key, str):
            raise ValueError("EMA parameter keys must be strings")
        parts = key.split(":", _EMA_KEY_PARTS - 1)
        if len(parts) != _EMA_KEY_PARTS:
            raise ValueError(f"unrecognised EMA parameter key: {key!r}")
        name = parts[-1]
        if not name.startswith(prefix):
            continue
        name = name[len(prefix) :]
        shadow = shadows[index]
        if not isinstance(shadow, Tensor):
            raise ValueError(f"EMA shadow for {key!r} is not a tensor")
        if shapes is not None and tuple(shadow.shape) != tuple(shapes[index]):
            raise ValueError(
                f"EMA shadow for {key!r} does not match its recorded shape"
            )
        if name in overrides:
            raise ValueError(f"EMA state names {name!r} twice")
        overrides[name] = shadow
    return overrides


def _apply_ema(
    state: dict[str, Tensor],
    ema_state: dict[str, object],
    prefix: str,
    *,
    module: str,
) -> tuple[dict[str, Tensor], int]:
    overrides = ema_shadow_overrides(ema_state, prefix)
    merged = dict(state)
    for name, shadow in overrides.items():
        current = merged.get(name)
        if current is None:
            raise ValueError(
                f"EMA tracks {module} parameter {name!r} that the module does not have"
            )
        if tuple(current.shape) != tuple(shadow.shape):
            raise ValueError(
                f"EMA shadow for {module} parameter {name!r} has shape "
                f"{list(shadow.shape)}, expected {list(current.shape)}"
            )
        merged[name] = shadow.to(current.dtype)
    return merged, len(overrides)


def load_module_states(
    checkpoint_dir: str | Path,
    *,
    weights: WeightSource = "ema",
    text_decoder_path: str | Path | None = None,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, int]]:
    """Read backbone and text-decoder weights straight off disk.

    Bypasses ``CheckpointManager`` deliberately: it validates the training world
    size, requires one ``rank_states`` file per rank, re-hashes every payload, and
    pins the checkpoint's absolute path. None of that applies to inference, and all
    of it refuses a single-GPU load of a 64-rank checkpoint.
    """

    if weights not in ("ema", "raw"):
        raise ValueError("weights must be 'ema' or 'raw'")
    root = Path(checkpoint_dir).expanduser().resolve(strict=True)
    hf_model = root / _HF_MODEL_FILE
    hf_decoder = (
        Path(text_decoder_path).expanduser()
        if text_decoder_path is not None
        else root / _HF_TEXT_DECODER_FILE
    )
    if not hf_decoder.is_absolute():
        hf_decoder = root / hf_decoder
    if hf_model.is_file() or hf_decoder.is_file():
        if not hf_model.is_file() or not hf_decoder.is_file():
            raise ValueError(
                "HF checkpoint must contain model.safetensors and "
                "text_decoder.safetensors"
            )
        if weights == "raw":
            raise ValueError("HF checkpoint contains EMA weights only")
        from safetensors.torch import load_file

        model_state = dict(load_file(hf_model, device="cpu"))
        decoder_state = dict(load_file(hf_decoder, device="cpu"))
        return (
            model_state,
            decoder_state,
            {
                "model_entries": len(model_state),
                "text_decoder_entries": len(decoder_state),
                "model_from_ema": len(model_state),
                "text_decoder_from_ema": len(decoder_state),
            },
        )
    if not (root / "COMPLETED").is_file():
        raise ValueError(f"checkpoint is not marked COMPLETED: {root}")

    model_state = _module_state(
        torch.load(root / "model.pt", map_location="cpu", weights_only=True, mmap=True),
    )
    decoder_state = _module_state(
        torch.load(
            root / "text_decoder.pt", map_location="cpu", weights_only=True, mmap=True
        ),
    )
    counts = {
        "model_entries": len(model_state),
        "text_decoder_entries": len(decoder_state),
        "model_from_ema": 0,
        "text_decoder_from_ema": 0,
    }
    if weights == "raw":
        return model_state, decoder_state, counts

    ema_payload = torch.load(
        root / "ema.pt", map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(ema_payload, dict) or "state" not in ema_payload:
        raise ValueError("ema.pt must carry a 'state' entry")
    ema_state = ema_payload["state"]
    if not isinstance(ema_state, dict):
        raise ValueError("ema.pt 'state' must be a mapping")

    model_state, model_count = _apply_ema(
        model_state, ema_state, _BACKBONE_PREFIX, module="model"
    )
    decoder_state, decoder_count = _apply_ema(
        decoder_state, ema_state, _TEXT_DECODER_PREFIX, module="text_decoder"
    )
    counts["model_from_ema"] = model_count
    counts["text_decoder_from_ema"] = decoder_count
    if model_count == 0 and decoder_count == 0:
        raise ValueError("ema.pt covers neither the model nor the text decoder")
    return model_state, decoder_state, counts


def load_evaluation_states(
    checkpoint_dir: str | Path,
    *,
    model: nn.Module,
    text_decoder: nn.Module,
    ema: object,
) -> None:
    """Restore module and EMA state without restoring a training process.

    Evaluation may use a different data-parallel world size from training. Rank
    RNG, data-stream, optimizer, and scheduler states therefore do not belong in
    this load path. The evaluator still owns the EMA parameter swap, so modules
    receive their raw states and the EMA tracker receives the saved shadows.
    """

    root = Path(checkpoint_dir).expanduser().resolve(strict=True)
    model_state, decoder_state, _ = load_module_states(root, weights="raw")
    model.load_state_dict(model_state, strict=True)
    text_decoder.load_state_dict(decoder_state, strict=True)

    ema_payload = torch.load(
        root / "ema.pt", map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(ema_payload, dict) or "state" not in ema_payload:
        raise ValueError("ema.pt must carry a 'state' entry")
    ema_state = ema_payload["state"]
    if not isinstance(ema_state, dict):
        raise ValueError("ema.pt 'state' must be a mapping")
    load_state_dict = getattr(ema, "load_state_dict", None)
    if not callable(load_state_dict):
        raise TypeError("ema must provide load_state_dict")
    load_state_dict(ema_state)


__all__ = [
    "CheckpointConfig",
    "HFCheckpointConfig",
    "WeightSource",
    "ema_shadow_overrides",
    "load_evaluation_states",
    "load_module_states",
    "placeholder_stats",
    "read_checkpoint_config",
    "validate_evaluation_checkpoint",
]
