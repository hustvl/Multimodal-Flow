from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from mf._compat import Self
from mf.codecs.registry import TEXT_CODEC_REGISTRY, VISION_CODEC_REGISTRY
from mf.contracts.task_registry import task_definitions

NonEmptyStr = Annotated[str, Field(min_length=1)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
OpenUnitFloat = Annotated[float, Field(gt=0.0, lt=1.0)]


def _tuple_from_yaml_sequence(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def _integer_keys_from_json_mapping(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    normalized: dict[object, object] = {}
    for raw_key, item in value.items():
        key = raw_key
        if (
            isinstance(raw_key, str)
            and raw_key.isascii()
            and raw_key.isdigit()
            and str(int(raw_key)) == raw_key
        ):
            key = int(raw_key)
        if key in normalized:
            raise ValueError(
                f"duplicate integer mapping key after JSON normalization: {key!r}"
            )
        normalized[key] = item
    return normalized


SDEGammaByStep = Annotated[
    dict[PositiveInt, NonNegativeFloat],
    BeforeValidator(_integer_keys_from_json_mapping),
]


PositiveIntTuple = Annotated[
    tuple[PositiveInt, ...],
    BeforeValidator(_tuple_from_yaml_sequence),
]
GridSizeTuple = Annotated[
    tuple[PositiveInt, PositiveInt],
    BeforeValidator(_tuple_from_yaml_sequence),
]
Grid16Tuple = Annotated[
    tuple[Literal[16], Literal[16]],
    BeforeValidator(_tuple_from_yaml_sequence),
]
CaptionTypeTuple = Annotated[
    tuple[Literal["short", "medium", "long"], ...],
    BeforeValidator(_tuple_from_yaml_sequence),
    Field(min_length=1),
]
TimestepBucketTuple = Annotated[
    tuple[OpenUnitFloat, ...],
    BeforeValidator(_tuple_from_yaml_sequence),
    Field(min_length=1),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class RunConfig(StrictModel):
    profile: NonEmptyStr
    name: NonEmptyStr
    seed: NonNegativeInt
    output_dir: NonEmptyStr
    experiment_key: NonEmptyStr | None = None


class DistributedConfig(StrictModel):
    backend: Literal["torchrun_ddp"]
    world_size: PositiveInt
    micro_batch_size_per_rank: PositiveInt
    gradient_accumulation_steps: PositiveInt
    global_batch_size: PositiveInt
    amp_dtype: Literal["bf16"]
    process_group_timeout_seconds: PositiveInt = 600
    ddp_bucket_cap_mb: PositiveInt = 25
    ddp_gradient_compression: Literal["none", "bf16"] = "none"
    ddp_init_sync: bool = True
    # None defers to the planner-derived default; False and True are explicit
    # opt-outs that keep gradient reduction independent of task scheduling.
    ddp_static_graph: bool | None = None
    ddp_find_unused_parameters: bool | None = None


class ObjectiveConfig(StrictModel):
    mode: Literal["multimodal_flow"] = "multimodal_flow"
    eos_token_id: NonNegativeInt = 1
    pad_token_id: NonNegativeInt | None = None
    null_condition_probability: Probability = 0.1


class LLaVA15SFTConfig(StrictModel):
    """Configuration for the official LLaVA-1.5 instruction records."""

    dataset: Literal["llava_1_5_instruct"] = "llava_1_5_instruct"
    train_json: NonEmptyStr
    image_root: NonEmptyStr
    prompt_max_length: PositiveInt = 32
    answer_max_length: PositiveInt = 512
    max_consecutive_invalid_records: PositiveInt = 1024


TextToImageSource = Literal["blip3_ft60k", "dalle3", "sharegpt4o"]
TextToImageSourceTuple = Annotated[
    tuple[TextToImageSource, ...],
    BeforeValidator(_tuple_from_yaml_sequence),
    Field(min_length=1),
]


class TextToImageSFTWeights(StrictModel):
    blip3_ft60k: Probability = 0.06
    dalle3: Probability = 0.016
    sharegpt4o: Probability = 0.04


class TextToImageSFTConfig(StrictModel):
    """Wave9-compatible sources for supervised text-to-image training."""

    root: NonEmptyStr
    format: Literal["huggingface", "tar"] = "huggingface"
    sources: TextToImageSourceTuple = (
        "blip3_ft60k",
        "dalle3",
        "sharegpt4o",
    )
    weights: TextToImageSFTWeights = Field(default_factory=TextToImageSFTWeights)
    sharegpt4o_caption_file: NonEmptyStr = "sharegpt4o/text_to_image.json"
    sharegpt4o_max_tokens: PositiveInt = 256
    max_consecutive_invalid_records: PositiveInt = 1024

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("sft.text_to_image.sources must not contain duplicates")
        return self


class SFTConfig(StrictModel):
    enabled: bool = False
    text_to_image: TextToImageSFTConfig | None = None
    vqa: LLaVA15SFTConfig | None = None

    @model_serializer(mode="wrap")
    def serialize_active(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        payload = handler(self)
        if not self.enabled and self.text_to_image is None and self.vqa is None:
            payload.clear()
        return payload


class TextDecoderConfig(StrictModel):
    hidden_size: PositiveInt
    depth: PositiveInt
    num_heads: PositiveInt
    head_dim: PositiveInt
    mlp_ratio: PositiveFloat
    bottleneck_dim: PositiveInt
    max_length: PositiveInt
    vocab_size: PositiveInt
    decoder_noise_type: Literal["interpolate"]
    decoder_noise_scale: PositiveFloat
    decoder_p_mean: float
    decoder_p_std: PositiveFloat
    use_attention_mask: bool = True
    compile_forward: bool = False
    sparse_block_loss: bool = True
    input_space: Literal["raw", "normalized"]
    checkpoint_path: NonEmptyStr | None = None
    trainable: bool = True


class ModelConfig(StrictModel):
    name: NonEmptyStr
    architecture: Literal["dense"] = "dense"
    sequence_layout: Literal["chunk_causal"] = "chunk_causal"
    image_chunk_conditioning: Literal[
        "token_additive",
        "legacy_active_prefix",
    ] = "token_additive"
    t2i_chunk_semantics: Literal[
        "chunk_native",
        "legacy_block_exact",
    ] = "chunk_native"
    hidden_size: PositiveInt
    depth: PositiveInt
    num_heads: PositiveInt
    head_dim: PositiveInt
    ffn_hidden_size: PositiveInt
    vision_ffn_hidden_size: PositiveInt | None = None
    attention_mode: Literal["shared", "modality_specific"]
    ffn_mode: Literal["shared", "modality_specific"]
    fp32_boundaries: bool = False
    gradient_checkpointing: bool = False
    compile_packed_blocks: bool = False
    text_input_bottleneck_dim: PositiveInt
    text_input_projection_mode: Literal["bottleneck", "linear"] = "bottleneck"
    norm: Literal["pre_rmsnorm"] = "pre_rmsnorm"
    position_encoding: Literal["qwen3vl_style_3d_mrope"] = "qwen3vl_style_3d_mrope"
    prefix_position_encoding: Literal["semantic_group_mrope"] = "semantic_group_mrope"
    mrope_section: PositiveIntTuple
    use_flash_attention: Literal[True] = True
    text_decoder: TextDecoderConfig

    @model_serializer(mode="wrap")
    def omit_native_t2i_chunk_semantics(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        payload = handler(self)
        if self.vision_ffn_hidden_size is None:
            payload.pop("vision_ffn_hidden_size", None)
        if self.architecture == "dense":
            payload.pop("architecture", None)
        if self.t2i_chunk_semantics == "chunk_native":
            payload.pop("t2i_chunk_semantics", None)
        return payload


class _VisionCodecConfigBase(StrictModel):
    model_path: NonEmptyStr
    decoder_config_path: NonEmptyStr
    decoder_checkpoint_path: NonEmptyStr
    encoder_trainable: Literal[False]
    decoder_trainable: Literal[False]


class DinoRAECodecConfig(_VisionCodecConfigBase):
    name: Literal["DINO RAE"]
    encoder_input_resolution: Literal[224]
    patch_size: Literal[14]
    grid_size: Grid16Tuple
    latent_tokens: Literal[256]
    latent_dim: Literal[768]


class SigLIP2RAECodecConfig(_VisionCodecConfigBase):
    name: Literal["SigLIP2 RAE"]
    encoder_input_resolution: Literal[256]
    patch_size: Literal[16]
    grid_size: Grid16Tuple
    latent_tokens: Literal[256]
    latent_dim: Literal[768]


class ScaleRAESigLIP2CodecConfig(_VisionCodecConfigBase):
    name: Literal["Scale RAE SigLIP2"]
    image_preprocessing: Literal[
        "legacy_center_crop_bicubic_v1",
        "siglip2_resize_bicubic_v1",
    ] = "legacy_center_crop_bicubic_v1"
    encoder_input_resolution: Literal[224]
    patch_size: Literal[14]
    grid_size: Grid16Tuple
    latent_tokens: Literal[256]
    latent_dim: Literal[1152]


class ScaleRAEWebSSLCodecConfig(_VisionCodecConfigBase):
    name: Literal["Scale RAE WebSSL"]
    encoder_input_resolution: Literal[224]
    patch_size: Literal[14]
    grid_size: Grid16Tuple
    latent_tokens: Literal[256]
    latent_dim: Literal[1024]


class RawPixelCodecConfig(StrictModel):
    name: Literal["Raw Pixel"]
    encoder_input_resolution: Literal[256]
    patch_size: Literal[16]
    grid_size: Grid16Tuple
    latent_tokens: Literal[256]
    latent_dim: Literal[768]
    normalization: Literal["minus_one_to_one"]
    encoder_trainable: Literal[False]
    decoder_trainable: Literal[False]


_BUILTIN_VISION_CODEC_NAMES = frozenset(
    {
        "DINO RAE",
        "SigLIP2 RAE",
        "Scale RAE SigLIP2",
        "Scale RAE WebSSL",
        "Raw Pixel",
    }
)


class ExtensionVisionCodecConfig(StrictModel):
    """Configuration boundary for a codec registered by an extension."""

    name: NonEmptyStr
    encoder_input_resolution: PositiveInt
    patch_size: PositiveInt
    grid_size: GridSizeTuple
    latent_tokens: PositiveInt
    latent_dim: PositiveInt
    model_path: NonEmptyStr | None = None
    decoder_config_path: NonEmptyStr | None = None
    decoder_checkpoint_path: NonEmptyStr | None = None
    encoder_trainable: Literal[False] = False
    decoder_trainable: Literal[False] = False
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_registered_codec(self) -> Self:
        if self.name in _BUILTIN_VISION_CODEC_NAMES:
            raise ValueError(
                f"{self.name!r} must use its built-in strict codec configuration"
            )
        if self.name not in VISION_CODEC_REGISTRY.names():
            available = ", ".join(VISION_CODEC_REGISTRY.names()) or "<none>"
            raise ValueError(
                f"extension vision codec {self.name!r} is not registered; "
                f"available codecs: {available}"
            )
        return self


VisionCodecConfig = (
    DinoRAECodecConfig
    | SigLIP2RAECodecConfig
    | ScaleRAESigLIP2CodecConfig
    | ScaleRAEWebSSLCodecConfig
    | RawPixelCodecConfig
    | ExtensionVisionCodecConfig
)


class TextEncoderConfig(StrictModel):
    name: Literal["T5-small"]
    model_path: NonEmptyStr
    tokenizer_path: NonEmptyStr
    max_length: PositiveInt
    latent_dim: PositiveInt
    tokenizer_revision: NonEmptyStr | None = None
    trainable: Literal[False]


class ExtensionTextEncoderConfig(StrictModel):
    """Configuration boundary for a text codec registered by an extension."""

    name: NonEmptyStr
    model_path: NonEmptyStr
    tokenizer_path: NonEmptyStr
    max_length: PositiveInt
    latent_dim: PositiveInt
    tokenizer_revision: NonEmptyStr | None = None
    trainable: Literal[False] = False
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_registered_codec(self) -> Self:
        if self.name == "T5-small":
            raise ValueError(
                "'T5-small' must use its built-in strict text codec configuration"
            )
        if self.name not in TEXT_CODEC_REGISTRY.names():
            available = ", ".join(TEXT_CODEC_REGISTRY.names()) or "<none>"
            raise ValueError(
                f"extension text codec {self.name!r} is not registered; "
                f"available codecs: {available}"
            )
        return self


TextCodecConfig = TextEncoderConfig | ExtensionTextEncoderConfig


class CodecsConfig(StrictModel):
    online_encoding: Literal[True]
    vision: VisionCodecConfig
    text: TextCodecConfig


class TensorStatsConfig(StrictModel):
    path: NonEmptyStr
    mean_key: NonEmptyStr
    std_key: NonEmptyStr
    expected_shape: PositiveIntTuple


class ScalarTextStatsConfig(StrictModel):
    mean: float = 0.0
    std: PositiveFloat = 0.17


class LatentStatsConfig(StrictModel):
    vision: TensorStatsConfig
    text_normal: TensorStatsConfig | ScalarTextStatsConfig = Field(
        default_factory=ScalarTextStatsConfig
    )


class CaptionSamplingConfig(StrictModel):
    strategy: Literal["uniform_over_available_types"]
    missing_type_policy: Literal["sample_from_existing_types"]
    caption_types: CaptionTypeTuple
    seed_formula: Literal["run_seed+global_sample_index"]

    @model_validator(mode="after")
    def validate_caption_types(self) -> Self:
        if len(set(self.caption_types)) != len(self.caption_types):
            raise ValueError("caption_types must not contain duplicates")
        return self


class CorruptionPolicyConfig(StrictModel):
    warning_limit: NonNegativeInt


class ImageToTextInstructionConfig(StrictModel):
    prompt: NonEmptyStr = "USER: Describe this image in words.\nASSISTANT:"
    max_length: PositiveInt = 32


class ImageTextDataConfig(StrictModel):
    source: Literal["GPIC"]
    root: NonEmptyStr
    split: Literal["train", "test"]
    min_age_minutes: NonNegativeFloat = 30.0
    caption_sampling: CaptionSamplingConfig
    corruption_policy: CorruptionPolicyConfig
    image_to_text_instruction: ImageToTextInstructionConfig = Field(
        default_factory=ImageToTextInstructionConfig
    )


class TextSourceConfig(StrictModel):
    root: NonEmptyStr
    split: NonEmptyStr
    weight: Probability


class TextDataConfig(StrictModel):
    ultrafineweb_multi_style: TextSourceConfig
    ultrafineweb_qa: TextSourceConfig
    gpic_caption_text: TextSourceConfig

    @model_validator(mode="after")
    def validate_source_weights(self) -> Self:
        weights = [
            self.ultrafineweb_multi_style.weight,
            self.ultrafineweb_qa.weight,
            self.gpic_caption_text.weight,
        ]
        weight_sum = math.fsum(weights)
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"text source weights must sum to 1 (got {weight_sum:.12g})"
            )
        return self


class DataLoaderConfig(StrictModel):
    num_workers: NonNegativeInt
    prefetch_factor: PositiveInt


class BundleDataConfig(StrictModel):
    root: NonEmptyStr
    split: NonEmptyStr = "train"
    records_path: NonEmptyStr | None = None
    image_to_text_instruction: ImageToTextInstructionConfig = Field(
        default_factory=ImageToTextInstructionConfig
    )


class DataConfig(StrictModel):
    input_mode: Literal["online"]
    loader: DataLoaderConfig
    bundle: BundleDataConfig | None = None
    image_text: ImageTextDataConfig | None = None
    text: TextDataConfig | None = None
    text_max_length: PositiveInt
    pad_is_inactive: Literal[True]

    @property
    def image_to_text_prompt_max_length(self) -> int:
        if self.bundle is not None:
            return self.bundle.image_to_text_instruction.max_length
        if self.image_text is not None:
            return self.image_text.image_to_text_instruction.max_length
        return ImageToTextInstructionConfig().max_length

    @model_serializer(mode="wrap")
    def serialize_active_sources(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        payload = handler(self)
        if self.bundle is None:
            payload.pop("bundle", None)
        if self.image_text is None:
            payload.pop("image_text", None)
        if self.text is None:
            payload.pop("text", None)
        return payload


class TaskWeightsConfig(StrictModel):
    text_to_image: Probability
    image_to_text: Probability
    text_only: Probability
    image_only: Probability
    custom: dict[NonEmptyStr, Probability] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_custom_tasks(self) -> Self:
        known = {definition.name for definition in task_definitions("planner")}
        unknown = sorted(set(self.custom) - known)
        if unknown:
            raise ValueError(f"task weights reference unregistered tasks: {unknown}")
        return self

    def as_mapping(self) -> dict[str, float]:
        """Expose configured weights using the registered task names."""

        values = dict(self.custom)
        for definition in task_definitions("planner"):
            if definition.name not in values:
                values[definition.name] = float(
                    getattr(self, definition.name, 0.0)
                )
        return values

    def as_tuple(self) -> tuple[float, ...]:
        return tuple(self.as_mapping()[definition.name] for definition in task_definitions("planner"))


class TaskSlotsConfig(StrictModel):
    text_to_image: NonNegativeInt
    image_to_text: NonNegativeInt
    text_only: NonNegativeInt
    image_only: NonNegativeInt
    custom: dict[NonEmptyStr, NonNegativeInt] = Field(default_factory=dict)

    def as_mapping(self) -> dict[str, int]:
        values = dict(self.custom)
        for definition in task_definitions("planner"):
            if definition.name not in values:
                values[definition.name] = int(getattr(self, definition.name, 0))
        return values

    def as_tuple(self) -> tuple[int, ...]:
        return tuple(self.as_mapping()[definition.name] for definition in task_definitions("planner"))


class ChunkPackTokenBudgetsConfig(StrictModel):
    text_to_image: NonNegativeInt
    image_to_text: NonNegativeInt
    text_only: NonNegativeInt
    image_only: NonNegativeInt
    custom: dict[NonEmptyStr, NonNegativeInt] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_custom_tasks(self) -> Self:
        known = {definition.name for definition in task_definitions("packing")}
        unknown = sorted(set(self.custom) - known)
        if unknown:
            raise ValueError(f"chunk budgets reference unregistered tasks: {unknown}")
        return self

    def as_mapping(self) -> dict[str, int]:
        values = dict(self.custom)
        for definition in task_definitions("packing"):
            if definition.name not in values:
                values[definition.name] = int(getattr(self, definition.name, 0))
        return values

    def as_tuple(self) -> tuple[int, ...]:
        return tuple(self.as_mapping()[definition.name] for definition in task_definitions("packing"))


class ChunkPackTextPackingConfig(StrictModel):
    mode: Literal["continuous", "block_aligned_eos"]
    block_size: Literal[8, 16] = 8
    block_aligned_record_policy: Literal[
        "whole_record_defer_pad",
        "record_chunk_defer_pad",
        "legacy_sentence_units_hard_pack",
        "record_units_hard_pack",
    ] = "whole_record_defer_pad"


class ChunkPackConfig(StrictModel):
    sequence_length: PositiveInt
    allocation: Literal["strict", "elastic"] = "strict"
    exposure_basis: Literal[
        "physical_tokens",
        "target_supervised_tokens",
        "logical_samples",
    ] = "physical_tokens"
    token_budgets: ChunkPackTokenBudgetsConfig
    logical_chunks_per_pack: PositiveInt | None = None
    text_packing: ChunkPackTextPackingConfig | None = None

    @model_validator(mode="after")
    def validate_budget_total(self) -> Self:
        total = sum(self.token_budgets.as_tuple())
        if self.allocation == "strict" and total != self.sequence_length:
            raise ValueError(
                "chunk pack token budgets must sum to sequence_length "
                f"({total} != {self.sequence_length})"
            )
        if self.allocation == "elastic" and any(
            budget > self.sequence_length for budget in self.token_budgets.as_tuple()
        ):
            raise ValueError(
                "elastic chunk pack task budgets cannot exceed sequence_length"
            )
        if (
            self.exposure_basis == "target_supervised_tokens"
            and self.allocation != "elastic"
        ):
            raise ValueError(
                "target-supervised-token exposure requires allocation=elastic"
            )
        if self.exposure_basis == "logical_samples" and (
            self.allocation != "elastic" or self.logical_chunks_per_pack is None
        ):
            raise ValueError(
                "logical-sample exposure requires allocation=elastic and logical_chunks_per_pack"
            )
        return self


class TasksConfig(StrictModel):
    planner: Literal[
        "deterministic_global",
        "rank_balanced_global",
        "chunk_token_packed",
    ]
    weights: TaskWeightsConfig
    chunk_pack: ChunkPackConfig | None = None

    @model_validator(mode="after")
    def validate_chunk_pack(self) -> Self:
        if self.planner == "chunk_token_packed" and self.chunk_pack is None:
            raise ValueError("chunk_token_packed planner requires tasks.chunk_pack")
        if self.planner != "chunk_token_packed" and self.chunk_pack is not None:
            raise ValueError("tasks.chunk_pack requires planner=chunk_token_packed")
        if self.chunk_pack is not None:
            budget_by_task = self.chunk_pack.token_budgets.as_mapping()
            weight_by_task = self.weights.as_mapping()
            mismatched = tuple(
                task
                for task in budget_by_task
                if (budget_by_task[task] > 0) != (weight_by_task[task] > 0.0)
            )
            if mismatched:
                raise ValueError(
                    "chunk-pack positive budgets and task weights must enable the same tasks: "
                    + ", ".join(mismatched)
                )
        return self


_TASK_FIELD_ORDER = tuple(
    field for field in TaskWeightsConfig.model_fields if field != "custom"
)


def derive_task_slots(
    weights: TaskWeightsConfig,
    global_batch_size: int,
) -> TaskSlotsConfig:
    """Convert task weights into exact deterministic global-batch counts."""
    if type(global_batch_size) is not int or global_batch_size <= 0:
        raise ValueError("global_batch_size must be a positive integer")
    weight_values = weights.as_tuple()
    weight_sum = math.fsum(weight_values)
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"task weights must sum to 1 (got {weight_sum:.12g})")

    exact = tuple(global_batch_size * weight for weight in weight_values)
    counts = [math.floor(value) for value in exact]
    remaining = global_batch_size - sum(counts)
    ranked = sorted(
        range(len(counts)),
        key=lambda index: (-(exact[index] - counts[index]), index),
    )
    for index in ranked[:remaining]:
        counts[index] += 1

    definitions = task_definitions("planner")
    built_in_fields = {
        definition.name: count
        for definition, count in zip(definitions, counts, strict=True)
        if definition.name in _TASK_FIELD_ORDER
    }
    custom_fields = {
        definition.name: count
        for definition, count in zip(definitions, counts, strict=True)
        if definition.name not in _TASK_FIELD_ORDER
    }
    return TaskSlotsConfig.model_validate(
        {**built_in_fields, "custom": custom_fields}
    )


class TimestepShiftConfig(StrictModel):
    distribution: Literal["logit_normal"]
    t_lognorm_mu: float
    t_lognorm_sigma: PositiveFloat
    image_alpha: PositiveFloat = 6.0
    text_alpha: PositiveFloat = 6.0
    shift_on: Literal["noise_level"]


class TextBlockCausalConfig(StrictModel):
    block_size: Literal[8, 16] = 8
    clean_noisy_layout: Literal["concatenate"] = "concatenate"
    timestep_sampling: Literal["per_block"] = "per_block"
    attention_backend: Literal["flex"] = "flex"
    flex_backend: Literal["triton", "fa4"] = "triton"
    flex_kernel_block_size: Literal[64, 128] = 128
    flex_sequence_bucket_size: PositiveInt = 4096
    flex_fixed_sequence_length: bool = False
    block_aligned_record_policy: Literal[
        "whole_record_defer_pad",
        "record_chunk_defer_pad",
        "legacy_sentence_units_hard_pack",
        "record_units_hard_pack",
    ] = "whole_record_defer_pad"
    flex_prewarm_sequence_lengths: PositiveIntTuple = ()
    target_encoding: Literal["full_sequence", "block_local"] = "full_sequence"
    packing: Literal["continuous", "block_aligned_eos"] = "continuous"
    image_to_text_eos_stop: bool = False

    @model_validator(mode="after")
    def validate_flex_sequence_bucket(self) -> Self:
        if self.flex_backend == "fa4" and self.flex_kernel_block_size != 128:
            raise ValueError("FA4 requires flex_kernel_block_size=128")
        if self.flex_sequence_bucket_size % self.flex_kernel_block_size:
            raise ValueError(
                "flex_sequence_bucket_size must be a kernel-block multiple"
            )
        if any(
            length % self.flex_kernel_block_size
            for length in self.flex_prewarm_sequence_lengths
        ):
            raise ValueError(
                "flex_prewarm_sequence_lengths must contain only kernel-block multiples"
            )
        if tuple(sorted(set(self.flex_prewarm_sequence_lengths))) != (
            self.flex_prewarm_sequence_lengths
        ):
            raise ValueError(
                "flex_prewarm_sequence_lengths must be strictly increasing and unique"
            )
        return self


class FlowConfig(StrictModel):
    t_convention: Literal["clean_is_one"]
    interpolation: Literal["x_t=t*x0+(1-t)*eps"]
    training_objective: Literal["masked_velocity_mse"]
    velocity_t_eps: Annotated[float, Field(gt=0.0, le=1.0)]
    prediction: Literal["normalized_x0"]
    latent_space: Literal["normalized"]
    vision_noise_scale: PositiveFloat
    text_noise_scale: PositiveFloat
    source_condition_noise: None = None
    text_block_causal: TextBlockCausalConfig
    timestep_shift: TimestepShiftConfig


class ScheduleConfig(StrictModel):
    warmup_steps: NonNegativeInt
    max_steps: PositiveInt
    schedule: Literal["cosine", "constant"]


class BackboneOptimizerConfig(StrictModel):
    matrix_optimizer: Literal["Muon"]
    fallback_optimizer: Literal["NesterovAdamW"]
    peak_lr: PositiveFloat
    min_lr: NonNegativeFloat
    weight_decay: NonNegativeFloat


class TextDecoderOptimizerConfig(StrictModel):
    optimizer: Literal["NesterovAdamW", "Muon"]
    peak_lr: PositiveFloat
    min_lr: NonNegativeFloat
    weight_decay: NonNegativeFloat


class EMAConfig(StrictModel):
    enabled: Literal[True]
    decay: Annotated[float, Field(ge=0.0, lt=1.0)]
    update_cadence: Literal["immediately_after_each_optimizer_step"]
    use_for_evaluation: Literal[True]


class OptimizersConfig(StrictModel):
    muon_distributed_mode: Literal["sharded", "replicated"] = "sharded"
    muon_shard_group_size: PositiveInt | None = None
    muon_sync_chunk_mb: PositiveInt = 64
    muon_matmul_precision: Literal["highest", "high"] = "highest"
    schedule: ScheduleConfig
    backbone: BackboneOptimizerConfig
    text_decoder: TextDecoderOptimizerConfig
    ema: EMAConfig


class TrainerConfig(StrictModel):
    max_steps: PositiveInt
    max_grad_norm: PositiveFloat
    save_steps: int
    eval_steps: int
    eval_at_final_step: bool
    checkpoint_root: NonEmptyStr
    synchronous_evaluation: Literal[True]


class SamplingConfig(StrictModel):
    num_inference_steps: PositiveInt
    cfg_scale: NonNegativeFloat
    method: Literal["ode", "sde"] = "ode"
    sde_gamma: NonNegativeFloat = 1.0
    ar_temperature: PositiveFloat = 0.8
    ar_top_p: Annotated[float, Field(gt=0.0, le=1.0)] = 0.95
    amp_dtype: Literal["bf16"]
    checkpoint_target: Literal["ema"]
    model_latent_shape: PositiveIntTuple
    codec_input_resolution: PositiveInt
    artifact_resolution: PositiveInt

class EvaluationConfig(StrictModel):
    artifact_dir: NonEmptyStr
    sampling: SamplingConfig
    image_to_text_target_length: PositiveInt = 64


class LoggingConfig(StrictModel):
    enabled: bool
    backend: Literal["tensorboard", "jsonl"]
    log_steps: PositiveInt
    output_dir: NonEmptyStr
    project: NonEmptyStr
    include_task_metrics: bool = True
    defer_writes_until_close: bool = False
    async_writes: bool = False
    async_queue_size: PositiveInt = 64
    task_metric_log_steps: PositiveInt | None = None
    dense_task_metric_steps: NonNegativeInt = 0

    @model_serializer(mode="wrap")
    def serialize_optional_performance_contract(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        data = handler(self)
        if not self.async_writes:
            data.pop("async_writes", None)
            data.pop("async_queue_size", None)
        if self.task_metric_log_steps is None:
            data.pop("task_metric_log_steps", None)
        if self.dense_task_metric_steps == 0:
            data.pop("dense_task_metric_steps", None)
        return data


class MFConfig(StrictModel):
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)
    run: RunConfig
    distributed: DistributedConfig
    model: ModelConfig
    codecs: CodecsConfig
    latent_stats: LatentStatsConfig
    data: DataConfig
    tasks: TasksConfig
    flow: FlowConfig
    optimizers: OptimizersConfig
    trainer: TrainerConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_training_contract(self) -> Self:
        if self.sft.enabled:
            if self.sft.text_to_image is None:
                raise ValueError(
                    "sft.text_to_image is required when sft.enabled is true"
                )
            if self.sft.vqa is None:
                raise ValueError("sft.vqa is required when sft.enabled is true")
            selected_sources = set(self.sft.text_to_image.sources)
            for source in selected_sources:
                if getattr(self.sft.text_to_image.weights, source) <= 0:
                    raise ValueError(
                        f"sft.text_to_image.weights.{source} must be positive for a selected source"
                    )
            if self.tasks.planner != "chunk_token_packed":
                raise ValueError("SFT requires tasks.planner=chunk_token_packed")
            if self.tasks.weights.text_to_image <= 0:
                raise ValueError(
                    "SFT requires a non-zero text_to_image task weight"
                )
            if self.tasks.weights.image_to_text <= 0:
                raise ValueError(
                    "SFT requires a non-zero image_to_text task weight"
                )
            if self.tasks.weights.text_only != 0 or self.tasks.weights.image_only != 0:
                raise ValueError(
                    "SFT exposes only text_to_image and image_to_text task weights"
                )
            if self.sft.vqa.answer_max_length > self.data.text_max_length:
                raise ValueError(
                    "sft.vqa.answer_max_length cannot exceed data.text_max_length"
                )
        elif self.sft.text_to_image is not None or self.sft.vqa is not None:
            raise ValueError("sft configuration requires sft.enabled=true")
        if self.model.t2i_chunk_semantics == "legacy_block_exact":
            if self.model.image_chunk_conditioning != "legacy_active_prefix":
                raise ValueError(
                    "legacy_block_exact T2I semantics require "
                    "model.image_chunk_conditioning=legacy_active_prefix"
                )
            if self.tasks.weights.text_to_image == 0:
                raise ValueError(
                    "legacy_block_exact T2I semantics require a positive tasks.weights.text_to_image"
                )
        if self.objective.mode != "multimodal_flow":
            raise ValueError("chunk_causal layout requires multimodal_flow")
        block_causal = self.flow.text_block_causal
        expected_hidden_size = self.model.num_heads * self.model.head_dim
        if self.model.hidden_size != expected_hidden_size:
            raise ValueError(
                "model.hidden_size must equal model.num_heads * model.head_dim "
                f"({self.model.hidden_size} != {expected_hidden_size})"
            )

        expected_global_batch = (
            self.distributed.world_size
            * self.distributed.micro_batch_size_per_rank
            * self.distributed.gradient_accumulation_steps
        )
        if self.distributed.global_batch_size != expected_global_batch:
            raise ValueError(
                "distributed.global_batch_size must equal world_size * "
                "micro_batch_size_per_rank * gradient_accumulation_steps "
                f"({self.distributed.global_batch_size} != {expected_global_batch})"
            )

        muon_group_size = self.optimizers.muon_shard_group_size
        if muon_group_size is not None:
            if self.optimizers.muon_distributed_mode != "sharded":
                raise ValueError("muon_shard_group_size requires sharded Muon")
            if muon_group_size > self.distributed.world_size:
                raise ValueError("muon_shard_group_size cannot exceed world_size")
            if self.distributed.world_size % muon_group_size:
                raise ValueError(
                    "muon_shard_group_size must evenly divide distributed.world_size"
                )

        chunk_pack = self.tasks.chunk_pack
        if chunk_pack is not None:
            if not block_causal.flex_fixed_sequence_length:
                raise ValueError(
                    "chunk token packing requires flex_fixed_sequence_length=true"
                )
            if block_causal.flex_sequence_bucket_size != chunk_pack.sequence_length:
                raise ValueError(
                    "chunk pack sequence_length must equal the fixed Flex sequence length"
                )
            if any(
                budget % block_causal.flex_kernel_block_size
                for budget in chunk_pack.token_budgets.as_tuple()
            ):
                raise ValueError(
                    "chunk pack token budgets must be Flex-kernel-block aligned"
                )
            if self.distributed.micro_batch_size_per_rank != 1:
                raise ValueError(
                    "chunk token packing requires physical micro batch size 1"
                )
        weight_sum = math.fsum(self.tasks.weights.as_tuple())
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"task weights must sum to 1 (got {weight_sum:.12g})")

        text_tokens = self.data.text_max_length
        if self.codecs.text.max_length != text_tokens:
            raise ValueError(
                "codecs.text.max_length must equal data.text_max_length "
                f"({self.codecs.text.max_length} != {text_tokens})"
            )
        decoder = self.model.text_decoder
        block_causal = self.flow.text_block_causal
        expected_decoder_length = text_tokens
        block_local_decoder = block_causal.target_encoding == "block_local"
        if block_local_decoder:
            expected_decoder_length = block_causal.block_size
            if block_causal.packing != "block_aligned_eos":
                raise ValueError(
                    "block_local target encoding requires block_aligned_eos packing"
                )
            if not block_causal.image_to_text_eos_stop:
                raise ValueError(
                    "block_local target encoding requires image_to_text_eos_stop=true"
                )
        if decoder.max_length != expected_decoder_length:
            if block_local_decoder:
                raise ValueError(
                    "model.text_decoder.max_length must equal text_block_causal.block_size "
                    f"({decoder.max_length} != {expected_decoder_length})"
                )
            raise ValueError(
                "model.text_decoder.max_length must equal data.text_max_length "
                f"({decoder.max_length} != {expected_decoder_length})"
            )
        if not decoder.trainable and decoder.checkpoint_path is None:
            raise ValueError("frozen model.text_decoder requires checkpoint_path")

        if self.trainer.save_steps <= 0:
            raise ValueError("trainer.save_steps must be greater than 0")
        if self.trainer.eval_steps <= 0:
            raise ValueError("trainer.eval_steps must be greater than 0")
        return self
