"""One typed entry point for MF multimodal generation tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, overload

import torch
from torch import Tensor, nn

from mf.config.schema import MFConfig
from mf.contracts.geometry import GeometryContract
from mf.contracts.model import MFModelInput, MFOutput
from mf.contracts.physical import PhysicalSequenceLayout
from mf.latents.stats import TextLatentStatsType
from mf.modeling.chunk_causal_layout import ChunkCausalRoutingLayout
from mf.evaluation.generators import (
    CaptionGenerator,
    ImageOnlyGenerator,
    PromptContinuationGenerator,
    T2IImageGenerator,
    TextContinuationRequest,
    TextOnlyGenerator,
)
from mf.evaluation.sampling import SamplerConfig, TextBlockSessionFactory
from mf.inference.bundle import InferenceBundle, load_bundle
from mf.contracts.text import resolve_text_contract
from mf.inference.loading import HFCheckpointConfig, WeightSource
from mf.inference.protocol import (
    GenerationConfig,
    apply_overrides,
    checkpoint_sampler_config,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageToTextRequest,
    InferenceRequest,
    InferenceResult,
    InferenceTask,
    TextGenerationRequest,
    TextGenerationResult,
    TextToImageRequest,
    TextToTextRequest,
    PhysicalGenerationRequest,
)
from mf.registries import GENERATION_REGISTRY
from mf.instructions import IMAGE_CAPTION_PROMPT, format_instruction_prompt

_DEFAULT_TEXT_INFERENCE_STEPS = 16

@dataclass(frozen=True)
class _ImageRequest:
    """The fields T2IImageGenerator reads off a generation request."""

    prompt: str
    global_index: int


@dataclass(frozen=True)
class _ImageSample:
    """The one field ``CaptionGenerator`` reads off a caption sample."""

    image: Tensor


class MFPipeline:
    """Run all supported text and image tasks from one loaded checkpoint.

    The tasks are driven by the same generator classes the evaluation
    harness uses, so a given seed produces the same tokens and latents here as it
    does in a formal evaluation run.
    """

    def __init__(self, bundle: InferenceBundle) -> None:
        self.bundle = bundle
        self.config = bundle.config
        self.device = bundle.device
        self._sampler = checkpoint_sampler_config(bundle.config)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: str | torch.device = "cuda",
        weights: WeightSource = "ema",
        compile_blocks: bool = False,
        extensions: Sequence[str] = (),
    ) -> MFPipeline:
        from mf.extensions import load_extensions

        load_extensions(extensions)
        return cls(
            load_bundle(
                checkpoint_dir,
                device=device,
                weights=weights,
                compile_blocks=compile_blocks,
                extensions=extensions,
            )
        )

    @classmethod
    def from_components(
        cls,
        *,
        config: MFConfig,
        model: nn.Module,
        text_decoder: nn.Module,
        tokenizer: object,
        device: str | torch.device | None = None,
        text_encoder: nn.Module | None = None,
        vision_encoder: nn.Module | None = None,
        vision_decoder: nn.Module | None = None,
    ) -> MFPipeline:
        """Bind the pipeline to already-loaded modules, as used by online evaluation."""

        if device is None:
            try:
                resolved_device = next(model.parameters()).device
            except StopIteration as error:
                raise ValueError(
                    "device is required when model has no parameters"
                ) from error
        else:
            resolved_device = torch.device(device)
        return cls(
            InferenceBundle(
                config=config,
                device=resolved_device,
                model=model,
                text_decoder=text_decoder,
                tokenizer=tokenizer,
                checkpoint_dir=Path("."),
                text_contract=resolve_text_contract(
                    tokenizer=tokenizer,
                    eos_token_id=config.objective.eos_token_id,
                    pad_token_id=config.objective.pad_token_id,
                    latent_dim=config.codecs.text.latent_dim,
                    max_length=config.codecs.text.max_length,
                    tokenizer_revision=config.codecs.text.tokenizer_revision,
                ),
                _text_encoder=text_encoder,
                _vision_encoder=vision_encoder,
                _vision_decoder=vision_decoder,
            )
        )

    @torch.inference_mode()
    def forward_physical(
        self,
        physical_layout: PhysicalSequenceLayout,
        *,
        routing_layout: ChunkCausalRoutingLayout | None = None,
    ) -> MFOutput:
        """Run an adapter-produced multimodal sequence through the model.

        The adapter owns codec-specific encoding and sequence construction. MF
        only supplies the shared routing, backbone, and registered output heads,
        so video, editing, audio, or other modalities use this same entry point.
        ``physical_layout.position_ids`` is the sole positional source for both
        cached and uncached callers.
        """

        if not isinstance(physical_layout, PhysicalSequenceLayout):
            raise TypeError("physical_layout must be a PhysicalSequenceLayout")
        physical = physical_layout.to(self.device)
        batch_size = physical.token_embeddings.shape[0]
        if isinstance(self.config, HFCheckpointConfig):
            text_tokens = self.config.text.max_length
        else:
            text_tokens = self.config.data.text_max_length
        zero_role = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        zero_timestep = torch.zeros(
            batch_size,
            dtype=physical.token_embeddings.dtype,
            device=self.device,
        )
        model_input = MFModelInput(
            vision_latents_norm=None,
            text_latents_norm=None,
            text_prompt_latents_norm=None,
            text_prompt_content_mask=torch.zeros(
                (batch_size, 0), dtype=torch.bool, device=self.device
            ),
            vision_timestep=zero_timestep,
            text_timestep=zero_timestep.clone(),
            vision_role=zero_role,
            text_role=zero_role.clone(),
            text_content_mask=torch.zeros(
                (batch_size, text_tokens), dtype=torch.bool, device=self.device
            ),
            text_latent_stats_type=torch.full(
                (batch_size, text_tokens),
                int(TextLatentStatsType.PAD_IGNORE),
                dtype=torch.long,
                device=self.device,
            ),
            active_token_mask=physical.active_token_mask,
            text_previous_x0_norm=None,
            text_block_size=None,
            text_segment_ids=None,
            vision_latent_dim=self.config.codecs.vision.latent_dim,
            geometry=GeometryContract.from_config(self.config),
            compiled_sequences=physical.compiled_sequences,
            physical_layout=physical,
        ).validate()
        return self.bundle.model(model_input, routing_layout=routing_layout)

    @torch.inference_mode()
    def generate_physical(
        self,
        request: PhysicalGenerationRequest,
        *,
        generator_name: str,
        **kwargs: object,
    ) -> object:
        """Run a registered multimodal denoise/decode loop.

        The registered generator owns modality-specific timestep integration,
        cache use, and decoding. It receives this pipeline so it can reuse
        ``forward_physical`` for every step instead of rebuilding a second
        model path outside MF.
        """

        if not isinstance(request, PhysicalGenerationRequest):
            raise TypeError("request must be a PhysicalGenerationRequest")
        generator = GENERATION_REGISTRY.resolve(generator_name)
        return generator(self, request, **kwargs)

    def close(self) -> None:
        """Release task-specific codecs loaded by this pipeline."""

        self.bundle.close()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    @property
    def default_sampler_config(self) -> SamplerConfig:
        return self._sampler

    def _eos_token_id(self) -> int:
        if self.bundle.text_contract is not None:
            return self.bundle.text_contract.eos_token_id
        objective = getattr(self.config, "objective", None)
        configured = (
            getattr(objective, "eos_token_id", None)
            if objective is not None
            else getattr(self.config, "eos_token_id", None)
        )
        if configured is None:
            configured = self.bundle.tokenizer.eos_token_id
        return int(configured)

    def _resolve(self, config: GenerationConfig | None) -> tuple[SamplerConfig, int]:
        sampler = apply_overrides(self._sampler, config)
        seed = (
            (
                self.config.seed
                if isinstance(self.config, HFCheckpointConfig)
                else self.config.run.seed
            )
            if config is None or config.seed is None
            else config.seed
        )
        return sampler, seed

    def _resolve_text(
        self,
        config: GenerationConfig | None,
    ) -> tuple[SamplerConfig, int]:
        sampler, seed = self._resolve(config)
        if config is None or config.num_inference_steps is None:
            sampler = replace(
                sampler,
                num_inference_steps=_DEFAULT_TEXT_INFERENCE_STEPS,
            )
        return sampler, seed

    def _block_session_factory(
        self,
        sampler: SamplerConfig,
        use_cache: bool | None,
    ) -> TextBlockSessionFactory | None:
        if use_cache is None:
            use_cache = (
                self.device.type == "cuda"
                and getattr(self.bundle.model, "sequence_layout", None)
                == "chunk_causal"
            )
        if not use_cache:
            return None
        if self.device.type != "cuda":
            raise RuntimeError("cached block generation requires a CUDA device")
        if sampler.amp_dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError("cached block generation requires BF16 or FP16 autocast")
        from mf.inference.executor import BlockGenerationSession

        def factory(
            *,
            batch_size: int,
            max_cache_len: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> BlockGenerationSession:
            return BlockGenerationSession(
                self.bundle.model,
                batch_size=batch_size,
                max_cache_len=max_cache_len,
                device=device,
                dtype=dtype,
            )

        return factory

    @torch.inference_mode()
    def generate_text(
        self,
        num_samples: int,
        *,
        config: GenerationConfig | None = None,
        start_index: int = 0,
        sample_indices: Sequence[int] | None = None,
        max_batch_size: int = 8,
        use_cache: bool | None = None,
    ) -> tuple[str, ...]:
        """Sample unconditional text.

        Indices address the sample, not the call order: ``start_index`` selects
        which noise draws are used, so a given index always yields the same text.
        """

        if type(num_samples) is not int or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        if type(start_index) is not int or start_index < 0:
            raise ValueError("start_index must be a non-negative integer")
        indices = _sample_indices(start_index, num_samples, sample_indices)
        sampler, seed = self._resolve_text(config)
        block_session_factory = self._block_session_factory(sampler, use_cache)
        generator = TextOnlyGenerator(
            model=self.bundle.model,
            text_decoder=self.bundle.text_decoder,
            tokenizer=self.bundle.tokenizer,
            sampler_config=sampler,
            seed=seed,
            max_batch_size=max_batch_size,
            block_session_factory=block_session_factory,
            eos_token_id=self._eos_token_id(),
        )
        return generator(indices)

    @torch.inference_mode()
    def complete_text(
        self,
        prompts: Sequence[str],
        *,
        config: GenerationConfig | None = None,
        start_index: int = 0,
        sample_indices: Sequence[int] | None = None,
        target_length: int | None = None,
        max_prompt_tokens: int | None = None,
        prompt_truncation: Literal["error", "reject", "left_keep_suffix"] = (
            "left_keep_suffix"
        ),
        stop: Sequence[str] = (),
        max_batch_size: int = 8,
        use_cache: bool | None = None,
    ) -> tuple[str, ...]:
        """Continue text prefixes and stop each suffix at its first EOS token."""

        prompt_rows = tuple(prompts)
        if not prompt_rows:
            raise ValueError("complete_text requires at least one prompt")
        if any(not isinstance(prompt, str) or not prompt for prompt in prompt_rows):
            raise ValueError("prompts must contain non-empty strings")
        if type(start_index) is not int or start_index < 0:
            raise ValueError("start_index must be a non-negative integer")
        indices = _sample_indices(start_index, len(prompt_rows), sample_indices)
        sampler, seed = self._resolve_text(config)
        length = (
            (
                self.config.text.image_to_text_target_length
                if isinstance(self.config, HFCheckpointConfig)
                else self.config.evaluation.image_to_text_target_length
            )
            if target_length is None
            else target_length
        )
        prompt_limit = (
            (
                self.config.text.max_length
                if isinstance(self.config, HFCheckpointConfig)
                else self.config.data.text_max_length
            )
            - length
            if max_prompt_tokens is None
            else max_prompt_tokens
        )
        if prompt_limit <= 0:
            raise ValueError("max_prompt_tokens must leave room for generated text")
        generator = PromptContinuationGenerator(
            model=self.bundle.model,
            text_encoder=self.bundle.text_encoder,
            text_decoder=self.bundle.text_decoder,
            tokenizer=self.bundle.tokenizer,
            stats=self.bundle.model.latent_stats_registry,
            sampler_config=sampler,
            seed=seed,
            target_length=length,
            max_prompt_tokens=prompt_limit,
            prompt_truncation=(
                "reject" if prompt_truncation == "error" else prompt_truncation
            ),
            max_batch_size=max_batch_size,
            block_session_factory=self._block_session_factory(sampler, use_cache),
            eos_token_id=self._eos_token_id(),
        )
        stops = tuple(stop)
        requests = tuple(
            TextContinuationRequest(
                global_index=global_index,
                prompt=prompt,
                until=stops,
            )
            for global_index, prompt in zip(indices, prompt_rows, strict=True)
        )
        return generator(requests)

    @torch.inference_mode()
    def generate_image(
        self,
        prompts: Sequence[str],
        *,
        config: GenerationConfig | None = None,
        start_index: int = 0,
        sample_indices: Sequence[int] | None = None,
    ) -> Tensor:
        """Generate images from text prompts. Returns ``[B, 3, H, W]`` in ``[-1, 1]``."""

        if type(start_index) is not int or start_index < 0:
            raise ValueError("start_index must be a non-negative integer")
        prompt_rows = tuple(prompts)
        indices = _sample_indices(start_index, len(prompt_rows), sample_indices)
        requests = tuple(
            _ImageRequest(prompt=str(prompt), global_index=global_index)
            for prompt, global_index in zip(prompt_rows, indices, strict=True)
        )
        if not requests:
            raise ValueError("generate_image requires at least one prompt")
        sampler, seed = self._resolve(config)
        generator = T2IImageGenerator(
            model=self.bundle.model,
            text_encoder=self.bundle.text_encoder,
            tokenizer=self.bundle.tokenizer,
            stats=self.bundle.model.latent_stats_registry,
            sampler_config=sampler,
            seed=seed,
            decoder_factory=lambda: self.bundle.vision_decoder,
            eos_token_id=self._eos_token_id(),
        )
        return generator(requests)

    @torch.inference_mode()
    def generate_image_unconditional(
        self,
        num_samples: int,
        *,
        config: GenerationConfig | None = None,
        start_index: int = 0,
        sample_indices: Sequence[int] | None = None,
    ) -> Tensor:
        """Generate images with no conditioning. Returns ``[B, 3, H, W]`` in ``[-1, 1]``."""

        if type(num_samples) is not int or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        if type(start_index) is not int or start_index < 0:
            raise ValueError("start_index must be a non-negative integer")
        indices = _sample_indices(start_index, num_samples, sample_indices)
        sampler, seed = self._resolve(config)
        generator = ImageOnlyGenerator(
            model=self.bundle.model,
            sampler_config=sampler,
            seed=seed,
            decoder_factory=lambda: self.bundle.vision_decoder,
        )
        return generator(indices)

    @torch.inference_mode()
    def caption(
        self,
        images: Sequence[Tensor],
        *,
        config: GenerationConfig | None = None,
        start_index: int = 0,
        sample_indices: Sequence[int] | None = None,
        target_length: int | None = None,
        prompt: str | Sequence[str] = IMAGE_CAPTION_PROMPT,
        max_prompt_tokens: int | None = None,
        stop_at_eos: bool | None = None,
        skip_special_tokens: bool = True,
        use_cache: bool | None = None,
    ) -> tuple[str, ...]:
        """Describe images. Each image is ``[3, R, R]`` in ``[0, 1]``.

        ``R`` must be the encoder's input resolution, which the checkpoint
        records as ``evaluation.sampling.codec_input_resolution``.
        """

        samples = tuple(_ImageSample(image=image) for image in images)
        if not samples:
            raise ValueError("caption requires at least one image")
        if type(start_index) is not int or start_index < 0:
            raise ValueError("start_index must be a non-negative integer")
        indices = _sample_indices(start_index, len(samples), sample_indices)
        prompts = (prompt,) * len(samples) if isinstance(prompt, str) else tuple(prompt)
        if len(prompts) != len(samples):
            raise ValueError("prompt must contain one string per image")
        if any(not isinstance(value, str) or not value for value in prompts):
            raise ValueError("prompt must contain non-empty strings")
        prompts = tuple(format_instruction_prompt(value) for value in prompts)
        sampler, seed = self._resolve_text(config)
        length = (
            (
                self.config.text.image_to_text_target_length
                if isinstance(self.config, HFCheckpointConfig)
                else self.config.evaluation.image_to_text_target_length
            )
            if target_length is None
            else target_length
        )
        eos_stop = (
            self.config.flow.text_block_causal.image_to_text_eos_stop
            if stop_at_eos is None
            else stop_at_eos
        )
        generator = CaptionGenerator(
            model=self.bundle.model,
            text_decoder=self.bundle.text_decoder,
            text_encoder=self.bundle.text_encoder,
            tokenizer=self.bundle.tokenizer,
            vision_encoder=self.bundle.vision_encoder,
            stats=self.bundle.model.latent_stats_registry,
            sampler_config=sampler,
            seed=seed,
            target_noise_length=length,
            stop_at_eos=eos_stop,
            prompt=prompts[0],
            prompt_tokens=(
                (
                    self.config.text.image_to_text_prompt_max_length
                    if isinstance(self.config, HFCheckpointConfig)
                    else self.config.data.image_to_text_prompt_max_length
                )
                if max_prompt_tokens is None
                else max_prompt_tokens
            ),
            block_session_factory=self._block_session_factory(sampler, use_cache),
            eos_token_id=self._eos_token_id(),
        )
        token_ids = generator(samples, indices, prompts=prompts)
        return self.decode_tokens(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            stop_at_eos=eos_stop,
        )

    def decode_tokens(
        self,
        token_ids: Tensor,
        *,
        skip_special_tokens: bool = True,
        stop_at_eos: bool = True,
    ) -> tuple[str, ...]:
        """Render generated token ids, optionally cutting each row at its first EOS."""

        eos = self._eos_token_id()
        pad = self.bundle.tokenizer.pad_token_id
        rendered: list[str] = []
        for row in token_ids.detach().cpu().tolist():
            if stop_at_eos and eos in row:
                row = row[: row.index(eos)]
            row = [value for value in row if value != pad]
            rendered.append(
                self.bundle.tokenizer.decode(
                    row, skip_special_tokens=skip_special_tokens
                )
            )
        return tuple(rendered)

    @overload
    def run(
        self,
        request: TextGenerationRequest | TextToTextRequest | ImageToTextRequest,
    ) -> TextGenerationResult: ...

    @overload
    def run(
        self,
        request: TextToImageRequest | ImageGenerationRequest,
    ) -> ImageGenerationResult: ...

    def run(self, request: InferenceRequest) -> InferenceResult:
        """Dispatch a typed request and return a typed, index-aligned result."""

        if isinstance(request, TextGenerationRequest):
            texts = self.generate_text(
                request.num_samples,
                config=request.config,
                start_index=request.start_index,
                sample_indices=request.sample_indices,
                max_batch_size=request.max_batch_size,
                use_cache=request.use_cache,
            )
            return TextGenerationResult(
                task=InferenceTask.TEXT,
                texts=texts,
                sample_indices=request.resolved_sample_indices,
            )
        if isinstance(request, TextToTextRequest):
            texts = self.complete_text(
                request.prompts,
                config=request.config,
                start_index=request.start_index,
                sample_indices=request.sample_indices,
                target_length=request.target_length,
                max_prompt_tokens=request.max_prompt_tokens,
                prompt_truncation=request.prompt_truncation,
                stop=request.stop,
                max_batch_size=request.max_batch_size,
                use_cache=request.use_cache,
            )
            return TextGenerationResult(
                task=InferenceTask.TEXT_TO_TEXT,
                texts=texts,
                sample_indices=request.resolved_sample_indices,
            )
        if isinstance(request, ImageToTextRequest):
            texts = self.caption(
                request.images,
                config=request.config,
                start_index=request.start_index,
                sample_indices=request.sample_indices,
                target_length=request.target_length,
                prompt=request.resolved_prompts,
                max_prompt_tokens=request.max_prompt_tokens,
                stop_at_eos=request.stop_at_eos,
                skip_special_tokens=request.skip_special_tokens,
                use_cache=request.use_cache,
            )
            return TextGenerationResult(
                task=InferenceTask.IMAGE_TO_TEXT,
                texts=texts,
                sample_indices=request.resolved_sample_indices,
            )
        if isinstance(request, TextToImageRequest):
            images = self.generate_image(
                request.prompts,
                config=request.config,
                start_index=request.start_index,
                sample_indices=request.sample_indices,
            )
            return ImageGenerationResult(
                task=InferenceTask.TEXT_TO_IMAGE,
                images=images,
                sample_indices=request.resolved_sample_indices,
            )
        if isinstance(request, ImageGenerationRequest):
            images = self.generate_image_unconditional(
                request.num_samples,
                config=request.config,
                start_index=request.start_index,
                sample_indices=request.sample_indices,
            )
            return ImageGenerationResult(
                task=InferenceTask.IMAGE,
                images=images,
                sample_indices=request.resolved_sample_indices,
            )
        raise TypeError(f"unsupported inference request: {type(request).__name__}")


def _sample_indices(
    start_index: int,
    count: int,
    sample_indices: Sequence[int] | None = None,
) -> tuple[int, ...]:
    if sample_indices is None:
        return tuple(range(start_index, start_index + count))
    if start_index != 0:
        raise ValueError(
            "sample_indices cannot be combined with a non-zero start_index"
        )
    indices = tuple(sample_indices)
    if len(indices) != count:
        raise ValueError("sample_indices must contain one entry per request sample")
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("sample_indices must contain non-negative integers")
    return indices


__all__ = ["MFPipeline"]
