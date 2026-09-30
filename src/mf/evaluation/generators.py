from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from mf.contracts.batch import TEXT_TOKENS
from mf.data.text import TokenizedTextBlock, tokenize_caption, tokenize_condition
from mf.evaluation.sampling import (
    SamplerConfig,
    TextBlockSessionFactory,
    TextCondition,
    TextTarget,
    VisionCondition,
    VisionDecoder,
    sample_text,
    sample_text_continuation,
    sample_text_unconditional,
    sample_vision,
    sample_vision_unconditional,
)
from mf.evaluation.text_metrics import render_token_ids
from mf.instructions import IMAGE_CAPTION_PROMPT
from mf.latents.stats import LatentStatsRegistry, TextLatentStatsType


class EvaluationTokenizer(Protocol):
    eos_token_id: int
    pad_token_id: int

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
    ) -> str: ...


class ImagePrompt(Protocol):
    global_index: int
    prompt: str


@dataclass(frozen=True, slots=True)
class TextContinuationRequest:
    global_index: int
    prompt: str
    until: tuple[str, ...] = ("\n",)

    def __post_init__(self) -> None:
        if type(self.global_index) is not int or self.global_index < 0:
            raise ValueError("global_index must be a non-negative integer")
        if type(self.prompt) is not str or not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        if (
            type(self.until) is not tuple
            or any(type(stop) is not str or not stop for stop in self.until)
            or len(self.until) != len(set(self.until))
        ):
            raise ValueError("until must contain unique non-empty strings")


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is None:
        raise ValueError("evaluation module must contain parameters")
    return parameter.device


def _model_text_tokens(model: nn.Module) -> int:
    text_tokens = getattr(model, "text_tokens", TEXT_TOKENS)
    if type(text_tokens) is not int or text_tokens <= 0:
        raise ValueError("evaluation model must expose a positive text_tokens value")
    return text_tokens


def _resolve_eos_token_id(
    tokenizer: EvaluationTokenizer,
    eos_token_id: int | None,
) -> int:
    value = tokenizer.eos_token_id if eos_token_id is None else eos_token_id
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("eos_token_id must be a non-negative integer")
    return value


def _require_max_batch_size(value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("max_batch_size must be a positive integer")
    return value


def text_stats_layout(
    token_ids: Tensor,
    content_mask: Tensor,
    *,
    eos_token_id: int,
) -> Tensor:
    if token_ids.ndim != 2 or token_ids.shape != content_mask.shape:
        raise ValueError("token_ids and content_mask must have matching [B, T] shapes")
    if content_mask.dtype is not torch.bool:
        raise ValueError("content_mask must have dtype torch.bool")
    stats_type = torch.full_like(token_ids, int(TextLatentStatsType.PAD_IGNORE))
    stats_type.masked_fill_(content_mask, int(TextLatentStatsType.NORMAL_TEXT))
    stats_type.masked_fill_(
        token_ids.eq(eos_token_id) & content_mask,
        int(TextLatentStatsType.EOS),
    )
    return stats_type


def _stack_text_blocks(
    blocks: Sequence[TokenizedTextBlock],
    *,
    eos_token_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    token_ids = torch.stack([block.token_ids for block in blocks]).to(device)
    content_mask = torch.stack([block.content_mask for block in blocks]).to(device)
    return (
        token_ids,
        content_mask,
        text_stats_layout(token_ids, content_mask, eos_token_id=eos_token_id),
    )


def _all_text_target(
    batch_size: int,
    text_tokens: int,
    device: torch.device,
) -> TextTarget:
    return TextTarget(
        content_mask=torch.ones(
            (batch_size, text_tokens), dtype=torch.bool, device=device
        ),
        latent_stats_type=torch.full(
            (batch_size, text_tokens),
            int(TextLatentStatsType.NORMAL_TEXT),
            dtype=torch.long,
            device=device,
        ),
    )


def _prefix_text_target(
    batch_size: int,
    text_tokens: int,
    active_text_tokens: int,
    device: torch.device,
) -> TextTarget:
    if (
        type(active_text_tokens) is not int
        or active_text_tokens <= 0
        or active_text_tokens > text_tokens
    ):
        raise ValueError("active_text_tokens must be within the model text capacity")
    content_mask = torch.zeros(
        (batch_size, text_tokens),
        dtype=torch.bool,
        device=device,
    )
    content_mask[:, :active_text_tokens] = True
    latent_stats_type = torch.full(
        (batch_size, text_tokens),
        int(TextLatentStatsType.PAD_IGNORE),
        dtype=torch.long,
        device=device,
    )
    latent_stats_type[:, :active_text_tokens] = int(TextLatentStatsType.NORMAL_TEXT)
    return TextTarget(
        content_mask=content_mask,
        latent_stats_type=latent_stats_type,
    )


class T2IImageGenerator:
    """Online T5 condition encoding, normalized-x0 sampling, and lazy RAE decode."""

    def __init__(
        self,
        *,
        model: nn.Module,
        text_encoder: nn.Module,
        tokenizer: EvaluationTokenizer,
        stats: LatentStatsRegistry,
        sampler_config: SamplerConfig,
        seed: int,
        decoder_factory: Callable[[], VisionDecoder],
        eos_token_id: int | None = None,
    ) -> None:
        self.model = model
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.eos_token_id = _resolve_eos_token_id(tokenizer, eos_token_id)
        self.stats = stats
        self.sampler_config = sampler_config
        self.seed = seed
        self.decoder_factory = decoder_factory
        self.device = _module_device(model)
        self.text_tokens = _model_text_tokens(model)
        self._decoder: VisionDecoder | None = None

    def _vision_decoder(self) -> VisionDecoder:
        if self._decoder is None:
            decoder = self.decoder_factory()
            if isinstance(decoder, nn.Module):
                decoder = decoder.to(self.device)
            self._decoder = decoder
        return self._decoder

    def __call__(self, requests: Sequence[ImagePrompt]) -> Tensor:
        if not requests:
            raise ValueError("image generation requests must not be empty")
        blocks = tuple(
            tokenize_caption(
                self.tokenizer,
                request.prompt,
                text_tokens=self.text_tokens,
            )
            for request in requests
        )
        token_ids, content_mask, stats_type = _stack_text_blocks(
            blocks,
            eos_token_id=self.eos_token_id,
            device=self.device,
        )
        raw_text = self.text_encoder.encode(token_ids, content_mask)
        normalized_text = self.stats.normalize_text(raw_text, stats_type, content_mask)
        sample = sample_vision(
            self.model,
            TextCondition(
                latents_norm=normalized_text,
                content_mask=content_mask,
                latent_stats_type=stats_type,
            ),
            self.sampler_config,
            self.seed,
            global_sample_indices=tuple(request.global_index for request in requests),
            decoder=self._vision_decoder(),
        )
        return sample.images.mul(2.0).sub(1.0)

    def close(self) -> None:
        self._decoder = None


class ImageOnlyGenerator:
    """Generate images with the vision branch alone and no text condition."""

    def __init__(
        self,
        *,
        model: nn.Module,
        sampler_config: SamplerConfig,
        seed: int,
        decoder_factory: Callable[[], VisionDecoder],
    ) -> None:
        self.model = model
        self.sampler_config = sampler_config
        self.seed = seed
        self.decoder_factory = decoder_factory
        self.device = _module_device(model)
        self._decoder: VisionDecoder | None = None

    def _vision_decoder(self) -> VisionDecoder:
        if self._decoder is None:
            decoder = self.decoder_factory()
            if isinstance(decoder, nn.Module):
                decoder = decoder.to(self.device)
            self._decoder = decoder
        return self._decoder

    def __call__(self, global_indices: Sequence[int]) -> Tensor:
        indices = tuple(global_indices)
        if not indices:
            raise ValueError("image-only global indices must not be empty")
        sample = sample_vision_unconditional(
            self.model,
            self.sampler_config,
            self.seed,
            global_sample_indices=indices,
            device=self.device,
            decoder=self._vision_decoder(),
        )
        return sample.images.mul(2.0).sub(1.0)

    def close(self) -> None:
        self._decoder = None


class CaptionGenerator:
    """Generate configured-length text from an online vision condition."""

    def __init__(
        self,
        *,
        model: nn.Module,
        text_decoder: nn.Module,
        text_encoder: nn.Module,
        tokenizer: EvaluationTokenizer,
        vision_encoder: nn.Module,
        stats: LatentStatsRegistry,
        sampler_config: SamplerConfig,
        seed: int,
        target_noise_length: int = 64,
        stop_at_eos: bool = False,
        prompt: str = IMAGE_CAPTION_PROMPT,
        prompt_tokens: int = 32,
        block_session_factory: TextBlockSessionFactory | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        self.model = model
        self.text_decoder = text_decoder
        self.vision_encoder = vision_encoder
        self.stats = stats
        self.sampler_config = sampler_config
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.eos_token_id = _resolve_eos_token_id(tokenizer, eos_token_id)
        self.seed = seed
        self.stop_at_eos = stop_at_eos
        self.block_session_factory = block_session_factory
        self.device = _module_device(model)
        self.prompt_tokens = prompt_tokens
        model_text_tokens = _model_text_tokens(model)
        if (
            type(target_noise_length) is not int
            or target_noise_length <= 0
            or target_noise_length > model_text_tokens
        ):
            raise ValueError(
                "caption target_noise_length must be positive and no larger than "
                "the model text capacity"
            )
        self.model_text_tokens = model_text_tokens
        self.text_tokens = target_noise_length
        self.prompt_condition = self._encode_prompts((prompt,))

    def _encode_prompts(self, prompts: Sequence[str]) -> TextCondition:
        blocks = tuple(
            tokenize_condition(
                self.tokenizer,
                prompt,
                text_tokens=self.prompt_tokens,
            )
            for prompt in prompts
        )
        prompt_ids = torch.stack(tuple(block.token_ids for block in blocks)).to(
            self.device
        )
        prompt_mask = torch.stack(tuple(block.content_mask for block in blocks)).to(
            self.device
        )
        prompt_stats = torch.full_like(
            prompt_ids,
            int(TextLatentStatsType.PAD_IGNORE),
        )
        prompt_stats.masked_fill_(
            prompt_mask,
            int(TextLatentStatsType.NORMAL_TEXT),
        )
        with torch.inference_mode():
            raw_prompt = self.text_encoder.encode(prompt_ids, prompt_mask)
        return TextCondition(
            latents_norm=self.stats.normalize_text(
                raw_prompt, prompt_stats, prompt_mask
            ).detach(),
            content_mask=prompt_mask,
            latent_stats_type=prompt_stats,
        )

    def _prompt(self, batch_size: int) -> TextCondition:
        return TextCondition(
            latents_norm=self.prompt_condition.latents_norm.expand(batch_size, -1, -1),
            content_mask=self.prompt_condition.content_mask.expand(batch_size, -1),
            latent_stats_type=self.prompt_condition.latent_stats_type.expand(
                batch_size, -1
            ),
        )

    def __call__(
        self,
        samples: Sequence[object],
        global_indices: Sequence[int],
        *,
        prompts: Sequence[str] | None = None,
    ) -> Tensor:
        if len(samples) != len(global_indices) or not samples:
            raise ValueError(
                "caption samples and global indices must be non-empty and aligned"
            )
        if prompts is None:
            prompt_condition = self._prompt(len(samples))
        else:
            prompt_rows = tuple(prompts)
            if len(prompt_rows) != len(samples):
                raise ValueError("caption prompts must contain one string per sample")
            if any(not isinstance(prompt, str) or not prompt for prompt in prompt_rows):
                raise ValueError("caption prompts must contain non-empty strings")
            prompt_condition = self._encode_prompts(prompt_rows)
        images = torch.stack([sample.image for sample in samples]).to(self.device)
        raw_vision = self.vision_encoder.encode(images)
        normalized_vision = self.stats.normalize_vision(raw_vision)
        target = _prefix_text_target(
            len(samples),
            self.model_text_tokens,
            self.text_tokens,
            self.device,
        )
        result = sample_text(
            self.model,
            VisionCondition(latents_norm=normalized_vision),
            prompt_condition,
            target,
            self.sampler_config,
            self.seed,
            global_sample_indices=tuple(global_indices),
            decoder=self.text_decoder,
            stop_at_eos_token_id=(
                self.eos_token_id if self.stop_at_eos else None
            ),
            block_session_factory=self.block_session_factory,
        )
        return result.token_ids[:, : self.text_tokens]


class TextOnlyGenerator:
    """Generate the standalone text distribution and render EOS as newline."""

    def __init__(
        self,
        *,
        model: nn.Module,
        text_decoder: nn.Module,
        tokenizer: EvaluationTokenizer,
        sampler_config: SamplerConfig,
        seed: int,
        max_batch_size: int = 8,
        block_session_factory: TextBlockSessionFactory | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        self.model = model
        self.text_decoder = text_decoder
        self.tokenizer = tokenizer
        self.eos_token_id = _resolve_eos_token_id(tokenizer, eos_token_id)
        self.sampler_config = sampler_config
        self.seed = seed
        self.device = _module_device(model)
        self.text_tokens = _model_text_tokens(model)
        self.max_batch_size = _require_max_batch_size(max_batch_size)
        self.block_session_factory = block_session_factory

    def __call__(self, global_indices: Sequence[int]) -> tuple[str, ...]:
        indices = tuple(global_indices)
        if not indices:
            return ()
        rendered: list[str] = []
        for start in range(0, len(indices), self.max_batch_size):
            batch_indices = indices[start : start + self.max_batch_size]
            target = _all_text_target(len(batch_indices), self.text_tokens, self.device)
            result = sample_text_unconditional(
                self.model,
                target,
                self.sampler_config,
                self.seed,
                global_sample_indices=batch_indices,
                decoder=self.text_decoder,
                block_session_factory=self.block_session_factory,
            )
            rendered.extend(
                render_token_ids(
                    row,
                    tokenizer=self.tokenizer,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.eos_token_id,
                )
                for row in result.token_ids
            )
        return tuple(rendered)


class PromptContinuationGenerator:
    """Encode a clean prefix and generate a suffix with the block-causal solver."""

    def __init__(
        self,
        *,
        model: nn.Module,
        text_encoder: nn.Module,
        text_decoder: nn.Module,
        tokenizer: EvaluationTokenizer,
        stats: LatentStatsRegistry,
        sampler_config: SamplerConfig,
        seed: int,
        target_length: int,
        max_prompt_tokens: int,
        prompt_truncation: str = "left_keep_suffix",
        max_batch_size: int = 8,
        block_session_factory: TextBlockSessionFactory | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        if type(target_length) is not int or target_length <= 0:
            raise ValueError("target_length must be a positive integer")
        if type(max_prompt_tokens) is not int or max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be a positive integer")
        if prompt_truncation not in {"left_keep_suffix", "reject"}:
            raise ValueError("prompt_truncation must be left_keep_suffix or reject")
        block_size = getattr(model, "text_block_size", None)
        if type(block_size) is not int or block_size <= 0 or target_length % block_size:
            raise ValueError(
                "target_length must be divisible by the model text block size"
            )
        self.model = model
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder
        self.tokenizer = tokenizer
        self.eos_token_id = _resolve_eos_token_id(tokenizer, eos_token_id)
        self.stats = stats
        self.sampler_config = sampler_config
        self.seed = seed
        self.target_length = target_length
        self.max_prompt_tokens = max_prompt_tokens
        self.prompt_truncation = prompt_truncation
        self.max_batch_size = _require_max_batch_size(max_batch_size)
        self.block_session_factory = block_session_factory
        self.device = _module_device(model)

    def _encode_prompts(self, prompts: Sequence[str]) -> TextCondition:
        encoded_rows = tuple(
            self.tokenizer.encode(prompt, add_special_tokens=False)
            for prompt in prompts
        )
        if any(not row for row in encoded_rows):
            raise ValueError("every benchmark prompt must encode to at least one token")
        overlong = tuple(
            len(row) for row in encoded_rows if len(row) > self.max_prompt_tokens
        )
        if overlong and self.prompt_truncation == "reject":
            raise ValueError(
                "benchmark prompt exceeds the reviewed token limit: "
                f"{max(overlong)} > {self.max_prompt_tokens}"
            )
        rows = tuple(row[-self.max_prompt_tokens :] for row in encoded_rows)
        longest = max(len(row) for row in rows)
        token_ids = torch.full(
            (len(rows), longest),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        content_mask = torch.zeros(
            (len(rows), longest), dtype=torch.bool, device=self.device
        )
        for index, row in enumerate(rows):
            token_ids[index, : len(row)] = torch.tensor(row, device=self.device)
            content_mask[index, : len(row)] = True
        stats_type = text_stats_layout(
            token_ids,
            content_mask,
            eos_token_id=self.eos_token_id,
        )
        raw = self.text_encoder.encode(token_ids, content_mask)
        return TextCondition(
            latents_norm=self.stats.normalize_text(raw, stats_type, content_mask),
            content_mask=content_mask,
            latent_stats_type=stats_type,
        )

    @staticmethod
    def _apply_stops(text: str, stops: tuple[str, ...]) -> str:
        positions: list[int] = []
        for stop in stops:
            current = 0
            while (position := text.find(stop, current)) >= 0:
                if stop != "\n" or re.search(r"\w", text[:position]):
                    positions.append(position)
                    break
                current = position + len(stop)
        return text[: min(positions)] if positions else text

    def __call__(self, requests: Sequence[TextContinuationRequest]) -> tuple[str, ...]:
        if not requests:
            return ()
        rendered: list[str] = []
        for start in range(0, len(requests), self.max_batch_size):
            batch = tuple(requests[start : start + self.max_batch_size])
            prompt = self._encode_prompts(tuple(request.prompt for request in batch))
            target = _all_text_target(len(batch), self.target_length, self.device)
            result = sample_text_continuation(
                self.model,
                prompt,
                target,
                self.sampler_config,
                self.seed,
                global_sample_indices=tuple(request.global_index for request in batch),
                decoder=self.text_decoder,
                stop_at_eos_token_id=self.eos_token_id,
                block_session_factory=self.block_session_factory,
            )
            for request, row in zip(batch, result.token_ids, strict=True):
                values = row.detach().cpu().tolist()
                try:
                    values = values[: values.index(self.eos_token_id)]
                except ValueError:
                    pass
                text = self.tokenizer.decode(values, skip_special_tokens=True)
                rendered.append(self._apply_stops(text, request.until).strip())
        return tuple(rendered)


__all__ = [
    "CaptionGenerator",
    "EvaluationTokenizer",
    "ImageOnlyGenerator",
    "PromptContinuationGenerator",
    "T2IImageGenerator",
    "TextContinuationRequest",
    "TextOnlyGenerator",
    "text_stats_layout",
]
