from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import queue
import threading
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from mf.config.schema import MFConfig
from mf.contracts.batch import RawTaskBatch, TaskType
from mf.contracts.task_registry import task_definition, task_label, task_value
from mf.data.collate import RawTaskSample, collate_raw_task_batch
from mf.data.bundle import BundleImageSample, BundleImageStream, BundleQAStream, BundleTextStream
from mf.data.chunk_pack import (
    chunk_pack_task_order,
    ElasticChunkPackLedger,
    ChunkPackUsage,
    chunk_pack_target_weights,
    chunk_pack_token_budgets,
    chunk_sample_physical_tokens,
    chunk_sample_target_tokens,
)
from mf.data.sources import normalized_data_config
from mf.data.gpic import GPICCorruptionPolicy, GPICSample, select_caption
from mf.data.physical import GPICWebTarStream, UltraFineWebParquetStream
from mf.data.planner import GlobalTaskBatchPlanner
from mf.data.streams import select_text_source
from mf.data.text import (
    TextTokenizer,
    TokenizedConditionBlock,
    TokenizedTextBlock,
    tokenize_caption,
    tokenize_condition,
    tokenizer_resume_signature,
)
from mf.registries import SAMPLER_REGISTRY, register_sampler

_PREFETCH_STALL_WARNING_SECONDS = 120.0
_PREFETCH_STALL_TIMEOUT_SECONDS = 1800.0
_PREFETCH_JOIN_TIMEOUT_SECONDS = 5.0
_PREFETCH_FACTOR_OVERRIDE_ENV = "MF_DATA_PREFETCH_FACTOR"
_PREFETCH_CAPACITY_OVERRIDE_ENV = "MF_DATA_PREFETCH_CAPACITY"
_PREFETCH_STALL_TIMEOUT_OVERRIDE_ENV = "MF_DATA_PREFETCH_STALL_TIMEOUT_SECONDS"


class _GPICStream(Protocol):
    def next_sample(self) -> GPICSample | BundleImageSample: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


class _TextStream(Protocol):
    def next_block(self, tokenizer: TextTokenizer) -> TokenizedTextBlock: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


class _RawTaskStream(Protocol):
    def next_sample(self) -> RawTaskSample: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


_RawTaskStreamFactory = Callable[
    [MFConfig, TextTokenizer, int, int],
    _RawTaskStream,
]


def _sample_text(runtime: object, task: TaskType, index: int) -> RawTaskSample:
    return RawTaskSample(
        task_type=task,
        text=runtime._text_only_block(index),
    )


def _sample_image(runtime: object, task: TaskType, index: int) -> RawTaskSample:
    del index
    sample = runtime.gpic_stream.next_sample()
    return RawTaskSample(task_type=task, image=sample.image)


def _sample_image_to_text(
    runtime: object, task: TaskType, index: int
) -> RawTaskSample:
    stream = runtime.image_to_text_task_stream
    if stream is not None:
        sample = stream.next_sample()
        if task_definition(sample.task_type).sampler != "image_to_text":
            raise RuntimeError("image-to-text task stream returned a different task")
        return sample
    sample = runtime.gpic_stream.next_sample()
    return RawTaskSample(
        task_type=task,
        image=sample.image,
        text=runtime._caption_block(sample, index),
    )


def _sample_text_to_image(
    runtime: object, task: TaskType, index: int
) -> RawTaskSample:
    stream = runtime.text_to_image_task_stream
    if stream is not None:
        sample = stream.next_sample()
        if task_definition(sample.task_type).sampler != "text_to_image":
            raise RuntimeError("text-to-image task stream returned a different task")
        return sample
    image = runtime.gpic_stream.next_sample()
    return RawTaskSample(
        task_type=task,
        image=image.image,
        text=runtime._caption_block(image, index),
    )


register_sampler("text", _sample_text, replace=True)
register_sampler("image", _sample_image, replace=True)
register_sampler("image_to_text", _sample_image_to_text, replace=True)
register_sampler("text_to_image", _sample_text_to_image, replace=True)


class _DisabledTextStream:
    def __init__(self, label: str) -> None:
        self._label = label

    def next_block(self, tokenizer: TextTokenizer) -> TokenizedTextBlock:
        del tokenizer
        raise RuntimeError(f"text stream {self._label!r} is disabled for this run")

    def state_dict(self) -> dict[str, object]:
        return {"version": 1, "label": self._label}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if dict(state) != self.state_dict():
            raise ValueError(f"disabled text stream {self._label!r} state is malformed")


class _DisabledGPICStream:
    def __init__(self, label: str) -> None:
        self._label = label

    def next_sample(self) -> GPICSample:
        raise RuntimeError(f"image stream {self._label!r} is disabled for this run")

    def state_dict(self) -> dict[str, object]:
        return {"version": 1, "label": self._label}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if dict(state) != self.state_dict():
            raise ValueError(f"disabled image stream {self._label!r} state is malformed")


class _SteppedBatchFetcher(Protocol):
    global_step: int

    def __call__(self) -> RawTaskBatch: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


def _signature(
    config: MFConfig,
    *,
    rank: int,
    tokenizer: TextTokenizer,
    stream_rank: int,
    stream_world_size: int,
    step_stride: int,
) -> str:
    payload = {
        "format": "mixed-batch-fetcher-v1",
        "data": normalized_data_config(config),
        "run_seed": config.run.seed,
        "rank": rank,
        "world_size": config.distributed.world_size,
        "stream_rank": stream_rank,
        "stream_world_size": stream_world_size,
        "step_stride": step_stride,
        "tokenizer": tokenizer_resume_signature(tokenizer),
    }
    if config.sft.enabled:
        payload["sft"] = config.sft.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class MixedBatchFetcher:
    """Build one deterministic rank-local mixed-task batch from physical streams."""

    _STATE_KEYS = frozenset(("version", "signature", "global_step", "gpic", "multi_style", "qa"))
    _VQA_STATE_KEYS = _STATE_KEYS | {"image_to_text_task"}
    _T2I_STATE_KEYS = _STATE_KEYS | {"text_to_image_task"}
    _SFT_STATE_KEYS = _STATE_KEYS | {"image_to_text_task", "text_to_image_task"}

    def __init__(
        self,
        *,
        config: MFConfig,
        rank: int,
        tokenizer: TextTokenizer,
        gpic_stream: _GPICStream,
        multi_style_stream: _TextStream,
        qa_stream: _TextStream,
        image_to_text_task_stream: _RawTaskStream | None = None,
        text_to_image_task_stream: _RawTaskStream | None = None,
        initial_global_step: int = 0,
        step_stride: int = 1,
        stream_rank: int | None = None,
        stream_world_size: int | None = None,
    ) -> None:
        normalized_data_config(config)
        if not 0 <= rank < config.distributed.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")
        if type(initial_global_step) is not int or initial_global_step < 0:
            raise ValueError("initial_global_step must be a non-negative integer")
        if type(step_stride) is not int or step_stride <= 0:
            raise ValueError("step_stride must be a positive integer")
        resolved_stream_rank = rank if stream_rank is None else stream_rank
        resolved_stream_world_size = (
            config.distributed.world_size if stream_world_size is None else stream_world_size
        )
        if not 0 <= resolved_stream_rank < resolved_stream_world_size:
            raise ValueError("stream_rank must be in [0, stream_world_size)")
        self.config = config
        self.rank = rank
        self.tokenizer = tokenizer
        self.planner = (
            None
            if config.tasks.planner == "chunk_token_packed"
            else GlobalTaskBatchPlanner.from_config(config)
        )
        self.gpic_stream = gpic_stream
        self.image_to_text_task_stream = image_to_text_task_stream
        self.text_to_image_task_stream = text_to_image_task_stream

        self.multi_style_stream = multi_style_stream
        self.qa_stream = qa_stream
        self.global_step = initial_global_step
        self._step_stride = step_stride
        self._signature = _signature(
            config,
            rank=rank,
            tokenizer=tokenizer,
            stream_rank=resolved_stream_rank,
            stream_world_size=resolved_stream_world_size,
            step_stride=step_stride,
        )
        self._local_batch_size = config.distributed.micro_batch_size_per_rank
        self._rank_batch_size = (
            self._local_batch_size * config.distributed.gradient_accumulation_steps
        )
        self._chunk_pack_ledger = ElasticChunkPackLedger()
        self.image_to_text_prompt: TokenizedConditionBlock | None = None
        if config.tasks.weights.image_to_text > 0 and image_to_text_task_stream is None:
            source = config.data.bundle or config.data.image_text
            if source is None:
                raise ValueError("image-to-text tasks require an image data source")
            instruction = source.image_to_text_instruction
            self.image_to_text_prompt = tokenize_condition(
                tokenizer,
                instruction.prompt,
                text_tokens=instruction.max_length,
            )

    @classmethod
    def from_config(
        cls,
        *,
        config: MFConfig,
        rank: int,
        tokenizer: TextTokenizer,
        stream_rank: int | None = None,
        stream_world_size: int | None = None,
        initial_global_step: int = 0,
        step_stride: int = 1,
        image_to_text_task_stream_factory: _RawTaskStreamFactory | None = None,
        text_to_image_task_stream_factory: _RawTaskStreamFactory | None = None,
    ) -> MixedBatchFetcher:
        if config.data.bundle is not None:
            resolved_stream_rank = rank if stream_rank is None else stream_rank
            resolved_stream_world_size = (
                config.distributed.world_size if stream_world_size is None else stream_world_size
            )
            image_to_text_task_stream = (
                image_to_text_task_stream_factory(
                    config,
                    tokenizer,
                    resolved_stream_rank,
                    resolved_stream_world_size,
                )
                if image_to_text_task_stream_factory is not None
                else BundleQAStream(
                    config=config,
                    rank=resolved_stream_rank,
                    world_size=resolved_stream_world_size,
                    tokenizer=tokenizer,
                )
            )
            text_to_image_task_stream = (
                text_to_image_task_stream_factory(
                    config,
                    tokenizer,
                    resolved_stream_rank,
                    resolved_stream_world_size,
                )
                if text_to_image_task_stream_factory is not None
                else None
            )
            return cls(
                config=config,
                rank=rank,
                tokenizer=tokenizer,
                initial_global_step=initial_global_step,
                step_stride=step_stride,
                stream_rank=resolved_stream_rank,
                stream_world_size=resolved_stream_world_size,
                gpic_stream=(
                    _DisabledGPICStream("bundle_image")
                    if text_to_image_task_stream_factory is not None
                    else BundleImageStream(
                        config=config,
                        rank=resolved_stream_rank,
                        world_size=resolved_stream_world_size,
                    )
                ),
                multi_style_stream=BundleTextStream(
                    config=config, rank=resolved_stream_rank, world_size=resolved_stream_world_size
                ),
                qa_stream=BundleTextStream(
                    config=config, rank=resolved_stream_rank, world_size=resolved_stream_world_size
                ),
                image_to_text_task_stream=image_to_text_task_stream,
                text_to_image_task_stream=text_to_image_task_stream,
            )
        seed = config.run.seed
        resolved_stream_rank = rank if stream_rank is None else stream_rank
        resolved_stream_world_size = (
            config.distributed.world_size if stream_world_size is None else stream_world_size
        )
        text_to_image_task_stream = (
            text_to_image_task_stream_factory(
                config,
                tokenizer,
                resolved_stream_rank,
                resolved_stream_world_size,
            )
            if text_to_image_task_stream_factory is not None
            else None
        )
        if text_to_image_task_stream_factory is not None:
            image_text_stream: _GPICStream = _DisabledGPICStream("gpic")
        else:
            if config.data.image_text is None:
                raise ValueError("image_text data is required for the default image stream")
            image_preprocessing = getattr(
                config.codecs.vision,
                "image_preprocessing",
                "legacy_center_crop_bicubic_v1",
            )
            image_text_stream = GPICWebTarStream(
                root=config.data.image_text.root,
                split=config.data.image_text.split,
                rank=resolved_stream_rank,
                world_size=resolved_stream_world_size,
                seed=seed,
                min_age_minutes=config.data.image_text.min_age_minutes,
                corruption_policy=GPICCorruptionPolicy.from_settings(
                    getattr(config.data.image_text, "corruption_policy", None)
                ),
                image_resolution=config.codecs.vision.encoder_input_resolution,
                image_preprocessing=image_preprocessing,
            )
        block_causal = config.flow.text_block_causal
        chunk_pack = config.tasks.chunk_pack
        text_packing = None if chunk_pack is None else chunk_pack.text_packing
        if text_packing is not None:
            packing_mode = text_packing.mode
            block_size = text_packing.block_size
            block_aligned_record_policy = text_packing.block_aligned_record_policy
        else:
            packing_mode = block_causal.packing
            block_size = block_causal.block_size
            block_aligned_record_policy = block_causal.block_aligned_record_policy
        image_to_text_task_stream = (
            image_to_text_task_stream_factory(
                config,
                tokenizer,
                resolved_stream_rank,
                resolved_stream_world_size,
            )
            if image_to_text_task_stream_factory is not None
            else None
        )
        text_config = config.data.text
        multi_style_stream: _TextStream = _DisabledTextStream("ultrafineweb_multi_style")
        qa_stream: _TextStream = _DisabledTextStream("ultrafineweb_qa")
        if text_config is not None and config.tasks.weights.text_only > 0:
            multi_style_stream = UltraFineWebParquetStream(
                root=text_config.ultrafineweb_multi_style.root,
                source_name="ultrafineweb_multi_style",
                rank=resolved_stream_rank,
                world_size=resolved_stream_world_size,
                seed=seed,
                text_tokens=config.data.text_max_length,
                packing_mode=packing_mode,
                block_size=block_size,
                block_aligned_record_policy=block_aligned_record_policy,
            )
        if text_config is not None and config.tasks.weights.image_only > 0:
            qa_stream = UltraFineWebParquetStream(
                root=text_config.ultrafineweb_qa.root,
                source_name="ultrafineweb_qa",
                rank=resolved_stream_rank,
                world_size=resolved_stream_world_size,
                seed=seed,
                text_tokens=config.data.text_max_length,
                packing_mode=packing_mode,
                block_size=block_size,
                block_aligned_record_policy=block_aligned_record_policy,
            )
        return cls(
            config=config,
            rank=rank,
            tokenizer=tokenizer,
            initial_global_step=initial_global_step,
            step_stride=step_stride,
            stream_rank=resolved_stream_rank,
            stream_world_size=resolved_stream_world_size,
            gpic_stream=image_text_stream,
            multi_style_stream=multi_style_stream,
            qa_stream=qa_stream,
            image_to_text_task_stream=image_to_text_task_stream,
            text_to_image_task_stream=text_to_image_task_stream,
        )

    def _plan_position(self) -> tuple[int, int]:
        accumulation_steps = self.config.distributed.gradient_accumulation_steps
        return divmod(self.global_step, accumulation_steps)

    def _caption_eos_fill_block_size(self) -> int | None:
        chunk_pack = self.config.tasks.chunk_pack
        text_packing = None if chunk_pack is None else chunk_pack.text_packing
        if text_packing is not None:
            return text_packing.block_size if text_packing.mode == "block_aligned_eos" else None
        block_causal = self.config.flow.text_block_causal
        if block_causal.packing != "block_aligned_eos":
            return None
        return block_causal.block_size

    def _global_sample_index(self, local_index: int) -> int:
        optimizer_step, accumulation_index = self._plan_position()
        return (
            optimizer_step * self.config.distributed.global_batch_size
            + self.rank * self._rank_batch_size
            + accumulation_index * self._local_batch_size
            + local_index
        )

    def _caption_block(
        self, sample: GPICSample | BundleImageSample, global_sample_index: int
    ) -> TokenizedTextBlock:
        if self.config.data.bundle is not None:
            caption_text = sample.captions["short"]
        else:
            caption_text = select_caption(
                sample.captions,
                config=self.config,
                global_sample_index=global_sample_index,
            ).text
        return tokenize_caption(
            self.tokenizer,
            caption_text,
            text_tokens=self.config.data.text_max_length,
            content_tokens=self.config.data.text_max_length,
            eos_fill_block_size=self._caption_eos_fill_block_size(),
        )

    def _text_only_block(self, global_sample_index: int) -> TokenizedTextBlock:
        if self.config.data.bundle is not None:
            return self.multi_style_stream.next_block(self.tokenizer)
        source_name = select_text_source(
            config=self.config,
            global_sample_index=global_sample_index,
        )
        if source_name == "ultrafineweb_multi_style":
            return self.multi_style_stream.next_block(self.tokenizer)
        if source_name == "ultrafineweb_qa":
            return self.qa_stream.next_block(self.tokenizer)
        if source_name == "gpic_caption_text":
            return self._caption_block(self.gpic_stream.next_sample(), global_sample_index)
        raise RuntimeError(f"unknown mixed text source {source_name!r}")

    def _sample(self, task: TaskType, global_sample_index: int) -> RawTaskSample:
        sampler = task_definition(task).sampler
        return SAMPLER_REGISTRY.resolve(sampler)(
            self,
            task,
            global_sample_index,
        )

    def _packed_sample_index(self, local_index: int) -> int:
        chunk_pack = self.config.tasks.chunk_pack
        if chunk_pack is None:
            raise RuntimeError("packed sample indexing requires an chunk-pack contract")
        optimizer_step, accumulation_index = self._plan_position()
        accumulation_steps = self.config.distributed.gradient_accumulation_steps
        pack_index = (
            optimizer_step * self.config.distributed.world_size * accumulation_steps
            + self.rank * accumulation_steps
            + accumulation_index
        )
        logical_chunks = chunk_pack.logical_chunks_per_pack
        if logical_chunks is not None:
            if not 0 <= local_index < logical_chunks:
                raise RuntimeError("chunk pack exceeds logical_chunks_per_pack")
            return pack_index * logical_chunks + local_index
        if not 0 <= local_index < 1_000_000:
            raise RuntimeError("chunk pack contains too many logical samples")
        return pack_index * 1_000_000 + local_index

    def _stream_snapshot(self) -> dict[str, object]:
        snapshot = {
            "gpic": copy.deepcopy(self.gpic_stream.state_dict()),
            "multi_style": copy.deepcopy(self.multi_style_stream.state_dict()),
            "qa": copy.deepcopy(self.qa_stream.state_dict()),
        }
        task_stream = self.image_to_text_task_stream
        if task_stream is not None:
            snapshot["image_to_text_task"] = copy.deepcopy(task_stream.state_dict())
        task_stream = self.text_to_image_task_stream
        if task_stream is not None:
            snapshot["text_to_image_task"] = copy.deepcopy(task_stream.state_dict())
        return snapshot

    def _restore_stream_snapshot(self, snapshot: Mapping[str, object]) -> None:
        gpic = snapshot.get("gpic")
        multi_style = snapshot.get("multi_style")
        qa = snapshot.get("qa")
        if not all(isinstance(value, Mapping) for value in (gpic, multi_style, qa)):
            raise RuntimeError("chunk pack stream snapshot is malformed")
        self.gpic_stream.load_state_dict(gpic)
        self.multi_style_stream.load_state_dict(multi_style)
        self.qa_stream.load_state_dict(qa)
        task_stream = self.image_to_text_task_stream
        if task_stream is not None:
            task_state = snapshot.get("image_to_text_task")
            if not isinstance(task_state, Mapping):
                raise RuntimeError("chunk pack image-to-text snapshot is malformed")
            task_stream.load_state_dict(task_state)
        task_stream = self.text_to_image_task_stream
        if task_stream is not None:
            task_state = snapshot.get("text_to_image_task")
            if not isinstance(task_state, Mapping):
                raise RuntimeError("chunk pack text-to-image snapshot is malformed")
            task_stream.load_state_dict(task_state)

    def _elastic_chunk_token_packed_batch(
        self,
        *,
        budgets: Mapping[TaskType, int],
        prompt_tokens: int,
    ) -> tuple[RawTaskBatch, ChunkPackUsage]:
        chunk_pack = self.config.tasks.chunk_pack
        if chunk_pack is None:
            raise RuntimeError("chunk-token-packed planner is missing its pack contract")
        target_weights = chunk_pack_target_weights(self.config.tasks.weights)
        planning_vision_tokens = self.config.codecs.vision.latent_tokens
        deferred_samples = getattr(self, "_chunk_pack_deferred_samples", None)
        if deferred_samples is None:
            deferred_samples = {}
            self._chunk_pack_deferred_samples = deferred_samples

        samples: list[RawTaskSample] = []
        physical_by_task = {task: 0 for task in chunk_pack_task_order()}
        target_by_task = {task: 0 for task in chunk_pack_task_order()}
        sample_counts = {task: 0 for task in chunk_pack_task_order()}
        overflow_counts = {task: 0 for task in chunk_pack_task_order()}
        defer_counts = {task: 0 for task in chunk_pack_task_order()}
        task_reassignment_counts = {task: 0 for task in chunk_pack_task_order()}
        blocked: set[TaskType] = set()
        logical_index = 0
        used_total = 0
        expected_chunks = chunk_pack.logical_chunks_per_pack
        while used_total < chunk_pack.sequence_length:
            if expected_chunks is not None and logical_index >= expected_chunks:
                break
            task = self._chunk_pack_ledger.select_task(
                budgets=budgets,
                current_tokens=physical_by_task,
                current_target_tokens=target_by_task,
                current_sample_counts=sample_counts,
                target_weights=target_weights,
                exposure_basis=chunk_pack.exposure_basis,
                blocked=blocked,
            )
            if task is None:
                break
            sample = deferred_samples.pop(task, None)
            snapshot = None
            if sample is None:
                snapshot = self._stream_snapshot()
                sample = self._sample(task, self._packed_sample_index(logical_index))
            if sample.task_type is not task:
                self._chunk_pack_ledger.record_task_reassignment(task)
                raise RuntimeError(
                    "deferred chunk-pack sample changed task identity "
                    f"from {task_label(sample.task_type)} to {task_label(task)}"
                )

            physical_cost = chunk_sample_physical_tokens(
                sample,
                image_to_text_prompt_tokens=prompt_tokens,
                image_chunk_conditioning=self.config.model.image_chunk_conditioning,
                t2i_chunk_semantics=self.config.model.t2i_chunk_semantics,
                vision_tokens=planning_vision_tokens,
            )
            if chunk_pack.exposure_basis == "physical_tokens":
                target_cost = physical_cost
            elif planning_vision_tokens == 256:
                target_cost = chunk_sample_target_tokens(sample)
            else:
                target_cost = chunk_sample_target_tokens(
                    sample,
                    vision_tokens=planning_vision_tokens,
                )
            if physical_cost <= 0 or target_cost <= 0:
                if snapshot is not None:
                    self._restore_stream_snapshot(snapshot)
                raise RuntimeError(
                    f"{task_label(task)} chunk has non-positive physical or target token cost"
                )
            if physical_cost > chunk_pack.sequence_length:
                if snapshot is not None:
                    self._restore_stream_snapshot(snapshot)
                raise RuntimeError(
                    f"{task_label(task)} chunk costs {physical_cost} physical tokens, exceeding "
                    f"the fixed pack length {chunk_pack.sequence_length}"
                )
            if used_total + physical_cost > chunk_pack.sequence_length:
                deferred_samples[task] = sample
                overflow_counts[task] += 1
                defer_counts[task] += 1
                blocked.add(task)
                continue

            samples.append(sample)
            physical_by_task[task] += physical_cost
            target_by_task[task] += target_cost
            sample_counts[task] += 1
            used_total += physical_cost
            logical_index += 1

        if not samples:
            raise RuntimeError("elastic chunk pack contains no complete chunk")
        if expected_chunks is not None and len(samples) != expected_chunks:
            raise RuntimeError(
                f"chunk pack contains {len(samples)} chunks; expected {expected_chunks}"
            )
        batch = collate_raw_task_batch(
            samples,
            pad_token_id=int(self.tokenizer.pad_token_id),
            text_tokens=self.config.data.text_max_length,
            vision_tokens=self.config.codecs.vision.latent_tokens,
            image_to_text_prompt=self.image_to_text_prompt,
            image_resolution=self.config.codecs.vision.encoder_input_resolution,
        )
        usage = ChunkPackUsage(
            sample_counts=sample_counts,
            physical_tokens=physical_by_task,
            target_tokens=target_by_task,
            overflow_counts=overflow_counts,
            defer_counts=defer_counts,
            task_reassignment_counts=task_reassignment_counts,
        )
        return batch, usage

    def _chunk_token_packed_batch(self) -> RawTaskBatch:
        chunk_pack = self.config.tasks.chunk_pack
        if chunk_pack is None:
            raise RuntimeError("chunk-token-packed planner is missing its pack contract")
        prompt_tokens = (
            int(self.image_to_text_prompt.content_mask.sum().item())
            if self.image_to_text_prompt is not None
            else (
                self.config.sft.vqa.prompt_max_length
                if self.config.sft.enabled and self.config.sft.vqa is not None
                else 0
            )
        )
        budgets = chunk_pack_token_budgets(chunk_pack)
        if chunk_pack.allocation == "elastic":
            batch, usage = self._elastic_chunk_token_packed_batch(
                budgets=budgets,
                prompt_tokens=prompt_tokens,
            )
            self._chunk_pack_ledger.record_pack(usage)
            return batch
        samples: list[RawTaskSample] = []
        logical_index = 0
        used_total = 0
        expected_chunks = chunk_pack.logical_chunks_per_pack
        for task in chunk_pack_task_order():
            budget = budgets[task]
            if budget == 0:
                continue
            used = 0
            while used < budget:
                snapshot = self._stream_snapshot()
                if expected_chunks is not None and logical_index >= expected_chunks:
                    break
                sample = self._sample(task, self._packed_sample_index(logical_index))
                token_cost = chunk_sample_physical_tokens(
                    sample,
                    image_to_text_prompt_tokens=prompt_tokens,
                    image_chunk_conditioning=self.config.model.image_chunk_conditioning,
                    t2i_chunk_semantics=self.config.model.t2i_chunk_semantics,
                    vision_tokens=self.config.codecs.vision.latent_tokens,
                )
                if token_cost > budget:
                    self._restore_stream_snapshot(snapshot)
                    raise RuntimeError(
                        f"{task_label(task)} chunk costs {token_cost} tokens, exceeding "
                        f"its per-pack budget {budget}"
                    )
                if used + token_cost > budget:
                    self._restore_stream_snapshot(snapshot)
                    break
                samples.append(sample)
                used += token_cost
                used_total += token_cost
                logical_index += 1
            if used == 0:
                raise RuntimeError(f"chunk pack contains no {task_label(task)} chunk")
        if used_total > chunk_pack.sequence_length:
            raise RuntimeError("chunk pack exceeds its fixed physical sequence length")
        if expected_chunks is not None and len(samples) != expected_chunks:
            raise RuntimeError(
                f"chunk pack contains {len(samples)} chunks; expected {expected_chunks}"
            )
        return collate_raw_task_batch(
            samples,
            pad_token_id=int(self.tokenizer.pad_token_id),
            text_tokens=self.config.data.text_max_length,
            vision_tokens=self.config.codecs.vision.latent_tokens,
            image_to_text_prompt=self.image_to_text_prompt,
            image_resolution=self.config.codecs.vision.encoder_input_resolution,
        )

    def __call__(self) -> RawTaskBatch:
        if self.config.tasks.planner == "chunk_token_packed":
            batch = self._chunk_token_packed_batch()
            self.global_step += self._step_stride
            return batch
        optimizer_step, accumulation_index = self._plan_position()
        if self.planner is None:
            raise RuntimeError("non-packed task planner is unavailable")
        rank_plan = self.planner.plan_for_rank(optimizer_step, rank=self.rank)
        start = accumulation_index * self._local_batch_size
        plan = rank_plan[start : start + self._local_batch_size]
        samples = tuple(
            self._sample(task, self._global_sample_index(local_index))
            for local_index, task in enumerate(plan)
        )
        batch = collate_raw_task_batch(
            samples,
            pad_token_id=int(self.tokenizer.pad_token_id),
            text_tokens=self.config.data.text_max_length,
            vision_tokens=self.config.codecs.vision.latent_tokens,
            image_to_text_prompt=self.image_to_text_prompt,
            image_resolution=self.config.codecs.vision.encoder_input_resolution,
        )
        self.global_step += self._step_stride
        return batch

    @property
    def chunk_pack_counters(self) -> Mapping[str, object]:
        chunk_pack = self.config.tasks.chunk_pack
        if chunk_pack is None or chunk_pack.allocation != "elastic":
            return {}
        return self._chunk_pack_ledger.counter_snapshot()

    def warm_prefetch(self) -> None:
        for stream in (
            self.gpic_stream,
            self.image_to_text_task_stream,
            self.text_to_image_task_stream,
        ):
            if stream is None:
                continue
            warm_next_shard = getattr(stream, "warm_next_shard", None)
            if warm_next_shard is not None:
                warm_next_shard()

    def state_dict(self) -> dict[str, object]:
        image_to_text_stream = self.image_to_text_task_stream
        text_to_image_stream = self.text_to_image_task_stream
        if image_to_text_stream is not None and text_to_image_stream is not None:
            version = 5
        elif text_to_image_stream is not None:
            version = 4
        elif image_to_text_stream is not None:
            version = 3
        else:
            version = 1
        state: dict[str, object] = {
            "version": version,
            "signature": self._signature,
            "global_step": self.global_step,
            "gpic": self.gpic_stream.state_dict(),
            "multi_style": self.multi_style_stream.state_dict(),
            "qa": self.qa_stream.state_dict(),
        }
        if image_to_text_stream is not None:
            state["image_to_text_task"] = image_to_text_stream.state_dict()
        if text_to_image_stream is not None:
            state["text_to_image_task"] = text_to_image_stream.state_dict()
        chunk_pack = self.config.tasks.chunk_pack
        if chunk_pack is not None and chunk_pack.allocation == "elastic":
            state["chunk_pack"] = self._chunk_pack_ledger.state_dict()
            deferred = getattr(self, "_chunk_pack_deferred_samples", {})
            if deferred:
                state["deferred_samples"] = {
                    task_label(task): {**asdict(sample), "task_type": int(sample.task_type)}
                    for task, sample in deferred.items()
                }
        return state

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        image_to_text_stream = self.image_to_text_task_stream
        text_to_image_stream = self.text_to_image_task_stream
        if image_to_text_stream is not None and text_to_image_stream is not None:
            expected_keys = self._SFT_STATE_KEYS
            expected_version = 5
        elif text_to_image_stream is not None:
            expected_keys = self._T2I_STATE_KEYS
            expected_version = 4
        elif image_to_text_stream is not None:
            expected_keys = self._VQA_STATE_KEYS
            expected_version = 3
        else:
            expected_keys = self._STATE_KEYS
            expected_version = 1
        chunk_pack = self.config.tasks.chunk_pack
        if chunk_pack is not None and chunk_pack.allocation == "elastic":
            expected_keys = expected_keys | {"chunk_pack"}
            if "deferred_samples" in state:
                expected_keys = expected_keys | {"deferred_samples"}
        if set(state) != expected_keys:
            raise ValueError("mixed batch fetcher state fields are malformed")
        signature = state.get("signature")
        if state.get("version") != expected_version or signature != self._signature:
            raise ValueError("mixed batch fetcher state does not match this run")
        global_step = state.get("global_step")
        if type(global_step) is not int or global_step < 0:
            raise ValueError("mixed batch fetcher global_step is malformed")
        deferred = {}
        raw_deferred = state.get("deferred_samples", {})
        if not isinstance(raw_deferred, Mapping):
            raise ValueError("chunk pack deferred samples are malformed")
        for label, raw_sample in raw_deferred.items():
            if not isinstance(raw_sample, Mapping):
                raise ValueError("chunk pack deferred sample is malformed")
            values = copy.deepcopy(dict(raw_sample))
            task = task_value(task_definition(int(values["task_type"])))
            if task not in chunk_pack_task_order() or task_label(task) != label:
                raise ValueError("chunk pack deferred sample task mismatch")
            values["task_type"] = task
            for field, cls in (
                ("text", TokenizedTextBlock),
                ("text_prompt", TokenizedConditionBlock),
            ):
                if values.get(field) is not None:
                    values[field] = cls(**values[field])
            deferred[task] = RawTaskSample(**values)
        bindings = [
            ("gpic", self.gpic_stream),
            ("multi_style", self.multi_style_stream),
            ("qa", self.qa_stream),
        ]
        if image_to_text_stream is not None:
            bindings.append(("image_to_text_task", image_to_text_stream))
        if text_to_image_stream is not None:
            bindings.append(("text_to_image_task", text_to_image_stream))
        for name, stream in bindings:
            stream_state = state.get(name)
            if not isinstance(stream_state, Mapping):
                raise ValueError("mixed batch fetcher stream state is malformed")
            stream.load_state_dict(stream_state)
        if chunk_pack is not None and chunk_pack.allocation == "elastic":
            chunk_pack_state = state.get("chunk_pack")
            if not isinstance(chunk_pack_state, Mapping):
                raise ValueError("chunk pack fetcher state is malformed")
            self._chunk_pack_ledger = ElasticChunkPackLedger.from_state_dict(chunk_pack_state)
        self.global_step = global_step
        self._chunk_pack_deferred_samples = deferred


@dataclass(frozen=True, slots=True)
class _PrefetchResult:
    step: int
    batch: RawTaskBatch | None
    post_state: Mapping[str, object] | None
    error: Exception | None


class _PrefetchWorker:
    def __init__(
        self,
        worker_id: int,
        fetcher: _SteppedBatchFetcher,
        *,
        prefetch_factor: int,
        prepare_batch: Callable[[RawTaskBatch], RawTaskBatch] | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.fetcher = fetcher
        self.prepare_batch = prepare_batch
        self.results: queue.Queue[_PrefetchResult] = queue.Queue(maxsize=prefetch_factor)
        self.consumed_state = copy.deepcopy(fetcher.state_dict())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("prefetch worker is already running")
        if not self.results.empty():
            raise RuntimeError("prefetch worker queue must be empty before start")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"mf-prefetch-{self.worker_id}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            step = self.fetcher.global_step
            try:
                if not self._wait_for_capacity():
                    return
                step = self.fetcher.global_step
                batch = self.fetcher()
                if self.prepare_batch is not None:
                    batch = self.prepare_batch(batch)
                result = _PrefetchResult(
                    step=step,
                    batch=batch,
                    post_state=copy.deepcopy(self.fetcher.state_dict()),
                    error=None,
                )
            except Exception as error:
                result = _PrefetchResult(
                    step=step,
                    batch=None,
                    post_state=None,
                    error=error,
                )
            if not self._put(result):
                return
            if result.error is not None:
                return

    def _wait_for_capacity(self) -> bool:
        warm_prefetch = getattr(self.fetcher, "warm_prefetch", None)
        while self.results.full():
            if warm_prefetch is not None:
                warm_prefetch()
            if self._stop.wait(timeout=0.05):
                return False
        return not self._stop.is_set()

    def _put(self, result: _PrefetchResult) -> bool:
        while not self._stop.is_set():
            try:
                self.results.put(result, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def take(self) -> _PrefetchResult:
        started = time.monotonic()
        next_warning = started + _PREFETCH_STALL_WARNING_SECONDS
        stall_timeout = _prefetch_stall_timeout_seconds()
        while True:
            wait_seconds = min(
                0.05,
                max(next_warning - time.monotonic(), 0.0),
            )
            try:
                return self.results.get(timeout=wait_seconds)
            except queue.Empty:
                if self._thread is None or not self._thread.is_alive():
                    raise RuntimeError(f"prefetch worker {self.worker_id} stopped without a result")
                stalled_seconds = time.monotonic() - started
                if stalled_seconds >= stall_timeout:
                    # A live-but-stuck worker (an uninterruptible storage read, for
                    # example) would otherwise block this rank forever and strand every
                    # other rank in the next collective until the process group times
                    # out. Fail closed so the job restarts instead of idling.
                    raise RuntimeError(
                        f"prefetch worker {self.worker_id} produced no result for "
                        f"{stalled_seconds:.0f} seconds while its thread stayed alive; "
                        f"exceeded the {stall_timeout:.0f}s stall budget "
                        f"(raise {_PREFETCH_STALL_TIMEOUT_OVERRIDE_ENV} for slower storage)"
                    )
                if time.monotonic() >= next_warning:
                    warnings.warn(
                        f"prefetch worker {self.worker_id} has produced no result for "
                        f"{stalled_seconds:.0f} seconds; continuing to wait for slow "
                        f"storage until the {stall_timeout:.0f}s stall budget expires",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    next_warning = time.monotonic() + _PREFETCH_STALL_WARNING_SECONDS

    def mark_consumed(self, state: Mapping[str, object]) -> None:
        self.consumed_state = copy.deepcopy(state)

    def request_stop(self) -> None:
        self._stop.set()

    def stop_and_reset(self, *, strict: bool) -> None:
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=_PREFETCH_JOIN_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                if strict:
                    raise RuntimeError(
                        f"prefetch worker {self.worker_id} did not stop within "
                        f"{_PREFETCH_JOIN_TIMEOUT_SECONDS:.0f} seconds"
                    )
                return
        self._thread = None
        while True:
            try:
                self.results.get_nowait()
            except queue.Empty:
                break
        self.fetcher.load_state_dict(copy.deepcopy(self.consumed_state))

    def load_consumed_state(self, state: Mapping[str, object]) -> None:
        self.fetcher.load_state_dict(copy.deepcopy(state))
        self.consumed_state = copy.deepcopy(state)


def _resolved_prefetch_factor(configured: int, *, worker_count: int = 1) -> int:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    raw_capacity = os.environ.get(_PREFETCH_CAPACITY_OVERRIDE_ENV)
    raw = os.environ.get(_PREFETCH_FACTOR_OVERRIDE_ENV)
    if raw_capacity is not None and raw is not None:
        raise ValueError(
            f"set only one of {_PREFETCH_CAPACITY_OVERRIDE_ENV} and {_PREFETCH_FACTOR_OVERRIDE_ENV}"
        )
    if raw_capacity is not None:
        capacity = _parse_positive_override(_PREFETCH_CAPACITY_OVERRIDE_ENV, raw_capacity)
        return max(1, (capacity + worker_count - 1) // worker_count)
    if raw is None:
        return configured
    return _parse_positive_override(_PREFETCH_FACTOR_OVERRIDE_ENV, raw)


def _parse_positive_override(name: str, raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _prefetch_stall_timeout_seconds() -> float:
    raw = os.environ.get(_PREFETCH_STALL_TIMEOUT_OVERRIDE_ENV)
    if raw is None:
        return _PREFETCH_STALL_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            f"{_PREFETCH_STALL_TIMEOUT_OVERRIDE_ENV} must be a positive number"
        ) from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{_PREFETCH_STALL_TIMEOUT_OVERRIDE_ENV} must be a positive number")
    return value


def _prefetch_signature(
    config: MFConfig,
    *,
    rank: int,
    tokenizer: TextTokenizer,
) -> str:
    payload = {
        "format": "ordered-prefetch-batch-fetcher-v1",
        "data": normalized_data_config(config),
        "run_seed": config.run.seed,
        "rank": rank,
        "world_size": config.distributed.world_size,
        "loader": config.data.loader.model_dump(mode="json"),
        "tokenizer": tokenizer_resume_signature(tokenizer),
    }
    if config.sft.enabled:
        payload["sft"] = config.sft.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class OrderedPrefetchBatchFetcher:
    """Deliver deterministic worker batches in order with checkpoint-safe rollback."""

    _STATE_KEYS = frozenset(("version", "signature", "next_step", "workers"))

    def __init__(
        self,
        workers: Sequence[_SteppedBatchFetcher],
        *,
        prefetch_factor: int,
        signature: str,
        prepare_batch: Callable[[RawTaskBatch], RawTaskBatch] | None = None,
    ) -> None:
        if not workers:
            raise ValueError("ordered prefetch requires at least one worker")
        if type(prefetch_factor) is not int or prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be a positive integer")
        if not isinstance(signature, str) or not signature:
            raise ValueError("signature must be a non-empty string")
        self._workers = tuple(
            _PrefetchWorker(
                index, worker, prefetch_factor=prefetch_factor, prepare_batch=prepare_batch
            )
            for index, worker in enumerate(workers)
        )
        self._prefetch_factor = prefetch_factor
        self._signature = signature
        self._next_step = 0
        self._lock = threading.RLock()
        self._closed = False
        self._started = False
        self._validate_worker_steps()

    @classmethod
    def from_config(
        cls,
        *,
        config: MFConfig,
        rank: int,
        tokenizer: TextTokenizer,
        prepare_batch: Callable[[RawTaskBatch], RawTaskBatch] | None = None,
        image_to_text_task_stream_factory: _RawTaskStreamFactory | None = None,
        text_to_image_task_stream_factory: _RawTaskStreamFactory | None = None,
    ) -> OrderedPrefetchBatchFetcher:
        worker_count = config.data.loader.num_workers
        if worker_count <= 0:
            raise ValueError("ordered prefetch requires data.loader.num_workers > 0")
        stream_world_size = config.distributed.world_size * worker_count
        workers = tuple(
            MixedBatchFetcher.from_config(
                config=config,
                rank=rank,
                tokenizer=tokenizer,
                stream_rank=rank * worker_count + worker_id,
                stream_world_size=stream_world_size,
                initial_global_step=worker_id,
                step_stride=worker_count,
                image_to_text_task_stream_factory=image_to_text_task_stream_factory,
                text_to_image_task_stream_factory=text_to_image_task_stream_factory,
            )
            for worker_id in range(worker_count)
        )
        return cls(
            workers,
            prefetch_factor=_resolved_prefetch_factor(
                config.data.loader.prefetch_factor,
                worker_count=worker_count,
            ),
            signature=_prefetch_signature(config, rank=rank, tokenizer=tokenizer),
            prepare_batch=prepare_batch,
        )

    @property
    def buffered_batches(self) -> int:
        return sum(worker.results.qsize() for worker in self._workers)

    @property
    def prefetch_capacity(self) -> int:
        return len(self._workers) * self._prefetch_factor

    def __call__(self) -> RawTaskBatch:
        with self._lock:
            if self._closed:
                raise RuntimeError("ordered prefetch batch fetcher is closed")
            if not self._started:
                self._start_workers()
            worker = self._workers[self._next_step % len(self._workers)]
            result = worker.take()
            if result.step != self._next_step:
                raise RuntimeError(
                    "prefetch result order mismatch: "
                    f"expected step {self._next_step}, got {result.step}"
                )
            if result.error is not None:
                raise RuntimeError(
                    f"prefetch worker {worker.worker_id} failed at step {result.step}"
                ) from result.error
            if result.batch is None or result.post_state is None:
                raise RuntimeError("successful prefetch result is incomplete")
            worker.mark_consumed(result.post_state)
            self._next_step += 1
            return result.batch

    def state_dict(self) -> dict[str, object]:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot checkpoint a closed prefetch batch fetcher")
            return {
                "version": 1,
                "signature": self._signature,
                "next_step": self._next_step,
                "workers": tuple(copy.deepcopy(worker.consumed_state) for worker in self._workers),
            }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != self._STATE_KEYS:
            raise ValueError("ordered prefetch state fields are malformed")
        signature = state.get("signature")
        if state.get("version") != 1 or signature != self._signature:
            raise ValueError("ordered prefetch state does not match this run")
        next_step = state.get("next_step")
        worker_states = state.get("workers")
        if type(next_step) is not int or next_step < 0:
            raise ValueError("ordered prefetch next_step is malformed")
        if (
            not isinstance(worker_states, Sequence)
            or isinstance(worker_states, (str, bytes))
            or len(worker_states) != len(self._workers)
            or any(not isinstance(worker_state, Mapping) for worker_state in worker_states)
        ):
            raise ValueError("ordered prefetch worker states are malformed")

        with self._lock:
            if self._closed:
                raise RuntimeError("cannot restore a closed prefetch batch fetcher")
            if next_step == self._next_step and all(
                worker_state == worker.consumed_state
                for worker, worker_state in zip(self._workers, worker_states, strict=True)
            ):
                return
            was_started = self._started
            if was_started:
                self._stop_and_reset_workers()
            for worker, worker_state in zip(
                self._workers,
                worker_states,
                strict=True,
            ):
                worker.load_consumed_state(worker_state)
            self._next_step = next_step
            self._validate_worker_steps()
            if was_started:
                self._start_workers()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._started:
                self._stop_and_reset_workers(strict=False)
            self._closed = True

    def _start_workers(self) -> None:
        if self._started:
            raise RuntimeError("ordered prefetch workers are already running")
        for worker in self._workers:
            worker.start()
        self._started = True

    def _stop_and_reset_workers(self, *, strict: bool = True) -> None:
        if not self._started:
            return
        for worker in self._workers:
            worker.request_stop()
        for worker in self._workers:
            worker.stop_and_reset(strict=strict)
        self._started = False

    def _validate_worker_steps(self) -> None:
        worker_count = len(self._workers)
        for worker_id, worker in enumerate(self._workers):
            expected = self._next_step + (worker_id - self._next_step) % worker_count
            if worker.fetcher.global_step != expected:
                raise ValueError(
                    f"prefetch worker {worker_id} expected step {expected}; "
                    f"got {worker.fetcher.global_step}"
                )


BatchFetcher = MixedBatchFetcher | OrderedPrefetchBatchFetcher


def build_batch_fetcher(
    *,
    config: MFConfig,
    rank: int,
        tokenizer: TextTokenizer,
        prepare_batch: Callable[[RawTaskBatch], RawTaskBatch] | None = None,
        image_to_text_task_stream_factory: _RawTaskStreamFactory | None = None,
        text_to_image_task_stream_factory: _RawTaskStreamFactory | None = None,
    ) -> BatchFetcher:
    normalized_data_config(config)
    if config.data.loader.num_workers == 0:
        if prepare_batch is not None:
            raise ValueError("CPU batch preparation requires ordered prefetch workers")
        return MixedBatchFetcher.from_config(
            config=config,
            rank=rank,
            tokenizer=tokenizer,
            image_to_text_task_stream_factory=image_to_text_task_stream_factory,
            text_to_image_task_stream_factory=text_to_image_task_stream_factory,
        )
    return OrderedPrefetchBatchFetcher.from_config(
        config=config,
        rank=rank,
        tokenizer=tokenizer,
        prepare_batch=prepare_batch,
        image_to_text_task_stream_factory=image_to_text_task_stream_factory,
        text_to_image_task_stream_factory=text_to_image_task_stream_factory,
    )
