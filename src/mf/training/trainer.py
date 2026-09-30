from __future__ import annotations

import json
import math
import os
import queue
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Protocol, TextIO

import torch
import torch.distributed as dist
from torch import Tensor, nn

from mf.config.fingerprint import training_fingerprint
from mf.config.schema import MFConfig
from mf.contracts.batch import EncodedTaskBatch, RawTaskBatch
from mf.contracts.task_registry import task_definitions
from mf.contracts.model import MFOutput
from mf.contracts.trainer import TrainerState, TrainingBatch
from mf.distributed.context import DistributedContext
from mf.modeling.chunk_adapter import chunk_causal_physical_token_counts
from mf.storage import durable_write_json
from mf.training.checkpoint import CheckpointManager
from mf.training.ema import ExponentialMovingAverage
from mf.training.execution import split_generator, training_execution_policy
from mf.training.metrics import (
    TaskMetrics,
    compose_flow_only_training_loss,
    compose_training_loss,
)
from mf.training.optimizers import OptimizerBundle
from mf.training.schedulers import SchedulerBundle

MetricValue = float | int
_LOSS_COMPONENT_NAMES = (
    "total",
    "vision_flow",
    "text_flow",
    "text_decoder_ce",
    "physical_flow",
)
_ASYNC_EMA_STREAM_ENV = "MF_ASYNC_EMA_STREAM"


class Evaluator(Protocol):
    def run(self, step: int, checkpoint: Path) -> object: ...


def _supervised_token_counts(batch: TrainingBatch) -> Tensor:
    """Return text/image loss positions; conditioning copies never count."""

    text_mask = batch.text_target_mask
    image_tokens = batch.vision_target_mask.sum(dtype=torch.float64)
    return torch.stack(
        (
            text_mask.sum(dtype=torch.float64),
            image_tokens,
        )
    )


def _physical_active_token_counts(
    config: MFConfig,
    batch: TrainingBatch,
) -> Tensor:
    if config.tasks.planner != "chunk_token_packed":
        return batch.model_input.active_token_mask.sum(dim=1, dtype=torch.float64)
    routing = batch.model_input.chunk_routing_cpu
    if routing is None:
        raise RuntimeError("chunk-token-packed metrics require CPU routing metadata")
    return chunk_causal_physical_token_counts(
        routing,
        image_chunk_conditioning=config.model.image_chunk_conditioning,
        t2i_chunk_semantics=config.model.t2i_chunk_semantics,
    ).to(dtype=torch.float64)


@contextmanager
def _temporary_float32_matmul_precision(precision: str):
    previous = torch.get_float32_matmul_precision()
    if precision == previous:
        yield
        return
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _async_ema_stream_requested() -> bool:
    raw_value = os.environ.get(_ASYNC_EMA_STREAM_ENV, "0")
    if raw_value not in {"0", "1"}:
        raise ValueError(f"{_ASYNC_EMA_STREAM_ENV} must be either 0 or 1")
    return raw_value == "1"


class BatchEncoder(Protocol):
    def encode(self, batch: RawTaskBatch) -> EncodedTaskBatch: ...


class TrainingTaskBuilder(Protocol):
    def build(
        self,
        batch: EncodedTaskBatch,
        generator: torch.Generator,
    ) -> TrainingBatch: ...


class MetricSink(Protocol):
    def write(self, metrics: dict[str, MetricValue]) -> None: ...

    def flush(self) -> None: ...


class TensorBoardMetricSink:
    """Publish the small set of metrics useful for a training dashboard."""

    @staticmethod
    def _is_dashboard_metric(name: str) -> bool:
        return (
            name.startswith("loss/")
            or name.startswith("lr/")
            or name.startswith("grad_norm")
            or name in {
                "samples_per_second",
                "active_tokens_per_second",
                "time/wall_step_seconds",
                "time/max_rank_train_step_seconds",
            }
        )

    def __init__(self, log_dir: str | Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard logging was enabled but tensorboard is not installed"
            ) from error
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def write(self, metrics: dict[str, MetricValue]) -> None:
        step = int(metrics["step"])
        for name, value in metrics.items():
            if name != "step" and self._is_dashboard_metric(name):
                self._writer.add_scalar(name, value, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class CompositeMetricSink:
    """Fan metrics out to a small fixed set of rank-zero sinks."""

    def __init__(self, *sinks: MetricSink) -> None:
        if not sinks:
            raise ValueError("CompositeMetricSink requires at least one sink")
        self._sinks = sinks

    def write(self, metrics: dict[str, MetricValue]) -> None:
        for sink in self._sinks:
            sink.write(metrics)

    def flush(self) -> None:
        for sink in self._sinks:
            flush = getattr(sink, "flush", None)
            if callable(flush):
                flush()

    def close(self) -> None:
        for sink in self._sinks:
            close = getattr(sink, "close", None)
            if callable(close):
                close()


class DeferredMetricSink:
    """Buffer bounded-run metrics and flush them when training closes."""

    def __init__(self, sink: MetricSink) -> None:
        self._sink = sink
        self._records: list[dict[str, MetricValue]] = []
        self._closed = False

    def write(self, metrics: dict[str, MetricValue]) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed deferred metric sink")
        self._records.append(dict(metrics))

    def flush(self) -> None:
        return

    def close(self) -> None:
        if self._closed:
            return
        try:
            for metrics in self._records:
                self._sink.write(metrics)
        finally:
            self._records.clear()
            close = getattr(self._sink, "close", None)
            if callable(close):
                close()
            self._closed = True


class ProgressMetricSink:
    """Write concise human progress and lossless JSON metrics on rank zero."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        max_steps: int,
        stream: TextIO | None = None,
    ) -> None:
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer")
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self._stream = sys.stdout if stream is None else stream
        self._train_log = (directory / "train.log").open("a", encoding="utf-8")
        self._metrics_log = (directory / "metrics.jsonl").open("a", encoding="utf-8")
        self._closed = False

    def write(self, metrics: dict[str, MetricValue]) -> None:
        if self._closed:
            raise RuntimeError("progress metric sink is closed")
        payload = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
        line = self._format_line(metrics)
        print(line, file=self._stream, flush=True)
        self._train_log.write(line + "\n")
        self._metrics_log.write(payload + "\n")
        self._train_log.flush()
        self._metrics_log.flush()

    def flush(self) -> None:
        if self._closed:
            return
        self._train_log.flush()
        self._metrics_log.flush()
        flush = getattr(self._stream, "flush", None)
        if callable(flush):
            flush()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._train_log.close()
        self._metrics_log.close()
        self._closed = True

    def _format_line(self, metrics: dict[str, MetricValue]) -> str:
        step = int(metrics["step"])
        step_seconds = float(metrics.get("time/train_step_seconds", 0.0))
        wall_step_seconds = float(metrics.get("time/wall_step_seconds", step_seconds))
        decoder_lr = float(
            metrics.get(
                "lr/text_decoder_muon",
                metrics.get("lr/text_decoder_adamw", 0.0),
            )
        )
        fields = [
            f"step={step}/{self.max_steps}",
            f"loss={float(metrics.get('loss/total', 0.0)):.6f}",
            f"vision={float(metrics.get('loss/vision_flow', 0.0)):.6f}",
            f"text={float(metrics.get('loss/text_flow', 0.0)):.6f}",
            f"decoder_ce={float(metrics.get('loss/text_decoder_ce', 0.0)):.6f}",
            f"lr={float(metrics.get('lr/backbone_muon', metrics.get('lr', 0.0))):.8g}",
            f"lr_decoder={float(decoder_lr or metrics.get('lr', 0.0)):.8g}",
            f"grad_norm={float(metrics.get('grad_norm', metrics.get('grad/norm', 0.0))):.6f}",
            f"step_s={step_seconds:.4f}",
            f"tokens/s={float(metrics.get('active_tokens_per_second', 0.0)):.2f}",
            f"eta={self._format_duration((self.max_steps - step) * wall_step_seconds)}",
        ]
        return " ".join(fields)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class AsyncMetricSink:
    """Move rank-zero metric serialization and I/O off the training thread."""

    _STOP = object()

    def __init__(self, sink: MetricSink, *, queue_size: int) -> None:
        if (
            isinstance(queue_size, bool)
            or not isinstance(queue_size, int)
            or queue_size <= 0
        ):
            raise ValueError("queue_size must be a positive integer")
        self._sink = sink
        self._queue: queue.Queue[dict[str, MetricValue] | object] = queue.Queue(
            queue_size
        )
        self._error: BaseException | None = None
        self._write_seconds = 0.0
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="mf-metric-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def write_seconds(self) -> float:
        return self._write_seconds

    def write(self, metrics: dict[str, MetricValue]) -> None:
        if self._closed:
            raise RuntimeError("async metric sink is closed")
        self._raise_worker_error()
        self._queue.put(dict(metrics))
        self._raise_worker_error()

    def flush(self) -> None:
        if self._closed:
            return
        self._queue.join()
        self._raise_worker_error()
        flush = getattr(self._sink, "flush", None)
        if callable(flush):
            started = time.perf_counter()
            try:
                flush()
            finally:
                self._write_seconds += time.perf_counter() - started

    def close(self) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        try:
            self.flush()
        except BaseException as caught:
            error = caught
        self._queue.put(self._STOP)
        self._thread.join()
        try:
            close = getattr(self._sink, "close", None)
            if callable(close):
                close()
        finally:
            self._closed = True
        if error is not None:
            raise error
        self._raise_worker_error()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                if self._error is None:
                    started = time.perf_counter()
                    try:
                        assert isinstance(item, dict)
                        self._sink.write(item)
                    except BaseException as error:
                        self._error = error
                    finally:
                        self._write_seconds += time.perf_counter() - started
            finally:
                self._queue.task_done()

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("asynchronous metric writer failed") from self._error


def build_metric_sink(
    log_dir: str | Path,
    *,
    max_steps: int,
    backend: str,
    defer_writes_until_close: bool = False,
    async_writes: bool = False,
    async_queue_size: int = 64,
) -> MetricSink:
    sinks: list[MetricSink] = [
        ProgressMetricSink(log_dir, max_steps=max_steps),
    ]
    if backend == "tensorboard":
        sinks.insert(0, TensorBoardMetricSink(log_dir))
    elif backend != "jsonl":
        raise ValueError(f"unsupported metric backend: {backend}")
    sink: MetricSink = CompositeMetricSink(*sinks)
    if defer_writes_until_close:
        sink = DeferredMetricSink(sink)
    if async_writes:
        sink = AsyncMetricSink(sink, queue_size=async_queue_size)
    return sink


class Trainer:
    """Explicit synchronous trainer coordinating existing MF components."""

    def __init__(
        self,
        *,
        config: MFConfig,
        model: nn.Module,
        text_decoder: nn.Module,
        batch_fetcher: Callable[[], RawTaskBatch],
        batch_encoder: BatchEncoder,
        task_builder: TrainingTaskBuilder,
        optimizers: OptimizerBundle,
        schedulers: SchedulerBundle,
        ema: ExponentialMovingAverage,
        checkpoint_manager: CheckpointManager,
        evaluator: Evaluator | None,
        training_generator: torch.Generator,
        distributed: DistributedContext,
        metric_sink: MetricSink | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not isinstance(config, MFConfig):
            raise TypeError("config must be a strict MFConfig")
        if not isinstance(model, nn.Module) or not isinstance(text_decoder, nn.Module):
            raise TypeError("model and text_decoder must be torch modules")
        if not callable(batch_fetcher):
            raise TypeError("batch_fetcher must be callable")
        if not isinstance(training_generator, torch.Generator):
            raise TypeError("training_generator must be a torch.Generator")
        if not isinstance(distributed, DistributedContext):
            raise TypeError("distributed must be a DistributedContext")
        if config.distributed.world_size != distributed.world_size:
            raise ValueError("config world_size does not match the distributed context")
        expected_global_batch = (
            distributed.world_size
            * config.distributed.micro_batch_size_per_rank
            * config.distributed.gradient_accumulation_steps
        )
        if config.distributed.global_batch_size != expected_global_batch:
            raise ValueError(
                "global_batch_size must equal world_size * micro_batch_size_per_rank * "
                "gradient_accumulation_steps"
            )
        if not optimizers.optimizers:
            raise ValueError("training requires at least one optimizer")
        if config.trainer.max_steps > config.optimizers.schedule.max_steps:
            raise ValueError(
                "trainer.max_steps cannot exceed optimizer schedule.max_steps"
            )
        self._validate_distributed_runtime(distributed)
        if _async_ema_stream_requested():
            ema.enable_async_stream()

        self.config = config
        self.model = model
        self.text_decoder = text_decoder
        self.batch_fetcher = batch_fetcher
        self.batch_encoder = batch_encoder
        self.task_builder = task_builder
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.ema = ema
        self.checkpoint_manager = checkpoint_manager
        self.evaluator = evaluator
        self.training_generator = training_generator
        self.distributed = distributed
        self.metric_sink = metric_sink
        self.clock = clock
        text_target_weight = (
            config.tasks.weights.image_to_text
            + config.tasks.weights.text_only
        )
        decoder_trainable = any(
            parameter.requires_grad for parameter in text_decoder.parameters()
        )
        self.loss_composer = (
            compose_training_loss
            if text_target_weight > 0.0 and decoder_trainable
            else compose_flow_only_training_loss
        )
        self._backbone_trainable_parameters = tuple(
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        self._decoder_trainable_parameters = tuple(
            parameter
            for parameter in self.text_decoder.parameters()
            if parameter.requires_grad
        )
        backbone_parameter_ids = {
            id(parameter) for parameter in self._backbone_trainable_parameters
        }
        decoder_parameter_ids = {
            id(parameter) for parameter in self._decoder_trainable_parameters
        }
        if backbone_parameter_ids & decoder_parameter_ids:
            raise RuntimeError("backbone and text decoder parameters must be disjoint")
        self.state = TrainerState()
        self._wall_window_started = self.clock()
        self._wall_window_step = 0
        self._last_metric_sink_write_seconds = 0.0

    def fit(self, *, resume_from: str | Path | None = None) -> TrainerState:
        with training_execution_policy(rank=self.distributed.rank) as policy:
            self.execution_gc_policy = policy
            try:
                return self._fit(resume_from=resume_from)
            finally:
                try:
                    self.ema.synchronize()
                finally:
                    self._close_metric_sink()
                    self._close_batch_fetcher()

    def _fit(self, *, resume_from: str | Path | None = None) -> TrainerState:
        if resume_from is not None:
            if self.state.global_step != 0:
                raise RuntimeError(
                    "cannot restore into a Trainer that has already advanced"
                )
            restored = self.checkpoint_manager.load(
                resume_from,
                training_fingerprint(self.config),
            )
            self.state = restored.trainer_state
            self._complete_pending_evaluations(restored.path)

        self.model.train()
        self.text_decoder.train(bool(self._decoder_trainable_parameters))
        self._wall_window_started = self.clock()
        self._wall_window_step = self.state.global_step

        while self.state.global_step < self.config.trainer.max_steps:
            self.execution_gc_policy.before_step(self.state.global_step)
            metrics = self._train_update()
            step = self.state.global_step
            checkpoint_seconds = 0.0
            evaluation_seconds = 0.0
            checkpoint: Path | None = None
            should_evaluate = self._should_evaluate(step)
            should_checkpoint = (
                step % self.config.trainer.save_steps == 0 or should_evaluate
            )

            if should_checkpoint:
                self._flush_metric_sink()
                started = self.clock()
                checkpoint_state = replace(self.state, last_checkpoint_step=step)
                checkpoint = self.checkpoint_manager.save(step, checkpoint_state)
                self.state = checkpoint_state
                checkpoint_seconds = self.clock() - started

            if should_evaluate:
                if checkpoint is None:
                    raise RuntimeError(
                        "evaluation requires a checkpoint from the same step"
                    )
                started = self.clock()
                self._run_evaluation(step, checkpoint)
                self.state = replace(self.state, last_evaluation_step=step)
                evaluation_seconds = self.clock() - started

            metrics["time/checkpoint_seconds"] = checkpoint_seconds
            metrics["time/evaluation_seconds"] = evaluation_seconds
            self._write_metrics(metrics)
            if should_checkpoint or should_evaluate:
                self._flush_metric_sink()
                self._wall_window_started = self.clock()
                self._wall_window_step = step

        return self.state

    def _train_update(self) -> dict[str, MetricValue]:
        accumulation_steps = self.config.distributed.gradient_accumulation_steps
        next_step = self.state.global_step + 1
        collect_metrics = self._should_log(next_step)
        collect_task_metrics = self._should_collect_task_metrics(next_step)
        timings = dict.fromkeys(
            (
                "data_wait",
                "encode",
                "build",
                "forward",
                "loss",
                "backward",
                "optimizer",
            ),
            0.0,
        )
        metric_totals: Tensor | None = None
        loss_components_finite: Tensor | None = None
        metric_components_finite: Tensor | None = None
        task_counts: Tensor | None = None
        active_tokens: Tensor | None = None
        active_tokens_by_task: Tensor | None = None
        supervised_tokens: Tensor | None = None
        task_metrics: TaskMetrics | None = None
        train_started = self.clock()
        self._zero_grad()

        try:
            for micro_step in range(accumulation_steps):
                started = self.clock()
                raw_batch = self.batch_fetcher()
                timings["data_wait"] += self.clock() - started
                self._validate_micro_batch(raw_batch)

                started = self.clock()
                encoded_batch = self.batch_encoder.encode(raw_batch)
                timings["encode"] += self.clock() - started

                (
                    builder_generator,
                    forward_generator,
                    loss_generator,
                ) = split_generator(self.training_generator, 3)
                started = self.clock()
                training_batch = self.task_builder.build(
                    encoded_batch,
                    builder_generator,
                )
                timings["build"] += self.clock() - started

                if collect_metrics:
                    micro_supervised = _supervised_token_counts(training_batch)
                    supervised_tokens = (
                        micro_supervised
                        if supervised_tokens is None
                        else supervised_tokens + micro_supervised
                    )
                    task_defs = task_definitions("metrics")
                    counts = torch.zeros(
                        len(task_defs),
                        dtype=torch.float64,
                        device=encoded_batch.task_type.device,
                    )
                    task_indices = torch.full_like(
                        encoded_batch.task_type,
                        fill_value=-1,
                        dtype=torch.long,
                    )
                    for index, definition in enumerate(task_defs):
                        rows = encoded_batch.task_type == definition.task_id
                        counts[index] = rows.sum(dtype=torch.float64)
                        task_indices.masked_fill_(rows, index)
                    if bool((task_indices < 0).any()):
                        raise RuntimeError("encoded batch contains an unregistered task")
                    row_active = _physical_active_token_counts(
                        self.config,
                        training_batch,
                    ).to(device=counts.device)
                    micro_active_by_task = torch.zeros_like(counts)
                    micro_active_by_task.scatter_add_(
                        0,
                        task_indices,
                        row_active,
                    )
                    task_counts = (
                        counts if task_counts is None else task_counts + counts
                    )
                    active_tokens_by_task = (
                        micro_active_by_task
                        if active_tokens_by_task is None
                        else active_tokens_by_task + micro_active_by_task
                    )
                    micro_active = row_active.sum()
                    active_tokens = (
                        micro_active
                        if active_tokens is None
                        else active_tokens + micro_active
                    )

                synchronize_gradients = micro_step == accumulation_steps - 1
                with self._gradient_sync_context(synchronize_gradients):
                    started = self.clock()
                    model_output, vision_velocity_target, text_velocity_target = (
                        self._forward_training_batch(training_batch, forward_generator)
                    )
                    timings["forward"] += self.clock() - started

                    with self._autocast_context():
                        started = self.clock()
                        losses, loss_metrics = self.loss_composer(
                            model_output,
                            training_batch,
                            self.text_decoder,
                            loss_generator,
                            self.distributed,
                            self.config,
                            collect_task_metrics=collect_task_metrics,
                            collect_loss_metrics=collect_metrics,
                            vision_velocity_target=vision_velocity_target,
                            text_velocity_target=text_velocity_target,
                        )
                        timings["loss"] += self.clock() - started

                    loss_components = torch.stack(
                        tuple(
                            getattr(losses, name).detach()
                            for name in _LOSS_COMPONENT_NAMES
                        )
                    )
                    micro_loss_components_finite = torch.isfinite(loss_components)
                    loss_components_finite = (
                        micro_loss_components_finite
                        if loss_components_finite is None
                        else loss_components_finite & micro_loss_components_finite
                    )
                    if collect_metrics:
                        loss_metric_values = torch.stack(
                            tuple(
                                getattr(loss_metrics, name).detach()
                                for name in _LOSS_COMPONENT_NAMES
                            )
                        ).to(dtype=torch.float64)
                        metric_totals = (
                            loss_metric_values
                            if metric_totals is None
                            else metric_totals + loss_metric_values
                        )
                        micro_metric_components_finite = torch.isfinite(
                            loss_metric_values
                        )
                        metric_components_finite = (
                            micro_metric_components_finite
                            if metric_components_finite is None
                            else metric_components_finite
                            & micro_metric_components_finite
                        )
                    if collect_metrics and loss_metrics.task is not None:
                        task_metrics = (
                            loss_metrics.task
                            if task_metrics is None
                            else task_metrics + loss_metrics.task
                        )
                    started = self.clock()
                    (losses.total / accumulation_steps).backward()
                    timings["backward"] += self.clock() - started

            if loss_components_finite is None:
                raise RuntimeError("training update completed without any losses")
            started = self.clock()
            backbone_grad_norm, decoder_grad_norm, backbone_clipped, decoder_clipped = (
                self._clip_gradient_norms(
                    self.config.trainer.max_grad_norm,
                    loss_components_finite=loss_components_finite,
                    metric_components_finite=metric_components_finite,
                )
            )
            with _temporary_float32_matmul_precision(
                self.config.optimizers.muon_matmul_precision
            ):
                for optimizer in self.optimizers.optimizers:
                    self.ema.step_optimizer(optimizer)
            self.schedulers.step(next_step)
            timings["optimizer"] += self.clock() - started
        except BaseException as error:
            self._zero_grad()
            self._record_rank_failure(error, step=next_step)
            raise

        self.state = TrainerState(
            global_step=next_step,
            samples_seen=self.state.samples_seen
            + self.config.distributed.global_batch_size,
            last_checkpoint_step=self.state.last_checkpoint_step,
            last_evaluation_step=self.state.last_evaluation_step,
        )
        train_seconds = max(
            self.clock() - train_started, torch.finfo(torch.float64).eps
        )
        if not collect_metrics:
            return {"step": next_step}
        if (
            task_counts is None
            or active_tokens is None
            or active_tokens_by_task is None
            or supervised_tokens is None
            or metric_totals is None
        ):
            raise RuntimeError("training update completed without any microbatches")
        if collect_task_metrics and task_metrics is None:
            raise RuntimeError("logging update completed without task metrics")
        timing_names = ("train_step", *timings, "unaccounted")
        timing_values = (
            train_seconds,
            *(timings[name] for name in timings),
            max(train_seconds - sum(timings.values()), 0.0),
        )
        local_timing = torch.tensor(
            timing_values,
            dtype=torch.float64,
            device=task_counts.device,
        )
        timing_value_count = len(timing_values)
        telemetry_started = self.clock()
        max_rank_values = self.distributed.all_reduce_detached_max(
            torch.cat(
                (
                    local_timing,
                    active_tokens.reshape(1),
                    active_tokens_by_task,
                )
            )
        )
        sum_parts = [
            task_counts,
            active_tokens.reshape(1),
            active_tokens_by_task,
            supervised_tokens,
        ]
        if task_metrics is not None:
            sum_parts.append(task_metrics.packed())
        global_values = self.distributed.all_reduce_detached_sum(torch.cat(sum_parts))
        max_rank_values_cpu = max_rank_values.detach().cpu()
        global_values_cpu = global_values.detach().cpu()
        telemetry_reduce_seconds = self.clock() - telemetry_started

        max_rank_timing = max_rank_values_cpu[:timing_value_count]
        max_rank_active_tokens = max_rank_values_cpu[timing_value_count]
        max_rank_active_tokens_by_task = max_rank_values_cpu[timing_value_count + 1 :]
        max_rank_timing_values = tuple(
            float(value) for value in max_rank_timing.tolist()
        )
        task_defs = task_definitions("metrics")
        task_count = len(task_defs)
        global_task_counts = global_values_cpu[:task_count]
        global_active_tokens = global_values_cpu[task_count]
        global_active_tokens_by_task = global_values_cpu[
            task_count + 1 : 2 * task_count + 1
        ]
        global_supervised_tokens = global_values_cpu[
            2 * task_count + 1 : 2 * task_count + 3
        ]
        reduced_task_metrics = None
        if task_metrics is not None:
            reduced_task_metrics = TaskMetrics.from_packed(
                global_values_cpu[2 * task_count + 3 :]
            )
        task_total = float(global_task_counts.sum().item())
        self._validate_observed_task_total(task_total)
        wall_window_finished = self.clock()
        wall_window_steps = next_step - self._wall_window_step
        if wall_window_steps <= 0:
            raise RuntimeError(
                "wall-clock telemetry requires monotonically increasing steps"
            )
        wall_step_seconds = max(
            (wall_window_finished - self._wall_window_started) / wall_window_steps,
            torch.finfo(torch.float64).eps,
        )
        self._wall_window_started = wall_window_finished
        self._wall_window_step = next_step
        throughput_seconds = wall_step_seconds
        used_lrs = self._current_learning_rates()
        loss_metric_values = (
            (metric_totals / accumulation_steps).detach().cpu().tolist()
        )

        metrics: dict[str, MetricValue] = {
            "step": next_step,
            "loss/total": loss_metric_values[0],
            "loss/vision_flow": loss_metric_values[1],
            "loss/text_flow": loss_metric_values[2],
            "loss/text_decoder_ce": loss_metric_values[3],
            "loss/physical_flow": loss_metric_values[4],
            "grad_norm/backbone": backbone_grad_norm,
            "grad_norm/text_decoder": decoder_grad_norm,
            "grad_norm": max(backbone_grad_norm, decoder_grad_norm),
            "grad_clip_max": self.config.trainer.max_grad_norm,
            "grad_clipped/backbone": int(backbone_clipped),
            "grad_clipped/text_decoder": int(decoder_clipped),
            "grad_clipped": int(backbone_clipped or decoder_clipped),
            "samples_per_second": (
                self.config.distributed.global_batch_size / throughput_seconds
            ),
            "active_tokens_per_second": (
                float(global_active_tokens.item()) / throughput_seconds
            ),
            "tokens/max_rank_active": int(max_rank_active_tokens.item()),
            "tokens/supervised_text": int(global_supervised_tokens[0].item()),
            "tokens/supervised_image": int(global_supervised_tokens[1].item()),
            "tokens/supervised_total": int(global_supervised_tokens.sum().item()),
            "supervised_text_tokens_per_second": (
                float(global_supervised_tokens[0].item()) / throughput_seconds
            ),
            "memory/allocated_bytes": self._cuda_memory("allocated"),
            "memory/reserved_bytes": self._cuda_memory("reserved"),
            "time/train_step_seconds": train_seconds,
            "time/wall_step_seconds": wall_step_seconds,
            "time/telemetry_reduce_seconds": telemetry_reduce_seconds,
            "time/unaccounted_seconds": timing_values[-1],
            "data/prefetch_buffered_batches": int(
                getattr(self.batch_fetcher, "buffered_batches", 0)
            ),
            "data/prefetch_capacity": int(
                getattr(self.batch_fetcher, "prefetch_capacity", 0)
            ),
        }
        for name, seconds in timings.items():
            metrics[f"time/{name}_seconds"] = seconds
        for name, seconds in zip(
            timing_names,
            max_rank_timing_values,
            strict=True,
        ):
            metrics[f"time/max_rank_{name}_seconds"] = seconds
        for index, definition in enumerate(task_defs):
            metrics[f"tokens/max_rank_active/{definition.label}"] = int(
                max_rank_active_tokens_by_task[index].item()
            )
            metrics[f"task/{definition.label}_ratio"] = (
                float(global_task_counts[index].item()) / task_total
            )
            metrics[f"task/{definition.label}_active_token_ratio"] = float(
                global_active_tokens_by_task[index].item()
            ) / float(global_active_tokens.item())
        block_causal = self.config.flow.text_block_causal
        bucket_size = block_causal.flex_sequence_bucket_size
        metrics["tokens/max_rank_flex_sequence_length"] = (
            math.ceil(int(max_rank_active_tokens.item()) / bucket_size) * bucket_size
        )
        if self.config.tasks.planner == "chunk_token_packed":
            chunk_pack = self.config.tasks.chunk_pack
            if chunk_pack is None:
                raise RuntimeError(
                    "chunk-token-packed planner is missing its pack contract"
                )
            physical_packs = (
                self.config.distributed.world_size
                * self.config.distributed.gradient_accumulation_steps
            )
            physical_capacity = physical_packs * chunk_pack.sequence_length
            metrics["logical_chunks_per_second"] = task_total / throughput_seconds
            metrics["physical_packs_per_second"] = physical_packs / throughput_seconds
            metrics["pack/active_token_utilization"] = (
                float(global_active_tokens.item()) / physical_capacity
            )
        if reduced_task_metrics is not None:
            metrics.update(reduced_task_metrics.to_scalars())
        metrics.update(used_lrs)
        return metrics

    def _forward_training_batch(
        self,
        training_batch: TrainingBatch,
        generator: torch.Generator | None = None,
    ) -> tuple[MFOutput, Tensor | None, Tensor | None]:
        generator = self.training_generator if generator is None else generator
        with self._autocast_context():
            output = self.model(training_batch.model_input)
        if not isinstance(output, MFOutput):
            raise TypeError("multimodal_flow model must return MFOutput")
        return output, None, None

    def _complete_pending_evaluations(
        self,
        restored_checkpoint: str | Path | None = None,
    ) -> None:
        if restored_checkpoint is not None:
            restored = Path(restored_checkpoint)
            local = (
                Path(self.checkpoint_manager.root)
                / f"step_{self.state.global_step:06d}"
            )
            if restored != local and self._should_evaluate(self.state.global_step):
                self._run_evaluation(self.state.global_step, restored)
                self.state = replace(
                    self.state,
                    last_evaluation_step=self.state.global_step,
                )

        while True:
            pending_step = self.checkpoint_manager.pending_eval_step()
            if pending_step is None:
                return
            if pending_step != self.state.global_step:
                raise RuntimeError(
                    "pending evaluation does not match the restored checkpoint step"
                )
            checkpoint = Path(self.checkpoint_manager.root) / f"step_{pending_step:06d}"
            if not checkpoint.is_dir():
                raise FileNotFoundError(
                    f"pending evaluation checkpoint is missing: {checkpoint}"
                )
            self._run_evaluation(pending_step, checkpoint)
            previous_evaluation = self.state.last_evaluation_step
            self.state = replace(
                self.state,
                last_evaluation_step=max(previous_evaluation or 0, pending_step),
            )
            if self.checkpoint_manager.pending_eval_step() == pending_step:
                raise RuntimeError(
                    f"evaluator did not publish completion for pending step {pending_step}"
                )

    def _run_evaluation(self, step: int, checkpoint: Path) -> None:
        if self.evaluator is None:
            raise RuntimeError("evaluation is disabled for this run")
        self.evaluator.run(step, checkpoint)

    def _should_evaluate(self, step: int) -> bool:
        return step % self.config.trainer.eval_steps == 0 or (
            self.config.trainer.eval_at_final_step
            and step == self.config.trainer.max_steps
        )

    def _validate_micro_batch(self, batch: RawTaskBatch) -> None:
        if not isinstance(batch, RawTaskBatch):
            raise TypeError("batch_fetcher must return a RawTaskBatch")
        batch.validate()
        actual = batch.task_type.shape[0]
        expected = self.config.distributed.micro_batch_size_per_rank
        if self.config.tasks.planner == "chunk_token_packed":
            if expected != 1:
                raise ValueError(
                    "chunk-token-packed training requires physical micro batch size 1"
                )
            if actual <= 0:
                raise ValueError(
                    "chunk-token-packed batch must contain at least one chunk"
                )
            return
        if actual != expected:
            raise ValueError(f"local micro batch size must be {expected}; got {actual}")

    def _validate_observed_task_total(self, task_total: float) -> None:
        if task_total <= 0:
            raise RuntimeError("observed global task count must be positive")
        if self.config.tasks.planner == "chunk_token_packed":
            return
        expected = float(self.config.distributed.global_batch_size)
        if task_total != expected:
            raise RuntimeError(
                "observed global task count does not match configured global_batch_size"
            )

    def _current_learning_rates(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for optimizer_index, optimizer in enumerate(self.optimizers.optimizers):
            for group_index, group in enumerate(optimizer.param_groups):
                group_name = group.get(
                    "group_name", f"optimizer_{optimizer_index}_{group_index}"
                )
                values[f"lr/{group_name}"] = float(group["lr"])
        return values

    def _clip_gradient_norms(
        self,
        max_norm: float,
        *,
        loss_components_finite: Tensor | None = None,
        metric_components_finite: Tensor | None = None,
    ) -> tuple[float, float, bool, bool]:
        backbone_parameters = getattr(self, "_backbone_trainable_parameters", None)
        if backbone_parameters is None:
            backbone_parameters = tuple(
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
        decoder_parameters = getattr(self, "_decoder_trainable_parameters", None)
        if decoder_parameters is None:
            decoder_parameters = tuple(
                parameter
                for parameter in self.text_decoder.parameters()
                if parameter.requires_grad
            )
        backbone_norm = self._clip_parameter_gradient_norm(
            backbone_parameters,
            max_norm,
            require_gradients=True,
        )
        if backbone_norm is None:
            raise RuntimeError("training update produced no backbone gradients")
        decoder_norm = self._clip_parameter_gradient_norm(
            decoder_parameters,
            max_norm,
            require_gradients=False,
        )
        if decoder_norm is None:
            decoder_norm = backbone_norm.new_zeros(())

        # One device-to-host transfer validates loss telemetry and independently
        # clipped modules without a synchronization bubble between backward and clipping.
        host_values = [backbone_norm, decoder_norm]
        if loss_components_finite is not None:
            host_values.insert(
                0,
                loss_components_finite.to(
                    device=backbone_norm.device,
                    dtype=backbone_norm.dtype,
                ).reshape(-1),
            )
        if metric_components_finite is not None:
            host_values.insert(
                1 if loss_components_finite is not None else 0,
                metric_components_finite.to(
                    device=backbone_norm.device,
                    dtype=backbone_norm.dtype,
                ).reshape(-1),
            )
        values = (
            torch.cat(tuple(value.reshape(-1) for value in host_values))
            .detach()
            .cpu()
            .tolist()
        )
        offset = 0
        if loss_components_finite is not None:
            loss_finite_values = values[: len(_LOSS_COMPONENT_NAMES)]
            offset += len(_LOSS_COMPONENT_NAMES)
            for name, value in zip(
                _LOSS_COMPONENT_NAMES, loss_finite_values, strict=True
            ):
                if value != 1.0:
                    raise FloatingPointError(f"non-finite {name} loss")
        if metric_components_finite is not None:
            metric_finite_values = values[offset : offset + len(_LOSS_COMPONENT_NAMES)]
            offset += len(_LOSS_COMPONENT_NAMES)
            for name, value in zip(
                _LOSS_COMPONENT_NAMES, metric_finite_values, strict=True
            ):
                if value != 1.0:
                    raise FloatingPointError(f"non-finite {name} metric")
        backbone_value, decoder_value = values[offset:]
        for name, value in (
            ("backbone", backbone_value),
            ("text decoder", decoder_value),
        ):
            if not math.isfinite(value):
                raise RuntimeError(
                    f"The total norm of order 2.0 for {name} gradients is non-finite, "
                    "so it cannot be clipped"
                )
        return (
            backbone_value,
            decoder_value,
            backbone_value > max_norm,
            decoder_value > max_norm,
        )

    @staticmethod
    def _clip_parameter_gradient_norm(
        trainable_parameters: tuple[nn.Parameter, ...],
        max_norm: float,
        *,
        require_gradients: bool,
    ) -> Tensor | None:
        parameters = [
            parameter
            for parameter in trainable_parameters
            if parameter.grad is not None
        ]
        if not parameters:
            if require_gradients:
                raise RuntimeError("training update produced no backbone gradients")
            return None
        return torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=max_norm,
            norm_type=2.0,
            error_if_nonfinite=False,
        )

    def _zero_grad(self) -> None:
        for optimizer in self.optimizers.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def _record_rank_failure(self, error: BaseException, *, step: int) -> None:
        """Leave a durable root-cause record before this rank leaves the collective.

        A rank that dies mid-step strands its peers in the next collective until the
        process group times out, and the surviving logs then show only that timeout.
        This file names the rank that actually failed and why.
        """

        try:
            directory = Path(self.config.logging.output_dir) / "rank_failures"
            directory.mkdir(parents=True, exist_ok=True)
            payload = {
                "rank": int(self.distributed.rank),
                "world_size": int(self.distributed.world_size),
                "step": int(step),
                "error_type": type(error).__name__,
                "error": str(error),
                "hostname": socket.gethostname(),
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            }
            durable_write_json(
                directory
                / f"rank_{int(self.distributed.rank):05d}_step_{int(step):09d}.json",
                payload,
            )
        except BaseException:
            # Never let diagnostics mask the original failure.
            pass

    @contextmanager
    def _gradient_sync_context(self, synchronize: bool):
        if synchronize:
            yield
            return
        with ExitStack() as stack:
            seen: set[int] = set()
            for module in (self.model, self.text_decoder):
                if id(module) in seen:
                    continue
                seen.add(id(module))
                no_sync = getattr(module, "no_sync", None)
                if callable(no_sync):
                    stack.enter_context(no_sync())
            yield

    def _autocast_context(self) -> AbstractContextManager[object]:
        device = self._training_device()
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _training_device(self) -> torch.device:
        for module in (self.model, self.text_decoder):
            parameter = next(module.parameters(), None)
            if parameter is not None:
                return parameter.device
        return torch.device(self.training_generator.device)

    def _cuda_memory(self, kind: str) -> int:
        device = self._training_device()
        if device.type != "cuda":
            return 0
        if kind == "allocated":
            return torch.cuda.memory_allocated(device)
        return torch.cuda.memory_reserved(device)

    def _write_metrics(self, metrics: dict[str, MetricValue]) -> None:
        step = int(metrics["step"])
        if not self._should_log(step) or self.distributed.rank != 0:
            return
        sink = self.metric_sink
        if sink is not None:
            write_seconds = float(getattr(sink, "write_seconds", 0.0))
            metrics["time/metric_write_seconds"] = max(
                write_seconds - self._last_metric_sink_write_seconds,
                0.0,
            )
            self._last_metric_sink_write_seconds = write_seconds
            sink.write(metrics)

    def _should_log(self, step: int) -> bool:
        return (
            step % self.config.logging.log_steps == 0
            or step == self.config.trainer.max_steps
        )

    def _should_collect_task_metrics(self, step: int) -> bool:
        if not self._should_log(step) or not self.config.logging.include_task_metrics:
            return False
        interval = self.config.logging.task_metric_log_steps
        if interval is None:
            return True
        return (
            step <= self.config.logging.dense_task_metric_steps
            or step % interval == 0
            or step == self.config.trainer.max_steps
        )

    def _flush_metric_sink(self) -> None:
        if self.distributed.rank != 0:
            return
        flush = getattr(self.metric_sink, "flush", None)
        if callable(flush):
            flush()

    def _close_metric_sink(self) -> None:
        sink = self.metric_sink
        self.metric_sink = None
        close = getattr(sink, "close", None)
        if callable(close):
            close()

    def _close_batch_fetcher(self) -> None:
        close = getattr(self.batch_fetcher, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _validate_distributed_runtime(context: DistributedContext) -> None:
        if context.world_size == 1:
            return
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized for multi-rank training"
            )
        runtime = (
            dist.get_rank(group=context.process_group),
            dist.get_world_size(group=context.process_group),
        )
        if runtime != (context.rank, context.world_size):
            raise RuntimeError(
                "distributed context does not match the initialized process group"
            )


__all__ = [
    "AsyncMetricSink",
    "CompositeMetricSink",
    "DeferredMetricSink",
    "MetricSink",
    "ProgressMetricSink",
    "TensorBoardMetricSink",
    "Trainer",
    "build_metric_sink",
]
