from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from torch.optim import Optimizer

from mf.config.schema import OptimizersConfig
from mf.training.optimizers import OptimizerBundle


class AbsoluteLRScheduler:
    """Prime the LR for the next update after each completed optimizer step.

    The documented loop is optimizer.step() followed by scheduler.step().
    completed_steps counts optimizer updates already performed, while
    lr_at_step(k) is the LR consumed by one-indexed optimizer update k.
    Step zero remains the schedule origin and is never consumed by that loop.
    """

    def __init__(
        self,
        optimizers: Sequence[Optimizer],
        *,
        warmup_steps: int,
        max_steps: int,
        peak_lr: float,
        min_lr: float,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if warmup_steps >= max_steps:
            raise ValueError("warmup_steps must be less than max_steps")
        if not 0.0 <= min_lr <= peak_lr:
            raise ValueError("learning rates must satisfy 0 <= min_lr <= peak_lr")
        self._optimizers = tuple(optimizers)
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.completed_steps = 0
        self._set_lr(self.lr_at_step(1))

    def lr_at_step(self, step: int) -> float:
        """Return the documented LR for optimizer update step."""

        _validate_step(step)
        if step <= self.warmup_steps:
            if self.warmup_steps == 0:
                return self.peak_lr
            return self.peak_lr * step / self.warmup_steps
        if step >= self.max_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def step(self, completed_steps: int | None = None) -> None:
        """Record completed updates and prime the LR for the next update."""

        next_completed = (
            self.completed_steps + 1 if completed_steps is None else completed_steps
        )
        _validate_step(next_completed)
        self._restore_completed_steps(next_completed)

    def get_last_lr(self) -> list[float]:
        """Return the LR currently primed for the next optimizer update."""

        return [
            float(group["lr"])
            for optimizer in self._optimizers
            for group in optimizer.param_groups
        ]

    def state_dict(self) -> dict[str, int]:
        return {"completed_steps": self.completed_steps}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        completed_steps = _completed_steps_from_state(state_dict)
        self._restore_completed_steps(completed_steps)

    def _restore_completed_steps(self, completed_steps: int) -> None:
        self.completed_steps = completed_steps
        self._set_lr(self.lr_at_step(completed_steps + 1))

    def _set_lr(self, lr: float) -> None:
        for optimizer in self._optimizers:
            for group in optimizer.param_groups:
                group["lr"] = lr


@dataclass(frozen=True, slots=True)
class SchedulerBundle:
    backbone: AbsoluteLRScheduler
    text_decoder: AbsoluteLRScheduler

    @property
    def completed_steps(self) -> int:
        """Return the synchronized number of completed optimizer updates."""

        if self.backbone.completed_steps != self.text_decoder.completed_steps:
            raise RuntimeError("backbone and text decoder schedulers are out of sync")
        return self.backbone.completed_steps

    def step(self, completed_steps: int | None = None) -> None:
        next_completed = (
            self.completed_steps + 1 if completed_steps is None else completed_steps
        )
        _validate_step(next_completed)
        self.backbone.step(next_completed)
        self.text_decoder.step(next_completed)

    def state_dict(self) -> dict[str, dict[str, int]]:
        return {
            "backbone": self.backbone.state_dict(),
            "text_decoder": self.text_decoder.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        if set(state_dict) != {"backbone", "text_decoder"}:
            raise ValueError("scheduler bundle state has invalid keys")
        backbone_state = state_dict["backbone"]
        decoder_state = state_dict["text_decoder"]
        if not isinstance(backbone_state, Mapping) or not isinstance(
            decoder_state,
            Mapping,
        ):
            raise TypeError("scheduler states must be mappings")
        backbone_completed = _completed_steps_from_state(backbone_state)
        decoder_completed = _completed_steps_from_state(decoder_state)
        if backbone_completed != decoder_completed:
            raise ValueError("scheduler completed_steps must match")
        self.backbone.load_state_dict(backbone_state)
        self.text_decoder.load_state_dict(decoder_state)


def _completed_steps_from_state(state_dict: Mapping[str, object]) -> int:
    if set(state_dict) != {"completed_steps"}:
        raise ValueError("scheduler state must contain only completed_steps")
    completed_steps = state_dict["completed_steps"]
    _validate_step(completed_steps)
    return completed_steps


def _validate_step(step: object) -> None:
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step must be an integer")
    if step < 0:
        raise ValueError("step must be non-negative")


def build_schedulers(
    bundle: OptimizerBundle,
    config: OptimizersConfig,
) -> SchedulerBundle:
    """Build schedules with LR primed for one-indexed optimizer update one."""

    schedule = config.schedule
    backbone_optimizers = tuple(
        optimizer
        for optimizer in (bundle.backbone_muon, bundle.backbone_adamw)
        if optimizer is not None
    )
    decoder_optimizers = tuple(
        optimizer
        for optimizer in (bundle.text_decoder_muon, bundle.text_decoder_adamw)
        if optimizer is not None
    )
    return SchedulerBundle(
        backbone=AbsoluteLRScheduler(
            backbone_optimizers,
            warmup_steps=schedule.warmup_steps,
            max_steps=schedule.max_steps,
            peak_lr=config.backbone.peak_lr,
            min_lr=config.backbone.min_lr,
        ),
        text_decoder=AbsoluteLRScheduler(
            decoder_optimizers,
            warmup_steps=schedule.warmup_steps,
            max_steps=schedule.max_steps,
            peak_lr=config.text_decoder.peak_lr,
            min_lr=config.text_decoder.min_lr,
        ),
    )
