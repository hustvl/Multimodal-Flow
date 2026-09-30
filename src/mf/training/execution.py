"""Opt-in training execution policies."""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from contextlib import contextmanager

import torch

_ENV = "MF_TRAIN_GC_FREEZE_AFTER_UPDATES"
_MAX_SEED = torch.iinfo(torch.int64).max


def split_generator(
    generator: torch.Generator,
    count: int,
) -> tuple[torch.Generator, ...]:
    """Create deterministic RNG domains while advancing the parent a fixed amount."""

    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    device = torch.device(generator.device)
    seeds = torch.randint(
        0,
        _MAX_SEED,
        (count,),
        generator=generator,
        device=device,
        dtype=torch.int64,
    )
    return tuple(
        torch.Generator(device=device).manual_seed(int(seed.item())) for seed in seeds
    )


class TrainingGCPolicy:
    def __init__(self, after_updates: int | None, *, rank: int = 0) -> None:
        self.after_updates = after_updates
        self.rank = rank
        self._attempted = False
        self._owns_freeze = False
        self._closed = False
        self._metrics: dict[str, int | float] = {}

    @classmethod
    def from_environment(cls, *, rank: int) -> TrainingGCPolicy:
        raw = os.environ.get(_ENV)
        if raw is None:
            return cls(None, rank=rank)
        try:
            count = int(raw)
        except ValueError as error:
            raise ValueError(f"{_ENV} must be a positive integer") from error
        if not 1 <= count <= 10000:
            raise ValueError(f"{_ENV} must be in [1, 10000]")
        return cls(count, rank=rank)

    def before_step(self, completed_updates: int) -> None:
        if (
            self.after_updates is None
            or self._attempted
            or self._closed
            or completed_updates < self.after_updates
        ):
            return
        self._attempted = True
        # Respect process-global GC ownership; never alter an existing policy.
        if not gc.isenabled() or gc.get_freeze_count():
            print(
                json.dumps(
                    {
                        "event": "training_gc_freeze_skipped",
                        "rank": self.rank,
                        "reason": "preexisting_gc_policy",
                    }
                ),
                flush=True,
            )
            return
        started = time.perf_counter()
        collected = gc.collect(2)
        gc.freeze()
        self._owns_freeze = True
        elapsed = time.perf_counter() - started
        frozen = gc.get_freeze_count()
        self._metrics = {
            "runtime/gc_freeze_completed_updates": completed_updates,
            "runtime/gc_freeze_initial_objects": frozen,
            "time/gc_freeze_once_seconds": elapsed,
        }
        print(
            json.dumps(
                {
                    "event": "training_gc_freeze",
                    "rank": self.rank,
                    "completed_updates": completed_updates,
                    "collected": collected,
                    "frozen_objects": frozen,
                    "seconds": elapsed,
                    "automatic_gc_enabled": gc.isenabled(),
                }
            ),
            flush=True,
        )

    def metrics(self) -> dict[str, int | float]:
        if not self._metrics:
            return {}
        result = dict(self._metrics)
        result["runtime/gc_frozen_objects"] = gc.get_freeze_count()
        return result

    def close(self) -> None:
        if self._owns_freeze:
            gc.unfreeze()
            self._owns_freeze = False
        self._closed = True


@contextmanager
def training_execution_policy(*, rank: int):
    previous = sys.getswitchinterval()
    raw = os.environ.get("MF_PYTHON_SWITCH_INTERVAL_SECONDS")
    if raw is not None:
        interval = float(raw)
        if not math.isfinite(interval) or not 0.0001 <= interval <= 0.01:
            raise ValueError(
                "MF_PYTHON_SWITCH_INTERVAL_SECONDS must be in [0.0001, 0.01]"
            )
    policy = TrainingGCPolicy.from_environment(rank=rank)
    try:
        if raw is not None:
            sys.setswitchinterval(interval)
        yield policy
    finally:
        policy.close()
        sys.setswitchinterval(previous)
