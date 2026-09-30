from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch.distributed as dist

from mf.config.fingerprint import full_config_hash
from mf.config.schema import MFConfig
from mf.contracts.trainer import TrainerState
from mf.distributed.context import DistributedContext, DistributedNotInitializedError
from mf.storage import durable_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _final_metrics(config: MFConfig, state: TrainerState) -> tuple[int, dict[str, Any]]:
    if not config.logging.enabled:
        return 0, {}
    path = Path(config.logging.output_dir) / "metrics.jsonl"
    if not path.is_file():
        raise RuntimeError(f"training metrics are missing: {path}")
    count = 0
    final: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"training metric line {line_number} is not an object"
                )
            count += 1
            final = payload
    if final is None:
        raise RuntimeError("training metrics are empty")
    if final.get("step") != state.global_step:
        raise RuntimeError(
            "final training metric step does not match trainer state: "
            f"{final.get('step')!r} != {state.global_step}"
        )
    return count, final


def finalize_training_run(
    config: MFConfig,
    state: TrainerState,
    distributed: DistributedContext,
) -> dict[str, str] | None:
    """Publish authoritative run-level completion artifacts after successful training."""

    if state.global_step != config.trainer.max_steps:
        raise RuntimeError(
            "cannot finalize an incomplete training run: "
            f"{state.global_step} != {config.trainer.max_steps}"
        )
    if distributed.world_size > 1:
        if not dist.is_available() or not dist.is_initialized():
            raise DistributedNotInitializedError(
                "training finalization requires the initialized process group"
            )
        dist.barrier(group=distributed.process_group)
    if distributed.rank != 0:
        return None

    output = Path(config.run.output_dir)
    marker_path = output / "COMPLETED"
    if marker_path.exists():
        raise FileExistsError(
            f"training completion marker already exists: {marker_path}"
        )

    metric_count, final_metrics = _final_metrics(config, state)
    finalized_at = datetime.now(timezone.utc).isoformat()  # noqa: UP017
    metrics_path = durable_write_json(
        output / "metrics.json",
        {
            "schema_version": 1,
            "run_name": config.run.name,
            "global_step": state.global_step,
            "samples_seen": state.samples_seen,
            "record_count": metric_count,
            "final": final_metrics,
        },
        ensure_ascii=True,
    )
    status_path = durable_write_json(
        output / "status.json",
        {
            "schema_version": 1,
            "status": "completed",
            "run_name": config.run.name,
            "config_hash": full_config_hash(config),
            "global_step": state.global_step,
            "expected_steps": config.trainer.max_steps,
            "samples_seen": state.samples_seen,
            "last_checkpoint_step": state.last_checkpoint_step,
            "last_evaluation_step": state.last_evaluation_step,
            "finalized_at": finalized_at,
        },
        ensure_ascii=True,
    )
    marker = {
        "schema_version": 1,
        "status": "completed",
        "run_name": config.run.name,
        "config_hash": full_config_hash(config),
        "global_step": state.global_step,
        "samples_seen": state.samples_seen,
        "finalized_at": finalized_at,
        "artifacts": {
            "metrics.json": _sha256(metrics_path),
            "status.json": _sha256(status_path),
        },
    }
    durable_write_json(marker_path, marker, ensure_ascii=True)
    return {
        "completed": str(marker_path),
        "metrics": str(metrics_path),
        "status": str(status_path),
    }


__all__ = ["finalize_training_run"]
