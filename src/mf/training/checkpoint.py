from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar, cast, runtime_checkable

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.optim import Optimizer

from mf.config.fingerprint import (
    full_config_hash,
    full_config_mapping_hash,
    legacy_training_fingerprint_mapping,
    training_fingerprint,
    training_fingerprint_mapping,
)
from mf.config.schema import MFConfig
from mf.contracts.evaluation import EvaluationIdentity
from mf.contracts.trainer import TrainerState
from mf.storage import durable_write_bytes, durable_write_stream, uses_direct_writes
from mf.training.ema import ExponentialMovingAverage
from mf.training.optimizers import OptimizerBundle
from mf.training.schedulers import SchedulerBundle

_CHECKPOINT_VERSION = 1
_CHECKPOINT_FORMAT = "mf-checkpoint-v1"
_CHECKPOINT_HASH_WORKERS = 4
_STEP_PATTERN = re.compile(r"step_(\d{6,})\Z")
_OPTIMIZER_NAMES = (
    "backbone_muon",
    "backbone_adamw",
    "text_decoder_muon",
    "text_decoder_adamw",
)
_SHARED_PAYLOAD_FILES = frozenset(
    {
        "model.pt",
        "text_decoder.pt",
        "ema.pt",
        "optimizers.pt",
        "schedulers.pt",
        "trainer_state.json",
        "resolved_config.json",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "format",
        "version",
        "step",
        "world_size",
        "checkpoint_name",
        "checkpoint_path",
        "training_fingerprint",
        "full_config_hash",
        "evaluation_identity",
        "rank_files",
        "files",
        "git",
        "dependencies",
    }
)
_T = TypeVar("_T")


class CheckpointError(RuntimeError):
    """Base error for checkpoint publication or restoration failures."""


class CheckpointValidationError(CheckpointError):
    """Raised when a checkpoint does not satisfy the exact restore contract."""


@runtime_checkable
class StatefulDataStream(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


@dataclass(frozen=True, slots=True)
class RunMetadata:
    git_commit: str
    git_dirty: bool
    dependency_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.git_commit, str) or not self.git_commit:
            raise ValueError("git_commit must be a non-empty string")
        if not isinstance(self.git_dirty, bool):
            raise TypeError("git_dirty must be a bool")
        if not isinstance(self.dependency_versions, Mapping) or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in self.dependency_versions.items()
        ):
            raise ValueError(
                "dependency_versions must map non-empty strings to strings"
            )


@dataclass(frozen=True, slots=True)
class CheckpointBindings:
    model: nn.Module
    text_decoder: nn.Module
    optimizers: OptimizerBundle
    schedulers: SchedulerBundle
    ema: ExponentialMovingAverage
    training_generator: torch.Generator
    evaluation_generator: torch.Generator
    data_stream: StatefulDataStream

    def __post_init__(self) -> None:
        expected_types = {
            "model": nn.Module,
            "text_decoder": nn.Module,
            "optimizers": OptimizerBundle,
            "schedulers": SchedulerBundle,
            "ema": ExponentialMovingAverage,
            "training_generator": torch.Generator,
            "evaluation_generator": torch.Generator,
        }
        for name, expected_type in expected_types.items():
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} must be a {expected_type.__name__}")
        if not isinstance(self.data_stream, StatefulDataStream):
            raise TypeError("data_stream must implement state_dict/load_state_dict")


@dataclass(frozen=True, slots=True)
class RestoredState:
    path: Path
    step: int
    trainer_state: TrainerState
    saved_config: MFConfig
    training_fingerprint: str
    saved_full_config_hash: str
    current_full_config_hash: str
    run_metadata: RunMetadata


@dataclass(slots=True)
class _RuntimeState:
    model_state: Mapping[str, Tensor]
    decoder_state: Mapping[str, Tensor]
    optimizer_states: Mapping[str, Mapping[str, object] | None]
    scheduler_state: Mapping[str, object]
    ema_state: Mapping[str, object]
    cpu_rng_state: Tensor
    cuda_rng_states: tuple[Tensor, ...]
    training_generator_state: Tensor
    evaluation_generator_state: Tensor
    data_stream_state: Mapping[str, object]


@dataclass(slots=True)
class _PreparedCheckpoint:
    path: Path
    step: int
    trainer_state: TrainerState
    saved_config: MFConfig
    fingerprint: str
    saved_full_hash: str
    runtime: _RuntimeState
    run_metadata: RunMetadata


class CheckpointManager:
    """Publish complete checkpoints atomically and restore bound runtime state exactly."""

    def __init__(
        self,
        root: str | Path,
        *,
        bindings: CheckpointBindings,
        config: MFConfig,
        evaluation_identity: EvaluationIdentity,
        eval_root: str | Path | None = None,
        rank: int = 0,
        world_size: int = 1,
        process_group: dist.ProcessGroup | None = None,
        run_metadata: RunMetadata,
    ) -> None:
        _validate_rank_world(rank, world_size)
        if not isinstance(config, MFConfig):
            raise TypeError("config must be a MFConfig")
        if not isinstance(evaluation_identity, EvaluationIdentity):
            raise TypeError("evaluation_identity must be an EvaluationIdentity")
        if evaluation_identity.config_hash != full_config_hash(config):
            raise ValueError(
                "evaluation_identity.config_hash must match the resolved config"
            )
        if config.distributed.world_size != world_size:
            raise ValueError(
                "config distributed world_size does not match checkpoint manager"
            )
        self.root = Path(root).expanduser().resolve()
        self.eval_root = (
            Path(eval_root).expanduser().resolve()
            if eval_root is not None
            else self.root.parent / "eval"
        )
        self.bindings = bindings
        self.config = config
        self.evaluation_identity = evaluation_identity
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group
        self.run_metadata = run_metadata

    def save(self, step: int, state: TrainerState) -> Path:
        """Atomically publish one complete checkpoint for all ranks."""

        def preflight() -> tuple[str, str, Mapping[str, object]]:
            self._validate_distributed_runtime()
            _validate_step(step)
            if not isinstance(state, TrainerState):
                raise TypeError("state must be a TrainerState")
            if state.global_step != step:
                raise ValueError("TrainerState.global_step must equal checkpoint step")
            _validate_runtime_progress(
                step,
                state,
                self.bindings.schedulers.state_dict(),
                self.bindings.ema.state_dict(),
                optimizer_count=len(self.bindings.optimizers.optimizers),
            )
            return (
                training_fingerprint(self.config),
                full_config_hash(self.config),
                _trainer_payload(step, state),
            )

        fingerprint, config_hash, trainer_payload = self._run_collective_stage(
            "checkpoint preflight",
            preflight,
        )

        checkpoint = self.root / _step_name(step)
        incomplete = self.root / f"{_step_name(step)}.incomplete"
        direct_write = uses_direct_writes(checkpoint)
        write_root = checkpoint if direct_write else incomplete
        consensus = {
            "step": step,
            "trainer_state": trainer_payload,
            "training_fingerprint": fingerprint,
            "full_config_hash": config_hash,
            "evaluation_identity": self.evaluation_identity.as_dict(),
        }
        self._require_consensus(consensus)

        def capture() -> tuple[Mapping[str, object], Mapping[str, object] | None]:
            rank_payload = self._rank_payload(step, fingerprint, config_hash)
            shared_payloads = self._shared_payloads() if self.rank == 0 else None
            return rank_payload, shared_payloads

        rank_payload, shared_payloads = self._run_collective_stage(
            "checkpoint state capture",
            capture,
        )

        def create_incomplete() -> None:
            self.root.mkdir(parents=True, exist_ok=True)
            if direct_write:
                if checkpoint.exists():
                    if checkpoint.is_symlink() or not checkpoint.is_dir():
                        raise FileExistsError(
                            f"checkpoint path is not a directory: {checkpoint}"
                        )
                    if (checkpoint / "COMPLETED").exists():
                        raise FileExistsError(
                            f"checkpoint path already exists: {checkpoint}"
                        )
                else:
                    checkpoint.mkdir()
                (checkpoint / "rank_states").mkdir(exist_ok=True)
            else:
                if checkpoint.exists():
                    raise FileExistsError(
                        f"checkpoint path already exists: {checkpoint}"
                    )
                if incomplete.exists():
                    raise FileExistsError(
                        f"incomplete checkpoint already exists: {incomplete}"
                    )
                incomplete.mkdir()
                (incomplete / "rank_states").mkdir()
                _fsync_directory(self.root)

        self._run_rank_zero_stage("checkpoint directory creation", create_incomplete)

        def write_payloads() -> None:
            _atomic_torch_save(
                write_root / "rank_states" / _rank_file_name(self.rank),
                rank_payload,
                rank=self.rank,
            )
            if self.rank == 0:
                assert shared_payloads is not None
                for filename, payload in shared_payloads.items():
                    _atomic_torch_save(
                        write_root / filename,
                        payload,
                        rank=self.rank,
                    )
                _atomic_json(
                    write_root / "trainer_state.json",
                    trainer_payload,
                    rank=self.rank,
                )
                _atomic_json(
                    write_root / "resolved_config.json",
                    self.config.model_dump(mode="json"),
                    rank=self.rank,
                )

        self._run_collective_stage("checkpoint payload write", write_payloads)

        def finalize() -> None:
            if not direct_write:
                _validate_directory_layout(write_root, self.world_size, stage="payload")
            relative_payloads = sorted(_expected_payload_paths(self.world_size))
            file_records = _build_file_records(write_root, relative_payloads)
            manifest = {
                "format": _CHECKPOINT_FORMAT,
                "version": _CHECKPOINT_VERSION,
                "step": step,
                "world_size": self.world_size,
                "checkpoint_name": checkpoint.name,
                "checkpoint_path": str(checkpoint),
                "training_fingerprint": fingerprint,
                "full_config_hash": config_hash,
                "evaluation_identity": self.evaluation_identity.as_dict(),
                "rank_files": [
                    f"rank_states/{_rank_file_name(rank)}"
                    for rank in range(self.world_size)
                ],
                "files": file_records,
                "git": {
                    "commit": self.run_metadata.git_commit,
                    "dirty": self.run_metadata.git_dirty,
                },
                "dependencies": dict(
                    sorted(self.run_metadata.dependency_versions.items())
                ),
            }
            _atomic_json(write_root / "run_manifest.json", manifest, rank=self.rank)
            _validate_directory_layout(write_root, self.world_size, stage="manifest")
            if not direct_write:
                os.rename(incomplete, checkpoint)
                _fsync_directory(self.root)
            marker = {
                "format": _CHECKPOINT_FORMAT,
                "version": _CHECKPOINT_VERSION,
                "step": step,
                "manifest_sha256": _sha256_file(checkpoint / "run_manifest.json"),
            }
            _atomic_json(checkpoint / "COMPLETED", marker, rank=self.rank)
            _validate_directory_layout(checkpoint, self.world_size, stage="completed")
            if not direct_write:
                _fsync_directory(self.root)

        self._run_rank_zero_stage("checkpoint finalization", finalize)
        return checkpoint

    def load(self, path: str | Path, expected_fingerprint: str) -> RestoredState:
        """Validate a complete checkpoint, then transactionally restore bound objects."""

        def preflight() -> tuple[Path, Mapping[str, object]]:
            self._validate_distributed_runtime()
            return self._validated_header(path, expected_fingerprint)

        checkpoint, manifest = self._run_collective_stage(
            "checkpoint preflight",
            preflight,
            error_type=CheckpointValidationError,
        )
        prepared = self._run_collective_stage(
            "checkpoint validation",
            lambda: self._prepare_checkpoint(
                checkpoint, manifest, expected_fingerprint
            ),
            error_type=CheckpointValidationError,
        )

        snapshot = self._run_collective_stage(
            "live state snapshot",
            self._runtime_snapshot,
        )

        apply_error: str | None = None
        rollback_error: str | None = None
        try:
            self._apply_runtime_state(prepared.runtime)
        except BaseException as error:
            apply_error = _exception_text(error)
            try:
                self._apply_runtime_state(snapshot)
            except BaseException as rollback:
                rollback_error = _exception_text(rollback)

        errors = self._collect_rank_errors(apply_error)
        if any(error is not None for error in errors):
            if apply_error is None:
                try:
                    self._apply_runtime_state(snapshot)
                except BaseException as rollback:
                    rollback_error = _exception_text(rollback)
            rollback_errors = self._collect_rank_errors(rollback_error)
            if any(error is not None for error in rollback_errors):
                raise CheckpointError(
                    "checkpoint restore and rollback failed: "
                    f"restore={_join_errors(errors)}; rollback={_join_errors(rollback_errors)}"
                )
            raise CheckpointError(f"checkpoint restore failed: {_join_errors(errors)}")

        self._barrier()
        return RestoredState(
            path=prepared.path,
            step=prepared.step,
            trainer_state=prepared.trainer_state,
            saved_config=prepared.saved_config,
            training_fingerprint=prepared.fingerprint,
            saved_full_config_hash=prepared.saved_full_hash,
            current_full_config_hash=full_config_hash(self.config),
            run_metadata=prepared.run_metadata,
        )

    def require_completed(self, step: int, checkpoint: str | Path) -> None:
        """Collectively validate the authoritative completed-checkpoint precondition."""

        self._run_collective_stage(
            "completed checkpoint validation",
            lambda: self.require_completed_local(step, checkpoint),
            error_type=CheckpointValidationError,
        )

    def require_completed_local(self, step: int, checkpoint: str | Path) -> None:
        """Validate one completed checkpoint without entering a collective."""

        self._validate_distributed_runtime()
        _validate_step(step)
        path, manifest = self._completed_header(checkpoint)
        if manifest["step"] != step or path.name != _step_name(step):
            raise CheckpointValidationError("completed checkpoint step mismatch")

    def pending_eval_step(self) -> int | None:
        """Return the oldest completed checkpoint without matching completed eval."""

        if not self.root.is_dir():
            return None
        completed: list[tuple[int, Path]] = []
        for candidate in self.root.iterdir():
            match = _STEP_PATTERN.fullmatch(candidate.name)
            if match is None or not candidate.is_dir():
                continue
            step = int(match.group(1))
            try:
                checkpoint, _ = self._completed_header(candidate)
            except CheckpointValidationError:
                continue
            completed.append((step, checkpoint))
        for step, checkpoint in sorted(completed):
            scheduled = step % self.config.trainer.eval_steps == 0 or (
                self.config.trainer.eval_at_final_step
                and step == self.config.trainer.max_steps
            )
            if not scheduled:
                continue
            if not _evaluation_is_completed(
                self.eval_root,
                step,
                checkpoint,
                self.evaluation_identity,
            ):
                return step
        return None

    def _shared_payloads(self) -> Mapping[str, object]:
        optimizer_payloads: dict[str, object] = {}
        for name, optimizer in _optimizer_slots(self.bindings.optimizers).items():
            optimizer_payloads[name] = (
                None
                if optimizer is None
                else {
                    "signature": _optimizer_signature(optimizer),
                    "state": _clone_cpu_tree(optimizer.state_dict()),
                }
            )
        payloads = {
            "model.pt": _module_payload(self.bindings.model),
            "text_decoder.pt": _module_payload(self.bindings.text_decoder),
            "ema.pt": {
                "version": _CHECKPOINT_VERSION,
                "state": _clone_cpu_tree(self.bindings.ema.state_dict()),
            },
            "optimizers.pt": {
                "version": _CHECKPOINT_VERSION,
                "optimizers": optimizer_payloads,
            },
            "schedulers.pt": {
                "version": _CHECKPOINT_VERSION,
                "state": _clone_cpu_tree(self.bindings.schedulers.state_dict()),
            },
        }
        _require_safe_tree(payloads, "shared checkpoint state")
        return payloads

    def _rank_payload(
        self,
        step: int,
        fingerprint: str,
        config_hash: str,
    ) -> Mapping[str, object]:
        payload = {
            "version": _CHECKPOINT_VERSION,
            "step": step,
            "rank": self.rank,
            "world_size": self.world_size,
            "training_fingerprint": fingerprint,
            "full_config_hash": config_hash,
            "cpu_rng_state": torch.get_rng_state().cpu().clone(),
            "cuda_rng_states": tuple(
                state.cpu().clone() for state in torch.cuda.get_rng_state_all()
            ),
            "generators": {
                "training": _generator_payload(self.bindings.training_generator),
                "evaluation": _generator_payload(self.bindings.evaluation_generator),
            },
            "data_stream": _clone_cpu_tree(self.bindings.data_stream.state_dict()),
        }
        _require_safe_tree(payload, "rank checkpoint state")
        return payload

    def _validated_header(
        self,
        path: str | Path,
        expected_fingerprint: str,
    ) -> tuple[Path, Mapping[str, object]]:
        if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
            raise TypeError("expected_fingerprint must be a non-empty string")
        current_fingerprint = training_fingerprint(self.config)
        if current_fingerprint != expected_fingerprint:
            raise CheckpointValidationError(
                "expected training fingerprint does not match the current resolved config"
            )

        checkpoint, manifest = self._completed_header(path)
        return checkpoint, manifest

    def _completed_header(
        self,
        path: str | Path,
    ) -> tuple[Path, Mapping[str, object]]:
        raw_path = Path(path).expanduser()
        try:
            checkpoint = raw_path.resolve(strict=True)
        except OSError as error:
            raise CheckpointValidationError(
                f"checkpoint path does not exist: {raw_path}"
            ) from error
        if not checkpoint.is_dir():
            raise CheckpointValidationError("checkpoint path must be a directory")
        step = _step_from_name(checkpoint.name)

        marker_path = checkpoint / "COMPLETED"
        if not marker_path.is_file() or marker_path.is_symlink():
            raise CheckpointValidationError(
                "checkpoint is missing the authoritative COMPLETED marker"
            )
        marker = _read_json(marker_path)
        if set(marker) != {"format", "version", "step", "manifest_sha256"}:
            raise CheckpointValidationError("COMPLETED marker fields are invalid")
        if (
            marker["format"] != _CHECKPOINT_FORMAT
            or type(marker["version"]) is not int
            or marker["version"] != _CHECKPOINT_VERSION
            or type(marker["step"]) is not int
            or marker["step"] != step
        ):
            raise CheckpointValidationError("COMPLETED marker identity is invalid")
        manifest_path = checkpoint / "run_manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise CheckpointValidationError("checkpoint manifest is missing")
        if marker["manifest_sha256"] != _sha256_file(manifest_path):
            raise CheckpointValidationError("checkpoint manifest hash is corrupt")
        manifest = _read_json(manifest_path)
        _validate_manifest_identity(manifest, checkpoint, step, self.world_size)
        _validate_directory_layout(checkpoint, self.world_size, stage="completed")
        _validate_file_records(checkpoint, manifest)
        return checkpoint, manifest

    def _prepare_checkpoint(
        self,
        checkpoint: Path,
        manifest: Mapping[str, object],
        expected_fingerprint: str,
    ) -> _PreparedCheckpoint:
        resolved_mapping = _read_json(checkpoint / "resolved_config.json")
        try:
            raw_saved_fingerprint = training_fingerprint_mapping(resolved_mapping)
            legacy_saved_fingerprint = legacy_training_fingerprint_mapping(
                resolved_mapping
            )
            saved_config = MFConfig.model_validate(resolved_mapping)
        except Exception as error:
            raise CheckpointValidationError(
                "saved resolved config is invalid"
            ) from error
        saved_full_hash = full_config_mapping_hash(resolved_mapping)
        saved_fingerprint = training_fingerprint(saved_config)
        if saved_full_hash != manifest["full_config_hash"]:
            raise CheckpointValidationError("saved full config hash is corrupt")
        stored_fingerprint = manifest["training_fingerprint"]
        if stored_fingerprint not in (
            raw_saved_fingerprint,
            legacy_saved_fingerprint,
        ):
            raise CheckpointValidationError("saved training fingerprint is corrupt")
        legacy_current_fingerprint = legacy_training_fingerprint_mapping(
            self.config.model_dump(mode="json")
        )
        legacy_resume_compatible = (
            stored_fingerprint == legacy_saved_fingerprint
            and legacy_current_fingerprint == legacy_saved_fingerprint
            and saved_config.objective.model_dump(mode="json")
            == self.config.objective.model_dump(mode="json")
        )
        if (
            saved_fingerprint != expected_fingerprint
            and not legacy_resume_compatible
            and not _constant_lr_horizon_extension_is_compatible(
                saved_config, self.config
            )
        ):
            raise CheckpointValidationError("checkpoint training fingerprint mismatch")

        step = manifest["step"]
        assert type(step) is int
        trainer_state = _load_trainer_state(checkpoint / "trainer_state.json", step)
        model_payload = _load_torch_payload(checkpoint / "model.pt")
        decoder_payload = _load_torch_payload(checkpoint / "text_decoder.pt")
        model_state = _validate_module_payload(
            model_payload, self.bindings.model, "model"
        )
        decoder_state = _validate_module_payload(
            decoder_payload,
            self.bindings.text_decoder,
            "text decoder",
        )

        optimizer_payload = _load_torch_payload(checkpoint / "optimizers.pt")
        optimizer_states = _validate_optimizer_payload(
            optimizer_payload,
            self.bindings.optimizers,
        )
        scheduler_payload = _load_torch_payload(checkpoint / "schedulers.pt")
        scheduler_state = _versioned_state(scheduler_payload, "scheduler")
        ema_payload = _load_torch_payload(checkpoint / "ema.pt")
        ema_state = _versioned_state(ema_payload, "EMA")
        _validate_runtime_progress(
            step,
            trainer_state,
            scheduler_state,
            ema_state,
            optimizer_count=len(self.bindings.optimizers.optimizers),
        )

        rank_path = checkpoint / "rank_states" / _rank_file_name(self.rank)
        rank_payload = _load_torch_payload(rank_path)
        rank_state = _validate_rank_payload(
            rank_payload,
            step=step,
            rank=self.rank,
            world_size=self.world_size,
            fingerprint=stored_fingerprint,
            full_hash=saved_full_hash,
            training_generator=self.bindings.training_generator,
            evaluation_generator=self.bindings.evaluation_generator,
        )
        git = manifest["git"]
        dependencies = manifest["dependencies"]
        assert isinstance(git, Mapping) and isinstance(dependencies, Mapping)
        metadata = RunMetadata(
            git_commit=git["commit"],
            git_dirty=git["dirty"],
            dependency_versions=dependencies,
        )
        return _PreparedCheckpoint(
            path=checkpoint,
            step=step,
            trainer_state=trainer_state,
            saved_config=saved_config,
            fingerprint=stored_fingerprint,
            saved_full_hash=saved_full_hash,
            runtime=_RuntimeState(
                model_state=model_state,
                decoder_state=decoder_state,
                optimizer_states=optimizer_states,
                scheduler_state=scheduler_state,
                ema_state=ema_state,
                cpu_rng_state=rank_state["cpu_rng_state"],
                cuda_rng_states=rank_state["cuda_rng_states"],
                training_generator_state=rank_state["training_generator_state"],
                evaluation_generator_state=rank_state["evaluation_generator_state"],
                data_stream_state=rank_state["data_stream_state"],
            ),
            run_metadata=metadata,
        )

    def _runtime_snapshot(self) -> _RuntimeState:
        optimizer_states = {
            name: None if optimizer is None else _clone_cpu_tree(optimizer.state_dict())
            for name, optimizer in _optimizer_slots(self.bindings.optimizers).items()
        }
        return _RuntimeState(
            model_state=_plain_module_state(self.bindings.model),
            decoder_state=_plain_module_state(self.bindings.text_decoder),
            optimizer_states=optimizer_states,
            scheduler_state=_clone_cpu_tree(self.bindings.schedulers.state_dict()),
            ema_state=_clone_cpu_tree(self.bindings.ema.state_dict()),
            cpu_rng_state=torch.get_rng_state().cpu().clone(),
            cuda_rng_states=tuple(
                state.cpu().clone() for state in torch.cuda.get_rng_state_all()
            ),
            training_generator_state=self.bindings.training_generator.get_state()
            .cpu()
            .clone(),
            evaluation_generator_state=(
                self.bindings.evaluation_generator.get_state().cpu().clone()
            ),
            data_stream_state=copy.deepcopy(self.bindings.data_stream.state_dict()),
        )

    def _apply_runtime_state(self, state: _RuntimeState) -> None:
        self.bindings.model.load_state_dict(state.model_state, strict=True)
        self.bindings.text_decoder.load_state_dict(state.decoder_state, strict=True)
        for name, optimizer in _optimizer_slots(self.bindings.optimizers).items():
            optimizer_state = state.optimizer_states[name]
            if optimizer is not None:
                assert optimizer_state is not None
                optimizer.load_state_dict(optimizer_state)
        self.bindings.schedulers.load_state_dict(state.scheduler_state)
        self.bindings.ema.load_state_dict(state.ema_state)
        self.bindings.data_stream.load_state_dict(state.data_stream_state)
        self.bindings.training_generator.set_state(state.training_generator_state)
        self.bindings.evaluation_generator.set_state(state.evaluation_generator_state)
        torch.set_rng_state(state.cpu_rng_state)
        torch.cuda.set_rng_state_all(list(state.cuda_rng_states))

    def _validate_distributed_runtime(self) -> None:
        _validate_rank_world(self.rank, self.world_size)
        if not isinstance(self.bindings, CheckpointBindings):
            raise CheckpointError("checkpoint runtime bindings are invalid")
        if not isinstance(self.config, MFConfig):
            raise CheckpointError("checkpoint runtime config is invalid")
        if not isinstance(self.evaluation_identity, EvaluationIdentity):
            raise CheckpointError("checkpoint evaluation identity is invalid")
        if self.config.distributed.world_size != self.world_size:
            raise CheckpointError(
                "checkpoint config world_size does not match the manager"
            )
        if self.evaluation_identity.world_size != self.world_size:
            raise CheckpointError(
                "evaluation identity world_size does not match the manager"
            )
        if self.evaluation_identity.config_hash != full_config_hash(self.config):
            raise CheckpointError(
                "evaluation identity config_hash does not match the resolved config"
            )

        if dist.is_available() and dist.is_initialized():
            runtime_rank = dist.get_rank(group=self.process_group)
            runtime_world_size = dist.get_world_size(group=self.process_group)
            if (runtime_rank, runtime_world_size) != (self.rank, self.world_size):
                raise CheckpointError(
                    "checkpoint rank/world_size do not match the process group"
                )
        elif self.world_size != 1:
            raise CheckpointError(
                "distributed checkpointing requires an initialized process group"
            )

    def _require_consensus(self, value: Mapping[str, object]) -> None:
        if self.world_size == 1:
            return
        gathered: list[object] = [None] * self.world_size
        dist.all_gather_object(gathered, dict(value), group=self.process_group)
        if any(candidate != gathered[0] for candidate in gathered[1:]):
            raise CheckpointError(
                "checkpoint step/config/trainer state differ across ranks"
            )

    def _collect_rank_errors(self, error: str | None) -> tuple[str | None, ...]:
        if not dist.is_available() or not dist.is_initialized():
            return (error,)
        runtime_world_size = dist.get_world_size(group=self.process_group)
        if runtime_world_size == 1:
            return (error,)
        gathered: list[str | None] = [None] * runtime_world_size
        dist.all_gather_object(gathered, error, group=self.process_group)
        return tuple(gathered)

    def _run_collective_stage(
        self,
        name: str,
        action: Callable[[], _T],
        *,
        error_type: type[CheckpointError] = CheckpointError,
    ) -> _T:
        result: _T | None = None
        local_error: str | None = None
        try:
            result = action()
        except BaseException as error:
            local_error = _exception_text(error)
        errors = self._collect_rank_errors(local_error)
        if any(error is not None for error in errors):
            raise error_type(f"{name} failed: {_join_errors(errors)}")
        return cast(_T, result)

    def _run_rank_zero_stage(self, name: str, action: Callable[[], None]) -> None:
        error: str | None = None
        if self.rank == 0:
            try:
                action()
            except BaseException as caught:
                error = _exception_text(caught)
        if self.world_size > 1:
            payload: list[str | None] = [error]
            dist.broadcast_object_list(payload, src=0, group=self.process_group)
            error = payload[0]
        if error is not None:
            raise CheckpointError(f"{name} failed: {error}")

    def _barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier(group=self.process_group)


def _validate_rank_world(rank: object, world_size: object) -> None:
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
    ):
        raise ValueError("world_size must be a positive integer")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 0 <= rank < world_size
    ):
        raise ValueError("rank must be an integer in [0, world_size)")


def _validate_step(step: object) -> None:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")


def _validate_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _step_name(step: int) -> str:
    return f"step_{step:06d}"


def _step_from_name(name: str) -> int:
    match = _STEP_PATTERN.fullmatch(name)
    if match is None:
        raise CheckpointValidationError("checkpoint directory name is invalid")
    return int(match.group(1))


def _rank_file_name(rank: int) -> str:
    return f"rank_{rank:05d}.pt"


def _optimizer_slots(bundle: OptimizerBundle) -> dict[str, Optimizer | None]:
    return {name: getattr(bundle, name) for name in _OPTIMIZER_NAMES}


def _class_key(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _plain_module_state(module: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, value in module.state_dict().items():
        if not isinstance(value, Tensor):
            raise TypeError(f"module state {name!r} is not a tensor")
        state[name] = value.detach().cpu().clone()
    return state


def _module_signature(module: nn.Module) -> Mapping[str, object]:
    state = module.state_dict()
    entries: list[tuple[str, tuple[int, ...], str]] = []
    for name, value in state.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"module state {name!r} is not a tensor")
        entries.append((name, tuple(value.shape), str(value.dtype)))
    return {"class": _class_key(module), "entries": tuple(entries)}


def _module_payload(module: nn.Module) -> Mapping[str, object]:
    state = _plain_module_state(module)
    return {
        "version": _CHECKPOINT_VERSION,
        "signature": _module_signature(module),
        "state": state,
    }


def _validate_module_payload(
    payload: Mapping[str, object],
    module: nn.Module,
    label: str,
) -> Mapping[str, Tensor]:
    if set(payload) != {"version", "signature", "state"}:
        raise CheckpointValidationError(f"{label} payload fields are invalid")
    if payload["version"] != _CHECKPOINT_VERSION:
        raise CheckpointValidationError(f"unsupported {label} payload version")
    signature = payload["signature"]
    state = payload["state"]
    if not isinstance(signature, Mapping) or not isinstance(state, Mapping):
        raise CheckpointValidationError(f"{label} payload is malformed")
    if any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in state.items()
    ):
        raise CheckpointValidationError(f"{label} state must map names to tensors")
    typed_state = dict(state)
    saved_signature = {
        "class": signature.get("class"),
        "entries": tuple(
            (name, tuple(value.shape), str(value.dtype))
            for name, value in typed_state.items()
        ),
    }
    if signature != saved_signature:
        raise CheckpointValidationError(f"{label} state signature is corrupt")
    if signature != _module_signature(module):
        raise CheckpointValidationError(f"{label} object identity/shape mismatch")
    return typed_state


def _optimizer_signature(optimizer: Optimizer) -> Mapping[str, object]:
    groups: list[Mapping[str, object]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = tuple(group["params"])
        names_value = group.get("parameter_names")
        if names_value is None:
            parameter_names = tuple(
                f"group_{group_index}.parameter_{index}"
                for index in range(len(parameters))
            )
        else:
            parameter_names = tuple(names_value)
        if len(parameter_names) != len(parameters) or any(
            not isinstance(name, str) or not name for name in parameter_names
        ):
            raise ValueError("optimizer parameter identities are invalid")
        group_name = group.get("group_name", f"group_{group_index}")
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("optimizer group identity is invalid")
        groups.append(
            {
                "group_name": group_name,
                "parameter_names": parameter_names,
                "parameter_shapes": tuple(
                    tuple(parameter.shape) for parameter in parameters
                ),
                "parameter_dtypes": tuple(
                    str(parameter.dtype) for parameter in parameters
                ),
            }
        )
    return {"class": _class_key(optimizer), "groups": tuple(groups)}


def _validate_optimizer_payload(
    payload: Mapping[str, object],
    bundle: OptimizerBundle,
) -> Mapping[str, Mapping[str, object] | None]:
    if set(payload) != {"version", "optimizers"}:
        raise CheckpointValidationError("optimizer payload fields are invalid")
    if payload["version"] != _CHECKPOINT_VERSION:
        raise CheckpointValidationError("unsupported optimizer payload version")
    saved = payload["optimizers"]
    if not isinstance(saved, Mapping) or set(saved) != set(_OPTIMIZER_NAMES):
        raise CheckpointValidationError("optimizer slots are invalid")
    result: dict[str, Mapping[str, object] | None] = {}
    for name, optimizer in _optimizer_slots(bundle).items():
        entry = saved[name]
        if optimizer is None:
            if entry is not None:
                raise CheckpointValidationError(
                    f"optimizer presence mismatch for {name}"
                )
            result[name] = None
            continue
        if not isinstance(entry, Mapping) or set(entry) != {"signature", "state"}:
            raise CheckpointValidationError(f"optimizer state for {name} is malformed")
        if entry["signature"] != _optimizer_signature(optimizer):
            raise CheckpointValidationError(
                f"optimizer identity/shape mismatch for {name}"
            )
        state = entry["state"]
        if not isinstance(state, Mapping):
            raise CheckpointValidationError(f"optimizer state for {name} is malformed")
        result[name] = state
    return result


def _versioned_state(payload: Mapping[str, object], label: str) -> Mapping[str, object]:
    if set(payload) != {"version", "state"}:
        raise CheckpointValidationError(f"{label} payload fields are invalid")
    if payload["version"] != _CHECKPOINT_VERSION:
        raise CheckpointValidationError(f"unsupported {label} payload version")
    state = payload["state"]
    if not isinstance(state, Mapping):
        raise CheckpointValidationError(f"{label} state is malformed")
    return state


def _generator_payload(generator: torch.Generator) -> Mapping[str, object]:
    return {
        "device": str(generator.device),
        "state": generator.get_state().cpu().clone(),
    }


def _valid_rng_state(value: object) -> bool:
    return (
        isinstance(value, Tensor)
        and value.device.type == "cpu"
        and value.dtype is torch.uint8
        and value.ndim == 1
    )


def _validate_generator_payload(
    value: object,
    generator: torch.Generator,
    label: str,
) -> Tensor:
    if not isinstance(value, Mapping) or set(value) != {"device", "state"}:
        raise CheckpointValidationError(f"{label} generator state is malformed")
    if value["device"] != str(generator.device):
        raise CheckpointValidationError(f"{label} generator device mismatch")
    state = value["state"]
    if not _valid_rng_state(state):
        raise CheckpointValidationError(f"{label} generator RNG state is invalid")
    return state


def _validate_rank_payload(
    payload: Mapping[str, object],
    *,
    step: int,
    rank: int,
    world_size: int,
    fingerprint: str,
    full_hash: str,
    training_generator: torch.Generator,
    evaluation_generator: torch.Generator,
) -> Mapping[str, object]:
    expected_keys = {
        "version",
        "step",
        "rank",
        "world_size",
        "training_fingerprint",
        "full_config_hash",
        "cpu_rng_state",
        "cuda_rng_states",
        "generators",
        "data_stream",
    }
    if set(payload) != expected_keys:
        raise CheckpointValidationError("rank state fields are invalid")
    if payload["version"] != _CHECKPOINT_VERSION:
        raise CheckpointValidationError("unsupported rank state version")
    if payload["step"] != step:
        raise CheckpointValidationError("rank state step mismatch")
    if payload["rank"] != rank or payload["world_size"] != world_size:
        raise CheckpointValidationError("rank state topology mismatch")
    if payload["training_fingerprint"] != fingerprint:
        raise CheckpointValidationError("rank state training fingerprint mismatch")
    if payload["full_config_hash"] != full_hash:
        raise CheckpointValidationError("rank state full config hash mismatch")
    cpu_rng_state = payload["cpu_rng_state"]
    if not _valid_rng_state(cpu_rng_state):
        raise CheckpointValidationError("CPU RNG state is invalid")
    cuda_rng_states = payload["cuda_rng_states"]
    if not isinstance(cuda_rng_states, tuple) or any(
        not _valid_rng_state(state) for state in cuda_rng_states
    ):
        raise CheckpointValidationError("CUDA RNG states are invalid")
    if len(cuda_rng_states) != torch.cuda.device_count():
        raise CheckpointValidationError(
            "visible CUDA device count does not match checkpoint"
        )
    generators = payload["generators"]
    if not isinstance(generators, Mapping) or set(generators) != {
        "training",
        "evaluation",
    }:
        raise CheckpointValidationError("explicit generator states are invalid")
    training_state = _validate_generator_payload(
        generators["training"], training_generator, "training"
    )
    evaluation_state = _validate_generator_payload(
        generators["evaluation"], evaluation_generator, "evaluation"
    )
    data_stream_state = payload["data_stream"]
    if not isinstance(data_stream_state, Mapping):
        raise CheckpointValidationError("data stream state is malformed")
    return {
        "cpu_rng_state": cpu_rng_state,
        "cuda_rng_states": cuda_rng_states,
        "training_generator_state": training_state,
        "evaluation_generator_state": evaluation_state,
        "data_stream_state": data_stream_state,
    }


def _validate_runtime_progress(
    step: int,
    trainer_state: TrainerState,
    scheduler_state: Mapping[str, object],
    ema_state: Mapping[str, object],
    *,
    optimizer_count: int,
) -> None:
    if trainer_state.global_step != step:
        raise CheckpointValidationError(
            "TrainerState progress does not match checkpoint step"
        )
    if set(scheduler_state) != {"backbone", "text_decoder"}:
        raise CheckpointValidationError("scheduler progress state is malformed")
    completed_steps: list[int] = []
    for name in ("backbone", "text_decoder"):
        value = scheduler_state[name]
        if (
            not isinstance(value, Mapping)
            or set(value) != {"completed_steps"}
            or type(value["completed_steps"]) is not int
        ):
            raise CheckpointValidationError(
                f"{name} scheduler progress state is malformed"
            )
        completed_steps.append(value["completed_steps"])
    if any(completed != step for completed in completed_steps):
        raise CheckpointValidationError(
            "scheduler progress does not match checkpoint step"
        )

    optimizer_steps = ema_state.get("optimizer_steps")
    if (
        not isinstance(optimizer_steps, tuple)
        or len(optimizer_steps) != optimizer_count
        or any(
            type(completed) is not int or completed != step
            for completed in optimizer_steps
        )
    ):
        raise CheckpointValidationError(
            "EMA optimizer progress does not match checkpoint step"
        )


def _constant_lr_horizon_extension_is_compatible(
    saved: MFConfig,
    current: MFConfig,
) -> bool:
    """Allow only a longer trainer/scheduler horizon under an unchanged constant LR."""

    saved_horizon = saved.trainer.max_steps
    current_horizon = current.trainer.max_steps
    if current_horizon <= saved_horizon:
        return False
    if saved.optimizers.schedule.max_steps != saved_horizon:
        return False
    if current.optimizers.schedule.max_steps != current_horizon:
        return False
    if (
        saved.optimizers.schedule.schedule != "constant"
        or current.optimizers.schedule.schedule != "constant"
    ):
        return False
    for config in (saved, current):
        for optimizer in (
            config.optimizers.backbone,
            config.optimizers.text_decoder,
        ):
            if optimizer.peak_lr != optimizer.min_lr:
                return False

    normalized_current = current.model_copy(
        update={
            "optimizers": current.optimizers.model_copy(
                update={
                    "schedule": current.optimizers.schedule.model_copy(
                        update={"max_steps": saved.optimizers.schedule.max_steps}
                    )
                }
            )
        }
    )
    return training_fingerprint(normalized_current) == training_fingerprint(saved)


def _trainer_payload(step: int, state: TrainerState) -> Mapping[str, object]:
    return {
        "version": _CHECKPOINT_VERSION,
        "step": step,
        "trainer_state": {
            "global_step": state.global_step,
            "samples_seen": state.samples_seen,
            "last_checkpoint_step": state.last_checkpoint_step,
            "last_evaluation_step": state.last_evaluation_step,
        },
    }


def _load_trainer_state(path: Path, step: int) -> TrainerState:
    payload = _read_json(path)
    if set(payload) != {"version", "step", "trainer_state"}:
        raise CheckpointValidationError("trainer state fields are invalid")
    if type(payload["version"]) is not int or payload["version"] != _CHECKPOINT_VERSION:
        raise CheckpointValidationError("unsupported trainer state version")
    if type(payload["step"]) is not int or payload["step"] != step:
        raise CheckpointValidationError("trainer state step mismatch")
    state = payload["trainer_state"]
    expected_keys = {
        "global_step",
        "samples_seen",
        "last_checkpoint_step",
        "last_evaluation_step",
    }
    if not isinstance(state, Mapping) or set(state) != expected_keys:
        raise CheckpointValidationError("TrainerState fields are invalid")
    try:
        trainer_state = TrainerState(**state)
    except (TypeError, ValueError) as error:
        raise CheckpointValidationError("TrainerState values are invalid") from error
    if trainer_state.global_step != step:
        raise CheckpointValidationError("TrainerState global step mismatch")
    return trainer_state


def _validate_manifest_identity(
    manifest: Mapping[str, object],
    checkpoint: Path,
    step: int,
    world_size: int,
) -> None:
    if set(manifest) != _MANIFEST_KEYS:
        raise CheckpointValidationError("checkpoint manifest fields are invalid")
    if manifest["format"] != _CHECKPOINT_FORMAT:
        raise CheckpointValidationError("checkpoint format is unsupported")
    if (
        type(manifest["version"]) is not int
        or manifest["version"] != _CHECKPOINT_VERSION
    ):
        raise CheckpointValidationError("checkpoint version is unsupported")
    if type(manifest["step"]) is not int or manifest["step"] != step:
        raise CheckpointValidationError("checkpoint manifest step mismatch")
    if type(manifest["world_size"]) is not int or manifest["world_size"] != world_size:
        raise CheckpointValidationError("checkpoint world_size mismatch")
    if manifest["checkpoint_name"] != checkpoint.name:
        raise CheckpointValidationError("checkpoint manifest directory name mismatch")
    if manifest["checkpoint_path"] != str(checkpoint):
        raise CheckpointValidationError("checkpoint manifest path mismatch")
    for name in ("training_fingerprint", "full_config_hash"):
        value = manifest[name]
        try:
            _validate_sha256(name, value)
        except ValueError as error:
            raise CheckpointValidationError(
                f"checkpoint manifest {name} is invalid"
            ) from error
    _evaluation_identity_from_manifest(manifest)
    expected_rank_files = [
        f"rank_states/{_rank_file_name(rank)}" for rank in range(world_size)
    ]
    if manifest["rank_files"] != expected_rank_files:
        raise CheckpointValidationError("checkpoint rank file manifest is invalid")
    git = manifest["git"]
    if (
        not isinstance(git, Mapping)
        or set(git) != {"commit", "dirty"}
        or not isinstance(git["commit"], str)
        or not git["commit"]
        or not isinstance(git["dirty"], bool)
    ):
        raise CheckpointValidationError("checkpoint Git metadata is invalid")
    dependencies = manifest["dependencies"]
    if not isinstance(dependencies, Mapping) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(version, str)
        or not version
        for name, version in dependencies.items()
    ):
        raise CheckpointValidationError("checkpoint dependency metadata is invalid")


def _evaluation_identity_from_manifest(
    manifest: Mapping[str, object],
) -> EvaluationIdentity:
    try:
        return EvaluationIdentity.from_mapping(manifest.get("evaluation_identity"))
    except ValueError as error:
        raise CheckpointValidationError(
            "checkpoint evaluation identity is invalid"
        ) from error


def _expected_payload_paths(world_size: int) -> set[str]:
    return set(_SHARED_PAYLOAD_FILES) | {
        f"rank_states/{_rank_file_name(rank)}" for rank in range(world_size)
    }


def _validate_directory_layout(path: Path, world_size: int, *, stage: str) -> None:
    expected_top = set(_SHARED_PAYLOAD_FILES) | {"rank_states"}
    if stage in {"manifest", "completed"}:
        expected_top.add("run_manifest.json")
    if stage == "completed":
        expected_top.add("COMPLETED")
    actual_top = {entry.name for entry in path.iterdir()}
    if actual_top != expected_top:
        raise CheckpointValidationError(
            f"checkpoint files mismatch (missing={sorted(expected_top - actual_top)}, "
            f"unexpected={sorted(actual_top - expected_top)})"
        )
    rank_directory = path / "rank_states"
    if not rank_directory.is_dir() or rank_directory.is_symlink():
        raise CheckpointValidationError("rank_states must be a real directory")
    expected_ranks = {_rank_file_name(rank) for rank in range(world_size)}
    actual_ranks = {entry.name for entry in rank_directory.iterdir()}
    if actual_ranks != expected_ranks:
        raise CheckpointValidationError(
            f"rank files mismatch (missing={sorted(expected_ranks - actual_ranks)}, "
            f"unexpected={sorted(actual_ranks - expected_ranks)})"
        )
    for relative_path in _expected_payload_paths(world_size):
        payload_path = path / relative_path
        if not payload_path.is_file() or payload_path.is_symlink():
            raise CheckpointValidationError(
                f"checkpoint payload is not a regular file: {relative_path}"
            )


def _build_file_records(
    root: Path,
    relative_paths: Iterable[str],
) -> dict[str, dict[str, int | str]]:
    ordered_paths = tuple(sorted(relative_paths))
    if not ordered_paths:
        return {}

    def hash_record(relative_path: str) -> tuple[str, dict[str, int | str]]:
        payload_path = root / relative_path
        size_before = payload_path.stat().st_size
        sha256 = _sha256_file(payload_path)
        size_after = payload_path.stat().st_size
        if size_after != size_before:
            raise CheckpointValidationError(
                f"checkpoint payload changed while hashing: {relative_path}"
            )
        return relative_path, {"size": size_after, "sha256": sha256}

    worker_count = min(_CHECKPOINT_HASH_WORKERS, len(ordered_paths))
    if worker_count == 1:
        relative_path, record = hash_record(ordered_paths[0])
        return {relative_path: record}
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="mf-checkpoint-hash",
    ) as executor:
        return dict(executor.map(hash_record, ordered_paths))


def _validate_file_records(root: Path, manifest: Mapping[str, object]) -> None:
    files = manifest.get("files")
    expected_paths = _expected_payload_paths(manifest["world_size"])
    if not isinstance(files, Mapping) or set(files) != expected_paths:
        raise CheckpointValidationError("checkpoint manifest payload list is invalid")
    for relative_path in sorted(expected_paths):
        record = files[relative_path]
        if not isinstance(record, Mapping) or set(record) != {"size", "sha256"}:
            raise CheckpointValidationError(f"file record is invalid: {relative_path}")
        if (
            type(record["size"]) is not int
            or record["size"] < 0
            or not isinstance(record["sha256"], str)
            or len(record["sha256"]) != 64
        ):
            raise CheckpointValidationError(
                f"file record values are invalid: {relative_path}"
            )

    actual_records = _build_file_records(root, expected_paths)
    for relative_path in sorted(expected_paths):
        record = files[relative_path]
        actual = actual_records[relative_path]
        if actual["size"] != record["size"]:
            raise CheckpointValidationError(
                f"checkpoint payload size mismatch: {relative_path}"
            )
        if actual["sha256"] != record["sha256"]:
            raise CheckpointValidationError(
                f"checkpoint payload hash is corrupt: {relative_path}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise CheckpointValidationError(
            f"JSON file is missing or invalid: {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise CheckpointValidationError(
            f"JSON file must contain an object: {path.name}"
        )
    return value


def _evaluation_is_completed(
    eval_root: Path,
    step: int,
    checkpoint: Path,
    evaluation_identity: EvaluationIdentity,
) -> bool:
    directory = eval_root / _step_name(step)
    manifest_path = directory / "manifest.json"
    try:
        status = _read_json(directory / "status.json")
        manifest = _read_json(manifest_path)
        manifest_sha256 = status.get("manifest_sha256")
        _validate_sha256("evaluation manifest_sha256", manifest_sha256)
        if manifest_sha256 != _sha256_file(manifest_path):
            return False
    except (CheckpointValidationError, OSError, ValueError):
        return False
    checkpoint_identity = str(checkpoint)
    suites = manifest.get("suites")
    if not isinstance(suites, Mapping) or set(suites) != set(
        evaluation_identity.required_suites
    ):
        return False
    matrix_fields_present = {
        name for name in ("matrix_sha256", "matrix_variants") if name in manifest
    }
    expected_matrix_fields = (
        {"matrix_sha256", "matrix_variants"}
        if evaluation_identity.matrix_sha256 is not None
        else set()
    )
    if matrix_fields_present != expected_matrix_fields:
        return False
    try:
        identity_payload = {
            name: manifest.get(name) for name in evaluation_identity.as_dict()
        }
        identity_payload["required_suites"] = list(evaluation_identity.required_suites)
        manifest_identity = EvaluationIdentity.from_mapping(identity_payload)
    except ValueError:
        return False
    return (
        status.get("status") == "completed"
        and type(status.get("step")) is int
        and status["step"] == step
        and status.get("manifest") == "manifest.json"
        and status.get("checkpoint") == checkpoint_identity
        and type(manifest.get("step")) is int
        and manifest["step"] == step
        and manifest.get("checkpoint") == checkpoint_identity
        and manifest_identity == evaluation_identity
    )


def _atomic_json(path: Path, value: object, *, rank: int) -> None:
    serialized = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    del rank
    durable_write_bytes(path, serialized)


def _atomic_torch_save(path: Path, value: object, *, rank: int) -> None:
    _require_safe_tree(value, path.name)
    del rank
    durable_write_stream(path, lambda handle: torch.save(value, handle))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_torch_payload(path: Path) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointValidationError(
            f"safe checkpoint load failed: {path.name}"
        ) from error
    if not isinstance(payload, Mapping):
        raise CheckpointValidationError(
            f"checkpoint payload must be a mapping: {path.name}"
        )
    return payload


def _clone_cpu_tree(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_cpu_tree(item) for item in value]
    return copy.deepcopy(value)


def _require_safe_tree(value: object, label: str) -> None:
    if value is None or isinstance(value, (str, int, float, bool, Tensor)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise TypeError(f"{label} contains an unsafe mapping key")
            _require_safe_tree(item, label)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _require_safe_tree(item, label)
        return
    raise TypeError(f"{label} contains unsupported type {_class_key(value)}")


def _exception_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _join_errors(errors: tuple[str | None, ...]) -> str:
    return "; ".join(
        f"rank {rank}: {error}"
        for rank, error in enumerate(errors)
        if error is not None
    )
