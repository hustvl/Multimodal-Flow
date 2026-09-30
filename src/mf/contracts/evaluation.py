from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from mf._compat import Self

CheckpointTarget = Literal["ema", "model"]
_LEGACY_STANDARD_SUITES = (
    "fid30k",
    "geneval",
    "dpgbench",
    "text",
    "gpic_captions",
)
_EVALUATION_IDENTITY_HASHES = (
    "config_hash",
    "stats_hash",
    "codec_hash",
    "prompt_hash",
)
_EVALUATION_IDENTITY_FIELDS = frozenset(
    {
        "checkpoint_target",
        *_EVALUATION_IDENTITY_HASHES,
        "cfg_scale",
        "num_inference_steps",
        "seed",
        "world_size",
        "required_suites",
    }
)
_MATRIX_EVALUATION_IDENTITY_FIELDS = _EVALUATION_IDENTITY_FIELDS | {
    "matrix_sha256",
    "matrix_variants",
}


def _require_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_negative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_sha256(name: str, value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class EvaluationIdentity:
    """Immutable content and runtime identity for one synchronous evaluation."""

    config_hash: str
    stats_hash: str
    codec_hash: str
    prompt_hash: str
    cfg_scale: float
    num_inference_steps: int
    seed: int
    world_size: int
    required_suites: tuple[str, ...]
    checkpoint_target: Literal["ema"] = "ema"
    matrix_sha256: str | None = None
    matrix_variants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.required_suites == _LEGACY_STANDARD_SUITES:
            object.__setattr__(
                self,
                "required_suites",
                (*self.required_suites, "fixed_validation"),
            )
        if type(self.checkpoint_target) is not str or self.checkpoint_target != "ema":
            raise ValueError("checkpoint_target must be exactly 'ema'")
        for name in _EVALUATION_IDENTITY_HASHES:
            _require_sha256(name, getattr(self, name))
        if (
            type(self.cfg_scale) is not float
            or not math.isfinite(self.cfg_scale)
            or self.cfg_scale < 0.0
        ):
            raise ValueError("cfg_scale must be a finite non-negative float")
        if type(self.num_inference_steps) is not int or self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.world_size) is not int or self.world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        if (
            type(self.required_suites) is not tuple
            or any(
                type(name) is not str or not name or name != name.strip()
                for name in self.required_suites
            )
            or len(self.required_suites) != len(set(self.required_suites))
        ):
            raise ValueError("required_suites must be unique non-empty names")
        if type(self.matrix_variants) is not tuple:
            raise ValueError("matrix_variants must be a tuple")
        if self.matrix_sha256 is None:
            if self.matrix_variants:
                raise ValueError(
                    "matrix_sha256 and matrix_variants must be provided together"
                )
        else:
            if not self.matrix_variants:
                raise ValueError(
                    "matrix_sha256 and matrix_variants must be provided together"
                )
            _require_sha256("matrix_sha256", self.matrix_sha256)
            if any(
                type(name) is not str or not name or name != name.strip()
                for name in self.matrix_variants
            ) or len(self.matrix_variants) != len(set(self.matrix_variants)):
                raise ValueError("matrix_variants must contain unique non-empty names")

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "checkpoint_target": self.checkpoint_target,
            "config_hash": self.config_hash,
            "stats_hash": self.stats_hash,
            "codec_hash": self.codec_hash,
            "prompt_hash": self.prompt_hash,
            "cfg_scale": self.cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "world_size": self.world_size,
            "required_suites": list(self.required_suites),
        }
        if self.matrix_sha256 is not None:
            payload["matrix_sha256"] = self.matrix_sha256
            payload["matrix_variants"] = list(self.matrix_variants)
        return payload

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        if not isinstance(value, Mapping) or set(value) not in (
            _EVALUATION_IDENTITY_FIELDS,
            _MATRIX_EVALUATION_IDENTITY_FIELDS,
        ):
            raise ValueError("evaluation identity fields are invalid")
        required_suites = value["required_suites"]
        if not isinstance(required_suites, (list, tuple)):
            raise ValueError("evaluation identity required_suites are invalid")
        matrix_variants = value.get("matrix_variants", ())
        if not isinstance(matrix_variants, (list, tuple)):
            raise ValueError("evaluation identity matrix_variants are invalid")
        return cls(
            checkpoint_target=value["checkpoint_target"],
            config_hash=value["config_hash"],
            stats_hash=value["stats_hash"],
            codec_hash=value["codec_hash"],
            prompt_hash=value["prompt_hash"],
            cfg_scale=value["cfg_scale"],
            num_inference_steps=value["num_inference_steps"],
            seed=value["seed"],
            world_size=value["world_size"],
            required_suites=tuple(required_suites),
            matrix_sha256=value.get("matrix_sha256"),
            matrix_variants=tuple(matrix_variants),
        )
