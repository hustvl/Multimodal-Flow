from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from mf.config.schema import ChunkPackConfig, TaskWeightsConfig
from mf.contracts.batch import (
    TEXT_PREFIX_TOKENS,
    VISION_PREFIX_TOKENS,
    VISION_TOKENS,
    TaskType,
)
from mf.contracts.task_registry import (
    BranchRole,
    task_definition,
    task_definitions,
    task_label,
    task_value,
)
from mf.data.collate import RawTaskSample

def chunk_pack_task_order() -> tuple[TaskType, ...]:
    """Return the current packing order from the live task registry."""

    return tuple(task_value(definition) for definition in task_definitions("packing"))


def _chunk_pack_task_priority() -> dict[TaskType, int]:
    return {task: -index for index, task in enumerate(chunk_pack_task_order())}
_ELASTIC_LEDGER_STATE_VERSION = 2
_PHYSICAL_EXPOSURE_BASIS = "physical_tokens"
_TARGET_EXPOSURE_BASIS = "target_supervised_tokens"
_SAMPLE_EXPOSURE_BASIS = "logical_samples"


def _empty_task_counters() -> dict[TaskType, int]:
    return {task: 0 for task in chunk_pack_task_order()}


def _validated_task_counters(
    values: Mapping[TaskType, int],
    *,
    name: str,
) -> dict[TaskType, int]:
    if set(values) != set(chunk_pack_task_order()):
        raise ValueError(f"elastic chunk pack {name} fields are malformed")
    counters: dict[TaskType, int] = {}
    for task in chunk_pack_task_order():
        value = values[task]
        if type(value) is not int or value < 0:
            raise ValueError(f"elastic chunk pack {name} must be non-negative integers")
        counters[task] = value
    return counters


@dataclass(frozen=True, slots=True)
class ChunkPackUsage:
    sample_counts: Mapping[TaskType, int]
    physical_tokens: Mapping[TaskType, int]
    target_tokens: Mapping[TaskType, int]
    overflow_counts: Mapping[TaskType, int]
    defer_counts: Mapping[TaskType, int]
    task_reassignment_counts: Mapping[TaskType, int]


@dataclass(slots=True)
class ElasticChunkPackLedger:
    """Track deterministic task exposure and chunk-pack diagnostics."""

    pack_count: int = 0
    token_totals: dict[TaskType, int] = field(default_factory=_empty_task_counters)
    target_token_totals: dict[TaskType, int] = field(default_factory=_empty_task_counters)
    sample_totals: dict[TaskType, int] = field(default_factory=_empty_task_counters)
    overflow_totals: dict[TaskType, int] = field(default_factory=_empty_task_counters)
    defer_totals: dict[TaskType, int] = field(default_factory=_empty_task_counters)
    task_reassignment_totals: dict[TaskType, int] = field(default_factory=_empty_task_counters)

    def select_task(
        self,
        *,
        budgets: Mapping[TaskType, int],
        current_tokens: Mapping[TaskType, int],
        blocked: set[TaskType],
        current_target_tokens: Mapping[TaskType, int] | None = None,
        current_sample_counts: Mapping[TaskType, int] | None = None,
        target_weights: Mapping[TaskType, float] | None = None,
        exposure_basis: str = _PHYSICAL_EXPOSURE_BASIS,
    ) -> TaskType | None:
        candidates = [
            task for task in chunk_pack_task_order() if task not in blocked and budgets[task] > 0
        ]
        if not candidates:
            return None
        if exposure_basis == _PHYSICAL_EXPOSURE_BASIS:
            target_pack_count = self.pack_count + 1
            return max(
                candidates,
                key=lambda task: (
                    target_pack_count * budgets[task]
                    - self.token_totals[task]
                    - current_tokens[task],
                    _chunk_pack_task_priority()[task],
                ),
            )
        if exposure_basis == _SAMPLE_EXPOSURE_BASIS:
            if current_sample_counts is None or target_weights is None:
                raise ValueError("logical-sample exposure requires sample counters and weights")
            exposure = {
                task: self.sample_totals[task] + current_sample_counts[task]
                for task in chunk_pack_task_order()
            }
        elif exposure_basis == _TARGET_EXPOSURE_BASIS:
            if current_target_tokens is None or target_weights is None:
                raise ValueError("target-supervised exposure requires target counters and weights")
            exposure = {
                task: self.target_token_totals[task] + current_target_tokens[task]
                for task in chunk_pack_task_order()
            }
        else:
            raise ValueError(f"unknown chunk pack exposure basis {exposure_basis!r}")

        assert target_weights is not None
        if set(target_weights) != set(chunk_pack_task_order()):
            raise ValueError("chunk pack target weights are malformed")
        if any(weight < 0.0 for weight in target_weights.values()):
            raise ValueError("chunk pack target weights must be non-negative")
        total_weight = math.fsum(target_weights.values())
        if total_weight <= 0.0:
            raise ValueError("chunk pack target weights must have positive mass")

        # In target mode the fixed physical pack is only a capacity constraint.
        # Task weights, rather than physical lane budgets, define supervised exposure.
        total_exposure = sum(exposure.values())
        deficits = {
            task: target_weights[task] * total_exposure - total_weight * exposure[task]
            for task in candidates
            if target_weights[task] > 0.0
        }
        eligible = [task for task in deficits if deficits[task] >= 0.0]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda task: (
                deficits[task],
                _chunk_pack_task_priority()[task],
            ),
        )

    def record_pack(
        self,
        usage: ChunkPackUsage | Mapping[TaskType, int],
    ) -> None:
        if isinstance(usage, Mapping):
            usage = ChunkPackUsage(
                sample_counts=_empty_task_counters(),
                physical_tokens=usage,
                target_tokens=_empty_task_counters(),
                overflow_counts=_empty_task_counters(),
                defer_counts=_empty_task_counters(),
                task_reassignment_counts=_empty_task_counters(),
            )
        fields = (
            ("sample counts", usage.sample_counts, self.sample_totals),
            ("physical token counts", usage.physical_tokens, self.token_totals),
            ("target token counts", usage.target_tokens, self.target_token_totals),
            ("overflow counts", usage.overflow_counts, self.overflow_totals),
            ("defer counts", usage.defer_counts, self.defer_totals),
            (
                "task reassignment counts",
                usage.task_reassignment_counts,
                self.task_reassignment_totals,
            ),
        )
        for name, values, totals in fields:
            validated = _validated_task_counters(values, name=name)
            for task in chunk_pack_task_order():
                totals[task] += validated[task]
        self.pack_count += 1

    def record_task_reassignment(self, task: TaskType) -> None:
        if task not in chunk_pack_task_order():
            raise ValueError("task reassignment counter requires a pack task")
        self.task_reassignment_totals[task] += 1

    def counter_snapshot(self) -> dict[str, object]:
        def labeled(values: Mapping[TaskType, int]) -> dict[str, int]:
            return {task_label(task): values[task] for task in chunk_pack_task_order()}

        supervised_text_tokens = sum(
            self.target_token_totals[task_value(definition)]
            for definition in task_definitions("metrics")
            if definition.role("text") is BranchRole.TARGET
        )
        supervised_image_tokens = sum(
            self.target_token_totals[task_value(definition)]
            for definition in task_definitions("metrics")
            if definition.role("image") is BranchRole.TARGET
        )
        return {
            "pack_count": self.pack_count,
            "sample_counts": labeled(self.sample_totals),
            "physical_tokens": labeled(self.token_totals),
            "target_tokens": labeled(self.target_token_totals),
            "supervised_text_tokens": supervised_text_tokens,
            "supervised_image_tokens": supervised_image_tokens,
            "overflow_counts": labeled(self.overflow_totals),
            "defer_counts": labeled(self.defer_totals),
            "task_reassignment_counts": labeled(self.task_reassignment_totals),
        }

    def state_dict(self) -> dict[str, object]:
        return {"version": _ELASTIC_LEDGER_STATE_VERSION, **self.counter_snapshot()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> ElasticChunkPackLedger:
        version = state.get("version")
        if version == 1:
            return cls._from_v1_state(state)
        if version != _ELASTIC_LEDGER_STATE_VERSION:
            raise ValueError("elastic chunk pack ledger state is malformed")
        count = state.get("pack_count")
        if type(count) is not int or count < 0:
            raise ValueError("elastic chunk pack ledger state is malformed")
        return cls(
            pack_count=count,
            sample_totals=cls._task_counters_from_state(state, "sample_counts"),
            token_totals=cls._task_counters_from_state(state, "physical_tokens"),
            target_token_totals=cls._task_counters_from_state(state, "target_tokens"),
            overflow_totals=cls._task_counters_from_state(state, "overflow_counts"),
            defer_totals=cls._task_counters_from_state(state, "defer_counts"),
            task_reassignment_totals=cls._task_counters_from_state(
                state, "task_reassignment_counts"
            ),
        )

    @classmethod
    def _from_v1_state(cls, state: Mapping[str, object]) -> ElasticChunkPackLedger:
        count = state.get("count")
        if type(count) is not int or count < 0:
            raise ValueError("elastic chunk pack ledger state is malformed")
        raw_totals = state.get("token_totals")
        if not isinstance(raw_totals, Mapping):
            raise ValueError("elastic chunk pack ledger state is malformed")
        return cls(
            pack_count=count,
            token_totals=cls._labeled_task_counters(raw_totals, "token totals"),
        )

    @classmethod
    def _task_counters_from_state(
        cls,
        state: Mapping[str, object],
        name: str,
    ) -> dict[TaskType, int]:
        raw_values = state.get(name)
        if not isinstance(raw_values, Mapping):
            raise ValueError("elastic chunk pack ledger state is malformed")
        return cls._labeled_task_counters(raw_values, name)

    @staticmethod
    def _labeled_task_counters(
        raw_values: Mapping[object, object],
        name: str,
    ) -> dict[TaskType, int]:
        if set(raw_values) != {task_label(task) for task in chunk_pack_task_order()}:
            raise ValueError(f"elastic chunk pack ledger {name} are malformed")
        counters: dict[TaskType, int] = {}
        for task in chunk_pack_task_order():
            value = raw_values.get(task_label(task))
            if type(value) is not int or value < 0:
                raise ValueError(f"elastic chunk pack ledger {name} are malformed")
            counters[task] = value
        return counters


def chunk_pack_token_budgets(config: ChunkPackConfig) -> Mapping[TaskType, int]:
    budgets = config.token_budgets.as_mapping()
    return {
        task_value(definition): budgets[definition.name]
        for definition in task_definitions("packing")
    }


def chunk_pack_target_weights(
    config: TaskWeightsConfig,
) -> Mapping[TaskType, float]:
    weights = config.as_mapping()
    return {
        task_value(definition): weights[definition.name]
        for definition in task_definitions("planner")
    }


def chunk_sample_physical_tokens(
    sample: RawTaskSample,
    *,
    image_to_text_prompt_tokens: int,
    image_chunk_conditioning: Literal[
        "token_additive",
        "legacy_active_prefix",
    ] = "token_additive",
    t2i_chunk_semantics: Literal[
        "chunk_native",
        "legacy_block_exact",
    ] = "chunk_native",
    vision_tokens: int = VISION_TOKENS,
) -> int:
    if type(image_to_text_prompt_tokens) is not int or image_to_text_prompt_tokens < 0:
        raise ValueError("image_to_text_prompt_tokens must be non-negative")
    if image_chunk_conditioning not in {"token_additive", "legacy_active_prefix"}:
        raise ValueError("unknown image chunk conditioning mode")
    if t2i_chunk_semantics not in {"chunk_native", "legacy_block_exact"}:
        raise ValueError("unknown T2I chunk semantics")
    if (
        t2i_chunk_semantics == "legacy_block_exact"
        and image_chunk_conditioning != "legacy_active_prefix"
    ):
        raise ValueError(
            "legacy_block_exact T2I semantics require legacy_active_prefix image conditioning"
        )
    if type(vision_tokens) is not int or vision_tokens <= 0:
        raise ValueError("vision_tokens must be a positive integer")
    image_tokens = vision_tokens + (
        VISION_PREFIX_TOKENS if image_chunk_conditioning == "legacy_active_prefix" else 0
    )
    definition = task_definition(sample.task_type)
    text_role = definition.role("text")
    vision_role = definition.role("image")
    if text_role is not BranchRole.ABSENT and sample.text is None:
        raise ValueError(f"{definition.label} chunk is missing text")
    if vision_role is not BranchRole.ABSENT and sample.image is None:
        raise ValueError(f"{definition.label} chunk is missing an image")
    text_tokens = (
        0 if sample.text is None else int(sample.text.content_mask.sum(dtype=None).item())
    )
    prompt_tokens = image_to_text_prompt_tokens if definition.prompt_policy != "forbidden" else 0
    if sample.text_prompt is not None:
        prompt_tokens = int(sample.text_prompt.content_mask.sum(dtype=None).item())
    text_prefix_tokens = 0
    if (
        t2i_chunk_semantics == "legacy_block_exact"
        and vision_role is BranchRole.TARGET
        and text_role is BranchRole.CONDITION
    ):
        text_prefix_tokens = TEXT_PREFIX_TOKENS
    text_copies = 2 if text_role is BranchRole.TARGET else 1
    return (
        (image_tokens if vision_role is not BranchRole.ABSENT else 0)
        + (text_copies * text_tokens if text_role is not BranchRole.ABSENT else 0)
        + (prompt_tokens if definition.prompt_policy != "forbidden" else 0)
        + text_prefix_tokens
    )


def chunk_sample_text_target_tokens(sample: RawTaskSample) -> int:
    """Count text positions that receive loss, never physical conditioning copies.

    Multimodal-flow layouts materialize a clean text condition next to the noisy
    target.  The clean copy consumes attention compute but is not a second unit
    of language training exposure and must not enter cross-architecture matching.
    """

    if task_definition(sample.task_type).role("text") is not BranchRole.TARGET:
        return 0
    if sample.text is None:
        raise ValueError(f"{sample.task_type.label} chunk is missing text")
    return int(sample.text.target_mask.sum(dtype=None).item())


def chunk_sample_image_target_tokens(
    sample: RawTaskSample,
    *,
    vision_tokens: int = VISION_TOKENS,
) -> int:
    """Count image positions that receive a generation loss."""

    if task_definition(sample.task_type).role("image") is BranchRole.TARGET:
        if sample.image is None:
            raise ValueError(f"{sample.task_type.label} chunk is missing an image")
        if type(vision_tokens) is not int or vision_tokens <= 0:
            raise ValueError("vision_tokens must be a positive integer")
        return vision_tokens
    return 0


def chunk_sample_target_tokens(
    sample: RawTaskSample,
    *,
    vision_tokens: int = VISION_TOKENS,
) -> int:
    """Count supervised targets, excluding clean conditions and prompts."""

    text_target_tokens = chunk_sample_text_target_tokens(sample)
    return text_target_tokens + chunk_sample_image_target_tokens(
        sample,
        vision_tokens=vision_tokens,
    )
