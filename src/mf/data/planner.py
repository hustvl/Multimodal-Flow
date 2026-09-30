from __future__ import annotations

import hashlib

import torch
from mf.config.schema import MFConfig, derive_task_slots
from mf.contracts.batch import TaskType
from mf.contracts.task_registry import task_definitions, task_value
from mf.data.sources import normalized_data_config


def stable_hash(*parts: int | str) -> int:
    """Hash seed components reproducibly across processes and Python versions."""
    digest = hashlib.sha256()
    for part in parts:
        prefix = b"i" if isinstance(part, int) else b"s"
        payload = str(part).encode("utf-8")
        digest.update(prefix)
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return int.from_bytes(digest.digest()[:8], byteorder="big") & ((1 << 63) - 1)


class GlobalTaskBatchPlanner:
    def __init__(self, config: MFConfig) -> None:
        normalized_data_config(config)
        slots = derive_task_slots(
            config.tasks.weights,
            config.distributed.global_batch_size,
        )
        slot_by_task = slots.as_mapping()
        self._plan = tuple(
            task_value(definition)
            for definition in task_definitions("planner")
            for _ in range(slot_by_task[definition.name])
        )
        if len(self._plan) % config.distributed.world_size != 0:
            raise ValueError("configured world_size must divide the global task plan")
        self._run_seed = config.run.seed
        self._planner = config.tasks.planner
        self._world_size = config.distributed.world_size
        self._accumulation_steps = config.distributed.gradient_accumulation_steps
        self._micro_batch_size = config.distributed.micro_batch_size_per_rank

    @classmethod
    def from_config(cls, config: MFConfig) -> GlobalTaskBatchPlanner:
        return cls(config)

    def plan(self, global_step: int) -> tuple[TaskType, ...]:
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_hash(self._run_seed, global_step))
        if self._planner == "rank_balanced_global":
            return self._balanced_plan(generator)
        order = torch.randperm(len(self._plan), generator=generator).tolist()
        return tuple(self._plan[index] for index in order)

    def _balanced_plan(self, generator: torch.Generator) -> tuple[TaskType, ...]:
        local_batch_count = self._world_size * self._accumulation_steps
        plans: list[list[TaskType]] = [[] for _ in range(local_batch_count)]
        task_counts = {
            task_value(definition): self._plan.count(task_value(definition))
            for definition in task_definitions("planner")
        }

        for task, count in task_counts.items():
            base, _ = divmod(count, local_batch_count)
            if base:
                for local_plan in plans:
                    local_plan.extend((task,) * base)

        residual_counts = {
            task: count % local_batch_count
            for task, count in task_counts.items()
            if count % local_batch_count
        }
        if residual_counts:
            tasks = tuple(residual_counts)
            task_order = torch.randperm(len(tasks), generator=generator).tolist()
            for task_index in task_order:
                task = tasks[task_index]
                tie_order = torch.randperm(local_batch_count, generator=generator).tolist()
                tie_priority = {index: priority for priority, index in enumerate(tie_order)}
                destinations = sorted(
                    range(local_batch_count),
                    key=lambda index: (len(plans[index]), tie_priority[index]),
                )[: residual_counts[task]]
                for destination in destinations:
                    plans[destination].append(task)

        if any(len(local_plan) != self._micro_batch_size for local_plan in plans):
            sizes = tuple(len(local_plan) for local_plan in plans)
            raise RuntimeError(f"rank-balanced task planner produced invalid sizes: {sizes}")
        for local_plan in plans:
            order = torch.randperm(len(local_plan), generator=generator).tolist()
            local_plan[:] = [local_plan[index] for index in order]

        rank_major = (
            plans[rank * self._accumulation_steps + accumulation]
            for rank in range(self._world_size)
            for accumulation in range(self._accumulation_steps)
        )
        return tuple(task for local_plan in rank_major for task in local_plan)

    def plan_for_rank(
        self,
        global_step: int,
        *,
        rank: int,
    ) -> tuple[TaskType, ...]:
        if not 0 <= rank < self._world_size:
            raise ValueError("rank must be in [0, world_size)")
        local_batch_size = len(self._plan) // self._world_size
        start = rank * local_batch_size
        return self.plan(global_step)[start : start + local_batch_size]
