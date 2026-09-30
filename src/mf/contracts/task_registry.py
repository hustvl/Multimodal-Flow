"""Declarative task definitions used by the MF data and training contracts.

The paper configuration selects a task recipe; it should not be responsible for
reimplementing task semantics in every planner, packer, and metric collector.
This registry keeps the stable storage IDs while making those semantics one
explicit extension point.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Literal


class TaskType(IntEnum):
    """Built-in task IDs kept stable for checkpoint and data compatibility."""

    TEXT_ONLY = 1
    IMAGE_ONLY = 2
    IMAGE_TO_TEXT = 3
    TEXT_TO_IMAGE = 4

    @property
    def label(self) -> str:
        return self.name.lower()

    def __str__(self) -> str:
        return self.label


class BranchRole(IntEnum):
    """Role of one modality in a task instance."""

    ABSENT = 0
    CONDITION = 1
    TARGET = 2


TaskIdentifier = int
PromptPolicy = Literal["required", "optional", "forbidden"]

_MODALITY_ALIASES = {"vision": "image", "image": "image", "text": "text"}


def canonical_modality_name(name: str) -> str:
    """Normalize public modality aliases to one sequence-level identifier."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("modality name must be a non-empty string")
    return _MODALITY_ALIASES.get(name.strip(), name.strip())


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """The declarative contract for one multimodal task.

    ``modality_roles`` is keyed by the canonical sequence modality name.  The
    legacy ``vision`` alias is normalized to ``image`` at registration time.
    Future codecs can add modalities without changing the task vocabulary.
    """

    task_id: TaskIdentifier
    name: str
    modality_roles: Mapping[str, BranchRole]
    loss_components: tuple[str, ...]
    prompt_policy: PromptPolicy = "forbidden"
    sampler: str = "generic"
    planner_order: int = 0
    packing_order: int = 0
    metric_order: int = 0

    def __post_init__(self) -> None:
        if type(self.task_id) is not int or self.task_id <= 0:
            raise ValueError("task_id must be a positive integer")
        if not self.name or self.name != self.name.strip():
            raise ValueError("task name must be a non-empty, trimmed string")
        if not self.modality_roles:
            raise ValueError("task modality_roles must not be empty")
        roles: dict[str, BranchRole] = {}
        for name, role in self.modality_roles.items():
            canonical = canonical_modality_name(name)
            previous = roles.get(canonical)
            if previous is not None and previous is not role:
                raise ValueError(
                    f"conflicting roles for modality alias {canonical!r}"
                )
            roles[canonical] = role
        if any(not name or not name.strip() for name in roles):
            raise ValueError("task modality names must be non-empty")
        if any(not isinstance(role, BranchRole) for role in roles.values()):
            raise TypeError("task modality roles must be BranchRole values")
        if any(not component or not component.strip() for component in self.loss_components):
            raise ValueError("task loss components must be non-empty")
        if self.prompt_policy not in {"required", "optional", "forbidden"}:
            raise ValueError("unknown task prompt policy")
        if not self.sampler or self.sampler != self.sampler.strip():
            raise ValueError("task sampler must be a non-empty, trimmed string")
        object.__setattr__(self, "modality_roles", MappingProxyType(roles))

    @property
    def label(self) -> str:
        return self.name

    def role(self, modality: str) -> BranchRole:
        """Return a modality role, treating an omitted modality as absent."""

        return self.modality_roles.get(
            canonical_modality_name(modality), BranchRole.ABSENT
        )


class TaskRegistry:
    """Registry of task contracts with stable ID and name lookup."""

    def __init__(self, definitions: tuple[TaskDefinition, ...] = ()) -> None:
        self._by_id: dict[int, TaskDefinition] = {}
        self._by_name: dict[str, TaskDefinition] = {}
        self._frozen = False
        for definition in definitions:
            self.register(definition)

    def register(self, definition: TaskDefinition, *, replace: bool = False) -> None:
        if self._frozen:
            raise RuntimeError(
                "task registry is frozen; register extensions before "
                "constructing the model and data pipeline"
            )
        if not isinstance(definition, TaskDefinition):
            raise TypeError("definition must be a TaskDefinition")
        existing_id = self._by_id.get(definition.task_id)
        existing_name = self._by_name.get(definition.name)
        if not replace and (existing_id is not None or existing_name is not None):
            raise ValueError(f"task is already registered: {definition.name!r}")
        if replace:
            if existing_id is not None:
                self._by_name.pop(existing_id.name, None)
            if existing_name is not None:
                self._by_id.pop(existing_name.task_id, None)
        self._by_id[definition.task_id] = definition
        self._by_name[definition.name] = definition

    def freeze(self) -> None:
        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def resolve(self, task: TaskIdentifier | str | TaskType) -> TaskDefinition:
        if isinstance(task, str):
            definition = self._by_name.get(task)
        else:
            definition = self._by_id.get(int(task))
        if definition is None:
            raise ValueError(f"unknown task: {task!r}")
        return definition

    def definitions(self, order: Literal["id", "planner", "packing", "metrics"] = "id") -> tuple[TaskDefinition, ...]:
        order_key = {
            "id": lambda item: item.task_id,
            "planner": lambda item: (item.planner_order, item.task_id),
            "packing": lambda item: (item.packing_order, item.task_id),
            "metrics": lambda item: (item.metric_order, item.task_id),
        }.get(order)
        if order_key is None:
            raise ValueError(f"unknown task registry order: {order!r}")
        return tuple(sorted(self._by_id.values(), key=order_key))

    def task_types(self, order: Literal["id", "planner", "packing", "metrics"] = "id") -> tuple[TaskIdentifier, ...]:
        return tuple(item.task_id for item in self.definitions(order))

    def index(self, task: TaskIdentifier | str | TaskType, *, order: Literal["id", "planner", "packing", "metrics"] = "metrics") -> int:
        task_id = self.resolve(task).task_id
        for index, definition in enumerate(self.definitions(order)):
            if definition.task_id == task_id:
                return index
        raise AssertionError("registered task disappeared")

    def __iter__(self) -> Iterator[TaskDefinition]:
        return iter(self.definitions())

    def manifest(self) -> tuple[dict[str, object], ...]:
        """Return stable task semantics for checkpoint fingerprints."""

        return tuple(
            {
                "task_id": definition.task_id,
                "name": definition.name,
                "modality_roles": {
                    name: role.name for name, role in definition.modality_roles.items()
                },
                "loss_components": definition.loss_components,
                "prompt_policy": definition.prompt_policy,
                "sampler": definition.sampler,
                "planner_order": definition.planner_order,
                "packing_order": definition.packing_order,
                "metric_order": definition.metric_order,
            }
            for definition in self.definitions()
        )

DEFAULT_TASK_REGISTRY = TaskRegistry(
    (
        TaskDefinition(
            task_id=int(TaskType.TEXT_ONLY),
            name="text_only",
            modality_roles={"image": BranchRole.ABSENT, "text": BranchRole.TARGET},
            loss_components=("text_flow", "text_decoder_ce"),
            prompt_policy="optional",
            sampler="text",
            planner_order=2,
            packing_order=0,
            metric_order=0,
        ),
        TaskDefinition(
            task_id=int(TaskType.IMAGE_ONLY),
            name="image_only",
            modality_roles={"image": BranchRole.TARGET, "text": BranchRole.ABSENT},
            loss_components=("vision_flow",),
            sampler="image",
            planner_order=3,
            packing_order=3,
            metric_order=1,
        ),
        TaskDefinition(
            task_id=int(TaskType.IMAGE_TO_TEXT),
            name="image_to_text",
            modality_roles={"image": BranchRole.CONDITION, "text": BranchRole.TARGET},
            loss_components=("text_flow", "text_decoder_ce"),
            prompt_policy="required",
            sampler="image_to_text",
            planner_order=1,
            packing_order=2,
            metric_order=2,
        ),
        TaskDefinition(
            task_id=int(TaskType.TEXT_TO_IMAGE),
            name="text_to_image",
            modality_roles={"image": BranchRole.TARGET, "text": BranchRole.CONDITION},
            loss_components=("vision_flow",),
            sampler="text_to_image",
            planner_order=0,
            packing_order=1,
            metric_order=3,
        ),
    )
)


def register_task(definition: TaskDefinition, *, replace: bool = False) -> None:
    """Register a task extension for data-factory based experiments."""

    DEFAULT_TASK_REGISTRY.register(definition, replace=replace)


def task_definition(task: TaskIdentifier | str | TaskType) -> TaskDefinition:
    return DEFAULT_TASK_REGISTRY.resolve(task)


def task_value(definition: TaskDefinition) -> TaskIdentifier | TaskType:
    """Return the stable enum value when one exists, otherwise the registered ID."""

    try:
        return TaskType(definition.task_id)
    except ValueError:
        return definition.task_id


def task_label(task: TaskIdentifier | str | TaskType) -> str:
    return task_definition(task).label


def task_definitions(
    order: Literal["id", "planner", "packing", "metrics"] = "id",
) -> tuple[TaskDefinition, ...]:
    return DEFAULT_TASK_REGISTRY.definitions(order)


def task_types(
    order: Literal["id", "planner", "packing", "metrics"] = "id",
) -> tuple[TaskIdentifier, ...]:
    return DEFAULT_TASK_REGISTRY.task_types(order)


def task_index(
    task: TaskIdentifier | str | TaskType,
    *,
    order: Literal["id", "planner", "packing", "metrics"] = "metrics",
) -> int:
    return DEFAULT_TASK_REGISTRY.index(task, order=order)


__all__ = [
    "BranchRole",
    "DEFAULT_TASK_REGISTRY",
    "TaskDefinition",
    "TaskIdentifier",
    "TaskRegistry",
    "TaskType",
    "canonical_modality_name",
    "register_task",
    "task_definition",
    "task_definitions",
    "task_index",
    "task_label",
    "task_types",
    "task_value",
]
