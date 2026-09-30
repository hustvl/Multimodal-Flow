from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import torch
from torch import Tensor

from mf._compat import Self
from mf.contracts.chunks import ChunkRoutingMetadata
from mf.contracts.geometry import GeometryContract
from mf.contracts.physical import PhysicalSequenceLayout
from mf.contracts.sequence import MultimodalSequence
from mf.contracts.sequence import CompiledSequence
from mf.contracts.task_registry import (
    BranchRole,
    TaskType,
    task_definition,
    task_definitions,
    task_types,
)

VISION_TOKENS = 256
VISION_LATENT_DIM = 768
VISION_IMAGE_RESOLUTIONS = (224, 256, 384)
TEXT_TOKENS = 256
TEXT_LATENT_DIM = 512
TIME_TOKENS = 4
MODALITY_TOKENS = 4
VISION_PREFIX_TOKENS = TIME_TOKENS + MODALITY_TOKENS
TEXT_PREFIX_TOKENS = TIME_TOKENS + MODALITY_TOKENS
VISION_LAYOUT_TOKENS = VISION_PREFIX_TOKENS + VISION_TOKENS
TEXT_LAYOUT_TOKENS = TEXT_PREFIX_TOKENS + TEXT_TOKENS
TOTAL_LAYOUT_TOKENS = VISION_LAYOUT_TOKENS + TEXT_LAYOUT_TOKENS

_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


def task_roles(task: TaskType) -> tuple[BranchRole, BranchRole]:
    """Return the image/text role pair used by the paper adapter."""

    definition = task_definition(task)
    return definition.role("image"), definition.role("text")


def task_branch_roles(task_type: Tensor) -> tuple[Tensor, Tensor]:
    """Derive image/text roles from the only task-routing source of truth."""

    tasks, _ = _validate_task_type(task_type)
    vision = torch.empty_like(tasks)
    text = torch.empty_like(tasks)
    for definition in task_definitions():
        rows = tasks == definition.task_id
        vision.masked_fill_(rows, int(definition.role("image")))
        text.masked_fill_(rows, int(definition.role("text")))
    return vision, text


def _require_tensor(name: str, value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    return value


def _require_shape(
    name: str, tensor: Tensor, shape: tuple[int, ...], text: str
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {text}; got {list(tensor.shape)}")


def _require_bool(name: str, tensor: Tensor) -> None:
    if tensor.dtype is not torch.bool:
        raise ValueError(f"{name} must have dtype torch.bool; got {tensor.dtype}")


def _require_integer(name: str, tensor: Tensor) -> None:
    if tensor.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"{name} must have an integer dtype; got {tensor.dtype}")


def _require_floating(name: str, tensor: Tensor) -> None:
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{name} must have a floating-point dtype; got {tensor.dtype}")


def _require_same_device(
    reference_name: str, reference: Tensor, **tensors: Tensor
) -> None:
    for name, tensor in tensors.items():
        if tensor.device != reference.device:
            raise ValueError(
                f"{name} must be on the same device as {reference_name}; "
                f"got {tensor.device} and {reference.device}"
            )


def _validate_task_type_metadata(task_type: object) -> tuple[Tensor, int]:
    tensor = _require_tensor("task_type", task_type)
    if tensor.ndim != 1:
        raise ValueError(f"task_type must have shape [B]; got {list(tensor.shape)}")
    _require_integer("task_type", tensor)
    return tensor, tensor.shape[0]


def _validate_task_type(task_type: object) -> tuple[Tensor, int]:
    tensor, batch_size = _validate_task_type_metadata(task_type)

    invalid = torch.ones_like(tensor, dtype=torch.bool)
    for task in task_types():
        invalid &= tensor != task
    if bool(invalid.any()):
        index = int(invalid.nonzero(as_tuple=False)[0, 0].item())
        value = int(tensor[index].item())
        raise ValueError(f"task_type[{index}] has unknown value {value}")
    return tensor, tensor.shape[0]


def _validate_presence(
    vision_present: object,
    text_present: object,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    vision = _require_tensor("vision_present", vision_present)
    text = _require_tensor("text_present", text_present)
    for name, tensor in (("vision_present", vision), ("text_present", text)):
        if tensor.ndim != 1:
            raise ValueError(f"{name} must have shape [B]; got {list(tensor.shape)}")
        if tensor.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch dimension must match task_type; "
                f"got {tensor.shape[0]} and {batch_size}"
            )
    _require_bool("vision_present", vision)
    _require_bool("text_present", text)
    _require_same_device("vision_present", vision, text_present=text)
    return vision, text


def _validate_task_presence(
    task_type: Tensor,
    vision_present: Tensor,
    text_present: Tensor,
) -> None:
    _require_same_device(
        "task_type",
        task_type,
        vision_present=vision_present,
        text_present=text_present,
    )
    expected_vision_role, expected_text_role = task_branch_roles(task_type)
    expected_vision = expected_vision_role != int(BranchRole.ABSENT)
    expected_text = expected_text_role != int(BranchRole.ABSENT)
    wrong_presence = (vision_present != expected_vision) | (
        text_present != expected_text
    )
    if bool(wrong_presence.any()):
        index = int(wrong_presence.nonzero(as_tuple=False)[0, 0].item())
        task = task_definition(int(task_type[index].item()))
        expected_vision_value = bool(expected_vision[index].item())
        expected_text_value = bool(expected_text[index].item())
        raise ValueError(
            f"{task.label} at batch index {index} requires "
            f"vision_present={expected_vision_value} and text_present={expected_text_value}"
        )


def _validate_routing(
    task_type: object,
    vision_present: object,
    text_present: object,
) -> tuple[Tensor, Tensor, Tensor, int]:
    tasks, batch_size = _validate_task_type(task_type)
    vision, text = _validate_presence(vision_present, text_present, batch_size)
    _validate_task_presence(tasks, vision, text)
    return tasks, vision, text, batch_size


def _validate_physical_routing(
    task_type: object,
    vision_present: object,
    text_present: object,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Validate a registry-owned physical task without legacy task assumptions."""

    tasks, batch_size = _validate_task_type_metadata(task_type)
    for value in torch.unique(tasks).tolist():
        task_definition(int(value))
    vision, text = _validate_presence(vision_present, text_present, batch_size)
    if bool(vision.any()) or bool(text.any()):
        raise ValueError(
            "physical sequence batches must route through registered modalities, "
            "not the legacy image/text task roles"
        )
    return tasks, vision, text, batch_size


def _require_present_branch(name: str, tensor: Tensor | None, present: bool) -> None:
    if present and tensor is None:
        raise ValueError(f"{name} is required when its modality is present")


def _validate_absent_text_mask(text_content_mask: Tensor, text_present: Tensor) -> None:
    absent_content = text_content_mask & ~text_present[:, None]
    if bool(absent_content.any()):
        raise ValueError("text_content_mask must be false when text_present is false")


def _task_rows(task_type: Tensor, predicate: Callable[[object], bool]) -> Tensor:
    rows = torch.zeros_like(task_type, dtype=torch.bool)
    for definition in task_definitions():
        if predicate(definition):
            rows |= task_type == definition.task_id
    return rows


def _image_to_text_rows(task_type: Tensor) -> Tensor:
    return _task_rows(task_type, lambda definition: definition.prompt_policy == "required")


def _prompt_allowed_rows(task_type: Tensor) -> Tensor:
    return _task_rows(
        task_type,
        lambda definition: definition.prompt_policy in {"required", "optional"},
    )


def _validate_prompt_mask(task_type: Tensor, prompt_mask: Tensor) -> None:
    if prompt_mask.ndim != 2 or prompt_mask.shape[0] != task_type.shape[0]:
        raise ValueError("text_prompt_content_mask must have shape [B, P]")
    _require_bool("text_prompt_content_mask", prompt_mask)
    _require_same_device("task_type", task_type, text_prompt_content_mask=prompt_mask)
    if prompt_mask.shape[1] <= 0:
        raise ValueError("text_prompt_content_mask must have P > 0 when materialized")
    if bool((~prompt_mask[:, :-1] & prompt_mask[:, 1:]).any()):
        raise ValueError("text_prompt_content_mask must be right padded")
    active_rows = prompt_mask.any(dim=1)
    if bool((active_rows & ~_prompt_allowed_rows(task_type)).any()):
        raise ValueError(
            "only image_to_text and text_only rows may contain an instruction prompt"
        )
    if not bool(active_rows[_image_to_text_rows(task_type)].all()):
        raise ValueError("image_to_text requires a materialized instruction prompt")


@dataclass
class RawTaskBatch:
    """CPU-side task routing with optional collated codec inputs."""

    task_type: Tensor
    vision_present: Tensor
    text_present: Tensor
    images: Tensor | None = None
    text_token_ids: Tensor | None = None
    text_content_mask: Tensor | None = None
    text_prompt_token_ids: Tensor | None = None
    text_prompt_content_mask: Tensor | None = None
    sequence_contracts: tuple[MultimodalSequence, ...] | None = None
    compiled_sequences: tuple[CompiledSequence, ...] | None = None
    physical_layout: PhysicalSequenceLayout | None = None

    cpu_preparation: RawBatchPreparation | None = field(
        default=None, repr=False, compare=False
    )

    def validate(self) -> Self:
        if self.physical_layout is not None:
            return self._validate_physical()
        task_type, vision_present, text_present, batch_size = _validate_routing(
            self.task_type, self.vision_present, self.text_present
        )
        has_vision = bool(vision_present.any())
        has_text = bool(text_present.any())
        _require_present_branch("images", self.images, has_vision)
        _require_present_branch("text_token_ids", self.text_token_ids, has_text)
        _require_present_branch("text_content_mask", self.text_content_mask, has_text)
        if self.sequence_contracts is not None:
            if not isinstance(self.sequence_contracts, tuple):
                raise ValueError("sequence_contracts must be a tuple when provided")
            if len(self.sequence_contracts) != batch_size:
                raise ValueError(
                    "sequence_contracts must contain one sequence per batch row"
                )
            for sequence in self.sequence_contracts:
                if not isinstance(sequence, MultimodalSequence):
                    raise ValueError(
                        "sequence_contracts must contain multimodal sequences"
                    )
                sequence.validate()
            compiled = tuple(sequence.compile() for sequence in self.sequence_contracts)
            if self.compiled_sequences is None:
                self.compiled_sequences = compiled
            if len(self.compiled_sequences) != batch_size:
                raise ValueError(
                    "compiled_sequences must contain one sequence per batch row"
                )
            for sequence, expected in zip(
                self.compiled_sequences, compiled, strict=True
            ):
                if sequence.routing_signature() != expected.routing_signature():
                    raise ValueError(
                        "compiled_sequences must be derived from sequence_contracts"
                    )
        elif self.compiled_sequences is not None:
            if not isinstance(self.compiled_sequences, tuple):
                raise ValueError("compiled_sequences must be a tuple when provided")
            if len(self.compiled_sequences) != batch_size:
                raise ValueError(
                    "compiled_sequences must contain one sequence per batch row"
                )
            for sequence in self.compiled_sequences:
                if not isinstance(sequence, CompiledSequence):
                    raise ValueError(
                        "compiled_sequences must contain CompiledSequence values"
                    )
                sequence.validate()

        tensors: dict[str, Tensor] = {
            "vision_present": vision_present,
            "text_present": text_present,
        }
        if self.images is not None:
            images = _require_tensor("images", self.images)
            valid_shape = (
                images.ndim == 4
                and images.shape[:2] == (batch_size, 3)
                and images.shape[2] == images.shape[3]
                and images.shape[2] > 0
            )
            if not valid_shape:
                raise ValueError(
                    "images must have shape [B, 3, R, R] with a positive square R; "
                    f"got {list(images.shape)}"
                )
            _require_floating("images", images)
            tensors["images"] = images
        text_tokens: int | None = None
        if self.text_token_ids is not None:
            token_ids = _require_tensor("text_token_ids", self.text_token_ids)
            if (
                token_ids.ndim != 2
                or token_ids.shape[0] != batch_size
                or token_ids.shape[1] <= 0
            ):
                raise ValueError("text_token_ids must have shape [B, T] with T > 0")
            text_tokens = token_ids.shape[1]
            _require_integer("text_token_ids", token_ids)
            tensors["text_token_ids"] = token_ids
        if self.text_content_mask is not None:
            content_mask = _require_tensor("text_content_mask", self.text_content_mask)
            if (
                content_mask.ndim != 2
                or content_mask.shape[0] != batch_size
                or content_mask.shape[1] <= 0
            ):
                raise ValueError("text_content_mask must have shape [B, T] with T > 0")
            if text_tokens is not None and content_mask.shape[1] != text_tokens:
                raise ValueError(
                    f"text_content_mask must have shape [B, {text_tokens}] to match text_token_ids"
                )
            _require_bool("text_content_mask", content_mask)
            _require_same_device(
                "text_present", text_present, text_content_mask=content_mask
            )
            _validate_absent_text_mask(content_mask, text_present)
            tensors["text_content_mask"] = content_mask

        prompt_ids = self.text_prompt_token_ids
        prompt_mask = self.text_prompt_content_mask
        if (prompt_ids is None) != (prompt_mask is None):
            raise ValueError(
                "text prompt token ids and content mask must be provided together"
            )
        if prompt_ids is None:
            if bool(_image_to_text_rows(task_type).any()):
                raise ValueError(
                    "image_to_text requires a materialized instruction prompt"
                )
        else:
            ids = _require_tensor("text_prompt_token_ids", prompt_ids)
            mask = _require_tensor("text_prompt_content_mask", prompt_mask)
            if ids.ndim != 2 or ids.shape[0] != batch_size or ids.shape[1] <= 0:
                raise ValueError(
                    "text_prompt_token_ids must have shape [B, P] with P > 0"
                )
            _require_integer("text_prompt_token_ids", ids)
            _require_shape(
                "text_prompt_content_mask",
                mask,
                tuple(ids.shape),
                f"[B, {ids.shape[1]}]",
            )
            _validate_prompt_mask(task_type, mask)
            tensors["text_prompt_token_ids"] = ids
            tensors["text_prompt_content_mask"] = mask

        _require_same_device("task_type", task_type, **tensors)
        return self

    def _validate_physical(self) -> Self:
        task_type, vision_present, text_present, batch_size = _validate_physical_routing(
            self.task_type,
            self.vision_present,
            self.text_present,
        )
        physical = self.physical_layout
        assert physical is not None
        if physical.token_embeddings.shape[0] != batch_size:
            raise ValueError("physical_layout batch dimension must match task_type")
        if any(
            value is not None
            for value in (
                self.images,
                self.text_token_ids,
                self.text_content_mask,
                self.text_prompt_token_ids,
                self.text_prompt_content_mask,
            )
        ):
            raise ValueError("physical sequence batches cannot contain legacy payloads")
        physical.validate()
        _require_same_device(
            "task_type",
            task_type,
            vision_present=vision_present,
            text_present=text_present,
        )
        if self.sequence_contracts is not None:
            if len(self.sequence_contracts) != batch_size:
                raise ValueError(
                    "sequence_contracts must contain one sequence per batch row"
                )
            for sequence in self.sequence_contracts:
                sequence.validate()
            compiled = tuple(sequence.compile() for sequence in self.sequence_contracts)
            if self.compiled_sequences is None:
                self.compiled_sequences = compiled
            if tuple(
                sequence.routing_signature() for sequence in self.compiled_sequences
            ) != tuple(sequence.routing_signature() for sequence in compiled):
                raise ValueError(
                    "compiled_sequences must be derived from sequence_contracts"
                )
        elif self.compiled_sequences is not None:
            if len(self.compiled_sequences) != batch_size:
                raise ValueError(
                    "compiled_sequences must contain one sequence per batch row"
                )
            for sequence in self.compiled_sequences:
                sequence.validate()
        return self


def tensor_mutation_version(value: Tensor | None) -> int | None:
    # Inference tensors have no version counter, so never reuse their prepared work.
    if value is None:
        return None
    return None if value.is_inference() else value._version


@dataclass(frozen=True)
class RawBatchPreparation:
    """Disposable CPU work for one raw batch; never part of data/RNG checkpoints."""

    source_tensors: tuple[Tensor | None, ...]
    source_versions: tuple[int | None, ...]
    chunk_routing: ChunkRoutingMetadata
    vision_indices: Tensor
    text_indices: Tensor
    prompt_indices: Tensor
    compact_images: Tensor | None
    sequence_signatures: tuple[tuple[tuple[object, ...], ...], ...] | None = None

    @staticmethod
    def inputs(batch: RawTaskBatch) -> tuple[Tensor | None, ...]:
        return (
            batch.task_type,
            batch.vision_present,
            batch.text_present,
            batch.images,
            batch.text_token_ids,
            batch.text_content_mask,
            batch.text_prompt_token_ids,
            batch.text_prompt_content_mask,
        )

    def matches(self, batch: RawTaskBatch) -> bool:
        current = self.inputs(batch)
        tensors_match = all(
            actual is expected
            and (
                actual is None
                or version is not None
                and tensor_mutation_version(actual) == version
            )
            for actual, expected, version in zip(
                current, self.source_tensors, self.source_versions, strict=True
            )
        )
        if not tensors_match:
            return False
        current_signatures = None
        if batch.sequence_contracts is not None:
            current_signatures = tuple(
                sequence.routing_signature()
                for sequence in batch.sequence_contracts
            )
        return current_signatures == self.sequence_signatures


@dataclass
class EncodedTaskBatch:
    """Online codec outputs in raw codec latent space."""

    task_type: Tensor
    vision_present: Tensor
    text_present: Tensor
    vision_latents_raw: Tensor | None = None
    vision_token_ids: Tensor | None = None
    semantic_vision_latents_raw: Tensor | None = None
    text_latents_raw: Tensor | None = None
    text_token_ids: Tensor | None = None
    text_content_mask: Tensor | None = None
    text_latent_stats_type: Tensor | None = None
    text_prompt_latents_raw: Tensor | None = None
    text_prompt_token_ids: Tensor | None = None
    text_prompt_content_mask: Tensor | None = None
    text_prompt_latent_stats_type: Tensor | None = None
    chunk_routing_cpu: ChunkRoutingMetadata | None = None
    sequence_contracts: tuple[MultimodalSequence, ...] | None = None
    compiled_sequences: tuple[CompiledSequence, ...] | None = None
    vision_tokens: int = VISION_TOKENS
    vision_latent_dim: int = VISION_LATENT_DIM
    text_latent_dim: int = TEXT_LATENT_DIM
    geometry: GeometryContract | None = None
    semantic_vision_latent_dim: int | None = None
    physical_layout: PhysicalSequenceLayout | None = None

    def validate_metadata(self) -> Self:
        task_type, batch_size = _validate_task_type_metadata(self.task_type)
        vision_present, text_present = _validate_presence(
            self.vision_present,
            self.text_present,
            batch_size,
        )
        if self.sequence_contracts is not None:
            if not isinstance(self.sequence_contracts, tuple):
                raise ValueError("sequence_contracts must be a tuple when provided")
            if len(self.sequence_contracts) != batch_size:
                raise ValueError(
                    "sequence_contracts must contain one sequence per batch row"
                )
            for sequence in self.sequence_contracts:
                if not isinstance(sequence, MultimodalSequence):
                    raise ValueError(
                        "sequence_contracts must contain multimodal sequences"
                    )
                sequence.validate()
            compiled = tuple(sequence.compile() for sequence in self.sequence_contracts)
            if self.compiled_sequences is None:
                self.compiled_sequences = compiled
            if len(self.compiled_sequences) != batch_size:
                raise ValueError(
                    "compiled_sequences must contain one sequence per batch row"
                )
            for sequence, expected in zip(
                self.compiled_sequences, compiled, strict=True
            ):
                if sequence.routing_signature() != expected.routing_signature():
                    raise ValueError(
                        "compiled_sequences must be derived from sequence_contracts"
                    )
        elif self.compiled_sequences is not None:
            if not isinstance(self.compiled_sequences, tuple):
                raise ValueError("compiled_sequences must be a tuple when provided")
            if len(self.compiled_sequences) != batch_size:
                raise ValueError(
                    "compiled_sequences must contain one sequence per batch row"
                )
            for sequence in self.compiled_sequences:
                if not isinstance(sequence, CompiledSequence):
                    raise ValueError(
                        "compiled_sequences must contain CompiledSequence values"
                    )
                sequence.validate()
        token_ids: Tensor | None = None
        text_tokens: int | None = None
        if self.text_token_ids is not None:
            token_ids = _require_tensor("text_token_ids", self.text_token_ids)
            if (
                token_ids.ndim != 2
                or token_ids.shape[0] != batch_size
                or token_ids.shape[1] <= 0
            ):
                raise ValueError("text_token_ids must have shape [B, T] with T > 0")
            _require_integer("text_token_ids", token_ids)
            text_tokens = token_ids.shape[1]

        if self.text_content_mask is None:
            raise ValueError("text_content_mask is required and must have shape [B, T]")
        content_mask = _require_tensor("text_content_mask", self.text_content_mask)
        if (
            content_mask.ndim != 2
            or content_mask.shape[0] != batch_size
            or content_mask.shape[1] <= 0
        ):
            raise ValueError("text_content_mask must have shape [B, T] with T > 0")
        if text_tokens is None:
            text_tokens = content_mask.shape[1]
        else:
            _require_shape(
                "text_content_mask",
                content_mask,
                (batch_size, text_tokens),
                f"[B, {text_tokens}]",
            )

        tensors: dict[str, Tensor] = {
            "vision_present": vision_present,
            "text_present": text_present,
        }
        if type(self.vision_tokens) is not int or self.vision_tokens <= 0:
            raise ValueError("vision_tokens must be a positive integer")
        if type(self.vision_latent_dim) is not int or self.vision_latent_dim <= 0:
            raise ValueError("vision_latent_dim must be a positive integer")
        if type(self.text_latent_dim) is not int or self.text_latent_dim <= 0:
            raise ValueError("text_latent_dim must be a positive integer")
        geometry = self.geometry or GeometryContract(
            vision_tokens=self.vision_tokens,
            vision_latent_dim=self.vision_latent_dim,
            text_latent_dim=self.text_latent_dim,
        )
        geometry.validate()
        self.geometry = geometry
        if self.physical_layout is not None:
            physical = self.physical_layout.validate()
            if physical.token_embeddings.shape[0] != batch_size:
                raise ValueError("physical_layout batch dimension must match encoded batch")
            tensors["physical_layout_token_embeddings"] = physical.token_embeddings
        if self.vision_latents_raw is not None:
            vision_latents = _require_tensor(
                "vision_latents_raw", self.vision_latents_raw
            )
            _require_shape(
                "vision_latents_raw",
                vision_latents,
                (batch_size, self.vision_tokens, self.vision_latent_dim),
                f"[B, {self.vision_tokens}, D] with D={self.vision_latent_dim}",
            )
            _require_floating("vision_latents_raw", vision_latents)
            tensors["vision_latents_raw"] = vision_latents
        if self.vision_token_ids is not None:
            vision_token_ids = _require_tensor(
                "vision_token_ids", self.vision_token_ids
            )
            _require_shape(
                "vision_token_ids",
                vision_token_ids,
                (batch_size, self.vision_tokens),
                f"[B, {self.vision_tokens}]",
            )
            _require_integer("vision_token_ids", vision_token_ids)
            if vision_token_ids.numel() and bool((vision_token_ids < 0).any()):
                raise ValueError("vision_token_ids must be non-negative")
            tensors["vision_token_ids"] = vision_token_ids
        if self.vision_latents_raw is not None and self.vision_token_ids is not None:
            raise ValueError(
                "vision_latents_raw and vision_token_ids are mutually exclusive"
            )
        if self.semantic_vision_latents_raw is not None:
            semantic_dim = self.semantic_vision_latent_dim
            if type(semantic_dim) is not int or semantic_dim <= 0:
                raise ValueError(
                    "semantic_vision_latent_dim must be a positive integer"
                )
            semantic = _require_tensor(
                "semantic_vision_latents_raw", self.semantic_vision_latents_raw
            )
            _require_shape(
                "semantic_vision_latents_raw",
                semantic,
                (batch_size, self.vision_tokens, semantic_dim),
                f"[B, {self.vision_tokens}, {semantic_dim}]",
            )
            _require_floating("semantic_vision_latents_raw", semantic)
            tensors["semantic_vision_latents_raw"] = semantic
        elif self.semantic_vision_latent_dim is not None:
            raise ValueError(
                "semantic_vision_latent_dim requires semantic_vision_latents_raw"
            )
        if self.text_latents_raw is not None:
            text_latents = _require_tensor("text_latents_raw", self.text_latents_raw)
            _require_shape(
                "text_latents_raw",
                text_latents,
                (batch_size, text_tokens, geometry.text_latent_dim),
                f"[B, {text_tokens}, {geometry.text_latent_dim}]",
            )
            _require_floating("text_latents_raw", text_latents)
            tensors["text_latents_raw"] = text_latents
        if token_ids is not None:
            tensors["text_token_ids"] = token_ids

        _require_bool("text_content_mask", content_mask)
        tensors["text_content_mask"] = content_mask

        if self.text_latent_stats_type is not None:
            stats_type = _require_tensor(
                "text_latent_stats_type", self.text_latent_stats_type
            )
            _require_shape(
                "text_latent_stats_type",
                stats_type,
                (batch_size, text_tokens),
                f"[B, {text_tokens}]",
            )
            _require_integer("text_latent_stats_type", stats_type)
            tensors["text_latent_stats_type"] = stats_type

        prompt_mask = self.text_prompt_content_mask
        prompt_tokens: int | None = None
        if prompt_mask is not None:
            mask = _require_tensor("text_prompt_content_mask", prompt_mask)
            _validate_prompt_mask(task_type, mask)
            prompt_tokens = mask.shape[1]
            tensors["text_prompt_content_mask"] = mask
        if self.text_prompt_token_ids is not None:
            if prompt_tokens is None:
                raise ValueError(
                    "text_prompt_token_ids requires text_prompt_content_mask"
                )
            prompt_ids = _require_tensor(
                "text_prompt_token_ids",
                self.text_prompt_token_ids,
            )
            _require_shape(
                "text_prompt_token_ids",
                prompt_ids,
                (batch_size, prompt_tokens),
                f"[B, {prompt_tokens}]",
            )
            _require_integer("text_prompt_token_ids", prompt_ids)
            tensors["text_prompt_token_ids"] = prompt_ids
        if self.text_prompt_latents_raw is not None:
            if prompt_tokens is None:
                raise ValueError(
                    "text_prompt_latents_raw requires text_prompt_content_mask"
                )
            prompt_latents = _require_tensor(
                "text_prompt_latents_raw",
                self.text_prompt_latents_raw,
            )
            _require_shape(
                "text_prompt_latents_raw",
                prompt_latents,
                (batch_size, prompt_tokens, geometry.text_latent_dim),
                f"[B, {prompt_tokens}, {geometry.text_latent_dim}]",
            )
            _require_floating("text_prompt_latents_raw", prompt_latents)
            tensors["text_prompt_latents_raw"] = prompt_latents
        if self.text_prompt_latent_stats_type is not None:
            if prompt_tokens is None:
                raise ValueError(
                    "text_prompt_latent_stats_type requires text_prompt_content_mask"
                )
            prompt_stats = _require_tensor(
                "text_prompt_latent_stats_type",
                self.text_prompt_latent_stats_type,
            )
            _require_shape(
                "text_prompt_latent_stats_type",
                prompt_stats,
                (batch_size, prompt_tokens),
                f"[B, {prompt_tokens}]",
            )
            _require_integer("text_prompt_latent_stats_type", prompt_stats)
            tensors["text_prompt_latent_stats_type"] = prompt_stats

        if self.chunk_routing_cpu is not None:
            routing = self.chunk_routing_cpu.validate()
            if routing.vision_role.shape[0] != batch_size:
                raise ValueError("chunk routing batch dimension must match task_type")
            if routing.text_content_mask.shape != content_mask.shape:
                raise ValueError(
                    "chunk routing text_content_mask must match encoded metadata"
                )
            expected_prompt_shape = (batch_size, prompt_tokens or 0)
            if routing.text_prompt_content_mask.shape != expected_prompt_shape:
                raise ValueError(
                    "chunk routing text_prompt_content_mask must match encoded metadata"
                )
            if routing.geometry is not None and routing.geometry != geometry:
                raise ValueError("chunk routing geometry must match encoded geometry")
            if routing.compiled_sequences is not None:
                if self.compiled_sequences is None:
                    raise ValueError(
                        "encoded batch must carry compiled sequences with chunk routing"
                    )
                if tuple(
                    item.routing_signature() for item in routing.compiled_sequences
                ) != tuple(
                    item.routing_signature() for item in self.compiled_sequences
                ):
                    raise ValueError(
                        "chunk routing compiled sequences must match encoded metadata"
                    )

        _require_same_device("task_type", task_type, **tensors)
        return self

    def validate(self) -> Self:
        self.validate_metadata()
        if self.physical_layout is not None:
            if bool(self.vision_present.any()) or bool(self.text_present.any()):
                raise ValueError(
                    "physical sequence batches must not declare legacy branches"
                )
            if any(
                value is not None
                for value in (
                    self.vision_latents_raw,
                    self.vision_token_ids,
                    self.semantic_vision_latents_raw,
                    self.text_latents_raw,
                    self.text_prompt_latents_raw,
                )
            ):
                raise ValueError("physical sequence batches cannot contain legacy latents")
            return self
        _, vision_present, text_present, _ = _validate_routing(
            self.task_type,
            self.vision_present,
            self.text_present,
        )
        has_vision = bool(vision_present.any())
        has_text = bool(text_present.any())
        if (
            has_vision
            and self.vision_latents_raw is None
            and self.vision_token_ids is None
        ):
            raise ValueError(
                "vision_latents_raw or vision_token_ids is required when vision is present"
            )
        _require_present_branch("text_latents_raw", self.text_latents_raw, has_text)
        _require_present_branch("text_token_ids", self.text_token_ids, has_text)
        _require_present_branch(
            "text_latent_stats_type", self.text_latent_stats_type, has_text
        )

        has_prompt = bool(_image_to_text_rows(self.task_type).any())
        _require_present_branch(
            "text_prompt_content_mask",
            self.text_prompt_content_mask,
            has_prompt,
        )
        _require_present_branch(
            "text_prompt_latents_raw",
            self.text_prompt_latents_raw,
            has_prompt,
        )
        _require_present_branch(
            "text_prompt_latent_stats_type",
            self.text_prompt_latent_stats_type,
            has_prompt,
        )

        content_mask = _require_tensor("text_content_mask", self.text_content_mask)
        _validate_absent_text_mask(content_mask, text_present)
        return self
