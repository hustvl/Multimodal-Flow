"""Generic multimodal sequence and modality extension contracts.

MF operates on ordered chunks, not on a fixed ``image``/``text`` pair. The
paper recipe is one useful default sequence; this module keeps the transport
contract general enough for video, editing, audio, depth, and future codecs.
The model-specific compiler is responsible for turning a validated sequence
into physical tokens.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from mf.contracts.task_registry import BranchRole, canonical_modality_name


PositionBuilder = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class ModalityDefinition:
    """Runtime capabilities needed to route one named modality."""

    name: str
    stable_id: int
    codec_name: str
    token_count: int | None = None
    latent_dim: int | None = None
    head_name: str = "linear"
    supports_temporal: bool = False
    supports_condition: bool = True
    supports_target: bool = True
    position_builder: PositionBuilder | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", canonical_modality_name(self.name))
        if not self.name or self.name.strip() != self.name:
            raise ValueError("modality name must be a non-empty trimmed string")
        if type(self.stable_id) is not int or self.stable_id < 0:
            raise ValueError("modality stable_id must be a non-negative integer")
        if not self.codec_name or self.codec_name.strip() != self.codec_name:
            raise ValueError("modality codec_name must be non-empty and trimmed")
        if not isinstance(self.head_name, str) or not self.head_name.strip():
            raise ValueError("modality head_name must be non-empty")
        for name, value in (("token_count", self.token_count), ("latent_dim", self.latent_dim)):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"modality {name} must be positive when provided")


class ModalityRegistry:
    """Name and stable-id registry used by sequence planners and adapters."""

    def __init__(self, definitions: Iterable[ModalityDefinition] = ()) -> None:
        self._by_name: dict[str, ModalityDefinition] = {}
        self._by_id: dict[int, ModalityDefinition] = {}
        self._frozen = False
        for definition in definitions:
            self.register(definition)

    def register(
        self, definition: ModalityDefinition, *, replace: bool = False
    ) -> ModalityDefinition:
        if self._frozen:
            raise RuntimeError(
                "modality registry is frozen; register extensions before "
                "constructing the model and data pipeline"
            )
        if not isinstance(definition, ModalityDefinition):
            raise TypeError("definition must be a ModalityDefinition")
        by_name = self._by_name.get(definition.name)
        by_id = self._by_id.get(definition.stable_id)
        if not replace and (by_name is not None or by_id is not None):
            raise ValueError(f"modality name or stable_id already registered: {definition.name!r}")
        if replace:
            if by_name is not None:
                self._by_id.pop(by_name.stable_id, None)
            if by_id is not None:
                self._by_name.pop(by_id.name, None)
        self._by_name[definition.name] = definition
        self._by_id[definition.stable_id] = definition
        return definition

    def freeze(self) -> None:
        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def resolve(self, modality: str | int) -> ModalityDefinition:
        definition = (
            self._by_name.get(canonical_modality_name(modality))
            if isinstance(modality, str)
            else self._by_id.get(modality)
        )
        if definition is None:
            known = ", ".join(sorted(self._by_name)) or "<none>"
            raise ValueError(f"unknown modality {modality!r}; available: {known}")
        return definition

    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def ids(self) -> tuple[int, ...]:
        """Return stable ids in deterministic order for physical model wiring."""

        return tuple(sorted(self._by_id))

    def definitions(self) -> tuple[ModalityDefinition, ...]:
        """Return registered modality contracts in stable-id order."""

        return tuple(self._by_id[modality_id] for modality_id in self.ids())

    def manifest(self) -> tuple[dict[str, object], ...]:
        """Return stable semantic metadata for checkpoint fingerprints."""

        return tuple(
            {
                "name": definition.name,
                "stable_id": definition.stable_id,
                "codec_name": definition.codec_name,
                "token_count": definition.token_count,
                "latent_dim": definition.latent_dim,
                "head_name": definition.head_name,
                "supports_temporal": definition.supports_temporal,
                "supports_condition": definition.supports_condition,
                "supports_target": definition.supports_target,
            }
            for definition in self.definitions()
        )

MODALITY_REGISTRY = ModalityRegistry(
    (
        # Token counts are adapter capabilities, not semantic sequence
        # invariants. A codec may expose a fixed count, but a sequence can
        # legally carry a variable budget until that adapter compiles it.
        ModalityDefinition("image", 0, "vision", latent_dim=768, supports_temporal=True),
        ModalityDefinition("text", 1, "text", latent_dim=512),
    )
)


@dataclass(frozen=True, slots=True)
class SequenceChunk:
    """One semantic chunk before tokenization or physical packing."""

    sequence_id: int
    chunk_index: int
    modality: str
    role: BranchRole
    token_count: int
    frame_index: int | None = None
    timestamp: float | None = None
    source_id: str | None = None
    output_slot: str | None = None
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def validate(self, *, modalities: ModalityRegistry | None = None) -> "SequenceChunk":
        if type(self.sequence_id) is not int or self.sequence_id < 0:
            raise ValueError("sequence_id must be a non-negative integer")
        if type(self.chunk_index) is not int or self.chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative integer")
        if not self.modality or self.modality.strip() != self.modality:
            raise ValueError("chunk modality must be a non-empty trimmed string")
        if not isinstance(self.role, BranchRole) or self.role is BranchRole.ABSENT:
            raise ValueError("sequence chunks must be CONDITION or TARGET")
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("chunk token_count must be a positive integer")
        if self.frame_index is not None and (
            type(self.frame_index) is not int or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer when provided")
        if self.timestamp is not None and not isfinite(self.timestamp):
            raise ValueError("timestamp must be finite when provided")
        for name, value in (("source_id", self.source_id), ("output_slot", self.output_slot)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when provided")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        if modalities is not None:
            definition = modalities.resolve(self.modality)
            if self.role is BranchRole.CONDITION and not definition.supports_condition:
                raise ValueError(f"modality {self.modality!r} cannot be a condition")
            if self.role is BranchRole.TARGET and not definition.supports_target:
                raise ValueError(f"modality {self.modality!r} cannot be a target")
            if self.frame_index is not None and not definition.supports_temporal:
                raise ValueError(f"modality {self.modality!r} does not support temporal metadata")
            if definition.token_count is not None and self.token_count != definition.token_count:
                raise ValueError(
                    f"{self.modality!r} chunks require token_count={definition.token_count}; got {self.token_count}"
                )
        return self


@dataclass(frozen=True, slots=True)
class SequenceRoutingItem:
    """Compiler-neutral route for one semantic chunk."""

    chunk_index: int
    modality_id: int
    modality: str
    role: BranchRole
    token_count: int
    frame_index: int | None
    timestamp: float | None
    source_id: str | None
    output_slot: str | None


@dataclass(frozen=True, slots=True)
class CompiledChunk:
    """Compiler output for one semantic chunk."""

    chunk_index: int
    modality_id: int
    modality: str
    role: BranchRole
    token_count: int
    frame_index: int | None
    timestamp: float | None
    source_id: str | None
    output_slot: str | None
    target_mask: bool
    decoder_target: str | None

    def validate(self) -> "CompiledChunk":
        if type(self.chunk_index) is not int or self.chunk_index < 0:
            raise ValueError("compiled chunk_index must be non-negative")
        if type(self.modality_id) is not int or self.modality_id < 0:
            raise ValueError("compiled modality_id must be non-negative")
        if not self.modality:
            raise ValueError("compiled modality must be non-empty")
        if self.role not in {BranchRole.CONDITION, BranchRole.TARGET}:
            raise ValueError("compiled chunks must be CONDITION or TARGET")
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("compiled token_count must be positive")
        if self.target_mask != (self.role is BranchRole.TARGET):
            raise ValueError("compiled target_mask must be derived from role")
        if self.decoder_target is not None and not self.target_mask:
            raise ValueError("decoder targets are valid only for target chunks")
        return self


@dataclass(frozen=True, slots=True)
class PhysicalTokenSpan:
    """Physical token span owned by a compiled semantic chunk."""

    chunk_index: int
    start: int
    end: int
    view: str
    output_slot: str | None = None

    def validate(self) -> "PhysicalTokenSpan":
        if type(self.chunk_index) is not int or self.chunk_index < 0:
            raise ValueError("physical span chunk_index must be non-negative")
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("physical span offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("physical span must have positive length")
        if self.view not in {"content", "noisy_text"}:
            raise ValueError("unknown physical span view")
        return self


@dataclass(frozen=True, slots=True)
class CompiledSequence:
    """Single source of truth for semantic and physical sequence semantics."""

    sequence_id: int
    chunks: tuple[CompiledChunk, ...]
    physical_spans: tuple[PhysicalTokenSpan, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunks", tuple(self.chunks))
        object.__setattr__(self, "physical_spans", tuple(self.physical_spans))

    @classmethod
    def from_sequence(
        cls,
        sequence: "MultimodalSequence",
        *,
        modalities: ModalityRegistry = MODALITY_REGISTRY,
    ) -> "CompiledSequence":
        sequence.validate(modalities=modalities)
        chunks = tuple(
            CompiledChunk(
                chunk_index=chunk.chunk_index,
                modality_id=modalities.resolve(chunk.modality).stable_id,
                modality=chunk.modality,
                role=chunk.role,
                token_count=chunk.token_count,
                frame_index=chunk.frame_index,
                timestamp=chunk.timestamp,
                source_id=chunk.source_id,
                output_slot=chunk.output_slot,
                target_mask=chunk.role is BranchRole.TARGET,
                decoder_target=(
                    "text"
                    if chunk.role is BranchRole.TARGET and chunk.modality == "text"
                    else None
                ),
            ).validate()
            for chunk in sequence.chunks
        )
        return cls(sequence.sequence_id, chunks).validate()

    def validate(self) -> "CompiledSequence":
        if type(self.sequence_id) is not int or self.sequence_id < 0:
            raise ValueError("compiled sequence_id must be non-negative")
        if not self.chunks:
            raise ValueError("compiled sequence must contain chunks")
        indices = tuple(chunk.chunk_index for chunk in self.chunks)
        if indices != tuple(range(len(self.chunks))):
            raise ValueError("compiled chunk indices must be contiguous and ordered")
        for chunk in self.chunks:
            chunk.validate()
        target_indices = tuple(
            chunk.chunk_index for chunk in self.chunks if chunk.target_mask
        )
        if not target_indices:
            raise ValueError("compiled sequence must contain a target")
        if target_indices != tuple(range(target_indices[0], len(self.chunks))):
            raise ValueError("compiled targets must form the final ordered suffix")
        for span in self.physical_spans:
            span.validate()
            if span.chunk_index not in indices:
                raise ValueError("physical span references an unknown chunk")
        if self.physical_spans:
            spans = tuple(sorted(self.physical_spans, key=lambda span: span.chunk_index))
            if tuple(span.chunk_index for span in spans) != indices:
                raise ValueError(
                    "physical spans must contain exactly one span per chunk"
                )
            offset = 0
            for span, chunk in zip(spans, self.chunks, strict=True):
                if span.start != offset or span.end - span.start != chunk.token_count:
                    raise ValueError(
                        "physical spans must cover ordered chunk tokens contiguously"
                    )
                offset = span.end
        return self

    @property
    def target_chunks(self) -> tuple[CompiledChunk, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.target_mask)

    @property
    def condition_chunks(self) -> tuple[CompiledChunk, ...]:
        return tuple(chunk for chunk in self.chunks if not chunk.target_mask)

    def role_for(self, modality: str) -> BranchRole:
        roles = {chunk.role for chunk in self.chunks if chunk.modality == modality}
        if not roles:
            return BranchRole.ABSENT
        if len(roles) != 1:
            raise ValueError(
                f"modality {modality!r} has multiple roles; use chunk-level routing"
            )
        return next(iter(roles))

    def primary_role(self, modality: str) -> BranchRole:
        """Return the role of a modality's model target, if it has one.

        Prompt chunks and target chunks may share a codec modality. In that
        case target is the branch role exposed to the paper model, while the
        chunk-level roles remain available to the physical compiler.
        """

        roles = {chunk.role for chunk in self.chunks if chunk.modality == modality}
        if not roles:
            return BranchRole.ABSENT
        if BranchRole.TARGET in roles:
            return BranchRole.TARGET
        return BranchRole.CONDITION

    def routing_signature(self) -> tuple[tuple[object, ...], ...]:
        self.validate()
        return tuple(
            (
                chunk.chunk_index,
                chunk.modality_id,
                chunk.modality,
                int(chunk.role),
                chunk.token_count,
                chunk.frame_index,
                chunk.timestamp,
                chunk.source_id,
                chunk.output_slot,
            )
            for chunk in self.chunks
        )

    def with_physical_spans(
        self, spans: Iterable[PhysicalTokenSpan]
    ) -> "CompiledSequence":
        spans = tuple(spans)
        expected = {chunk.chunk_index for chunk in self.chunks}
        actual = {span.chunk_index for span in spans}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "physical compiler must emit spans for every semantic chunk; "
                f"missing={missing}, unknown={extra}"
            )
        return CompiledSequence(
            self.sequence_id,
            self.chunks,
            spans,
        ).validate()


@dataclass(frozen=True, slots=True)
class MultimodalSequence:
    """Ordered semantic sequence consumed by a task/layout adapter."""

    sequence_id: int
    chunks: tuple[SequenceChunk, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunks", tuple(self.chunks))

    def validate(
        self, *, modalities: ModalityRegistry | None = MODALITY_REGISTRY
    ) -> "MultimodalSequence":
        if type(self.sequence_id) is not int or self.sequence_id < 0:
            raise ValueError("sequence_id must be a non-negative integer")
        if not self.chunks:
            raise ValueError("a multimodal sequence must contain at least one chunk")
        for chunk in self.chunks:
            chunk.validate(modalities=modalities)
            if chunk.sequence_id != self.sequence_id:
                raise ValueError("all chunks must belong to the sequence_id")
        indices = tuple(chunk.chunk_index for chunk in self.chunks)
        if indices != tuple(range(len(self.chunks))):
            raise ValueError("chunk_index values must be contiguous and ordered from zero")
        target_indices = tuple(
            chunk.chunk_index for chunk in self.chunks if chunk.role is BranchRole.TARGET
        )
        if not target_indices:
            raise ValueError("a multimodal sequence must contain a target chunk")
        if target_indices != tuple(range(target_indices[0], len(self.chunks))):
            raise ValueError("target chunks must form the final ordered suffix")
        last_frame = -1
        last_time = float("-inf")
        for chunk in self.chunks:
            if chunk.frame_index is not None:
                if chunk.frame_index < last_frame:
                    raise ValueError("frame_index must be non-decreasing")
                last_frame = chunk.frame_index
            if chunk.timestamp is not None:
                if chunk.timestamp < last_time:
                    raise ValueError("timestamp must be non-decreasing")
                last_time = chunk.timestamp
        return self

    @property
    def target_chunks(self) -> tuple[SequenceChunk, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.role is BranchRole.TARGET)

    @property
    def condition_chunks(self) -> tuple[SequenceChunk, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.role is BranchRole.CONDITION)

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(chunk.modality for chunk in self.chunks))

    def routing_signature(self) -> tuple[tuple[object, ...], ...]:
        self.validate()
        return tuple(
            (
                chunk.chunk_index,
                chunk.modality,
                int(chunk.role),
                chunk.token_count,
                chunk.frame_index,
                chunk.timestamp,
                chunk.source_id,
                chunk.output_slot,
            )
            for chunk in self.chunks
        )

    def routing(
        self,
        *,
        modalities: ModalityRegistry = MODALITY_REGISTRY,
    ) -> tuple[SequenceRoutingItem, ...]:
        """Compile semantic chunks to stable modality ids without token packing."""

        self.validate(modalities=modalities)
        return tuple(
            SequenceRoutingItem(
                chunk_index=chunk.chunk_index,
                modality_id=modalities.resolve(chunk.modality).stable_id,
                modality=chunk.modality,
                role=chunk.role,
                token_count=chunk.token_count,
                frame_index=chunk.frame_index,
                timestamp=chunk.timestamp,
                source_id=chunk.source_id,
                output_slot=chunk.output_slot,
            )
            for chunk in self.chunks
        )

    def compile(
        self,
        *,
        modalities: ModalityRegistry = MODALITY_REGISTRY,
    ) -> CompiledSequence:
        """Compile semantic roles once for all downstream physical consumers."""

        return CompiledSequence.from_sequence(self, modalities=modalities)

    @classmethod
    def video(
        cls,
        *,
        sequence_id: int,
        frame_count: int,
        token_count: int = 256,
        target_index: int | None = None,
        timestamps: Iterable[float] | None = None,
    ) -> "MultimodalSequence":
        if type(frame_count) is not int or frame_count <= 0:
            raise ValueError("frame_count must be a positive integer")
        target = frame_count - 1 if target_index is None else target_index
        if type(target) is not int or not 0 <= target < frame_count:
            raise ValueError("target_index must identify one video frame")
        times = None if timestamps is None else tuple(timestamps)
        if times is not None and len(times) != frame_count:
            raise ValueError("timestamps must contain one value per frame")
        chunks = tuple(
            SequenceChunk(
                sequence_id,
                index,
                "image",
                BranchRole.TARGET if index >= target else BranchRole.CONDITION,
                token_count,
                index,
                None if times is None else times[index],
                f"frame:{index}",
                "image" if index >= target else None,
            )
            for index in range(frame_count)
        )
        return cls(sequence_id, chunks).validate()

    @classmethod
    def image_edit(
        cls, *, sequence_id: int, source_token_count: int = 256, target_token_count: int = 256
    ) -> "MultimodalSequence":
        return cls(
            sequence_id,
            (
                SequenceChunk(
                    sequence_id,
                    0,
                    "image",
                    BranchRole.CONDITION,
                    source_token_count,
                    source_id="source",
                ),
                SequenceChunk(
                    sequence_id,
                    1,
                    "image",
                    BranchRole.TARGET,
                    target_token_count,
                    source_id="target",
                    output_slot="image",
                ),
            ),
        ).validate()


def register_modality(definition: ModalityDefinition, *, replace: bool = False) -> None:
    """Register a codec-backed modality extension before loading a config."""

    MODALITY_REGISTRY.register(definition, replace=replace)


__all__ = [
    "MODALITY_REGISTRY",
    "CompiledChunk",
    "CompiledSequence",
    "ModalityDefinition",
    "ModalityRegistry",
    "MultimodalSequence",
    "PhysicalTokenSpan",
    "SequenceChunk",
    "SequenceRoutingItem",
    "register_modality",
]
