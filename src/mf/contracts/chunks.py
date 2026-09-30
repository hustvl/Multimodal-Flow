from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

from mf._compat import Self
from mf.contracts.sequence import (
    MODALITY_REGISTRY,
    CompiledSequence,
    MultimodalSequence,
    SequenceChunk,
)
from mf.contracts.geometry import GeometryContract
from mf.contracts.task_registry import BranchRole


class ChunkModality(IntEnum):
    """Stable modality ids shared with packed backbone routing."""

    IMAGE = 0
    TEXT = 1


class ChunkTokenView(IntEnum):
    """Compiler-only views of one semantic chunk."""

    CONTENT = 0
    NOISY_TEXT = 1


def build_text_segment_ids(
    token_ids: Tensor,
    content_mask: Tensor,
    *,
    eos_token_id: int,
) -> Tensor:
    """Assign packed records to segments while keeping terminal EOS runs together."""

    if token_ids.shape != content_mask.shape:
        raise ValueError(
            "text token ids and content mask must have matching [B, T] shape"
        )
    if token_ids.ndim != 2 or content_mask.dtype is not torch.bool:
        raise ValueError("text token ids and content mask must have shape [B, T]")
    starts = torch.zeros_like(content_mask)
    starts[:, 0] = content_mask[:, 0]
    starts[:, 1:] = (
        content_mask[:, 1:]
        & (token_ids[:, 1:] != eos_token_id)
        & (token_ids[:, :-1] == eos_token_id)
    )
    segment_ids = starts.to(torch.long).cumsum(dim=1) - 1
    return torch.where(
        content_mask,
        segment_ids,
        torch.full_like(segment_ids, -1),
    )


@dataclass(frozen=True)
class ChunkRoutingMetadata:
    """CPU-resident semantic metadata used to compile one chunk-causal route."""

    vision_role: Tensor
    text_role: Tensor
    text_content_mask: Tensor
    text_prompt_content_mask: Tensor
    text_segment_ids: Tensor
    sequence_contracts: tuple[MultimodalSequence, ...] | None = None
    compiled_sequences: tuple[CompiledSequence, ...] | None = None
    geometry: GeometryContract | None = None
    null_conditioning: Tensor | None = None

    def validate(self) -> Self:
        tensors = {
            "vision_role": self.vision_role,
            "text_role": self.text_role,
            "text_content_mask": self.text_content_mask,
            "text_prompt_content_mask": self.text_prompt_content_mask,
            "text_segment_ids": self.text_segment_ids,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, Tensor):
                raise ValueError(f"{name} must be a torch.Tensor")
            if tensor.device.type != "cpu":
                raise ValueError(f"{name} must remain on CPU")
        if self.vision_role.ndim != 1 or self.text_role.shape != self.vision_role.shape:
            raise ValueError("chunk routing roles must have matching shape [B]")
        if self.vision_role.dtype != torch.long or self.text_role.dtype != torch.long:
            raise ValueError("chunk routing roles must use int64")
        batch_size = self.vision_role.shape[0]
        if (
            self.text_content_mask.ndim != 2
            or self.text_content_mask.shape[0] != batch_size
            or self.text_content_mask.dtype is not torch.bool
        ):
            raise ValueError("chunk routing text_content_mask must be bool [B, T]")
        if (
            self.text_prompt_content_mask.ndim != 2
            or self.text_prompt_content_mask.shape[0] != batch_size
            or self.text_prompt_content_mask.dtype is not torch.bool
        ):
            raise ValueError(
                "chunk routing text_prompt_content_mask must be bool [B, P]"
            )
        if (
            self.text_segment_ids.shape != self.text_content_mask.shape
            or self.text_segment_ids.dtype != torch.long
        ):
            raise ValueError("chunk routing text_segment_ids must be int64 [B, T]")
        if bool((self.text_content_mask & self.text_segment_ids.lt(0)).any()):
            raise ValueError(
                "active chunk routing text tokens require non-negative segment ids"
            )
        if bool((~self.text_content_mask & self.text_segment_ids.ne(-1)).any()):
            raise ValueError("inactive chunk routing text tokens require segment id -1")
        if self.sequence_contracts is not None:
            if not isinstance(self.sequence_contracts, tuple):
                raise ValueError("sequence_contracts must be a tuple when provided")
            for sequence in self.sequence_contracts:
                if not isinstance(sequence, MultimodalSequence):
                    raise ValueError("sequence_contracts must contain multimodal sequences")
                sequence.validate()
            compiled = tuple(sequence.compile() for sequence in self.sequence_contracts)
            if self.compiled_sequences is None:
                raise ValueError(
                    "chunk routing must carry compiled_sequences with sequence_contracts"
                )
            if len(self.compiled_sequences) != len(self.sequence_contracts):
                raise ValueError(
                    "compiled_sequences must match sequence_contracts length"
                )
            for sequence, expected in zip(
                self.compiled_sequences, compiled, strict=True
            ):
                if sequence.routing_signature() != expected.routing_signature():
                    raise ValueError(
                        "chunk routing compiled_sequences must derive from sequence_contracts"
                    )
        elif self.compiled_sequences is not None:
            if not isinstance(self.compiled_sequences, tuple):
                raise ValueError("compiled_sequences must be a tuple when provided")
            for sequence in self.compiled_sequences:
                if not isinstance(sequence, CompiledSequence):
                    raise ValueError(
                        "compiled_sequences must contain CompiledSequence values"
                    )
                sequence.validate()
        if self.geometry is not None:
            self.geometry.validate()
        if self.null_conditioning is not None:
            if self.null_conditioning.shape != self.vision_role.shape:
                raise ValueError("null_conditioning must have shape [B]")
            if self.null_conditioning.dtype is not torch.bool:
                raise ValueError("null_conditioning must have dtype torch.bool")
            if self.null_conditioning.device.type != "cpu":
                raise ValueError("null_conditioning must remain on CPU")
        return self


@dataclass(frozen=True)
class ChunkDescriptor:
    """Identity and semantic metadata for one sequence chunk.

    The compiler still accepts the two paper modalities as ``ChunkModality``.
    Generic adapters may carry a registered integer modality id together with
    ``modality_name`` until their physical token compiler is invoked.
    """

    sequence_id: int
    chunk_index: int
    modality: ChunkModality | int
    speaker: str | None = None
    modality_name: str | None = None
    role: BranchRole = BranchRole.CONDITION
    frame_index: int | None = None
    timestamp: float | None = None
    source_id: str | None = None

    @classmethod
    def from_sequence_chunk(cls, chunk: SequenceChunk) -> "ChunkDescriptor":
        chunk.validate()
        definition = MODALITY_REGISTRY.resolve(chunk.modality)
        return cls(
            sequence_id=chunk.sequence_id,
            chunk_index=chunk.chunk_index,
            modality=definition.stable_id,
            modality_name=definition.name,
            role=chunk.role,
            frame_index=chunk.frame_index,
            timestamp=chunk.timestamp,
            source_id=chunk.source_id,
        )

    def validate(self) -> Self:
        for name, value in (
            ("sequence_id", self.sequence_id),
            ("chunk_index", self.chunk_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(self.modality, bool) or not isinstance(self.modality, int):
            raise ValueError("modality must be a non-negative modality id")
        if self.modality < 0:
            raise ValueError("modality must be a non-negative modality id")
        if not isinstance(self.role, BranchRole) or self.role is BranchRole.ABSENT:
            raise ValueError("chunk role must be CONDITION or TARGET")
        if self.modality_name is not None and (
            not isinstance(self.modality_name, str) or not self.modality_name.strip()
        ):
            raise ValueError("modality_name must be a non-empty string when provided")
        if self.frame_index is not None and (
            type(self.frame_index) is not int or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer when provided")
        if self.timestamp is not None and not isinstance(self.timestamp, (int, float)):
            raise ValueError("timestamp must be numeric when provided")
        if self.source_id is not None and (
            not isinstance(self.source_id, str) or not self.source_id.strip()
        ):
            raise ValueError("source_id must be a non-empty string when provided")
        if self.speaker is not None and (
            not isinstance(self.speaker, str) or not self.speaker.strip()
        ):
            raise ValueError("speaker must be a non-empty string when provided")
        return self


def target_chunk_index(chunks: tuple[ChunkDescriptor, ...]) -> int:
    """Return the externally guided sole target: the final ordered chunk."""

    if not chunks:
        raise ValueError("an chunk sequence must contain at least one chunk")
    checked = tuple(chunk.validate() for chunk in chunks)
    if len({chunk.sequence_id for chunk in checked}) != 1:
        raise ValueError("target_chunk_index requires one semantic sequence")
    indices = tuple(chunk.chunk_index for chunk in checked)
    if indices != tuple(range(len(indices))):
        raise ValueError("chunk_index values must be contiguous and ordered from zero")
    return indices[-1]
