from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from mf._compat import Self
from mf.contracts.chunks import ChunkTokenView
from mf.contracts.sequence import (
    MODALITY_REGISTRY,
    CompiledSequence,
    MultimodalSequence,
    PhysicalTokenSpan,
)
from mf.contracts.task_registry import BranchRole

_INTEGER_DTYPES = frozenset(
    {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)
_INDEX_DTYPES = frozenset({torch.int32, torch.int64})


@dataclass(frozen=True)
class EncodedChunk:
    """Codec output consumed by the shared physical sequence compiler."""

    modality: str
    codec_name: str
    embeddings: Tensor
    position_ids: Tensor

    def validate(self) -> "EncodedChunk":
        if not isinstance(self.modality, str) or not self.modality.strip():
            raise ValueError("encoded chunk modality must be non-empty")
        if not isinstance(self.codec_name, str) or not self.codec_name.strip():
            raise ValueError("encoded chunk codec_name must be non-empty")
        if self.embeddings.ndim != 2 or not self.embeddings.is_floating_point():
            raise ValueError("encoded chunk embeddings must have shape [tokens, hidden]")
        if self.position_ids.shape != (3, self.embeddings.shape[0]):
            raise ValueError("encoded chunk position_ids must have shape [3, tokens]")
        if self.position_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("encoded chunk position_ids must be integer")
        if self.position_ids.device != self.embeddings.device:
            raise ValueError("encoded chunk positions must share the embedding device")
        return self


@dataclass(frozen=True)
class PhysicalSequenceSample:
    """One already-encoded sequence emitted by a modality adapter.

    The adapter owns codec-specific encoding. MF only consumes the resulting
    hidden tokens and routing metadata, so adding a modality never requires
    teaching the core collator about that modality's payload format.
    """

    token_embeddings: Tensor
    position_ids: Tensor
    sequence_ids: Tensor
    chunk_indices: Tensor
    modality_ids: Tensor
    view_ids: Tensor
    block_indices: Tensor
    target_mask: Tensor
    target_latents: Mapping[int, Tensor] | None = None
    noisy_latents: Mapping[int, Tensor] | None = None
    target_timesteps: Mapping[int, Tensor] | None = None
    compiled_sequence: CompiledSequence | None = None

    @classmethod
    def from_codec_outputs(
        cls,
        sequence: MultimodalSequence,
        *,
        encoded_chunks: Mapping[int, EncodedChunk],
        target_latents_by_chunk: Mapping[int, Tensor] | None = None,
        noisy_latents_by_chunk: Mapping[int, Tensor] | None = None,
        target_timesteps_by_chunk: Mapping[int, Tensor] | None = None,
    ) -> "PhysicalSequenceSample":
        """Materialize one semantic sequence into the physical model contract.

        Codec adapters provide ``EncodedChunk`` values; MF owns ordering, role
        masks, modality ids, physical spans, and target grouping. This keeps
        video, editing, and new modalities on one compiler path.
        """

        sequence.validate()
        chunks = sequence.chunks
        chunk_indices = {chunk.chunk_index for chunk in chunks}
        if set(encoded_chunks) != chunk_indices:
            raise ValueError("encoded_chunks must cover every sequence chunk")

        target_chunks = tuple(sequence.target_chunks)
        target_indices = {chunk.chunk_index for chunk in target_chunks}
        target_maps = (
            target_latents_by_chunk,
            noisy_latents_by_chunk,
            target_timesteps_by_chunk,
        )
        if any(value is not None for value in target_maps) and not all(
            value is not None for value in target_maps
        ):
            raise ValueError(
                "target_latents_by_chunk, noisy_latents_by_chunk, and "
                "target_timesteps_by_chunk must be provided together"
            )
        if all(value is not None for value in target_maps):
            for name, values in zip(
                (
                    "target_latents_by_chunk",
                    "noisy_latents_by_chunk",
                    "target_timesteps_by_chunk",
                ),
                target_maps,
                strict=True,
            ):
                assert values is not None
                if set(values) != target_indices:
                    raise ValueError(f"{name} must cover every target chunk")

        first_output = next(iter(encoded_chunks.values())).validate()
        device = first_output.embeddings.device
        hidden_size = first_output.embeddings.shape[-1]
        embeddings: list[Tensor] = []
        positions: list[Tensor] = []
        sequence_ids: list[Tensor] = []
        chunk_ids: list[Tensor] = []
        modality_ids: list[Tensor] = []
        view_ids: list[Tensor] = []
        block_ids: list[Tensor] = []
        target_mask: list[Tensor] = []
        target_latents: dict[int, list[Tensor]] = {}
        noisy_latents: dict[int, list[Tensor]] = {}
        target_timesteps: dict[int, list[Tensor]] = {}
        spans = []
        offset = 0

        for chunk in chunks:
            encoded = encoded_chunks[chunk.chunk_index].validate()
            cls._validate_codec_binding(chunk.modality, encoded)
            embedding = encoded.embeddings
            position = encoded.position_ids
            if (
                embedding.ndim != 2
                or embedding.shape[0] != chunk.token_count
                or embedding.shape[1] != hidden_size
                or embedding.device != device
                or not embedding.is_floating_point()
            ):
                raise ValueError(
                    f"chunk {chunk.chunk_index} embeddings must have shape "
                    f"[{chunk.token_count}, {hidden_size}] on one device"
                )
            if (
                position.shape != (3, chunk.token_count)
                or position.device != device
                or position.dtype not in _INTEGER_DTYPES
            ):
                raise ValueError(
                    f"chunk {chunk.chunk_index} positions must have shape "
                    f"[3, {chunk.token_count}] and share the embedding device"
                )
            modality_id = MODALITY_REGISTRY.resolve(chunk.modality).stable_id
            embeddings.append(embedding)
            positions.append(position)
            sequence_ids.append(torch.full((chunk.token_count,), chunk.sequence_id, dtype=torch.long, device=device))
            chunk_ids.append(torch.full((chunk.token_count,), chunk.chunk_index, dtype=torch.long, device=device))
            modality_ids.append(torch.full((chunk.token_count,), modality_id, dtype=torch.long, device=device))
            view_ids.append(torch.full((chunk.token_count,), int(ChunkTokenView.CONTENT), dtype=torch.long, device=device))
            block_ids.append(torch.full((chunk.token_count,), chunk.chunk_index, dtype=torch.long, device=device))
            target_mask.append(torch.full((chunk.token_count,), chunk.role is BranchRole.TARGET, dtype=torch.bool, device=device))
            spans.append((chunk.chunk_index, offset, offset + chunk.token_count))
            offset += chunk.token_count

            if chunk.role is BranchRole.TARGET and all(value is not None for value in target_maps):
                assert target_latents_by_chunk is not None
                assert noisy_latents_by_chunk is not None
                assert target_timesteps_by_chunk is not None
                target = target_latents_by_chunk[chunk.chunk_index]
                noisy = noisy_latents_by_chunk[chunk.chunk_index]
                timestep = target_timesteps_by_chunk[chunk.chunk_index]
                for name, value in (("target", target), ("noisy", noisy)):
                    if (
                        value.ndim != 2
                        or value.shape[0] != chunk.token_count
                        or value.device != device
                        or not value.is_floating_point()
                    ):
                        raise ValueError(
                            f"chunk {chunk.chunk_index} {name} latents must align with tokens"
                        )
                if (
                    timestep.shape != (chunk.token_count,)
                    or timestep.device != device
                    or not timestep.is_floating_point()
                    or not bool(torch.isfinite(timestep).all())
                    or bool(((timestep < 0) | (timestep > 1)).any())
                ):
                    raise ValueError(
                        f"chunk {chunk.chunk_index} target timesteps must be finite in [0, 1]"
                    )
                target_latents.setdefault(modality_id, []).append(target)
                noisy_latents.setdefault(modality_id, []).append(noisy)
                target_timesteps.setdefault(modality_id, []).append(timestep)

        compiled = sequence.compile()
        compiled = compiled.with_physical_spans(
            PhysicalTokenSpan(index, start, end, "content")
            for index, start, end in spans
        )
        return cls(
            token_embeddings=torch.cat(embeddings, dim=0),
            position_ids=torch.cat(positions, dim=1),
            sequence_ids=torch.cat(sequence_ids),
            chunk_indices=torch.cat(chunk_ids),
            modality_ids=torch.cat(modality_ids),
            view_ids=torch.cat(view_ids),
            block_indices=torch.cat(block_ids),
            target_mask=torch.cat(target_mask),
            target_latents=(
                None
                if not target_latents
                else {key: torch.cat(values, dim=0) for key, values in target_latents.items()}
            ),
            noisy_latents=(
                None
                if not noisy_latents
                else {key: torch.cat(values, dim=0) for key, values in noisy_latents.items()}
            ),
            target_timesteps=(
                None
                if not target_timesteps
                else {key: torch.cat(values, dim=0) for key, values in target_timesteps.items()}
            ),
            compiled_sequence=compiled,
        ).validate()

    @classmethod
    def from_sequence(
        cls,
        sequence: MultimodalSequence,
        *,
        encoded_chunks: Mapping[int, EncodedChunk],
        target_latents_by_chunk: Mapping[int, Tensor] | None = None,
        noisy_latents_by_chunk: Mapping[int, Tensor] | None = None,
        target_timesteps_by_chunk: Mapping[int, Tensor] | None = None,
    ) -> "PhysicalSequenceSample":
        """Compatibility alias for :meth:`from_codec_outputs`."""

        return cls.from_codec_outputs(
            sequence,
            encoded_chunks=encoded_chunks,
            target_latents_by_chunk=target_latents_by_chunk,
            noisy_latents_by_chunk=noisy_latents_by_chunk,
            target_timesteps_by_chunk=target_timesteps_by_chunk,
        )

    @staticmethod
    def _validate_codec_binding(modality: str, encoded: EncodedChunk) -> None:
        from mf.codecs.registry import CODEC_REGISTRY

        definition = MODALITY_REGISTRY.resolve(modality)
        codec = CODEC_REGISTRY.resolve(encoded.codec_name)
        if definition.codec_name in {"vision", "text"}:
            if codec.modality != definition.codec_name:
                raise ValueError(
                    f"chunk modality {modality!r} expects codec family "
                    f"{definition.codec_name!r}, got {codec.modality!r}"
                )
            return
        if codec.modality != definition.name:
            raise ValueError(
                f"codec {encoded.codec_name!r} belongs to modality "
                f"{codec.modality!r}, not {definition.name!r}"
            )

    def validate(self) -> Self:
        if self.token_embeddings.ndim != 2:
            raise ValueError("token_embeddings must have shape [L, H]")
        token_count = self.token_embeddings.shape[0]
        if not self.token_embeddings.is_floating_point():
            raise ValueError("token_embeddings must be floating point")
        fields = {
            "sequence_ids": self.sequence_ids,
            "chunk_indices": self.chunk_indices,
            "modality_ids": self.modality_ids,
            "view_ids": self.view_ids,
            "block_indices": self.block_indices,
            "target_mask": self.target_mask,
        }
        for name, value in fields.items():
            if value.ndim != 1 or value.shape[0] != token_count:
                raise ValueError(f"{name} must have shape [L]")
            if value.device != self.token_embeddings.device:
                raise ValueError(f"{name} must share token_embeddings device")
        if self.position_ids.ndim != 2 or self.position_ids.shape[0] != 3:
            raise ValueError("position_ids must have shape [3, L]")
        if self.position_ids.shape[1] != token_count:
            raise ValueError("position_ids must have shape [3, L]")
        if self.target_mask.dtype is not torch.bool:
            raise ValueError("target_mask must be bool")
        if self.position_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("position_ids must have an integer dtype")
        for name in (
            "sequence_ids",
            "chunk_indices",
            "modality_ids",
            "view_ids",
            "block_indices",
        ):
            if fields[name].dtype not in _INTEGER_DTYPES:
                raise ValueError(f"{name} must have an integer dtype")
        if self.compiled_sequence is not None:
            self.compiled_sequence.validate()
        target_counts = self._target_counts()
        target_modalities = set(target_counts)
        for name, values in (
            ("target_latents", self.target_latents),
            ("noisy_latents", self.noisy_latents),
        ):
            if values is None:
                continue
            if not isinstance(values, Mapping):
                raise ValueError(f"{name} must be a mapping when provided")
            if set(values) != target_modalities:
                raise ValueError(
                    f"{name} keys must exactly match target modalities "
                    f"{sorted(target_modalities)}"
                )
            for modality_id, value in values.items():
                if type(modality_id) is not int or modality_id < 0:
                    raise ValueError(f"{name} keys must be non-negative integers")
                if value.ndim != 2 or value.shape[0] != target_counts.get(modality_id, 0):
                    raise ValueError(
                        f"{name}[{modality_id}] must have shape "
                        f"[{target_counts.get(modality_id, 0)}, D]"
                    )
                if value.device != self.token_embeddings.device or not value.is_floating_point():
                    raise ValueError(
                        f"{name}[{modality_id}] must share device and be floating point"
                    )
        if self.target_timesteps is not None:
            if not isinstance(self.target_timesteps, Mapping):
                raise ValueError("target_timesteps must be a mapping when provided")
            for modality_id, value in self.target_timesteps.items():
                if type(modality_id) is not int or modality_id < 0:
                    raise ValueError(
                        "target_timesteps keys must be non-negative integers"
                    )
                if value.shape != (target_counts.get(modality_id, 0),):
                    raise ValueError(
                        f"target_timesteps[{modality_id}] must align with target tokens"
                    )
                if value.device != self.token_embeddings.device or not value.is_floating_point():
                    raise ValueError(
                        "target timesteps must share device and be floating point"
                    )
                if not bool(torch.isfinite(value).all()) or bool(
                    ((value < 0) | (value > 1)).any()
                ):
                    raise ValueError("target timesteps must be finite and lie in [0, 1]")
        return self

    def _target_counts(self) -> dict[int, int]:
        active_targets = self.target_mask
        return {
            int(modality_id): int(
                (active_targets & (self.modality_ids == modality_id)).sum().item()
            )
            for modality_id in torch.unique(self.modality_ids[active_targets]).tolist()
        }


@dataclass(frozen=True)
class PhysicalSequenceLayout:
    """Pre-embedded physical sequence supplied by a registered codec adapter."""

    token_embeddings: Tensor
    active_token_mask: Tensor
    position_ids: Tensor
    sequence_ids: Tensor
    chunk_indices: Tensor
    modality_ids: Tensor
    view_ids: Tensor
    block_indices: Tensor
    target_mask: Tensor
    target_latents: Mapping[int, Tensor] | None = None
    noisy_latents: Mapping[int, Tensor] | None = None
    target_timesteps: Mapping[int, Tensor] | None = None
    ordered_active_indices: Tensor | None = None
    compiled_sequences: tuple[CompiledSequence, ...] | None = None
    layout_name: str = "chunk_causal"

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "PhysicalSequenceLayout":
        """Move the complete physical contract, including per-modality targets."""

        def move(value: Tensor) -> Tensor:
            return value.to(device=device, non_blocking=non_blocking)
        return PhysicalSequenceLayout(
            token_embeddings=move(self.token_embeddings),
            active_token_mask=move(self.active_token_mask),
            position_ids=move(self.position_ids),
            sequence_ids=move(self.sequence_ids),
            chunk_indices=move(self.chunk_indices),
            modality_ids=move(self.modality_ids),
            view_ids=move(self.view_ids),
            block_indices=move(self.block_indices),
            target_mask=move(self.target_mask),
            target_latents=(
                None
                if self.target_latents is None
                else {key: move(value) for key, value in self.target_latents.items()}
            ),
            noisy_latents=(
                None
                if self.noisy_latents is None
                else {key: move(value) for key, value in self.noisy_latents.items()}
            ),
            target_timesteps=(
                None
                if self.target_timesteps is None
                else {
                    key: move(value) for key, value in self.target_timesteps.items()
                }
            ),
            ordered_active_indices=(
                None
                if self.ordered_active_indices is None
                else move(self.ordered_active_indices)
            ),
            compiled_sequences=self.compiled_sequences,
            layout_name=self.layout_name,
        ).validate()

    def validate(self) -> Self:
        if not isinstance(self.layout_name, str) or not self.layout_name.strip():
            raise ValueError("layout_name must be a non-empty string")
        if self.token_embeddings.ndim != 3:
            raise ValueError("token_embeddings must have shape [B, L, H]")
        batch_size, sequence_length, _ = self.token_embeddings.shape
        if not self.token_embeddings.is_floating_point():
            raise ValueError("token_embeddings must be floating point")
        fields = {
            "active_token_mask": self.active_token_mask,
            "sequence_ids": self.sequence_ids,
            "chunk_indices": self.chunk_indices,
            "modality_ids": self.modality_ids,
            "view_ids": self.view_ids,
            "block_indices": self.block_indices,
            "target_mask": self.target_mask,
        }
        for name, value in fields.items():
            if value.shape != (batch_size, sequence_length):
                raise ValueError(f"{name} must have shape [B, L]")
            if value.device != self.token_embeddings.device:
                raise ValueError(f"{name} must share token_embeddings device")
        if self.active_token_mask.dtype is not torch.bool:
            raise ValueError("active_token_mask must be bool")
        if self.target_mask.dtype is not torch.bool:
            raise ValueError("target_mask must be bool")
        if self.target_latents is not None and not isinstance(
            self.target_latents, Mapping
        ):
            raise ValueError("target_latents must be a mapping when provided")
        if self.noisy_latents is not None and not isinstance(
            self.noisy_latents, Mapping
        ):
            raise ValueError("noisy_latents must be a mapping when provided")
        target_mask = self.active_token_mask & self.target_mask
        target_counts = {
            int(modality_id): int(
                (target_mask & (self.modality_ids == modality_id)).sum().item()
            )
            for modality_id in torch.unique(self.modality_ids[target_mask]).tolist()
        }
        target_modalities = set(target_counts)
        for name, values in (
            ("target_latents", self.target_latents),
            ("noisy_latents", self.noisy_latents),
        ):
            if values is None:
                continue
            if set(values) != target_modalities:
                raise ValueError(
                    f"{name} keys must exactly match target modalities "
                    f"{sorted(target_modalities)}"
                )
            for modality_id, value in values.items():
                if type(modality_id) is not int or modality_id < 0:
                    raise ValueError(f"{name} keys must be non-negative integers")
                if value.ndim != 2 or value.shape[0] != target_counts.get(modality_id, 0):
                    raise ValueError(f"{name}[{modality_id}] must align with target tokens")
                if value.device != self.token_embeddings.device or not value.is_floating_point():
                    raise ValueError(
                        f"{name}[{modality_id}] must share device and be floating point"
                    )
        if self.target_timesteps is not None:
            if not isinstance(self.target_timesteps, Mapping):
                raise ValueError("target_timesteps must be a mapping when provided")
            for modality_id, value in self.target_timesteps.items():
                if type(modality_id) is not int or modality_id < 0:
                    raise ValueError(
                        "target_timesteps keys must be non-negative integers"
                    )
                if value.shape != (target_counts.get(modality_id, 0),):
                    raise ValueError("target_timesteps must align with target tokens")
                if value.device != self.token_embeddings.device or not value.is_floating_point():
                    raise ValueError(
                        "target timesteps must share device and be floating point"
                    )
                if not bool(torch.isfinite(value).all()) or bool(
                    ((value < 0) | (value > 1)).any()
                ):
                    raise ValueError("target timesteps must be finite and lie in [0, 1]")
            if set(self.target_timesteps) != target_modalities:
                raise ValueError(
                    "target_timesteps keys must exactly match target modalities "
                    f"{sorted(target_modalities)}"
                )
        if self.position_ids.shape != (3, batch_size, sequence_length):
            raise ValueError("position_ids must have shape [3, B, L]")
        if self.position_ids.device != self.token_embeddings.device:
            raise ValueError("position_ids must share token_embeddings device")
        if self.position_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("position_ids must have an integer dtype")
        for name in (
            "sequence_ids",
            "chunk_indices",
            "modality_ids",
            "view_ids",
            "block_indices",
        ):
            if fields[name].dtype not in _INTEGER_DTYPES:
                raise ValueError(f"{name} must have an integer dtype")
        if bool((self.active_token_mask & (self.sequence_ids < 0)).any()):
            raise ValueError("active physical tokens require non-negative sequence ids")
        if bool((self.active_token_mask & (self.chunk_indices < 0)).any()):
            raise ValueError("active physical tokens require non-negative chunk indices")
        if bool((self.active_token_mask & (self.modality_ids < 0)).any()):
            raise ValueError("active physical tokens require non-negative modality ids")
        if self.ordered_active_indices is not None:
            indices = self.ordered_active_indices
            expected = torch.nonzero(
                self.active_token_mask.reshape(-1), as_tuple=False
            ).flatten()
            if indices.ndim != 1 or indices.dtype not in _INDEX_DTYPES:
                raise ValueError("ordered_active_indices must be an integer vector")
            if indices.device != self.token_embeddings.device:
                raise ValueError(
                    "ordered_active_indices must share token_embeddings device"
                )
            if indices.numel() != expected.numel() or not torch.equal(
                torch.sort(indices).values, expected
            ):
                raise ValueError(
                    "ordered_active_indices must partition active tokens exactly"
                )
        if self.compiled_sequences is not None:
            if len(self.compiled_sequences) != batch_size:
                raise ValueError("compiled_sequences must contain one sequence per row")
            for row, sequence in enumerate(self.compiled_sequences):
                sequence.validate()
                active = self.active_token_mask[row]
                expected_tokens = sum(
                    chunk.token_count for chunk in sequence.chunks
                )
                if int(active.sum().item()) != expected_tokens:
                    raise ValueError(
                        "compiled sequence token_count does not match physical active tokens"
                    )
                active_sequence_ids = self.sequence_ids[row][active]
                if active_sequence_ids.numel() and bool(
                    (active_sequence_ids != sequence.sequence_id).any()
                ):
                    raise ValueError(
                        "physical sequence ids do not match the compiled sequence"
                    )
                for chunk in sequence.chunks:
                    chunk_mask = active & (self.chunk_indices[row] == chunk.chunk_index)
                    if int(chunk_mask.sum().item()) != chunk.token_count:
                        raise ValueError(
                            "compiled chunk token_count does not match its physical span"
                        )
                    modality_id = MODALITY_REGISTRY.resolve(chunk.modality).stable_id
                    if bool(
                        (self.modality_ids[row][chunk_mask] != modality_id).any()
                    ):
                        raise ValueError(
                            "physical modality ids do not match the compiled sequence"
                        )
                    if bool(
                        (
                            self.target_mask[row][chunk_mask]
                            != (chunk.role is BranchRole.TARGET)
                        ).any()
                    ):
                        raise ValueError(
                            "physical target mask does not match the compiled sequence"
                        )
        return self


def collate_physical_sequences(
    samples: Sequence[PhysicalSequenceSample],
) -> PhysicalSequenceLayout:
    """Pad adapter-produced sequences into one validated physical batch."""

    if not samples:
        raise ValueError("at least one physical sequence is required")
    validated = tuple(sample.validate() for sample in samples)
    hidden_size = validated[0].token_embeddings.shape[-1]
    device = validated[0].token_embeddings.device
    if any(sample.token_embeddings.shape[-1] != hidden_size for sample in validated):
        raise ValueError("all physical sequences must share hidden size")
    if any(sample.token_embeddings.device != device for sample in validated):
        raise ValueError("all physical sequences must share device")
    batch_size = len(validated)
    max_tokens = max(sample.token_embeddings.shape[0] for sample in validated)
    token_embeddings = validated[0].token_embeddings.new_zeros(
        batch_size, max_tokens, hidden_size
    )
    active_token_mask = torch.zeros(
        batch_size, max_tokens, dtype=torch.bool, device=device
    )
    position_ids = torch.zeros(
        3,
        batch_size,
        max_tokens,
        dtype=validated[0].position_ids.dtype,
        device=device,
    )
    integer_fields = {
        name: torch.full(
            (batch_size, max_tokens),
            -1,
            dtype=getattr(validated[0], name).dtype,
            device=device,
        )
        for name in (
            "sequence_ids",
            "chunk_indices",
            "modality_ids",
            "view_ids",
            "block_indices",
        )
    }
    target_mask = torch.zeros(
        batch_size, max_tokens, dtype=torch.bool, device=device
    )
    per_modality_targets: dict[int, list[Tensor]] = {}
    per_modality_noisy: dict[int, list[Tensor]] = {}
    per_modality_timesteps: dict[int, list[Tensor]] = {}
    for row, sample in enumerate(validated):
        length = sample.token_embeddings.shape[0]
        token_embeddings[row, :length] = sample.token_embeddings
        active_token_mask[row, :length] = True
        position_ids[:, row, :length] = sample.position_ids
        for name, values in integer_fields.items():
            values[row, :length] = getattr(sample, name)
        target_mask[row, :length] = sample.target_mask
        for modality_id, values in (sample.target_latents or {}).items():
            per_modality_targets.setdefault(modality_id, []).append(values)
        for modality_id, values in (sample.noisy_latents or {}).items():
            per_modality_noisy.setdefault(modality_id, []).append(values)
        for modality_id, values in (sample.target_timesteps or {}).items():
            per_modality_timesteps.setdefault(modality_id, []).append(values)
    target_latents = {
        modality_id: torch.cat(values, dim=0)
        for modality_id, values in per_modality_targets.items()
    } or None
    noisy_latents = {
        modality_id: torch.cat(values, dim=0)
        for modality_id, values in per_modality_noisy.items()
    } or None
    target_timesteps = {
        modality_id: torch.cat(values, dim=0)
        for modality_id, values in per_modality_timesteps.items()
    } or None
    compiled = (
        tuple(sample.compiled_sequence for sample in validated)
        if all(sample.compiled_sequence is not None for sample in validated)
        else None
    )
    return PhysicalSequenceLayout(
        token_embeddings=token_embeddings,
        active_token_mask=active_token_mask,
        position_ids=position_ids,
        sequence_ids=integer_fields["sequence_ids"],
        chunk_indices=integer_fields["chunk_indices"],
        modality_ids=integer_fields["modality_ids"],
        view_ids=integer_fields["view_ids"],
        block_indices=integer_fields["block_indices"],
        target_mask=target_mask,
        target_latents=target_latents,
        noisy_latents=noisy_latents,
        target_timesteps=target_timesteps,
        compiled_sequences=compiled,
    ).validate()
