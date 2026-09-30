from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mf._compat import Self
from mf.contracts.batch import (
    TEXT_LATENT_DIM,
    VISION_TOKENS,
    BranchRole,
    _require_bool,
    _require_floating,
    _require_integer,
    _require_same_device,
    _require_shape,
    _require_tensor,
)
from mf.contracts.chunks import ChunkRoutingMetadata
from mf.contracts.geometry import GeometryContract
from mf.contracts.physical import PhysicalSequenceLayout
from mf.contracts.sequence import CompiledSequence, MultimodalSequence


def _validate_role(name: str, role: object) -> tuple[Tensor, int]:
    tensor = _require_tensor(name, role)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must have shape [B]; got {list(tensor.shape)}")
    _require_integer(name, tensor)
    valid = torch.zeros_like(tensor, dtype=torch.bool)
    for value in BranchRole:
        valid |= tensor == int(value)
    if bool((~valid).any()):
        index = int((~valid).nonzero(as_tuple=False)[0, 0].item())
        raise ValueError(
            f"{name}[{index}] has unknown role {int(tensor[index].item())}"
        )
    return tensor, tensor.shape[0]


@dataclass
class MFModelInput:
    """MF normalized input with roles as the routing source of truth."""

    vision_latents_norm: Tensor | None
    text_latents_norm: Tensor | None
    text_prompt_latents_norm: Tensor | None
    text_prompt_content_mask: Tensor
    vision_timestep: Tensor
    text_timestep: Tensor
    vision_role: Tensor
    text_role: Tensor
    text_content_mask: Tensor
    text_latent_stats_type: Tensor
    active_token_mask: Tensor
    text_previous_x0_norm: Tensor | None
    null_conditioning: Tensor | None = None
    text_clean_latents_norm: Tensor | None = None
    text_token_timestep: Tensor | None = None
    text_block_size: int | None = None
    text_segment_ids: Tensor | None = None
    chunk_routing_cpu: ChunkRoutingMetadata | None = None
    sequence_contracts: tuple[MultimodalSequence, ...] | None = None
    compiled_sequences: tuple[CompiledSequence, ...] | None = None
    vision_latent_dim: int = 768
    geometry: GeometryContract | None = None
    physical_layout: PhysicalSequenceLayout | None = None

    @property
    def vision_present(self) -> Tensor:
        return self.vision_role != int(BranchRole.ABSENT)

    @property
    def text_present(self) -> Tensor:
        return self.text_role != int(BranchRole.ABSENT)

    @property
    def vision_target_rows(self) -> Tensor:
        return self.vision_role == int(BranchRole.TARGET)

    @property
    def text_target_rows(self) -> Tensor:
        return self.text_role == int(BranchRole.TARGET)

    @property
    def vision_branch_active(self) -> bool:
        return bool(self.vision_present.any())

    @property
    def text_branch_active(self) -> bool:
        return bool(self.text_present.any())

    @property
    def vision_layout_active(self) -> Tensor:
        return self.vision_present

    @property
    def text_layout_active(self) -> Tensor:
        return self.text_present

    @property
    def text_prompt_present(self) -> Tensor:
        return self.text_prompt_content_mask.any(dim=1)

    @property
    def block_causal_text(self) -> bool:
        return self.text_block_size is not None

    @property
    def text_loss_timestep(self) -> Tensor:
        return (
            self.text_token_timestep
            if self.text_token_timestep is not None
            else self.text_timestep
        )

    def validate_metadata(self) -> Self:
        vision_role, batch_size = _validate_role("vision_role", self.vision_role)
        text_role, text_batch_size = _validate_role("text_role", self.text_role)
        if text_batch_size != batch_size:
            raise ValueError(
                "vision_role and text_role must have the same batch dimension"
            )
        inferred_vision_tokens = (
            self.vision_latents_norm.shape[1]
            if isinstance(self.vision_latents_norm, Tensor)
            else VISION_TOKENS
        )
        inferred_text_dim = next(
            (
                value.shape[-1]
                for value in (
                    self.text_latents_norm,
                    self.text_clean_latents_norm,
                    self.text_prompt_latents_norm,
                )
                if isinstance(value, Tensor)
            ),
            TEXT_LATENT_DIM,
        )
        geometry = self.geometry or GeometryContract(
            vision_tokens=inferred_vision_tokens,
            vision_latent_dim=self.vision_latent_dim,
            text_latent_dim=inferred_text_dim,
        )
        geometry.validate()
        if geometry.vision_latent_dim != self.vision_latent_dim:
            raise ValueError("geometry.vision_latent_dim must match vision_latent_dim")
        self.geometry = geometry
        _require_same_device("vision_role", vision_role, text_role=text_role)
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
                raise ValueError(
                    "model input must carry compiled_sequences with sequence_contracts"
                )
            if len(self.compiled_sequences) != len(compiled):
                raise ValueError(
                    "compiled_sequences must match sequence_contracts length"
                )
            for sequence, expected in zip(
                self.compiled_sequences, compiled, strict=True
            ):
                if sequence.routing_signature() != expected.routing_signature():
                    raise ValueError(
                        "model input compiled_sequences must derive from sequence_contracts"
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

        content_mask = _require_tensor("text_content_mask", self.text_content_mask)
        if (
            content_mask.ndim != 2
            or content_mask.shape[0] != batch_size
            or content_mask.shape[1] <= 0
        ):
            raise ValueError("text_content_mask must have shape [B, T] with T > 0")
        _require_bool("text_content_mask", content_mask)
        text_tokens = content_mask.shape[1]

        prompt_mask = _require_tensor(
            "text_prompt_content_mask",
            self.text_prompt_content_mask,
        )
        if prompt_mask.ndim != 2 or prompt_mask.shape[0] != batch_size:
            raise ValueError("text_prompt_content_mask must have shape [B, P]")
        _require_bool("text_prompt_content_mask", prompt_mask)
        if bool((~prompt_mask[:, :-1] & prompt_mask[:, 1:]).any()):
            raise ValueError("text_prompt_content_mask must be right padded")
        required_prompt_rows = (vision_role == int(BranchRole.CONDITION)) & (
            text_role == int(BranchRole.TARGET)
        )
        prompt_rows = prompt_mask.any(dim=1)
        allowed_prompt_rows = (text_role == int(BranchRole.TARGET)) & (
            (vision_role == int(BranchRole.CONDITION))
            | (vision_role == int(BranchRole.ABSENT))
        )
        if bool((required_prompt_rows & ~prompt_rows).any()):
            raise ValueError("image_to_text requires a clean text prompt")
        if bool((prompt_rows & ~allowed_prompt_rows).any()):
            raise ValueError("plain text prompts require a text target")
        prompt_tokens = prompt_mask.shape[1]
        if self.chunk_routing_cpu is not None:
            routing = self.chunk_routing_cpu.validate()
            if routing.geometry is not None and routing.geometry != geometry:
                raise ValueError("chunk routing geometry must match model input geometry")
            if routing.compiled_sequences is not None:
                if self.compiled_sequences is None:
                    raise ValueError(
                        "model input must carry compiled sequences with chunk routing"
                    )
                if tuple(
                    item.routing_signature() for item in routing.compiled_sequences
                ) != tuple(
                    item.routing_signature() for item in self.compiled_sequences
                ):
                    raise ValueError(
                        "chunk routing compiled sequences must match model input"
                    )
            if routing.vision_role.shape[0] != batch_size:
                raise ValueError("chunk routing batch dimension must match model input")
            routing_vision_role = routing.vision_role.to(device=vision_role.device)
            routing_text_role = routing.text_role.to(device=text_role.device)
            if not torch.equal(routing_vision_role, vision_role):
                raise ValueError(
                    "chunk routing vision_role must match model input"
                )
            if not torch.equal(routing_text_role, text_role):
                raise ValueError("chunk routing text_role must match model input")
            if routing.text_content_mask.shape != content_mask.shape:
                raise ValueError(
                    "chunk routing text_content_mask must match model input"
                )
            routing_content_mask = routing.text_content_mask.to(
                device=content_mask.device
            )
            if not torch.equal(routing_content_mask, content_mask):
                raise ValueError(
                    "chunk routing text_content_mask must match model input"
                )
            if routing.text_prompt_content_mask.shape != prompt_mask.shape:
                raise ValueError(
                    "chunk routing text_prompt_content_mask must match model input"
                )
            routing_prompt_mask = routing.text_prompt_content_mask.to(
                device=prompt_mask.device
            )
            if not torch.equal(routing_prompt_mask, prompt_mask):
                raise ValueError(
                    "chunk routing text_prompt_content_mask must match model input"
                )
            if self.text_segment_ids is None:
                raise ValueError(
                    "chunk routing requires model input text_segment_ids"
                )
            routing_segment_ids = routing.text_segment_ids.to(
                device=self.text_segment_ids.device
            )
            if not torch.equal(routing_segment_ids, self.text_segment_ids):
                raise ValueError(
                    "chunk routing text_segment_ids must match model input"
                )
            if self.sequence_contracts is not None and (
                routing.sequence_contracts is None
                or tuple(
                    sequence.routing_signature() for sequence in routing.sequence_contracts
                )
                != tuple(
                    sequence.routing_signature() for sequence in self.sequence_contracts
                )
            ):
                raise ValueError(
                    "chunk routing sequence_contracts must match model input"
                )
            if self.null_conditioning is not None:
                routing_null = routing.null_conditioning
                if routing_null is None or not torch.equal(
                    routing_null.to(device=self.null_conditioning.device),
                    self.null_conditioning,
                ):
                    raise ValueError(
                        "chunk routing null_conditioning must match model input"
                    )
        if bool(prompt_rows.any()) and self.text_prompt_latents_norm is None:
            raise ValueError("active clean text prompts require prompt latents")
        if self.text_prompt_latents_norm is not None:
            prompt_latents = _require_tensor(
                "text_prompt_latents_norm",
                self.text_prompt_latents_norm,
            )
            _require_shape(
                "text_prompt_latents_norm",
                prompt_latents,
                (batch_size, prompt_tokens, geometry.text_latent_dim),
                f"[B, {prompt_tokens}, {geometry.text_latent_dim}]",
            )
            _require_floating("text_prompt_latents_norm", prompt_latents)

        tensors: dict[str, Tensor] = {
            "text_role": text_role,
            "text_content_mask": content_mask,
            "text_prompt_content_mask": prompt_mask,
        }
        if self.text_prompt_latents_norm is not None:
            tensors["text_prompt_latents_norm"] = self.text_prompt_latents_norm
        vision_active = vision_role != int(BranchRole.ABSENT)
        text_active = text_role != int(BranchRole.ABSENT)

        if type(self.vision_latent_dim) is not int or self.vision_latent_dim <= 0:
            raise ValueError("vision_latent_dim must be a positive integer")
        if bool(vision_active.any()) and self.vision_latents_norm is None:
            raise ValueError(
                "vision_latents_norm is required for an active vision role"
            )
        if self.vision_latents_norm is not None:
            vision_latents = _require_tensor(
                "vision_latents_norm", self.vision_latents_norm
            )
            _require_shape(
                "vision_latents_norm",
                vision_latents,
                (batch_size, geometry.vision_tokens, self.vision_latent_dim),
                f"[B, {geometry.vision_tokens}, {self.vision_latent_dim}]",
            )
            _require_floating("vision_latents_norm", vision_latents)
            tensors["vision_latents_norm"] = vision_latents

        if bool(text_active.any()) and self.text_latents_norm is None:
            raise ValueError("text_latents_norm is required for an active text role")
        if bool(text_active.any()) and self.text_previous_x0_norm is None:
            raise ValueError(
                "text_previous_x0_norm is required for an active text role"
            )
        if self.text_latents_norm is not None:
            text_latents = _require_tensor("text_latents_norm", self.text_latents_norm)
            _require_shape(
                "text_latents_norm",
                text_latents,
                (batch_size, text_tokens, geometry.text_latent_dim),
                f"[B, {text_tokens}, {geometry.text_latent_dim}]",
            )
            _require_floating("text_latents_norm", text_latents)
            tensors["text_latents_norm"] = text_latents
        block_fields = (
            self.text_clean_latents_norm,
            self.text_token_timestep,
            self.text_block_size,
            self.text_segment_ids,
        )
        if any(value is not None for value in block_fields) and not all(
            value is not None for value in block_fields
        ):
            raise ValueError(
                "text_clean_latents_norm, text_token_timestep, text_segment_ids, and "
                "text_block_size must be enabled together"
            )
        if self.block_causal_text:
            if type(self.text_block_size) is not int or self.text_block_size <= 0:
                raise ValueError("text_block_size must be a positive integer")
            clean_text = _require_tensor(
                "text_clean_latents_norm", self.text_clean_latents_norm
            )
            _require_shape(
                "text_clean_latents_norm",
                clean_text,
                (batch_size, text_tokens, geometry.text_latent_dim),
                f"[B, {text_tokens}, {geometry.text_latent_dim}]",
            )
            _require_floating("text_clean_latents_norm", clean_text)
            token_timestep = _require_tensor(
                "text_token_timestep", self.text_token_timestep
            )
            _require_shape(
                "text_token_timestep",
                token_timestep,
                (batch_size, text_tokens),
                f"[B, {text_tokens}]",
            )
            _require_floating("text_token_timestep", token_timestep)
            segment_ids = _require_tensor("text_segment_ids", self.text_segment_ids)
            _require_shape(
                "text_segment_ids",
                segment_ids,
                (batch_size, text_tokens),
                f"[B, {text_tokens}]",
            )
            _require_integer("text_segment_ids", segment_ids)
            if bool((content_mask & (segment_ids < 0)).any()):
                raise ValueError(
                    "active text tokens must have non-negative segment ids"
                )
            if bool(((~content_mask) & (segment_ids != -1)).any()):
                raise ValueError("inactive text tokens must have segment id -1")
            if text_tokens > 1 and bool(
                (
                    content_mask[:, 1:]
                    & content_mask[:, :-1]
                    & (segment_ids[:, 1:] < segment_ids[:, :-1])
                ).any()
            ):
                raise ValueError("text segment ids must be non-decreasing")
            tensors["text_segment_ids"] = segment_ids
            tensors["text_clean_latents_norm"] = clean_text
            tensors["text_token_timestep"] = token_timestep
        if self.text_previous_x0_norm is not None:
            previous = _require_tensor(
                "text_previous_x0_norm", self.text_previous_x0_norm
            )
            _require_shape(
                "text_previous_x0_norm",
                previous,
                (batch_size, text_tokens, geometry.text_latent_dim),
                f"[B, {text_tokens}, {geometry.text_latent_dim}]",
            )
            _require_floating("text_previous_x0_norm", previous)
            tensors["text_previous_x0_norm"] = previous

        for name, timestep in (
            ("vision_timestep", self.vision_timestep),
            ("text_timestep", self.text_timestep),
        ):
            value = _require_tensor(name, timestep)
            _require_shape(name, value, (batch_size,), "[B]")
            _require_floating(name, value)
            tensors[name] = value

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

        if self.null_conditioning is not None:
            null_conditioning = _require_tensor(
                "null_conditioning", self.null_conditioning
            )
            _require_shape("null_conditioning", null_conditioning, (batch_size,), "[B]")
            _require_bool("null_conditioning", null_conditioning)
            tensors["null_conditioning"] = null_conditioning

        prompt_layout_tokens = (
            geometry.text_prefix_tokens + prompt_tokens if prompt_tokens > 0 else 0
        )
        total_tokens = (
            self.physical_layout.active_token_mask.shape[1]
            if self.physical_layout is not None
            else geometry.vision_layout_tokens
            + prompt_layout_tokens
            + geometry.text_prefix_tokens
            + text_tokens * (2 if self.block_causal_text else 1)
        )
        active_mask = _require_tensor("active_token_mask", self.active_token_mask)
        _require_shape(
            "active_token_mask",
            active_mask,
            (batch_size, total_tokens),
            f"[B, {total_tokens}]",
        )
        _require_bool("active_token_mask", active_mask)
        tensors["active_token_mask"] = active_mask
        if self.physical_layout is not None:
            physical = self.physical_layout.validate()
            if physical.token_embeddings.shape[:2] != active_mask.shape:
                raise ValueError("physical_layout must match active_token_mask")
            if not torch.equal(physical.active_token_mask, active_mask):
                raise ValueError("physical_layout active mask must match model input")
            tensors["physical_layout_token_embeddings"] = physical.token_embeddings
        _require_same_device("vision_role", vision_role, **tensors)
        return self

    def validate(self) -> Self:
        self.validate_metadata()
        if self.physical_layout is not None:
            if bool(self.vision_present.any()) or bool(self.text_present.any()):
                raise ValueError(
                    "physical sequence inputs must not also declare legacy branches"
                )
            for name, value in (
                ("vision_timestep", self.vision_timestep),
                ("text_timestep", self.text_timestep),
            ):
                if bool((~torch.isfinite(value) | value.ne(0)).any()):
                    raise ValueError(
                        f"{name} must be zero for a physical sequence input"
                    )
            return self
        geometry = self.geometry
        assert geometry is not None
        vision_active = self.vision_present
        text_active = self.text_present

        for name, timestep, role in (
            ("vision_timestep", self.vision_timestep, self.vision_role),
            ("text_timestep", self.text_timestep, self.text_role),
        ):
            invalid = ~torch.isfinite(timestep) | (timestep < 0) | (timestep > 1)
            if bool(invalid.any()):
                raise ValueError(f"{name} must contain finite clean-t values in [0, 1]")
            if bool(((role == int(BranchRole.ABSENT)) & timestep.ne(0)).any()):
                raise ValueError(f"{name} ABSENT rows must use t=0")
            condition_rows = role == int(BranchRole.CONDITION)
            if bool((condition_rows & timestep.ne(1)).any()):
                raise ValueError(f"{name} CONDITION rows must use clean-t=1")

        if self.text_token_timestep is not None:
            token_timestep = self.text_token_timestep
            invalid_token_t = (
                ~torch.isfinite(token_timestep)
                | (token_timestep < 0)
                | (token_timestep > 1)
            )
            if bool(invalid_token_t.any()):
                raise ValueError(
                    "text_token_timestep must contain finite clean-t values in [0, 1]"
                )
            # Inference starts every block at the exact all-noise endpoint clean-t=0.
            expected_token_active = (
                self.text_target_rows[:, None] & self.text_content_mask
            )
            if bool((~expected_token_active & token_timestep.ne(0)).any()):
                raise ValueError(
                    "text_token_timestep must be zero outside active text targets"
                )

        content_mask = self.text_content_mask
        if bool((content_mask & ~text_active[:, None]).any()):
            raise ValueError(
                "text_content_mask must be false for an ABSENT text branch"
            )

        text_target_rows = self.text_target_rows
        previous = self.text_previous_x0_norm
        if previous is not None:
            allowed_target_tokens = text_target_rows[:, None] & content_mask
            outside_target = torch.where(
                allowed_target_tokens.unsqueeze(-1),
                torch.zeros_like(previous),
                previous,
            )
            if bool(outside_target.ne(0).any()):
                raise ValueError(
                    "text_previous_x0_norm may be nonzero only on valid TARGET tokens"
                )
        if self.null_conditioning is not None:
            condition_rows = (self.vision_role == int(BranchRole.CONDITION)) | (
                self.text_role == int(BranchRole.CONDITION)
            )
            if bool((self.null_conditioning & ~condition_rows).any()):
                raise ValueError(
                    "null_conditioning may only be enabled on rows with a condition role"
                )

        expected_active = torch.zeros_like(self.active_token_mask)
        expected_active[:, :geometry.vision_layout_tokens] = vision_active[:, None]
        prompt_tokens = self.text_prompt_content_mask.shape[1]
        prompt_prefix_start = geometry.vision_layout_tokens
        prompt_content_start = prompt_prefix_start
        if prompt_tokens > 0:
            prompt_content_start += geometry.text_prefix_tokens
            expected_active[:, prompt_prefix_start:prompt_content_start] = (
                self.text_prompt_present[:, None]
            )
        text_prefix_start = prompt_content_start + prompt_tokens
        text_latent_start = text_prefix_start + geometry.text_prefix_tokens
        expected_active[:, prompt_content_start:text_prefix_start] = (
            self.text_prompt_content_mask
        )
        expected_active[:, text_prefix_start:text_latent_start] = text_active[:, None]
        if self.block_causal_text:
            clean_end = text_latent_start + content_mask.shape[1]
            expected_active[:, text_latent_start:clean_end] = (
                content_mask & text_active[:, None]
            )
            expected_active[:, clean_end:] = content_mask & text_target_rows[:, None]
        else:
            expected_active[:, text_latent_start:] = content_mask & text_active[:, None]
        if not torch.equal(self.active_token_mask, expected_active):
            raise ValueError(
                "active_token_mask must be derived exactly from branch roles and valid text tokens"
            )
        return self


@dataclass
class MFOutput:
    """Normalized-x0 runtime predictions for active MF branches."""

    vision_pred_norm: Tensor | None
    text_pred_norm: Tensor | None
    active_token_mask: Tensor
    vision_latent_dim: int = 768
    geometry: GeometryContract | None = None
    modality_pred_norm: dict[int, Tensor] | None = None
    modality_target_indices: dict[int, Tensor] | None = None
    physical_layout: PhysicalSequenceLayout | None = None

    def validate_metadata(self) -> Self:
        geometry = self.geometry or GeometryContract(
            vision_tokens=(
                self.vision_pred_norm.shape[1]
                if isinstance(self.vision_pred_norm, Tensor)
                else VISION_TOKENS
            ),
            vision_latent_dim=self.vision_latent_dim,
            text_latent_dim=(
                self.text_pred_norm.shape[-1]
                if isinstance(self.text_pred_norm, Tensor)
                else TEXT_LATENT_DIM
            ),
        )
        geometry.validate()
        self.geometry = geometry
        active_mask = _require_tensor("active_token_mask", self.active_token_mask)
        minimum_layout_tokens = (
            1
            if self.vision_pred_norm is None and self.text_pred_norm is None
            else geometry.vision_layout_tokens + geometry.text_prefix_tokens + 1
        )
        if active_mask.ndim != 2 or active_mask.shape[1] < minimum_layout_tokens:
            raise ValueError(
                "active_token_mask must contain both modality prefixes and T > 0; "
                f"got {list(active_mask.shape)}"
            )
        _require_bool("active_token_mask", active_mask)
        batch_size = active_mask.shape[0]

        tensors: dict[str, Tensor] = {}
        if type(self.vision_latent_dim) is not int or self.vision_latent_dim <= 0:
            raise ValueError("vision_latent_dim must be a positive integer")
        if self.vision_pred_norm is not None:
            vision_pred = _require_tensor("vision_pred_norm", self.vision_pred_norm)
            _require_shape(
                "vision_pred_norm",
                vision_pred,
                (batch_size, geometry.vision_tokens, self.vision_latent_dim),
                f"[B, {geometry.vision_tokens}, {self.vision_latent_dim}]",
            )
            _require_floating("vision_pred_norm", vision_pred)
            tensors["vision_pred_norm"] = vision_pred
        if self.text_pred_norm is not None:
            text_pred = _require_tensor("text_pred_norm", self.text_pred_norm)
            if (
                text_pred.ndim != 3
                or text_pred.shape[0] != batch_size
                or text_pred.shape[1] <= 0
                or text_pred.shape[2] != geometry.text_latent_dim
            ):
                raise ValueError(
                    "text_pred_norm must have shape [B, T, text_latent_dim] with T > 0"
                )
            _require_floating("text_pred_norm", text_pred)
            tensors["text_pred_norm"] = text_pred
        if self.modality_pred_norm is not None:
            if not isinstance(self.modality_pred_norm, dict):
                raise ValueError("modality_pred_norm must be a dict when provided")
            for modality_id, prediction in self.modality_pred_norm.items():
                if type(modality_id) is not int or modality_id < 0:
                    raise ValueError("modality prediction ids must be non-negative integers")
                value = _require_tensor(
                    f"modality_pred_norm[{modality_id}]",
                    prediction,
                )
                if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
                    raise ValueError(
                        "dynamic modality predictions must have shape [tokens, latent_dim]"
                    )
                _require_floating(f"modality_pred_norm[{modality_id}]", value)
                tensors[f"modality_pred_norm[{modality_id}]"] = value
        if self.modality_target_indices is not None:
            if not isinstance(self.modality_target_indices, dict):
                raise ValueError("modality_target_indices must be a dict when provided")
            for modality_id, indices in self.modality_target_indices.items():
                if type(modality_id) is not int or modality_id < 0:
                    raise ValueError(
                        "modality target ids must be non-negative integers"
                    )
                if indices.ndim != 1 or indices.dtype not in {
                    torch.int32,
                    torch.int64,
                }:
                    raise ValueError(
                        "modality target indices must be integer vectors"
                    )
                if indices.device != active_mask.device:
                    raise ValueError(
                        "modality target indices must share active mask device"
                    )
                if self.modality_pred_norm is not None and modality_id in self.modality_pred_norm:
                    prediction_length = self.modality_pred_norm[modality_id].shape[0]
                    if indices.numel() and int(indices.max().item()) >= prediction_length:
                        raise ValueError(
                            "modality target indices must index their prediction tensor"
                        )
                tensors[f"modality_target_indices[{modality_id}]"] = indices
        _require_same_device("active_token_mask", active_mask, **tensors)
        if self.physical_layout is not None:
            physical = self.physical_layout.validate()
            if physical.active_token_mask.shape != active_mask.shape:
                raise ValueError(
                    "physical layout active_token_mask must match output active_token_mask"
                )
            if not torch.equal(physical.active_token_mask, active_mask):
                raise ValueError(
                    "physical layout and output active_token_mask disagree"
                )
            target_mask = physical.active_token_mask & physical.target_mask
            target_modalities = {
                int(modality_id)
                for modality_id in torch.unique(physical.modality_ids[target_mask]).tolist()
            }
            predictions = self.modality_pred_norm or {}
            indices = self.modality_target_indices or {}
            if not target_modalities.issubset(predictions):
                missing = sorted(target_modalities.difference(predictions))
                raise ValueError(
                    f"physical output is missing modality predictions: {missing}"
                )
            if set(indices) != target_modalities:
                raise ValueError(
                    "physical output target indices must exactly match target modalities"
                )
            for modality_id in target_modalities:
                expected = int(
                    (target_mask & physical.modality_ids.eq(modality_id)).sum().item()
                )
                if indices[modality_id].numel() != expected:
                    raise ValueError(
                        f"physical output alignment for modality {modality_id} "
                        f"has {indices[modality_id].numel()} rows; expected {expected}"
                    )
        return self

    def validate(self) -> Self:
        self.validate_metadata()
        return self
