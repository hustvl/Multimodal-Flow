from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mf._compat import Self
from mf.contracts.batch import (
    TEXT_LATENT_DIM,
    VISION_TOKENS,
    BranchRole,
    TaskType,
    _require_bool,
    _require_floating,
    _require_integer,
    _require_same_device,
    _require_shape,
    _require_tensor,
    _validate_task_type,
    _validate_task_type_metadata,
    task_branch_roles,
)
from mf.contracts.model import MFModelInput


@dataclass
class TrainingBatch:
    """A validated model input paired with normalized clean targets."""

    task_type: Tensor
    model_input: MFModelInput
    vision_target_norm: Tensor | None
    text_target_norm: Tensor | None
    vision_target_mask: Tensor
    text_target_mask: Tensor
    vision_noisy_input_norm: Tensor | None = None
    text_noisy_input_norm: Tensor | None = None
    text_decoder_latents_raw: Tensor | None = None
    text_decoder_input_latents: Tensor | None = None
    text_token_ids: Tensor | None = None

    def validate_metadata(self) -> Self:
        if not isinstance(self.model_input, MFModelInput):
            raise ValueError("model_input must be a MFModelInput")
        self.model_input.validate_metadata()
        model_batch_size = self.model_input.vision_present.shape[0]
        text_tokens = self.model_input.text_content_mask.shape[1]
        vision_latent_dim = self.model_input.vision_latent_dim
        geometry = self.model_input.geometry
        vision_tokens = (
            geometry.vision_tokens
            if geometry is not None
            else (
                self.model_input.vision_latents_norm.shape[1]
                if self.model_input.vision_latents_norm is not None
                else VISION_TOKENS
            )
        )
        text_latent_dim = (
            geometry.text_latent_dim
            if geometry is not None
            else next(
                (
                    value.shape[-1]
                    for value in (
                        self.model_input.text_latents_norm,
                        self.text_target_norm,
                        self.text_noisy_input_norm,
                        self.text_decoder_latents_raw,
                        self.text_decoder_input_latents,
                    )
                    if value is not None
                ),
                TEXT_LATENT_DIM,
            )
        )

        task_type, task_batch_size = _validate_task_type_metadata(self.task_type)
        if task_batch_size != model_batch_size:
            raise ValueError(
                "task_type batch dimension must match model_input batch dimension; "
                f"got {task_batch_size} and {model_batch_size}"
            )

        vision_target_mask = _require_tensor(
            "vision_target_mask", self.vision_target_mask
        )
        text_target_mask = _require_tensor("text_target_mask", self.text_target_mask)
        _require_shape(
            "vision_target_mask",
            vision_target_mask,
            (model_batch_size, vision_tokens),
            f"[B, {vision_tokens}]",
        )
        _require_shape(
            "text_target_mask",
            text_target_mask,
            (model_batch_size, text_tokens),
            f"[B, {text_tokens}]",
        )
        _require_bool("vision_target_mask", vision_target_mask)
        _require_bool("text_target_mask", text_target_mask)

        tensors: dict[str, Tensor] = {
            "vision_present": self.model_input.vision_present,
            "vision_target_mask": vision_target_mask,
            "text_target_mask": text_target_mask,
        }
        if self.vision_target_norm is not None:
            vision_target = _require_tensor(
                "vision_target_norm", self.vision_target_norm
            )
            _require_shape(
                "vision_target_norm",
                vision_target,
                (model_batch_size, vision_tokens, vision_latent_dim),
                f"[B, {vision_tokens}, {vision_latent_dim}]",
            )
            _require_floating("vision_target_norm", vision_target)
            tensors["vision_target_norm"] = vision_target
        if self.text_target_norm is not None:
            text_target = _require_tensor("text_target_norm", self.text_target_norm)
            _require_shape(
                "text_target_norm",
                text_target,
                (model_batch_size, text_tokens, text_latent_dim),
                f"[B, {text_tokens}, {text_latent_dim}]",
            )
            _require_floating("text_target_norm", text_target)
            tensors["text_target_norm"] = text_target
        if self.vision_noisy_input_norm is not None:
            vision_noisy_input = _require_tensor(
                "vision_noisy_input_norm",
                self.vision_noisy_input_norm,
            )
            _require_shape(
                "vision_noisy_input_norm",
                vision_noisy_input,
                (model_batch_size, vision_tokens, vision_latent_dim),
                f"[B, {vision_tokens}, {vision_latent_dim}]",
            )
            _require_floating("vision_noisy_input_norm", vision_noisy_input)
            tensors["vision_noisy_input_norm"] = vision_noisy_input
        if self.text_noisy_input_norm is not None:
            text_noisy_input = _require_tensor(
                "text_noisy_input_norm",
                self.text_noisy_input_norm,
            )
            _require_shape(
                "text_noisy_input_norm",
                text_noisy_input,
                (model_batch_size, text_tokens, text_latent_dim),
                f"[B, {text_tokens}, {text_latent_dim}]",
            )
            _require_floating("text_noisy_input_norm", text_noisy_input)
            tensors["text_noisy_input_norm"] = text_noisy_input
        if self.text_decoder_latents_raw is not None:
            decoder_latents = _require_tensor(
                "text_decoder_latents_raw", self.text_decoder_latents_raw
            )
            _require_shape(
                "text_decoder_latents_raw",
                decoder_latents,
                (model_batch_size, text_tokens, text_latent_dim),
                f"[B, {text_tokens}, {text_latent_dim}]",
            )
            _require_floating("text_decoder_latents_raw", decoder_latents)
            tensors["text_decoder_latents_raw"] = decoder_latents
        if self.text_decoder_input_latents is not None:
            decoder_input = _require_tensor(
                "text_decoder_input_latents", self.text_decoder_input_latents
            )
            _require_shape(
                "text_decoder_input_latents",
                decoder_input,
                (model_batch_size, text_tokens, text_latent_dim),
                f"[B, {text_tokens}, {text_latent_dim}]",
            )
            _require_floating("text_decoder_input_latents", decoder_input)
            tensors["text_decoder_input_latents"] = decoder_input
        if self.text_token_ids is not None:
            token_ids = _require_tensor("text_token_ids", self.text_token_ids)
            _require_shape(
                "text_token_ids",
                token_ids,
                (model_batch_size, text_tokens),
                f"[B, {text_tokens}]",
            )
            _require_integer("text_token_ids", token_ids)
            tensors["text_token_ids"] = token_ids

        _require_same_device("task_type", task_type, **tensors)
        return self

    def validate(self) -> Self:
        self.validate_metadata()
        self.model_input.validate()
        if self.model_input.physical_layout is not None:
            if bool(self.vision_target_mask.any()) or bool(self.text_target_mask.any()):
                raise ValueError(
                    "physical sequence batches must use physical_layout.target_mask"
                )
            return self
        task_type, _ = _validate_task_type(self.task_type)
        expected_vision_role, expected_text_role = task_branch_roles(task_type)
        if not torch.equal(self.model_input.vision_role, expected_vision_role):
            raise ValueError("model_input.vision_role must match task_type")
        if not torch.equal(self.model_input.text_role, expected_text_role):
            raise ValueError("model_input.text_role must match task_type")

        vision_target_mask = _require_tensor(
            "vision_target_mask", self.vision_target_mask
        )
        text_target_mask = _require_tensor("text_target_mask", self.text_target_mask)
        pad_target = text_target_mask & ~self.model_input.text_content_mask
        if bool(pad_target.any()):
            raise ValueError("text PAD tokens must never be marked as a target")

        expected_vision_target_rows = expected_vision_role == int(BranchRole.TARGET)
        expected_vision_target = expected_vision_target_rows[:, None].expand(
            -1, self.vision_target_mask.shape[1]
        )
        expected_text_target_rows = expected_text_role == int(BranchRole.TARGET)
        expected_text_target = (
            expected_text_target_rows[:, None] & self.model_input.text_content_mask
        )

        wrong_vision = vision_target_mask != expected_vision_target
        if bool(wrong_vision.any()):
            location = wrong_vision.nonzero(as_tuple=False)[0]
            batch_index = int(location[0].item())
            token_index = int(location[1].item())
            task = TaskType(int(task_type[batch_index].item()))
            expected_target = bool(expected_vision_target_rows[batch_index].item())
            raise ValueError(
                f"{task.label} at batch index {batch_index}, vision token {token_index} "
                f"requires vision_target_mask={expected_target}"
            )
        wrong_text = (text_target_mask != expected_text_target).any(dim=1)
        if bool(wrong_text.any()):
            index = int(wrong_text.nonzero(as_tuple=False)[0, 0].item())
            task = TaskType(int(task_type[index].item()))
            raise ValueError(
                f"{task.label} text_target_mask must equal its non-PAD target tokens"
            )

        has_vision_target = bool(vision_target_mask.any())
        has_text_target = bool(text_target_mask.any())
        if has_vision_target and self.vision_target_norm is None:
            raise ValueError("vision_target_norm is required for a vision target")
        if has_text_target and self.text_target_norm is None:
            raise ValueError("text_target_norm is required for a text target")
        if has_text_target and self.text_decoder_latents_raw is None:
            raise ValueError(
                "text_decoder_latents_raw is required when any text target is active"
            )
        if has_text_target and self.text_token_ids is None:
            raise ValueError(
                "text_token_ids is required when any text target is active"
            )
        if has_text_target and self.text_decoder_input_latents is None:
            raise ValueError(
                "text_decoder_input_latents is required when any text target is active"
            )

        vision_condition_rows = expected_vision_role == int(BranchRole.CONDITION)
        invalid_vision_condition = vision_condition_rows & (
            self.model_input.vision_timestep != 1
        )
        if bool(invalid_vision_condition.any()):
            raise ValueError("vision CONDITION role requires t=1")
        text_condition_rows = expected_text_role == int(BranchRole.CONDITION)
        invalid_text_condition = text_condition_rows & (
            self.model_input.text_timestep != 1
        )
        if bool(invalid_text_condition.any()):
            raise ValueError("text CONDITION role requires t=1")
        return self


@dataclass
class LossOutput:
    """Scalar losses with names matching trainer metrics."""

    total: Tensor
    vision_flow: Tensor
    text_flow: Tensor
    text_decoder_ce: Tensor
    physical_flow: Tensor

    def validate(self) -> Self:
        tensors: dict[str, Tensor] = {}
        for name in (
            "total",
            "vision_flow",
            "text_flow",
            "text_decoder_ce",
            "physical_flow",
        ):
            tensor = _require_tensor(name, getattr(self, name))
            if tensor.ndim != 0:
                raise ValueError(
                    f"{name} must be a scalar tensor; got {list(tensor.shape)}"
                )
            _require_floating(name, tensor)
            tensors[name] = tensor
        total = tensors.pop("total")
        _require_same_device("total", total, **tensors)
        return self


def _validate_non_negative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class TrainerState:
    """Immutable checkpointable counters for the trainer state machine."""

    global_step: int = 0
    samples_seen: int = 0
    last_checkpoint_step: int | None = None
    last_evaluation_step: int | None = None

    def __post_init__(self) -> None:
        _validate_non_negative_int("global_step", self.global_step)
        _validate_non_negative_int("samples_seen", self.samples_seen)
        for name, value in (
            ("last_checkpoint_step", self.last_checkpoint_step),
            ("last_evaluation_step", self.last_evaluation_step),
        ):
            if value is not None:
                _validate_non_negative_int(name, value)
                if value > self.global_step:
                    raise ValueError(f"{name} cannot exceed global_step")
