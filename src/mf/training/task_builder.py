from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from mf.codecs.online import prepare_raw_batch_cpu
from mf.config.schema import MFConfig
from mf.contracts.batch import (
    BranchRole,
    EncodedTaskBatch,
    RawTaskBatch,
    task_branch_roles,
)
from mf.contracts.chunks import build_text_segment_ids as _build_text_segment_ids
from mf.contracts.geometry import GeometryContract
from mf.contracts.model import MFModelInput
from mf.contracts.trainer import TrainingBatch
from mf.latents.flow import sample_flow_input
from mf.latents.stats import LatentStatsRegistry, TextLatentStatsType
from mf.latents.time import sample_shifted_clean_t
from mf.modeling.chunk_adapter import prepare_chunk_routing_cpu
from mf.training.execution import split_generator


def build_cpu_batch_preparer(
    config: MFConfig,
) -> Callable[[RawTaskBatch], RawTaskBatch] | None:
    """Prepare CPU-only batch work ahead of device execution."""

    value = os.environ.get("MF_CPU_BATCH_PREPARATION", "auto")
    if value not in {"auto", "0", "1"}:
        raise ValueError("MF_CPU_BATCH_PREPARATION must be auto, 0 or 1")
    block = config.flow.text_block_causal
    if value == "0" or (value == "auto" and config.data.loader.num_workers <= 0):
        return None
    if config.data.loader.num_workers <= 0:
        raise ValueError("CPU batch preparation requires ordered prefetch workers")

    @torch.no_grad()
    def prepare(batch: RawTaskBatch) -> RawTaskBatch:
        prepared = prepare_raw_batch_cpu(
            batch,
            eos_token_id=config.objective.eos_token_id,
            geometry=GeometryContract.from_config(config),
        )
        routing = prepare_chunk_routing_cpu(
            prepared.chunk_routing,
            text_block_size=block.block_size,
            hidden_size=config.model.hidden_size,
            attention_backend=block.attention_backend,
            image_chunk_conditioning=config.model.image_chunk_conditioning,
            t2i_chunk_semantics=config.model.t2i_chunk_semantics,
        )
        return replace(batch, cpu_preparation=replace(prepared, chunk_routing=routing))

    return prepare


@dataclass(frozen=True)
class ImageFlowBranch:
    noisy_input_norm: Tensor
    target_norm: Tensor
    target_mask: Tensor


def build_image_flow_branch(
    clean_norm: Tensor,
    clean_timestep: Tensor,
    target_rows: Tensor,
    active_rows: Tensor,
    generator: torch.Generator,
    *,
    noise_scale: float,
) -> ImageFlowBranch:
    if clean_norm.ndim != 3:
        raise ValueError("clean_norm must have shape [B, V, D]")
    batch_size, vision_tokens, _ = clean_norm.shape
    for name, value in (
        ("clean_timestep", clean_timestep),
        ("target_rows", target_rows),
        ("active_rows", active_rows),
    ):
        if value.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [B]")
        if value.device != clean_norm.device:
            raise ValueError(f"{name} must share the clean_norm device")
    if target_rows.dtype is not torch.bool or active_rows.dtype is not torch.bool:
        raise ValueError("target_rows and active_rows must have dtype torch.bool")
    if bool((target_rows & ~active_rows).any()):
        raise ValueError("target image rows must also be active")
    noised, _ = sample_flow_input(
        clean_norm,
        clean_timestep,
        generator,
        noise_scale=noise_scale,
    )
    noisy_input = torch.where(
        target_rows[:, None, None],
        noised,
        clean_norm,
    )
    noisy_input = _masked_rows(noisy_input, active_rows)
    target = _masked_rows(clean_norm, target_rows)
    target_mask = target_rows[:, None].expand(-1, vision_tokens)
    return ImageFlowBranch(
        noisy_input_norm=noisy_input,
        target_norm=target,
        target_mask=target_mask,
    )


def _validate_config_contract(config: MFConfig) -> None:
    geometry = GeometryContract.from_config(config)
    if config.codecs.text.max_length != config.data.text_max_length:
        raise ValueError(
            "codecs.text.max_length must equal data.text_max_length; "
            f"got {config.codecs.text.max_length} and {config.data.text_max_length}"
        )
    geometry.validate()


def _masked_rows(value: Tensor, row_mask: Tensor) -> Tensor:
    broadcast = row_mask.view((row_mask.shape[0],) + (1,) * (value.ndim - 1))
    return torch.where(broadcast, value, torch.zeros_like(value))


def _canonical_device(device: torch.device) -> torch.device:
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


class TaskBuilder:
    """Build validated mixed-task model inputs and clean training targets."""

    def __init__(
        self,
        *,
        config: MFConfig,
        latent_stats_registry: LatentStatsRegistry,
    ) -> None:
        if not isinstance(config, MFConfig):
            raise TypeError("config must be a strict MFConfig")
        if not isinstance(latent_stats_registry, LatentStatsRegistry):
            raise TypeError("latent_stats_registry must be a LatentStatsRegistry")
        _validate_config_contract(config)
        if latent_stats_registry.vision_latent_dim != config.codecs.vision.latent_dim:
            raise ValueError(
                "vision stats dimension must match codecs.vision.latent_dim; "
                f"got {latent_stats_registry.vision_latent_dim} and "
                f"{config.codecs.vision.latent_dim}"
            )
        self.config = config
        self.latent_stats_registry = latent_stats_registry
        self.geometry = GeometryContract.from_config(config)

    def _sample_shifted_t(
        self,
        *,
        batch_size: int,
        alpha: float,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> Tensor:
        shift = self.config.flow.timestep_shift
        return sample_shifted_clean_t(
            batch_size=batch_size,
            alpha=alpha,
            mu=shift.t_lognorm_mu,
            sigma=shift.t_lognorm_sigma,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _flow_input(
        clean: Tensor,
        clean_t: Tensor,
        target_rows: Tensor,
        generator: torch.Generator,
        noise_scale: float,
    ) -> Tensor:
        noised, _ = sample_flow_input(
            clean, clean_t, generator, noise_scale=noise_scale
        )
        broadcast = target_rows.view((target_rows.shape[0],) + (1,) * (clean.ndim - 1))
        return torch.where(broadcast, noised, clean)

    def _decoder_boundary(
        self,
        batch: EncodedTaskBatch,
        text_target_rows: Tensor,
        normalized_latents: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if batch.text_latents_raw is None or batch.text_token_ids is None:
            raise RuntimeError(
                "validated text targets are missing raw decoder boundary data"
            )
        # The historical raw space is after scalar text normalization and
        # before optional vision codec statistics.
        decoder_source = (
            normalized_latents
            if self.config.model.text_decoder.input_space == "normalized"
            else batch.text_latents_raw
        )
        raw_latents = _masked_rows(batch.text_latents_raw.detach(), text_target_rows)
        decoder_input = _masked_rows(decoder_source.detach(), text_target_rows)
        token_ids = torch.where(
            text_target_rows[:, None],
            batch.text_token_ids,
            torch.zeros_like(batch.text_token_ids),
        )
        return raw_latents, decoder_input, token_ids

    def _build_physical_batch(self, batch: EncodedTaskBatch) -> TrainingBatch:
        """Build a training contract for an adapter-produced multimodal sequence."""

        physical = batch.physical_layout
        if physical is None:
            raise RuntimeError("physical batch builder requires a physical layout")
        physical.validate()
        batch_size = batch.task_type.shape[0]
        device = physical.token_embeddings.device
        dtype = physical.token_embeddings.dtype
        text_tokens = self.config.data.text_max_length
        absent_role = torch.zeros(batch_size, dtype=torch.long, device=device)
        empty_text = torch.zeros(
            (batch_size, text_tokens), dtype=torch.bool, device=device
        )
        text_stats = torch.full(
            (batch_size, text_tokens),
            int(TextLatentStatsType.PAD_IGNORE),
            dtype=torch.long,
            device=device,
        )
        zero_timestep = torch.zeros(batch_size, dtype=dtype, device=device)
        model_input = MFModelInput(
            vision_latents_norm=None,
            text_latents_norm=None,
            text_prompt_latents_norm=None,
            text_prompt_content_mask=torch.zeros(
                (batch_size, 0), dtype=torch.bool, device=device
            ),
            vision_timestep=zero_timestep,
            text_timestep=zero_timestep.clone(),
            vision_role=absent_role,
            text_role=absent_role.clone(),
            text_content_mask=empty_text,
            text_latent_stats_type=text_stats,
            active_token_mask=physical.active_token_mask,
            text_previous_x0_norm=None,
            text_block_size=None,
            text_segment_ids=None,
            vision_latent_dim=self.config.codecs.vision.latent_dim,
            geometry=self.geometry,
            sequence_contracts=batch.sequence_contracts,
            compiled_sequences=batch.compiled_sequences,
            physical_layout=physical,
        )
        return TrainingBatch(
            task_type=batch.task_type,
            model_input=model_input,
            vision_target_norm=None,
            text_target_norm=None,
            vision_target_mask=torch.zeros(
                (batch_size, self.geometry.vision_tokens),
                dtype=torch.bool,
                device=device,
            ),
            text_target_mask=torch.zeros(
                (batch_size, text_tokens), dtype=torch.bool, device=device
            ),
        ).validate()

    def build(
        self,
        batch: EncodedTaskBatch,
        generator: torch.Generator,
    ) -> TrainingBatch:
        if not isinstance(batch, EncodedTaskBatch):
            raise TypeError("batch must be an EncodedTaskBatch")
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        batch.validate()
        if batch.vision_latent_dim != self.config.codecs.vision.latent_dim:
            raise ValueError(
                "EncodedTaskBatch vision_latent_dim must match the configured codec; "
                f"got {batch.vision_latent_dim} and {self.config.codecs.vision.latent_dim}"
            )

        batch_size = batch.task_type.shape[0]
        if batch_size == 0:
            raise ValueError("EncodedTaskBatch must contain at least one sample")
        device = batch.task_type.device
        registry_device = self.latent_stats_registry.vision_mean.device
        if registry_device != device:
            raise ValueError(
                "EncodedTaskBatch and latent_stats_registry must share a device; "
                f"got {device} and {registry_device}"
            )
        if _canonical_device(generator.device) != _canonical_device(device):
            raise ValueError(
                "generator must be on the EncodedTaskBatch device; "
                f"got {generator.device} and {device}"
            )
        if batch.physical_layout is not None:
            return self._build_physical_batch(batch)
        vision_generator, text_generator = split_generator(generator, 2)
        task_type = batch.task_type
        vision_role, text_role = task_branch_roles(task_type)
        vision_active_rows = vision_role != int(BranchRole.ABSENT)
        text_active_rows = text_role != int(BranchRole.ABSENT)
        vision_target_rows = vision_role == int(BranchRole.TARGET)
        text_target_rows = text_role == int(BranchRole.TARGET)
        vision_condition_rows = vision_role == int(BranchRole.CONDITION)
        text_condition_rows = text_role == int(BranchRole.CONDITION)

        content_mask = batch.text_content_mask
        if content_mask is None:
            raise RuntimeError(
                "validated EncodedTaskBatch is missing text_content_mask"
            )
        text_tokens = content_mask.shape[1]
        if text_tokens != self.config.data.text_max_length:
            raise ValueError(
                "batch text length must match data.text_max_length; "
                f"got {text_tokens} and {self.config.data.text_max_length}"
            )
        stats_type = torch.full(
            (batch_size, text_tokens),
            int(TextLatentStatsType.PAD_IGNORE),
            dtype=torch.long,
            device=device,
        )

        vision_clean: Tensor | None = None
        if batch.vision_latents_raw is not None:
            vision_clean = self.latent_stats_registry.normalize_vision(
                batch.vision_latents_raw
            )
            vision_clean = _masked_rows(vision_clean, vision_active_rows)

        text_clean: Tensor | None = None
        if batch.text_latents_raw is not None:
            if batch.text_latent_stats_type is None or batch.text_token_ids is None:
                raise RuntimeError(
                    "a materialized text branch requires text_latent_stats_type and text_token_ids"
                )
            stats_type = batch.text_latent_stats_type.clone()
            stats_type.masked_fill_(~content_mask, int(TextLatentStatsType.PAD_IGNORE))
            text_clean = self.latent_stats_registry.normalize_text(
                batch.text_latents_raw,
                stats_type,
                content_mask,
            )
            text_clean = torch.where(
                content_mask.unsqueeze(-1),
                text_clean,
                torch.zeros_like(text_clean),
            )

        prompt_mask = batch.text_prompt_content_mask
        if prompt_mask is None:
            prompt_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=device)
        prompt_clean: Tensor | None = None
        if batch.text_prompt_latents_raw is not None:
            if batch.text_prompt_latent_stats_type is None:
                raise RuntimeError("materialized prompt latents require prompt stats")
            prompt_clean = self.latent_stats_registry.normalize_text(
                batch.text_prompt_latents_raw,
                batch.text_prompt_latent_stats_type,
                prompt_mask,
            )
            prompt_clean = torch.where(
                prompt_mask.unsqueeze(-1),
                prompt_clean,
                torch.zeros_like(prompt_clean),
            )

        condition_rows = vision_condition_rows | text_condition_rows
        conditioning_reference = vision_clean if vision_clean is not None else text_clean
        if conditioning_reference is None:
            raise RuntimeError("null-conditioning requires a materialized modality")
        null_conditioning = (
            torch.rand(
                batch_size,
                device=device,
                dtype=conditioning_reference.dtype,
                generator=text_generator,
            )
            < self.config.objective.null_condition_probability
        ) & condition_rows
        if vision_clean is not None:
            vision_clean = _masked_rows(
                vision_clean, ~(null_conditioning & vision_condition_rows)
            )
        if text_clean is not None:
            text_clean = _masked_rows(
                text_clean, ~(null_conditioning & text_condition_rows)
            )
        if prompt_clean is not None:
            prompt_clean = _masked_rows(prompt_clean, ~null_conditioning)

        chunk_routing = batch.chunk_routing_cpu
        if chunk_routing is not None:
            chunk_routing = replace(
                chunk_routing,
                null_conditioning=null_conditioning.detach().cpu(),
            ).validate()

        reference = vision_clean if vision_clean is not None else text_clean
        if reference is None:
            raise ValueError(
                "EncodedTaskBatch must contain at least one present modality"
            )
        shift = self.config.flow.timestep_shift
        vision_timestep = torch.zeros(
            batch_size,
            device=device,
            dtype=reference.dtype,
        )
        sampled_vision_timestep = torch.zeros_like(vision_timestep)
        if vision_clean is not None:
            sampled_vision_timestep = self._sample_shifted_t(
                batch_size=batch_size,
                alpha=shift.image_alpha,
                device=device,
                dtype=reference.dtype,
                generator=vision_generator,
            )
        sampled_text_timestep = torch.zeros_like(vision_timestep)
        if text_clean is not None:
            sampled_text_timestep = self._sample_shifted_t(
                batch_size=batch_size,
                alpha=shift.text_alpha,
                device=device,
                dtype=reference.dtype,
                generator=text_generator,
            )
        vision_target_timestep = torch.where(
            vision_target_rows,
            sampled_vision_timestep,
            torch.zeros_like(vision_timestep),
        )
        vision_timestep = torch.where(
            vision_target_rows,
            vision_target_timestep,
            vision_timestep,
        )
        vision_timestep = torch.where(
            vision_condition_rows,
            torch.ones_like(vision_timestep),
            vision_timestep,
        )
        text_target_timestep = torch.where(
            text_target_rows,
            sampled_text_timestep,
            torch.zeros_like(vision_timestep),
        )
        text_timestep = torch.where(
            text_target_rows,
            text_target_timestep,
            torch.zeros_like(vision_timestep),
        )
        text_timestep = torch.where(
            text_condition_rows,
            torch.ones_like(text_timestep),
            text_timestep,
        )
        block_causal = self.config.flow.text_block_causal
        if batch.text_token_ids is None:
            if bool(text_active_rows.any()):
                raise RuntimeError("block-causal text requires source token ids")
            text_segment_ids = torch.full_like(content_mask, -1, dtype=torch.long)
        else:
            text_segment_ids = _build_text_segment_ids(
                batch.text_token_ids,
                content_mask,
                eos_token_id=self.config.objective.eos_token_id,
            )
        block_count = (
            text_tokens + block_causal.block_size - 1
        ) // block_causal.block_size
        sampled_block_timestep = self._sample_shifted_t(
            batch_size=batch_size * block_count,
            alpha=shift.text_alpha,
            device=device,
            dtype=reference.dtype,
            generator=text_generator,
        ).view(batch_size, block_count)
        text_token_timestep = sampled_block_timestep.repeat_interleave(
            block_causal.block_size, dim=1
        )[:, :text_tokens]
        text_token_timestep = torch.where(
            text_target_rows[:, None] & content_mask,
            text_token_timestep,
            torch.zeros_like(text_token_timestep),
        )
        text_timestep = torch.where(
            text_active_rows,
            torch.ones_like(text_timestep),
            torch.zeros_like(text_timestep),
        )
        image_flow_branch: ImageFlowBranch | None = None
        vision_input: Tensor | None = None
        if vision_clean is not None:
            image_flow_branch = build_image_flow_branch(
                vision_clean,
                vision_timestep,
                vision_target_rows,
                vision_active_rows,
                vision_generator,
                noise_scale=self.config.flow.vision_noise_scale,
            )
            vision_input = image_flow_branch.noisy_input_norm

        text_input: Tensor | None = None
        if text_clean is not None:
            text_input = self._flow_input(
                text_clean,
                text_token_timestep,
                text_target_rows,
                text_generator,
                self.config.flow.text_noise_scale,
            )
            text_input = torch.where(
                content_mask.unsqueeze(-1),
                text_input,
                torch.zeros_like(text_input),
            )

        vision_target_mask = (
            image_flow_branch.target_mask
            if image_flow_branch is not None
            else vision_target_rows[:, None].expand(-1, self.geometry.vision_tokens)
        )
        text_target_mask = text_target_rows[:, None] & content_mask
        vision_target_norm = (
            image_flow_branch.target_norm if image_flow_branch is not None else None
        )
        text_target_norm = (
            torch.where(
                text_target_mask.unsqueeze(-1),
                text_clean,
                torch.zeros_like(text_clean),
            )
            if text_clean is not None
            else None
        )

        decoder_latents_raw: Tensor | None = None
        decoder_input_latents: Tensor | None = None
        decoder_token_ids: Tensor | None = None
        if text_clean is not None:
            (
                decoder_latents_raw,
                decoder_input_latents,
                decoder_token_ids,
            ) = self._decoder_boundary(
                batch,
                text_target_rows,
                text_clean,
            )

        prompt_tokens = prompt_mask.shape[1]
        prompt_layout_tokens = (
            self.geometry.text_prefix_tokens + prompt_tokens if prompt_tokens > 0 else 0
        )
        active_token_mask = torch.zeros(
            (
                batch_size,
                self.geometry.vision_layout_tokens
                + prompt_layout_tokens
                + self.geometry.text_prefix_tokens
                + 2 * text_tokens,
            ),
            dtype=torch.bool,
            device=device,
        )
        active_token_mask[:, : self.geometry.vision_layout_tokens] = vision_active_rows[:, None]
        prompt_prefix_start = self.geometry.vision_layout_tokens
        prompt_content_start = prompt_prefix_start
        if prompt_tokens > 0:
            prompt_content_start += self.geometry.text_prefix_tokens
            active_token_mask[:, prompt_prefix_start:prompt_content_start] = (
                prompt_mask.any(dim=1)[:, None]
            )
        text_prefix_start = prompt_content_start + prompt_tokens
        text_latent_start = text_prefix_start + self.geometry.text_prefix_tokens
        active_token_mask[:, prompt_content_start:text_prefix_start] = prompt_mask
        active_token_mask[:, text_prefix_start:text_latent_start] = text_active_rows[
            :, None
        ]
        clean_end = text_latent_start + text_tokens
        active_token_mask[:, text_latent_start:clean_end] = (
            content_mask & text_active_rows[:, None]
        )
        active_token_mask[:, clean_end:] = content_mask & text_target_rows[:, None]

        model_input = MFModelInput(
            vision_latents_norm=vision_input,
            text_latents_norm=text_input,
            text_prompt_latents_norm=prompt_clean,
            text_prompt_content_mask=prompt_mask,
            vision_timestep=vision_timestep,
            text_timestep=text_timestep,
            vision_role=vision_role,
            text_role=text_role,
            text_content_mask=content_mask,
            text_latent_stats_type=stats_type,
            active_token_mask=active_token_mask,
            text_previous_x0_norm=(
                torch.zeros_like(text_input) if text_input is not None else None
            ),
            null_conditioning=null_conditioning,
            text_clean_latents_norm=(
                text_clean
                if text_clean is not None
                else reference.new_zeros(
                    (batch_size, text_tokens, self.geometry.text_latent_dim)
                )
            ),
            text_token_timestep=text_token_timestep,
            text_block_size=block_causal.block_size,
            vision_latent_dim=self.config.codecs.vision.latent_dim,
            geometry=self.geometry,
            text_segment_ids=text_segment_ids,
            chunk_routing_cpu=chunk_routing,
            sequence_contracts=batch.sequence_contracts,
            compiled_sequences=batch.compiled_sequences,
            physical_layout=batch.physical_layout,
        )
        return TrainingBatch(
            task_type=task_type,
            model_input=model_input,
            vision_target_norm=vision_target_norm,
            text_target_norm=text_target_norm,
            vision_target_mask=vision_target_mask,
            text_target_mask=text_target_mask,
            vision_noisy_input_norm=vision_input,
            text_noisy_input_norm=text_input,
            text_decoder_latents_raw=decoder_latents_raw,
            text_decoder_input_latents=decoder_input_latents,
            text_token_ids=decoder_token_ids,
        ).validate_metadata()
