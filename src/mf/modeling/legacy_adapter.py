"""Compatibility adapter for the original image/text model contract.

The physical sequence path is the canonical MF model interface. This adapter
keeps the paper recipe and older callers working without placing its
image/text layout assumptions in the main model forward method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mf.contracts.batch import BranchRole
from mf.contracts.model import MFModelInput, MFOutput
from mf.modeling.chunk_causal_layout import ChunkCausalRoutingLayout

if TYPE_CHECKING:
    from mf.modeling.model import MFModel


class LegacyImageTextAdapter:
    """Translate the legacy image/text input contract into model activations."""

    __slots__ = ("model",)

    def __init__(self, model: MFModel) -> None:
        self.model = model

    def forward(
        self,
        model_input: MFModelInput,
        routing_layout: ChunkCausalRoutingLayout,
    ) -> MFOutput:
        model = self.model
        routing_layout.validate_cache_identity(
            (
                model_input.active_token_mask,
                model_input.vision_role,
                model_input.text_role,
                model_input.text_content_mask,
                model_input.text_prompt_content_mask,
                model_input.text_segment_ids,
                model_input.null_conditioning,
            )
        )
        packed_routing_layout = routing_layout.sequence_plan.packed_routing_layout
        has_vision = packed_routing_layout.vision_indices.numel() > 0
        has_text = packed_routing_layout.text_indices.numel() > 0
        if not has_vision and not has_text:
            return MFOutput(
                vision_pred_norm=None,
                text_pred_norm=None,
                active_token_mask=model_input.active_token_mask,
                vision_latent_dim=model.vision_latent_dim,
                geometry=model.geometry,
            ).validate()

        vision_tokens = None
        if has_vision:
            if model_input.vision_latents_norm is None:
                raise RuntimeError("metadata-validated vision branch is missing latents")
            if model.image_chunk_conditioning == "legacy_active_prefix":
                vision_tokens = model.embeddings.embed_vision(
                    model_input.vision_latents_norm,
                    model_input.vision_timestep,
                )
            else:
                vision_content = model.embeddings.embed_chunk_vision(
                    model_input.vision_latents_norm,
                    model_input.vision_timestep,
                )
                vision_tokens = torch.cat(
                    (
                        vision_content.new_zeros(
                            vision_content.shape[0],
                            model.geometry.vision_layout_tokens - vision_content.shape[1],
                            vision_content.shape[2],
                        ),
                        vision_content,
                    ),
                    dim=1,
                )

        text_tokens = None
        if has_text:
            if (
                model_input.text_latents_norm is None
                or model_input.text_previous_x0_norm is None
            ):
                raise RuntimeError("metadata-validated text branch is missing latent state")
            if (
                model_input.text_clean_latents_norm is None
                or model_input.text_token_timestep is None
                or model_input.text_block_size != model.text_block_size
            ):
                raise RuntimeError("block-causal text state is incomplete")
            clean_content = model.embeddings.embed_chunk_text(
                model_input.text_clean_latents_norm,
                previous_x0_norm=torch.zeros_like(model_input.text_clean_latents_norm),
                token_timestep=torch.ones_like(model_input.text_token_timestep),
            )
            noisy_content = model.embeddings.embed_chunk_text(
                model_input.text_latents_norm,
                previous_x0_norm=model_input.text_previous_x0_norm,
                token_timestep=model_input.text_token_timestep,
            )
            prefix = clean_content.new_zeros(
                clean_content.shape[0],
                model.geometry.text_prefix_tokens,
                clean_content.shape[2],
            )
            if model.t2i_chunk_semantics == "legacy_block_exact":
                t2i_rows = (model_input.vision_role == int(BranchRole.TARGET)) & (
                    model_input.text_role == int(BranchRole.CONDITION)
                )
                t2i_row_mask = t2i_rows[:, None, None]
                legacy_prefix = model.embeddings.embed_text_prefix(
                    model_input.text_timestep,
                )
                legacy_clean_content = model.embeddings.embed_text_content(
                    model_input.text_clean_latents_norm,
                    previous_x0_norm=torch.zeros_like(
                        model_input.text_clean_latents_norm
                    ),
                )
                legacy_clean_time = model.embeddings.timestep_embedder(
                    torch.ones_like(model_input.text_token_timestep).reshape(-1)
                ).view_as(legacy_clean_content)
                prefix = torch.where(t2i_row_mask, legacy_prefix, prefix)
                clean_content = torch.where(
                    t2i_row_mask,
                    legacy_clean_content + legacy_clean_time,
                    clean_content,
                )
            text_tokens = torch.cat((prefix, clean_content, noisy_content), dim=1)

        prompt_layout = None
        if model_input.text_prompt_latents_norm is not None:
            prompt_content = model.embeddings.embed_chunk_text(
                model_input.text_prompt_latents_norm,
                previous_x0_norm=torch.zeros_like(
                    model_input.text_prompt_latents_norm,
                ),
                token_timestep=torch.ones(
                    model_input.text_prompt_latents_norm.shape[:2],
                    dtype=model_input.text_timestep.dtype,
                    device=model_input.text_timestep.device,
                ),
            )
            prompt_prefix = prompt_content.new_zeros(
                prompt_content.shape[0],
                model.geometry.text_prefix_tokens,
                prompt_content.shape[2],
            )
            prompt_layout = torch.cat((prompt_prefix, prompt_content), dim=1)

        reference = vision_tokens
        if reference is None:
            reference = prompt_layout
        if reference is None:
            reference = text_tokens
        if reference is None:
            raise RuntimeError("at least one validated modality must be active")
        batch_size = model_input.vision_role.shape[0]
        if vision_tokens is None:
            vision_tokens = reference.new_zeros(
                (batch_size, model.geometry.vision_layout_tokens, model.embeddings.hidden_size)
            )
        prompt_capacity = model_input.text_prompt_content_mask.shape[1]
        prompt_layout_tokens = (
            model.geometry.text_prefix_tokens + prompt_capacity
            if prompt_capacity > 0
            else 0
        )
        if prompt_layout is None:
            prompt_layout = reference.new_zeros(
                (batch_size, prompt_layout_tokens, model.embeddings.hidden_size)
            )
        if text_tokens is None:
            text_tokens = reference.new_zeros(
                (batch_size, model.text_layout_tokens, model.embeddings.hidden_size)
            )

        hidden_states = torch.cat((vision_tokens, prompt_layout, text_tokens), dim=1)
        active_token_mask = routing_layout.sequence_plan.active_token_mask
        hidden_states = hidden_states * active_token_mask.unsqueeze(-1).to(
            dtype=hidden_states.dtype
        )
        hidden_states = model.backbone(
            hidden_states, active_token_mask, packed_routing_layout
        )
        vision_pred_norm, text_pred_norm = model.heads(
            hidden_states,
            model_input.vision_present,
            model_input.text_content_mask,
            predict_vision=has_vision,
            predict_text=has_text,
            text_prompt_layout_tokens=prompt_layout_tokens,
        )
        modality_pred_norm = model.heads.forward_modalities(
            hidden_states,
            packed_routing_layout,
        )
        return MFOutput(
            vision_pred_norm=vision_pred_norm,
            text_pred_norm=text_pred_norm,
            active_token_mask=model_input.active_token_mask,
            vision_latent_dim=model.vision_latent_dim,
            geometry=model.geometry,
            modality_pred_norm=modality_pred_norm,
        ).validate()
