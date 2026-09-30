from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn

from mf.contracts.batch import (
    TEXT_TOKENS,
)
from mf.contracts.model import MFModelInput, MFOutput
from mf.contracts.geometry import GeometryContract
from mf.latents.stats import LatentStatsRegistry
from mf.modeling.backbone import MFBackbone
from mf.modeling.chunk_adapter import build_chunk_causal_layout
from mf.modeling.chunk_causal_layout import ChunkCausalRoutingLayout
from mf.modeling.embeddings import MFInputEmbeddings
from mf.modeling.heads import MFOutputHeads
from mf.modeling.legacy_adapter import LegacyImageTextAdapter
from mf.modeling.mrope import MROPE_SECTION
from mf.contracts.sequence import MODALITY_REGISTRY


class OptimizerParameterRoles(NamedTuple):
    """Explicit, disjoint trainable parameter owners for optimizer routing."""

    muon: tuple[nn.Parameter, ...]
    adamw: tuple[nn.Parameter, ...]


class MFModel(nn.Module):
    """Multimodal Flow backbone with normalized inputs."""

    def __init__(
        self,
        latent_stats_registry: LatentStatsRegistry,
        *,
        vision_latent_dim: int | None = None,
        vision_tokens: int = 256,
        text_latent_dim: int = 512,
        vision_grid_size: tuple[int, int] = (16, 16),
        hidden_size: int = 1024,
        depth: int = 28,
        num_heads: int = 16,
        head_dim: int = 64,
        ffn_hidden_size: int = 2816,
        mrope_section: tuple[int, int, int] = MROPE_SECTION,
        attention_mode: str = "shared",
        ffn_mode: str = "modality_specific",
        text_tokens: int = TEXT_TOKENS,
        text_input_bottleneck_dim: int = 128,
        text_input_projection_mode: Literal["bottleneck", "linear"] = "bottleneck",
        blocks: Sequence[nn.Module] | None = None,
        fp32_boundaries: bool = False,
        gradient_checkpointing: bool = False,
        compile_packed_blocks: bool = False,
        sequence_layout: Literal["chunk_causal"] = "chunk_causal",
        image_chunk_conditioning: Literal[
            "token_additive",
            "legacy_active_prefix",
        ] = "token_additive",
        t2i_chunk_semantics: Literal[
            "chunk_native",
            "legacy_block_exact",
        ] = "chunk_native",
        block_causal_attention_backend: Literal["flex"] = "flex",
        block_causal_flex_backend: Literal["triton", "fa4"] = "triton",
        block_causal_flex_kernel_block_size: int = 128,
        block_causal_flex_sequence_bucket_size: int = 4096,
        block_causal_flex_fixed_sequence_length: bool = False,
        text_block_size: int = 8,
    ) -> None:
        super().__init__()
        if not isinstance(latent_stats_registry, LatentStatsRegistry):
            raise TypeError("latent_stats_registry must be a LatentStatsRegistry")
        if vision_latent_dim is None:
            vision_latent_dim = latent_stats_registry.vision_latent_dim
        elif type(vision_latent_dim) is not int or vision_latent_dim <= 0:
            raise ValueError("vision_latent_dim must be a positive integer")
        if vision_latent_dim != latent_stats_registry.vision_latent_dim:
            raise ValueError(
                "vision latent dimension must match the latent statistics; "
                f"got {vision_latent_dim} and {latent_stats_registry.vision_latent_dim}"
            )
        if vision_tokens != latent_stats_registry.vision_tokens:
            raise ValueError(
                "vision token count must match the latent statistics; "
                f"got {vision_tokens} and {latent_stats_registry.vision_tokens}"
            )
        if text_latent_dim != latent_stats_registry.text_latent_dim:
            raise ValueError(
                "text latent dimension must match the latent statistics; "
                f"got {text_latent_dim} and {latent_stats_registry.text_latent_dim}"
            )
        self.geometry = GeometryContract(
            vision_tokens=vision_tokens,
            vision_latent_dim=vision_latent_dim,
            text_latent_dim=text_latent_dim,
            vision_grid_size=tuple(vision_grid_size),
        ).validate()
        if type(text_tokens) is not int or text_tokens <= 0:
            raise ValueError("text_tokens must be a positive integer")
        if sequence_layout != "chunk_causal":
            raise ValueError("sequence_layout must be chunk_causal")
        if image_chunk_conditioning not in {
            "token_additive",
            "legacy_active_prefix",
        }:
            raise ValueError("unknown image chunk conditioning mode")
        if t2i_chunk_semantics not in {"chunk_native", "legacy_block_exact"}:
            raise ValueError("unknown T2I chunk semantics")
        if (
            t2i_chunk_semantics == "legacy_block_exact"
            and image_chunk_conditioning != "legacy_active_prefix"
        ):
            raise ValueError(
                "legacy_block_exact T2I semantics require legacy_active_prefix image conditioning"
            )
        if block_causal_attention_backend != "flex":
            raise ValueError("unknown block-causal attention backend")
        if block_causal_flex_backend not in {"triton", "fa4"}:
            raise ValueError("unknown Flex backend")
        if block_causal_flex_kernel_block_size not in {64, 128}:
            raise ValueError("block-causal Flex kernel block size must be 64 or 128")
        if (
            type(block_causal_flex_sequence_bucket_size) is not int
            or block_causal_flex_sequence_bucket_size <= 0
            or block_causal_flex_sequence_bucket_size
            % block_causal_flex_kernel_block_size
        ):
            raise ValueError(
                "block-causal Flex sequence bucket must be a positive block multiple"
            )
        if type(block_causal_flex_fixed_sequence_length) is not bool:
            raise TypeError("block-causal fixed Flex sequence flag must be a bool")
        if type(text_block_size) is not int or text_block_size <= 0:
            raise ValueError("text_block_size must be a positive integer")
        self.sequence_layout = sequence_layout
        self.image_chunk_conditioning = image_chunk_conditioning
        self.t2i_chunk_semantics = t2i_chunk_semantics
        self.block_causal_attention_backend = block_causal_attention_backend
        self.block_causal_flex_backend = block_causal_flex_backend
        self.block_causal_flex_kernel_block_size = block_causal_flex_kernel_block_size
        self.block_causal_flex_sequence_bucket_size = (
            block_causal_flex_sequence_bucket_size
        )
        self.block_causal_flex_fixed_sequence_length = (
            block_causal_flex_fixed_sequence_length
        )
        self.text_block_size = text_block_size
        self.vision_latent_dim = vision_latent_dim
        self.vision_tokens = self.geometry.vision_tokens
        self.text_latent_dim = self.geometry.text_latent_dim
        self.text_tokens = text_tokens
        self.text_layout_copies = 2
        self.text_layout_tokens = (
            self.geometry.text_prefix_tokens + text_tokens * self.text_layout_copies
        )
        self.latent_stats_registry = latent_stats_registry
        self.embeddings = MFInputEmbeddings(
            hidden_size=hidden_size,
            vision_latent_dim=self.vision_latent_dim,
            vision_tokens=self.geometry.vision_tokens,
            vision_grid_size=self.geometry.vision_grid_size,
            text_latent_dim=self.geometry.text_latent_dim,
            text_tokens=text_tokens,
            text_input_bottleneck_dim=text_input_bottleneck_dim,
            text_input_projection_mode=text_input_projection_mode,
            fp32_boundaries=fp32_boundaries,
        )
        self.backbone = MFBackbone(
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            ffn_hidden_size=ffn_hidden_size,
            mrope_section=mrope_section,
            attention_mode=attention_mode,
            ffn_mode=ffn_mode,
            attention_implementation=block_causal_attention_backend,
            flex_backend=block_causal_flex_backend,
            fp32_boundaries=fp32_boundaries,
            gradient_checkpointing=gradient_checkpointing,
            compile_packed_blocks=compile_packed_blocks,
            blocks=blocks,
        )
        self.heads = MFOutputHeads(
            hidden_size=hidden_size,
            vision_latent_dim=self.vision_latent_dim,
            vision_tokens=self.geometry.vision_tokens,
            text_latent_dim=self.geometry.text_latent_dim,
            text_tokens=text_tokens,
            text_prefix_tokens=self.geometry.text_prefix_tokens,
            text_layout_copies=self.text_layout_copies,
            fp32_boundaries=fp32_boundaries,
            modality_latent_dims={
                definition.stable_id: definition.latent_dim
                for definition in MODALITY_REGISTRY.definitions()
                if definition.latent_dim is not None
            },
            modality_head_names={
                definition.stable_id: definition.head_name
                for definition in MODALITY_REGISTRY.definitions()
            },
        )
        if image_chunk_conditioning == "token_additive":
            self.embeddings.learned_time.requires_grad_(False)
        self.legacy_adapter = LegacyImageTextAdapter(self)

    def optimizer_parameter_roles(self) -> OptimizerParameterRoles:
        """Assign Muon to every trainable non-embedding matrix."""

        embedding_weight_ids = {
            id(module.weight)
            for module in self.modules()
            if isinstance(module, nn.Embedding) and module.weight.requires_grad
        }
        muon: list[nn.Parameter] = []
        adamw: list[nn.Parameter] = []
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            target = (
                muon
                if parameter.ndim == 2 and id(parameter) not in embedding_weight_ids
                else adamw
            )
            target.append(parameter)

        owned = muon + adamw
        owned_ids = [id(parameter) for parameter in owned]
        trainable_ids = {
            id(parameter) for parameter in self.parameters() if parameter.requires_grad
        }
        if (
            any(parameter.ndim != 2 for parameter in muon)
            or len(owned_ids) != len(set(owned_ids))
            or set(owned_ids) != trainable_ids
        ):
            raise RuntimeError(
                "optimizer roles must own every trainable parameter exactly once"
            )
        return OptimizerParameterRoles(muon=tuple(muon), adamw=tuple(adamw))

    def prepare_routing_layout(
        self,
        model_input: MFModelInput,
    ) -> ChunkCausalRoutingLayout:
        """Build token-free compact metadata for an unchanged prediction layout."""

        return build_chunk_causal_layout(
            model_input,
            hidden_size=self.backbone.hidden_size,
            attention_backend=self.block_causal_attention_backend,
            flex_backend=self.block_causal_flex_backend,
            flex_kernel_block_size=self.block_causal_flex_kernel_block_size,
            flex_sequence_bucket_size=self.block_causal_flex_sequence_bucket_size,
            flex_fixed_sequence_length=self.block_causal_flex_fixed_sequence_length,
            image_chunk_conditioning=self.image_chunk_conditioning,
            t2i_chunk_semantics=self.t2i_chunk_semantics,
        )

    def denormalize_vision_prediction(self, normalized: Tensor) -> Tensor:
        return self.latent_stats_registry.denormalize_vision(normalized)

    def denormalize_text_prediction(
        self,
        normalized: Tensor,
        stats_type: Tensor,
        content_mask: Tensor,
    ) -> Tensor:
        return self.latent_stats_registry.denormalize_text(
            normalized,
            stats_type,
            content_mask,
        )

    def predict_normalized(
        self,
        model_input: MFModelInput,
        routing_layout: ChunkCausalRoutingLayout | None = None,
    ) -> MFOutput:
        if routing_layout is None:
            return self.forward(model_input)
        return self.forward(model_input, routing_layout=routing_layout)

    def forward(
        self,
        model_input: MFModelInput,
        routing_layout: ChunkCausalRoutingLayout | None = None,
    ) -> MFOutput:
        """Return normalized x0 for each active modality branch."""

        if routing_layout is None:
            routing_layout = self.prepare_routing_layout(model_input)
        if type(routing_layout) is not ChunkCausalRoutingLayout:
            raise TypeError("routing_layout must be a ChunkCausalRoutingLayout")
        if model_input.physical_layout is not None:
            physical = model_input.physical_layout
            routing_layout.validate_cache_identity(
                (
                    physical.active_token_mask,
                    physical.position_ids,
                    physical.sequence_ids,
                    physical.chunk_indices,
                    physical.modality_ids,
                    physical.view_ids,
                    physical.block_indices,
                    physical.target_mask,
                )
            )
            packed_routing_layout = routing_layout.sequence_plan.packed_routing_layout
            hidden_states = physical.token_embeddings
            active_token_mask = routing_layout.sequence_plan.active_token_mask
            hidden_states = hidden_states * active_token_mask.unsqueeze(-1).to(
                dtype=hidden_states.dtype
            )
            hidden_states = self.backbone(
                hidden_states, active_token_mask, packed_routing_layout
            )
            modality_pred_norm = self.heads.forward_modalities(
                hidden_states,
                packed_routing_layout,
                include_legacy=True,
            )
            modality_target_indices = None
            if packed_routing_layout.target_mask is not None:
                modality_target_indices = {}
                for modality_id, indices in packed_routing_layout.modality_indices.items():
                    target_positions = torch.nonzero(
                        packed_routing_layout.target_mask.index_select(0, indices),
                        as_tuple=False,
                    ).flatten()
                    if target_positions.numel():
                        modality_target_indices[modality_id] = target_positions
            return MFOutput(
                vision_pred_norm=None,
                text_pred_norm=None,
                active_token_mask=active_token_mask,
                vision_latent_dim=self.vision_latent_dim,
                geometry=self.geometry,
                modality_pred_norm=modality_pred_norm,
                modality_target_indices=modality_target_indices,
                physical_layout=physical,
            ).validate()
        return self.legacy_adapter.forward(model_input, routing_layout)
