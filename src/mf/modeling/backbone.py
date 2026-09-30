from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from mf.modeling.block import MFBlock, run_compiled_packed_block
from mf.modeling.mrope import MROPE_SECTION
from mf.modeling.layers import ModalitySpecificRMSNorm
from mf.modeling.packing import PackedRoutingLayout
from mf.contracts.sequence import MODALITY_REGISTRY


class MFBackbone(nn.Module):
    """Pack one explicit token layout and apply the shared MF block stack."""

    def __init__(
        self,
        hidden_size: int = 1024,
        depth: int = 28,
        num_heads: int = 16,
        head_dim: int = 64,
        ffn_hidden_size: int = 2816,
        mrope_section: tuple[int, int, int] = MROPE_SECTION,
        attention_mode: str = "shared",
        ffn_mode: str = "modality_specific",
        attention_implementation: str = "flex",
        flex_backend: str = "triton",
        *,
        fp32_boundaries: bool = False,
        gradient_checkpointing: bool = False,
        compile_packed_blocks: bool = False,
        blocks: Sequence[nn.Module] | None = None,
        vision_ffn_hidden_size: int | None = None,
        full_mot_norms: bool = False,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        if type(fp32_boundaries) is not bool:
            raise ValueError("fp32_boundaries must be a Python bool")
        self.fp32_boundaries = fp32_boundaries
        if type(gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be a Python bool")
        self.gradient_checkpointing = gradient_checkpointing
        if type(compile_packed_blocks) is not bool:
            raise ValueError("compile_packed_blocks must be a Python bool")
        if compile_packed_blocks and gradient_checkpointing:
            raise ValueError("compiled packed blocks cannot use gradient checkpointing")
        self.compile_packed_blocks = compile_packed_blocks
        modality_ids = MODALITY_REGISTRY.ids()
        if blocks is None:
            block_list = [
                MFBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ffn_hidden_size=ffn_hidden_size,
                    mrope_section=mrope_section,
                    attention_mode=attention_mode,
                    ffn_mode=ffn_mode,
                    attention_implementation=attention_implementation,
                    flex_backend=flex_backend,
                    vision_ffn_hidden_size=vision_ffn_hidden_size,
                    modality_specific_qk_norms=full_mot_norms,
                    modality_ids=modality_ids,
                )
                for _ in range(depth)
            ]
        else:
            block_list = list(blocks)
            if len(block_list) != depth:
                raise ValueError("injected blocks must contain exactly depth modules")
        self.hidden_size = hidden_size
        self.depth = depth
        self.blocks = nn.ModuleList(block_list)
        self.final_norm = (
            ModalitySpecificRMSNorm(
                hidden_size,
                eps=1e-6,
                modality_ids=modality_ids,
            )
            if full_mot_norms
            else nn.RMSNorm(hidden_size, eps=1e-6)
        )
        self._packed_block_stack = all(type(block) is MFBlock for block in block_list)

    def _apply_final_norm(
        self,
        hidden_states: Tensor,
        routing_layout: PackedRoutingLayout,
    ) -> Tensor:
        if isinstance(self.final_norm, ModalitySpecificRMSNorm):
            packed = routing_layout.with_tokens(hidden_states)
            normalized = self.final_norm(
                packed.tokens,
                packed.vision_indices,
                packed.text_indices,
                modality_indices=packed.modality_indices,
            )
            return routing_layout.unpack(normalized)
        return self.final_norm(hidden_states)

    def _finalize(
        self,
        hidden_states: Tensor,
        active_token_mask: Tensor,
        routing_layout: PackedRoutingLayout,
    ) -> Tensor:
        if self.fp32_boundaries:
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                hidden_states = self._apply_final_norm(
                    hidden_states.float(), routing_layout
                )
        else:
            hidden_states = self._apply_final_norm(hidden_states, routing_layout)
        return hidden_states * active_token_mask.unsqueeze(-1).to(
            dtype=hidden_states.dtype
        )

    def forward(
        self,
        x: Tensor,
        active_token_mask: Tensor,
        routing_layout: PackedRoutingLayout,
    ) -> Tensor:
        """Run any validated physical layout through the shared block stack."""

        if x.ndim != 3 or x.shape[0] != active_token_mask.shape[0]:
            raise ValueError("x must have shape [B, L, H] matching active_token_mask")
        if x.shape[1:] != (active_token_mask.shape[1], self.hidden_size):
            raise ValueError(
                "x token and hidden dimensions do not match the routed layout"
            )
        if type(routing_layout) is not PackedRoutingLayout:
            raise TypeError(
                "routing_layout must be a metadata-only PackedRoutingLayout"
            )
        if routing_layout.active_mask is not active_token_mask:
            raise ValueError(
                "routing_layout must reference the same active_token_mask tensor"
            )

        if self._packed_block_stack:
            packed_layout = routing_layout.with_tokens(x)
            for block in self.blocks:
                if self.gradient_checkpointing and self.training:
                    block_tokens = checkpoint(
                        lambda tokens, current_block=block, layout=packed_layout: (
                            current_block.forward_packed(replace(layout, tokens=tokens))
                        ),
                        packed_layout.tokens,
                        use_reentrant=False,
                    )
                elif self.compile_packed_blocks:
                    block_tokens = run_compiled_packed_block(block, packed_layout)
                else:
                    block_tokens = block.forward_packed(packed_layout)
                packed_layout = replace(packed_layout, tokens=block_tokens)
            hidden_states = routing_layout.unpack(packed_layout.tokens)
        else:
            layout = routing_layout.with_tokens(x)
            hidden_states = x
            for block in self.blocks:
                if self.gradient_checkpointing and self.training:
                    hidden_states = checkpoint(
                        lambda tokens, current_block=block, current_layout=layout: (
                            current_block(tokens, current_layout)
                        ),
                        hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = block(hidden_states, layout)
        return self._finalize(hidden_states, active_token_mask, routing_layout)
