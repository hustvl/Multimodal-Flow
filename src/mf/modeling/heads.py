from __future__ import annotations

import torch
from collections.abc import Mapping

from torch import Tensor, nn

from mf.contracts.batch import (
    TEXT_LATENT_DIM,
    TEXT_PREFIX_TOKENS,
    TEXT_TOKENS,
    VISION_TOKENS,
)
from mf.modeling.layers import make_linear, zero_linear
from mf.registries import OUTPUT_HEAD_REGISTRY, register_output_head


def _build_linear_output_head(hidden_size: int, latent_dim: int) -> nn.Module:
    return zero_linear(make_linear(hidden_size, latent_dim))


register_output_head("linear", _build_linear_output_head, replace=True)


class MFOutputHeads(nn.Module):
    """Apply modality heads only to their latent-token positions."""

    def __init__(
        self,
        hidden_size: int = 1024,
        *,
        vision_latent_dim: int = 768,
        vision_tokens: int = VISION_TOKENS,
        text_latent_dim: int = TEXT_LATENT_DIM,
        text_tokens: int = TEXT_TOKENS,
        text_prefix_tokens: int = TEXT_PREFIX_TOKENS,
        text_layout_copies: int = 1,
        fp32_boundaries: bool = False,
        modality_latent_dims: Mapping[int, int] | None = None,
        modality_head_names: Mapping[int, str] | None = None,
    ) -> None:
        super().__init__()
        if type(vision_latent_dim) is not int or vision_latent_dim <= 0:
            raise ValueError("vision_latent_dim must be a positive integer")
        if type(vision_tokens) is not int or vision_tokens <= 0:
            raise ValueError("vision_tokens must be a positive integer")
        if type(text_latent_dim) is not int or text_latent_dim <= 0:
            raise ValueError("text_latent_dim must be a positive integer")
        if type(text_tokens) is not int or text_tokens <= 0:
            raise ValueError("text_tokens must be a positive integer")
        if type(text_prefix_tokens) is not int or text_prefix_tokens <= 0:
            raise ValueError("text_prefix_tokens must be a positive integer")
        if text_layout_copies not in (1, 2):
            raise ValueError("text_layout_copies must be 1 or 2")
        if type(fp32_boundaries) is not bool:
            raise ValueError("fp32_boundaries must be a Python bool")
        self.hidden_size = hidden_size
        self.text_tokens = text_tokens
        self.text_prefix_tokens = text_prefix_tokens
        self.text_layout_copies = text_layout_copies
        self.fp32_boundaries = fp32_boundaries
        self.vision_latent_dim = vision_latent_dim
        self.vision_tokens = vision_tokens
        self.vision_prefix_tokens = 8
        self.vision_layout_tokens = self.vision_prefix_tokens + vision_tokens
        self.text_latent_dim = text_latent_dim
        self.total_layout_tokens = (
            self.vision_layout_tokens + text_prefix_tokens + text_tokens * text_layout_copies
        )
        self.vision_output_head = zero_linear(
            make_linear(hidden_size, vision_latent_dim)
        )
        self.text_output_head = zero_linear(make_linear(hidden_size, text_latent_dim))
        head_names = modality_head_names or {}
        self.modality_output_heads = nn.ModuleDict()
        for modality_id, latent_dim in (modality_latent_dims or {}).items():
            if modality_id in (0, 1):
                continue
            head_name = head_names.get(modality_id, "linear")
            self.modality_output_heads[str(modality_id)] = (
                OUTPUT_HEAD_REGISTRY.resolve(head_name)(hidden_size, latent_dim)
            )

    def forward_modalities(
        self,
        hidden_states: Tensor,
        packed_routing_layout: object,
        *,
        include_legacy: bool = False,
    ) -> dict[int, Tensor]:
        """Project every routed modality from packed hidden states."""

        if not self.modality_output_heads and not include_legacy:
            return {}
        packed_tokens = packed_routing_layout.gather(hidden_states)
        predictions: dict[int, Tensor] = {}
        for modality_id, indices in packed_routing_layout.modality_indices.items():
            key = str(modality_id)
            head = None
            if modality_id == 0 and include_legacy:
                head = self.vision_output_head
            elif modality_id == 1 and include_legacy:
                head = self.text_output_head
            elif key in self.modality_output_heads:
                head = self.modality_output_heads[key]
            if head is not None and indices.numel():
                predictions[modality_id] = head(
                    packed_tokens.index_select(0, indices)
                )
        return predictions

    def forward(
        self,
        hidden_states: Tensor,
        vision_present: Tensor,
        text_content_mask: Tensor,
        *,
        predict_vision: bool,
        predict_text: bool,
        text_prompt_layout_tokens: int = 0,
    ) -> tuple[Tensor | None, Tensor | None]:
        if type(text_prompt_layout_tokens) is not int or text_prompt_layout_tokens < 0:
            raise ValueError("text_prompt_layout_tokens must be a non-negative integer")
        if text_content_mask.ndim != 2:
            raise ValueError("text_content_mask must have shape [B, T]")
        runtime_text_tokens = text_content_mask.shape[1]
        expected_tokens = (
            self.vision_layout_tokens
            + text_prompt_layout_tokens
            + self.text_prefix_tokens
            + runtime_text_tokens * self.text_layout_copies
        )
        if hidden_states.ndim != 3 or hidden_states.shape != (
            text_content_mask.shape[0],
            expected_tokens,
            self.hidden_size,
        ):
            raise ValueError(
                f"hidden_states must have shape [B, {expected_tokens}, {self.hidden_size}]; "
                f"got {list(hidden_states.shape)} "
                f"(vision_layout={self.vision_layout_tokens}, "
                f"text_prefix={TEXT_PREFIX_TOKENS}, runtime_text={runtime_text_tokens}, "
                f"copies={self.text_layout_copies}, prompt={text_prompt_layout_tokens})"
            )

        vision_prediction = None
        if predict_vision:
            vision_hidden = hidden_states[
                :, self.vision_prefix_tokens : self.vision_layout_tokens
            ]
            if vision_hidden.shape[1] != self.vision_tokens:
                raise RuntimeError(
                    f"vision latent slice must contain {self.vision_tokens} tokens"
                )
            if self.fp32_boundaries:
                with torch.autocast(
                    device_type=vision_hidden.device.type,
                    enabled=False,
                ):
                    vision_prediction = self.vision_output_head(
                        vision_hidden.to(dtype=self.vision_output_head.weight.dtype)
                    )
            else:
                vision_prediction = self.vision_output_head(vision_hidden)
            vision_prediction = vision_prediction * vision_present[:, None, None]

        text_prediction = None
        if predict_text:
            text_start = self.vision_layout_tokens + self.text_prefix_tokens + text_prompt_layout_tokens
            text_start += (self.text_layout_copies - 1) * runtime_text_tokens
            text_hidden = hidden_states[
                :, text_start : text_start + runtime_text_tokens
            ]
            if text_hidden.shape[1] != runtime_text_tokens:
                raise RuntimeError(
                    f"text latent slice must contain {runtime_text_tokens} tokens"
                )
            if self.fp32_boundaries:
                with torch.autocast(
                    device_type=text_hidden.device.type,
                    enabled=False,
                ):
                    text_prediction = self.text_output_head(
                        text_hidden.to(dtype=self.text_output_head.weight.dtype)
                    )
            else:
                text_prediction = self.text_output_head(text_hidden)
            text_prediction = text_prediction * text_content_mask[:, :, None]
        return vision_prediction, text_prediction
