from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import lru_cache

import torch
from torch import Tensor, nn

from mf.modeling.attention import (
    ModalitySpecificQKVOAttention,
    SharedQKVOAttention,
)
from mf.modeling.layers import (
    ModalitySpecificRMSNorm,
    ModalitySpecificSwiGLU,
    SharedRMSNorm,
    SharedSwiGLU,
)
from mf.modeling.mrope import MROPE_SECTION
from mf.modeling.packing import PackedLayout


def _forward_packed_block(
    block: MFBlock,
    tokens: Tensor,
    packed_layout: PackedLayout,
) -> Tensor:
    return block.forward_packed(replace(packed_layout, tokens=tokens))


@lru_cache(maxsize=1)
def _load_compiled_packed_block() -> Callable[[MFBlock, Tensor, PackedLayout], Tensor]:
    if not hasattr(torch, "compile"):
        raise RuntimeError("compiled MF blocks require torch.compile")
    policy = os.environ.get("MF_COMPILED_BLOCK_MIX_REDUCTION_NON_STRICT", "0")
    if policy not in {"0", "1"}:
        raise ValueError("MF_COMPILED_BLOCK_MIX_REDUCTION_NON_STRICT must be 0 or 1")
    options = (
        {"triton.mix_order_reduction_non_strict_mode": True} if policy == "1" else {}
    )
    return torch.compile(
        _forward_packed_block,
        dynamic=True,
        fullgraph=True,
        **({"options": options} if options else {}),
    )


def run_compiled_packed_block(block: MFBlock, packed_layout: PackedLayout) -> Tensor:
    return _load_compiled_packed_block()(block, packed_layout.tokens, packed_layout)


def prewarm_compiled_packed_block(
    block: MFBlock,
    *,
    device: torch.device,
    dtype: torch.dtype,
    sequence_lengths: tuple[int, ...],
    kernel_block_size: int,
    text_block_size: int,
    vision_token_count: int | None = None,
) -> tuple[float, ...]:
    """Compile one reusable packed block graph for every production Flex bucket."""

    from mf.modeling.chunk_adapter import build_chunk_flex_prewarm_mask

    if device.type != "cuda":
        raise RuntimeError("compiled MF block prewarm requires a CUDA device")
    if dtype not in {torch.bfloat16, torch.float16}:
        raise ValueError("compiled MF block prewarm requires a 16-bit floating dtype")
    if not sequence_lengths:
        return ()

    elapsed_seconds: list[float] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    for sequence_length in sequence_lengths:
        if sequence_length < 2:
            raise ValueError("compiled MF block prewarm lengths must be at least two")
        vision_tokens = (
            min(256, max(1, sequence_length // 4))
            if vision_token_count is None
            else vision_token_count
        )
        if not 0 <= vision_tokens <= sequence_length:
            raise ValueError("compiled MF block prewarm vision tokens are out of range")
        modality_ids = torch.cat(
            (
                torch.zeros(vision_tokens, device=device, dtype=torch.long),
                torch.ones(
                    sequence_length - vision_tokens, device=device, dtype=torch.long
                ),
            )
        )
        active_mask = torch.ones(1, sequence_length, device=device, dtype=torch.bool)
        active_indices = torch.arange(sequence_length, device=device)
        positions = active_indices.view(1, -1).expand(3, -1)
        block_mask = build_chunk_flex_prewarm_mask(
            device,
            sequence_length=sequence_length,
            kernel_block_size=kernel_block_size,
            text_block_size=text_block_size,
            flex_backend=block.flex_backend,
        )
        tokens = torch.randn(
            sequence_length,
            block.hidden_size,
            device=device,
            dtype=dtype,
            generator=generator,
            requires_grad=True,
        )
        layout = PackedLayout(
            positions=positions,
            modality_ids=modality_ids,
            cu_seqlens=torch.tensor(
                (0, sequence_length), device=device, dtype=torch.int32
            ),
            active_indices=active_indices,
            active_mask=active_mask,
            dense_modality_ids=modality_ids.view(1, -1),
            batch_size=1,
            sequence_length=sequence_length,
            hidden_size=block.hidden_size,
            max_seqlen=sequence_length,
            flex_block_mask=block_mask,
            flex_sequence_length=sequence_length,
            tokens=tokens,
        )
        block.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = run_compiled_packed_block(block, layout)
            loss = output.float().square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed_seconds.append(time.perf_counter() - started)

        if not torch.isfinite(output).all():
            raise RuntimeError("compiled MF block prewarm output is non-finite")
        if tokens.grad is None or not torch.isfinite(tokens.grad).all():
            raise RuntimeError("compiled MF block prewarm input gradient is non-finite")
        if torch.count_nonzero(tokens.grad) == 0:
            raise RuntimeError("compiled MF block prewarm input gradient is zero")
        block.zero_grad(set_to_none=True)
        del block_mask, layout, loss, output, tokens

    torch.cuda.empty_cache()
    return tuple(elapsed_seconds)


class MFBlock(nn.Module):
    """Pre-RMSNorm block with packed modality routing."""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        ffn_hidden_size: int = 2816,
        mrope_section: tuple[int, int, int] = MROPE_SECTION,
        rms_norm_eps: float = 1e-6,
        attention_mode: str = "shared",
        ffn_mode: str = "modality_specific",
        attention_implementation: str = "flex",
        flex_backend: str = "triton",
        vision_ffn_hidden_size: int | None = None,
        modality_specific_qk_norms: bool = False,
        modality_ids: Sequence[int] = (0, 1),
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_implementation = attention_implementation
        self.flex_backend = flex_backend
        if attention_mode == "shared":
            self.attention_norm = SharedRMSNorm(hidden_size, rms_norm_eps)
            self.attention = SharedQKVOAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                mrope_section=mrope_section,
            )
        elif attention_mode == "modality_specific":
            self.attention_norm = ModalitySpecificRMSNorm(
                hidden_size, rms_norm_eps, modality_ids
            )
            self.attention = ModalitySpecificQKVOAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                mrope_section=mrope_section,
                modality_specific_qk_norms=modality_specific_qk_norms,
                modality_ids=modality_ids,
            )
        else:
            raise ValueError("attention_mode must be shared or modality_specific")

        if ffn_mode == "shared":
            self.ffn_norm = SharedRMSNorm(hidden_size, rms_norm_eps)
            self.ffn = SharedSwiGLU(hidden_size, ffn_hidden_size)
        elif ffn_mode == "modality_specific":
            self.ffn_norm = ModalitySpecificRMSNorm(
                hidden_size, rms_norm_eps, modality_ids
            )
            self.ffn = ModalitySpecificSwiGLU(
                hidden_size,
                ffn_hidden_size,
                vision_ffn_hidden_size=vision_ffn_hidden_size,
                modality_ids=modality_ids,
            )
        else:
            raise ValueError("ffn_mode must be shared or modality_specific")

    def forward_packed(self, packed_layout: PackedLayout) -> Tensor:
        token_count = (
            packed_layout.flex_sequence_length
            if packed_layout.flex_sequence_length is not None
            else packed_layout.active_indices.shape[0]
        )
        expected_shape = (token_count, self.hidden_size)
        if packed_layout.tokens.shape != expected_shape:
            raise ValueError(f"packed tokens must have shape {expected_shape}")
        if packed_layout.tokens.device != packed_layout.active_mask.device:
            raise ValueError("packed tokens and layout must be on the same device")

        attention_tokens = self.attention_norm(
            packed_layout.tokens,
            packed_layout.vision_indices,
            packed_layout.text_indices,
            modality_indices=packed_layout.modality_indices,
        )
        attention_layout = replace(packed_layout, tokens=attention_tokens)
        attention_output = self.attention(
            attention_layout,
            implementation=self.attention_implementation,
            flex_backend=self.flex_backend,
        )
        hidden_tokens = packed_layout.tokens + attention_output

        ffn_tokens = self.ffn_norm(
            hidden_tokens,
            packed_layout.vision_indices,
            packed_layout.text_indices,
            modality_indices=packed_layout.modality_indices,
        )
        ffn_output = self.ffn(
            ffn_tokens,
            packed_layout.vision_indices,
            packed_layout.text_indices,
            modality_indices=packed_layout.modality_indices,
        )
        return hidden_tokens + ffn_output

    def forward(self, x: Tensor, packed_layout: PackedLayout) -> Tensor:
        expected_shape = (
            packed_layout.batch_size,
            packed_layout.sequence_length,
            self.hidden_size,
        )
        if x.shape != expected_shape:
            raise ValueError(f"x must have shape {expected_shape}")
        if x.device != packed_layout.active_mask.device:
            raise ValueError("x and packed layout must be on the same device")

        output_tokens = self.forward_packed(packed_layout.with_tokens(x))
        return packed_layout.unpack(output_tokens)
