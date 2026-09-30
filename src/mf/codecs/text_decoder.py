from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

TEXT_ENCODER_DIM = 512
TEXT_DECODER_HIDDEN_SIZE = 512
TEXT_DECODER_DEPTH = 6
TEXT_DECODER_HEADS = 8
TEXT_DECODER_HEAD_DIM = 64
TEXT_DECODER_BOTTLENECK = 128
TEXT_DECODER_MAX_LENGTH = 256
TEXT_DECODER_VOCAB_SIZE = 32128
_MLP_RATIO = 4.0


def _forward_text_decoder(
    decoder: LatentTextDecoder,
    raw_latents: Tensor,
    attention_mask: Tensor | None,
) -> Tensor:
    return decoder._forward_eager(raw_latents, attention_mask)


@lru_cache(maxsize=1)
def _load_compiled_text_decoder() -> Callable[
    [LatentTextDecoder, Tensor, Tensor | None], Tensor
]:
    if not hasattr(torch, "compile"):
        raise RuntimeError("compiled text decoder requires torch.compile")
    return torch.compile(
        _forward_text_decoder,
        dynamic=True,
        fullgraph=True,
    )


def run_compiled_text_decoder(
    decoder: LatentTextDecoder,
    raw_latents: Tensor,
    attention_mask: Tensor | None,
) -> Tensor:
    return _load_compiled_text_decoder()(decoder, raw_latents, attention_mask)


def load_text_decoder_checkpoint(
    decoder: LatentTextDecoder,
    checkpoint_path: str | Path,
) -> None:
    path = Path(checkpoint_path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"text decoder checkpoint must be a regular file: {path}")
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"failed to load text decoder checkpoint: {path}") from error
    if not isinstance(payload, Mapping) or payload.get("version") != 1:
        raise ValueError("text decoder checkpoint has an unsupported payload")
    state = payload.get("state")
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in state.items()
    ):
        raise ValueError("text decoder checkpoint state must map names to tensors")
    decoder.load_state_dict(state, strict=True)


def _make_linear(
    in_features: int, out_features: int, *, bias: bool = True
) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden: Tensor) -> Tensor:
        input_dtype = hidden.dtype
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden * torch.rsqrt(variance + self.eps).to(input_dtype)
        return normalized * self.weight.to(input_dtype)


def _rotate_half(hidden: Tensor) -> Tensor:
    paired = hidden.unflatten(-1, (-1, 2))
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


class _TextRotaryEmbedding(nn.Module):
    def __init__(
        self, head_dim: int, max_length: int, *, theta: float = 10000.0
    ) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)[: head_dim // 2]
                / head_dim
            )
        )
        positions = torch.arange(max_length, dtype=torch.float32)
        frequencies = torch.einsum("l,d->ld", positions, inverse_frequency)
        frequencies = frequencies.repeat_interleave(2, dim=-1)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(self, hidden: Tensor) -> Tensor:
        length = hidden.shape[-2]
        cos = self.cos[:length].to(device=hidden.device, dtype=hidden.dtype)
        sin = self.sin[:length].to(device=hidden.device, dtype=hidden.dtype)
        return hidden * cos + _rotate_half(hidden) * sin


class _BottleneckTextProjection(nn.Module):
    def __init__(self, input_dim: int = TEXT_ENCODER_DIM) -> None:
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        self.proj1 = _make_linear(
            input_dim,
            TEXT_DECODER_BOTTLENECK,
            bias=False,
        )
        self.proj2 = _make_linear(
            TEXT_DECODER_BOTTLENECK,
            TEXT_DECODER_HIDDEN_SIZE,
        )

    def forward(self, raw_latents: Tensor) -> Tensor:
        return self.proj2(self.proj1(raw_latents))


class _SelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = TEXT_DECODER_HEADS
        self.head_dim = TEXT_DECODER_HEAD_DIM
        self.qkv = _make_linear(
            TEXT_DECODER_HIDDEN_SIZE,
            3 * TEXT_DECODER_HIDDEN_SIZE,
        )
        self.q_norm = _RMSNorm(TEXT_DECODER_HEAD_DIM)
        self.k_norm = _RMSNorm(TEXT_DECODER_HEAD_DIM)
        self.proj = _make_linear(TEXT_DECODER_HIDDEN_SIZE, TEXT_DECODER_HIDDEN_SIZE)

    def forward(
        self,
        hidden: Tensor,
        *,
        rope: _TextRotaryEmbedding,
        attention_mask: Tensor | None,
    ) -> Tensor:
        batch_size, length, _ = hidden.shape
        qkv = self.qkv(hidden).reshape(
            batch_size,
            length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        query = rope(self.q_norm(query))
        key = rope(self.k_norm(key))
        mask = None if attention_mask is None else attention_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            length,
            TEXT_DECODER_HIDDEN_SIZE,
        )
        return self.proj(attended)


class _SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_dim = int(TEXT_DECODER_HIDDEN_SIZE * _MLP_RATIO * 2 / 3)
        self.w12 = _make_linear(TEXT_DECODER_HIDDEN_SIZE, 2 * hidden_dim)
        self.w3 = _make_linear(hidden_dim, TEXT_DECODER_HIDDEN_SIZE)

    def forward(self, hidden: Tensor) -> Tensor:
        gate, value = self.w12(hidden).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * value)


class _DecoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(TEXT_DECODER_HIDDEN_SIZE)
        self.attention = _SelfAttention()
        self.norm2 = _RMSNorm(TEXT_DECODER_HIDDEN_SIZE)
        self.mlp = _SwiGLU()

    def forward(
        self,
        hidden: Tensor,
        *,
        rope: _TextRotaryEmbedding,
        attention_mask: Tensor | None,
    ) -> Tensor:
        hidden = hidden + self.attention(
            self.norm1(hidden),
            rope=rope,
            attention_mask=attention_mask,
        )
        return hidden + self.mlp(self.norm2(hidden))


class LatentTextDecoder(nn.Module):
    """Latent-to-token decoder whose input width follows the text codec."""

    def __init__(
        self,
        *,
        input_dim: int = TEXT_ENCODER_DIM,
        max_length: int = TEXT_DECODER_MAX_LENGTH,
        fp32_boundaries: bool = False,
        compile_forward: bool = False,
    ) -> None:
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if type(fp32_boundaries) is not bool:
            raise ValueError("fp32_boundaries must be a Python bool")
        if type(compile_forward) is not bool:
            raise ValueError("compile_forward must be a Python bool")
        self.fp32_boundaries = fp32_boundaries
        self.compile_forward = compile_forward
        self.text_encoder_dim = input_dim
        self.hidden_size = TEXT_DECODER_HIDDEN_SIZE
        self.depth = TEXT_DECODER_DEPTH
        self.num_heads = TEXT_DECODER_HEADS
        self.head_dim = TEXT_DECODER_HEAD_DIM
        self.mlp_ratio = _MLP_RATIO
        self.bottleneck_dim = TEXT_DECODER_BOTTLENECK
        self.max_length = max_length
        self.vocab_size = TEXT_DECODER_VOCAB_SIZE

        self.text_proj = _BottleneckTextProjection(input_dim)
        self.rope = _TextRotaryEmbedding(self.head_dim, self.max_length)
        self.blocks = nn.ModuleList(_DecoderBlock() for _ in range(self.depth))
        self.norm = _RMSNorm(self.hidden_size)

        self.proj_kernel = nn.Parameter(
            torch.empty(self.hidden_size, self.text_encoder_dim)
        )
        self.proj_bias = nn.Parameter(torch.zeros(self.text_encoder_dim))
        self.unembed_kernel = nn.Parameter(
            torch.empty(self.text_encoder_dim, self.vocab_size)
        )
        self.unembed_bias = nn.Parameter(torch.zeros(self.vocab_size))
        nn.init.xavier_uniform_(self.proj_kernel)
        nn.init.xavier_uniform_(self.unembed_kernel)

    def forward(
        self,
        raw_latents: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if self.compile_forward:
            return run_compiled_text_decoder(self, raw_latents, attention_mask)
        return self._forward_eager(raw_latents, attention_mask)

    def _forward_eager(
        self,
        raw_latents: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if raw_latents.ndim != 3 or raw_latents.shape[-1] != self.text_encoder_dim:
            raise ValueError(
                "raw_latents must have shape "
                f"[N, L, {self.text_encoder_dim}]; got {list(raw_latents.shape)}"
            )
        batch_size, length, _ = raw_latents.shape
        if length > self.max_length:
            raise ValueError(
                f"raw latent sequence length {length} exceeds max_length={self.max_length}"
            )
        if attention_mask is not None:
            expected_mask = (batch_size, length)
            if tuple(attention_mask.shape) != expected_mask:
                raise ValueError(
                    "attention_mask must have shape [N, L]; "
                    f"expected {list(expected_mask)}, got {list(attention_mask.shape)}"
                )
            if attention_mask.device != raw_latents.device:
                raise ValueError(
                    "attention_mask and raw_latents must be on the same device"
                )
            attention_mask = attention_mask.to(dtype=torch.bool)

        parameter_dtype = self.text_proj.proj1.weight.dtype
        if self.fp32_boundaries:
            with torch.autocast(
                device_type=raw_latents.device.type,
                enabled=False,
            ):
                hidden = self.text_proj(raw_latents.float())
        else:
            hidden = self.text_proj(raw_latents.to(dtype=parameter_dtype))
        for block in self.blocks:
            hidden = block(hidden, rope=self.rope, attention_mask=attention_mask)
        if self.fp32_boundaries:
            with torch.autocast(
                device_type=hidden.device.type,
                enabled=False,
            ):
                hidden = self.norm(hidden.float()).float()
                projected = F.gelu(
                    hidden @ self.proj_kernel + self.proj_bias,
                    approximate="tanh",
                )
                return projected @ self.unembed_kernel + self.unembed_bias
        hidden = self.norm(hidden)
        projected = F.gelu(
            hidden @ self.proj_kernel + self.proj_bias,
            approximate="tanh",
        )
        return projected @ self.unembed_kernel + self.unembed_bias


def prewarm_compiled_text_decoder(
    decoder: LatentTextDecoder,
    *,
    device: torch.device,
    dtype: torch.dtype,
    block_size: int,
    active_blocks: int,
    use_attention_mask: bool,
) -> float:
    """Compile the production block-local decoder graph before DDP collectives."""

    if not decoder.compile_forward:
        return 0.0
    if device.type != "cuda":
        raise RuntimeError("compiled text decoder prewarm requires a CUDA device")
    if dtype not in {torch.bfloat16, torch.float16}:
        raise ValueError(
            "compiled text decoder prewarm requires a 16-bit floating dtype"
        )
    if block_size <= 0 or active_blocks <= 0:
        raise ValueError("block_size and active_blocks must be positive")

    generator = torch.Generator(device=device).manual_seed(0)
    raw_latents = torch.randn(
        active_blocks,
        block_size,
        decoder.text_encoder_dim,
        device=device,
        dtype=torch.float32,
        generator=generator,
        requires_grad=True,
    )
    attention_mask = (
        torch.ones(active_blocks, block_size, device=device, dtype=torch.bool)
        if use_attention_mask
        else None
    )
    decoder.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=dtype):
        logits = decoder(raw_latents, attention_mask)
        loss = logits.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    decoder.zero_grad(set_to_none=True)
    raw_latents.grad = None
    return elapsed
