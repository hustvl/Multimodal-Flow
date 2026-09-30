from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


WeightInitializer = Callable[[Tensor], None]


def xavier_uniform_(weight: Tensor) -> None:
    nn.init.xavier_uniform_(weight)


def normal_002_(weight: Tensor) -> None:
    nn.init.normal_(weight, mean=0.0, std=0.02)


def make_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    weight_initializer: WeightInitializer = xavier_uniform_,
) -> nn.Linear:
    """Create a linear layer with the model's explicit initialization."""

    layer = nn.Linear(in_features, out_features, bias=bias)
    weight_initializer(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def zero_linear(layer: nn.Linear) -> nn.Linear:
    """Zero-initialize an output projection without replacing its parameters."""

    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def route_by_modality(
    x: Tensor,
    vision_indices: Tensor,
    text_indices: Tensor,
    vision_module: nn.Module,
    text_module: nn.Module,
) -> Tensor:
    """Apply independent modules to packed vision/text tokens and restore order."""

    if x.ndim < 2:
        raise ValueError("x must have a leading token dimension")
    for name, indices in (
        ("vision_indices", vision_indices),
        ("text_indices", text_indices),
    ):
        if indices.ndim != 1:
            raise ValueError(f"{name} must have shape [route_tokens]")
        if indices.dtype is not torch.long:
            raise ValueError(f"{name} must have dtype torch.long")
        if indices.device != x.device:
            raise ValueError(f"{name} must be on the same device as x")

    if vision_indices.numel() == 0 and text_indices.numel() != 0:
        text_output = text_module(x.index_select(0, text_indices))
        if text_output.ndim != 2:
            raise RuntimeError("routed modules must return rank-2 tensors")
        output = text_output.new_zeros(x.shape[0], text_output.shape[-1])
        output = output.index_copy(0, text_indices, text_output)
        return output + _zero_parameter_dependency(vision_module, output)
    if text_indices.numel() == 0 and vision_indices.numel() != 0:
        vision_output = vision_module(x.index_select(0, vision_indices))
        if vision_output.ndim != 2:
            raise RuntimeError("routed modules must return rank-2 tensors")
        output = vision_output.new_zeros(x.shape[0], vision_output.shape[-1])
        output = output.index_copy(0, vision_indices, vision_output)
        return output + _zero_parameter_dependency(text_module, output)

    vision_output = vision_module(x.index_select(0, vision_indices))
    text_output = text_module(x.index_select(0, text_indices))
    if vision_output.ndim != 2 or text_output.ndim != 2:
        raise RuntimeError("routed modules must return rank-2 tensors")
    if vision_output.shape[-1] != text_output.shape[-1]:
        raise RuntimeError("routed modules must return the same hidden size")

    output = vision_output.new_zeros(x.shape[0], vision_output.shape[-1])
    output = output.index_copy(0, vision_indices, vision_output)
    return output.index_copy(0, text_indices, text_output)


def _zero_parameter_dependency(module: nn.Module, output: Tensor) -> Tensor:
    dependency = output.new_zeros(())
    for parameter in module.parameters():
        if parameter.numel() != 0:
            dependency = dependency + parameter.reshape(-1)[0] * 0
    return dependency


def route_by_modality_map(
    x: Tensor,
    modality_indices: Mapping[int, Tensor],
    modules: Mapping[str, nn.Module],
    *,
    fallback: nn.Module,
) -> Tensor:
    """Route every packed token through the module for its canonical modality id."""

    if x.ndim < 2:
        raise ValueError("x must have a leading token dimension")
    output: Tensor | None = None
    used: set[int] = set()
    for modality_id, indices in sorted(modality_indices.items()):
        if indices.numel() == 0:
            continue
        if indices.dtype is not torch.long or indices.device != x.device:
            raise ValueError("modality route indices must be device-local torch.long tensors")
        module = modules[str(modality_id)] if str(modality_id) in modules else fallback
        routed = module(x.index_select(0, indices))
        if routed.ndim != x.ndim:
            raise RuntimeError("routed modality modules must preserve tensor rank")
        if output is None:
            output = routed.new_zeros((x.shape[0],) + tuple(routed.shape[1:]))
        elif routed.shape[1:] != output.shape[1:]:
            raise RuntimeError("all modality modules must return the same feature shape")
        output = output.index_copy(0, indices, routed)
        used.add(modality_id)
    if output is None:
        output = x.new_zeros(x.shape)
    for modality_id, module in modules.items():
        if int(modality_id) not in used:
            output = output + _zero_parameter_dependency(module, output)
    output = output + _zero_parameter_dependency(fallback, output)
    return output


class RMSNorm(nn.Module):
    """RMSNorm with FP32 variance accumulation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        inv_std = (variance + self.eps).rsqrt().to(input_dtype)
        return self.weight.to(input_dtype) * (hidden_states * inv_std)


class SharedRMSNorm(nn.Module):
    """One token-wise RMSNorm affine shared by vision and text."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)

    @property
    def weight(self) -> nn.Parameter:
        return self.norm.weight

    def forward(
        self,
        x: Tensor,
        vision_indices: Tensor,
        text_indices: Tensor,
        *,
        modality_indices: Mapping[int, Tensor] | None = None,
    ) -> Tensor:
        del vision_indices, text_indices, modality_indices
        return self.norm(x)


class ModalitySpecificRMSNorm(nn.Module):
    """Independent token-wise norms for registered modalities with a safe fallback."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        modality_ids: Sequence[int] = (0, 1),
    ) -> None:
        super().__init__()
        self.vision = nn.RMSNorm(hidden_size, eps=eps)
        self.text = nn.RMSNorm(hidden_size, eps=eps)
        self.extra = nn.ModuleDict(
            {
                str(modality_id): nn.RMSNorm(hidden_size, eps=eps)
                for modality_id in modality_ids
                if modality_id not in (0, 1)
            }
        )

    def forward(
        self,
        x: Tensor,
        vision_indices: Tensor,
        text_indices: Tensor,
        *,
        modality_indices: Mapping[int, Tensor] | None = None,
    ) -> Tensor:
        if modality_indices is not None:
            return route_by_modality_map(
                x,
                modality_indices,
                {"0": self.vision, "1": self.text, **self.extra},
                fallback=self.text,
            )
        return route_by_modality(
            x,
            vision_indices,
            text_indices,
            self.vision,
            self.text,
        )


class SwiGLU(nn.Module):
    """Fused gate/up SwiGLU projection."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.gate_up = make_linear(hidden_size, 2 * ffn_hidden_size)
        self.down = make_linear(ffn_hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class SharedSwiGLU(nn.Module):
    """One SwiGLU projection shared by vision and text tokens."""

    def __init__(self, hidden_size: int = 1024, ffn_hidden_size: int = 2816) -> None:
        super().__init__()
        self.projection = SwiGLU(hidden_size, ffn_hidden_size)

    def forward(
        self,
        x: Tensor,
        vision_indices: Tensor,
        text_indices: Tensor,
        *,
        modality_indices: Mapping[int, Tensor] | None = None,
    ) -> Tensor:
        del vision_indices, text_indices, modality_indices
        return self.projection(x)


class ModalitySpecificSwiGLU(nn.Module):
    """Route active tokens through independent SwiGLU weights by modality id."""

    def __init__(
        self,
        hidden_size: int = 1024,
        ffn_hidden_size: int = 2816,
        *,
        vision_ffn_hidden_size: int | None = None,
        modality_ids: Sequence[int] = (0, 1),
    ) -> None:
        super().__init__()
        self.vision = SwiGLU(hidden_size, vision_ffn_hidden_size or ffn_hidden_size)
        self.text = SwiGLU(hidden_size, ffn_hidden_size)
        self.extra = nn.ModuleDict(
            {
                str(modality_id): SwiGLU(hidden_size, ffn_hidden_size)
                for modality_id in modality_ids
                if modality_id not in (0, 1)
            }
        )

    def forward(
        self,
        x: Tensor,
        vision_indices: Tensor,
        text_indices: Tensor,
        *,
        modality_indices: Mapping[int, Tensor] | None = None,
    ) -> Tensor:
        if modality_indices is not None:
            return route_by_modality_map(
                x,
                modality_indices,
                {"0": self.vision, "1": self.text, **self.extra},
                fallback=self.text,
            )
        return route_by_modality(
            x,
            vision_indices,
            text_indices,
            self.vision,
            self.text,
        )
