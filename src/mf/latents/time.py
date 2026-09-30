import math

import torch
from torch import Tensor

_REAL_FLOAT_DTYPES = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
)


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _require_positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive; got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}")
    return result


def sample_shifted_clean_t(
    batch_size: int,
    alpha: float,
    mu: float,
    sigma: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Sample explicitly shifted noise levels and convert them to clean timesteps."""
    batch_size = _require_positive_int("batch_size", batch_size)
    alpha = _require_positive_float("alpha", alpha)
    if not isinstance(mu, (int, float)) or not math.isfinite(mu):
        raise ValueError(f"mu must be finite; got {mu!r}")
    if not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma must be finite and nonnegative; got {sigma!r}")
    if not isinstance(generator, torch.Generator):
        raise ValueError("generator must be a torch.Generator")
    if not isinstance(device, torch.device):
        raise ValueError(f"device must be a torch.device; got {type(device).__name__}")
    if dtype not in _REAL_FLOAT_DTYPES:
        raise ValueError(f"dtype must be a real floating-point dtype; got {dtype}")

    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    normal = torch.randn(
        (batch_size,),
        device=device,
        dtype=compute_dtype,
        generator=generator,
    )
    normal = normal * sigma + mu
    raw_noise = torch.sigmoid(normal)
    shifted_noise = alpha * raw_noise / (1.0 + (alpha - 1.0) * raw_noise)
    return (1.0 - shifted_noise).to(dtype=dtype)
