import torch
from torch import Tensor


def _validate_clean_latent(x0: object) -> Tensor:
    if not isinstance(x0, Tensor):
        raise ValueError("x0 must be a torch.Tensor")
    if x0.ndim < 1:
        raise ValueError("x0 must have a batch dimension")
    if not torch.is_floating_point(x0):
        raise ValueError(f"x0 must have a floating-point dtype; got {x0.dtype}")
    return x0


def interpolate_with_noise(x0: Tensor, eps: Tensor, clean_t: Tensor) -> Tensor:
    """Interpolate with the project convention t=1 clean and t=0 noise."""
    clean = _validate_clean_latent(x0)
    if not isinstance(eps, Tensor):
        raise ValueError("eps must be a torch.Tensor")
    if eps.shape != clean.shape:
        raise ValueError(
            f"eps shape must match x0; got {list(eps.shape)} and {list(clean.shape)}"
        )
    if eps.dtype != clean.dtype:
        raise ValueError(f"eps dtype must match x0; got {eps.dtype} and {clean.dtype}")
    if eps.device != clean.device:
        raise ValueError(
            f"eps device must match x0; got {eps.device} and {clean.device}"
        )
    if not isinstance(clean_t, Tensor):
        raise ValueError("clean_t must be a torch.Tensor")
    valid_timestep_shapes = {(clean.shape[0],), clean.shape[:-1]}
    if tuple(clean_t.shape) not in valid_timestep_shapes:
        raise ValueError(
            "clean_t must have shape [B] or match x0 without its latent dimension; "
            f"got {list(clean_t.shape)}"
        )
    if clean_t.device != clean.device:
        raise ValueError(
            f"clean_t device must match x0; got {clean_t.device} and {clean.device}"
        )

    if clean_t.ndim == 1:
        broadcast_shape = (clean.shape[0],) + (1,) * (clean.ndim - 1)
    else:
        broadcast_shape = (*clean_t.shape, 1)
    timestep = clean_t.to(dtype=clean.dtype).view(broadcast_shape)
    return timestep * clean + (1.0 - timestep) * eps


def sample_flow_input(
    x0: Tensor,
    clean_t: Tensor,
    generator: torch.Generator,
    *,
    noise_scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Sample scaled Gaussian flow noise and return both x_t and epsilon."""
    clean = _validate_clean_latent(x0)
    if not isinstance(generator, torch.Generator):
        raise ValueError("generator must be a torch.Generator")
    eps = torch.randn(
        clean.shape,
        device=clean.device,
        dtype=clean.dtype,
        generator=generator,
    )
    eps = eps * float(noise_scale)
    return interpolate_with_noise(clean, eps, clean_t), eps
