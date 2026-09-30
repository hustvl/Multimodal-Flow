from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from mf.distributed.context import DistributedContext
from mf.distributed.reductions import masked_ddp_mean, weighted_ddp_mean


def x0_to_velocity(
    x0_prediction: Tensor,
    noisy_input: Tensor,
    clean_timestep: Tensor,
    *,
    t_eps: float,
) -> Tensor:
    """Convert a normalized x0 prediction into flow velocity."""

    if x0_prediction.shape != noisy_input.shape:
        raise ValueError(
            "x0_prediction and noisy_input must have the same shape; "
            f"got {list(x0_prediction.shape)} and {list(noisy_input.shape)}"
        )
    if x0_prediction.ndim < 2 or x0_prediction.shape[-1] == 0:
        raise ValueError(
            "x0_prediction and noisy_input must have a non-empty latent dimension"
        )
    valid_timestep_shapes = {x0_prediction.shape[:1], x0_prediction.shape[:-1]}
    if clean_timestep.shape not in valid_timestep_shapes:
        raise ValueError(
            "clean_timestep must have shape [B] or match x0_prediction without its latent "
            f"dimension; got {list(clean_timestep.shape)}"
        )
    if not 0.0 < t_eps <= 1.0:
        raise ValueError("t_eps must be in (0, 1]")

    denominator = (1.0 - clean_timestep).clamp_min(t_eps)
    while denominator.ndim < x0_prediction.ndim:
        denominator = denominator.unsqueeze(-1)
    return (x0_prediction - noisy_input) / denominator


def mean_of_per_sample_masked_means(
    local_values: Tensor,
    mask: Tensor,
    context: DistributedContext,
    *,
    weights: Tensor | None = None,
    collect_metric: bool = True,
) -> tuple[Tensor, Tensor]:
    """Give each non-empty sample equal weight regardless of active token count."""
    if not local_values.is_floating_point():
        raise TypeError("local_values must be a floating-point tensor")
    if local_values.ndim != 2 or local_values.shape != mask.shape:
        raise ValueError("local_values and mask must have matching shape [B, T]")
    if mask.dtype is not torch.bool:
        raise TypeError("mask must have dtype torch.bool")
    if local_values.device != mask.device:
        raise ValueError("local_values and mask must share a device")
    reduction_values = (
        local_values.float()
        if local_values.dtype in (torch.float16, torch.bfloat16)
        else local_values
    )
    if weights is None:
        active_weights = mask.to(dtype=reduction_values.dtype)
    else:
        if weights.shape != mask.shape or not weights.is_floating_point():
            raise ValueError("weights must be floating-point with shape [B, T]")
        if weights.device != mask.device:
            raise ValueError("weights and mask must share a device")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("weights must be finite and non-negative")
        active_weights = torch.where(
            mask,
            weights.to(dtype=reduction_values.dtype),
            0.0,
        )
    per_sample_weight = active_weights.sum(dim=1)
    sample_present = per_sample_weight > 0
    safe_values = torch.where(mask, reduction_values, 0.0)
    per_sample_sum = (safe_values * active_weights).sum(dim=1)
    per_sample_mean = per_sample_sum / per_sample_weight.clamp_min(1)
    return masked_ddp_mean(
        per_sample_mean,
        sample_present,
        context,
        collect_metric=collect_metric,
    )


def mean_of_per_sample_fixed_window_sums(
    local_values: Tensor,
    mask: Tensor,
    context: DistributedContext,
    *,
    denominator: int,
    weights: Tensor | None = None,
    collect_metric: bool = True,
) -> tuple[Tensor, Tensor]:
    """Average per-sample active sums against one fixed token window."""
    if not local_values.is_floating_point():
        raise TypeError("local_values must be a floating-point tensor")
    if local_values.ndim != 2 or local_values.shape != mask.shape:
        raise ValueError("local_values and mask must have matching shape [B, T]")
    if mask.dtype is not torch.bool or local_values.device != mask.device:
        raise ValueError("mask must be bool and share the local_values device")
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise ValueError("denominator must be a positive integer")
    reduction_values = (
        local_values.float()
        if local_values.dtype in (torch.float16, torch.bfloat16)
        else local_values
    )
    active_weights = mask.to(dtype=reduction_values.dtype)
    if weights is not None:
        if weights.shape != mask.shape or not weights.is_floating_point():
            raise ValueError("weights must be floating-point with shape [B, T]")
        if weights.device != mask.device:
            raise ValueError("weights and mask must share a device")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("weights must be finite and non-negative")
        active_weights = torch.where(
            mask, weights.to(dtype=reduction_values.dtype), 0.0
        )
    safe_values = torch.where(mask, reduction_values, 0.0)
    per_sample = (safe_values * active_weights).sum(dim=1) / float(denominator)
    return masked_ddp_mean(
        per_sample, mask.any(dim=1), context, collect_metric=collect_metric
    )


def velocity_mse_values(
    x0_prediction: Tensor,
    noisy_input: Tensor,
    clean_target: Tensor,
    clean_timestep: Tensor,
    active_mask: Tensor,
    *,
    t_eps: float,
    velocity_target: Tensor | None = None,
) -> Tensor:
    """Return finite per-token velocity MSE with inactive positions zeroed."""
    if x0_prediction.shape != clean_target.shape:
        raise ValueError(
            "x0_prediction and clean_target must have the same shape; "
            f"got {list(x0_prediction.shape)} and {list(clean_target.shape)}"
        )
    velocity_prediction = x0_to_velocity(
        x0_prediction,
        noisy_input,
        clean_timestep,
        t_eps=t_eps,
    )
    if velocity_target is None:
        velocity_target = x0_to_velocity(
            clean_target,
            noisy_input,
            clean_timestep,
            t_eps=t_eps,
        )
    elif velocity_target.shape != x0_prediction.shape:
        raise ValueError("velocity_target must match x0_prediction shape")
    expected_mask_shape = x0_prediction.shape[:-1]
    if active_mask.shape != expected_mask_shape:
        raise ValueError(
            "active_mask must match x0_prediction without its latent dimension; "
            f"expected {list(expected_mask_shape)}, got {list(active_mask.shape)}"
        )

    expanded_mask = active_mask.unsqueeze(-1)
    safe_prediction = velocity_prediction.masked_fill(~expanded_mask, 0.0)
    safe_target = velocity_target.masked_fill(~expanded_mask, 0.0)
    return (safe_prediction - safe_target).square().mean(dim=-1)


def masked_velocity_mse_loss(
    x0_prediction: Tensor,
    noisy_input: Tensor,
    clean_target: Tensor,
    clean_timestep: Tensor,
    active_mask: Tensor,
    context: DistributedContext,
    *,
    t_eps: float,
    velocity_target: Tensor | None = None,
    collect_metric: bool = True,
    token_weights: Tensor | None = None,
    per_sample: bool = False,
    per_sample_denominator: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Reduce normalized velocity MSE over active answer tokens."""
    if per_sample and per_sample_denominator is not None:
        raise ValueError("per-sample reductions are mutually exclusive")

    per_token_mse = velocity_mse_values(
        x0_prediction,
        noisy_input,
        clean_target,
        clean_timestep,
        active_mask,
        t_eps=t_eps,
        velocity_target=velocity_target,
    )
    if per_sample_denominator is not None:
        return mean_of_per_sample_fixed_window_sums(
            per_token_mse,
            active_mask,
            context,
            denominator=per_sample_denominator,
            weights=token_weights,
            collect_metric=collect_metric,
        )
    if per_sample:
        return mean_of_per_sample_masked_means(
            per_token_mse,
            active_mask,
            context,
            weights=token_weights,
            collect_metric=collect_metric,
        )
    if token_weights is not None:
        return weighted_ddp_mean(
            per_token_mse,
            active_mask,
            token_weights,
            context,
            collect_metric=collect_metric,
        )
    return masked_ddp_mean(
        per_token_mse,
        active_mask,
        context,
        collect_metric=collect_metric,
    )


flow_mse_loss = masked_velocity_mse_loss


def decoder_cross_entropy_values(
    logits: Tensor,
    targets: Tensor,
    active_mask: Tensor,
) -> Tensor:
    """Return per-token decoder cross entropy with inactive positions isolated."""
    if logits.ndim != targets.ndim + 1 or logits.shape[:-1] != targets.shape:
        raise ValueError(
            "logits must have shape targets.shape + [vocab_size]; "
            f"got {list(logits.shape)} and {list(targets.shape)}"
        )
    if active_mask.shape != targets.shape:
        raise ValueError(
            "active_mask and targets must have the same shape; "
            f"got {list(active_mask.shape)} and {list(targets.shape)}"
        )
    if logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")

    safe_logits = logits.masked_fill(~active_mask.unsqueeze(-1), 0.0)
    safe_targets = targets.masked_fill(~active_mask, 0)
    return F.cross_entropy(
        safe_logits.reshape(-1, logits.shape[-1]),
        safe_targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)


def decoder_cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    active_mask: Tensor,
    context: DistributedContext,
    *,
    collect_metric: bool = True,
    token_weights: Tensor | None = None,
    per_sample: bool = False,
    per_sample_denominator: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Reduce token cross entropy over active decoder targets."""
    if per_sample and per_sample_denominator is not None:
        raise ValueError("per-sample reductions are mutually exclusive")

    token_losses = decoder_cross_entropy_values(logits, targets, active_mask)
    if per_sample_denominator is not None:
        return mean_of_per_sample_fixed_window_sums(
            token_losses,
            active_mask,
            context,
            denominator=per_sample_denominator,
            weights=token_weights,
            collect_metric=collect_metric,
        )
    if per_sample:
        return mean_of_per_sample_masked_means(
            token_losses,
            active_mask,
            context,
            weights=token_weights,
            collect_metric=collect_metric,
        )
    if token_weights is not None:
        return weighted_ddp_mean(
            token_losses,
            active_mask,
            token_weights,
            context,
            collect_metric=collect_metric,
        )
    return masked_ddp_mean(
        token_losses,
        active_mask,
        context,
        collect_metric=collect_metric,
    )
