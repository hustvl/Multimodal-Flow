from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from mf.distributed.context import DistributedContext


def masked_ddp_means(
    components: Sequence[tuple[Tensor, Tensor]],
    context: DistributedContext,
    *,
    collect_metric: bool = True,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Reduce multiple masked means with one rank-synchronous collective."""

    if not components:
        raise ValueError("components must not be empty")

    local_sums: list[Tensor] = []
    local_counts: list[Tensor] = []
    device = components[0][0].device
    for local_values, mask in components:
        if not local_values.is_floating_point():
            raise TypeError("local_values must be a floating-point tensor")
        if mask.dtype is not torch.bool:
            raise TypeError("mask must have dtype torch.bool")
        if local_values.shape != mask.shape:
            raise ValueError(
                "local_values and mask must have the same shape; "
                f"got {list(local_values.shape)} and {list(mask.shape)}"
            )
        if local_values.device != mask.device:
            raise ValueError("local_values and mask must be on the same device")
        if local_values.device != device:
            raise ValueError("all components must be on the same device")

        reduction_values = (
            local_values.float()
            if local_values.dtype in (torch.float16, torch.bfloat16)
            else local_values
        )
        local_sums.append(torch.where(mask, reduction_values, 0.0).sum())
        local_counts.append(mask.sum(dtype=torch.int64))

    if collect_metric:
        packed = torch.stack(
            [
                value
                for local_sum, local_count in zip(local_sums, local_counts, strict=True)
                for value in (
                    local_sum.detach().to(dtype=torch.float64),
                    local_count.to(dtype=torch.float64),
                )
            ]
        )
    else:
        packed = torch.stack(
            [local_count.to(dtype=torch.float64) for local_count in local_counts]
        )
    global_values = context.all_reduce_detached_sum(packed)

    backward_losses: list[Tensor] = []
    detached_metrics: list[Tensor] = []
    for index, local_sum in enumerate(local_sums):
        count_index = 2 * index + 1 if collect_metric else index
        global_count = global_values[count_index]
        safe_global_count = global_count.clamp_min(1)
        backward_losses.append(
            local_sum * context.world_size / safe_global_count.to(dtype=local_sum.dtype)
        )
        if collect_metric:
            detached_metrics.append(global_values[2 * index] / safe_global_count)
        else:
            detached_metrics.append(
                local_sum.detach().new_zeros((), dtype=torch.float64)
            )
    return tuple(backward_losses), tuple(detached_metrics)


def masked_ddp_mean(
    local_values: Tensor,
    mask: Tensor,
    context: DistributedContext,
    *,
    collect_metric: bool = True,
) -> tuple[Tensor, Tensor]:
    """Build a DDP-correct backward mean and a detached global metric."""

    losses, metrics = masked_ddp_means(
        ((local_values, mask),),
        context,
        collect_metric=collect_metric,
    )
    return losses[0], metrics[0]


def weighted_ddp_mean(
    local_values: Tensor,
    mask: Tensor,
    weights: Tensor,
    context: DistributedContext,
    *,
    collect_metric: bool = True,
) -> tuple[Tensor, Tensor]:
    """Build a DDP-correct mean normalized by the global active weight sum."""

    if not local_values.is_floating_point():
        raise TypeError("local_values must be a floating-point tensor")
    if mask.dtype is not torch.bool:
        raise TypeError("mask must have dtype torch.bool")
    if not weights.is_floating_point():
        raise TypeError("weights must be a floating-point tensor")
    if local_values.shape != mask.shape or weights.shape != mask.shape:
        raise ValueError("local_values, mask, and weights must have the same shape")
    if local_values.device != mask.device or weights.device != mask.device:
        raise ValueError("local_values, mask, and weights must be on the same device")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("weights must be finite and non-negative")

    reduction_values = (
        local_values.float()
        if local_values.dtype in (torch.float16, torch.bfloat16)
        else local_values
    )
    active_weights = torch.where(
        mask,
        weights.to(dtype=reduction_values.dtype),
        0.0,
    )
    local_sum = (reduction_values * active_weights).sum()
    local_weight = active_weights.sum()
    global_metric_sum = (
        context.all_reduce_detached_sum(local_sum.to(torch.float64))
        if collect_metric
        else None
    )
    global_weight = context.all_reduce_detached_sum(local_weight.to(torch.float64))
    safe_global_weight = global_weight.clamp_min(torch.finfo(torch.float64).tiny)
    backward_loss = (
        local_sum * context.world_size / safe_global_weight.to(local_sum.dtype)
    )
    if global_metric_sum is not None:
        detached_metric = global_metric_sum / safe_global_weight
    else:
        detached_metric = local_sum.detach().new_zeros((), dtype=torch.float64)
    return backward_loss, detached_metric
