from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist
from torch import Tensor


class DistributedError(RuntimeError):
    """Invalid or unavailable distributed state."""


class DistributedNotInitializedError(DistributedError):
    """A multi-rank operation has no initialized process group."""


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Validated rank metadata and detached collective access."""

    rank: int
    world_size: int
    process_group: dist.ProcessGroup | None = None

    def __post_init__(self) -> None:
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int):
            raise TypeError("world_size must be an integer")
        if self.world_size < 1:
            raise ValueError("world_size must be at least 1")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("rank must be an integer")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")

    @classmethod
    def single_process(cls) -> DistributedContext:
        return cls(rank=0, world_size=1)

    @classmethod
    def from_process_group(
        cls,
        process_group: dist.ProcessGroup | None = None,
    ) -> DistributedContext:
        if not dist.is_available() or not dist.is_initialized():
            raise DistributedNotInitializedError("torch.distributed is not initialized")
        return cls(
            rank=dist.get_rank(group=process_group),
            world_size=dist.get_world_size(group=process_group),
            process_group=process_group,
        )

    def all_reduce_detached_sum(self, value: Tensor) -> Tensor:
        """Return a detached summed clone, leaving the input tensor untouched."""

        result = value.detach().clone()
        if self.world_size == 1:
            return result
        if not dist.is_available() or not dist.is_initialized():
            raise DistributedNotInitializedError(
                "a process group must be initialized when world_size is greater than 1"
            )

        runtime_rank = dist.get_rank(group=self.process_group)
        runtime_world_size = dist.get_world_size(group=self.process_group)
        if (runtime_rank, runtime_world_size) != (self.rank, self.world_size):
            raise DistributedError(
                "context rank/world_size do not match the initialized process group"
            )
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=self.process_group)
        return result

    def all_reduce_detached_max(self, value: Tensor) -> Tensor:
        """Return a detached elementwise maximum clone across ranks."""

        result = value.detach().clone()
        if self.world_size == 1:
            return result
        if not dist.is_available() or not dist.is_initialized():
            raise DistributedNotInitializedError(
                "a process group must be initialized when world_size is greater than 1"
            )

        runtime_rank = dist.get_rank(group=self.process_group)
        runtime_world_size = dist.get_world_size(group=self.process_group)
        if (runtime_rank, runtime_world_size) != (self.rank, self.world_size):
            raise DistributedError(
                "context rank/world_size do not match the initialized process group"
            )
        dist.all_reduce(result, op=dist.ReduceOp.MAX, group=self.process_group)
        return result
