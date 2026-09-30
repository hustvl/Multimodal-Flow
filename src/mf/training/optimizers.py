from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.optim import Optimizer

from mf.config.schema import OptimizersConfig

# Sharded Muon must reconstruct the full updated parameter set on every rank.
# A large chunk avoids paying dozens of collective launch latencies per step for
# the 1.6B model while preserving the same parameter ownership and update math.
_DEFAULT_MUON_SYNC_CHUNK_BYTES = 1 << 26
_MUON_OVERLAP_ALL_GATHER_ENV = "MF_MUON_OVERLAP_ALL_GATHER"
_MUON_ASYNC_BUFFER_COUNT = 2

_MUON_PROCESS_GROUPS: dict[tuple[int, int, int], tuple[dist.ProcessGroup, ...]] = {}


class ParameterOwnershipError(RuntimeError):
    """Raised when trainable parameters do not have one unambiguous owner."""


class _OptimizerParameterRoles(Protocol):
    muon: Iterable[nn.Parameter]
    adamw: Iterable[nn.Parameter]


@runtime_checkable
class _OptimizerRoleProvider(Protocol):
    def optimizer_parameter_roles(self) -> _OptimizerParameterRoles: ...


def _zeropower_via_newton_schulz5(matrix: Tensor, steps: int) -> Tensor:
    # Newton-Schulz coefficients used by the Muon optimizer.
    a, b, c = (3.4445, -4.7750, 2.0315)
    value = matrix.to(torch.float32)
    transposed = value.size(-2) > value.size(-1)
    if transposed:
        value = value.mT
    value = value / (value.norm(dim=(-2, -1), keepdim=True) + 1e-8)
    for _ in range(steps):
        gram = value @ value.mT
        polynomial = b * gram + c * gram @ gram
        value = a * value + polynomial @ value
    return value.mT if transposed else value


def _advance_muon_state(
    gradient: Tensor,
    momentum_buffer: Tensor,
    step: int,
    *,
    momentum: float,
) -> int:
    momentum_buffer.lerp_(gradient, 1.0 - momentum)
    return step + 1


def _muon_update_from_state(
    gradient: Tensor,
    momentum_buffer: Tensor,
    step: int,
    *,
    momentum: float,
    ns_steps: int,
    flax_layout: bool,
) -> Tensor:
    corrected_momentum = momentum_buffer / (1.0 - momentum ** (step + 1))
    corrected_gradient = gradient / (1.0 - momentum**step)
    update = momentum * corrected_momentum + (1.0 - momentum) * corrected_gradient
    update = _zeropower_via_newton_schulz5(update, ns_steps)
    rows, columns = gradient.shape
    aspect_ratio = columns / rows if flax_layout else rows / columns
    return update * max(1.0, aspect_ratio) ** 0.5


def _muon_parameter_cost(parameter: nn.Parameter) -> int:
    rows, columns = parameter.shape
    inner = min(rows, columns)
    outer = max(rows, columns)
    # Two inner^2*outer products and one inner^3 product per NS iteration.
    return 2 * inner * inner * outer + inner * inner * inner


def _balanced_parameter_owners(
    parameters: list[nn.Parameter],
    world_size: int,
) -> tuple[int, ...]:
    loads = [0] * world_size
    owners = [0] * len(parameters)
    for index in sorted(
        range(len(parameters)),
        key=lambda item: (-_muon_parameter_cost(parameters[item]), item),
    ):
        owner = min(
            range(world_size), key=lambda candidate: (loads[candidate], candidate)
        )
        owners[index] = owner
        loads[owner] += _muon_parameter_cost(parameters[index])
    return tuple(owners)


def _muon_shard_process_group(
    shard_group_size: int | None,
) -> tuple[dist.ProcessGroup | None, int, int]:
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    if shard_group_size is None or shard_group_size == world_size:
        return None, global_rank, world_size
    if shard_group_size > world_size:
        raise ValueError("Muon shard_group_size cannot exceed distributed world size")
    if world_size % shard_group_size:
        raise ValueError(
            "Muon shard_group_size must evenly divide distributed world size"
        )

    world_group_id = id(dist.group.WORLD)
    cache_key = (world_group_id, world_size, shard_group_size)
    process_groups = _MUON_PROCESS_GROUPS.get(cache_key)
    if process_groups is None:
        process_groups = tuple(
            dist.new_group(ranks=list(range(start, start + shard_group_size)))
            for start in range(0, world_size, shard_group_size)
        )
        _MUON_PROCESS_GROUPS[cache_key] = process_groups
    process_group = process_groups[global_rank // shard_group_size]
    return process_group, dist.get_rank(group=process_group), shard_group_size


@dataclass(frozen=True)
class _MuonSyncGroup:
    spans_by_owner: tuple[tuple[tuple[nn.Parameter, int, int], ...], ...]
    max_owner_load: int
    chunk_numel: int
    local_chunks: tuple[Tensor, ...]
    gathered_chunks: tuple[Tensor, ...]


@dataclass(frozen=True)
class _MuonSyncChunk:
    group: _MuonSyncGroup
    chunk_start: int
    chunk_end: int
    buffer_index: int
    required_parameter_ids: frozenset[int]


@dataclass(frozen=True)
class _PendingMuonSync:
    chunk: _MuonSyncChunk
    work: Any


def _muon_overlap_all_gather_enabled(configured: bool | None) -> bool:
    if configured is not None:
        if type(configured) is not bool:
            raise ValueError("Muon overlap_all_gather must be a boolean")
        return configured
    value = os.environ.get(_MUON_OVERLAP_ALL_GATHER_ENV)
    if value is None or value.strip().lower() in {"", "0", "false", "no", "off"}:
        return False
    if value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"{_MUON_OVERLAP_ALL_GATHER_ENV} must be a boolean value")


@dataclass(frozen=True)
class _MuonSyncPlan:
    groups: tuple[_MuonSyncGroup, ...]


def _build_muon_sync_plan(
    parameters: list[nn.Parameter],
    owners: tuple[int, ...],
    *,
    world_size: int,
    sync_chunk_bytes: int,
    buffer_count: int = 1,
) -> _MuonSyncPlan:
    parameter_groups: dict[
        tuple[torch.dtype, torch.device],
        list[tuple[int, nn.Parameter]],
    ] = {}
    for index, parameter in enumerate(parameters):
        parameter_groups.setdefault((parameter.dtype, parameter.device), []).append(
            (index, parameter)
        )

    sync_groups: list[_MuonSyncGroup] = []
    for (dtype, device), indexed_parameters in parameter_groups.items():
        spans_by_owner: list[list[tuple[nn.Parameter, int, int]]] = [
            [] for _ in range(world_size)
        ]
        owner_loads = [0] * world_size
        for index, parameter in indexed_parameters:
            owner = owners[index]
            start = owner_loads[owner]
            end = start + parameter.numel()
            spans_by_owner[owner].append((parameter, start, end))
            owner_loads[owner] = end

        max_owner_load = max(owner_loads)
        if max_owner_load == 0:
            continue

        element_size = torch.empty((), dtype=dtype).element_size()
        chunk_numel = max(1, min(max_owner_load, sync_chunk_bytes // element_size))
        chunk_count = (max_owner_load + chunk_numel - 1) // chunk_numel
        allocated_buffer_count = min(buffer_count, chunk_count)
        local_chunks = tuple(
            torch.empty(chunk_numel, dtype=dtype, device=device)
            for _ in range(allocated_buffer_count)
        )
        gathered_chunks = tuple(
            torch.empty(
                world_size * chunk_numel,
                dtype=dtype,
                device=device,
            )
            for _ in range(allocated_buffer_count)
        )
        sync_groups.append(
            _MuonSyncGroup(
                spans_by_owner=tuple(tuple(spans) for spans in spans_by_owner),
                max_owner_load=max_owner_load,
                chunk_numel=chunk_numel,
                local_chunks=local_chunks,
                gathered_chunks=gathered_chunks,
            )
        )
    return _MuonSyncPlan(groups=tuple(sync_groups))


def _foreach_copy(destinations: list[Tensor], sources: list[Tensor]) -> None:
    if not destinations:
        return
    if len(destinations) != len(sources):
        raise RuntimeError("Muon sync copy lists must have the same length")
    if len(destinations) == 1:
        destinations[0].copy_(sources[0])
        return
    torch._foreach_copy_(destinations, sources)


def _foreach_lerp_(
    destinations: list[Tensor], sources: list[Tensor], weight: float
) -> None:
    if not destinations:
        return
    if len(destinations) != len(sources):
        raise RuntimeError("foreach lerp lists must have the same length")
    if len(destinations) == 1:
        destinations[0].lerp_(sources[0], weight)
        return
    torch._foreach_lerp_(destinations, sources, weight)


def _foreach_add_(
    destinations: list[Tensor], sources: list[Tensor], alpha: float
) -> None:
    if not destinations:
        return
    if len(destinations) != len(sources):
        raise RuntimeError("foreach add lists must have the same length")
    if len(destinations) == 1:
        destinations[0].add_(sources[0], alpha=alpha)
        return
    torch._foreach_add_(destinations, sources, alpha=alpha)


def _pack_muon_chunk(
    chunk: _MuonSyncChunk,
    *,
    rank: int,
) -> tuple[Tensor, Tensor]:
    group = chunk.group
    local_chunk = group.local_chunks[chunk.buffer_index]
    gathered_chunks = group.gathered_chunks[chunk.buffer_index]
    local_chunk.zero_()
    pack_destinations: list[Tensor] = []
    pack_sources: list[Tensor] = []
    for parameter, span_start, span_end in group.spans_by_owner[rank]:
        overlap_start = max(chunk.chunk_start, span_start)
        overlap_end = min(chunk.chunk_end, span_end)
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - chunk.chunk_start
        parameter_start = overlap_start - span_start
        count = overlap_end - overlap_start
        pack_destinations.append(local_chunk[local_start : local_start + count])
        pack_sources.append(
            parameter.view(-1)[parameter_start : parameter_start + count]
        )
    _foreach_copy(pack_destinations, pack_sources)
    return local_chunk, gathered_chunks


def _unpack_muon_chunk(chunk: _MuonSyncChunk) -> None:
    group = chunk.group
    gathered_chunks = group.gathered_chunks[chunk.buffer_index]
    unpack_destinations: list[Tensor] = []
    unpack_sources: list[Tensor] = []
    for owner, spans in enumerate(group.spans_by_owner):
        owner_chunk = gathered_chunks.narrow(
            0,
            owner * group.chunk_numel,
            group.chunk_numel,
        )
        for parameter, span_start, span_end in spans:
            overlap_start = max(chunk.chunk_start, span_start)
            overlap_end = min(chunk.chunk_end, span_end)
            if overlap_start >= overlap_end:
                continue
            local_start = overlap_start - chunk.chunk_start
            parameter_start = overlap_start - span_start
            count = overlap_end - overlap_start
            unpack_destinations.append(
                parameter.view(-1)[parameter_start : parameter_start + count]
            )
            unpack_sources.append(owner_chunk[local_start : local_start + count])
    _foreach_copy(unpack_destinations, unpack_sources)


def _muon_sync_chunks(plan: _MuonSyncPlan, *, rank: int) -> tuple[_MuonSyncChunk, ...]:
    chunks: list[_MuonSyncChunk] = []
    for group in plan.groups:
        for chunk_index, chunk_start in enumerate(
            range(0, group.max_owner_load, group.chunk_numel)
        ):
            chunk_end = min(chunk_start + group.chunk_numel, group.max_owner_load)
            required_parameter_ids = frozenset(
                id(parameter)
                for parameter, span_start, span_end in group.spans_by_owner[rank]
                if max(chunk_start, span_start) < min(chunk_end, span_end)
            )
            chunks.append(
                _MuonSyncChunk(
                    group=group,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    buffer_index=chunk_index % len(group.local_chunks),
                    required_parameter_ids=required_parameter_ids,
                )
            )
    return tuple(chunks)


def _synchronize_muon_parameters(
    plan: _MuonSyncPlan,
    *,
    rank: int,
    process_group: dist.ProcessGroup | None,
) -> None:
    for group in plan.groups:
        for chunk_start in range(0, group.max_owner_load, group.chunk_numel):
            chunk_end = min(chunk_start + group.chunk_numel, group.max_owner_load)
            chunk = _MuonSyncChunk(
                group=group,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                buffer_index=0,
                required_parameter_ids=frozenset(),
            )
            local_chunk, gathered_chunks = _pack_muon_chunk(chunk, rank=rank)
            dist.all_gather_into_tensor(
                gathered_chunks,
                local_chunk,
                group=process_group,
            )
            _unpack_muon_chunk(chunk)


class _OverlappedMuonSynchronizer:
    def __init__(
        self,
        plan: _MuonSyncPlan,
        *,
        rank: int,
        process_group: dist.ProcessGroup | None,
    ) -> None:
        self._rank = rank
        self._process_group = process_group
        self._chunks = _muon_sync_chunks(plan, rank=rank)
        self._next_chunk_index = 0
        self._pending: list[_PendingMuonSync] = []

    def launch_ready(
        self,
        updated_parameter_ids: set[int],
        *,
        max_new: int = 1,
    ) -> None:
        launched = 0
        while self._next_chunk_index < len(self._chunks) and launched < max_new:
            chunk = self._chunks[self._next_chunk_index]
            if not chunk.required_parameter_ids.issubset(updated_parameter_ids):
                return
            if len(self._pending) == _MUON_ASYNC_BUFFER_COUNT:
                self._complete_oldest()
            local_chunk, gathered_chunks = _pack_muon_chunk(chunk, rank=self._rank)
            work = dist.all_gather_into_tensor(
                gathered_chunks,
                local_chunk,
                group=self._process_group,
                async_op=True,
            )
            if work is None:
                raise RuntimeError("Muon async all-gather did not return a work handle")
            self._pending.append(_PendingMuonSync(chunk=chunk, work=work))
            self._next_chunk_index += 1
            launched += 1

    def finish(self, updated_parameter_ids: set[int]) -> None:
        while self._next_chunk_index < len(self._chunks):
            before = self._next_chunk_index
            self.launch_ready(updated_parameter_ids)
            if self._next_chunk_index == before:
                raise RuntimeError(
                    "Muon async sync chunk was not ready after all owned updates"
                )
        while self._pending:
            self._complete_oldest()

    def _complete_oldest(self) -> None:
        pending = self._pending.pop(0)
        if pending.work.wait() is False:
            raise RuntimeError("Muon async all-gather did not complete")
        _unpack_muon_chunk(pending.chunk)


class Muon(Optimizer):
    """Matrix-only Muon optimizer; unsupported parameters are rejected."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float,
        *,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        flax_layout_parameter_ids: Iterable[int] = (),
        distributed_mode: Literal["sharded", "replicated"] = "sharded",
        shard_group_size: int | None = None,
        sync_chunk_bytes: int = _DEFAULT_MUON_SYNC_CHUNK_BYTES,
        overlap_all_gather: bool | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if ns_steps < 1:
            raise ValueError("ns_steps must be at least 1")
        if distributed_mode not in {"sharded", "replicated"}:
            raise ValueError("Muon distributed_mode must be sharded or replicated")
        if shard_group_size is not None and shard_group_size < 1:
            raise ValueError("Muon shard_group_size must be positive")
        if shard_group_size is not None and distributed_mode != "sharded":
            raise ValueError("Muon shard_group_size requires sharded mode")
        if type(sync_chunk_bytes) is not int or sync_chunk_bytes < 1:
            raise ValueError("Muon sync_chunk_bytes must be a positive integer")
        super().__init__(
            params,
            defaults={
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "ns_steps": ns_steps,
            },
        )
        self._distributed_mode = distributed_mode
        self._shard_group_size = shard_group_size
        self._overlap_all_gather = _muon_overlap_all_gather_enabled(overlap_all_gather)
        parameter_ids: set[int] = set()
        self._sync_chunk_bytes = sync_chunk_bytes
        for group in self.param_groups:
            if group["weight_decay"] < 0.0:
                raise ValueError("weight_decay must be non-negative")
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError(
                        "Muon accepts only two-dimensional matrix parameters"
                    )
                parameter_ids.add(id(parameter))
        self._flax_layout_parameter_ids = frozenset(flax_layout_parameter_ids)
        unknown_layout_ids = self._flax_layout_parameter_ids - parameter_ids
        if unknown_layout_ids:
            raise ValueError("flax-layout ids must identify Muon parameters")
        self._sharded_sync_plans: dict[
            tuple[int, int], tuple[tuple[int, ...], _MuonSyncPlan]
        ] = {}

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], float | Tensor] | None = None,
    ) -> float | Tensor | None:
        loss: float | Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        def parameter_update(
            parameter: nn.Parameter,
            gradient: Tensor,
            settings: dict[str, Any],
        ) -> Tensor:
            state = self.state[parameter]
            return _muon_update_from_state(
                gradient,
                state["momentum_buffer"],
                state["step"],
                momentum=settings["momentum"],
                ns_steps=settings["ns_steps"],
                flax_layout=(id(parameter) in self._flax_layout_parameter_ids),
            )

        for group in self.param_groups:
            parameters = group["params"]
            momentum_groups: dict[
                tuple[torch.device, torch.dtype], tuple[list[Tensor], list[Tensor]]
            ] = {}
            gradients: dict[nn.Parameter, Tensor] = {}
            for parameter in parameters:
                gradient = parameter.grad
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                    parameter.grad = gradient
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                    state["step"] = 0
                state["step"] += 1
                buffers, grouped_gradients = momentum_groups.setdefault(
                    (parameter.device, parameter.dtype), ([], [])
                )
                buffers.append(state["momentum_buffer"])
                grouped_gradients.append(gradient)
                gradients[parameter] = gradient
            for buffers, grouped_gradients in momentum_groups.values():
                _foreach_lerp_(
                    buffers,
                    grouped_gradients,
                    1.0 - group["momentum"],
                )

            sharded = distributed and self._distributed_mode == "sharded"
            owners: tuple[int, ...] | None = None
            sync_plan: _MuonSyncPlan | None = None
            process_group: dist.ProcessGroup | None = None
            shard_rank = rank
            shard_world_size = world_size
            if sharded:
                process_group, shard_rank, shard_world_size = _muon_shard_process_group(
                    self._shard_group_size
                )
                cache_key = (id(group), shard_world_size)
                cached = self._sharded_sync_plans.get(cache_key)
                if cached is None:
                    owners = _balanced_parameter_owners(parameters, shard_world_size)
                    sync_plan = _build_muon_sync_plan(
                        parameters,
                        owners,
                        world_size=shard_world_size,
                        sync_chunk_bytes=self._sync_chunk_bytes,
                        buffer_count=(
                            _MUON_ASYNC_BUFFER_COUNT if self._overlap_all_gather else 1
                        ),
                    )
                    self._sharded_sync_plans[cache_key] = (owners, sync_plan)
                else:
                    owners, sync_plan = cached
            update_groups: dict[
                tuple[torch.device, torch.dtype], tuple[list[Tensor], list[Tensor]]
            ] = {}

            if sync_plan is not None and self._overlap_all_gather:
                synchronizer = _OverlappedMuonSynchronizer(
                    sync_plan,
                    rank=shard_rank,
                    process_group=process_group,
                )
                updated_parameter_ids: set[int] = set()
                synchronizer.launch_ready(updated_parameter_ids)
                for index, parameter in enumerate(parameters):
                    if owners[index] != shard_rank:
                        continue
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(
                        parameter_update(parameter, gradients[parameter], group),
                        alpha=-group["lr"],
                    )
                    updated_parameter_ids.add(id(parameter))
                    synchronizer.launch_ready(updated_parameter_ids)
                synchronizer.finish(updated_parameter_ids)
            else:
                for index, parameter in enumerate(parameters):
                    if owners is not None and owners[index] != shard_rank:
                        continue
                    update = parameter_update(parameter, gradients[parameter], group)
                    destinations, updates = update_groups.setdefault(
                        (parameter.device, parameter.dtype), ([], [])
                    )
                    destinations.append(parameter)
                    updates.append(update)
                for destinations, updates in update_groups.values():
                    if group["weight_decay"]:
                        torch._foreach_mul_(
                            destinations,
                            1.0 - group["lr"] * group["weight_decay"],
                        )
                    _foreach_add_(destinations, updates, -group["lr"])

                if sync_plan is not None:
                    _synchronize_muon_parameters(
                        sync_plan,
                        rank=shard_rank,
                        process_group=process_group,
                    )
        return loss


class NesterovAdamW(Optimizer):
    """Nesterov-Adam fallback with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1)")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        super().__init__(
            params,
            defaults={
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            },
        )

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], float | Tensor] | None = None,
    ) -> float | Tensor | None:
        loss: float | Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            tensor_groups: dict[
                tuple[torch.device, torch.dtype, int],
                tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]],
            ] = {}
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                    parameter.grad = gradient
                if gradient.is_sparse:
                    raise RuntimeError(
                        "NesterovAdamW does not support sparse gradients"
                    )
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] = 0
                state["step"] += 1
                parameters, gradients, exp_avgs, exp_avg_sqs = tensor_groups.setdefault(
                    (parameter.device, parameter.dtype, state["step"]),
                    ([], [], [], []),
                )
                parameters.append(parameter)
                gradients.append(gradient)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

            for (_, _, step), (
                parameters,
                gradients,
                exp_avgs,
                exp_avg_sqs,
            ) in tensor_groups.items():
                _foreach_lerp_(exp_avgs, gradients, 1.0 - beta1)
                gradient_squares = list(torch._foreach_mul(gradients, gradients))
                _foreach_lerp_(exp_avg_sqs, gradient_squares, 1.0 - beta2)

                corrected_momentum = list(
                    torch._foreach_mul(
                        exp_avgs,
                        beta1 / (1.0 - beta1 ** (step + 1)),
                    )
                )
                torch._foreach_add_(
                    corrected_momentum,
                    gradients,
                    alpha=(1.0 - beta1) / (1.0 - beta1**step),
                )
                corrected_variance = list(
                    torch._foreach_mul(exp_avg_sqs, 1.0 / (1.0 - beta2**step))
                )
                torch._foreach_sqrt_(corrected_variance)
                torch._foreach_add_(corrected_variance, group["eps"])
                torch._foreach_div_(corrected_momentum, corrected_variance)
                if group["weight_decay"] != 0.0:
                    torch._foreach_mul_(
                        parameters,
                        1.0 - group["lr"] * group["weight_decay"],
                    )
                _foreach_add_(parameters, corrected_momentum, -group["lr"])
        return loss


@dataclass(frozen=True, slots=True)
class OptimizerBundle:
    backbone_muon: Muon | None
    backbone_adamw: NesterovAdamW | None
    text_decoder_muon: Muon | None
    text_decoder_adamw: NesterovAdamW | None

    @property
    def optimizers(self) -> tuple[Optimizer, ...]:
        return tuple(
            optimizer
            for optimizer in (
                self.backbone_muon,
                self.backbone_adamw,
                self.text_decoder_muon,
                self.text_decoder_adamw,
            )
            if optimizer is not None
        )


def _named_trainable_parameters(
    module: nn.Module,
    *,
    owner: str,
) -> list[tuple[str, nn.Parameter]]:
    named = [
        (name, parameter)
        for name, parameter in module.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    ]
    names_by_id: dict[int, list[str]] = {}
    for name, parameter in named:
        names_by_id.setdefault(id(parameter), []).append(name)
    aliases = [names for names in names_by_id.values() if len(names) > 1]
    if aliases:
        alias_text = "; ".join(", ".join(names) for names in aliases)
        raise ParameterOwnershipError(
            f"{owner} contains aliased trainable parameters: {alias_text}"
        )
    return named


def _optimizer_role_parameters(
    model: nn.Module,
    expected_ids: set[int],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    if not isinstance(model, _OptimizerRoleProvider):
        raise ParameterOwnershipError(
            "backbone must implement optimizer_parameter_roles()"
        )

    roles = model.optimizer_parameter_roles()
    try:
        muon_parameters = list(roles.muon)
        adamw_parameters = list(roles.adamw)
    except (AttributeError, TypeError) as error:
        raise ParameterOwnershipError(
            "optimizer_parameter_roles() must return iterable muon and adamw roles"
        ) from error

    role_parameters = muon_parameters + adamw_parameters
    if any(not isinstance(parameter, nn.Parameter) for parameter in role_parameters):
        raise ParameterOwnershipError(
            "optimizer_parameter_roles() may return nn.Parameter objects only"
        )

    muon_counts = Counter(id(parameter) for parameter in muon_parameters)
    adamw_counts = Counter(id(parameter) for parameter in adamw_parameters)
    duplicated_ids = {
        parameter_id for parameter_id, count in muon_counts.items() if count > 1
    } | {parameter_id for parameter_id, count in adamw_counts.items() if count > 1}
    shared_ids = muon_counts.keys() & adamw_counts.keys()
    owned_ids = muon_counts.keys() | adamw_counts.keys()
    missing_ids = expected_ids - owned_ids
    unknown_ids = owned_ids - expected_ids
    if missing_ids or unknown_ids or duplicated_ids or shared_ids:
        raise ParameterOwnershipError(
            "optimizer_parameter_roles() must own every backbone trainable exactly once "
            f"(missing={len(missing_ids)}, unknown={len(unknown_ids)}, "
            f"duplicated={len(duplicated_ids)}, shared={len(shared_ids)})"
        )
    if any(parameter.ndim != 2 for parameter in muon_parameters):
        raise ParameterOwnershipError(
            "optimizer_parameter_roles() muon role may contain matrices only"
        )
    return muon_parameters, adamw_parameters


def _new_nesterov_adamw(
    parameters: list[nn.Parameter],
    parameter_names: tuple[str, ...],
    *,
    group_name: str,
    lr: float,
    min_lr: float,
    weight_decay: float,
) -> NesterovAdamW | None:
    if not parameters:
        return None
    return NesterovAdamW(
        [
            {
                "params": parameters,
                "parameter_names": parameter_names,
                "group_name": group_name,
                "lr": lr,
                "peak_lr": lr,
                "min_lr": min_lr,
                "weight_decay": weight_decay,
            }
        ],
        lr=lr,
        weight_decay=weight_decay,
    )


def _decoder_optimizer_parameters(
    decoder: nn.Module,
    decoder_named: list[tuple[str, nn.Parameter]],
    *,
    use_muon: bool,
) -> tuple[list[nn.Parameter], list[nn.Parameter], set[int]]:
    parameters = [parameter for _, parameter in decoder_named]
    if not use_muon:
        return [], parameters, set()

    embedding_weight_ids = {
        id(module.weight)
        for module in decoder.modules()
        if isinstance(module, nn.Embedding) and module.weight.requires_grad
    }
    linear_weight_ids = {
        id(module.weight)
        for module in decoder.modules()
        if isinstance(module, nn.Linear) and module.weight.requires_grad
    }
    muon_parameters = [
        parameter
        for parameter in parameters
        if parameter.ndim == 2 and id(parameter) not in embedding_weight_ids
    ]
    muon_ids = {id(parameter) for parameter in muon_parameters}
    fallback_parameters = [
        parameter for parameter in parameters if id(parameter) not in muon_ids
    ]
    flax_layout_ids = {
        id(parameter)
        for parameter in muon_parameters
        if id(parameter) not in linear_weight_ids
    }
    return muon_parameters, fallback_parameters, flax_layout_ids


def build_optimizers(
    model: nn.Module,
    decoder: nn.Module,
    config: OptimizersConfig,
) -> OptimizerBundle:
    """Create disjoint Muon and Nesterov-AdamW owners for backbone and decoder."""

    backbone_named = _named_trainable_parameters(model, owner="backbone")
    decoder_named = _named_trainable_parameters(decoder, owner="decoder")
    backbone_ids = {id(parameter) for _, parameter in backbone_named}
    decoder_ids = {id(parameter) for _, parameter in decoder_named}
    if backbone_ids & decoder_ids:
        raise ParameterOwnershipError(
            "a trainable parameter cannot belong to both backbone and text decoder"
        )

    muon_parameters, fallback_parameters = _optimizer_role_parameters(
        model,
        backbone_ids,
    )
    backbone_names = {
        id(parameter): f"backbone.{name}" for name, parameter in backbone_named
    }
    muon_parameter_names = tuple(
        backbone_names[id(parameter)] for parameter in muon_parameters
    )
    fallback_parameter_names = tuple(
        backbone_names[id(parameter)] for parameter in fallback_parameters
    )

    linear_weight_ids = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, nn.Linear) and module.weight.requires_grad
    }
    flax_layout_parameter_ids = {
        id(parameter)
        for parameter in muon_parameters
        if id(parameter) not in linear_weight_ids
    }

    backbone_config = config.backbone
    decoder_config = config.text_decoder
    decoder_muon_parameters, decoder_fallback_parameters, decoder_flax_layout_ids = (
        _decoder_optimizer_parameters(
            decoder,
            decoder_named,
            use_muon=decoder_config.optimizer == "Muon",
        )
    )
    decoder_names = {
        id(parameter): f"text_decoder.{name}" for name, parameter in decoder_named
    }
    decoder_muon_names = tuple(
        decoder_names[id(parameter)] for parameter in decoder_muon_parameters
    )
    decoder_fallback_names = tuple(
        decoder_names[id(parameter)] for parameter in decoder_fallback_parameters
    )

    backbone_muon = (
        Muon(
            [
                {
                    "params": muon_parameters,
                    "parameter_names": muon_parameter_names,
                    "group_name": "backbone_muon",
                    "lr": backbone_config.peak_lr,
                    "peak_lr": backbone_config.peak_lr,
                    "min_lr": backbone_config.min_lr,
                    "weight_decay": backbone_config.weight_decay,
                }
            ],
            lr=backbone_config.peak_lr,
            weight_decay=backbone_config.weight_decay,
            flax_layout_parameter_ids=flax_layout_parameter_ids,
            distributed_mode=config.muon_distributed_mode,
            shard_group_size=config.muon_shard_group_size,
            sync_chunk_bytes=config.muon_sync_chunk_mb << 20,
        )
        if muon_parameters
        else None
    )
    backbone_adamw = _new_nesterov_adamw(
        fallback_parameters,
        fallback_parameter_names,
        group_name="backbone_adamw",
        lr=backbone_config.peak_lr,
        min_lr=backbone_config.min_lr,
        weight_decay=backbone_config.weight_decay,
    )
    text_decoder_muon = (
        Muon(
            [
                {
                    "params": decoder_muon_parameters,
                    "parameter_names": decoder_muon_names,
                    "group_name": "text_decoder_muon",
                    "lr": decoder_config.peak_lr,
                    "peak_lr": decoder_config.peak_lr,
                    "min_lr": decoder_config.min_lr,
                    "weight_decay": decoder_config.weight_decay,
                }
            ],
            lr=decoder_config.peak_lr,
            weight_decay=decoder_config.weight_decay,
            flax_layout_parameter_ids=decoder_flax_layout_ids,
            distributed_mode=config.muon_distributed_mode,
            shard_group_size=config.muon_shard_group_size,
            sync_chunk_bytes=config.muon_sync_chunk_mb << 20,
        )
        if decoder_muon_parameters
        else None
    )
    text_decoder_adamw = _new_nesterov_adamw(
        decoder_fallback_parameters,
        decoder_fallback_names,
        group_name="text_decoder_adamw",
        lr=decoder_config.peak_lr,
        min_lr=decoder_config.min_lr,
        weight_decay=decoder_config.weight_decay,
    )
    return OptimizerBundle(
        backbone_muon=backbone_muon,
        backbone_adamw=backbone_adamw,
        text_decoder_muon=text_decoder_muon,
        text_decoder_adamw=text_decoder_adamw,
    )
