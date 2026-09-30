from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

EMA_DECAY = 0.9999
_EMA_STATE_VERSION = 1
_EMA_STATE_KEYS = frozenset(
    {
        "version",
        "decay",
        "optimizer_keys",
        "parameter_keys",
        "optimizer_parameter_indices",
        "parameter_shapes",
        "optimizer_steps",
        "shadows",
    }
)


@dataclass(frozen=True, slots=True)
class _FlatNonFp32Group:
    parameters: tuple[nn.Parameter, ...]
    current_views: tuple[Tensor, ...]
    current_flat: Tensor
    shadow_flat: Tensor


class ExponentialMovingAverage:
    """Optimizer-owned fp32 EMA with atomic step/update and exact live restores."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        *,
        decay: float = EMA_DECAY,
        parameter_keys: Iterable[str] | None = None,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        parameter_list = tuple(parameters)
        parameter_ids = [id(parameter) for parameter in parameter_list]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("EMA parameters must not contain duplicates")
        if any(not parameter.requires_grad for parameter in parameter_list):
            raise ValueError("EMA tracks trainable parameters only")

        if parameter_keys is None:
            keys = tuple(f"parameter_{index}" for index in range(len(parameter_list)))
        else:
            keys = tuple(parameter_keys)
        if (
            len(keys) != len(parameter_list)
            or any(not isinstance(key, str) or not key for key in keys)
            or len(keys) != len(set(keys))
        ):
            raise ValueError("EMA parameter identity keys must be unique and ordered")

        self.decay = decay
        self._parameters = parameter_list
        self._parameter_keys = keys
        self._parameter_shapes = tuple(
            tuple(parameter.shape) for parameter in parameter_list
        )
        self._shadows = tuple(
            parameter.detach().to(dtype=torch.float32).clone()
            for parameter in parameter_list
        )
        self._index_by_parameter_id = {
            id(parameter): index for index, parameter in enumerate(parameter_list)
        }
        self._optimizer_keys: tuple[str, ...] = ()
        self._optimizer_parameter_indices: tuple[tuple[int, ...], ...] = ()
        self._optimizer_fp32_indices: tuple[tuple[int, ...], ...] = ()
        self._optimizer_non_fp32_indices: tuple[tuple[int, ...], ...] = ()
        self._optimizer_non_fp32_flat_groups: tuple[
            tuple[_FlatNonFp32Group, ...], ...
        ] = ()
        self._optimizer_steps: list[int] = []
        self._optimizer_position_by_id: dict[int, int] = {}
        self._swapped = False
        self._ema_stream: torch.cuda.Stream | None = None
        self._ema_device: torch.device | None = None
        self._pending_update_events: list[torch.cuda.Event | None] = []

    @classmethod
    def from_optimizers(
        cls,
        optimizers: Iterable[Optimizer],
        *,
        decay: float = EMA_DECAY,
    ) -> ExponentialMovingAverage:
        optimizer_list = tuple(optimizers)
        optimizer_ids = [id(optimizer) for optimizer in optimizer_list]
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise ValueError("EMA optimizers must not contain duplicates")

        parameters: list[nn.Parameter] = []
        parameter_keys: list[str] = []
        optimizer_keys: list[str] = []
        optimizer_parameter_indices: list[tuple[int, ...]] = []
        seen_parameter_ids: set[int] = set()
        for optimizer_index, optimizer in enumerate(optimizer_list):
            group_names: list[str] = []
            owned_indices: list[int] = []
            for group_index, group in enumerate(optimizer.param_groups):
                raw_group_name = group.get("group_name")
                if raw_group_name is None:
                    group_name = f"group_{group_index}"
                elif isinstance(raw_group_name, str) and raw_group_name:
                    group_name = raw_group_name
                else:
                    raise ValueError(
                        "EMA optimizer group identity must be a non-empty string"
                    )
                group_names.append(group_name)

                group_parameters = tuple(group["params"])
                raw_parameter_names = group.get("parameter_names")
                if raw_parameter_names is None:
                    group_parameter_names = tuple(
                        f"parameter_{index}" for index in range(len(group_parameters))
                    )
                else:
                    try:
                        group_parameter_names = tuple(raw_parameter_names)
                    except TypeError as error:
                        raise ValueError(
                            "EMA optimizer parameter identities must be iterable"
                        ) from error
                if (
                    len(group_parameter_names) != len(group_parameters)
                    or any(
                        not isinstance(name, str) or not name
                        for name in group_parameter_names
                    )
                    or len(group_parameter_names) != len(set(group_parameter_names))
                ):
                    raise ValueError(
                        "EMA optimizer parameter identities must be unique and ordered"
                    )

                for parameter_index, (parameter, parameter_name) in enumerate(
                    zip(group_parameters, group_parameter_names, strict=True)
                ):
                    if (
                        not isinstance(parameter, nn.Parameter)
                        or not parameter.requires_grad
                    ):
                        raise ValueError(
                            "EMA optimizers must own trainable parameters only"
                        )
                    parameter_id = id(parameter)
                    if parameter_id in seen_parameter_ids:
                        raise ValueError(
                            "EMA optimizer parameter ownership must be disjoint"
                        )
                    seen_parameter_ids.add(parameter_id)
                    owned_indices.append(len(parameters))
                    parameters.append(parameter)
                    parameter_keys.append(
                        f"{optimizer_index}:{group_index}:{group_name}:"
                        f"{parameter_index}:{parameter_name}"
                    )

            optimizer_class = type(optimizer)
            class_key = f"{optimizer_class.__module__}.{optimizer_class.__qualname__}"
            optimizer_keys.append(
                f"{optimizer_index}:{class_key}:{'|'.join(group_names)}"
            )
            optimizer_parameter_indices.append(tuple(owned_indices))

        instance = cls(
            parameters,
            decay=decay,
            parameter_keys=parameter_keys,
        )
        instance._optimizer_keys = tuple(optimizer_keys)
        instance._optimizer_parameter_indices = tuple(optimizer_parameter_indices)
        instance._optimizer_fp32_indices = tuple(
            tuple(
                index
                for index in owned_indices
                if parameters[index].dtype is torch.float32
            )
            for owned_indices in optimizer_parameter_indices
        )
        instance._optimizer_non_fp32_indices = tuple(
            tuple(
                index
                for index in owned_indices
                if parameters[index].dtype is not torch.float32
            )
            for owned_indices in optimizer_parameter_indices
        )
        shadows = list(instance._shadows)
        optimizer_flat_groups: list[tuple[_FlatNonFp32Group, ...]] = []
        for owned_indices in instance._optimizer_non_fp32_indices:
            indices_by_storage: dict[tuple[torch.device, torch.dtype], list[int]] = {}
            for index in owned_indices:
                parameter = parameters[index]
                indices_by_storage.setdefault(
                    (parameter.device, parameter.dtype), []
                ).append(index)

            flat_groups: list[_FlatNonFp32Group] = []
            for indices in indices_by_storage.values():
                group_parameters = tuple(parameters[index] for index in indices)
                shadow_flat = torch._utils._flatten_dense_tensors(group_parameters).to(
                    dtype=torch.float32
                )
                shadow_views = tuple(
                    torch._utils._unflatten_dense_tensors(shadow_flat, group_parameters)
                )
                current_flat = torch.empty_like(shadow_flat)
                current_views = tuple(
                    torch._utils._unflatten_dense_tensors(
                        current_flat, group_parameters
                    )
                )
                for index, shadow in zip(indices, shadow_views, strict=True):
                    shadows[index] = shadow
                flat_groups.append(
                    _FlatNonFp32Group(
                        parameters=group_parameters,
                        current_views=current_views,
                        current_flat=current_flat,
                        shadow_flat=shadow_flat,
                    )
                )
            optimizer_flat_groups.append(tuple(flat_groups))
        instance._shadows = tuple(shadows)
        instance._optimizer_non_fp32_flat_groups = tuple(optimizer_flat_groups)
        instance._optimizer_steps = [0] * len(optimizer_list)
        instance._optimizer_position_by_id = {
            id(optimizer): index for index, optimizer in enumerate(optimizer_list)
        }
        instance._pending_update_events = [None] * len(optimizer_list)
        return instance

    @property
    def parameter_ids(self) -> frozenset[int]:
        return frozenset(self._index_by_parameter_id)

    def enable_async_stream(self) -> None:
        """Move future EMA updates to one ordered CUDA-compatible stream."""

        if self._swapped:
            raise RuntimeError(
                "EMA async stream cannot be enabled while parameters are swapped"
            )
        if self._ema_stream is not None:
            return
        if not self._optimizer_keys:
            raise RuntimeError("EMA async stream requires registered optimizers")
        devices = {parameter.device for parameter in self._parameters}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise RuntimeError(
                "EMA async stream requires all tracked parameters on one CUDA-compatible device"
            )

        device = next(iter(devices))
        stream = torch.cuda.Stream(device=device)
        self._ema_device = device
        self._ema_stream = stream

    def synchronize(self) -> None:
        """Wait for every enqueued EMA update before an external state boundary."""

        pending_events = tuple(
            event for event in self._pending_update_events if event is not None
        )
        for event in pending_events:
            event.synchronize()
        if pending_events:
            self._pending_update_events[:] = [None] * len(self._pending_update_events)

    def _wait_before_optimizer_mutation(self, optimizer_position: int) -> None:
        event = self._pending_update_events[optimizer_position]
        if event is None:
            return
        if self._ema_device is None:
            raise RuntimeError("EMA async stream device is not initialized")
        torch.cuda.current_stream(self._ema_device).wait_event(event)

    def _update_optimizer_shadows(self, optimizer_position: int) -> None:
        update_weight = 1.0 - self.decay
        fp32_indices = self._optimizer_fp32_indices[optimizer_position]
        if fp32_indices:
            torch._foreach_lerp_(
                tuple(self._shadows[index] for index in fp32_indices),
                tuple(self._parameters[index].detach() for index in fp32_indices),
                update_weight,
            )
        for group in self._optimizer_non_fp32_flat_groups[optimizer_position]:
            torch._foreach_copy_(group.current_views, group.parameters)
            group.shadow_flat.lerp_(group.current_flat, update_weight)

    def _enqueue_optimizer_update(self, optimizer_position: int) -> None:
        if self._ema_stream is None or self._ema_device is None:
            self._update_optimizer_shadows(optimizer_position)
            return
        optimizer_complete = torch.cuda.Event()
        optimizer_complete.record(torch.cuda.current_stream(self._ema_device))
        self._ema_stream.wait_event(optimizer_complete)
        with torch.cuda.stream(self._ema_stream):
            self._update_optimizer_shadows(optimizer_position)
            update_complete = torch.cuda.Event()
            update_complete.record(self._ema_stream)
        self._pending_update_events[optimizer_position] = update_complete

    @torch.no_grad()
    def step_optimizer(
        self,
        optimizer: Optimizer,
        closure: Callable[[], float | Tensor] | None = None,
    ) -> float | Tensor | None:
        """Step one registered optimizer, then update its EMA subset exactly once."""

        if self._swapped:
            raise RuntimeError("EMA cannot step while parameters are swapped")
        optimizer_position = self._optimizer_position_by_id.get(id(optimizer))
        if optimizer_position is None:
            raise ValueError("optimizer is not registered with this EMA")
        self._wait_before_optimizer_mutation(optimizer_position)

        if closure is None:
            loss = optimizer.step()
        else:
            loss = optimizer.step(closure)

        self._enqueue_optimizer_update(optimizer_position)
        self._optimizer_steps[optimizer_position] += 1
        return loss

    def state_dict(self) -> dict[str, object]:
        if self._swapped:
            raise RuntimeError("EMA state cannot be saved while parameters are swapped")
        self.synchronize()
        return {
            "version": _EMA_STATE_VERSION,
            "decay": self.decay,
            "optimizer_keys": self._optimizer_keys,
            "parameter_keys": self._parameter_keys,
            "optimizer_parameter_indices": self._optimizer_parameter_indices,
            "parameter_shapes": self._parameter_shapes,
            "optimizer_steps": tuple(self._optimizer_steps),
            "shadows": tuple(shadow.detach().clone() for shadow in self._shadows),
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if self._swapped:
            raise RuntimeError(
                "EMA state cannot be loaded while parameters are swapped"
            )
        if not isinstance(state, Mapping):
            raise TypeError("EMA state must be a mapping")
        state_keys = frozenset(state)
        if state_keys != _EMA_STATE_KEYS:
            missing = sorted(_EMA_STATE_KEYS - state_keys)
            unexpected = sorted(state_keys - _EMA_STATE_KEYS)
            raise ValueError(
                f"EMA state keys mismatch (missing={missing}, unexpected={unexpected})"
            )
        if state["version"] != _EMA_STATE_VERSION:
            raise ValueError("EMA state version mismatch")
        if state["decay"] != self.decay:
            raise ValueError("EMA state decay mismatch")
        if state["optimizer_keys"] != self._optimizer_keys:
            raise ValueError("EMA optimizer identity mismatch")
        if state["parameter_keys"] != self._parameter_keys:
            raise ValueError("EMA parameter identity/order mismatch")
        if state["optimizer_parameter_indices"] != self._optimizer_parameter_indices:
            raise ValueError("EMA optimizer parameter order mismatch")
        if state["parameter_shapes"] != self._parameter_shapes:
            raise ValueError("EMA parameter shape mismatch")

        optimizer_steps = state["optimizer_steps"]
        if (
            not isinstance(optimizer_steps, tuple)
            or len(optimizer_steps) != len(self._optimizer_keys)
            or any(
                isinstance(step, bool) or not isinstance(step, int) or step < 0
                for step in optimizer_steps
            )
        ):
            raise ValueError("EMA optimizer cadence state is invalid")

        saved_shadows = state["shadows"]
        if not isinstance(saved_shadows, tuple) or len(saved_shadows) != len(
            self._shadows
        ):
            raise ValueError("EMA shadow shape/count mismatch")
        validated_shadows: list[Tensor] = []
        for saved_shadow, expected_shape, live_shadow in zip(
            saved_shadows,
            self._parameter_shapes,
            self._shadows,
            strict=True,
        ):
            if (
                not isinstance(saved_shadow, Tensor)
                or saved_shadow.dtype is not torch.float32
                or tuple(saved_shadow.shape) != expected_shape
            ):
                raise ValueError("EMA shadow dtype/shape mismatch")
            validated_shadows.append(
                saved_shadow.detach().to(device=live_shadow.device).clone()
            )

        self.synchronize()
        for live_shadow, saved_shadow in zip(
            self._shadows,
            validated_shadows,
            strict=True,
        ):
            live_shadow.copy_(saved_shadow)
        self._optimizer_steps[:] = optimizer_steps

    @contextmanager
    def swap_parameters(self) -> Iterator[None]:
        if self._swapped:
            raise RuntimeError("nested EMA parameter swaps are not supported")
        self.synchronize()
        self._swapped = True
        backups: list[Tensor] = []
        try:
            with torch.no_grad():
                for parameter, shadow in zip(
                    self._parameters,
                    self._shadows,
                    strict=True,
                ):
                    backups.append(parameter.detach().clone())
                    parameter.copy_(shadow)
        except BaseException:
            with torch.no_grad():
                for parameter, backup in reversed(
                    list(zip(self._parameters[: len(backups)], backups, strict=True))
                ):
                    parameter.copy_(backup)
            self._swapped = False
            raise

        try:
            yield
        finally:
            try:
                with torch.no_grad():
                    for parameter, backup in reversed(
                        list(zip(self._parameters, backups, strict=True))
                    ):
                        parameter.copy_(backup)
            finally:
                self._swapped = False
