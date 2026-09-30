from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mf.codecs.text_decoder import LatentTextDecoder
from mf.config.schema import MFConfig
from mf.contracts.task_registry import task_definitions, task_value
from mf.contracts.model import MFOutput
from mf.contracts.trainer import LossOutput, TrainingBatch
from mf.distributed.context import DistributedContext
from mf.distributed.reductions import masked_ddp_mean, masked_ddp_means
from mf.latents.decoder_noise import corrupt_text_decoder_latents
from mf.registries import OBJECTIVE_REGISTRY, register_objective
from mf.training.losses import (
    decoder_cross_entropy_values,
    masked_velocity_mse_loss,
    velocity_mse_values,
    x0_to_velocity,
)


class LossMetrics(NamedTuple):
    """Detached global means matching the three backward loss components."""

    total: Tensor
    vision_flow: Tensor
    text_flow: Tensor
    text_decoder_ce: Tensor
    physical_flow: Tensor
    task: TaskMetrics | None


@dataclass(frozen=True, slots=True)
class ObjectiveDefinition:
    """Composable physical objective and its detached metric projection."""

    compose: Callable[..., tuple[Tensor, Tensor]]
    metric_sum: Callable[..., Tensor]

    def __call__(self, *args: object, **kwargs: object) -> tuple[Tensor, Tensor]:
        return self.compose(*args, **kwargs)


_SUPPORTED_LOSS_COMPONENTS = frozenset(
    {"vision_flow", "text_flow", "text_decoder_ce", "modality_flow"}
)


def _metric_registry_snapshot() -> tuple[
    tuple[object, ...], dict[int, tuple[str, ...]], tuple[tuple[int, str], ...]
]:
    """Read task semantics at call time so late plugins cannot go stale."""

    definitions = task_definitions("metrics")
    components = {
        int(task_value(definition)): definition.loss_components
        for definition in definitions
    }
    supported = _SUPPORTED_LOSS_COMPONENTS | frozenset(OBJECTIVE_REGISTRY.names())
    unknown = sorted(
        {
            component
            for values in components.values()
            for component in values
            if component not in supported
        }
    )
    if unknown:
        raise RuntimeError(
            "task registry contains loss components without a registered composer: "
            + ", ".join(unknown)
        )
    loss_keys = tuple(
        (int(task_value(definition)), component)
        for definition in definitions
        for component in definition.loss_components
    )
    return definitions, components, loss_keys


@dataclass(frozen=True, slots=True)
class TaskMetrics(Mapping[str, Tensor]):
    """Detached per-task sufficient statistics exposed as flat scalar metrics."""

    sample_counts: dict[int, Tensor]
    active_tokens: dict[tuple[int, str], Tensor]
    loss_sums: dict[tuple[int, str], Tensor]

    def _values(self) -> dict[str, Tensor]:
        definitions, task_components, _ = _metric_registry_snapshot()
        values: dict[str, Tensor] = {}
        for definition in definitions:
            task = task_value(definition)
            label = definition.label
            values[f"samples/{label}"] = self.sample_counts[task]
            for modality in ("vision", "text"):
                values[f"active_tokens/{label}/{modality}"] = self.active_tokens[
                    task, modality
                ]

            component_values: list[Tensor] = []
            for component in task_components[int(task)]:
                key = (task, component)
                if key not in self.loss_sums:
                    continue
                if component == "vision_flow":
                    modality = "vision"
                elif component in {"text_flow", "text_decoder_ce"}:
                    modality = "text"
                elif component == "modality_flow":
                    support = self.sample_counts[task]
                    if not bool(support):
                        continue
                    component_value = self.loss_sums[key] / support.clamp_min(1).to(
                        dtype=self.loss_sums[key].dtype
                    )
                    values[f"loss/{label}/{component}"] = component_value
                    component_values.append(component_value)
                    continue
                else:
                    raise RuntimeError(
                        f"loss component {component!r} has no metric support"
                    )
                support = self.active_tokens[task, modality]
                if not bool(support):
                    continue
                component_value = self.loss_sums[key] / support.clamp_min(1).to(
                    dtype=self.loss_sums[key].dtype
                )
                values[f"loss/{label}/{component}"] = component_value
                component_values.append(component_value)
            if component_values:
                values[f"loss/{label}/total"] = sum(
                    component_values,
                    self.sample_counts[task].new_zeros((), dtype=torch.float64),
                )
        return values

    def __getitem__(self, key: str) -> Tensor:
        return self._values()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values())

    def __len__(self) -> int:
        return len(self._values())

    def __add__(self, other: TaskMetrics) -> TaskMetrics:
        definitions, _, _ = _metric_registry_snapshot()
        return TaskMetrics(
            sample_counts={
                task: self.sample_counts[task] + other.sample_counts[task]
                for definition in definitions
                for task in (task_value(definition),)
            },
            active_tokens={
                key: self.active_tokens[key] + other.active_tokens[key]
                for key in self.active_tokens
            },
            loss_sums={
                key: self.loss_sums.get(key, value.new_zeros(())) + value
                for key, value in other.loss_sums.items()
            }
            | {
                key: value
                for key, value in self.loss_sums.items()
                if key not in other.loss_sums
            },
        )

    def packed(self) -> Tensor:
        """Pack sufficient statistics in a stable order for one collective."""

        definitions, _, loss_keys = _metric_registry_snapshot()
        local_values = [
            self.sample_counts[task_value(definition)]
            for definition in definitions
        ]
        local_values.extend(
            self.active_tokens[task, modality]
            for definition in definitions
            for task in (task_value(definition),)
            for modality in ("vision", "text")
        )
        zero = local_values[0].new_zeros(())
        local_values.extend(self.loss_sums.get(key, zero) for key in loss_keys)
        return torch.stack([value.to(dtype=torch.float64) for value in local_values])

    @classmethod
    def from_packed(cls, values: Tensor) -> TaskMetrics:
        """Restore sufficient statistics produced by :meth:`packed`."""

        definitions, _, loss_keys = _metric_registry_snapshot()
        expected = 3 * len(definitions) + len(loss_keys)
        if values.ndim != 1 or values.numel() != expected:
            raise ValueError(
                f"packed task metrics must have shape ({expected},), got {tuple(values.shape)}"
            )
        offset = 0
        sample_counts = {}
        for definition in definitions:
            task = task_value(definition)
            sample_counts[task] = values[offset].to(dtype=torch.int64)
            offset += 1
        active_tokens: dict[tuple[int, str], Tensor] = {}
        for definition in definitions:
            task = task_value(definition)
            for modality in ("vision", "text"):
                active_tokens[task, modality] = values[offset].to(dtype=torch.int64)
                offset += 1
        loss_sums = {key: values[offset + index] for index, key in enumerate(loss_keys)}
        return cls(
            sample_counts=sample_counts,
            active_tokens=active_tokens,
            loss_sums=loss_sums,
        )

    def reduced(self, context: DistributedContext) -> TaskMetrics:
        return self.from_packed(context.all_reduce_detached_sum(self.packed()))

    def to_scalars(self) -> dict[str, float | int]:
        values = self.from_packed(self.packed().detach().cpu())._values()
        return {
            name: int(value.item())
            if not value.is_floating_point()
            else float(value.item())
            for name, value in values.items()
        }


def _flow_tensors(
    prediction: Tensor | None,
    noisy_input: Tensor | None,
    target: Tensor | None,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if prediction is None:
        if target is not None or noisy_input is not None:
            raise ValueError("flow inputs cannot exist without their prediction branch")
        prediction = torch.zeros(
            (*mask.shape, 1),
            device=mask.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        noisy_input = torch.zeros_like(prediction)
        target = torch.zeros_like(prediction)
    elif target is None or noisy_input is None:
        raise ValueError(
            "a flow prediction branch requires noisy input and normalized clean target"
        )
    return prediction, noisy_input, target


def _decoder_inputs(
    batch: TrainingBatch,
    text_decoder: LatentTextDecoder,
) -> tuple[Tensor, Tensor]:
    decoder_input = batch.text_decoder_input_latents
    targets = batch.text_token_ids
    if decoder_input is None:
        parameter = next(text_decoder.parameters())
        geometry = batch.model_input.geometry
        text_latent_dim = (
            geometry.text_latent_dim
            if geometry is not None
            else int(parameter.shape[-1])
        )
        decoder_input = torch.zeros(
            (*batch.text_target_mask.shape, text_latent_dim),
            device=batch.text_target_mask.device,
            dtype=parameter.dtype,
        )
    if targets is None:
        targets = torch.zeros_like(batch.text_target_mask, dtype=torch.long)
    return decoder_input.detach(), targets


def _component_task_mask(
    task_type: Tensor,
    base_mask: Tensor,
    component: str,
    definitions: tuple[object, ...],
) -> Tensor:
    """Select only rows whose registered task declares this objective."""

    rows = torch.zeros_like(task_type, dtype=torch.bool)
    for definition in definitions:
        if component in definition.loss_components:
            rows |= task_type == int(task_value(definition))
    return base_mask & rows[:, None]


def _decoder_logits(
    decoder_input: Tensor,
    batch: TrainingBatch,
    text_decoder: LatentTextDecoder,
    config: MFConfig,
) -> Tensor:
    decoder_config = config.model.text_decoder
    block_causal = config.flow.text_block_causal
    block_local = block_causal.target_encoding == "block_local"
    if not block_local:
        return text_decoder(
            decoder_input,
            attention_mask=(
                batch.model_input.text_content_mask
                if decoder_config.use_attention_mask
                else None
            ),
        )

    block_size = block_causal.block_size
    batch_size, text_tokens, latent_dim = decoder_input.shape
    if text_tokens % block_size != 0:
        raise ValueError("decoder text length must be divisible by block_size")
    block_count = text_tokens // block_size
    flat_input = decoder_input.reshape(batch_size * block_count, block_size, latent_dim)
    flat_mask = batch.text_target_mask.reshape(batch_size * block_count, block_size)
    active_blocks = flat_mask.any(dim=1)
    active_indices = active_blocks.nonzero(as_tuple=False).flatten()
    if active_indices.numel() == 0:
        dummy_mask = torch.zeros(
            (1, block_size),
            dtype=torch.bool,
            device=decoder_input.device,
        )
        dummy = text_decoder(flat_input[:1], attention_mask=dummy_mask)
        flat_logits = (
            dummy.new_zeros((batch_size * block_count, block_size, dummy.shape[-1]))
            + dummy.sum() * 0.0
        )
    else:
        selected_mask = flat_mask.index_select(0, active_indices)
        active_logits = text_decoder(
            flat_input.index_select(0, active_indices),
            attention_mask=(
                selected_mask if decoder_config.use_attention_mask else None
            ),
        )
        flat_logits = active_logits.new_zeros(
            (batch_size * block_count, block_size, active_logits.shape[-1])
        ).index_copy(0, active_indices, active_logits)
    return flat_logits.view(batch_size, text_tokens, -1)


def _decoder_loss_values(
    decoder_input: Tensor,
    decoder_targets: Tensor,
    batch: TrainingBatch,
    text_decoder: LatentTextDecoder,
    config: MFConfig,
    *,
    keep_logits: bool,
) -> tuple[Tensor, Tensor | None]:
    decoder_config = config.model.text_decoder
    block_causal = config.flow.text_block_causal
    block_local = block_causal.target_encoding == "block_local"
    if not block_local or not decoder_config.sparse_block_loss:
        logits = _decoder_logits(decoder_input, batch, text_decoder, config)
        return (
            decoder_cross_entropy_values(
                logits,
                decoder_targets,
                batch.text_target_mask,
            ),
            logits if keep_logits else None,
        )

    block_size = block_causal.block_size
    batch_size, text_tokens, latent_dim = decoder_input.shape
    if text_tokens % block_size != 0:
        raise ValueError("decoder text length must be divisible by block_size")
    block_count = text_tokens // block_size
    flat_input = decoder_input.reshape(batch_size * block_count, block_size, latent_dim)
    flat_mask = batch.text_target_mask.reshape(batch_size * block_count, block_size)
    flat_targets = decoder_targets.reshape(batch_size * block_count, block_size)
    active_indices = flat_mask.any(dim=1).nonzero(as_tuple=False).flatten()

    if active_indices.numel() == 0:
        dummy_mask = torch.zeros(
            (1, block_size),
            dtype=torch.bool,
            device=decoder_input.device,
        )
        dummy = text_decoder(flat_input[:1], attention_mask=dummy_mask)
        zero_anchor = dummy.sum() * 0.0
        flat_values = dummy.new_zeros((batch_size * block_count, block_size))
        flat_values = flat_values + zero_anchor
        logits = (
            dummy.new_zeros(
                batch_size * block_count,
                block_size,
                dummy.shape[-1],
            )
            + zero_anchor
            if keep_logits
            else None
        )
    else:
        selected_mask = flat_mask.index_select(0, active_indices)
        active_logits = text_decoder(
            flat_input.index_select(0, active_indices),
            attention_mask=(
                selected_mask if decoder_config.use_attention_mask else None
            ),
        )
        active_values = decoder_cross_entropy_values(
            active_logits,
            flat_targets.index_select(0, active_indices),
            selected_mask,
        )
        flat_values = active_values.new_zeros(
            batch_size * block_count,
            block_size,
        ).index_copy(0, active_indices, active_values)
        logits = (
            active_logits.new_zeros(
                batch_size * block_count,
                block_size,
                active_logits.shape[-1],
            ).index_copy(0, active_indices, active_logits)
            if keep_logits
            else None
        )

    return (
        flat_values.view(batch_size, text_tokens),
        logits.view(batch_size, text_tokens, -1) if logits is not None else None,
    )


def _flow_metric_sum(
    prediction: Tensor,
    noisy_input: Tensor,
    target: Tensor,
    timestep: Tensor,
    mask: Tensor,
    *,
    t_eps: float,
    velocity_target: Tensor | None = None,
) -> Tensor:
    velocity_prediction = x0_to_velocity(
        prediction,
        noisy_input,
        timestep,
        t_eps=t_eps,
    )
    if velocity_target is None:
        velocity_target = x0_to_velocity(
            target,
            noisy_input,
            timestep,
            t_eps=t_eps,
        )
    expanded_mask = mask.unsqueeze(-1)
    safe_prediction = velocity_prediction.masked_fill(~expanded_mask, 0.0)
    safe_target = velocity_target.masked_fill(~expanded_mask, 0.0)
    per_token_mse = (safe_prediction - safe_target).square().mean(dim=-1)
    if per_token_mse.dtype in (torch.float16, torch.bfloat16):
        per_token_mse = per_token_mse.float()
    return torch.where(mask, per_token_mse, 0.0).sum().to(dtype=torch.float64)


def _decoder_metric_sum(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    safe_logits = logits.masked_fill(~mask.unsqueeze(-1), 0.0)
    safe_targets = targets.masked_fill(~mask, 0)
    token_losses = F.cross_entropy(
        safe_logits.reshape(-1, logits.shape[-1]),
        safe_targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    if token_losses.dtype in (torch.float16, torch.bfloat16):
        token_losses = token_losses.float()
    return torch.where(mask, token_losses, 0.0).sum().to(dtype=torch.float64)


def _compose_physical_flow_loss(
    output: MFOutput,
    batch: TrainingBatch,
    context: DistributedContext,
    *,
    t_eps: float,
    collect_metric: bool,
) -> tuple[Tensor, Tensor]:
    """Compose flow loss from registered physical modality heads.

    The physical layout stores target tensors grouped by modality. The output
    contract carries the matching target indices, so no image/text assumptions
    are needed here.
    """

    physical = batch.model_input.physical_layout
    if physical is None:
        raise RuntimeError("physical flow composition requires a physical layout")
    targets = physical.target_latents or {}
    noisy_inputs = physical.noisy_latents or {}
    timesteps = physical.target_timesteps or {}
    target_mask = physical.active_token_mask & physical.target_mask
    expected_modalities = {
        int(modality_id)
        for modality_id in torch.unique(physical.modality_ids[target_mask]).tolist()
    }
    if expected_modalities != set(targets):
        raise ValueError(
            "physical target_latents must cover every target modality; "
            f"expected {sorted(expected_modalities)}, got {sorted(targets)}"
        )
    if expected_modalities != set(noisy_inputs) or expected_modalities != set(timesteps):
        raise ValueError(
            "physical noisy_latents and target_timesteps must cover every target modality"
        )
    predictions = output.modality_pred_norm or {}
    target_indices = output.modality_target_indices or {}
    local_values: list[Tensor] = []
    local_masks: list[Tensor] = []
    for modality_id, target in targets.items():
        if modality_id not in predictions:
            raise RuntimeError(
                f"physical target modality {modality_id} has no model output head"
            )
        if modality_id not in noisy_inputs or modality_id not in timesteps:
            raise ValueError(
                f"physical modality {modality_id} requires noisy inputs and timesteps"
            )
        indices = target_indices.get(modality_id)
        if indices is None:
            raise RuntimeError(
                f"physical target modality {modality_id} has no output alignment"
            )
        prediction = predictions[modality_id].index_select(0, indices)
        noisy_input = noisy_inputs[modality_id]
        timestep = timesteps[modality_id]
        if prediction.shape != target.shape or noisy_input.shape != target.shape:
            raise ValueError(
                f"physical modality {modality_id} target/prediction shapes disagree"
            )
        values = velocity_mse_values(
            prediction,
            noisy_input,
            target,
            timestep,
            torch.ones(target.shape[:-1], dtype=torch.bool, device=target.device),
            t_eps=t_eps,
        )
        local_values.append(values)
        local_masks.append(torch.ones_like(values, dtype=torch.bool))
    if not local_values:
        anchor = next(iter(predictions.values()), None)
        if anchor is None:
            zero = batch.model_input.active_token_mask.sum() * 0.0
        else:
            zero = anchor.sum() * 0.0
        return zero, zero.detach().to(dtype=torch.float64)
    values, masks = torch.cat(local_values), torch.cat(local_masks)
    return masked_ddp_mean(
        values,
        masks,
        context,
        collect_metric=collect_metric,
    )


def _physical_metric_sum(
    output: MFOutput,
    batch: TrainingBatch,
    row_mask: Tensor,
    *,
    t_eps: float,
) -> Tensor:
    """Return the local physical-flow sum for the selected batch rows.

    Physical target maps are concatenated in row-major order by the collator.
    Recovering the corresponding row positions here keeps task metrics aligned
    with the same compiled physical layout used by the objective composer.
    """

    physical = batch.model_input.physical_layout
    if physical is None:
        raise RuntimeError("physical metrics require a physical layout")
    targets = physical.target_latents or {}
    noisy_inputs = physical.noisy_latents or {}
    timesteps = physical.target_timesteps or {}
    target_mask = physical.active_token_mask & physical.target_mask
    predictions = output.modality_pred_norm or {}
    target_indices = output.modality_target_indices or {}
    expected_modalities = {
        int(modality_id)
        for modality_id in torch.unique(physical.modality_ids[target_mask]).tolist()
    }
    if expected_modalities != set(targets):
        raise ValueError(
            "physical target_latents must cover every target modality; "
            f"expected {sorted(expected_modalities)}, got {sorted(targets)}"
        )
    if expected_modalities != set(noisy_inputs) or expected_modalities != set(timesteps):
        raise ValueError(
            "physical noisy_latents and target_timesteps must cover every target modality"
        )

    total = torch.zeros((), dtype=torch.float64, device=target_mask.device)
    for modality_id, target in targets.items():
        if modality_id not in predictions or modality_id not in target_indices:
            raise RuntimeError(
                f"physical modality {modality_id} has no output alignment"
            )
        modality_mask = target_mask & physical.modality_ids.eq(modality_id)
        all_positions = modality_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        selected_positions = (
            (modality_mask & row_mask[:, None])
            .reshape(-1)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if selected_positions.numel() == 0:
            continue
        if all_positions.numel() != target.shape[0]:
            raise ValueError(
                f"physical modality {modality_id} target rows disagree with layout"
            )
        local_positions = torch.searchsorted(all_positions, selected_positions)
        indices = target_indices[modality_id]
        if indices.numel() != target.shape[0]:
            raise ValueError(
                f"physical modality {modality_id} output alignment disagrees with layout"
            )
        selected_indices = indices.index_select(0, local_positions)
        prediction = predictions[modality_id].index_select(0, selected_indices)
        noisy_input = noisy_inputs[modality_id].index_select(0, local_positions)
        selected_target = target.index_select(0, local_positions)
        timestep = timesteps[modality_id].index_select(0, local_positions)
        if prediction.shape != selected_target.shape or noisy_input.shape != selected_target.shape:
            raise ValueError(
                f"physical modality {modality_id} target/prediction shapes disagree"
            )
        values = velocity_mse_values(
            prediction,
            noisy_input,
            selected_target,
            timestep,
            torch.ones(
                selected_target.shape[:-1],
                dtype=torch.bool,
                device=selected_target.device,
            ),
            t_eps=t_eps,
        )
        total = total + values.sum().to(dtype=torch.float64)
    return total


register_objective(
    "modality_flow",
    ObjectiveDefinition(_compose_physical_flow_loss, _physical_metric_sum),
    replace=True,
)


def _physical_objective_name(batch: TrainingBatch) -> str:
    """Resolve one registered objective for the active physical pack."""

    active_tasks = {int(value) for value in batch.task_type.unique().tolist()}
    definitions = {
        int(task_value(definition)): definition
        for definition in task_definitions("metrics")
    }
    names = {
        component
        for task in active_tasks
        for component in definitions[task].loss_components
        if component in OBJECTIVE_REGISTRY.names()
    }
    if not names:
        raise RuntimeError(
            "physical training requires a registered objective in the active task definitions"
        )
    if len(names) != 1:
        raise RuntimeError(
            "one physical pack cannot mix objective composers: "
            + ", ".join(sorted(names))
        )
    return next(iter(names))


def compose_task_metrics(
    output: MFOutput,
    batch: TrainingBatch,
    decoder_logits: Tensor | None,
    decoder_targets: Tensor | None,
    *,
    t_eps: float = 0.05,
    vision_velocity_target: Tensor | None = None,
    text_velocity_target: Tensor | None = None,
) -> TaskMetrics:
    """Build detached local statistics for a later packed logging reduction."""

    definitions, task_components, _ = _metric_registry_snapshot()

    if (decoder_logits is None) != (decoder_targets is None):
        raise ValueError(
            "decoder logits and targets must either both exist or both be absent"
        )

    vision_prediction, vision_noisy_input, vision_target = _flow_tensors(
        output.vision_pred_norm,
        batch.vision_noisy_input_norm,
        batch.vision_target_norm,
        batch.vision_target_mask,
    )
    text_prediction, text_noisy_input, text_target = _flow_tensors(
        output.text_pred_norm,
        batch.text_noisy_input_norm,
        batch.text_target_norm,
        batch.text_target_mask,
    )
    sample_counts: dict[int, Tensor] = {}
    active_tokens: dict[tuple[int, str], Tensor] = {}
    loss_sums: dict[tuple[int, str], Tensor] = {}

    with torch.no_grad():
        for definition in definitions:
            task = task_value(definition)
            rows = batch.task_type == int(task)
            vision_mask = batch.vision_target_mask & rows[:, None]
            text_mask = batch.text_target_mask & rows[:, None]
            sample_counts[task] = rows.sum()
            active_tokens[task, "vision"] = vision_mask.sum()
            active_tokens[task, "text"] = text_mask.sum()

            for component in task_components[int(task)]:
                if component == "vision_flow":
                    value = _flow_metric_sum(
                        vision_prediction,
                        vision_noisy_input,
                        vision_target,
                        batch.model_input.vision_timestep,
                        vision_mask,
                        t_eps=t_eps,
                        velocity_target=vision_velocity_target,
                    )
                elif component == "text_flow":
                    value = _flow_metric_sum(
                        text_prediction,
                        text_noisy_input,
                        text_target,
                        batch.model_input.text_loss_timestep,
                        text_mask,
                        t_eps=t_eps,
                        velocity_target=text_velocity_target,
                    )
                elif component == "text_decoder_ce":
                    if decoder_logits is None or decoder_targets is None:
                        continue
                    value = _decoder_metric_sum(
                        decoder_logits,
                        decoder_targets,
                        text_mask,
                    )
                elif component in OBJECTIVE_REGISTRY.names():
                    objective = OBJECTIVE_REGISTRY.resolve(component)
                    metric_sum = getattr(objective, "metric_sum", None)
                    if not callable(metric_sum):
                        raise RuntimeError(
                            f"registered objective {component!r} has no metric_sum"
                        )
                    value = metric_sum(output, batch, rows, t_eps=t_eps)
                loss_sums[task, component] = value

    return TaskMetrics(
        sample_counts=sample_counts,
        active_tokens=active_tokens,
        loss_sums=loss_sums,
    )


def compose_training_loss(
    output: MFOutput,
    batch: TrainingBatch,
    text_decoder: LatentTextDecoder,
    generator: torch.Generator,
    context: DistributedContext,
    config: MFConfig,
    *,
    collect_task_metrics: bool = False,
    collect_loss_metrics: bool = True,
    vision_velocity_target: Tensor | None = None,
    text_velocity_target: Tensor | None = None,
) -> tuple[LossOutput, LossMetrics]:
    """Compose normalized flow losses and raw-latent decoder CE in DDP order."""

    if batch.model_input.physical_layout is not None:
        objective = OBJECTIVE_REGISTRY.resolve(_physical_objective_name(batch))
        physical_flow, physical_metric = objective(
            output,
            batch,
            context,
            t_eps=config.flow.velocity_t_eps,
            collect_metric=collect_loss_metrics,
        )
        zero = physical_flow.new_zeros(())
        zero_metric = physical_metric.new_zeros(())
        task_metrics = (
            compose_task_metrics(
                output,
                batch,
                None,
                None,
                t_eps=config.flow.velocity_t_eps,
            )
            if collect_task_metrics
            else None
        )
        return (
            LossOutput(
                total=physical_flow,
                vision_flow=zero,
                text_flow=zero,
                text_decoder_ce=zero,
                physical_flow=physical_flow,
            ).validate(),
            LossMetrics(
                total=physical_metric,
                vision_flow=zero_metric,
                text_flow=zero_metric,
                text_decoder_ce=zero_metric,
                physical_flow=physical_metric,
                task=task_metrics,
            ),
        )

    vision_prediction, vision_noisy_input, vision_target = _flow_tensors(
        output.vision_pred_norm,
        batch.vision_noisy_input_norm,
        batch.vision_target_norm,
        batch.vision_target_mask,
    )
    vision_values = velocity_mse_values(
        vision_prediction,
        vision_noisy_input,
        vision_target,
        batch.model_input.vision_timestep,
        batch.vision_target_mask,
        t_eps=config.flow.velocity_t_eps,
        velocity_target=vision_velocity_target,
    )

    text_prediction, text_noisy_input, text_target = _flow_tensors(
        output.text_pred_norm,
        batch.text_noisy_input_norm,
        batch.text_target_norm,
        batch.text_target_mask,
    )
    text_values = velocity_mse_values(
        text_prediction,
        text_noisy_input,
        text_target,
        batch.model_input.text_loss_timestep,
        batch.text_target_mask,
        t_eps=config.flow.velocity_t_eps,
        velocity_target=text_velocity_target,
    )

    raw_decoder_latents, decoder_targets = _decoder_inputs(batch, text_decoder)
    decoder_config = config.model.text_decoder
    corruption = corrupt_text_decoder_latents(
        raw_decoder_latents,
        generator,
        p_mean=decoder_config.decoder_p_mean,
        p_std=decoder_config.decoder_p_std,
        noise_scale=decoder_config.decoder_noise_scale,
    )
    decoder_values, decoder_logits = _decoder_loss_values(
        corruption.corrupted,
        decoder_targets,
        batch,
        text_decoder,
        config,
        keep_logits=collect_task_metrics,
    )
    definitions, _, _ = _metric_registry_snapshot()
    vision_component_mask = _component_task_mask(
        batch.task_type,
        batch.vision_target_mask,
        "vision_flow",
        definitions,
    )
    text_component_mask = _component_task_mask(
        batch.task_type,
        batch.text_target_mask,
        "text_flow",
        definitions,
    )
    decoder_component_mask = _component_task_mask(
        batch.task_type,
        batch.text_target_mask,
        "text_decoder_ce",
        definitions,
    )
    reduced_losses, reduced_metrics = masked_ddp_means(
        (
            (vision_values, vision_component_mask),
            (text_values, text_component_mask),
            (decoder_values, decoder_component_mask),
        ),
        context,
        collect_metric=collect_loss_metrics,
    )
    vision_flow, text_flow, text_decoder_ce = reduced_losses
    vision_metric, text_metric, decoder_metric = reduced_metrics

    losses = LossOutput(
        total=vision_flow + text_flow + text_decoder_ce,
        vision_flow=vision_flow,
        text_flow=text_flow,
        text_decoder_ce=text_decoder_ce,
        physical_flow=vision_flow.new_zeros(()),
    ).validate()
    metrics = LossMetrics(
        total=vision_metric + text_metric + decoder_metric,
        vision_flow=vision_metric,
        text_flow=text_metric,
        text_decoder_ce=decoder_metric,
        physical_flow=vision_metric.new_zeros(()),
        task=(
            compose_task_metrics(
                output,
                batch,
                decoder_logits,
                decoder_targets,
                t_eps=config.flow.velocity_t_eps,
                vision_velocity_target=vision_velocity_target,
                text_velocity_target=text_velocity_target,
            )
            if collect_task_metrics
            else None
        ),
    )
    return losses, metrics


def compose_flow_only_training_loss(
    output: MFOutput,
    batch: TrainingBatch,
    text_decoder: LatentTextDecoder | None,
    generator: torch.Generator,
    context: DistributedContext,
    config: MFConfig,
    *,
    collect_task_metrics: bool = False,
    collect_loss_metrics: bool = True,
    vision_velocity_target: Tensor | None = None,
    text_velocity_target: Tensor | None = None,
) -> tuple[LossOutput, LossMetrics]:
    """Compose only normalized-latent DiT losses for diagnostics."""

    if batch.model_input.physical_layout is not None:
        objective = OBJECTIVE_REGISTRY.resolve(_physical_objective_name(batch))
        physical_flow, physical_metric = objective(
            output,
            batch,
            context,
            t_eps=config.flow.velocity_t_eps,
            collect_metric=collect_loss_metrics,
        )
        zero = physical_flow.new_zeros(())
        zero_metric = physical_metric.new_zeros(())
        task_metrics = (
            compose_task_metrics(
                output,
                batch,
                None,
                None,
                t_eps=config.flow.velocity_t_eps,
            )
            if collect_task_metrics
            else None
        )
        return (
            LossOutput(
                total=physical_flow,
                vision_flow=zero,
                text_flow=zero,
                text_decoder_ce=zero,
                physical_flow=physical_flow,
            ).validate(),
            LossMetrics(
                total=physical_metric,
                vision_flow=zero_metric,
                text_flow=zero_metric,
                text_decoder_ce=zero_metric,
                physical_flow=physical_metric,
                task=task_metrics,
            ),
        )

    del text_decoder, generator
    vision_prediction, vision_noisy_input, vision_target = _flow_tensors(
        output.vision_pred_norm,
        batch.vision_noisy_input_norm,
        batch.vision_target_norm,
        batch.vision_target_mask,
    )
    vision_flow, vision_metric = masked_velocity_mse_loss(
        vision_prediction,
        vision_noisy_input,
        vision_target,
        batch.model_input.vision_timestep,
        batch.vision_target_mask,
        context,
        t_eps=config.flow.velocity_t_eps,
        velocity_target=vision_velocity_target,
        collect_metric=collect_loss_metrics,
    )

    text_prediction, text_noisy_input, text_target = _flow_tensors(
        output.text_pred_norm,
        batch.text_noisy_input_norm,
        batch.text_target_norm,
        batch.text_target_mask,
    )
    text_flow, text_metric = masked_velocity_mse_loss(
        text_prediction,
        text_noisy_input,
        text_target,
        batch.model_input.text_loss_timestep,
        batch.text_target_mask,
        context,
        t_eps=config.flow.velocity_t_eps,
        velocity_target=text_velocity_target,
        collect_metric=collect_loss_metrics,
    )

    decoder_zero = vision_flow.new_zeros(())
    decoder_metric_zero = vision_metric.new_zeros(())
    losses = LossOutput(
        total=vision_flow + text_flow,
        vision_flow=vision_flow,
        text_flow=text_flow,
        text_decoder_ce=decoder_zero,
        physical_flow=decoder_zero,
    ).validate()
    metrics = LossMetrics(
        total=vision_metric + text_metric,
        vision_flow=vision_metric,
        text_flow=text_metric,
        text_decoder_ce=decoder_metric_zero,
        physical_flow=decoder_metric_zero,
        task=(
            compose_task_metrics(
                output,
                batch,
                None,
                None,
                t_eps=config.flow.velocity_t_eps,
                vision_velocity_target=vision_velocity_target,
                text_velocity_target=text_velocity_target,
            )
            if collect_task_metrics
            else None
        ),
    )
    return losses, metrics
