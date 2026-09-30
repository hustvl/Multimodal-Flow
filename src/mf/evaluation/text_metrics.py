from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

import torch
from torch import Tensor


class TokenizerDecoder(Protocol):
    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class TextMetrics:
    perplexity: float
    mean_entropy: float
    token_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "ppl": self.perplexity,
            "mean_entropy": self.mean_entropy,
            "token_count": self.token_count,
        }


def _metric_sums(
    logits: Tensor,
    target_ids: Tensor,
    *,
    pad_token_id: int,
) -> tuple[float, float, int]:
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if (
        not isinstance(target_ids, Tensor)
        or target_ids.ndim != 2
        or tuple(target_ids.shape) != tuple(logits.shape[:2])
    ):
        raise ValueError("target_ids shape must match logits [B, T]")
    if target_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("target_ids must have an integer dtype")
    if target_ids.device != logits.device:
        raise ValueError("target_ids and logits must be on the same device")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise ValueError("pad_token_id must be an integer")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must have a floating-point dtype")

    active = target_ids.ne(pad_token_id)
    token_count = int(active.sum().item())
    if token_count == 0:
        raise ValueError("text metrics require at least one non-PAD token")
    active_targets = target_ids[active]
    if bool(((active_targets < 0) | (active_targets >= logits.shape[-1])).any()):
        raise ValueError("non-PAD target ids must be valid vocabulary indices")

    stable_logits = logits.to(dtype=torch.float64)
    log_probs = torch.log_softmax(stable_logits, dim=-1)
    gather_ids = target_ids.clamp(0, logits.shape[-1] - 1).unsqueeze(-1)
    nll = -log_probs.gather(-1, gather_ids).squeeze(-1)
    probabilities = log_probs.exp()
    entropy = -(probabilities * log_probs).sum(dim=-1)
    return (
        float(nll[active].sum().item()),
        float(entropy[active].sum().item()),
        token_count,
    )


def _metrics(nll_sum: float, entropy_sum: float, token_count: int) -> TextMetrics:
    if token_count <= 0:
        raise ValueError("text metrics require at least one non-PAD token")
    return TextMetrics(
        perplexity=math.exp(nll_sum / token_count),
        mean_entropy=entropy_sum / token_count,
        token_count=token_count,
    )


def text_metrics_from_logits(
    logits: Tensor,
    target_ids: Tensor,
    *,
    pad_token_id: int,
) -> TextMetrics:
    return _metrics(*_metric_sums(logits, target_ids, pad_token_id=pad_token_id))


class TextMetricAccumulator:
    def __init__(self, *, pad_token_id: int) -> None:
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
            raise ValueError("pad_token_id must be an integer")
        self.pad_token_id = pad_token_id
        self._nll_sum = 0.0
        self._entropy_sum = 0.0
        self._token_count = 0

    def update(self, logits: Tensor, target_ids: Tensor) -> None:
        nll_sum, entropy_sum, token_count = _metric_sums(
            logits,
            target_ids,
            pad_token_id=self.pad_token_id,
        )
        self._nll_sum += nll_sum
        self._entropy_sum += entropy_sum
        self._token_count += token_count

    def state_dict(self) -> dict[str, float | int]:
        return {
            "nll_sum": self._nll_sum,
            "entropy_sum": self._entropy_sum,
            "token_count": self._token_count,
        }

    def merge(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping) or set(state) != {
            "nll_sum",
            "entropy_sum",
            "token_count",
        }:
            raise ValueError("text metric state fields are invalid")
        nll_sum = state["nll_sum"]
        entropy_sum = state["entropy_sum"]
        token_count = state["token_count"]
        if (
            isinstance(nll_sum, bool)
            or not isinstance(nll_sum, Real)
            or not math.isfinite(float(nll_sum))
            or isinstance(entropy_sum, bool)
            or not isinstance(entropy_sum, Real)
            or not math.isfinite(float(entropy_sum))
            or isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise ValueError("text metric state values are invalid")
        self._nll_sum += float(nll_sum)
        self._entropy_sum += float(entropy_sum)
        self._token_count += token_count

    def compute(self) -> TextMetrics:
        return _metrics(self._nll_sum, self._entropy_sum, self._token_count)


def render_token_ids(
    token_ids: Tensor | Sequence[int],
    *,
    tokenizer: TokenizerDecoder,
    pad_token_id: int,
    eos_token_id: int,
) -> str:
    if isinstance(token_ids, Tensor):
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be one-dimensional")
        values = token_ids.detach().cpu().tolist()
    else:
        values = list(token_ids)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("token_ids must contain integers")

    segments: list[list[int]] = [[]]
    for token_id in values:
        if token_id == pad_token_id:
            continue
        if token_id == eos_token_id:
            segments.append([])
        else:
            segments[-1].append(token_id)
    lines: list[str] = []
    for segment in segments:
        if not segment:
            continue
        decoded = tokenizer.decode(segment, skip_special_tokens=True)
        normalized = " ".join(decoded.split())
        if normalized:
            lines.append(normalized)
    return "\n".join(lines)
