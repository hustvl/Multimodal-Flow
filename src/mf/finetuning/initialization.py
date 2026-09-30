from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class InitializationReport:
    checkpoint: Path
    used_ema: bool
    loaded_parameter_count: int


def _load_mapping(path: Path) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint payload must be a mapping: {path}")
    return payload


def _module_state(
    payload: Mapping[str, object], *, filename: str
) -> Mapping[str, Tensor]:
    state = payload.get("state")
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in state.items()
    ):
        raise ValueError(f"{filename} must contain a tensor state mapping")
    return state


def _ema_parameter_state(
    payload: Mapping[str, object],
    *,
    owner: str,
) -> dict[str, Tensor]:
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("ema.pt must contain an EMA state mapping")
    keys = state.get("parameter_keys")
    shadows = state.get("shadows")
    if (
        not isinstance(keys, tuple)
        or not isinstance(shadows, tuple)
        or len(keys) != len(shadows)
    ):
        raise ValueError("EMA parameter identities and shadows must be aligned tuples")

    prefix = owner + "."
    output: dict[str, Tensor] = {}
    for key, shadow in zip(keys, shadows, strict=True):
        if not isinstance(key, str) or not isinstance(shadow, Tensor):
            raise ValueError("EMA parameter entries are malformed")
        fields = key.split(":", 4)
        if len(fields) != 5 or not fields[4].startswith(prefix):
            continue
        name = fields[4].removeprefix(prefix)
        if name in output:
            raise ValueError(f"duplicate EMA parameter identity: {owner}.{name}")
        output[name] = shadow
    return output


@torch.no_grad()
def _apply_ema(module: nn.Module, state: Mapping[str, Tensor], *, owner: str) -> int:
    named_parameters = dict(module.named_parameters())
    unexpected = sorted(set(state) - set(named_parameters))
    if unexpected:
        raise ValueError(f"EMA/{owner} parameter mismatch (unexpected={unexpected})")

    # EMA tracks optimizer-owned parameters only. Parameters frozen during
    # pretraining remain at the live values loaded from the complete model state.
    for name, shadow in state.items():
        parameter = named_parameters[name]
        if shadow.shape != parameter.shape:
            raise ValueError(f"EMA shape mismatch for {owner}.{name}")
        parameter.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))
    return len(state)


@torch.no_grad()
def initialize_from_pretrain_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    text_decoder: nn.Module | None = None,
    use_ema: bool = True,
) -> InitializationReport:
    """Load pretrained live weights only; optimizer, schedule and cursors remain new."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    if text_decoder is not None and not isinstance(text_decoder, nn.Module):
        raise TypeError("text_decoder must be a torch module")
    root = Path(checkpoint).expanduser().resolve()
    if not (root / "COMPLETED").is_file():
        raise ValueError(f"initialization checkpoint is not complete: {root}")
    native_format = (root / "model.pt").is_file()
    epoch_format = (root / "shared.pt").is_file()
    if native_format == epoch_format:
        raise ValueError(
            "initialization checkpoint must contain exactly one supported weight format"
        )
    if epoch_format:
        shared = _load_mapping(root / "shared.pt")
        model_payload: Mapping[str, object] = {"state": shared.get("model")}
        decoder_payload: Mapping[str, object] = {"state": shared.get("text_decoder")}
        ema_payload: Mapping[str, object] = {"state": shared.get("ema")}
        model_filename = "shared.pt:model"
        decoder_filename = "shared.pt:text_decoder"
    else:
        model_payload = _load_mapping(root / "model.pt")
        decoder_payload = (
            _load_mapping(root / "text_decoder.pt") if text_decoder is not None else {}
        )
        ema_payload = _load_mapping(root / "ema.pt") if use_ema else {}
        model_filename = "model.pt"
        decoder_filename = "text_decoder.pt"

    model.load_state_dict(
        _module_state(model_payload, filename=model_filename),
        strict=True,
    )
    if text_decoder is not None:
        text_decoder.load_state_dict(
            _module_state(decoder_payload, filename=decoder_filename),
            strict=True,
        )

    loaded = len(tuple(model.parameters()))
    if text_decoder is not None:
        loaded += len(tuple(text_decoder.parameters()))
    if use_ema:
        ema_loaded = _apply_ema(
            model,
            _ema_parameter_state(ema_payload, owner="backbone"),
            owner="backbone",
        )
        if text_decoder is not None:
            ema_loaded += _apply_ema(
                text_decoder,
                _ema_parameter_state(ema_payload, owner="text_decoder"),
                owner="text_decoder",
            )
        if ema_loaded == 0:
            raise ValueError("ema.pt covers neither the backbone nor the text decoder")

    return InitializationReport(
        checkpoint=root,
        used_ema=use_ema,
        loaded_parameter_count=loaded,
    )
