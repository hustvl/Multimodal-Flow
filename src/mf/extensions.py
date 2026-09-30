"""Shared extension loading for training, inference, and evaluation.

An extension is a normal Python module whose import registers codecs, tasks,
recipes, or physical layout adapters.  The same loader is used by every
public entry point so a checkpoint cannot be trained with one registry and
silently evaluated with another.
"""

from __future__ import annotations

import importlib
import hashlib
import os
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

from mf.registries import EXTENSION_REGISTRY


EXTENSION_CONTRACT_VERSION = "1"
_LOADED_EXTENSIONS: dict[str, ModuleType] = {}
_REGISTRIES_FROZEN = False


def _module_source_hash(module: ModuleType) -> str | None:
    origin = getattr(module, "__file__", None)
    if not origin:
        return None
    source = Path(origin)
    if source.suffix in {".pyc", ".pyo"}:
        source = source.with_suffix(".py")
    if not source.is_file():
        return None
    return hashlib.sha256(source.read_bytes()).hexdigest()


def load_extensions(specs: Iterable[str] = ()) -> tuple[str, ...]:
    """Import explicit modules and ``MF_EXTENSION_MODULES`` exactly once."""

    configured = [item.strip() for item in os.environ.get("MF_EXTENSION_MODULES", "").split(",")]
    requested = tuple(item for item in (*configured, *specs) if item)
    if _REGISTRIES_FROZEN:
        late = tuple(item for item in requested if item not in _LOADED_EXTENSIONS)
        if late:
            raise RuntimeError(
                "extensions are frozen; load all extension modules before "
                f"freezing the registries: {', '.join(late)}"
            )
    for name in requested:
        if name not in _LOADED_EXTENSIONS:
            _LOADED_EXTENSIONS[name] = importlib.import_module(name)
    return tuple(_LOADED_EXTENSIONS)


def freeze_extensions() -> dict[str, object]:
    """Freeze all extension registries before model and pipeline construction."""

    global _REGISTRIES_FROZEN
    # Import built-ins before freezing. These modules perform the same normal
    # registration work as user extensions, but must be part of every runtime.
    for module_name in (
        "mf.codecs.factory",
        "mf.data.runtime",
        "mf.modeling.chunk_causal_layout",
        "mf.modeling.heads",
        "mf.training.metrics",
        "mf.inference.generation",
    ):
        importlib.import_module(module_name)
    EXTENSION_REGISTRY.freeze()
    _REGISTRIES_FROZEN = True
    return extension_manifest()


def extension_manifest() -> dict[str, object]:
    """Return the registry identity included in training fingerprints."""

    return {
        "contract_version": EXTENSION_CONTRACT_VERSION,
        "modules": tuple(
            {
                "name": name,
                "version": getattr(module, "__mf_extension_version__", "unversioned"),
                "source_sha256": _module_source_hash(module),
            }
            for name, module in sorted(_LOADED_EXTENSIONS.items())
        ),
        "registries_frozen": _REGISTRIES_FROZEN,
        "registries": EXTENSION_REGISTRY.manifest(),
    }


__all__ = [
    "EXTENSION_CONTRACT_VERSION",
    "EXTENSION_REGISTRY",
    "extension_manifest",
    "freeze_extensions",
    "load_extensions",
]
