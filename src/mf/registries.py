"""Small named registries shared by the train and inference contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class NamedRegistry:
    """Register one callable per stable extension name."""

    def __init__(self, kind: str) -> None:
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("registry kind must be non-empty")
        self.kind = kind.strip()
        self._builders: dict[str, Callable[..., Any]] = {}
        self._frozen = False

    def register(
        self,
        name: str,
        builder: Callable[..., Any],
        *,
        replace: bool = False,
    ) -> Callable[..., Any]:
        if self._frozen:
            raise RuntimeError(
                f"{self.kind} registry is frozen; register extensions before "
                "constructing the model and data pipeline"
            )
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{self.kind} name must be non-empty")
        if not callable(builder):
            raise TypeError(f"{self.kind} builder must be callable")
        name = name.strip()
        if name in self._builders and not replace:
            raise ValueError(f"{self.kind} is already registered: {name!r}")
        self._builders[name] = builder
        return builder

    def freeze(self) -> None:
        """Freeze the registry for one model/data-pipeline lifecycle."""

        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def resolve(self, name: str) -> Callable[..., Any]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{self.kind} name must be non-empty")
        try:
            return self._builders[name.strip()]
        except KeyError as error:
            known = ", ".join(sorted(self._builders)) or "<none>"
            raise ValueError(
                f"unknown {self.kind} {name!r}; available: {known}"
            ) from error

    def names(self) -> tuple[str, ...]:
        return tuple(self._builders)

    def manifest(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "name": name,
                "builder": (
                    f"{getattr(builder, '__module__', type(builder).__module__)}:"
                    f"{getattr(builder, '__qualname__', type(builder).__qualname__)}"
                ),
            }
            for name, builder in self._builders.items()
        )


class ExtensionRegistry:
    """Unified lifecycle facade for all MF extension namespaces.

    The component registries remain typed so that a physical layout cannot be
    confused with a codec or an objective. This facade is the single public
    boundary for extension validation, freezing, registration and manifest
    generation.
    """

    @property
    def named(self) -> tuple[NamedRegistry, ...]:
        return (
            PHYSICAL_LAYOUT_REGISTRY,
            OBJECTIVE_REGISTRY,
            OUTPUT_HEAD_REGISTRY,
            SAMPLER_REGISTRY,
            DECODER_REGISTRY,
            GENERATION_REGISTRY,
        )

    def validate_bindings(self) -> None:
        from mf.codecs.registry import (
            CODEC_REGISTRY,
            TEXT_CODEC_REGISTRY,
            VISION_CODEC_REGISTRY,
        )
        from mf.contracts.sequence import MODALITY_REGISTRY

        for definition in MODALITY_REGISTRY.definitions():
            if definition.codec_name == "vision":
                if not VISION_CODEC_REGISTRY.names():
                    raise ValueError(
                        f"modality {definition.name!r} uses the vision codec "
                        "family, but no vision codec is registered"
                    )
                continue
            if definition.codec_name == "text":
                if not TEXT_CODEC_REGISTRY.names():
                    raise ValueError(
                        f"modality {definition.name!r} uses the text codec "
                        "family, but no text codec is registered"
                    )
                continue
            codec = CODEC_REGISTRY.resolve(definition.codec_name)
            if codec.modality != definition.name:
                raise ValueError(
                    f"modality {definition.name!r} references codec "
                    f"{definition.codec_name!r} for modality {codec.modality!r}"
                )

    def freeze(self) -> None:
        """Validate and freeze one complete model/data lifecycle."""

        from mf.codecs.registry import CODEC_REGISTRY
        from mf.contracts.sequence import MODALITY_REGISTRY
        from mf.contracts.task_registry import DEFAULT_TASK_REGISTRY

        self.validate_bindings()
        CODEC_REGISTRY.freeze()
        MODALITY_REGISTRY.freeze()
        DEFAULT_TASK_REGISTRY.freeze()
        self.freeze_named()

    def freeze_named(self) -> None:
        for registry in self.named:
            registry.freeze()

    def manifest(self) -> dict[str, object]:
        from mf.codecs.registry import (
            CODEC_REGISTRY,
            TEXT_CODEC_REGISTRY,
            VISION_CODEC_REGISTRY,
        )
        from mf.contracts.sequence import MODALITY_REGISTRY
        from mf.contracts.task_registry import DEFAULT_TASK_REGISTRY

        return {
            "physical_layout": PHYSICAL_LAYOUT_REGISTRY.manifest(),
            "objective": OBJECTIVE_REGISTRY.manifest(),
            "output_head": OUTPUT_HEAD_REGISTRY.manifest(),
            "sampler": SAMPLER_REGISTRY.manifest(),
            "decoder": DECODER_REGISTRY.manifest(),
            "generation": GENERATION_REGISTRY.manifest(),
            "modality": MODALITY_REGISTRY.manifest(),
            "task": DEFAULT_TASK_REGISTRY.manifest(),
            "codec": CODEC_REGISTRY.manifest(),
            "vision_codec": VISION_CODEC_REGISTRY.manifest(),
            "text_codec": TEXT_CODEC_REGISTRY.manifest(),
        }

    def register_codec(self, *args: Any, **kwargs: Any) -> Any:
        from mf.codecs.registry import register_codec

        return register_codec(*args, **kwargs)

    def register_modality(self, *args: Any, **kwargs: Any) -> Any:
        from mf.contracts.sequence import register_modality

        return register_modality(*args, **kwargs)

    def register_task(self, *args: Any, **kwargs: Any) -> Any:
        from mf.contracts.task_registry import register_task

        return register_task(*args, **kwargs)

    def register_physical_layout(self, *args: Any, **kwargs: Any) -> Any:
        return PHYSICAL_LAYOUT_REGISTRY.register(*args, **kwargs)

    def register_objective(self, *args: Any, **kwargs: Any) -> Any:
        return OBJECTIVE_REGISTRY.register(*args, **kwargs)

    def register_output_head(self, *args: Any, **kwargs: Any) -> Any:
        return OUTPUT_HEAD_REGISTRY.register(*args, **kwargs)

    def register_sampler(self, *args: Any, **kwargs: Any) -> Any:
        return SAMPLER_REGISTRY.register(*args, **kwargs)

    def register_decoder(self, *args: Any, **kwargs: Any) -> Any:
        return DECODER_REGISTRY.register(*args, **kwargs)

    def register_generation(self, *args: Any, **kwargs: Any) -> Any:
        return GENERATION_REGISTRY.register(*args, **kwargs)


PHYSICAL_LAYOUT_REGISTRY = NamedRegistry("physical layout")
OBJECTIVE_REGISTRY = NamedRegistry("objective")
OUTPUT_HEAD_REGISTRY = NamedRegistry("output head")
SAMPLER_REGISTRY = NamedRegistry("sampler")
DECODER_REGISTRY = NamedRegistry("decoder")
GENERATION_REGISTRY = NamedRegistry("generation")
EXTENSION_REGISTRY = ExtensionRegistry()


def freeze_named_registries() -> None:
    EXTENSION_REGISTRY.freeze_named()


def registry_manifest() -> dict[str, object]:
    return EXTENSION_REGISTRY.manifest()


def register_physical_layout(
    name: str, builder: Callable[..., Any], *, replace: bool = False
) -> Callable[..., Any]:
    return PHYSICAL_LAYOUT_REGISTRY.register(name, builder, replace=replace)


def register_objective(
    name: str, builder: Callable[..., Any], *, replace: bool = False
) -> Callable[..., Any]:
    return OBJECTIVE_REGISTRY.register(name, builder, replace=replace)


def register_output_head(
    name: str, builder: Callable[..., Any], *, replace: bool = False
) -> Callable[..., Any]:
    return OUTPUT_HEAD_REGISTRY.register(name, builder, replace=replace)


def register_sampler(
    name: str, builder: Callable[..., Any], *, replace: bool = False
) -> Callable[..., Any]:
    return SAMPLER_REGISTRY.register(name, builder, replace=replace)


def register_decoder(
    name: str, builder: Callable[..., Any], *, replace: bool = False
) -> Callable[..., Any]:
    return DECODER_REGISTRY.register(name, builder, replace=replace)


def register_generation(
    name: str, builder: Callable[..., Any], *, replace: bool = False
) -> Callable[..., Any]:
    """Register a complete physical denoise/decode generation loop."""

    return GENERATION_REGISTRY.register(name, builder, replace=replace)


__all__ = [
    "OBJECTIVE_REGISTRY",
    "DECODER_REGISTRY",
    "EXTENSION_REGISTRY",
    "ExtensionRegistry",
    "GENERATION_REGISTRY",
    "OUTPUT_HEAD_REGISTRY",
    "PHYSICAL_LAYOUT_REGISTRY",
    "SAMPLER_REGISTRY",
    "NamedRegistry",
    "freeze_named_registries",
    "registry_manifest",
    "register_decoder",
    "register_generation",
    "register_objective",
    "register_output_head",
    "register_physical_layout",
    "register_sampler",
]
