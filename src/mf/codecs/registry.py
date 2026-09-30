"""Registry for configuration-driven codec construction across modalities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from torch import nn

from mf.registries import DECODER_REGISTRY, register_decoder


CodecConfig = Any
CodecBuilder = Callable[[CodecConfig], nn.Module]


@dataclass(frozen=True, slots=True)
class CodecDefinition:
    """Construction contract for one named codec."""

    name: str
    encoder_builder: CodecBuilder
    decoder_builder: CodecBuilder | None = None
    capabilities: frozenset[str] = frozenset()
    modality: str = "vision"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("codec name must be non-empty")
        if not self.modality:
            raise ValueError("codec modality must be non-empty")
        if not callable(self.encoder_builder):
            raise TypeError("codec encoder_builder must be callable")
        if self.decoder_builder is not None and not callable(self.decoder_builder):
            raise TypeError("codec decoder_builder must be callable")


class CodecRegistry:
    """Ordered registry for codecs across all modalities.

    A registry may be used as the generic codec namespace or as a typed view
    over that namespace. The built-in vision/text registries are views, not
    separate stores, so custom modalities share name resolution and the
    checkpoint fingerprint contract with built-in codecs.
    """

    def __init__(
        self,
        modality: str | None = None,
        *,
        definitions: dict[str, CodecDefinition] | None = None,
        freeze_state: dict[str, bool] | None = None,
    ) -> None:
        if modality is not None and not modality:
            raise ValueError("codec registry modality must be non-empty")
        self.modality = modality
        self._definitions = definitions if definitions is not None else {}
        self._freeze_state = freeze_state if freeze_state is not None else {"frozen": False}

    @property
    def _label(self) -> str:
        return self.modality or "generic"

    def register(
        self,
        definition: CodecDefinition,
        *,
        replace: bool = False,
    ) -> CodecDefinition:
        if self.is_frozen:
            raise RuntimeError(
                f"{self._label} codec registry is frozen; register extensions "
                "before constructing the model and data pipeline"
            )
        if self.modality is not None and definition.modality != self.modality:
            raise ValueError(
                f"codec {definition.name!r} belongs to modality "
                f"{definition.modality!r}, expected {self.modality!r}"
            )
        if definition.name in self._definitions and not replace:
            raise ValueError(f"codec already registered: {definition.name!r}")
        self._definitions[definition.name] = definition
        if definition.decoder_builder is not None:
            register_decoder(
                definition.name,
                definition.decoder_builder,
                replace=replace,
            )
        return definition

    def freeze(self) -> None:
        self._freeze_state["frozen"] = True

    @property
    def is_frozen(self) -> bool:
        return self._freeze_state["frozen"]

    def _visible_definitions(self) -> tuple[CodecDefinition, ...]:
        definitions = tuple(self._definitions.values())
        if self.modality is None:
            return definitions
        return tuple(item for item in definitions if item.modality == self.modality)

    def resolve(self, config_or_name: CodecConfig | str) -> CodecDefinition:
        name = (
            config_or_name
            if isinstance(config_or_name, str)
            else getattr(config_or_name, "name", None)
        )
        if not isinstance(name, str) or not name:
            raise TypeError("codec config must expose a non-empty name")
        try:
            definition = self._definitions[name]
        except KeyError as error:
            known = ", ".join(sorted(self.names())) or "<none>"
            raise ValueError(
                f"{self._label} codec {name!r} is not registered; available: {known}"
            ) from error
        if self.modality is not None and definition.modality != self.modality:
            known = ", ".join(sorted(self.names())) or "<none>"
            raise ValueError(
                f"{self._label} codec {name!r} is not registered; available: {known}"
            )
        return definition

    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self._visible_definitions())

    def definitions(self) -> tuple[CodecDefinition, ...]:
        return self._visible_definitions()

    @staticmethod
    def _builder_identity(builder: CodecBuilder | None) -> str | None:
        if builder is None:
            return None
        return (
            f"{getattr(builder, '__module__', type(builder).__module__)}:"
            f"{getattr(builder, '__qualname__', type(builder).__qualname__)}"
        )

    def manifest(self) -> tuple[dict[str, object], ...]:
        """Return stable codec semantics for checkpoint fingerprints."""

        return tuple(
            {
                "name": definition.name,
                "modality": definition.modality,
                "capabilities": tuple(sorted(definition.capabilities)),
                "encoder": self._builder_identity(definition.encoder_builder),
                "decoder": self._builder_identity(definition.decoder_builder),
            }
            for definition in self.definitions()
        )

    def build_encoder(self, config: CodecConfig) -> nn.Module:
        return self.resolve(config).encoder_builder(config)

    def build_decoder(self, config: CodecConfig) -> nn.Module:
        definition = self.resolve(config)
        builder = definition.decoder_builder
        if builder is None:
            try:
                builder = DECODER_REGISTRY.resolve(definition.name)
            except ValueError as error:
                raise ValueError(f"codec has no decoder: {definition.name!r}") from error
        return builder(config)


VisionCodecDefinition = CodecDefinition


class VisionCodecRegistry(CodecRegistry):
    def __init__(self, parent: CodecRegistry | None = None) -> None:
        super().__init__(
            "vision",
            definitions=None if parent is None else parent._definitions,
            freeze_state=None if parent is None else parent._freeze_state,
        )


class TextCodecRegistry(CodecRegistry):
    def __init__(self, parent: CodecRegistry | None = None) -> None:
        super().__init__(
            "text",
            definitions=None if parent is None else parent._definitions,
            freeze_state=None if parent is None else parent._freeze_state,
        )


CODEC_REGISTRY = CodecRegistry()
VISION_CODEC_REGISTRY = VisionCodecRegistry(CODEC_REGISTRY)
TEXT_CODEC_REGISTRY = TextCodecRegistry(CODEC_REGISTRY)


def register_codec(
    name: str,
    *,
    modality: str,
    encoder: CodecBuilder,
    decoder: CodecBuilder | None = None,
    capabilities: Iterable[str] = (),
    replace: bool = False,
) -> CodecDefinition:
    """Register a codec for any named modality."""

    return CODEC_REGISTRY.register(
        CodecDefinition(
            name=name,
            encoder_builder=encoder,
            decoder_builder=decoder,
            capabilities=frozenset(capabilities),
            modality=modality,
        ),
        replace=replace,
    )


def register_vision_codec(
    name: str,
    *,
    encoder: CodecBuilder,
    decoder: CodecBuilder | None = None,
    capabilities: Iterable[str] = (),
    replace: bool = False,
) -> VisionCodecDefinition:
    """Register a named vision codec for configuration-driven construction."""

    return register_codec(
        name,
        modality="vision",
        encoder=encoder,
        decoder=decoder,
        capabilities=capabilities,
        replace=replace,
    )


def register_text_codec(
    name: str,
    *,
    encoder: CodecBuilder,
    decoder: CodecBuilder | None = None,
    capabilities: Iterable[str] = (),
    replace: bool = False,
) -> CodecDefinition:
    """Register a named text codec for configuration-driven construction."""

    return register_codec(
        name,
        modality="text",
        encoder=encoder,
        decoder=decoder,
        capabilities=capabilities,
        replace=replace,
    )


__all__ = [
    "CodecBuilder",
    "CodecDefinition",
    "CodecRegistry",
    "CODEC_REGISTRY",
    "TEXT_CODEC_REGISTRY",
    "TextCodecRegistry",
    "VISION_CODEC_REGISTRY",
    "VisionCodecDefinition",
    "VisionCodecRegistry",
    "register_codec",
    "register_vision_codec",
    "register_text_codec",
]
