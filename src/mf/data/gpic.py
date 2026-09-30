from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from torch import Tensor
from mf.config.schema import MFConfig
from mf.data.sources import expected_source, normalized_data_config
from mf.data.planner import stable_hash
from mf.data.readers import LazyIndexedReader, SourceProvenance, require_reader_metadata

CaptionType = Literal["short", "medium", "long"]
_CAPTION_TYPES: tuple[CaptionType, ...] = ("short", "medium", "long")
_GPIC_OFFICIAL_SPLITS = frozenset(("train", "test"))
_COUNTER_KEYS = frozenset(
    (
        "seen",
        "skipped",
        "warnings_emitted",
        "by_reason",
        "by_source",
        "by_shard",
        "by_rank",
    )
)


@dataclass(frozen=True, slots=True)
class GPICPairIndex:
    stem: str
    json_member_name: str | None
    image_member_name: str | None


@dataclass(frozen=True, slots=True)
class GPICCorruptionPolicy:
    warning_limit: int = 20

    def __post_init__(self) -> None:
        if type(self.warning_limit) is not int or self.warning_limit < 0:
            raise ValueError("GPIC corruption policy warning_limit must be non-negative")

    @classmethod
    def from_settings(cls, settings: object | None) -> GPICCorruptionPolicy:
        if settings is None:
            return DEFAULT_GPIC_CORRUPTION_POLICY
        try:
            return cls(warning_limit=getattr(settings, "warning_limit"))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("GPIC corruption policy settings are malformed") from error


DEFAULT_GPIC_CORRUPTION_POLICY = GPICCorruptionPolicy()


@dataclass(slots=True)
class GPICSkipCounters:
    seen: int = 0
    skipped: int = 0
    warnings_emitted: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    by_shard: dict[str, int] = field(default_factory=dict)
    by_rank: dict[str, int] = field(default_factory=dict)

    def note_seen(self) -> None:
        self.seen += 1

    def note_skip(self, *, reason: str, source: str, shard: str, rank: int) -> None:
        self.skipped += 1
        for counters, key in (
            (self.by_reason, reason),
            (self.by_source, source),
            (self.by_shard, shard),
            (self.by_rank, str(rank)),
        ):
            counters[key] = counters.get(key, 0) + 1

    def as_dict(self) -> dict[str, object]:
        return {
            "seen": self.seen,
            "skipped": self.skipped,
            "warnings_emitted": self.warnings_emitted,
            "by_reason": dict(sorted(self.by_reason.items())),
            "by_source": dict(sorted(self.by_source.items())),
            "by_shard": dict(sorted(self.by_shard.items())),
            "by_rank": dict(sorted(self.by_rank.items())),
        }

    @classmethod
    def from_state(cls, state: object) -> GPICSkipCounters:
        if not isinstance(state, Mapping) or set(state) != _COUNTER_KEYS:
            raise ValueError("GPIC skip counters are malformed")
        seen = state.get("seen")
        skipped = state.get("skipped")
        warnings_emitted = state.get("warnings_emitted")
        if (
            type(seen) is not int
            or seen < 0
            or type(skipped) is not int
            or not 0 <= skipped <= seen
            or type(warnings_emitted) is not int
            or not 0 <= warnings_emitted <= skipped
        ):
            raise ValueError("GPIC skip counter totals are malformed")

        mappings: dict[str, dict[str, int]] = {}
        for name in ("by_reason", "by_source", "by_shard", "by_rank"):
            raw = state.get(name)
            if (
                not isinstance(raw, Mapping)
                or any(not isinstance(key, str) or not key for key in raw)
                or any(type(value) is not int or value <= 0 for value in raw.values())
                or sum(raw.values()) != skipped
            ):
                raise ValueError(f"GPIC skip counter {name} is malformed")
            mappings[name] = dict(raw)
        return cls(
            seen=seen,
            skipped=skipped,
            warnings_emitted=warnings_emitted,
            by_reason=mappings["by_reason"],
            by_source=mappings["by_source"],
            by_shard=mappings["by_shard"],
            by_rank=mappings["by_rank"],
        )


def _official_split_from_sample_id(sample_id: str) -> str:
    if not isinstance(sample_id, str):
        raise ValueError("GPIC sample_id must encode its official split")
    components = sample_id.split("/")
    if (
        "\\" in sample_id
        or len(components) < 2
        or any(component in ("", ".", "..") for component in components)
    ):
        raise ValueError("GPIC sample_id must encode its official split as a canonical locator")
    official_split = components[0]
    if official_split not in _GPIC_OFFICIAL_SPLITS:
        raise ValueError("GPIC sample_id official split must be train or test")
    return official_split


@dataclass(frozen=True)
class GPICSample:
    sample_id: str
    image: Tensor
    captions: Mapping[CaptionType, str]
    provenance: SourceProvenance


@dataclass(frozen=True)
class CaptionSelection:
    caption_type: CaptionType
    text: str


class GPICTrainingSource:
    def __init__(
        self,
        *,
        config: MFConfig,
        reader: LazyIndexedReader[GPICSample],
    ) -> None:
        normalized_data_config(config)
        provenance, fingerprint = require_reader_metadata(reader)
        kind, split, root = expected_source(config, "gpic")
        if (
            provenance.source_kind != kind
            or provenance.official_split != split
            or provenance.source_identity != root
        ):
            raise ValueError(
                "reader is not the configured GPIC train source "
                "from the official GPIC train partition"
            )
        self._reader = reader
        self._provenance = provenance
        self._fingerprint = fingerprint

    @classmethod
    def from_config(
        cls,
        *,
        config: MFConfig,
        reader: LazyIndexedReader[GPICSample],
    ) -> GPICTrainingSource:
        return cls(config=config, reader=reader)

    @property
    def provenance(self) -> SourceProvenance:
        return self._provenance

    @property
    def fingerprint(self) -> str | None:
        return self._fingerprint

    def __len__(self) -> int:
        return len(self._reader)

    def __getitem__(self, index: int) -> GPICSample:
        sample = self._reader[index]
        if not isinstance(sample, GPICSample):
            raise TypeError("GPIC reader returned a non-GPICSample record")
        if sample.provenance != self._provenance:
            raise ValueError("GPIC record provenance does not match its reader provenance")
        if _official_split_from_sample_id(sample.sample_id) != self._provenance.official_split:
            raise ValueError("GPIC sample_id does not match its reader official split")
        return sample


def select_caption(
    captions: Mapping[CaptionType, str],
    *,
    config: MFConfig,
    global_sample_index: int,
) -> CaptionSelection:
    normalized_data_config(config)
    if global_sample_index < 0:
        raise ValueError("global_sample_index must be non-negative")
    configured_types = config.data.image_text.caption_sampling.caption_types
    available_types = tuple(
        caption_type for caption_type in configured_types if captions.get(caption_type)
    )
    if not available_types:
        raise ValueError("GPIC sample must provide at least one non-empty caption")

    selected_index = stable_hash(config.run.seed, global_sample_index) % len(available_types)
    caption_type = available_types[selected_index]
    return CaptionSelection(caption_type=caption_type, text=captions[caption_type])
