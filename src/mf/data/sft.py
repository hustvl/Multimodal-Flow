"""Reusable mechanics for checkpointable supervised task streams.

Dataset recipes should provide parsing and validation policies; this module
owns the small pieces that are shared by those policies and the batch runtime.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from mf.config.schema import MFConfig
from mf.contracts.batch import RawTaskBatch
from mf.data.runtime import BatchFetcher, build_batch_fetcher
from mf.data.text import TextTokenizer


RawTaskStreamFactory = Callable[[MFConfig, TextTokenizer, int, int], object]

_SFT_RECIPES: dict[
    str, tuple[RawTaskStreamFactory, RawTaskStreamFactory]
] = {}


def register_sft_recipe(
    name: str,
    *,
    image_to_text_task_stream_factory: RawTaskStreamFactory,
    text_to_image_task_stream_factory: RawTaskStreamFactory,
    replace: bool = False,
) -> None:
    """Register dataset parsing outside the generic SFT runtime.

    A recipe is deliberately just two checkpointable stream factories. The
    core package owns mixing, batching, and resume semantics; a project or
    downstream package owns the dataset field and archive conventions.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("SFT recipe name must be a non-empty string")
    if not callable(image_to_text_task_stream_factory):
        raise TypeError("image-to-text SFT recipe factory must be callable")
    if not callable(text_to_image_task_stream_factory):
        raise TypeError("text-to-image SFT recipe factory must be callable")
    if name in _SFT_RECIPES and not replace:
        raise ValueError(f"SFT recipe is already registered: {name!r}")
    _SFT_RECIPES[name] = (
        image_to_text_task_stream_factory,
        text_to_image_task_stream_factory,
    )


def resolve_sft_recipe(
    name: str,
) -> tuple[RawTaskStreamFactory, RawTaskStreamFactory]:
    """Resolve a recipe registered by the CLI or a downstream package."""

    try:
        return _SFT_RECIPES[name]
    except KeyError as error:
        known = ", ".join(sorted(_SFT_RECIPES)) or "<none>"
        raise ValueError(
            f"unknown SFT recipe {name!r}; import a recipe plugin first "
            f"(registered: {known})"
        ) from error


def eos_fill_block_size(config: MFConfig) -> int | None:
    """Return the EOS-aligned caption block size selected by the config."""
    chunk_pack = config.tasks.chunk_pack
    text_packing = None if chunk_pack is None else chunk_pack.text_packing
    if text_packing is not None:
        return text_packing.block_size if text_packing.mode == "block_aligned_eos" else None
    block_causal = config.flow.text_block_causal
    return block_causal.block_size if block_causal.packing == "block_aligned_eos" else None


def validate_stream_position(stream_rank: int, stream_world_size: int) -> None:
    if type(stream_rank) is not int or type(stream_world_size) is not int:
        raise TypeError("stream rank and world size must be integers")
    if not 0 <= stream_rank < stream_world_size:
        raise ValueError("stream rank must be in [0, stream_world_size)")


def manifest_signature(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class SkipTracker:
    """Bounded warning and deterministic counters for invalid records."""

    warnings_emitted: int = 0
    counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = {}

    def note(self, source: str, reason: str) -> None:
        key = f"{source}:{reason}"
        assert self.counts is not None
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.warnings_emitted < 20:
            warnings.warn(
                f"skipping SFT sample source={source} reason={reason}",
                RuntimeWarning,
                stacklevel=3,
            )
            self.warnings_emitted += 1


def build_sft_batch_fetcher(
    *,
    config: MFConfig,
    rank: int,
    tokenizer: TextTokenizer,
    prepare_batch: Callable[[RawTaskBatch], RawTaskBatch] | None = None,
    recipe: str = "public",
    image_to_text_task_stream_factory: RawTaskStreamFactory | None = None,
    text_to_image_task_stream_factory: RawTaskStreamFactory | None = None,
) -> BatchFetcher:
    """Connect recipe-provided task streams to the generic mixed-batch runtime."""
    if not config.sft.enabled:
        raise ValueError("build_sft_batch_fetcher requires sft.enabled=true")
    if image_to_text_task_stream_factory is None or text_to_image_task_stream_factory is None:
        image_to_text_task_stream_factory, text_to_image_task_stream_factory = (
            resolve_sft_recipe(recipe)
        )
    return build_batch_fetcher(
        config=config,
        rank=rank,
        tokenizer=tokenizer,
        prepare_batch=prepare_batch,
        image_to_text_task_stream_factory=image_to_text_task_stream_factory,
        text_to_image_task_stream_factory=text_to_image_task_stream_factory,
    )


__all__ = [
    "RawTaskStreamFactory",
    "SkipTracker",
    "build_sft_batch_fetcher",
    "eos_fill_block_size",
    "manifest_signature",
    "register_sft_recipe",
    "resolve_sft_recipe",
    "validate_stream_position",
]
