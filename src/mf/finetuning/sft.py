"""Compatibility entry point for the generic supervised fine-tuning runtime."""

from __future__ import annotations

from collections.abc import Callable

from mf.config.schema import MFConfig
from mf.contracts.batch import RawTaskBatch
from mf.data.runtime import BatchFetcher
from mf.data.sft import build_sft_batch_fetcher as _build_sft_batch_fetcher
from mf.data.text import TextTokenizer


def build_sft_batch_fetcher(
    *,
    config: MFConfig,
    rank: int,
    tokenizer: TextTokenizer,
    prepare_batch: Callable[[RawTaskBatch], RawTaskBatch] | None = None,
) -> BatchFetcher:
    return _build_sft_batch_fetcher(
        config=config,
        rank=rank,
        tokenizer=tokenizer,
        prepare_batch=prepare_batch,
    )


__all__ = ["build_sft_batch_fetcher"]
