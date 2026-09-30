from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from mf.config.schema import MFConfig
from mf.data.sources import expected_source, normalized_data_config
from mf.data.readers import (
    LazyIndexedReader,
    SourceProvenance,
    require_reader_metadata,
)
from mf.data.text import (
    TextTokenizer,
    TokenizedTextBlock,
    tokenize_document_window,
    tokenizer_resume_signature,
)


@dataclass(frozen=True)
class UltraFineWebDocument:
    document_id: str
    text: str
    provenance: SourceProvenance


@dataclass(frozen=True, slots=True)
class UltraFineWebWindow:
    document_id: str
    value: TokenizedTextBlock
    provenance: SourceProvenance


@runtime_checkable
class UltraFineWebWindowReader(Protocol):
    def read_window(
        self,
        index: int,
        *,
        cycle: int,
    ) -> UltraFineWebWindow: ...


class UltraFineWebTrainingSource:
    def __init__(
        self,
        *,
        config: MFConfig,
        source_name: str,
        reader: LazyIndexedReader[UltraFineWebDocument],
        tokenizer: TextTokenizer,
    ) -> None:
        normalized_data_config(config)
        if source_name not in ("ultrafineweb_multi_style", "ultrafineweb_qa"):
            raise ValueError(f"not a configured UltraFineWeb source: {source_name!r}")
        provenance, fingerprint = require_reader_metadata(reader)
        kind, split, root = expected_source(config, source_name)
        if (
            provenance.source_kind != kind
            or provenance.official_split != split
            or provenance.source_identity != root
        ):
            raise ValueError("reader is not the configured official UltraFineWeb train source")
        self._reader = reader
        self._provenance = provenance
        self._fingerprint = json.dumps(
            {
                "format": "ultrafineweb-window-source-v2",
                "reader_fingerprint": fingerprint,
                "run_seed": config.run.seed,
                "source_name": source_name,
                "tokenizer": tokenizer_resume_signature(tokenizer),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        self._tokenizer = tokenizer
        self._run_seed = config.run.seed
        self._source_name = source_name

    @classmethod
    def from_config(
        cls,
        *,
        config: MFConfig,
        source_name: str,
        reader: LazyIndexedReader[UltraFineWebDocument],
        tokenizer: TextTokenizer,
    ) -> UltraFineWebTrainingSource:
        return cls(
            config=config,
            source_name=source_name,
            reader=reader,
            tokenizer=tokenizer,
        )

    @property
    def provenance(self) -> SourceProvenance:
        return self._provenance

    @property
    def fingerprint(self) -> str | None:
        return self._fingerprint

    def __len__(self) -> int:
        return len(self._reader)

    def __getitem__(self, index: int) -> UltraFineWebDocument:
        document = self._reader[index]
        if not isinstance(document, UltraFineWebDocument):
            raise TypeError("UltraFineWeb reader returned a non-document record")
        if document.provenance != self._provenance:
            raise ValueError(
                "UltraFineWeb document provenance does not match its reader provenance"
            )
        return document

    def read_window(self, index: int, *, cycle: int) -> UltraFineWebWindow:
        if type(cycle) is not int or cycle < 0:
            raise ValueError("cycle must be a non-negative integer")
        document = self[index]
        value = tokenize_document_window(
            self._tokenizer,
            document.text,
            run_seed=self._run_seed,
            source_id=self._source_name,
            sample_id=document.document_id,
            cycle=cycle,
        )
        return UltraFineWebWindow(
            document_id=document.document_id,
            value=value,
            provenance=self._provenance,
        )
