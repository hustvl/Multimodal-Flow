from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Immutable partition identity issued by a format-specific indexed reader."""

    source_kind: str
    official_split: str
    source_identity: str
    external_fingerprint: str

    def __post_init__(self) -> None:
        for name, value in (
            ("source_kind", self.source_kind),
            ("official_split", self.official_split),
            ("source_identity", self.source_identity),
            ("external_fingerprint", self.external_fingerprint),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")


class LazyIndexedReader(Protocol[T_co]):
    """Disk-format boundary: metadata plus constant-time indexed record access."""

    provenance: SourceProvenance
    fingerprint: str | None

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> T_co: ...


@dataclass(frozen=True, slots=True)
class ReaderBlock:
    document_id: str
    block_index: int
    is_final: bool
    value: object
    provenance: SourceProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id:
            raise ValueError("document_id must be a non-empty string")
        if type(self.block_index) is not int or self.block_index < 0:
            raise ValueError("block_index must be a non-negative integer")
        if type(self.is_final) is not bool:
            raise ValueError("is_final must be a bool")
        if not isinstance(self.provenance, SourceProvenance):
            raise ValueError("block provenance must be SourceProvenance")


@runtime_checkable
class BlockIndexedReader(Protocol):
    def read_block(self, index: int, block_index: int) -> ReaderBlock: ...


def require_reader_metadata(reader: object) -> tuple[SourceProvenance, str | None]:
    provenance = getattr(reader, "provenance", None)
    fingerprint = getattr(reader, "fingerprint", None)
    if not isinstance(provenance, SourceProvenance):
        raise TypeError("indexed reader must expose SourceProvenance")
    if fingerprint is not None and (
        not isinstance(fingerprint, str) or not fingerprint
    ):
        raise ValueError("reader fingerprint must be None or a non-empty string")
    return provenance, fingerprint
