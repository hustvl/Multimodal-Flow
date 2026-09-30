from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_DIRECT_WRITE_ROOTS: tuple[Path, ...] = ()


def uses_direct_writes(path: str | Path) -> bool:
    """Return whether ``path`` is on storage that cannot publish through rename."""

    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    return any(
        absolute == root or root in absolute.parents for root in _DIRECT_WRITE_ROOTS
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _flush_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _copy_file(source: Path, destination: Path) -> None:
    with source.open("rb") as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


@contextmanager
def durable_artifact_path(path: str | Path) -> Iterator[Path]:
    """Stage an artifact, then publish it with the filesystem's supported semantics.

    Local filesystems use a sibling temporary file and atomic replacement.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    direct = uses_direct_writes(destination)
    temporary_parent = None if direct else destination.parent
    descriptor, temporary_name = tempfile.mkstemp(
        dir=temporary_parent,
        prefix=f".{destination.stem}.",
        suffix=f".tmp{destination.suffix}",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        _flush_file(temporary)
        if direct:
            _copy_file(temporary, destination)
        else:
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def durable_write_artifact(path: str | Path, writer: Callable[[Path], None]) -> Path:
    """Publish a path-based writer without requiring atomic rename."""

    if not callable(writer):
        raise TypeError("writer must be callable")
    destination = Path(path)
    with durable_artifact_path(destination) as temporary:
        writer(temporary)
    return destination


def durable_write_stream(
    path: str | Path,
    writer: Callable[[Any], None],
) -> Path:
    """Publish a binary stream writer without staging large payloads locally."""

    if not callable(writer):
        raise TypeError("writer must be callable")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if uses_direct_writes(destination):
        with destination.open("wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        return destination
    with durable_artifact_path(destination) as temporary:
        with temporary.open("wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
    return destination


def durable_write_bytes(path: str | Path, payload: bytes) -> Path:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    destination = Path(path)
    if uses_direct_writes(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return destination
    return durable_write_artifact(
        destination, lambda temporary: temporary.write_bytes(payload)
    )


def durable_write_text(path: str | Path, payload: str) -> Path:
    if not isinstance(payload, str):
        raise TypeError("payload must be a string")
    return durable_write_bytes(path, payload.encode("utf-8"))


def durable_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    ensure_ascii: bool = False,
) -> Path:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=ensure_ascii,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return durable_write_bytes(path, encoded)
