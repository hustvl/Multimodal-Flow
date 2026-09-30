from __future__ import annotations

import hashlib
import io
import json
import random
import struct
import tarfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from mf.data.images import preprocess_image
from mf.data.gpic import (
    DEFAULT_GPIC_CORRUPTION_POLICY,
    GPICCorruptionPolicy,
    GPICPairIndex,
    GPICSample,
    GPICSkipCounters,
)
from mf.data.planner import stable_hash
from mf.data.readers import SourceProvenance
from mf.data.text import (
    DEFAULT_TEXT_TOKENS,
    TextTokenizer,
    TokenizedTextBlock,
    make_hard_packed_block,
    make_padded_packed_block,
    tokenize_block_aligned_document_ids,
    tokenize_block_aligned_document_units,
    tokenize_block_aligned_record_unit,
    tokenize_document_ids,
    tokenizer_resume_signature,
)

_GPIC_CAPTION_TYPES = frozenset(("short", "medium", "long"))
_GPIC_IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".png", ".webp"))

_RECOVERABLE_IMAGE_ERRORS = (
    EOFError,
    IndexError,
    OSError,
    OverflowError,
    SyntaxError,
    ValueError,
    struct.error,
    Image.DecompressionBombError,
)


class CorruptImageSampleError(Exception):
    """A recoverable image payload error that must not terminate training."""


class CorruptGPICShardError(Exception):
    """A recoverable structural tar error."""


class GPICCorruptionError(RuntimeError):
    """GPIC corruption exceeded the configured fail-closed threshold."""


def _paths_digest(paths: tuple[Path, ...], *, root: Path) -> str:
    snapshot: list[dict[str, int | str]] = []
    for path in paths:
        stat = path.stat()
        snapshot.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "st_size": stat.st_size,
                "st_mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_partition(rank: int, world_size: int) -> None:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")


def _cycle_partition(
    paths: tuple[Path, ...],
    *,
    seed: int,
    cycle: int,
    rank: int,
    world_size: int,
    label: str,
) -> tuple[Path, ...]:
    order = list(paths)
    random.Random(stable_hash(seed, cycle, label)).shuffle(order)
    local = tuple(order[rank::world_size])
    if not local:
        raise RuntimeError(f"{label} has fewer shards than distributed ranks")
    return local


def _prepare_image(
    payload: bytes,
    *,
    image_resolution: int = 224,
    preprocessing: str = "legacy_center_crop_bicubic_v1",
) -> torch.Tensor:
    if type(image_resolution) is not int or image_resolution <= 0:
        raise ValueError("image_resolution must be a positive integer")
    if preprocessing not in {
        "legacy_center_crop_bicubic_v1",
        "siglip2_resize_bicubic_v1",
    }:
        raise ValueError(f"unsupported image preprocessing policy: {preprocessing!r}")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            return preprocess_image(
                source, resolution=image_resolution, policy=preprocessing
            )
    except _RECOVERABLE_IMAGE_ERRORS as error:
        raise CorruptImageSampleError(f"{type(error).__name__}: {error}") from error


class GPICWebTarStream:
    """Semantic GPIC tar reader with recoverable skips and exact cursor state."""

    _STATE_KEYS = frozenset(
        (
            "version",
            "signature",
            "shard_snapshot_digest",
            "cycle",
            "shard_position",
            "semantic_pair_index",
            "counters",
        )
    )

    def __init__(
        self,
        *,
        root: str | Path,
        split: str,
        rank: int,
        world_size: int,
        seed: int,
        corruption_policy: GPICCorruptionPolicy = DEFAULT_GPIC_CORRUPTION_POLICY,
        min_age_minutes: float = 0.0,
        image_resolution: int = 224,
        image_preprocessing: str = "legacy_center_crop_bicubic_v1",
    ) -> None:
        _validate_partition(rank, world_size)
        if split not in ("train", "test"):
            raise ValueError("GPIC split must be train or test")
        if not isinstance(corruption_policy, GPICCorruptionPolicy):
            raise TypeError("corruption_policy must be GPICCorruptionPolicy")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.corruption_policy = corruption_policy
        if type(image_resolution) is not int or image_resolution <= 0:
            raise ValueError("image_resolution must be a positive integer")
        self.image_resolution = image_resolution
        if image_preprocessing not in {
            "legacy_center_crop_bicubic_v1",
            "siglip2_resize_bicubic_v1",
        }:
            raise ValueError(f"unsupported image preprocessing policy: {image_preprocessing!r}")
        self.image_preprocessing = image_preprocessing
        if isinstance(min_age_minutes, bool) or not isinstance(min_age_minutes, (int, float)):
            raise TypeError("GPIC min_age_minutes must be numeric")
        if float(min_age_minutes) < 0.0:
            raise ValueError("GPIC min_age_minutes must be non-negative")
        self.min_age_minutes = float(min_age_minutes)
        min_age_seconds = self.min_age_minutes * 60.0
        now = time.time()
        paths: list[Path] = []
        for path in sorted((self.root / split).glob("*.tar")):
            try:
                stat = path.stat()
            except OSError:
                continue
            if not path.is_file() or stat.st_size <= 0:
                continue
            if min_age_seconds > 0.0 and now - stat.st_mtime < min_age_seconds:
                continue
            paths.append(path)
        self._paths = tuple(paths)
        if not self._paths:
            raise FileNotFoundError(
                f"no ready GPIC tar shards found under {self.root / split}; "
                f"min_age_minutes={self.min_age_minutes}"
            )
        self._shard_snapshot_digest = _paths_digest(self._paths, root=self.root)
        signature_payload = {
            "format": "gpic-webtar-v2",
            "root": str(self.root),
            "split": split,
            "rank": rank,
            "world_size": world_size,
            "seed": seed,
            "min_age_minutes": self.min_age_minutes,
            "shards": self._shard_snapshot_digest,
            "corruption_policy": {
                "warning_limit": corruption_policy.warning_limit,
            },
            "image_preprocessing": image_preprocessing,
        }
        if image_resolution != 224:
            signature_payload["image_resolution"] = image_resolution
        self._signature = json.dumps(
            signature_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.provenance = SourceProvenance(
            source_kind="gpic",
            official_split=split,
            source_identity=str(self.root),
            external_fingerprint=self._shard_snapshot_digest,
        )
        self._cycle = 0
        self._shard_position = 0
        self._semantic_pair_index = 0
        self._counters = GPICSkipCounters()
        self._archive: tarfile.TarFile | None = None
        self._archive_path: Path | None = None
        self._pairs: tuple[GPICPairIndex, ...] = ()
        self._pair_issues: tuple[str | None, ...] = ()
        self._members: dict[str, tarfile.TarInfo] = {}
        self._warm_archive: tarfile.TarFile | None = None
        self._warm_archive_path: Path | None = None
        self._warm_members: list[tarfile.TarInfo] = []
        self._warm_pairs: tuple[GPICPairIndex, ...] | None = None
        self._warm_pair_issues: tuple[str | None, ...] = ()
        self._warm_member_lookup: dict[str, tarfile.TarInfo] = {}
        self._warm_error: str | None = None

    def _paths_for_cycle(self, cycle: int) -> tuple[Path, ...]:
        return _cycle_partition(
            self._paths,
            seed=self.seed,
            cycle=cycle,
            rank=self.rank,
            world_size=self.world_size,
            label="gpic",
        )

    def _local_paths(self) -> tuple[Path, ...]:
        return self._paths_for_cycle(self._cycle)

    def _next_shard_path(self) -> Path:
        local_paths = self._local_paths()
        next_position = self._shard_position + 1
        if next_position < len(local_paths):
            return local_paths[next_position]
        return self._paths_for_cycle(self._cycle + 1)[0]

    def _close_current_archive(self) -> None:
        if self._archive is not None:
            self._archive.close()
        self._archive = None
        self._archive_path = None
        self._pairs = ()
        self._pair_issues = ()
        self._members = {}

    def _reset_warm_archive(self) -> None:
        if self._warm_archive is not None:
            self._warm_archive.close()
        self._warm_archive = None
        self._warm_archive_path = None
        self._warm_members = []
        self._warm_pairs = None
        self._warm_pair_issues = ()
        self._warm_member_lookup = {}
        self._warm_error = None

    def _close_archive(self) -> None:
        self._close_current_archive()
        self._reset_warm_archive()

    def _advance_shard(self) -> None:
        self._close_current_archive()
        self._shard_position += 1
        self._semantic_pair_index = 0
        if self._shard_position >= len(self._local_paths()):
            self._cycle += 1
            self._shard_position = 0
        if self._warm_archive_path != self._local_paths()[self._shard_position]:
            self._reset_warm_archive()

    @staticmethod
    def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
        file_object = archive.extractfile(member)
        if file_object is None:
            raise ValueError(f"unable to extract tar member {member.name!r}")
        return file_object.read()

    @staticmethod
    def _semantic_index(
        members: list[tarfile.TarInfo],
    ) -> tuple[
        tuple[GPICPairIndex, ...],
        tuple[str | None, ...],
        dict[str, tarfile.TarInfo],
    ]:
        json_members: dict[str, list[tarfile.TarInfo]] = {}
        image_members: dict[str, list[tarfile.TarInfo]] = {}
        member_lookup: dict[str, tarfile.TarInfo] = {}
        for member in members:
            if not member.isfile():
                continue
            suffix = PurePosixPath(member.name).suffix.lower()
            if suffix != ".json" and suffix not in _GPIC_IMAGE_SUFFIXES:
                continue
            stem = str(PurePosixPath(member.name).with_suffix(""))
            target = json_members if suffix == ".json" else image_members
            target.setdefault(stem, []).append(member)
            member_lookup[member.name] = member

        pairs: list[GPICPairIndex] = []
        issues: list[str | None] = []
        for stem in sorted(json_members.keys() | image_members.keys()):
            metadata = json_members.get(stem, [])
            images = image_members.get(stem, [])
            issue = None
            if len(metadata) > 1 or len(images) > 1:
                issue = "duplicate_stem"
            elif not metadata:
                issue = "missing_json"
            elif not images:
                issue = "missing_image"
            pairs.append(
                GPICPairIndex(
                    stem=stem,
                    json_member_name=metadata[0].name if metadata else None,
                    image_member_name=images[0].name if images else None,
                )
            )
            issues.append(issue)
        return tuple(pairs), tuple(issues), member_lookup

    def _continue_warm_archive(self, *, max_members: int) -> bool:
        if self._warm_archive is None:
            return self._warm_pairs is not None or self._warm_error is not None
        for _ in range(max_members):
            member = self._warm_archive.next()
            if member is None:
                pairs, issues, members = self._semantic_index(self._warm_members)
                self._warm_pairs = pairs
                self._warm_pair_issues = issues
                self._warm_member_lookup = members
                return True
            self._warm_members.append(member)
        return False

    def warm_next_shard(self, *, max_members: int = 512) -> bool:
        if type(max_members) is not int or max_members <= 0:
            raise ValueError("max_members must be a positive integer")
        if self._archive is None:
            return False
        path = self._next_shard_path()
        if self._warm_archive_path != path:
            self._reset_warm_archive()
            self._warm_archive_path = path
            try:
                self._warm_archive = tarfile.open(path, mode="r:*")
            except (EOFError, OSError, tarfile.TarError, ValueError) as error:
                self._warm_error = f"{type(error).__name__}: {error}"
                return True
        if self._warm_pairs is not None or self._warm_error is not None:
            return True
        try:
            return self._continue_warm_archive(max_members=max_members)
        except (EOFError, OSError, tarfile.TarError, ValueError) as error:
            if self._warm_archive is not None:
                self._warm_archive.close()
            self._warm_archive = None
            self._warm_members = []
            self._warm_error = f"{type(error).__name__}: {error}"
            return True

    def _activate_warm_archive(self, path: Path) -> bool:
        if (
            self._warm_archive_path != path
            or self._warm_archive is None
            or self._warm_pairs is None
        ):
            return False
        self._archive = self._warm_archive
        self._archive_path = path
        self._pairs = self._warm_pairs
        self._pair_issues = self._warm_pair_issues
        self._members = self._warm_member_lookup
        self._warm_archive = None
        self._warm_archive_path = None
        self._warm_members = []
        self._warm_pairs = None
        self._warm_pair_issues = ()
        self._warm_member_lookup = {}
        self._warm_error = None
        return True

    def _ensure_archive(self) -> None:
        path = self._local_paths()[self._shard_position]
        if self._archive is not None and self._archive_path == path:
            return
        self._close_current_archive()
        if self._warm_archive_path == path:
            if self._warm_error is None:
                try:
                    while not self._continue_warm_archive(max_members=4096):
                        pass
                except (EOFError, OSError, tarfile.TarError, ValueError) as error:
                    self._warm_error = f"{type(error).__name__}: {error}"
            if self._warm_error is not None:
                detail = self._warm_error
                self._reset_warm_archive()
                raise CorruptGPICShardError(detail)
            if not self._activate_warm_archive(path):
                raise RuntimeError("completed GPIC shard warmup could not be activated")
        else:
            self._reset_warm_archive()
            try:
                archive = tarfile.open(path, mode="r:*")
                pairs, issues, members = self._semantic_index(archive.getmembers())
            except (EOFError, OSError, tarfile.TarError, ValueError) as error:
                if "archive" in locals():
                    archive.close()
                raise CorruptGPICShardError(f"{type(error).__name__}: {error}") from error
            self._archive = archive
            self._archive_path = path
            self._pairs = pairs
            self._pair_issues = issues
            self._members = members
        if self._semantic_pair_index > len(self._pairs):
            raise ValueError("GPIC semantic checkpoint cursor exceeds its shard")

    def _record_skip(self, *, reason: str, stem: str, detail: str) -> None:
        path = self._local_paths()[self._shard_position]
        self._counters.note_skip(
            reason=reason,
            source="gpic",
            shard=path.name,
            rank=self.rank,
        )
        if self._counters.warnings_emitted < self.corruption_policy.warning_limit:
            warnings.warn(
                (
                    f"skipping corrupt GPIC sample {self.split}/{stem} reason={reason} "
                    f"detail={detail} in shard {path}; rank={self.rank} cycle={self._cycle} "
                    f"shard_position={self._shard_position} "
                    f"semantic_pair_index={self._semantic_pair_index}"
                ),
                RuntimeWarning,
                stacklevel=3,
            )
            self._counters.warnings_emitted += 1

        # Recoverable sample/shard corruption is skipped and counted, never
        # promoted into a distributed training abort.
        # next_sample still fails if a complete rank-local cycle has no valid sample.

    def _skip_pair(self, pair: GPICPairIndex, reason: str, detail: str) -> None:
        self._record_skip(reason=reason, stem=pair.stem, detail=detail)

    def next_sample(self) -> GPICSample:
        scanning_full_cycle = self._shard_position == 0 and self._semantic_pair_index == 0
        structurally_corrupt_shards = 0
        while True:
            try:
                self._ensure_archive()
            except CorruptGPICShardError as error:
                self._counters.note_seen()
                self._record_skip(
                    reason="shard_error",
                    stem="<shard>",
                    detail=f"structurally corrupt GPIC shard: {error}",
                )
                previous_cycle = self._cycle
                self._advance_shard()
                structurally_corrupt_shards += 1
                if self._cycle != previous_cycle:
                    if scanning_full_cycle:
                        if structurally_corrupt_shards == len(self._local_paths()):
                            raise RuntimeError(
                                "all rank-local GPIC shards are structurally corrupt"
                            ) from error
                        raise RuntimeError(
                            "all rank-local GPIC shards contain no valid samples"
                        ) from error
                    scanning_full_cycle = True
                    structurally_corrupt_shards = 0
                continue

            if self._semantic_pair_index >= len(self._pairs):
                previous_cycle = self._cycle
                self._advance_shard()
                if self._cycle != previous_cycle:
                    if scanning_full_cycle:
                        raise RuntimeError("all rank-local GPIC shards contain no valid samples")
                    scanning_full_cycle = True
                    structurally_corrupt_shards = 0
                continue

            pair_index = self._semantic_pair_index
            pair = self._pairs[pair_index]
            issue = self._pair_issues[pair_index]
            self._semantic_pair_index += 1
            self._counters.note_seen()
            if issue is not None:
                self._skip_pair(pair, issue, "semantic pair is incomplete or duplicated")
                continue
            if pair.json_member_name is None or pair.image_member_name is None:
                raise RuntimeError("complete GPIC pair index lost a member")
            if self._archive is None:
                raise RuntimeError("GPIC archive is not open")

            try:
                metadata_payload = self._read_member(
                    self._archive,
                    self._members[pair.json_member_name],
                )
                metadata = json.loads(metadata_payload)
            except (UnicodeError, json.JSONDecodeError) as error:
                self._skip_pair(pair, "invalid_json", f"{type(error).__name__}: {error}")
                continue
            except (EOFError, OSError, tarfile.TarError, ValueError) as error:
                self._skip_pair(pair, "member_read", f"{type(error).__name__}: {error}")
                continue
            if not isinstance(metadata, Mapping):
                self._skip_pair(pair, "invalid_metadata", "JSON metadata is not an object")
                continue

            key = metadata.get("key")
            if not isinstance(key, str) or not key or PurePosixPath(pair.stem).name != key:
                self._skip_pair(pair, "invalid_metadata", "metadata key does not match stem")
                continue
            caption_type = metadata.get("caption_type")
            caption = metadata.get("caption")
            if (
                caption_type not in _GPIC_CAPTION_TYPES
                or not isinstance(caption, str)
                or not caption
            ):
                # GPIC also contains auxiliary caption types such as "tag". These
                # are ordinary filtering cases, not dataset corruption.
                continue

            try:
                image_payload = self._read_member(
                    self._archive,
                    self._members[pair.image_member_name],
                )
                image = _prepare_image(
                    image_payload,
                    image_resolution=self.image_resolution,
                    preprocessing=self.image_preprocessing,
                )
            except CorruptImageSampleError as error:
                self._skip_pair(pair, "image_decode", str(error))
                continue
            except (EOFError, OSError, tarfile.TarError, ValueError) as error:
                self._skip_pair(pair, "member_read", f"{type(error).__name__}: {error}")
                continue
            return GPICSample(
                sample_id=f"{self.split}/{key}",
                image=image,
                captions={caption_type: caption},
                provenance=self.provenance,
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 2,
            "signature": self._signature,
            "shard_snapshot_digest": self._shard_snapshot_digest,
            "cycle": self._cycle,
            "shard_position": self._shard_position,
            "semantic_pair_index": self._semantic_pair_index,
            "counters": self._counters.as_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        version = state.get("version")
        if version != 2:
            raise ValueError(
                "GPIC resume requires stream state version 2; "
                "earlier checkpoints are unsupported"
            )
        if set(state) != self._STATE_KEYS:
            raise ValueError("GPIC stream state fields or version are malformed")
        if state.get("shard_snapshot_digest") != self._shard_snapshot_digest:
            raise ValueError(
                "GPIC shard snapshot digest mismatch; restart from a compatible checkpoint"
            )
        if state.get("signature") != self._signature:
            raise ValueError("GPIC stream state does not match this dataset and policy")
        values = tuple(
            state.get(name) for name in ("cycle", "shard_position", "semantic_pair_index")
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("GPIC semantic cursor values must be non-negative integers")
        cycle, shard_position, semantic_pair_index = values
        counters = GPICSkipCounters.from_state(state.get("counters"))
        same_shard = (
            int(cycle) == self._cycle
            and int(shard_position) == self._shard_position
        )

        self._cycle = int(cycle)
        self._shard_position = int(shard_position)
        self._semantic_pair_index = int(semantic_pair_index)
        if self._shard_position >= len(self._local_paths()):
            raise ValueError("GPIC stream shard cursor is out of range")
        self._counters = counters
        if not same_shard:
            self._close_archive()


class UltraFineWebParquetStream:
    """Row-group UltraFineWeb stream with hard packing and exact state."""

    _HARD_PACK_STATE_KEYS = frozenset(
        (
            "version",
            "signature",
            "cycle",
            "shard_position",
            "row_group",
            "row_index",
            "run_seed",
            "text_tokens",
            "max_chars",
            "token_buffer",
            "tokenizer_signature",
        )
    )
    _BLOCK_ALIGNED_STATE_KEYS = _HARD_PACK_STATE_KEYS | {"packing_mode", "block_size"}

    def __init__(
        self,
        *,
        root: str | Path,
        source_name: str,
        rank: int,
        world_size: int,
        seed: int,
        text_tokens: int = DEFAULT_TEXT_TOKENS,
        max_chars: int = 8192,
        packing_mode: str = "continuous",
        block_size: int = 8,
        block_aligned_record_policy: str = "whole_record_defer_pad",
    ) -> None:
        _validate_partition(rank, world_size)
        if type(text_tokens) is not int or text_tokens <= 0:
            raise ValueError("text_tokens must be a positive integer")
        if type(max_chars) is not int or max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        self.root = Path(root).expanduser().resolve()
        if packing_mode not in {"continuous", "block_aligned_eos"}:
            raise ValueError("packing_mode must be continuous or block_aligned_eos")
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if block_aligned_record_policy not in {
            "whole_record_defer_pad",
            "record_chunk_defer_pad",
            "legacy_sentence_units_hard_pack",
            "record_units_hard_pack",
        }:
            raise ValueError("block_aligned_record_policy must select a supported packing policy")
        if text_tokens % block_size != 0:
            raise ValueError("text_tokens must be divisible by block_size")
        self.source_name = source_name
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.text_tokens = text_tokens
        self.max_chars = max_chars
        self.packing_mode = packing_mode
        self.block_size = block_size
        self.block_aligned_record_policy = block_aligned_record_policy
        paths: list[Path] = []
        for path in sorted(self.root.glob("*.parquet")):
            try:
                stat = path.stat()
            except OSError:
                continue
            if path.is_file() and stat.st_size > 0:
                paths.append(path)
        self._paths = tuple(paths)
        if not self._paths:
            raise FileNotFoundError(f"no UltraFineWeb parquet shards found under {self.root}")
        shards_digest = _paths_digest(self._paths, root=self.root)
        signature_payload: dict[str, int | str] = {
            "format": "ultrafineweb-parquet-hard-pack-v4",
            "root": str(self.root),
            "source_name": source_name,
            "rank": rank,
            "world_size": world_size,
            "seed": seed,
            "text_tokens": text_tokens,
            "max_chars": max_chars,
            "append_eos": "complete_documents_only",
            "pack_strategy": "hard",
            "shards": shards_digest,
        }
        if (
            packing_mode == "block_aligned_eos"
            and block_aligned_record_policy == "legacy_sentence_units_hard_pack"
        ):
            signature_payload.update(
                format="ultrafineweb-parquet-sentence-unit-eos-v7",
                append_eos="sentence_or_paragraph_unit_end_fill_terminal_block",
                content_boundary="complete_sentence_or_paragraph_unit",
                pack_boundary="hard_pack_with_eos_fill",
                oversized_content="truncate_unit_and_preserve_terminal_eos",
                cycle_boundary="clear_partial_hard_pack",
                pack_strategy=block_aligned_record_policy,
                block_size=block_size,
            )
        elif (
            packing_mode == "block_aligned_eos"
            and block_aligned_record_policy == "record_units_hard_pack"
        ):
            signature_payload.update(
                format="ultrafineweb-parquet-record-unit-eos-v9",
                append_eos="source_record_end_if_capacity_then_fill_terminal_block",
                content_boundary="parquet_content_record",
                pack_boundary="hard_pack_with_eos_fill",
                capacity_exhausted_content="preserve_prefix_without_terminal_eos",
                cycle_boundary="clear_partial_hard_pack",
                pack_strategy=block_aligned_record_policy,
                block_size=block_size,
            )
        elif (
            packing_mode == "block_aligned_eos"
            and block_aligned_record_policy == "record_chunk_defer_pad"
        ):
            signature_payload.update(
                format="ultrafineweb-parquet-record-chunk-eos-v1",
                append_eos="source_record_end_if_capacity_then_fill_terminal_block",
                content_boundary="parquet_content_record",
                pack_boundary="one_source_record_per_logical_chunk",
                capacity_exhausted_content="preserve_prefix_without_terminal_eos",
                cycle_boundary="independent_record_chunks",
                pack_strategy=block_aligned_record_policy,
                block_size=block_size,
            )
        elif packing_mode == "block_aligned_eos":
            signature_payload.update(
                format="ultrafineweb-parquet-content-aligned-eos-v7",
                append_eos="content_end_fill_terminal_block",
                content_boundary="parquet_content_record",
                pack_boundary="defer_complete_content_and_pad_inactive_tail",
                oversized_content="truncate_once_and_discard_overflow",
                cycle_boundary="preserve_complete_content_records",
                pack_strategy=packing_mode,
                block_size=block_size,
            )
        self._signature = json.dumps(
            signature_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._cycle = 0
        self._shard_position = 0
        self._row_group = 0
        self._row_index = 0
        self._tokenizer_signature: dict[str, int | str] | None = None
        self._token_buffer: list[int] = []
        self._parquet: pq.ParquetFile | None = None
        self._table: pa.Table | None = None
        self._table_row_group: int | None = None

    def _local_paths(self) -> tuple[Path, ...]:
        return _cycle_partition(
            self._paths,
            seed=self.seed,
            cycle=self._cycle,
            rank=self.rank,
            world_size=self.world_size,
            label=self.source_name,
        )

    def _reset_cache(self) -> None:
        self._parquet = None
        self._table = None
        self._table_row_group = None

    def _advance_shard(self) -> None:
        self._shard_position += 1
        self._row_group = 0
        self._row_index = 0
        self._reset_cache()
        if self._shard_position >= len(self._local_paths()):
            self._cycle += 1
            self._shard_position = 0

    def _current_row(self) -> tuple[str, str]:
        scanning_full_cycle = (
            self._shard_position == 0 and self._row_group == 0 and self._row_index == 0
        )
        while True:
            path = self._local_paths()[self._shard_position]
            try:
                if self._parquet is None:
                    self._parquet = pq.ParquetFile(path)
            except (OSError, ValueError, pa.ArrowException) as error:
                previous_cycle = self._cycle
                warnings.warn(
                    f"skipping unreadable UltraFineWeb shard {path}: "
                    f"{type(error).__name__}: {error}",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._advance_shard()
                if self._cycle != previous_cycle:
                    if scanning_full_cycle:
                        raise RuntimeError(
                            "all rank-local UltraFineWeb parquet shards are unreadable"
                        ) from error
                    scanning_full_cycle = True
                continue
            if self._row_group >= self._parquet.num_row_groups:
                previous_cycle = self._cycle
                self._advance_shard()
                if self._cycle != previous_cycle:
                    if scanning_full_cycle:
                        raise RuntimeError(
                            "all rank-local UltraFineWeb parquet shards contain no rows"
                        )
                    scanning_full_cycle = True
                continue
            if self._table is None or self._table_row_group != self._row_group:
                try:
                    available_columns = set(self._parquet.schema_arrow.names)
                    if "content" not in available_columns:
                        raise ValueError(
                            f"UltraFineWeb shard {path} is missing required content column"
                        )
                    columns = (
                        ("uid", "content")
                        if "uid" in available_columns
                        else ("content",)
                    )
                    self._table = self._parquet.read_row_group(
                        self._row_group,
                        columns=columns,
                    )
                except (OSError, ValueError, pa.ArrowException) as error:
                    previous_cycle = self._cycle
                    warnings.warn(
                        f"skipping unreadable UltraFineWeb shard {path}: "
                        f"{type(error).__name__}: {error}",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    self._advance_shard()
                    if self._cycle != previous_cycle:
                        if scanning_full_cycle:
                            raise RuntimeError(
                                "all rank-local UltraFineWeb parquet shards are unreadable"
                            ) from error
                        scanning_full_cycle = True
                    continue
                self._table_row_group = self._row_group
            if self._row_index >= self._table.num_rows:
                self._row_group += 1
                self._row_index = 0
                self._table = None
                continue
            uid = (
                self._table.column("uid")[self._row_index].as_py()
                if "uid" in self._table.column_names
                else None
            )
            content = self._table.column("content")[self._row_index].as_py()
            row_index = self._row_index
            self._row_index += 1
            if content is None:
                continue
            content = str(content)
            if not content:
                continue
            if not isinstance(uid, str) or not uid:
                uid = f"{path.name}:{self._row_group}:{row_index}"
            return uid, content

    def next_block(self, tokenizer: TextTokenizer) -> TokenizedTextBlock:
        signature = tokenizer_resume_signature(tokenizer)
        if self._tokenizer_signature is None:
            self._tokenizer_signature = signature
        elif signature != self._tokenizer_signature:
            raise ValueError("UltraFineWeb tokenizer does not match checkpoint state")

        if self.packing_mode == "continuous":
            while len(self._token_buffer) < self.text_tokens:
                previous_cycle = self._cycle
                _, content = self._current_row()
                if self._cycle != previous_cycle:
                    self._token_buffer.clear()
                self._token_buffer.extend(
                    tokenize_document_ids(tokenizer, content, max_chars=self.max_chars)
                )
            token_ids = self._token_buffer[: self.text_tokens]
            del self._token_buffer[: self.text_tokens]
            return make_hard_packed_block(token_ids, text_tokens=self.text_tokens)

        if self.block_aligned_record_policy == "record_chunk_defer_pad":
            while True:
                _, content = self._current_row()
                record_ids = tokenize_block_aligned_record_unit(
                    tokenizer,
                    content,
                    text_tokens=self.text_tokens,
                    block_size=self.block_size,
                )
                if record_ids:
                    return make_padded_packed_block(
                        record_ids,
                        pad_token_id=int(signature["pad_token_id"]),
                        text_tokens=self.text_tokens,
                    )

        if self.block_aligned_record_policy in {
            "legacy_sentence_units_hard_pack",
            "record_units_hard_pack",
        }:
            eos_token_id = tokenizer.eos_token_id
            if eos_token_id is None:
                raise ValueError("tokenizer must define eos_token_id")
            while len(self._token_buffer) < self.text_tokens:
                previous_cycle = self._cycle
                _, content = self._current_row()
                if self._cycle != previous_cycle:
                    self._token_buffer.clear()
                if self.block_aligned_record_policy == "record_units_hard_pack":
                    record_ids = tokenize_block_aligned_record_unit(
                        tokenizer,
                        content,
                        text_tokens=self.text_tokens,
                        block_size=self.block_size,
                    )
                    units = (record_ids,) if record_ids else ()
                else:
                    units = tokenize_block_aligned_document_units(
                        tokenizer,
                        content,
                        text_tokens=self.text_tokens,
                        block_size=self.block_size,
                    )
                for unit_ids in units:
                    offset = len(self._token_buffer) % self.text_tokens
                    remaining = self.text_tokens - offset if offset else self.text_tokens
                    if len(unit_ids) > remaining and remaining < self.text_tokens:
                        self._token_buffer.extend([int(eos_token_id)] * remaining)
                    self._token_buffer.extend(unit_ids)
            token_ids = self._token_buffer[: self.text_tokens]
            del self._token_buffer[: self.text_tokens]
            return make_hard_packed_block(token_ids, text_tokens=self.text_tokens)

        packed_ids: list[int] = []
        while len(packed_ids) < self.text_tokens:
            if self._token_buffer:
                content_ids = self._token_buffer
                self._token_buffer = []
            else:
                _, content = self._current_row()
                content_ids = tokenize_block_aligned_document_ids(
                    tokenizer,
                    content,
                    text_tokens=self.text_tokens,
                    block_size=self.block_size,
                )
            if not content_ids:
                continue

            if len(content_ids) > self.text_tokens:
                if packed_ids:
                    self._token_buffer = content_ids
                    break
                # Only an individually oversized content may be truncated. Its
                # overflow is discarded instead of leaking into the next pack.
                packed_ids.extend(content_ids[: self.text_tokens])
                break

            remaining = self.text_tokens - len(packed_ids)
            if len(content_ids) > remaining:
                # A fitting content starts intact in the next physical pack.
                self._token_buffer = content_ids
                break
            packed_ids.extend(content_ids)

        return make_padded_packed_block(
            packed_ids,
            pad_token_id=int(signature["pad_token_id"]),
            text_tokens=self.text_tokens,
        )

    def state_dict(self) -> dict[str, object]:
        state: dict[str, object] = {
            "version": 4,
            "signature": self._signature,
            "run_seed": self.seed,
            "text_tokens": self.text_tokens,
            "max_chars": self.max_chars,
            "token_buffer": list(self._token_buffer),
            "cycle": self._cycle,
            "shard_position": self._shard_position,
            "row_group": self._row_group,
            "row_index": self._row_index,
            "tokenizer_signature": self._tokenizer_signature,
        }
        if self.packing_mode == "block_aligned_eos":
            state.update(
                version=7,
                packing_mode=self.packing_mode,
                block_size=self.block_size,
            )
        return state

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        version = state.get("version")
        expected_version = 7 if self.packing_mode == "block_aligned_eos" else 4
        if version != expected_version:
            if self.packing_mode == "continuous":
                raise ValueError(
                    "UltraFineWeb resume requires hard-pack stream state version 4; "
                    "earlier checkpoints are unsupported"
                )
            raise ValueError("content-aligned EOS packing requires stream state version 7")
        expected_keys = (
            self._BLOCK_ALIGNED_STATE_KEYS
            if self.packing_mode == "block_aligned_eos"
            else self._HARD_PACK_STATE_KEYS
        )
        if set(state) != expected_keys:
            raise ValueError("UltraFineWeb stream state fields or version are malformed")
        if state.get("signature") != self._signature:
            raise ValueError("UltraFineWeb stream state does not match this dataset")
        if state.get("run_seed") != self.seed:
            raise ValueError("UltraFineWeb stream run seed does not match this dataset")
        if state.get("text_tokens") != self.text_tokens:
            raise ValueError("UltraFineWeb stream text_tokens does not match this run")
        if state.get("max_chars") != self.max_chars:
            raise ValueError("UltraFineWeb stream max_chars does not match this run")
        if version == 7 and state.get("packing_mode") != self.packing_mode:
            raise ValueError("UltraFineWeb stream packing_mode does not match this dataset")
        if version == 7 and state.get("block_size") != self.block_size:
            raise ValueError("UltraFineWeb stream block_size does not match this dataset")
        names = ("cycle", "shard_position", "row_group", "row_index")
        values = tuple(state.get(name) for name in names)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("UltraFineWeb cursor values must be non-negative integers")
        tokenizer_signature = state.get("tokenizer_signature")
        if tokenizer_signature is not None and not isinstance(tokenizer_signature, Mapping):
            raise ValueError("UltraFineWeb tokenizer signature is malformed")
        token_buffer = state.get("token_buffer")
        if not isinstance(token_buffer, list) or any(
            type(token_id) is not int or token_id < 0 for token_id in token_buffer
        ):
            raise ValueError("UltraFineWeb hard-pack token buffer is malformed")
        same_shard = (
            int(values[0]) == self._cycle
            and int(values[1]) == self._shard_position
        )
        (
            self._cycle,
            self._shard_position,
            self._row_group,
            self._row_index,
        ) = (int(value) for value in values)
        if self._shard_position >= len(self._local_paths()):
            raise ValueError("UltraFineWeb shard cursor is out of range")
        self._tokenizer_signature = (
            None if tokenizer_signature is None else dict(tokenizer_signature)
        )
        self._token_buffer = list(token_buffer)
        if not same_shard:
            self._reset_cache()
