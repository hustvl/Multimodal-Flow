"""Public SFT dataset recipe adapters.

The public SFT recipe combines the three configured text-to-image sources with
the official LLaVA-1.5 instruction JSON. Both streams emit the same raw task
sample contract used by the chunk-native trainer and are checkpointable.
"""

from __future__ import annotations

__mf_extension_version__ = "1"

import hashlib
from io import BytesIO
import json
import random
import re
import tarfile
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

from PIL import Image

from mf.config.schema import MFConfig
from mf.contracts.batch import TaskType
from mf.data.collate import RawTaskSample
from mf.data.images import load_image, preprocess_image
from mf.data.planner import stable_hash
from mf.data.sft import (
    eos_fill_block_size,
    register_sft_recipe,
    validate_stream_position,
)
from mf.data.text import (
    TextTokenizer,
    TokenizedTextBlock,
    tokenize_caption,
    tokenize_condition,
    tokenizer_resume_signature,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_TEXT_TO_IMAGE_SOURCES = ("blip3_ft60k", "dalle3", "sharegpt4o")
_BLIP3_ARCHIVES = (
    "dalle3.tar",
    "geneval_train.tar",
    "human_gestures.tar",
    "journeyDB.tar",
    "mscoco_human.tar",
    "object_1.tar",
    "object_2.tar",
    "occupation_1.tar",
    "occupation_2.tar",
    "text_1.tar",
    "text_2.tar",
)
_IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".png", ".webp"))
_DALLE_PATTERN = re.compile(r"shard-(\d+)\.tar")
_SHARE_PART_PATTERN = re.compile(r"text_to_image_part_(\d+)\.tar")
_SHARE_SHARD_PATTERN = re.compile(r"shard-(\d+)\.tar")
_CURSOR_KEYS = frozenset(("cycle", "shard_position", "pair_index"))


@dataclass(frozen=True, slots=True)
class _TextToImagePair:
    stem: str
    text_member: str | None
    image_member: str


@dataclass(frozen=True, slots=True)
class _CaptionIndex:
    captions: Mapping[str, str]
    sha256: str
    fingerprint: str


def _numbered_archives(
    source: str,
    directory: Path,
    pattern: re.Pattern[str],
    expected_count: int,
) -> tuple[Path, ...]:
    numbered = []
    for path in sorted(directory.glob("*.tar")):
        match = pattern.fullmatch(path.name)
        if match is not None:
            numbered.append((int(match.group(1)), path))
    if not numbered:
        raise FileNotFoundError(f"missing {source} tar archives under {directory}")
    numbers = tuple(number for number, _ in numbered)
    expected = tuple(range(expected_count))
    if numbers != expected:
        raise ValueError(f"{source} requires contiguous archives {expected}; got {numbers}")
    files = tuple(path for _, path in numbered)
    if any(path.is_symlink() or path.stat().st_size <= 0 for path in files):
        raise ValueError(f"{source} archives must be non-empty regular files")
    return files


def _source_archives(root: Path, source: str) -> tuple[Path, ...]:
    directory = root / source
    if not directory.is_dir():
        raise FileNotFoundError(f"missing text-to-image source directory: {directory}")
    if source == "blip3_ft60k":
        files = tuple(directory / name for name in _BLIP3_ARCHIVES)
        if any(path.is_symlink() or not path.is_file() or path.stat().st_size <= 0 for path in files):
            raise FileNotFoundError(f"incomplete blip3_ft60k source under {directory}")
    elif source == "dalle3":
        files = _numbered_archives(source, directory, _DALLE_PATTERN, 14)
    elif source == "sharegpt4o":
        part_files = tuple(
            path
            for path in sorted(directory.glob("*.tar"))
            if _SHARE_PART_PATTERN.fullmatch(path.name)
        )
        pattern = _SHARE_PART_PATTERN if part_files else _SHARE_SHARD_PATTERN
        files = _numbered_archives(source, directory, pattern, 10)
    else:
        raise ValueError(f"unsupported text-to-image source: {source!r}")
    return files


def _resolve_sidecar(root: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file() and path.name == "text_to_image.json":
        for alternate in ("text_and_image_to_image.json", "text_to_image.json"):
            candidate = (path.parent / alternate).resolve()
            if candidate.is_file():
                path = candidate
                break
    if not path.is_relative_to(root.resolve()):
        raise ValueError("ShareGPT4o caption file must remain under the dataset root")
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing ShareGPT4o caption file: {path}")
    return path


def _load_caption_index(
    root: Path,
    configured_path: str,
    tokenizer: TextTokenizer,
    max_tokens: int,
) -> _CaptionIndex:
    path = _resolve_sidecar(root, configured_path)
    raw = path.read_bytes()
    try:
        rows = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid ShareGPT4o caption file: {path}") from error
    if not isinstance(rows, list) or not rows:
        raise ValueError("ShareGPT4o caption file must contain a non-empty list")
    captions: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"ShareGPT4o row {index} must be a mapping")
        image_name = row.get("output_image")
        prompt = row.get("input_prompt")
        if not isinstance(image_name, str) or not image_name:
            raise ValueError(f"ShareGPT4o row {index} has no output_image")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"ShareGPT4o row {index} has no input_prompt")
        key = str(PurePosixPath(image_name).with_suffix(""))
        if key in captions:
            raise ValueError(f"duplicate ShareGPT4o output_image key: {key}")
        if len(tokenizer.encode(prompt, add_special_tokens=True)) <= max_tokens:
            captions[key] = prompt.strip()
    if not captions:
        raise ValueError("ShareGPT4o caption filtering removed every record")
    sha256 = hashlib.sha256(raw).hexdigest()
    identity = {
        "format": "mf-sharegpt4o-caption-index-v1",
        "sha256": sha256,
        "max_tokens": max_tokens,
        "retained": len(captions),
        "tokenizer": tokenizer_resume_signature(tokenizer),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return _CaptionIndex(captions=captions, sha256=sha256, fingerprint=fingerprint)


class TextToImageSFTStream:
    """Checkpointable stream for the configured text-to-image sources."""

    _STATE_KEYS = frozenset(
        ("version", "signature", "selection_index", "cursors", "skipped")
    )

    def __init__(
        self,
        *,
        config: MFConfig,
        tokenizer: TextTokenizer,
        stream_rank: int,
        stream_world_size: int,
    ) -> None:
        settings = config.sft.text_to_image
        if settings is None:
            raise ValueError("text-to-image SFT configuration is missing")
        validate_stream_position(stream_rank, stream_world_size)
        self.config = config
        self.tokenizer = tokenizer
        self.stream_rank = stream_rank
        self.stream_world_size = stream_world_size
        self.settings = settings
        self.root = Path(settings.root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"text-to-image root does not exist: {self.root}")
        self.sources = tuple(settings.sources)
        self.weights = tuple(
            float(getattr(settings.weights, source)) for source in self.sources
        )
        self.archives = {
            source: _source_archives(self.root, source) for source in self.sources
        }
        self.caption_index = (
            _load_caption_index(
                self.root,
                settings.sharegpt4o_caption_file,
                tokenizer,
                settings.sharegpt4o_max_tokens,
            )
            if "sharegpt4o" in self.sources
            else None
        )
        self._local_files: dict[str, tuple[Path, ...]] = {}
        self._pair_lanes: dict[str, tuple[int, int]] = {}
        for source in self.sources:
            files = self.archives[source]
            if stream_world_size > len(files):
                shard_index = stream_rank % len(files)
                lane_index = stream_rank // len(files)
                lane_count = ((stream_world_size - 1 - shard_index) // len(files)) + 1
                self._local_files[source] = (files[shard_index],)
                self._pair_lanes[source] = (lane_index, lane_count)
            else:
                self._local_files[source] = files[stream_rank::stream_world_size]
                self._pair_lanes[source] = (0, 1)
        self._cursors = {
            source: {"cycle": 0, "shard_position": 0, "pair_index": 0}
            for source in self.sources
        }
        self._selection_index = 0
        self._skipped: dict[str, int] = {}
        self._warnings_emitted = 0
        self._archives: dict[str, tarfile.TarFile] = {}
        self._archive_paths: dict[str, Path] = {}
        self._pairs: dict[str, tuple[_TextToImagePair, ...]] = {}
        manifest = {
            "root": str(self.root),
            "sources": {
                source: [
                    {
                        "path": str(path.relative_to(self.root)),
                        "size": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns,
                    }
                    for path in self.archives[source]
                ]
                for source in self.sources
            },
            "caption_index": None if self.caption_index is None else self.caption_index.fingerprint,
            "sources_selected": self.sources,
            "weights": self.weights,
            "rank": stream_rank,
            "world_size": stream_world_size,
            "seed": config.run.seed,
            "tokenizer": tokenizer_resume_signature(tokenizer),
        }
        self._signature = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._rng_seed = config.run.seed

    def _source_name(self) -> str:
        total = sum(self.weights)
        bucket = stable_hash(self._rng_seed, self._selection_index, "mf-text-to-image-source")
        pick = (bucket % 1_000_000_000) / 1_000_000_000 * total
        cumulative = 0.0
        for source, weight in zip(self.sources, self.weights, strict=True):
            cumulative += weight
            if pick < cumulative:
                return source
        return self.sources[-1]

    @staticmethod
    def _semantic_pairs(members: Sequence[tarfile.TarInfo]) -> tuple[_TextToImagePair, ...]:
        images: dict[str, list[str]] = {}
        texts: dict[str, list[str]] = {}
        for member in members:
            if not member.isfile():
                continue
            suffix = PurePosixPath(member.name).suffix.lower()
            if suffix == ".txt":
                texts.setdefault(str(PurePosixPath(member.name).with_suffix("")), []).append(member.name)
            elif suffix in _IMAGE_SUFFIXES:
                images.setdefault(str(PurePosixPath(member.name).with_suffix("")), []).append(member.name)
        return tuple(
            _TextToImagePair(stem=stem, text_member=texts[stem][0], image_member=images[stem][0])
            for stem in sorted(texts.keys() & images.keys())
            if len(texts[stem]) == 1 and len(images[stem]) == 1
        )

    def _caption_pairs(self, members: Sequence[tarfile.TarInfo]) -> tuple[_TextToImagePair, ...]:
        if self.caption_index is None:
            raise RuntimeError("ShareGPT4o caption index is unavailable")
        images: dict[str, list[str]] = {}
        for member in members:
            if member.isfile() and PurePosixPath(member.name).suffix.lower() in _IMAGE_SUFFIXES:
                images.setdefault(str(PurePosixPath(member.name).with_suffix("")), []).append(member.name)
        return tuple(
            _TextToImagePair(stem=stem, text_member=None, image_member=images[stem][0])
            for stem in sorted(images.keys() & self.caption_index.captions.keys())
            if len(images[stem]) == 1
        )

    def _close_source(self, source: str) -> None:
        archive = self._archives.pop(source, None)
        if archive is not None:
            archive.close()
        self._archive_paths.pop(source, None)
        self._pairs.pop(source, None)

    def close(self) -> None:
        for source in tuple(self._archives):
            self._close_source(source)

    def _ensure_archive(self, source: str) -> tuple[tarfile.TarFile, tuple[_TextToImagePair, ...]]:
        cursor = self._cursors[source]
        path = self._local_files[source][cursor["shard_position"]]
        if self._archive_paths.get(source) == path:
            return self._archives[source], self._pairs[source]
        self._close_source(source)
        archive = tarfile.open(path, mode="r:*")
        members = archive.getmembers()
        pairs = self._caption_pairs(members) if source == "sharegpt4o" else self._semantic_pairs(members)
        lane_index, lane_count = self._pair_lanes[source]
        pairs = pairs[lane_index::lane_count]
        self._archives[source] = archive
        self._archive_paths[source] = path
        self._pairs[source] = pairs
        if cursor["pair_index"] > len(pairs):
            raise ValueError(f"text-to-image cursor exceeds pair count for {source}")
        return archive, pairs

    def _advance_shard(self, source: str) -> None:
        cursor = self._cursors[source]
        self._close_source(source)
        cursor["pair_index"] = 0
        cursor["shard_position"] += 1
        if cursor["shard_position"] >= len(self._local_files[source]):
            cursor["shard_position"] = 0
            cursor["cycle"] += 1

    @staticmethod
    def _read_member(archive: tarfile.TarFile, member_name: str) -> bytes:
        stream = archive.extractfile(archive.getmember(member_name))
        if stream is None:
            raise ValueError(f"unable to extract text-to-image member {member_name!r}")
        return stream.read()

    def _note_skip(self, source: str, reason: str) -> None:
        key = f"{source}:{reason}"
        self._skipped[key] = self._skipped.get(key, 0) + 1
        if self._warnings_emitted < 20:
            warnings.warn(
                f"skipping text-to-image sample source={source} reason={reason}",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warnings_emitted += 1

    def _decode_image(self, payload: bytes):
        with Image.open(BytesIO(payload)) as image:
            return preprocess_image(
                image,
                resolution=self.config.codecs.vision.encoder_input_resolution,
                policy=getattr(
                    self.config.codecs.vision,
                    "image_preprocessing",
                    "legacy_center_crop_bicubic_v1",
                ),
            )

    def _next_from_source(self, source: str) -> RawTaskSample:
        for _ in range(self.settings.max_consecutive_invalid_records):
            try:
                archive, pairs = self._ensure_archive(source)
            except (EOFError, OSError, tarfile.TarError, ValueError) as error:
                self._note_skip(source, f"shard:{type(error).__name__}")
                self._advance_shard(source)
                continue
            cursor = self._cursors[source]
            if cursor["pair_index"] >= len(pairs):
                self._advance_shard(source)
                continue
            pair = pairs[cursor["pair_index"]]
            cursor["pair_index"] += 1
            try:
                if pair.text_member is None:
                    if self.caption_index is None:
                        raise RuntimeError("ShareGPT4o caption index is unavailable")
                    caption = self.caption_index.captions[pair.stem]
                else:
                    caption = self._read_member(archive, pair.text_member).decode("utf-8").strip()
                if not caption:
                    self._note_skip(source, "empty_caption")
                    continue
                image = self._decode_image(self._read_member(archive, pair.image_member))
                text = tokenize_caption(
                    self.tokenizer,
                    caption,
                    text_tokens=self.config.data.text_max_length,
                    content_tokens=self.config.data.text_max_length,
                    eos_fill_block_size=eos_fill_block_size(self.config),
                )
            except (UnicodeError, KeyError):
                self._note_skip(source, "invalid_caption")
                continue
            except (EOFError, OSError, tarfile.TarError, ValueError):
                self._note_skip(source, "invalid_sample")
                continue
            return RawTaskSample(task_type=TaskType.TEXT_TO_IMAGE, image=image, text=text)
        raise RuntimeError(
            "text-to-image stream exceeded its invalid-record retry budget "
            f"({self.settings.max_consecutive_invalid_records}) for {source}"
        )

    def next_sample(self) -> RawTaskSample:
        source = self._source_name()
        self._selection_index += 1
        return self._next_from_source(source)

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "signature": self._signature,
            "selection_index": self._selection_index,
            "cursors": json.loads(json.dumps(self._cursors)),
            "skipped": dict(sorted(self._skipped.items())),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != self._STATE_KEYS or state.get("version") != 1:
            raise ValueError("text-to-image stream state fields are malformed")
        if state.get("signature") != self._signature:
            raise ValueError("text-to-image stream state does not match this run")
        selection_index = state.get("selection_index")
        cursors = state.get("cursors")
        skipped = state.get("skipped")
        if type(selection_index) is not int or selection_index < 0:
            raise ValueError("text-to-image selection index is malformed")
        if not isinstance(cursors, Mapping) or set(cursors) != set(self._cursors):
            raise ValueError("text-to-image source cursors are malformed")
        resolved_cursors: dict[str, dict[str, int]] = {}
        for source, raw in cursors.items():
            if not isinstance(raw, Mapping) or set(raw) != _CURSOR_KEYS:
                raise ValueError("text-to-image cursor fields are malformed")
            values = {key: raw.get(key) for key in _CURSOR_KEYS}
            if any(type(value) is not int or value < 0 for value in values.values()):
                raise ValueError("text-to-image cursor values are malformed")
            if values["shard_position"] >= len(self._local_files[source]):
                raise ValueError("text-to-image shard cursor is out of range")
            resolved_cursors[source] = {key: int(value) for key, value in values.items()}
        if not isinstance(skipped, Mapping) or any(
            not isinstance(key, str) or type(value) is not int or value <= 0
            for key, value in skipped.items()
        ):
            raise ValueError("text-to-image skip counters are malformed")
        self.close()
        self._selection_index = selection_index
        self._cursors = resolved_cursors
        self._skipped = dict(skipped)


def _huggingface_files(root: Path, source: str) -> tuple[Path, ...]:
    directory = root / source
    if not directory.is_dir():
        raise FileNotFoundError(f"missing Hugging Face source directory: {directory}")
    search_roots = (directory, directory / "data")
    files = tuple(sorted(
        path
        for search_root in search_roots
        if search_root.is_dir()
        for pattern in ("*.parquet", "*.jsonl", "*.json")
        for path in search_root.glob(pattern)
        if path.name not in {"dataset_info.json", "state.json"}
        and path.is_file()
        and not path.is_symlink()
        and path.stat().st_size > 0
    ))
    if not files:
        raise FileNotFoundError(
            f"no Hugging Face parquet/json files found under {directory}"
        )
    return files


def _has_huggingface_records(root: Path, source: str) -> bool:
    directory = root / source
    if not directory.is_dir():
        return False
    for search_root in (directory, directory / "data"):
        if not search_root.is_dir():
            continue
        if any(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size > 0
            and path.name not in {"dataset_info.json", "state.json"}
            for pattern in ("*.parquet", "*.jsonl")
            for path in search_root.glob(pattern)
        ):
            return True
        if any(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size > 0
            and path.name not in {"dataset_info.json", "state.json", "text_to_image.json"}
            for path in search_root.glob("*.json")
        ):
            return True
    return False


class _HuggingFaceTextToImageSFTStream:
    """Read local Hugging Face parquet/JSON dataset snapshots lazily."""

    _STATE_KEYS = frozenset(
        ("version", "signature", "selection_index", "cursors", "skipped")
    )
    _CURSOR_KEYS = frozenset(("file_index", "row_group", "row_index"))

    def __init__(
        self,
        *,
        config: MFConfig,
        tokenizer: TextTokenizer,
        stream_rank: int,
        stream_world_size: int,
    ) -> None:
        settings = config.sft.text_to_image
        if settings is None:
            raise ValueError("text-to-image SFT configuration is missing")
        validate_stream_position(stream_rank, stream_world_size)
        self.config = config
        self.tokenizer = tokenizer
        self.settings = settings
        self.stream_rank = stream_rank
        self.stream_world_size = stream_world_size
        self.root = Path(settings.root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"text-to-image root does not exist: {self.root}")
        self.sources = tuple(settings.sources)
        self.files = {source: _huggingface_files(self.root, source) for source in self.sources}
        self.weights = tuple(
            float(getattr(settings.weights, source)) for source in self.sources
        )
        self.caption_index = None
        if "sharegpt4o" in self.sources:
            sidecar = Path(settings.sharegpt4o_caption_file).expanduser()
            if not sidecar.is_absolute():
                sidecar = self.root / sidecar
            if sidecar.is_file() and not sidecar.is_symlink():
                self.caption_index = _load_caption_index(
                    self.root,
                    settings.sharegpt4o_caption_file,
                    tokenizer,
                    settings.sharegpt4o_max_tokens,
                )
        self._local_files: dict[str, tuple[Path, ...]] = {}
        self._row_lanes: dict[str, tuple[int, int]] = {}
        for source in self.sources:
            source_files = self.files[source]
            if stream_world_size > len(source_files):
                file_index = stream_rank % len(source_files)
                lane_index = stream_rank // len(source_files)
                lane_count = ((stream_world_size - 1 - file_index) // len(source_files)) + 1
                self._local_files[source] = (source_files[file_index],)
                self._row_lanes[source] = (lane_index, lane_count)
            else:
                self._local_files[source] = source_files[stream_rank::stream_world_size]
                self._row_lanes[source] = (0, 1)
        self._cursors = {
            source: {"file_index": 0, "row_group": 0, "row_index": 0}
            for source in self.sources
        }
        self._selection_index = 0
        self._skipped: dict[str, int] = {}
        self._warnings_emitted = 0
        self._row_cache: dict[str, tuple[Path, int, list[Mapping[str, object]]]] = {}
        manifest = {
            "format": "mf-huggingface-text-to-image-v1",
            "root": str(self.root),
            "sources": {
                source: [
                    {
                        "path": str(path.relative_to(self.root)),
                        "size": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns,
                    }
                    for path in self.files[source]
                ]
                for source in self.sources
            },
            "caption_index": None if self.caption_index is None else self.caption_index.fingerprint,
            "weights": self.weights,
            "rank": stream_rank,
            "world_size": stream_world_size,
            "row_lanes": self._row_lanes,
            "seed": config.run.seed,
            "tokenizer": tokenizer_resume_signature(tokenizer),
        }
        self._signature = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _source_name(self) -> str:
        total = sum(self.weights)
        bucket = stable_hash(self.config.run.seed, self._selection_index, "mf-text-to-image-source")
        pick = (bucket % 1_000_000_000) / 1_000_000_000 * total
        cumulative = 0.0
        for source, weight in zip(self.sources, self.weights, strict=True):
            cumulative += weight
            if pick < cumulative:
                return source
        return self.sources[-1]

    @staticmethod
    def _json_rows(path: Path) -> list[Mapping[str, object]]:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError(f"Hugging Face JSON file must contain a list of records: {path}")
        return list(rows)

    def _row_group_count(self, path: Path) -> int:
        if path.suffix != ".parquet":
            return 1
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).num_row_groups

    def _load_rows(self, source: str) -> list[Mapping[str, object]]:
        cursor = self._cursors[source]
        path = self._local_files[source][cursor["file_index"]]
        row_group = cursor["row_group"]
        cached = self._row_cache.get(source)
        if cached is not None and cached[:2] == (path, row_group):
            return cached[2]
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            rows = pq.ParquetFile(path).read_row_group(row_group).to_pylist()
        else:
            if row_group != 0:
                raise ValueError("JSON source row group must be zero")
            rows = self._json_rows(path)
        if not rows:
            raise ValueError(f"Hugging Face source row group is empty: {path}")
        lane_index, lane_count = self._row_lanes[source]
        rows = rows[lane_index::lane_count]
        resolved = [row for row in rows if isinstance(row, Mapping)]
        self._row_cache[source] = (path, row_group, resolved)
        return resolved

    def _advance(self, source: str) -> None:
        cursor = self._cursors[source]
        path = self._local_files[source][cursor["file_index"]]
        cursor["row_index"] = 0
        cursor["row_group"] += 1
        if cursor["row_group"] >= self._row_group_count(path):
            cursor["row_group"] = 0
            cursor["file_index"] += 1
        if cursor["file_index"] >= len(self._local_files[source]):
            cursor["file_index"] = 0
        self._row_cache.pop(source, None)

    @staticmethod
    def _image_value(row: Mapping[str, object]) -> object:
        for key in ("image", "images", "image_data", "output_image"):
            if key in row and row[key] is not None:
                return row[key]
        return None

    @staticmethod
    def _caption_value(row: Mapping[str, object]) -> str | None:
        for key in ("caption", "text", "txt", "prompt", "input_prompt", "description"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _resolve_image(self, source: str, value: object):
        payload: bytes | None = None
        image_path: Path | None = None
        if isinstance(value, Mapping):
            raw_bytes = value.get("bytes")
            raw_path = value.get("path")
            if isinstance(raw_bytes, (bytes, bytearray, memoryview)):
                payload = bytes(raw_bytes)
            elif isinstance(raw_path, str) and raw_path:
                value = raw_path
        if payload is None and isinstance(value, (bytes, bytearray, memoryview)):
            payload = bytes(value)
        if payload is None and isinstance(value, str) and value:
            root = (self.root / source).resolve()
            image_path = (root / value).resolve()
            if not image_path.is_relative_to(root) or not image_path.is_file():
                image_path = (self.root / value).resolve()
                if not image_path.is_relative_to(self.root) or not image_path.is_file():
                    raise FileNotFoundError(f"image path is outside the dataset root: {value}")
        if image_path is not None:
            return load_image(image_path, self.config)
        if payload is None:
            raise ValueError("Hugging Face record has no supported image value")
        with Image.open(BytesIO(payload)) as image:
            return preprocess_image(
                image,
                resolution=self.config.codecs.vision.encoder_input_resolution,
                policy=getattr(
                    self.config.codecs.vision,
                    "image_preprocessing",
                    "legacy_center_crop_bicubic_v1",
                ),
            )

    def _next_from_source(self, source: str) -> RawTaskSample:
        for _ in range(self.settings.max_consecutive_invalid_records):
            try:
                rows = self._load_rows(source)
                cursor = self._cursors[source]
                if cursor["row_index"] >= len(rows):
                    self._advance(source)
                    continue
                row = rows[cursor["row_index"]]
                cursor["row_index"] += 1
                image_value = self._image_value(row)
                caption = self._caption_value(row)
                if caption is None and source == "sharegpt4o" and self.caption_index is not None:
                    if isinstance(image_value, Mapping):
                        image_value = image_value.get("path") or image_value.get("bytes")
                    if isinstance(image_value, str):
                        key = str(PurePosixPath(image_value).with_suffix(""))
                        caption = self.caption_index.captions.get(key)
                if caption is None:
                    raise ValueError("record has no non-empty caption")
                if source == "sharegpt4o" and len(
                    self.tokenizer.encode(caption, add_special_tokens=True)
                ) > self.settings.sharegpt4o_max_tokens:
                    raise ValueError("caption exceeds ShareGPT4o eligibility limit")
                image = self._resolve_image(source, image_value)
                text = tokenize_caption(
                    self.tokenizer,
                    caption,
                    text_tokens=self.config.data.text_max_length,
                    content_tokens=self.config.data.text_max_length,
                    eos_fill_block_size=eos_fill_block_size(self.config),
                )
            except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
                self._note_skip(source, type(error).__name__)
                continue
            return RawTaskSample(task_type=TaskType.TEXT_TO_IMAGE, image=image, text=text)
        raise RuntimeError(
            "Hugging Face text-to-image stream exceeded its invalid-record retry budget "
            f"({self.settings.max_consecutive_invalid_records}) for {source}"
        )

    def _note_skip(self, source: str, reason: str) -> None:
        key = f"{source}:{reason}"
        self._skipped[key] = self._skipped.get(key, 0) + 1
        if self._warnings_emitted < 20:
            warnings.warn(
                f"skipping text-to-image sample source={source} reason={reason}",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warnings_emitted += 1

    def next_sample(self) -> RawTaskSample:
        source = self._source_name()
        self._selection_index += 1
        return self._next_from_source(source)

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "signature": self._signature,
            "selection_index": self._selection_index,
            "cursors": json.loads(json.dumps(self._cursors)),
            "skipped": dict(sorted(self._skipped.items())),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != self._STATE_KEYS or state.get("version") != 1:
            raise ValueError("Hugging Face text-to-image stream state is malformed")
        if state.get("signature") != self._signature:
            raise ValueError("Hugging Face text-to-image state does not match this run")
        selection_index = state.get("selection_index")
        cursors = state.get("cursors")
        skipped = state.get("skipped")
        if type(selection_index) is not int or selection_index < 0:
            raise ValueError("Hugging Face selection index is malformed")
        if not isinstance(cursors, Mapping) or set(cursors) != set(self._cursors):
            raise ValueError("Hugging Face source cursors are malformed")
        resolved_cursors: dict[str, dict[str, int]] = {}
        for source, raw in cursors.items():
            if not isinstance(raw, Mapping) or not self._CURSOR_KEYS.issubset(raw):
                raise ValueError("Hugging Face cursor fields are malformed")
            values = {key: raw.get(key) for key in self._CURSOR_KEYS}
            if any(type(value) is not int or value < 0 for value in values.values()):
                raise ValueError("Hugging Face cursor values are malformed")
            if values["file_index"] >= len(self._local_files[source]):
                raise ValueError("Hugging Face file cursor is out of range")
            resolved_cursors[source] = {key: int(value) for key, value in values.items()}
        if not isinstance(skipped, Mapping) or any(
            not isinstance(key, str) or type(value) is not int or value <= 0
            for key, value in skipped.items()
        ):
            raise ValueError("Hugging Face skip counters are malformed")
        self._selection_index = selection_index
        self._cursors = resolved_cursors
        self._skipped = dict(skipped)
        self._row_cache.clear()


def _build_text_to_image_stream(
    config: MFConfig,
    tokenizer: TextTokenizer,
    stream_rank: int,
    stream_world_size: int,
) -> TextToImageSFTStream | _HuggingFaceTextToImageSFTStream:
    settings = config.sft.text_to_image
    if settings is None:
        raise ValueError("text-to-image SFT configuration is missing")
    dataset_root = Path(settings.root).expanduser().resolve()
    use_huggingface_records = settings.format == "huggingface" and all(
        _has_huggingface_records(dataset_root, source) for source in settings.sources
    )
    stream_type = (
        _HuggingFaceTextToImageSFTStream
        if use_huggingface_records
        else TextToImageSFTStream
    )
    return stream_type(
        config=config,
        tokenizer=tokenizer,
        stream_rank=stream_rank,
        stream_world_size=stream_world_size,
    )


class LLaVA15InstructionStream:
    """Checkpointable sampler for the official LLaVA-1.5 instruction JSON."""

    _STATE_KEYS = frozenset(
        (
            "version",
            "signature",
            "sample_count",
            "skipped_invalid_count",
            "skipped_overlength_count",
            "rng_state",
        )
    )

    def __init__(
        self,
        *,
        config: MFConfig,
        tokenizer: TextTokenizer,
        stream_rank: int,
        stream_world_size: int,
    ) -> None:
        if config.sft.vqa is None:
            raise ValueError("LLaVA-1.5 SFT configuration is missing")
        validate_stream_position(stream_rank, stream_world_size)
        self.config = config
        self.tokenizer = tokenizer
        self.stream_rank = stream_rank
        self.stream_world_size = stream_world_size
        self.settings = config.sft.vqa
        self.json_path = Path(self.settings.train_json).expanduser().resolve()
        self.image_root = Path(self.settings.image_root).expanduser().resolve()
        if not self.json_path.is_file():
            raise FileNotFoundError(f"LLaVA-1.5 instruction file does not exist: {self.json_path}")
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"LLaVA-1.5 image root does not exist: {self.image_root}")
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            payload = payload.get("data")
        if not isinstance(payload, list) or not payload:
            raise ValueError("LLaVA-1.5 instruction JSON must contain a non-empty list")
        self.records = tuple(payload)
        self._signature = self._make_signature()
        self._rng = random.Random(config.run.seed + 7001 + stream_rank)
        self.sample_count = 0
        self.skipped_invalid_count = 0
        self.skipped_overlength_count = 0

    def _make_signature(self) -> str:
        payload = {
            "format": "mf_llava15_instruction_stream_v1",
            "json": str(self.json_path),
            "json_sha256": _file_sha256(self.json_path),
            "image_root": str(self.image_root),
            "record_count": len(self.records),
            "seed": self.config.run.seed,
            "stream_rank": self.stream_rank,
            "stream_world_size": self.stream_world_size,
            "tokenizer": tokenizer_resume_signature(self.tokenizer),
            "prompt_max_length": self.settings.prompt_max_length,
            "answer_max_length": self.settings.answer_max_length,
            "text_max_length": self.config.data.text_max_length,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _turn_text(turn: object, role: str) -> str | None:
        if not isinstance(turn, Mapping) or turn.get("from") != role:
            return None
        value = turn.get("value")
        return value.strip() if isinstance(value, str) and value.strip() else None

    def _parse_record(self, record: object) -> tuple[Path, str, str] | None:
        if not isinstance(record, Mapping):
            return None
        image_name = record.get("image")
        conversations = record.get("conversations")
        if not isinstance(image_name, str) or not image_name.strip():
            return None
        if not isinstance(conversations, list):
            return None
        question: str | None = None
        answer: str | None = None
        for turn in conversations:
            if question is None:
                question = self._turn_text(turn, "human")
                if question is not None:
                    question = question.replace("<image>", "").strip()
            elif answer is None:
                answer = self._turn_text(turn, "gpt")
                if answer is not None:
                    break
        if not question or not answer:
            return None
        roots = [self.image_root]
        if Path(image_name).parts[:1] == (self.image_root.name,):
            roots.append(self.image_root.parent)
        for root in roots:
            image_path = (root / image_name).resolve()
            if image_path.is_relative_to(root.resolve()) and image_path.is_file():
                return image_path, question, answer
        return None

    def _answer_block(self, answer: str) -> TokenizedTextBlock:
        return tokenize_caption(
            self.tokenizer,
            answer,
            text_tokens=self.config.data.text_max_length,
            content_tokens=self.settings.answer_max_length,
            eos_fill_block_size=eos_fill_block_size(self.config),
        )

    def _answer_content_length(self, answer: str) -> int:
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None or pad_token_id is None:
            raise ValueError("tokenizer must define eos_token_id and pad_token_id")
        return sum(
            token_id not in (eos_token_id, pad_token_id)
            for token_id in self.tokenizer.encode(answer, add_special_tokens=False)
        )

    def next_sample(self) -> RawTaskSample:
        for _ in range(self.settings.max_consecutive_invalid_records):
            self.sample_count += 1
            record = self.records[self._rng.randrange(len(self.records))]
            parsed = self._parse_record(record)
            if parsed is None:
                self.skipped_invalid_count += 1
                continue
            image_path, question, answer = parsed
            if self._answer_content_length(answer) > self.settings.answer_max_length:
                self.skipped_overlength_count += 1
                continue
            answer_block = self._answer_block(answer)
            return RawTaskSample(
                task_type=TaskType.IMAGE_TO_TEXT,
                image=load_image(image_path, self.config),
                text=answer_block,
                text_prompt=tokenize_condition(
                    self.tokenizer,
                    question,
                    text_tokens=self.settings.prompt_max_length,
                ),
            )
        raise RuntimeError(
            "LLaVA-1.5 instruction stream exceeded its invalid-record retry budget "
            f"({self.settings.max_consecutive_invalid_records})"
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "signature": self._signature,
            "sample_count": self.sample_count,
            "skipped_invalid_count": self.skipped_invalid_count,
            "skipped_overlength_count": self.skipped_overlength_count,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != self._STATE_KEYS or state.get("version") != 1:
            raise ValueError("LLaVA-1.5 instruction stream state fields are malformed")
        if state.get("signature") != self._signature:
            raise ValueError("LLaVA-1.5 instruction stream state does not match this run")
        for name in ("sample_count", "skipped_invalid_count", "skipped_overlength_count"):
            value = state.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"LLaVA-1.5 instruction stream {name} is malformed")
        rng_state = state.get("rng_state")
        if not isinstance(rng_state, tuple):
            raise ValueError("LLaVA-1.5 instruction stream RNG state is malformed")
        self._rng.setstate(rng_state)
        self.sample_count = state["sample_count"]
        self.skipped_invalid_count = state["skipped_invalid_count"]
        self.skipped_overlength_count = state["skipped_overlength_count"]


def build_vqa_stream(
    config: MFConfig,
    tokenizer: TextTokenizer,
    stream_rank: int,
    stream_world_size: int,
) -> LLaVA15InstructionStream:
    return LLaVA15InstructionStream(
        config=config,
        tokenizer=tokenizer,
        stream_rank=stream_rank,
        stream_world_size=stream_world_size,
    )


build_text_to_image_stream = _build_text_to_image_stream

register_sft_recipe(
    "public",
    image_to_text_task_stream_factory=build_vqa_stream,
    text_to_image_task_stream_factory=build_text_to_image_stream,
)


__all__ = [
    "LLaVA15InstructionStream",
    "TextToImageSFTStream",
    "build_vqa_stream",
    "build_text_to_image_stream",
]
