"""Small, self-contained image/text dataset bundles for MF training."""

from __future__ import annotations

import hashlib
import io
import json
import random
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from torch import Tensor
from PIL import Image

from mf.config.schema import MFConfig
from mf.contracts.batch import TaskType
from mf.data.collate import RawTaskSample
from mf.data.images import load_image, preprocess_image
from mf.data.text import (
    TextTokenizer,
    TokenizedTextBlock,
    tokenize_caption,
    tokenize_condition,
)


@dataclass(frozen=True)
class BundleImageSample:
    image: Tensor
    captions: Mapping[str, str]


def _qa_records(record: Mapping[str, object]) -> list[dict[str, object]]:
    if not isinstance(record.get("image"), str):
        return []
    qa = record.get("qa")
    if isinstance(qa, list) and qa:
        return [
            {**record, "question": pair.get("question"), "answer": pair.get("answer")}
            for pair in qa
            if isinstance(pair, dict)
        ]
    return [dict(record)] if isinstance(record.get("caption"), str) else []


class _ShardedRecords:
    def __init__(
        self, root: Path, split: str, kind: str, rank: int, world_size: int, seed: int
    ) -> None:
        image = kind in {"image", "qa"}
        directory = "image_shards" if image else "text_shards"
        name = "images.manifest.json" if image else "text.manifest.json"
        manifest_path = root / split / name
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
        entries = manifest["shards"]
        if not entries:
            raise ValueError(f"empty bundle manifest: {manifest_path}")
        self.paths = []
        for entry in entries:
            archive = entry["archive"]
            if Path(archive).name != archive:
                raise ValueError(f"invalid archive name: {archive}")
            self.paths.append((root / split / directory / archive).resolve(strict=True))
        self.image = image
        self.kind = kind
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.signature = hashlib.sha256(
            json.dumps(
                [
                    str(root),
                    hashlib.sha256(payload).hexdigest(),
                    kind,
                    rank,
                    world_size,
                    seed,
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.cycle = 0
        self.shard_position = 0
        self.row_position = 0
        self._iterator = None

    def _local_shards(self) -> tuple[list[Path], int, int]:
        order = list(self.paths)
        random.Random(f"{self.seed}:{self.cycle}:{self.image}").shuffle(order)
        if len(order) >= self.world_size:
            return order[self.rank :: self.world_size], 1, 0
        group = self.rank % len(order)
        group_size = (self.world_size - 1 - group) // len(order) + 1
        return [order[group]], group_size, self.rank // len(order)

    def _rows(self, path: Path, stride: int, offset: int):
        position = 0
        if self.image:
            metadata = None
            with tarfile.open(path, mode="r|") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        continue
                    if member.name.endswith(".json"):
                        metadata = json.load(source)
                    elif member.name.endswith(".webp") and metadata is not None:
                        if metadata.get("image") != member.name:
                            raise ValueError(f"unpaired image in {path}: {member.name}")
                        if position % stride == offset:
                            record = {**metadata, "_image_bytes": source.read()}
                            if self.kind == "qa":
                                yield from _qa_records(record)
                            else:
                                yield record
                        metadata = None
                        position += 1
        else:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    if position % stride == offset:
                        yield row
                    position += 1

    def next_record(self) -> Mapping[str, object]:
        while True:
            shards, stride, offset = self._local_shards()
            if self.shard_position >= len(shards):
                self.cycle += 1
                self.shard_position = 0
                self.row_position = 0
                self._iterator = None
                continue
            if self._iterator is None:
                self._iterator = iter(
                    self._rows(shards[self.shard_position], stride, offset)
                )
                for _ in range(self.row_position):
                    next(self._iterator)
            try:
                record = next(self._iterator)
            except StopIteration:
                self.shard_position += 1
                self.row_position = 0
                self._iterator = None
                continue
            self.row_position += 1
            return record

    def state_dict(self) -> dict[str, object]:
        return {
            "version": 2,
            "signature": self.signature,
            "cycle": self.cycle,
            "shard_position": self.shard_position,
            "row_position": self.row_position,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            set(state)
            != {"version", "signature", "cycle", "shard_position", "row_position"}
            or state.get("version") != 2
            or state.get("signature") != self.signature
            or any(
                type(state.get(key)) is not int or state[key] < 0
                for key in ("cycle", "shard_position", "row_position")
            )
        ):
            raise ValueError("bundle stream state does not match this dataset")
        self.cycle = state["cycle"]
        self.shard_position = state["shard_position"]
        self.row_position = state["row_position"]
        self._iterator = None


class _ManifestStream:
    def __init__(
        self,
        *,
        root: str,
        split: str,
        kind: str,
        rank: int,
        world_size: int,
        seed: int,
        records_path: str | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        manifest = (
            self.root / split / "records.jsonl"
            if records_path is None
            else Path(records_path).expanduser().resolve(strict=True)
        )
        if not 0 <= rank < world_size:
            raise ValueError("bundle rank must be within world size")
        self._sharded = None
        if records_path is None and not manifest.exists():
            self._sharded = _ShardedRecords(
                self.root, split, kind, rank, world_size, seed
            )
            self.signature = self._sharded.signature
            return
        manifest = manifest.resolve(strict=True)
        if not manifest.is_file():
            raise ValueError(f"records_path must be a JSONL file: {manifest}")
        if records_path is None and not manifest.is_relative_to(self.root):
            raise ValueError("bundle manifest must be inside the bundle root")
        payload = manifest.read_bytes()
        manifest_hash = hashlib.sha256(payload).hexdigest()
        records = [json.loads(line) for line in payload.splitlines() if line.strip()]
        if not records or any(not isinstance(item, dict) for item in records):
            raise ValueError(f"bundle manifest must contain JSON objects: {manifest}")
        if kind == "image":
            selected = [
                item
                for item in records
                if isinstance(item.get("image"), str)
                and isinstance(item.get("caption"), str)
            ]
        elif kind == "qa":
            selected = [
                qa_record for item in records for qa_record in _qa_records(item)
            ]
        elif kind == "text":
            selected = [item for item in records if isinstance(item.get("text"), str)]
        else:
            raise ValueError(f"unknown bundle record kind: {kind}")
        if not selected:
            raise ValueError(f"bundle has no {kind} records: {manifest}")
        self.records = selected
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.cursor = 0
        self.signature = hashlib.sha256(
            json.dumps(
                [
                    str(manifest),
                    manifest_hash,
                    kind,
                    rank,
                    world_size,
                    seed,
                    "rank_shared_shuffle_v2",
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self._shuffle_seed = hashlib.sha256(
            json.dumps(
                [str(manifest), manifest_hash, kind, seed],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self._cycle = -1
        self._order: list[int] = []

    def _next_record(self) -> Mapping[str, object]:
        if self._sharded is not None:
            return self._sharded.next_record()
        cycle, offset = divmod(
            self.rank + self.cursor * self.world_size, len(self.records)
        )
        if cycle != self._cycle:
            self._order = list(range(len(self.records)))
            random.Random(f"{self.seed}:{cycle}:{self._shuffle_seed}").shuffle(
                self._order
            )
            self._cycle = cycle
        self.cursor += 1
        return self.records[self._order[offset]]

    def _image(self, record: Mapping[str, object], config: MFConfig) -> Tensor:
        payload = record.get("_image_bytes")
        if isinstance(payload, bytes):
            with Image.open(io.BytesIO(payload)) as source:
                return preprocess_image(
                    source,
                    resolution=config.codecs.vision.encoder_input_resolution,
                    policy=getattr(
                        config.codecs.vision,
                        "image_preprocessing",
                        "legacy_center_crop_bicubic_v1",
                    ),
                )
        name = record.get("image")
        if not isinstance(name, str) or not name:
            raise ValueError("bundle image record requires an image path")
        path = (self.root / name).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError("bundle image path must remain inside the bundle root")
        return load_image(path, config)

    def state_dict(self) -> dict[str, object]:
        if self._sharded is not None:
            return self._sharded.state_dict()
        return {"version": 2, "signature": self.signature, "cursor": self.cursor}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if self._sharded is not None:
            self._sharded.load_state_dict(state)
            return
        cursor = state.get("cursor")
        if (
            set(state) != {"version", "signature", "cursor"}
            or state.get("version") != 2
            or state.get("signature") != self.signature
            or type(cursor) is not int
            or cursor < 0
        ):
            raise ValueError("bundle stream state does not match this dataset")
        self.cursor = cursor
        self._cycle = -1


class BundleImageStream(_ManifestStream):
    def __init__(self, *, config: MFConfig, rank: int, world_size: int) -> None:
        bundle = config.data.bundle
        if bundle is None:
            raise ValueError("bundle data is not configured")
        super().__init__(
            root=bundle.root,
            split=bundle.split,
            kind="image",
            rank=rank,
            world_size=world_size,
            seed=config.run.seed,
            records_path=bundle.records_path,
        )
        self.config = config

    def next_sample(self) -> BundleImageSample:
        record = self._next_record()
        caption = record.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("bundle image-text record requires a caption")
        return BundleImageSample(self._image(record, self.config), {"short": caption})


class BundleQAStream(_ManifestStream):
    def __init__(
        self, *, config: MFConfig, rank: int, world_size: int, tokenizer: TextTokenizer
    ) -> None:
        bundle = config.data.bundle
        if bundle is None:
            raise ValueError("bundle data is not configured")
        super().__init__(
            root=bundle.root,
            split=bundle.split,
            kind="qa",
            rank=rank,
            world_size=world_size,
            seed=config.run.seed + 1,
            records_path=bundle.records_path,
        )
        self.config = config
        self.tokenizer = tokenizer

    def next_sample(self) -> RawTaskSample:
        record = self._next_record()
        question, answer = record.get("question"), record.get("answer")
        if (
            question is None
            and answer is None
            and isinstance(record.get("caption"), str)
        ):
            bundle = self.config.data.bundle
            assert bundle is not None
            question = bundle.image_to_text_instruction.prompt
            answer = record["caption"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError("bundle image-QA record requires a question")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("bundle image-QA record requires an answer")
        bundle = self.config.data.bundle
        assert bundle is not None
        return RawTaskSample(
            task_type=TaskType.IMAGE_TO_TEXT,
            image=self._image(record, self.config),
            text=tokenize_caption(
                self.tokenizer,
                answer,
                text_tokens=self.config.data.text_max_length,
                eos_fill_block_size=_eos_block_size(self.config),
            ),
            text_prompt=tokenize_condition(
                self.tokenizer,
                question,
                text_tokens=bundle.image_to_text_instruction.max_length,
            ),
        )


class BundleTextStream(_ManifestStream):
    def __init__(self, *, config: MFConfig, rank: int, world_size: int) -> None:
        bundle = config.data.bundle
        if bundle is None:
            raise ValueError("bundle data is not configured")
        super().__init__(
            root=bundle.root,
            split=bundle.split,
            kind="text",
            rank=rank,
            world_size=world_size,
            seed=config.run.seed + 2,
            records_path=bundle.records_path,
        )
        self.config = config

    def next_block(self, tokenizer: TextTokenizer) -> TokenizedTextBlock:
        text = self._next_record().get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("bundle text record requires non-empty text")
        return tokenize_caption(
            tokenizer,
            text,
            text_tokens=self.config.data.text_max_length,
            eos_fill_block_size=_eos_block_size(self.config),
        )


def _eos_block_size(config: MFConfig) -> int | None:
    packing = config.flow.text_block_causal
    if packing.packing != "block_aligned_eos":
        return None
    return packing.block_size
