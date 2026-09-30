from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import accumulate
from typing import cast

import torch
from mf.config.schema import MFConfig
from mf.data.sources import (
    configured_source_names,
    configured_text_source_names,
    expected_source,
    normalized_data_config,
)
from mf.data.planner import stable_hash
from mf.data.readers import (
    BlockIndexedReader,
    LazyIndexedReader,
    ReaderBlock,
    SourceProvenance,
    require_reader_metadata,
)
from mf.data.ultrafineweb import UltraFineWebWindow, UltraFineWebWindowReader

_STATE_KEYS = frozenset(
    {
        "version",
        "signature",
        "signature_hash",
        "rank",
        "world_size",
        "run_seed",
        "shard_order_seed",
        "source_counters",
        "source_cycles",
        "consumed_batches",
        "next_batch_index",
        "pending_blocks",
        "cursor_hash",
    }
)
_PENDING_KEYS = frozenset(
    {
        "cycle",
        "shard_id",
        "sample_index",
        "document_id",
        "block_index",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class DataShard:
    shard_id: str
    reader: LazyIndexedReader[object]
    length: int
    provenance: SourceProvenance
    fingerprint: str | None

    def __init__(self, shard_id: str, reader: LazyIndexedReader[object]) -> None:
        if not isinstance(shard_id, str) or not shard_id:
            raise ValueError("shard_id must be a non-empty string")
        provenance, fingerprint = require_reader_metadata(reader)
        length = len(reader)
        if length <= 0:
            raise ValueError(f"data shard {shard_id!r} must be non-empty")
        object.__setattr__(self, "shard_id", shard_id)
        object.__setattr__(self, "reader", reader)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "fingerprint", fingerprint)


@dataclass(frozen=True, slots=True)
class SampleIdentity:
    source_name: str
    shard_id: str
    sample_index: int
    cycle: int
    document_id: str | None = None
    block_index: int | None = None


@dataclass(frozen=True, slots=True)
class StreamSample:
    identity: SampleIdentity
    value: object = field(compare=False)


@dataclass(frozen=True, slots=True)
class _SourceLayout:
    shards: tuple[DataShard, ...]
    total_size: int


@dataclass(frozen=True, slots=True)
class _CycleLayout:
    cycle: int
    order: tuple[int, ...]
    boundaries: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Location:
    cycle: int
    shard: DataShard
    sample_index: int


@dataclass(frozen=True, slots=True)
class _PendingBlock:
    cycle: int
    shard_id: str
    sample_index: int
    document_id: str
    block_index: int

    def as_dict(self) -> dict[str, object]:
        return {
            "cycle": self.cycle,
            "shard_id": self.shard_id,
            "sample_index": self.sample_index,
            "document_id": self.document_id,
            "block_index": self.block_index,
        }


def text_source_for_bucket(config: MFConfig, bucket: int) -> str:
    normalized_data_config(config)
    if type(bucket) is not int or not 0 <= bucket < 10_000_000:
        raise ValueError("text source bucket must be an integer in [0, 10000000)")
    unit_value = (bucket + 0.5) / 10_000_000
    cumulative = 0.0
    source_names = configured_text_source_names(config)
    for source_name in source_names:
        source = getattr(config.data.text, source_name)
        cumulative += float(source.weight)
        if unit_value <= cumulative:
            return source_name
    return source_names[-1]


def select_text_source(
    *,
    config: MFConfig,
    global_sample_index: int,
) -> str:
    if type(global_sample_index) is not int or global_sample_index < 0:
        raise ValueError("global_sample_index must be a non-negative integer")
    bucket = stable_hash(config.run.seed, global_sample_index, "text-source") % 10_000_000
    return text_source_for_bucket(config, bucket)


class StatefulMixedDataStream:
    def __init__(
        self,
        *,
        config: MFConfig,
        sources: Mapping[str, Sequence[DataShard]],
        rank: int,
    ) -> None:
        normalized_config = normalized_data_config(config)
        world_size = config.distributed.world_size
        if type(world_size) is not int or world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("rank must be an integer in [0, world_size)")
        source_names = configured_source_names(config)
        if set(sources) != set(source_names):
            raise ValueError(f"source names must be exactly {list(source_names)}")

        source_layouts: dict[str, _SourceLayout] = {}
        for source_name in source_names:
            shards = tuple(sources[source_name])
            if not shards:
                raise ValueError(f"source {source_name!r} must contain at least one shard")
            if any(not isinstance(shard, DataShard) for shard in shards):
                raise TypeError(f"source {source_name!r} must contain only DataShard objects")
            shard_ids = [shard.shard_id for shard in shards]
            if len(set(shard_ids)) != len(shard_ids):
                raise ValueError(f"source {source_name!r} has duplicate shard ids")

            expected_kind, expected_split, expected_identity = expected_source(config, source_name)
            for shard in shards:
                provenance = shard.provenance
                if (
                    provenance.source_kind != expected_kind
                    or provenance.official_split != expected_split
                    or provenance.source_identity != expected_identity
                ):
                    raise ValueError(
                        f"source {source_name!r} shard {shard.shard_id!r} "
                        "does not match its configured source identity"
                    )
                if expected_kind == "ultrafineweb" and not isinstance(
                    shard.reader, UltraFineWebWindowReader
                ):
                    raise TypeError(
                        f"UltraFineWeb source {source_name!r} must expose indexed window reads"
                    )
            source_layouts[source_name] = _SourceLayout(
                shards=shards,
                total_size=sum(shard.length for shard in shards),
            )

        self._sources = source_layouts
        self._source_names = source_names
        self._rank = rank
        self._world_size = world_size
        self._run_seed = config.run.seed
        self._shard_order_seed = stable_hash(self._run_seed, "shard-order")
        self._source_counters = dict.fromkeys(self._source_names, 0)
        self._pending_blocks: dict[str, _PendingBlock] = {}
        self._consumed_batches = 0
        self._active_cycle_layouts: dict[str, _CycleLayout] = {}

        signature_payload = {
            "normalized_data_config": normalized_config,
            "rank": rank,
            "world_size": world_size,
            "run_seed": self._run_seed,
            "shard_order_seed": self._shard_order_seed,
            "sources": [
                self._source_signature(source_name) for source_name in sorted(self._source_names)
            ],
        }
        self._signature = _canonical_json(signature_payload)
        self._signature_hash = _sha256(self._signature)

    @classmethod
    def from_config(
        cls,
        *,
        config: MFConfig,
        sources: Mapping[str, Sequence[DataShard]],
        rank: int,
    ) -> StatefulMixedDataStream:
        return cls(
            config=config,
            sources=sources,
            rank=rank,
        )

    def _source_signature(self, source_name: str) -> dict[str, object]:
        source = self._sources[source_name]
        first = source.shards[0].provenance
        return {
            "name": source_name,
            "source_kind": first.source_kind,
            "official_split": first.official_split,
            "source_identity": first.source_identity,
            "shards": [
                {
                    "shard_id": shard.shard_id,
                    "length": shard.length,
                    "external_fingerprint": shard.provenance.external_fingerprint,
                    "fingerprint": shard.fingerprint,
                }
                for shard in source.shards
            ],
        }

    def _next_cycle(self, source_name: str, counter: int) -> int:
        source_position = counter * self._world_size + self._rank
        return source_position // self._sources[source_name].total_size

    def _cycle_layout(self, source_name: str, cycle: int) -> _CycleLayout:
        cached = self._active_cycle_layouts.get(source_name)
        if cached is not None and cached.cycle == cycle:
            return cached

        source = self._sources[source_name]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_hash(self._shard_order_seed, source_name, cycle))
        order = tuple(torch.randperm(len(source.shards), generator=generator).tolist())
        boundaries = tuple(accumulate(source.shards[index].length for index in order))
        layout = _CycleLayout(cycle=cycle, order=order, boundaries=boundaries)
        self._active_cycle_layouts[source_name] = layout
        return layout

    def _locate(self, source_name: str, counter: int) -> _Location:
        source = self._sources[source_name]
        source_position = counter * self._world_size + self._rank
        cycle, cycle_offset = divmod(source_position, source.total_size)
        layout = self._cycle_layout(source_name, cycle)
        order_position = bisect_right(layout.boundaries, cycle_offset)
        previous_boundary = 0 if order_position == 0 else layout.boundaries[order_position - 1]
        shard = source.shards[layout.order[order_position]]
        return _Location(
            cycle=cycle,
            shard=shard,
            sample_index=cycle_offset - previous_boundary,
        )

    @staticmethod
    def _validate_reader_block(
        block: ReaderBlock,
        block_index: int,
        provenance: SourceProvenance,
    ) -> None:
        if not isinstance(block, ReaderBlock):
            raise TypeError("block reader must return ReaderBlock")
        if block.block_index != block_index:
            raise ValueError("block reader returned a mismatched block_index")
        if block.provenance != provenance:
            raise ValueError("block provenance does not match its indexed reader")

    def next_sample(self, source_name: str) -> StreamSample:
        if source_name not in self._sources:
            raise KeyError(f"unknown data source {source_name!r}")
        counter = self._source_counters[source_name]
        location = self._locate(source_name, counter)
        reader = location.shard.reader
        pending = self._pending_blocks.get(source_name)

        if isinstance(reader, UltraFineWebWindowReader):
            if pending is not None:
                raise RuntimeError("UltraFineWeb window source has a pending block cursor")
            window = reader.read_window(location.sample_index, cycle=location.cycle)
            if not isinstance(window, UltraFineWebWindow):
                raise TypeError("UltraFineWeb window reader returned an invalid window")
            if window.provenance != location.shard.provenance:
                raise ValueError("window provenance does not match its indexed reader")
            self._source_counters[source_name] = counter + 1
            return StreamSample(
                identity=SampleIdentity(
                    source_name=source_name,
                    shard_id=location.shard.shard_id,
                    sample_index=location.sample_index,
                    cycle=location.cycle,
                    document_id=window.document_id,
                ),
                value=window.value,
            )

        if isinstance(reader, BlockIndexedReader):
            block_index = 0 if pending is None else pending.block_index
            if pending is not None and (
                pending.cycle != location.cycle
                or pending.shard_id != location.shard.shard_id
                or pending.sample_index != location.sample_index
            ):
                raise RuntimeError("pending block cursor does not match the source counter")
            block = reader.read_block(location.sample_index, block_index)
            self._validate_reader_block(
                block,
                block_index,
                location.shard.provenance,
            )
            if pending is not None and block.document_id != pending.document_id:
                raise RuntimeError("pending document identity changed while reading")
            identity = SampleIdentity(
                source_name=source_name,
                shard_id=location.shard.shard_id,
                sample_index=location.sample_index,
                cycle=location.cycle,
                document_id=block.document_id,
                block_index=block.block_index,
            )
            if block.is_final:
                self._source_counters[source_name] = counter + 1
                self._pending_blocks.pop(source_name, None)
            else:
                self._pending_blocks[source_name] = _PendingBlock(
                    cycle=location.cycle,
                    shard_id=location.shard.shard_id,
                    sample_index=location.sample_index,
                    document_id=block.document_id,
                    block_index=block.block_index + 1,
                )
            return StreamSample(identity=identity, value=block.value)

        if pending is not None:
            raise RuntimeError("non-block source has a pending block cursor")
        value = reader[location.sample_index]
        if getattr(value, "provenance", None) != location.shard.provenance:
            raise ValueError("record provenance does not match its indexed reader")
        self._source_counters[source_name] = counter + 1
        return StreamSample(
            identity=SampleIdentity(
                source_name=source_name,
                shard_id=location.shard.shard_id,
                sample_index=location.sample_index,
                cycle=location.cycle,
            ),
            value=value,
        )

    def next_batch(self, source_names: Sequence[str]) -> tuple[StreamSample, ...]:
        if not source_names:
            raise ValueError("source_names must be non-empty")
        unknown_sources = set(source_names) - self._sources.keys()
        if unknown_sources:
            raise KeyError(f"unknown data sources: {sorted(unknown_sources)}")
        batch = tuple(self.next_sample(source_name) for source_name in source_names)
        self._consumed_batches += 1
        return batch

    def _source_cycles(self, counters: Mapping[str, int]) -> dict[str, int]:
        return {name: self._next_cycle(name, counter) for name, counter in counters.items()}

    def _pending_payload(self) -> dict[str, dict[str, object]]:
        return {name: pending.as_dict() for name, pending in sorted(self._pending_blocks.items())}

    @staticmethod
    def _cursor_payload(
        *,
        source_counters: object,
        source_cycles: object,
        consumed_batches: object,
        next_batch_index: object,
        pending_blocks: object,
    ) -> dict[str, object]:
        return {
            "source_counters": source_counters,
            "source_cycles": source_cycles,
            "consumed_batches": consumed_batches,
            "next_batch_index": next_batch_index,
            "pending_blocks": pending_blocks,
        }

    def state_dict(self) -> dict[str, object]:
        source_counters = dict(self._source_counters)
        source_cycles = self._source_cycles(source_counters)
        pending_blocks = self._pending_payload()
        cursor = self._cursor_payload(
            source_counters=source_counters,
            source_cycles=source_cycles,
            consumed_batches=self._consumed_batches,
            next_batch_index=self._consumed_batches,
            pending_blocks=pending_blocks,
        )
        return {
            "version": 2,
            "signature": self._signature,
            "signature_hash": self._signature_hash,
            "rank": self._rank,
            "world_size": self._world_size,
            "run_seed": self._run_seed,
            "shard_order_seed": self._shard_order_seed,
            **cursor,
            "cursor_hash": _sha256(_canonical_json(cursor)),
        }

    def _validate_pending_blocks(
        self,
        pending_blocks: object,
        counters: Mapping[str, int],
    ) -> dict[str, _PendingBlock]:
        if not isinstance(pending_blocks, Mapping):
            raise ValueError("pending block state must be a mapping")
        if not set(pending_blocks).issubset(self._sources):
            raise ValueError("pending block state contains an unknown source")

        validated: dict[str, _PendingBlock] = {}
        for source_name, raw_pending in pending_blocks.items():
            if not isinstance(source_name, str) or not isinstance(raw_pending, Mapping):
                raise ValueError("pending block cursor is malformed")
            if set(raw_pending) != _PENDING_KEYS:
                raise ValueError("pending block cursor fields are malformed")
            cycle = raw_pending.get("cycle")
            shard_id = raw_pending.get("shard_id")
            sample_index = raw_pending.get("sample_index")
            document_id = raw_pending.get("document_id")
            block_index = raw_pending.get("block_index")
            if (
                type(cycle) is not int
                or cycle < 0
                or not isinstance(shard_id, str)
                or not shard_id
                or type(sample_index) is not int
                or sample_index < 0
                or not isinstance(document_id, str)
                or not document_id
                or type(block_index) is not int
                or block_index <= 0
            ):
                raise ValueError("pending block cursor values are malformed")

            location = self._locate(source_name, counters[source_name])
            reader = location.shard.reader
            if not isinstance(reader, BlockIndexedReader):
                raise ValueError("pending block cursor refers to a non-block source")
            if (
                cycle != location.cycle
                or shard_id != location.shard.shard_id
                or sample_index != location.sample_index
            ):
                raise ValueError("pending block cursor does not match its source counter")
            try:
                previous = reader.read_block(location.sample_index, block_index - 1)
                current = reader.read_block(location.sample_index, block_index)
                self._validate_reader_block(
                    previous,
                    block_index - 1,
                    location.shard.provenance,
                )
                self._validate_reader_block(
                    current,
                    block_index,
                    location.shard.provenance,
                )
            except (IndexError, TypeError, ValueError) as error:
                raise ValueError("pending block cursor is outside its document") from error
            if (
                previous.is_final
                or previous.document_id != document_id
                or current.document_id != document_id
            ):
                raise ValueError("pending block document identity is inconsistent")
            validated[source_name] = _PendingBlock(
                cycle=cycle,
                shard_id=shard_id,
                sample_index=sample_index,
                document_id=document_id,
                block_index=block_index,
            )
        return validated

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != _STATE_KEYS:
            raise ValueError("data stream state fields are malformed")
        version = state.get("version")
        if type(version) is not int or version != 2:
            raise ValueError("unsupported data stream state version")
        checkpoint_rank = state.get("rank")
        if type(checkpoint_rank) is not int or checkpoint_rank != self._rank:
            raise ValueError("data stream rank does not match checkpoint")
        checkpoint_world_size = state.get("world_size")
        if type(checkpoint_world_size) is not int or checkpoint_world_size != self._world_size:
            raise ValueError("data stream world_size does not match checkpoint")
        checkpoint_seed = state.get("run_seed")
        if type(checkpoint_seed) is not int or checkpoint_seed != self._run_seed:
            raise ValueError("data stream run seed does not match checkpoint")
        checkpoint_shard_order_seed = state.get("shard_order_seed")
        if (
            type(checkpoint_shard_order_seed) is not int
            or checkpoint_shard_order_seed != self._shard_order_seed
        ):
            raise ValueError("data stream shard order seed does not match checkpoint")

        signature = state.get("signature")
        signature_hash = state.get("signature_hash")
        if not isinstance(signature, str) or not isinstance(signature_hash, str):
            raise ValueError("data stream signature is malformed")
        if _sha256(signature) != signature_hash:
            raise ValueError("data stream signature hash is corrupt")
        if signature != self._signature or signature_hash != self._signature_hash:
            raise ValueError("data stream signature does not match configured sources")

        cursor = self._cursor_payload(
            source_counters=state.get("source_counters"),
            source_cycles=state.get("source_cycles"),
            consumed_batches=state.get("consumed_batches"),
            next_batch_index=state.get("next_batch_index"),
            pending_blocks=state.get("pending_blocks"),
        )
        cursor_hash = state.get("cursor_hash")
        try:
            expected_cursor_hash = _sha256(_canonical_json(cursor))
        except (TypeError, ValueError) as error:
            raise ValueError("data stream cursor state is not canonical JSON") from error
        if not isinstance(cursor_hash, str) or cursor_hash != expected_cursor_hash:
            raise ValueError("data stream cursor hash is corrupt")

        source_counters = state.get("source_counters")
        if not isinstance(source_counters, Mapping) or set(source_counters) != set(
            self._source_names
        ):
            raise ValueError("source counters do not match configured source names")
        if any(type(value) is not int or value < 0 for value in source_counters.values()):
            raise ValueError("source counters must be non-negative integers")
        counters = {name: cast(int, source_counters[name]) for name in self._source_names}

        source_cycles = state.get("source_cycles")
        expected_cycles = self._source_cycles(counters)
        if (
            not isinstance(source_cycles, Mapping)
            or set(source_cycles) != set(self._source_names)
            or any(type(value) is not int or value < 0 for value in source_cycles.values())
            or dict(source_cycles) != expected_cycles
        ):
            raise ValueError("source cycles do not match source counters")

        consumed_batches = state.get("consumed_batches")
        next_batch_index = state.get("next_batch_index")
        if (
            type(consumed_batches) is not int
            or consumed_batches < 0
            or type(next_batch_index) is not int
            or next_batch_index != consumed_batches
        ):
            raise ValueError("local consumed/next batch index is malformed")

        pending = self._validate_pending_blocks(state.get("pending_blocks"), counters)

        self._source_counters = counters
        self._pending_blocks = pending
        self._consumed_batches = consumed_batches
        self._active_cycle_layouts.clear()
