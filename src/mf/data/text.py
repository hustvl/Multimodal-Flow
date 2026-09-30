from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class TextTokenizer(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None
    tokenizer_fingerprint: str
    tokenizer_path: str
    tokenizer_version: str

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


DEFAULT_TEXT_TOKENS = 256
DEFAULT_I2T_PROMPT_TOKENS = 32


@dataclass(frozen=True)
class TokenizedTextBlock:
    token_ids: Tensor
    content_mask: Tensor
    target_mask: Tensor


@dataclass(frozen=True)
class TokenizedConditionBlock:
    token_ids: Tensor
    content_mask: Tensor


def _special_token_ids(tokenizer: TextTokenizer) -> tuple[int, int]:
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if eos_token_id is None or pad_token_id is None:
        raise ValueError("tokenizer must define eos_token_id and pad_token_id")
    if eos_token_id == pad_token_id:
        raise ValueError("eos_token_id and pad_token_id must differ")
    return eos_token_id, pad_token_id


def tokenizer_resume_signature(tokenizer: TextTokenizer) -> dict[str, int | str]:
    eos_token_id, pad_token_id = _special_token_ids(tokenizer)
    identity = {
        "tokenizer_fingerprint": getattr(tokenizer, "tokenizer_fingerprint", None),
        "tokenizer_path": getattr(tokenizer, "tokenizer_path", None),
        "tokenizer_version": getattr(tokenizer, "tokenizer_version", None),
    }
    for name, value in identity.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"tokenizer must define non-empty {name}")
    return {
        **identity,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "tokenization_format": "mf_text_blocks",
        "tokenization_version": 2,
    }


def _content_ids(
    tokenizer: TextTokenizer,
    text: str,
    eos_token_id: int,
    pad_token_id: int,
) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return [token_id for token_id in encoded if token_id not in (eos_token_id, pad_token_id)]


def _make_block(
    active_ids: list[int],
    pad_token_id: int,
    *,
    text_tokens: int,
) -> TokenizedTextBlock:
    if type(text_tokens) is not int or text_tokens <= 0:
        raise ValueError("text_tokens must be a positive integer")
    if len(active_ids) > text_tokens:
        raise ValueError("active text tokens exceed the configured sequence length")
    token_ids = torch.full((text_tokens,), pad_token_id, dtype=torch.long)
    token_ids[: len(active_ids)] = torch.tensor(active_ids, dtype=torch.long)
    content_mask = torch.zeros(text_tokens, dtype=torch.bool)
    content_mask[: len(active_ids)] = True
    return TokenizedTextBlock(
        token_ids=token_ids,
        content_mask=content_mask,
        target_mask=content_mask.clone(),
    )


def tokenize_caption(
    tokenizer: TextTokenizer,
    text: str,
    *,
    text_tokens: int = DEFAULT_TEXT_TOKENS,
    content_tokens: int | None = None,
    eos_fill_block_size: int | None = None,
) -> TokenizedTextBlock:
    if content_tokens is None:
        content_tokens = text_tokens
    if type(content_tokens) is not int or not 0 < content_tokens <= text_tokens:
        raise ValueError("content_tokens must be a positive integer no greater than text_tokens")
    if eos_fill_block_size is not None:
        if type(eos_fill_block_size) is not int or eos_fill_block_size <= 0:
            raise ValueError("eos_fill_block_size must be a positive integer")
        if text_tokens % eos_fill_block_size != 0:
            raise ValueError("text_tokens must be divisible by eos_fill_block_size")
    eos_token_id, pad_token_id = _special_token_ids(tokenizer)
    content_ids = _content_ids(tokenizer, text, eos_token_id, pad_token_id)
    has_terminal_capacity = len(content_ids) < content_tokens
    active_ids = content_ids[:content_tokens]
    if has_terminal_capacity:
        active_ids.append(eos_token_id)
    if eos_fill_block_size is not None and has_terminal_capacity:
        active_ids.extend([eos_token_id] * ((-len(active_ids)) % eos_fill_block_size))
    return _make_block(active_ids, pad_token_id, text_tokens=text_tokens)


def tokenize_condition(
    tokenizer: TextTokenizer,
    text: str,
    *,
    text_tokens: int = DEFAULT_I2T_PROMPT_TOKENS,
) -> TokenizedConditionBlock:
    """Tokenize a clean condition without appending target EOS."""

    if type(text_tokens) is not int or text_tokens <= 0:
        raise ValueError("text_tokens must be a positive integer")
    eos_token_id, pad_token_id = _special_token_ids(tokenizer)
    active_ids = _content_ids(tokenizer, text, eos_token_id, pad_token_id)[:text_tokens]
    if not active_ids:
        raise ValueError("condition text must contain at least one token")
    token_ids = torch.full((text_tokens,), pad_token_id, dtype=torch.long)
    token_ids[: len(active_ids)] = torch.tensor(active_ids, dtype=torch.long)
    content_mask = torch.zeros(text_tokens, dtype=torch.bool)
    content_mask[: len(active_ids)] = True
    return TokenizedConditionBlock(
        token_ids=token_ids,
        content_mask=content_mask,
    )


def tokenize_document_ids(
    tokenizer: TextTokenizer,
    text: str,
    *,
    max_chars: int | None = None,
) -> list[int]:
    """Tokenize one hard-pack document with the configured character/EOS policy."""

    eos_token_id, pad_token_id = _special_token_ids(tokenizer)
    if max_chars is not None:
        if type(max_chars) is not int or max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        truncated = len(text) > max_chars
        text = text[:max_chars]
    else:
        truncated = False
    token_ids = _content_ids(tokenizer, text, eos_token_id, pad_token_id)
    return token_ids if truncated else [*token_ids, eos_token_id]


_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？])|(?<=[.!?])\s+")


def split_complete_text_units(text: str) -> tuple[str, ...]:
    """Split a document without discarding punctuation or non-empty text."""

    units: list[str] = []
    for paragraph in re.split(r"[\r\n]+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        units.extend(
            sentence
            for sentence in (part.strip() for part in _SENTENCE_BOUNDARY.split(paragraph))
            if sentence
        )
    return tuple(units)


def tokenize_block_aligned_document_units(
    tokenizer: TextTokenizer,
    text: str,
    *,
    text_tokens: int,
    block_size: int,
) -> tuple[list[int], ...]:
    """Tokenize sentence units with a supervised EOS in every terminal block."""

    if type(text_tokens) is not int or text_tokens <= 0:
        raise ValueError("text_tokens must be a positive integer")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if text_tokens % block_size != 0:
        raise ValueError("text_tokens must be divisible by block_size")
    eos_token_id, pad_token_id = _special_token_ids(tokenizer)
    units: list[list[int]] = []
    for unit in split_complete_text_units(text):
        content_ids = _content_ids(tokenizer, unit, eos_token_id, pad_token_id)
        if not content_ids:
            continue
        active_ids = [*content_ids[: text_tokens - 1], eos_token_id]
        active_ids.extend([eos_token_id] * ((-len(active_ids)) % block_size))
        units.append(active_ids)
    return tuple(units)


def tokenize_block_aligned_record_unit(
    tokenizer: TextTokenizer,
    text: str,
    *,
    text_tokens: int,
    block_size: int,
) -> list[int]:
    """Tokenize one source record with one terminal EOS and block padding."""

    if type(text_tokens) is not int or text_tokens <= 0:
        raise ValueError("text_tokens must be a positive integer")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if text_tokens % block_size != 0:
        raise ValueError("text_tokens must be divisible by block_size")
    eos_token_id, pad_token_id = _special_token_ids(tokenizer)
    content_ids = _content_ids(tokenizer, text, eos_token_id, pad_token_id)
    if not content_ids:
        return []
    has_terminal_capacity = len(content_ids) < text_tokens
    active_ids = content_ids[:text_tokens]
    if has_terminal_capacity:
        active_ids.append(eos_token_id)
        active_ids.extend([eos_token_id] * ((-len(active_ids)) % block_size))
    return active_ids


def tokenize_block_aligned_document_ids(
    tokenizer: TextTokenizer,
    text: str,
    *,
    text_tokens: int,
    block_size: int,
) -> list[int]:
    """Tokenize one source record and EOS-fill its terminal block."""

    if type(text_tokens) is not int or text_tokens <= 0:
        raise ValueError("text_tokens must be a positive integer")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if text_tokens % block_size != 0:
        raise ValueError("text_tokens must be divisible by block_size")
    eos_token_id, pad_token_id = _special_token_ids(tokenizer)
    content_ids = _content_ids(tokenizer, text, eos_token_id, pad_token_id)
    if not content_ids:
        return []
    active_ids = [*content_ids, eos_token_id]
    active_ids.extend([eos_token_id] * ((-len(active_ids)) % block_size))
    return active_ids


def make_hard_packed_block(
    token_ids: list[int],
    *,
    text_tokens: int,
) -> TokenizedTextBlock:
    """Build one fully active fixed-length block from a continuous token stream."""

    if len(token_ids) != text_tokens:
        raise ValueError("hard-packed token block must exactly match text_tokens")
    tensor = torch.tensor(token_ids, dtype=torch.long)
    mask = torch.ones(text_tokens, dtype=torch.bool)
    return TokenizedTextBlock(token_ids=tensor, content_mask=mask, target_mask=mask.clone())


def make_padded_packed_block(
    token_ids: list[int],
    *,
    pad_token_id: int,
    text_tokens: int,
) -> TokenizedTextBlock:
    """Build a packed block whose unused physical tail is inactive PAD."""

    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise ValueError("pad_token_id must be a non-negative integer")
    return _make_block(token_ids, pad_token_id, text_tokens=text_tokens)


def stable_text_window_start(
    *,
    run_seed: int,
    source_id: str,
    sample_id: str,
    cycle: int,
    choices: int,
) -> int:
    if choices <= 1:
        return 0
    payload = f"{run_seed}\0{source_id}\0{sample_id}\0{cycle}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % choices


def tokenize_document_window(
    tokenizer: TextTokenizer,
    text: str,
    *,
    run_seed: int,
    source_id: str,
    sample_id: str,
    cycle: int,
    text_tokens: int = DEFAULT_TEXT_TOKENS,
) -> TokenizedTextBlock:
    eos_token_id, pad_token_id = _special_token_ids(tokenizer)
    content_ids = _content_ids(tokenizer, text, eos_token_id, pad_token_id)
    content_limit = text_tokens - 1
    start = stable_text_window_start(
        run_seed=run_seed,
        source_id=source_id,
        sample_id=sample_id,
        cycle=cycle,
        choices=len(content_ids) - content_limit + 1,
    )
    active_ids = [*content_ids[start : start + content_limit], eos_token_id]
    return _make_block(active_ids, pad_token_id, text_tokens=text_tokens)
