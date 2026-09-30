"""Resolved text semantics shared by training, evaluation, and inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResolvedTextContract:
    """Tokenizer-independent text boundary and latent semantics."""

    eos_token_id: int
    pad_token_id: int
    latent_dim: int
    max_length: int
    tokenizer_revision: str

    def validate(self, *, vocab_size: int | None = None) -> "ResolvedTextContract":
        for name, value in (
            ("eos_token_id", self.eos_token_id),
            ("pad_token_id", self.pad_token_id),
            ("latent_dim", self.latent_dim),
            ("max_length", self.max_length),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.latent_dim == 0 or self.max_length == 0:
            raise ValueError("latent_dim and max_length must be positive")
        if not self.tokenizer_revision:
            raise ValueError("tokenizer_revision must be non-empty")
        if vocab_size is not None and (
            type(vocab_size) is not int
            or vocab_size <= max(self.eos_token_id, self.pad_token_id)
        ):
            raise ValueError(
                "text contract token ids must be inside the tokenizer vocabulary"
            )
        return self


def resolve_text_contract(
    *,
    tokenizer: object,
    eos_token_id: int | None,
    pad_token_id: int | None,
    latent_dim: int,
    max_length: int,
    tokenizer_revision: str | None = None,
) -> ResolvedTextContract:
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    tokenizer_pad = getattr(tokenizer, "pad_token_id", None)
    if tokenizer_eos is None or tokenizer_pad is None:
        raise ValueError("tokenizer must define EOS and PAD token ids")
    contract = ResolvedTextContract(
        eos_token_id=int(tokenizer_eos if eos_token_id is None else eos_token_id),
        pad_token_id=int(tokenizer_pad if pad_token_id is None else pad_token_id),
        latent_dim=int(latent_dim),
        max_length=int(max_length),
        tokenizer_revision=str(
            tokenizer_revision
            or getattr(tokenizer, "tokenizer_fingerprint", None)
            or getattr(tokenizer, "name_or_path", None)
            or "runtime-tokenizer"
        ),
    )
    return contract.validate(vocab_size=getattr(tokenizer, "vocab_size", None))
