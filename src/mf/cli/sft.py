"""Supervised fine-tuning command for Multimodal Flow."""

from collections.abc import Sequence

from mf.cli.train import main as _train_main
from mf.extensions import load_extensions


def main(argv: Sequence[str] | None = None) -> int:
    load_extensions(("mf.recipes.public_sft",))
    return _train_main(argv, prog="mf sft")

__all__ = ["main"]
