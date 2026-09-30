"""Small public command line interface for Multimodal Flow."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from mf import __version__


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "train": "mf.cli.train",
        "sft": "mf.cli.sft",
        "infer": "mf.cli.infer",
        "evaluate": "mf.cli.evaluate",
    }
    parser = argparse.ArgumentParser(prog="mf")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("command", nargs="?", choices=tuple(commands))
    if arguments[:1] == ["--version"]:
        parser.parse_args(arguments)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command = arguments.pop(0)
    module = commands.get(command)
    if module is None:
        parser.error(f"unknown command: {command}")
    return importlib.import_module(module).main(arguments)


__all__ = ["main"]
