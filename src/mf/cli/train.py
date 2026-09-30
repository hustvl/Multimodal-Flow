from __future__ import annotations

import argparse
import importlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch.distributed as dist
import yaml

from mf.extensions import freeze_extensions, load_extensions


def _parser(*, prog: str = "mf train") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, description="Train Multimodal Flow."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="override a configuration value; repeat as needed",
    )
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume", type=Path)
    checkpoint.add_argument(
        "--init-from",
        type=Path,
        help="initialize from checkpoint EMA weights without restoring training state",
    )
    parser.add_argument(
        "--data-factory",
        metavar="MODULE:FUNCTION",
        help="batch-fetcher factory for custom training data",
    )
    parser.add_argument(
        "--extension",
        dest="extensions",
        action="append",
        default=[],
        metavar="MODULE",
        help="import an extension module before loading the configuration; repeatable",
    )
    return parser


def _load_data_factory(spec: str) -> Callable[..., object]:
    module_name, separator, attribute_name = spec.rpartition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("data factory must use MODULE:FUNCTION syntax")
    factory = getattr(importlib.import_module(module_name), attribute_name, None)
    if not callable(factory):
        raise TypeError(f"data factory is not callable: {spec}")
    return factory


def _rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def main(argv: Sequence[str] | None = None, *, prog: str = "mf train") -> int:
    args = _parser(prog=prog).parse_args(argv)
    # Registration must happen before schema import: validation and several
    # training modules snapshot the available tasks/codecs at import time.
    load_extensions(args.extensions)
    if args.data_factory is not None:
        args.data_factory_callable = _load_data_factory(args.data_factory)
    importlib.import_module("mf.codecs.factory")

    from mf.application import build_train_runtime
    from mf.config.fingerprint import full_config_hash
    from mf.config.loader import load_config, resolve_config_path
    from mf.storage import durable_write_text
    from mf.training.run_artifacts import finalize_training_run

    args.config = resolve_config_path(args.config)
    config = load_config(args.config, args.overrides)
    freeze_extensions()
    args.config = args.config.resolve(strict=True)
    if args.data_factory is None:
        args.data_factory_callable = None
    if _rank_zero():
        durable_write_text(
            Path(config.run.output_dir) / "config.resolved.yaml",
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        )
        print(
            json.dumps(
                {
                    "config": str(args.config),
                    "config_hash": full_config_hash(config),
                    "output_dir": config.run.output_dir,
                    "resume": str(args.resume) if args.resume else None,
                    "init_from": str(args.init_from) if args.init_from else None,
                    "data_factory": args.data_factory,
                },
                sort_keys=True,
            )
        )
    try:
        runtime = build_train_runtime(config=config, args=args)
        state = runtime.fit(resume_from=args.resume)
        if args.resume is not None:
            saved_step = json.loads(
                (args.resume / "run_manifest.json").read_text(encoding="utf-8")
            )["step"]
            if state.global_step == saved_step:
                if _rank_zero():
                    print(json.dumps({"result": asdict(state), "new_steps": 0}))
                return 0
        finalize_training_run(config, state, runtime.distributed)
        if _rank_zero():
            result = asdict(state) if is_dataclass(state) else state
            print(json.dumps({"result": result}, sort_keys=True, default=str))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    return 0


__all__ = ["main"]
