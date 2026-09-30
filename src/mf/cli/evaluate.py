"""Basic image-question answering evaluation with saved model responses."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mf.extensions import load_extensions


def _normalized(answer: str) -> str:
    return " ".join(answer.casefold().split())


def _questions(item: dict[str, object]) -> list[tuple[str, str, str]]:
    image = item.get("image")
    if not isinstance(image, str) or not image:
        return []
    qa = item.get("qa")
    if isinstance(qa, list):
        return [
            (image, str(pair["question"]), str(pair["answer"]))
            for pair in qa
            if isinstance(pair, dict) and "question" in pair and "answer" in pair
        ]
    if "question" in item and "answer" in item:
        return [(image, str(item["question"]), str(item["answer"]))]
    return []


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mf evaluate")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input", type=Path, required=True, help="JSONL with image, question, answer"
    )
    parser.add_argument("--output", type=Path, required=True, help="prediction JSONL")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-length", type=int, default=32)
    parser.add_argument(
        "--extension",
        dest="extensions",
        action="append",
        default=[],
        metavar="MODULE",
        help="import an extension module before loading the checkpoint; repeatable",
    )
    args = parser.parse_args(argv)
    load_extensions(args.extensions)
    from mf.data.images import load_image
    from mf.inference.pipeline import MFPipeline


    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must be different files")

    pipeline = MFPipeline.from_checkpoint(args.checkpoint, device=args.device)
    image_root = args.image_root or args.input.parent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = total = 0
    with (
        args.input.open(encoding="utf-8") as source,
        args.output.open("w", encoding="utf-8") as destination,
    ):
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            for image, question, answer in _questions(item):
                prediction = pipeline.caption(
                    (load_image(image_root / image, pipeline.config),),
                    prompt=question,
                    target_length=args.target_length,
                )[0]
                match = _normalized(prediction) == _normalized(answer)
                destination.write(
                    json.dumps(
                        {
                            "id": item.get("id"),
                            "image": image,
                            "question": question,
                            "answer": answer,
                            "prediction": prediction,
                            "exact_match": match,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                correct += int(match)
                total += 1
    print(
        json.dumps({"samples": total, "exact_match": correct / total if total else 0.0})
    )
    return 0


__all__ = ["main"]
