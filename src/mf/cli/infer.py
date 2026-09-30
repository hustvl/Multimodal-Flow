"""Generate text or images from an MF checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from torchvision.transforms.functional import to_pil_image

from mf.extensions import load_extensions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mf infer")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument(
        "--extension",
        dest="extensions",
        action="append",
        default=[],
        metavar="MODULE",
        help="import an extension module before loading the checkpoint; repeatable",
    )
    tasks = parser.add_subparsers(dest="task", required=True)

    text = tasks.add_parser("text", help="continue a text prompt")
    text.add_argument("--prompt", required=True)
    text.add_argument("--target-length", type=int, default=128)

    caption = tasks.add_parser("caption", help="answer a question about an image")
    caption.add_argument("--image", type=Path, required=True)
    caption.add_argument("--prompt", default="Describe this image.")
    caption.add_argument("--target-length", type=int, default=128)

    image = tasks.add_parser("image", help="generate an image from text")
    image.add_argument("--prompt", required=True)
    image.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    load_extensions(args.extensions)
    from mf.data.images import load_image
    from mf.inference.pipeline import MFPipeline

    pipeline = MFPipeline.from_checkpoint(
        args.checkpoint, device=args.device, weights=args.weights
    )
    if args.task == "text":
        print(
            pipeline.complete_text((args.prompt,), target_length=args.target_length)[0]
        )
    elif args.task == "caption":
        sample = load_image(args.image, pipeline.config)
        print(
            pipeline.caption(
                (sample,), prompt=args.prompt, target_length=args.target_length
            )[0]
        )
    else:
        generated = pipeline.generate_image((args.prompt,))[0]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        to_pil_image(((generated.detach().cpu().float() + 1) / 2).clamp(0, 1)).save(
            args.output
        )
        print(args.output)
    return 0


__all__ = ["main"]
