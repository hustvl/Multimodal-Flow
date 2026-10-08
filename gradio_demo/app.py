"""Gradio demo for MF-1 (Multimodal Flow): entry point.

Text to image (with a live denoising preview), image to text, and text continuation.
Use --mock (or MF_MOCK=1) to preview the UI without a GPU or model weights.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mf_demo.backend import MFBackend, MockBackend  # noqa: E402
from mf_demo.ui import build_ui, launch_kwargs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="UI preview without a model")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = parser.parse_args()

    mock = args.mock or os.environ.get("MF_MOCK") == "1"
    backend = MockBackend() if mock else MFBackend()
    demo = build_ui(backend)
    demo.queue(max_size=16, default_concurrency_limit=8).launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share, **launch_kwargs()
    )


if __name__ == "__main__":
    main()
