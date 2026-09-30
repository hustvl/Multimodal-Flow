"""Image preprocessing shared by training, inference, and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor

from mf.config.schema import MFConfig

def preprocess_image(image: Image.Image, *, resolution: int, policy: str) -> Tensor:
    if type(resolution) is not int or resolution <= 0:
        raise ValueError("image resolution must be a positive integer")
    if policy not in {"legacy_center_crop_bicubic_v1", "siglip2_resize_bicubic_v1"}:
        raise ValueError(f"unsupported image preprocessing policy: {policy!r}")
    image = ImageOps.exif_transpose(image).convert("RGB")
    if policy == "legacy_center_crop_bicubic_v1":
        width, height = image.size
        size = min(width, height)
        image = image.crop(
            ((width - size) // 2, (height - size) // 2,
             (width + size) // 2, (height + size) // 2)
        )
    image = image.resize((resolution, resolution), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.uint8).copy()
    if array.shape != (resolution, resolution, 3):
        raise ValueError(f"unexpected decoded image shape {array.shape}")
    return torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div_(255.0)


def load_image(path: str | Path, config: MFConfig) -> Tensor:
    with Image.open(path) as source:
        return preprocess_image(
            source,
            resolution=config.codecs.vision.encoder_input_resolution,
            policy=getattr(
                config.codecs.vision,
                "image_preprocessing",
                "legacy_center_crop_bicubic_v1",
            ),
        )
