"""MF inference: load a checkpoint and run multimodal generation tasks."""

from __future__ import annotations

from mf.inference.bundle import InferenceBundle, load_bundle
from mf.inference.executor import BlockCache, BlockGenerationSession, LayerKVCache
from mf.inference.loading import read_checkpoint_config
from mf.inference.pipeline import MFPipeline
from mf.inference.protocol import (
    GenerationConfig,
    checkpoint_sampler_config,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageToTextRequest,
    InferenceRequest,
    InferenceResult,
    InferenceTask,
    TextGenerationRequest,
    TextGenerationResult,
    TextToImageRequest,
    TextToTextRequest,
    PhysicalGenerationRequest,
)
from mf.registries import GENERATION_REGISTRY, register_generation

__all__ = [
    "BlockCache",
    "BlockGenerationSession",
    "GenerationConfig",
    "GENERATION_REGISTRY",
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "ImageToTextRequest",
    "PhysicalGenerationRequest",
    "InferenceBundle",
    "InferenceRequest",
    "InferenceResult",
    "InferenceTask",
    "LayerKVCache",
    "TextGenerationRequest",
    "TextGenerationResult",
    "TextToImageRequest",
    "TextToTextRequest",
    "MFPipeline",
    "checkpoint_sampler_config",
    "load_bundle",
    "read_checkpoint_config",
    "register_generation",
]
