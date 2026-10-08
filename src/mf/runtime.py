from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_hooks
from torch.nn.parallel import DistributedDataParallel

from mf.application import load_latent_stats_registry
from mf.codecs.factory import build_text_encoder, build_vision_encoder
from mf.codecs.online import OnlineBatchEncoder
from mf.codecs.text_decoder import (
    LatentTextDecoder,
    load_text_decoder_checkpoint,
    prewarm_compiled_text_decoder,
)
from mf.config.fingerprint import full_config_hash
from mf.config.schema import (
    MFConfig,
    RawPixelCodecConfig,
    TensorStatsConfig,
    derive_task_slots,
)
from mf.contracts.evaluation import EvaluationIdentity
from mf.contracts.geometry import GeometryContract
from mf.contracts.text import resolve_text_contract
from mf.data.runtime import (
    BatchFetcher,
    build_batch_fetcher,
)
from mf.distributed.context import DistributedContext
from mf.latents.stats import LatentStatsRegistry
from mf.modeling.attention import prewarm_compiled_flex_attention
from mf.modeling.block import prewarm_compiled_packed_block
from mf.modeling.model import MFModel
from mf.training.checkpoint import (
    CheckpointBindings,
    CheckpointManager,
    RunMetadata,
)
from mf.training.ema import ExponentialMovingAverage
from mf.training.optimizers import OptimizerBundle, build_optimizers
from mf.training.schedulers import SchedulerBundle, build_schedulers
from mf.training.task_builder import TaskBuilder, build_cpu_batch_preparer
from mf.training.trainer import (
    Evaluator,
    Trainer,
    build_metric_sink,
)


class RuntimeTokenizer:
    """Local-only tokenizer adapter with a stable checkpoint identity."""

    def __init__(self, model_path: str | Path) -> None:
        from transformers import AutoTokenizer

        path = Path(model_path).expanduser()
        local = path.exists()
        source = path.resolve(strict=True) if local else str(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            source,
            local_files_only=local,
        )
        if self._tokenizer.eos_token_id is None or self._tokenizer.pad_token_id is None:
            raise ValueError("configured tokenizer must define EOS and PAD token ids")
        self.eos_token_id = int(self._tokenizer.eos_token_id)
        self.pad_token_id = int(self._tokenizer.pad_token_id)
        self.vocab_size = int(
            getattr(self._tokenizer, "vocab_size", len(self._tokenizer))
        )
        self.tokenizer_path = str(source)
        self.tokenizer_fingerprint = (
            _hash_files(_tokenizer_files(source))
            if local
            else hashlib.sha256(str(source).encode("utf-8")).hexdigest()
        )
        self.tokenizer_version = importlib.metadata.version("transformers")

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(
            self._tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                verbose=False,
            )
        )

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        return str(
            self._tokenizer.decode(
                list(token_ids),
                skip_special_tokens=skip_special_tokens,
            )
        )


@dataclass(frozen=True, slots=True)
class DistributedRuntime:
    context: DistributedContext
    device: torch.device
    local_rank: int


@dataclass(slots=True)
class _RuntimeBundle:
    config: MFConfig
    distributed: DistributedRuntime
    tokenizer: RuntimeTokenizer
    stats: LatentStatsRegistry
    model: nn.Module
    training_model: nn.Module
    text_decoder: nn.Module
    training_text_decoder: nn.Module
    vision_encoder: nn.Module
    text_encoder: nn.Module
    batch_encoder: nn.Module
    batch_fetcher: BatchFetcher
    task_builder: object
    optimizers: OptimizerBundle
    schedulers: SchedulerBundle
    ema: ExponentialMovingAverage
    training_generator: torch.Generator
    evaluation_generator: torch.Generator
    checkpoint_manager: CheckpointManager
    evaluator: Evaluator | None


@dataclass(frozen=True, slots=True)
class TrainableModules:
    model: MFModel
    training_model: nn.Module
    text_decoder: LatentTextDecoder
    training_text_decoder: nn.Module


def _candidate_weight(path: Path) -> Path:
    for name in (
        "model.safetensors",
        "diffusion_pytorch_model.safetensors",
        "pytorch_model.bin",
    ):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"model weights are missing under {path}")


def _tokenizer_files(path: Path) -> tuple[Path, ...]:
    names = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "vocab.json",
        "merges.txt",
    )
    files = tuple(path / name for name in names if (path / name).is_file())
    if not files:
        raise FileNotFoundError(f"tokenizer metadata is missing under {path}")
    return files


def _vision_codec_files(vision: object) -> tuple[Path, ...]:
    if isinstance(vision, RawPixelCodecConfig):
        return ()
    model_path = getattr(vision, "model_path", None)
    files: list[Path] = []
    if model_path is not None:
        root = Path(model_path).expanduser().resolve(strict=True)
        if root.is_file():
            files.append(root)
        else:
            files.extend((root / "config.json", _candidate_weight(root)))
        preprocessor = root / "preprocessor_config.json"
        if preprocessor.is_file():
            files.append(preprocessor)
    decoder_config = getattr(vision, "decoder_config_path", None)
    decoder_checkpoint = getattr(vision, "decoder_checkpoint_path", None)
    if (decoder_config is None) != (decoder_checkpoint is None):
        raise ValueError(
            "vision decoder config and checkpoint must either both be set or both be null"
        )
    if decoder_config is not None and decoder_checkpoint is not None:
        files.extend((Path(decoder_config) / "config.json", Path(decoder_checkpoint)))
    return tuple(files)


def _hash_files(paths: Iterable[str | Path]) -> str:
    normalized = tuple(
        sorted({Path(path).expanduser().resolve(strict=True) for path in paths})
    )
    if not normalized:
        raise ValueError("at least one file is required for an asset hash")
    digest = hashlib.sha256()
    for path in normalized:
        if not path.is_file():
            raise FileNotFoundError(f"asset hash requires a regular file: {path}")
        name = str(path).encode("utf-8")
        digest.update(len(name).to_bytes(8, byteorder="big"))
        digest.update(name)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _broadcast_string(value: str | None, context: DistributedContext) -> str:
    values: list[object] = [value]
    if context.world_size > 1:
        dist.broadcast_object_list(values, src=0, group=context.process_group)
    result = values[0]
    if not isinstance(result, str):
        raise RuntimeError("rank zero did not publish a runtime identity")
    return result


def _identity_hashes(
    config: MFConfig, context: DistributedContext
) -> tuple[str, str, str]:
    stats_hash: str | None = None
    codec_hash: str | None = None
    prompt_hash: str | None = None
    if context.rank == 0:
        stats_paths = [config.latent_stats.vision.path]
        text_stats = config.latent_stats.text_normal
        if isinstance(text_stats, TensorStatsConfig):
            stats_paths.append(text_stats.path)
        stats_hash = _hash_files(stats_paths)
        text_root = Path(config.codecs.text.model_path).resolve(strict=True)
        vision_codec_files = list(_vision_codec_files(config.codecs.vision))
        codec_files: list[str | Path] = [
            *vision_codec_files,
            text_root / "config.json",
            _candidate_weight(text_root),
            *_tokenizer_files(text_root),
        ]
        codec_hash = _hash_files(codec_files)
        prompt_hash = hashlib.sha256(b"").hexdigest()
    return (
        _broadcast_string(stats_hash, context),
        _broadcast_string(codec_hash, context),
        _broadcast_string(prompt_hash, context),
    )


def build_evaluation_identity(
    config: MFConfig,
    context: DistributedContext,
) -> EvaluationIdentity:
    stats_hash, codec_hash, prompt_hash = _identity_hashes(config, context)
    sampling = config.evaluation.sampling
    return EvaluationIdentity(
        config_hash=full_config_hash(config),
        stats_hash=stats_hash,
        codec_hash=codec_hash,
        prompt_hash=prompt_hash,
        cfg_scale=float(sampling.cfg_scale),
        num_inference_steps=sampling.num_inference_steps,
        seed=config.run.seed,
        world_size=context.world_size,
        required_suites=(),
        matrix_sha256=None,
        matrix_variants=(),
    )


def initialize_distributed(config: MFConfig) -> DistributedRuntime:
    if not torch.cuda.is_available():
        raise RuntimeError("the standard MF runtime requires CUDA")
    configured_world = config.distributed.world_size
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK is outside the visible CUDA device range")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        environment_world = int(os.environ.get("WORLD_SIZE", "1"))
        if environment_world != configured_world:
            raise RuntimeError(
                f"torchrun WORLD_SIZE={environment_world} does not match config "
                f"world_size={configured_world}"
            )
        if configured_world > 1:
            dist.init_process_group(
                backend="nccl",
                device_id=device,
                timeout=timedelta(
                    seconds=config.distributed.process_group_timeout_seconds
                ),
            )
    if dist.is_initialized():
        context = DistributedContext.from_process_group()
    else:
        context = DistributedContext.single_process()
    if context.world_size != configured_world:
        raise RuntimeError(
            "initialized process group does not match configured world_size"
        )
    return DistributedRuntime(
        context=context,
        device=device,
        local_rank=local_rank,
    )


def _prewarm_flex_attention(
    config: MFConfig,
    runtime: DistributedRuntime,
) -> tuple[float, ...]:
    block_causal = config.flow.text_block_causal
    if not block_causal.flex_prewarm_sequence_lengths:
        return ()

    context = runtime.context
    if context.world_size > 1:
        dist.barrier(
            group=context.process_group,
            device_ids=[runtime.local_rank],
        )
    local_elapsed = prewarm_compiled_flex_attention(
        device=runtime.device,
        dtype=torch.bfloat16
        if config.distributed.amp_dtype == "bf16"
        else torch.float16,
        num_heads=config.model.num_heads,
        head_dim=config.model.head_dim,
        sequence_lengths=block_causal.flex_prewarm_sequence_lengths,
        kernel_block_size=block_causal.flex_kernel_block_size,
        text_block_size=block_causal.block_size,
        flex_backend=block_causal.flex_backend,
    )
    elapsed = torch.tensor(
        local_elapsed,
        device=runtime.device,
        dtype=torch.float64,
    )
    max_elapsed = context.all_reduce_detached_max(elapsed)
    if context.world_size > 1:
        dist.barrier(
            group=context.process_group,
            device_ids=[runtime.local_rank],
        )
    result = tuple(float(value) for value in max_elapsed.tolist())
    if context.rank == 0:
        details = ", ".join(
            f"{length}:{seconds:.2f}s"
            for length, seconds in zip(
                block_causal.flex_prewarm_sequence_lengths,
                result,
                strict=True,
            )
        )
        print(f"Flex Attention prewarm completed ({details})", flush=True)
    return result


def _prewarm_compiled_blocks(
    config: MFConfig,
    runtime: DistributedRuntime,
    model: MFModel,
) -> tuple[float, ...]:
    block_causal = config.flow.text_block_causal
    if (
        not config.model.compile_packed_blocks
        or not block_causal.flex_prewarm_sequence_lengths
    ):
        return ()

    context = runtime.context
    if context.world_size > 1:
        dist.barrier(group=context.process_group, device_ids=[runtime.local_rank])
    weights = config.tasks.weights
    has_vision_task = any(
        weight > 0.0
        for weight in (
            weights.text_to_image,
            weights.image_to_text,
            weights.image_only,
        )
    )
    elapsed = prewarm_compiled_packed_block(
        model.backbone.blocks[0],
        device=runtime.device,
        dtype=torch.bfloat16
        if config.distributed.amp_dtype == "bf16"
        else torch.float16,
        sequence_lengths=block_causal.flex_prewarm_sequence_lengths,
        kernel_block_size=block_causal.flex_kernel_block_size,
        text_block_size=block_causal.block_size,
        vision_token_count=None if has_vision_task else 0,
    )
    elapsed_tensor = torch.tensor(elapsed, device=runtime.device, dtype=torch.float64)
    max_elapsed = context.all_reduce_detached_max(elapsed_tensor)
    if context.world_size > 1:
        dist.barrier(group=context.process_group, device_ids=[runtime.local_rank])
    result = tuple(float(value) for value in max_elapsed.tolist())
    if context.rank == 0:
        details = ", ".join(
            f"{length}:{seconds:.2f}s"
            for length, seconds in zip(
                block_causal.flex_prewarm_sequence_lengths,
                result,
                strict=True,
            )
        )
        print(f"Compiled MF block prewarm completed ({details})", flush=True)
    return result


def _prewarm_compiled_decoder(
    config: MFConfig,
    runtime: DistributedRuntime,
    decoder: LatentTextDecoder,
) -> float:
    block_causal = config.flow.text_block_causal
    if not config.model.compile_packed_blocks:
        return 0.0
    if block_causal.target_encoding != "block_local":
        raise ValueError("compiled text decoder requires block-local text targets")

    context = runtime.context
    if context.world_size > 1:
        dist.barrier(group=context.process_group, device_ids=[runtime.local_rank])
    active_blocks = (
        config.distributed.micro_batch_size_per_rank
        * config.data.text_max_length
        // block_causal.block_size
    )
    local_elapsed = prewarm_compiled_text_decoder(
        decoder,
        device=runtime.device,
        dtype=torch.bfloat16
        if config.distributed.amp_dtype == "bf16"
        else torch.float16,
        block_size=block_causal.block_size,
        active_blocks=active_blocks,
        use_attention_mask=config.model.text_decoder.use_attention_mask,
    )
    elapsed = torch.tensor(local_elapsed, device=runtime.device, dtype=torch.float64)
    max_elapsed = context.all_reduce_detached_max(elapsed)
    if context.world_size > 1:
        dist.barrier(group=context.process_group, device_ids=[runtime.local_rank])
    result = float(max_elapsed.item())
    if context.rank == 0:
        print(f"Compiled text decoder prewarm completed ({result:.2f}s)", flush=True)
    return result


def new_model(config: object, stats: LatentStatsRegistry) -> MFModel:
    model = config.model
    block_causal = config.flow.text_block_causal
    inference_text = getattr(config, "text", None)
    text_tokens = (
        inference_text.max_length
        if inference_text is not None
        else config.data.text_max_length
    )
    return MFModel(
        stats,
        vision_latent_dim=config.codecs.vision.latent_dim,
        vision_tokens=config.codecs.vision.latent_tokens,
        text_latent_dim=config.codecs.text.latent_dim,
        vision_grid_size=tuple(config.codecs.vision.grid_size),
        hidden_size=model.hidden_size,
        depth=model.depth,
        num_heads=model.num_heads,
        head_dim=model.head_dim,
        ffn_hidden_size=model.ffn_hidden_size,
        mrope_section=tuple(model.mrope_section),
        attention_mode=model.attention_mode,
        ffn_mode=model.ffn_mode,
        text_tokens=text_tokens,
        text_input_bottleneck_dim=model.text_input_bottleneck_dim,
        text_input_projection_mode=model.text_input_projection_mode,
        fp32_boundaries=model.fp32_boundaries,
        gradient_checkpointing=getattr(model, "gradient_checkpointing", False),
        compile_packed_blocks=model.compile_packed_blocks,
        sequence_layout=model.sequence_layout,
        image_chunk_conditioning=model.image_chunk_conditioning,
        t2i_chunk_semantics=model.t2i_chunk_semantics,
        block_causal_attention_backend=block_causal.attention_backend,
        block_causal_flex_backend=block_causal.flex_backend,
        block_causal_flex_kernel_block_size=block_causal.flex_kernel_block_size,
        block_causal_flex_sequence_bucket_size=block_causal.flex_sequence_bucket_size,
        block_causal_flex_fixed_sequence_length=block_causal.flex_fixed_sequence_length,
        text_block_size=block_causal.block_size,
    )


def verify_decoder_config(config: MFConfig, decoder: LatentTextDecoder) -> None:
    expected = config.model.text_decoder
    actual = {
        "hidden_size": decoder.hidden_size,
        "depth": decoder.depth,
        "num_heads": decoder.num_heads,
        "head_dim": decoder.head_dim,
        "mlp_ratio": decoder.mlp_ratio,
        "bottleneck_dim": decoder.bottleneck_dim,
        "max_length": decoder.max_length,
        "vocab_size": decoder.vocab_size,
    }
    for name, value in actual.items():
        if value != getattr(expected, name):
            raise ValueError(f"text decoder {name} does not match the resolved config")


def _balanced_plan_uses_all_trainable_branches(config: MFConfig) -> bool:
    if config.tasks.planner == "chunk_token_packed":
        # Token budgets do not guarantee each rank uses every branch in a pack.
        return False
    if config.tasks.planner != "rank_balanced_global":
        return False
    local_batch_count = (
        config.distributed.world_size * config.distributed.gradient_accumulation_steps
    )
    slots = derive_task_slots(
        config.tasks.weights,
        config.distributed.global_batch_size,
    )
    # Every local microbatch must exercise both target heads. Those tasks also
    # carry both modalities, so every modality-specific FFN and decoder branch
    # participates in every backward pass.
    return (
        slots.text_to_image >= local_batch_count
        and slots.image_to_text >= local_batch_count
    )


def wrap_ddp(
    module: nn.Module,
    runtime: DistributedRuntime,
    *,
    find_unused_parameters: bool = True,
    static_graph: bool = False,
    bucket_cap_mb: int = 25,
    gradient_compression: str = "none",
    init_sync: bool = True,
) -> nn.Module:
    if runtime.context.world_size == 1 or not any(
        parameter.requires_grad for parameter in module.parameters()
    ):
        return module
    wrapped = DistributedDataParallel(
        module,
        device_ids=(runtime.local_rank,),
        output_device=runtime.local_rank,
        broadcast_buffers=False,
        init_sync=init_sync,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=True,
        static_graph=static_graph,
        bucket_cap_mb=bucket_cap_mb,
    )
    if gradient_compression == "bf16":
        wrapped.register_comm_hook(
            runtime.context.process_group,
            ddp_hooks.bf16_compress_hook,
        )
    return wrapped


def build_trainable_modules(
    config: MFConfig,
    stats: LatentStatsRegistry,
    runtime: DistributedRuntime,
    *,
    initializer: Callable[[MFModel, LatentTextDecoder], object] | None = None,
) -> TrainableModules:
    # Trainable parameters must be identical before EMA captures its initial shadow.
    torch.manual_seed(config.run.seed)
    torch.cuda.manual_seed(config.run.seed)
    model = new_model(config, stats).to(runtime.device)
    text_decoder = LatentTextDecoder(
        input_dim=config.codecs.text.latent_dim,
        max_length=config.model.text_decoder.max_length,
        fp32_boundaries=config.model.fp32_boundaries,
        compile_forward=config.model.text_decoder.compile_forward,
    ).to(runtime.device)
    verify_decoder_config(config, text_decoder)
    if initializer is not None:
        initializer(model, text_decoder)
    decoder_checkpoint = config.model.text_decoder.checkpoint_path
    if decoder_checkpoint is not None:
        load_text_decoder_checkpoint(text_decoder, decoder_checkpoint)
    if not config.model.text_decoder.trainable:
        text_decoder.requires_grad_(False)
        text_decoder.eval()
    if config.model.compile_packed_blocks:
        _prewarm_compiled_blocks(config, runtime, model)
    _prewarm_compiled_decoder(config, runtime, text_decoder)
    planner_static_graph = _balanced_plan_uses_all_trainable_branches(config)
    static_graph = (
        planner_static_graph
        if config.distributed.ddp_static_graph is None
        else config.distributed.ddp_static_graph
    )
    find_unused_parameters = (
        not planner_static_graph
        if config.distributed.ddp_find_unused_parameters is None
        else config.distributed.ddp_find_unused_parameters
    )
    training_model = wrap_ddp(
        model,
        runtime,
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
        bucket_cap_mb=config.distributed.ddp_bucket_cap_mb,
        gradient_compression=config.distributed.ddp_gradient_compression,
        init_sync=config.distributed.ddp_init_sync,
    )
    training_text_decoder = wrap_ddp(
        text_decoder,
        runtime,
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
        bucket_cap_mb=config.distributed.ddp_bucket_cap_mb,
        gradient_compression=config.distributed.ddp_gradient_compression,
        init_sync=config.distributed.ddp_init_sync,
    )

    rank_seed = config.run.seed + runtime.context.rank
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    return TrainableModules(
        model=model,
        training_model=training_model,
        text_decoder=text_decoder,
        training_text_decoder=training_text_decoder,
    )


def _git_metadata() -> tuple[str, bool]:
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ("git", "-C", str(repo), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        dirty = bool(
            subprocess.run(
                ("git", "-C", str(repo), "status", "--porcelain"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unversioned", True
    commit = commit.strip()
    return commit, dirty


def _run_metadata(config: MFConfig) -> RunMetadata:
    commit, dirty = _git_metadata()
    packages = ("torch", "transformers", "pyarrow", "pydantic")
    versions = {name: importlib.metadata.version(name) for name in packages}
    for package in ("flash-attn", "flash-attn-4"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return RunMetadata(
        git_commit=commit,
        git_dirty=dirty,
        dependency_versions=versions,
    )


def _build_bundle(
    config: MFConfig,
    *,
    initializer: Callable[[MFModel, LatentTextDecoder], object] | None = None,
    batch_fetcher_factory: Callable[..., BatchFetcher] | None = None,
) -> _RuntimeBundle:
    runtime = initialize_distributed(config)
    rank = runtime.context.rank
    device = runtime.device
    _prewarm_flex_attention(config, runtime)

    tokenizer = RuntimeTokenizer(config.codecs.text.tokenizer_path)
    text_contract = resolve_text_contract(
        tokenizer=tokenizer,
        eos_token_id=config.objective.eos_token_id,
        pad_token_id=config.objective.pad_token_id,
        latent_dim=config.codecs.text.latent_dim,
        max_length=config.codecs.text.max_length,
        tokenizer_revision=config.codecs.text.tokenizer_revision,
    )
    geometry = GeometryContract.from_config(config)
    stats = load_latent_stats_registry(config).to(device)
    trainables = build_trainable_modules(
        config, stats, runtime, initializer=initializer
    )
    model = trainables.model
    text_decoder = trainables.text_decoder
    vision_encoder = build_vision_encoder(config.codecs.vision).to(
        device=device,
        dtype=torch.bfloat16,
    )
    text_encoder = build_text_encoder(config.codecs.text).to(
        device=device,
        dtype=torch.bfloat16,
    )
    block_causal = config.flow.text_block_causal
    batch_encoder = OnlineBatchEncoder(
        vision=vision_encoder,
        text=text_encoder,
        eos_token_id=text_contract.eos_token_id,
        geometry=geometry,
        target_block_size=(
            block_causal.block_size
            if block_causal.target_encoding == "block_local"
            else None
        ),
    )
    task_builder = TaskBuilder(config=config, latent_stats_registry=stats)
    cpu_preparer = build_cpu_batch_preparer(config)
    preparation_kwargs = {} if cpu_preparer is None else {"prepare_batch": cpu_preparer}
    if batch_fetcher_factory is None:
        batch_fetcher = build_batch_fetcher(
            config=config,
            rank=rank,
            tokenizer=tokenizer,
            **preparation_kwargs,
        )
    else:
        batch_fetcher = batch_fetcher_factory(
            config=config,
            rank=rank,
            tokenizer=tokenizer,
            prepare_batch=cpu_preparer,
        )
    optimizers = build_optimizers(model, text_decoder, config.optimizers)
    schedulers = build_schedulers(optimizers, config.optimizers)
    ema = ExponentialMovingAverage.from_optimizers(
        optimizers.optimizers,
        decay=config.optimizers.ema.decay,
    )
    training_generator = torch.Generator(device=device).manual_seed(
        config.run.seed + 10_000 + rank
    )
    evaluation_generator = torch.Generator(device=device).manual_seed(
        config.run.seed + 20_000 + rank
    )
    identity = build_evaluation_identity(config, runtime.context)
    bindings = CheckpointBindings(
        model=model,
        text_decoder=text_decoder,
        optimizers=optimizers,
        schedulers=schedulers,
        ema=ema,
        training_generator=training_generator,
        evaluation_generator=evaluation_generator,
        data_stream=batch_fetcher,
    )
    checkpoint_manager = CheckpointManager(
        config.trainer.checkpoint_root,
        bindings=bindings,
        config=config,
        evaluation_identity=identity,
        eval_root=config.evaluation.artifact_dir,
        rank=rank,
        world_size=runtime.context.world_size,
        process_group=runtime.context.process_group,
        run_metadata=_run_metadata(config),
    )
    evaluator = None
    return _RuntimeBundle(
        config=config,
        distributed=runtime,
        tokenizer=tokenizer,
        stats=stats,
        model=model,
        training_model=trainables.training_model,
        text_decoder=text_decoder,
        training_text_decoder=trainables.training_text_decoder,
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        batch_encoder=batch_encoder,
        batch_fetcher=batch_fetcher,
        task_builder=task_builder,
        optimizers=optimizers,
        schedulers=schedulers,
        ema=ema,
        training_generator=training_generator,
        evaluation_generator=evaluation_generator,
        checkpoint_manager=checkpoint_manager,
        evaluator=evaluator,
    )


def _trainer_from_bundle(config: MFConfig, bundle: _RuntimeBundle) -> Trainer:
    sink = None
    if config.logging.enabled and bundle.distributed.context.rank == 0:
        sink = build_metric_sink(
            config.logging.output_dir,
            max_steps=config.trainer.max_steps,
            backend=config.logging.backend,
            defer_writes_until_close=config.logging.defer_writes_until_close,
            async_writes=config.logging.async_writes,
            async_queue_size=config.logging.async_queue_size,
        )
    return Trainer(
        config=config,
        model=bundle.training_model,
        text_decoder=bundle.training_text_decoder,
        batch_fetcher=bundle.batch_fetcher,
        batch_encoder=bundle.batch_encoder,
        task_builder=bundle.task_builder,
        optimizers=bundle.optimizers,
        schedulers=bundle.schedulers,
        ema=bundle.ema,
        checkpoint_manager=bundle.checkpoint_manager,
        evaluator=bundle.evaluator,
        training_generator=bundle.training_generator,
        distributed=bundle.distributed.context,
        metric_sink=sink,
    )


def build_train_runtime(*, config: MFConfig, args: argparse.Namespace) -> Trainer:
    init_from = getattr(args, "init_from", None)
    resume = getattr(args, "resume", None)
    initializer = None
    if init_from is not None:
        from mf.finetuning.initialization import initialize_from_pretrain_checkpoint

        def initializer(model: MFModel, text_decoder: LatentTextDecoder) -> object:
            return initialize_from_pretrain_checkpoint(
                model,
                init_from,
                text_decoder=text_decoder,
                use_ema=True,
            )

    batch_fetcher_factory = getattr(args, "data_factory_callable", None)
    if config.sft.enabled:
        if batch_fetcher_factory is not None:
            raise ValueError("SFT owns its data path; do not combine it with --data-factory")
        if init_from is None and resume is None:
            raise ValueError("SFT requires --init-from or --resume")
        from mf.data.sft import build_sft_batch_fetcher

        batch_fetcher_factory = build_sft_batch_fetcher
    return _trainer_from_bundle(
        config,
        _build_bundle(
            config,
            initializer=initializer,
            batch_fetcher_factory=batch_fetcher_factory,
        ),
    )
