import hashlib
import json
from collections.abc import Mapping

from mf.config.schema import MFConfig
from mf.extensions import extension_manifest


def _canonical_hash(payload: object) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _normalize_optional_fields(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Make omitted optional settings hash like their default values."""
    normalized = dict(payload)
    flow = dict(payload["flow"])
    if flow.get("source_condition_noise") is None:
        flow.pop("source_condition_noise", None)
    if flow.get("text_block_causal") is None:
        flow.pop("text_block_causal", None)
    elif isinstance(flow["text_block_causal"], Mapping):
        text_block_causal = dict(flow["text_block_causal"])
        if text_block_causal.get("flex_fixed_sequence_length") is False:
            text_block_causal.pop("flex_fixed_sequence_length")
        if (
            text_block_causal.get("block_aligned_record_policy")
            == "whole_record_defer_pad"
        ):
            text_block_causal.pop("block_aligned_record_policy", None)
        flow["text_block_causal"] = text_block_causal
    normalized["flow"] = flow
    tasks = dict(payload["tasks"])
    if tasks.get("chunk_pack") is None:
        tasks.pop("chunk_pack", None)
    elif isinstance(tasks["chunk_pack"], Mapping):
        chunk_pack = dict(tasks["chunk_pack"])
        if chunk_pack.get("exposure_basis", "physical_tokens") == "physical_tokens":
            chunk_pack.pop("exposure_basis", None)
        if chunk_pack.get("text_packing") is None:
            chunk_pack.pop("text_packing", None)
        tasks["chunk_pack"] = chunk_pack
    normalized["tasks"] = tasks
    evaluation = dict(payload["evaluation"])
    if evaluation.get("cola_lm_eval") is None:
        evaluation.pop("cola_lm_eval", None)
    if evaluation.get("lmms_vqa") is None:
        evaluation.pop("lmms_vqa", None)
    normalized["evaluation"] = evaluation
    distributed = dict(payload["distributed"])
    if distributed.get("ddp_init_sync", True) is True:
        distributed.pop("ddp_init_sync", None)
    if distributed.get("ddp_gradient_compression", "none") == "none":
        distributed.pop("ddp_gradient_compression", None)
    if distributed.get("ddp_static_graph") is None:
        distributed.pop("ddp_static_graph", None)
    if distributed.get("ddp_find_unused_parameters") is None:
        distributed.pop("ddp_find_unused_parameters", None)
    normalized["distributed"] = distributed
    optimizers = dict(payload["optimizers"])
    if optimizers.get("muon_distributed_mode") == "sharded":
        optimizers.pop("muon_distributed_mode")
    if optimizers.get("muon_shard_group_size") is None:
        optimizers.pop("muon_shard_group_size", None)
    if optimizers.get("muon_matmul_precision", "highest") == "highest":
        optimizers.pop("muon_matmul_precision", None)
    normalized["optimizers"] = optimizers
    return normalized


def _config_mapping(config: MFConfig) -> dict[str, object]:
    return _normalize_optional_fields(config.model_dump(mode="json"))


def full_config_hash(config: MFConfig) -> str:
    return _canonical_hash(_config_mapping(config))


def full_config_mapping_hash(config: Mapping[str, object]) -> str:
    return _canonical_hash(_normalize_optional_fields(config))


def _training_fingerprint_payload(
    config: Mapping[str, object],
    *,
    include_objective: bool,
) -> dict[str, object]:
    """Build the semantic fingerprint payload, with legacy support."""
    config = _normalize_optional_fields(config)
    flow_value = config["flow"]
    if not isinstance(flow_value, Mapping):
        raise TypeError("flow config must be a mapping")
    flow = dict(flow_value)
    run = config["run"]
    distributed = config["distributed"]
    if not isinstance(run, Mapping) or not isinstance(distributed, Mapping):
        raise TypeError("run and distributed configs must be mappings")
    distributed_batch_equation: dict[str, object] = {
        "world_size": distributed["world_size"],
        "micro_batch_size_per_rank": distributed["micro_batch_size_per_rank"],
        "gradient_accumulation_steps": distributed["gradient_accumulation_steps"],
        "global_batch_size": distributed["global_batch_size"],
    }
    gradient_compression = distributed.get("ddp_gradient_compression", "none")
    if gradient_compression != "none":
        distributed_batch_equation["ddp_gradient_compression"] = gradient_compression
    static_graph = distributed.get("ddp_static_graph")
    if static_graph is not None:
        distributed_batch_equation["ddp_static_graph"] = static_graph
    find_unused_parameters = distributed.get("ddp_find_unused_parameters")
    if find_unused_parameters is not None:
        distributed_batch_equation["ddp_find_unused_parameters"] = (
            find_unused_parameters
        )
    payload: dict[str, object] = {
        "run": {"profile": run["profile"], "seed": run["seed"]},
        "model": config["model"],
        "codecs": config["codecs"],
        "latent_stats": config["latent_stats"],
        "data": config["data"],
        "tasks": config["tasks"],
        "flow": flow,
        "optimizers": config["optimizers"],
        "distributed_batch_equation": distributed_batch_equation,
    }
    if include_objective:
        payload["objective"] = config.get(
            "objective",
            {
                "mode": "multimodal_flow",
                "eos_token_id": 1,
                "pad_token_id": None,
                "null_condition_probability": 0.1,
            },
        )
        payload["extension_manifest"] = extension_manifest()
    sft = config.get("sft")
    if isinstance(sft, Mapping) and sft.get("enabled") is True:
        payload["sft"] = sft
    return payload


def training_fingerprint_mapping(config: Mapping[str, object]) -> str:
    """Hash training semantics, including the configured EOS contract."""
    return _canonical_hash(
        _training_fingerprint_payload(config, include_objective=True)
    )


def legacy_training_fingerprint_mapping(config: Mapping[str, object]) -> str:
    """Hash the pre-objective fingerprint used by older checkpoints."""
    return _canonical_hash(
        _training_fingerprint_payload(config, include_objective=False)
    )


def training_fingerprint(config: MFConfig) -> str:
    return training_fingerprint_mapping(_config_mapping(config))
