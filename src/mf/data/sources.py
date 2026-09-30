from __future__ import annotations

import math

from mf.config.schema import MFConfig, derive_task_slots

TEXT_SOURCE_NAMES = (
    "ultrafineweb_multi_style",
    "ultrafineweb_qa",
    "gpic_caption_text",
)


def configured_text_source_names(config: MFConfig) -> tuple[str, ...]:
    if config.data.text is None:
        raise _data_error("text sources for the built-in data reader")
    return TEXT_SOURCE_NAMES


def configured_source_names(config: MFConfig) -> tuple[str, ...]:
    return ("gpic", *configured_text_source_names(config))


def _data_error(requirement: str) -> ValueError:
    return ValueError(f"data config requires {requirement}")


def normalized_data_config(config: MFConfig) -> dict[str, object]:
    bundle = config.data.bundle
    if bundle is None and (config.data.image_text is None or config.data.text is None):
        raise _data_error(
            "a bundle or image_text and text sources for the built-in data reader; "
            "use --data-factory for other datasets"
        )
    if bundle is not None and any(
        source is not None for source in (config.data.image_text, config.data.text)
    ):
        raise _data_error("bundle to be used without legacy data sources")
    slots = derive_task_slots(
        config.tasks.weights,
        config.distributed.global_batch_size,
    )

    data = config.data
    text_tokens = data.text_max_length
    block_causal = config.flow.text_block_causal
    block_local = block_causal.target_encoding == "block_local"
    expected_decoder_tokens = block_causal.block_size if block_local else text_tokens
    if (
        config.codecs.text.max_length != text_tokens
        or config.model.text_decoder.max_length != expected_decoder_tokens
    ):
        if block_local:
            raise _data_error(
                "codecs.text.max_length to match data.text_max_length and block-local "
                "model.text_decoder.max_length to match text_block_causal.block_size"
            )
        else:
            raise _data_error(
                "data.text_max_length, codecs.text.max_length, and "
                "model.text_decoder.max_length to match"
            )
    if bundle is None:
        text_source_names = configured_text_source_names(config)
        text_sources = tuple(getattr(data.text, name) for name in text_source_names)
        text_weight_sum = math.fsum(source.weight for source in text_sources)
        if not math.isclose(text_weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise _data_error(f"text source weights to sum to 1 (got {text_weight_sum:.12g})")
        if (
            data.text.gpic_caption_text.weight > 0
            and data.text.gpic_caption_text.root != data.image_text.root
        ):
            raise _data_error("gpic_caption_text root to equal the GPIC image_text root")

    normalized_data = data.model_dump(mode="json")

    normalized_tasks = config.tasks.model_dump(mode="json")
    chunk_pack = normalized_tasks.get("chunk_pack")
    if (
        isinstance(chunk_pack, dict)
        and chunk_pack.get("exposure_basis", "physical_tokens") == "physical_tokens"
    ):
        chunk_pack.pop("exposure_basis", None)

    return {
        "profile": config.run.profile,
        "data": normalized_data,
        "tasks": {
            **normalized_tasks,
            "derived_slots": slots.model_dump(mode="json"),
        },
    }


def expected_source(config: MFConfig, source_name: str) -> tuple[str, str, str]:
    normalized_data_config(config)
    if source_name == "gpic":
        return "gpic", config.data.image_text.split, config.data.image_text.root
    if source_name not in TEXT_SOURCE_NAMES:
        raise ValueError(f"unknown data source {source_name!r}")
    source = getattr(config.data.text, source_name)
    if source is None:
        raise ValueError(f"inactive data source {source_name!r}")
    kind = {
        "gpic_caption_text": "gpic",
    }.get(source_name, "ultrafineweb")
    return kind, source.split, source.root
