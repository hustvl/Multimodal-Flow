from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from mf.contracts.batch import BranchRole, RawTaskBatch, TaskType, task_branch_roles
from mf.contracts.physical import (
    PhysicalSequenceSample,
    collate_physical_sequences,
)
from mf.contracts.task_registry import task_definition
from mf.contracts.sequence import MultimodalSequence, SequenceChunk
from mf.data.text import (
    DEFAULT_TEXT_TOKENS,
    TokenizedConditionBlock,
    TokenizedTextBlock,
)


@dataclass(frozen=True)
class RawTaskSample:
    task_type: TaskType | int
    image: Tensor | None = None
    text: TokenizedTextBlock | None = None
    text_prompt: TokenizedConditionBlock | None = None
    sequence: MultimodalSequence | None = None
    physical: PhysicalSequenceSample | None = None


def _canonical_sequence(
    *,
    sequence_id: int,
    task_type: TaskType,
    text_tokens: int,
    vision_tokens: int,
    prompt_tokens: int = 0,
) -> MultimodalSequence:
    """Represent the four paper tasks using the generic sequence contract."""

    definition = task_definition(task_type)
    chunks: list[SequenceChunk] = []
    for modality in ("image", "text"):
        role = definition.role(modality)
        if role is BranchRole.ABSENT:
            continue
        canonical_modality = modality
        if (
            canonical_modality == "text"
            and role is BranchRole.TARGET
            and prompt_tokens > 0
        ):
            chunks.append(
                SequenceChunk(
                    sequence_id=sequence_id,
                    chunk_index=len(chunks),
                    modality="text",
                    role=BranchRole.CONDITION,
                    token_count=prompt_tokens,
                    source_id="text_prompt",
                )
            )
        chunks.append(
            SequenceChunk(
                sequence_id=sequence_id,
                chunk_index=len(chunks),
                modality=canonical_modality,
                role=role,
                token_count=vision_tokens if canonical_modality == "image" else text_tokens,
                source_id=canonical_modality,
                output_slot=canonical_modality if role is BranchRole.TARGET else None,
            )
        )
    if not chunks:
        raise ValueError(f"{definition.label} does not define a sequence modality")
    # Conditions must precede targets for chunk-causal generation. The sort is
    # stable, so vision/text retain their paper ordering within each role.
    chunks.sort(key=lambda chunk: (chunk.role is BranchRole.TARGET, chunk.chunk_index))
    chunks = [
        SequenceChunk(
            sequence_id=chunk.sequence_id,
            chunk_index=index,
            modality=chunk.modality,
            role=chunk.role,
            token_count=chunk.token_count,
            source_id=chunk.source_id,
            output_slot=chunk.output_slot,
        )
        for index, chunk in enumerate(chunks)
    ]
    return MultimodalSequence(sequence_id, tuple(chunks)).validate()


def _validate_text_block(
    block: TokenizedTextBlock,
    *,
    pad_token_id: int,
    sample_index: int,
    text_tokens: int,
) -> None:
    tensors = (
        ("token_ids", block.token_ids, torch.long),
        ("content_mask", block.content_mask, torch.bool),
        ("target_mask", block.target_mask, torch.bool),
    )
    for name, tensor, dtype in tensors:
        if not isinstance(tensor, Tensor):
            raise ValueError(f"text {name} at index {sample_index} must be a torch.Tensor")
        if tuple(tensor.shape) != (text_tokens,):
            raise ValueError(
                f"text {name} at index {sample_index} must have shape [{text_tokens}]; "
                f"got {list(tensor.shape)}"
            )
        if tensor.dtype is not dtype:
            expected_dtype = "torch.long" if dtype is torch.long else str(dtype)
            raise ValueError(
                f"text {name} at index {sample_index} must have dtype {expected_dtype}; "
                f"got {tensor.dtype}"
            )
    if not (block.token_ids.device == block.content_mask.device == block.target_mask.device):
        raise ValueError(f"text block tensors at index {sample_index} must share a device")

    pad_positions = block.token_ids == pad_token_id
    if bool(((block.content_mask | block.target_mask) & pad_positions).any()):
        raise ValueError(
            f"text PAD positions at index {sample_index} must be inactive in both masks"
        )
    if not torch.equal(block.target_mask, block.content_mask):
        raise ValueError(f"text target_mask at index {sample_index} must equal content_mask")
    if not torch.equal(block.content_mask, ~pad_positions):
        raise ValueError(
            f"text content_mask at index {sample_index} must activate every non-PAD token"
        )


def _validate_condition_block(block: TokenizedConditionBlock) -> None:
    if not isinstance(block, TokenizedConditionBlock):
        raise TypeError("image_to_text_prompt must be a TokenizedConditionBlock")
    if block.token_ids.ndim != 1 or block.token_ids.shape[0] <= 0:
        raise ValueError("image_to_text_prompt token_ids must have shape [P] with P > 0")
    if block.token_ids.dtype is not torch.long:
        raise ValueError("image_to_text_prompt token_ids must have dtype torch.long")
    if block.content_mask.shape != block.token_ids.shape:
        raise ValueError("image_to_text_prompt content_mask must match token_ids")
    if block.content_mask.dtype is not torch.bool:
        raise ValueError("image_to_text_prompt content_mask must have dtype torch.bool")
    if block.content_mask.device != block.token_ids.device:
        raise ValueError("image_to_text_prompt tensors must share a device")
    if not bool(block.content_mask.any()):
        raise ValueError("image_to_text_prompt must contain at least one active token")
    if bool((~block.content_mask[:-1] & block.content_mask[1:]).any()):
        raise ValueError("image_to_text_prompt must be right padded")


def collate_raw_task_batch(
    samples: Sequence[RawTaskSample],
    *,
    pad_token_id: int,
    text_tokens: int = DEFAULT_TEXT_TOKENS,
    vision_tokens: int = 256,
    image_to_text_prompt: TokenizedConditionBlock | None = None,
    image_resolution: int = 224,
) -> RawTaskBatch:
    if type(text_tokens) is not int or text_tokens <= 0:
        raise ValueError("text_tokens must be a positive integer")
    if type(vision_tokens) is not int or vision_tokens <= 0:
        raise ValueError("vision_tokens must be a positive integer")
    if type(image_resolution) is not int or image_resolution <= 0:
        raise ValueError("image_resolution must be a positive integer")
    batch_size = len(samples)
    task_type = torch.tensor([sample.task_type for sample in samples], dtype=torch.long)
    physical_values = tuple(sample.physical for sample in samples)
    if any(value is not None for value in physical_values):
        vision_present = torch.zeros(batch_size, dtype=torch.bool)
        text_present = torch.zeros(batch_size, dtype=torch.bool)
        if not all(value is not None for value in physical_values):
            raise ValueError(
                "a physical sequence batch must provide one physical sample per row"
            )
        if any(
            sample.image is not None
            or sample.text is not None
            or sample.text_prompt is not None
            for sample in samples
        ):
            raise ValueError(
                "physical sequence samples cannot be mixed with legacy image/text payloads"
            )
        physical_layout = collate_physical_sequences(
            tuple(value for value in physical_values if value is not None)
        )
        if bool(vision_present.any()) or bool(text_present.any()):
            raise ValueError(
                "physical sequence samples use registered modalities; "
                "do not also declare legacy image/text task roles"
            )
        sequences = tuple(
            sample.sequence for sample in samples if sample.sequence is not None
        )
        if sequences and len(sequences) != batch_size:
            raise ValueError(
                "physical sequence batches must provide sequence contracts for every row"
            )
        return RawTaskBatch(
            task_type=task_type,
            vision_present=vision_present,
            text_present=text_present,
            sequence_contracts=tuple(sequences) if sequences else None,
            compiled_sequences=physical_layout.compiled_sequences,
            physical_layout=physical_layout,
        ).validate()
    vision_role, text_role = task_branch_roles(task_type)
    vision_present = vision_role != int(BranchRole.ABSENT)
    text_present = text_role != int(BranchRole.ABSENT)
    images = torch.zeros(batch_size, 3, image_resolution, image_resolution, dtype=torch.float32)
    text_token_ids = torch.full(
        (batch_size, text_tokens),
        pad_token_id,
        dtype=torch.long,
    )
    text_content_mask = torch.zeros(batch_size, text_tokens, dtype=torch.bool)

    per_sample_prompts = tuple(
        sample.text_prompt
        for sample in samples
        if sample.text_prompt is not None
    )
    for prompt in per_sample_prompts:
        _validate_condition_block(prompt)
    if image_to_text_prompt is not None:
        _validate_condition_block(image_to_text_prompt)

    prompt_token_ids: Tensor | None = None
    prompt_content_mask: Tensor | None = None
    sequences: list[MultimodalSequence] = []
    compiled_sequences = []
    prompt_widths = [prompt.token_ids.shape[0] for prompt in per_sample_prompts]
    if image_to_text_prompt is not None:
        prompt_widths.append(image_to_text_prompt.token_ids.shape[0])
    if prompt_widths:
        prompt_tokens = max(prompt_widths)
        prompt_token_ids = torch.full(
            (batch_size, prompt_tokens),
            pad_token_id,
            dtype=torch.long,
        )
        prompt_content_mask = torch.zeros(batch_size, prompt_tokens, dtype=torch.bool)

    for index, sample in enumerate(samples):
        definition = task_definition(sample.task_type)
        prompt = sample.text_prompt
        if prompt is None and definition.prompt_policy == "required":
            prompt = image_to_text_prompt
        sequence = sample.sequence or _canonical_sequence(
            sequence_id=index,
            task_type=sample.task_type,
            text_tokens=text_tokens,
            vision_tokens=vision_tokens,
            prompt_tokens=0 if prompt is None else prompt.token_ids.shape[0],
        )
        sequence.validate()
        sequences.append(sequence)
        compiled = sequence.compile()
        compiled_sequences.append(compiled)
        definition = task_definition(sample.task_type)
        for semantic_modality in ("image", "text"):
            sequence_role = compiled.primary_role(semantic_modality)
            task_role = definition.role(semantic_modality)
            if sequence_role is not task_role:
                raise ValueError(
                    f"{definition.label} sequence role for {semantic_modality!r} "
                    f"must match its registered task definition"
                )
        unsupported = tuple(
            chunk.modality
            for chunk in compiled.chunks
            if chunk.modality not in {"image", "text"}
        )
        if unsupported:
            raise ValueError(
                "the default image/text codec path cannot consume modalities "
                f"{unsupported}; register a physical codec adapter"
            )
        requires_vision = bool(vision_present[index])
        requires_text = bool(text_present[index])
        if requires_vision != (sample.image is not None):
            raise ValueError(
                f"image payload does not match {sample.task_type.label} task at index {index}"
            )
        if requires_text != (sample.text is not None):
            raise ValueError(
                f"text payload does not match {sample.task_type.label} task at index {index}"
            )
        if sample.image is not None:
            expected_image_shape = (3, image_resolution, image_resolution)
            if tuple(sample.image.shape) != expected_image_shape:
                raise ValueError(
                    f"image at index {index} must have shape "
                    f"{list(expected_image_shape)}; got {list(sample.image.shape)}"
                )
            images[index].copy_(sample.image)
        if sample.text is not None:
            _validate_text_block(
                sample.text,
                pad_token_id=pad_token_id,
                sample_index=index,
                text_tokens=text_tokens,
            )
            text_token_ids[index].copy_(sample.text.token_ids)
            text_content_mask[index].copy_(sample.text.content_mask)
        if definition.prompt_policy == "required":
            prompt = sample.text_prompt or image_to_text_prompt
            if prompt is None:
                raise ValueError(f"{definition.label} requires a clean text prompt")
            assert prompt_token_ids is not None
            assert prompt_content_mask is not None
            width = prompt.token_ids.shape[0]
            prompt_token_ids[index, :width].copy_(prompt.token_ids)
            prompt_content_mask[index, :width].copy_(prompt.content_mask)
        elif definition.prompt_policy == "optional" and sample.text_prompt is not None:
            assert prompt_token_ids is not None
            assert prompt_content_mask is not None
            width = sample.text_prompt.token_ids.shape[0]
            prompt_token_ids[index, :width].copy_(sample.text_prompt.token_ids)
            prompt_content_mask[index, :width].copy_(sample.text_prompt.content_mask)
        elif sample.text_prompt is not None:
            raise ValueError(
                f"{definition.label} does not allow text_prompt"
            )

    return RawTaskBatch(
        task_type=task_type,
        vision_present=vision_present,
        text_present=text_present,
        images=images,
        text_token_ids=text_token_ids,
        text_content_mask=text_content_mask,
        text_prompt_token_ids=prompt_token_ids,
        text_prompt_content_mask=prompt_content_mask,
        sequence_contracts=tuple(sequences),
        compiled_sequences=tuple(compiled_sequences),
    ).validate()
