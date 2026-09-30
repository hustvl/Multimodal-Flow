from __future__ import annotations

USER_MARKER = "USER:"
ASSISTANT_MARKER = "ASSISTANT:"
IMAGE_CAPTION_PROMPT = (
    f"{USER_MARKER} Describe this image in words.\n{ASSISTANT_MARKER}"
)


def format_instruction_prompt(text: str) -> str:
    """Return the single-turn condition boundary shared by I2T and VQA."""

    prompt = str(text or "").replace("<image>", "").strip()
    if not prompt:
        raise ValueError("instruction prompt must not be empty")
    if prompt.endswith(ASSISTANT_MARKER):
        prompt = prompt[: -len(ASSISTANT_MARKER)].rstrip()
    if prompt.startswith(USER_MARKER):
        prompt = prompt[len(USER_MARKER) :].lstrip()
    if not prompt:
        raise ValueError("instruction prompt must contain user text")
    return f"{USER_MARKER} {prompt}\n{ASSISTANT_MARKER}"
