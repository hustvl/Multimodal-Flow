"""Gradio layout and handlers for the MF-1 demo."""

from __future__ import annotations

import html
import os
import random
import time
from pathlib import Path

import gradio as gr

from mf_demo.backend import Cancelled
from mf_demo.theme import build_theme

PKG = Path(__file__).resolve().parent
STATIC = PKG / "static"
MAX_SEED = 2**31 - 1
POLL_SECONDS = 0.25
HISTORY_LIMIT = 8

# (chip label, text placed in the box)
T2I_EXAMPLES = [
    ("🔭 Observatory", "A quiet observatory above the clouds."),
    ("🚲 Red bicycle", "A red bicycle leaning against a blue door."),
    ("🍓 Strawberries", "A bowl of fresh strawberries on a wooden table."),
    ("🏔️ Snowy village", "A snowy mountain village at sunrise."),
]
TEXT_EXAMPLES = [
    ("🤖 Language model", "A short language model can"),
    ("🌍 Learning languages", "The quickest way to learn a new language is"),
    ("📖 Once upon a time", "Once upon a time, in a small village by the sea,"),
]
QUESTION_CHIPS = [
    "Describe this image.",
    "What is the main object in this picture?",
    "What colors do you see?",
    "Is there a person in the image?",
]

# logical name -> file under the repo root (the repo already ships these images in docs/assets)
ASSET_FILES = {
    "astronaut": ["docs/assets/astronaut.png"],
    "venice": ["docs/assets/samples/image_01.jpg"],
    "nebula": ["docs/assets/nebula.png"],
    "architecture": ["docs/assets/multimodal-flow-architecture.png"],
}


def _asset_roots() -> list[Path]:
    return [PKG.parent, PKG.parent.parent]


def find_asset(name: str) -> str | None:
    for root in _asset_roots():
        for rel in ASSET_FILES[name]:
            candidate = root / rel
            if candidate.is_file():
                return str(candidate)
    return None


def _gradio_major() -> int:
    return int(gr.__version__.split(".")[0])


def page_js() -> str:
    """Gradio 5 evaluates `js` as `(<js>)()`, so it must be a function expression;
    Gradio 6 takes raw code."""

    code = (STATIC / "effects.js").read_text()
    return code if _gradio_major() >= 6 else f"() => {{\n{code}\n}}"


def style_kwargs() -> dict:
    """Theme, CSS and JS for the page."""

    return {
        "theme": build_theme(),
        "css": (STATIC / "style.css").read_text(),
        "js": page_js(),
    }


def blocks_kwargs() -> dict:
    """Gradio 5 takes the styling in gr.Blocks(); Gradio 6 moved it to launch()."""

    return style_kwargs() if _gradio_major() < 6 else {}


def launch_kwargs() -> dict:
    allowed = {str(Path(p).parent) for p in (find_asset(n) for n in ASSET_FILES) if p}
    kwargs: dict = {"allowed_paths": sorted(allowed)}
    if _gradio_major() >= 6:
        kwargs.update(style_kwargs())
    return kwargs


# ───────────────────────── small HTML builders ─────────────────────────

TITLE_HTML = """
<header class="mf-title">
  <h1>Multimodal Flow <span>MF-1</span></h1>
  <p>Text and images from one flow model.</p>
</header>
"""

STEPS_HTML = """
<div class="mf-steps">
  <div class="mf-step"><b>1 &middot; Start from noise</b>
    <p>Text and images are turned into smooth vectors. Generation begins with random noise in that space.</p></div>
  <div class="mf-step"><b>2 &middot; Clean it step by step</b>
    <p>One flow model nudges the noise toward clean data over many steps. More steps are slower and usually cleaner.</p></div>
  <div class="mf-step"><b>3 &middot; Decode</b>
    <p>Frozen decoders turn the clean vectors back into pixels or words. The same process serves both.</p></div>
</div>
"""

FOOTER_HTML = """
<div class="mf-footer">
  MF-1 is a research prototype trained on 150B tokens: images are generated at 256&nbsp;px from a 224&nbsp;px
  encoder, and the language model is small, so expect imperfect results.<br>
  Requests share one GPU and run one at a time; you will see your place in line.
  &middot; <a href="https://github.com/hustvl/Multimodal-Flow" target="_blank" rel="noopener">hustvl/Multimodal-Flow</a>
</div>
"""

EMPTY_TEXT_HTML = '<div class="mf-empty">The answer will appear here.</div>'

STAGE_LABEL = {
    "image": "Denoising",
    "surprise": "Denoising",
    "caption": "Reading the image and writing",
    "text": "Writing",
}


def pick_seed(seed, randomize) -> int:
    return random.randint(0, MAX_SEED) if randomize else int(seed or 0)


def fmt_seconds(value: float) -> str:
    value = max(float(value), 0.0)
    if value < 90:
        return f"{value:.0f}s"
    minutes, seconds = divmod(int(value), 60)
    return f"{minutes}m {seconds:02d}s"


def progress_html(backend, state) -> str:
    esc = html.escape
    if state.waiting:
        ahead = backend.queue_ahead(state)
        label = "Waiting for the GPU" + (f" &middot; {ahead} ahead of you" if ahead else "")
        return (
            '<div class="mf-progress queued indet" role="status" aria-live="polite">'
            f'<div class="mf-progress-top"><span>{label}</span><span>{fmt_seconds(time.monotonic() - state.created)}</span></div>'
            '<div class="mf-bar"><i></i></div></div>'
        )
    stage = esc(STAGE_LABEL.get(state.kind, "Working"))
    fraction = state.fraction
    if fraction is None:
        right = fmt_seconds(state.elapsed)
        body = '<div class="mf-bar"><i></i></div>'
        cls = "indet"
    else:
        right = f"{state.steps_done}/{state.steps_total} steps"
        eta = backend.eta.remaining(state)
        if eta is not None:
            right += f" &middot; ~{fmt_seconds(eta)} left"
        body = f'<div class="mf-bar"><i style="width:{fraction * 100:.1f}%"></i></div>'
        cls = ""
    extra = ""
    if state.want_preview and state.preview is None and fraction is not None:
        extra = (
            f'<div class="mf-noise-wrap"><div class="mf-noise" style="--p:{fraction:.2f}"></div>'
            '<div class="mf-noise-cap">Illustration of denoising, not the actual image.</div></div>'
        )
    return (
        f'<div class="mf-progress {cls}" role="status" aria-live="polite">'
        f'<div class="mf-progress-top"><span>{stage}&hellip;</span><span>{right}</span></div>{body}{extra}</div>'
    )


def notice_html(kind: str, message: str) -> str:
    return (
        f'<div class="mf-progress {kind}" role="status" aria-live="polite">'
        f'<div class="mf-progress-top"><span>{html.escape(message)}</span><span></span></div></div>'
    )


def chips_html(*items: str) -> str:
    spans = "".join(f"<span>{html.escape(i)}</span>" for i in items if i)
    return f'<div class="mf-meta mf-fade">{spans}</div>'


def answer_html(text: str, question: str | None, kind: str, chips: str) -> str:
    q = f'<p class="mf-question">{html.escape(question)}</p>' if question else ""
    body = html.escape(text.strip() or "(empty answer)")
    return f'<div class="mf-answer {kind}">{q}<div class="mf-reveal">{body}</div></div>{chips}'


def friendly_error(error: BaseException) -> str:
    name = type(error).__name__
    if name == "OutOfMemoryError":
        return "The GPU ran out of memory. Try fewer steps or try again in a moment."
    return f"Something went wrong ({name}). Check the server log for details."


def gallery_value(history: list[dict]) -> list[tuple]:
    return [(h["image"], f'{h["label"]} · seed {h["seed"]}') for h in history]


# ───────────────────────── watching a run ─────────────────────────


def watch(backend, state):
    """Yield (progress html, preview image, preview version) until the run finishes.

    If the client disconnects the generator is closed and the run is cancelled, so an
    abandoned request never keeps the GPU busy.
    """

    try:
        while not state.finished:
            yield progress_html(backend, state), state.preview, state.preview_version
            time.sleep(POLL_SECONDS)
    finally:
        if not state.finished:
            state.cancel.set()


def build_ui(backend, *, surprise: bool | None = None) -> gr.Blocks:
    defaults = backend.defaults
    default_steps = int(defaults["steps"])
    if surprise is None:
        surprise = os.environ.get("MF_SURPRISE") == "1"

    # ── handlers ────────────────────────────────────────────────────
    def _image_run(state, used, meta, history, label):
        """Shared by prompt-based and unconditional image generation."""

        seen = -1
        yield progress_html(backend, state), gr.skip(), "", history, gr.skip(), gr.skip(), state.id
        for card, preview, version in watch(backend, state):
            image = preview if (preview is not None and version != seen) else gr.skip()
            seen = version
            yield card, image, "", history, gr.skip(), gr.skip(), state.id
        if isinstance(state.error, Cancelled):
            yield notice_html("cancelled", "Cancelled."), gr.skip(), "", history, gr.skip(), gr.skip(), state.id
            return
        if state.error is not None:
            yield (
                notice_html("error", friendly_error(state.error)),
                gr.skip(), "", history, gr.skip(), gr.skip(), state.id,
            )
            raise gr.Error(friendly_error(state.error))
        image = state.result["image"]
        seconds = state.elapsed
        entry = {"image": image, "label": label, **meta, "seed": used}
        history = ([entry] + history)[:HISTORY_LIMIT]
        info = chips_html(
            f"seed {used}", f'{meta["steps"]} steps', meta["method"].upper(),
            f'CFG {meta["cfg"]:g}' if meta.get("cfg") is not None else "", f"{seconds:.1f}s",
        )
        yield "", image, info, history, gallery_value(history), used, state.id

    def run_image(prompt, steps, cfg, method, seed, randomize, history):
        prompt = (prompt or "").strip()
        if not prompt:
            raise gr.Error("Write a prompt first, or pick one of the examples.")
        used = pick_seed(seed, randomize)
        state = backend.start_image(prompt, int(steps), float(cfg), method, used)
        meta = {"prompt": prompt, "steps": int(steps), "cfg": float(cfg), "method": method}
        yield from _image_run(state, used, meta, history, prompt[:40])

    def run_surprise(steps, cfg, method, seed, randomize, history):
        used = pick_seed(seed, randomize)
        state = backend.start_unconditional(int(steps), float(cfg), method, used)
        meta = {"prompt": "", "steps": int(steps), "cfg": None, "method": method}  # no guidance without a prompt
        yield from _image_run(state, used, meta, history, "Surprise")

    def _text_run(state, question, kind, used, length):
        yield progress_html(backend, state), gr.skip(), gr.skip(), state.id
        for card, _, _ in watch(backend, state):
            yield card, gr.skip(), gr.skip(), state.id
        if isinstance(state.error, Cancelled):
            yield notice_html("cancelled", "Cancelled."), gr.skip(), gr.skip(), state.id
            return
        if state.error is not None:
            yield notice_html("error", friendly_error(state.error)), gr.skip(), gr.skip(), state.id
            raise gr.Error(friendly_error(state.error))
        chips = chips_html(f"seed {used}", f"up to {int(length)} tokens", f"{state.elapsed:.1f}s")
        yield "", answer_html(state.result["text"], question, kind, chips), used, state.id

    def run_caption(image, question, length, seed, randomize):
        if image is None:
            raise gr.Error("Upload an image first, or pick one of the examples.")
        question = (question or "").strip()
        if not question:
            raise gr.Error("Write a question or instruction about the image.")
        used = pick_seed(seed, randomize)
        state = backend.start_caption(image, question, int(length), used)
        yield from _text_run(state, question, "vision", used, length)

    def run_text(prompt, length, seed, randomize):
        prompt = (prompt or "").strip()
        if not prompt:
            raise gr.Error("Write the start of a sentence first.")
        used = pick_seed(seed, randomize)
        state = backend.start_text(prompt, int(length), used)
        yield from _text_run(state, None, "text", used, length)

    def cancel_run(run_id):
        backend.cancel(run_id)
        return notice_html("cancelled", "Cancelling…")

    def apply_preset(preset):
        steps = {"fast": 16, "balanced": 32, "best": default_steps}.get(preset, default_steps)
        return gr.update(value=steps)

    def restore(history, evt: gr.SelectData):
        if not history or evt.index >= len(history):
            return [gr.skip()] * 7
        h = history[evt.index]
        return (
            h.get("prompt", ""), h["seed"], False, h["steps"],
            h["cfg"] if h.get("cfg") is not None else gr.skip(), h["method"], h["image"],
        )

    # ── layout ──────────────────────────────────────────────────────
    with gr.Blocks(title="MF-1 · Multimodal Flow", **blocks_kwargs()) as demo:
        gr.HTML(TITLE_HTML, padding=False)
        with gr.Accordion("How does it work?", open=False):
            gr.HTML(STEPS_HTML, padding=False)
            arch = find_asset("architecture")
            if arch:
                gr.Image(value=arch, interactive=False, show_label=False, container=False)

        history = gr.State([])
        last_image_seed = gr.State(None)
        image_run_id = gr.State(None)
        caption_run_id = gr.State(None)
        text_run_id = gr.State(None)
        caption_seed_state = gr.State(None)
        text_seed_state = gr.State(None)

        with gr.Tabs():
            # ── Text → Image ──
            with gr.Tab("🎨 Text → Image"):
                gr.HTML('<p class="mf-lead">Describe a picture and watch it emerge from noise.</p>', padding=False)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        prompt = gr.Textbox(
                            label="Describe the image", lines=3,
                            placeholder="A quiet observatory above the clouds.",
                        )
                        with gr.Row(elem_classes="mf-chiprow"):
                            chip_buttons = [
                                gr.Button(label, size="sm", elem_classes="mf-chip") for label, _ in T2I_EXAMPLES
                            ]
                        preset = gr.Radio(
                            choices=[
                                ("⚡ Fast · 16 steps", "fast"),
                                ("⚖ Balanced · 32 steps", "balanced"),
                                (f"✨ Best · {default_steps} steps", "best"),
                            ],
                            value="best", label="Quality",
                            info="More steps take longer and are usually cleaner.",
                        )
                        with gr.Accordion("Advanced", open=False):
                            steps = gr.Slider(4, 128, value=default_steps, step=1, label="Flow steps",
                                              info="How many cleaning steps the model takes.")
                            cfg = gr.Slider(0.0, 10.0, value=float(defaults["cfg"]), step=0.1, label="Prompt strength (CFG)",
                                            info="Higher follows the prompt harder; too high can look harsh.")
                            method = gr.Radio(["sde", "ode"], value=defaults["method"], label="Sampler",
                                              info="SDE adds a little fresh noise each step; ODE is a smooth, fixed path.")
                            with gr.Row():
                                seed = gr.Number(value=0, precision=0, minimum=0, maximum=MAX_SEED, label="Seed",
                                                 info="Same seed and settings give the same image.")
                                randomize = gr.Checkbox(value=True, label="Random seed")
                            reuse = gr.Button("Reuse last seed", size="sm")
                        with gr.Row():
                            generate = gr.Button("Generate", variant="primary", scale=3)
                            cancel_image = gr.Button("Cancel", variant="stop", scale=1)
                            surprise_btn = gr.Button("🎲 Surprise me", scale=2, visible=surprise)
                    with gr.Column(scale=6):
                        progress = gr.HTML(padding=False)
                        image_out = gr.Image(
                            label="Result", type="pil", interactive=False, height=420,
                            elem_classes="mf-result", placeholder="Your image will appear here.",
                        )
                        meta_out = gr.HTML(padding=False)
                        gallery = gr.Gallery(label="This session", columns=4, height="auto", allow_preview=False,
                                             object_fit="cover")

            # ── Image → Text ──
            with gr.Tab("🔍 Image → Text"):
                gr.HTML('<p class="mf-lead">Upload a photo and ask the model about it.</p>', padding=False)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        image_in = gr.Image(label="Image", type="pil", sources=["upload", "clipboard"], height=300)
                        examples = [[p] for p in (find_asset(n) for n in ("astronaut", "venice", "nebula")) if p]
                        if examples:
                            gr.Examples(examples=examples, inputs=[image_in], label="Try an example photo")
                        question = gr.Textbox(label="Question or instruction", value="Describe this image.")
                        with gr.Row(elem_classes="mf-chiprow"):
                            q_chips = [gr.Button(t, size="sm", elem_classes="mf-chip") for t in QUESTION_CHIPS]
                        with gr.Accordion("Advanced", open=False):
                            length = gr.Slider(16, 256, value=128, step=8, label="Max new tokens",
                                               info="Upper limit on the answer length.")
                            with gr.Row():
                                seed2 = gr.Number(value=0, precision=0, minimum=0, maximum=MAX_SEED, label="Seed")
                                randomize2 = gr.Checkbox(value=True, label="Random seed")
                        with gr.Row():
                            describe = gr.Button("Ask", variant="primary", scale=3)
                            cancel_caption = gr.Button("Cancel", variant="stop", scale=1)
                    with gr.Column(scale=5):
                        progress2 = gr.HTML(padding=False)
                        answer = gr.HTML(EMPTY_TEXT_HTML, padding=False)

            # ── Text continuation ──
            with gr.Tab("✍️ Text continuation"):
                gr.HTML('<p class="mf-lead">Give the start of a sentence and let the model continue it.</p>', padding=False)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        text_prompt = gr.Textbox(label="Start of the text", lines=3, placeholder="A short language model can")
                        with gr.Row(elem_classes="mf-chiprow"):
                            t_chips = [gr.Button(label, size="sm", elem_classes="mf-chip") for label, _ in TEXT_EXAMPLES]
                        with gr.Accordion("Advanced", open=False):
                            length3 = gr.Slider(16, 256, value=128, step=8, label="Max new tokens")
                            with gr.Row():
                                seed3 = gr.Number(value=0, precision=0, minimum=0, maximum=MAX_SEED, label="Seed")
                                randomize3 = gr.Checkbox(value=True, label="Random seed")
                        with gr.Row():
                            continue_btn = gr.Button("Continue", variant="primary", scale=3)
                            cancel_text = gr.Button("Cancel", variant="stop", scale=1)
                    with gr.Column(scale=5):
                        progress3 = gr.HTML(padding=False)
                        continuation = gr.HTML(EMPTY_TEXT_HTML, padding=False)

        gr.HTML(FOOTER_HTML, padding=False)

        # ── wiring ──────────────────────────────────────────────────
        for button, (_, text) in zip(chip_buttons, T2I_EXAMPLES):
            button.click(lambda t=text: t, outputs=prompt, show_progress="hidden")
        for button, text in zip(q_chips, QUESTION_CHIPS):
            button.click(lambda t=text: t, outputs=question, show_progress="hidden")
        for button, (_, text) in zip(t_chips, TEXT_EXAMPLES):
            button.click(lambda t=text: t, outputs=text_prompt, show_progress="hidden")

        preset.change(apply_preset, preset, steps, show_progress="hidden")
        reuse.click(lambda s: (s, False) if s is not None else (gr.skip(), gr.skip()),
                    last_image_seed, [seed, randomize], show_progress="hidden")

        image_outputs = [progress, image_out, meta_out, history, gallery, last_image_seed, image_run_id]
        run_kwargs = {"show_progress": "hidden", "concurrency_limit": 8, "api_name": False}
        gen_event = generate.click(
            run_image, [prompt, steps, cfg, method, seed, randomize, history], image_outputs,
            **{**run_kwargs, "api_name": "generate_image"},
        )
        prompt.submit(
            run_image, [prompt, steps, cfg, method, seed, randomize, history], image_outputs,
            **{**run_kwargs, "api_name": False},
        )
        events = [gen_event]
        if surprise:
            events.append(surprise_btn.click(
                run_surprise, [steps, cfg, method, seed, randomize, history], image_outputs,
                **{**run_kwargs, "api_name": "surprise"},
            ))
        cancel_image.click(cancel_run, image_run_id, progress, cancels=events, show_progress="hidden", api_name=False)
        gallery.select(restore, history, [prompt, seed, randomize, steps, cfg, method, image_out],
                       show_progress="hidden", api_name=False)

        caption_event = describe.click(
            run_caption, [image_in, question, length, seed2, randomize2],
            [progress2, answer, caption_seed_state, caption_run_id],
            **{**run_kwargs, "api_name": "describe_image"},
        )
        cancel_caption.click(cancel_run, caption_run_id, progress2, cancels=[caption_event],
                             show_progress="hidden", api_name=False)

        text_event = continue_btn.click(
            run_text, [text_prompt, length3, seed3, randomize3],
            [progress3, continuation, text_seed_state, text_run_id],
            **{**run_kwargs, "api_name": "continue_text"},
        )
        cancel_text.click(cancel_run, text_run_id, progress3, cancels=[text_event],
                          show_progress="hidden", api_name=False)

    return demo
