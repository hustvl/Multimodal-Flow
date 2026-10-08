"""Model backends for the MF-1 demo.

MFBackend runs the real pipeline; MockBackend simulates progress and live previews
so the UI can be developed and tested without a GPU. Both expose the same start_*
methods, which return a RunState that the UI polls from a Gradio generator.

Progress without touching src/: the sampler has no callback, but it calls
model.predict_normalized once per branch per step (twice per step with CFG). A
ProgressTap wraps that method on the model instance, counts calls, and can decode
the current x0 estimate into a live preview image.
"""

from __future__ import annotations

import os
import random
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

DEMO_DIR = Path(__file__).resolve().parent.parent
PREVIEW_INTERVAL = 1.5
MAX_TRACKED_RUNS = 64


class Cancelled(Exception):
    """Raised inside the model call when the user cancels a run."""


@dataclass(eq=False)
class RunState:
    kind: str  # image | surprise | caption | text
    total: int = 0  # expected predict_normalized calls; 0 means unknown
    calls_per_step: int = 2
    want_preview: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    done: int = 0
    waiting: bool = True
    finished: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)
    created: float = field(default_factory=time.monotonic)
    started: float | None = None
    ended: float | None = None
    preview: Image.Image | None = None
    preview_version: int = 0
    last_preview_at: float = 0.0
    preview_seconds: float = 0.0
    result: object = None
    error: BaseException | None = None

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.ended or time.monotonic()) - self.started

    @property
    def fraction(self) -> float | None:
        if self.total <= 0:
            return None
        return min(self.done / self.total, 1.0)

    @property
    def steps_done(self) -> int:
        return self.done // max(self.calls_per_step, 1)

    @property
    def steps_total(self) -> int:
        return self.total // max(self.calls_per_step, 1)


class EtaTracker:
    """Seconds-per-call moving average, used to estimate time left."""

    def __init__(self, smoothing: float = 0.5) -> None:
        self._smoothing = smoothing
        self._rate: float | None = None
        self._lock = threading.Lock()

    def record(self, calls: int, seconds: float) -> None:
        if calls <= 0 or seconds <= 0:
            return
        rate = seconds / calls
        with self._lock:
            if self._rate is None:
                self._rate = rate
            else:
                self._rate = self._smoothing * rate + (1 - self._smoothing) * self._rate

    def remaining(self, state: RunState) -> float | None:
        if state.total <= 0 or state.started is None:
            return None
        left = max(state.total - state.done, 0)
        if state.done >= 2:
            return left * (state.elapsed / state.done)
        with self._lock:
            rate = self._rate
        return None if rate is None else left * rate


class FifoGate:
    """One run at a time, served in arrival order. Waiting runs can be cancelled."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._queue: deque[RunState] = deque()
        self._busy = False

    def enter(self, state: RunState) -> None:
        with self._cv:
            self._queue.append(state)
            try:
                while self._busy or self._queue[0] is not state:
                    if state.cancel.is_set():
                        raise Cancelled()
                    self._cv.wait(0.2)
                if state.cancel.is_set():
                    raise Cancelled()
            except BaseException:
                if state in self._queue:
                    self._queue.remove(state)
                self._cv.notify_all()
                raise
            self._queue.popleft()
            self._busy = True

    def leave(self) -> None:
        with self._cv:
            self._busy = False
            self._cv.notify_all()

    def ahead(self, state: RunState) -> int:
        """How many runs must finish before this one starts."""
        with self._cv:
            try:
                position = list(self._queue).index(state)
            except ValueError:
                return 0
            return position + (1 if self._busy else 0)


class ProgressTap:
    """Counts model calls and optionally decodes live previews."""

    def __init__(self, model, decode_preview=None, interval: float = PREVIEW_INTERVAL) -> None:
        self.state: RunState | None = None
        self.preview_enabled = decode_preview is not None
        self._decode = decode_preview
        self._interval = interval
        self._original = model.predict_normalized
        model.predict_normalized = self._wrapped

    def _wrapped(self, model_input, routing_layout=None):
        state = self.state
        if state is None:
            return self._original(model_input, routing_layout)
        if state.cancel.is_set():
            raise Cancelled()
        output = self._original(model_input, routing_layout)
        state.done += 1
        self._maybe_preview(state, output)
        return output

    def _maybe_preview(self, state: RunState, output) -> None:
        if not (self.preview_enabled and state.want_preview):
            return
        if (state.done - 1) % max(state.calls_per_step, 1):
            return  # only the conditional branch carries the prompt-guided estimate
        now = time.monotonic()
        if now - state.last_preview_at < self._interval:
            return
        prediction = getattr(output, "vision_pred_norm", None)
        if prediction is None:
            return
        try:
            image = self._decode(prediction)
        except Exception:
            self.preview_enabled = False
            print("[mf-demo] live preview disabled after an error:")
            traceback.print_exc()
            return
        state.preview = image
        state.preview_version += 1
        state.last_preview_at = time.monotonic()
        state.preview_seconds += state.last_preview_at - now


class BaseBackend:
    """Run registry, FIFO GPU gate and ETA shared by the real and mock backends."""

    defaults: dict = {"steps": 64, "cfg": 3.0, "method": "sde"}

    def __init__(self) -> None:
        self.gate = FifoGate()
        self.eta = EtaTracker()
        self._runs: OrderedDict[str, RunState] = OrderedDict()
        self._runs_lock = threading.Lock()

    # -- run bookkeeping -------------------------------------------------
    def _register(self, state: RunState) -> None:
        with self._runs_lock:
            self._runs[state.id] = state
            while len(self._runs) > MAX_TRACKED_RUNS:
                self._runs.popitem(last=False)

    def get_run(self, run_id: str | None) -> RunState | None:
        with self._runs_lock:
            return self._runs.get(run_id) if run_id else None

    def cancel(self, run_id: str | None) -> bool:
        state = self.get_run(run_id)
        if state is None or state.finished:
            return False
        state.cancel.set()
        return True

    def queue_ahead(self, state: RunState) -> int:
        return self.gate.ahead(state)

    # -- hooks for subclasses --------------------------------------------
    def _begin(self, state: RunState) -> None:
        pass

    def _end(self, state: RunState) -> None:
        pass

    def _launch(self, state: RunState, work) -> RunState:
        self._register(state)

        def target() -> None:
            try:
                self.gate.enter(state)
            except Cancelled as exc:
                state.error = exc
                state.waiting = False
                state.ended = time.monotonic()
                state.finished = True
                return
            try:
                state.waiting = False
                state.started = time.monotonic()
                self._begin(state)
                state.result = work()
            except BaseException as exc:  # noqa: BLE001 - reported to the UI
                state.error = exc
                if not isinstance(exc, Cancelled):
                    traceback.print_exc()
            finally:
                self._end(state)
                state.ended = time.monotonic()
                self.gate.leave()
                if state.error is None and state.kind in ("image", "surprise") and state.done:
                    self.eta.record(state.done, state.ended - state.started)
                    print(
                        f"[mf-demo] {state.kind}: {state.steps_total} steps in "
                        f"{state.elapsed:.1f}s, preview overhead {state.preview_seconds:.1f}s"
                    )
                state.finished = True

        threading.Thread(target=target, daemon=True).start()
        return state

    # -- public API used by the UI ---------------------------------------
    def start_image(self, prompt: str, steps: int, cfg: float, method: str, seed: int) -> RunState:
        state = RunState("image", total=2 * steps, calls_per_step=2, want_preview=True)
        return self._launch(state, lambda: self._run_image(state, prompt, steps, cfg, method, seed))

    def start_unconditional(self, steps: int, cfg: float, method: str, seed: int) -> RunState:
        state = RunState("surprise", total=steps, calls_per_step=1, want_preview=True)
        return self._launch(state, lambda: self._run_unconditional(state, steps, cfg, method, seed))

    def start_caption(self, image: Image.Image, question: str, length: int, seed: int) -> RunState:
        state = RunState("caption")
        return self._launch(state, lambda: self._run_caption(state, image, question, length, seed))

    def start_text(self, prompt: str, length: int, seed: int) -> RunState:
        state = RunState("text")
        return self._launch(state, lambda: self._run_text(state, prompt, length, seed))

    def _run_image(self, state, prompt, steps, cfg, method, seed):
        raise NotImplementedError

    def _run_unconditional(self, state, steps, cfg, method, seed):
        raise NotImplementedError

    def _run_caption(self, state, image, question, length, seed):
        raise NotImplementedError

    def _run_text(self, state, prompt, length, seed):
        raise NotImplementedError


class MFBackend(BaseBackend):
    """Loads the MF checkpoint once and serves requests one at a time."""

    def __init__(self) -> None:
        super().__init__()
        os.environ.setdefault("MF_ASSETS_ROOT", str(DEMO_DIR / "assets"))
        checkpoint = os.environ.get("MF_CHECKPOINT")
        if not checkpoint:
            raise RuntimeError(
                "Set MF_CHECKPOINT to the MF checkpoint folder (for example <dir>/MF/sft). "
                "See README.md for how to download it."
            )

        import torch
        from torchvision.transforms.functional import to_pil_image

        from mf.inference.pipeline import MFPipeline
        from mf.inference.protocol import GenerationConfig

        self._torch = torch
        self._to_pil = to_pil_image
        self._config_cls = GenerationConfig
        self.pipe = MFPipeline.from_checkpoint(
            checkpoint,
            device=os.environ.get("MF_DEVICE", "cuda"),
            weights=os.environ.get("MF_WEIGHTS", "ema"),
        )
        sampler = self.pipe.default_sampler_config
        self.defaults = {
            "steps": sampler.num_inference_steps,
            "cfg": sampler.cfg_scale,
            "method": sampler.method,
        }
        live = os.environ.get("MF_LIVE_PREVIEW", "1") != "0"
        self.tap = ProgressTap(
            self.pipe.bundle.model,
            decode_preview=self._decode_preview if live else None,
        )

    def _to_image(self, tensor) -> Image.Image:
        return self._to_pil(((tensor.detach().cpu().float() + 1) / 2).clamp(0, 1))

    def _decode_preview(self, prediction) -> Image.Image:
        bundle = self.pipe.bundle
        with self._torch.autocast(device_type="cuda", enabled=False):
            raw = bundle.model.denormalize_vision_prediction(prediction[:1].float())
            images = bundle.vision_decoder.decode(raw)
        return self._to_image(images[0])

    def _begin(self, state: RunState) -> None:
        self.tap.state = state

    def _end(self, state: RunState) -> None:
        self.tap.state = None
        if state.error is not None:
            self._torch.cuda.empty_cache()

    def _config(self, **kwargs):
        return self._config_cls(**kwargs)

    def _run_image(self, state, prompt, steps, cfg, method, seed):
        config = self._config(
            num_inference_steps=int(steps), cfg_scale=float(cfg), method=method, seed=int(seed)
        )
        image = self.pipe.generate_image((prompt,), config=config)[0]
        return {"image": self._to_image(image)}

    def _run_unconditional(self, state, steps, cfg, method, seed):
        config = self._config(
            num_inference_steps=int(steps), cfg_scale=float(cfg), method=method, seed=int(seed)
        )
        image = self.pipe.generate_image_unconditional(1, config=config)[0]
        return {"image": self._to_image(image)}

    def _run_caption(self, state, image, question, length, seed):
        from mf.data.images import preprocess_image

        vision = self.pipe.config.codecs.vision
        tensor = preprocess_image(
            image,
            resolution=vision.encoder_input_resolution,
            policy=getattr(vision, "image_preprocessing", "legacy_center_crop_bicubic_v1"),
        )
        config = self._config(seed=int(seed))
        text = self.pipe.caption(
            (tensor,), prompt=question, target_length=int(length), config=config
        )[0]
        return {"text": text}

    def _run_text(self, state, prompt, length, seed):
        config = self._config(seed=int(seed))
        text = self.pipe.complete_text((prompt,), target_length=int(length), config=config)[0]
        return {"text": text}


class MockBackend(BaseBackend):
    """Simulates progress and previews so the UI works on a machine without a GPU."""

    def __init__(self, delay: float | None = None) -> None:
        super().__init__()
        self._delay = float(os.environ.get("MF_MOCK_DELAY", "0.03")) if delay is None else delay

    @staticmethod
    def _picture(seed: int, progress: float) -> Image.Image:
        rng = random.Random(seed)
        top = tuple(rng.randint(30, 120) for _ in range(3))
        bottom = tuple(rng.randint(120, 230) for _ in range(3))
        base = Image.new("RGB", (256, 256))
        draw = ImageDraw.Draw(base)
        for y in range(256):
            mix = y / 255
            draw.line([(0, y), (256, y)], fill=tuple(int(a + (b - a) * mix) for a, b in zip(top, bottom)))
        cx, cy, r = rng.randint(70, 186), rng.randint(70, 186), rng.randint(30, 60)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(rng.randint(200, 255) for _ in range(3)))
        noise = Image.effect_noise((256, 256), 90).convert("RGB")
        blended = Image.blend(noise, base, min(max(progress, 0.0), 1.0) ** 0.8)
        return blended.filter(ImageFilter.GaussianBlur((1 - progress) * 3))

    def _simulate(self, state: RunState, seed: int) -> Image.Image:
        for _ in range(state.total):
            if state.cancel.is_set():
                raise Cancelled()
            time.sleep(self._delay)
            state.done += 1
            now = time.monotonic()
            if (state.done - 1) % state.calls_per_step == 0 and now - state.last_preview_at >= 0.4:
                state.preview = self._picture(seed, state.done / state.total)
                state.preview_version += 1
                state.last_preview_at = now
        return self._picture(seed, 1.0)

    def _run_image(self, state, prompt, steps, cfg, method, seed):
        return {"image": self._simulate(state, seed)}

    def _run_unconditional(self, state, steps, cfg, method, seed):
        return {"image": self._simulate(state, seed)}

    def _think(self, state: RunState, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if state.cancel.is_set():
                raise Cancelled()
            time.sleep(0.05)

    def _run_caption(self, state, image, question, length, seed):
        self._think(state, 1.2)
        return {"text": f"[mock] Q: {question} A: a {image.size[0]}x{image.size[1]} picture (seed {seed})."}

    def _run_text(self, state, prompt, length, seed):
        self._think(state, 1.2)
        return {"text": f"{prompt} ... [mock continuation of up to {length} tokens, seed {seed}]"}
