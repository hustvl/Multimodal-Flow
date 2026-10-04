import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from mf_demo.backend import (
    Cancelled,
    EtaTracker,
    FifoGate,
    MockBackend,
    ProgressTap,
    RunState,
)


class FakeModel:
    """Stands in for MFModel: predict_normalized returns an object with vision_pred_norm."""

    def __init__(self):
        self.calls = 0

    def predict_normalized(self, model_input, routing_layout=None):
        self.calls += 1
        return SimpleNamespace(vision_pred_norm=f"pred{self.calls}")


def wait(state, timeout=10):
    end = time.monotonic() + timeout
    while not state.finished:
        assert time.monotonic() < end, "run did not finish"
        time.sleep(0.01)


def test_tap_passes_through_without_state():
    model = FakeModel()
    tap = ProgressTap(model)
    assert model.predict_normalized("x").vision_pred_norm == "pred1"
    assert tap.state is None


def test_tap_counts_calls_and_cancels():
    model = FakeModel()
    tap = ProgressTap(model)
    tap.state = RunState("image", total=8, calls_per_step=2)
    model.predict_normalized("a")
    model.predict_normalized("b")
    assert tap.state.done == 2 and tap.state.steps_done == 1 and tap.state.steps_total == 4
    tap.state.cancel.set()
    with pytest.raises(Cancelled):
        model.predict_normalized("c")
    assert model.calls == 2  # the cancelled call never reached the model


def test_tap_previews_only_conditional_calls_and_throttles():
    model = FakeModel()
    decoded = []

    def decode(prediction):
        decoded.append(prediction)
        return Image.new("RGB", (4, 4))

    tap = ProgressTap(model, decode_preview=decode, interval=0.0)
    tap.state = RunState("image", total=8, calls_per_step=2, want_preview=True)
    for _ in range(4):
        model.predict_normalized("x")
    assert decoded == ["pred1", "pred3"]  # calls 1 and 3 are the conditional branch
    assert tap.state.preview_version == 2

    throttled = ProgressTap(FakeModel(), decode_preview=decode, interval=3600)
    throttled.state = RunState("image", total=8, calls_per_step=2, want_preview=True)
    model2 = throttled._original.__self__
    model2.predict_normalized("a"), model2.predict_normalized("b")
    assert throttled.state.preview_version == 1  # second conditional call is throttled


def test_tap_disables_preview_after_decoder_error():
    model = FakeModel()

    def broken(prediction):
        raise RuntimeError("boom")

    tap = ProgressTap(model, decode_preview=broken, interval=0.0)
    tap.state = RunState("image", total=4, calls_per_step=2, want_preview=True)
    model.predict_normalized("a")
    model.predict_normalized("b")
    assert tap.preview_enabled is False and tap.state.done == 2  # generation unaffected


def test_unconditional_runs_preview_every_call():
    model = FakeModel()
    tap = ProgressTap(model, decode_preview=lambda p: Image.new("RGB", (4, 4)), interval=0.0)
    tap.state = RunState("surprise", total=3, calls_per_step=1, want_preview=True)
    for _ in range(3):
        model.predict_normalized("x")
    assert tap.state.preview_version == 3


def test_eta_uses_history_then_live_rate():
    eta = EtaTracker()
    state = RunState("image", total=10, calls_per_step=2)
    assert eta.remaining(state) is None  # not started
    state.started = time.monotonic()
    assert eta.remaining(state) is None  # no history, no calls yet
    eta.record(10, 5.0)
    assert eta.remaining(state) == pytest.approx(5.0)  # 10 calls left at 0.5 s/call
    state.done = 5
    state.started = time.monotonic() - 10  # live rate: 2 s/call -> 5 left = 10 s
    assert eta.remaining(state) == pytest.approx(10.0, rel=0.05)
    assert eta.remaining(RunState("text")) is None  # unknown total


def test_fifo_gate_orders_runs_and_cancels_waiters():
    gate = FifoGate()
    order = []
    first, second, third = RunState("text"), RunState("text"), RunState("text")
    gate.enter(first)

    def worker(state):
        try:
            gate.enter(state)
        except Cancelled:
            order.append(("cancelled", state))
            return
        order.append(("ran", state))
        gate.leave()

    t2 = threading.Thread(target=worker, args=(second,))
    t3 = threading.Thread(target=worker, args=(third,))
    t2.start(); time.sleep(0.05); t3.start(); time.sleep(0.05)
    assert gate.ahead(second) == 1
    assert gate.ahead(third) == 2
    third.cancel.set(); time.sleep(0.4)
    assert ("cancelled", third) in order
    gate.leave()
    t2.join(2); t3.join(2)
    assert order == [("cancelled", third), ("ran", second)]


def test_mock_image_run_reports_progress_and_preview():
    backend = MockBackend(delay=0.0)
    state = backend.start_image("a cat", 8, 3.0, "sde", 7)
    wait(state)
    assert state.error is None
    assert state.done == 16 and state.fraction == 1.0
    assert state.preview_version >= 1
    assert isinstance(state.result["image"], Image.Image)
    assert backend.eta._rate is not None  # successful image runs feed the ETA


def test_mock_cancel_mid_run_and_lookup():
    backend = MockBackend(delay=0.01)
    state = backend.start_image("a cat", 64, 3.0, "sde", 1)
    time.sleep(0.1)
    assert backend.get_run(state.id) is state
    assert backend.cancel(state.id) is True
    wait(state)
    assert isinstance(state.error, Cancelled)
    assert backend.cancel(state.id) is False  # already finished
    assert backend.cancel("nope") is False


def test_second_request_waits_for_first():
    backend = MockBackend(delay=0.01)
    first = backend.start_image("one", 16, 3.0, "sde", 1)
    time.sleep(0.05)
    second = backend.start_text("two", 32, 2)
    time.sleep(0.05)
    assert second.waiting and backend.queue_ahead(second) >= 1
    wait(first); wait(second)
    assert second.started >= first.ended and second.error is None


def test_cancel_while_queued_never_runs():
    backend = MockBackend(delay=0.01)
    first = backend.start_image("one", 32, 3.0, "sde", 1)
    time.sleep(0.05)
    queued = backend.start_text("two", 32, 2)
    time.sleep(0.05)
    backend.cancel(queued.id)
    wait(queued)
    assert isinstance(queued.error, Cancelled) and queued.started is None
    backend.cancel(first.id); wait(first)


def test_text_and_caption_results():
    backend = MockBackend(delay=0.0)
    text = backend.start_text("Hello", 16, 3)
    caption = backend.start_caption(Image.new("RGB", (30, 20)), "What?", 16, 3)
    wait(text); wait(caption)
    assert "Hello" in text.result["text"]
    assert "What?" in caption.result["text"]
