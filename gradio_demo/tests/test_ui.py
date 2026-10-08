import socket

import pytest
from gradio_client import Client, handle_file

from mf_demo.backend import MockBackend
from mf_demo.ui import TITLE_HTML, blocks_kwargs, build_ui, find_asset, launch_kwargs, page_js, style_kwargs


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def client():
    backend = MockBackend(delay=0.0)
    demo = build_ui(backend, surprise=True)
    port = free_port()
    demo.queue(max_size=16, default_concurrency_limit=8).launch(
        server_port=port, prevent_thread_lock=True, quiet=True, **launch_kwargs()
    )
    yield Client(f"http://127.0.0.1:{port}", verbose=False)
    demo.close()


def test_style_kwargs_contain_css_js_theme():
    kw = style_kwargs()
    assert ".mf-progress" in kw["css"] and "prefers-reduced-motion" in kw["css"]
    assert "mf-reveal" in kw["js"] and "MutationObserver" in kw["js"]
    assert kw["theme"] is not None


def test_styling_goes_where_this_gradio_version_expects_it():
    import gradio as gr

    major = int(gr.__version__.split(".")[0])
    assert ("css" in blocks_kwargs()) == (major < 6)
    assert ("css" in launch_kwargs()) == (major >= 6)
    assert "allowed_paths" in launch_kwargs()


def test_example_assets_resolve_in_repo():
    for name in ("astronaut", "venice", "nebula", "architecture"):
        assert find_asset(name), name


def test_generate_image_endpoint(client):
    # State components (history, last seed, run id) are not part of the client API
    progress, image, meta, gallery = client.predict(
        "a red bicycle", 8, 3.0, "sde", 5, False, api_name="/generate_image"
    )
    assert progress == ""  # progress card is cleared on success
    assert image and "seed 5" in meta and "8 steps" in meta and "SDE" in meta
    assert len(gallery) >= 1


def test_random_seed_is_reported(client):
    first = client.predict("x", 4, 3.0, "ode", 0, True, api_name="/generate_image")[2]
    second = client.predict("x", 4, 3.0, "ode", 0, True, api_name="/generate_image")[2]
    assert "seed " in first and first != second  # random seeds differ and are reported


def test_empty_prompt_is_a_friendly_error(client):
    with pytest.raises(Exception) as err:
        client.predict("   ", 8, 3.0, "sde", 0, True, api_name="/generate_image")
    assert "prompt" in str(err.value).lower()


def test_history_is_capped_at_eight(client):
    fresh = Client(client.src, verbose=False)  # a new session has its own history
    sizes = []
    for i in range(10):
        out = fresh.predict(f"p{i}", 4, 3.0, "sde", i, False, api_name="/generate_image")
        sizes.append(len(out[3]))
    assert sizes[:3] == [1, 2, 3] and max(sizes) == 8


def test_surprise_endpoint(client):
    out = client.predict(4, 3.0, "sde", 3, False, api_name="/surprise")
    assert out[1] and "seed 3" in out[2] and "CFG" not in out[2]


def test_text_continue_endpoint(client):
    progress, answer = client.predict("A short language model can", 32, 9, False, api_name="/continue_text")
    assert progress == "" and "mf-reveal" in answer and "A short language model can" in answer
    assert "seed 9" in answer


def test_describe_endpoint_escapes_html(client):
    image = find_asset("astronaut")
    _, answer = client.predict(handle_file(image), "<b>What</b> is it?", 32, 1, False, api_name="/describe_image")
    assert "&lt;b&gt;What&lt;/b&gt;" in answer and "<b>What</b>" not in answer


def test_describe_requires_image_and_question(client):
    with pytest.raises(Exception) as err:
        client.predict(None, "Describe this image.", 32, 1, False, api_name="/describe_image")
    assert "image" in str(err.value).lower()
    with pytest.raises(Exception) as err:
        client.predict(handle_file(find_asset("astronaut")), "  ", 32, 1, False, api_name="/describe_image")
    assert "question" in str(err.value).lower()


def test_job_cancel_frees_the_queue():
    backend = MockBackend(delay=0.01)
    demo = build_ui(backend)
    port = free_port()
    demo.queue(max_size=16, default_concurrency_limit=8).launch(
        server_port=port, prevent_thread_lock=True, quiet=True, **launch_kwargs()
    )
    try:
        client = Client(f"http://127.0.0.1:{port}", verbose=False)
        job = client.submit("slow", 128, 3.0, "sde", 1, False, api_name="/generate_image")
        import time
        time.sleep(0.6)
        job.cancel()
        time.sleep(0.5)
        # the closed generator must cancel the backend run, not leave the GPU busy
        assert all(r.finished for r in backend._runs.values())
        out = client.predict("next", 4, 3.0, "sde", 2, False, api_name="/generate_image")
        assert out[1]
    finally:
        demo.close()


def test_page_js_has_the_shape_each_gradio_version_expects():
    import gradio as gr

    js = page_js().strip()
    if int(gr.__version__.split(".")[0]) < 6:
        # evaluated as `(<js>)()`: must be one function expression with no trailing ';'
        assert js.startswith("() => {") and js.endswith("}") and not js.endswith(";")
    else:
        assert js.endswith("})();")


def test_small_title_is_on_the_page():
    assert "<h1>" in TITLE_HTML and "MF-1" in TITLE_HTML and TITLE_HTML.count("<h1>") == 1
    assert ".mf-title" in style_kwargs()["css"]
    demo = build_ui(MockBackend(delay=0.0))
    values = [str(getattr(b, "value", "")) for b in demo.blocks.values()]
    assert any('class="mf-title"' in v for v in values)
    assert not any("mf-hero" in v or "mf-flowfield" in v for v in values)  # the big banner stays gone


def test_css_gradients_do_not_use_var():
    """Gradio's CSS handling drops a background whose gradient uses a variable, which hid the title's
    gradient text and the progress-bar fill. Gradients must use literal colors."""
    import re

    css = style_kwargs()["css"]
    assert "linear-gradient(" in css
    assert not re.search(r"gradient\([^;]*var\(", css)
