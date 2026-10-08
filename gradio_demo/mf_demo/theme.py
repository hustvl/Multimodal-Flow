"""Gradio theme following the project page palette (blue text, green vision, purple flow)."""

from __future__ import annotations

import gradio as gr


def build_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        button_primary_background_fill="linear-gradient(135deg, #3f72b0 0%, #7a6ba6 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #24497a 0%, #5f5190 100%)",
        button_primary_text_color="white",
        block_shadow="0 1px 2px rgba(16,24,40,.05), 0 4px 14px rgba(16,24,40,.06)",
        block_border_width="1px",
    )
