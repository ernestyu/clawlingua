from __future__ import annotations

import gradio as gr

from clawlearn_web.app import build_interface


def test_build_interface_smoke() -> None:
    demo = build_interface()
    assert isinstance(demo, gr.Blocks)
    if hasattr(demo, "close"):
        demo.close()
