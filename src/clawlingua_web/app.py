"""Local-only web UI for ClawLingua.

This module exposes a thin Gradio-based frontend over the existing
`clawlingua` CLI/pipeline. It does **not** change CLI behavior and is
intended as an optional convenience for users who prefer a browser UI.

Usage (development):

    python -m clawlingua_web.app

This will start a Gradio app bound to 127.0.0.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr

from clawlingua.config import load_config
from clawlingua.pipeline.build_deck import BuildDeckOptions, run_build_deck
from clawlingua.logger import setup_logging


def _resolve_env_file() -> Optional[Path]:
    """Best-effort resolution of the default .env file.

    For now we simply look at the current working directory and use
    ``./.env`` if it exists. This keeps behavior explicit and avoids
    surprising resolution rules.
    """

    candidate = Path(".env").resolve()
    return candidate if candidate.exists() else None


def _load_app_config() -> Any:
    env_file = _resolve_env_file()
    cfg = load_config(env_file=env_file)
    setup_logging(cfg.log_level, log_dir=cfg.log_dir)
    return cfg


def _run_single_build(
    uploaded_file: Any,
    deck_title: str,
    source_lang: str,
    target_lang: str,
    content_profile: str,
    difficulty: str,
    max_notes: Optional[int],
    input_char_limit: Optional[int],
    cloze_min_chars: Optional[int],
    chunk_max_chars: Optional[int],
    temperature: Optional[float],
    save_intermediate: bool,
    continue_on_error: bool,
) -> Dict[str, Any]:
    """Run a single build using the in-process pipeline.

    Returns a dict suitable for Gradio consumption with keys like
    ``status``, ``run_id``, ``cards_count``, ``errors_count`` and
    ``output_path``.
    """

    if uploaded_file is None:
        return {"status": "error", "message": "No input file provided."}

    cfg = _load_app_config()

    # Persist uploaded file to a temporary location under workspace_root/tmp.
    workspace_root = cfg.workspace_root
    tmp_dir = (workspace_root / "tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # gradio's UploadedFile has attributes name, size, and a path-like object.
    # We re-save it to a deterministic path.
    uploaded_path = Path(uploaded_file.name)
    suffix = uploaded_path.suffix or ""
    safe_name = uploaded_path.stem or "input"
    dst = tmp_dir / f"{safe_name}{suffix}"
    with open(dst, "wb") as f_out:
        f_out.write(uploaded_file.read())

    # Build options based on UI inputs.
    options = BuildDeckOptions(
        input_value=str(dst),
        source_lang=source_lang or None,
        target_lang=target_lang or None,
        content_profile=content_profile or None,
        input_char_limit=input_char_limit,
        deck_name=deck_title or None,
        max_chars=chunk_max_chars,
        cloze_min_chars=cloze_min_chars,
        max_notes=max_notes,
        temperature=temperature,
        cloze_difficulty=difficulty or None,
        save_intermediate=save_intermediate,
        continue_on_error=continue_on_error,
    )

    try:
        result = run_build_deck(cfg, options)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {"status": "error", "message": str(exc)}

    return {
        "status": "ok",
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "output_path": str(result.output_path),
        "cards_count": result.cards_count,
        "errors_count": result.errors_count,
    }


def build_interface() -> gr.Blocks:
    """Construct the Gradio Blocks interface.

    Two tabs:
    - Run: upload file + per-run overrides + download link
    - Config: (reserved for future work, not implemented yet)
    """

    cfg = _load_app_config()

    with gr.Blocks(title="ClawLingua Web UI") as demo:
        gr.Markdown("# ClawLingua Web UI\nUpload a local file and build an Anki cloze deck.")

        with gr.Tab("Run"):
            with gr.Row():
                input_file = gr.File(label="Input file (.txt/.md/.epub)", file_types=[".txt", ".md", ".markdown", ".epub"], file_count="single")
                deck_title = gr.Textbox(label="Deck title (optional)")

            with gr.Row():
                source_lang = gr.Dropdown(
                    choices=["en", "zh", "ja"],
                    value=cfg.default_source_lang,
                    label="Source language",
                )
                target_lang = gr.Dropdown(
                    choices=["zh", "en", "ja"],
                    value=cfg.default_target_lang,
                    label="Target language",
                )
                content_profile = gr.Dropdown(
                    choices=["general", "textbook_examples"],
                    value=cfg.content_profile,
                    label="Content profile",
                )
                difficulty = gr.Dropdown(
                    choices=["beginner", "intermediate", "advanced"],
                    value=cfg.cloze_difficulty,
                    label="Difficulty",
                )

            with gr.Row():
                max_notes = gr.Number(label="Max notes (0 = no limit)", value=None, precision=0)
                input_char_limit = gr.Number(label="Input char limit (for quick tests)", value=None, precision=0)

            with gr.Accordion("Advanced", open=False):
                cloze_min_chars = gr.Number(
                    label="Cloze min chars (override env)",
                    value=cfg.cloze_min_chars,
                    precision=0,
                )
                chunk_max_chars = gr.Number(
                    label="Chunk max chars (override env)",
                    value=cfg.chunk_max_chars,
                    precision=0,
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=cfg.llm_temperature,
                    step=0.05,
                    label="LLM temperature (override env)",
                )
                save_intermediate = gr.Checkbox(
                    label="Save intermediate files",
                    value=cfg.save_intermediate,
                )
                continue_on_error = gr.Checkbox(
                    label="Continue on error",
                    value=False,
                )

            run_button = gr.Button("Run")

            status = gr.Markdown(label="Status")
            output_file = gr.File(label="Download .apkg", interactive=False)

            def _on_run(
                file_obj,
                deck_title_val,
                src,
                tgt,
                profile,
                diff,
                max_notes_val,
                input_limit_val,
                cloze_min_val,
                chunk_max_val,
                temperature_val,
                save_inter_val,
                continue_on_error_val,
            ):
                result = _run_single_build(
                    uploaded_file=file_obj,
                    deck_title=deck_title_val or "",
                    source_lang=src,
                    target_lang=tgt,
                    content_profile=profile,
                    difficulty=diff,
                    max_notes=int(max_notes_val) if max_notes_val and max_notes_val > 0 else None,
                    input_char_limit=int(input_limit_val) if input_limit_val and input_limit_val > 0 else None,
                    cloze_min_chars=int(cloze_min_val) if cloze_min_val and cloze_min_val >= 0 else None,
                    chunk_max_chars=int(chunk_max_val) if chunk_max_val and chunk_max_val > 0 else None,
                    temperature=float(temperature_val) if temperature_val is not None else None,
                    save_intermediate=bool(save_inter_val),
                    continue_on_error=bool(continue_on_error_val),
                )
                if result.get("status") != "ok":
                    msg = result.get("message") or "Unknown error"
                    return f"❌ Error: {msg}", None

                run_id = result["run_id"]
                cards = result["cards_count"]
                errors = result["errors_count"]
                out_path = result["output_path"]
                status_md = f"✅ Run complete\n\n- run_id: `{run_id}`\n- cards: **{cards}**\n- errors: **{errors}**\n- output: `{out_path}`"
                return status_md, out_path

            run_button.click(
                _on_run,
                inputs=[
                    input_file,
                    deck_title,
                    source_lang,
                    target_lang,
                    content_profile,
                    difficulty,
                    max_notes,
                    input_char_limit,
                    cloze_min_chars,
                    chunk_max_chars,
                    temperature,
                    save_intermediate,
                    continue_on_error,
                ],
                outputs=[status, output_file],
            )

        with gr.Tab("Config"):
            gr.Markdown(
                """### Config (read-only placeholder)

                This tab is reserved for future work to edit `.env` values
                directly from the browser. For now, please edit `.env`
                manually or use `clawlingua config show/validate`.
                """
            )

    return demo


def launch(*, server_port: int = 7860) -> None:
    """Launch the Gradio app bound to 127.0.0.1."""

    demo = build_interface()
    demo.queue().launch(server_name="127.0.0.1", server_port=server_port)


if __name__ == "__main__":  # pragma: no cover
    launch()
