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
from dotenv import dotenv_values
import logging

from clawlingua.config import (
    load_config,
    validate_base_config,
    validate_runtime_config,
)
from clawlingua.pipeline.build_deck import BuildDeckOptions, run_build_deck
from clawlingua.logger import setup_logging

logger = logging.getLogger("clawlingua.web")


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


# Keys that the Config tab allows editing. These map directly to
# CLAWLINGUA_* environment variables used by AppConfig.
_EDITABLE_ENV_KEYS = [
    # Defaults / language
    "CLAWLINGUA_DEFAULT_SOURCE_LANG",
    "CLAWLINGUA_DEFAULT_TARGET_LANG",
    # LLM (primary)
    "CLAWLINGUA_LLM_BASE_URL",
    "CLAWLINGUA_LLM_API_KEY",
    "CLAWLINGUA_LLM_MODEL",
    "CLAWLINGUA_LLM_TIMEOUT_SECONDS",
    "CLAWLINGUA_LLM_MAX_RETRIES",
    "CLAWLINGUA_LLM_RETRY_BACKOFF_SECONDS",
    "CLAWLINGUA_LLM_REQUEST_SLEEP_SECONDS",
    "CLAWLINGUA_LLM_TEMPERATURE",
    # Translation LLM
    "CLAWLINGUA_TRANSLATE_LLM_BASE_URL",
    "CLAWLINGUA_TRANSLATE_LLM_API_KEY",
    "CLAWLINGUA_TRANSLATE_LLM_MODEL",
    "CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE",
    # Chunk & cloze
    "CLAWLINGUA_CHUNK_MAX_CHARS",
    "CLAWLINGUA_CHUNK_MIN_CHARS",
    "CLAWLINGUA_CHUNK_OVERLAP_SENTENCES",
    "CLAWLINGUA_CLOZE_MAX_SENTENCES",
    "CLAWLINGUA_CLOZE_MIN_CHARS",
    "CLAWLINGUA_CLOZE_MAX_PER_CHUNK",
    "CLAWLINGUA_LLM_CHUNK_BATCH_SIZE",
    "CLAWLINGUA_INGEST_SHORT_LINE_MAX_WORDS",
    "CLAWLINGUA_CONTENT_PROFILE",
    "CLAWLINGUA_CLOZE_DIFFICULTY",
    "CLAWLINGUA_PROMPT_LANG",
    # Paths & defaults
    "CLAWLINGUA_OUTPUT_DIR",
    "CLAWLINGUA_EXPORT_DIR",
    "CLAWLINGUA_LOG_DIR",
    "CLAWLINGUA_DEFAULT_DECK_NAME",
    # TTS (common voices)
    "CLAWLINGUA_TTS_EDGE_EN_VOICES",
    "CLAWLINGUA_TTS_EDGE_ZH_VOICES",
    "CLAWLINGUA_TTS_EDGE_JA_VOICES",
]


def _load_env_view(cfg: Any, env_file: Optional[Path]) -> Dict[str, str]:
    """Build a view of config values for the Config tab.

    Preference order per key:
    - If key is present in .env, use its string value
    - Otherwise, fall back to AppConfig attribute when possible
    - Otherwise, empty string
    """

    file_values: Dict[str, str] = {}
    if env_file is not None and env_file.exists():
        for k, v in dotenv_values(env_file).items():
            if v is not None:
                file_values[k] = str(v)

    view: Dict[str, str] = {}
    for key in _EDITABLE_ENV_KEYS:
        if key in file_values:
            view[key] = file_values[key]
            continue
        # Fallback: derive from cfg when possible.
        attr_name = key.removeprefix("CLAWLINGUA_").lower()
        if hasattr(cfg, attr_name):
            value = getattr(cfg, attr_name)
            view[key] = "" if value is None else str(value)
        else:
            view[key] = ""
    return view


def _save_env(updated: Dict[str, str]) -> str:
    """Persist selected config values back to .env and validate.

    Behavior:
    - Only CLAWLINGUA_* keys in _EDITABLE_ENV_KEYS are modified.
    - Other keys in existing .env are preserved as-is.
    - Empty string means "remove from .env" so defaults apply.
    - On validation failure, the original .env content is restored.
    """

    env_file = _resolve_env_file() or Path(".env").resolve()
    original_text: Optional[str] = None
    if env_file.exists():
        original_text = env_file.read_text(encoding="utf-8")
        current = {k: v for k, v in dotenv_values(env_file).items() if v is not None}
    else:
        current = {}

    # Preserve non-editable keys
    new_env: Dict[str, str] = {
        k: str(v) for k, v in current.items() if k not in _EDITABLE_ENV_KEYS
    }

    # Apply edits
    for key in _EDITABLE_ENV_KEYS:
        val = updated.get(key)
        if val is None:
            continue
        val_str = str(val).strip()
        if val_str == "":
            # Empty string => drop from .env so defaults apply
            new_env.pop(key, None)
        else:
            new_env[key] = val_str

    # Write new .env content (simple KEY=VALUE lines, no comments)
    lines = [f"{k}={v}\n" for k, v in sorted(new_env.items())]
    env_file.write_text("".join(lines), encoding="utf-8")

    # Validate new config
    try:
        cfg = load_config(env_file=env_file)
        validate_base_config(cfg)
        validate_runtime_config(cfg)
    except Exception as exc:  # pragma: no cover - defensive
        # Roll back on failure
        if original_text is not None:
            env_file.write_text(original_text, encoding="utf-8")
        else:
            env_file.unlink(missing_ok=True)
        return f"❌ Failed to save config: {exc}"

    return "✅ Config saved and validated."


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
            gr.Markdown("### Config (.env editor)")
            env_file = _resolve_env_file()
            cfg_view = _load_env_view(cfg, env_file)

            with gr.Accordion("LLM (primary)", open=True):
                llm_base_url = gr.Textbox(
                    label="CLAWLINGUA_LLM_BASE_URL",
                    value=cfg_view.get("CLAWLINGUA_LLM_BASE_URL", ""),
                )
                llm_api_key = gr.Textbox(
                    label="CLAWLINGUA_LLM_API_KEY",
                    value=cfg_view.get("CLAWLINGUA_LLM_API_KEY", ""),
                    type="password",
                )
                llm_model = gr.Textbox(
                    label="CLAWLINGUA_LLM_MODEL",
                    value=cfg_view.get("CLAWLINGUA_LLM_MODEL", ""),
                )
                llm_timeout = gr.Textbox(
                    label="CLAWLINGUA_LLM_TIMEOUT_SECONDS",
                    value=cfg_view.get("CLAWLINGUA_LLM_TIMEOUT_SECONDS", "120"),
                )
                llm_temperature_env = gr.Textbox(
                    label="CLAWLINGUA_LLM_TEMPERATURE",
                    value=cfg_view.get("CLAWLINGUA_LLM_TEMPERATURE", "0.2"),
                )

            with gr.Accordion("Translation LLM", open=False):
                translate_base_url = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_BASE_URL",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_BASE_URL", ""),
                )
                translate_api_key = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_API_KEY",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_API_KEY", ""),
                    type="password",
                )
                translate_model = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_MODEL",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_MODEL", ""),
                )

            with gr.Accordion("Chunk & Cloze", open=False):
                chunk_max_chars_env = gr.Textbox(
                    label="CLAWLINGUA_CHUNK_MAX_CHARS",
                    value=cfg_view.get("CLAWLINGUA_CHUNK_MAX_CHARS", "1800"),
                )
                chunk_min_chars_env = gr.Textbox(
                    label="CLAWLINGUA_CHUNK_MIN_CHARS",
                    value=cfg_view.get("CLAWLINGUA_CHUNK_MIN_CHARS", "120"),
                )
                cloze_min_chars_env = gr.Textbox(
                    label="CLAWLINGUA_CLOZE_MIN_CHARS",
                    value=cfg_view.get("CLAWLINGUA_CLOZE_MIN_CHARS", "0"),
                )
                cloze_max_per_chunk_env = gr.Textbox(
                    label="CLAWLINGUA_CLOZE_MAX_PER_CHUNK",
                    value=cfg_view.get("CLAWLINGUA_CLOZE_MAX_PER_CHUNK", ""),
                )
                content_profile_env = gr.Textbox(
                    label="CLAWLINGUA_CONTENT_PROFILE",
                    value=cfg_view.get("CLAWLINGUA_CONTENT_PROFILE", "general"),
                )
                cloze_difficulty_env = gr.Textbox(
                    label="CLAWLINGUA_CLOZE_DIFFICULTY",
                    value=cfg_view.get("CLAWLINGUA_CLOZE_DIFFICULTY", "intermediate"),
                )
                prompt_lang_env = gr.Textbox(
                    label="CLAWLINGUA_PROMPT_LANG",
                    value=cfg_view.get("CLAWLINGUA_PROMPT_LANG", "zh"),
                    info="Prompt language for multi-lingual prompts (en|zh).",
                )

            with gr.Accordion("Paths & defaults", open=False):
                output_dir_env = gr.Textbox(
                    label="CLAWLINGUA_OUTPUT_DIR",
                    value=cfg_view.get("CLAWLINGUA_OUTPUT_DIR", "./runs"),
                )
                export_dir_env = gr.Textbox(
                    label="CLAWLINGUA_EXPORT_DIR",
                    value=cfg_view.get("CLAWLINGUA_EXPORT_DIR", "./outputs"),
                )
                log_dir_env = gr.Textbox(
                    label="CLAWLINGUA_LOG_DIR",
                    value=cfg_view.get("CLAWLINGUA_LOG_DIR", "./logs"),
                )
                default_deck_name_env = gr.Textbox(
                    label="CLAWLINGUA_DEFAULT_DECK_NAME",
                    value=cfg_view.get("CLAWLINGUA_DEFAULT_DECK_NAME", cfg.default_deck_name),
                )

            with gr.Row():
                load_defaults_btn = gr.Button("Load defaults from ENV_EXAMPLE.md")
                save_config_btn = gr.Button("Save config")
            save_config_status = gr.Markdown()
@@
            save_config_btn.click(
                _on_save_config,
@@
                outputs=[save_config_status],
            )
+
+            def _on_load_defaults():
+                defaults = _read_env_example()
+
+                def dv(key: str, current: str) -> str:
+                    return defaults.get(key, current or "")
+
+                return (
+                    dv("CLAWLINGUA_LLM_BASE_URL", llm_base_url.value),
+                    dv("CLAWLINGUA_LLM_API_KEY", llm_api_key.value),
+                    dv("CLAWLINGUA_LLM_MODEL", llm_model.value),
+                    dv("CLAWLINGUA_LLM_TIMEOUT_SECONDS", llm_timeout.value),
+                    dv("CLAWLINGUA_LLM_TEMPERATURE", llm_temperature_env.value),
+                    dv("CLAWLINGUA_TRANSLATE_LLM_BASE_URL", translate_base_url.value),
+                    dv("CLAWLINGUA_TRANSLATE_LLM_API_KEY", translate_api_key.value),
+                    dv("CLAWLINGUA_TRANSLATE_LLM_MODEL", translate_model.value),
+                    dv("CLAWLINGUA_CHUNK_MAX_CHARS", chunk_max_chars_env.value),
+                    dv("CLAWLINGUA_CHUNK_MIN_CHARS", chunk_min_chars_env.value),
+                    dv("CLAWLINGUA_CLOZE_MIN_CHARS", cloze_min_chars_env.value),
+                    dv("CLAWLINGUA_CLOZE_MAX_PER_CHUNK", cloze_max_per_chunk_env.value),
+                    dv("CLAWLINGUA_CONTENT_PROFILE", content_profile_env.value),
+                    dv("CLAWLINGUA_CLOZE_DIFFICULTY", cloze_difficulty_env.value),
+                    dv("CLAWLINGUA_PROMPT_LANG", prompt_lang_env.value),
+                    dv("CLAWLINGUA_OUTPUT_DIR", output_dir_env.value),
+                    dv("CLAWLINGUA_EXPORT_DIR", export_dir_env.value),
+                    dv("CLAWLINGUA_LOG_DIR", log_dir_env.value),
+                    dv("CLAWLINGUA_DEFAULT_DECK_NAME", default_deck_name_env.value),
+                    "Loaded defaults from ENV_EXAMPLE.md (not yet saved).",
+                )
+
+            load_defaults_btn.click(
+                _on_load_defaults,
+                inputs=[],
+                outputs=[
+                    llm_base_url,
+                    llm_api_key,
+                    llm_model,
+                    llm_timeout,
+                    llm_temperature_env,
+                    translate_base_url,
+                    translate_api_key,
+                    translate_model,
+                    chunk_max_chars_env,
+                    chunk_min_chars_env,
+                    cloze_min_chars_env,
+                    cloze_max_per_chunk_env,
+                    content_profile_env,
+                    cloze_difficulty_env,
+                    prompt_lang_env,
+                    output_dir_env,
+                    export_dir_env,
+                    log_dir_env,
+                    default_deck_name_env,
+                    save_config_status,
+                ],
+            )

            def _on_save_config(
                llm_base_url_val,
                llm_api_key_val,
                llm_model_val,
                llm_timeout_val,
                llm_temperature_val,
                translate_base_url_val,
                translate_api_key_val,
                translate_model_val,
                chunk_max_chars_val,
                chunk_min_chars_val,
                cloze_min_chars_val,
                cloze_max_per_chunk_val,
                content_profile_val,
                cloze_difficulty_val,
                prompt_lang_val,
                output_dir_val,
                export_dir_val,
                log_dir_val,
                default_deck_name_val,
            ):
                updated = {
                    "CLAWLINGUA_LLM_BASE_URL": llm_base_url_val or "",
                    "CLAWLINGUA_LLM_API_KEY": llm_api_key_val or "",
                    "CLAWLINGUA_LLM_MODEL": llm_model_val or "",
                    "CLAWLINGUA_LLM_TIMEOUT_SECONDS": llm_timeout_val or "",
                    "CLAWLINGUA_LLM_TEMPERATURE": llm_temperature_val or "",
                    "CLAWLINGUA_TRANSLATE_LLM_BASE_URL": translate_base_url_val or "",
                    "CLAWLINGUA_TRANSLATE_LLM_API_KEY": translate_api_key_val or "",
                    "CLAWLINGUA_TRANSLATE_LLM_MODEL": translate_model_val or "",
                    "CLAWLINGUA_CHUNK_MAX_CHARS": chunk_max_chars_val or "",
                    "CLAWLINGUA_CHUNK_MIN_CHARS": chunk_min_chars_val or "",
                    "CLAWLINGUA_CLOZE_MIN_CHARS": cloze_min_chars_val or "",
                    "CLAWLINGUA_CLOZE_MAX_PER_CHUNK": cloze_max_per_chunk_val or "",
                    "CLAWLINGUA_CONTENT_PROFILE": content_profile_val or "",
                    "CLAWLINGUA_CLOZE_DIFFICULTY": cloze_difficulty_val or "",
                    "CLAWLINGUA_PROMPT_LANG": prompt_lang_val or "",
                    "CLAWLINGUA_OUTPUT_DIR": output_dir_val or "",
                    "CLAWLINGUA_EXPORT_DIR": export_dir_val or "",
                    "CLAWLINGUA_LOG_DIR": log_dir_val or "",
                    "CLAWLINGUA_DEFAULT_DECK_NAME": default_deck_name_val or "",
                }
                msg = _save_env(updated)
                return msg

            save_config_btn.click(
                _on_save_config,
                inputs=[
                    llm_base_url,
                    llm_api_key,
                    llm_model,
                    llm_timeout,
                    llm_temperature_env,
                    translate_base_url,
                    translate_api_key,
                    translate_model,
                    chunk_max_chars_env,
                    chunk_min_chars_env,
                    cloze_min_chars_env,
                    cloze_max_per_chunk_env,
                    content_profile_env,
                    cloze_difficulty_env,
                    prompt_lang_env,
                    output_dir_env,
                    export_dir_env,
                    log_dir_env,
                    default_deck_name_env,
                ],
                outputs=[save_config_status],
            )

    return demo


def launch(*, server_port: int = 7860) -> None:
    """Launch the Gradio app bound to 127.0.0.1.

    Logging is configured via the shared `setup_logging` function when
    loading the application config. Web-specific events are logged under
    the `clawlingua.web` logger.
    """

    logger.info("starting ClawLingua web UI | port=%d", server_port)
    demo = build_interface()
    demo.queue().launch(server_name="127.0.0.1", server_port=server_port)
    logger.info("ClawLingua web UI stopped")


if __name__ == "__main__":  # pragma: no cover
    launch()
