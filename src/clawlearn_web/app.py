"""Local-only web UI for ClawLearn.

This module exposes a thin Gradio-based frontend over the existing
`clawlearn` CLI/pipeline. It does **not** change CLI behavior and is
intended as an optional convenience for users who prefer a browser UI.

Usage (development):

    python -m clawlearn_web.app

This will start a Gradio app bound to 0.0.0.0 by default.
"""

from __future__ import annotations

import logging
import os
import functools
from typing import Any

import gradio as gr

from clawlearn.config import (
    load_config,
)
from clawlearn.logger import setup_logging
from clawlearn_web import (
    config_io,
    handlers_config,
    handlers_prompt,
    handlers_run,
    handlers_ui,
    i18n,
    prompt_io,
    run_analysis,
    run_history,
)
from clawlearn_web.ui import tab_analytics, tab_config, tab_prompt, tab_run

logger = logging.getLogger("clawlearn.web")

_PROMPT_CONTENT_TYPE_OPTIONS = prompt_io.PROMPT_CONTENT_TYPE_OPTIONS
_PROMPT_LEARNING_MODE_OPTIONS = prompt_io.PROMPT_LEARNING_MODE_OPTIONS
_PROMPT_DIFFICULTY_OPTIONS = prompt_io.PROMPT_DIFFICULTY_OPTIONS


def _load_app_config() -> Any:
    env_file = config_io.resolve_env_file()
    cfg = load_config(env_file=env_file)
    setup_logging(cfg.log_level, log_dir=cfg.log_dir)
    return cfg


def _normalize_ui_lang(value: str | None) -> str:
    return i18n.normalize_ui_lang(value)


def _tr(lang: str, en: str, zh: str) -> str:
    return i18n.tr(lang, en, zh)


def _to_optional_int(value: Any, *, min_value: int | None = None) -> int | None:
    if value in (None, ""):
        return None
    try:
        num = int(float(value))
    except (TypeError, ValueError):
        return None
    if min_value is not None and num < min_value:
        return None
    return num


def _to_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_timeout_seconds(value: Any, default: float = 20.0) -> float:
    timeout = _to_optional_float(value)
    if timeout is None or timeout <= 0:
        return default
    return timeout


def _refresh_recent_runs(
    cfg: Any, *, lang: str, preferred_run_id: str | None = None
) -> tuple[Any, str, str | None]:
    choices, selected, detail, download_path = run_history.recent_runs_view(
        cfg,
        lang=lang,
        tr=_tr,
        preferred_run_id=preferred_run_id,
        limit=30,
    )
    return gr.update(choices=choices, value=selected), detail, download_path


def build_interface() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""

    cfg = _load_app_config()
    try:
        cfg.resolve_extract_prompt_path(
            material_profile=cfg.material_profile,
            difficulty=cfg.cloze_difficulty,
            learning_mode=getattr(cfg, "learning_mode", "expression_mining"),
        )
        cfg.resolve_explain_prompt_path(
            material_profile=cfg.material_profile,
            difficulty=cfg.cloze_difficulty,
            learning_mode=getattr(cfg, "learning_mode", "expression_mining"),
        )
    except Exception:
        logger.warning("prompt auto-seed during web startup failed", exc_info=True)

    prompt_mode_label = functools.partial(prompt_io.prompt_mode_label, tr=_tr)
    prompt_choices_from_map = functools.partial(prompt_io.prompt_choices_from_map, tr=_tr)
    prompt_path_choices = functools.partial(prompt_io.prompt_path_choices, tr=_tr)
    load_prompt_template = functools.partial(prompt_io.load_prompt_template, tr=_tr)
    load_prompt_mode = functools.partial(prompt_io.load_prompt_mode, tr=_tr)
    load_prompt_filter_metadata = functools.partial(
        prompt_io.load_prompt_filter_metadata, tr=_tr
    )
    run_choice_label = functools.partial(run_history.run_choice_label, tr=_tr)
    load_run_detail = functools.partial(run_history.load_run_detail, tr=_tr)
    save_env = functools.partial(config_io.save_env, tr=_tr)

    env_file = config_io.resolve_env_file()
    cfg_view = config_io.load_env_view(cfg, env_file)
    prompt_files = prompt_io.prompt_file_map(cfg)
    initial_ui_lang = _normalize_ui_lang(getattr(cfg, "prompt_lang", "en"))
    initial_run_content_type = prompt_io.normalize_prompt_content_type(cfg.content_profile)
    initial_run_learning_mode = prompt_io.normalize_prompt_learning_mode(
        getattr(cfg, "learning_mode", "expression_mining")
    )
    initial_run_difficulty = prompt_io.normalize_prompt_difficulty(cfg.cloze_difficulty)

    initial_prompt_key = ""
    for key in prompt_files:
        if load_prompt_mode(key, prompt_files, lang=initial_ui_lang) == "extraction":
            initial_prompt_key = key
            break
    if not initial_prompt_key and prompt_files:
        initial_prompt_key = next(iter(prompt_files))

    initial_prompt_text, initial_prompt_status = load_prompt_template(
        initial_prompt_key, prompt_files, lang=initial_ui_lang
    )
    initial_prompt_mode = load_prompt_mode(
        initial_prompt_key, prompt_files, lang=initial_ui_lang
    )
    (
        initial_prompt_content_type,
        initial_prompt_learning_mode,
        initial_prompt_difficulty,
    ) = load_prompt_filter_metadata(
        initial_prompt_key, prompt_files, lang=initial_ui_lang
    )
    initial_prompt_choices = prompt_io.prompt_choices(
        cfg,
        lang=initial_ui_lang,
        tr=_tr,
        mode_filter=initial_prompt_mode or "extraction",
        content_type_filter=initial_prompt_content_type,
        learning_mode_filter=initial_prompt_learning_mode,
        difficulty_filter=initial_prompt_difficulty,
    )

    run_extract_prompt_choices = prompt_path_choices(
        cfg,
        lang=initial_ui_lang,
        mode_filter="extraction",
        content_type_filter=initial_run_content_type,
        learning_mode_filter=initial_run_learning_mode,
        difficulty_filter=initial_run_difficulty,
        include_auto=True,
    )
    run_explain_prompt_choices = prompt_path_choices(
        cfg,
        lang=initial_ui_lang,
        mode_filter="explanation",
        content_type_filter=initial_run_content_type,
        learning_mode_filter=initial_run_learning_mode,
        difficulty_filter=initial_run_difficulty,
        include_auto=True,
    )
    config_extract_prompt_choices = prompt_path_choices(
        cfg,
        lang=initial_ui_lang,
        mode_filter="extraction",
        include_auto=True,
    )
    config_explain_prompt_choices = prompt_path_choices(
        cfg,
        lang=initial_ui_lang,
        mode_filter="explanation",
        include_auto=True,
    )

    initial_runs = run_history.scan_runs(cfg, limit=30)
    if initial_runs:
        initial_run_choices = [
            (run_choice_label(run, lang=initial_ui_lang), run.run_id)
            for run in initial_runs
        ]
        initial_run_selected = initial_runs[0].run_id
        initial_run_detail, initial_run_download = load_run_detail(
            initial_run_selected, cfg, lang=initial_ui_lang
        )
    else:
        initial_run_choices = []
        initial_run_selected = None
        initial_run_detail = _tr(initial_ui_lang, "No runs found.", "No runs found.")
        initial_run_download = None

    build_run_analysis = functools.partial(run_analysis.build_run_analysis, tr=_tr)
    (
        initial_analysis_md,
        initial_samples_rows,
        initial_taxonomy_choices,
        initial_rejection_choices,
        initial_chunk_choices,
    ) = build_run_analysis(
        initial_run_selected,
        cfg,
        lang=initial_ui_lang,
        taxonomy_filter="all",
        transfer_filter="all",
        rejection_filter="all",
        chunk_filter="all",
    )

    run_service_deps = handlers_run.RunServiceDeps(
        load_app_config=_load_app_config,
        normalize_ui_lang=_normalize_ui_lang,
        as_str=run_history.as_str,
        logger=logger,
    )
    run_deps = handlers_run.RunDeps(
        normalize_ui_lang=_normalize_ui_lang,
        tr=_tr,
        run_single_build=functools.partial(
            handlers_run.run_single_build,
            deps=run_service_deps,
        ),
        to_optional_int=_to_optional_int,
        to_optional_float=_to_optional_float,
        as_str=run_history.as_str,
        load_app_config=_load_app_config,
        refresh_recent_runs=_refresh_recent_runs,
        load_run_detail=load_run_detail,
        build_run_analysis=build_run_analysis,
    )
    config_deps = handlers_config.ConfigDeps(
        to_timeout_seconds=_to_timeout_seconds,
        normalize_ui_lang=_normalize_ui_lang,
        read_env_example=config_io.read_env_example,
        tr=_tr,
        save_env_v2=save_env,
    )
    prompt_deps = handlers_prompt.PromptDeps(
        normalize_ui_lang=_normalize_ui_lang,
        tr=_tr,
        load_app_config=_load_app_config,
        normalize_prompt_mode=prompt_io.normalize_prompt_mode,
        normalize_prompt_content_type=prompt_io.normalize_prompt_content_type,
        normalize_prompt_learning_mode=prompt_io.normalize_prompt_learning_mode,
        normalize_prompt_difficulty=prompt_io.normalize_prompt_difficulty,
        prompt_mode_label=prompt_mode_label,
        prompt_file_map=prompt_io.prompt_file_map,
        prompt_choices_from_map=prompt_choices_from_map,
        prompt_path_choices=prompt_path_choices,
        load_prompt_template=load_prompt_template,
        load_prompt_mode=load_prompt_mode,
        load_prompt_filter_metadata=load_prompt_filter_metadata,
        sanitize_prompt_filename=prompt_io.sanitize_prompt_filename,
        prompt_content_type_options=_PROMPT_CONTENT_TYPE_OPTIONS,
        prompt_learning_mode_options=_PROMPT_LEARNING_MODE_OPTIONS,
        prompt_difficulty_options=_PROMPT_DIFFICULTY_OPTIONS,
        prompt_io=prompt_io,
    )

    with gr.Blocks(title="ClawLearn Web UI") as demo:
        with gr.Row():
            ui_lang = gr.Dropdown(
                choices=[("English", "en"), ("中文", "zh")],
                value=initial_ui_lang,
                label=_tr(initial_ui_lang, "UI language", "鐣岄潰璇█"),
                scale=1,
            )
        title_md = gr.Markdown(
            _tr(
                initial_ui_lang,
                "# ClawLearn Web UI\nLocal deck builder for text learning.",
                "# ClawLearn Web UI\nLocal deck builder for text learning.",
            )
        )

        run_ui = tab_run.build_tab(
            initial_ui_lang=initial_ui_lang,
            cfg=cfg,
            initial_run_choices=initial_run_choices,
            initial_run_selected=initial_run_selected,
            initial_run_detail=initial_run_detail,
            initial_run_download=initial_run_download,
            run_extract_prompt_choices=run_extract_prompt_choices,
            run_explain_prompt_choices=run_explain_prompt_choices,
            tr=_tr,
        )
        config_ui = tab_config.build_tab(
            initial_ui_lang=initial_ui_lang,
            cfg=cfg,
            cfg_view=cfg_view,
            config_extract_prompt_choices=config_extract_prompt_choices,
            config_explain_prompt_choices=config_explain_prompt_choices,
            tr=_tr,
        )
        prompt_ui = tab_prompt.build_tab(
            initial_ui_lang=initial_ui_lang,
            initial_prompt_content_type=initial_prompt_content_type,
            initial_prompt_learning_mode=initial_prompt_learning_mode,
            initial_prompt_difficulty=initial_prompt_difficulty,
            initial_prompt_choices=initial_prompt_choices,
            initial_prompt_key=initial_prompt_key,
            initial_prompt_mode=initial_prompt_mode,
            initial_prompt_text=initial_prompt_text,
            initial_prompt_status=initial_prompt_status,
            prompt_content_type_options=_PROMPT_CONTENT_TYPE_OPTIONS,
            prompt_learning_mode_options=_PROMPT_LEARNING_MODE_OPTIONS,
            prompt_difficulty_options=_PROMPT_DIFFICULTY_OPTIONS,
            prompt_mode_label=prompt_mode_label,
            tr=_tr,
        )
        analytics_ui = tab_analytics.build_tab(
            initial_ui_lang=initial_ui_lang,
            initial_analysis_md=initial_analysis_md,
            initial_samples_rows=initial_samples_rows,
            initial_taxonomy_choices=initial_taxonomy_choices,
            initial_rejection_choices=initial_rejection_choices,
            initial_chunk_choices=initial_chunk_choices,
            tr=_tr,
        )

        tab_run.bind_events(
            components=run_ui,
            analytics=analytics_ui,
            ui_lang=ui_lang,
            deps=run_deps,
        )
        tab_config.bind_events(
            components=config_ui,
            ui_lang=ui_lang,
            deps=config_deps,
        )
        tab_prompt.bind_events(
            components=prompt_ui,
            run_tab=run_ui,
            config_tab=config_ui,
            ui_lang=ui_lang,
            deps=prompt_deps,
        )
        tab_analytics.bind_events(
            components=analytics_ui,
            run_selector=run_ui.run_selector,
            ui_lang=ui_lang,
            deps=run_deps,
        )

        ui_deps = handlers_ui.UiDeps(
            normalize_ui_lang=_normalize_ui_lang,
            load_app_config=_load_app_config,
            prompt_file_map=prompt_io.prompt_file_map,
            normalize_prompt_mode=prompt_io.normalize_prompt_mode,
            normalize_prompt_content_type=prompt_io.normalize_prompt_content_type,
            normalize_prompt_learning_mode=prompt_io.normalize_prompt_learning_mode,
            normalize_prompt_difficulty=prompt_io.normalize_prompt_difficulty,
            load_prompt_mode=load_prompt_mode,
            pick_prompt_key=lambda prompt_files_now, **kwargs: handlers_prompt.pick_prompt_key(
                prompt_files_now, deps=prompt_deps, **kwargs
            ),
            load_prompt_template=load_prompt_template,
            load_prompt_filter_metadata=load_prompt_filter_metadata,
            prompt_mode_choices_for_ui=lambda lang: handlers_prompt.prompt_mode_choices_for_ui(
                lang, deps=prompt_deps
            ),
            prompt_path_choices=prompt_path_choices,
            refresh_recent_runs=_refresh_recent_runs,
            normalize_dropdown_value=lambda current, choices: handlers_prompt.normalize_dropdown_value(
                current, choices
            ),
            tr=_tr,
            prompt_choices_from_map=prompt_choices_from_map,
            prompt_content_type_options=_PROMPT_CONTENT_TYPE_OPTIONS,
            prompt_learning_mode_options=_PROMPT_LEARNING_MODE_OPTIONS,
            prompt_difficulty_options=_PROMPT_DIFFICULTY_OPTIONS,
        )

        def _on_ui_lang_change(
            lang_value: str,
            prompt_lang_current: str,
            prompt_key_current: str,
            prompt_mode_current: str,
            prompt_content_type_current: str,
            prompt_learning_mode_current: str,
            prompt_difficulty_current: str,
            run_content_type_current: str,
            run_learning_mode_current: str,
            run_difficulty_current: str,
            run_extract_prompt_current: str,
            run_explain_prompt_current: str,
            run_id_current: str | None,
        ) -> tuple[Any, ...]:
            return handlers_ui.on_ui_lang_change(
                lang_value,
                prompt_lang_current,
                prompt_key_current,
                prompt_mode_current,
                prompt_content_type_current,
                prompt_learning_mode_current,
                prompt_difficulty_current,
                run_content_type_current,
                run_learning_mode_current,
                run_difficulty_current,
                run_extract_prompt_current,
                run_explain_prompt_current,
                run_id_current,
                deps=ui_deps,
            )

        ui_lang.change(
            _on_ui_lang_change,
            inputs=[
                ui_lang,
                config_ui.prompt_lang_env,
                prompt_ui.prompt_file_selector,
                prompt_ui.prompt_mode_selector,
                prompt_ui.prompt_content_type_selector,
                prompt_ui.prompt_learning_mode_selector,
                prompt_ui.prompt_difficulty_selector,
                run_ui.content_profile,
                run_ui.learning_mode,
                run_ui.difficulty,
                run_ui.run_extract_prompt,
                run_ui.run_explain_prompt,
                run_ui.run_selector,
            ],
            outputs=[
                ui_lang,
                title_md,
                run_ui.run_tab,
                config_ui.config_tab,
                prompt_ui.prompt_tab,
                run_ui.input_file,
                run_ui.deck_title,
                run_ui.source_lang,
                run_ui.target_lang,
                run_ui.content_profile,
                run_ui.difficulty,
                run_ui.run_extract_prompt,
                run_ui.run_explain_prompt,
                run_ui.max_notes,
                run_ui.input_char_limit,
                run_ui.run_advanced,
                run_ui.cloze_min_chars,
                run_ui.chunk_max_chars,
                run_ui.temperature,
                run_ui.save_intermediate,
                run_ui.continue_on_error,
                run_ui.run_button,
                run_ui.run_status,
                run_ui.output_file,
                run_ui.recent_runs_heading,
                run_ui.refresh_runs_button,
                run_ui.run_selector,
                run_ui.run_detail,
                run_ui.run_download_file,
                config_ui.config_heading,
                config_ui.llm_accordion,
                config_ui.llm_base_url,
                config_ui.llm_api_key,
                config_ui.llm_model,
                config_ui.llm_timeout,
                config_ui.llm_temperature_env,
                config_ui.llm_list_models_btn,
                config_ui.llm_test_btn,
                config_ui.llm_status,
                config_ui.translate_accordion,
                config_ui.translate_base_url,
                config_ui.translate_api_key,
                config_ui.translate_model,
                config_ui.translate_temperature,
                config_ui.translate_list_models_btn,
                config_ui.translate_test_btn,
                config_ui.translate_status,
                config_ui.chunk_accordion,
                config_ui.chunk_max_chars_env,
                config_ui.cloze_min_chars_env,
                config_ui.cloze_max_per_chunk_env,
                config_ui.prompt_lang_env,
                config_ui.extract_prompt_env,
                config_ui.explain_prompt_env,
                config_ui.paths_accordion,
                config_ui.output_dir_env,
                config_ui.export_dir_env,
                config_ui.log_dir_env,
                config_ui.load_defaults_btn,
                config_ui.save_config_btn,
                config_ui.tts_accordion,
                config_ui.tts_hint_md,
                config_ui.tts_voice1_env,
                prompt_ui.prompt_heading,
                prompt_ui.prompt_file_selector,
                prompt_ui.prompt_mode_selector,
                prompt_ui.prompt_content_type_selector,
                prompt_ui.prompt_learning_mode_selector,
                prompt_ui.prompt_difficulty_selector,
                prompt_ui.prompt_new_name,
                prompt_ui.prompt_rename_name,
                prompt_ui.prompt_editor,
                prompt_ui.prompt_new_btn,
                prompt_ui.prompt_save_btn,
                prompt_ui.prompt_rename_btn,
                prompt_ui.prompt_load_default_btn,
                prompt_ui.prompt_save_confirm,
                prompt_ui.prompt_delete_confirm,
                prompt_ui.prompt_status,
                analytics_ui.analytics_tab,
            ],
        )

    return demo


def launch(*, server_port: int | None = None, server_host: str | None = None) -> None:
    """Launch the Gradio app.

    Logging is configured via the shared `setup_logging` function when
    loading the application config. Web-specific events are logged under
    the `clawlearn.web` logger.
    """

    port_value = server_port
    if port_value is None:
        env_port = _to_optional_int(os.getenv("CLAWLEARN_WEB_PORT"), min_value=1)
        port_value = env_port or 7860
    host_value = (
        server_host or os.getenv("CLAWLEARN_WEB_HOST") or "0.0.0.0"
    ).strip() or "0.0.0.0"

    logger.info("starting ClawLearn web UI | host=%s port=%d", host_value, port_value)
    demo = build_interface()
    demo.queue().launch(server_name=host_value, server_port=port_value)
    logger.info("ClawLearn web UI stopped")


if __name__ == "__main__":  # pragma: no cover
    launch()
