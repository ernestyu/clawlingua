"""Run-tab event handlers extracted from app.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gradio as gr


@dataclass(frozen=True)
class RunDeps:
    normalize_ui_lang: Callable[[str | None], str]
    tr: Callable[[str, str, str], str]
    run_single_build_v2: Callable[..., dict[str, Any]]
    to_optional_int: Callable[..., int | None]
    to_optional_float: Callable[[Any], float | None]
    as_str: Callable[..., str]
    load_app_config: Callable[[], Any]
    refresh_recent_runs: Callable[..., tuple[Any, str, str | None]]
    load_run_detail: Callable[..., tuple[str, str | None]]
    build_run_analysis: Callable[..., tuple[str, list[list[Any]], list[Any], list[Any], list[Any]]]


def on_run_start(ui_lang_val: str, *, deps: RunDeps) -> tuple[str, None]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    return deps.tr(lang, "Running", "Running"), None


def on_run(
    file_obj: Any,
    deck_title_val: Any,
    src: Any,
    tgt: Any,
    profile: Any,
    mode: Any,
    diff: Any,
    extract_prompt_val: Any,
    explain_prompt_val: Any,
    max_notes_val: Any,
    input_limit_val: Any,
    cloze_min_val: Any,
    chunk_max_val: Any,
    temperature_val: Any,
    save_inter_val: Any,
    continue_on_error_val: Any,
    ui_lang_val: Any,
    *,
    deps: RunDeps,
) -> tuple[Any, ...]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    result = deps.run_single_build_v2(
        uploaded_file=file_obj,
        deck_title=deck_title_val or "",
        source_lang=src,
        target_lang=tgt,
        content_profile=profile,
        learning_mode=mode,
        difficulty=diff,
        max_notes=deps.to_optional_int(max_notes_val, min_value=1),
        input_char_limit=deps.to_optional_int(input_limit_val, min_value=1),
        cloze_min_chars=deps.to_optional_int(cloze_min_val, min_value=0),
        chunk_max_chars=deps.to_optional_int(chunk_max_val, min_value=1),
        temperature=deps.to_optional_float(temperature_val),
        save_intermediate=bool(save_inter_val),
        continue_on_error=bool(continue_on_error_val),
        prompt_lang=lang,
        extract_prompt=deps.as_str(extract_prompt_val),
        explain_prompt=deps.as_str(explain_prompt_val),
    )
    cfg_now = deps.load_app_config()
    run_id = deps.as_str(result.get("run_id")) or None
    selector_update, detail_md, history_download = deps.refresh_recent_runs(
        cfg_now,
        lang=lang,
        preferred_run_id=run_id,
    )
    (
        analysis_md,
        sample_rows,
        taxonomy_choices,
        rejection_choices,
        chunk_choices,
    ) = deps.build_run_analysis(
        run_id,
        cfg_now,
        lang=lang,
        taxonomy_filter="all",
        transfer_filter="all",
        rejection_filter="all",
        chunk_filter="all",
    )
    taxonomy_update = gr.update(choices=taxonomy_choices, value="all")
    rejection_update = gr.update(choices=rejection_choices, value="all")
    chunk_update = gr.update(choices=chunk_choices, value="all")
    transfer_update = gr.update(value="all")

    if result.get("status") != "ok":
        msg = result.get("message") or "Unknown error"
        run_line = f"- run_id: `{run_id}`\n" if run_id else ""
        status_md = (
            f"{deps.tr(lang, 'Failed', 'Failed')}\n\n"
            f"{run_line}- {deps.tr(lang, 'Error', 'Error')}: `{msg}`"
        )
        return (
            status_md,
            None,
            selector_update,
            detail_md,
            history_download,
            analysis_md,
            sample_rows,
            taxonomy_update,
            transfer_update,
            rejection_update,
            chunk_update,
        )

    cards = result["cards_count"]
    errors = result["errors_count"]
    out_path = result["output_path"]
    status_md = (
        f"{deps.tr(lang, 'Completed', 'Completed')}\n\n"
        f"- run_id: `{run_id}`\n"
        f"- cards: **{cards}**\n"
        f"- errors: **{errors}**\n"
        f"- output: `{out_path}`"
    )
    return (
        status_md,
        out_path,
        selector_update,
        detail_md,
        history_download,
        analysis_md,
        sample_rows,
        taxonomy_update,
        transfer_update,
        rejection_update,
        chunk_update,
    )


def on_refresh_runs(
    ui_lang_val: str,
    selected_run_id: str | None,
    *,
    deps: RunDeps,
) -> tuple[Any, str, str | None, str, list[list[Any]], Any, Any, Any, Any]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    cfg_now = deps.load_app_config()
    selector_update, detail_md, history_download = deps.refresh_recent_runs(
        cfg_now,
        lang=lang,
        preferred_run_id=selected_run_id,
    )
    run_id_next = deps.as_str(
        selector_update.get("value") if isinstance(selector_update, dict) else selected_run_id
    )
    (
        analysis_md,
        sample_rows,
        taxonomy_choices,
        rejection_choices,
        chunk_choices,
    ) = deps.build_run_analysis(
        run_id_next,
        cfg_now,
        lang=lang,
        taxonomy_filter="all",
        transfer_filter="all",
        rejection_filter="all",
        chunk_filter="all",
    )
    return (
        selector_update,
        detail_md,
        history_download,
        analysis_md,
        sample_rows,
        gr.update(choices=taxonomy_choices, value="all"),
        gr.update(value="all"),
        gr.update(choices=rejection_choices, value="all"),
        gr.update(choices=chunk_choices, value="all"),
    )


def on_run_selected(
    run_id_val: str | None,
    ui_lang_val: str,
    *,
    deps: RunDeps,
) -> tuple[str, str | None, str, list[list[Any]], Any, Any, Any, Any]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    cfg_now = deps.load_app_config()
    detail_md, download_path = deps.load_run_detail(run_id_val, cfg_now, lang=lang)
    (
        analysis_md,
        sample_rows,
        taxonomy_choices,
        rejection_choices,
        chunk_choices,
    ) = deps.build_run_analysis(
        run_id_val,
        cfg_now,
        lang=lang,
        taxonomy_filter="all",
        transfer_filter="all",
        rejection_filter="all",
        chunk_filter="all",
    )
    return (
        detail_md,
        download_path,
        analysis_md,
        sample_rows,
        gr.update(choices=taxonomy_choices, value="all"),
        gr.update(value="all"),
        gr.update(choices=rejection_choices, value="all"),
        gr.update(choices=chunk_choices, value="all"),
    )


def on_apply_analysis_filters(
    run_id_val: str | None,
    ui_lang_val: str,
    taxonomy_val: str,
    transfer_val: str,
    rejection_val: str,
    chunk_val: str,
    *,
    deps: RunDeps,
) -> tuple[str, list[list[Any]]]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    cfg_now = deps.load_app_config()
    analysis_md, sample_rows, _, _, _ = deps.build_run_analysis(
        run_id_val,
        cfg_now,
        lang=lang,
        taxonomy_filter=taxonomy_val or "all",
        transfer_filter=transfer_val or "all",
        rejection_filter=rejection_val or "all",
        chunk_filter=chunk_val or "all",
    )
    return analysis_md, sample_rows
