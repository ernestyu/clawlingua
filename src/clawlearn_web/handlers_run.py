"""Run-tab event handlers extracted from app.py."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterator

import gradio as gr

from clawlearn.pipeline.build_lingua_deck import BuildDeckOptions, run_build_lingua_deck
from clawlearn.pipeline.build_textbook_deck import (
    BuildTextbookDeckOptions,
    run_build_textbook_deck,
)
from clawlearn.utils.time import make_run_id, utc_now_iso
from clawlearn_web import run_history, upload_io

_RUN_PROGRESS_POLL_SECONDS = 2.0


@dataclass(frozen=True)
class RunDeps:
    normalize_ui_lang: Callable[[str | None], str]
    tr: Callable[[str, str, str], str]
    run_single_build: Callable[..., dict[str, Any]]
    to_optional_int: Callable[..., int | None]
    to_optional_float: Callable[[Any], float | None]
    as_str: Callable[..., str]
    load_app_config: Callable[[], Any]
    refresh_recent_runs: Callable[..., tuple[Any, str, str | None]]
    load_run_detail: Callable[..., tuple[str, str | None]]
    build_run_analysis: Callable[..., tuple[str, list[list[Any]], list[Any], list[Any], list[Any]]]


@dataclass(frozen=True)
class RunServiceDeps:
    load_app_config: Callable[[], Any]
    normalize_ui_lang: Callable[[str | None], str]
    as_str: Callable[..., str]
    logger: logging.Logger | None = None


def run_single_build(
    uploaded_file: Any,
    deck_title: str,
    source_lang: str,
    target_lang: str,
    content_profile: str,
    learning_mode: str,
    difficulty: str,
    max_notes: int | None,
    input_char_limit: int | None,
    cloze_min_chars: int | None,
    textbook_max_concepts_per_chunk: int | None,
    textbook_keep_source_excerpt: bool,
    chunk_max_chars: int | None,
    temperature: float | None,
    save_intermediate: bool,
    continue_on_error: bool,
    prompt_lang: str | None = None,
    extract_prompt: str | None = None,
    explain_prompt: str | None = None,
    *,
    deps: RunServiceDeps,
) -> dict[str, Any]:
    if uploaded_file is None:
        return {"status": "error", "message": "No input file provided.", "run_id": None}

    logger = deps.logger or logging.getLogger("clawlearn.web")
    cfg = deps.load_app_config()
    if prompt_lang:
        cfg.prompt_lang = deps.normalize_ui_lang(prompt_lang)
    workspace_root = cfg.workspace_root
    tmp_dir = (workspace_root / "tmp").resolve()
    try:
        local_input = upload_io.materialize_uploaded_file(uploaded_file, tmp_dir)
    except Exception as exc:
        logger.exception("failed to materialize uploaded file")
        return {"status": "error", "message": str(exc), "run_id": None}

    run_id = make_run_id()
    run_dir = cfg.resolve_path(cfg.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    default_output_path = (cfg.resolve_path(cfg.export_dir) / run_id / "output.apkg").resolve()
    summary_path = run_dir / "run_summary.json"

    source_lang_value = source_lang or cfg.default_source_lang
    target_lang_value = target_lang or cfg.default_target_lang
    learning_mode_value = str(learning_mode or cfg.learning_mode or "").strip().lower()
    if learning_mode_value.startswith("textbook_"):
        domain_value = "textbook"
    else:
        domain_value = "lingua"
    profile_value = (
        content_profile or getattr(cfg, "material_profile", None) or cfg.content_profile
    )
    title_value = (deck_title or "").strip() or local_input.stem
    run_history.record_run_start(
        summary_path,
        run_id=run_id,
        started_at=utc_now_iso(),
        title=title_value,
        source_lang=source_lang_value,
        target_lang=target_lang_value,
        domain=domain_value,
        content_profile=profile_value,
        learning_mode=learning_mode_value,
        difficulty=difficulty or cfg.cloze_difficulty,
        extract_prompt_override=deps.as_str(extract_prompt),
        explain_prompt_override=deps.as_str(explain_prompt),
        output_path=str(default_output_path),
        cfg=cfg,
        env_snapshot_overrides={
            "CLAWLEARN_MATERIAL_PROFILE": profile_value,
            "CLAWLEARN_LEARNING_MODE": learning_mode_value,
            "CLAWLEARN_EXTRACT_PROMPT": deps.as_str(extract_prompt),
            "CLAWLEARN_EXPLAIN_PROMPT": deps.as_str(explain_prompt),
        },
    )

    try:
        if domain_value == "textbook":
            options = BuildTextbookDeckOptions(
                input_value=str(local_input),
                run_id=run_id,
                source_lang=source_lang or None,
                target_lang=target_lang or None,
                learning_mode=learning_mode_value or None,
                input_char_limit=input_char_limit,
                deck_name=deck_title or None,
                max_chars=chunk_max_chars,
                max_notes=max_notes,
                max_concepts_per_chunk=textbook_max_concepts_per_chunk,
                keep_source_excerpt=textbook_keep_source_excerpt,
                save_intermediate=save_intermediate,
                continue_on_error=continue_on_error,
            )
            result = run_build_textbook_deck(cfg, options)
        else:
            options = BuildDeckOptions(
                input_value=str(local_input),
                run_id=run_id,
                source_lang=source_lang or None,
                target_lang=target_lang or None,
                content_profile=content_profile or None,
                material_profile=content_profile or None,
                learning_mode=learning_mode_value or None,
                input_char_limit=input_char_limit,
                deck_name=deck_title or None,
                max_chars=chunk_max_chars,
                cloze_min_chars=cloze_min_chars,
                max_notes=max_notes,
                temperature=temperature,
                cloze_difficulty=difficulty or None,
                extract_prompt=Path(extract_prompt) if deps.as_str(extract_prompt) else None,
                explain_prompt=Path(explain_prompt) if deps.as_str(explain_prompt) else None,
                save_intermediate=save_intermediate,
                continue_on_error=continue_on_error,
            )
            result = run_build_lingua_deck(cfg, options)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception(
            "web build failed | run_id=%s domain=%s input=%s profile=%s difficulty=%s",
            run_id,
            domain_value,
            str(local_input),
            content_profile or cfg.content_profile,
            difficulty or cfg.cloze_difficulty,
        )
        run_history.record_run_failed(
            summary_path,
            finished_at=utc_now_iso(),
            error=str(exc),
        )
        return {
            "status": "error",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "message": str(exc),
        }

    output_path = str(result.output_path)
    run_history.record_run_completed(
        summary_path,
        finished_at=utc_now_iso(),
        cards=result.cards_count,
        errors=result.errors_count,
        output_path=output_path,
    )

    return {
        "status": "ok",
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "output_path": output_path,
        "cards_count": result.cards_count,
        "errors_count": result.errors_count,
    }


def on_run_start(ui_lang_val: str, *, deps: RunDeps) -> tuple[str, None]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    return deps.tr(lang, "Running", "Running"), None


def _safe_refresh_recent_runs(
    *,
    cfg_now: Any,
    lang: str,
    preferred_run_id: str | None,
    deps: RunDeps,
) -> tuple[Any, str, str | None]:
    fallback_selector = gr.update(value=preferred_run_id)
    fallback_detail = deps.tr(lang, "Run details unavailable", "Run details unavailable")
    fallback_download = None
    try:
        result = deps.refresh_recent_runs(
            cfg_now,
            lang=lang,
            preferred_run_id=preferred_run_id,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return (
            fallback_selector,
            f"{fallback_detail}\n\n- {deps.tr(lang, 'Error', 'Error')}: `{exc}`",
            fallback_download,
        )
    if not isinstance(result, tuple) or len(result) != 3:
        return (
            fallback_selector,
            (
                f"{fallback_detail}\n\n"
                f"- {deps.tr(lang, 'Error', 'Error')}: "
                f"`{deps.tr(lang, 'invalid response while refreshing run history', 'invalid response while refreshing run history')}`"
            ),
            fallback_download,
        )
    selector_update, detail_md, history_download = result
    return selector_update, detail_md, history_download


def _safe_build_run_analysis(
    *,
    run_id: str | None,
    cfg_now: Any,
    lang: str,
    deps: RunDeps,
) -> tuple[str, list[list[Any]], list[Any], list[Any], list[Any]]:
    fallback_title = deps.tr(lang, "Run analytics unavailable", "Run analytics unavailable")
    fallback_error = deps.tr(
        lang,
        "invalid response while building analysis",
        "invalid response while building analysis",
    )
    fallback = (
        (
            f"{fallback_title}\n\n"
            f"- {deps.tr(lang, 'Error', 'Error')}: `{fallback_error}`"
        ),
        [],
        [("all", "all")],
        [("all", "all")],
        [("all", "all")],
    )
    try:
        result = deps.build_run_analysis(
            run_id,
            cfg_now,
            lang=lang,
            taxonomy_filter="all",
            transfer_filter="all",
            rejection_filter="all",
            chunk_filter="all",
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return (
            (
                f"{deps.tr(lang, 'Run analytics unavailable', 'Run analytics unavailable')}\n\n"
                f"- {deps.tr(lang, 'Error', 'Error')}: `{exc}`"
            ),
            [],
            [("all", "all")],
            [("all", "all")],
            [("all", "all")],
        )
    if not isinstance(result, tuple) or len(result) != 5:
        return fallback
    return result


def _safe_build_run_analysis_with_filters(
    *,
    run_id: str | None,
    cfg_now: Any,
    lang: str,
    deps: RunDeps,
    taxonomy_filter: str,
    transfer_filter: str,
    rejection_filter: str,
    chunk_filter: str,
) -> tuple[str, list[list[Any]]]:
    fallback_title = deps.tr(lang, "Run analytics unavailable", "Run analytics unavailable")
    fallback_error = deps.tr(
        lang,
        "invalid response while building analysis",
        "invalid response while building analysis",
    )
    fallback = (
        (
            f"{fallback_title}\n\n"
            f"- {deps.tr(lang, 'Error', 'Error')}: `{fallback_error}`"
        ),
        [],
    )
    try:
        result = deps.build_run_analysis(
            run_id,
            cfg_now,
            lang=lang,
            taxonomy_filter=taxonomy_filter,
            transfer_filter=transfer_filter,
            rejection_filter=rejection_filter,
            chunk_filter=chunk_filter,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return (
            (
                f"{deps.tr(lang, 'Run analytics unavailable', 'Run analytics unavailable')}\n\n"
                f"- {deps.tr(lang, 'Error', 'Error')}: `{exc}`"
            ),
            [],
        )
    if not isinstance(result, tuple) or len(result) != 5:
        return fallback
    analysis_md, sample_rows, _taxonomy_choices, _rejection_choices, _chunk_choices = result
    return analysis_md, sample_rows


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
    textbook_max_concepts_val: Any,
    textbook_keep_excerpt_val: Any,
    chunk_max_val: Any,
    temperature_val: Any,
    save_inter_val: Any,
    continue_on_error_val: Any,
    ui_lang_val: Any,
    *,
    deps: RunDeps,
) -> Iterator[tuple[Any, ...]]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    worker_payload: dict[str, Any] = {}
    worker_error: dict[str, Exception] = {}

    def _run_worker() -> None:
        try:
            worker_payload["result"] = deps.run_single_build(
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
                textbook_max_concepts_per_chunk=deps.to_optional_int(
                    textbook_max_concepts_val, min_value=1
                ),
                textbook_keep_source_excerpt=bool(textbook_keep_excerpt_val),
                chunk_max_chars=deps.to_optional_int(chunk_max_val, min_value=1),
                temperature=deps.to_optional_float(temperature_val),
                save_intermediate=bool(save_inter_val),
                continue_on_error=bool(continue_on_error_val),
                prompt_lang=lang,
                extract_prompt=deps.as_str(extract_prompt_val),
                explain_prompt=deps.as_str(explain_prompt_val),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            worker_error["error"] = exc

    worker = threading.Thread(target=_run_worker, daemon=True)
    worker.start()
    started = time.monotonic()
    while worker.is_alive():
        worker.join(timeout=_RUN_PROGRESS_POLL_SECONDS)
        if worker.is_alive():
            elapsed_seconds = int(max(0.0, time.monotonic() - started))
            status_md = (
                f"{deps.tr(lang, 'Running', 'Running')}\n\n"
                f"- {deps.tr(lang, 'Elapsed', 'Elapsed')}: **{elapsed_seconds}s**"
            )
            yield (
                status_md,
                None,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )

    if "error" in worker_error:
        result = {"status": "error", "message": str(worker_error["error"]), "run_id": None}
    else:
        result = worker_payload.get("result") or {
            "status": "error",
            "message": "run failed without result",
            "run_id": None,
        }

    cfg_now = deps.load_app_config()
    run_id = deps.as_str(result.get("run_id")) or None
    selector_update, detail_md, history_download = _safe_refresh_recent_runs(
        cfg_now=cfg_now,
        lang=lang,
        preferred_run_id=run_id,
        deps=deps,
    )
    (
        analysis_md,
        sample_rows,
        taxonomy_choices,
        rejection_choices,
        chunk_choices,
    ) = _safe_build_run_analysis(
        run_id=run_id,
        cfg_now=cfg_now,
        lang=lang,
        deps=deps,
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
        yield (
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
        return

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
    yield (
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
    selector_update, detail_md, history_download = _safe_refresh_recent_runs(
        cfg_now=cfg_now,
        lang=lang,
        preferred_run_id=selected_run_id,
        deps=deps,
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
    ) = _safe_build_run_analysis(
        run_id=run_id_next,
        cfg_now=cfg_now,
        lang=lang,
        deps=deps,
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
    ) = _safe_build_run_analysis(
        run_id=run_id_val,
        cfg_now=cfg_now,
        lang=lang,
        deps=deps,
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
    analysis_md, sample_rows = _safe_build_run_analysis_with_filters(
        run_id=run_id_val,
        cfg_now=cfg_now,
        lang=lang,
        deps=deps,
        taxonomy_filter=taxonomy_val or "all",
        transfer_filter=transfer_val or "all",
        rejection_filter=rejection_val or "all",
        chunk_filter=chunk_val or "all",
    )
    return analysis_md, sample_rows
