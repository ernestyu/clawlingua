"""Local-only web UI for ClawLingua.

This module exposes a thin Gradio-based frontend over the existing
`clawlingua` CLI/pipeline. It does **not** change CLI behavior and is
intended as an optional convenience for users who prefer a browser UI.

Usage (development):

    python -m clawlingua_web.app

This will start a Gradio app bound to 0.0.0.0 by default.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import shutil
import functools
from typing import Any

import gradio as gr
import httpx

from clawlingua.config import (
    load_config,
)
from clawlingua.logger import setup_logging
from clawlingua.pipeline.build_deck import BuildDeckOptions, run_build_deck
from clawlingua.pipeline.validators import classify_rejection_reason
from clawlingua.utils.time import make_run_id, utc_now_iso
from clawlingua_web import (
    config_io,
    handlers_config,
    handlers_prompt,
    handlers_run,
    handlers_ui,
    i18n,
    prompt_io,
    run_history,
)

logger = logging.getLogger("clawlingua.web")

_PROMPT_CONTENT_TYPE_OPTIONS = prompt_io.PROMPT_CONTENT_TYPE_OPTIONS
_PROMPT_LEARNING_MODE_OPTIONS = prompt_io.PROMPT_LEARNING_MODE_OPTIONS
_PROMPT_DIFFICULTY_OPTIONS = prompt_io.PROMPT_DIFFICULTY_OPTIONS


def _resolve_env_file() -> Path | None:
    return config_io.resolve_env_file()


def _load_app_config() -> Any:
    env_file = _resolve_env_file()
    cfg = load_config(env_file=env_file)
    setup_logging(cfg.log_level, log_dir=cfg.log_dir)
    return cfg


def _normalize_ui_lang(value: str | None) -> str:
    return i18n.normalize_ui_lang(value)


def _tr(lang: str, en: str, zh: str) -> str:
    return i18n.tr(lang, en, zh)


def _read_env_example() -> dict[str, str]:
    return config_io.read_env_example()


def _load_env_view(cfg: Any, env_file: Path | None) -> dict[str, str]:
    return config_io.load_env_view(cfg, env_file)


def _safe_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe or "input"


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


def _materialize_uploaded_file(uploaded_file: Any, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(uploaded_file, (str, Path)):
        src = Path(uploaded_file)
        if not src.exists():
            raise ValueError(f"Uploaded file path does not exist: {src}")
        dst = tmp_dir / f"{_safe_stem(src.stem)}{src.suffix}"
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        return dst

    if isinstance(uploaded_file, dict):
        src_path = uploaded_file.get("path") or uploaded_file.get("name")
        if src_path:
            src = Path(str(src_path))
            if src.exists():
                dst = tmp_dir / f"{_safe_stem(src.stem)}{src.suffix}"
                shutil.copyfile(src, dst)
                return dst

    name = getattr(uploaded_file, "name", "input.txt")
    suffix = Path(str(name)).suffix
    stem = Path(str(name)).stem
    dst = tmp_dir / f"{_safe_stem(stem)}{suffix}"
    if not hasattr(uploaded_file, "read"):
        raise ValueError("Unsupported uploaded file payload.")
    data = uploaded_file.read()
    if isinstance(data, str):
        dst.write_text(data, encoding="utf-8")
    else:
        dst.write_bytes(data)
    return dst


def _run_single_build_v2(
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
    chunk_max_chars: int | None,
    temperature: float | None,
    save_intermediate: bool,
    continue_on_error: bool,
    prompt_lang: str | None = None,
    extract_prompt: str | None = None,
    explain_prompt: str | None = None,
) -> dict[str, Any]:
    if uploaded_file is None:
        return {"status": "error", "message": "No input file provided.", "run_id": None}

    cfg = _load_app_config()
    if prompt_lang:
        cfg.prompt_lang = _normalize_ui_lang(prompt_lang)
    workspace_root = cfg.workspace_root
    tmp_dir = (workspace_root / "tmp").resolve()
    try:
        local_input = _materialize_uploaded_file(uploaded_file, tmp_dir)
    except Exception as exc:
        logger.exception("failed to materialize uploaded file")
        return {"status": "error", "message": str(exc), "run_id": None}

    run_id = make_run_id()
    run_dir = cfg.resolve_path(cfg.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    default_output_path = (
        cfg.resolve_path(cfg.export_dir) / run_id / "output.apkg"
    ).resolve()
    summary_path = run_dir / "run_summary.json"

    source_lang_value = source_lang or cfg.default_source_lang
    target_lang_value = target_lang or cfg.default_target_lang
    profile_value = (
        content_profile or getattr(cfg, "material_profile", None) or cfg.content_profile
    )
    learning_mode_value = learning_mode or cfg.learning_mode
    title_value = (deck_title or "").strip() or local_input.stem
    run_history.record_run_start(
        summary_path,
        run_id=run_id,
        started_at=utc_now_iso(),
        title=title_value,
        source_lang=source_lang_value,
        target_lang=target_lang_value,
        content_profile=profile_value,
        learning_mode=learning_mode_value,
        difficulty=difficulty or cfg.cloze_difficulty,
        extract_prompt_override=_as_str(extract_prompt),
        explain_prompt_override=_as_str(explain_prompt),
        output_path=str(default_output_path),
        cfg=cfg,
    )

    options = BuildDeckOptions(
        input_value=str(local_input),
        run_id=run_id,
        source_lang=source_lang or None,
        target_lang=target_lang or None,
        content_profile=content_profile or None,
        material_profile=content_profile or None,
        learning_mode=learning_mode or None,
        input_char_limit=input_char_limit,
        deck_name=deck_title or None,
        max_chars=chunk_max_chars,
        cloze_min_chars=cloze_min_chars,
        max_notes=max_notes,
        temperature=temperature,
        cloze_difficulty=difficulty or None,
        extract_prompt=Path(extract_prompt) if _as_str(extract_prompt) else None,
        explain_prompt=Path(explain_prompt) if _as_str(explain_prompt) else None,
        save_intermediate=save_intermediate,
        continue_on_error=continue_on_error,
    )
    try:
        result = run_build_deck(cfg, options)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception(
            "web build failed | run_id=%s input=%s profile=%s difficulty=%s",
            run_id,
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


def _save_env_v2(updated: dict[str, str], *, lang: str) -> str:
    return config_io.save_env(updated, lang=lang, tr=_tr)


def _normalize_base_url(base_url: str) -> str:
    value = (base_url or "").strip().rstrip("/")
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")]
    return value


def _build_models_url(base_url: str) -> str:
    root = _normalize_base_url(base_url)
    if not root:
        return ""
    if root.endswith("/models"):
        return root
    return f"{root}/models"


def _request_models(
    base_url: str, api_key: str, timeout_seconds: float
) -> tuple[str, httpx.Response]:
    endpoint = _build_models_url(base_url)
    if not endpoint:
        raise ValueError("missing base URL")
    headers = {"Accept": "application/json"}
    token = (api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.get(endpoint, headers=headers)
    return endpoint, response


def _list_models_markdown(
    base_url: str, api_key: str, timeout_seconds: float, *, lang: str
) -> str:
    if not _normalize_base_url(base_url):
        return f"⚠️ {_tr(lang, 'Missing base URL.', 'Missing base URL.')}"
    try:
        endpoint, response = _request_models(base_url, api_key, timeout_seconds)
    except ValueError:
        return f"⚠️ {_tr(lang, 'Missing base URL.', 'Missing base URL.')}"
    except httpx.RequestError as exc:
        return f"❌ {_tr(lang, 'Request failed', 'Request failed')}: `{exc}`"
    except Exception as exc:
        return f"❌ {_tr(lang, 'Request failed', 'Request failed')}: `{exc}`"

    if response.status_code >= 400:
        body = response.text[:500] if response.text else ""
        return (
            f"❌ {_tr(lang, 'HTTP error', 'HTTP error')}: **{response.status_code}**\n\n"
            f"- endpoint: `{endpoint}`\n"
            f"- body: `{body}`"
        )

    try:
        payload = response.json()
    except ValueError:
        return f"❌ {_tr(lang, 'Response is not valid JSON.', 'Response is not valid JSON.')}"

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return (
            f"⚠️ {_tr(lang, 'Response JSON has no list field `data`.', 'Response JSON has no list field `data`.')}\n\n"
            f"- endpoint: `{endpoint}`\n"
            f"- status: `{response.status_code}`"
        )

    model_ids: list[str] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = _as_str(item.get("id"))
        if not model_id or model_id in seen:
            continue
        model_ids.append(model_id)
        seen.add(model_id)

    if not model_ids:
        return (
            f"✅ {_tr(lang, 'Found models', 'Found models')}: **0**\n\n"
            f"- endpoint: `{endpoint}`\n"
            f"- status: `{response.status_code}`\n"
            f"- {_tr(lang, 'No model ids found in `data`.', 'No model ids found in `data`.')}"
        )

    lines = [f"- `{model_id}`" for model_id in model_ids]
    return (
        f"✅ {_tr(lang, 'Found models', 'Found models')}: **{len(model_ids)}**\n\n"
        f"- endpoint: `{endpoint}`\n"
        f"- status: `{response.status_code}`\n\n"
        f"{chr(10).join(lines)}"
    )


def _test_models_markdown(
    base_url: str, api_key: str, timeout_seconds: float, *, lang: str
) -> str:
    if not _normalize_base_url(base_url):
        return f"⚠️ {_tr(lang, 'Missing base URL.', 'Missing base URL.')}"
    try:
        endpoint, response = _request_models(base_url, api_key, timeout_seconds)
    except ValueError:
        return f"⚠️ {_tr(lang, 'Missing base URL.', 'Missing base URL.')}"
    except httpx.RequestError as exc:
        return f"❌ {_tr(lang, 'Request failed', 'Request failed')}: `{exc}`"
    except Exception as exc:
        return f"❌ {_tr(lang, 'Request failed', 'Request failed')}: `{exc}`"
    return (
        f"✅ {_tr(lang, 'Connectivity OK', 'Connectivity OK')}\n\n"
        f"- endpoint: `{endpoint}`\n"
        f"- status: `{response.status_code}`"
    )


def _normalize_prompt_mode(value: Any) -> str:
    return prompt_io.normalize_prompt_mode(value)


def _normalize_prompt_content_type(value: Any) -> str:
    return prompt_io.normalize_prompt_content_type(value)


def _normalize_prompt_learning_mode(value: Any) -> str:
    return prompt_io.normalize_prompt_learning_mode(value)


def _normalize_prompt_difficulty(value: Any) -> str:
    return prompt_io.normalize_prompt_difficulty(value)


def _prompt_mode_label(mode: str, *, lang: str) -> str:
    return prompt_io.prompt_mode_label(mode, lang=lang, tr=_tr)


def _prompt_file_map(
    cfg: Any,
    *,
    mode_filter: str | None = None,
    content_type_filter: str | None = None,
    learning_mode_filter: str | None = None,
    difficulty_filter: str | None = None,
    include_templates: bool = False,
) -> dict[str, Path]:
    return prompt_io.prompt_file_map(
        cfg,
        mode_filter=mode_filter,
        content_type_filter=content_type_filter,
        learning_mode_filter=learning_mode_filter,
        difficulty_filter=difficulty_filter,
        include_templates=include_templates,
    )


def _prompt_choices_from_map(
    prompt_files: dict[str, Path], *, lang: str
) -> list[tuple[str, str]]:
    return prompt_io.prompt_choices_from_map(prompt_files, lang=lang, tr=_tr)


def _prompt_choices(
    lang: str,
    *,
    mode_filter: str | None = None,
    content_type_filter: str | None = None,
    learning_mode_filter: str | None = None,
    difficulty_filter: str | None = None,
    include_templates: bool = False,
) -> list[tuple[str, str]]:
    cfg = _load_app_config()
    return prompt_io.prompt_choices(
        cfg,
        lang=lang,
        tr=_tr,
        mode_filter=mode_filter,
        content_type_filter=content_type_filter,
        learning_mode_filter=learning_mode_filter,
        difficulty_filter=difficulty_filter,
        include_templates=include_templates,
    )


def _prompt_path_choices(
    cfg: Any,
    *,
    lang: str,
    mode_filter: str,
    content_type_filter: str | None = None,
    learning_mode_filter: str | None = None,
    difficulty_filter: str | None = None,
    include_auto: bool = False,
) -> list[tuple[str, str]]:
    return prompt_io.prompt_path_choices(
        cfg,
        lang=lang,
        tr=_tr,
        mode_filter=mode_filter,
        content_type_filter=content_type_filter,
        learning_mode_filter=learning_mode_filter,
        difficulty_filter=difficulty_filter,
        include_auto=include_auto,
    )


def _sanitize_prompt_filename(raw: str) -> str:
    return prompt_io.sanitize_prompt_filename(raw)


def _load_prompt_template(
    prompt_key: str,
    prompt_files: dict[str, Path],
    *,
    lang: str,
) -> tuple[str, str]:
    return prompt_io.load_prompt_template(prompt_key, prompt_files, lang=lang, tr=_tr)


def _load_prompt_mode(
    prompt_key: str,
    prompt_files: dict[str, Path],
    *,
    lang: str,
) -> str:
    return prompt_io.load_prompt_mode(prompt_key, prompt_files, lang=lang, tr=_tr)


def _load_prompt_filter_metadata(
    prompt_key: str,
    prompt_files: dict[str, Path],
    *,
    lang: str,
) -> tuple[str, str, str]:
    return prompt_io.load_prompt_filter_metadata(
        prompt_key, prompt_files, lang=lang, tr=_tr
    )


RunInfo = run_history.RunInfo


def _as_str(value: Any, *, default: str = "") -> str:
    return run_history.as_str(value, default=default)


def _as_int(value: Any, *, default: int = 0) -> int:
    return run_history.as_int(value, default=default)


def _read_run_summary(path: Path) -> dict[str, Any]:
    return run_history.read_run_summary(path)


def _scan_runs(cfg: Any, *, limit: int = 30) -> list[RunInfo]:
    return run_history.scan_runs(cfg, limit=limit)


def _run_choice_label(info: RunInfo, *, lang: str) -> str:
    return run_history.run_choice_label(info, lang=lang, tr=_tr)


def _load_run_detail(
    run_id: str | None, cfg: Any, *, lang: str
) -> tuple[str, str | None]:
    return run_history.load_run_detail(run_id, cfg, lang=lang, tr=_tr)


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


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _bar(value: float, max_value: float, *, width: int = 18) -> str:
    if max_value <= 0:
        return "." * width
    ratio = max(0.0, min(1.0, value / max_value))
    filled = int(round(ratio * width))
    return ("#" * filled) + ("-" * max(0, width - filled))


def _render_count_histogram(title: str, hist: dict[str, int]) -> str:
    if not hist:
        return f"#### {title}\n- n/a"
    max_value = max(hist.values())
    lines = [f"#### {title}"]
    for key, value in sorted(hist.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- `{key}` `{value}` {_bar(float(value), float(max_value))}")
    return "\n".join(lines)


def _render_score_histogram(title: str, hist: dict[str, float]) -> str:
    if not hist:
        return f"#### {title}\n- n/a"
    max_value = max(hist.values())
    lines = [f"#### {title}"]
    for key, value in sorted(hist.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- `{key}` `{value:.3f}` {_bar(float(value), float(max_value))}")
    return "\n".join(lines)


def _short_text(value: str, *, limit: int = 120) -> str:
    text = _as_str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _run_analysis_payload(run_id: str | None, cfg: Any) -> dict[str, Any]:
    selected = _as_str(run_id)
    if not selected:
        return {}
    run_dir = cfg.resolve_path(cfg.output_dir) / selected
    if not run_dir.exists() or not run_dir.is_dir():
        return {}
    summary = _read_run_summary(run_dir / "run_summary.json")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    selected_candidates = _read_jsonl_dicts(run_dir / "text_candidates.validated.jsonl")
    raw_candidates = _read_jsonl_dicts(run_dir / "text_candidates.raw.jsonl")
    cards = _read_jsonl_dicts(run_dir / "cards.final.jsonl")
    errors = _read_jsonl_dicts(run_dir / "errors.jsonl")
    chunks = _read_jsonl_dicts(run_dir / "chunks.jsonl")

    rejected: list[dict[str, Any]] = []
    for item in errors:
        if _as_str(item.get("stage")) != "validate_text":
            continue
        reason = _as_str(item.get("reason"))
        category = classify_rejection_reason(reason)
        candidate = item.get("item")
        candidate_map = candidate if isinstance(candidate, dict) else {}
        rejected.append(
            {
                "reason": reason,
                "reason_category": category,
                "chunk_id": _as_str(
                    item.get("chunk_id") or candidate_map.get("chunk_id")
                ),
                "item": candidate_map,
            }
        )

    return {
        "run_id": selected,
        "run_dir": run_dir,
        "summary": summary,
        "metrics": metrics,
        "selected_candidates": selected_candidates,
        "raw_candidates": raw_candidates,
        "cards": cards,
        "rejected": rejected,
        "chunks": chunks,
    }


def _analysis_filter_choices(
    payload: dict[str, Any],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    metrics = (
        payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    )
    selected_candidates = (
        payload.get("selected_candidates", [])
        if isinstance(payload.get("selected_candidates"), list)
        else []
    )
    rejected = (
        payload.get("rejected", []) if isinstance(payload.get("rejected"), list) else []
    )

    taxonomy_keys: set[str] = set()
    for key in (
        "taxonomy_model_histogram",
        "taxonomy_candidate_histogram",
        "taxonomy_selected_histogram",
        "taxonomy_average_score",
    ):
        value = metrics.get(key)
        if isinstance(value, dict):
            taxonomy_keys.update(
                _as_str(item) for item in value.keys() if _as_str(item)
            )
    for item in selected_candidates:
        if not isinstance(item, dict):
            continue
        for ptype in item.get("phrase_types", []) or []:
            key = _as_str(ptype)
            if key:
                taxonomy_keys.add(key)

    rejection_keys: set[str] = set()
    for item in rejected:
        if not isinstance(item, dict):
            continue
        key = _as_str(item.get("reason_category"))
        if key:
            rejection_keys.add(key)

    chunk_keys: set[str] = set()
    for item in selected_candidates:
        if not isinstance(item, dict):
            continue
        key = _as_str(item.get("chunk_id"))
        if key:
            chunk_keys.add(key)
    for item in rejected:
        if not isinstance(item, dict):
            continue
        key = _as_str(item.get("chunk_id"))
        if key:
            chunk_keys.add(key)

    taxonomy_choices = [("all", "all")] + [(key, key) for key in sorted(taxonomy_keys)]
    rejection_choices = [("all", "all")] + [
        (key, key) for key in sorted(rejection_keys)
    ]
    chunk_choices = [("all", "all")] + [(key, key) for key in sorted(chunk_keys)]
    return taxonomy_choices, rejection_choices, chunk_choices


def _build_run_analysis(
    run_id: str | None,
    cfg: Any,
    *,
    lang: str,
    taxonomy_filter: str = "all",
    transfer_filter: str = "all",
    rejection_filter: str = "all",
    chunk_filter: str = "all",
) -> tuple[
    str,
    list[list[Any]],
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    payload = _run_analysis_payload(run_id, cfg)
    if not payload:
        return (
            _tr(lang, "No run selected.", "No run selected."),
            [],
            [("all", "all")],
            [("all", "all")],
            [("all", "all")],
        )

    summary = payload["summary"]
    metrics = payload["metrics"]
    selected_candidates = payload["selected_candidates"]
    rejected = payload["rejected"]
    taxonomy_choices, rejection_choices, chunk_choices = _analysis_filter_choices(
        payload
    )

    taxonomy_filter = _as_str(taxonomy_filter, default="all")
    transfer_filter = _as_str(transfer_filter, default="all")
    rejection_filter = _as_str(rejection_filter, default="all")
    chunk_filter = _as_str(chunk_filter, default="all")

    filtered_selected: list[dict[str, Any]] = []
    for item in selected_candidates:
        if not isinstance(item, dict):
            continue
        item_types = [
            _as_str(x) for x in (item.get("phrase_types", []) or []) if _as_str(x)
        ]
        has_transfer = bool(_as_str(item.get("expression_transfer")))
        item_chunk = _as_str(item.get("chunk_id"))

        if taxonomy_filter != "all" and taxonomy_filter not in item_types:
            continue
        if transfer_filter == "with_transfer" and not has_transfer:
            continue
        if transfer_filter == "without_transfer" and has_transfer:
            continue
        if chunk_filter != "all" and item_chunk != chunk_filter:
            continue
        filtered_selected.append(item)

    filtered_rejected: list[dict[str, Any]] = []
    for item in rejected:
        if not isinstance(item, dict):
            continue
        item_chunk = _as_str(item.get("chunk_id"))
        reason_category = _as_str(item.get("reason_category"))
        if rejection_filter != "all" and reason_category != rejection_filter:
            continue
        if chunk_filter != "all" and item_chunk != chunk_filter:
            continue
        filtered_rejected.append(item)

    learning_mode = _as_str(
        summary.get("learning_mode"),
        default=_as_str(metrics.get("learning_mode"), default="expression_mining"),
    )
    material_profile = _as_str(
        summary.get("material_profile"),
        default=_as_str(summary.get("content_profile"), default="-"),
    )
    difficulty = _as_str(
        summary.get("difficulty"),
        default=_as_str(metrics.get("difficulty"), default="-"),
    )
    chunks_total = _as_int(metrics.get("chunks_total"), default=0)
    raw_total = _as_int(metrics.get("raw_candidates"), default=0)
    valid_total = _as_int(metrics.get("validated_candidates"), default=0)
    selected_total = _as_int(metrics.get("deduped_candidates"), default=0)
    transfer_ratio = float(metrics.get("expression_transfer_non_empty_ratio") or 0.0)
    avg_clozes = float(metrics.get("avg_clozes_per_candidate") or 0.0)
    avg_phrases = float(metrics.get("avg_target_phrases_per_candidate") or 0.0)
    avg_selected_per_chunk = float(
        metrics.get("avg_selected_candidates_per_chunk") or 0.0
    )

    model_hist = (
        metrics.get("taxonomy_model_histogram")
        if isinstance(metrics.get("taxonomy_model_histogram"), dict)
        else {}
    )
    candidate_hist = (
        metrics.get("taxonomy_candidate_histogram")
        if isinstance(metrics.get("taxonomy_candidate_histogram"), dict)
        else {}
    )
    selected_hist = (
        metrics.get("taxonomy_selected_histogram")
        if isinstance(metrics.get("taxonomy_selected_histogram"), dict)
        else {}
    )
    avg_score_hist = (
        metrics.get("taxonomy_average_score")
        if isinstance(metrics.get("taxonomy_average_score"), dict)
        else {}
    )
    rejection_hist = (
        metrics.get("rejection_reason_histogram")
        if isinstance(metrics.get("rejection_reason_histogram"), dict)
        else {}
    )
    transfer_by_tax = (
        metrics.get("expression_transfer_non_empty_ratio_by_taxonomy")
        if isinstance(
            metrics.get("expression_transfer_non_empty_ratio_by_taxonomy"), dict
        )
        else {}
    )

    lines = [
        f"### {_tr(lang, 'Run analytics', 'Run analytics')}",
        f"- {_tr(lang, 'Learning mode', 'Learning mode')}: `{learning_mode}`",
        f"- {_tr(lang, 'Material profile', 'Material profile')}: `{material_profile}`",
        f"- {_tr(lang, 'Difficulty', 'Difficulty')}: `{difficulty}`",
        f"- {_tr(lang, 'Chunks', 'Chunks')}: **{chunks_total}**",
        f"- {_tr(lang, 'Raw candidates', 'Raw candidates')}: **{raw_total}**",
        f"- {_tr(lang, 'Validated candidates', 'Validated candidates')}: **{valid_total}**",
        f"- {_tr(lang, 'Selected cards', 'Selected cards')}: **{selected_total}**",
        f"- {_tr(lang, 'Transfer non-empty ratio', 'Transfer non-empty ratio')}: **{transfer_ratio:.2%}**",
        f"- {_tr(lang, 'Avg clozes per candidate', 'Avg clozes per candidate')}: **{avg_clozes:.2f}**",
        f"- {_tr(lang, 'Avg target phrases per candidate', 'Avg target phrases per candidate')}: **{avg_phrases:.2f}**",
        f"- {_tr(lang, 'Avg selected per chunk', 'Avg selected per chunk')}: **{avg_selected_per_chunk:.2f}**",
        f"- {_tr(lang, 'Filtered selected items', 'Filtered selected items')}: **{len(filtered_selected)}**",
        f"- {_tr(lang, 'Filtered rejected items', 'Filtered rejected items')}: **{len(filtered_rejected)}**",
        "",
        _render_count_histogram(
            _tr(lang, "Model taxonomy histogram", "Model taxonomy histogram"),
            model_hist,
        ),
        _render_count_histogram(
            _tr(lang, "Candidate taxonomy histogram", "Candidate taxonomy histogram"),
            candidate_hist,
        ),
        _render_count_histogram(
            _tr(lang, "Selected taxonomy histogram", "Selected taxonomy histogram"),
            selected_hist,
        ),
        _render_score_histogram(
            _tr(lang, "Taxonomy average score", "Taxonomy average score"),
            avg_score_hist,
        ),
        _render_count_histogram(
            _tr(lang, "Rejection reason histogram", "Rejection reason histogram"),
            rejection_hist,
        ),
        _render_score_histogram(
            _tr(
                lang,
                "Transfer non-empty ratio by taxonomy",
                "Transfer non-empty ratio by taxonomy",
            ),
            transfer_by_tax,
        ),
    ]

    sample_rows: list[list[Any]] = []
    seen: set[tuple[str, str, str]] = set()

    if rejection_filter == "all":
        top_selected = sorted(
            filtered_selected,
            key=lambda row: float(row.get("learning_value_score", 0.0)),
            reverse=True,
        )[:5]
        for item in top_selected:
            key = (
                "top_selected",
                _as_str(item.get("chunk_id")),
                _as_str(item.get("text")),
            )
            if key in seen:
                continue
            seen.add(key)
            sample_rows.append(
                [
                    "top_selected",
                    _as_str(item.get("chunk_id")),
                    " | ".join(
                        [
                            _as_str(x)
                            for x in (item.get("phrase_types", []) or [])
                            if _as_str(x)
                        ]
                    ),
                    float(item.get("learning_value_score", 0.0)),
                    _short_text(_as_str(item.get("text"))),
                    _short_text(_as_str(item.get("expression_transfer"))),
                    _short_text(_as_str(item.get("selection_reason"))),
                ]
            )

        taxonomy_bucket: dict[str, dict[str, Any]] = {}
        for item in sorted(
            filtered_selected,
            key=lambda row: float(row.get("learning_value_score", 0.0)),
            reverse=True,
        ):
            for ptype in item.get("phrase_types", []) or []:
                key = _as_str(ptype)
                if not key or key in taxonomy_bucket:
                    continue
                taxonomy_bucket[key] = item
        for ptype, item in list(taxonomy_bucket.items())[:5]:
            key = (
                "taxonomy_example",
                _as_str(item.get("chunk_id")),
                _as_str(item.get("text")),
            )
            if key in seen:
                continue
            seen.add(key)
            sample_rows.append(
                [
                    f"taxonomy_example:{ptype}",
                    _as_str(item.get("chunk_id")),
                    " | ".join(
                        [
                            _as_str(x)
                            for x in (item.get("phrase_types", []) or [])
                            if _as_str(x)
                        ]
                    ),
                    float(item.get("learning_value_score", 0.0)),
                    _short_text(_as_str(item.get("text"))),
                    _short_text(_as_str(item.get("expression_transfer"))),
                    _short_text(_as_str(item.get("selection_reason"))),
                ]
            )

        transfer_examples = [
            item
            for item in filtered_selected
            if _as_str(item.get("expression_transfer"))
        ][:5]
        for item in transfer_examples:
            key = (
                "transfer_example",
                _as_str(item.get("chunk_id")),
                _as_str(item.get("text")),
            )
            if key in seen:
                continue
            seen.add(key)
            sample_rows.append(
                [
                    "transfer_example",
                    _as_str(item.get("chunk_id")),
                    " | ".join(
                        [
                            _as_str(x)
                            for x in (item.get("phrase_types", []) or [])
                            if _as_str(x)
                        ]
                    ),
                    float(item.get("learning_value_score", 0.0)),
                    _short_text(_as_str(item.get("text"))),
                    _short_text(_as_str(item.get("expression_transfer"))),
                    _short_text(_as_str(item.get("selection_reason"))),
                ]
            )

    for item in filtered_rejected[:3]:
        candidate = item.get("item") if isinstance(item.get("item"), dict) else {}
        key = (
            "rejected_example",
            _as_str(item.get("chunk_id")),
            _as_str(candidate.get("text")),
        )
        if key in seen:
            continue
        seen.add(key)
        sample_rows.append(
            [
                f"rejected:{_as_str(item.get('reason_category'))}",
                _as_str(item.get("chunk_id")),
                " | ".join(
                    [
                        _as_str(x)
                        for x in (candidate.get("phrase_types", []) or [])
                        if _as_str(x)
                    ]
                ),
                float(candidate.get("learning_value_score", 0.0)),
                _short_text(_as_str(candidate.get("text"))),
                _short_text(_as_str(candidate.get("expression_transfer"))),
                _short_text(_as_str(item.get("reason"))),
            ]
        )

    return (
        "\n".join(lines),
        sample_rows,
        taxonomy_choices,
        rejection_choices,
        chunk_choices,
    )


def build_interface() -> gr.Blocks:
    """Construct the Gradio Blocks interface.

    Two tabs:
    - Run: upload file + per-run overrides + download link
    - Config: (reserved for future work, not implemented yet)
    """

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
        # Keep UI startup resilient; users can still inspect/fix prompt files in Prompt tab.
        logger.warning("prompt auto-seed during web startup failed", exc_info=True)
    env_file = _resolve_env_file()
    cfg_view = _load_env_view(cfg, env_file)
    prompt_files = _prompt_file_map(cfg)
    initial_ui_lang = _normalize_ui_lang(getattr(cfg, "prompt_lang", "en"))
    initial_run_content_type = _normalize_prompt_content_type(cfg.content_profile)
    initial_run_learning_mode = _normalize_prompt_learning_mode(
        getattr(cfg, "learning_mode", "expression_mining")
    )
    initial_run_difficulty = _normalize_prompt_difficulty(cfg.cloze_difficulty)
    initial_prompt_key = ""
    for key in prompt_files:
        if _load_prompt_mode(key, prompt_files, lang=initial_ui_lang) == "extraction":
            initial_prompt_key = key
            break
    if not initial_prompt_key and prompt_files:
        initial_prompt_key = next(iter(prompt_files))
    initial_prompt_text, initial_prompt_status = _load_prompt_template(
        initial_prompt_key, prompt_files, lang=initial_ui_lang
    )
    initial_prompt_mode = _load_prompt_mode(
        initial_prompt_key, prompt_files, lang=initial_ui_lang
    )
    (
        initial_prompt_content_type,
        initial_prompt_learning_mode,
        initial_prompt_difficulty,
    ) = _load_prompt_filter_metadata(
        initial_prompt_key, prompt_files, lang=initial_ui_lang
    )
    run_extract_prompt_choices = _prompt_path_choices(
        cfg,
        lang=initial_ui_lang,
        mode_filter="extraction",
        content_type_filter=initial_run_content_type,
        learning_mode_filter=initial_run_learning_mode,
        difficulty_filter=initial_run_difficulty,
        include_auto=True,
    )
    run_explain_prompt_choices = _prompt_path_choices(
        cfg,
        lang=initial_ui_lang,
        mode_filter="explanation",
        content_type_filter=initial_run_content_type,
        learning_mode_filter=initial_run_learning_mode,
        difficulty_filter=initial_run_difficulty,
        include_auto=True,
    )
    config_extract_prompt_choices = _prompt_path_choices(
        cfg,
        lang=initial_ui_lang,
        mode_filter="extraction",
        include_auto=True,
    )
    config_explain_prompt_choices = _prompt_path_choices(
        cfg,
        lang=initial_ui_lang,
        mode_filter="explanation",
        include_auto=True,
    )
    initial_runs = _scan_runs(cfg, limit=30)
    if initial_runs:
        initial_run_choices = [
            (_run_choice_label(run, lang=initial_ui_lang), run.run_id)
            for run in initial_runs
        ]
        initial_run_selected = initial_runs[0].run_id
        initial_run_detail, initial_run_download = _load_run_detail(
            initial_run_selected, cfg, lang=initial_ui_lang
        )
    else:
        initial_run_choices = []
        initial_run_selected = None
        initial_run_detail = _tr(initial_ui_lang, "No runs found.", "No runs found.")
        initial_run_download = None
    (
        initial_analysis_md,
        initial_samples_rows,
        initial_taxonomy_choices,
        initial_rejection_choices,
        initial_chunk_choices,
    ) = _build_run_analysis(
        initial_run_selected,
        cfg,
        lang=initial_ui_lang,
        taxonomy_filter="all",
        transfer_filter="all",
        rejection_filter="all",
        chunk_filter="all",
    )
    run_deps = handlers_run.RunDeps(
        normalize_ui_lang=_normalize_ui_lang,
        tr=_tr,
        run_single_build_v2=_run_single_build_v2,
        to_optional_int=_to_optional_int,
        to_optional_float=_to_optional_float,
        as_str=_as_str,
        load_app_config=_load_app_config,
        refresh_recent_runs=_refresh_recent_runs,
        load_run_detail=_load_run_detail,
        build_run_analysis=_build_run_analysis,
    )
    config_deps = handlers_config.ConfigDeps(
        list_models_markdown=_list_models_markdown,
        test_models_markdown=_test_models_markdown,
        to_timeout_seconds=_to_timeout_seconds,
        normalize_ui_lang=_normalize_ui_lang,
        read_env_example=_read_env_example,
        tr=_tr,
        save_env_v2=_save_env_v2,
    )
    prompt_deps = handlers_prompt.PromptDeps(
        normalize_ui_lang=_normalize_ui_lang,
        tr=_tr,
        load_app_config=_load_app_config,
        normalize_prompt_mode=_normalize_prompt_mode,
        normalize_prompt_content_type=_normalize_prompt_content_type,
        normalize_prompt_learning_mode=_normalize_prompt_learning_mode,
        normalize_prompt_difficulty=_normalize_prompt_difficulty,
        prompt_mode_label=_prompt_mode_label,
        prompt_file_map=_prompt_file_map,
        prompt_choices_from_map=_prompt_choices_from_map,
        prompt_path_choices=_prompt_path_choices,
        load_prompt_template=_load_prompt_template,
        load_prompt_mode=_load_prompt_mode,
        load_prompt_filter_metadata=_load_prompt_filter_metadata,
        sanitize_prompt_filename=_sanitize_prompt_filename,
        prompt_content_type_options=_PROMPT_CONTENT_TYPE_OPTIONS,
        prompt_learning_mode_options=_PROMPT_LEARNING_MODE_OPTIONS,
        prompt_difficulty_options=_PROMPT_DIFFICULTY_OPTIONS,
        prompt_io=prompt_io,
    )

    with gr.Blocks(title="ClawLingua Web UI") as demo:
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
                "# ClawLingua Web UI\nLocal deck builder for text learning.",
                "# ClawLingua Web UI\nLocal deck builder for text learning.",
            )
        )

        with gr.Tab(_tr(initial_ui_lang, "Run", "杩愯")) as run_tab:
            with gr.Row():
                input_file = gr.File(
                    label=_tr(initial_ui_lang, "Input file", "杈撳叆鏂囦欢"),
                    file_types=[".txt", ".md", ".markdown", ".epub"],
                    file_count="single",
                )
                deck_title = gr.Textbox(
                    label=_tr(
                        initial_ui_lang,
                        "Deck title (optional)",
                        "鐗岀粍鍚嶇О锛堝彲閫夛級",
                    )
                )

            with gr.Row():
                source_lang = gr.Dropdown(
                    choices=["en", "zh", "ja", "de", "fr"],
                    value=cfg.default_source_lang,
                    label=_tr(initial_ui_lang, "Source language", "婧愯瑷€"),
                )
                target_lang = gr.Dropdown(
                    choices=["zh", "en", "ja", "de", "fr"],
                    value=cfg.default_target_lang,
                    label=_tr(initial_ui_lang, "Target language", "鐩爣璇█"),
                )
                content_profile = gr.Dropdown(
                    choices=[
                        "prose_article",
                        "transcript_dialogue",
                        "textbook_examples",
                    ],
                    value=cfg.content_profile,
                    label=_tr(initial_ui_lang, "Content profile", "鍐呭绫诲瀷"),
                )
                learning_mode = gr.Dropdown(
                    choices=["expression_mining", "reading_support"],
                    value=getattr(cfg, "learning_mode", "expression_mining"),
                    label=_tr(initial_ui_lang, "Learning mode", "瀛︿範妯″紡"),
                )
                difficulty = gr.Dropdown(
                    choices=["beginner", "intermediate", "advanced"],
                    value=cfg.cloze_difficulty,
                    label=_tr(initial_ui_lang, "Difficulty", "闅惧害"),
                )

            with gr.Row():
                run_extract_prompt = gr.Dropdown(
                    choices=run_extract_prompt_choices,
                    value="",
                    label=_tr(
                        initial_ui_lang,
                        "Extraction prompt (run override)",
                        "Extraction prompt (run override)",
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "Equivalent to CLI --extract-prompt.",
                        "Equivalent to CLI --extract-prompt.",
                    ),
                )
                run_explain_prompt = gr.Dropdown(
                    choices=run_explain_prompt_choices,
                    value="",
                    label=_tr(
                        initial_ui_lang,
                        "Explanation prompt (run override)",
                        "Explanation prompt (run override)",
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "Equivalent to CLI --explain-prompt.",
                        "Equivalent to CLI --explain-prompt.",
                    ),
                )

            with gr.Row():
                max_notes = gr.Number(
                    label=_tr(
                        initial_ui_lang,
                        "Max notes (0 = no limit)",
                        "Max notes (0 = no limit)",
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "Maximum notes after dedupe. Empty/0 means no limit.",
                        "Maximum notes after dedupe. Empty/0 means no limit.",
                    ),
                    value=None,
                    precision=0,
                )
                input_char_limit = gr.Number(
                    label=_tr(initial_ui_lang, "Input char limit", "Input char limit"),
                    info=_tr(
                        initial_ui_lang,
                        "Only process the first N chars of input. Empty means no limit.",
                        "Only process the first N chars of input. Empty means no limit.",
                    ),
                    value=None,
                    precision=0,
                )

            with gr.Accordion(
                _tr(initial_ui_lang, "Advanced", "Advanced"), open=False
            ) as run_advanced:
                cloze_min_chars = gr.Number(
                    label=_tr(
                        initial_ui_lang,
                        "Cloze min chars (override env)",
                        "Cloze min chars (override env)",
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "One-run override for CLAWLINGUA_CLOZE_MIN_CHARS.",
                        "One-run override for CLAWLINGUA_CLOZE_MIN_CHARS.",
                    ),
                    value=cfg.cloze_min_chars,
                    precision=0,
                )
                chunk_max_chars = gr.Number(
                    label=_tr(
                        initial_ui_lang,
                        "Chunk max chars (override env)",
                        "Chunk max chars (override env)",
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "One-run override for CLAWLINGUA_CHUNK_MAX_CHARS.",
                        "One-run override for CLAWLINGUA_CHUNK_MAX_CHARS.",
                    ),
                    value=cfg.chunk_max_chars,
                    precision=0,
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=cfg.llm_temperature,
                    step=0.05,
                    label=_tr(
                        initial_ui_lang,
                        "Temperature (override env)",
                        "Temperature (override env)",
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "0 is more deterministic; higher values are more random.",
                        "0 is more deterministic; higher values are more random.",
                    ),
                )
                save_intermediate = gr.Checkbox(
                    label=_tr(
                        initial_ui_lang,
                        "Save intermediate files",
                        "Save intermediate files",
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.",
                        "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.",
                    ),
                    value=cfg.save_intermediate,
                )
                continue_on_error = gr.Checkbox(
                    label=_tr(
                        initial_ui_lang, "Continue on error", "Continue on error"
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "If enabled, continue processing after per-item failures.",
                        "If enabled, continue processing after per-item failures.",
                    ),
                    value=False,
                )

            run_button = gr.Button(_tr(initial_ui_lang, "Run", "Run"))

            run_status = gr.Markdown(label=_tr(initial_ui_lang, "Status", "Status"))
            output_file = gr.File(
                label=_tr(initial_ui_lang, "Download .apkg", "Download .apkg"),
                interactive=False,
            )
            recent_runs_heading = gr.Markdown(
                _tr(initial_ui_lang, "### Recent runs", "### Recent runs")
            )
            with gr.Row():
                refresh_runs_button = gr.Button(
                    _tr(initial_ui_lang, "Refresh runs", "Refresh runs")
                )
                run_selector = gr.Dropdown(
                    choices=initial_run_choices,
                    value=initial_run_selected,
                    label=_tr(initial_ui_lang, "Run ID", "Run ID"),
                )
            run_detail = gr.Markdown(value=initial_run_detail)
            run_download_file = gr.File(
                label=_tr(initial_ui_lang, "Download .apkg", "Download .apkg"),
                interactive=False,
                value=initial_run_download,
            )
            analytics_heading = gr.Markdown(
                _tr(initial_ui_lang, "### Run analytics", "### 杩愯缁熻鍒嗘瀽"),
                render=False,
            )
            taxonomy_filter = gr.Dropdown(
                choices=initial_taxonomy_choices,
                value="all",
                label=_tr(initial_ui_lang, "Taxonomy filter", "taxonomy 杩囨护"),
                render=False,
            )
            transfer_filter = gr.Dropdown(
                choices=[
                    ("all", "all"),
                    ("with_transfer", "with_transfer"),
                    ("without_transfer", "without_transfer"),
                ],
                value="all",
                label=_tr(initial_ui_lang, "Transfer filter", "transfer 杩囨护"),
                render=False,
            )
            rejection_filter = gr.Dropdown(
                choices=initial_rejection_choices,
                value="all",
                label=_tr(initial_ui_lang, "Rejection filter", "鎷掔粷鍘熷洜杩囨护"),
                render=False,
            )
            chunk_filter = gr.Dropdown(
                choices=initial_chunk_choices,
                value="all",
                label=_tr(initial_ui_lang, "Chunk filter", "chunk 杩囨护"),
                render=False,
            )
            apply_analysis_filter_btn = gr.Button(
                _tr(initial_ui_lang, "Apply filters", "搴旂敤杩囨护"),
                render=False,
            )
            run_analysis = gr.Markdown(value=initial_analysis_md, render=False)
            run_samples = gr.Dataframe(
                headers=[
                    "kind",
                    "chunk_id",
                    "phrase_types",
                    "score",
                    "text",
                    "expression_transfer",
                    "reason",
                ],
                datatype=["str", "str", "str", "number", "str", "str", "str"],
                interactive=False,
                wrap=True,
                value=initial_samples_rows,
                label=_tr(
                    initial_ui_lang, "Representative samples", "Representative samples"
                ),
                render=False,
            )

            def _on_run_start(ui_lang_val: str) -> tuple[str, None]:
                return handlers_run.on_run_start(ui_lang_val, deps=run_deps)

            def _on_run(
                file_obj,
                deck_title_val,
                src,
                tgt,
                profile,
                mode,
                diff,
                extract_prompt_val,
                explain_prompt_val,
                max_notes_val,
                input_limit_val,
                cloze_min_val,
                chunk_max_val,
                temperature_val,
                save_inter_val,
                continue_on_error_val,
                ui_lang_val,
            ):
                return handlers_run.on_run(
                    file_obj,
                    deck_title_val,
                    src,
                    tgt,
                    profile,
                    mode,
                    diff,
                    extract_prompt_val,
                    explain_prompt_val,
                    max_notes_val,
                    input_limit_val,
                    cloze_min_val,
                    chunk_max_val,
                    temperature_val,
                    save_inter_val,
                    continue_on_error_val,
                    ui_lang_val,
                    deps=run_deps,
                )

            def _on_refresh_runs(
                ui_lang_val: str,
                selected_run_id: str | None,
            ) -> tuple[Any, str, str | None, str, list[list[Any]], Any, Any, Any, Any]:
                return handlers_run.on_refresh_runs(
                    ui_lang_val,
                    selected_run_id,
                    deps=run_deps,
                )

            def _on_run_selected(
                run_id_val: str | None,
                ui_lang_val: str,
            ) -> tuple[str, str | None, str, list[list[Any]], Any, Any, Any, Any]:
                return handlers_run.on_run_selected(
                    run_id_val,
                    ui_lang_val,
                    deps=run_deps,
                )

            def _on_apply_analysis_filters(
                run_id_val: str | None,
                ui_lang_val: str,
                taxonomy_val: str,
                transfer_val: str,
                rejection_val: str,
                chunk_val: str,
            ) -> tuple[str, list[list[Any]]]:
                return handlers_run.on_apply_analysis_filters(
                    run_id_val,
                    ui_lang_val,
                    taxonomy_val,
                    transfer_val,
                    rejection_val,
                    chunk_val,
                    deps=run_deps,
                )

            run_button.click(
                _on_run_start,
                inputs=[ui_lang],
                outputs=[run_status, output_file],
                queue=False,
            ).then(
                _on_run,
                inputs=[
                    input_file,
                    deck_title,
                    source_lang,
                    target_lang,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    max_notes,
                    input_char_limit,
                    cloze_min_chars,
                    chunk_max_chars,
                    temperature,
                    save_intermediate,
                    continue_on_error,
                    ui_lang,
                ],
                outputs=[
                    run_status,
                    output_file,
                    run_selector,
                    run_detail,
                    run_download_file,
                    run_analysis,
                    run_samples,
                    taxonomy_filter,
                    transfer_filter,
                    rejection_filter,
                    chunk_filter,
                ],
            )

            refresh_runs_button.click(
                _on_refresh_runs,
                inputs=[ui_lang, run_selector],
                outputs=[
                    run_selector,
                    run_detail,
                    run_download_file,
                    run_analysis,
                    run_samples,
                    taxonomy_filter,
                    transfer_filter,
                    rejection_filter,
                    chunk_filter,
                ],
            )

            run_selector.change(
                _on_run_selected,
                inputs=[run_selector, ui_lang],
                outputs=[
                    run_detail,
                    run_download_file,
                    run_analysis,
                    run_samples,
                    taxonomy_filter,
                    transfer_filter,
                    rejection_filter,
                    chunk_filter,
                ],
            )
            apply_analysis_filter_btn.click(
                _on_apply_analysis_filters,
                inputs=[
                    run_selector,
                    ui_lang,
                    taxonomy_filter,
                    transfer_filter,
                    rejection_filter,
                    chunk_filter,
                ],
                outputs=[run_analysis, run_samples],
            )
        with gr.Tab(_tr(initial_ui_lang, "Config", "閰嶇疆")) as config_tab:
            config_heading = gr.Markdown(
                _tr(
                    initial_ui_lang,
                    "### Config (.env editor)",
                    "### 閰嶇疆锛?env 缂栬緫鍣級",
                )
            )

            with gr.Accordion(
                _tr(initial_ui_lang, "Extraction LLM", "Extraction LLM"), open=True
            ) as llm_accordion:
                llm_base_url = gr.Textbox(
                    label="CLAWLINGUA_LLM_BASE_URL",
                    value=cfg_view.get("CLAWLINGUA_LLM_BASE_URL", ""),
                    info=_tr(
                        initial_ui_lang,
                        "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).",
                        "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).",
                    ),
                )
                llm_api_key = gr.Textbox(
                    label="CLAWLINGUA_LLM_API_KEY",
                    value=cfg_view.get("CLAWLINGUA_LLM_API_KEY", ""),
                    type="password",
                    info=_tr(
                        initial_ui_lang,
                        "API key for extraction LLM, when required.",
                        "API key for extraction LLM, when required.",
                    ),
                )
                llm_model = gr.Textbox(
                    label="CLAWLINGUA_LLM_MODEL",
                    value=cfg_view.get("CLAWLINGUA_LLM_MODEL", ""),
                    info=_tr(
                        initial_ui_lang,
                        "Model name for extraction LLM.",
                        "Model name for extraction LLM.",
                    ),
                )
                llm_timeout = gr.Textbox(
                    label="CLAWLINGUA_LLM_TIMEOUT_SECONDS",
                    value=cfg_view.get("CLAWLINGUA_LLM_TIMEOUT_SECONDS", "120"),
                    info=_tr(
                        initial_ui_lang,
                        "Request timeout in seconds.",
                        "Request timeout in seconds.",
                    ),
                )
                llm_temperature_env = gr.Textbox(
                    label="CLAWLINGUA_LLM_TEMPERATURE",
                    value=cfg_view.get("CLAWLINGUA_LLM_TEMPERATURE", "0.2"),
                    info=_tr(
                        initial_ui_lang,
                        "Default temperature for extraction LLM.",
                        "Default temperature for extraction LLM.",
                    ),
                )
                llm_chunk_batch_size_env = gr.Textbox(
                    label="CLAWLINGUA_LLM_CHUNK_BATCH_SIZE",
                    value=cfg_view.get("CLAWLINGUA_LLM_CHUNK_BATCH_SIZE", "1"),
                    info="Chunk batch size for cloze LLM calls; 1 means per-chunk requests.",
                )
                extract_prompt_env = gr.Dropdown(
                    choices=config_extract_prompt_choices,
                    value=cfg_view.get("CLAWLINGUA_EXTRACT_PROMPT", ""),
                    label="CLAWLINGUA_EXTRACT_PROMPT",
                    info=_tr(
                        initial_ui_lang,
                        "Default extraction prompt path.",
                        "Default extraction prompt path.",
                    ),
                )
                with gr.Row():
                    llm_list_models_btn = gr.Button(
                        _tr(initial_ui_lang, "List models", "鍒楀嚭妯″瀷")
                    )
                    llm_test_btn = gr.Button(_tr(initial_ui_lang, "Test", "Test"))
                llm_status = gr.Markdown(
                    label=_tr(
                        initial_ui_lang,
                        "Extraction LLM status",
                        "Extraction LLM status",
                    )
                )

            with gr.Accordion(
                _tr(initial_ui_lang, "Explanation LLM", "Explanation LLM"), open=False
            ) as translate_accordion:
                translate_base_url = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_BASE_URL",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_BASE_URL", ""),
                    info=_tr(
                        initial_ui_lang,
                        "Optional base URL for explanation model.",
                        "Optional base URL for explanation model.",
                    ),
                )
                translate_api_key = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_API_KEY",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_API_KEY", ""),
                    type="password",
                    info=_tr(
                        initial_ui_lang,
                        "API key for explanation LLM, when required.",
                        "API key for explanation LLM, when required.",
                    ),
                )
                translate_model = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_MODEL",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_MODEL", ""),
                    info=_tr(
                        initial_ui_lang,
                        "Model name for explanation LLM.",
                        "Model name for explanation LLM.",
                    ),
                )
                translate_temperature = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE", ""),
                    info=_tr(
                        initial_ui_lang,
                        "Default temperature for explanation LLM.",
                        "Default temperature for explanation LLM.",
                    ),
                )
                explain_prompt_env = gr.Dropdown(
                    choices=config_explain_prompt_choices,
                    value=cfg_view.get("CLAWLINGUA_EXPLAIN_PROMPT", ""),
                    label="CLAWLINGUA_EXPLAIN_PROMPT",
                    info=_tr(
                        initial_ui_lang,
                        "Default explanation prompt path.",
                        "Default explanation prompt path.",
                    ),
                )
                with gr.Row():
                    translate_list_models_btn = gr.Button(
                        _tr(
                            initial_ui_lang,
                            "List models (explanation)",
                            "鍒楀嚭瑙ｉ噴妯″瀷",
                        )
                    )
                    translate_test_btn = gr.Button(
                        _tr(initial_ui_lang, "Test (explanation)", "Test (explanation)")
                    )
                translate_status = gr.Markdown(
                    label=_tr(
                        initial_ui_lang,
                        "Explanation LLM status",
                        "Explanation LLM status",
                    )
                )

            with gr.Accordion(
                _tr(initial_ui_lang, "Chunk & Cloze", "Chunk & Cloze"), open=False
            ) as chunk_accordion:
                chunk_max_chars_env = gr.Textbox(
                    label="CLAWLINGUA_CHUNK_MAX_CHARS",
                    value=cfg_view.get("CLAWLINGUA_CHUNK_MAX_CHARS", "1800"),
                    info=_tr(
                        initial_ui_lang,
                        "Default max chars per chunk.",
                        "Default max chars per chunk.",
                    ),
                )
                chunk_min_chars_env = gr.Textbox(
                    label="CLAWLINGUA_CHUNK_MIN_CHARS",
                    value=cfg_view.get("CLAWLINGUA_CHUNK_MIN_CHARS", "120"),
                )
                cloze_min_chars_env = gr.Textbox(
                    label="CLAWLINGUA_CLOZE_MIN_CHARS",
                    value=cfg_view.get("CLAWLINGUA_CLOZE_MIN_CHARS", "0"),
                    info=_tr(
                        initial_ui_lang,
                        "Minimum chars required for cloze text.",
                        "Minimum chars required for cloze text.",
                    ),
                )
                cloze_max_per_chunk_env = gr.Textbox(
                    label="CLAWLINGUA_CLOZE_MAX_PER_CHUNK",
                    value=cfg_view.get("CLAWLINGUA_CLOZE_MAX_PER_CHUNK", ""),
                    info=_tr(
                        initial_ui_lang,
                        "Max cards per chunk after dedupe. Empty/0 means unlimited.",
                        "Max cards per chunk after dedupe. Empty/0 means unlimited.",
                    ),
                )
                validate_retry_enable_env = gr.Textbox(
                    label="CLAWLINGUA_VALIDATE_FORMAT_RETRY_ENABLE",
                    value=cfg_view.get(
                        "CLAWLINGUA_VALIDATE_FORMAT_RETRY_ENABLE", "true"
                    ),
                )
                validate_retry_max_env = gr.Textbox(
                    label="CLAWLINGUA_VALIDATE_FORMAT_RETRY_MAX",
                    value=cfg_view.get("CLAWLINGUA_VALIDATE_FORMAT_RETRY_MAX", "3"),
                )
                validate_retry_llm_enable_env = gr.Textbox(
                    label="CLAWLINGUA_VALIDATE_FORMAT_RETRY_LLM_ENABLE",
                    value=cfg_view.get(
                        "CLAWLINGUA_VALIDATE_FORMAT_RETRY_LLM_ENABLE", "true"
                    ),
                )
                content_profile_env = gr.Textbox(
                    label="CLAWLINGUA_CONTENT_PROFILE",
                    value=cfg_view.get("CLAWLINGUA_CONTENT_PROFILE", "prose_article"),
                )
                cloze_difficulty_env = gr.Textbox(
                    label="CLAWLINGUA_CLOZE_DIFFICULTY",
                    value=cfg_view.get("CLAWLINGUA_CLOZE_DIFFICULTY", "intermediate"),
                )
                prompt_lang_env = gr.Textbox(
                    label="CLAWLINGUA_PROMPT_LANG",
                    value=cfg_view.get("CLAWLINGUA_PROMPT_LANG", "zh"),
                    info=_tr(
                        initial_ui_lang,
                        "Prompt language for multi-lingual prompts (en/zh).",
                        "Prompt language for multi-lingual prompts (en/zh).",
                    ),
                )

            with gr.Accordion(
                _tr(initial_ui_lang, "Paths & defaults", "Paths & defaults"), open=False
            ) as paths_accordion:
                output_dir_env = gr.Textbox(
                    label="CLAWLINGUA_OUTPUT_DIR",
                    value=cfg_view.get("CLAWLINGUA_OUTPUT_DIR", "./runs"),
                    info=_tr(
                        initial_ui_lang,
                        "Directory for intermediate run data (JSONL, media).",
                        "Directory for intermediate run data (JSONL, media).",
                    ),
                )
                export_dir_env = gr.Textbox(
                    label="CLAWLINGUA_EXPORT_DIR",
                    value=cfg_view.get("CLAWLINGUA_EXPORT_DIR", "./outputs"),
                    info=_tr(
                        initial_ui_lang,
                        "Default directory for exported decks.",
                        "Default directory for exported decks.",
                    ),
                )
                log_dir_env = gr.Textbox(
                    label="CLAWLINGUA_LOG_DIR",
                    value=cfg_view.get("CLAWLINGUA_LOG_DIR", "./logs"),
                    info=_tr(
                        initial_ui_lang,
                        "Directory for log files.",
                        "Directory for log files.",
                    ),
                )
                default_deck_name_env = gr.Textbox(
                    label="CLAWLINGUA_DEFAULT_DECK_NAME",
                    value=cfg_view.get(
                        "CLAWLINGUA_DEFAULT_DECK_NAME", cfg.default_deck_name
                    ),
                )

            with gr.Accordion(
                _tr(initial_ui_lang, "TTS voices (Edge)", "TTS voices (Edge)"),
                open=False,
            ) as tts_accordion:
                tts_hint_md = gr.Markdown(
                    _tr(
                        initial_ui_lang,
                        "Voice reference: [Edge TTS Voice Samples](https://tts.travisvn.com/)",
                        "鍏蜂綋鐨勯煶鑹插彲浠ュ弬鑰僛Edge TTS Voice Samples](https://tts.travisvn.com/)",
                    )
                )
                tts_voice1_env = gr.Textbox(
                    label="CLAWLINGUA_TTS_EDGE_VOICE1",
                    value=cfg_view.get("CLAWLINGUA_TTS_EDGE_VOICE1", ""),
                    info=_tr(
                        initial_ui_lang,
                        "Configure 4 voice slots used for random selection.",
                        "Configure 4 voice slots used for random selection.",
                    ),
                )
                tts_voice2_env = gr.Textbox(
                    label="CLAWLINGUA_TTS_EDGE_VOICE2",
                    value=cfg_view.get("CLAWLINGUA_TTS_EDGE_VOICE2", ""),
                )
                tts_voice3_env = gr.Textbox(
                    label="CLAWLINGUA_TTS_EDGE_VOICE3",
                    value=cfg_view.get("CLAWLINGUA_TTS_EDGE_VOICE3", ""),
                )
                tts_voice4_env = gr.Textbox(
                    label="CLAWLINGUA_TTS_EDGE_VOICE4",
                    value=cfg_view.get("CLAWLINGUA_TTS_EDGE_VOICE4", ""),
                )

            with gr.Row():
                load_defaults_btn = gr.Button(
                    _tr(
                        initial_ui_lang,
                        "Load defaults from ENV_EXAMPLE.md",
                        "Load defaults from ENV_EXAMPLE.md",
                    )
                )
                save_config_btn = gr.Button(
                    _tr(initial_ui_lang, "Save config", "淇濆瓨閰嶇疆")
                )
            save_config_status = gr.Markdown()

            def _on_list_models(
                base_url: str, api_key: str, timeout_raw: Any, ui_lang_val: str
            ) -> str:
                return handlers_config.on_list_models(
                    base_url,
                    api_key,
                    timeout_raw,
                    ui_lang_val,
                    deps=config_deps,
                )

            def _on_test_models(
                base_url: str, api_key: str, timeout_raw: Any, ui_lang_val: str
            ) -> str:
                return handlers_config.on_test_models(
                    base_url,
                    api_key,
                    timeout_raw,
                    ui_lang_val,
                    deps=config_deps,
                )

            llm_list_models_btn.click(
                _on_list_models,
                inputs=[llm_base_url, llm_api_key, llm_timeout, ui_lang],
                outputs=[llm_status],
            )
            llm_test_btn.click(
                _on_test_models,
                inputs=[llm_base_url, llm_api_key, llm_timeout, ui_lang],
                outputs=[llm_status],
            )
            translate_list_models_btn.click(
                _on_list_models,
                inputs=[translate_base_url, translate_api_key, llm_timeout, ui_lang],
                outputs=[translate_status],
            )
            translate_test_btn.click(
                _on_test_models,
                inputs=[translate_base_url, translate_api_key, llm_timeout, ui_lang],
                outputs=[translate_status],
            )

            def _on_load_defaults(
                llm_base_url_val: str,
                llm_api_key_val: str,
                llm_model_val: str,
                llm_timeout_val: str,
                llm_temperature_val: str,
                llm_chunk_batch_size_val: str,
                translate_base_url_val: str,
                translate_api_key_val: str,
                translate_model_val: str,
                translate_temperature_val: str,
                chunk_max_chars_val: str,
                chunk_min_chars_val: str,
                cloze_min_chars_val: str,
                cloze_max_per_chunk_val: str,
                validate_retry_enable_val: str,
                validate_retry_max_val: str,
                validate_retry_llm_enable_val: str,
                content_profile_val: str,
                cloze_difficulty_val: str,
                prompt_lang_val: str,
                extract_prompt_env_val: str,
                explain_prompt_env_val: str,
                output_dir_val: str,
                export_dir_val: str,
                log_dir_val: str,
                default_deck_name_val: str,
                tts_voice1_val: str,
                tts_voice2_val: str,
                tts_voice3_val: str,
                tts_voice4_val: str,
                ui_lang_val: str,
            ) -> tuple[str, ...]:
                return handlers_config.on_load_defaults(
                    llm_base_url_val,
                    llm_api_key_val,
                    llm_model_val,
                    llm_timeout_val,
                    llm_temperature_val,
                    llm_chunk_batch_size_val,
                    translate_base_url_val,
                    translate_api_key_val,
                    translate_model_val,
                    translate_temperature_val,
                    chunk_max_chars_val,
                    chunk_min_chars_val,
                    cloze_min_chars_val,
                    cloze_max_per_chunk_val,
                    validate_retry_enable_val,
                    validate_retry_max_val,
                    validate_retry_llm_enable_val,
                    content_profile_val,
                    cloze_difficulty_val,
                    prompt_lang_val,
                    extract_prompt_env_val,
                    explain_prompt_env_val,
                    output_dir_val,
                    export_dir_val,
                    log_dir_val,
                    default_deck_name_val,
                    tts_voice1_val,
                    tts_voice2_val,
                    tts_voice3_val,
                    tts_voice4_val,
                    ui_lang_val,
                    deps=config_deps,
                )

            load_defaults_btn.click(
                _on_load_defaults,
                inputs=[
                    llm_base_url,
                    llm_api_key,
                    llm_model,
                    llm_timeout,
                    llm_temperature_env,
                    llm_chunk_batch_size_env,
                    translate_base_url,
                    translate_api_key,
                    translate_model,
                    translate_temperature,
                    chunk_max_chars_env,
                    chunk_min_chars_env,
                    cloze_min_chars_env,
                    cloze_max_per_chunk_env,
                    validate_retry_enable_env,
                    validate_retry_max_env,
                    validate_retry_llm_enable_env,
                    content_profile_env,
                    cloze_difficulty_env,
                    prompt_lang_env,
                    extract_prompt_env,
                    explain_prompt_env,
                    output_dir_env,
                    export_dir_env,
                    log_dir_env,
                    default_deck_name_env,
                    tts_voice1_env,
                    tts_voice2_env,
                    tts_voice3_env,
                    tts_voice4_env,
                    ui_lang,
                ],
                outputs=[
                    llm_base_url,
                    llm_api_key,
                    llm_model,
                    llm_timeout,
                    llm_temperature_env,
                    llm_chunk_batch_size_env,
                    translate_base_url,
                    translate_api_key,
                    translate_model,
                    translate_temperature,
                    chunk_max_chars_env,
                    chunk_min_chars_env,
                    cloze_min_chars_env,
                    cloze_max_per_chunk_env,
                    validate_retry_enable_env,
                    validate_retry_max_env,
                    validate_retry_llm_enable_env,
                    content_profile_env,
                    cloze_difficulty_env,
                    prompt_lang_env,
                    extract_prompt_env,
                    explain_prompt_env,
                    output_dir_env,
                    export_dir_env,
                    log_dir_env,
                    default_deck_name_env,
                    tts_voice1_env,
                    tts_voice2_env,
                    tts_voice3_env,
                    tts_voice4_env,
                    save_config_status,
                ],
            )

            def _on_save_config(
                llm_base_url_val,
                llm_api_key_val,
                llm_model_val,
                llm_timeout_val,
                llm_temperature_val,
                llm_chunk_batch_size_val,
                translate_base_url_val,
                translate_api_key_val,
                translate_model_val,
                translate_temperature_val,
                chunk_max_chars_val,
                chunk_min_chars_val,
                cloze_min_chars_val,
                cloze_max_per_chunk_val,
                validate_retry_enable_val,
                validate_retry_max_val,
                validate_retry_llm_enable_val,
                content_profile_val,
                cloze_difficulty_val,
                prompt_lang_val,
                extract_prompt_env_val,
                explain_prompt_env_val,
                output_dir_val,
                export_dir_val,
                log_dir_val,
                default_deck_name_val,
                tts_voice1_val,
                tts_voice2_val,
                tts_voice3_val,
                tts_voice4_val,
                ui_lang_val,
            ):
                return handlers_config.on_save_config(
                    llm_base_url_val,
                    llm_api_key_val,
                    llm_model_val,
                    llm_timeout_val,
                    llm_temperature_val,
                    llm_chunk_batch_size_val,
                    translate_base_url_val,
                    translate_api_key_val,
                    translate_model_val,
                    translate_temperature_val,
                    chunk_max_chars_val,
                    chunk_min_chars_val,
                    cloze_min_chars_val,
                    cloze_max_per_chunk_val,
                    validate_retry_enable_val,
                    validate_retry_max_val,
                    validate_retry_llm_enable_val,
                    content_profile_val,
                    cloze_difficulty_val,
                    prompt_lang_val,
                    extract_prompt_env_val,
                    explain_prompt_env_val,
                    output_dir_val,
                    export_dir_val,
                    log_dir_val,
                    default_deck_name_val,
                    tts_voice1_val,
                    tts_voice2_val,
                    tts_voice3_val,
                    tts_voice4_val,
                    ui_lang_val,
                    deps=config_deps,
                )

            save_config_btn.click(
                _on_save_config,
                inputs=[
                    llm_base_url,
                    llm_api_key,
                    llm_model,
                    llm_timeout,
                    llm_temperature_env,
                    llm_chunk_batch_size_env,
                    translate_base_url,
                    translate_api_key,
                    translate_model,
                    translate_temperature,
                    chunk_max_chars_env,
                    chunk_min_chars_env,
                    cloze_min_chars_env,
                    cloze_max_per_chunk_env,
                    validate_retry_enable_env,
                    validate_retry_max_env,
                    validate_retry_llm_enable_env,
                    content_profile_env,
                    cloze_difficulty_env,
                    prompt_lang_env,
                    extract_prompt_env,
                    explain_prompt_env,
                    output_dir_env,
                    export_dir_env,
                    log_dir_env,
                    default_deck_name_env,
                    tts_voice1_env,
                    tts_voice2_env,
                    tts_voice3_env,
                    tts_voice4_env,
                    ui_lang,
                ],
                outputs=[save_config_status],
            )

        with gr.Tab(_tr(initial_ui_lang, "Prompt", "Prompt")) as prompt_tab:
            prompt_heading = gr.Markdown(
                _tr(
                    initial_ui_lang,
                    "### Prompt template editor",
                    "### Prompt template editor",
                )
            )
            with gr.Row():
                prompt_content_type_selector = gr.Dropdown(
                    choices=_PROMPT_CONTENT_TYPE_OPTIONS,
                    value=initial_prompt_content_type,
                    label=_tr(
                        initial_ui_lang,
                        "Prompt content type",
                        "Prompt content type",
                    ),
                    scale=1,
                )
                prompt_learning_mode_selector = gr.Dropdown(
                    choices=_PROMPT_LEARNING_MODE_OPTIONS,
                    value=initial_prompt_learning_mode,
                    label=_tr(
                        initial_ui_lang,
                        "Prompt learning mode",
                        "Prompt learning mode",
                    ),
                    scale=1,
                )
                prompt_difficulty_selector = gr.Dropdown(
                    choices=_PROMPT_DIFFICULTY_OPTIONS,
                    value=initial_prompt_difficulty,
                    label=_tr(
                        initial_ui_lang,
                        "Prompt difficulty",
                        "Prompt difficulty",
                    ),
                    scale=1,
                )
            with gr.Row():
                prompt_file_selector = gr.Dropdown(
                    choices=_prompt_choices(
                        initial_ui_lang,
                        mode_filter=initial_prompt_mode or "extraction",
                        content_type_filter=initial_prompt_content_type,
                        learning_mode_filter=initial_prompt_learning_mode,
                        difficulty_filter=initial_prompt_difficulty,
                    ),
                    value=initial_prompt_key,
                    label=_tr(initial_ui_lang, "Prompt file", "Prompt 鏂囦欢"),
                    scale=2,
                )
                prompt_mode_selector = gr.Dropdown(
                    choices=[
                        (
                            _prompt_mode_label("extraction", lang=initial_ui_lang),
                            "extraction",
                        ),
                        (
                            _prompt_mode_label("explanation", lang=initial_ui_lang),
                            "explanation",
                        ),
                    ],
                    value=initial_prompt_mode or "extraction",
                    label=_tr(initial_ui_lang, "Prompt type", "Prompt type"),
                    scale=1,
                )
            with gr.Row():
                prompt_new_name = gr.Textbox(
                    label=_tr(
                        initial_ui_lang, "New prompt file name", "New prompt file name"
                    ),
                    placeholder="my_prompt.json",
                )
                prompt_rename_name = gr.Textbox(
                    label=_tr(initial_ui_lang, "Rename to", "閲嶅懡鍚嶄负"),
                    placeholder="renamed_prompt.json",
                )
            prompt_editor = gr.Textbox(
                label=_tr(initial_ui_lang, "Prompt template", "Prompt 妯℃澘"),
                value=initial_prompt_text,
                lines=24,
            )
            with gr.Row():
                prompt_new_btn = gr.Button(_tr(initial_ui_lang, "New", "鏂板缓"))
                prompt_save_btn = gr.Button(_tr(initial_ui_lang, "Save", "淇濆瓨"))
                prompt_rename_btn = gr.Button(_tr(initial_ui_lang, "Rename", "Rename"))
                prompt_load_default_btn = gr.Button(
                    _tr(initial_ui_lang, "Delete", "鍒犻櫎"), variant="stop"
                )
            prompt_save_confirm = gr.Checkbox(value=False, visible=False)
            prompt_delete_confirm = gr.Checkbox(value=False, visible=False)
            prompt_status = gr.Markdown(
                label=_tr(initial_ui_lang, "Prompt status", "Prompt status"),
                value=initial_prompt_status,
            )

            def _prompt_mode_choices_for_ui(lang: str) -> list[tuple[str, str]]:
                return handlers_prompt.prompt_mode_choices_for_ui(
                    lang,
                    deps=prompt_deps,
                )

            def _normalize_dropdown_value(current: str, choices: list[Any]) -> str:
                return handlers_prompt.normalize_dropdown_value(current, choices)

            def _pick_prompt_key(
                prompt_files_now: dict[str, Path],
                *,
                lang: str,
                preferred_key: str = "",
                preferred_mode: str = "",
            ) -> str:
                return handlers_prompt.pick_prompt_key(
                    prompt_files_now,
                    lang=lang,
                    preferred_key=preferred_key,
                    preferred_mode=preferred_mode,
                    deps=prompt_deps,
                )

            def _refresh_prompt_controls(
                *,
                lang: str,
                prompt_key: str,
                preferred_mode: str,
                preferred_content_type: str = "all",
                preferred_learning_mode: str = "all",
                preferred_difficulty: str = "all",
                status: str,
                editor_override: str | None,
                run_content_type: str = "all",
                run_learning_mode: str = "all",
                run_difficulty: str = "all",
                run_extract_current: str = "",
                run_explain_current: str = "",
                config_extract_current: str = "",
                config_explain_current: str = "",
            ) -> tuple[Any, ...]:
                return handlers_prompt.refresh_prompt_controls(
                    lang=lang,
                    prompt_key=prompt_key,
                    preferred_mode=preferred_mode,
                    preferred_content_type=preferred_content_type,
                    preferred_learning_mode=preferred_learning_mode,
                    preferred_difficulty=preferred_difficulty,
                    status=status,
                    editor_override=editor_override,
                    run_content_type=run_content_type,
                    run_learning_mode=run_learning_mode,
                    run_difficulty=run_difficulty,
                    run_extract_current=run_extract_current,
                    run_explain_current=run_explain_current,
                    config_extract_current=config_extract_current,
                    config_explain_current=config_explain_current,
                    deps=prompt_deps,
                )

            def _append_prompt_aux_updates(
                updates: tuple[Any, ...],
                *,
                new_name_value: str = "",
                rename_name_value: str = "",
            ) -> tuple[Any, ...]:
                return handlers_prompt.append_prompt_aux_updates(
                    updates,
                    new_name_value=new_name_value,
                    rename_name_value=rename_name_value,
                )

            def _on_prompt_file_change(
                prompt_key: str,
                prompt_mode: str,
                prompt_content_type: str,
                prompt_learning_mode: str,
                prompt_difficulty: str,
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                config_extract_val: str,
                config_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, ...]:
                return handlers_prompt.on_prompt_file_change(
                    prompt_key,
                    prompt_mode,
                    prompt_content_type,
                    prompt_learning_mode,
                    prompt_difficulty,
                    run_content_type,
                    run_learning_mode,
                    run_difficulty,
                    run_extract_val,
                    run_explain_val,
                    config_extract_val,
                    config_explain_val,
                    ui_lang_val,
                    deps=prompt_deps,
                )

            def _on_prompt_mode_change(
                prompt_mode: str,
                prompt_key: str,
                prompt_content_type: str,
                prompt_learning_mode: str,
                prompt_difficulty: str,
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                config_extract_val: str,
                config_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, ...]:
                return handlers_prompt.on_prompt_mode_change(
                    prompt_mode,
                    prompt_key,
                    prompt_content_type,
                    prompt_learning_mode,
                    prompt_difficulty,
                    run_content_type,
                    run_learning_mode,
                    run_difficulty,
                    run_extract_val,
                    run_explain_val,
                    config_extract_val,
                    config_explain_val,
                    ui_lang_val,
                    deps=prompt_deps,
                )

            def _on_prompt_filter_change(
                prompt_content_type: str,
                prompt_learning_mode: str,
                prompt_difficulty: str,
                prompt_mode: str,
                prompt_key: str,
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                config_extract_val: str,
                config_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, ...]:
                return handlers_prompt.on_prompt_filter_change(
                    prompt_content_type,
                    prompt_learning_mode,
                    prompt_difficulty,
                    prompt_mode,
                    prompt_key,
                    run_content_type,
                    run_learning_mode,
                    run_difficulty,
                    run_extract_val,
                    run_explain_val,
                    config_extract_val,
                    config_explain_val,
                    ui_lang_val,
                    deps=prompt_deps,
                )

            def _on_prompt_new(
                prompt_key: str,
                new_name: str,
                prompt_mode: str,
                prompt_content_type: str,
                prompt_learning_mode: str,
                prompt_difficulty: str,
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                config_extract_val: str,
                config_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, ...]:
                return handlers_prompt.on_prompt_new(
                    prompt_key,
                    new_name,
                    prompt_mode,
                    prompt_content_type,
                    prompt_learning_mode,
                    prompt_difficulty,
                    run_content_type,
                    run_learning_mode,
                    run_difficulty,
                    run_extract_val,
                    run_explain_val,
                    config_extract_val,
                    config_explain_val,
                    ui_lang_val,
                    deps=prompt_deps,
                )

            def _on_prompt_save(
                prompt_key: str,
                prompt_mode: str,
                prompt_content_type: str,
                prompt_learning_mode: str,
                prompt_difficulty: str,
                prompt_template: str,
                save_confirmed: bool,
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                config_extract_val: str,
                config_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, ...]:
                return handlers_prompt.on_prompt_save(
                    prompt_key,
                    prompt_mode,
                    prompt_content_type,
                    prompt_learning_mode,
                    prompt_difficulty,
                    prompt_template,
                    save_confirmed,
                    run_content_type,
                    run_learning_mode,
                    run_difficulty,
                    run_extract_val,
                    run_explain_val,
                    config_extract_val,
                    config_explain_val,
                    ui_lang_val,
                    deps=prompt_deps,
                )

            def _on_prompt_rename(
                prompt_key: str,
                rename_name: str,
                prompt_mode: str,
                prompt_content_type: str,
                prompt_learning_mode: str,
                prompt_difficulty: str,
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                config_extract_val: str,
                config_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, ...]:
                return handlers_prompt.on_prompt_rename(
                    prompt_key,
                    rename_name,
                    prompt_mode,
                    prompt_content_type,
                    prompt_learning_mode,
                    prompt_difficulty,
                    run_content_type,
                    run_learning_mode,
                    run_difficulty,
                    run_extract_val,
                    run_explain_val,
                    config_extract_val,
                    config_explain_val,
                    ui_lang_val,
                    deps=prompt_deps,
                )

            def _on_prompt_delete(
                prompt_key: str,
                delete_confirmed: bool,
                prompt_mode: str,
                prompt_content_type: str,
                prompt_learning_mode: str,
                prompt_difficulty: str,
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                config_extract_val: str,
                config_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, ...]:
                return handlers_prompt.on_prompt_delete(
                    prompt_key,
                    delete_confirmed,
                    prompt_mode,
                    prompt_content_type,
                    prompt_learning_mode,
                    prompt_difficulty,
                    run_content_type,
                    run_learning_mode,
                    run_difficulty,
                    run_extract_val,
                    run_explain_val,
                    config_extract_val,
                    config_explain_val,
                    ui_lang_val,
                    deps=prompt_deps,
                )

            def _on_run_prompt_filters_change(
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, Any]:
                return handlers_prompt.on_run_prompt_filters_change(
                    run_content_type,
                    run_learning_mode,
                    run_difficulty,
                    run_extract_val,
                    run_explain_val,
                    ui_lang_val,
                    deps=prompt_deps,
                )

            prompt_file_selector.change(
                _on_prompt_file_change,
                inputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                ],
            )
            prompt_mode_selector.change(
                _on_prompt_mode_change,
                inputs=[
                    prompt_mode_selector,
                    prompt_file_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                ],
            )
            prompt_content_type_selector.change(
                _on_prompt_filter_change,
                inputs=[
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_mode_selector,
                    prompt_file_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                ],
            )
            prompt_learning_mode_selector.change(
                _on_prompt_filter_change,
                inputs=[
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_mode_selector,
                    prompt_file_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                ],
            )
            prompt_difficulty_selector.change(
                _on_prompt_filter_change,
                inputs=[
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_mode_selector,
                    prompt_file_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                ],
            )
            content_profile.change(
                _on_run_prompt_filters_change,
                inputs=[
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    ui_lang,
                ],
                outputs=[run_extract_prompt, run_explain_prompt],
            )
            learning_mode.change(
                _on_run_prompt_filters_change,
                inputs=[
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    ui_lang,
                ],
                outputs=[run_extract_prompt, run_explain_prompt],
            )
            difficulty.change(
                _on_run_prompt_filters_change,
                inputs=[
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    ui_lang,
                ],
                outputs=[run_extract_prompt, run_explain_prompt],
            )
            prompt_new_btn.click(
                _on_prompt_new,
                inputs=[
                    prompt_file_selector,
                    prompt_new_name,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    prompt_new_name,
                    prompt_rename_name,
                    prompt_save_confirm,
                    prompt_delete_confirm,
                ],
            )
            prompt_save_btn.click(
                _on_prompt_save,
                inputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_save_confirm,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    prompt_new_name,
                    prompt_rename_name,
                    prompt_save_confirm,
                    prompt_delete_confirm,
                ],
                js="""
(prompt_key, prompt_mode, prompt_content_type, prompt_learning_mode, prompt_difficulty, prompt_template, _save_confirmed, run_content_type, run_learning_mode, run_difficulty, run_extract_val, run_explain_val, config_extract_val, config_explain_val, ui_lang_val) => {
    const message = ui_lang_val === "zh" ? "确认保存当前提示词文件？" : "Confirm saving the current prompt file?";
    return [
        prompt_key,
        prompt_mode,
        prompt_content_type,
        prompt_learning_mode,
        prompt_difficulty,
        prompt_template,
        window.confirm(message),
        run_content_type,
        run_learning_mode,
        run_difficulty,
        run_extract_val,
        run_explain_val,
        config_extract_val,
        config_explain_val,
        ui_lang_val,
    ];
}
""",
            )
            prompt_rename_btn.click(
                _on_prompt_rename,
                inputs=[
                    prompt_file_selector,
                    prompt_rename_name,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    prompt_new_name,
                    prompt_rename_name,
                    prompt_save_confirm,
                    prompt_delete_confirm,
                ],
            )
            prompt_load_default_btn.click(
                _on_prompt_delete,
                inputs=[
                    prompt_file_selector,
                    prompt_delete_confirm,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    prompt_new_name,
                    prompt_rename_name,
                    prompt_save_confirm,
                    prompt_delete_confirm,
                ],
                js="""
(prompt_key, _delete_confirmed, prompt_mode, prompt_content_type, prompt_learning_mode, prompt_difficulty, run_content_type, run_learning_mode, run_difficulty, run_extract_val, run_explain_val, config_extract_val, config_explain_val, ui_lang_val) => {
    const message = ui_lang_val === "zh" ? "确认删除当前提示词文件？" : "Confirm deleting the current prompt file?";
    return [
        prompt_key,
        window.confirm(message),
        prompt_mode,
        prompt_content_type,
        prompt_learning_mode,
        prompt_difficulty,
        run_content_type,
        run_learning_mode,
        run_difficulty,
        run_extract_val,
        run_explain_val,
        config_extract_val,
        config_explain_val,
        ui_lang_val,
    ];
}
""",
            )

        with gr.Tab(
            _tr(initial_ui_lang, "Run analytics", "杩愯缁熻鍒嗘瀽")
        ) as analytics_tab:
            analytics_heading.render()
            with gr.Row():
                taxonomy_filter.render()
                transfer_filter.render()
                rejection_filter.render()
                chunk_filter.render()
            apply_analysis_filter_btn.render()
        ui_deps = handlers_ui.UiDeps(
            normalize_ui_lang=_normalize_ui_lang,
            load_app_config=_load_app_config,
            prompt_file_map=_prompt_file_map,
            normalize_prompt_mode=_normalize_prompt_mode,
            normalize_prompt_content_type=_normalize_prompt_content_type,
            normalize_prompt_learning_mode=_normalize_prompt_learning_mode,
            normalize_prompt_difficulty=_normalize_prompt_difficulty,
            load_prompt_mode=_load_prompt_mode,
            pick_prompt_key=lambda prompt_files_now, **kwargs: handlers_prompt.pick_prompt_key(  # placeholder
                prompt_files_now, deps=prompt_deps, **kwargs
            ),
            load_prompt_template=_load_prompt_template,
            load_prompt_filter_metadata=_load_prompt_filter_metadata,
            prompt_mode_choices_for_ui=lambda lang: handlers_prompt.prompt_mode_choices_for_ui(  # placeholder
                lang, deps=prompt_deps
            ),
            prompt_path_choices=_prompt_path_choices,
            refresh_recent_runs=_refresh_recent_runs,
            normalize_dropdown_value=lambda current, choices: handlers_prompt.normalize_dropdown_value(
                current, choices
            ),
            tr=_tr,
            prompt_choices_from_map=_prompt_choices_from_map,
            prompt_content_type_options=_PROMPT_CONTENT_TYPE_OPTIONS,
            prompt_learning_mode_options=_PROMPT_LEARNING_MODE_OPTIONS,
            prompt_difficulty_options=_PROMPT_DIFFICULTY_OPTIONS,
        )
            run_id_current: str | None,
        ) -> tuple[Any, ...]:
            lang = _normalize_ui_lang(lang_value)
            _ = prompt_lang_current
            prompt_lang_next = lang
            cfg_now = _load_app_config()
            prompt_files_now = _prompt_file_map(cfg_now)
            prompt_mode_pref = _normalize_prompt_mode(prompt_mode_current)
            prompt_content_type_pref = _normalize_prompt_content_type(
                prompt_content_type_current
            )
            prompt_learning_mode_pref = _normalize_prompt_learning_mode(
                prompt_learning_mode_current
            )
            prompt_difficulty_pref = _normalize_prompt_difficulty(
                prompt_difficulty_current
            )
            if not prompt_mode_pref and prompt_key_current in prompt_files_now:
                prompt_mode_pref = _load_prompt_mode(
                    prompt_key_current, prompt_files_now, lang=lang
                )
            if not prompt_mode_pref:
                prompt_mode_pref = "extraction"
            prompt_files_filtered_next = _prompt_file_map(
                cfg_now,
                mode_filter=prompt_mode_pref,
                content_type_filter=prompt_content_type_pref,
                learning_mode_filter=prompt_learning_mode_pref,
                difficulty_filter=prompt_difficulty_pref,
            )
            prompt_key_next = _pick_prompt_key(
                prompt_files_filtered_next,
                lang=lang,
                preferred_key=prompt_key_current,
                preferred_mode=prompt_mode_pref,
            )
            if not prompt_key_next and prompt_files_now:
                prompt_key_next = _pick_prompt_key(
                    prompt_files_now,
                    lang=lang,
                    preferred_key=prompt_key_current,
                    preferred_mode=prompt_mode_pref,
                )
            prompt_mode_next = (
                _load_prompt_mode(prompt_key_next, prompt_files_now, lang=lang)
                or prompt_mode_pref
            )
            prompt_files_filtered_next = _prompt_file_map(
                cfg_now,
                mode_filter=prompt_mode_next,
                content_type_filter=prompt_content_type_pref,
                learning_mode_filter=prompt_learning_mode_pref,
                difficulty_filter=prompt_difficulty_pref,
            )
            if prompt_key_next:
                prompt_template_next, prompt_status_next = _load_prompt_template(
                    prompt_key_next,
                    prompt_files_now,
                    lang=lang,
                )
                if (
                    prompt_content_type_pref == "all"
                    and prompt_learning_mode_pref == "all"
                    and prompt_difficulty_pref == "all"
                ):
                    (
                        prompt_content_type_pref,
                        prompt_learning_mode_pref,
                        prompt_difficulty_pref,
                    ) = _load_prompt_filter_metadata(
                        prompt_key_next, prompt_files_now, lang=lang
                    )
                    prompt_files_filtered_next = _prompt_file_map(
                        cfg_now,
                        mode_filter=prompt_mode_next,
                        content_type_filter=prompt_content_type_pref,
                        learning_mode_filter=prompt_learning_mode_pref,
                        difficulty_filter=prompt_difficulty_pref,
                    )
            else:
                prompt_template_next, prompt_status_next = "", ""
            prompt_mode_choices_next = _prompt_mode_choices_for_ui(lang)
            run_content_type_pref = _normalize_prompt_content_type(
                run_content_type_current
            )
            run_learning_mode_pref = _normalize_prompt_learning_mode(
                run_learning_mode_current
            )
            run_difficulty_pref = _normalize_prompt_difficulty(run_difficulty_current)
            run_extract_prompt_choices_next = _prompt_path_choices(
                cfg_now,
                lang=lang,
                mode_filter="extraction",
                content_type_filter=run_content_type_pref,
                learning_mode_filter=run_learning_mode_pref,
                difficulty_filter=run_difficulty_pref,
                include_auto=True,
            )
            run_explain_prompt_choices_next = _prompt_path_choices(
                cfg_now,
                lang=lang,
                mode_filter="explanation",
                content_type_filter=run_content_type_pref,
                learning_mode_filter=run_learning_mode_pref,
                difficulty_filter=run_difficulty_pref,
                include_auto=True,
            )
            config_extract_prompt_choices_next = _prompt_path_choices(
                cfg_now,
                lang=lang,
                mode_filter="extraction",
                include_auto=True,
            )
            config_explain_prompt_choices_next = _prompt_path_choices(
                cfg_now,
                lang=lang,
                mode_filter="explanation",
                include_auto=True,
            )
            run_selector_next, run_detail_next, run_download_next = (
                _refresh_recent_runs(
                    cfg_now,
                    lang=lang,
                    preferred_run_id=run_id_current,
                )
            )
            selector_choices = run_selector_next.get("choices", [])
            selector_value = run_selector_next.get("value")
            return (
                gr.update(label=_tr(lang, "UI language", "UI language")),
                gr.update(
                    value=_tr(
                        lang,
                        "# ClawLingua Web UI\nLocal deck builder for text learning.",
                        "# ClawLingua Web UI\nLocal deck builder for text learning.",
                    )
                ),
                gr.update(label=_tr(lang, "Run", "Run")),
                gr.update(label=_tr(lang, "Config", "Config")),
                gr.update(label=_tr(lang, "Prompt", "Prompt")),
                gr.update(label=_tr(lang, "Input file", "Input file")),
                gr.update(
                    label=_tr(lang, "Deck title (optional)", "Deck title (optional)")
                ),
                gr.update(label=_tr(lang, "Source language", "Source language")),
                gr.update(label=_tr(lang, "Target language", "Target language")),
                gr.update(label=_tr(lang, "Content profile", "Content profile")),
                gr.update(label=_tr(lang, "Difficulty", "Difficulty")),
                gr.update(
                    label=_tr(
                        lang,
                        "Extraction prompt (run override)",
                        "Extraction prompt (run override)",
                    ),
                    info=_tr(
                        lang,
                        "Equivalent to CLI --extract-prompt.",
                        "Equivalent to CLI --extract-prompt.",
                    ),
                    choices=run_extract_prompt_choices_next,
                    value=_normalize_dropdown_value(
                        run_extract_prompt_current, run_extract_prompt_choices_next
                    ),
                ),
                gr.update(
                    label=_tr(
                        lang,
                        "Explanation prompt (run override)",
                        "Explanation prompt (run override)",
                    ),
                    info=_tr(
                        lang,
                        "Equivalent to CLI --explain-prompt.",
                        "Equivalent to CLI --explain-prompt.",
                    ),
                    choices=run_explain_prompt_choices_next,
                    value=_normalize_dropdown_value(
                        run_explain_prompt_current, run_explain_prompt_choices_next
                    ),
                ),
                gr.update(
                    label=_tr(
                        lang, "Max notes (0 = no limit)", "Max notes (0 = no limit)"
                    ),
                    info=_tr(
                        lang,
                        "Maximum notes after dedupe. Empty/0 means no limit.",
                        "Maximum notes after dedupe. Empty/0 means no limit.",
                    ),
                ),
                gr.update(
                    label=_tr(lang, "Input char limit", "Input char limit"),
                    info=_tr(
                        lang,
                        "Only process the first N chars of input. Empty means no limit.",
                        "Only process the first N chars of input. Empty means no limit.",
                    ),
                ),
                gr.update(label=_tr(lang, "Advanced", "Advanced")),
                gr.update(
                    label=_tr(
                        lang,
                        "Cloze min chars (override env)",
                        "Cloze min chars (override env)",
                    ),
                    info=_tr(
                        lang,
                        "One-run override for CLAWLINGUA_CLOZE_MIN_CHARS.",
                        "One-run override for CLAWLINGUA_CLOZE_MIN_CHARS.",
                    ),
                ),
                gr.update(
                    label=_tr(
                        lang,
                        "Chunk max chars (override env)",
                        "Chunk max chars (override env)",
                    ),
                    info=_tr(
                        lang,
                        "One-run override for CLAWLINGUA_CHUNK_MAX_CHARS.",
                        "One-run override for CLAWLINGUA_CHUNK_MAX_CHARS.",
                    ),
                ),
                gr.update(
                    label=_tr(
                        lang, "Temperature (override env)", "Temperature (override env)"
                    ),
                    info=_tr(
                        lang,
                        "0 is more deterministic; higher values are more random.",
                        "0 is more deterministic; higher values are more random.",
                    ),
                ),
                gr.update(
                    label=_tr(
                        lang, "Save intermediate files", "Save intermediate files"
                    ),
                    info=_tr(
                        lang,
                        "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.",
                        "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.",
                    ),
                ),
                gr.update(
                    label=_tr(lang, "Continue on error", "Continue on error"),
                    info=_tr(
                        lang,
                        "If enabled, continue processing after per-item failures.",
                        "If enabled, continue processing after per-item failures.",
                    ),
                ),
                gr.update(value=_tr(lang, "Run", "Run")),
                gr.update(label=_tr(lang, "Status", "Status")),
                gr.update(label=_tr(lang, "Download .apkg", "Download .apkg")),
                gr.update(value=_tr(lang, "### Recent runs", "### Recent runs")),
                gr.update(value=_tr(lang, "Refresh runs", "Refresh runs")),
                gr.update(
                    label=_tr(lang, "Run ID", "Run ID"),
                    choices=selector_choices,
                    value=selector_value,
                ),
                gr.update(value=run_detail_next),
                gr.update(
                    label=_tr(lang, "Download .apkg", "Download .apkg"),
                    value=run_download_next,
                ),
                gr.update(
                    value=_tr(
                        lang, "### Config (.env editor)", "### Config (.env editor)"
                    )
                ),
                gr.update(label=_tr(lang, "Extraction LLM", "Extraction LLM")),
                gr.update(
                    info=_tr(
                        lang,
                        "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).",
                        "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "API key for extraction LLM, when required.",
                        "API key for extraction LLM, when required.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Model name for extraction LLM.",
                        "Model name for extraction LLM.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Request timeout in seconds.",
                        "Request timeout in seconds.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Default temperature for extraction LLM.",
                        "Default temperature for extraction LLM.",
                    )
                ),
                gr.update(value=_tr(lang, "List models", "List models")),
                gr.update(value=_tr(lang, "Test", "Test")),
                gr.update(
                    label=_tr(lang, "Extraction LLM status", "Extraction LLM status")
                ),
                gr.update(label=_tr(lang, "Explanation LLM", "Explanation LLM")),
                gr.update(
                    info=_tr(
                        lang,
                        "Optional base URL for explanation model.",
                        "Optional base URL for explanation model.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "API key for explanation LLM, when required.",
                        "API key for explanation LLM, when required.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Model name for explanation LLM.",
                        "Model name for explanation LLM.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Default temperature for explanation LLM.",
                        "Default temperature for explanation LLM.",
                    )
                ),
                gr.update(
                    value=_tr(
                        lang, "List models (explanation)", "List models (explanation)"
                    )
                ),
                gr.update(value=_tr(lang, "Test (explanation)", "Test (explanation)")),
                gr.update(
                    label=_tr(lang, "Explanation LLM status", "Explanation LLM status")
                ),
                gr.update(label=_tr(lang, "Chunk & Cloze", "Chunk & Cloze")),
                gr.update(
                    info=_tr(
                        lang,
                        "Default max chars per chunk.",
                        "Default max chars per chunk.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Minimum chars required for cloze text.",
                        "Minimum chars required for cloze text.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Max cards per chunk after dedupe. Empty/0 means unlimited.",
                        "Max cards per chunk after dedupe. Empty/0 means unlimited.",
                    )
                ),
                gr.update(
                    label="CLAWLINGUA_PROMPT_LANG",
                    info=_tr(
                        lang,
                        "Prompt language for multi-lingual prompts (en/zh).",
                        "Prompt language for multi-lingual prompts (en/zh).",
                    ),
                    value=prompt_lang_next,
                ),
                gr.update(
                    label="CLAWLINGUA_EXTRACT_PROMPT",
                    info=_tr(
                        lang,
                        "Default extraction prompt path.",
                        "Default extraction prompt path.",
                    ),
                    choices=config_extract_prompt_choices_next,
                ),
                gr.update(
                    label="CLAWLINGUA_EXPLAIN_PROMPT",
                    info=_tr(
                        lang,
                        "Default explanation prompt path.",
                        "Default explanation prompt path.",
                    ),
                    choices=config_explain_prompt_choices_next,
                ),
                gr.update(label=_tr(lang, "Paths & defaults", "Paths & defaults")),
                gr.update(
                    info=_tr(
                        lang,
                        "Directory for intermediate run data (JSONL, media).",
                        "Directory for intermediate run data (JSONL, media).",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Default directory for exported decks.",
                        "Default directory for exported decks.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang, "Directory for log files.", "Directory for log files."
                    )
                ),
                gr.update(
                    value=_tr(
                        lang,
                        "Load defaults from ENV_EXAMPLE.md",
                        "Load defaults from ENV_EXAMPLE.md",
                    )
                ),
                gr.update(value=_tr(lang, "Save config", "Save config")),
                gr.update(label=_tr(lang, "TTS voices (Edge)", "TTS voices (Edge)")),
                gr.update(
                    value=_tr(
                        lang,
                        "Voice reference: [Edge TTS Voice Samples](https://tts.travisvn.com/)",
                        "Voice reference: [Edge TTS Voice Samples](https://tts.travisvn.com/)",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Configure 4 voice slots used for random selection.",
                        "Configure 4 voice slots used for random selection.",
                    )
                ),
                gr.update(
                    value=_tr(
                        lang, "### Prompt template editor", "### Prompt template editor"
                    )
                ),
                gr.update(
                    label=_tr(lang, "Prompt file", "Prompt file"),
                    choices=_prompt_choices_from_map(
                        prompt_files_filtered_next, lang=lang
                    ),
                    value=prompt_key_next,
                ),
                gr.update(
                    label=_tr(lang, "Prompt type", "Prompt type"),
                    choices=prompt_mode_choices_next,
                    value=prompt_mode_next,
                ),
                gr.update(
                    label=_tr(lang, "Prompt content type", "Prompt content type"),
                    choices=_PROMPT_CONTENT_TYPE_OPTIONS,
                    value=prompt_content_type_pref,
                ),
                gr.update(
                    label=_tr(lang, "Prompt learning mode", "Prompt learning mode"),
                    choices=_PROMPT_LEARNING_MODE_OPTIONS,
                    value=prompt_learning_mode_pref,
                ),
                gr.update(
                    label=_tr(lang, "Prompt difficulty", "Prompt difficulty"),
                    choices=_PROMPT_DIFFICULTY_OPTIONS,
                    value=prompt_difficulty_pref,
                ),
                gr.update(
                    label=_tr(lang, "New prompt file name", "New prompt file name")
                ),
                gr.update(label=_tr(lang, "Rename to", "Rename to")),
                gr.update(
                    label=_tr(lang, "Prompt template", "Prompt template"),
                    value=prompt_template_next,
                ),
                gr.update(value=_tr(lang, "New", "New")),
                gr.update(value=_tr(lang, "Save", "Save")),
                gr.update(value=_tr(lang, "Rename", "Rename")),
                gr.update(value=_tr(lang, "Delete", "Delete")),
                gr.update(value=False, visible=False),
                gr.update(value=False, visible=False),
                gr.update(
                    label=_tr(lang, "Prompt status", "Prompt status"),
                    value=prompt_status_next,
                ),
                gr.update(label=_tr(lang, "Run analytics", "杩愯缁熻鍒嗘瀽")),
            )

        ui_lang.change(
            _on_ui_lang_change,
            inputs=[
                ui_lang,
                prompt_lang_env,
                prompt_file_selector,
                prompt_mode_selector,
                prompt_content_type_selector,
                prompt_learning_mode_selector,
                prompt_difficulty_selector,
                content_profile,
                learning_mode,
                difficulty,
                run_extract_prompt,
                run_explain_prompt,
                run_selector,
            ],
            outputs=[
                ui_lang,
                title_md,
                run_tab,
                config_tab,
                prompt_tab,
                input_file,
                deck_title,
                source_lang,
                target_lang,
                content_profile,
                difficulty,
                run_extract_prompt,
                run_explain_prompt,
                max_notes,
                input_char_limit,
                run_advanced,
                cloze_min_chars,
                chunk_max_chars,
                temperature,
                save_intermediate,
                continue_on_error,
                run_button,
                run_status,
                output_file,
                recent_runs_heading,
                refresh_runs_button,
                run_selector,
                run_detail,
                run_download_file,
                config_heading,
                llm_accordion,
                llm_base_url,
                llm_api_key,
                llm_model,
                llm_timeout,
                llm_temperature_env,
                llm_list_models_btn,
                llm_test_btn,
                llm_status,
                translate_accordion,
                translate_base_url,
                translate_api_key,
                translate_model,
                translate_temperature,
                translate_list_models_btn,
                translate_test_btn,
                translate_status,
                chunk_accordion,
                chunk_max_chars_env,
                cloze_min_chars_env,
                cloze_max_per_chunk_env,
                prompt_lang_env,
                extract_prompt_env,
                explain_prompt_env,
                paths_accordion,
                output_dir_env,
                export_dir_env,
                log_dir_env,
                load_defaults_btn,
                save_config_btn,
                tts_accordion,
                tts_hint_md,
                tts_voice1_env,
                prompt_heading,
                prompt_file_selector,
                prompt_mode_selector,
                prompt_content_type_selector,
                prompt_learning_mode_selector,
                prompt_difficulty_selector,
                prompt_new_name,
                prompt_rename_name,
                prompt_editor,
                prompt_new_btn,
                prompt_save_btn,
                prompt_rename_btn,
                prompt_load_default_btn,
                prompt_save_confirm,
                prompt_delete_confirm,
                prompt_status,
                analytics_tab,
            ],
        )

    return demo


def launch(*, server_port: int | None = None, server_host: str | None = None) -> None:
    """Launch the Gradio app.

    Logging is configured via the shared `setup_logging` function when
    loading the application config. Web-specific events are logged under
    the `clawlingua.web` logger.
    """

    port_value = server_port
    if port_value is None:
        env_port = _to_optional_int(os.getenv("CLAWLINGUA_WEB_PORT"), min_value=1)
        port_value = env_port or 7860
    host_value = (
        server_host or os.getenv("CLAWLINGUA_WEB_HOST") or "0.0.0.0"
    ).strip() or "0.0.0.0"

    logger.info("starting ClawLingua web UI | host=%s port=%d", host_value, port_value)
    demo = build_interface()
    demo.queue().launch(server_name=host_value, server_port=port_value)
    logger.info("ClawLingua web UI stopped")


if __name__ == "__main__":  # pragma: no cover
    launch()
