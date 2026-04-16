"""Run analytics helpers extracted from app.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from clawlearn.pipeline.validators import classify_rejection_reason
from clawlearn_web import run_history

_DEFAULT_SAMPLE_MAX_LINES = 200
_DEFAULT_SAMPLE_MAX_BYTES = 5 * 1024 * 1024
_DEFAULT_FULL_CANDIDATE_MAX_LINES = 2000
_DEFAULT_FULL_CANDIDATE_MAX_BYTES = 20 * 1024 * 1024


def _as_str(value: Any, *, default: str = "") -> str:
    return run_history.as_str(value, default=default)


def _as_int(value: Any, *, default: int = 0) -> int:
    return run_history.as_int(value, default=default)


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_jsonl_dicts(
    path: Path,
    *,
    max_lines: int | None = None,
    max_bytes: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    if not path.exists():
        return [], False
    rows: list[dict[str, Any]] = []
    line_limit = max_lines if max_lines and max_lines > 0 else None
    byte_limit = max_bytes if max_bytes and max_bytes > 0 else None
    consumed_bytes = 0
    read_lines = 0

    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                raw_bytes = len(raw.encode("utf-8"))
                if byte_limit is not None and (consumed_bytes + raw_bytes) > byte_limit:
                    return rows, True
                consumed_bytes += raw_bytes
                read_lines += 1
                if line_limit is not None and read_lines > line_limit:
                    return rows, True

                line = raw.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
    except Exception:
        return rows, False

    return rows, False


def _read_candidates_stage(
    run_dir: Path,
    *,
    stage: str,
    max_lines: int | None = None,
    max_bytes: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    stage_name = "validated" if stage == "validated" else "raw"
    primary = run_dir / f"candidates.{stage_name}.jsonl"
    if primary.exists():
        return _read_jsonl_dicts(primary, max_lines=max_lines, max_bytes=max_bytes)
    legacy = run_dir / f"text_candidates.{stage_name}.jsonl"
    return _read_jsonl_dicts(legacy, max_lines=max_lines, max_bytes=max_bytes)


def _bar(value: float, max_value: float, *, width: int = 18) -> str:
    if max_value <= 0:
        return "." * width
    ratio = max(0.0, min(1.0, value / max_value))
    filled = int(round(ratio * width))
    return ("#" * filled) + ("-" * max(0, width - filled))


def _render_count_histogram(title: str, hist: dict[str, int]) -> str:
    if not hist:
        return f"#### {title}\n- n/a"
    normalized = {str(key): _as_int(value, default=0) for key, value in hist.items()}
    if not normalized:
        return f"#### {title}\n- n/a"
    max_value = max(normalized.values())
    lines = [f"#### {title}"]
    for key, value in sorted(normalized.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- `{key}` `{value}` {_bar(float(value), float(max_value))}")
    return "\n".join(lines)


def _render_score_histogram(title: str, hist: dict[str, float]) -> str:
    if not hist:
        return f"#### {title}\n- n/a"
    normalized = {
        str(key): _as_float(value, default=0.0) for key, value in hist.items()
    }
    if not normalized:
        return f"#### {title}\n- n/a"
    max_value = max(normalized.values())
    lines = [f"#### {title}"]
    for key, value in sorted(normalized.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- `{key}` `{value:.3f}` {_bar(float(value), float(max_value))}")
    return "\n".join(lines)


def _short_text(value: str, *, limit: int = 120) -> str:
    text = _as_str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _score_sort_value(item: dict[str, Any]) -> float:
    return _as_float(item.get("learning_value_score"), default=0.0)


def _score_display_value(item: dict[str, Any]) -> float | None:
    score = item.get("learning_value_score")
    if score is None:
        return None
    if isinstance(score, str) and not score.strip():
        return None
    return _as_float(score, default=0.0)


def _run_analysis_payload(
    run_id: str | None,
    cfg: Any,
    *,
    load_full_candidates: bool = False,
) -> dict[str, Any]:
    selected = _as_str(run_id)
    if not selected:
        return {}
    run_dir = cfg.resolve_path(cfg.output_dir) / selected
    if not run_dir.exists() or not run_dir.is_dir():
        return {}
    summary = run_history.read_run_summary(run_dir / "run_summary.json")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    cards, cards_truncated = _read_jsonl_dicts(
        run_dir / "cards.final.jsonl",
        max_lines=_DEFAULT_SAMPLE_MAX_LINES,
        max_bytes=_DEFAULT_SAMPLE_MAX_BYTES,
    )
    errors, errors_truncated = _read_jsonl_dicts(
        run_dir / "errors.jsonl",
        max_lines=_DEFAULT_SAMPLE_MAX_LINES,
        max_bytes=_DEFAULT_SAMPLE_MAX_BYTES,
    )
    chunks, chunks_truncated = _read_jsonl_dicts(
        run_dir / "chunks.jsonl",
        max_lines=_DEFAULT_SAMPLE_MAX_LINES,
        max_bytes=_DEFAULT_SAMPLE_MAX_BYTES,
    )

    selected_candidates, validated_truncated = _read_candidates_stage(
        run_dir,
        stage="validated",
        max_lines=_DEFAULT_SAMPLE_MAX_LINES,
        max_bytes=_DEFAULT_SAMPLE_MAX_BYTES,
    )
    if not selected_candidates:
        selected_candidates = cards
    raw_candidates: list[dict[str, Any]] = []
    raw_truncated = False

    if load_full_candidates:
        selected_candidates, validated_truncated = _read_candidates_stage(
            run_dir,
            stage="validated",
            max_lines=_DEFAULT_FULL_CANDIDATE_MAX_LINES,
            max_bytes=_DEFAULT_FULL_CANDIDATE_MAX_BYTES,
        )
        raw_candidates, raw_truncated = _read_candidates_stage(
            run_dir,
            stage="raw",
            max_lines=_DEFAULT_FULL_CANDIDATE_MAX_LINES,
            max_bytes=_DEFAULT_FULL_CANDIDATE_MAX_BYTES,
        )

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
        "truncated_files": [
            name
            for name, truncated in (
                ("cards.final.jsonl", cards_truncated),
                ("errors.jsonl", errors_truncated),
                ("chunks.jsonl", chunks_truncated),
                ("candidates.validated.jsonl", validated_truncated),
                ("candidates.raw.jsonl", raw_truncated),
            )
            if truncated
        ],
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
            taxonomy_keys.update(_as_str(item) for item in value.keys() if _as_str(item))
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


def build_run_analysis(
    run_id: str | None,
    cfg: Any,
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
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
            tr(lang, "No run selected.", "No run selected."),
            [],
            [("all", "all")],
            [("all", "all")],
            [("all", "all")],
        )

    summary = payload["summary"]
    metrics = payload["metrics"]
    selected_candidates = payload["selected_candidates"]
    rejected = payload["rejected"]
    truncated_files = (
        payload.get("truncated_files", [])
        if isinstance(payload.get("truncated_files"), list)
        else []
    )
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
        default=_as_str(metrics.get("learning_mode"), default="lingua_expression"),
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
    transfer_ratio = _as_float(
        metrics.get("expression_transfer_non_empty_ratio"),
        default=0.0,
    )
    avg_clozes = _as_float(metrics.get("avg_clozes_per_candidate"), default=0.0)
    avg_phrases = _as_float(
        metrics.get("avg_target_phrases_per_candidate"),
        default=0.0,
    )
    avg_selected_per_chunk = _as_float(
        metrics.get("avg_selected_candidates_per_chunk"),
        default=0.0,
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
    secondary_metrics = (
        metrics.get("secondary_extraction")
        if isinstance(metrics.get("secondary_extraction"), dict)
        else {}
    )
    secondary_enabled = bool(secondary_metrics.get("enabled", False))
    secondary_requested = bool(secondary_metrics.get("requested", False))
    secondary_model = _as_str(secondary_metrics.get("secondary_model"))
    secondary_primary_count = _as_int(
        secondary_metrics.get("candidates_primary_count"),
        default=0,
    )
    secondary_count = _as_int(
        secondary_metrics.get("candidates_secondary_count"),
        default=0,
    )
    secondary_merged_count = _as_int(
        secondary_metrics.get("candidates_merged_count"),
        default=0,
    )
    secondary_dedup_removed = _as_int(
        secondary_metrics.get("dedup_removed_count"),
        default=0,
    )
    secondary_unique_gain = _as_int(
        secondary_metrics.get("unique_phrase_gain_from_secondary"),
        default=0,
    )
    secondary_error_type = _as_str(secondary_metrics.get("secondary_error_type"))
    secondary_error_message = _as_str(secondary_metrics.get("secondary_error_message"))
    secondary_fallback = bool(secondary_metrics.get("fallback_to_primary", False))

    lines = [
        f"### {tr(lang, 'Run analytics', 'Run analytics')}",
        f"- {tr(lang, 'Learning mode', 'Learning mode')}: `{learning_mode}`",
        f"- {tr(lang, 'Material profile', 'Material profile')}: `{material_profile}`",
        f"- {tr(lang, 'Difficulty', 'Difficulty')}: `{difficulty}`",
        f"- {tr(lang, 'Chunks', 'Chunks')}: **{chunks_total}**",
        f"- {tr(lang, 'Raw candidates', 'Raw candidates')}: **{raw_total}**",
        f"- {tr(lang, 'Validated candidates', 'Validated candidates')}: **{valid_total}**",
        f"- {tr(lang, 'Selected cards', 'Selected cards')}: **{selected_total}**",
        f"- {tr(lang, 'Transfer non-empty ratio', 'Transfer non-empty ratio')}: **{transfer_ratio:.2%}**",
        f"- {tr(lang, 'Avg clozes per candidate', 'Avg clozes per candidate')}: **{avg_clozes:.2f}**",
        f"- {tr(lang, 'Avg target phrases per candidate', 'Avg target phrases per candidate')}: **{avg_phrases:.2f}**",
        f"- {tr(lang, 'Avg selected per chunk', 'Avg selected per chunk')}: **{avg_selected_per_chunk:.2f}**",
        f"- {tr(lang, 'Filtered selected items', 'Filtered selected items')}: **{len(filtered_selected)}**",
        f"- {tr(lang, 'Filtered rejected items', 'Filtered rejected items')}: **{len(filtered_rejected)}**",
        f"- {tr(lang, 'Secondary extraction requested', 'Secondary extraction requested')}: **{secondary_requested}**",
        f"- {tr(lang, 'Secondary extraction enabled', 'Secondary extraction enabled')}: **{secondary_enabled}**",
        f"- {tr(lang, 'Secondary extraction model', 'Secondary extraction model')}: `{secondary_model or '-'}`",
        f"- {tr(lang, 'Secondary candidates (primary/secondary/merged)', 'Secondary candidates (primary/secondary/merged)')}: **{secondary_primary_count}/{secondary_count}/{secondary_merged_count}**",
        f"- {tr(lang, 'Secondary dedup removed', 'Secondary dedup removed')}: **{secondary_dedup_removed}**",
        f"- {tr(lang, 'Secondary unique phrase gain', 'Secondary unique phrase gain')}: **{secondary_unique_gain}**",
        f"- {tr(lang, 'Secondary fallback to primary', 'Secondary fallback to primary')}: **{secondary_fallback}**",
    ]
    if secondary_error_type or secondary_error_message:
        lines.append(
            f"- {tr(lang, 'Secondary error', 'Secondary error')}: "
            f"`{secondary_error_type or 'other'}` {secondary_error_message}"
        )
    if truncated_files:
        lines.append(
            f"- {tr(lang, 'Samples truncated', 'Samples truncated')}: "
            f"`{', '.join(sorted(_as_str(name) for name in truncated_files if _as_str(name)))}`"
        )
    lines.extend(
        [
        "",
        _render_count_histogram(
            tr(lang, "Model taxonomy histogram", "Model taxonomy histogram"),
            model_hist,
        ),
        _render_count_histogram(
            tr(lang, "Candidate taxonomy histogram", "Candidate taxonomy histogram"),
            candidate_hist,
        ),
        _render_count_histogram(
            tr(lang, "Selected taxonomy histogram", "Selected taxonomy histogram"),
            selected_hist,
        ),
        _render_score_histogram(
            tr(lang, "Taxonomy average score", "Taxonomy average score"),
            avg_score_hist,
        ),
        _render_count_histogram(
            tr(lang, "Rejection reason histogram", "Rejection reason histogram"),
            rejection_hist,
        ),
        _render_score_histogram(
            tr(
                lang,
                "Transfer non-empty ratio by taxonomy",
                "Transfer non-empty ratio by taxonomy",
            ),
            transfer_by_tax,
        ),
        ]
    )

    sample_rows: list[list[Any]] = []
    seen: set[tuple[str, str, str]] = set()

    if rejection_filter == "all":
        top_selected = sorted(
            filtered_selected,
            key=_score_sort_value,
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
                    _score_display_value(item),
                    _short_text(_as_str(item.get("text"))),
                    _short_text(_as_str(item.get("expression_transfer"))),
                    _short_text(_as_str(item.get("selection_reason"))),
                ]
            )

        taxonomy_bucket: dict[str, dict[str, Any]] = {}
        for item in sorted(
            filtered_selected,
            key=_score_sort_value,
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
                    _score_display_value(item),
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
                    _score_display_value(item),
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
                _score_display_value(candidate),
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
