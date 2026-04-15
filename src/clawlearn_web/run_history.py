"""Run history persistence and rendering helpers for the web UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("clawlearn.web")


@dataclass
class RunInfo:
    run_id: str
    started_at: str
    finished_at: str | None
    title: str
    source_lang: str
    target_lang: str
    content_profile: str
    material_profile: str
    learning_mode: str
    status: str
    cards: int
    errors: int
    output_path: str | None
    last_error: str | None = None


def as_str(value: Any, *, default: str = "") -> str:
    if isinstance(value, str):
        text = value.strip()
        return text if text else default
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def as_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_iso_datetime(value: str | None) -> datetime | None:
    text = as_str(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def normalize_run_status(value: Any) -> str:
    status = as_str(value).lower()
    if status in {"running", "completed", "failed", "unknown"}:
        return status
    return "unknown"


def status_text(lang: str, status: str, *, tr: Callable[[str, str, str], str]) -> str:
    normalized = normalize_run_status(status)
    return tr(lang, normalized, normalized)


def build_env_snapshot(cfg: Any) -> dict[str, str]:
    return {
        "CLAWLEARN_LLM_MODEL": as_str(getattr(cfg, "llm_model", "")),
        "CLAWLEARN_TRANSLATE_LLM_MODEL": as_str(
            getattr(cfg, "translate_llm_model", "")
        ),
        "CLAWLEARN_PROMPT_LANG": as_str(getattr(cfg, "prompt_lang", "")),
        "CLAWLEARN_EXTRACT_PROMPT": as_str(getattr(cfg, "extract_prompt", "")),
        "CLAWLEARN_EXPLAIN_PROMPT": as_str(getattr(cfg, "explain_prompt", "")),
        "CLAWLEARN_MATERIAL_PROFILE": as_str(getattr(cfg, "material_profile", "")),
        "CLAWLEARN_LEARNING_MODE": as_str(getattr(cfg, "learning_mode", "")),
    }


def write_run_summary(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:  # pragma: no cover - defensive
        logger.exception("failed to write run summary | path=%s", path)


def read_run_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def update_run_summary(path: Path, updates: dict[str, Any]) -> None:
    payload = read_run_summary(path)
    payload.update(updates)
    write_run_summary(path, payload)


def record_run_start(
    summary_path: Path,
    *,
    run_id: str,
    started_at: str,
    title: str,
    source_lang: str,
    target_lang: str,
    domain: str,
    content_profile: str,
    learning_mode: str,
    difficulty: str,
    extract_prompt_override: str,
    explain_prompt_override: str,
    output_path: str,
    cfg: Any,
) -> None:
    write_run_summary(
        summary_path,
        {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": None,
            "status": "running",
            "title": title,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "domain": as_str(domain, default="lingua"),
            "content_profile": content_profile,
            "material_profile": content_profile,
            "learning_mode": learning_mode,
            "difficulty": difficulty,
            "extract_prompt_override": as_str(extract_prompt_override),
            "explain_prompt_override": as_str(explain_prompt_override),
            "cards": 0,
            "errors": 0,
            "output_path": output_path,
            "env_snapshot": build_env_snapshot(cfg),
        },
    )


def record_run_failed(summary_path: Path, *, finished_at: str, error: str) -> None:
    previous = read_run_summary(summary_path)
    previous_errors = as_int(previous.get("errors"), default=0) if previous else 0
    update_run_summary(
        summary_path,
        {
            "finished_at": finished_at,
            "status": "failed",
            "errors": max(1, previous_errors),
            "last_error": str(error),
        },
    )


def record_run_completed(
    summary_path: Path,
    *,
    finished_at: str,
    cards: int,
    errors: int,
    output_path: str,
) -> None:
    update_run_summary(
        summary_path,
        {
            "finished_at": finished_at,
            "status": "completed",
            "cards": max(0, int(cards)),
            "errors": max(0, int(errors)),
            "output_path": output_path,
            "last_error": None,
        },
    )


def resolve_output_path(cfg: Any, run_dir: Path, output_path: Any) -> Path | None:
    text = as_str(output_path)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    workspace_candidate = (cfg.workspace_root / path).resolve()
    run_dir_candidate = (run_dir / path).resolve()
    if workspace_candidate.exists() or not run_dir_candidate.exists():
        return workspace_candidate
    return run_dir_candidate


def run_started_sort_key(value: str) -> float:
    dt = parse_iso_datetime(value)
    return dt.timestamp() if dt is not None else 0.0


def run_info_from_dir(cfg: Any, run_dir: Path) -> RunInfo:
    run_id = run_dir.name
    summary = read_run_summary(run_dir / "run_summary.json")
    fallback_started = datetime.fromtimestamp(
        run_dir.stat().st_mtime, tz=timezone.utc
    ).isoformat()

    started_at = as_str(summary.get("started_at"), default=fallback_started)
    finished_at_text = as_str(summary.get("finished_at"))
    finished_at = finished_at_text or None
    status = normalize_run_status(summary.get("status"))
    title = as_str(summary.get("title"), default=run_id)
    source_lang = as_str(summary.get("source_lang"))
    target_lang = as_str(summary.get("target_lang"))
    content_profile = as_str(summary.get("content_profile"))
    material_profile = as_str(summary.get("material_profile"), default=content_profile)
    learning_mode = as_str(summary.get("learning_mode"), default="expression_mining")
    cards = max(0, as_int(summary.get("cards"), default=0))
    errors = max(0, as_int(summary.get("errors"), default=0))
    output_path_resolved = resolve_output_path(cfg, run_dir, summary.get("output_path"))
    output_path_text = (
        str(output_path_resolved) if output_path_resolved is not None else None
    )
    output_exists = bool(output_path_resolved and output_path_resolved.exists())
    last_error = as_str(summary.get("last_error")) or None

    if status == "completed" and not output_exists:
        status = "failed"
    if status == "running" and finished_at is not None:
        status = "failed"

    return RunInfo(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        title=title,
        source_lang=source_lang,
        target_lang=target_lang,
        content_profile=content_profile,
        material_profile=material_profile,
        learning_mode=learning_mode,
        status=status,
        cards=cards,
        errors=errors,
        output_path=output_path_text,
        last_error=last_error,
    )


def scan_runs(cfg: Any, *, limit: int = 30) -> list[RunInfo]:
    runs_root = cfg.resolve_path(cfg.output_dir)
    if not runs_root.exists() or not runs_root.is_dir():
        return []

    infos: list[RunInfo] = []
    for entry in runs_root.iterdir():
        if not entry.is_dir():
            continue
        infos.append(run_info_from_dir(cfg, entry))

    infos.sort(key=lambda item: run_started_sort_key(item.started_at), reverse=True)
    max_items = max(0, int(limit))
    if max_items:
        infos = infos[:max_items]
    return infos


def run_choice_label(
    info: RunInfo, *, lang: str, tr: Callable[[str, str, str], str]
) -> str:
    started = info.started_at or "-"
    title = info.title or "-"
    return f"{info.run_id} | {started} | {status_text(lang, info.status, tr=tr)} | {title}"


def load_run_detail(
    run_id: str | None,
    cfg: Any,
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
) -> tuple[str, str | None]:
    selected = as_str(run_id)
    if not selected:
        return tr(lang, "No run selected.", "No run selected."), None

    run_dir = cfg.resolve_path(cfg.output_dir) / selected
    if not run_dir.exists() or not run_dir.is_dir():
        return tr(lang, "No run selected.", "No run selected."), None

    info = run_info_from_dir(cfg, run_dir)
    download_path = None
    if info.output_path:
        candidate = Path(info.output_path)
        if candidate.exists():
            download_path = str(candidate)

    lines = [
        f"### {tr(lang, 'Run details', 'Run details')}",
        f"- {tr(lang, 'Run ID', 'Run ID')}: `{info.run_id}`",
        f"- {tr(lang, 'Status', 'Status')}: **{status_text(lang, info.status, tr=tr)}**",
        f"- {tr(lang, 'Started at', 'Started at')}: `{info.started_at or '-'}`",
        f"- {tr(lang, 'Finished at', 'Finished at')}: `{info.finished_at or '-'}`",
        f"- {tr(lang, 'Title', 'Title')}: `{info.title or '-'}`",
        f"- {tr(lang, 'Source language', 'Source language')}: `{info.source_lang or '-'}`",
        f"- {tr(lang, 'Target language', 'Target language')}: `{info.target_lang or '-'}`",
        f"- {tr(lang, 'Learning mode', 'Learning mode')}: `{info.learning_mode or '-'}`",
        f"- {tr(lang, 'Content profile', 'Content profile')}: `{info.content_profile or '-'}`",
        f"- {tr(lang, 'Material profile', 'Material profile')}: `{info.material_profile or '-'}`",
        f"- {tr(lang, 'Cards', 'Cards')}: **{info.cards}**",
        f"- {tr(lang, 'Errors', 'Errors')}: **{info.errors}**",
        f"- {tr(lang, 'Output path', 'Output path')}: `{info.output_path or '-'}`",
    ]
    if info.last_error:
        lines.append(f"- {tr(lang, 'Last error', 'Last error')}: `{info.last_error}`")
    if download_path is None:
        lines.append(
            f"- {tr(lang, 'Output file not available yet.', 'Output file not available yet.')}"
        )
    return "\n".join(lines), download_path


def recent_runs_view(
    cfg: Any,
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
    preferred_run_id: str | None = None,
    limit: int = 30,
) -> tuple[list[tuple[str, str]], str | None, str, str | None]:
    runs = scan_runs(cfg, limit=limit)
    if not runs:
        return [], None, tr(lang, "No runs found.", "No runs found."), None

    choices = [(run_choice_label(run, lang=lang, tr=tr), run.run_id) for run in runs]
    run_ids = {run.run_id for run in runs}
    selected = preferred_run_id if preferred_run_id in run_ids else runs[0].run_id
    detail, download_path = load_run_detail(selected, cfg, lang=lang, tr=tr)
    return choices, selected, detail, download_path
