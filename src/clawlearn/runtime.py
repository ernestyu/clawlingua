"""Runtime context for a build run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .utils.fs import ensure_dir
from .utils.time import make_run_id


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    media_dir: Path


def create_run_context(cfg: AppConfig, *, name: str = "build_deck", run_id: str | None = None) -> RunContext:
    """Create a run context.

    - ``run_dir`` is always under ``output_dir`` (typically ./runs/<run_id>) and
      holds intermediate JSONL files and media.
    - The final exported deck path is derived separately in the pipeline using
      ``export_dir`` (typically ./outputs/<run_id>/output.apkg) unless
      ``--output`` is provided.
    """

    resolved_run_id = (run_id or "").strip() or make_run_id(name)
    run_dir = ensure_dir(cfg.resolve_path(cfg.output_dir) / resolved_run_id)
    media_dir = ensure_dir(run_dir / "media")
    return RunContext(run_id=resolved_run_id, run_dir=run_dir, media_dir=media_dir)
