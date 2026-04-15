"""Shared candidate-stage artifact helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.jsonx import dump_jsonl


def dump_candidate_artifacts(
    *,
    run_dir: Path,
    raw_candidates: list[dict[str, Any]],
    validated_candidates: list[dict[str, Any]],
    write_legacy_text_candidates: bool = False,
) -> None:
    dump_jsonl(run_dir / "candidates.raw.jsonl", raw_candidates)
    dump_jsonl(run_dir / "candidates.validated.jsonl", validated_candidates)
    if write_legacy_text_candidates:
        dump_jsonl(run_dir / "text_candidates.raw.jsonl", raw_candidates)
        dump_jsonl(run_dir / "text_candidates.validated.jsonl", validated_candidates)
