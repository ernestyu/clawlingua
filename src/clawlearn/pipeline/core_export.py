"""Core export helpers shared by pipeline domains."""

from __future__ import annotations

from pathlib import Path


def resolve_output_path(
    *,
    workspace_root: Path,
    export_dir: Path,
    run_id: str,
    explicit_output: Path | None,
) -> Path:
    if explicit_output is not None:
        output_path = explicit_output
    else:
        export_root = (workspace_root / export_dir).resolve() if not export_dir.is_absolute() else export_dir
        target_dir = export_root / run_id
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / "output.apkg"
    if output_path.is_absolute():
        return output_path
    return (workspace_root / output_path).resolve()

