from __future__ import annotations

from pathlib import Path

from clawlearn.config import AppConfig
from clawlearn_web import run_history


def _fake_info(run_id: str) -> run_history.RunInfo:
    return run_history.RunInfo(
        run_id=run_id,
        started_at="2026-04-16T00:00:00+00:00",
        finished_at=None,
        title=run_id,
        source_lang="en",
        target_lang="zh",
        content_profile="transcript_dialogue",
        material_profile="transcript_dialogue",
        learning_mode="lingua_expression",
        status="completed",
        cards=0,
        errors=0,
        output_path=None,
    )


def test_scan_runs_reads_only_recent_candidate_dirs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = AppConfig(workspace_root=tmp_path, output_dir=Path("runs"))
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    for idx in range(100):
        (runs_root / f"20260416T1000{idx:02d}Z_run{idx:03d}").mkdir()

    calls: list[str] = []

    def _stub_run_info(_cfg: AppConfig, run_dir: Path) -> run_history.RunInfo:
        calls.append(run_dir.name)
        return _fake_info(run_dir.name)

    monkeypatch.setattr(run_history, "run_info_from_dir", _stub_run_info)
    infos = run_history.scan_runs(cfg, limit=5)

    assert len(calls) == 30
    assert len(infos) == 5


def test_scan_runs_limit_zero_keeps_full_scan(tmp_path: Path, monkeypatch) -> None:
    cfg = AppConfig(workspace_root=tmp_path, output_dir=Path("runs"))
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    for idx in range(3):
        (runs_root / f"run_{idx:02d}").mkdir()

    calls: list[str] = []

    def _stub_run_info(_cfg: AppConfig, run_dir: Path) -> run_history.RunInfo:
        calls.append(run_dir.name)
        return _fake_info(run_dir.name)

    monkeypatch.setattr(run_history, "run_info_from_dir", _stub_run_info)
    infos = run_history.scan_runs(cfg, limit=0)

    assert len(calls) == 3
    assert len(infos) == 3
