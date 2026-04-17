from __future__ import annotations

import json
from pathlib import Path

from clawlearn.config import AppConfig
from clawlearn_web import run_analysis


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def test_run_analysis_prefers_unified_candidates_filenames(tmp_path: Path) -> None:
    cfg = AppConfig(workspace_root=tmp_path, output_dir=Path("runs"))
    run_dir = tmp_path / "runs" / "run_new"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text("{}", encoding="utf-8")
    _write_jsonl(run_dir / "candidates.raw.jsonl", [{"chunk_id": "chunk_0001"}])
    _write_jsonl(
        run_dir / "candidates.validated.jsonl",
        [{"chunk_id": "chunk_0001", "phrase_types": ["strong_collocation"]}],
    )

    payload = run_analysis._run_analysis_payload(  # noqa: SLF001
        "run_new",
        cfg,
        load_full_candidates=True,
    )

    assert len(payload["raw_candidates"]) == 1
    assert len(payload["selected_candidates"]) == 1
    assert payload["selected_candidates"][0]["chunk_id"] == "chunk_0001"


def test_run_analysis_falls_back_to_legacy_candidates_filenames(tmp_path: Path) -> None:
    cfg = AppConfig(workspace_root=tmp_path, output_dir=Path("runs"))
    run_dir = tmp_path / "runs" / "run_legacy"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text("{}", encoding="utf-8")
    _write_jsonl(run_dir / "text_candidates.raw.jsonl", [{"chunk_id": "chunk_0002"}])
    _write_jsonl(
        run_dir / "text_candidates.validated.jsonl",
        [{"chunk_id": "chunk_0002", "phrase_types": ["strong_collocation"]}],
    )

    payload = run_analysis._run_analysis_payload(  # noqa: SLF001
        "run_legacy",
        cfg,
        load_full_candidates=True,
    )

    assert len(payload["raw_candidates"]) == 1
    assert len(payload["selected_candidates"]) == 1
    assert payload["selected_candidates"][0]["chunk_id"] == "chunk_0002"


def test_run_analysis_payload_default_prefers_validated_candidates_for_samples(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(workspace_root=tmp_path, output_dir=Path("runs"))
    run_dir = tmp_path / "runs" / "run_light"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text("{}", encoding="utf-8")
    _write_jsonl(run_dir / "candidates.raw.jsonl", [{"chunk_id": "chunk_raw"}])
    _write_jsonl(
        run_dir / "candidates.validated.jsonl",
        [{"chunk_id": "chunk_validated", "phrase_types": ["strong_collocation"]}],
    )
    _write_jsonl(
        run_dir / "cards.final.jsonl",
        [{"chunk_id": "chunk_card", "phrase_types": ["reusable_high_frequency_chunk"]}],
    )

    payload = run_analysis._run_analysis_payload("run_light", cfg)  # noqa: SLF001

    assert payload["raw_candidates"] == []
    assert len(payload["selected_candidates"]) == 1
    assert payload["selected_candidates"][0]["chunk_id"] == "chunk_validated"


def test_run_analysis_payload_default_falls_back_to_cards_when_validated_missing(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(workspace_root=tmp_path, output_dir=Path("runs"))
    run_dir = tmp_path / "runs" / "run_cards_fallback"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text("{}", encoding="utf-8")
    _write_jsonl(
        run_dir / "cards.final.jsonl",
        [{"chunk_id": "chunk_card", "phrase_types": ["reusable_high_frequency_chunk"]}],
    )

    payload = run_analysis._run_analysis_payload("run_cards_fallback", cfg)  # noqa: SLF001

    assert len(payload["selected_candidates"]) == 1
    assert payload["selected_candidates"][0]["chunk_id"] == "chunk_card"


def test_read_jsonl_dicts_streaming_limits_and_invalid_lines(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"i": 1}),
                "invalid json",
                json.dumps({"i": 2}),
                json.dumps({"i": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows, truncated = run_analysis._read_jsonl_dicts(path, max_lines=2)  # noqa: SLF001
    assert truncated is True
    assert rows == [{"i": 1}]

    rows2, truncated2 = run_analysis._read_jsonl_dicts(path, max_bytes=8)  # noqa: SLF001
    assert truncated2 is True
    assert rows2 == []


def test_build_run_analysis_marks_truncated_samples(tmp_path: Path) -> None:
    cfg = AppConfig(workspace_root=tmp_path, output_dir=Path("runs"))
    run_dir = tmp_path / "runs" / "run_truncated"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps({"metrics": {"deduped_candidates": 250}}),
        encoding="utf-8",
    )
    _write_jsonl(
        run_dir / "cards.final.jsonl",
        [
            {
                "chunk_id": f"chunk_{idx:04d}",
                "text": f"text {idx}",
                "learning_value_score": float(idx),
                "phrase_types": ["reusable_high_frequency_chunk"],
            }
            for idx in range(250)
        ],
    )

    analysis_md, _rows, _tax, _rej, _chunk = run_analysis.build_run_analysis(
        "run_truncated",
        cfg,
        lang="en",
        tr=lambda _lang, en, _zh: en,
    )

    assert "Samples truncated" in analysis_md
    assert "cards.final.jsonl" in analysis_md


def test_build_run_analysis_tolerates_none_or_invalid_scores(tmp_path: Path) -> None:
    cfg = AppConfig(workspace_root=tmp_path, output_dir=Path("runs"))
    run_dir = tmp_path / "runs" / "run_safe_score"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "taxonomy_average_score": {"good": None, "bad": "not-a-number"},
                    "expression_transfer_non_empty_ratio": None,
                }
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        run_dir / "candidates.validated.jsonl",
        [
            {
                "chunk_id": "chunk_0001",
                "text": "sample text",
                "learning_value_score": None,
                "phrase_types": ["reusable_high_frequency_chunk"],
            }
        ],
    )

    analysis_md, rows, _tax, _rej, _chunk = run_analysis.build_run_analysis(
        "run_safe_score",
        cfg,
        lang="en",
        tr=lambda _lang, en, _zh: en,
    )

    assert "Run analytics" in analysis_md
    assert rows
    assert rows[0][3] is None


def test_build_run_analysis_renders_secondary_extraction_metrics(tmp_path: Path) -> None:
    cfg = AppConfig(workspace_root=tmp_path, output_dir=Path("runs"))
    run_dir = tmp_path / "runs" / "run_secondary_metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "secondary_extraction": {
                        "requested": True,
                        "enabled": True,
                        "configured": True,
                        "parallel": True,
                        "execution_mode": "parallel",
                        "secondary_model": "mistral-small3.2:latest",
                        "candidates_primary_count": 11,
                        "candidates_secondary_count": 7,
                        "candidates_merged_count": 15,
                        "dedup_removed_count": 3,
                        "unique_phrase_gain_from_secondary": 4,
                        "secondary_error_type": "",
                        "secondary_error_message": "",
                        "fallback_to_primary": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        run_dir / "cards.final.jsonl",
        [{"chunk_id": "chunk_0001", "text": "sample", "phrase_types": ["strong_collocation"]}],
    )

    analysis_md, _rows, _tax, _rej, _chunk = run_analysis.build_run_analysis(
        "run_secondary_metrics",
        cfg,
        lang="en",
        tr=lambda _lang, en, _zh: en,
    )

    assert "Secondary extraction enabled" in analysis_md
    assert "Secondary extraction parallel" in analysis_md
    assert "`parallel`" in analysis_md
    assert "11/7/15" in analysis_md
    assert "mistral-small3.2:latest" in analysis_md
