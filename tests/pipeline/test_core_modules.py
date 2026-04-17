from __future__ import annotations

from pathlib import Path

from clawlearn.pipeline.core_candidates import dump_candidate_artifacts
from clawlearn.pipeline.core_export import resolve_output_path
from clawlearn.pipeline.core_io import resolve_input_path
from clawlearn.pipeline.core_llm import iter_batches


def test_resolve_input_path_handles_relative_and_absolute(tmp_path: Path) -> None:
    rel = resolve_input_path(workspace_root=tmp_path, input_value="data/input.txt")
    assert rel == (tmp_path / "data/input.txt").resolve()

    abs_path = tmp_path / "abs.txt"
    resolved_abs = resolve_input_path(workspace_root=tmp_path, input_value=str(abs_path))
    assert resolved_abs == abs_path


def test_iter_batches_handles_edge_sizes() -> None:
    assert iter_batches([1, 2, 3], 1) == [[1], [2], [3]]
    assert iter_batches([1, 2, 3], 2) == [[1, 2], [3]]
    # defensive normalization: zero/negative become size=1
    assert iter_batches([1, 2], 0) == [[1], [2]]
    assert iter_batches([1, 2], -4) == [[1], [2]]


def test_resolve_output_path_default_and_explicit(tmp_path: Path) -> None:
    run_id = "run_001"
    output = resolve_output_path(
        workspace_root=tmp_path,
        export_dir=Path("outputs"),
        run_id=run_id,
        explicit_output=None,
    )
    assert output == (tmp_path / "outputs" / run_id / "output.apkg").resolve()
    assert output.parent.exists()

    explicit_rel = resolve_output_path(
        workspace_root=tmp_path,
        export_dir=Path("outputs"),
        run_id=run_id,
        explicit_output=Path("custom/out.apkg"),
    )
    assert explicit_rel == (tmp_path / "custom/out.apkg").resolve()

    explicit_abs = tmp_path / "abs" / "deck.apkg"
    resolved_abs = resolve_output_path(
        workspace_root=tmp_path,
        export_dir=Path("outputs"),
        run_id=run_id,
        explicit_output=explicit_abs,
    )
    assert resolved_abs == explicit_abs


def test_dump_candidate_artifacts_writes_unified_and_optional_legacy_names(
    tmp_path: Path,
) -> None:
    raw = [{"chunk_id": "chunk_0001"}]
    validated = [{"chunk_id": "chunk_0001", "text": "x"}]

    dump_candidate_artifacts(
        run_dir=tmp_path / "run_a",
        raw_candidates=raw,
        validated_candidates=validated,
        write_legacy_text_candidates=True,
    )
    assert (tmp_path / "run_a" / "candidates.raw.jsonl").exists()
    assert (tmp_path / "run_a" / "candidates.validated.jsonl").exists()
    assert (tmp_path / "run_a" / "text_candidates.raw.jsonl").exists()
    assert (tmp_path / "run_a" / "text_candidates.validated.jsonl").exists()

    dump_candidate_artifacts(
        run_dir=tmp_path / "run_b",
        raw_candidates=raw,
        validated_candidates=validated,
        write_legacy_text_candidates=False,
    )
    assert (tmp_path / "run_b" / "candidates.raw.jsonl").exists()
    assert (tmp_path / "run_b" / "candidates.validated.jsonl").exists()
    assert not (tmp_path / "run_b" / "text_candidates.raw.jsonl").exists()
    assert not (tmp_path / "run_b" / "text_candidates.validated.jsonl").exists()
