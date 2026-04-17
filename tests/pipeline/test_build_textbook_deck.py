from __future__ import annotations

import json
from pathlib import Path

import pytest

from clawlearn.config import AppConfig
from clawlearn.errors import ClawLearnError
from clawlearn.pipeline.build_textbook_deck import (
    BuildTextbookDeckOptions,
    _build_textbook_cloze,
    _extract_concept_candidate,
    _extract_concept_candidates,
    _resolve_learning_mode,
    _split_sentences,
    run_build_textbook_deck,
)


def test_split_sentences_handles_basic_punctuation() -> None:
    text = "First idea here. Second point follows! Third question?"
    assert _split_sentences(text) == [
        "First idea here.",
        "Second point follows!",
        "Third question?",
    ]


def test_extract_concept_candidate_returns_none_for_short_sentence() -> None:
    assert _extract_concept_candidate("Hi there.") is None


def test_extract_concept_candidate_picks_first_sentence_and_title() -> None:
    concept = _extract_concept_candidate(
        "Neural networks optimize parameters for robust predictions. Another sentence."
    )
    assert concept is not None
    title, sentence, explanation = concept
    assert title == "Neural networks optimize parameters for robust"
    assert sentence == "Neural networks optimize parameters for robust predictions."
    assert explanation.startswith("Key idea:")


def test_extract_concept_candidates_default_preserves_first_sentence_only() -> None:
    concepts = _extract_concept_candidates(
        "Hi there. Neural networks optimize parameters for robust predictions.",
        max_concepts_per_chunk=1,
    )
    assert concepts == []


def test_extract_concept_candidates_supports_multiple_when_requested() -> None:
    concepts = _extract_concept_candidates(
        "Neural networks optimize parameters for robust predictions. "
        "Gradient descent updates weights for better convergence.",
        max_concepts_per_chunk=2,
    )
    assert len(concepts) == 2
    assert concepts[0][0] == "Neural networks optimize parameters for robust"
    assert concepts[1][0] == "Gradient descent updates weights for better"


def test_build_textbook_cloze_inserts_match_case_insensitive() -> None:
    sentence = "Neural Networks optimize parameters for robust predictions."
    text = _build_textbook_cloze(sentence, "neural networks")
    assert text.startswith("{{c1::<b>Neural Networks</b>}}(concept)")
    assert "optimize parameters" in text


def test_build_textbook_cloze_fallback_when_not_found() -> None:
    sentence = "A complete sentence without the target phrase."
    text = _build_textbook_cloze(sentence, "different phrase")
    assert text.startswith("{{c1::<b>different phrase</b>}}(concept): ")
    assert sentence in text


def test_resolve_learning_mode_defaults_to_textbook_focus() -> None:
    options = BuildTextbookDeckOptions(input_value="input.txt")
    assert _resolve_learning_mode(options) == "textbook_focus"


def test_resolve_learning_mode_rejects_non_textbook_mode() -> None:
    options = BuildTextbookDeckOptions(
        input_value="input.txt",
        learning_mode="lingua_expression",
    )
    with pytest.raises(ClawLearnError) as exc_info:
        _resolve_learning_mode(options)
    assert exc_info.value.error_code == "ARG_LEARNING_MODE_INVALID"


def test_run_build_textbook_deck_writes_observability_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text(
        "Neural networks optimize parameters for robust predictions. "
        "Gradient descent updates weights for better convergence.",
        encoding="utf-8",
    )

    cfg = AppConfig(
        workspace_root=tmp_path,
        output_dir=Path("runs"),
        export_dir=Path("outputs"),
        save_intermediate=True,
        allow_empty_deck=True,
    )

    monkeypatch.setattr(
        "clawlearn.pipeline.build_textbook_deck.validate_base_config",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        "clawlearn.pipeline.build_textbook_deck.validate_runtime_config",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        "clawlearn.pipeline.build_textbook_deck.load_anki_template",
        lambda _path: {"name": "stub"},
    )

    def _fake_export_apkg(**kwargs):  # noqa: ANN003, ANN202
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr("clawlearn.pipeline.build_textbook_deck.export_apkg", _fake_export_apkg)

    result = run_build_textbook_deck(
        cfg,
        BuildTextbookDeckOptions(
            input_value=str(input_path),
            run_id="run_test_textbook",
            learning_mode="textbook_review",
            max_concepts_per_chunk=2,
            save_intermediate=True,
        ),
    )

    run_dir = result.run_dir
    expected_files = [
        "document.json",
        "document.md",
        "chunks.jsonl",
        "candidates.raw.jsonl",
        "candidates.validated.jsonl",
        "cards.final.jsonl",
        "errors.jsonl",
        "run_summary.json",
    ]
    for name in expected_files:
        assert (run_dir / name).exists(), name

    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["domain"] == "textbook"
    assert summary["learning_mode"] == "textbook_review"
    assert summary["schema_name"] == "textbook_concepts_v1"
    assert isinstance(summary["models"], dict)
    assert summary["models"]["candidate_extraction_model"] == "heuristic_local"
    assert summary["models"]["card_generation_model"] == "heuristic_local"
    assert summary["raw_candidates"] >= summary["validated_candidates"]
    assert summary["validated_candidates"] >= summary["cards"]
