from __future__ import annotations

from clawlearn.pipeline.ranking import (
    _extract_pattern_type_scores,
    _model_label_supported,
    score_candidate,
)


def test_extract_pattern_scores_no_longer_adds_reusable_by_phrase_length() -> None:
    scores, reasons = _extract_pattern_type_scores(
        text="We evaluate many ideas.",
        original="We evaluate many ideas.",
        phrases=["pretty hard evals"],
    )
    assert "reusable_high_frequency_chunk" not in scores
    assert "pattern:reusable_chunk" not in reasons


def test_model_reusable_label_requires_strong_transfer_evidence() -> None:
    assert (
        _model_label_supported(
            label="reusable_high_frequency_chunk",
            pattern_scores={},
            text="We evaluate many ideas.",
            phrases=["pretty hard evals"],
        )
        is False
    )
    assert (
        _model_label_supported(
            label="reusable_high_frequency_chunk",
            pattern_scores={},
            text="It turns out this works in practice.",
            phrases=["turns out this works"],
        )
        is True
    )


def test_score_candidate_debug_types_do_not_mark_reusable_from_length_only() -> None:
    _, _, _, _, debug_types = score_candidate(
        {
            "text": "We {{c1::<b>pretty hard evals</b>}}(hint) this quarter.",
            "original": "We pretty hard evals this quarter.",
            "target_phrases": ["pretty hard evals"],
            "phrase_types": [],
        },
        difficulty="advanced",
        material_profile="transcript_dialogue",
        learning_mode="lingua_expression",
    )
    assert "reusable_high_frequency_chunk" not in debug_types["programmatic_phrase_types"]


def test_score_candidate_keeps_model_structural_label_as_final_signal() -> None:
    _score, reasons, phrase_types, _transfer, _debug = score_candidate(
        {
            "text": "But there is {{c1::<b>another explanation</b>}}(hint).",
            "original": "But there is another explanation.",
            "target_phrases": ["another explanation"],
            "phrase_types": ["discourse_organizer"],
        },
        difficulty="advanced",
        material_profile="transcript_dialogue",
        learning_mode="lingua_expression",
    )
    assert phrase_types == ["discourse_organizer"]
    assert "model_structural_label_dropped:discourse_organizer" not in reasons


def test_score_candidate_does_not_add_programmatic_label_when_model_label_is_empty() -> None:
    _score, _reasons, phrase_types, _transfer, _debug = score_candidate(
        {
            "text": "On the one hand this helps, on the other hand it hurts.",
            "original": "On the one hand this helps, on the other hand it hurts.",
            "target_phrases": ["on the one hand"],
            "phrase_types": [],
        },
        difficulty="advanced",
        material_profile="transcript_dialogue",
        learning_mode="lingua_expression",
    )
    assert phrase_types == []
