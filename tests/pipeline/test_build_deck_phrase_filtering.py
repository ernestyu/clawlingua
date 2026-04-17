from __future__ import annotations

import threading

from clawlearn.pipeline.build_lingua_deck import (
    _build_pipeline_metrics,
    _merge_secondary_extraction_candidates,
    _materialize_cloze_candidates_from_phrase_items,
    _run_extraction_passes,
)


def test_materialize_phrase_candidates_filters_english_noise() -> None:
    raw, stats = _materialize_cloze_candidates_from_phrase_items(
        [
            {
                "chunk_id": "chunk_0001",
                "sentence_text": "This is one of the very confusing things right now.",
                "phrase_text": "one of the",
            },
            {
                "chunk_id": "chunk_0001",
                "sentence_text": "This is one of the very confusing things right now.",
                "phrase_text": "very confusing things",
            },
        ],
        source_lang="en",
        learning_mode="lingua_expression",
        difficulty="advanced",
    )

    assert len(raw) == 1
    assert raw[0]["target_phrases"] == ["very confusing things"]
    assert stats["enabled"] is True
    assert stats["dropped_count"] == 1
    assert stats["dropped_by_rule"]["drop:blacklist_ngram"] == 1


def test_materialize_phrase_candidates_routes_non_english_to_generic() -> None:
    raw, stats = _materialize_cloze_candidates_from_phrase_items(
        [
            {
                "chunk_id": "chunk_0002",
                "sentence_text": "Nous gardons one of the pour un test minimal.",
                "phrase_text": "one of the",
            },
            {
                "chunk_id": "chunk_0002",
                "sentence_text": "Nous gardons one of the pour un test minimal.",
                "phrase_text": "test minimal",
            },
        ],
        source_lang="fr",
        learning_mode="lingua_expression",
        difficulty="advanced",
    )
    assert len(raw) == 1
    assert raw[0]["target_phrases"] == ["one of the", "test minimal"]
    assert stats["language"] == "fr"
    assert stats["dropped_count"] == 0


def test_materialize_phrase_candidates_drops_empty_candidates_after_filtering() -> None:
    raw, stats = _materialize_cloze_candidates_from_phrase_items(
        [
            {
                "chunk_id": "chunk_0003",
                "sentence_text": "You look at the numbers and move on.",
                "phrase_text": "look at the",
            }
        ],
        source_lang="en",
        learning_mode="lingua_expression",
        difficulty="advanced",
    )
    assert raw == []
    assert stats["dropped_count"] == 1


def test_pipeline_metrics_exposes_phrase_filter_summary() -> None:
    metrics = _build_pipeline_metrics(
        chunks=[],
        raw_candidates=[],
        valid_candidates=[],
        ranked_candidates=[],
        deduped_candidates=[],
        errors=[],
        phrase_filter_stats={
            "enabled": True,
            "source_lang": "en",
            "language": "en",
            "dropped_count": 2,
            "dropped_by_rule": {"drop:blacklist_ngram": 2},
            "examples_by_rule": {"drop:blacklist_ngram": ["one of the"]},
        },
    )
    assert metrics["phrase_filter"]["enabled"] is True
    assert metrics["phrase_filter"]["source_lang"] == "en"
    assert metrics["phrase_filter"]["dropped_count"] == 2
    assert metrics["secondary_extraction"]["enabled"] is False
    assert metrics["secondary_extraction"]["parallel"] is False
    assert metrics["secondary_extraction"]["execution_mode"] == "disabled"


def test_merge_secondary_extraction_candidates_unions_overlapping_phrase_context() -> None:
    merged, unique_gain = _merge_secondary_extraction_candidates(
        use_phrase_pipeline=True,
        primary_items=[
            {
                "chunk_id": "chunk_0001",
                "sentence_text": "Today we made progress and kept going despite delays.",
                "phrase_text": "made progress",
            }
        ],
        secondary_items=[
            {
                "chunk_id": "chunk_0002",
                "sentence_text": "We made progress and kept going despite long delays today.",
                "phrase_text": "kept going",
            }
        ],
    )
    merged_phrases = {item["phrase_text"] for item in merged}
    assert merged_phrases == {"made progress", "kept going"}
    assert any(item["chunk_id"] == "chunk_0001" for item in merged)
    assert unique_gain == 1


def test_pipeline_metrics_exposes_secondary_extract_summary() -> None:
    metrics = _build_pipeline_metrics(
        chunks=[],
        raw_candidates=[],
        valid_candidates=[],
        ranked_candidates=[],
        deduped_candidates=[],
        errors=[],
        secondary_extract_stats={
            "requested": True,
            "enabled": True,
            "configured": True,
            "parallel": True,
            "execution_mode": "parallel",
            "secondary_model": "mistral-small3.2:latest",
            "candidates_primary_count": 10,
            "candidates_secondary_count": 6,
            "candidates_merged_count": 14,
            "dedup_removed_count": 2,
            "unique_phrase_gain_from_secondary": 4,
            "secondary_error_type": "",
            "secondary_error_message": "",
            "fallback_to_primary": False,
        },
    )
    sec = metrics["secondary_extraction"]
    assert sec["requested"] is True
    assert sec["enabled"] is True
    assert sec["parallel"] is True
    assert sec["execution_mode"] == "parallel"
    assert sec["secondary_model"] == "mistral-small3.2:latest"
    assert sec["candidates_merged_count"] == 14
    assert sec["unique_phrase_gain_from_secondary"] == 4


def test_merge_secondary_extraction_candidates_for_cloze_union_sources() -> None:
    merged, unique_gain = _merge_secondary_extraction_candidates(
        use_phrase_pipeline=False,
        primary_items=[
            {
                "chunk_id": "chunk_0001",
                "text": "We {{c1::<b>made progress</b>}}(hint).",
                "original": "We made progress.",
                "target_phrases": ["made progress"],
            }
        ],
        secondary_items=[
            {
                "chunk_id": "chunk_0001",
                "text": "We {{c1::<b>made progress</b>}}(hint).",
                "original": "We made progress.",
                "target_phrases": ["made progress", "made"],
            }
        ],
    )
    assert len(merged) == 1
    assert merged[0]["extract_sources"] == ["primary", "secondary"]
    assert merged[0]["target_phrases"] == ["made progress", "made"]
    assert unique_gain == 1


def test_run_extraction_passes_serial_runs_primary_then_secondary() -> None:
    calls: list[str] = []

    def _primary() -> None:
        calls.append("primary")
        return None

    def _secondary() -> None:
        calls.append("secondary")
        return None

    primary_error, secondary_error = _run_extraction_passes(
        run_primary=_primary,
        run_secondary=_secondary,
        secondary_parallel=False,
    )
    assert primary_error is None
    assert secondary_error is None
    assert calls == ["primary", "secondary"]


def test_run_extraction_passes_parallel_runs_both_workers() -> None:
    barrier = threading.Barrier(2)
    calls: list[str] = []

    def _primary() -> None:
        calls.append("primary:start")
        barrier.wait(timeout=1.0)
        calls.append("primary:done")
        return None

    def _secondary() -> None:
        calls.append("secondary:start")
        barrier.wait(timeout=1.0)
        calls.append("secondary:done")
        return None

    primary_error, secondary_error = _run_extraction_passes(
        run_primary=_primary,
        run_secondary=_secondary,
        secondary_parallel=True,
    )
    assert primary_error is None
    assert secondary_error is None
    assert "primary:done" in calls
    assert "secondary:done" in calls
