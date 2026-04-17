from __future__ import annotations

from clawlearn.config import AppConfig
from clawlearn.errors import build_error
from clawlearn.exit_codes import ExitCode
from clawlearn.pipeline.build_lingua_deck import (
    _annotate_lingua_candidates_pre_rank,
    _drop_candidates_without_taxonomy_labels,
)


def _base_items() -> list[dict]:
    return [
        {
            "chunk_id": "chunk_0001",
            "text": "We {{c1::<b>made progress</b>}}(hint) today. It {{c2::<b>looks at the</b>}}(hint) chart.",
            "original": "We made progress today. It looks at the chart.",
            "target_phrases": ["made progress", "looks at the"],
            "phrase_types": ["strong_collocation", "phrasal_verb"],
        },
        {
            "chunk_id": "chunk_0002",
            "text": "It {{c1::<b>turned out</b>}}(hint) well.",
            "original": "It turned out well.",
            "target_phrases": ["turned out"],
            "phrase_types": ["phrasal_verb"],
        },
    ]


def test_annotate_lingua_candidates_pre_rank_is_gated_by_enable_and_mode() -> None:
    items = _base_items()
    errors: list[dict] = []
    cfg_disabled = AppConfig(lingua_annotate_enable=False)
    updated, stats = _annotate_lingua_candidates_pre_rank(
        items=items,
        cfg=cfg_disabled,
        learning_mode="lingua_expression",
        source_lang="en",
        client=object(),
        errors=errors,
        classify_fn=lambda **kwargs: [],
    )
    assert len(updated) == 2
    assert stats.candidates_output == 2
    assert stats.phrases_total == 0
    assert errors == []

    cfg_enabled = AppConfig(lingua_annotate_enable=True)
    updated, _stats = _annotate_lingua_candidates_pre_rank(
        items=items,
        cfg=cfg_enabled,
        learning_mode="textbook_focus",
        source_lang="en",
        client=object(),
        errors=errors,
        classify_fn=lambda **kwargs: [],
    )
    assert len(updated) == 2


def test_annotate_lingua_candidates_pre_rank_phrase_level_none_downgrades_cloze() -> None:
    cfg = AppConfig(
        lingua_annotate_enable=True,
        lingua_annotate_batch_size=10,
    )
    items = _base_items()
    errors: list[dict] = []

    def _classify(**kwargs):  # noqa: ANN003, ANN202
        payload = kwargs["items"]
        assert len(payload) == 3
        return [
            {"id": "candidate_000000_phrase_00", "label": "strong_collocation", "keep": True, "confidence": 0.9},
            {"id": "candidate_000000_phrase_01", "label": "none", "keep": False, "confidence": 0.7},
            {"id": "candidate_000001_phrase_00", "label": "phrasal_verb", "keep": True, "confidence": 0.8},
        ]

    updated, stats = _annotate_lingua_candidates_pre_rank(
        items=items,
        cfg=cfg,
        learning_mode="lingua_expression",
        source_lang="en",
        client=object(),
        errors=errors,
        classify_fn=_classify,
    )
    assert len(updated) == 2
    assert stats.phrases_total == 3
    assert stats.phrases_kept == 2
    assert stats.phrases_none == 1
    assert "{{c2::" not in updated[0]["text"]
    assert "looks at the" in updated[0]["text"]
    assert updated[0]["target_phrases"] == ["made progress"]
    assert updated[1]["target_phrases"] == ["turned out"]


def test_annotate_lingua_candidates_pre_rank_all_none_removes_candidate() -> None:
    cfg = AppConfig(
        lingua_annotate_enable=True,
        lingua_annotate_batch_size=10,
    )
    items = [
        {
            "chunk_id": "chunk_weak",
            "text": "We {{c1::<b>one of the</b>}}(hint) and {{c2::<b>look at the</b>}}(hint) case.",
            "original": "We one of the and look at the case.",
            "target_phrases": ["one of the", "look at the"],
            "phrase_types": ["strong_collocation"],
        }
    ]
    errors: list[dict] = []

    def _classify(**kwargs):  # noqa: ANN003, ANN202
        payload = kwargs["items"]
        return [
            {"id": str(row["id"]), "label": "none", "keep": False, "confidence": 0.5}
            for row in payload
        ]

    updated, stats = _annotate_lingua_candidates_pre_rank(
        items=items,
        cfg=cfg,
        learning_mode="lingua_expression",
        source_lang="en",
        client=object(),
        errors=errors,
        classify_fn=_classify,
    )
    assert updated == []
    assert stats.candidates_context_only == 1
    assert stats.phrases_kept == 0


def test_annotate_lingua_candidates_pre_rank_context_only_drop_emits_debug_event() -> None:
    cfg = AppConfig(
        lingua_annotate_enable=True,
        lingua_annotate_batch_size=10,
    )
    items = [
        {
            "chunk_id": "chunk_low",
            "text": "We {{c1::<b>one of the</b>}}(hint) and {{c2::<b>look at the</b>}}(hint) parts.",
            "original": "We one of the and look at the parts.",
            "target_phrases": ["one of the", "look at the"],
            "phrase_types": ["strong_collocation"],
        }
    ]
    errors: list[dict] = []

    def _classify(**kwargs):  # noqa: ANN003, ANN202
        payload = kwargs["items"]
        return [
            {"id": str(row["id"]), "label": "none", "keep": False, "confidence": 0.35}
            for row in payload
        ]

    updated, stats = _annotate_lingua_candidates_pre_rank(
        items=items,
        cfg=cfg,
        learning_mode="lingua_expression",
        source_lang="en",
        client=object(),
        errors=errors,
        classify_fn=_classify,
    )
    assert updated == []
    assert stats.candidates_context_only == 1
    drop_events = [
        err for err in errors if err.get("stage") == "lingua_taxonomy_pre_rank_context_only_drop"
    ]
    assert len(drop_events) == 1
    event = drop_events[0]
    assert event["candidate_id"] == "candidate_000000"
    assert event["chunk_id"] == "chunk_low"
    assert event["original"] == "We one of the and look at the parts."
    assert event["phrases"] == [
        {
            "phrase_index": 0,
            "phrase_text": "one of the",
            "label": "none",
            "keep": False,
            "confidence": 0.35,
        },
        {
            "phrase_index": 1,
            "phrase_text": "look at the",
            "label": "none",
            "keep": False,
            "confidence": 0.35,
        },
    ]


def test_annotate_lingua_candidates_pre_rank_all_none_fallback_rescues_high_value_phrase() -> None:
    cfg = AppConfig(
        lingua_annotate_enable=True,
        lingua_annotate_batch_size=10,
    )
    items = [
        {
            "chunk_id": "chunk_rescue",
            "text": "We {{c1::<b>made progress</b>}}(hint) despite {{c2::<b>one of the</b>}}(hint) delays.",
            "original": "We made progress despite one of the delays.",
            "target_phrases": ["made progress", "one of the"],
            "phrase_types": ["strong_collocation"],
        }
    ]
    errors: list[dict] = []

    def _classify(**kwargs):  # noqa: ANN003, ANN202
        payload = kwargs["items"]
        return [
            {"id": str(row["id"]), "label": "none", "keep": False, "confidence": 0.25}
            for row in payload
        ]

    updated, stats = _annotate_lingua_candidates_pre_rank(
        items=items,
        cfg=cfg,
        learning_mode="lingua_expression",
        source_lang="en",
        client=object(),
        errors=errors,
        classify_fn=_classify,
    )
    assert len(updated) == 1
    assert updated[0]["target_phrases"] == ["made progress"]
    assert "one of the" in updated[0]["text"]
    assert "{{c1::<b>made progress</b>}}" in updated[0]["text"]
    assert updated[0]["phrase_types"] == ["strong_collocation"]
    assert stats.phrases_kept == 1
    assert stats.phrases_none == 1
    assert any(err.get("stage") == "lingua_taxonomy_pre_rank_all_none_fallback" for err in errors)


def test_annotate_lingua_candidates_pre_rank_batch_error_degrades() -> None:
    cfg = AppConfig(
        lingua_annotate_enable=True,
        lingua_annotate_batch_size=50,
    )
    items = _base_items()
    errors: list[dict] = []

    def _raise_error(**kwargs):  # noqa: ANN003, ANN202
        raise build_error(
            error_code="LLM_RESPONSE_PARSE_FAILED",
            cause="bad response",
            detail="bad response",
            next_steps=[],
            exit_code=ExitCode.LLM_PARSE_ERROR,
        )

    updated, stats = _annotate_lingua_candidates_pre_rank(
        items=items,
        cfg=cfg,
        learning_mode="lingua_expression",
        source_lang="en",
        client=object(),
        errors=errors,
        classify_fn=_raise_error,
    )
    assert len(updated) == 2
    assert updated[0]["target_phrases"] == ["made progress"]
    assert updated[1]["target_phrases"] == ["turned out"]
    assert stats.batch_error_count == 1
    assert stats.exhausted_count == 3
    assert any(err.get("stage") == "lingua_taxonomy_pre_rank_batch_error" for err in errors)


def test_annotate_lingua_candidates_pre_rank_partial_response_retries_remaining() -> None:
    cfg = AppConfig(
        lingua_annotate_enable=True,
        lingua_annotate_batch_size=10,
        llm_retry_backoff_seconds=0.0,
    )
    items = _base_items()
    errors: list[dict] = []
    calls = {"n": 0}

    def _classify_partial(**kwargs):  # noqa: ANN003, ANN202
        calls["n"] += 1
        if calls["n"] == 1:
            return [
                {"id": "candidate_000000_phrase_00", "label": "strong_collocation", "keep": True, "confidence": 0.9},
                {"id": "candidate_000000_phrase_01", "label": "none", "keep": False, "confidence": 0.7},
            ]
        return [{"id": "candidate_000001_phrase_00", "label": "phrasal_verb", "keep": True, "confidence": 0.8}]

    updated, stats = _annotate_lingua_candidates_pre_rank(
        items=items,
        cfg=cfg,
        learning_mode="lingua_expression",
        source_lang="en",
        client=object(),
        errors=errors,
        classify_fn=_classify_partial,
    )
    assert calls["n"] == 2
    assert len(updated) == 2
    assert stats.partial_retry_count >= 1
    assert any(err.get("stage") == "lingua_taxonomy_pre_rank_batch_partial_retry" for err in errors)


def test_drop_candidates_without_taxonomy_labels_hard_gate() -> None:
    errors: list[dict] = []
    items = [
        {
            "chunk_id": "chunk_0001",
            "text": "A {{c1::<b>x</b>}}(hint)",
            "original": "A x",
            "target_phrases": ["x"],
            "phrase_types": [],
        },
        {
            "chunk_id": "chunk_0002",
            "text": "B {{c1::<b>y</b>}}(hint)",
            "original": "B y",
            "target_phrases": ["y"],
            "phrase_types": ["strong_collocation"],
        },
    ]
    filtered, dropped = _drop_candidates_without_taxonomy_labels(
        items=items,
        errors=errors,
    )
    assert dropped == 1
    assert len(filtered) == 1
    assert filtered[0]["chunk_id"] == "chunk_0002"
    assert filtered[0]["phrase_types"] == ["strong_collocation"]
    assert any(err.get("stage") == "lingua_taxonomy_pre_rank_empty_label_drop" for err in errors)
