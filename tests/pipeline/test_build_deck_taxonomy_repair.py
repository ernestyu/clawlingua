from __future__ import annotations

from clawlearn.config import AppConfig
from clawlearn.errors import build_error
from clawlearn.exit_codes import ExitCode
from clawlearn.pipeline.build_lingua_deck import _collect_valid_candidates, _repair_taxonomy_candidates


def _candidate_with_too_many_labels() -> dict:
    return {
        "chunk_id": "chunk_0001",
        "chunk_text": "During the weekly review, we made progress across three teams today.",
        "text": "During the weekly review, we {{c1::<b>made progress</b>}}(hint) across three teams today.",
        "original": "During the weekly review, we made progress across three teams today.",
        "target_phrases": ["made progress"],
        "phrase_types": ["strong_collocation", "phrasal_verb", "stance_positioning"],
        "note_hint": "",
    }


def _candidate_for_invalid_combo() -> dict:
    return {
        "chunk_id": "chunk_0001",
        "chunk_text": "Now now.",
        "text": "Now {{c1::<b>now</b>}}(hint).",
        "original": "Now now.",
        "target_phrases": ["now"],
        "phrase_types": ["stance_positioning", "concession_contrast"],
        "note_hint": "",
    }


def test_collect_valid_candidates_without_taxonomy_repair_rejects() -> None:
    cfg = AppConfig(validate_format_retry_enable=False, taxonomy_repair_enable=False)
    errors: list[dict] = []
    valid, rejected, _first, final_reason, _retry_stats, queue, taxonomy_count = _collect_valid_candidates(
        raw_candidates=[_candidate_with_too_many_labels()],
        cfg=cfg,
        material_profile="prose_article",
        learning_mode="lingua_expression",
        errors=errors,
        taxonomy_repair_enable=cfg.taxonomy_repair_enable,
    )
    assert valid == []
    assert rejected == 1
    assert queue == []
    assert taxonomy_count == 1
    assert str(final_reason).startswith("taxonomy:too_many_labels:")
    assert any(err.get("stage") == "validate_text" for err in errors)


def test_collect_valid_candidates_with_taxonomy_repair_queues() -> None:
    cfg = AppConfig(validate_format_retry_enable=False, taxonomy_repair_enable=True)
    errors: list[dict] = []
    valid, rejected, _first, final_reason, _retry_stats, queue, taxonomy_count = _collect_valid_candidates(
        raw_candidates=[_candidate_with_too_many_labels()],
        cfg=cfg,
        material_profile="prose_article",
        learning_mode="lingua_expression",
        errors=errors,
        taxonomy_repair_enable=cfg.taxonomy_repair_enable,
    )
    assert valid == []
    assert rejected == 0
    assert final_reason is None
    assert len(queue) == 1
    assert taxonomy_count == 1
    assert any(err.get("stage") == "taxonomy_repair_queued" for err in errors)


def test_repair_taxonomy_candidates_successfully_recovers_candidate() -> None:
    cfg = AppConfig(
        validate_format_retry_enable=False,
        taxonomy_repair_enable=True,
        llm_retry_backoff_seconds=0.0,
    )
    queued = [
        {
            "item": _candidate_with_too_many_labels(),
            "reason": "taxonomy:too_many_labels:phrase_types has too many labels",
            "original_reason": "taxonomy:too_many_labels:phrase_types has too many labels",
        }
    ]
    errors: list[dict] = []
    repaired, stats = _repair_taxonomy_candidates(
        queued_candidates=queued,
        cfg=cfg,
        material_profile="prose_article",
        learning_mode="lingua_expression",
        client=object(),  # classify_fn ignores client in this unit test
        errors=errors,
        temperature=0.0,
        classify_fn=lambda **kwargs: [["strong_collocation"]],
    )
    assert len(repaired) == 1
    assert repaired[0]["phrase_types"] == ["strong_collocation"]
    assert stats.taxonomy_reject_count == 1
    assert stats.taxonomy_repair_attempted == 1
    assert stats.taxonomy_repair_success == 1
    assert stats.taxonomy_repair_failed == 0
    assert any(err.get("stage") == "taxonomy_repair_recovered" for err in errors)


def test_repair_taxonomy_candidates_failed_revalidation_is_counted() -> None:
    cfg = AppConfig(
        validate_format_retry_enable=False,
        taxonomy_repair_enable=True,
        llm_retry_backoff_seconds=0.0,
    )
    queued = [
        {
            "item": _candidate_for_invalid_combo(),
            "reason": "taxonomy:invalid_combo:phrase_types combination is inconsistent with candidate context",
            "original_reason": "taxonomy:invalid_combo:phrase_types combination is inconsistent with candidate context",
        }
    ]
    errors: list[dict] = []
    repaired, stats = _repair_taxonomy_candidates(
        queued_candidates=queued,
        cfg=cfg,
        material_profile="prose_article",
        learning_mode="lingua_expression",
        client=object(),
        errors=errors,
        temperature=0.0,
        classify_fn=lambda **kwargs: [["stance_positioning", "concession_contrast"]],
    )
    assert repaired == []
    assert stats.taxonomy_repair_success == 0
    assert stats.taxonomy_repair_failed == 1
    assert any(err.get("stage") == "taxonomy_repair_failed" for err in errors)


def test_repair_taxonomy_candidates_batch_error_marks_failed() -> None:
    cfg = AppConfig(
        validate_format_retry_enable=False,
        taxonomy_repair_enable=True,
        llm_retry_backoff_seconds=0.0,
    )
    queued = [
        {
            "item": _candidate_with_too_many_labels(),
            "reason": "taxonomy:too_many_labels:phrase_types has too many labels",
            "original_reason": "taxonomy:too_many_labels:phrase_types has too many labels",
        }
    ]
    errors: list[dict] = []
    calls = {"n": 0}

    def _raise_retryable(**kwargs):  # noqa: ANN003, ANN202
        calls["n"] += 1
        raise build_error(
            error_code="LLM_RESPONSE_PARSE_FAILED",
            cause="bad json",
            detail="bad json",
            next_steps=[],
            exit_code=ExitCode.LLM_PARSE_ERROR,
        )

    repaired, stats = _repair_taxonomy_candidates(
        queued_candidates=queued,
        cfg=cfg,
        material_profile="prose_article",
        learning_mode="lingua_expression",
        client=object(),
        errors=errors,
        temperature=0.0,
        classify_fn=_raise_retryable,
    )
    assert repaired == []
    assert calls["n"] == 3
    assert stats.taxonomy_repair_success == 0
    assert stats.taxonomy_repair_failed == 1
    assert any(err.get("stage") == "taxonomy_repair_batch_error" for err in errors)


def test_repair_taxonomy_candidates_partial_response_retries_remaining() -> None:
    cfg = AppConfig(
        validate_format_retry_enable=False,
        taxonomy_repair_enable=True,
        llm_retry_backoff_seconds=0.0,
    )
    queued = [
        {
            "item": _candidate_with_too_many_labels(),
            "reason": "taxonomy:too_many_labels:phrase_types has too many labels",
            "original_reason": "taxonomy:too_many_labels:phrase_types has too many labels",
        },
        {
            "item": _candidate_with_too_many_labels(),
            "reason": "taxonomy:too_many_labels:phrase_types has too many labels",
            "original_reason": "taxonomy:too_many_labels:phrase_types has too many labels",
        },
    ]
    errors: list[dict] = []
    calls = {"n": 0}

    def _classify_partial(**kwargs):  # noqa: ANN003, ANN202
        calls["n"] += 1
        return [["strong_collocation"]]

    repaired, stats = _repair_taxonomy_candidates(
        queued_candidates=queued,
        cfg=cfg,
        material_profile="prose_article",
        learning_mode="lingua_expression",
        client=object(),
        errors=errors,
        temperature=0.0,
        classify_fn=_classify_partial,
    )
    assert calls["n"] == 2
    assert len(repaired) == 2
    assert all(item["phrase_types"] == ["strong_collocation"] for item in repaired)
    assert stats.taxonomy_repair_success == 2
    assert stats.taxonomy_repair_failed == 0
    assert any(err.get("stage") == "taxonomy_repair_batch_partial_retry" for err in errors)
    assert not any(err.get("stage") == "taxonomy_repair_batch_error" for err in errors)


def test_repair_taxonomy_candidates_partial_response_exhausted_marks_failed() -> None:
    cfg = AppConfig(
        validate_format_retry_enable=False,
        taxonomy_repair_enable=True,
        llm_retry_backoff_seconds=0.0,
    )
    queued = [
        {
            "item": _candidate_with_too_many_labels(),
            "reason": "taxonomy:too_many_labels:phrase_types has too many labels",
            "original_reason": "taxonomy:too_many_labels:phrase_types has too many labels",
        },
        {
            "item": _candidate_with_too_many_labels(),
            "reason": "taxonomy:too_many_labels:phrase_types has too many labels",
            "original_reason": "taxonomy:too_many_labels:phrase_types has too many labels",
        },
    ]
    errors: list[dict] = []
    calls = {"n": 0}

    def _classify_empty(**kwargs):  # noqa: ANN003, ANN202
        calls["n"] += 1
        return []

    repaired, stats = _repair_taxonomy_candidates(
        queued_candidates=queued,
        cfg=cfg,
        material_profile="prose_article",
        learning_mode="lingua_expression",
        client=object(),
        errors=errors,
        temperature=0.0,
        classify_fn=_classify_empty,
    )
    assert repaired == []
    assert calls["n"] == 3
    assert stats.taxonomy_repair_success == 0
    assert stats.taxonomy_repair_failed == 2
    assert any(err.get("stage") == "taxonomy_repair_batch_error" for err in errors)
