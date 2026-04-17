from __future__ import annotations

from clawlearn.pipeline.build_lingua_deck import (
    _is_structurally_empty_raw_candidate,
    _should_fail_for_structurally_empty_raw_candidates,
)


def test_is_structurally_empty_raw_candidate_detects_empty_payload() -> None:
    item = {"text": "", "original": "", "target_phrases": []}
    assert _is_structurally_empty_raw_candidate(item) is True


def test_is_structurally_empty_raw_candidate_accepts_non_empty_payload() -> None:
    item = {
        "text": "A {{c1::<b>phrase</b>}}(hint).",
        "original": "A phrase.",
        "target_phrases": ["phrase"],
    }
    assert _is_structurally_empty_raw_candidate(item) is False


def test_should_fail_for_structurally_empty_raw_candidates_when_all_empty() -> None:
    raw = [{"text": "", "original": "", "target_phrases": []} for _ in range(3)]
    should_fail, empty_count, ratio = _should_fail_for_structurally_empty_raw_candidates(raw)
    assert should_fail is True
    assert empty_count == 3
    assert ratio == 1.0


def test_should_fail_for_structurally_empty_raw_candidates_when_mostly_valid() -> None:
    raw = [{"text": "", "original": "", "target_phrases": []}]
    raw.extend(
        {
            "text": "A {{c1::<b>phrase</b>}}(hint).",
            "original": "A phrase.",
            "target_phrases": ["phrase"],
        }
        for _ in range(9)
    )
    should_fail, empty_count, ratio = _should_fail_for_structurally_empty_raw_candidates(raw)
    assert should_fail is False
    assert empty_count == 1
    assert ratio == 0.1


def test_should_fail_for_structurally_empty_raw_candidates_when_large_and_nearly_all_empty() -> None:
    raw = [{"text": "", "original": "", "target_phrases": []} for _ in range(19)]
    raw.append(
        {
            "text": "A {{c1::<b>phrase</b>}}(hint).",
            "original": "A phrase.",
            "target_phrases": ["phrase"],
        }
    )
    should_fail, empty_count, ratio = _should_fail_for_structurally_empty_raw_candidates(raw)
    assert should_fail is True
    assert empty_count == 19
    assert ratio == 0.95
