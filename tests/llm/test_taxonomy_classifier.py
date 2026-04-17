from __future__ import annotations

import pytest

from clawlearn.errors import ClawLearnError
from clawlearn.llm.taxonomy_classifier import (
    classify_lingua_prerank_phrases_batch,
    classify_lingua_prerank_taxonomy_batch,
    classify_phrase_annotations_batch,
    classify_phrase_types_batch,
)


class _FakeClient:
    def __init__(self, response: str) -> None:
        self._response = response

    def chat(self, messages, *, temperature=None, max_retries=None):  # noqa: ANN001
        return self._response


def test_classify_phrase_types_batch_normalizes_labels() -> None:
    client = _FakeClient(
        '[{"phrase_types":["discourse_marker","invalid"]}, {"phrase_types":"stance,contrast"}]'
    )
    items = [
        {
            "text_cloze": "A {{c1::<b>in other words</b>}}(hint) B",
            "text_original": "A in other words B",
            "target_phrases": ["in other words"],
            "difficulty": "intermediate",
            "learning_mode": "lingua_expression",
        },
        {
            "text_cloze": "I {{c1::<b>think</b>}}(hint) this works",
            "text_original": "I think this works",
            "target_phrases": ["think"],
            "difficulty": "intermediate",
            "learning_mode": "lingua_expression",
        },
    ]
    labels = classify_phrase_types_batch(client=client, items=items)
    assert labels == [["discourse_organizer"], ["stance_positioning", "concession_contrast"]]


def test_classify_phrase_types_batch_length_mismatch_raises() -> None:
    client = _FakeClient('[{"phrase_types":["strong_collocation"]}]')
    items = [
        {"text_cloze": "x", "text_original": "x", "target_phrases": ["x"], "difficulty": "intermediate", "learning_mode": "lingua_expression"},
        {"text_cloze": "y", "text_original": "y", "target_phrases": ["y"], "difficulty": "intermediate", "learning_mode": "lingua_expression"},
    ]
    with pytest.raises(ClawLearnError) as exc_info:
        classify_phrase_types_batch(client=client, items=items)
    assert exc_info.value.error_code == "LLM_RESPONSE_SHAPE_INVALID"


def test_classify_phrase_types_batch_length_mismatch_allow_partial() -> None:
    client = _FakeClient('[{"phrase_types":["strong_collocation"]}]')
    items = [
        {"text_cloze": "x", "text_original": "x", "target_phrases": ["x"], "difficulty": "intermediate", "learning_mode": "lingua_expression"},
        {"text_cloze": "y", "text_original": "y", "target_phrases": ["y"], "difficulty": "intermediate", "learning_mode": "lingua_expression"},
    ]
    labels = classify_phrase_types_batch(client=client, items=items, allow_partial=True)
    assert labels == [["strong_collocation"]]


def test_classify_phrase_types_batch_invalid_json_raises() -> None:
    client = _FakeClient("not-json")
    items = [
        {"text_cloze": "x", "text_original": "x", "target_phrases": ["x"], "difficulty": "intermediate", "learning_mode": "lingua_expression"},
    ]
    with pytest.raises(ClawLearnError) as exc_info:
        classify_phrase_types_batch(client=client, items=items)
    assert exc_info.value.error_code == "LLM_RESPONSE_PARSE_FAILED"


def test_classify_phrase_annotations_batch_normalizes_label_and_reason() -> None:
    client = _FakeClient(
        '[{"phrase_types":["invalid","strong_collocation"],"reason":"Very reusable in daily speaking."}]'
    )
    items = [
        {
            "text_cloze": "We {{c1::<b>made progress</b>}}(hint) today",
            "text_original": "We made progress today",
            "target_phrases": ["made progress"],
            "difficulty": "intermediate",
            "learning_mode": "lingua_expression",
        }
    ]
    annotations = classify_phrase_annotations_batch(client=client, items=items)
    assert annotations == [{"phrase_types": ["strong_collocation"], "reason": "Very reusable in daily speaking."}]


def test_classify_phrase_annotations_batch_invalid_label_degrades_reason() -> None:
    client = _FakeClient('[{"phrase_types":["not_in_taxonomy"],"reason":"should drop"}]')
    items = [
        {
            "text_cloze": "A {{c1::<b>thing</b>}}(hint)",
            "text_original": "A thing",
            "target_phrases": ["thing"],
            "difficulty": "intermediate",
            "learning_mode": "lingua_expression",
        }
    ]
    annotations = classify_phrase_annotations_batch(client=client, items=items)
    assert annotations == [{"phrase_types": [], "reason": ""}]


def test_classify_phrase_annotations_batch_length_mismatch_allow_partial() -> None:
    client = _FakeClient('[{"phrase_types":["strong_collocation"],"reason":"ok"}]')
    items = [
        {
            "text_cloze": "A {{c1::<b>thing</b>}}(hint)",
            "text_original": "A thing",
            "target_phrases": ["thing"],
            "difficulty": "intermediate",
            "learning_mode": "lingua_expression",
        },
        {
            "text_cloze": "B {{c1::<b>item</b>}}(hint)",
            "text_original": "B item",
            "target_phrases": ["item"],
            "difficulty": "intermediate",
            "learning_mode": "lingua_expression",
        },
    ]
    annotations = classify_phrase_annotations_batch(
        client=client,
        items=items,
        allow_partial=True,
    )
    assert annotations == [{"phrase_types": ["strong_collocation"], "reason": "ok"}]


def test_classify_lingua_prerank_taxonomy_batch_normalizes_id_and_label() -> None:
    client = _FakeClient(
        '[{"id":"item_1","phrase_types":["discourse_marker"]},{"id":"item_2","phrase_types":["invalid"]}]'
    )
    items = [
        {
            "id": "item_1",
            "text_cloze": "A {{c1::<b>in other words</b>}}(hint)",
            "text_original": "A in other words",
            "target_phrases": ["in other words"],
            "difficulty": "intermediate",
            "learning_mode": "lingua_expression",
        },
        {
            "id": "item_2",
            "text_cloze": "B {{c1::<b>thing</b>}}(hint)",
            "text_original": "B thing",
            "target_phrases": ["thing"],
            "difficulty": "intermediate",
            "learning_mode": "lingua_expression",
        },
    ]
    annotations = classify_lingua_prerank_taxonomy_batch(
        client=client,
        items=items,
        source_lang="en",
    )
    assert annotations == [
        {"id": "item_1", "phrase_types": ["discourse_organizer"]},
        {"id": "item_2", "phrase_types": []},
    ]


def test_classify_lingua_prerank_taxonomy_batch_length_mismatch_allow_partial() -> None:
    client = _FakeClient('[{"id":"item_1","phrase_types":["strong_collocation"]}]')
    items = [
        {
            "id": "item_1",
            "text_cloze": "A {{c1::<b>x</b>}}(hint)",
            "text_original": "A x",
            "target_phrases": ["x"],
            "difficulty": "intermediate",
            "learning_mode": "lingua_expression",
        },
        {
            "id": "item_2",
            "text_cloze": "B {{c1::<b>y</b>}}(hint)",
            "text_original": "B y",
            "target_phrases": ["y"],
            "difficulty": "intermediate",
            "learning_mode": "lingua_expression",
        },
    ]
    annotations = classify_lingua_prerank_taxonomy_batch(
        client=client,
        items=items,
        allow_partial=True,
    )
    assert annotations == [{"id": "item_1", "phrase_types": ["strong_collocation"]}]


def test_classify_lingua_prerank_phrases_batch_normalizes_none_and_keep() -> None:
    client = _FakeClient(
        '[{"id":"p1","label":"discourse_marker","keep":true,"confidence":0.93},'
        '{"id":"p2","label":"none","keep":false,"confidence":0.4}]'
    )
    items = [
        {
            "id": "p1",
            "phrase_text": "in other words",
            "text_original": "A in other words B",
            "text_cloze": "A {{c1::<b>in other words</b>}}(hint) B",
            "local_context": "A in other words B",
            "difficulty": "advanced",
            "learning_mode": "lingua_expression",
        },
        {
            "id": "p2",
            "phrase_text": "look at the",
            "text_original": "look at the chart",
            "text_cloze": "{{c1::<b>look at the</b>}}(hint) chart",
            "local_context": "look at the chart",
            "difficulty": "advanced",
            "learning_mode": "lingua_expression",
        },
    ]
    out = classify_lingua_prerank_phrases_batch(client=client, items=items, source_lang="en")
    assert out == [
        {"id": "p1", "label": "discourse_organizer", "keep": True, "confidence": 0.93},
        {"id": "p2", "label": "none", "keep": False, "confidence": 0.4},
    ]


def test_classify_lingua_prerank_phrases_batch_length_mismatch_allow_partial() -> None:
    client = _FakeClient('[{"id":"p1","label":"strong_collocation","keep":true,"confidence":1}]')
    items = [
        {
            "id": "p1",
            "phrase_text": "make sense of",
            "text_original": "A make sense of B",
            "text_cloze": "A {{c1::<b>make sense of</b>}}(hint) B",
            "local_context": "A make sense of B",
            "difficulty": "advanced",
            "learning_mode": "lingua_expression",
        },
        {
            "id": "p2",
            "phrase_text": "have this new",
            "text_original": "you have this new bug",
            "text_cloze": "you {{c1::<b>have this new</b>}}(hint) bug",
            "local_context": "you have this new bug",
            "difficulty": "advanced",
            "learning_mode": "lingua_expression",
        },
    ]
    out = classify_lingua_prerank_phrases_batch(
        client=client,
        items=items,
        allow_partial=True,
        source_lang="en",
    )
    assert out == [{"id": "p1", "label": "strong_collocation", "keep": True, "confidence": 1.0}]
