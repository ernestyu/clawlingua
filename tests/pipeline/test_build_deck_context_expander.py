from __future__ import annotations

from clawlearn.pipeline.build_lingua_deck import (
    _apply_transcript_context_fallback,
    _expand_candidate_context_in_chunk,
)


def test_expand_candidate_context_prefers_next_sentence() -> None:
    text, original, expanded_with = _expand_candidate_context_in_chunk(
        text="I {{c1::<b>have two possible explanations</b>}}(hint).",
        original="I have two possible explanations.",
        chunk_text="I have two possible explanations. The first one is simple.",
        min_sentences=2,
    )
    assert expanded_with == "next_sentence"
    assert original == "I have two possible explanations. The first one is simple."
    assert text.endswith("The first one is simple.")


def test_expand_candidate_context_uses_previous_when_last_sentence() -> None:
    text, original, expanded_with = _expand_candidate_context_in_chunk(
        text="This {{c1::<b>works</b>}}(hint).",
        original="This works.",
        chunk_text="We tested it yesterday. This works.",
        min_sentences=2,
    )
    assert expanded_with == "previous_sentence"
    assert original == "We tested it yesterday. This works."
    assert text.startswith("We tested it yesterday.")


def test_expand_candidate_context_skips_extreme_next_sentence() -> None:
    very_long = "a" * 361 + "."
    text, original, expanded_with = _expand_candidate_context_in_chunk(
        text="I {{c1::<b>agree</b>}}(hint).",
        original="I agree.",
        chunk_text=f"I agree. {very_long}",
        min_sentences=2,
    )
    assert expanded_with == ""
    assert original == "I agree."
    assert text == "I {{c1::<b>agree</b>}}(hint)."


def test_apply_transcript_context_fallback_is_scope_gated() -> None:
    items = [
        {
            "text": "I {{c1::<b>agree</b>}}(hint).",
            "original": "I agree.",
            "chunk_text": "I agree. This is clearer now.",
        }
    ]
    count = _apply_transcript_context_fallback(
        items=items,
        material_profile="prose_article",
        learning_mode="lingua_expression",
        min_sentences=2,
    )
    assert count == 0
    assert items[0]["original"] == "I agree."


def test_apply_transcript_context_fallback_is_mode_gated() -> None:
    items = [
        {
            "text": "I {{c1::<b>agree</b>}}(hint).",
            "original": "I agree.",
            "chunk_text": "I agree. This is clearer now.",
        }
    ]
    count = _apply_transcript_context_fallback(
        items=items,
        material_profile="transcript_dialogue",
        learning_mode="lingua_reading",
        min_sentences=2,
    )
    assert count == 0
    assert items[0]["original"] == "I agree."


def test_apply_transcript_context_fallback_marks_diagnostics() -> None:
    items = [
        {
            "text": "I {{c1::<b>agree</b>}}(hint).",
            "original": "I agree.",
            "chunk_text": "I agree. This is clearer now.",
        }
    ]
    count = _apply_transcript_context_fallback(
        items=items,
        material_profile="transcript_dialogue",
        learning_mode="lingua_expression",
        min_sentences=2,
    )
    assert count == 1
    assert items[0]["context_expanded"] is True
    assert items[0]["context_expanded_with"] == "next_sentence"
    assert items[0]["original"] == "I agree. This is clearer now."


def test_expand_candidate_context_matches_fragment_suffix_in_chunk() -> None:
    text, original, expanded_with = _expand_candidate_context_in_chunk(
        text="and {{c1::<b>brings back the first bug</b>}}(hint), and you can alternate between those.",
        original="and brings back the first bug, and you can alternate between those.",
        chunk_text=(
            "Then it brings back the first bug, and you can alternate between those. "
            "How is that possible?"
        ),
        min_sentences=2,
    )
    assert expanded_with == "next_sentence"
    assert original.endswith("How is that possible?")
    assert text.endswith("How is that possible?")
