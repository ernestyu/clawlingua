from __future__ import annotations

from clawlearn.pipeline.taxonomy import (
    PHRASE_TAXONOMY,
    PRERANK_HIGH_VALUE_TAXONOMY,
    get_allowed_taxonomy,
    get_prerank_taxonomy,
    normalize_prerank_phrase_label,
)


def test_get_allowed_taxonomy_known_language() -> None:
    labels, aliases = get_allowed_taxonomy("en")
    assert labels == PHRASE_TAXONOMY
    assert "discourse_marker" in aliases


def test_get_allowed_taxonomy_unknown_language_uses_fallback() -> None:
    labels, aliases = get_allowed_taxonomy("it")
    assert labels == PHRASE_TAXONOMY
    assert "high_frequency_chunk" in aliases


def test_get_prerank_taxonomy_returns_v2_labels() -> None:
    labels, aliases = get_prerank_taxonomy("en")
    assert labels == PRERANK_HIGH_VALUE_TAXONOMY
    assert "discourse_marker" in aliases


def test_normalize_prerank_phrase_label() -> None:
    assert normalize_prerank_phrase_label("discourse_marker") == "discourse_organizer"
    assert normalize_prerank_phrase_label("reusable_high_frequency_chunk") == ""
    assert normalize_prerank_phrase_label("none") == ""
