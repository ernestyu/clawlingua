"""Taxonomy helpers shared by prompt outputs, ranking, validation, and summary."""

from __future__ import annotations

import re

PHRASE_TAXONOMY: tuple[str, ...] = (
    "metaphor_imagery",
    "stance_positioning",
    "concession_contrast",
    "discourse_organizer",
    "abstraction_bridge",
    "reusable_high_frequency_chunk",
    "phrasal_verb",
    "strong_collocation",
)

_TAXONOMY_SET = set(PHRASE_TAXONOMY)
PRERANK_HIGH_VALUE_TAXONOMY: tuple[str, ...] = (
    "discourse_organizer",
    "concession_contrast",
    "stance_positioning",
    "abstraction_bridge",
    "strong_collocation",
    "phrasal_verb",
)
_PRERANK_HIGH_VALUE_SET = set(PRERANK_HIGH_VALUE_TAXONOMY)

_TAXONOMY_ALIASES: dict[str, str] = {
    # Legacy/debug labels from earlier versions.
    "discourse_collocation": "discourse_organizer",
    "multiword_expression": "reusable_high_frequency_chunk",
    "general_expression": "reusable_high_frequency_chunk",
    # Common prompt-side variants.
    "discourse_marker": "discourse_organizer",
    "discourse_markers": "discourse_organizer",
    "stance": "stance_positioning",
    "concession": "concession_contrast",
    "contrast": "concession_contrast",
    "abstraction": "abstraction_bridge",
    "imagery": "metaphor_imagery",
    "metaphor": "metaphor_imagery",
    "collocation": "strong_collocation",
    "high_frequency_chunk": "reusable_high_frequency_chunk",
}

HIGH_VALUE_ADVANCED_TYPES: set[str] = {
    "metaphor_imagery",
    "stance_positioning",
    "concession_contrast",
    "discourse_organizer",
    "abstraction_bridge",
    "reusable_high_frequency_chunk",
    "strong_collocation",
}
STRUCTURAL_DISCOURSE_TYPES: set[str] = {
    "metaphor_imagery",
    "stance_positioning",
    "concession_contrast",
    "discourse_organizer",
    "abstraction_bridge",
}

TRANSFER_MAX_CHARS = 120
TRANSFER_MIN_CHARS = 8
TRANSFER_PREFIX_RE = re.compile(
    r"^(?:means?|meaning|translation|\u7ffb\u8bd1|\u610f\u601d\u662f)[:\uff1a\s]",
    re.IGNORECASE,
)
_NORMALIZE_LABEL_RE = re.compile(r"[\s\-]+")

_TYPE_WEIGHTS: dict[str, dict[str, float]] = {
    "beginner": {
        "metaphor_imagery": 0.3,
        "stance_positioning": 0.4,
        "concession_contrast": 0.4,
        "discourse_organizer": 0.5,
        "abstraction_bridge": 0.2,
        "reusable_high_frequency_chunk": 0.9,
        "phrasal_verb": 0.9,
        "strong_collocation": 0.8,
    },
    "intermediate": {
        "metaphor_imagery": 0.8,
        "stance_positioning": 0.9,
        "concession_contrast": 1.0,
        "discourse_organizer": 1.0,
        "abstraction_bridge": 0.9,
        "reusable_high_frequency_chunk": 1.1,
        "phrasal_verb": 0.6,
        "strong_collocation": 1.0,
    },
    "advanced": {
        "metaphor_imagery": 1.6,
        "stance_positioning": 1.5,
        "concession_contrast": 1.7,
        "discourse_organizer": 1.6,
        "abstraction_bridge": 1.6,
        "reusable_high_frequency_chunk": 1.3,
        "phrasal_verb": 0.1,
        "strong_collocation": 1.1,
    },
}

_LANG_PLACEHOLDER_CODES: set[str] = {"en", "zh", "ja", "de", "fr", "es"}


def get_allowed_taxonomy(source_lang: str | None) -> tuple[tuple[str, ...], dict[str, str]]:
    """Return taxonomy labels and aliases for a source language.

    v1 keeps a shared internal taxonomy for all languages while preserving
    a language-routed entry point for future language-specific schemas.
    """

    lang = str(source_lang or "").strip().lower()
    if lang in _LANG_PLACEHOLDER_CODES:
        return PHRASE_TAXONOMY, _TAXONOMY_ALIASES
    return PHRASE_TAXONOMY, _TAXONOMY_ALIASES


def get_prerank_taxonomy(source_lang: str | None) -> tuple[tuple[str, ...], dict[str, str]]:
    """Return v2 high-value taxonomy labels for pre-rank phrase classification."""

    _labels, aliases = get_allowed_taxonomy(source_lang)
    return PRERANK_HIGH_VALUE_TAXONOMY, aliases


def normalize_phrase_type(value: str) -> str | None:
    key = _NORMALIZE_LABEL_RE.sub("_", str(value or "").strip().lower())
    if not key:
        return None
    key = _TAXONOMY_ALIASES.get(key, key)
    if key in _TAXONOMY_SET:
        return key
    return None


def normalize_prerank_phrase_label(value: object) -> str:
    """Normalize phrase label for pre-rank classifier; empty string means none."""

    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text or text in {"none", "null", "unknown", "n/a", "na"}:
        return ""
    normalized = normalize_phrase_type(text)
    if not normalized:
        return ""
    if normalized not in _PRERANK_HIGH_VALUE_SET:
        return ""
    return normalized


def normalize_phrase_types(value: object, *, max_items: int = 2) -> list[str]:
    if value is None:
        return []

    items: list[str]
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, list):
        items = [str(part).strip() for part in value]
    else:
        items = [str(value).strip()]

    output: list[str] = []
    seen: set[str] = set()
    for raw in items:
        normalized = normalize_phrase_type(raw)
        if not normalized or normalized in seen:
            continue
        output.append(normalized)
        seen.add(normalized)
        if len(output) >= max_items:
            break
    return output


def phrase_type_weight(*, label: str, difficulty: str) -> float:
    diff = (difficulty or "intermediate").strip().lower()
    if diff not in _TYPE_WEIGHTS:
        diff = "intermediate"
    return _TYPE_WEIGHTS[diff].get(label, 0.0)


def normalize_expression_transfer(value: object, *, max_chars: int = TRANSFER_MAX_CHARS) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" ,;:.") + "..."


def looks_like_translation_style_transfer(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    if TRANSFER_PREFIX_RE.search(cleaned):
        return True
    # Very short "X = Y" style mapping usually indicates literal translation.
    if len(cleaned) <= 40 and ("=" in cleaned or "->" in cleaned):
        return True
    return False
