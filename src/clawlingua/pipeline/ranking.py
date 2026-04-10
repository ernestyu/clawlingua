"""Candidate ranking and learning-value scoring."""

from __future__ import annotations

import re
from typing import Any

from ..utils.text import count_sentences

_TOKEN_RE = re.compile(r"[A-Za-z']+")
_PHRASAL_VERB_RE = re.compile(
    r"\b(?:give|take|put|set|get|go|come|look|work|turn|bring|carry|call|show|point|run|break)\s+"
    r"(?:up|down|out|off|over|through|back|away|around|into|across)\b",
    re.IGNORECASE,
)
_COLLOCATION_RE = re.compile(
    r"\b(?:in terms of|as a result|on the other hand|in order to|tend to|be likely to|at least)\b",
    re.IGNORECASE,
)
_LOW_VALUE_RE = re.compile(
    r"\b(?:think if|this data or that data|very very|good thing|bad thing|basic things?)\b",
    re.IGNORECASE,
)


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _phrase_types(item: dict[str, Any]) -> list[str]:
    text = str(item.get("text", ""))
    phrases = [str(x) for x in (item.get("target_phrases") or []) if str(x).strip()]
    output: set[str] = set()
    if _PHRASAL_VERB_RE.search(text):
        output.add("phrasal_verb")
    if _COLLOCATION_RE.search(text):
        output.add("discourse_collocation")
    if any(len(p.split()) >= 3 for p in phrases):
        output.add("multiword_expression")
    if not output:
        output.add("general_expression")
    return sorted(output)


def _difficulty_score_adjustment(score: float, *, difficulty: str) -> float:
    diff = (difficulty or "intermediate").strip().lower()
    if diff == "beginner":
        return score + 0.4
    if diff == "advanced":
        return score - 0.2
    return score


def _material_score_adjustment(score: float, *, material_profile: str, text: str) -> float:
    profile = (material_profile or "prose_article").strip().lower()
    if profile == "transcript_dialogue":
        if "'" in text or "?" in text:
            return score + 0.6
        return score + 0.2
    if profile == "prose_article":
        if count_sentences(text) >= 2:
            return score + 0.3
    return score


def score_candidate(
    item: dict[str, Any],
    *,
    difficulty: str,
    material_profile: str,
) -> tuple[float, list[str], list[str]]:
    text = str(item.get("text", "")).strip()
    original = str(item.get("original", "")).strip()
    phrases = [str(x).strip() for x in (item.get("target_phrases") or []) if str(x).strip()]
    phrase_types = _phrase_types(item)

    score = 0.0
    reasons: list[str] = []

    if len(original) >= 60:
        score += 1.2
        reasons.append("context_complete")
    elif len(original) >= 35:
        score += 0.5
        reasons.append("context_ok")
    else:
        score -= 1.0
        reasons.append("context_short")

    sentence_count = count_sentences(text)
    if sentence_count <= 3:
        score += 0.6
    else:
        score -= 0.5
        reasons.append("too_long")

    phrase_token_counts = [_token_count(p) for p in phrases] if phrases else [0]
    avg_phrase_tokens = sum(phrase_token_counts) / max(1, len(phrase_token_counts))
    if avg_phrase_tokens >= 2:
        score += 0.8
        reasons.append("reusable_phrase")
    if avg_phrase_tokens >= 3:
        score += 0.4
        reasons.append("multiword_focus")
    if any(len(p) >= 12 for p in phrases):
        score += 0.3

    if _PHRASAL_VERB_RE.search(text):
        score += 0.6
        reasons.append("phrasal_verb")
    if _COLLOCATION_RE.search(text):
        score += 0.5
        reasons.append("discourse_pattern")
    if _LOW_VALUE_RE.search(text):
        score -= 2.2
        reasons.append("low_value_pattern")

    score = _difficulty_score_adjustment(score, difficulty=difficulty)
    score = _material_score_adjustment(score, material_profile=material_profile, text=text)
    return round(score, 4), reasons, phrase_types


def rank_candidates(
    items: list[dict[str, Any]],
    *,
    difficulty: str,
    material_profile: str,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for item in items:
        score, reasons, phrase_types = score_candidate(
            item,
            difficulty=difficulty,
            material_profile=material_profile,
        )
        enriched = dict(item)
        enriched["learning_value_score"] = score
        enriched["selection_reason"] = ", ".join(reasons[:4]) if reasons else "generic"
        enriched["phrase_types"] = phrase_types
        ranked.append(enriched)

    ranked.sort(
        key=lambda row: (
            float(row.get("learning_value_score", 0.0)),
            len(str(row.get("original", ""))),
            len(str(row.get("text", ""))),
        ),
        reverse=True,
    )
    return ranked
