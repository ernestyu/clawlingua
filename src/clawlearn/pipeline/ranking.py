"""Candidate ranking and learning-value scoring."""

from __future__ import annotations

import re
from typing import Any

from ..utils.text import count_sentences
from .taxonomy import (
    HIGH_VALUE_ADVANCED_TYPES,
    STRUCTURAL_DISCOURSE_TYPES,
    normalize_expression_transfer,
    normalize_phrase_types,
    phrase_type_weight,
)

_TOKEN_RE = re.compile(r"[A-Za-z']+")
_PHRASAL_VERB_RE = re.compile(
    r"\b(?:give|take|put|set|get|go|come|look|work|turn|bring|carry|call|show|point|run|break|end)\s+"
    r"(?:up|down|out|off|over|through|back|away|around|into|across|with)\b",
    re.IGNORECASE,
)
_LOW_VALUE_RE = re.compile(
    r"\b(?:think if|this data or that data|very very|good thing|bad thing|basic things?)\b",
    re.IGNORECASE,
)
_SCOPE_LIMIT_RE = re.compile(
    r"\b(?:to (?:some|a certain) extent|to the extent that|strictly speaking|for the most part|at least|at most|in this sense|within the scope of)\b",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"\b(?:to sum up|in short|all in all|the bottom line is|what it boils down to|boils down to)\b",
    re.IGNORECASE,
)
_COGNITIVE_FRAME_RE = re.compile(
    r"\b(?:think of .* as|frame(?:d)? as|amounts to|this suggests|this implies|this points to|seen as|not so much .* as)\b",
    re.IGNORECASE,
)
_STANCE_FRAME_RE = re.compile(
    r"\b(?:i would argue|one could argue|i'd say|it seems to me|from where i stand|to my mind|my view is|the point here is)\b",
    re.IGNORECASE,
)
_ABSTRACTION_BRIDGE_RE = re.compile(
    r"\b(?:stepping back|at a higher level|in broader terms|from a broader perspective|the bigger picture|as a general rule|in structural terms|this reflects|this reveals)\b",
    re.IGNORECASE,
)
_TRANSFER_CHUNK_RE = re.compile(
    r"\b(?:it turns out|if anything|at the end of the day|the point is that|what matters is|on that note|that being said)\b",
    re.IGNORECASE,
)
_STRONG_COLLOCATION_RE = re.compile(
    r"\b(?:in terms of|as a result|play a role|make a case for|raise a question|bear in mind|in light of|by contrast)\b",
    re.IGNORECASE,
)

_PATTERN_LAYER: dict[str, list[re.Pattern[str]]] = {
    "concession_contrast": [
        re.compile(
            r"\b(?:although|though|even though|while|whereas|however|yet|nevertheless|nonetheless|still|having said that|that said|despite|in spite of)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:on (?:the )?one hand .* on the other hand|rather than)\b", re.IGNORECASE),
    ],
    "stance_positioning": [
        re.compile(
            r"\b(?:i think|i believe|i'd argue|in my view|from my perspective|to me|the key point is|what matters is|my take is)\b",
            re.IGNORECASE,
        ),
        _SCOPE_LIMIT_RE,
        _STANCE_FRAME_RE,
    ],
    "discourse_organizer": [
        re.compile(
            r"\b(?:in other words|to put it differently|for example|for instance|to begin with|first of all|on the other hand|more importantly|to sum up|in short)\b",
            re.IGNORECASE,
        ),
        _SUMMARY_RE,
    ],
    "abstraction_bridge": [
        re.compile(
            r"\b(?:at a broader level|more generally|in essence|at its core|this suggests that|which points to|in principle|in practice)\b",
            re.IGNORECASE,
        ),
        _COGNITIVE_FRAME_RE,
        _ABSTRACTION_BRIDGE_RE,
    ],
    "metaphor_imagery": [
        re.compile(
            r"\b(?:under the hood|at the heart of|on the table|in the wild|move the needle|a slippery slope|paint(?:s|ing)? a picture|the landscape of)\b",
            re.IGNORECASE,
        ),
    ],
    "strong_collocation": [
        _STRONG_COLLOCATION_RE,
    ],
    "reusable_high_frequency_chunk": [
        _TRANSFER_CHUNK_RE,
    ],
}


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _difficulty_score_adjustment(score: float, *, difficulty: str) -> float:
    diff = (difficulty or "intermediate").strip().lower()
    if diff == "beginner":
        return score + 0.5
    if diff == "advanced":
        return score - 0.1
    return score


def _material_score_adjustment(score: float, *, material_profile: str, text: str) -> float:
    profile = (material_profile or "prose_article").strip().lower()
    if profile == "transcript_dialogue":
        if "'" in text or "?" in text:
            return score + 0.6
        return score + 0.2
    if profile == "prose_article" and count_sentences(text) >= 2:
        return score + 0.3
    return score


def _default_expression_transfer(label: str) -> str:
    templates = {
        "metaphor_imagery": "Useful for adding vivid framing when presenting an abstract idea.",
        "stance_positioning": "Useful for signaling your stance before giving evidence.",
        "concession_contrast": "Useful for acknowledging a point before pivoting to contrast.",
        "discourse_organizer": "Useful for organizing argument flow across longer explanations.",
        "abstraction_bridge": "Useful for lifting specific examples into a broader claim.",
        "reusable_high_frequency_chunk": "Useful as a reusable chunk for fluent topic transitions.",
        "phrasal_verb": "Useful for natural everyday actions in spoken or written contexts.",
        "strong_collocation": "Useful for producing native-like collocations in formal discussion.",
    }
    return templates.get(label, "")


def _resolve_expression_transfer(
    item: dict[str, Any],
    *,
    phrase_types: list[str],
    difficulty: str,
    learning_mode: str,
) -> str:
    raw = normalize_expression_transfer(item.get("expression_transfer", ""))
    diff = (difficulty or "intermediate").strip().lower()
    mode = (learning_mode or "lingua_expression").strip().lower()
    if raw:
        return raw
    if mode == "lingua_reading":
        return ""
    if diff == "beginner" or not phrase_types:
        return ""
    return _default_expression_transfer(phrase_types[0])


def _extract_pattern_type_scores(*, text: str, original: str, phrases: list[str]) -> tuple[dict[str, float], list[str]]:
    merged = " ".join([text, original, *phrases])
    scores: dict[str, float] = {}
    reasons: list[str] = []
    for label, patterns in _PATTERN_LAYER.items():
        for pattern in patterns:
            if not pattern.search(merged):
                continue
            scores[label] = max(scores.get(label, 0.0), 0.9)
            reasons.append(f"pattern:{label}")
            break
    if _PHRASAL_VERB_RE.search(merged):
        scores["phrasal_verb"] = max(scores.get("phrasal_verb", 0.0), 0.8)
        reasons.append("pattern:phrasal_verb")
    return scores, reasons


def _model_label_supported(
    *,
    label: str,
    pattern_scores: dict[str, float],
    text: str,
    phrases: list[str],
) -> bool:
    if label in pattern_scores:
        return True
    merged = " ".join([text, *phrases])
    if label == "phrasal_verb":
        return bool(_PHRASAL_VERB_RE.search(merged))
    if label == "strong_collocation":
        return bool(_STRONG_COLLOCATION_RE.search(merged))
    if label == "reusable_high_frequency_chunk":
        return bool(_TRANSFER_CHUNK_RE.search(merged))
    # Structural discourse labels should have explicit contextual cues to remain stable.
    if label in STRUCTURAL_DISCOURSE_TYPES:
        return False
    return True


def _compose_phrase_types(
    *,
    model_phrase_types: list[str],
    pattern_scores: dict[str, float],
    text: str,
    phrases: list[str],
    difficulty: str,
) -> tuple[list[str], list[str]]:
    # Keep taxonomy as a strict pre-rank signal from LLM stage.
    # Do not add/replace labels programmatically in ranking.
    _ = pattern_scores, text, phrases, difficulty
    return list(model_phrase_types[:2]), []


def score_candidate(
    item: dict[str, Any],
    *,
    difficulty: str,
    material_profile: str,
    learning_mode: str = "lingua_expression",
) -> tuple[float, list[str], list[str], str, dict[str, list[str]]]:
    text = str(item.get("text", "")).strip()
    original = str(item.get("original", "")).strip()
    phrases = [str(x).strip() for x in (item.get("target_phrases") or []) if str(x).strip()]
    model_phrase_types = normalize_phrase_types(item.get("phrase_types"), max_items=2)
    pattern_scores, pattern_reasons = _extract_pattern_type_scores(text=text, original=original, phrases=phrases)
    phrase_types, correction_reasons = _compose_phrase_types(
        model_phrase_types=model_phrase_types,
        pattern_scores=pattern_scores,
        text=text,
        phrases=phrases,
        difficulty=difficulty,
    )
    expression_transfer = _resolve_expression_transfer(
        item,
        phrase_types=phrase_types,
        difficulty=difficulty,
        learning_mode=learning_mode,
    )
    mode = (learning_mode or "lingua_expression").strip().lower()

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
    sentence_limit = 2 if mode == "lingua_reading" else 3
    if sentence_count <= sentence_limit:
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
        score += 0.5
        reasons.append("multiword_focus")
    if any(len(p) >= 12 for p in phrases):
        score += 0.3

    if _LOW_VALUE_RE.search(text):
        score -= 2.5
        reasons.append("low_value_pattern")

    for ptype in phrase_types:
        type_weight = phrase_type_weight(label=ptype, difficulty=difficulty)
        if mode == "lingua_reading":
            type_weight *= 0.35
        score += type_weight
    if pattern_reasons:
        reasons.extend(pattern_reasons[:2])
    if correction_reasons:
        reasons.extend(correction_reasons[:2])

    diff = (difficulty or "intermediate").strip().lower()
    high_value_count = len([ptype for ptype in phrase_types if ptype in HIGH_VALUE_ADVANCED_TYPES])
    if mode == "lingua_reading":
        # lingua_reading: prioritize readability and discourse continuity.
        if sentence_count > 2:
            score -= 0.8
            reasons.append("lingua_reading_too_many_sentences")
        cloze_count = text.count("{{c")
        if cloze_count > 2:
            score -= 1.0
            reasons.append("lingua_reading_too_many_clozes")
        discourse_help = len(
            [
                p
                for p in phrase_types
                if p in {"discourse_organizer", "concession_contrast", "stance_positioning", "abstraction_bridge"}
            ]
        )
        if discourse_help:
            score += 0.9 + 0.2 * (discourse_help - 1)
            reasons.append("lingua_reading_discourse_help")
        if high_value_count and diff == "advanced":
            score += 0.25
            reasons.append("lingua_reading_advanced_signal")
        if expression_transfer:
            score += 0.1
            reasons.append("transfer_optional")
    else:
        if diff == "advanced":
            if high_value_count > 0:
                score += 0.8 + 0.3 * (high_value_count - 1)
                reasons.append("advanced_high_value")
            elif phrase_types == ["phrasal_verb"]:
                score -= 1.2
                reasons.append("advanced_phrasal_only")
        if expression_transfer:
            score += 0.5 if diff == "advanced" else 0.3
            reasons.append("transfer_ready")

    score = _difficulty_score_adjustment(score, difficulty=difficulty)
    score = _material_score_adjustment(score, material_profile=material_profile, text=text)
    debug_types = {
        "model_phrase_types": model_phrase_types,
        "programmatic_phrase_types": sorted(pattern_scores.keys()),
    }
    return round(score, 4), reasons, phrase_types, expression_transfer, debug_types


def rank_candidates(
    items: list[dict[str, Any]],
    *,
    difficulty: str,
    material_profile: str,
    learning_mode: str = "lingua_expression",
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for item in items:
        score, reasons, phrase_types, expression_transfer, debug_types = score_candidate(
            item,
            difficulty=difficulty,
            material_profile=material_profile,
            learning_mode=learning_mode,
        )
        enriched = dict(item)
        enriched["learning_value_score"] = score
        enriched["selection_reason"] = ", ".join(reasons[:5]) if reasons else "generic"
        enriched["phrase_types"] = phrase_types
        enriched["expression_transfer"] = expression_transfer
        enriched["model_phrase_types"] = debug_types["model_phrase_types"]
        enriched["programmatic_phrase_types"] = debug_types["programmatic_phrase_types"]
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
