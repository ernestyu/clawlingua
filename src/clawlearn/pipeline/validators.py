"""Validation logic for LLM outputs."""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..utils.text import count_sentences, normalize_for_dedupe
from .taxonomy import (
    HIGH_VALUE_ADVANCED_TYPES,
    PHRASE_TAXONOMY,
    STRUCTURAL_DISCOURSE_TYPES,
    TRANSFER_MAX_CHARS,
    TRANSFER_MIN_CHARS,
    looks_like_translation_style_transfer,
    normalize_expression_transfer,
    normalize_phrase_types,
)

# {{c1::...}} format
_CLOZE_MARK_RE = re.compile(r"\{\{c\d+::")
# {c1::...} format
_CLOZE_MARK_SINGLE_RE = re.compile(r"\{c\d+::")
_CLOZE_BLOCK_RE = re.compile(r"\{\{c(\d+)::(.*?)\}\}")
# target style: {{cN::<b>phrase</b>}}(hint)
_CLOZE_STYLE_RE = re.compile(r"\{\{c\d+::\s*<b>.*?</b>\s*\}\}\s*\([^)]+\)")

_TOKEN_RE = re.compile(r"[a-zA-Z']+")

_STOPWORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "it",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "to",
    "of",
    "for",
    "on",
    "in",
    "at",
    "as",
    "if",
    "or",
    "and",
    "but",
    "so",
    "you",
    "your",
    "i",
    "we",
    "they",
    "he",
    "she",
    "them",
    "their",
}

_EASY_SINGLE_WORDS = {
    "think",
    "thinking",
    "introduce",
    "introduces",
    "introduced",
    "suggest",
    "suggests",
    "suggested",
    "answer",
    "answered",
    "fix",
    "fixed",
    "make",
    "made",
    "do",
    "did",
    "get",
    "got",
    "have",
    "has",
    "had",
    "is",
    "are",
    "was",
    "were",
}

_PHRASAL_PARTICLES = {
    "up",
    "down",
    "out",
    "off",
    "over",
    "through",
    "around",
    "between",
    "behind",
    "into",
    "across",
}

_ADVANCED_LOW_VALUE_RE = re.compile(
    r"\b(?:think if|this data or that data|good thing|bad thing|some things?)\b",
    re.IGNORECASE,
)


def classify_rejection_reason(reason: str) -> str:
    text = str(reason or "")
    if ":" not in text:
        return "unknown"
    return text.split(":", 1)[0].strip().lower() or "unknown"


def _reject(category: str, message: str) -> tuple[bool, str]:
    return False, f"{category}:{message}"


def _normalize_phrase_type_input(value: object) -> tuple[list[str], int]:
    if value is None:
        return [], 0
    if isinstance(value, list):
        raw = [str(x).strip() for x in value if str(x).strip()]
        return normalize_phrase_types(raw, max_items=2), len(raw)
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",") if part.strip()]
        return normalize_phrase_types(raw, max_items=2), len(raw)
    raw = [str(value).strip()] if str(value).strip() else []
    return normalize_phrase_types(raw, max_items=2), len(raw)


def _transfer_unrelated_to_targets(*, transfer: str, target_phrases: list[str]) -> bool:
    if not transfer or not target_phrases:
        return False
    transfer_norm = normalize_for_dedupe(transfer)
    overlap = 0
    for phrase in target_phrases:
        phrase_norm = normalize_for_dedupe(phrase)
        if not phrase_norm:
            continue
        if phrase_norm in transfer_norm or any(tok in transfer_norm for tok in phrase_norm.split()):
            overlap += 1
    return overlap == 0


def _validate_expression_transfer(
    *,
    transfer: str,
    note_hint: str,
    target_phrases: list[str],
    difficulty: str,
    learning_mode: str,
) -> tuple[bool, str]:
    if not transfer:
        return True, ""
    if len(transfer) > TRANSFER_MAX_CHARS:
        return _reject("format", f"expression_transfer exceeds {TRANSFER_MAX_CHARS} chars")
    if len(transfer) < TRANSFER_MIN_CHARS:
        return _reject("format", "expression_transfer is too short")
    if looks_like_translation_style_transfer(transfer):
        return _reject("quality", "expression_transfer looks like dictionary translation")
    if note_hint and normalize_for_dedupe(note_hint) == normalize_for_dedupe(transfer):
        return _reject("quality", "expression_transfer duplicates note_hint")
    if any(normalize_for_dedupe(tp) == normalize_for_dedupe(transfer) for tp in target_phrases):
        return _reject("quality", "expression_transfer duplicates target phrase")
    if _transfer_unrelated_to_targets(transfer=transfer, target_phrases=target_phrases):
        return _reject("quality", "expression_transfer looks unrelated to target phrase")
    if "\n" in transfer:
        return _reject("format", "expression_transfer must be one short line")
    diff = (difficulty or "intermediate").strip().lower()
    mode = (learning_mode or "lingua_expression").strip().lower()
    if mode == "lingua_reading" and len(transfer.split()) > 18:
        return _reject("quality", "lingua_reading expression_transfer should stay concise")
    if diff == "beginner" and len(transfer) > 0 and len(transfer.split()) > 20:
        return _reject("difficulty", "beginner expression_transfer should stay lightweight")
    return True, ""


def _has_obviously_invalid_type_combo(
    *,
    phrase_types: list[str],
    text: str,
    target_phrases: list[str],
) -> bool:
    if len(phrase_types) < 2:
        return False
    structural_count = len([ptype for ptype in phrase_types if ptype in STRUCTURAL_DISCOURSE_TYPES])
    phrase_token_counts = [len(_TOKEN_RE.findall(p.lower())) for p in target_phrases if p.strip()]
    max_phrase_tokens = max(phrase_token_counts) if phrase_token_counts else 0
    # Two structural discourse labels on a very short context with tiny phrase targets
    # are typically over-tagging noise from model guesses.
    if structural_count >= 2 and len(text) < 48 and max_phrase_tokens <= 2:
        return True
    if "phrasal_verb" in phrase_types and structural_count >= 1 and len(text) < 40 and max_phrase_tokens <= 2:
        return True
    return False


def _normalize_single_cloze(text: str) -> str:
    """Normalize {c1::...} into {{c1::...}}."""
    pattern = re.compile(r"\{(c\d+::[^}]*)\}")
    return pattern.sub(r"{{\1}}", text)


def _reindex_cloze_numbers(text: str) -> str:
    """Reindex cloze numbers by appearance order to c1/c2/c3..."""
    counter = {"n": 0}

    def _repl(match: re.Match[str]) -> str:
        counter["n"] += 1
        body = match.group(2)
        return f"{{{{c{counter['n']}::{body}}}}}"

    return _CLOZE_BLOCK_RE.sub(_repl, text)


def _extract_cloze_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for match in _CLOZE_BLOCK_RE.finditer(text):
        body = match.group(2).strip()
        body = re.sub(r"</?b>", "", body, flags=re.IGNORECASE).strip()
        if body:
            phrases.append(body)
    return phrases


def _phrase_score(phrase: str) -> float:
    tokens = _TOKEN_RE.findall(phrase.lower())
    if not tokens:
        return -3.0

    score = 0.0
    token_count = len(tokens)
    unique_count = len(set(tokens))
    stop_count = sum(1 for t in tokens if t in _STOPWORDS)
    stop_ratio = stop_count / max(1, token_count)

    if token_count >= 2:
        score += 1.0
    if any(len(t) >= 8 for t in tokens):
        score += 1.0
    if "-" in phrase:
        score += 0.5
    if any(t in _PHRASAL_PARTICLES for t in tokens[1:]):
        score += 0.5

    if stop_ratio >= 0.6:
        score -= 1.5
    elif stop_ratio >= 0.4:
        score -= 0.5

    if token_count > 1 and unique_count <= token_count // 2:
        score -= 1.0

    if tokens[0] in {"this", "that", "these", "those"} and "or" in tokens:
        score -= 2.0
    if phrase.strip().lower() in {"this data or that data", "think if", "basic things"}:
        score -= 3.0
    if token_count == 1 and tokens[0] in _EASY_SINGLE_WORDS:
        score -= 2.0

    return score


def _passes_difficulty(phrases: Iterable[str], *, difficulty: str, learning_mode: str) -> bool:
    cleaned = [p.strip() for p in phrases if p.strip()]
    if not cleaned:
        return False

    scores = [_phrase_score(p) for p in cleaned]
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)

    diff = (difficulty or "intermediate").strip().lower()
    mode = (learning_mode or "lingua_expression").strip().lower()
    if mode == "lingua_reading":
        # lingua_reading favors understandability over aggressiveness.
        if diff == "beginner":
            return max_score >= -1.2
        if diff == "advanced":
            return max_score >= 0.3 and avg_score >= -0.5
        return max_score >= -0.4 and avg_score >= -0.6
    if diff == "beginner":
        # Beginner: allow simple phrases but still filter obvious low-value filler.
        return max_score >= -0.5
    if diff == "advanced":
        # Advanced: avoid all-basic expression sets.
        return max_score >= 1.0 and avg_score >= 0.2 and min_score > -2.0
    # intermediate
    return max_score >= 0.0 and avg_score >= -0.2


def _auto_inject_cloze(text: str, target_phrases: list[str]) -> str:
    """Inject a fallback cloze for the first matching target phrase."""
    if not target_phrases:
        return text
    for raw in target_phrases:
        phrase = str(raw).strip()
        if not phrase:
            continue
        idx = text.find(phrase)
        if idx == -1:
            continue
        injected = f"{{{{c1::<b>{phrase}</b>}}}}(hint)"
        return text.replace(phrase, injected, 1)
    return text


def _passes_material_profile(
    *,
    material_profile: str,
    text: str,
    original: str,
    learning_mode: str,
) -> tuple[bool, str]:
    profile = (material_profile or "prose_article").strip().lower()
    mode = (learning_mode or "lingua_expression").strip().lower()
    if profile == "transcript_dialogue":
        if len(original) < 20:
            return _reject("context", "transcript candidate context is too short")
        # transcript cards should stay concise and conversational
        transcript_max_sentences = 2 if mode == "lingua_reading" else 3
        if count_sentences(text) > transcript_max_sentences:
            return _reject("difficulty", "transcript candidate has too many sentences")
    elif profile == "prose_article":
        min_original_len = 24 if mode == "lingua_reading" else 28
        if len(original) < min_original_len:
            return _reject("context", "prose candidate context is too short")
    return True, ""


def validate_text_candidate(
    item: dict,
    *,
    max_sentences: int,
    min_chars: int = 0,
    difficulty: str = "intermediate",
    material_profile: str = "prose_article",
    learning_mode: str = "lingua_expression",
) -> tuple[bool, str]:
    text = str(item.get("text", "")).strip()
    original = str(item.get("original", "")).strip()
    note_hint = str(item.get("note_hint", "")).strip()
    item["note_hint"] = note_hint
    target_phrases_raw = item.get("target_phrases") or []
    target_phrases = [str(x).strip() for x in target_phrases_raw if str(x).strip()]
    raw_phrase_types_value = item.get("phrase_types")
    phrase_types, raw_phrase_type_count = _normalize_phrase_type_input(raw_phrase_types_value)
    item["phrase_types"] = phrase_types
    raw_transfer = str(item.get("expression_transfer", "")).strip()
    transfer_text = normalize_expression_transfer(raw_transfer, max_chars=TRANSFER_MAX_CHARS * 4)
    item["expression_transfer"] = transfer_text

    if not text:
        return _reject("format", "text is empty")
    if not original:
        return _reject("format", "original is empty")
    if min_chars and len(text) < min_chars:
        return _reject("format", f"text chars < {min_chars}")
    if _CLOZE_MARK_RE.search(original) or _CLOZE_MARK_SINGLE_RE.search(original):
        return _reject("format", "original contains cloze marker")
    if "<" in original and ">" in original:
        return _reject("format", "original contains html tags")

    has_double = bool(_CLOZE_MARK_RE.search(text))
    has_single = bool(_CLOZE_MARK_SINGLE_RE.search(text))

    # Normalize single-brace cloze markers.
    if has_single and not has_double:
        text = _normalize_single_cloze(text)
        item["text"] = text
        has_double = bool(_CLOZE_MARK_RE.search(text))

    # Fallback auto-injection when target phrases exist but cloze markers missing.
    if not has_double and not has_single and target_phrases:
        text = _auto_inject_cloze(text, target_phrases)
        item["text"] = text
        has_double = bool(_CLOZE_MARK_RE.search(text))

    if not has_double:
        return _reject("format", "missing cloze marker")

    # Reindex c1/c2/c3 by order.
    text = _reindex_cloze_numbers(text)
    item["text"] = text

    if not _CLOZE_STYLE_RE.search(text):
        return _reject("format", "cloze style must be {{cN::<b>...</b>}}(hint)")

    cloze_phrases = _extract_cloze_phrases(text)
    if not cloze_phrases:
        return _reject("format", "unable to parse cloze phrases")
    if "<b>" not in text.lower() or "</b>" not in text.lower():
        return _reject("format", "cloze text missing <b>...</b> emphasis")

    normalized_cloze_phrases = [normalize_for_dedupe(p) for p in cloze_phrases]
    if len(set(normalized_cloze_phrases)) < len(normalized_cloze_phrases):
        return _reject("quality", "duplicate cloze phrases in one candidate")

    if count_sentences(text) > max_sentences:
        return _reject("format", f"text exceeds {max_sentences} sentences")
    if not isinstance(target_phrases_raw, list) or len(target_phrases) < 1:
        return _reject("format", "target_phrases is empty or invalid")
    if raw_phrase_type_count > 2:
        return _reject("taxonomy", "too_many_labels:phrase_types has too many labels")
    if raw_phrase_type_count > 0 and not phrase_types:
        allowed = ", ".join(PHRASE_TAXONOMY)
        return _reject("taxonomy", f"invalid_labels:phrase_types must be in taxonomy: {allowed}")
    if _has_obviously_invalid_type_combo(
        phrase_types=phrase_types,
        text=text,
        target_phrases=target_phrases,
    ):
        return _reject("taxonomy", "invalid_combo:phrase_types combination is inconsistent with candidate context")

    transfer_ok, transfer_reason = _validate_expression_transfer(
        transfer=transfer_text,
        note_hint=note_hint,
        target_phrases=target_phrases,
        difficulty=difficulty,
        learning_mode=learning_mode,
    )
    if not transfer_ok:
        diff = (difficulty or "").strip().lower()
        mode = (learning_mode or "lingua_expression").strip().lower()
        if "duplicates note_hint" in transfer_reason or "duplicates target phrase" in transfer_reason:
            return transfer_ok, transfer_reason
        if diff == "advanced" or mode == "lingua_reading":
            # Advanced/lingua_reading can drop invalid transfer hints instead of
            # rejecting an otherwise useful card.
            item["expression_transfer"] = ""
        else:
            return transfer_ok, transfer_reason

    profile_ok, profile_reason = _passes_material_profile(
        material_profile=material_profile,
        text=text,
        original=original,
        learning_mode=learning_mode,
    )
    if not profile_ok:
        return profile_ok, profile_reason

    # Advanced-only low-value blacklist.
    if (difficulty or "").strip().lower() == "advanced" and _ADVANCED_LOW_VALUE_RE.search(text):
        return _reject("quality", "advanced candidate contains low-value expression")

    mode = (learning_mode or "lingua_expression").strip().lower()
    phrases_for_difficulty = target_phrases or cloze_phrases
    if not _passes_difficulty(
        phrases_for_difficulty,
        difficulty=difficulty,
        learning_mode=learning_mode,
    ):
        return _reject("difficulty", f"candidate does not match {difficulty} difficulty")

    if mode == "lingua_reading":
        cloze_count = len(cloze_phrases)
        if cloze_count > 2:
            return _reject("difficulty", "lingua_reading candidate has too many clozes")
        if len(target_phrases) > 2:
            return _reject("difficulty", "lingua_reading target_phrases should be 1-2 key items")
        if material_profile == "transcript_dialogue" and count_sentences(text) > 2:
            return _reject("difficulty", "lingua_reading transcript candidate should stay compact")
        if len(original) > 420:
            return _reject("context", "lingua_reading candidate context is too long")

    diff = (difficulty or "").strip().lower()
    if diff == "advanced" and mode != "lingua_reading":
        if phrase_types and not any(pt in HIGH_VALUE_ADVANCED_TYPES for pt in phrase_types):
            phrase_scores = [_phrase_score(p) for p in phrases_for_difficulty]
            if max(phrase_scores or [-10.0]) < 1.2:
                return _reject(
                    "taxonomy",
                    "advanced_support_missing:advanced candidate lacks high-value taxonomy support",
                )

    return True, ""


def validate_translation_text(text: str) -> tuple[bool, str]:
    value = text.strip()
    if not value:
        return _reject("format", "translation is empty")
    lowered = value.lower()
    if lowered.startswith("翻译:") or lowered.startswith("翻译："):
        return _reject("format", "translation has forbidden prefix")
    if "**" in value:
        return _reject("format", "translation contains markdown **")
    return True, ""
