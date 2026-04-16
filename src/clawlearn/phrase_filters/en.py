"""English phrase filtering rules."""

from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z']+")
_ARTICLES = {"a", "an", "the"}

# Keep v1 blacklist intentionally small and high precision.
_BLACKLIST = {
    "one of the",
    "look at the",
    "in the same",
}
_WHITELIST = {
    "as a result",
    "in spite of",
    "in terms of",
    "make things happen",
    "on the one hand",
    "on the other hand",
    "that said",
    "to be fair",
}
_PLACEHOLDER_TOKEN_WHITELIST = {
    "make things happen",
}
_DISCOURSE_PLACEHOLDER_BLACKLIST = {
    "another thing",
    "another explanation",
    "there is another explanation",
}
_PLACEHOLDER_SINGLE_TOKENS = {
    "stuff",
    "thing",
    "things",
    "somewhere",
    "someone",
    "something",
    "anyone",
    "anything",
}
_PLACEHOLDER_PHRASE_PATTERNS = (
    ("some", "place"),
)
_PLACEHOLDER_DETERMINERS = {
    "a",
    "an",
    "the",
    "some",
    "any",
    "another",
    "to",
    "of",
    "in",
    "on",
    "at",
}
_OPEN_ENDING_WORDS = {
    "from",
    "would",
    "could",
    "should",
    "might",
    "may",
    "can",
    "will",
}
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
_ADVANCED_MIN_SCORE = 0.8


def _empty_stats(*, source_lang: str) -> dict[str, Any]:
    return {
        "source_lang": source_lang,
        "language": "en",
        "dropped_count": 0,
        "dropped_by_rule": {},
        "examples_by_rule": {},
    }


def _record_drop(stats: dict[str, Any], *, rule: str, phrase: str) -> None:
    stats["dropped_count"] = int(stats.get("dropped_count", 0)) + 1
    dropped_by_rule = stats.setdefault("dropped_by_rule", {})
    dropped_by_rule[rule] = int(dropped_by_rule.get(rule, 0)) + 1
    examples_by_rule = stats.setdefault("examples_by_rule", {})
    bucket = examples_by_rule.setdefault(rule, [])
    text = str(phrase).strip()
    if text and text not in bucket and len(bucket) < 3:
        bucket.append(text)


def _stopword_ratio(phrase: str) -> float:
    tokens = [token.lower() for token in _TOKEN_RE.findall(phrase)]
    if not tokens:
        return 1.0
    stop_count = sum(1 for token in tokens if token in _STOPWORDS)
    return stop_count / len(tokens)


def phrase_quality_score(phrase: str) -> float:
    tokens = [token.lower() for token in _TOKEN_RE.findall(phrase)]
    if not tokens:
        return -3.0
    score = 0.0
    token_count = len(tokens)
    unique_count = len(set(tokens))
    stop_count = sum(1 for token in tokens if token in _STOPWORDS)
    stop_ratio = stop_count / max(1, token_count)

    if token_count >= 2:
        score += 1.0
    if any(len(token) >= 8 for token in tokens):
        score += 1.0
    if "-" in phrase:
        score += 0.5
    if any(token in _PHRASAL_PARTICLES for token in tokens[1:]):
        score += 0.5

    if stop_ratio >= 0.6:
        score -= 1.5
    elif stop_ratio >= 0.4:
        score -= 0.5

    if token_count > 1 and unique_count <= token_count // 2:
        score -= 1.0

    if tokens[0] in {"this", "that", "these", "those"} and "or" in tokens:
        score -= 2.0
    if token_count == 1 and tokens[0] in {"think", "do", "get", "have", "is", "are"}:
        score -= 2.0
    return score


def _contains_placeholder_pattern(tokens: list[str]) -> bool:
    if any(token in _PLACEHOLDER_SINGLE_TOKENS for token in tokens):
        return True
    for left, right in _PLACEHOLDER_PHRASE_PATTERNS:
        for idx in range(len(tokens) - 1):
            if tokens[idx] == left and tokens[idx + 1] == right:
                return True
    return False


def _has_descriptive_non_placeholder_token(tokens: list[str]) -> bool:
    for token in tokens:
        if token in _PLACEHOLDER_SINGLE_TOKENS:
            continue
        if token in _PLACEHOLDER_DETERMINERS:
            continue
        if len(token) >= 8:
            return True
    return False


def filter_phrases(
    *,
    source_lang: str,
    phrases: list[str],
    context: str,
    difficulty: str = "",
) -> tuple[list[str], dict[str, Any]]:
    del context

    stats = _empty_stats(source_lang=source_lang)
    kept: list[str] = []
    diff = (difficulty or "").strip().lower()
    advanced_mode = diff == "advanced"

    for phrase in phrases:
        value = str(phrase).strip()
        key = value.lower()
        if not value:
            _record_drop(stats, rule="drop:empty_phrase", phrase=value)
            continue
        if key in _WHITELIST:
            kept.append(value)
            continue
        tokens = [token.lower() for token in _TOKEN_RE.findall(value)]
        token_count = len(tokens)
        if advanced_mode and key in _DISCOURSE_PLACEHOLDER_BLACKLIST:
            _record_drop(stats, rule="drop:discourse_placeholder_blacklist", phrase=value)
            continue
        if (
            advanced_mode
            and 0 < token_count <= 4
            and key not in _PLACEHOLDER_TOKEN_WHITELIST
            and _contains_placeholder_pattern(tokens)
            and not _has_descriptive_non_placeholder_token(tokens)
        ):
            _record_drop(stats, rule="drop:placeholder_token", phrase=value)
            continue
        if key in _BLACKLIST:
            _record_drop(stats, rule="drop:blacklist_ngram", phrase=value)
            continue
        if tokens and tokens[-1] in _ARTICLES:
            _record_drop(stats, rule="drop:ends_with_article", phrase=value)
            continue
        if tokens and tokens[-1] in _OPEN_ENDING_WORDS:
            _record_drop(stats, rule="drop:ends_with_open_word", phrase=value)
            continue
        if _stopword_ratio(value) >= (2 / 3):
            _record_drop(stats, rule="drop:stopword_ratio", phrase=value)
            continue
        if advanced_mode and phrase_quality_score(value) < _ADVANCED_MIN_SCORE:
            _record_drop(stats, rule="drop:advanced_min_phrase_score", phrase=value)
            continue
        kept.append(value)

    return kept, stats
