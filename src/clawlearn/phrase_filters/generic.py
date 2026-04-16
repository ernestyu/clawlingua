"""Language-agnostic phrase filtering rules."""

from __future__ import annotations

import re
from typing import Any

from ..utils.text import normalize_for_dedupe

_MULTI_SPACE_RE = re.compile(r"\s+")


def _empty_stats(*, source_lang: str) -> dict[str, Any]:
    return {
        "source_lang": source_lang,
        "language": source_lang or "unknown",
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


def _normalize_phrase(phrase: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", str(phrase or "").strip())


def filter_phrases(
    *,
    source_lang: str,
    phrases: list[str],
    context: str,
    difficulty: str = "",
) -> tuple[list[str], dict[str, Any]]:
    del context
    del difficulty

    stats = _empty_stats(source_lang=source_lang)
    kept: list[str] = []
    seen_keys: set[str] = set()

    for raw_phrase in phrases:
        phrase = _normalize_phrase(str(raw_phrase))
        if not phrase:
            _record_drop(stats, rule="drop:empty_phrase", phrase=str(raw_phrase))
            continue
        if any(ch in phrase for ch in ",;:"):
            _record_drop(stats, rule="drop:forbidden_punctuation", phrase=phrase)
            continue
        key = normalize_for_dedupe(phrase)
        if not key:
            _record_drop(stats, rule="drop:empty_after_normalize", phrase=phrase)
            continue
        if key in seen_keys:
            _record_drop(stats, rule="drop:duplicate_phrase", phrase=phrase)
            continue
        seen_keys.add(key)
        kept.append(phrase)
    return kept, stats

