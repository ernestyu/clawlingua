"""Phrase-level filtering entrypoint with language routing."""

from __future__ import annotations

from typing import Any

from . import de, en, fr, generic, ja, zh


def _empty_stats(*, source_lang: str, language: str) -> dict[str, Any]:
    return {
        "source_lang": source_lang,
        "language": language,
        "dropped_count": 0,
        "dropped_by_rule": {},
        "examples_by_rule": {},
    }


def _merge_stats(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged_dropped = int(merged.get("dropped_count", 0)) + int(incoming.get("dropped_count", 0))
    merged["dropped_count"] = merged_dropped

    dropped_by_rule: dict[str, int] = {}
    for mapping in (merged.get("dropped_by_rule", {}), incoming.get("dropped_by_rule", {})):
        if not isinstance(mapping, dict):
            continue
        for key, value in mapping.items():
            rule = str(key).strip()
            if not rule:
                continue
            try:
                count = int(value)
            except (TypeError, ValueError):
                count = 0
            if count <= 0:
                continue
            dropped_by_rule[rule] = dropped_by_rule.get(rule, 0) + count
    merged["dropped_by_rule"] = dropped_by_rule

    examples_by_rule: dict[str, list[str]] = {}
    for mapping in (merged.get("examples_by_rule", {}), incoming.get("examples_by_rule", {})):
        if not isinstance(mapping, dict):
            continue
        for key, value in mapping.items():
            rule = str(key).strip()
            if not rule or not isinstance(value, list):
                continue
            bucket = examples_by_rule.setdefault(rule, [])
            for item in value:
                text = str(item).strip()
                if not text or text in bucket:
                    continue
                if len(bucket) >= 3:
                    break
                bucket.append(text)
    merged["examples_by_rule"] = examples_by_rule
    return merged


def filter_phrases(
    *,
    source_lang: str,
    phrases: list[str],
    context: str,
    difficulty: str = "",
) -> tuple[list[str], dict[str, Any]]:
    normalized_source = (source_lang or "").strip().lower()
    language = normalized_source or "unknown"
    generic_kept, generic_stats = generic.filter_phrases(
        source_lang=normalized_source,
        phrases=phrases,
        context=context,
        difficulty=difficulty,
    )

    language_filter = {
        "en": en.filter_phrases,
        "fr": fr.filter_phrases,
        "de": de.filter_phrases,
        "zh": zh.filter_phrases,
        "ja": ja.filter_phrases,
    }.get(normalized_source)
    if language_filter is None:
        stats = _empty_stats(source_lang=normalized_source, language=language)
        return generic_kept, _merge_stats(stats, generic_stats)

    kept, lang_stats = language_filter(
        source_lang=normalized_source,
        phrases=generic_kept,
        context=context,
        difficulty=difficulty,
    )
    stats = _empty_stats(source_lang=normalized_source, language=language)
    stats = _merge_stats(stats, generic_stats)
    stats = _merge_stats(stats, lang_stats)
    return kept, stats

