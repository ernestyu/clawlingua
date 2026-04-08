"""Deduplicate candidate cards."""

from __future__ import annotations

from ..utils.text import normalize_for_dedupe


def dedupe_candidates(items: list[dict]) -> list[dict]:
    seen_original: set[str] = set()
    seen_text: set[str] = set()
    output: list[dict] = []
    for item in items:
        key_original = normalize_for_dedupe(str(item.get("original", "")))
        key_text = normalize_for_dedupe(str(item.get("text", "")))
        if not key_original or not key_text:
            continue
        if key_original in seen_original:
            continue
        if key_text in seen_text:
            continue
        seen_original.add(key_original)
        seen_text.add(key_text)
        output.append(item)
    return output

