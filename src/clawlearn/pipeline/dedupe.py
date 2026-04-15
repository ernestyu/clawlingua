"""Deduplicate candidate cards."""

from __future__ import annotations

import re

from ..utils.text import normalize_for_dedupe

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def dedupe_candidates(items: list[dict]) -> list[dict]:
    seen_original: set[str] = set()
    seen_text: set[str] = set()
    seen_phrase_key_by_chunk: dict[str, set[str]] = {}
    seen_original_tokens: list[set[str]] = []

    output: list[dict] = []
    for item in items:
        key_original = normalize_for_dedupe(str(item.get("original", "")))
        key_text = normalize_for_dedupe(str(item.get("text", "")))
        chunk_id = str(item.get("chunk_id", "")).strip()
        phrases = [
            normalize_for_dedupe(str(p))
            for p in (item.get("target_phrases") or [])
            if normalize_for_dedupe(str(p))
        ]
        phrase_key = "|".join(sorted(set(phrases)))

        if not key_original or not key_text:
            continue
        if key_original in seen_original:
            continue
        if key_text in seen_text:
            continue

        # 同一 chunk 内同一组 target phrases 只保留一条，避免近似重复卡片。
        if chunk_id and phrase_key:
            bucket = seen_phrase_key_by_chunk.setdefault(chunk_id, set())
            if phrase_key in bucket:
                continue

        # 跨 chunk 的近似重复过滤：original token Jaccard 过高则跳过。
        original_tokens = _token_set(key_original)
        if any(_jaccard(original_tokens, prev) >= 0.9 for prev in seen_original_tokens):
            continue

        seen_original.add(key_original)
        seen_text.add(key_text)
        if chunk_id and phrase_key:
            seen_phrase_key_by_chunk.setdefault(chunk_id, set()).add(phrase_key)
        seen_original_tokens.append(original_tokens)
        output.append(item)
    return output
