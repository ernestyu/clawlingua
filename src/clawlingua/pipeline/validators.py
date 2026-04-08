"""Validation logic for LLM outputs."""

from __future__ import annotations

import re

from ..utils.text import count_sentences

_CLOZE_MARK_RE = re.compile(r"\{\{c\d+::")


def validate_text_candidate(item: dict, *, max_sentences: int) -> tuple[bool, str]:
    text = str(item.get("text", "")).strip()
    original = str(item.get("original", "")).strip()
    target_phrases = item.get("target_phrases") or []

    if not text:
        return False, "text 为空"
    if not original:
        return False, "original 为空"
    if _CLOZE_MARK_RE.search(original):
        return False, "original 不应包含 cloze 标记"
    if not _CLOZE_MARK_RE.search(text):
        return False, "text 缺少 cloze 标记"
    if count_sentences(text) > max_sentences:
        return False, f"text 超过 {max_sentences} 句"
    if not isinstance(target_phrases, list) or len([x for x in target_phrases if str(x).strip()]) < 1:
        return False, "target_phrases 不足"
    return True, ""


def validate_translation_text(text: str) -> tuple[bool, str]:
    value = text.strip()
    if not value:
        return False, "translation 为空"
    lowered = value.lower()
    if lowered.startswith("翻译:") or lowered.startswith("翻译："):
        return False, "translation 含不允许前缀"
    if "**" in value:
        return False, "translation 包含 Markdown **"
    return True, ""

