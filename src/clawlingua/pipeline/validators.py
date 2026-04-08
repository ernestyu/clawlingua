"""Validation logic for LLM outputs."""

from __future__ import annotations

import re

from ..utils.text import count_sentences

_CLOZE_MARK_RE = re.compile(r"\{\{c\d+::")


def _auto_inject_cloze(text: str, target_phrases: list[str]) -> str:
    """如果 text 中没有 cloze 标记，但有 target_phrases，尝试自动注入一个 cloze。

    策略：
    - 取第一个在 text 中出现的非空 target_phrase；
    - 将其首次出现替换为 {{c1::phrase}}；
    - 若没有任何 phrase 出现在 text 中，则原样返回。
    """
    if not target_phrases:
        return text
    lowered_text = text
    for raw in target_phrases:
        phrase = str(raw).strip()
        if not phrase:
            continue
        idx = lowered_text.find(phrase)
        if idx == -1:
            continue
        # 简单替换第一个匹配位置
        return lowered_text.replace(phrase, f"{{{{c1::{phrase}}}}}", 1)
    return text


def validate_text_candidate(item: dict, *, max_sentences: int) -> tuple[bool, str]:
    text = str(item.get("text", "")).strip()
    original = str(item.get("original", "")).strip()
    target_phrases_raw = item.get("target_phrases") or []
    target_phrases = [str(x).strip() for x in target_phrases_raw if str(x).strip()]

    if not text:
        return False, "text 为空"
    if not original:
        return False, "original 为空"
    if _CLOZE_MARK_RE.search(original):
        return False, "original 不应包含 cloze 标记"

    # 如果 text 中没有 cloze 标记，但有 target_phrases，尝试自动注入一个 {{c1::...}}。
    if not _CLOZE_MARK_RE.search(text) and target_phrases:
        text = _auto_inject_cloze(text, target_phrases)
        item["text"] = text

    if not _CLOZE_MARK_RE.search(text):
        return False, "text 缺少 cloze 标记"
    if count_sentences(text) > max_sentences:
        return False, f"text 超过 {max_sentences} 句"
    if not isinstance(target_phrases_raw, list) or len(target_phrases) < 1:
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

