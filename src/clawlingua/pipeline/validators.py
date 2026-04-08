"""Validation logic for LLM outputs."""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..utils.text import count_sentences, normalize_for_dedupe

# 双大括号形式 {{c1::...}}
_CLOZE_MARK_RE = re.compile(r"\{\{c\d+::")
# 单大括号形式 {c1::...}
_CLOZE_MARK_SINGLE_RE = re.compile(r"\{c\d+::")
_CLOZE_BLOCK_RE = re.compile(r"\{\{c(\d+)::(.*?)\}\}")
# 目标样式：{{cN::<b>phrase</b>}}(提示)
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


def _normalize_single_cloze(text: str) -> str:
    """将 {c1::...} 规范化为 {{c1::...}} 形式，避免出现 {c1::{{c1::...}}} 套娃。

    这里用一个简单规则：
    - 匹配 {cN::...}，将外层大括号替换为双大括号。
    """
    pattern = re.compile(r"\{(c\d+::[^}]*)\}")
    return pattern.sub(r"{{\1}}", text)


def _reindex_cloze_numbers(text: str) -> str:
    """将 cloze 编号按出现顺序重排为 c1/c2/c3...。"""

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


def _passes_difficulty(phrases: Iterable[str], *, difficulty: str) -> bool:
    cleaned = [p.strip() for p in phrases if p.strip()]
    if not cleaned:
        return False

    scores = [_phrase_score(p) for p in cleaned]
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)

    diff = (difficulty or "intermediate").strip().lower()
    if diff == "beginner":
        # 初级难度放宽，但仍过滤明显空洞短语。
        return max_score >= -0.5
    if diff == "advanced":
        # 高级难度更严格：不能全是常见/空洞表达。
        return max_score >= 1.0 and avg_score >= 0.2 and min_score > -2.0
    # intermediate
    return max_score >= 0.0 and avg_score >= -0.2


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
        # 简单替换第一个匹配位置，遵循 {{c1::<b>...</b>}}(提示) 形态。
        injected = f"{{{{c1::<b>{phrase}</b>}}}}(提示)"
        return lowered_text.replace(phrase, injected, 1)
    return text


def validate_text_candidate(
    item: dict,
    *,
    max_sentences: int,
    min_chars: int = 0,
    difficulty: str = "intermediate",
) -> tuple[bool, str]:
    text = str(item.get("text", "")).strip()
    original = str(item.get("original", "")).strip()
    target_phrases_raw = item.get("target_phrases") or []
    target_phrases = [str(x).strip() for x in target_phrases_raw if str(x).strip()]

    if not text:
        return False, "text 为空"
    if not original:
        return False, "original 为空"
    if min_chars and len(text) < min_chars:
        return False, f"text 字符数不足 {min_chars}"
    if _CLOZE_MARK_RE.search(original) or _CLOZE_MARK_SINGLE_RE.search(original):
        return False, "original 不应包含 cloze 标记"
    if "<" in original and ">" in original:
        return False, "original 不应包含 HTML 标记"

    has_double = bool(_CLOZE_MARK_RE.search(text))
    has_single = bool(_CLOZE_MARK_SINGLE_RE.search(text))

    # 如果已有单大括号 cloze，但没有双大括号，先规范化为双大括号
    if has_single and not has_double:
        text = _normalize_single_cloze(text)
        item["text"] = text
        has_double = bool(_CLOZE_MARK_RE.search(text))

    # 如果完全没有任何 cloze 标记，但有 target_phrases，则自动注入一个
    if not has_double and not has_single and target_phrases:
        text = _auto_inject_cloze(text, target_phrases)
        item["text"] = text
        has_double = bool(_CLOZE_MARK_RE.search(text))

    if not has_double:
        return False, "text 缺少 cloze 标记"

    # 统一重编号，避免同一条 text 中多个 c1 的情况。
    text = _reindex_cloze_numbers(text)
    item["text"] = text

    # 样式约束：至少一个 cloze 满足 {{cN::<b>...</b>}}(提示)
    if not _CLOZE_STYLE_RE.search(text):
        return False, "cloze 样式不符合 {{cN::<b>...</b>}}(提示)"

    # 追加严格性：每个 cloze 块都应带 <b>。
    cloze_phrases = _extract_cloze_phrases(text)
    if not cloze_phrases:
        return False, "无法解析 cloze 短语"
    if "<b>" not in text.lower() or "</b>" not in text.lower():
        return False, "cloze 缺少 <b>...</b> 标记"
    normalized_cloze_phrases = [normalize_for_dedupe(p) for p in cloze_phrases]
    if len(set(normalized_cloze_phrases)) < len(normalized_cloze_phrases):
        return False, "同一条 text 中重复挖空同一短语"

    if count_sentences(text) > max_sentences:
        return False, f"text 超过 {max_sentences} 句"
    if not isinstance(target_phrases_raw, list) or len(target_phrases) < 1:
        return False, "target_phrases 不足"

    phrases_for_difficulty = target_phrases or cloze_phrases
    if not _passes_difficulty(phrases_for_difficulty, difficulty=difficulty):
        return False, f"不符合 {difficulty} 难度要求"

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
