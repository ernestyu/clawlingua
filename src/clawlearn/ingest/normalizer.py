"""Text normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape

from ..utils.text import normalize_paragraph_text

_NOISE_LINES = (
    "cookie",
    "privacy policy",
    "terms of service",
    "subscribe",
    "sign in",
)
_TRANSCRIPT_HEADERS = (
    "transcript",
    "full transcript",
    "podcast transcript",
)
_SPEAKER_WITH_TIMESTAMP_RE = re.compile(
    r"^[A-Z][A-Za-z'.-]*(?:\s+[A-Z][A-Za-z'.-]*){0,6}\s+\d{1,2}:\d{2}(?::\d{2})?$"
)
_TIMESTAMP_HEADING_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?(?:\s*[-:]\s*.*)?$")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_MD_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_MD_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_AUTOLINK_RE = re.compile(r"<(https?://[^>]+)>", re.IGNORECASE)
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_MD_LIST_MARK_RE = re.compile(r"^\s{0,3}(?:[*+-]|\d+\.)\s+", re.MULTILINE)
_MD_HRULE_RE = re.compile(r"^\s{0,3}(?:[-*_]\s*){3,}$", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_BLOCK_TAG_RE = re.compile(
    r"</?(?:p|div|h[1-6]|li|section|article|br|tr|td|th|blockquote|ul|ol|hr|pre)[^>]*>",
    re.IGNORECASE,
)
_HTML_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>[\s\S]*?</\1>", re.IGNORECASE)
_HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->", re.MULTILINE)
_REPLACEMENT_CHAR_RE = re.compile(r"\uFFFD+")
_REPLACEMENT_CONTRACTION_RE = re.compile(
    r"(?<=[A-Za-z])\uFFFD+(?=(?:s|t|m|d|ll|re|ve)\b)",
    re.IGNORECASE,
)
_LEADING_BROKEN_IM_RE = re.compile(r"(?<!\w)\?['’]?m\b", re.IGNORECASE)
_BROKEN_APOSTROPHE_SPACING_RE = re.compile(r"\b([A-Za-z]+)[’']\s+(ll|re|ve|d|m|s|t)\b", re.IGNORECASE)
_BROKEN_YOURE_RE = re.compile(r"\bYoure\b")

_COMMON_MOJIBAKE_REPLACEMENTS = (
    ("隆炉", "'"),
    ("隆陋", "—"),
    ("鈥?", "—"),
    ("鈥", "—"),
)


@dataclass(frozen=True)
class NormalizeOptions:
    short_line_max_words: int = 3
    material_profile: str = "prose_article"


def _is_noise_line(line: str) -> bool:
    lowered = line.strip().lower()
    return bool(lowered) and any(marker in lowered for marker in _NOISE_LINES) and len(line.strip()) < 80


def _is_transcript_meta_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered in _TRANSCRIPT_HEADERS:
        return True
    if _SPEAKER_WITH_TIMESTAMP_RE.match(stripped):
        return True
    if _TIMESTAMP_HEADING_RE.match(stripped):
        return True
    return False


def _is_low_value_short_utterance(
    line: str,
    *,
    max_words: int,
) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(marker in stripped for marker in [":", "(", ")", "[", "]"]):
        return False
    words = _WORD_RE.findall(stripped)
    if not words:
        return False
    return len(words) <= max(1, max_words)


_INLINE_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_SPEAKER_PREFIX_RE = re.compile(r"^[A-Z][A-Za-z'.-]*(?:\s+[A-Z][A-Za-z'.-]*){0,6}\s*:\s*")
_FILLER_ONLY_RE = re.compile(
    r"^(?:uh+|um+|erm+|hmm+|yeah+|yep+|nope+|right+|okay+|ok+|mm+)\.?!?$",
    re.IGNORECASE,
)


def _normalize_transcript_line(line: str) -> str:
    value = line.strip()
    if not value:
        return ""
    value = _SPEAKER_PREFIX_RE.sub("", value)
    value = _INLINE_TIMESTAMP_RE.sub("", value)
    value = re.sub(r"\s{2,}", " ", value).strip(" -\t")
    return value.strip()


def _repair_common_encoding_noise(text: str) -> str:
    value = str(text or "")
    if not value:
        return ""

    for src, dst in _COMMON_MOJIBAKE_REPLACEMENTS:
        value = value.replace(src, dst)

    # Keep likely contractions/possessives, e.g. John��s -> John's.
    value = _REPLACEMENT_CONTRACTION_RE.sub("'", value)
    # Remaining replacement chars are treated as punctuation separators.
    value = _REPLACEMENT_CHAR_RE.sub(" — ", value)

    # Common OCR glitches in conversation material.
    value = _LEADING_BROKEN_IM_RE.sub("I'm", value)
    value = _BROKEN_APOSTROPHE_SPACING_RE.sub(r"\1'\2", value)
    value = _BROKEN_YOURE_RE.sub("You're", value)
    value = re.sub(r"\s*—\s*", " — ", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value


def strip_markdown_to_text(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    value = _repair_common_encoding_noise(value)
    value = _MD_FENCE_RE.sub("\n", value)
    value = _MD_IMAGE_RE.sub(r"\1", value)
    value = _MD_LINK_RE.sub(r"\1", value)
    value = _MD_AUTOLINK_RE.sub(r"\1", value)
    value = _MD_INLINE_CODE_RE.sub(r"\1", value)
    value = _MD_HEADING_RE.sub("", value)
    value = _MD_BLOCKQUOTE_RE.sub("", value)
    value = _MD_LIST_MARK_RE.sub("", value)
    value = _MD_HRULE_RE.sub("\n", value)
    value = value.replace("**", "").replace("__", "").replace("~~", "")
    value = re.sub(r"(?<!\w)[*_](?!\s)", "", value)
    value = _HTML_TAG_RE.sub("", value)
    value = unescape(value)
    return normalize_paragraph_text(value)


def strip_html_to_text(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    value = _repair_common_encoding_noise(value)
    value = _HTML_COMMENT_RE.sub("\n", value)
    value = _HTML_SCRIPT_STYLE_RE.sub("\n", value)
    value = _HTML_BLOCK_TAG_RE.sub("\n", value)
    value = _HTML_TAG_RE.sub("", value)
    value = unescape(value)
    return normalize_paragraph_text(value)


def normalize_text(text: str, *, options: NormalizeOptions | None = None) -> str:
    cfg = options or NormalizeOptions()
    text = _repair_common_encoding_noise(text)
    text = normalize_paragraph_text(text)
    raw_lines = [line.strip() for line in text.splitlines()]

    lines: list[str] = []
    profile = (cfg.material_profile or "prose_article").strip().lower()
    if profile == "general":
        profile = "prose_article"

    for line in raw_lines:
        if profile == "transcript_dialogue":
            line = _normalize_transcript_line(line)
            if not line:
                continue
            if _FILLER_ONLY_RE.match(line):
                continue
        if _is_noise_line(line):
            continue
        # Always remove obvious transcript/meta lines to keep only learnable prose.
        if _is_transcript_meta_line(line):
            continue
        if profile != "transcript_dialogue" and cfg.short_line_max_words > 0 and _is_low_value_short_utterance(
            line,
            max_words=cfg.short_line_max_words,
        ):
            continue
        lines.append(line)

    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized
