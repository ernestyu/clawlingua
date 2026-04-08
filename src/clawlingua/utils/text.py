"""Text helper utilities."""

from __future__ import annotations

import re

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def normalize_for_dedupe(text: str) -> str:
    lowered = text.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def count_sentences(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    parts = [p for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    return max(1, len(parts))


def split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [part.strip() for part in _SENT_SPLIT_RE.split(stripped) if part.strip()]


def normalize_paragraph_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()

