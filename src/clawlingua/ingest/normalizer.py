"""Text normalization."""

from __future__ import annotations

import re

from ..utils.text import normalize_paragraph_text

_NOISE_LINES = (
    "cookie",
    "privacy policy",
    "terms of service",
    "subscribe",
    "sign in",
)


def normalize_text(text: str) -> str:
    text = normalize_paragraph_text(text)
    lines = []
    for line in text.splitlines():
        lowered = line.strip().lower()
        if any(marker in lowered for marker in _NOISE_LINES) and len(line.strip()) < 80:
            continue
        lines.append(line.strip())

    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized

