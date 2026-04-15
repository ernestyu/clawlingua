"""TTS provider protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class BaseTTSProvider(Protocol):
    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        output_path: Path,
        lang: str | None = None,
    ) -> None:
        ...

