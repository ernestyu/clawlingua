"""Voice selection strategies."""

from __future__ import annotations

import random


class UniformVoiceSelector:
    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(self, voices: list[str]) -> str:
        if not voices:
            raise ValueError("voices must not be empty")
        return self._rng.choice(voices)

