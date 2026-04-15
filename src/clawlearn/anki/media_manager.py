"""Media file helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MediaFile:
    filename: str
    path: Path


class MediaManager:
    def __init__(self, media_dir: Path, *, ext: str = "mp3") -> None:
        self._media_dir = media_dir
        self._ext = ext.lstrip(".")
        self._counter = 0

    def next_audio_file(self) -> MediaFile:
        self._counter += 1
        filename = f"audio_{self._counter:06d}.{self._ext}"
        return MediaFile(filename=filename, path=self._media_dir / filename)

    @staticmethod
    def to_anki_sound_field(filename: str) -> str:
        return f"[sound:{filename}]"

