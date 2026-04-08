"""edge-tts provider."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ..errors import build_error
from ..exit_codes import ExitCode


class EdgeTTSProvider:
    def __init__(self, *, rate: str = "+0%", volume: str = "+0%") -> None:
        self._rate = rate
        self._volume = volume

    async def _synthesize_async(self, *, text: str, voice: str, output_path: Path) -> None:
        import edge_tts  # local import for optional dependency handling

        communicator = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=self._rate,
            volume=self._volume,
        )
        await communicator.save(str(output_path))

    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        output_path: Path,
        lang: str | None = None,
    ) -> None:
        try:
            asyncio.run(self._synthesize_async(text=text, voice=voice, output_path=output_path))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    self._synthesize_async(text=text, voice=voice, output_path=output_path)
                )
            finally:
                loop.close()
        except Exception as exc:
            raise build_error(
                error_code="TTS_SYNTHESIZE_FAILED",
                cause="TTS 合成失败。",
                detail=f"voice={voice}, reason={exc}",
                next_steps=["检查 edge_tts 是否可用", "确认文本内容与 voice 配置"],
                exit_code=ExitCode.TTS_ERROR,
            ) from exc

