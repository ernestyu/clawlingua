"""edge-tts provider."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from pathlib import Path

from ..errors import build_error
from ..exit_codes import ExitCode

logger = logging.getLogger(__name__)


class EdgeTTSProvider:
    def __init__(
        self,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self._rate = rate
        self._volume = volume
        self._retry_attempts = max(0, int(retry_attempts))
        self._retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

    async def _synthesize_async(self, *, text: str, voice: str, output_path: Path) -> None:
        import edge_tts  # local import for optional dependency handling

        communicator = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=self._rate,
            volume=self._volume,
        )
        await communicator.save(str(output_path))

    def _synthesize_once(self, *, text: str, voice: str, output_path: Path) -> None:
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

    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        output_path: Path,
        lang: str | None = None,
    ) -> None:
        total_attempts = 1 + self._retry_attempts
        last_exc: Exception | None = None

        for attempt in range(1, total_attempts + 1):
            try:
                self._synthesize_once(text=text, voice=voice, output_path=output_path)
                return
            except Exception as exc:
                last_exc = exc
                if attempt >= total_attempts or not _is_retryable_tts_error(exc):
                    break
                delay = _retry_delay_seconds(
                    attempt=attempt,
                    base_seconds=self._retry_backoff_seconds,
                )
                logger.warning(
                    "edge-tts synth failed, retrying | attempt=%d/%d voice=%s delay=%.2fs err=%s",
                    attempt,
                    total_attempts,
                    voice,
                    delay,
                    exc,
                )
                time.sleep(delay)

        raise build_error(
            error_code="TTS_SYNTHESIZE_FAILED",
            cause="TTS synthesize failed.",
            detail=f"voice={voice}, attempts={total_attempts}, reason={last_exc}",
            next_steps=[
                "Check edge_tts upstream/network availability.",
                "Verify configured voice is valid.",
                "Increase CLAWLINGUA_TTS_RETRY_ATTEMPTS or backoff if needed.",
            ],
            exit_code=ExitCode.TTS_ERROR,
        ) from last_exc


def _iter_exception_chain(exc: Exception) -> list[Exception]:
    chain: list[Exception] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while isinstance(current, Exception):
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)
        chain.append(current)
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        current = current.__context__
    return chain


def _status_code_from_exception(exc: Exception) -> int | None:
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def _is_retryable_tts_error(exc: Exception) -> bool:
    for candidate in _iter_exception_chain(exc):
        status = _status_code_from_exception(candidate)
        if status is not None:
            if status in {408, 409, 425, 429} or status >= 500:
                return True
            return False
        if isinstance(candidate, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
            return True
        name = candidate.__class__.__name__.lower()
        if "timeout" in name or "disconnect" in name or "connection" in name:
            return True
    return False


def _retry_delay_seconds(*, attempt: int, base_seconds: float) -> float:
    if base_seconds <= 0:
        return 0.0
    delay = base_seconds * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, base_seconds)
    return min(30.0, delay + jitter)
