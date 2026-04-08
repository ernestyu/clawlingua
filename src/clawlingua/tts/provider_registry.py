"""TTS provider registry."""

from __future__ import annotations

from ..config import AppConfig
from ..errors import build_error
from ..exit_codes import ExitCode
from .base import BaseTTSProvider
from .edge_tts_provider import EdgeTTSProvider


def get_tts_provider(cfg: AppConfig) -> BaseTTSProvider:
    if cfg.tts_provider == "edge_tts":
        return EdgeTTSProvider(rate=cfg.tts_rate, volume=cfg.tts_volume)
    raise build_error(
        error_code="TTS_PROVIDER_UNSUPPORTED",
        cause="TTS provider 不支持。",
        detail=f"provider={cfg.tts_provider}",
        next_steps=["将 CLAWLINGUA_TTS_PROVIDER 设置为 edge_tts"],
        exit_code=ExitCode.CONFIG_ERROR,
    )

