"""Configuration loading and validation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, Field, field_validator

from .constants import (
    DEFAULT_ANKI_TEMPLATE,
    DEFAULT_DECK_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_CLOZE,
    DEFAULT_PROMPT_TRANSLATE,
)
from .errors import build_error
from .exit_codes import ExitCode

_VOICE_ENV_RE = re.compile(r"^CLAWLINGUA_TTS_EDGE_([A-Z0-9_]+)_VOICES$")


class AppConfig(BaseModel):
    workspace_root: Path = Field(default_factory=Path.cwd)

    default_source_lang: str = "en"
    default_target_lang: str = "zh"

    llm_provider: str = "openai_compatible"
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_seconds: int = 120
    llm_max_retries: int = 3
    llm_retry_backoff_seconds: float = 2.0
    llm_temperature: float = 0.2

    http_timeout_seconds: int = 30
    http_user_agent: str = "ClawLingua/0.1"
    http_verify_ssl: bool = True

    chunk_max_chars: int = 1800
    chunk_max_sentences: int = 8
    chunk_min_chars: int = 120
    chunk_overlap_sentences: int = 1

    prompt_cloze: Path = DEFAULT_PROMPT_CLOZE
    prompt_translate: Path = DEFAULT_PROMPT_TRANSLATE
    anki_template: Path = DEFAULT_ANKI_TEMPLATE

    output_dir: Path = DEFAULT_OUTPUT_DIR
    log_level: str = "INFO"
    save_intermediate: bool = True
    default_deck_name: str = DEFAULT_DECK_NAME

    tts_provider: str = "edge_tts"
    tts_output_format: str = "mp3"
    tts_rate: str = "+0%"
    tts_volume: str = "+0%"
    tts_random_seed: int | None = None

    tts_edge_voice_map: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator(
        "default_source_lang",
        "default_target_lang",
        "llm_provider",
        "log_level",
        "tts_provider",
        mode="before",
    )
    @classmethod
    def _strip_lower(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    def resolve_path(self, value: Path) -> Path:
        if value.is_absolute():
            return value
        return (self.workspace_root / value).resolve()

    def get_source_voices(self, source_lang: str) -> list[str]:
        lang = source_lang.lower()
        fallback = lang.split("-")[0]
        if lang in self.tts_edge_voice_map:
            return self.tts_edge_voice_map[lang]
        if fallback in self.tts_edge_voice_map:
            return self.tts_edge_voice_map[fallback]
        raise build_error(
            error_code="TTS_VOICE_NOT_CONFIGURED",
            cause="当前源语言没有可用的 TTS voice 配置。",
            detail=(
                f"source_lang={source_lang}，未找到对应 "
                f"CLAWLINGUA_TTS_EDGE_{fallback.upper()}_VOICES。"
            ),
            next_steps=[
                "在 .env 中为该语言配置至少 3 个 edge_tts voice",
                "或切换为其它已支持 voice 的源语言",
                "运行 clawlingua doctor 检查 TTS 配置",
            ],
            exit_code=ExitCode.CONFIG_ERROR,
        )

    def masked_dump(self) -> dict[str, Any]:
        data = self.model_dump(mode="python")
        api_key = data.get("llm_api_key") or ""
        if api_key:
            data["llm_api_key"] = _mask_secret(api_key)
        return data


def _mask_secret(secret: str) -> str:
    if len(secret) <= 6:
        return "***"
    return f"{secret[:3]}***{secret[-3:]}"


def _parse_voices(data: dict[str, Any]) -> dict[str, list[str]]:
    voice_map: dict[str, list[str]] = {}
    for key, value in data.items():
        match = _VOICE_ENV_RE.match(key)
        if not match or value is None:
            continue
        lang = match.group(1).lower().replace("_", "-")
        voices = [item.strip() for item in str(value).split(",") if item.strip()]
        if voices:
            voice_map[lang] = voices
    return voice_map


def _normalize_env_aliases(data: dict[str, Any]) -> None:
    alias_pairs = [
        ("CLAWLINGUA_DEFAULT_CLOZE_PROMPT", "CLAWLINGUA_PROMPT_CLOZE"),
        ("CLAWLINGUA_DEFAULT_TRANSLATE_PROMPT", "CLAWLINGUA_PROMPT_TRANSLATE"),
        ("CLAWLINGUA_DEFAULT_ANKI_TEMPLATE", "CLAWLINGUA_ANKI_TEMPLATE"),
    ]
    for old_key, new_key in alias_pairs:
        if old_key in data and new_key not in data:
            data[new_key] = data[old_key]


def _env_value(value: Any, fallback: Any) -> Any:
    return fallback if value is None or value == "" else value


def load_config(
    *,
    env_file: Path | None = None,
    overrides: dict[str, Any] | None = None,
    workspace_root: Path | None = None,
) -> AppConfig:
    root = (workspace_root or Path.cwd()).resolve()
    effective_env_file = env_file or (root / ".env")
    merged: dict[str, Any] = {}

    if effective_env_file.exists():
        for k, v in dotenv_values(effective_env_file).items():
            if v is not None:
                merged[k] = v

    for k, v in os.environ.items():
        if k.startswith("CLAWLINGUA_"):
            merged[k] = v

    _normalize_env_aliases(merged)
    voice_map = _parse_voices(merged)

    payload: dict[str, Any] = {
        "workspace_root": root,
        "default_source_lang": _env_value(
            merged.get("CLAWLINGUA_DEFAULT_SOURCE_LANG"), "en"
        ),
        "default_target_lang": _env_value(
            merged.get("CLAWLINGUA_DEFAULT_TARGET_LANG"), "zh"
        ),
        "llm_provider": _env_value(merged.get("CLAWLINGUA_LLM_PROVIDER"), "openai_compatible"),
        "llm_base_url": _env_value(merged.get("CLAWLINGUA_LLM_BASE_URL"), "http://127.0.0.1:8000/v1"),
        "llm_api_key": _env_value(merged.get("CLAWLINGUA_LLM_API_KEY"), ""),
        "llm_model": _env_value(merged.get("CLAWLINGUA_LLM_MODEL"), ""),
        "llm_timeout_seconds": _env_value(merged.get("CLAWLINGUA_LLM_TIMEOUT_SECONDS"), 120),
        "llm_max_retries": _env_value(merged.get("CLAWLINGUA_LLM_MAX_RETRIES"), 3),
        "llm_retry_backoff_seconds": _env_value(
            merged.get("CLAWLINGUA_LLM_RETRY_BACKOFF_SECONDS"), 2.0
        ),
        "llm_temperature": _env_value(merged.get("CLAWLINGUA_LLM_TEMPERATURE"), 0.2),
        "http_timeout_seconds": _env_value(merged.get("CLAWLINGUA_HTTP_TIMEOUT_SECONDS"), 30),
        "http_user_agent": _env_value(merged.get("CLAWLINGUA_HTTP_USER_AGENT"), "ClawLingua/0.1"),
        "http_verify_ssl": _env_value(merged.get("CLAWLINGUA_HTTP_VERIFY_SSL"), True),
        "chunk_max_chars": _env_value(merged.get("CLAWLINGUA_CHUNK_MAX_CHARS"), 1800),
        "chunk_max_sentences": _env_value(merged.get("CLAWLINGUA_CHUNK_MAX_SENTENCES"), 8),
        "chunk_min_chars": _env_value(merged.get("CLAWLINGUA_CHUNK_MIN_CHARS"), 120),
        "chunk_overlap_sentences": _env_value(
            merged.get("CLAWLINGUA_CHUNK_OVERLAP_SENTENCES"), 1
        ),
        "prompt_cloze": _env_value(merged.get("CLAWLINGUA_PROMPT_CLOZE"), DEFAULT_PROMPT_CLOZE),
        "prompt_translate": _env_value(
            merged.get("CLAWLINGUA_PROMPT_TRANSLATE"), DEFAULT_PROMPT_TRANSLATE
        ),
        "anki_template": _env_value(merged.get("CLAWLINGUA_ANKI_TEMPLATE"), DEFAULT_ANKI_TEMPLATE),
        "output_dir": _env_value(merged.get("CLAWLINGUA_OUTPUT_DIR"), DEFAULT_OUTPUT_DIR),
        "log_level": _env_value(merged.get("CLAWLINGUA_LOG_LEVEL"), "INFO"),
        "save_intermediate": _env_value(merged.get("CLAWLINGUA_SAVE_INTERMEDIATE"), True),
        "default_deck_name": _env_value(
            merged.get("CLAWLINGUA_DEFAULT_DECK_NAME"), DEFAULT_DECK_NAME
        ),
        "tts_provider": _env_value(merged.get("CLAWLINGUA_TTS_PROVIDER"), "edge_tts"),
        "tts_output_format": _env_value(
            merged.get("CLAWLINGUA_TTS_OUTPUT_FORMAT"), "mp3"
        ),
        "tts_rate": _env_value(merged.get("CLAWLINGUA_TTS_RATE"), "+0%"),
        "tts_volume": _env_value(merged.get("CLAWLINGUA_TTS_VOLUME"), "+0%"),
        "tts_random_seed": _env_value(merged.get("CLAWLINGUA_TTS_RANDOM_SEED"), None),
        "tts_edge_voice_map": voice_map,
    }

    if overrides:
        payload.update({k: v for k, v in overrides.items() if v is not None})

    return AppConfig.model_validate(payload)


def validate_base_config(cfg: AppConfig) -> None:
    required_paths = [
        ("CLAWLINGUA_PROMPT_CLOZE", cfg.resolve_path(cfg.prompt_cloze)),
        ("CLAWLINGUA_PROMPT_TRANSLATE", cfg.resolve_path(cfg.prompt_translate)),
        ("CLAWLINGUA_ANKI_TEMPLATE", cfg.resolve_path(cfg.anki_template)),
    ]
    for key, path in required_paths:
        if not path.exists():
            raise build_error(
                error_code="CONFIG_PATH_NOT_FOUND",
                cause="关键配置文件不存在。",
                detail=f"{key} -> {path}",
                next_steps=[
                    "检查 .env 配置路径",
                    "运行 clawlingua init 生成默认配置文件",
                ],
                exit_code=ExitCode.CONFIG_ERROR,
            )


def validate_runtime_config(cfg: AppConfig) -> None:
    if cfg.llm_provider == "openai_compatible":
        if not cfg.llm_api_key:
            raise build_error(
                error_code="CONFIG_MISSING_API_KEY",
                cause="LLM API key 缺失。",
                detail="未在 .env 或命令行参数中找到 CLAWLINGUA_LLM_API_KEY。",
                next_steps=[
                    "在 .env 中添加 CLAWLINGUA_LLM_API_KEY",
                    "或通过 --env-file 指定正确的环境文件",
                    "运行 clawlingua config validate 检查配置",
                ],
                exit_code=ExitCode.CONFIG_ERROR,
            )
        if not cfg.llm_model:
            raise build_error(
                error_code="CONFIG_MISSING_MODEL",
                cause="LLM model 缺失。",
                detail="未在配置中找到 CLAWLINGUA_LLM_MODEL。",
                next_steps=["在 .env 中添加 CLAWLINGUA_LLM_MODEL"],
                exit_code=ExitCode.CONFIG_ERROR,
            )

