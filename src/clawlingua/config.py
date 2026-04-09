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
    DEFAULT_EXPORT_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_CLOZE,
    DEFAULT_PROMPT_CLOZE_TEXTBOOK,
    DEFAULT_PROMPT_TRANSLATE,
    SUPPORTED_CONTENT_PROFILES,
)
from .errors import build_error
from .exit_codes import ExitCode

_VOICE_ENV_RE = re.compile(r"^CLAWLINGUA_TTS_EDGE_([A-Z0-9_]+)_VOICES$")
_VOICE_SLOT_ENV_KEYS = (
    "CLAWLINGUA_TTS_EDGE_VOICE1",
    "CLAWLINGUA_TTS_EDGE_VOICE2",
    "CLAWLINGUA_TTS_EDGE_VOICE3",
    "CLAWLINGUA_TTS_EDGE_VOICE4",
)


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
    # Base sleep between successful LLM calls (seconds); actual sleep is randomized
    # within [llm_request_sleep_seconds, 3*llm_request_sleep_seconds].
    llm_request_sleep_seconds: float = 0.0
    llm_temperature: float = 0.2

    # 0 disables short-line filtering.
    ingest_short_line_max_words: int = Field(default=3, ge=0)

    chunk_max_chars: int = 1800
    # chunk_max_sentences was removed; chunking 閻滈浜掔€涙顑侀弫棰佽礋娑撲紮绱濋悽鍗炲綖鐎涙劘绔熼悾宀冭拫閸掑浄绱?    # Do not use per-chunk sentence count caps; chunking is char-count-first.
    chunk_min_chars: int = 120
    chunk_overlap_sentences: int = 1

    # Cloze-level controls for per-card validation.
    cloze_max_sentences: int = 3
    # Minimum chars per cloze text; 0 means no lower bound.
    cloze_min_chars: int = 0
    cloze_difficulty: str = "intermediate"  # beginner|intermediate|advanced
    cloze_max_per_chunk: int | None = None
    # LLM chunk-level batch size; 1 means per-chunk requests.
    llm_chunk_batch_size: int = 1

    # Prompt language: en|zh. Used when prompts provide multi-lingual
    # variants (e.g. system_prompt/user_prompt_template as {"en": ..., "zh": ...}).
    prompt_lang: str = "zh"

    # Translation LLM (small LLM) settings; fallback to main LLM when empty.
    translate_llm_base_url: str | None = None
    translate_llm_api_key: str | None = None
    translate_llm_model: str | None = None
    translate_llm_temperature: float | None = None

    prompt_cloze: Path = DEFAULT_PROMPT_CLOZE
    prompt_cloze_textbook: Path = DEFAULT_PROMPT_CLOZE_TEXTBOOK
    prompt_translate: Path = DEFAULT_PROMPT_TRANSLATE
    anki_template: Path = DEFAULT_ANKI_TEMPLATE
    content_profile: str = "general"

    output_dir: Path = DEFAULT_OUTPUT_DIR
    export_dir: Path = DEFAULT_EXPORT_DIR
    log_dir: Path = DEFAULT_LOG_DIR
    log_level: str = "INFO"
    save_intermediate: bool = True
    default_deck_name: str = DEFAULT_DECK_NAME

    tts_provider: str = "edge_tts"
    tts_output_format: str = "mp3"
    tts_rate: str = "+0%"
    tts_volume: str = "+0%"
    tts_random_seed: int | None = None

    tts_edge_voices: list[str] = Field(default_factory=list)
    tts_edge_voice_map: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator(
        "default_source_lang",
        "default_target_lang",
        "llm_provider",
        "log_level",
        "tts_provider",
        "cloze_difficulty",
        "content_profile",
        mode="before",
    )
    @classmethod
    def _strip_lower(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("cloze_difficulty")
    @classmethod
    def _validate_cloze_difficulty(cls, v: str) -> str:
        value = (v or "intermediate").strip().lower()
        if value not in {"beginner", "intermediate", "advanced"}:
            raise ValueError("cloze_difficulty must be beginner|intermediate|advanced")
        return value

    @field_validator("content_profile")
    @classmethod
    def _validate_content_profile(cls, v: str) -> str:
        value = (v or "general").strip().lower()
        if value not in SUPPORTED_CONTENT_PROFILES:
            allowed = ", ".join(sorted(SUPPORTED_CONTENT_PROFILES))
            raise ValueError(f"content_profile must be one of: {allowed}")
        return value

    def resolve_path(self, value: Path) -> Path:
        if value.is_absolute():
            return value
        return (self.workspace_root / value).resolve()

    def get_source_voices(self, source_lang: str) -> list[str]:
        if self.tts_edge_voices:
            return list(self.tts_edge_voices)
        lang = source_lang.lower()
        fallback = lang.split("-")[0]
        if lang in self.tts_edge_voice_map:
            return list(self.tts_edge_voice_map[lang])
        if fallback in self.tts_edge_voice_map:
            return list(self.tts_edge_voice_map[fallback])
        return []

    def masked_dump(self) -> dict[str, Any]:
        data = self.model_dump(mode="python")
        api_key = data.get("llm_api_key") or ""
        if api_key:
            data["llm_api_key"] = _mask_secret(api_key)
        translate_api_key = data.get("translate_llm_api_key") or ""
        if translate_api_key:
            data["translate_llm_api_key"] = _mask_secret(translate_api_key)
        return data


def _mask_secret(secret: str) -> str:
    if len(secret) <= 6:
        return "***"
    return f"{secret[:3]}***{secret[-3:]}"


def _parse_voice_slots(data: dict[str, Any]) -> list[str]:
    voices: list[str] = []
    seen: set[str] = set()
    for key in _VOICE_SLOT_ENV_KEYS:
        value = data.get(key)
        if value is None:
            continue
        for item in str(value).split(","):
            voice = item.strip()
            if not voice or voice in seen:
                continue
            voices.append(voice)
            seen.add(voice)
    return voices


def _parse_legacy_voice_map(data: dict[str, Any]) -> dict[str, list[str]]:
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
        ("CLAWLINGUA_INGEST_SHORT_UTTERANCE_MAX_WORDS", "CLAWLINGUA_INGEST_SHORT_LINE_MAX_WORDS"),
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
    voice_slots = _parse_voice_slots(merged)
    voice_map = _parse_legacy_voice_map(merged)

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
        "llm_request_sleep_seconds": _env_value(
            merged.get("CLAWLINGUA_LLM_REQUEST_SLEEP_SECONDS"), 0.0
        ),
        "llm_temperature": _env_value(merged.get("CLAWLINGUA_LLM_TEMPERATURE"), 0.2),
        "ingest_short_line_max_words": _env_value(
            merged.get("CLAWLINGUA_INGEST_SHORT_LINE_MAX_WORDS"), 3
        ),
        "chunk_max_chars": _env_value(merged.get("CLAWLINGUA_CHUNK_MAX_CHARS"), 1800),
        "chunk_min_chars": _env_value(merged.get("CLAWLINGUA_CHUNK_MIN_CHARS"), 120),
        "chunk_overlap_sentences": _env_value(
            merged.get("CLAWLINGUA_CHUNK_OVERLAP_SENTENCES"), 1
        ),
        "cloze_max_sentences": _env_value(
            merged.get("CLAWLINGUA_CLOZE_MAX_SENTENCES"), 3
        ),
        "cloze_min_chars": _env_value(
            merged.get("CLAWLINGUA_CLOZE_MIN_CHARS"), 0
        ),
        "cloze_difficulty": _env_value(
            merged.get("CLAWLINGUA_CLOZE_DIFFICULTY"), "intermediate"
        ),
        "cloze_max_per_chunk": _env_value(
            merged.get("CLAWLINGUA_CLOZE_MAX_PER_CHUNK"), None
        ),
        "llm_chunk_batch_size": _env_value(
            merged.get("CLAWLINGUA_LLM_CHUNK_BATCH_SIZE"), 1
        ),
        "prompt_lang": _env_value(
            merged.get("CLAWLINGUA_PROMPT_LANG"), "zh"
        ),
        "translate_llm_base_url": _env_value(            merged.get("CLAWLINGUA_TRANSLATE_LLM_BASE_URL"), None
        ),
        "translate_llm_api_key": _env_value(
            merged.get("CLAWLINGUA_TRANSLATE_LLM_API_KEY"), None
        ),
        "translate_llm_model": _env_value(
            merged.get("CLAWLINGUA_TRANSLATE_LLM_MODEL"), None
        ),
        "translate_llm_temperature": _env_value(
            merged.get("CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE"), None
        ),
        "prompt_cloze": _env_value(merged.get("CLAWLINGUA_PROMPT_CLOZE"), DEFAULT_PROMPT_CLOZE),
        "prompt_cloze_textbook": _env_value(
            merged.get("CLAWLINGUA_PROMPT_CLOZE_TEXTBOOK"),
            DEFAULT_PROMPT_CLOZE_TEXTBOOK,
        ),
        "prompt_translate": _env_value(
            merged.get("CLAWLINGUA_PROMPT_TRANSLATE"), DEFAULT_PROMPT_TRANSLATE
        ),
        "anki_template": _env_value(merged.get("CLAWLINGUA_ANKI_TEMPLATE"), DEFAULT_ANKI_TEMPLATE),
        "content_profile": _env_value(
            merged.get("CLAWLINGUA_CONTENT_PROFILE"), "general"
        ),
        "output_dir": _env_value(merged.get("CLAWLINGUA_OUTPUT_DIR"), DEFAULT_OUTPUT_DIR),
        "export_dir": _env_value(merged.get("CLAWLINGUA_EXPORT_DIR"), DEFAULT_EXPORT_DIR),
        "log_dir": _env_value(merged.get("CLAWLINGUA_LOG_DIR"), DEFAULT_LOG_DIR),
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
        "tts_edge_voices": voice_slots,
        "tts_edge_voice_map": voice_map,
    }

    if overrides:
        payload.update({k: v for k, v in overrides.items() if v is not None})

    return AppConfig.model_validate(payload)


def validate_base_config(cfg: AppConfig) -> None:
    required_paths = [
        ("CLAWLINGUA_PROMPT_CLOZE", cfg.resolve_path(cfg.prompt_cloze)),
        ("CLAWLINGUA_PROMPT_CLOZE_TEXTBOOK", cfg.resolve_path(cfg.prompt_cloze_textbook)),
        ("CLAWLINGUA_PROMPT_TRANSLATE", cfg.resolve_path(cfg.prompt_translate)),
        ("CLAWLINGUA_ANKI_TEMPLATE", cfg.resolve_path(cfg.anki_template)),
    ]
    for key, path in required_paths:
        if not path.exists():
            raise build_error(
                error_code="CONFIG_PATH_NOT_FOUND",
                cause="Required config file is missing.",
                detail=f"{key} -> {path}",
                next_steps=[
                    "Check the path values in .env",
                    "Run `clawlingua init` to generate default config files",
                ],
                exit_code=ExitCode.CONFIG_ERROR,
            )


def validate_runtime_config(cfg: AppConfig) -> None:
    if cfg.llm_provider == "openai_compatible":
        if not cfg.llm_api_key:
            raise build_error(
                error_code="CONFIG_MISSING_API_KEY",
                cause="LLM API key is missing.",
                detail="`CLAWLINGUA_LLM_API_KEY` was not found in .env or environment variables.",
                next_steps=[
                    "Add `CLAWLINGUA_LLM_API_KEY` to .env",
                    "Or point to the correct env file via `--env-file`",
                    "Run `clawlingua config validate` to check config",
                ],
                exit_code=ExitCode.CONFIG_ERROR,
            )
        if not cfg.llm_model:
            raise build_error(
                error_code="CONFIG_MISSING_MODEL",
                cause="LLM model is missing.",
                detail="`CLAWLINGUA_LLM_MODEL` was not found in configuration.",
                next_steps=["Add `CLAWLINGUA_LLM_MODEL` to .env"],
                exit_code=ExitCode.CONFIG_ERROR,
            )
