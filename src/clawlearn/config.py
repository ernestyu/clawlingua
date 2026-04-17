"""Configuration loading and validation."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, Field, ValidationError, field_validator

from .constants import (
    DEFAULT_ANKI_TEMPLATE,
    DEFAULT_DECK_NAME,
    DEFAULT_EXPORT_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_CONTENT_PROFILES,
    SUPPORTED_LEARNING_MODES,
    SUPPORTED_MATERIAL_PROFILES,
)
from .errors import build_error
from .exit_codes import ExitCode
from .models.prompt_schema import PromptSpec

_VOICE_ENV_RE = re.compile(r"^CLAWLEARN_TTS_EDGE_([A-Z0-9_]+)_VOICES$")
_VOICE_SLOT_ENV_KEYS = (
    "CLAWLEARN_TTS_EDGE_VOICE1",
    "CLAWLEARN_TTS_EDGE_VOICE2",
    "CLAWLEARN_TTS_EDGE_VOICE3",
    "CLAWLEARN_TTS_EDGE_VOICE4",
)
_PROMPT_META_FILENAMES = {"user_prompt_overrides.json"}
_PROMPT_TEMPLATE_FILENAMES = {"template_extraction.json", "template_explanation.json"}
_PROMPTS_DIR = Path("./prompts")
_AUTO_SEED_FILE_BY_MODE = {
    "extraction": "auto_seed_extraction.json",
    "explanation": "auto_seed_explanation.json",
}
_SEED_SOURCE_FILENAMES_BY_MODE = {
    "extraction": ("cloze_contextual_example.json", "template_extraction.json"),
    "explanation": ("translate_rewrite.json", "template_explanation.json"),
}

logger = logging.getLogger(__name__)


def _normalize_profile(value: str | None) -> str:
    profile = (value or "").strip().lower()
    if not profile:
        return "prose_article"
    if profile == "general":
        return "prose_article"
    return profile


def _normalize_prompt_mode(value: str | None) -> str:
    mode = (value or "").strip().lower()
    if mode == "cloze":
        return "extraction"
    if mode == "translate":
        return "explanation"
    if mode in {"extraction", "explanation"}:
        return mode
    return ""


def _normalize_prompt_content_type(value: str | None) -> str:
    content_type = (value or "").strip().lower()
    if content_type in {"general", "prose", "article"}:
        return "prose_article"
    if content_type in {"transcript", "dialogue"}:
        return "transcript_dialogue"
    if content_type in {"textbook", "example"}:
        return "textbook_examples"
    if content_type in {"prose_article", "transcript_dialogue", "textbook_examples"}:
        return content_type
    return "all"


def _normalize_prompt_learning_mode(value: str | None) -> str:
    learning_mode = (value or "").strip().lower()
    if learning_mode in SUPPORTED_LEARNING_MODES:
        return learning_mode
    return "all"


def _normalize_prompt_difficulty(value: str | None) -> str:
    difficulty = (value or "").strip().lower()
    if difficulty in {"beginner", "intermediate", "advanced"}:
        return difficulty
    return "all"


def _prompt_content_type_for_profile(material_profile: str | None) -> str:
    profile = _normalize_profile(material_profile)
    if profile == "transcript_dialogue":
        return "transcript_dialogue"
    if profile == "textbook_examples":
        return "textbook_examples"
    return "prose_article"


def _score_prompt_field(requested: str, actual: str) -> int | None:
    if requested == "all":
        return 0
    if actual == requested:
        return 3
    if actual == "all":
        return 1
    return None


def _embedded_seed_prompt_payload(mode: str) -> dict[str, Any]:
    normalized_mode = _normalize_prompt_mode(mode) or "extraction"
    if normalized_mode == "explanation":
        return {
            "name": "auto_seed_explanation",
            "version": "2026-04-13",
            "description": "Auto-seeded fallback explanation prompt.",
            "mode": "explanation",
            "content_type": "all",
            "learning_mode": "all",
            "difficulty_level": "all",
            "target_langs_supported": "all",
            "source_langs_suggested": "all",
            "system_prompt": (
                "You are a translation assistant. "
                "Return a JSON array only with objects: {\"translation\": \"...\"}."
            ),
            "user_prompt_template": (
                "source_lang={source_lang}\n"
                "target_lang={target_lang}\n"
                "document_title={document_title}\n"
                "source_url={source_url}\n"
                "batch_size={batch_size}\n"
                "text_originals_json={text_originals_json}\n\n"
                "Return a JSON array with exactly {batch_size} items in the same order. "
                "Each item must be an object with key \"translation\" only."
            ),
            "placeholders": [
                "source_lang",
                "target_lang",
                "document_title",
                "source_url",
                "chunk_text",
                "text_original",
                "text_originals_json",
                "batch_size",
            ],
            "output_format": {"type": "json", "schema_name": "translation_batch_v1"},
            "parser": {"strip_code_fences": True, "expect_json_array": True},
        }
    return {
        "name": "auto_seed_extraction",
        "version": "2026-04-13",
        "description": "Auto-seeded fallback extraction prompt.",
        "mode": "extraction",
        "content_type": "all",
        "learning_mode": "all",
        "difficulty_level": "all",
        "target_langs_supported": "all",
        "source_langs_suggested": "all",
        "system_prompt": (
            "You are a language-learning content editor. "
            "Output a JSON array only. No markdown, no explanations."
        ),
        "user_prompt_template": (
            "source_lang={source_lang}\n"
            "target_lang={target_lang}\n"
            "document_title={document_title}\n"
            "source_url={source_url}\n"
            "learning_mode={learning_mode}\n"
            "difficulty={difficulty}\n"
            "cloze_max_sentences={cloze_max_sentences}\n"
            "cloze_min_chars={cloze_min_chars}\n"
            "cloze_max_per_chunk={cloze_max_per_chunk}\n\n"
            "Return a JSON array. Each item must contain:\n"
            "{\"chunk_id\":\"...\",\"text\":\"...\",\"original\":\"...\","
            "\"target_phrases\":[\"...\"],\"note_hint\":\"...\"}\n\n"
            "Requirements:\n"
            "1) text must include at least one cloze in format {{cN::<b>...</b>}}(hint).\n"
            "2) original must not contain cloze markers or HTML tags.\n"
            "3) Respect cloze_max_sentences and cloze_min_chars.\n"
            "4) If no suitable candidate, return an empty JSON array.\n\n"
            "chunk_text:\n{chunk_text}"
        ),
        "placeholders": [
            "source_lang",
            "target_lang",
            "document_title",
            "source_url",
            "chunk_text",
            "learning_mode",
            "difficulty",
            "cloze_max_sentences",
            "cloze_min_chars",
            "cloze_max_per_chunk",
        ],
        "output_format": {"type": "json", "schema_name": "cloze_cards_v1"},
        "parser": {"strip_code_fences": True, "expect_json_array": True},
    }


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
    # Do not use per-chunk sentence count caps; chunking is char-count-first.
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
    # Optional second-pass extraction model settings.
    secondary_extract_enable: bool = False
    secondary_extract_parallel: bool = False
    secondary_extract_llm_base_url: str | None = None
    secondary_extract_llm_api_key: str | None = None
    secondary_extract_llm_model: str | None = None
    secondary_extract_llm_timeout_seconds: int | None = Field(default=None, ge=1)
    secondary_extract_llm_temperature: float | None = None
    secondary_extract_llm_max_retries: int | None = Field(default=None, ge=1)
    secondary_extract_llm_retry_backoff_seconds: float | None = Field(default=None, ge=0.0)
    secondary_extract_llm_chunk_batch_size: int | None = Field(default=None, ge=1)
    # Format-only validation retry controls.
    validate_format_retry_enable: bool = True
    # Retry attempts after initial validation failure.
    validate_format_retry_max: int = Field(default=3, ge=0, le=3)
    # Whether attempts >=2 are allowed to call LLM repair/regenerate.
    validate_format_retry_llm_enable: bool = True
    # Whether taxonomy rejects are sent to a small-model repair pass.
    taxonomy_repair_enable: bool = False
    # Lingua transcript context fallback minimum sentence count.
    lingua_transcript_min_context_sentences: int = Field(default=2, ge=1, le=3)
    # Whether to run Stage-2 taxonomy/reason annotation for final lingua transcript candidates.
    lingua_annotate_enable: bool = False
    # Stage-2 annotation batch size.
    lingua_annotate_batch_size: int = Field(default=50, ge=1)
    # Optional max items for Stage-2 annotation per run.
    lingua_annotate_max_items: int | None = Field(default=None, ge=1)

    # Prompt language: en|zh.
    prompt_lang: str = "zh"
    # Prompt override paths (default layer, lower priority than CLI args).
    extract_prompt: Path | None = None
    explain_prompt: Path | None = None

    # Translation LLM (small LLM) settings; fallback to main LLM when empty.
    translate_llm_base_url: str | None = None
    translate_llm_api_key: str | None = None
    translate_llm_model: str | None = None
    translate_llm_temperature: float | None = None
    # Number of originals translated in one LLM request.
    translate_batch_size: int = 4

    # Legacy prompt path fields (env overrides); optional and no hard-coded defaults.
    prompt_cloze: Path | None = None
    prompt_cloze_textbook: Path | None = None
    prompt_cloze_prose_beginner: Path | None = None
    prompt_cloze_prose_intermediate: Path | None = None
    prompt_cloze_prose_advanced: Path | None = None
    prompt_cloze_prose_reading_support_beginner: Path | None = None
    prompt_cloze_prose_reading_support_intermediate: Path | None = None
    prompt_cloze_prose_reading_support_advanced: Path | None = None
    prompt_cloze_transcript_beginner: Path | None = None
    prompt_cloze_transcript_intermediate: Path | None = None
    prompt_cloze_transcript_advanced: Path | None = None
    prompt_cloze_transcript_reading_support_beginner: Path | None = None
    prompt_cloze_transcript_reading_support_intermediate: Path | None = None
    prompt_cloze_transcript_reading_support_advanced: Path | None = None

    prompt_translate: Path | None = None
    anki_template: Path = DEFAULT_ANKI_TEMPLATE

    # Legacy field. Kept for backward compatibility.
    content_profile: str = "general"
    # New V2 fields.
    material_profile: str = "prose_article"
    learning_mode: str = "lingua_expression"

    # Allow export of empty deck (0 cards) instead of hard-failing the run.
    allow_empty_deck: bool = True

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
    # Retries after the first failed TTS request (e.g., transient 503/network jitter).
    tts_retry_attempts: int = Field(default=3, ge=0, le=10)
    # Base seconds for exponential backoff between TTS retries.
    tts_retry_backoff_seconds: float = Field(default=1.0, ge=0.0)

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
        "material_profile",
        "learning_mode",
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

    @field_validator("material_profile")
    @classmethod
    def _validate_material_profile(cls, v: str) -> str:
        value = _normalize_profile(v)
        if value not in SUPPORTED_MATERIAL_PROFILES:
            allowed = ", ".join(sorted(SUPPORTED_MATERIAL_PROFILES))
            raise ValueError(f"material_profile must be one of: {allowed}")
        return value

    @field_validator("learning_mode")
    @classmethod
    def _validate_learning_mode(cls, v: str) -> str:
        value = (v or "lingua_expression").strip().lower()
        if value not in SUPPORTED_LEARNING_MODES:
            allowed = ", ".join(sorted(SUPPORTED_LEARNING_MODES))
            raise ValueError(f"learning_mode must be one of: {allowed}")
        return value

    def resolve_path(self, value: Path) -> Path:
        if value.is_absolute():
            return value
        return (self.workspace_root / value).resolve()

    def _resolve_legacy_extract_prompt_path(
        self,
        *,
        material_profile: str | None = None,
        difficulty: str | None = None,
        learning_mode: str | None = None,
    ) -> Path | None:
        profile = _normalize_profile(material_profile or self.material_profile)
        diff = _normalize_prompt_difficulty(difficulty or self.cloze_difficulty)
        mode = _normalize_prompt_learning_mode(learning_mode or self.learning_mode)
        if diff == "all":
            diff = "intermediate"
        if mode == "all":
            mode = "lingua_expression"

        if profile == "textbook_examples":
            return self.prompt_cloze_textbook or self.prompt_cloze

        if mode == "lingua_reading":
            if profile == "transcript_dialogue":
                return {
                    "beginner": self.prompt_cloze_transcript_reading_support_beginner,
                    "intermediate": self.prompt_cloze_transcript_reading_support_intermediate,
                    "advanced": self.prompt_cloze_transcript_reading_support_advanced,
                }.get(diff)
            if profile == "prose_article":
                return {
                    "beginner": self.prompt_cloze_prose_reading_support_beginner,
                    "intermediate": self.prompt_cloze_prose_reading_support_intermediate,
                    "advanced": self.prompt_cloze_prose_reading_support_advanced,
                }.get(diff)
            return self.prompt_cloze

        if profile == "transcript_dialogue":
            return {
                "beginner": self.prompt_cloze_transcript_beginner,
                "intermediate": self.prompt_cloze_transcript_intermediate,
                "advanced": self.prompt_cloze_transcript_advanced,
            }.get(diff)
        if profile == "prose_article":
            return {
                "beginner": self.prompt_cloze_prose_beginner,
                "intermediate": self.prompt_cloze_prose_intermediate,
                "advanced": self.prompt_cloze_prose_advanced,
            }.get(diff)
        return self.prompt_cloze

    def _scan_prompt_files(
        self,
        *,
        mode: str,
        content_type: str = "all",
        learning_mode: str = "all",
        difficulty: str = "all",
        include_templates: bool = False,
    ) -> list[Path]:
        normalized_mode = _normalize_prompt_mode(mode)
        if normalized_mode not in {"extraction", "explanation"}:
            return []
        prompts_dir = self.resolve_path(_PROMPTS_DIR)
        if not prompts_dir.exists():
            return []

        requested_content = _normalize_prompt_content_type(content_type)
        requested_learning = _normalize_prompt_learning_mode(learning_mode)
        requested_difficulty = _normalize_prompt_difficulty(difficulty)
        scored: list[tuple[int, str, Path]] = []

        for path in sorted(prompts_dir.glob("*.json")):
            if path.name in _PROMPT_META_FILENAMES:
                continue
            if not include_templates and path.name in _PROMPT_TEMPLATE_FILENAMES:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                spec = PromptSpec.model_validate(payload)
            except (OSError, json.JSONDecodeError, ValidationError, ValueError):
                continue

            spec_mode = _normalize_prompt_mode(str(spec.mode))
            if spec_mode != normalized_mode:
                continue
            spec_content = _normalize_prompt_content_type(str(spec.content_type))
            spec_learning = _normalize_prompt_learning_mode(str(spec.learning_mode))
            spec_difficulty = _normalize_prompt_difficulty(str(spec.difficulty_level))

            score_parts = (
                _score_prompt_field(requested_content, spec_content),
                _score_prompt_field(requested_learning, spec_learning),
                _score_prompt_field(requested_difficulty, spec_difficulty),
            )
            if any(part is None for part in score_parts):
                continue
            total_score = int(sum(part for part in score_parts if part is not None))
            scored.append((total_score, path.name.lower(), path))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored]

    def _build_seed_payload_from_sources(
        self, *, mode: str, prompts_dir: Path
    ) -> tuple[dict[str, Any], str]:
        normalized_mode = _normalize_prompt_mode(mode) or "extraction"
        for name in _SEED_SOURCE_FILENAMES_BY_MODE[normalized_mode]:
            path = prompts_dir / name
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                spec = PromptSpec.model_validate(payload)
            except (OSError, json.JSONDecodeError, ValidationError, ValueError):
                continue
            if _normalize_prompt_mode(str(spec.mode)) != normalized_mode:
                continue
            result = spec.model_dump(mode="json")
            result["name"] = f"auto_seed_{normalized_mode}"
            result["mode"] = normalized_mode
            result["content_type"] = "all"
            result["learning_mode"] = "all"
            result["difficulty_level"] = "all"
            return result, name

        source_matches = self._scan_prompt_files(
            mode=normalized_mode,
            include_templates=True,
        )
        if source_matches:
            source_path = source_matches[0]
            try:
                payload = json.loads(source_path.read_text(encoding="utf-8"))
                spec = PromptSpec.model_validate(payload)
                result = spec.model_dump(mode="json")
                result["name"] = f"auto_seed_{normalized_mode}"
                result["mode"] = normalized_mode
                result["content_type"] = "all"
                result["learning_mode"] = "all"
                result["difficulty_level"] = "all"
                return result, source_path.name
            except (OSError, json.JSONDecodeError, ValidationError, ValueError):
                pass

        return _embedded_seed_prompt_payload(normalized_mode), "embedded"

    def _auto_seed_prompt_file(self, *, mode: str) -> Path | None:
        normalized_mode = _normalize_prompt_mode(mode)
        if normalized_mode not in {"extraction", "explanation"}:
            return None
        prompts_dir = self.resolve_path(_PROMPTS_DIR)
        prompts_dir.mkdir(parents=True, exist_ok=True)
        target = prompts_dir / _AUTO_SEED_FILE_BY_MODE[normalized_mode]

        if target.exists():
            try:
                payload = json.loads(target.read_text(encoding="utf-8"))
                spec = PromptSpec.model_validate(payload)
                if _normalize_prompt_mode(str(spec.mode)) == normalized_mode:
                    return target
            except (OSError, json.JSONDecodeError, ValidationError, ValueError):
                # Rewrite invalid existing seed file.
                pass

        payload, source = self._build_seed_payload_from_sources(
            mode=normalized_mode, prompts_dir=prompts_dir
        )
        payload["name"] = f"auto_seed_{normalized_mode}"
        payload["mode"] = normalized_mode
        payload["content_type"] = "all"
        payload["learning_mode"] = "all"
        payload["difficulty_level"] = "all"
        try:
            spec = PromptSpec.model_validate(payload)
            target.write_text(
                json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            logger.warning(
                "prompt auto-seeded | mode=%s path=%s source=%s",
                normalized_mode,
                target,
                source,
            )
            return target
        except (OSError, ValidationError, ValueError) as exc:
            logger.error(
                "prompt auto-seed failed | mode=%s path=%s error=%s",
                normalized_mode,
                target,
                exc,
            )
            return None

    def _ensure_mode_prompt_files(self, *, mode: str) -> list[Path]:
        matches = self._scan_prompt_files(mode=mode)
        if matches:
            return matches
        self._auto_seed_prompt_file(mode=mode)
        return self._scan_prompt_files(mode=mode)

    def resolve_cloze_prompt_path(
        self,
        *,
        material_profile: str | None = None,
        difficulty: str | None = None,
        learning_mode: str | None = None,
    ) -> Path:
        return self.resolve_extract_prompt_path(
            material_profile=material_profile,
            difficulty=difficulty,
            learning_mode=learning_mode,
        )

    def resolve_extract_prompt_path(
        self,
        *,
        material_profile: str | None = None,
        difficulty: str | None = None,
        learning_mode: str | None = None,
    ) -> Path:
        if self.extract_prompt is not None:
            return self.extract_prompt
        legacy = self._resolve_legacy_extract_prompt_path(
            material_profile=material_profile,
            difficulty=difficulty,
            learning_mode=learning_mode,
        )
        if legacy is not None and self.resolve_path(legacy).exists():
            return legacy

        requested_content = _prompt_content_type_for_profile(
            material_profile or self.material_profile
        )
        requested_learning = _normalize_prompt_learning_mode(
            learning_mode or self.learning_mode
        )
        requested_difficulty = _normalize_prompt_difficulty(
            difficulty or self.cloze_difficulty
        )

        candidates = self._scan_prompt_files(
            mode="extraction",
            content_type=requested_content,
            learning_mode=requested_learning,
            difficulty=requested_difficulty,
        )
        if not candidates:
            self._ensure_mode_prompt_files(mode="extraction")
            candidates = self._scan_prompt_files(
                mode="extraction",
                content_type=requested_content,
                learning_mode=requested_learning,
                difficulty=requested_difficulty,
            )
        if not candidates:
            candidates = self._scan_prompt_files(mode="extraction")
        if candidates:
            return candidates[0]

        raise build_error(
            error_code="PROMPT_EXTRACTION_NOT_FOUND",
            cause="No usable extraction prompt files found.",
            detail=f"prompt_dir={self.resolve_path(_PROMPTS_DIR)}",
            next_steps=[
                "Check prompt JSON schema files under prompts/",
                "Ensure at least one extraction prompt exists",
                "Use --extract-prompt to specify a valid prompt path",
            ],
            exit_code=ExitCode.CONFIG_ERROR,
        )

    def resolve_explain_prompt_path(
        self,
        *,
        material_profile: str | None = None,
        difficulty: str | None = None,
        learning_mode: str | None = None,
    ) -> Path:
        if self.explain_prompt is not None:
            return self.explain_prompt
        if self.prompt_translate is not None and self.resolve_path(
            self.prompt_translate
        ).exists():
            return self.prompt_translate

        requested_content = _prompt_content_type_for_profile(
            material_profile or self.material_profile
        )
        requested_learning = _normalize_prompt_learning_mode(
            learning_mode or self.learning_mode
        )
        requested_difficulty = _normalize_prompt_difficulty(
            difficulty or self.cloze_difficulty
        )

        candidates = self._scan_prompt_files(
            mode="explanation",
            content_type=requested_content,
            learning_mode=requested_learning,
            difficulty=requested_difficulty,
        )
        if not candidates:
            self._ensure_mode_prompt_files(mode="explanation")
            candidates = self._scan_prompt_files(
                mode="explanation",
                content_type=requested_content,
                learning_mode=requested_learning,
                difficulty=requested_difficulty,
            )
        if not candidates:
            candidates = self._scan_prompt_files(mode="explanation")
        if candidates:
            return candidates[0]

        raise build_error(
            error_code="PROMPT_EXPLANATION_NOT_FOUND",
            cause="No usable explanation prompt files found.",
            detail=f"prompt_dir={self.resolve_path(_PROMPTS_DIR)}",
            next_steps=[
                "Check prompt JSON schema files under prompts/",
                "Ensure at least one explanation prompt exists",
                "Use --explain-prompt to specify a valid prompt path",
            ],
            exit_code=ExitCode.CONFIG_ERROR,
        )

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
        ("CLAWLEARN_DEFAULT_CLOZE_PROMPT", "CLAWLEARN_PROMPT_CLOZE"),
        ("CLAWLEARN_DEFAULT_TRANSLATE_PROMPT", "CLAWLEARN_PROMPT_TRANSLATE"),
        ("CLAWLEARN_DEFAULT_ANKI_TEMPLATE", "CLAWLEARN_ANKI_TEMPLATE"),
        ("CLAWLEARN_INGEST_SHORT_UTTERANCE_MAX_WORDS", "CLAWLEARN_INGEST_SHORT_LINE_MAX_WORDS"),
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
        if k.startswith("CLAWLEARN_"):
            merged[k] = v

    _normalize_env_aliases(merged)
    voice_slots = _parse_voice_slots(merged)
    voice_map = _parse_legacy_voice_map(merged)

    raw_profile = _env_value(
        merged.get("CLAWLEARN_MATERIAL_PROFILE"),
        _env_value(merged.get("CLAWLEARN_CONTENT_PROFILE"), "prose_article"),
    )
    normalized_profile = _normalize_profile(str(raw_profile))

    payload: dict[str, Any] = {
        "workspace_root": root,
        "default_source_lang": _env_value(merged.get("CLAWLEARN_DEFAULT_SOURCE_LANG"), "en"),
        "default_target_lang": _env_value(merged.get("CLAWLEARN_DEFAULT_TARGET_LANG"), "zh"),
        "llm_provider": _env_value(merged.get("CLAWLEARN_LLM_PROVIDER"), "openai_compatible"),
        "llm_base_url": _env_value(merged.get("CLAWLEARN_LLM_BASE_URL"), "http://127.0.0.1:8000/v1"),
        "llm_api_key": _env_value(merged.get("CLAWLEARN_LLM_API_KEY"), ""),
        "llm_model": _env_value(merged.get("CLAWLEARN_LLM_MODEL"), ""),
        "llm_timeout_seconds": _env_value(merged.get("CLAWLEARN_LLM_TIMEOUT_SECONDS"), 120),
        "llm_max_retries": _env_value(merged.get("CLAWLEARN_LLM_MAX_RETRIES"), 3),
        "llm_retry_backoff_seconds": _env_value(merged.get("CLAWLEARN_LLM_RETRY_BACKOFF_SECONDS"), 2.0),
        "llm_request_sleep_seconds": _env_value(merged.get("CLAWLEARN_LLM_REQUEST_SLEEP_SECONDS"), 0.0),
        "llm_temperature": _env_value(merged.get("CLAWLEARN_LLM_TEMPERATURE"), 0.2),
        "ingest_short_line_max_words": _env_value(merged.get("CLAWLEARN_INGEST_SHORT_LINE_MAX_WORDS"), 3),
        "chunk_max_chars": _env_value(merged.get("CLAWLEARN_CHUNK_MAX_CHARS"), 1800),
        "chunk_min_chars": _env_value(merged.get("CLAWLEARN_CHUNK_MIN_CHARS"), 120),
        "chunk_overlap_sentences": _env_value(merged.get("CLAWLEARN_CHUNK_OVERLAP_SENTENCES"), 1),
        "cloze_max_sentences": _env_value(merged.get("CLAWLEARN_CLOZE_MAX_SENTENCES"), 3),
        "cloze_min_chars": _env_value(merged.get("CLAWLEARN_CLOZE_MIN_CHARS"), 0),
        "cloze_difficulty": _env_value(merged.get("CLAWLEARN_CLOZE_DIFFICULTY"), "intermediate"),
        "cloze_max_per_chunk": _env_value(merged.get("CLAWLEARN_CLOZE_MAX_PER_CHUNK"), None),
        "llm_chunk_batch_size": _env_value(merged.get("CLAWLEARN_LLM_CHUNK_BATCH_SIZE"), 1),
        "secondary_extract_enable": _env_value(merged.get("CLAWLEARN_SECONDARY_EXTRACT_ENABLE"), False),
        "secondary_extract_parallel": _env_value(
            merged.get("CLAWLEARN_SECONDARY_EXTRACT_PARALLEL"),
            False,
        ),
        "secondary_extract_llm_base_url": _env_value(
            merged.get("CLAWLEARN_SECONDARY_EXTRACT_LLM_BASE_URL"),
            None,
        ),
        "secondary_extract_llm_api_key": _env_value(
            merged.get("CLAWLEARN_SECONDARY_EXTRACT_LLM_API_KEY"),
            None,
        ),
        "secondary_extract_llm_model": _env_value(
            merged.get("CLAWLEARN_SECONDARY_EXTRACT_LLM_MODEL"),
            None,
        ),
        "secondary_extract_llm_timeout_seconds": _env_value(
            merged.get("CLAWLEARN_SECONDARY_EXTRACT_LLM_TIMEOUT_SECONDS"),
            None,
        ),
        "secondary_extract_llm_temperature": _env_value(
            merged.get("CLAWLEARN_SECONDARY_EXTRACT_LLM_TEMPERATURE"),
            None,
        ),
        "secondary_extract_llm_max_retries": _env_value(
            merged.get("CLAWLEARN_SECONDARY_EXTRACT_LLM_MAX_RETRIES"),
            None,
        ),
        "secondary_extract_llm_retry_backoff_seconds": _env_value(
            merged.get("CLAWLEARN_SECONDARY_EXTRACT_LLM_RETRY_BACKOFF_SECONDS"),
            None,
        ),
        "secondary_extract_llm_chunk_batch_size": _env_value(
            merged.get("CLAWLEARN_SECONDARY_EXTRACT_LLM_CHUNK_BATCH_SIZE"),
            None,
        ),
        "validate_format_retry_enable": _env_value(merged.get("CLAWLEARN_VALIDATE_FORMAT_RETRY_ENABLE"), True),
        "validate_format_retry_max": _env_value(merged.get("CLAWLEARN_VALIDATE_FORMAT_RETRY_MAX"), 3),
        "validate_format_retry_llm_enable": _env_value(
            merged.get("CLAWLEARN_VALIDATE_FORMAT_RETRY_LLM_ENABLE"),
            True,
        ),
        "taxonomy_repair_enable": _env_value(merged.get("CLAWLEARN_TAXONOMY_REPAIR_ENABLE"), False),
        "lingua_transcript_min_context_sentences": _env_value(
            merged.get("CLAWLEARN_LINGUA_TRANSCRIPT_MIN_CONTEXT_SENTENCES"),
            2,
        ),
        "lingua_annotate_enable": _env_value(merged.get("CLAWLEARN_LINGUA_ANNOTATE_ENABLE"), False),
        "lingua_annotate_batch_size": _env_value(merged.get("CLAWLEARN_LINGUA_ANNOTATE_BATCH_SIZE"), 50),
        "lingua_annotate_max_items": _env_value(merged.get("CLAWLEARN_LINGUA_ANNOTATE_MAX_ITEMS"), None),
        "prompt_lang": _env_value(merged.get("CLAWLEARN_PROMPT_LANG"), "zh"),
        "extract_prompt": _env_value(merged.get("CLAWLEARN_EXTRACT_PROMPT"), None),
        "explain_prompt": _env_value(merged.get("CLAWLEARN_EXPLAIN_PROMPT"), None),
        "translate_llm_base_url": _env_value(merged.get("CLAWLEARN_TRANSLATE_LLM_BASE_URL"), None),
        "translate_llm_api_key": _env_value(merged.get("CLAWLEARN_TRANSLATE_LLM_API_KEY"), None),
        "translate_llm_model": _env_value(merged.get("CLAWLEARN_TRANSLATE_LLM_MODEL"), None),
        "translate_llm_temperature": _env_value(merged.get("CLAWLEARN_TRANSLATE_LLM_TEMPERATURE"), None),
        "translate_batch_size": _env_value(merged.get("CLAWLEARN_TRANSLATE_BATCH_SIZE"), 4),
        # Legacy prompt path overrides. Defaults are intentionally empty to avoid file-name coupling.
        "prompt_cloze": _env_value(merged.get("CLAWLEARN_PROMPT_CLOZE"), None),
        "prompt_cloze_textbook": _env_value(merged.get("CLAWLEARN_PROMPT_CLOZE_TEXTBOOK"), None),
        "prompt_cloze_prose_beginner": _env_value(
            merged.get("CLAWLEARN_PROMPT_CLOZE_PROSE_BEGINNER"),
            None,
        ),
        "prompt_cloze_prose_intermediate": _env_value(
            merged.get("CLAWLEARN_PROMPT_CLOZE_PROSE_INTERMEDIATE"),
            None,
        ),
        "prompt_cloze_prose_advanced": _env_value(
            merged.get("CLAWLEARN_PROMPT_CLOZE_PROSE_ADVANCED"),
            None,
        ),
        "prompt_cloze_prose_reading_support_beginner": _env_value(
            merged.get("CLAWLEARN_PROMPT_CLOZE_PROSE_READING_SUPPORT_BEGINNER"),
            None,
        ),
        "prompt_cloze_prose_reading_support_intermediate": _env_value(
            merged.get("CLAWLEARN_PROMPT_CLOZE_PROSE_READING_SUPPORT_INTERMEDIATE"),
            None,
        ),
        "prompt_cloze_prose_reading_support_advanced": _env_value(
            merged.get("CLAWLEARN_PROMPT_CLOZE_PROSE_READING_SUPPORT_ADVANCED"),
            None,
        ),
        "prompt_cloze_transcript_beginner": _env_value(
            merged.get("CLAWLEARN_PROMPT_CLOZE_TRANSCRIPT_BEGINNER"),
            None,
        ),
        "prompt_cloze_transcript_intermediate": _env_value(
            merged.get("CLAWLEARN_PROMPT_CLOZE_TRANSCRIPT_INTERMEDIATE"),
            None,
        ),
        "prompt_cloze_transcript_advanced": _env_value(
            merged.get("CLAWLEARN_PROMPT_CLOZE_TRANSCRIPT_ADVANCED"),
            None,
        ),
        "prompt_cloze_transcript_reading_support_beginner": _env_value(
            merged.get("CLAWLEARN_PROMPT_CLOZE_TRANSCRIPT_READING_SUPPORT_BEGINNER"),
            None,
        ),
        "prompt_cloze_transcript_reading_support_intermediate": _env_value(
            merged.get("CLAWLEARN_PROMPT_CLOZE_TRANSCRIPT_READING_SUPPORT_INTERMEDIATE"),
            None,
        ),
        "prompt_cloze_transcript_reading_support_advanced": _env_value(
            merged.get("CLAWLEARN_PROMPT_CLOZE_TRANSCRIPT_READING_SUPPORT_ADVANCED"),
            None,
        ),
        "prompt_translate": _env_value(merged.get("CLAWLEARN_PROMPT_TRANSLATE"), None),
        "anki_template": _env_value(merged.get("CLAWLEARN_ANKI_TEMPLATE"), DEFAULT_ANKI_TEMPLATE),
        # Legacy + V2 profiles.
        "content_profile": _env_value(merged.get("CLAWLEARN_CONTENT_PROFILE"), normalized_profile),
        "material_profile": normalized_profile,
        "learning_mode": _env_value(merged.get("CLAWLEARN_LEARNING_MODE"), "lingua_expression"),
        "allow_empty_deck": _env_value(merged.get("CLAWLEARN_ALLOW_EMPTY_DECK"), True),
        "output_dir": _env_value(merged.get("CLAWLEARN_OUTPUT_DIR"), DEFAULT_OUTPUT_DIR),
        "export_dir": _env_value(merged.get("CLAWLEARN_EXPORT_DIR"), DEFAULT_EXPORT_DIR),
        "log_dir": _env_value(merged.get("CLAWLEARN_LOG_DIR"), DEFAULT_LOG_DIR),
        "log_level": _env_value(merged.get("CLAWLEARN_LOG_LEVEL"), "INFO"),
        "save_intermediate": _env_value(merged.get("CLAWLEARN_SAVE_INTERMEDIATE"), True),
        "default_deck_name": _env_value(merged.get("CLAWLEARN_DEFAULT_DECK_NAME"), DEFAULT_DECK_NAME),
        "tts_provider": _env_value(merged.get("CLAWLEARN_TTS_PROVIDER"), "edge_tts"),
        "tts_output_format": _env_value(merged.get("CLAWLEARN_TTS_OUTPUT_FORMAT"), "mp3"),
        "tts_rate": _env_value(merged.get("CLAWLEARN_TTS_RATE"), "+0%"),
        "tts_volume": _env_value(merged.get("CLAWLEARN_TTS_VOLUME"), "+0%"),
        "tts_random_seed": _env_value(merged.get("CLAWLEARN_TTS_RANDOM_SEED"), None),
        "tts_retry_attempts": _env_value(merged.get("CLAWLEARN_TTS_RETRY_ATTEMPTS"), 3),
        "tts_retry_backoff_seconds": _env_value(
            merged.get("CLAWLEARN_TTS_RETRY_BACKOFF_SECONDS"),
            1.0,
        ),
        "tts_edge_voices": voice_slots,
        "tts_edge_voice_map": voice_map,
    }

    if overrides:
        payload.update({k: v for k, v in overrides.items() if v is not None})

    return AppConfig.model_validate(payload)


def validate_base_config(cfg: AppConfig) -> None:
    required_paths = [
        ("CLAWLEARN_EFFECTIVE_EXTRACT_PROMPT", cfg.resolve_path(cfg.resolve_extract_prompt_path())),
        ("CLAWLEARN_EFFECTIVE_EXPLAIN_PROMPT", cfg.resolve_path(cfg.resolve_explain_prompt_path())),
        ("CLAWLEARN_ANKI_TEMPLATE", cfg.resolve_path(cfg.anki_template)),
    ]
    for key, path in required_paths:
        if not path.exists():
            raise build_error(
                error_code="CONFIG_PATH_NOT_FOUND",
                cause="Required config file is missing.",
                detail=f"{key} -> {path}",
                next_steps=[
                    "Check the path values in .env",
                    "Run `clawlearn init` to generate default config files",
                ],
                exit_code=ExitCode.CONFIG_ERROR,
            )


def validate_runtime_config(cfg: AppConfig) -> None:
    if cfg.llm_provider == "openai_compatible":
        if not cfg.llm_api_key:
            raise build_error(
                error_code="CONFIG_MISSING_API_KEY",
                cause="LLM API key is missing.",
                detail="`CLAWLEARN_LLM_API_KEY` was not found in .env or environment variables.",
                next_steps=[
                    "Add `CLAWLEARN_LLM_API_KEY` to .env",
                    "Or point to the correct env file via `--env-file`",
                    "Run `clawlearn config validate` to check config",
                ],
                exit_code=ExitCode.CONFIG_ERROR,
            )
        if not cfg.llm_model:
            raise build_error(
                error_code="CONFIG_MISSING_MODEL",
                cause="LLM model is missing.",
                detail="`CLAWLEARN_LLM_MODEL` was not found in configuration.",
                next_steps=["Add `CLAWLEARN_LLM_MODEL` to .env"],
                exit_code=ExitCode.CONFIG_ERROR,
            )
