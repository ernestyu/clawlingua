"""Config-tab event handlers extracted from app.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ConfigDeps:
    list_models_markdown: Callable[..., str]
    test_models_markdown: Callable[..., str]
    to_timeout_seconds: Callable[[Any], float]
    normalize_ui_lang: Callable[[str | None], str]
    read_env_example: Callable[[], dict[str, str]]
    tr: Callable[[str, str, str], str]
    save_env_v2: Callable[..., str]


def on_list_models(
    base_url: str, api_key: str, timeout_raw: Any, ui_lang_val: str, *, deps: ConfigDeps
) -> str:
    return deps.list_models_markdown(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=deps.to_timeout_seconds(timeout_raw),
        lang=deps.normalize_ui_lang(ui_lang_val),
    )


def on_test_models(
    base_url: str, api_key: str, timeout_raw: Any, ui_lang_val: str, *, deps: ConfigDeps
) -> str:
    return deps.test_models_markdown(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=deps.to_timeout_seconds(timeout_raw),
        lang=deps.normalize_ui_lang(ui_lang_val),
    )


def on_load_defaults(
    llm_base_url_val: str,
    llm_api_key_val: str,
    llm_model_val: str,
    llm_timeout_val: str,
    llm_temperature_val: str,
    llm_chunk_batch_size_val: str,
    translate_base_url_val: str,
    translate_api_key_val: str,
    translate_model_val: str,
    translate_temperature_val: str,
    chunk_max_chars_val: str,
    chunk_min_chars_val: str,
    cloze_min_chars_val: str,
    cloze_max_per_chunk_val: str,
    validate_retry_enable_val: str,
    validate_retry_max_val: str,
    validate_retry_llm_enable_val: str,
    content_profile_val: str,
    cloze_difficulty_val: str,
    prompt_lang_val: str,
    extract_prompt_env_val: str,
    explain_prompt_env_val: str,
    output_dir_val: str,
    export_dir_val: str,
    log_dir_val: str,
    default_deck_name_val: str,
    tts_voice1_val: str,
    tts_voice2_val: str,
    tts_voice3_val: str,
    tts_voice4_val: str,
    ui_lang_val: str,
    *,
    deps: ConfigDeps,
) -> tuple[str, ...]:
    defaults = deps.read_env_example()
    lang = deps.normalize_ui_lang(ui_lang_val)

    def dv(key: str, current: str) -> str:
        return defaults.get(key, current or "")

    return (
        dv("CLAWLINGUA_LLM_BASE_URL", llm_base_url_val),
        dv("CLAWLINGUA_LLM_API_KEY", llm_api_key_val),
        dv("CLAWLINGUA_LLM_MODEL", llm_model_val),
        dv("CLAWLINGUA_LLM_TIMEOUT_SECONDS", llm_timeout_val),
        dv("CLAWLINGUA_LLM_TEMPERATURE", llm_temperature_val),
        dv("CLAWLINGUA_LLM_CHUNK_BATCH_SIZE", llm_chunk_batch_size_val),
        dv("CLAWLINGUA_TRANSLATE_LLM_BASE_URL", translate_base_url_val),
        dv("CLAWLINGUA_TRANSLATE_LLM_API_KEY", translate_api_key_val),
        dv("CLAWLINGUA_TRANSLATE_LLM_MODEL", translate_model_val),
        dv(
            "CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE",
            translate_temperature_val,
        ),
        dv("CLAWLINGUA_CHUNK_MAX_CHARS", chunk_max_chars_val),
        dv("CLAWLINGUA_CHUNK_MIN_CHARS", chunk_min_chars_val),
        dv("CLAWLINGUA_CLOZE_MIN_CHARS", cloze_min_chars_val),
        dv("CLAWLINGUA_CLOZE_MAX_PER_CHUNK", cloze_max_per_chunk_val),
        dv(
            "CLAWLINGUA_VALIDATE_FORMAT_RETRY_ENABLE",
            validate_retry_enable_val,
        ),
        dv("CLAWLINGUA_VALIDATE_FORMAT_RETRY_MAX", validate_retry_max_val),
        dv(
            "CLAWLINGUA_VALIDATE_FORMAT_RETRY_LLM_ENABLE",
            validate_retry_llm_enable_val,
        ),
        dv("CLAWLINGUA_CONTENT_PROFILE", content_profile_val),
        dv("CLAWLINGUA_CLOZE_DIFFICULTY", cloze_difficulty_val),
        dv("CLAWLINGUA_PROMPT_LANG", prompt_lang_val),
        dv("CLAWLINGUA_EXTRACT_PROMPT", extract_prompt_env_val),
        dv("CLAWLINGUA_EXPLAIN_PROMPT", explain_prompt_env_val),
        dv("CLAWLINGUA_OUTPUT_DIR", output_dir_val),
        dv("CLAWLINGUA_EXPORT_DIR", export_dir_val),
        dv("CLAWLINGUA_LOG_DIR", log_dir_val),
        dv("CLAWLINGUA_DEFAULT_DECK_NAME", default_deck_name_val),
        dv("CLAWLINGUA_TTS_EDGE_VOICE1", tts_voice1_val),
        dv("CLAWLINGUA_TTS_EDGE_VOICE2", tts_voice2_val),
        dv("CLAWLINGUA_TTS_EDGE_VOICE3", tts_voice3_val),
        dv("CLAWLINGUA_TTS_EDGE_VOICE4", tts_voice4_val),
        f"✅ {deps.tr(lang, 'Loaded defaults from ENV_EXAMPLE.md (not yet saved).', 'Loaded defaults from ENV_EXAMPLE.md (not yet saved).')}",
    )


def on_save_config(
    llm_base_url_val: Any,
    llm_api_key_val: Any,
    llm_model_val: Any,
    llm_timeout_val: Any,
    llm_temperature_val: Any,
    llm_chunk_batch_size_val: Any,
    translate_base_url_val: Any,
    translate_api_key_val: Any,
    translate_model_val: Any,
    translate_temperature_val: Any,
    chunk_max_chars_val: Any,
    chunk_min_chars_val: Any,
    cloze_min_chars_val: Any,
    cloze_max_per_chunk_val: Any,
    validate_retry_enable_val: Any,
    validate_retry_max_val: Any,
    validate_retry_llm_enable_val: Any,
    content_profile_val: Any,
    cloze_difficulty_val: Any,
    prompt_lang_val: Any,
    extract_prompt_env_val: Any,
    explain_prompt_env_val: Any,
    output_dir_val: Any,
    export_dir_val: Any,
    log_dir_val: Any,
    default_deck_name_val: Any,
    tts_voice1_val: Any,
    tts_voice2_val: Any,
    tts_voice3_val: Any,
    tts_voice4_val: Any,
    ui_lang_val: Any,
    *,
    deps: ConfigDeps,
) -> str:
    updated = {
        "CLAWLINGUA_LLM_BASE_URL": llm_base_url_val or "",
        "CLAWLINGUA_LLM_API_KEY": llm_api_key_val or "",
        "CLAWLINGUA_LLM_MODEL": llm_model_val or "",
        "CLAWLINGUA_LLM_TIMEOUT_SECONDS": llm_timeout_val or "",
        "CLAWLINGUA_LLM_TEMPERATURE": llm_temperature_val or "",
        "CLAWLINGUA_LLM_CHUNK_BATCH_SIZE": llm_chunk_batch_size_val or "",
        "CLAWLINGUA_TRANSLATE_LLM_BASE_URL": translate_base_url_val or "",
        "CLAWLINGUA_TRANSLATE_LLM_API_KEY": translate_api_key_val or "",
        "CLAWLINGUA_TRANSLATE_LLM_MODEL": translate_model_val or "",
        "CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE": translate_temperature_val or "",
        "CLAWLINGUA_CHUNK_MAX_CHARS": chunk_max_chars_val or "",
        "CLAWLINGUA_CHUNK_MIN_CHARS": chunk_min_chars_val or "",
        "CLAWLINGUA_CLOZE_MIN_CHARS": cloze_min_chars_val or "",
        "CLAWLINGUA_CLOZE_MAX_PER_CHUNK": cloze_max_per_chunk_val or "",
        "CLAWLINGUA_VALIDATE_FORMAT_RETRY_ENABLE": validate_retry_enable_val or "",
        "CLAWLINGUA_VALIDATE_FORMAT_RETRY_MAX": validate_retry_max_val or "",
        "CLAWLINGUA_VALIDATE_FORMAT_RETRY_LLM_ENABLE": validate_retry_llm_enable_val or "",
        "CLAWLINGUA_CONTENT_PROFILE": content_profile_val or "",
        "CLAWLINGUA_CLOZE_DIFFICULTY": cloze_difficulty_val or "",
        "CLAWLINGUA_PROMPT_LANG": prompt_lang_val or "",
        "CLAWLINGUA_EXTRACT_PROMPT": extract_prompt_env_val or "",
        "CLAWLINGUA_EXPLAIN_PROMPT": explain_prompt_env_val or "",
        "CLAWLINGUA_OUTPUT_DIR": output_dir_val or "",
        "CLAWLINGUA_EXPORT_DIR": export_dir_val or "",
        "CLAWLINGUA_LOG_DIR": log_dir_val or "",
        "CLAWLINGUA_DEFAULT_DECK_NAME": default_deck_name_val or "",
        "CLAWLINGUA_TTS_EDGE_VOICE1": tts_voice1_val or "",
        "CLAWLINGUA_TTS_EDGE_VOICE2": tts_voice2_val or "",
        "CLAWLINGUA_TTS_EDGE_VOICE3": tts_voice3_val or "",
        "CLAWLINGUA_TTS_EDGE_VOICE4": tts_voice4_val or "",
    }
    return deps.save_env_v2(updated, lang=deps.normalize_ui_lang(ui_lang_val))
