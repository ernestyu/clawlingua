"""Config-tab event handlers extracted from app.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx


@dataclass(frozen=True)
class ConfigDeps:
    to_timeout_seconds: Callable[[Any], float]
    normalize_ui_lang: Callable[[str | None], str]
    read_env_example: Callable[[], dict[str, str]]
    resolve_env_file: Callable[[], Path | None]
    load_config: Callable[..., Any]
    load_env_view: Callable[..., dict[str, str]]
    tr: Callable[[str, str, str], str]
    save_env_v2: Callable[..., str]


def _config_values_from_view(cfg_view: dict[str, str]) -> tuple[str, ...]:
    return (
        cfg_view.get("CLAWLEARN_LLM_BASE_URL", ""),
        cfg_view.get("CLAWLEARN_LLM_API_KEY", ""),
        cfg_view.get("CLAWLEARN_LLM_MODEL", ""),
        cfg_view.get("CLAWLEARN_LLM_TIMEOUT_SECONDS", "120"),
        cfg_view.get("CLAWLEARN_LLM_TEMPERATURE", "0.2"),
        cfg_view.get("CLAWLEARN_LLM_CHUNK_BATCH_SIZE", "1"),
        cfg_view.get("CLAWLEARN_TRANSLATE_LLM_BASE_URL", ""),
        cfg_view.get("CLAWLEARN_TRANSLATE_LLM_API_KEY", ""),
        cfg_view.get("CLAWLEARN_TRANSLATE_LLM_MODEL", ""),
        cfg_view.get("CLAWLEARN_TRANSLATE_LLM_TEMPERATURE", ""),
        cfg_view.get("CLAWLEARN_CHUNK_MAX_CHARS", "1800"),
        cfg_view.get("CLAWLEARN_CHUNK_MIN_CHARS", "120"),
        cfg_view.get("CLAWLEARN_CLOZE_MIN_CHARS", "0"),
        cfg_view.get("CLAWLEARN_CLOZE_MAX_PER_CHUNK", ""),
        cfg_view.get("CLAWLEARN_VALIDATE_FORMAT_RETRY_ENABLE", "true"),
        cfg_view.get("CLAWLEARN_VALIDATE_FORMAT_RETRY_MAX", "3"),
        cfg_view.get("CLAWLEARN_VALIDATE_FORMAT_RETRY_LLM_ENABLE", "true"),
        cfg_view.get("CLAWLEARN_CONTENT_PROFILE", "prose_article"),
        cfg_view.get("CLAWLEARN_CLOZE_DIFFICULTY", "intermediate"),
        cfg_view.get("CLAWLEARN_PROMPT_LANG", "zh"),
        cfg_view.get("CLAWLEARN_EXTRACT_PROMPT", ""),
        cfg_view.get("CLAWLEARN_EXPLAIN_PROMPT", ""),
        cfg_view.get("CLAWLEARN_OUTPUT_DIR", "./runs"),
        cfg_view.get("CLAWLEARN_EXPORT_DIR", "./outputs"),
        cfg_view.get("CLAWLEARN_LOG_DIR", "./logs"),
        cfg_view.get("CLAWLEARN_DEFAULT_DECK_NAME", ""),
        cfg_view.get("CLAWLEARN_TTS_EDGE_VOICE1", ""),
        cfg_view.get("CLAWLEARN_TTS_EDGE_VOICE2", ""),
        cfg_view.get("CLAWLEARN_TTS_EDGE_VOICE3", ""),
        cfg_view.get("CLAWLEARN_TTS_EDGE_VOICE4", ""),
    )


def _reload_status(
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
    env_path: Path,
    found: bool,
    error: Exception | None = None,
) -> str:
    if error is not None:
        return (
            f"❌ {tr(lang, 'Failed to reload config from .env', 'Failed to reload config from .env')}: `{error}`\n\n"
            f"- env_file: `{env_path}`"
        )
    if not found:
        return (
            f"⚠️ {tr(lang, '.env not found', '.env not found')}\n\n"
            f"- env_file: `{env_path}`"
        )
    return (
        f"✅ {tr(lang, 'Reloaded config from .env', 'Reloaded config from .env')}\n\n"
        f"- env_file: `{env_path}`"
    )


def _normalize_base_url(base_url: str) -> str:
    value = (base_url or "").strip().rstrip("/")
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")]
    return value


def _build_models_url(base_url: str) -> str:
    root = _normalize_base_url(base_url)
    if not root:
        return ""
    if root.endswith("/models"):
        return root
    return f"{root}/models"


def _request_models(
    base_url: str, api_key: str, timeout_seconds: float
) -> tuple[str, httpx.Response]:
    endpoint = _build_models_url(base_url)
    if not endpoint:
        raise ValueError("missing base URL")
    headers = {"Accept": "application/json"}
    token = (api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.get(endpoint, headers=headers)
    return endpoint, response


def _list_models_markdown(
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
) -> str:
    if not _normalize_base_url(base_url):
        return f"⚠️ {tr(lang, 'Missing base URL.', 'Missing base URL.')}"
    try:
        endpoint, response = _request_models(base_url, api_key, timeout_seconds)
    except ValueError:
        return f"⚠️ {tr(lang, 'Missing base URL.', 'Missing base URL.')}"
    except httpx.RequestError as exc:
        return f"❌ {tr(lang, 'Request failed', 'Request failed')}: `{exc}`"
    except Exception as exc:
        return f"❌ {tr(lang, 'Request failed', 'Request failed')}: `{exc}`"

    if response.status_code >= 400:
        body = response.text[:500] if response.text else ""
        return (
            f"❌ {tr(lang, 'HTTP error', 'HTTP error')}: **{response.status_code}**\n\n"
            f"- endpoint: `{endpoint}`\n"
            f"- body: `{body}`"
        )

    try:
        payload = response.json()
    except ValueError:
        return f"❌ {tr(lang, 'Response is not valid JSON.', 'Response is not valid JSON.')}"

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return (
            f"⚠️ {tr(lang, 'Response JSON has no list field `data`.', 'Response JSON has no list field `data`.')}\n\n"
            f"- endpoint: `{endpoint}`\n"
            f"- status: `{response.status_code}`"
        )

    model_ids: list[str] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id or model_id in seen:
            continue
        model_ids.append(model_id)
        seen.add(model_id)

    if not model_ids:
        return (
            f"✅ {tr(lang, 'Found models', 'Found models')}: **0**\n\n"
            f"- endpoint: `{endpoint}`\n"
            f"- status: `{response.status_code}`\n"
            f"- {tr(lang, 'No model ids found in `data`.', 'No model ids found in `data`.')}"
        )

    lines = [f"- `{model_id}`" for model_id in model_ids]
    return (
        f"✅ {tr(lang, 'Found models', 'Found models')}: **{len(model_ids)}**\n\n"
        f"- endpoint: `{endpoint}`\n"
        f"- status: `{response.status_code}`\n\n"
        f"{chr(10).join(lines)}"
    )


def _test_models_markdown(
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
) -> str:
    if not _normalize_base_url(base_url):
        return f"⚠️ {tr(lang, 'Missing base URL.', 'Missing base URL.')}"
    try:
        endpoint, response = _request_models(base_url, api_key, timeout_seconds)
    except ValueError:
        return f"⚠️ {tr(lang, 'Missing base URL.', 'Missing base URL.')}"
    except httpx.RequestError as exc:
        return f"❌ {tr(lang, 'Request failed', 'Request failed')}: `{exc}`"
    except Exception as exc:
        return f"❌ {tr(lang, 'Request failed', 'Request failed')}: `{exc}`"
    return (
        f"✅ {tr(lang, 'Connectivity OK', 'Connectivity OK')}\n\n"
        f"- endpoint: `{endpoint}`\n"
        f"- status: `{response.status_code}`"
    )


def on_list_models(
    base_url: str, api_key: str, timeout_raw: Any, ui_lang_val: str, *, deps: ConfigDeps
) -> str:
    return _list_models_markdown(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=deps.to_timeout_seconds(timeout_raw),
        lang=deps.normalize_ui_lang(ui_lang_val),
        tr=deps.tr,
    )


def on_test_models(
    base_url: str, api_key: str, timeout_raw: Any, ui_lang_val: str, *, deps: ConfigDeps
) -> str:
    return _test_models_markdown(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=deps.to_timeout_seconds(timeout_raw),
        lang=deps.normalize_ui_lang(ui_lang_val),
        tr=deps.tr,
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
        dv("CLAWLEARN_LLM_BASE_URL", llm_base_url_val),
        dv("CLAWLEARN_LLM_API_KEY", llm_api_key_val),
        dv("CLAWLEARN_LLM_MODEL", llm_model_val),
        dv("CLAWLEARN_LLM_TIMEOUT_SECONDS", llm_timeout_val),
        dv("CLAWLEARN_LLM_TEMPERATURE", llm_temperature_val),
        dv("CLAWLEARN_LLM_CHUNK_BATCH_SIZE", llm_chunk_batch_size_val),
        dv("CLAWLEARN_TRANSLATE_LLM_BASE_URL", translate_base_url_val),
        dv("CLAWLEARN_TRANSLATE_LLM_API_KEY", translate_api_key_val),
        dv("CLAWLEARN_TRANSLATE_LLM_MODEL", translate_model_val),
        dv(
            "CLAWLEARN_TRANSLATE_LLM_TEMPERATURE",
            translate_temperature_val,
        ),
        dv("CLAWLEARN_CHUNK_MAX_CHARS", chunk_max_chars_val),
        dv("CLAWLEARN_CHUNK_MIN_CHARS", chunk_min_chars_val),
        dv("CLAWLEARN_CLOZE_MIN_CHARS", cloze_min_chars_val),
        dv("CLAWLEARN_CLOZE_MAX_PER_CHUNK", cloze_max_per_chunk_val),
        dv(
            "CLAWLEARN_VALIDATE_FORMAT_RETRY_ENABLE",
            validate_retry_enable_val,
        ),
        dv("CLAWLEARN_VALIDATE_FORMAT_RETRY_MAX", validate_retry_max_val),
        dv(
            "CLAWLEARN_VALIDATE_FORMAT_RETRY_LLM_ENABLE",
            validate_retry_llm_enable_val,
        ),
        dv("CLAWLEARN_CONTENT_PROFILE", content_profile_val),
        dv("CLAWLEARN_CLOZE_DIFFICULTY", cloze_difficulty_val),
        dv("CLAWLEARN_PROMPT_LANG", prompt_lang_val),
        dv("CLAWLEARN_EXTRACT_PROMPT", extract_prompt_env_val),
        dv("CLAWLEARN_EXPLAIN_PROMPT", explain_prompt_env_val),
        dv("CLAWLEARN_OUTPUT_DIR", output_dir_val),
        dv("CLAWLEARN_EXPORT_DIR", export_dir_val),
        dv("CLAWLEARN_LOG_DIR", log_dir_val),
        dv("CLAWLEARN_DEFAULT_DECK_NAME", default_deck_name_val),
        dv("CLAWLEARN_TTS_EDGE_VOICE1", tts_voice1_val),
        dv("CLAWLEARN_TTS_EDGE_VOICE2", tts_voice2_val),
        dv("CLAWLEARN_TTS_EDGE_VOICE3", tts_voice3_val),
        dv("CLAWLEARN_TTS_EDGE_VOICE4", tts_voice4_val),
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
) -> tuple[str, ...]:
    updated = {
        "CLAWLEARN_LLM_BASE_URL": llm_base_url_val or "",
        "CLAWLEARN_LLM_API_KEY": llm_api_key_val or "",
        "CLAWLEARN_LLM_MODEL": llm_model_val or "",
        "CLAWLEARN_LLM_TIMEOUT_SECONDS": llm_timeout_val or "",
        "CLAWLEARN_LLM_TEMPERATURE": llm_temperature_val or "",
        "CLAWLEARN_LLM_CHUNK_BATCH_SIZE": llm_chunk_batch_size_val or "",
        "CLAWLEARN_TRANSLATE_LLM_BASE_URL": translate_base_url_val or "",
        "CLAWLEARN_TRANSLATE_LLM_API_KEY": translate_api_key_val or "",
        "CLAWLEARN_TRANSLATE_LLM_MODEL": translate_model_val or "",
        "CLAWLEARN_TRANSLATE_LLM_TEMPERATURE": translate_temperature_val or "",
        "CLAWLEARN_CHUNK_MAX_CHARS": chunk_max_chars_val or "",
        "CLAWLEARN_CHUNK_MIN_CHARS": chunk_min_chars_val or "",
        "CLAWLEARN_CLOZE_MIN_CHARS": cloze_min_chars_val or "",
        "CLAWLEARN_CLOZE_MAX_PER_CHUNK": cloze_max_per_chunk_val or "",
        "CLAWLEARN_VALIDATE_FORMAT_RETRY_ENABLE": validate_retry_enable_val or "",
        "CLAWLEARN_VALIDATE_FORMAT_RETRY_MAX": validate_retry_max_val or "",
        "CLAWLEARN_VALIDATE_FORMAT_RETRY_LLM_ENABLE": validate_retry_llm_enable_val or "",
        "CLAWLEARN_CONTENT_PROFILE": content_profile_val or "",
        "CLAWLEARN_CLOZE_DIFFICULTY": cloze_difficulty_val or "",
        "CLAWLEARN_PROMPT_LANG": prompt_lang_val or "",
        "CLAWLEARN_EXTRACT_PROMPT": extract_prompt_env_val or "",
        "CLAWLEARN_EXPLAIN_PROMPT": explain_prompt_env_val or "",
        "CLAWLEARN_OUTPUT_DIR": output_dir_val or "",
        "CLAWLEARN_EXPORT_DIR": export_dir_val or "",
        "CLAWLEARN_LOG_DIR": log_dir_val or "",
        "CLAWLEARN_DEFAULT_DECK_NAME": default_deck_name_val or "",
        "CLAWLEARN_TTS_EDGE_VOICE1": tts_voice1_val or "",
        "CLAWLEARN_TTS_EDGE_VOICE2": tts_voice2_val or "",
        "CLAWLEARN_TTS_EDGE_VOICE3": tts_voice3_val or "",
        "CLAWLEARN_TTS_EDGE_VOICE4": tts_voice4_val or "",
    }
    save_status = deps.save_env_v2(updated, lang=deps.normalize_ui_lang(ui_lang_val))
    reloaded = on_reload_env(ui_lang_val, deps=deps)
    values, reload_status = reloaded[:-1], reloaded[-1]
    return (*values, f"{save_status}\n\n{reload_status}")


def on_reload_env(
    ui_lang_val: Any,
    *,
    deps: ConfigDeps,
) -> tuple[str, ...]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    resolved_env = deps.resolve_env_file()
    env_path = (resolved_env or Path(".env")).resolve()
    try:
        cfg = deps.load_config(env_file=resolved_env)
        cfg_view = deps.load_env_view(cfg, resolved_env)
        status = _reload_status(
            lang=lang,
            tr=deps.tr,
            env_path=env_path,
            found=bool(resolved_env and resolved_env.exists()),
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        try:
            cfg = deps.load_config(env_file=None)
            cfg_view = deps.load_env_view(cfg, None)
        except Exception:
            cfg_view = {}
        status = _reload_status(
            lang=lang,
            tr=deps.tr,
            env_path=env_path,
            found=False,
            error=exc,
        )
    return (*_config_values_from_view(cfg_view), status)
