"""Config file and environment IO helpers for the web UI."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, Optional

from dotenv import dotenv_values

from clawlingua.config import (
    load_config,
    validate_base_config,
    validate_runtime_config,
)

_ENV_LINE_RE = re.compile(r"^\s*(CLAWLINGUA_[A-Z0-9_]+)\s*=\s*(.*)\s*$")

# Keys that the Config tab allows editing. These map directly to
# CLAWLINGUA_* environment variables used by AppConfig.
EDITABLE_ENV_KEYS = [
    # Defaults / language
    "CLAWLINGUA_DEFAULT_SOURCE_LANG",
    "CLAWLINGUA_DEFAULT_TARGET_LANG",
    # LLM (primary)
    "CLAWLINGUA_LLM_BASE_URL",
    "CLAWLINGUA_LLM_API_KEY",
    "CLAWLINGUA_LLM_MODEL",
    "CLAWLINGUA_LLM_TIMEOUT_SECONDS",
    "CLAWLINGUA_LLM_MAX_RETRIES",
    "CLAWLINGUA_LLM_RETRY_BACKOFF_SECONDS",
    "CLAWLINGUA_LLM_REQUEST_SLEEP_SECONDS",
    "CLAWLINGUA_LLM_TEMPERATURE",
    # Translation LLM
    "CLAWLINGUA_TRANSLATE_LLM_BASE_URL",
    "CLAWLINGUA_TRANSLATE_LLM_API_KEY",
    "CLAWLINGUA_TRANSLATE_LLM_MODEL",
    "CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE",
    # Chunk & cloze
    "CLAWLINGUA_CHUNK_MAX_CHARS",
    "CLAWLINGUA_CHUNK_MIN_CHARS",
    "CLAWLINGUA_CHUNK_OVERLAP_SENTENCES",
    "CLAWLINGUA_CLOZE_MAX_SENTENCES",
    "CLAWLINGUA_CLOZE_MIN_CHARS",
    "CLAWLINGUA_CLOZE_MAX_PER_CHUNK",
    "CLAWLINGUA_LLM_CHUNK_BATCH_SIZE",
    "CLAWLINGUA_VALIDATE_FORMAT_RETRY_ENABLE",
    "CLAWLINGUA_VALIDATE_FORMAT_RETRY_MAX",
    "CLAWLINGUA_VALIDATE_FORMAT_RETRY_LLM_ENABLE",
    "CLAWLINGUA_INGEST_SHORT_LINE_MAX_WORDS",
    "CLAWLINGUA_CONTENT_PROFILE",
    "CLAWLINGUA_CLOZE_DIFFICULTY",
    "CLAWLINGUA_PROMPT_LANG",
    "CLAWLINGUA_EXTRACT_PROMPT",
    "CLAWLINGUA_EXPLAIN_PROMPT",
    # Paths & defaults
    "CLAWLINGUA_OUTPUT_DIR",
    "CLAWLINGUA_EXPORT_DIR",
    "CLAWLINGUA_LOG_DIR",
    "CLAWLINGUA_DEFAULT_DECK_NAME",
    # TTS voice slots
    "CLAWLINGUA_TTS_EDGE_VOICE1",
    "CLAWLINGUA_TTS_EDGE_VOICE2",
    "CLAWLINGUA_TTS_EDGE_VOICE3",
    "CLAWLINGUA_TTS_EDGE_VOICE4",
]


def resolve_env_file() -> Optional[Path]:
    """Best-effort resolution of the default .env file."""

    candidate = Path(".env").resolve()
    return candidate if candidate.exists() else None


def read_env_example(path: Path | None = None) -> Dict[str, str]:
    env_example = path or Path("ENV_EXAMPLE.md").resolve()
    if not env_example.exists():
        return {}
    defaults: Dict[str, str] = {}
    for line in env_example.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _ENV_LINE_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        defaults[key] = value.strip()
    return defaults


def load_env_view(
    cfg: Any,
    env_file: Optional[Path],
    *,
    editable_keys: Iterable[str] = EDITABLE_ENV_KEYS,
) -> Dict[str, str]:
    """Build a view of config values for the Config tab.

    Preference order per key:
    - If key is present in .env, use its string value
    - Otherwise, fall back to AppConfig attribute when possible
    - Otherwise, empty string
    """

    file_values: Dict[str, str] = {}
    if env_file is not None and env_file.exists():
        for key, value in dotenv_values(env_file).items():
            if value is not None:
                file_values[key] = str(value)

    view: Dict[str, str] = {}
    for key in editable_keys:
        if key in file_values:
            view[key] = file_values[key]
            continue
        attr_name = key.removeprefix("CLAWLINGUA_").lower()
        if hasattr(cfg, attr_name):
            value = getattr(cfg, attr_name)
            view[key] = "" if value is None else str(value)
        else:
            view[key] = ""
    return view


def save_env(
    updated: Dict[str, str],
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
    editable_keys: Iterable[str] = EDITABLE_ENV_KEYS,
    env_file: Path | None = None,
) -> str:
    """Persist selected config values back to .env and validate."""

    target_env = env_file or resolve_env_file() or Path(".env").resolve()
    original_text: Optional[str] = None
    if target_env.exists():
        original_text = target_env.read_text(encoding="utf-8")
        current = {k: v for k, v in dotenv_values(target_env).items() if v is not None}
    else:
        current = {}

    editable_set = set(editable_keys)
    new_env: Dict[str, str] = {
        k: str(v) for k, v in current.items() if k not in editable_set
    }
    for key in editable_set:
        if key not in updated:
            continue
        value = str(updated.get(key, "")).strip()
        if value:
            new_env[key] = value
        else:
            new_env.pop(key, None)

    target_env.write_text(
        "".join(f"{k}={v}\n" for k, v in sorted(new_env.items())),
        encoding="utf-8",
    )
    try:
        cfg = load_config(env_file=target_env)
        validate_base_config(cfg)
        validate_runtime_config(cfg)
    except Exception as exc:
        if original_text is not None:
            target_env.write_text(original_text, encoding="utf-8")
        else:
            target_env.unlink(missing_ok=True)
        return f"❌ {tr(lang, 'Failed to save config', 'Failed to save config')}: {exc}"

    return f"✅ {tr(lang, 'Config saved and validated.', 'Config saved and validated.')}"
