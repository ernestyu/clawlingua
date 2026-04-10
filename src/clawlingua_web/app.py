"""Local-only web UI for ClawLingua.

This module exposes a thin Gradio-based frontend over the existing
`clawlingua` CLI/pipeline. It does **not** change CLI behavior and is
intended as an optional convenience for users who prefer a browser UI.

Usage (development):

    python -m clawlingua_web.app

This will start a Gradio app bound to 0.0.0.0 by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, Optional

import gradio as gr
import httpx
from dotenv import dotenv_values
from pydantic import ValidationError

from clawlingua.config import (
    load_config,
    validate_base_config,
    validate_runtime_config,
)
from clawlingua.logger import setup_logging
from clawlingua.models.prompt_schema import PromptSpec
from clawlingua.pipeline.build_deck import BuildDeckOptions, run_build_deck
from clawlingua.utils.time import make_run_id, utc_now_iso

logger = logging.getLogger("clawlingua.web")

_SUPPORTED_UI_LANGS = {"en", "zh"}
_ENV_LINE_RE = re.compile(r"^\s*(CLAWLINGUA_[A-Z0-9_]+)\s*=\s*(.*)\s*$")
_PROMPT_DEFAULTS_FILE = Path("./prompts/user_prompt_overrides.json")
_PROMPT_DEFAULTS_VERSION = 1
_ZH_I18N = {
    "UI language": "\u754c\u9762\u8bed\u8a00",
    "Run": "\u8fd0\u884c",
    "Config": "\u914d\u7f6e",
    "Prompt": "\u63d0\u793a\u8bcd",
    "Input file": "\u8f93\u5165\u6587\u4ef6",
    "Deck title (optional)": "\u724c\u7ec4\u540d\u79f0\uff08\u53ef\u9009\uff09",
    "Source language": "\u6e90\u8bed\u8a00",
    "Target language": "\u76ee\u6807\u8bed\u8a00",
    "Content profile": "\u5185\u5bb9\u7c7b\u578b",
    "Difficulty": "\u96be\u5ea6",
    "Max notes (0 = no limit)": "\u6700\u5927 note \u6570\uff080=\u4e0d\u9650\uff09",
    "Maximum notes after dedupe. Empty/0 means no limit.": "\u53bb\u91cd\u540e\u6700\u591a\u751f\u6210\u591a\u5c11 note\u3002\u7a7a\u6216 0 \u8868\u793a\u4e0d\u9650\u5236\u3002",
    "Input char limit": "\u8f93\u5165\u5b57\u7b26\u4e0a\u9650",
    "Only process the first N chars of input. Empty means no limit.": "\u4ec5\u5904\u7406\u8f93\u5165\u524d N \u4e2a\u5b57\u7b26\u3002\u7559\u7a7a\u8868\u793a\u4e0d\u9650\u5236\u3002",
    "Advanced": "\u9ad8\u7ea7\u53c2\u6570",
    "Cloze min chars (override env)": "\u6700\u5c0f\u6316\u7a7a\u957f\u5ea6\uff08\u8986\u76d6 env\uff09",
    "One-run override for CLAWLINGUA_CLOZE_MIN_CHARS.": "\u4ec5\u672c\u6b21\u8fd0\u884c\u8986\u76d6 CLAWLINGUA_CLOZE_MIN_CHARS\u3002",
    "Chunk max chars (override env)": "chunk \u6700\u5927\u5b57\u7b26\uff08\u8986\u76d6 env\uff09",
    "One-run override for CLAWLINGUA_CHUNK_MAX_CHARS.": "\u4ec5\u672c\u6b21\u8fd0\u884c\u8986\u76d6 CLAWLINGUA_CHUNK_MAX_CHARS\u3002",
    "Temperature (override env)": "\u6e29\u5ea6\u53c2\u6570\uff08\u8986\u76d6 env\uff09",
    "0 is more deterministic; higher values are more random.": "0 \u66f4\u786e\u5b9a\uff0c\u9ad8\u503c\u66f4\u968f\u673a\u3002",
    "Save intermediate files": "\u4fdd\u5b58\u4e2d\u95f4\u6587\u4ef6",
    "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.": "\u5c06\u4e2d\u95f4 JSONL/media \u5199\u5165 OUTPUT_DIR/<run_id>\u3002",
    "Continue on error": "\u9047\u9519\u7ee7\u7eed",
    "If enabled, continue processing after per-item failures.": "\u52fe\u9009\u540e\u9047\u5230\u5c40\u90e8\u9519\u8bef\u4ecd\u7ee7\u7eed\u5904\u7406\u540e\u7eed\u5185\u5bb9\u3002",
    "Status": "\u72b6\u6001",
    "Download .apkg": "\u4e0b\u8f7d .apkg",
    "Error": "\u9519\u8bef",
    "No input file provided.": "\u672a\u63d0\u4f9b\u8f93\u5165\u6587\u4ef6\u3002",
    "Run complete": "\u8fd0\u884c\u5b8c\u6210",
    "Running": "\u8fd0\u884c\u4e2d",
    "Completed": "\u5b8c\u6210",
    "Failed": "\u5931\u8d25",
    "Run failed": "\u8fd0\u884c\u5931\u8d25",
    "Recent runs": "\u6700\u8fd1\u8fd0\u884c",
    "### Recent runs": "### \u6700\u8fd1\u8fd0\u884c",
    "Refresh runs": "\u5237\u65b0\u8fd0\u884c\u5217\u8868",
    "Run ID": "Run ID",
    "Started at": "\u5f00\u59cb\u65f6\u95f4",
    "Finished at": "\u7ed3\u675f\u65f6\u95f4",
    "Cards": "\u5361\u7247\u6570",
    "Errors": "\u9519\u8bef\u6570",
    "unknown": "\u672a\u77e5",
    "running": "\u8fd0\u884c\u4e2d",
    "completed": "\u5b8c\u6210",
    "failed": "\u5931\u8d25",
    "Run details": "\u8fd0\u884c\u8be6\u60c5",
    "Title": "\u6807\u9898",
    "Output path": "\u8f93\u51fa\u6587\u4ef6",
    "Last error": "\u6700\u540e\u9519\u8bef",
    "Output file not available yet.": "\u5c1a\u672a\u751f\u6210\u8f93\u51fa\u6587\u4ef6\u3002",
    "No runs found.": "\u6682\u672a\u627e\u5230\u8fd0\u884c\u8bb0\u5f55\u3002",
    "No run selected.": "\u672a\u9009\u62e9\u8fd0\u884c\u8bb0\u5f55\u3002",
    "### Config (.env editor)": "### \u914d\u7f6e\uff08.env \u7f16\u8f91\u5668\uff09",
    "LLM (primary)": "\u4e3b LLM",
    "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).": "OpenAI \u517c\u5bb9\u63a5\u53e3\u57fa\u7840\u5730\u5740\uff08/chat/completions \u4e4b\u524d\u7684\u90e8\u5206\uff0c\u5982 .../v1\uff09\u3002",
    "API key for primary LLM, when required.": "\u4e3b LLM \u7684 API Key\uff08\u5982\u9700\u8981\uff09\u3002",
    "Model name for primary LLM.": "\u4e3b LLM \u7684\u6a21\u578b\u540d\u3002",
    "Request timeout in seconds.": "\u8bf7\u6c42\u8d85\u65f6\uff08\u79d2\uff09\u3002",
    "Default temperature for primary LLM.": "\u4e3b LLM \u9ed8\u8ba4\u6e29\u5ea6\u53c2\u6570\u3002",
    "List models": "\u5217\u51fa\u6a21\u578b",
    "Test": "\u6d4b\u8bd5\u8fde\u901a",
    "Primary LLM status": "\u4e3b LLM \u72b6\u6001",
    "Translation LLM": "\u7ffb\u8bd1 LLM",
    "Optional base URL for translation model.": "\u7ffb\u8bd1\u6a21\u578b\u53ef\u9009\u57fa\u7840\u5730\u5740\u3002",
    "API key for translation LLM, when required.": "\u7ffb\u8bd1 LLM \u7684 API Key\uff08\u5982\u9700\u8981\uff09\u3002",
    "Model name for translation LLM.": "\u7ffb\u8bd1 LLM \u7684\u6a21\u578b\u540d\u3002",
    "Default temperature for translation LLM.": "\u7ffb\u8bd1 LLM \u9ed8\u8ba4\u6e29\u5ea6\u53c2\u6570\u3002",
    "List models (translate)": "\u5217\u51fa\u7ffb\u8bd1\u6a21\u578b",
    "Test (translate)": "\u6d4b\u8bd5\u7ffb\u8bd1\u8fde\u901a",
    "Translation LLM status": "\u7ffb\u8bd1 LLM \u72b6\u6001",
    "Chunk & Cloze": "\u5207\u5757\u4e0e\u6316\u7a7a",
    "Default max chars per chunk.": "\u9ed8\u8ba4\u6bcf\u4e2a chunk \u7684\u6700\u5927\u5b57\u7b26\u6570\u3002",
    "Minimum chars required for cloze text.": "\u6316\u7a7a\u6587\u672c\u6700\u5c0f\u5b57\u7b26\u6570\u3002",
    "Max cards per chunk after dedupe. Empty/0 means unlimited.": "\u53bb\u91cd\u540e\u6bcf\u4e2a chunk \u6700\u591a\u5361\u7247\u6570\u3002\u7a7a\u6216 0 \u8868\u793a\u4e0d\u9650\u5236\u3002",
    "Prompt language for multi-lingual prompts (en/zh).": "\u591a\u8bed\u8a00 prompt \u9009\u62e9\uff08en/zh\uff09\u3002",
    "Paths & defaults": "\u8def\u5f84\u4e0e\u9ed8\u8ba4\u503c",
    "Directory for intermediate run data (JSONL, media).": "\u4e2d\u95f4\u8fd0\u884c\u6570\u636e\u76ee\u5f55\uff08JSONL\u3001media\uff09\u3002",
    "Default directory for exported decks.": "\u9ed8\u8ba4\u724c\u7ec4\u5bfc\u51fa\u76ee\u5f55\u3002",
    "Directory for log files.": "\u65e5\u5fd7\u76ee\u5f55\u3002",
    "Load defaults from ENV_EXAMPLE.md": "\u4ece ENV_EXAMPLE.md \u8f7d\u5165\u9ed8\u8ba4\u503c",
    "Save config": "\u4fdd\u5b58\u914d\u7f6e",
    "Loaded defaults from ENV_EXAMPLE.md (not yet saved).": "\u5df2\u8f7d\u5165 ENV_EXAMPLE.md \u9ed8\u8ba4\u503c\uff08\u5c1a\u672a\u4fdd\u5b58\uff09\u3002",
    "Failed to save config": "\u4fdd\u5b58\u914d\u7f6e\u5931\u8d25",
    "Config saved and validated.": "\u914d\u7f6e\u5df2\u4fdd\u5b58\u5e76\u901a\u8fc7\u6821\u9a8c\u3002",
    "Missing base URL.": "\u7f3a\u5c11 base URL\u3002",
    "Request failed": "\u8bf7\u6c42\u5931\u8d25",
    "HTTP error": "HTTP \u9519\u8bef",
    "Response is not valid JSON.": "\u54cd\u5e94\u4e0d\u662f\u6709\u6548 JSON\u3002",
    "Response JSON has no list field `data`.": "\u54cd\u5e94 JSON \u4e2d\u7f3a\u5c11\u5217\u8868\u5b57\u6bb5 `data`\u3002",
    "Found models": "\u6a21\u578b\u5217\u8868",
    "No model ids found in `data`.": "`data` \u4e2d\u672a\u627e\u5230\u6a21\u578b id\u3002",
    "Connectivity OK": "\u8fde\u901a\u6027\u6b63\u5e38",
    "### Prompt template editor": "### Prompt \u6a21\u677f\u7f16\u8f91\u5668",
    "Prompt file": "Prompt \u6587\u4ef6",
    "Prompt template": "Prompt \u6a21\u677f",
    "Save": "\u4fdd\u5b58",
    "Load default": "\u8f7d\u5165\u9ed8\u8ba4",
    "Prompt status": "Prompt \u72b6\u6001",
    "Prompt template is empty.": "Prompt \u6a21\u677f\u4e3a\u7a7a\u3002",
    "Prompt defaults file missing.": "Prompt \u9ed8\u8ba4\u6a21\u677f\u6587\u4ef6\u4e0d\u5b58\u5728\u3002",
    "Prompt defaults file invalid.": "Prompt \u9ed8\u8ba4\u6a21\u677f\u6587\u4ef6\u683c\u5f0f\u65e0\u6548\u3002",
    "Unsupported prompt defaults version.": "Prompt \u9ed8\u8ba4\u6a21\u677f\u7248\u672c\u4e0d\u53d7\u652f\u6301\u3002",
    "Default template not found.": "\u672a\u627e\u5230\u5f53\u524d\u63d0\u793a\u8bcd\u7684\u9ed8\u8ba4\u6a21\u677f\u3002",
    "Prompt file JSON parse error": "Prompt \u6587\u4ef6 JSON \u89e3\u6790\u5931\u8d25",
    "Schema validation failed": "Schema \u6821\u9a8c\u5931\u8d25",
    "Failed to load prompt file": "\u52a0\u8f7d prompt \u6587\u4ef6\u5931\u8d25",
    "Failed to save prompt file": "\u4fdd\u5b58 prompt \u6587\u4ef6\u5931\u8d25",
    "Not saved because validation failed.": "\u672a\u4fdd\u5b58\uff1a\u6821\u9a8c\u672a\u901a\u8fc7\u3002",
    "Prompt template saved.": "Prompt \u6a21\u677f\u5df2\u4fdd\u5b58\u3002",
    "Prompt template restored from default.": "\u5df2\u4ece\u9ed8\u8ba4\u6a21\u677f\u8fd8\u539f Prompt\u3002",
    "Backup created": "\u5df2\u521b\u5efa\u5907\u4efd",
    "TTS voices (Edge)": "\u8bed\u97f3\u914d\u7f6e\uff08Edge\uff09",
    "Configure 4 voice slots used for random selection.": "\u914d\u7f6e 4 \u4e2a\u8bed\u97f3\u69fd\u4f4d\uff0c\u7528\u4e8e\u968f\u673a\u9009\u62e9\u3002",
    "Voice reference: [Edge TTS Voice Samples](https://tts.travisvn.com/)": "\u5177\u4f53\u7684\u97f3\u8272\u53ef\u4ee5\u53c2\u8003[Edge TTS Voice Samples](https://tts.travisvn.com/)",
}


def _resolve_env_file() -> Optional[Path]:
    """Best-effort resolution of the default .env file.

    For now we simply look at the current working directory and use
    ``./.env`` if it exists. This keeps behavior explicit and avoids
    surprising resolution rules.
    """

    candidate = Path(".env").resolve()
    return candidate if candidate.exists() else None


def _load_app_config() -> Any:
    env_file = _resolve_env_file()
    cfg = load_config(env_file=env_file)
    setup_logging(cfg.log_level, log_dir=cfg.log_dir)
    return cfg


def _normalize_ui_lang(value: str | None) -> str:
    lang = (value or "").strip().lower()
    return lang if lang in _SUPPORTED_UI_LANGS else "en"


def _tr(lang: str, en: str, zh: str) -> str:
    if _normalize_ui_lang(lang) != "zh":
        return en
    return _ZH_I18N.get(en, en)


# Keys that the Config tab allows editing. These map directly to
# CLAWLINGUA_* environment variables used by AppConfig.
_EDITABLE_ENV_KEYS = [
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
    "CLAWLINGUA_INGEST_SHORT_LINE_MAX_WORDS",
    "CLAWLINGUA_CONTENT_PROFILE",
    "CLAWLINGUA_CLOZE_DIFFICULTY",
    "CLAWLINGUA_PROMPT_LANG",
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


def _read_env_example() -> Dict[str, str]:
    env_example = Path("ENV_EXAMPLE.md").resolve()
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


def _load_env_view(cfg: Any, env_file: Optional[Path]) -> Dict[str, str]:
    """Build a view of config values for the Config tab.

    Preference order per key:
    - If key is present in .env, use its string value
    - Otherwise, fall back to AppConfig attribute when possible
    - Otherwise, empty string
    """

    file_values: Dict[str, str] = {}
    if env_file is not None and env_file.exists():
        for k, v in dotenv_values(env_file).items():
            if v is not None:
                file_values[k] = str(v)

    view: Dict[str, str] = {}
    for key in _EDITABLE_ENV_KEYS:
        if key in file_values:
            view[key] = file_values[key]
            continue
        # Fallback: derive from cfg when possible.
        attr_name = key.removeprefix("CLAWLINGUA_").lower()
        if hasattr(cfg, attr_name):
            value = getattr(cfg, attr_name)
            view[key] = "" if value is None else str(value)
        else:
            view[key] = ""
    return view


def _save_env(updated: Dict[str, str]) -> str:
    """Persist selected config values back to .env and validate.

    Behavior:
    - Only CLAWLINGUA_* keys in _EDITABLE_ENV_KEYS are modified.
    - Other keys in existing .env are preserved as-is.
    - Empty string means "remove from .env" so defaults apply.
    - On validation failure, the original .env content is restored.
    """

    env_file = _resolve_env_file() or Path(".env").resolve()
    original_text: Optional[str] = None
    if env_file.exists():
        original_text = env_file.read_text(encoding="utf-8")
        current = {k: v for k, v in dotenv_values(env_file).items() if v is not None}
    else:
        current = {}

    # Preserve non-editable keys
    new_env: Dict[str, str] = {
        k: str(v) for k, v in current.items() if k not in _EDITABLE_ENV_KEYS
    }

    # Apply edits
    for key in _EDITABLE_ENV_KEYS:
        val = updated.get(key)
        if val is None:
            continue
        val_str = str(val).strip()
        if val_str == "":
            # Empty string => drop from .env so defaults apply
            new_env.pop(key, None)
        else:
            new_env[key] = val_str

    # Write new .env content (simple KEY=VALUE lines, no comments)
    lines = [f"{k}={v}\n" for k, v in sorted(new_env.items())]
    env_file.write_text("".join(lines), encoding="utf-8")

    # Validate new config
    try:
        cfg = load_config(env_file=env_file)
        validate_base_config(cfg)
        validate_runtime_config(cfg)
    except Exception as exc:  # pragma: no cover - defensive
        # Roll back on failure
        if original_text is not None:
            env_file.write_text(original_text, encoding="utf-8")
        else:
            env_file.unlink(missing_ok=True)
        return f"❌ Failed to save config: {exc}"

    return "✅ Config saved and validated."


def _run_single_build(
    uploaded_file: Any,
    deck_title: str,
    source_lang: str,
    target_lang: str,
    content_profile: str,
    difficulty: str,
    max_notes: Optional[int],
    input_char_limit: Optional[int],
    cloze_min_chars: Optional[int],
    chunk_max_chars: Optional[int],
    temperature: Optional[float],
    save_intermediate: bool,
    continue_on_error: bool,
) -> Dict[str, Any]:
    """Run a single build using the in-process pipeline.

    Returns a dict suitable for Gradio consumption with keys like
    ``status``, ``run_id``, ``cards_count``, ``errors_count`` and
    ``output_path``.
    """

    if uploaded_file is None:
        return {"status": "error", "message": "No input file provided."}

    cfg = _load_app_config()

    # Persist uploaded file to a temporary location under workspace_root/tmp.
    workspace_root = cfg.workspace_root
    tmp_dir = (workspace_root / "tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # gradio's UploadedFile has attributes name, size, and a path-like object.
    # We re-save it to a deterministic path.
    uploaded_path = Path(uploaded_file.name)
    suffix = uploaded_path.suffix or ""
    safe_name = uploaded_path.stem or "input"
    dst = tmp_dir / f"{safe_name}{suffix}"
    with open(dst, "wb") as f_out:
        f_out.write(uploaded_file.read())

    # Build options based on UI inputs.
    options = BuildDeckOptions(
        input_value=str(dst),
        source_lang=source_lang or None,
        target_lang=target_lang or None,
        content_profile=content_profile or None,
        input_char_limit=input_char_limit,
        deck_name=deck_title or None,
        max_chars=chunk_max_chars,
        cloze_min_chars=cloze_min_chars,
        max_notes=max_notes,
        temperature=temperature,
        cloze_difficulty=difficulty or None,
        save_intermediate=save_intermediate,
        continue_on_error=continue_on_error,
    )

    try:
        result = run_build_deck(cfg, options)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {"status": "error", "message": str(exc)}

    return {
        "status": "ok",
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "output_path": str(result.output_path),
        "cards_count": result.cards_count,
        "errors_count": result.errors_count,
    }


def _safe_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe or "input"


def _to_optional_int(value: Any, *, min_value: int | None = None) -> int | None:
    if value in (None, ""):
        return None
    try:
        num = int(float(value))
    except (TypeError, ValueError):
        return None
    if min_value is not None and num < min_value:
        return None
    return num


def _to_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_timeout_seconds(value: Any, default: float = 20.0) -> float:
    timeout = _to_optional_float(value)
    if timeout is None or timeout <= 0:
        return default
    return timeout


def _materialize_uploaded_file(uploaded_file: Any, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(uploaded_file, (str, Path)):
        src = Path(uploaded_file)
        if not src.exists():
            raise ValueError(f"Uploaded file path does not exist: {src}")
        dst = tmp_dir / f"{_safe_stem(src.stem)}{src.suffix}"
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        return dst

    if isinstance(uploaded_file, dict):
        src_path = uploaded_file.get("path") or uploaded_file.get("name")
        if src_path:
            src = Path(str(src_path))
            if src.exists():
                dst = tmp_dir / f"{_safe_stem(src.stem)}{src.suffix}"
                shutil.copyfile(src, dst)
                return dst

    name = getattr(uploaded_file, "name", "input.txt")
    suffix = Path(str(name)).suffix
    stem = Path(str(name)).stem
    dst = tmp_dir / f"{_safe_stem(stem)}{suffix}"
    if not hasattr(uploaded_file, "read"):
        raise ValueError("Unsupported uploaded file payload.")
    data = uploaded_file.read()
    if isinstance(data, str):
        dst.write_text(data, encoding="utf-8")
    else:
        dst.write_bytes(data)
    return dst


def _run_single_build_v2(
    uploaded_file: Any,
    deck_title: str,
    source_lang: str,
    target_lang: str,
    content_profile: str,
    difficulty: str,
    max_notes: int | None,
    input_char_limit: int | None,
    cloze_min_chars: int | None,
    chunk_max_chars: int | None,
    temperature: float | None,
    save_intermediate: bool,
    continue_on_error: bool,
    prompt_lang: str | None = None,
) -> Dict[str, Any]:
    if uploaded_file is None:
        return {"status": "error", "message": "No input file provided.", "run_id": None}

    cfg = _load_app_config()
    if prompt_lang:
        cfg.prompt_lang = _normalize_ui_lang(prompt_lang)
    workspace_root = cfg.workspace_root
    tmp_dir = (workspace_root / "tmp").resolve()
    try:
        local_input = _materialize_uploaded_file(uploaded_file, tmp_dir)
    except Exception as exc:
        return {"status": "error", "message": str(exc), "run_id": None}

    run_id = make_run_id()
    run_dir = cfg.resolve_path(cfg.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    default_output_path = (cfg.resolve_path(cfg.export_dir) / run_id / "output.apkg").resolve()
    summary_path = run_dir / "run_summary.json"

    source_lang_value = source_lang or cfg.default_source_lang
    target_lang_value = target_lang or cfg.default_target_lang
    profile_value = content_profile or cfg.content_profile
    title_value = (deck_title or "").strip() or local_input.stem
    _write_run_summary(
        summary_path,
        {
            "run_id": run_id,
            "started_at": utc_now_iso(),
            "finished_at": None,
            "status": "running",
            "title": title_value,
            "source_lang": source_lang_value,
            "target_lang": target_lang_value,
            "content_profile": profile_value,
            "cards": 0,
            "errors": 0,
            "output_path": str(default_output_path),
            "env_snapshot": _build_env_snapshot(cfg),
        },
    )

    options = BuildDeckOptions(
        input_value=str(local_input),
        run_id=run_id,
        source_lang=source_lang or None,
        target_lang=target_lang or None,
        content_profile=content_profile or None,
        input_char_limit=input_char_limit,
        deck_name=deck_title or None,
        max_chars=chunk_max_chars,
        cloze_min_chars=cloze_min_chars,
        max_notes=max_notes,
        temperature=temperature,
        cloze_difficulty=difficulty or None,
        save_intermediate=save_intermediate,
        continue_on_error=continue_on_error,
    )
    try:
        result = run_build_deck(cfg, options)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        previous = _read_run_summary(summary_path)
        previous_errors = _as_int(previous.get("errors"), default=0) if previous else 0
        _update_run_summary(
            summary_path,
            {
                "finished_at": utc_now_iso(),
                "status": "failed",
                "errors": max(1, previous_errors),
                "last_error": str(exc),
            },
        )
        return {
            "status": "error",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "message": str(exc),
        }

    output_path = str(result.output_path)
    _update_run_summary(
        summary_path,
        {
            "finished_at": utc_now_iso(),
            "status": "completed",
            "cards": result.cards_count,
            "errors": result.errors_count,
            "output_path": output_path,
            "last_error": None,
        },
    )

    return {
        "status": "ok",
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "output_path": output_path,
        "cards_count": result.cards_count,
        "errors_count": result.errors_count,
    }


def _save_env_v2(updated: Dict[str, str], *, lang: str) -> str:
    env_file = _resolve_env_file() or Path(".env").resolve()
    original_text: Optional[str] = None
    if env_file.exists():
        original_text = env_file.read_text(encoding="utf-8")
        current = {k: v for k, v in dotenv_values(env_file).items() if v is not None}
    else:
        current = {}

    new_env: Dict[str, str] = {k: str(v) for k, v in current.items() if k not in _EDITABLE_ENV_KEYS}
    for key in _EDITABLE_ENV_KEYS:
        if key not in updated:
            continue
        val = str(updated.get(key, "")).strip()
        if val:
            new_env[key] = val
        else:
            new_env.pop(key, None)

    env_file.write_text("".join(f"{k}={v}\n" for k, v in sorted(new_env.items())), encoding="utf-8")
    try:
        cfg = load_config(env_file=env_file)
        validate_base_config(cfg)
        validate_runtime_config(cfg)
    except Exception as exc:
        if original_text is not None:
            env_file.write_text(original_text, encoding="utf-8")
        else:
            env_file.unlink(missing_ok=True)
        return f"❌ {_tr(lang, 'Failed to save config', '保存配置失败')}: {exc}"
    return f"✅ {_tr(lang, 'Config saved and validated.', '配置已保存并通过校验。')}"


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


def _request_models(base_url: str, api_key: str, timeout_seconds: float) -> tuple[str, httpx.Response]:
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


def _list_models_markdown(base_url: str, api_key: str, timeout_seconds: float, *, lang: str) -> str:
    if not _normalize_base_url(base_url):
        return f"⚠️ {_tr(lang, 'Missing base URL.', '缺少 base URL。')}"
    try:
        endpoint, response = _request_models(base_url, api_key, timeout_seconds)
    except ValueError:
        return f"⚠️ {_tr(lang, 'Missing base URL.', '缺少 base URL。')}"
    except httpx.RequestError as exc:
        return f"❌ {_tr(lang, 'Request failed', '请求失败')}: `{exc}`"
    except Exception as exc:
        return f"❌ {_tr(lang, 'Request failed', '请求失败')}: `{exc}`"

    if response.status_code >= 400:
        body = response.text[:500] if response.text else ""
        return (
            f"❌ {_tr(lang, 'HTTP error', 'HTTP 错误')}: **{response.status_code}**\n\n"
            f"- endpoint: `{endpoint}`\n"
            f"- body: `{body}`"
        )

    try:
        payload = response.json()
    except ValueError:
        return f"❌ {_tr(lang, 'Response is not valid JSON.', '响应不是有效 JSON。')}"

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return (
            f"⚠️ {_tr(lang, 'Response JSON has no list field `data`.', '响应 JSON 中缺少列表字段 `data`。')}\n\n"
            f"- endpoint: `{endpoint}`"
        )

    model_ids: list[str] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "")).strip()
        if not model_id or model_id in seen:
            continue
        model_ids.append(model_id)
        seen.add(model_id)

    if not model_ids:
        return (
            f"✅ {_tr(lang, 'Found models', '模型列表')}: **0**\n\n"
            f"- endpoint: `{endpoint}`\n"
            f"- status: `{response.status_code}`\n"
            f"- {_tr(lang, 'No model ids found in `data`.', '`data` 中未找到模型 id。')}"
        )

    lines = [f"- `{model_id}`" for model_id in model_ids]
    return (
        f"✅ {_tr(lang, 'Found models', '模型列表')}: **{len(model_ids)}**\n\n"
        f"- endpoint: `{endpoint}`\n"
        f"- status: `{response.status_code}`\n\n"
        f"{chr(10).join(lines)}"
    )


def _test_models_markdown(base_url: str, api_key: str, timeout_seconds: float, *, lang: str) -> str:
    if not _normalize_base_url(base_url):
        return f"⚠️ {_tr(lang, 'Missing base URL.', '缺少 base URL。')}"
    try:
        endpoint, response = _request_models(base_url, api_key, timeout_seconds)
    except ValueError:
        return f"⚠️ {_tr(lang, 'Missing base URL.', '缺少 base URL。')}"
    except httpx.RequestError as exc:
        return f"❌ {_tr(lang, 'Request failed', '请求失败')}: `{exc}`"
    except Exception as exc:
        return f"❌ {_tr(lang, 'Request failed', '请求失败')}: `{exc}`"
    return (
        f"✅ {_tr(lang, 'Connectivity OK', '连通性正常')}\n\n"
        f"- endpoint: `{endpoint}`\n"
        f"- status: `{response.status_code}`"
    )


def _prompt_file_map(cfg: Any) -> dict[str, Path]:
    return {
        "cloze_prose_beginner": cfg.resolve_path(cfg.prompt_cloze_prose_beginner),
        "cloze_prose_intermediate": cfg.resolve_path(cfg.prompt_cloze_prose_intermediate),
        "cloze_prose_advanced": cfg.resolve_path(cfg.prompt_cloze_prose_advanced),
        "cloze_transcript_beginner": cfg.resolve_path(cfg.prompt_cloze_transcript_beginner),
        "cloze_transcript_intermediate": cfg.resolve_path(cfg.prompt_cloze_transcript_intermediate),
        "cloze_transcript_advanced": cfg.resolve_path(cfg.prompt_cloze_transcript_advanced),
        "cloze_textbook_examples": cfg.resolve_path(cfg.prompt_cloze_textbook),
        # Legacy entry kept to support older deployments.
        "cloze_contextual": cfg.resolve_path(cfg.prompt_cloze),
        "translate_rewrite": cfg.resolve_path(cfg.prompt_translate),
    }


def _prompt_choices(lang: str) -> list[tuple[str, str]]:
    if _normalize_ui_lang(lang) == "zh":
        return [
            ("Prose 初级 (cloze_prose_beginner)", "cloze_prose_beginner"),
            ("Prose 中级 (cloze_prose_intermediate)", "cloze_prose_intermediate"),
            ("Prose 高级 (cloze_prose_advanced)", "cloze_prose_advanced"),
            ("Transcript 初级 (cloze_transcript_beginner)", "cloze_transcript_beginner"),
            ("Transcript 中级 (cloze_transcript_intermediate)", "cloze_transcript_intermediate"),
            ("Transcript 高级 (cloze_transcript_advanced)", "cloze_transcript_advanced"),
            ("教材例句模式 (cloze_textbook_examples)", "cloze_textbook_examples"),
            ("旧版上下文挖空 (cloze_contextual)", "cloze_contextual"),
            ("翻译改写 (translate_rewrite)", "translate_rewrite"),
        ]
    return [
        ("cloze_prose_beginner", "cloze_prose_beginner"),
        ("cloze_prose_intermediate", "cloze_prose_intermediate"),
        ("cloze_prose_advanced", "cloze_prose_advanced"),
        ("cloze_transcript_beginner", "cloze_transcript_beginner"),
        ("cloze_transcript_intermediate", "cloze_transcript_intermediate"),
        ("cloze_transcript_advanced", "cloze_transcript_advanced"),
        ("cloze_textbook_examples", "cloze_textbook_examples"),
        ("cloze_contextual (legacy)", "cloze_contextual"),
        ("translate_rewrite", "translate_rewrite"),
    ]


def _prompt_defaults_path(cfg: Any) -> Path:
    return cfg.resolve_path(_PROMPT_DEFAULTS_FILE)


def _read_prompt_payload(
    prompt_key: str,
    prompt_files: dict[str, Path],
    *,
    lang: str,
) -> tuple[Path | None, dict[str, Any] | None, str]:
    path = prompt_files.get(prompt_key)
    if path is None:
        return None, None, f"❌ {_tr(lang, 'Failed to load prompt file', '加载 prompt 文件失败')}: `{prompt_key}`"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (
            path,
            None,
            f"❌ {_tr(lang, 'Prompt file JSON parse error', 'Prompt 文件 JSON 解析失败')}: {exc.msg} (line {exc.lineno}, col {exc.colno})",
        )
    except Exception as exc:
        return None, None, f"❌ {_tr(lang, 'Failed to load prompt file', '加载 prompt 文件失败')}: `{exc}`"
    if not isinstance(payload, dict):
        return (
            path,
            None,
            f"❌ {_tr(lang, 'Prompt file JSON parse error', 'Prompt 文件 JSON 解析失败')}: root must be a JSON object.",
        )
    return path, payload, ""


def _resolve_template_for_lang(value: Any, *, lang: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""

    order = [lang, "zh", "en"]
    seen: set[str] = set()
    for key in order:
        if key in seen:
            continue
        seen.add(key)
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            return text
    for text in value.values():
        if isinstance(text, str) and text.strip():
            return text
    return ""


def _load_prompt_template(
    prompt_key: str,
    prompt_files: dict[str, Path],
    *,
    lang: str,
) -> tuple[str, str]:
    _path, payload, msg = _read_prompt_payload(prompt_key, prompt_files, lang=lang)
    if payload is None:
        return "", msg
    ok, validation_msg = _validate_prompt_payload(payload, lang=lang)
    if not ok:
        return "", validation_msg
    template = _resolve_template_for_lang(payload.get("user_prompt_template"), lang=lang)
    return template, ""


def _format_prompt_validation_error(exc: ValidationError) -> str:
    errors = exc.errors()
    lines: list[str] = []
    for err in errors[:5]:
        loc = ".".join(str(piece) for piece in err.get("loc", []))
        msg = str(err.get("msg", "invalid"))
        lines.append(f"- `{loc}`: {msg}" if loc else f"- {msg}")
    if len(errors) > 5:
        lines.append(f"- ... ({len(errors) - 5} more)")
    return "\n".join(lines)


def _validate_prompt_payload(payload: dict[str, Any], *, lang: str) -> tuple[bool, str]:
    try:
        PromptSpec.model_validate(payload)
    except ValidationError as exc:
        return False, f"❌ {_tr(lang, 'Schema validation failed', 'Schema 校验失败')}\n\n{_format_prompt_validation_error(exc)}"
    return True, ""


def _set_user_prompt_template(payload: dict[str, Any], *, lang: str, template: str) -> None:
    current = payload.get("user_prompt_template")
    mapping: dict[str, str] = {}
    if isinstance(current, str):
        text = current.strip()
        if text:
            mapping["en"] = text
            mapping["zh"] = text
    elif isinstance(current, dict):
        for key, value in current.items():
            if isinstance(value, str) and value.strip():
                mapping[str(key)] = value

    mapping[lang] = template
    if "en" not in mapping and "zh" in mapping:
        mapping["en"] = mapping["zh"]
    if "zh" not in mapping and "en" in mapping:
        mapping["zh"] = mapping["en"]
    payload["user_prompt_template"] = mapping


def _write_prompt_payload(path: Path, payload: dict[str, Any], *, lang: str) -> tuple[bool, str]:
    ok, msg = _validate_prompt_payload(payload, lang=lang)
    if not ok:
        return False, f"{msg}\n\n⚠️ {_tr(lang, 'Not saved because validation failed.', '未保存：校验未通过。')}"
    backup = path.with_suffix(path.suffix + ".bak")
    try:
        if path.exists():
            shutil.copyfile(path, backup)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        return False, f"❌ {_tr(lang, 'Failed to save prompt file', '保存 prompt 文件失败')}: `{exc}`"
    return True, f"- file: `{path}`\n- {_tr(lang, 'Backup created', '已创建备份')}: `{backup}`"


def _load_prompt_defaults(
    defaults_path: Path,
    *,
    lang: str,
) -> tuple[dict[str, Any] | None, str]:
    if not defaults_path.exists():
        return None, f"❌ {_tr(lang, 'Prompt defaults file missing.', 'Prompt 默认模板文件不存在。')}: `{defaults_path}`"
    try:
        payload = json.loads(defaults_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (
            None,
            f"❌ {_tr(lang, 'Prompt defaults file invalid.', 'Prompt 默认模板文件格式无效。')}: {exc.msg} (line {exc.lineno}, col {exc.colno})",
        )
    except Exception as exc:
        return None, f"❌ {_tr(lang, 'Prompt defaults file invalid.', 'Prompt 默认模板文件格式无效。')}: `{exc}`"
    if not isinstance(payload, dict):
        return None, f"❌ {_tr(lang, 'Prompt defaults file invalid.', 'Prompt 默认模板文件格式无效。')}"
    # v1 structure:
    # {
    #   "version": 1,
    #   "templates": { "prompt_key": {"en": "...", "zh": "..."} }
    # }
    if "version" in payload or "templates" in payload:
        version_raw = payload.get("version", _PROMPT_DEFAULTS_VERSION)
        try:
            version = int(str(version_raw).strip())
        except (TypeError, ValueError):
            return None, f"❌ {_tr(lang, 'Prompt defaults file invalid.', 'Prompt 默认模板文件格式无效。')}"
        if version != _PROMPT_DEFAULTS_VERSION:
            return (
                None,
                f"❌ {_tr(lang, 'Unsupported prompt defaults version.', 'Prompt 默认模板版本不受支持。')}: `{version}`",
            )
        templates = payload.get("templates")
        if not isinstance(templates, dict):
            return None, f"❌ {_tr(lang, 'Prompt defaults file invalid.', 'Prompt 默认模板文件格式无效。')}"
        return templates, ""

    # Legacy structure (no version): { "prompt_key": {"en": "...", "zh": "..."} }
    return payload, ""


def _default_template_for_lang(
    defaults_payload: dict[str, Any],
    *,
    prompt_key: str,
    lang: str,
) -> str:
    entry = defaults_payload.get(prompt_key)
    return _resolve_template_for_lang(entry, lang=lang)


@dataclass
class RunInfo:
    run_id: str
    started_at: str
    finished_at: str | None
    title: str
    source_lang: str
    target_lang: str
    content_profile: str
    status: str
    cards: int
    errors: int
    output_path: str | None
    last_error: str | None = None


def _as_str(value: Any, *, default: str = "") -> str:
    if isinstance(value, str):
        text = value.strip()
        return text if text else default
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_iso_datetime(value: str | None) -> datetime | None:
    text = _as_str(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _normalize_run_status(value: Any) -> str:
    status = _as_str(value).lower()
    if status in {"running", "completed", "failed", "unknown"}:
        return status
    return "unknown"


def _status_text(lang: str, status: str) -> str:
    normalized = _normalize_run_status(status)
    return _tr(lang, normalized, normalized)


def _build_env_snapshot(cfg: Any) -> dict[str, str]:
    return {
        "CLAWLINGUA_LLM_MODEL": _as_str(getattr(cfg, "llm_model", "")),
        "CLAWLINGUA_TRANSLATE_LLM_MODEL": _as_str(getattr(cfg, "translate_llm_model", "")),
        "CLAWLINGUA_PROMPT_LANG": _as_str(getattr(cfg, "prompt_lang", "")),
        "CLAWLINGUA_MATERIAL_PROFILE": _as_str(getattr(cfg, "material_profile", "")),
        "CLAWLINGUA_LEARNING_MODE": _as_str(getattr(cfg, "learning_mode", "")),
    }


def _write_run_summary(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:  # pragma: no cover - defensive
        logger.exception("failed to write run summary | path=%s", path)


def _read_run_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _update_run_summary(path: Path, updates: dict[str, Any]) -> None:
    payload = _read_run_summary(path)
    payload.update(updates)
    _write_run_summary(path, payload)


def _resolve_output_path(cfg: Any, run_dir: Path, output_path: Any) -> Path | None:
    text = _as_str(output_path)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    workspace_candidate = (cfg.workspace_root / path).resolve()
    run_dir_candidate = (run_dir / path).resolve()
    if workspace_candidate.exists() or not run_dir_candidate.exists():
        return workspace_candidate
    return run_dir_candidate


def _run_started_sort_key(value: str) -> float:
    dt = _parse_iso_datetime(value)
    return dt.timestamp() if dt is not None else 0.0


def _run_info_from_dir(cfg: Any, run_dir: Path) -> RunInfo:
    run_id = run_dir.name
    summary = _read_run_summary(run_dir / "run_summary.json")
    fallback_started = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()

    started_at = _as_str(summary.get("started_at"), default=fallback_started)
    finished_at_text = _as_str(summary.get("finished_at"))
    finished_at = finished_at_text or None
    status = _normalize_run_status(summary.get("status"))
    title = _as_str(summary.get("title"), default=run_id)
    source_lang = _as_str(summary.get("source_lang"))
    target_lang = _as_str(summary.get("target_lang"))
    content_profile = _as_str(summary.get("content_profile"))
    cards = max(0, _as_int(summary.get("cards"), default=0))
    errors = max(0, _as_int(summary.get("errors"), default=0))
    output_path_resolved = _resolve_output_path(cfg, run_dir, summary.get("output_path"))
    output_path_text = str(output_path_resolved) if output_path_resolved is not None else None
    output_exists = bool(output_path_resolved and output_path_resolved.exists())
    last_error = _as_str(summary.get("last_error")) or None

    if status == "completed" and not output_exists:
        status = "failed"
    if status == "running" and finished_at is not None:
        status = "failed"

    return RunInfo(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        title=title,
        source_lang=source_lang,
        target_lang=target_lang,
        content_profile=content_profile,
        status=status,
        cards=cards,
        errors=errors,
        output_path=output_path_text,
        last_error=last_error,
    )


def _scan_runs(cfg: Any, *, limit: int = 30) -> list[RunInfo]:
    runs_root = cfg.resolve_path(cfg.output_dir)
    if not runs_root.exists() or not runs_root.is_dir():
        return []

    infos: list[RunInfo] = []
    for entry in runs_root.iterdir():
        if not entry.is_dir():
            continue
        infos.append(_run_info_from_dir(cfg, entry))

    infos.sort(key=lambda item: _run_started_sort_key(item.started_at), reverse=True)
    max_items = max(0, int(limit))
    if max_items:
        infos = infos[:max_items]
    return infos


def _run_choice_label(info: RunInfo, *, lang: str) -> str:
    started = info.started_at or "-"
    title = info.title or "-"
    return f"{info.run_id} | {started} | {_status_text(lang, info.status)} | {title}"


def _load_run_detail(run_id: str | None, cfg: Any, *, lang: str) -> tuple[str, str | None]:
    selected = _as_str(run_id)
    if not selected:
        return _tr(lang, "No run selected.", "No run selected."), None

    run_dir = cfg.resolve_path(cfg.output_dir) / selected
    if not run_dir.exists() or not run_dir.is_dir():
        return _tr(lang, "No run selected.", "No run selected."), None

    info = _run_info_from_dir(cfg, run_dir)
    download_path = None
    if info.output_path:
        candidate = Path(info.output_path)
        if candidate.exists():
            download_path = str(candidate)

    lines = [
        f"### {_tr(lang, 'Run details', '运行详情')}",
        f"- {_tr(lang, 'Run ID', 'Run ID')}: `{info.run_id}`",
        f"- {_tr(lang, 'Status', '状态')}: **{_status_text(lang, info.status)}**",
        f"- {_tr(lang, 'Started at', '开始时间')}: `{info.started_at or '-'}`",
        f"- {_tr(lang, 'Finished at', '结束时间')}: `{info.finished_at or '-'}`",
        f"- {_tr(lang, 'Title', '标题')}: `{info.title or '-'}`",
        f"- {_tr(lang, 'Source language', '源语言')}: `{info.source_lang or '-'}`",
        f"- {_tr(lang, 'Target language', '目标语言')}: `{info.target_lang or '-'}`",
        f"- {_tr(lang, 'Content profile', '内容类型')}: `{info.content_profile or '-'}`",
        f"- {_tr(lang, 'Cards', '卡片数')}: **{info.cards}**",
        f"- {_tr(lang, 'Errors', '错误数')}: **{info.errors}**",
        f"- {_tr(lang, 'Output path', '输出文件')}: `{info.output_path or '-'}`",
    ]
    if info.last_error:
        lines.append(f"- {_tr(lang, 'Last error', '最后错误')}: `{info.last_error}`")
    if download_path is None:
        lines.append(f"- {_tr(lang, 'Output file not available yet.', '尚未生成输出文件。')}")
    return "\n".join(lines), download_path


def _refresh_recent_runs(cfg: Any, *, lang: str, preferred_run_id: str | None = None) -> tuple[Any, str, str | None]:
    runs = _scan_runs(cfg, limit=30)
    if not runs:
        detail = _tr(lang, "No runs found.", "No runs found.")
        return gr.update(choices=[], value=None), detail, None

    choices = [(_run_choice_label(run, lang=lang), run.run_id) for run in runs]
    run_ids = {run.run_id for run in runs}
    selected = preferred_run_id if preferred_run_id in run_ids else runs[0].run_id
    detail, download_path = _load_run_detail(selected, cfg, lang=lang)
    return gr.update(choices=choices, value=selected), detail, download_path


def build_interface() -> gr.Blocks:
    """Construct the Gradio Blocks interface.

    Two tabs:
    - Run: upload file + per-run overrides + download link
    - Config: (reserved for future work, not implemented yet)
    """

    cfg = _load_app_config()
    env_file = _resolve_env_file()
    cfg_view = _load_env_view(cfg, env_file)
    prompt_files = _prompt_file_map(cfg)
    prompt_defaults_file = _prompt_defaults_path(cfg)
    initial_ui_lang = _normalize_ui_lang(getattr(cfg, "prompt_lang", "en"))
    initial_prompt_key = (
        "cloze_prose_intermediate"
        if "cloze_prose_intermediate" in prompt_files
        else next(iter(prompt_files))
    )
    initial_prompt_text, initial_prompt_status = _load_prompt_template(
        initial_prompt_key, prompt_files, lang=initial_ui_lang
    )
    initial_runs = _scan_runs(cfg, limit=30)
    if initial_runs:
        initial_run_choices = [(_run_choice_label(run, lang=initial_ui_lang), run.run_id) for run in initial_runs]
        initial_run_selected = initial_runs[0].run_id
        initial_run_detail, initial_run_download = _load_run_detail(initial_run_selected, cfg, lang=initial_ui_lang)
    else:
        initial_run_choices = []
        initial_run_selected = None
        initial_run_detail = _tr(initial_ui_lang, "No runs found.", "No runs found.")
        initial_run_download = None

    with gr.Blocks(title="ClawLingua Web UI") as demo:
        with gr.Row():
            ui_lang = gr.Dropdown(
                choices=[("English", "en"), ("中文", "zh")],
                value=initial_ui_lang,
                label=_tr(initial_ui_lang, "UI language", "界面语言"),
                scale=1,
            )
        title_md = gr.Markdown(
            _tr(
                initial_ui_lang,
                "# ClawLingua Web UI\nLocal deck builder for text learning.",
                "# ClawLingua Web UI\n本地化文本学习牌组生成器。",
            )
        )

        with gr.Tab(_tr(initial_ui_lang, "Run", "运行")) as run_tab:
            with gr.Row():
                input_file = gr.File(
                    label=_tr(initial_ui_lang, "Input file", "输入文件"),
                    file_types=[".txt", ".md", ".markdown", ".epub"],
                    file_count="single",
                )
                deck_title = gr.Textbox(label=_tr(initial_ui_lang, "Deck title (optional)", "牌组名称（可选）"))

            with gr.Row():
                source_lang = gr.Dropdown(
                    choices=["en", "zh", "ja", "de", "fr"],
                    value=cfg.default_source_lang,
                    label=_tr(initial_ui_lang, "Source language", "源语言"),
                )
                target_lang = gr.Dropdown(
                    choices=["zh", "en", "ja", "de", "fr"],
                    value=cfg.default_target_lang,
                    label=_tr(initial_ui_lang, "Target language", "目标语言"),
                )
                content_profile = gr.Dropdown(
                    choices=["prose_article", "transcript_dialogue", "textbook_examples"],
                    value=cfg.content_profile,
                    label=_tr(initial_ui_lang, "Content profile", "内容类型"),
                )
                difficulty = gr.Dropdown(
                    choices=["beginner", "intermediate", "advanced"],
                    value=cfg.cloze_difficulty,
                    label=_tr(initial_ui_lang, "Difficulty", "难度"),
                )

            with gr.Row():
                max_notes = gr.Number(
                    label=_tr(initial_ui_lang, "Max notes (0 = no limit)", "最大 note 数（0=不限）"),
                    info=_tr(
                        initial_ui_lang,
                        "Maximum notes after dedupe. Empty/0 means no limit.",
                        "去重后最多生成多少 note。空或 0 表示不限制。",
                    ),
                    value=None,
                    precision=0,
                )
                input_char_limit = gr.Number(
                    label=_tr(initial_ui_lang, "Input char limit", "输入字符上限"),
                    info=_tr(
                        initial_ui_lang,
                        "Only process the first N chars of input. Empty means no limit.",
                        "仅处理输入前 N 个字符。留空表示不限制。",
                    ),
                    value=None,
                    precision=0,
                )

            with gr.Accordion(_tr(initial_ui_lang, "Advanced", "高级参数"), open=False) as run_advanced:
                cloze_min_chars = gr.Number(
                    label=_tr(initial_ui_lang, "Cloze min chars (override env)", "最小挖空长度（覆盖 env）"),
                    info=_tr(
                        initial_ui_lang,
                        "One-run override for CLAWLINGUA_CLOZE_MIN_CHARS.",
                        "仅本次运行覆盖 CLAWLINGUA_CLOZE_MIN_CHARS。",
                    ),
                    value=cfg.cloze_min_chars,
                    precision=0,
                )
                chunk_max_chars = gr.Number(
                    label=_tr(initial_ui_lang, "Chunk max chars (override env)", "chunk 最大字符（覆盖 env）"),
                    info=_tr(
                        initial_ui_lang,
                        "One-run override for CLAWLINGUA_CHUNK_MAX_CHARS.",
                        "仅本次运行覆盖 CLAWLINGUA_CHUNK_MAX_CHARS。",
                    ),
                    value=cfg.chunk_max_chars,
                    precision=0,
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=cfg.llm_temperature,
                    step=0.05,
                    label=_tr(initial_ui_lang, "Temperature (override env)", "温度参数（覆盖 env）"),
                    info=_tr(initial_ui_lang, "0 is more deterministic; higher values are more random.", "0 更确定，高值更随机。"),
                )
                save_intermediate = gr.Checkbox(
                    label=_tr(initial_ui_lang, "Save intermediate files", "保存中间文件"),
                    info=_tr(
                        initial_ui_lang,
                        "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.",
                        "将中间 JSONL/media 写入 OUTPUT_DIR/<run_id>。",
                    ),
                    value=cfg.save_intermediate,
                )
                continue_on_error = gr.Checkbox(
                    label=_tr(initial_ui_lang, "Continue on error", "遇错继续"),
                    info=_tr(
                        initial_ui_lang,
                        "If enabled, continue processing after per-item failures.",
                        "勾选后遇到局部错误仍继续处理后续内容。",
                    ),
                    value=False,
                )

            run_button = gr.Button(_tr(initial_ui_lang, "Run", "Run"))

            run_status = gr.Markdown(label=_tr(initial_ui_lang, "Status", "Status"))
            output_file = gr.File(label=_tr(initial_ui_lang, "Download .apkg", "Download .apkg"), interactive=False)
            recent_runs_heading = gr.Markdown(_tr(initial_ui_lang, "### Recent runs", "### Recent runs"))
            with gr.Row():
                refresh_runs_button = gr.Button(_tr(initial_ui_lang, "Refresh runs", "Refresh runs"))
                run_selector = gr.Dropdown(
                    choices=initial_run_choices,
                    value=initial_run_selected,
                    label=_tr(initial_ui_lang, "Run ID", "Run ID"),
                )
            run_detail = gr.Markdown(value=initial_run_detail)
            run_download_file = gr.File(
                label=_tr(initial_ui_lang, "Download .apkg", "Download .apkg"),
                interactive=False,
                value=initial_run_download,
            )

            def _on_run_start(ui_lang_val: str) -> tuple[str, None]:
                lang = _normalize_ui_lang(ui_lang_val)
                return _tr(lang, "Running", "Running"), None

            def _on_run(
                file_obj,
                deck_title_val,
                src,
                tgt,
                profile,
                diff,
                max_notes_val,
                input_limit_val,
                cloze_min_val,
                chunk_max_val,
                temperature_val,
                save_inter_val,
                continue_on_error_val,
                ui_lang_val,
            ):
                lang = _normalize_ui_lang(ui_lang_val)
                result = _run_single_build_v2(
                    uploaded_file=file_obj,
                    deck_title=deck_title_val or "",
                    source_lang=src,
                    target_lang=tgt,
                    content_profile=profile,
                    difficulty=diff,
                    max_notes=_to_optional_int(max_notes_val, min_value=1),
                    input_char_limit=_to_optional_int(input_limit_val, min_value=1),
                    cloze_min_chars=_to_optional_int(cloze_min_val, min_value=0),
                    chunk_max_chars=_to_optional_int(chunk_max_val, min_value=1),
                    temperature=_to_optional_float(temperature_val),
                    save_intermediate=bool(save_inter_val),
                    continue_on_error=bool(continue_on_error_val),
                    prompt_lang=lang,
                )
                cfg_now = _load_app_config()
                run_id = _as_str(result.get("run_id")) or None
                selector_update, detail_md, history_download = _refresh_recent_runs(
                    cfg_now,
                    lang=lang,
                    preferred_run_id=run_id,
                )

                if result.get("status") != "ok":
                    msg = result.get("message") or "Unknown error"
                    run_line = f"- run_id: `{run_id}`\n" if run_id else ""
                    status_md = f"{_tr(lang, 'Failed', 'Failed')}\n\n{run_line}- {_tr(lang, 'Error', 'Error')}: `{msg}`"
                    return status_md, None, selector_update, detail_md, history_download

                cards = result["cards_count"]
                errors = result["errors_count"]
                out_path = result["output_path"]
                status_md = (
                    f"{_tr(lang, 'Completed', 'Completed')}\n\n"
                    f"- run_id: `{run_id}`\n"
                    f"- cards: **{cards}**\n"
                    f"- errors: **{errors}**\n"
                    f"- output: `{out_path}`"
                )
                return status_md, out_path, selector_update, detail_md, history_download

            def _on_refresh_runs(ui_lang_val: str, selected_run_id: str | None) -> tuple[Any, str, str | None]:
                lang = _normalize_ui_lang(ui_lang_val)
                cfg_now = _load_app_config()
                return _refresh_recent_runs(cfg_now, lang=lang, preferred_run_id=selected_run_id)

            def _on_run_selected(run_id_val: str | None, ui_lang_val: str) -> tuple[str, str | None]:
                lang = _normalize_ui_lang(ui_lang_val)
                cfg_now = _load_app_config()
                return _load_run_detail(run_id_val, cfg_now, lang=lang)

            run_button.click(
                _on_run_start,
                inputs=[ui_lang],
                outputs=[run_status, output_file],
                queue=False,
            ).then(
                _on_run,
                inputs=[
                    input_file,
                    deck_title,
                    source_lang,
                    target_lang,
                    content_profile,
                    difficulty,
                    max_notes,
                    input_char_limit,
                    cloze_min_chars,
                    chunk_max_chars,
                    temperature,
                    save_intermediate,
                    continue_on_error,
                    ui_lang,
                ],
                outputs=[run_status, output_file, run_selector, run_detail, run_download_file],
            )

            refresh_runs_button.click(
                _on_refresh_runs,
                inputs=[ui_lang, run_selector],
                outputs=[run_selector, run_detail, run_download_file],
            )

            run_selector.change(
                _on_run_selected,
                inputs=[run_selector, ui_lang],
                outputs=[run_detail, run_download_file],
            )
        with gr.Tab(_tr(initial_ui_lang, "Config", "配置")) as config_tab:
            config_heading = gr.Markdown(_tr(initial_ui_lang, "### Config (.env editor)", "### 配置（.env 编辑器）"))

            with gr.Accordion(_tr(initial_ui_lang, "LLM (primary)", "主 LLM"), open=True) as llm_accordion:
                llm_base_url = gr.Textbox(
                    label="CLAWLINGUA_LLM_BASE_URL",
                    value=cfg_view.get("CLAWLINGUA_LLM_BASE_URL", ""),
                    info=_tr(
                        initial_ui_lang,
                        "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).",
                        "OpenAI 兼容接口基础地址（/chat/completions 之前的部分，如 .../v1）。",
                    ),
                )
                llm_api_key = gr.Textbox(
                    label="CLAWLINGUA_LLM_API_KEY",
                    value=cfg_view.get("CLAWLINGUA_LLM_API_KEY", ""),
                    type="password",
                    info=_tr(initial_ui_lang, "API key for primary LLM, when required.", "主 LLM 的 API Key（如需要）。"),
                )
                llm_model = gr.Textbox(
                    label="CLAWLINGUA_LLM_MODEL",
                    value=cfg_view.get("CLAWLINGUA_LLM_MODEL", ""),
                    info=_tr(initial_ui_lang, "Model name for primary LLM.", "主 LLM 的模型名。"),
                )
                llm_timeout = gr.Textbox(
                    label="CLAWLINGUA_LLM_TIMEOUT_SECONDS",
                    value=cfg_view.get("CLAWLINGUA_LLM_TIMEOUT_SECONDS", "120"),
                    info=_tr(initial_ui_lang, "Request timeout in seconds.", "请求超时（秒）。"),
                )
                llm_temperature_env = gr.Textbox(
                    label="CLAWLINGUA_LLM_TEMPERATURE",
                    value=cfg_view.get("CLAWLINGUA_LLM_TEMPERATURE", "0.2"),
                    info=_tr(initial_ui_lang, "Default temperature for primary LLM.", "主 LLM 默认温度参数。"),
                )
                with gr.Row():
                    llm_list_models_btn = gr.Button(_tr(initial_ui_lang, "List models", "列出模型"))
                    llm_test_btn = gr.Button(_tr(initial_ui_lang, "Test", "测试连通"))
                llm_status = gr.Markdown(label=_tr(initial_ui_lang, "Primary LLM status", "主 LLM 状态"))

            with gr.Accordion(_tr(initial_ui_lang, "Translation LLM", "翻译 LLM"), open=False) as translate_accordion:
                translate_base_url = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_BASE_URL",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_BASE_URL", ""),
                    info=_tr(initial_ui_lang, "Optional base URL for translation model.", "翻译模型可选基础地址。"),
                )
                translate_api_key = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_API_KEY",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_API_KEY", ""),
                    type="password",
                    info=_tr(initial_ui_lang, "API key for translation LLM, when required.", "翻译 LLM 的 API Key（如需要）。"),
                )
                translate_model = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_MODEL",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_MODEL", ""),
                    info=_tr(initial_ui_lang, "Model name for translation LLM.", "翻译 LLM 的模型名。"),
                )
                translate_temperature = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE", ""),
                    info=_tr(initial_ui_lang, "Default temperature for translation LLM.", "翻译 LLM 默认温度参数。"),
                )
                with gr.Row():
                    translate_list_models_btn = gr.Button(_tr(initial_ui_lang, "List models (translate)", "列出翻译模型"))
                    translate_test_btn = gr.Button(_tr(initial_ui_lang, "Test (translate)", "测试翻译连通"))
                translate_status = gr.Markdown(label=_tr(initial_ui_lang, "Translation LLM status", "翻译 LLM 状态"))

            with gr.Accordion(_tr(initial_ui_lang, "Chunk & Cloze", "切块与挖空"), open=False) as chunk_accordion:
                chunk_max_chars_env = gr.Textbox(
                    label="CLAWLINGUA_CHUNK_MAX_CHARS",
                    value=cfg_view.get("CLAWLINGUA_CHUNK_MAX_CHARS", "1800"),
                    info=_tr(initial_ui_lang, "Default max chars per chunk.", "默认每个 chunk 的最大字符数。"),
                )
                chunk_min_chars_env = gr.Textbox(
                    label="CLAWLINGUA_CHUNK_MIN_CHARS",
                    value=cfg_view.get("CLAWLINGUA_CHUNK_MIN_CHARS", "120"),
                )
                cloze_min_chars_env = gr.Textbox(
                    label="CLAWLINGUA_CLOZE_MIN_CHARS",
                    value=cfg_view.get("CLAWLINGUA_CLOZE_MIN_CHARS", "0"),
                    info=_tr(initial_ui_lang, "Minimum chars required for cloze text.", "挖空文本最小字符数。"),
                )
                cloze_max_per_chunk_env = gr.Textbox(
                    label="CLAWLINGUA_CLOZE_MAX_PER_CHUNK",
                    value=cfg_view.get("CLAWLINGUA_CLOZE_MAX_PER_CHUNK", ""),
                    info=_tr(
                        initial_ui_lang,
                        "Max cards per chunk after dedupe. Empty/0 means unlimited.",
                        "去重后每个 chunk 最多卡片数。空或 0 表示不限制。",
                    ),
                )
                content_profile_env = gr.Textbox(
                    label="CLAWLINGUA_CONTENT_PROFILE",
                    value=cfg_view.get("CLAWLINGUA_CONTENT_PROFILE", "prose_article"),
                )
                cloze_difficulty_env = gr.Textbox(
                    label="CLAWLINGUA_CLOZE_DIFFICULTY",
                    value=cfg_view.get("CLAWLINGUA_CLOZE_DIFFICULTY", "intermediate"),
                )
                prompt_lang_env = gr.Textbox(
                    label="CLAWLINGUA_PROMPT_LANG",
                    value=cfg_view.get("CLAWLINGUA_PROMPT_LANG", "zh"),
                    info=_tr(initial_ui_lang, "Prompt language for multi-lingual prompts (en/zh).", "多语言 prompt 选择（en/zh）。"),
                )

            with gr.Accordion(_tr(initial_ui_lang, "Paths & defaults", "路径与默认值"), open=False) as paths_accordion:
                output_dir_env = gr.Textbox(
                    label="CLAWLINGUA_OUTPUT_DIR",
                    value=cfg_view.get("CLAWLINGUA_OUTPUT_DIR", "./runs"),
                    info=_tr(initial_ui_lang, "Directory for intermediate run data (JSONL, media).", "中间运行数据目录（JSONL、media）。"),
                )
                export_dir_env = gr.Textbox(
                    label="CLAWLINGUA_EXPORT_DIR",
                    value=cfg_view.get("CLAWLINGUA_EXPORT_DIR", "./outputs"),
                    info=_tr(initial_ui_lang, "Default directory for exported decks.", "默认牌组导出目录。"),
                )
                log_dir_env = gr.Textbox(
                    label="CLAWLINGUA_LOG_DIR",
                    value=cfg_view.get("CLAWLINGUA_LOG_DIR", "./logs"),
                    info=_tr(initial_ui_lang, "Directory for log files.", "日志目录。"),
                )
                default_deck_name_env = gr.Textbox(
                    label="CLAWLINGUA_DEFAULT_DECK_NAME",
                    value=cfg_view.get("CLAWLINGUA_DEFAULT_DECK_NAME", cfg.default_deck_name),
                )

            with gr.Accordion(_tr(initial_ui_lang, "TTS voices (Edge)", "语音配置（Edge）"), open=False) as tts_accordion:
                tts_hint_md = gr.Markdown(
                    _tr(
                        initial_ui_lang,
                        "Voice reference: [Edge TTS Voice Samples](https://tts.travisvn.com/)",
                        "具体的音色可以参考[Edge TTS Voice Samples](https://tts.travisvn.com/)",
                    )
                )
                tts_voice1_env = gr.Textbox(
                    label="CLAWLINGUA_TTS_EDGE_VOICE1",
                    value=cfg_view.get("CLAWLINGUA_TTS_EDGE_VOICE1", ""),
                    info=_tr(
                        initial_ui_lang,
                        "Configure 4 voice slots used for random selection.",
                        "配置 4 个语音槽位，用于随机选择。",
                    ),
                )
                tts_voice2_env = gr.Textbox(
                    label="CLAWLINGUA_TTS_EDGE_VOICE2",
                    value=cfg_view.get("CLAWLINGUA_TTS_EDGE_VOICE2", ""),
                )
                tts_voice3_env = gr.Textbox(
                    label="CLAWLINGUA_TTS_EDGE_VOICE3",
                    value=cfg_view.get("CLAWLINGUA_TTS_EDGE_VOICE3", ""),
                )
                tts_voice4_env = gr.Textbox(
                    label="CLAWLINGUA_TTS_EDGE_VOICE4",
                    value=cfg_view.get("CLAWLINGUA_TTS_EDGE_VOICE4", ""),
                )

            with gr.Row():
                load_defaults_btn = gr.Button(_tr(initial_ui_lang, "Load defaults from ENV_EXAMPLE.md", "从 ENV_EXAMPLE.md 载入默认值"))
                save_config_btn = gr.Button(_tr(initial_ui_lang, "Save config", "保存配置"))
            save_config_status = gr.Markdown()

            def _on_list_models(base_url: str, api_key: str, timeout_raw: Any, ui_lang_val: str) -> str:
                return _list_models_markdown(
                    base_url=base_url,
                    api_key=api_key,
                    timeout_seconds=_to_timeout_seconds(timeout_raw),
                    lang=_normalize_ui_lang(ui_lang_val),
                )

            def _on_test_models(base_url: str, api_key: str, timeout_raw: Any, ui_lang_val: str) -> str:
                return _test_models_markdown(
                    base_url=base_url,
                    api_key=api_key,
                    timeout_seconds=_to_timeout_seconds(timeout_raw),
                    lang=_normalize_ui_lang(ui_lang_val),
                )

            llm_list_models_btn.click(
                _on_list_models,
                inputs=[llm_base_url, llm_api_key, llm_timeout, ui_lang],
                outputs=[llm_status],
            )
            llm_test_btn.click(
                _on_test_models,
                inputs=[llm_base_url, llm_api_key, llm_timeout, ui_lang],
                outputs=[llm_status],
            )
            translate_list_models_btn.click(
                _on_list_models,
                inputs=[translate_base_url, translate_api_key, llm_timeout, ui_lang],
                outputs=[translate_status],
            )
            translate_test_btn.click(
                _on_test_models,
                inputs=[translate_base_url, translate_api_key, llm_timeout, ui_lang],
                outputs=[translate_status],
            )

            def _on_load_defaults(
                llm_base_url_val: str,
                llm_api_key_val: str,
                llm_model_val: str,
                llm_timeout_val: str,
                llm_temperature_val: str,
                translate_base_url_val: str,
                translate_api_key_val: str,
                translate_model_val: str,
                translate_temperature_val: str,
                chunk_max_chars_val: str,
                chunk_min_chars_val: str,
                cloze_min_chars_val: str,
                cloze_max_per_chunk_val: str,
                content_profile_val: str,
                cloze_difficulty_val: str,
                prompt_lang_val: str,
                output_dir_val: str,
                export_dir_val: str,
                log_dir_val: str,
                default_deck_name_val: str,
                tts_voice1_val: str,
                tts_voice2_val: str,
                tts_voice3_val: str,
                tts_voice4_val: str,
                ui_lang_val: str,
            ) -> tuple[str, ...]:
                defaults = _read_env_example()
                lang = _normalize_ui_lang(ui_lang_val)

                def dv(key: str, current: str) -> str:
                    return defaults.get(key, current or "")

                return (
                    dv("CLAWLINGUA_LLM_BASE_URL", llm_base_url_val),
                    dv("CLAWLINGUA_LLM_API_KEY", llm_api_key_val),
                    dv("CLAWLINGUA_LLM_MODEL", llm_model_val),
                    dv("CLAWLINGUA_LLM_TIMEOUT_SECONDS", llm_timeout_val),
                    dv("CLAWLINGUA_LLM_TEMPERATURE", llm_temperature_val),
                    dv("CLAWLINGUA_TRANSLATE_LLM_BASE_URL", translate_base_url_val),
                    dv("CLAWLINGUA_TRANSLATE_LLM_API_KEY", translate_api_key_val),
                    dv("CLAWLINGUA_TRANSLATE_LLM_MODEL", translate_model_val),
                    dv("CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE", translate_temperature_val),
                    dv("CLAWLINGUA_CHUNK_MAX_CHARS", chunk_max_chars_val),
                    dv("CLAWLINGUA_CHUNK_MIN_CHARS", chunk_min_chars_val),
                    dv("CLAWLINGUA_CLOZE_MIN_CHARS", cloze_min_chars_val),
                    dv("CLAWLINGUA_CLOZE_MAX_PER_CHUNK", cloze_max_per_chunk_val),
                    dv("CLAWLINGUA_CONTENT_PROFILE", content_profile_val),
                    dv("CLAWLINGUA_CLOZE_DIFFICULTY", cloze_difficulty_val),
                    dv("CLAWLINGUA_PROMPT_LANG", prompt_lang_val),
                    dv("CLAWLINGUA_OUTPUT_DIR", output_dir_val),
                    dv("CLAWLINGUA_EXPORT_DIR", export_dir_val),
                    dv("CLAWLINGUA_LOG_DIR", log_dir_val),
                    dv("CLAWLINGUA_DEFAULT_DECK_NAME", default_deck_name_val),
                    dv("CLAWLINGUA_TTS_EDGE_VOICE1", tts_voice1_val),
                    dv("CLAWLINGUA_TTS_EDGE_VOICE2", tts_voice2_val),
                    dv("CLAWLINGUA_TTS_EDGE_VOICE3", tts_voice3_val),
                    dv("CLAWLINGUA_TTS_EDGE_VOICE4", tts_voice4_val),
                    f"✅ {_tr(lang, 'Loaded defaults from ENV_EXAMPLE.md (not yet saved).', '已载入 ENV_EXAMPLE.md 默认值（尚未保存）。')}",
                )

            load_defaults_btn.click(
                _on_load_defaults,
                inputs=[
                    llm_base_url,
                    llm_api_key,
                    llm_model,
                    llm_timeout,
                    llm_temperature_env,
                    translate_base_url,
                    translate_api_key,
                    translate_model,
                    translate_temperature,
                    chunk_max_chars_env,
                    chunk_min_chars_env,
                    cloze_min_chars_env,
                    cloze_max_per_chunk_env,
                    content_profile_env,
                    cloze_difficulty_env,
                    prompt_lang_env,
                    output_dir_env,
                    export_dir_env,
                    log_dir_env,
                    default_deck_name_env,
                    tts_voice1_env,
                    tts_voice2_env,
                    tts_voice3_env,
                    tts_voice4_env,
                    ui_lang,
                ],
                outputs=[
                    llm_base_url,
                    llm_api_key,
                    llm_model,
                    llm_timeout,
                    llm_temperature_env,
                    translate_base_url,
                    translate_api_key,
                    translate_model,
                    translate_temperature,
                    chunk_max_chars_env,
                    chunk_min_chars_env,
                    cloze_min_chars_env,
                    cloze_max_per_chunk_env,
                    content_profile_env,
                    cloze_difficulty_env,
                    prompt_lang_env,
                    output_dir_env,
                    export_dir_env,
                    log_dir_env,
                    default_deck_name_env,
                    tts_voice1_env,
                    tts_voice2_env,
                    tts_voice3_env,
                    tts_voice4_env,
                    save_config_status,
                ],
            )

            def _on_save_config(
                llm_base_url_val,
                llm_api_key_val,
                llm_model_val,
                llm_timeout_val,
                llm_temperature_val,
                translate_base_url_val,
                translate_api_key_val,
                translate_model_val,
                translate_temperature_val,
                chunk_max_chars_val,
                chunk_min_chars_val,
                cloze_min_chars_val,
                cloze_max_per_chunk_val,
                content_profile_val,
                cloze_difficulty_val,
                prompt_lang_val,
                output_dir_val,
                export_dir_val,
                log_dir_val,
                default_deck_name_val,
                tts_voice1_val,
                tts_voice2_val,
                tts_voice3_val,
                tts_voice4_val,
                ui_lang_val,
            ):
                updated = {
                    "CLAWLINGUA_LLM_BASE_URL": llm_base_url_val or "",
                    "CLAWLINGUA_LLM_API_KEY": llm_api_key_val or "",
                    "CLAWLINGUA_LLM_MODEL": llm_model_val or "",
                    "CLAWLINGUA_LLM_TIMEOUT_SECONDS": llm_timeout_val or "",
                    "CLAWLINGUA_LLM_TEMPERATURE": llm_temperature_val or "",
                    "CLAWLINGUA_TRANSLATE_LLM_BASE_URL": translate_base_url_val or "",
                    "CLAWLINGUA_TRANSLATE_LLM_API_KEY": translate_api_key_val or "",
                    "CLAWLINGUA_TRANSLATE_LLM_MODEL": translate_model_val or "",
                    "CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE": translate_temperature_val or "",
                    "CLAWLINGUA_CHUNK_MAX_CHARS": chunk_max_chars_val or "",
                    "CLAWLINGUA_CHUNK_MIN_CHARS": chunk_min_chars_val or "",
                    "CLAWLINGUA_CLOZE_MIN_CHARS": cloze_min_chars_val or "",
                    "CLAWLINGUA_CLOZE_MAX_PER_CHUNK": cloze_max_per_chunk_val or "",
                    "CLAWLINGUA_CONTENT_PROFILE": content_profile_val or "",
                    "CLAWLINGUA_CLOZE_DIFFICULTY": cloze_difficulty_val or "",
                    "CLAWLINGUA_PROMPT_LANG": prompt_lang_val or "",
                    "CLAWLINGUA_OUTPUT_DIR": output_dir_val or "",
                    "CLAWLINGUA_EXPORT_DIR": export_dir_val or "",
                    "CLAWLINGUA_LOG_DIR": log_dir_val or "",
                    "CLAWLINGUA_DEFAULT_DECK_NAME": default_deck_name_val or "",
                    "CLAWLINGUA_TTS_EDGE_VOICE1": tts_voice1_val or "",
                    "CLAWLINGUA_TTS_EDGE_VOICE2": tts_voice2_val or "",
                    "CLAWLINGUA_TTS_EDGE_VOICE3": tts_voice3_val or "",
                    "CLAWLINGUA_TTS_EDGE_VOICE4": tts_voice4_val or "",
                }
                msg = _save_env_v2(updated, lang=_normalize_ui_lang(ui_lang_val))
                return msg

            save_config_btn.click(
                _on_save_config,
                inputs=[
                    llm_base_url,
                    llm_api_key,
                    llm_model,
                    llm_timeout,
                    llm_temperature_env,
                    translate_base_url,
                    translate_api_key,
                    translate_model,
                    translate_temperature,
                    chunk_max_chars_env,
                    chunk_min_chars_env,
                    cloze_min_chars_env,
                    cloze_max_per_chunk_env,
                    content_profile_env,
                    cloze_difficulty_env,
                    prompt_lang_env,
                    output_dir_env,
                    export_dir_env,
                    log_dir_env,
                    default_deck_name_env,
                    tts_voice1_env,
                    tts_voice2_env,
                    tts_voice3_env,
                    tts_voice4_env,
                    ui_lang,
                ],
                outputs=[save_config_status],
            )

        with gr.Tab(_tr(initial_ui_lang, "Prompt", "提示词")) as prompt_tab:
            prompt_heading = gr.Markdown(_tr(initial_ui_lang, "### Prompt template editor", "### Prompt 模板编辑器"))
            prompt_file_selector = gr.Dropdown(
                choices=_prompt_choices(initial_ui_lang),
                value=initial_prompt_key,
                label=_tr(initial_ui_lang, "Prompt file", "Prompt 文件"),
            )
            prompt_editor = gr.Textbox(
                label=_tr(initial_ui_lang, "Prompt template", "Prompt 模板"),
                value=initial_prompt_text,
                lines=24,
            )
            with gr.Row():
                prompt_save_btn = gr.Button(_tr(initial_ui_lang, "Save", "保存"))
                prompt_load_default_btn = gr.Button(_tr(initial_ui_lang, "Load default", "载入默认"))
            prompt_status = gr.Markdown(
                label=_tr(initial_ui_lang, "Prompt status", "Prompt 状态"),
                value=initial_prompt_status,
            )

            def _on_prompt_file_change(prompt_key: str, ui_lang_val: str) -> tuple[str, str]:
                lang = _normalize_ui_lang(ui_lang_val)
                return _load_prompt_template(prompt_key, prompt_files, lang=lang)

            def _on_prompt_save(prompt_key: str, prompt_template: str, ui_lang_val: str) -> str:
                lang = _normalize_ui_lang(ui_lang_val)
                raw_template = prompt_template or ""
                if not raw_template.strip():
                    return f"❌ {_tr(lang, 'Prompt template is empty.', 'Prompt 模板为空。')}"
                template = raw_template.rstrip()
                path, payload, msg = _read_prompt_payload(prompt_key, prompt_files, lang=lang)
                if payload is None or path is None:
                    return msg
                _set_user_prompt_template(payload, lang=lang, template=template)
                ok, details = _write_prompt_payload(path, payload, lang=lang)
                if not ok:
                    return details
                return (
                    f"✅ {_tr(lang, 'Prompt template saved.', 'Prompt 模板已保存。')}\n\n"
                    f"{details}"
                )

            def _on_prompt_load_default(
                prompt_key: str,
                current_template: str,
                ui_lang_val: str,
            ) -> tuple[str, str]:
                lang = _normalize_ui_lang(ui_lang_val)
                defaults_payload, msg = _load_prompt_defaults(prompt_defaults_file, lang=lang)
                if defaults_payload is None:
                    return current_template, msg
                default_template = _default_template_for_lang(
                    defaults_payload,
                    prompt_key=prompt_key,
                    lang=lang,
                )
                if not default_template.strip():
                    return current_template, f"❌ {_tr(lang, 'Default template not found.', '未找到当前提示词的默认模板。')}"
                default_template = default_template.rstrip()

                path, payload, load_msg = _read_prompt_payload(prompt_key, prompt_files, lang=lang)
                if payload is None or path is None:
                    return current_template, load_msg
                _set_user_prompt_template(payload, lang=lang, template=default_template)
                ok, details = _write_prompt_payload(path, payload, lang=lang)
                if not ok:
                    return current_template, details
                return (
                    default_template,
                    f"✅ {_tr(lang, 'Prompt template restored from default.', '已从默认模板还原 Prompt。')}\n\n{details}",
                )

            prompt_file_selector.change(
                _on_prompt_file_change,
                inputs=[prompt_file_selector, ui_lang],
                outputs=[prompt_editor, prompt_status],
            )
            prompt_save_btn.click(
                _on_prompt_save,
                inputs=[prompt_file_selector, prompt_editor, ui_lang],
                outputs=[prompt_status],
            )
            prompt_load_default_btn.click(
                _on_prompt_load_default,
                inputs=[prompt_file_selector, prompt_editor, ui_lang],
                outputs=[prompt_editor, prompt_status],
            )

        def _on_ui_lang_change(
            lang_value: str,
            prompt_lang_current: str,
            prompt_key_current: str,
            run_id_current: str | None,
        ) -> tuple[Any, ...]:
            lang = _normalize_ui_lang(lang_value)
            _ = prompt_lang_current
            prompt_lang_next = lang
            prompt_key_next = prompt_key_current if prompt_key_current in prompt_files else initial_prompt_key
            prompt_template_next, prompt_status_next = _load_prompt_template(
                prompt_key_next,
                prompt_files,
                lang=lang,
            )
            cfg_now = _load_app_config()
            run_selector_next, run_detail_next, run_download_next = _refresh_recent_runs(
                cfg_now,
                lang=lang,
                preferred_run_id=run_id_current,
            )
            selector_choices = run_selector_next.get("choices", [])
            selector_value = run_selector_next.get("value")
            return (
                gr.update(label=_tr(lang, "UI language", "UI language")),
                gr.update(value=_tr(lang, "# ClawLingua Web UI\nLocal deck builder for text learning.", "# ClawLingua Web UI\nLocal deck builder for text learning.")),
                gr.update(label=_tr(lang, "Run", "Run")),
                gr.update(label=_tr(lang, "Config", "Config")),
                gr.update(label=_tr(lang, "Prompt", "Prompt")),
                gr.update(label=_tr(lang, "Input file", "Input file")),
                gr.update(label=_tr(lang, "Deck title (optional)", "Deck title (optional)")),
                gr.update(label=_tr(lang, "Source language", "Source language")),
                gr.update(label=_tr(lang, "Target language", "Target language")),
                gr.update(label=_tr(lang, "Content profile", "Content profile")),
                gr.update(label=_tr(lang, "Difficulty", "Difficulty")),
                gr.update(label=_tr(lang, "Max notes (0 = no limit)", "Max notes (0 = no limit)"), info=_tr(lang, "Maximum notes after dedupe. Empty/0 means no limit.", "Maximum notes after dedupe. Empty/0 means no limit.")),
                gr.update(label=_tr(lang, "Input char limit", "Input char limit"), info=_tr(lang, "Only process the first N chars of input. Empty means no limit.", "Only process the first N chars of input. Empty means no limit.")),
                gr.update(label=_tr(lang, "Advanced", "Advanced")),
                gr.update(label=_tr(lang, "Cloze min chars (override env)", "Cloze min chars (override env)"), info=_tr(lang, "One-run override for CLAWLINGUA_CLOZE_MIN_CHARS.", "One-run override for CLAWLINGUA_CLOZE_MIN_CHARS.")),
                gr.update(label=_tr(lang, "Chunk max chars (override env)", "Chunk max chars (override env)"), info=_tr(lang, "One-run override for CLAWLINGUA_CHUNK_MAX_CHARS.", "One-run override for CLAWLINGUA_CHUNK_MAX_CHARS.")),
                gr.update(label=_tr(lang, "Temperature (override env)", "Temperature (override env)"), info=_tr(lang, "0 is more deterministic; higher values are more random.", "0 is more deterministic; higher values are more random.")),
                gr.update(label=_tr(lang, "Save intermediate files", "Save intermediate files"), info=_tr(lang, "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.", "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.")),
                gr.update(label=_tr(lang, "Continue on error", "Continue on error"), info=_tr(lang, "If enabled, continue processing after per-item failures.", "If enabled, continue processing after per-item failures.")),
                gr.update(value=_tr(lang, "Run", "Run")),
                gr.update(label=_tr(lang, "Status", "Status")),
                gr.update(label=_tr(lang, "Download .apkg", "Download .apkg")),
                gr.update(value=_tr(lang, "### Recent runs", "### Recent runs")),
                gr.update(value=_tr(lang, "Refresh runs", "Refresh runs")),
                gr.update(label=_tr(lang, "Run ID", "Run ID"), choices=selector_choices, value=selector_value),
                gr.update(value=run_detail_next),
                gr.update(label=_tr(lang, "Download .apkg", "Download .apkg"), value=run_download_next),
                gr.update(value=_tr(lang, "### Config (.env editor)", "### Config (.env editor)")),
                gr.update(label=_tr(lang, "LLM (primary)", "LLM (primary)")),
                gr.update(info=_tr(lang, "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).", "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).")),
                gr.update(info=_tr(lang, "API key for primary LLM, when required.", "API key for primary LLM, when required.")),
                gr.update(info=_tr(lang, "Model name for primary LLM.", "Model name for primary LLM.")),
                gr.update(info=_tr(lang, "Request timeout in seconds.", "Request timeout in seconds.")),
                gr.update(info=_tr(lang, "Default temperature for primary LLM.", "Default temperature for primary LLM.")),
                gr.update(value=_tr(lang, "List models", "List models")),
                gr.update(value=_tr(lang, "Test", "Test")),
                gr.update(label=_tr(lang, "Primary LLM status", "Primary LLM status")),
                gr.update(label=_tr(lang, "Translation LLM", "Translation LLM")),
                gr.update(info=_tr(lang, "Optional base URL for translation model.", "Optional base URL for translation model.")),
                gr.update(info=_tr(lang, "API key for translation LLM, when required.", "API key for translation LLM, when required.")),
                gr.update(info=_tr(lang, "Model name for translation LLM.", "Model name for translation LLM.")),
                gr.update(info=_tr(lang, "Default temperature for translation LLM.", "Default temperature for translation LLM.")),
                gr.update(value=_tr(lang, "List models (translate)", "List models (translate)")),
                gr.update(value=_tr(lang, "Test (translate)", "Test (translate)")),
                gr.update(label=_tr(lang, "Translation LLM status", "Translation LLM status")),
                gr.update(label=_tr(lang, "Chunk & Cloze", "Chunk & Cloze")),
                gr.update(info=_tr(lang, "Default max chars per chunk.", "Default max chars per chunk.")),
                gr.update(info=_tr(lang, "Minimum chars required for cloze text.", "Minimum chars required for cloze text.")),
                gr.update(info=_tr(lang, "Max cards per chunk after dedupe. Empty/0 means unlimited.", "Max cards per chunk after dedupe. Empty/0 means unlimited.")),
                gr.update(label="CLAWLINGUA_PROMPT_LANG", info=_tr(lang, "Prompt language for multi-lingual prompts (en/zh).", "Prompt language for multi-lingual prompts (en/zh)."), value=prompt_lang_next),
                gr.update(label=_tr(lang, "Paths & defaults", "Paths & defaults")),
                gr.update(info=_tr(lang, "Directory for intermediate run data (JSONL, media).", "Directory for intermediate run data (JSONL, media).")),
                gr.update(info=_tr(lang, "Default directory for exported decks.", "Default directory for exported decks.")),
                gr.update(info=_tr(lang, "Directory for log files.", "Directory for log files.")),
                gr.update(value=_tr(lang, "Load defaults from ENV_EXAMPLE.md", "Load defaults from ENV_EXAMPLE.md")),
                gr.update(value=_tr(lang, "Save config", "Save config")),
                gr.update(label=_tr(lang, "TTS voices (Edge)", "TTS voices (Edge)")),
                gr.update(value=_tr(lang, "Voice reference: [Edge TTS Voice Samples](https://tts.travisvn.com/)", "Voice reference: [Edge TTS Voice Samples](https://tts.travisvn.com/)")),
                gr.update(info=_tr(lang, "Configure 4 voice slots used for random selection.", "Configure 4 voice slots used for random selection.")),
                gr.update(value=_tr(lang, "### Prompt template editor", "### Prompt template editor")),
                gr.update(label=_tr(lang, "Prompt file", "Prompt file"), choices=_prompt_choices(lang), value=prompt_key_next),
                gr.update(label=_tr(lang, "Prompt template", "Prompt template"), value=prompt_template_next),
                gr.update(value=_tr(lang, "Save", "Save")),
                gr.update(value=_tr(lang, "Load default", "Load default")),
                gr.update(label=_tr(lang, "Prompt status", "Prompt status"), value=prompt_status_next),
            )

        ui_lang.change(
            _on_ui_lang_change,
            inputs=[ui_lang, prompt_lang_env, prompt_file_selector, run_selector],
            outputs=[
                ui_lang,
                title_md,
                run_tab,
                config_tab,
                prompt_tab,
                input_file,
                deck_title,
                source_lang,
                target_lang,
                content_profile,
                difficulty,
                max_notes,
                input_char_limit,
                run_advanced,
                cloze_min_chars,
                chunk_max_chars,
                temperature,
                save_intermediate,
                continue_on_error,
                run_button,
                run_status,
                output_file,
                recent_runs_heading,
                refresh_runs_button,
                run_selector,
                run_detail,
                run_download_file,
                config_heading,
                llm_accordion,
                llm_base_url,
                llm_api_key,
                llm_model,
                llm_timeout,
                llm_temperature_env,
                llm_list_models_btn,
                llm_test_btn,
                llm_status,
                translate_accordion,
                translate_base_url,
                translate_api_key,
                translate_model,
                translate_temperature,
                translate_list_models_btn,
                translate_test_btn,
                translate_status,
                chunk_accordion,
                chunk_max_chars_env,
                cloze_min_chars_env,
                cloze_max_per_chunk_env,
                prompt_lang_env,
                paths_accordion,
                output_dir_env,
                export_dir_env,
                log_dir_env,
                load_defaults_btn,
                save_config_btn,
                tts_accordion,
                tts_hint_md,
                tts_voice1_env,
                prompt_heading,
                prompt_file_selector,
                prompt_editor,
                prompt_save_btn,
                prompt_load_default_btn,
                prompt_status,
            ],
        )


    return demo


def launch(*, server_port: int | None = None, server_host: str | None = None) -> None:
    """Launch the Gradio app.

    Logging is configured via the shared `setup_logging` function when
    loading the application config. Web-specific events are logged under
    the `clawlingua.web` logger.
    """

    port_value = server_port
    if port_value is None:
        env_port = _to_optional_int(os.getenv("CLAWLINGUA_WEB_PORT"), min_value=1)
        port_value = env_port or 7860
    host_value = (server_host or os.getenv("CLAWLINGUA_WEB_HOST") or "0.0.0.0").strip() or "0.0.0.0"

    logger.info("starting ClawLingua web UI | host=%s port=%d", host_value, port_value)
    demo = build_interface()
    demo.queue().launch(server_name=host_value, server_port=port_value)
    logger.info("ClawLingua web UI stopped")


if __name__ == "__main__":  # pragma: no cover
    launch()
