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
from clawlingua.pipeline.validators import classify_rejection_reason
from clawlingua.utils.time import make_run_id, utc_now_iso

logger = logging.getLogger("clawlingua.web")

_SUPPORTED_UI_LANGS = {"en", "zh"}
_ENV_LINE_RE = re.compile(r"^\s*(CLAWLINGUA_[A-Z0-9_]+)\s*=\s*(.*)\s*$")
_PROMPT_DIR = Path("./prompts")
_PROMPT_META_FILENAMES = {"user_prompt_overrides.json"}
_PROMPT_TEMPLATE_FILENAMES = {"template_extraction.json", "template_explanation.json"}
_PROMPT_TEMPLATE_BY_MODE = {
    "extraction": Path("./prompts/template_extraction.json"),
    "explanation": Path("./prompts/template_explanation.json"),
}
_PROMPT_CONTENT_TYPE_OPTIONS = [
    "all",
    "prose_article",
    "transcript_dialogue",
    "textbook_examples",
]
_PROMPT_LEARNING_MODE_OPTIONS = ["all", "expression_mining", "reading_support"]
_PROMPT_DIFFICULTY_OPTIONS = ["all", "beginner", "intermediate", "advanced"]
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
    "Run analytics": "\u8fd0\u884c\u7edf\u8ba1\u5206\u6790",
    "Title": "\u6807\u9898",
    "Output path": "\u8f93\u51fa\u6587\u4ef6",
    "Last error": "\u6700\u540e\u9519\u8bef",
    "Learning mode": "\u5b66\u4e60\u6a21\u5f0f",
    "Material profile": "\u6750\u6599\u7c7b\u578b",
    "Taxonomy filter": "taxonomy \u8fc7\u6ee4",
    "Transfer filter": "transfer \u8fc7\u6ee4",
    "Rejection filter": "\u62d2\u7edd\u539f\u56e0\u8fc7\u6ee4",
    "Chunk filter": "chunk \u8fc7\u6ee4",
    "Apply filters": "\u5e94\u7528\u8fc7\u6ee4",
    "Representative samples": "\u4ee3\u8868\u6027\u6837\u4f8b",
    "Transfer non-empty ratio": "Transfer \u975e\u7a7a\u6bd4\u4f8b",
    "Avg clozes per candidate": "\u6bcf\u6761\u5e73\u5747 cloze \u6570",
    "Avg target phrases per candidate": "\u6bcf\u6761\u5e73\u5747 target phrase \u6570",
    "Avg selected per chunk": "\u6bcf\u4e2a chunk \u5e73\u5747\u4fdd\u7559\u6761\u6570",
    "Filtered selected items": "\u8fc7\u6ee4\u540e\u4fdd\u7559\u6761\u76ee",
    "Filtered rejected items": "\u8fc7\u6ee4\u540e\u62d2\u7edd\u6761\u76ee",
    "Raw candidates": "\u539f\u59cb\u5019\u9009",
    "Validated candidates": "\u901a\u8fc7\u9a8c\u8bc1",
    "Selected cards": "\u6700\u7ec8\u4fdd\u7559",
    "Chunks": "\u5207\u5757\u6570",
    "Model taxonomy histogram": "\u6a21\u578b taxonomy \u5206\u5e03",
    "Candidate taxonomy histogram": "\u5019\u9009 taxonomy \u5206\u5e03",
    "Selected taxonomy histogram": "\u6700\u7ec8 taxonomy \u5206\u5e03",
    "Taxonomy average score": "taxonomy \u5e73\u5747\u5f97\u5206",
    "Rejection reason histogram": "\u62d2\u7edd\u539f\u56e0\u5206\u5e03",
    "Transfer non-empty ratio by taxonomy": "taxonomy \u7ef4\u5ea6 transfer \u975e\u7a7a\u7387",
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
    "Prompt type": "\u63d0\u793a\u8bcd\u7c7b\u578b",
    "Prompt content type": "\u63d0\u793a\u8bcd\u5185\u5bb9\u7c7b\u578b",
    "Prompt learning mode": "\u63d0\u793a\u8bcd\u5b66\u4e60\u6a21\u5f0f",
    "Prompt difficulty": "\u63d0\u793a\u8bcd\u96be\u5ea6",
    "New": "\u65b0\u5efa",
    "Rename": "\u91cd\u547d\u540d",
    "Delete": "\u5220\u9664",
    "Confirm save": "\u786e\u8ba4\u4fdd\u5b58",
    "Confirm delete": "\u786e\u8ba4\u5220\u9664",
    "Please confirm save first.": "\u8bf7\u5148\u786e\u8ba4\u4fdd\u5b58\u3002",
    "Please confirm delete first.": "\u8bf7\u5148\u786e\u8ba4\u5220\u9664\u3002",
    "Prompt file name is empty.": "\u63d0\u793a\u8bcd\u6587\u4ef6\u540d\u4e3a\u7a7a\u3002",
    "Prompt file already exists.": "\u63d0\u793a\u8bcd\u6587\u4ef6\u5df2\u5b58\u5728\u3002",
    "Prompt file created.": "\u63d0\u793a\u8bcd\u6587\u4ef6\u5df2\u521b\u5efa\u3002",
    "Prompt file renamed.": "\u63d0\u793a\u8bcd\u6587\u4ef6\u5df2\u91cd\u547d\u540d\u3002",
    "Prompt file deleted.": "\u63d0\u793a\u8bcd\u6587\u4ef6\u5df2\u5220\u9664\u3002",
    "Cannot delete the last Extraction prompt.": "\u4e0d\u80fd\u5220\u9664\u6700\u540e\u4e00\u4e2a\u63d0\u53d6\u63d0\u793a\u8bcd\u3002",
    "Cannot delete the last Explanation prompt.": "\u4e0d\u80fd\u5220\u9664\u6700\u540e\u4e00\u4e2a\u89e3\u91ca\u63d0\u793a\u8bcd\u3002",
    "Template prompt file missing.": "\u6a21\u677f\u63d0\u793a\u8bcd\u6587\u4ef6\u4e0d\u5b58\u5728\u3002",
    "New prompt file name": "\u65b0\u63d0\u793a\u8bcd\u6587\u4ef6\u540d",
    "Rename to": "\u91cd\u547d\u540d\u4e3a",
    "Extraction prompt (run override)": "\u63d0\u53d6\u63d0\u793a\u8bcd\uff08\u8fd0\u884c\u8986\u76d6\uff09",
    "Explanation prompt (run override)": "\u89e3\u91ca\u63d0\u793a\u8bcd\uff08\u8fd0\u884c\u8986\u76d6\uff09",
    "Equivalent to CLI --extract-prompt.": "\u7b49\u4ef7\u4e8e\u547d\u4ee4\u884c --extract-prompt\u3002",
    "Equivalent to CLI --explain-prompt.": "\u7b49\u4ef7\u4e8e\u547d\u4ee4\u884c --explain-prompt\u3002",
    "Default extraction prompt path.": "\u9ed8\u8ba4\u63d0\u53d6\u63d0\u793a\u8bcd\u8def\u5f84\u3002",
    "Default explanation prompt path.": "\u9ed8\u8ba4\u89e3\u91ca\u63d0\u793a\u8bcd\u8def\u5f84\u3002",
    "Extraction": "\u63d0\u53d6",
    "Explanation": "\u89e3\u91ca",
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
        return f"鉂?Failed to save config: {exc}"

    return "鉁?Config saved and validated."


def _run_single_build(
    uploaded_file: Any,
    deck_title: str,
    source_lang: str,
    target_lang: str,
    content_profile: str,
    learning_mode: str,
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
        learning_mode=learning_mode or None,
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
        logger.exception(
            "web build failed | input=%s profile=%s difficulty=%s",
            str(dst),
            content_profile or cfg.content_profile,
            difficulty or cfg.cloze_difficulty,
        )
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
    learning_mode: str,
    difficulty: str,
    max_notes: int | None,
    input_char_limit: int | None,
    cloze_min_chars: int | None,
    chunk_max_chars: int | None,
    temperature: float | None,
    save_intermediate: bool,
    continue_on_error: bool,
    prompt_lang: str | None = None,
    extract_prompt: str | None = None,
    explain_prompt: str | None = None,
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
        logger.exception("failed to materialize uploaded file")
        return {"status": "error", "message": str(exc), "run_id": None}

    run_id = make_run_id()
    run_dir = cfg.resolve_path(cfg.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    default_output_path = (
        cfg.resolve_path(cfg.export_dir) / run_id / "output.apkg"
    ).resolve()
    summary_path = run_dir / "run_summary.json"

    source_lang_value = source_lang or cfg.default_source_lang
    target_lang_value = target_lang or cfg.default_target_lang
    profile_value = (
        content_profile or getattr(cfg, "material_profile", None) or cfg.content_profile
    )
    learning_mode_value = learning_mode or cfg.learning_mode
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
            "material_profile": profile_value,
            "learning_mode": learning_mode_value,
            "difficulty": difficulty or cfg.cloze_difficulty,
            "extract_prompt_override": _as_str(extract_prompt),
            "explain_prompt_override": _as_str(explain_prompt),
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
        material_profile=content_profile or None,
        learning_mode=learning_mode or None,
        input_char_limit=input_char_limit,
        deck_name=deck_title or None,
        max_chars=chunk_max_chars,
        cloze_min_chars=cloze_min_chars,
        max_notes=max_notes,
        temperature=temperature,
        cloze_difficulty=difficulty or None,
        extract_prompt=Path(extract_prompt) if _as_str(extract_prompt) else None,
        explain_prompt=Path(explain_prompt) if _as_str(explain_prompt) else None,
        save_intermediate=save_intermediate,
        continue_on_error=continue_on_error,
    )
    try:
        result = run_build_deck(cfg, options)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception(
            "web build failed | run_id=%s input=%s profile=%s difficulty=%s",
            run_id,
            str(local_input),
            content_profile or cfg.content_profile,
            difficulty or cfg.cloze_difficulty,
        )
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

    new_env: Dict[str, str] = {
        k: str(v) for k, v in current.items() if k not in _EDITABLE_ENV_KEYS
    }
    for key in _EDITABLE_ENV_KEYS:
        if key not in updated:
            continue
        val = str(updated.get(key, "")).strip()
        if val:
            new_env[key] = val
        else:
            new_env.pop(key, None)

    env_file.write_text(
        "".join(f"{k}={v}\n" for k, v in sorted(new_env.items())), encoding="utf-8"
    )
    try:
        cfg = load_config(env_file=env_file)
        validate_base_config(cfg)
        validate_runtime_config(cfg)
    except Exception as exc:
        if original_text is not None:
            env_file.write_text(original_text, encoding="utf-8")
        else:
            env_file.unlink(missing_ok=True)
        return (
            f"❌ {_tr(lang, 'Failed to save config', 'Failed to save config')}: {exc}"
        )
    return (
        f"✅ {_tr(lang, 'Config saved and validated.', 'Config saved and validated.')}"
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
    base_url: str, api_key: str, timeout_seconds: float, *, lang: str
) -> str:
    if not _normalize_base_url(base_url):
        return f"⚠️ {_tr(lang, 'Missing base URL.', 'Missing base URL.')}"
    try:
        endpoint, response = _request_models(base_url, api_key, timeout_seconds)
    except ValueError:
        return f"⚠️ {_tr(lang, 'Missing base URL.', 'Missing base URL.')}"
    except httpx.RequestError as exc:
        return f"❌ {_tr(lang, 'Request failed', 'Request failed')}: `{exc}`"
    except Exception as exc:
        return f"❌ {_tr(lang, 'Request failed', 'Request failed')}: `{exc}`"

    if response.status_code >= 400:
        body = response.text[:500] if response.text else ""
        return (
            f"❌ {_tr(lang, 'HTTP error', 'HTTP error')}: **{response.status_code}**\n\n"
            f"- endpoint: `{endpoint}`\n"
            f"- body: `{body}`"
        )

    try:
        payload = response.json()
    except ValueError:
        return f"❌ {_tr(lang, 'Response is not valid JSON.', 'Response is not valid JSON.')}"

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return (
            f"⚠️ {_tr(lang, 'Response JSON has no list field `data`.', 'Response JSON has no list field `data`.')}\n\n"
            f"- endpoint: `{endpoint}`\n"
            f"- status: `{response.status_code}`"
        )

    model_ids: list[str] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = _as_str(item.get("id"))
        if not model_id or model_id in seen:
            continue
        model_ids.append(model_id)
        seen.add(model_id)

    if not model_ids:
        return (
            f"✅ {_tr(lang, 'Found models', 'Found models')}: **0**\n\n"
            f"- endpoint: `{endpoint}`\n"
            f"- status: `{response.status_code}`\n"
            f"- {_tr(lang, 'No model ids found in `data`.', 'No model ids found in `data`.')}"
        )

    lines = [f"- `{model_id}`" for model_id in model_ids]
    return (
        f"✅ {_tr(lang, 'Found models', 'Found models')}: **{len(model_ids)}**\n\n"
        f"- endpoint: `{endpoint}`\n"
        f"- status: `{response.status_code}`\n\n"
        f"{chr(10).join(lines)}"
    )


def _test_models_markdown(
    base_url: str, api_key: str, timeout_seconds: float, *, lang: str
) -> str:
    if not _normalize_base_url(base_url):
        return f"⚠️ {_tr(lang, 'Missing base URL.', 'Missing base URL.')}"
    try:
        endpoint, response = _request_models(base_url, api_key, timeout_seconds)
    except ValueError:
        return f"⚠️ {_tr(lang, 'Missing base URL.', 'Missing base URL.')}"
    except httpx.RequestError as exc:
        return f"❌ {_tr(lang, 'Request failed', 'Request failed')}: `{exc}`"
    except Exception as exc:
        return f"❌ {_tr(lang, 'Request failed', 'Request failed')}: `{exc}`"
    return (
        f"✅ {_tr(lang, 'Connectivity OK', 'Connectivity OK')}\n\n"
        f"- endpoint: `{endpoint}`\n"
        f"- status: `{response.status_code}`"
    )


def _normalize_prompt_mode(value: Any) -> str:
    mode = _as_str(value).lower()
    if mode == "cloze":
        return "extraction"
    if mode == "translate":
        return "explanation"
    if mode in {"extraction", "explanation"}:
        return mode
    return ""


def _normalize_prompt_content_type(value: Any) -> str:
    content_type = _as_str(value).lower()
    if content_type in {"", "auto"}:
        return "all"
    if content_type == "general":
        return "prose_article"
    if content_type in {"prose", "article"}:
        return "prose_article"
    if content_type in {"transcript", "dialogue"}:
        return "transcript_dialogue"
    if content_type in {"textbook", "example"}:
        return "textbook_examples"
    if content_type in _PROMPT_CONTENT_TYPE_OPTIONS:
        return content_type
    return "all"


def _normalize_prompt_learning_mode(value: Any) -> str:
    learning_mode = _as_str(value).lower()
    if learning_mode in {"", "auto"}:
        return "all"
    if learning_mode in _PROMPT_LEARNING_MODE_OPTIONS:
        return learning_mode
    return "all"


def _normalize_prompt_difficulty(value: Any) -> str:
    difficulty = _as_str(value).lower()
    if difficulty in {"", "auto"}:
        return "all"
    if difficulty in _PROMPT_DIFFICULTY_OPTIONS:
        return difficulty
    return "all"


def _normalize_prompt_metadata_from_payload(payload: dict[str, Any]) -> tuple[str, str, str]:
    content_type = _normalize_prompt_content_type(
        payload.get("content_type")
        or payload.get("material_profile")
        or payload.get("content_profile")
    )
    learning_mode = _normalize_prompt_learning_mode(payload.get("learning_mode"))
    difficulty = _normalize_prompt_difficulty(
        payload.get("difficulty_level")
        or payload.get("difficulty")
        or payload.get("cloze_difficulty")
    )
    return content_type, learning_mode, difficulty


def _prompt_meta_matches(filter_value: str, prompt_value: str) -> bool:
    if filter_value == "all":
        return True
    if prompt_value == "all":
        return True
    return filter_value == prompt_value


def _prompt_mode_label(mode: str, *, lang: str) -> str:
    if _normalize_prompt_mode(mode) == "explanation":
        return _tr(lang, "Explanation", "瑙ｉ噴")
    return _tr(lang, "Extraction", "鎻愬彇")


def _prompt_file_map(
    cfg: Any,
    *,
    mode_filter: str | None = None,
    content_type_filter: str | None = None,
    learning_mode_filter: str | None = None,
    difficulty_filter: str | None = None,
    include_templates: bool = False,
) -> dict[str, Path]:
    prompts_dir = cfg.resolve_path(_PROMPT_DIR)
    mode_value = _normalize_prompt_mode(mode_filter) if mode_filter else ""
    content_type_value = (
        _normalize_prompt_content_type(content_type_filter)
        if content_type_filter is not None
        else "all"
    )
    learning_mode_value = (
        _normalize_prompt_learning_mode(learning_mode_filter)
        if learning_mode_filter is not None
        else "all"
    )
    difficulty_value = (
        _normalize_prompt_difficulty(difficulty_filter)
        if difficulty_filter is not None
        else "all"
    )
    if not prompts_dir.exists():
        return {}
    result: dict[str, Path] = {}
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
        mode = _normalize_prompt_mode(spec.mode)
        if mode_value and mode != mode_value:
            continue
        prompt_content_type, prompt_learning_mode, prompt_difficulty = (
            _normalize_prompt_metadata_from_payload(payload)
        )
        if not _prompt_meta_matches(content_type_value, prompt_content_type):
            continue
        if not _prompt_meta_matches(learning_mode_value, prompt_learning_mode):
            continue
        if not _prompt_meta_matches(difficulty_value, prompt_difficulty):
            continue
        result[path.name] = path
    return result


def _prompt_choices_from_map(
    prompt_files: dict[str, Path], *, lang: str
) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for key, path in prompt_files.items():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            spec = PromptSpec.model_validate(payload)
            mode = _normalize_prompt_mode(spec.mode)
        except (OSError, json.JSONDecodeError, ValidationError, ValueError):
            continue
        mode_text = _prompt_mode_label(mode, lang=lang)
        choices.append((f"{path.name} ({mode_text})", key))
    return choices


def _prompt_files_for_mode(
    prompt_files: dict[str, Path], *, mode: str, lang: str
) -> dict[str, Path]:
    mode_value = _normalize_prompt_mode(mode)
    if not mode_value:
        return dict(prompt_files)
    filtered: dict[str, Path] = {}
    for key, path in prompt_files.items():
        if _load_prompt_mode(key, prompt_files, lang=lang) == mode_value:
            filtered[key] = path
    return filtered


def _prompt_choices(
    lang: str,
    *,
    mode_filter: str | None = None,
    content_type_filter: str | None = None,
    learning_mode_filter: str | None = None,
    difficulty_filter: str | None = None,
    include_templates: bool = False,
) -> list[tuple[str, str]]:
    cfg = _load_app_config()
    prompt_files = _prompt_file_map(
        cfg,
        mode_filter=mode_filter,
        content_type_filter=content_type_filter,
        learning_mode_filter=learning_mode_filter,
        difficulty_filter=difficulty_filter,
        include_templates=include_templates,
    )
    return _prompt_choices_from_map(prompt_files, lang=lang)


def _prompt_path_value(cfg: Any, path: Path) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(cfg.workspace_root.resolve())
        rel_text = rel.as_posix()
        if rel_text.startswith("./"):
            return rel_text
        return f"./{rel_text}"
    except ValueError:
        return str(resolved)


def _prompt_path_choices(
    cfg: Any,
    *,
    lang: str,
    mode_filter: str,
    content_type_filter: str | None = None,
    learning_mode_filter: str | None = None,
    difficulty_filter: str | None = None,
    include_auto: bool = False,
) -> list[tuple[str, str]]:
    prompt_files = _prompt_file_map(
        cfg,
        mode_filter=mode_filter,
        content_type_filter=content_type_filter,
        learning_mode_filter=learning_mode_filter,
        difficulty_filter=difficulty_filter,
        include_templates=False,
    )
    mode_text = _prompt_mode_label(mode_filter, lang=lang)
    choices: list[tuple[str, str]] = []
    if include_auto:
        choices.append(
            (_tr(lang, "Auto (default chain)", "鑷姩锛堥粯璁ら摼璺級"), "")
        )
    for path in prompt_files.values():
        choices.append((f"{path.name} ({mode_text})", _prompt_path_value(cfg, path)))
    return choices


def _prompt_template_path(cfg: Any, mode: str) -> Path | None:
    normalized_mode = _normalize_prompt_mode(mode)
    if not normalized_mode:
        return None
    template_rel = _PROMPT_TEMPLATE_BY_MODE.get(normalized_mode)
    if template_rel is None:
        return None
    return cfg.resolve_path(template_rel)


def _sanitize_prompt_filename(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", _as_str(raw))
    if not safe:
        return ""
    if not safe.lower().endswith(".json"):
        safe += ".json"
    return safe


def _read_prompt_payload(
    prompt_key: str,
    prompt_files: dict[str, Path],
    *,
    lang: str,
) -> tuple[Path | None, dict[str, Any] | None, str]:
    path = prompt_files.get(prompt_key)
    if path is None:
        return (
            None,
            None,
            f"鉂?{_tr(lang, 'Failed to load prompt file', '鍔犺浇 prompt 鏂囦欢澶辫触')}: `{prompt_key}`",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (
            path,
            None,
            f"鉂?{_tr(lang, 'Prompt file JSON parse error', 'Prompt 鏂囦欢 JSON 瑙ｆ瀽澶辫触')}: {exc.msg} (line {exc.lineno}, col {exc.colno})",
        )
    except Exception as exc:
        return (
            None,
            None,
            f"鉂?{_tr(lang, 'Failed to load prompt file', '鍔犺浇 prompt 鏂囦欢澶辫触')}: `{exc}`",
        )
    if not isinstance(payload, dict):
        return (
            path,
            None,
            f"鉂?{_tr(lang, 'Prompt file JSON parse error', 'Prompt 鏂囦欢 JSON 瑙ｆ瀽澶辫触')}: root must be a JSON object.",
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
    template = _resolve_template_for_lang(
        payload.get("user_prompt_template"), lang=lang
    )
    return template, ""


def _load_prompt_mode(
    prompt_key: str,
    prompt_files: dict[str, Path],
    *,
    lang: str,
) -> str:
    _path, payload, _msg = _read_prompt_payload(prompt_key, prompt_files, lang=lang)
    if payload is None:
        return ""
    return _normalize_prompt_mode(payload.get("mode"))


def _load_prompt_filter_metadata(
    prompt_key: str,
    prompt_files: dict[str, Path],
    *,
    lang: str,
) -> tuple[str, str, str]:
    _path, payload, _msg = _read_prompt_payload(prompt_key, prompt_files, lang=lang)
    if payload is None:
        return ("all", "all", "all")
    return _normalize_prompt_metadata_from_payload(payload)


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
        return (
            False,
            f"鉂?{_tr(lang, 'Schema validation failed', 'Schema 鏍￠獙澶辫触')}\n\n{_format_prompt_validation_error(exc)}",
        )
    return True, ""


def _set_user_prompt_template(
    payload: dict[str, Any], *, lang: str, template: str
) -> None:
    _ = lang
    payload["user_prompt_template"] = template


def _write_prompt_payload(
    path: Path, payload: dict[str, Any], *, lang: str
) -> tuple[bool, str]:
    ok, msg = _validate_prompt_payload(payload, lang=lang)
    if not ok:
        return (
            False,
            f"{msg}\n\n⚠️ {_tr(lang, 'Not saved because validation failed.', 'Not saved because validation failed.')}",
        )
    backup = path.with_suffix(path.suffix + ".bak")
    try:
        if path.exists():
            shutil.copyfile(path, backup)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        return (
            False,
            f"❌ {_tr(lang, 'Failed to save prompt file', 'Failed to save prompt file')}: `{exc}`",
        )
    return (
        True,
        f"- file: `{path}`\n- {_tr(lang, 'Backup created', 'Backup created')}: `{backup}`",
    )


@dataclass
class RunInfo:
    run_id: str
    started_at: str
    finished_at: str | None
    title: str
    source_lang: str
    target_lang: str
    content_profile: str
    material_profile: str
    learning_mode: str
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
        "CLAWLINGUA_TRANSLATE_LLM_MODEL": _as_str(
            getattr(cfg, "translate_llm_model", "")
        ),
        "CLAWLINGUA_PROMPT_LANG": _as_str(getattr(cfg, "prompt_lang", "")),
        "CLAWLINGUA_EXTRACT_PROMPT": _as_str(getattr(cfg, "extract_prompt", "")),
        "CLAWLINGUA_EXPLAIN_PROMPT": _as_str(getattr(cfg, "explain_prompt", "")),
        "CLAWLINGUA_MATERIAL_PROFILE": _as_str(getattr(cfg, "material_profile", "")),
        "CLAWLINGUA_LEARNING_MODE": _as_str(getattr(cfg, "learning_mode", "")),
    }


def _write_run_summary(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
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
    fallback_started = datetime.fromtimestamp(
        run_dir.stat().st_mtime, tz=timezone.utc
    ).isoformat()

    started_at = _as_str(summary.get("started_at"), default=fallback_started)
    finished_at_text = _as_str(summary.get("finished_at"))
    finished_at = finished_at_text or None
    status = _normalize_run_status(summary.get("status"))
    title = _as_str(summary.get("title"), default=run_id)
    source_lang = _as_str(summary.get("source_lang"))
    target_lang = _as_str(summary.get("target_lang"))
    content_profile = _as_str(summary.get("content_profile"))
    material_profile = _as_str(summary.get("material_profile"), default=content_profile)
    learning_mode = _as_str(summary.get("learning_mode"), default="expression_mining")
    cards = max(0, _as_int(summary.get("cards"), default=0))
    errors = max(0, _as_int(summary.get("errors"), default=0))
    output_path_resolved = _resolve_output_path(
        cfg, run_dir, summary.get("output_path")
    )
    output_path_text = (
        str(output_path_resolved) if output_path_resolved is not None else None
    )
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
        material_profile=material_profile,
        learning_mode=learning_mode,
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


def _load_run_detail(
    run_id: str | None, cfg: Any, *, lang: str
) -> tuple[str, str | None]:
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
        f"### {_tr(lang, 'Run details', 'Run details')}",
        f"- {_tr(lang, 'Run ID', 'Run ID')}: `{info.run_id}`",
        f"- {_tr(lang, 'Status', 'Status')}: **{_status_text(lang, info.status)}**",
        f"- {_tr(lang, 'Started at', 'Started at')}: `{info.started_at or '-'}`",
        f"- {_tr(lang, 'Finished at', 'Finished at')}: `{info.finished_at or '-'}`",
        f"- {_tr(lang, 'Title', 'Title')}: `{info.title or '-'}`",
        f"- {_tr(lang, 'Source language', 'Source language')}: `{info.source_lang or '-'}`",
        f"- {_tr(lang, 'Target language', 'Target language')}: `{info.target_lang or '-'}`",
        f"- {_tr(lang, 'Learning mode', 'Learning mode')}: `{info.learning_mode or '-'}`",
        f"- {_tr(lang, 'Content profile', 'Content profile')}: `{info.content_profile or '-'}`",
        f"- {_tr(lang, 'Material profile', 'Material profile')}: `{info.material_profile or '-'}`",
        f"- {_tr(lang, 'Cards', 'Cards')}: **{info.cards}**",
        f"- {_tr(lang, 'Errors', 'Errors')}: **{info.errors}**",
        f"- {_tr(lang, 'Output path', 'Output path')}: `{info.output_path or '-'}`",
    ]
    if info.last_error:
        lines.append(f"- {_tr(lang, 'Last error', 'Last error')}: `{info.last_error}`")
    if download_path is None:
        lines.append(
            f"- {_tr(lang, 'Output file not available yet.', 'Output file not available yet.')}"
        )
    return "\n".join(lines), download_path


def _refresh_recent_runs(
    cfg: Any, *, lang: str, preferred_run_id: str | None = None
) -> tuple[Any, str, str | None]:
    runs = _scan_runs(cfg, limit=30)
    if not runs:
        detail = _tr(lang, "No runs found.", "No runs found.")
        return gr.update(choices=[], value=None), detail, None

    choices = [(_run_choice_label(run, lang=lang), run.run_id) for run in runs]
    run_ids = {run.run_id for run in runs}
    selected = preferred_run_id if preferred_run_id in run_ids else runs[0].run_id
    detail, download_path = _load_run_detail(selected, cfg, lang=lang)
    return gr.update(choices=choices, value=selected), detail, download_path


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _bar(value: float, max_value: float, *, width: int = 18) -> str:
    if max_value <= 0:
        return "." * width
    ratio = max(0.0, min(1.0, value / max_value))
    filled = int(round(ratio * width))
    return ("#" * filled) + ("-" * max(0, width - filled))


def _render_count_histogram(title: str, hist: dict[str, int]) -> str:
    if not hist:
        return f"#### {title}\n- n/a"
    max_value = max(hist.values())
    lines = [f"#### {title}"]
    for key, value in sorted(hist.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- `{key}` `{value}` {_bar(float(value), float(max_value))}")
    return "\n".join(lines)


def _render_score_histogram(title: str, hist: dict[str, float]) -> str:
    if not hist:
        return f"#### {title}\n- n/a"
    max_value = max(hist.values())
    lines = [f"#### {title}"]
    for key, value in sorted(hist.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- `{key}` `{value:.3f}` {_bar(float(value), float(max_value))}")
    return "\n".join(lines)


def _short_text(value: str, *, limit: int = 120) -> str:
    text = _as_str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _run_analysis_payload(run_id: str | None, cfg: Any) -> dict[str, Any]:
    selected = _as_str(run_id)
    if not selected:
        return {}
    run_dir = cfg.resolve_path(cfg.output_dir) / selected
    if not run_dir.exists() or not run_dir.is_dir():
        return {}
    summary = _read_run_summary(run_dir / "run_summary.json")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    selected_candidates = _read_jsonl_dicts(run_dir / "text_candidates.validated.jsonl")
    raw_candidates = _read_jsonl_dicts(run_dir / "text_candidates.raw.jsonl")
    cards = _read_jsonl_dicts(run_dir / "cards.final.jsonl")
    errors = _read_jsonl_dicts(run_dir / "errors.jsonl")
    chunks = _read_jsonl_dicts(run_dir / "chunks.jsonl")

    rejected: list[dict[str, Any]] = []
    for item in errors:
        if _as_str(item.get("stage")) != "validate_text":
            continue
        reason = _as_str(item.get("reason"))
        category = classify_rejection_reason(reason)
        candidate = item.get("item")
        candidate_map = candidate if isinstance(candidate, dict) else {}
        rejected.append(
            {
                "reason": reason,
                "reason_category": category,
                "chunk_id": _as_str(
                    item.get("chunk_id") or candidate_map.get("chunk_id")
                ),
                "item": candidate_map,
            }
        )

    return {
        "run_id": selected,
        "run_dir": run_dir,
        "summary": summary,
        "metrics": metrics,
        "selected_candidates": selected_candidates,
        "raw_candidates": raw_candidates,
        "cards": cards,
        "rejected": rejected,
        "chunks": chunks,
    }


def _analysis_filter_choices(
    payload: dict[str, Any],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    metrics = (
        payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    )
    selected_candidates = (
        payload.get("selected_candidates", [])
        if isinstance(payload.get("selected_candidates"), list)
        else []
    )
    rejected = (
        payload.get("rejected", []) if isinstance(payload.get("rejected"), list) else []
    )

    taxonomy_keys: set[str] = set()
    for key in (
        "taxonomy_model_histogram",
        "taxonomy_candidate_histogram",
        "taxonomy_selected_histogram",
        "taxonomy_average_score",
    ):
        value = metrics.get(key)
        if isinstance(value, dict):
            taxonomy_keys.update(
                _as_str(item) for item in value.keys() if _as_str(item)
            )
    for item in selected_candidates:
        if not isinstance(item, dict):
            continue
        for ptype in item.get("phrase_types", []) or []:
            key = _as_str(ptype)
            if key:
                taxonomy_keys.add(key)

    rejection_keys: set[str] = set()
    for item in rejected:
        if not isinstance(item, dict):
            continue
        key = _as_str(item.get("reason_category"))
        if key:
            rejection_keys.add(key)

    chunk_keys: set[str] = set()
    for item in selected_candidates:
        if not isinstance(item, dict):
            continue
        key = _as_str(item.get("chunk_id"))
        if key:
            chunk_keys.add(key)
    for item in rejected:
        if not isinstance(item, dict):
            continue
        key = _as_str(item.get("chunk_id"))
        if key:
            chunk_keys.add(key)

    taxonomy_choices = [("all", "all")] + [(key, key) for key in sorted(taxonomy_keys)]
    rejection_choices = [("all", "all")] + [
        (key, key) for key in sorted(rejection_keys)
    ]
    chunk_choices = [("all", "all")] + [(key, key) for key in sorted(chunk_keys)]
    return taxonomy_choices, rejection_choices, chunk_choices


def _build_run_analysis(
    run_id: str | None,
    cfg: Any,
    *,
    lang: str,
    taxonomy_filter: str = "all",
    transfer_filter: str = "all",
    rejection_filter: str = "all",
    chunk_filter: str = "all",
) -> tuple[
    str,
    list[list[Any]],
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    payload = _run_analysis_payload(run_id, cfg)
    if not payload:
        return (
            _tr(lang, "No run selected.", "No run selected."),
            [],
            [("all", "all")],
            [("all", "all")],
            [("all", "all")],
        )

    summary = payload["summary"]
    metrics = payload["metrics"]
    selected_candidates = payload["selected_candidates"]
    rejected = payload["rejected"]
    taxonomy_choices, rejection_choices, chunk_choices = _analysis_filter_choices(
        payload
    )

    taxonomy_filter = _as_str(taxonomy_filter, default="all")
    transfer_filter = _as_str(transfer_filter, default="all")
    rejection_filter = _as_str(rejection_filter, default="all")
    chunk_filter = _as_str(chunk_filter, default="all")

    filtered_selected: list[dict[str, Any]] = []
    for item in selected_candidates:
        if not isinstance(item, dict):
            continue
        item_types = [
            _as_str(x) for x in (item.get("phrase_types", []) or []) if _as_str(x)
        ]
        has_transfer = bool(_as_str(item.get("expression_transfer")))
        item_chunk = _as_str(item.get("chunk_id"))

        if taxonomy_filter != "all" and taxonomy_filter not in item_types:
            continue
        if transfer_filter == "with_transfer" and not has_transfer:
            continue
        if transfer_filter == "without_transfer" and has_transfer:
            continue
        if chunk_filter != "all" and item_chunk != chunk_filter:
            continue
        filtered_selected.append(item)

    filtered_rejected: list[dict[str, Any]] = []
    for item in rejected:
        if not isinstance(item, dict):
            continue
        item_chunk = _as_str(item.get("chunk_id"))
        reason_category = _as_str(item.get("reason_category"))
        if rejection_filter != "all" and reason_category != rejection_filter:
            continue
        if chunk_filter != "all" and item_chunk != chunk_filter:
            continue
        filtered_rejected.append(item)

    learning_mode = _as_str(
        summary.get("learning_mode"),
        default=_as_str(metrics.get("learning_mode"), default="expression_mining"),
    )
    material_profile = _as_str(
        summary.get("material_profile"),
        default=_as_str(summary.get("content_profile"), default="-"),
    )
    difficulty = _as_str(
        summary.get("difficulty"),
        default=_as_str(metrics.get("difficulty"), default="-"),
    )
    chunks_total = _as_int(metrics.get("chunks_total"), default=0)
    raw_total = _as_int(metrics.get("raw_candidates"), default=0)
    valid_total = _as_int(metrics.get("validated_candidates"), default=0)
    selected_total = _as_int(metrics.get("deduped_candidates"), default=0)
    transfer_ratio = float(metrics.get("expression_transfer_non_empty_ratio") or 0.0)
    avg_clozes = float(metrics.get("avg_clozes_per_candidate") or 0.0)
    avg_phrases = float(metrics.get("avg_target_phrases_per_candidate") or 0.0)
    avg_selected_per_chunk = float(
        metrics.get("avg_selected_candidates_per_chunk") or 0.0
    )

    model_hist = (
        metrics.get("taxonomy_model_histogram")
        if isinstance(metrics.get("taxonomy_model_histogram"), dict)
        else {}
    )
    candidate_hist = (
        metrics.get("taxonomy_candidate_histogram")
        if isinstance(metrics.get("taxonomy_candidate_histogram"), dict)
        else {}
    )
    selected_hist = (
        metrics.get("taxonomy_selected_histogram")
        if isinstance(metrics.get("taxonomy_selected_histogram"), dict)
        else {}
    )
    avg_score_hist = (
        metrics.get("taxonomy_average_score")
        if isinstance(metrics.get("taxonomy_average_score"), dict)
        else {}
    )
    rejection_hist = (
        metrics.get("rejection_reason_histogram")
        if isinstance(metrics.get("rejection_reason_histogram"), dict)
        else {}
    )
    transfer_by_tax = (
        metrics.get("expression_transfer_non_empty_ratio_by_taxonomy")
        if isinstance(
            metrics.get("expression_transfer_non_empty_ratio_by_taxonomy"), dict
        )
        else {}
    )

    lines = [
        f"### {_tr(lang, 'Run analytics', 'Run analytics')}",
        f"- {_tr(lang, 'Learning mode', 'Learning mode')}: `{learning_mode}`",
        f"- {_tr(lang, 'Material profile', 'Material profile')}: `{material_profile}`",
        f"- {_tr(lang, 'Difficulty', 'Difficulty')}: `{difficulty}`",
        f"- {_tr(lang, 'Chunks', 'Chunks')}: **{chunks_total}**",
        f"- {_tr(lang, 'Raw candidates', 'Raw candidates')}: **{raw_total}**",
        f"- {_tr(lang, 'Validated candidates', 'Validated candidates')}: **{valid_total}**",
        f"- {_tr(lang, 'Selected cards', 'Selected cards')}: **{selected_total}**",
        f"- {_tr(lang, 'Transfer non-empty ratio', 'Transfer non-empty ratio')}: **{transfer_ratio:.2%}**",
        f"- {_tr(lang, 'Avg clozes per candidate', 'Avg clozes per candidate')}: **{avg_clozes:.2f}**",
        f"- {_tr(lang, 'Avg target phrases per candidate', 'Avg target phrases per candidate')}: **{avg_phrases:.2f}**",
        f"- {_tr(lang, 'Avg selected per chunk', 'Avg selected per chunk')}: **{avg_selected_per_chunk:.2f}**",
        f"- {_tr(lang, 'Filtered selected items', 'Filtered selected items')}: **{len(filtered_selected)}**",
        f"- {_tr(lang, 'Filtered rejected items', 'Filtered rejected items')}: **{len(filtered_rejected)}**",
        "",
        _render_count_histogram(
            _tr(lang, "Model taxonomy histogram", "Model taxonomy histogram"),
            model_hist,
        ),
        _render_count_histogram(
            _tr(lang, "Candidate taxonomy histogram", "Candidate taxonomy histogram"),
            candidate_hist,
        ),
        _render_count_histogram(
            _tr(lang, "Selected taxonomy histogram", "Selected taxonomy histogram"),
            selected_hist,
        ),
        _render_score_histogram(
            _tr(lang, "Taxonomy average score", "Taxonomy average score"),
            avg_score_hist,
        ),
        _render_count_histogram(
            _tr(lang, "Rejection reason histogram", "Rejection reason histogram"),
            rejection_hist,
        ),
        _render_score_histogram(
            _tr(
                lang,
                "Transfer non-empty ratio by taxonomy",
                "Transfer non-empty ratio by taxonomy",
            ),
            transfer_by_tax,
        ),
    ]

    sample_rows: list[list[Any]] = []
    seen: set[tuple[str, str, str]] = set()

    if rejection_filter == "all":
        top_selected = sorted(
            filtered_selected,
            key=lambda row: float(row.get("learning_value_score", 0.0)),
            reverse=True,
        )[:5]
        for item in top_selected:
            key = (
                "top_selected",
                _as_str(item.get("chunk_id")),
                _as_str(item.get("text")),
            )
            if key in seen:
                continue
            seen.add(key)
            sample_rows.append(
                [
                    "top_selected",
                    _as_str(item.get("chunk_id")),
                    " | ".join(
                        [
                            _as_str(x)
                            for x in (item.get("phrase_types", []) or [])
                            if _as_str(x)
                        ]
                    ),
                    float(item.get("learning_value_score", 0.0)),
                    _short_text(_as_str(item.get("text"))),
                    _short_text(_as_str(item.get("expression_transfer"))),
                    _short_text(_as_str(item.get("selection_reason"))),
                ]
            )

        taxonomy_bucket: dict[str, dict[str, Any]] = {}
        for item in sorted(
            filtered_selected,
            key=lambda row: float(row.get("learning_value_score", 0.0)),
            reverse=True,
        ):
            for ptype in item.get("phrase_types", []) or []:
                key = _as_str(ptype)
                if not key or key in taxonomy_bucket:
                    continue
                taxonomy_bucket[key] = item
        for ptype, item in list(taxonomy_bucket.items())[:5]:
            key = (
                "taxonomy_example",
                _as_str(item.get("chunk_id")),
                _as_str(item.get("text")),
            )
            if key in seen:
                continue
            seen.add(key)
            sample_rows.append(
                [
                    f"taxonomy_example:{ptype}",
                    _as_str(item.get("chunk_id")),
                    " | ".join(
                        [
                            _as_str(x)
                            for x in (item.get("phrase_types", []) or [])
                            if _as_str(x)
                        ]
                    ),
                    float(item.get("learning_value_score", 0.0)),
                    _short_text(_as_str(item.get("text"))),
                    _short_text(_as_str(item.get("expression_transfer"))),
                    _short_text(_as_str(item.get("selection_reason"))),
                ]
            )

        transfer_examples = [
            item
            for item in filtered_selected
            if _as_str(item.get("expression_transfer"))
        ][:5]
        for item in transfer_examples:
            key = (
                "transfer_example",
                _as_str(item.get("chunk_id")),
                _as_str(item.get("text")),
            )
            if key in seen:
                continue
            seen.add(key)
            sample_rows.append(
                [
                    "transfer_example",
                    _as_str(item.get("chunk_id")),
                    " | ".join(
                        [
                            _as_str(x)
                            for x in (item.get("phrase_types", []) or [])
                            if _as_str(x)
                        ]
                    ),
                    float(item.get("learning_value_score", 0.0)),
                    _short_text(_as_str(item.get("text"))),
                    _short_text(_as_str(item.get("expression_transfer"))),
                    _short_text(_as_str(item.get("selection_reason"))),
                ]
            )

    for item in filtered_rejected[:3]:
        candidate = item.get("item") if isinstance(item.get("item"), dict) else {}
        key = (
            "rejected_example",
            _as_str(item.get("chunk_id")),
            _as_str(candidate.get("text")),
        )
        if key in seen:
            continue
        seen.add(key)
        sample_rows.append(
            [
                f"rejected:{_as_str(item.get('reason_category'))}",
                _as_str(item.get("chunk_id")),
                " | ".join(
                    [
                        _as_str(x)
                        for x in (candidate.get("phrase_types", []) or [])
                        if _as_str(x)
                    ]
                ),
                float(candidate.get("learning_value_score", 0.0)),
                _short_text(_as_str(candidate.get("text"))),
                _short_text(_as_str(candidate.get("expression_transfer"))),
                _short_text(_as_str(item.get("reason"))),
            ]
        )

    return (
        "\n".join(lines),
        sample_rows,
        taxonomy_choices,
        rejection_choices,
        chunk_choices,
    )


def build_interface() -> gr.Blocks:
    """Construct the Gradio Blocks interface.

    Two tabs:
    - Run: upload file + per-run overrides + download link
    - Config: (reserved for future work, not implemented yet)
    """

    cfg = _load_app_config()
    try:
        cfg.resolve_extract_prompt_path(
            material_profile=cfg.material_profile,
            difficulty=cfg.cloze_difficulty,
            learning_mode=getattr(cfg, "learning_mode", "expression_mining"),
        )
        cfg.resolve_explain_prompt_path(
            material_profile=cfg.material_profile,
            difficulty=cfg.cloze_difficulty,
            learning_mode=getattr(cfg, "learning_mode", "expression_mining"),
        )
    except Exception:
        # Keep UI startup resilient; users can still inspect/fix prompt files in Prompt tab.
        logger.warning("prompt auto-seed during web startup failed", exc_info=True)
    env_file = _resolve_env_file()
    cfg_view = _load_env_view(cfg, env_file)
    prompt_files = _prompt_file_map(cfg)
    initial_ui_lang = _normalize_ui_lang(getattr(cfg, "prompt_lang", "en"))
    initial_run_content_type = _normalize_prompt_content_type(cfg.content_profile)
    initial_run_learning_mode = _normalize_prompt_learning_mode(
        getattr(cfg, "learning_mode", "expression_mining")
    )
    initial_run_difficulty = _normalize_prompt_difficulty(cfg.cloze_difficulty)
    initial_prompt_key = ""
    for key in prompt_files:
        if _load_prompt_mode(key, prompt_files, lang=initial_ui_lang) == "extraction":
            initial_prompt_key = key
            break
    if not initial_prompt_key and prompt_files:
        initial_prompt_key = next(iter(prompt_files))
    initial_prompt_text, initial_prompt_status = _load_prompt_template(
        initial_prompt_key, prompt_files, lang=initial_ui_lang
    )
    initial_prompt_mode = _load_prompt_mode(
        initial_prompt_key, prompt_files, lang=initial_ui_lang
    )
    (
        initial_prompt_content_type,
        initial_prompt_learning_mode,
        initial_prompt_difficulty,
    ) = _load_prompt_filter_metadata(
        initial_prompt_key, prompt_files, lang=initial_ui_lang
    )
    run_extract_prompt_choices = _prompt_path_choices(
        cfg,
        lang=initial_ui_lang,
        mode_filter="extraction",
        content_type_filter=initial_run_content_type,
        learning_mode_filter=initial_run_learning_mode,
        difficulty_filter=initial_run_difficulty,
        include_auto=True,
    )
    run_explain_prompt_choices = _prompt_path_choices(
        cfg,
        lang=initial_ui_lang,
        mode_filter="explanation",
        content_type_filter=initial_run_content_type,
        learning_mode_filter=initial_run_learning_mode,
        difficulty_filter=initial_run_difficulty,
        include_auto=True,
    )
    config_extract_prompt_choices = _prompt_path_choices(
        cfg,
        lang=initial_ui_lang,
        mode_filter="extraction",
        include_auto=True,
    )
    config_explain_prompt_choices = _prompt_path_choices(
        cfg,
        lang=initial_ui_lang,
        mode_filter="explanation",
        include_auto=True,
    )
    initial_runs = _scan_runs(cfg, limit=30)
    if initial_runs:
        initial_run_choices = [
            (_run_choice_label(run, lang=initial_ui_lang), run.run_id)
            for run in initial_runs
        ]
        initial_run_selected = initial_runs[0].run_id
        initial_run_detail, initial_run_download = _load_run_detail(
            initial_run_selected, cfg, lang=initial_ui_lang
        )
    else:
        initial_run_choices = []
        initial_run_selected = None
        initial_run_detail = _tr(initial_ui_lang, "No runs found.", "No runs found.")
        initial_run_download = None
    (
        initial_analysis_md,
        initial_samples_rows,
        initial_taxonomy_choices,
        initial_rejection_choices,
        initial_chunk_choices,
    ) = _build_run_analysis(
        initial_run_selected,
        cfg,
        lang=initial_ui_lang,
        taxonomy_filter="all",
        transfer_filter="all",
        rejection_filter="all",
        chunk_filter="all",
    )

    with gr.Blocks(title="ClawLingua Web UI") as demo:
        with gr.Row():
            ui_lang = gr.Dropdown(
                choices=[("English", "en"), ("中文", "zh")],
                value=initial_ui_lang,
                label=_tr(initial_ui_lang, "UI language", "鐣岄潰璇█"),
                scale=1,
            )
        title_md = gr.Markdown(
            _tr(
                initial_ui_lang,
                "# ClawLingua Web UI\nLocal deck builder for text learning.",
                "# ClawLingua Web UI\nLocal deck builder for text learning.",
            )
        )

        with gr.Tab(_tr(initial_ui_lang, "Run", "杩愯")) as run_tab:
            with gr.Row():
                input_file = gr.File(
                    label=_tr(initial_ui_lang, "Input file", "杈撳叆鏂囦欢"),
                    file_types=[".txt", ".md", ".markdown", ".epub"],
                    file_count="single",
                )
                deck_title = gr.Textbox(
                    label=_tr(
                        initial_ui_lang,
                        "Deck title (optional)",
                        "鐗岀粍鍚嶇О锛堝彲閫夛級",
                    )
                )

            with gr.Row():
                source_lang = gr.Dropdown(
                    choices=["en", "zh", "ja", "de", "fr"],
                    value=cfg.default_source_lang,
                    label=_tr(initial_ui_lang, "Source language", "婧愯瑷€"),
                )
                target_lang = gr.Dropdown(
                    choices=["zh", "en", "ja", "de", "fr"],
                    value=cfg.default_target_lang,
                    label=_tr(initial_ui_lang, "Target language", "鐩爣璇█"),
                )
                content_profile = gr.Dropdown(
                    choices=[
                        "prose_article",
                        "transcript_dialogue",
                        "textbook_examples",
                    ],
                    value=cfg.content_profile,
                    label=_tr(initial_ui_lang, "Content profile", "鍐呭绫诲瀷"),
                )
                learning_mode = gr.Dropdown(
                    choices=["expression_mining", "reading_support"],
                    value=getattr(cfg, "learning_mode", "expression_mining"),
                    label=_tr(initial_ui_lang, "Learning mode", "瀛︿範妯″紡"),
                )
                difficulty = gr.Dropdown(
                    choices=["beginner", "intermediate", "advanced"],
                    value=cfg.cloze_difficulty,
                    label=_tr(initial_ui_lang, "Difficulty", "闅惧害"),
                )

            with gr.Row():
                run_extract_prompt = gr.Dropdown(
                    choices=run_extract_prompt_choices,
                    value="",
                    label=_tr(
                        initial_ui_lang,
                        "Extraction prompt (run override)",
                        "Extraction prompt (run override)",
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "Equivalent to CLI --extract-prompt.",
                        "Equivalent to CLI --extract-prompt.",
                    ),
                )
                run_explain_prompt = gr.Dropdown(
                    choices=run_explain_prompt_choices,
                    value="",
                    label=_tr(
                        initial_ui_lang,
                        "Explanation prompt (run override)",
                        "Explanation prompt (run override)",
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "Equivalent to CLI --explain-prompt.",
                        "Equivalent to CLI --explain-prompt.",
                    ),
                )

            with gr.Row():
                max_notes = gr.Number(
                    label=_tr(
                        initial_ui_lang,
                        "Max notes (0 = no limit)",
                        "Max notes (0 = no limit)",
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "Maximum notes after dedupe. Empty/0 means no limit.",
                        "Maximum notes after dedupe. Empty/0 means no limit.",
                    ),
                    value=None,
                    precision=0,
                )
                input_char_limit = gr.Number(
                    label=_tr(initial_ui_lang, "Input char limit", "Input char limit"),
                    info=_tr(
                        initial_ui_lang,
                        "Only process the first N chars of input. Empty means no limit.",
                        "Only process the first N chars of input. Empty means no limit.",
                    ),
                    value=None,
                    precision=0,
                )

            with gr.Accordion(
                _tr(initial_ui_lang, "Advanced", "Advanced"), open=False
            ) as run_advanced:
                cloze_min_chars = gr.Number(
                    label=_tr(
                        initial_ui_lang,
                        "Cloze min chars (override env)",
                        "Cloze min chars (override env)",
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "One-run override for CLAWLINGUA_CLOZE_MIN_CHARS.",
                        "One-run override for CLAWLINGUA_CLOZE_MIN_CHARS.",
                    ),
                    value=cfg.cloze_min_chars,
                    precision=0,
                )
                chunk_max_chars = gr.Number(
                    label=_tr(
                        initial_ui_lang,
                        "Chunk max chars (override env)",
                        "Chunk max chars (override env)",
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "One-run override for CLAWLINGUA_CHUNK_MAX_CHARS.",
                        "One-run override for CLAWLINGUA_CHUNK_MAX_CHARS.",
                    ),
                    value=cfg.chunk_max_chars,
                    precision=0,
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=cfg.llm_temperature,
                    step=0.05,
                    label=_tr(
                        initial_ui_lang,
                        "Temperature (override env)",
                        "Temperature (override env)",
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "0 is more deterministic; higher values are more random.",
                        "0 is more deterministic; higher values are more random.",
                    ),
                )
                save_intermediate = gr.Checkbox(
                    label=_tr(
                        initial_ui_lang,
                        "Save intermediate files",
                        "Save intermediate files",
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.",
                        "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.",
                    ),
                    value=cfg.save_intermediate,
                )
                continue_on_error = gr.Checkbox(
                    label=_tr(
                        initial_ui_lang, "Continue on error", "Continue on error"
                    ),
                    info=_tr(
                        initial_ui_lang,
                        "If enabled, continue processing after per-item failures.",
                        "If enabled, continue processing after per-item failures.",
                    ),
                    value=False,
                )

            run_button = gr.Button(_tr(initial_ui_lang, "Run", "Run"))

            run_status = gr.Markdown(label=_tr(initial_ui_lang, "Status", "Status"))
            output_file = gr.File(
                label=_tr(initial_ui_lang, "Download .apkg", "Download .apkg"),
                interactive=False,
            )
            recent_runs_heading = gr.Markdown(
                _tr(initial_ui_lang, "### Recent runs", "### Recent runs")
            )
            with gr.Row():
                refresh_runs_button = gr.Button(
                    _tr(initial_ui_lang, "Refresh runs", "Refresh runs")
                )
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
            analytics_heading = gr.Markdown(
                _tr(initial_ui_lang, "### Run analytics", "### 杩愯缁熻鍒嗘瀽"),
                render=False,
            )
            taxonomy_filter = gr.Dropdown(
                choices=initial_taxonomy_choices,
                value="all",
                label=_tr(initial_ui_lang, "Taxonomy filter", "taxonomy 杩囨护"),
                render=False,
            )
            transfer_filter = gr.Dropdown(
                choices=[
                    ("all", "all"),
                    ("with_transfer", "with_transfer"),
                    ("without_transfer", "without_transfer"),
                ],
                value="all",
                label=_tr(initial_ui_lang, "Transfer filter", "transfer 杩囨护"),
                render=False,
            )
            rejection_filter = gr.Dropdown(
                choices=initial_rejection_choices,
                value="all",
                label=_tr(initial_ui_lang, "Rejection filter", "鎷掔粷鍘熷洜杩囨护"),
                render=False,
            )
            chunk_filter = gr.Dropdown(
                choices=initial_chunk_choices,
                value="all",
                label=_tr(initial_ui_lang, "Chunk filter", "chunk 杩囨护"),
                render=False,
            )
            apply_analysis_filter_btn = gr.Button(
                _tr(initial_ui_lang, "Apply filters", "搴旂敤杩囨护"),
                render=False,
            )
            run_analysis = gr.Markdown(value=initial_analysis_md, render=False)
            run_samples = gr.Dataframe(
                headers=[
                    "kind",
                    "chunk_id",
                    "phrase_types",
                    "score",
                    "text",
                    "expression_transfer",
                    "reason",
                ],
                datatype=["str", "str", "str", "number", "str", "str", "str"],
                interactive=False,
                wrap=True,
                value=initial_samples_rows,
                label=_tr(
                    initial_ui_lang, "Representative samples", "Representative samples"
                ),
                render=False,
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
                mode,
                diff,
                extract_prompt_val,
                explain_prompt_val,
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
                    learning_mode=mode,
                    difficulty=diff,
                    max_notes=_to_optional_int(max_notes_val, min_value=1),
                    input_char_limit=_to_optional_int(input_limit_val, min_value=1),
                    cloze_min_chars=_to_optional_int(cloze_min_val, min_value=0),
                    chunk_max_chars=_to_optional_int(chunk_max_val, min_value=1),
                    temperature=_to_optional_float(temperature_val),
                    save_intermediate=bool(save_inter_val),
                    continue_on_error=bool(continue_on_error_val),
                    prompt_lang=lang,
                    extract_prompt=_as_str(extract_prompt_val),
                    explain_prompt=_as_str(explain_prompt_val),
                )
                cfg_now = _load_app_config()
                run_id = _as_str(result.get("run_id")) or None
                selector_update, detail_md, history_download = _refresh_recent_runs(
                    cfg_now,
                    lang=lang,
                    preferred_run_id=run_id,
                )
                (
                    analysis_md,
                    sample_rows,
                    taxonomy_choices,
                    rejection_choices,
                    chunk_choices,
                ) = _build_run_analysis(
                    run_id,
                    cfg_now,
                    lang=lang,
                    taxonomy_filter="all",
                    transfer_filter="all",
                    rejection_filter="all",
                    chunk_filter="all",
                )
                taxonomy_update = gr.update(choices=taxonomy_choices, value="all")
                rejection_update = gr.update(choices=rejection_choices, value="all")
                chunk_update = gr.update(choices=chunk_choices, value="all")
                transfer_update = gr.update(value="all")

                if result.get("status") != "ok":
                    msg = result.get("message") or "Unknown error"
                    run_line = f"- run_id: `{run_id}`\n" if run_id else ""
                    status_md = f"{_tr(lang, 'Failed', 'Failed')}\n\n{run_line}- {_tr(lang, 'Error', 'Error')}: `{msg}`"
                    return (
                        status_md,
                        None,
                        selector_update,
                        detail_md,
                        history_download,
                        analysis_md,
                        sample_rows,
                        taxonomy_update,
                        transfer_update,
                        rejection_update,
                        chunk_update,
                    )

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
                return (
                    status_md,
                    out_path,
                    selector_update,
                    detail_md,
                    history_download,
                    analysis_md,
                    sample_rows,
                    taxonomy_update,
                    transfer_update,
                    rejection_update,
                    chunk_update,
                )

            def _on_refresh_runs(
                ui_lang_val: str,
                selected_run_id: str | None,
            ) -> tuple[Any, str, str | None, str, list[list[Any]], Any, Any, Any, Any]:
                lang = _normalize_ui_lang(ui_lang_val)
                cfg_now = _load_app_config()
                selector_update, detail_md, history_download = _refresh_recent_runs(
                    cfg_now,
                    lang=lang,
                    preferred_run_id=selected_run_id,
                )
                run_id_next = _as_str(
                    selector_update.get("value")
                    if isinstance(selector_update, dict)
                    else selected_run_id
                )
                (
                    analysis_md,
                    sample_rows,
                    taxonomy_choices,
                    rejection_choices,
                    chunk_choices,
                ) = _build_run_analysis(
                    run_id_next,
                    cfg_now,
                    lang=lang,
                    taxonomy_filter="all",
                    transfer_filter="all",
                    rejection_filter="all",
                    chunk_filter="all",
                )
                return (
                    selector_update,
                    detail_md,
                    history_download,
                    analysis_md,
                    sample_rows,
                    gr.update(choices=taxonomy_choices, value="all"),
                    gr.update(value="all"),
                    gr.update(choices=rejection_choices, value="all"),
                    gr.update(choices=chunk_choices, value="all"),
                )

            def _on_run_selected(
                run_id_val: str | None,
                ui_lang_val: str,
            ) -> tuple[str, str | None, str, list[list[Any]], Any, Any, Any, Any]:
                lang = _normalize_ui_lang(ui_lang_val)
                cfg_now = _load_app_config()
                detail_md, download_path = _load_run_detail(
                    run_id_val, cfg_now, lang=lang
                )
                (
                    analysis_md,
                    sample_rows,
                    taxonomy_choices,
                    rejection_choices,
                    chunk_choices,
                ) = _build_run_analysis(
                    run_id_val,
                    cfg_now,
                    lang=lang,
                    taxonomy_filter="all",
                    transfer_filter="all",
                    rejection_filter="all",
                    chunk_filter="all",
                )
                return (
                    detail_md,
                    download_path,
                    analysis_md,
                    sample_rows,
                    gr.update(choices=taxonomy_choices, value="all"),
                    gr.update(value="all"),
                    gr.update(choices=rejection_choices, value="all"),
                    gr.update(choices=chunk_choices, value="all"),
                )

            def _on_apply_analysis_filters(
                run_id_val: str | None,
                ui_lang_val: str,
                taxonomy_val: str,
                transfer_val: str,
                rejection_val: str,
                chunk_val: str,
            ) -> tuple[str, list[list[Any]]]:
                lang = _normalize_ui_lang(ui_lang_val)
                cfg_now = _load_app_config()
                analysis_md, sample_rows, _, _, _ = _build_run_analysis(
                    run_id_val,
                    cfg_now,
                    lang=lang,
                    taxonomy_filter=taxonomy_val or "all",
                    transfer_filter=transfer_val or "all",
                    rejection_filter=rejection_val or "all",
                    chunk_filter=chunk_val or "all",
                )
                return analysis_md, sample_rows

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
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    max_notes,
                    input_char_limit,
                    cloze_min_chars,
                    chunk_max_chars,
                    temperature,
                    save_intermediate,
                    continue_on_error,
                    ui_lang,
                ],
                outputs=[
                    run_status,
                    output_file,
                    run_selector,
                    run_detail,
                    run_download_file,
                    run_analysis,
                    run_samples,
                    taxonomy_filter,
                    transfer_filter,
                    rejection_filter,
                    chunk_filter,
                ],
            )

            refresh_runs_button.click(
                _on_refresh_runs,
                inputs=[ui_lang, run_selector],
                outputs=[
                    run_selector,
                    run_detail,
                    run_download_file,
                    run_analysis,
                    run_samples,
                    taxonomy_filter,
                    transfer_filter,
                    rejection_filter,
                    chunk_filter,
                ],
            )

            run_selector.change(
                _on_run_selected,
                inputs=[run_selector, ui_lang],
                outputs=[
                    run_detail,
                    run_download_file,
                    run_analysis,
                    run_samples,
                    taxonomy_filter,
                    transfer_filter,
                    rejection_filter,
                    chunk_filter,
                ],
            )
            apply_analysis_filter_btn.click(
                _on_apply_analysis_filters,
                inputs=[
                    run_selector,
                    ui_lang,
                    taxonomy_filter,
                    transfer_filter,
                    rejection_filter,
                    chunk_filter,
                ],
                outputs=[run_analysis, run_samples],
            )
        with gr.Tab(_tr(initial_ui_lang, "Config", "閰嶇疆")) as config_tab:
            config_heading = gr.Markdown(
                _tr(
                    initial_ui_lang,
                    "### Config (.env editor)",
                    "### 閰嶇疆锛?env 缂栬緫鍣級",
                )
            )

            with gr.Accordion(
                _tr(initial_ui_lang, "Extraction LLM", "Extraction LLM"), open=True
            ) as llm_accordion:
                llm_base_url = gr.Textbox(
                    label="CLAWLINGUA_LLM_BASE_URL",
                    value=cfg_view.get("CLAWLINGUA_LLM_BASE_URL", ""),
                    info=_tr(
                        initial_ui_lang,
                        "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).",
                        "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).",
                    ),
                )
                llm_api_key = gr.Textbox(
                    label="CLAWLINGUA_LLM_API_KEY",
                    value=cfg_view.get("CLAWLINGUA_LLM_API_KEY", ""),
                    type="password",
                    info=_tr(
                        initial_ui_lang,
                        "API key for extraction LLM, when required.",
                        "API key for extraction LLM, when required.",
                    ),
                )
                llm_model = gr.Textbox(
                    label="CLAWLINGUA_LLM_MODEL",
                    value=cfg_view.get("CLAWLINGUA_LLM_MODEL", ""),
                    info=_tr(
                        initial_ui_lang,
                        "Model name for extraction LLM.",
                        "Model name for extraction LLM.",
                    ),
                )
                llm_timeout = gr.Textbox(
                    label="CLAWLINGUA_LLM_TIMEOUT_SECONDS",
                    value=cfg_view.get("CLAWLINGUA_LLM_TIMEOUT_SECONDS", "120"),
                    info=_tr(
                        initial_ui_lang,
                        "Request timeout in seconds.",
                        "Request timeout in seconds.",
                    ),
                )
                llm_temperature_env = gr.Textbox(
                    label="CLAWLINGUA_LLM_TEMPERATURE",
                    value=cfg_view.get("CLAWLINGUA_LLM_TEMPERATURE", "0.2"),
                    info=_tr(
                        initial_ui_lang,
                        "Default temperature for extraction LLM.",
                        "Default temperature for extraction LLM.",
                    ),
                )
                llm_chunk_batch_size_env = gr.Textbox(
                    label="CLAWLINGUA_LLM_CHUNK_BATCH_SIZE",
                    value=cfg_view.get("CLAWLINGUA_LLM_CHUNK_BATCH_SIZE", "1"),
                    info="Chunk batch size for cloze LLM calls; 1 means per-chunk requests.",
                )
                extract_prompt_env = gr.Dropdown(
                    choices=config_extract_prompt_choices,
                    value=cfg_view.get("CLAWLINGUA_EXTRACT_PROMPT", ""),
                    label="CLAWLINGUA_EXTRACT_PROMPT",
                    info=_tr(
                        initial_ui_lang,
                        "Default extraction prompt path.",
                        "Default extraction prompt path.",
                    ),
                )
                with gr.Row():
                    llm_list_models_btn = gr.Button(
                        _tr(initial_ui_lang, "List models", "鍒楀嚭妯″瀷")
                    )
                    llm_test_btn = gr.Button(_tr(initial_ui_lang, "Test", "Test"))
                llm_status = gr.Markdown(
                    label=_tr(
                        initial_ui_lang,
                        "Extraction LLM status",
                        "Extraction LLM status",
                    )
                )

            with gr.Accordion(
                _tr(initial_ui_lang, "Explanation LLM", "Explanation LLM"), open=False
            ) as translate_accordion:
                translate_base_url = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_BASE_URL",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_BASE_URL", ""),
                    info=_tr(
                        initial_ui_lang,
                        "Optional base URL for explanation model.",
                        "Optional base URL for explanation model.",
                    ),
                )
                translate_api_key = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_API_KEY",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_API_KEY", ""),
                    type="password",
                    info=_tr(
                        initial_ui_lang,
                        "API key for explanation LLM, when required.",
                        "API key for explanation LLM, when required.",
                    ),
                )
                translate_model = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_MODEL",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_MODEL", ""),
                    info=_tr(
                        initial_ui_lang,
                        "Model name for explanation LLM.",
                        "Model name for explanation LLM.",
                    ),
                )
                translate_temperature = gr.Textbox(
                    label="CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE",
                    value=cfg_view.get("CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE", ""),
                    info=_tr(
                        initial_ui_lang,
                        "Default temperature for explanation LLM.",
                        "Default temperature for explanation LLM.",
                    ),
                )
                explain_prompt_env = gr.Dropdown(
                    choices=config_explain_prompt_choices,
                    value=cfg_view.get("CLAWLINGUA_EXPLAIN_PROMPT", ""),
                    label="CLAWLINGUA_EXPLAIN_PROMPT",
                    info=_tr(
                        initial_ui_lang,
                        "Default explanation prompt path.",
                        "Default explanation prompt path.",
                    ),
                )
                with gr.Row():
                    translate_list_models_btn = gr.Button(
                        _tr(
                            initial_ui_lang,
                            "List models (explanation)",
                            "鍒楀嚭瑙ｉ噴妯″瀷",
                        )
                    )
                    translate_test_btn = gr.Button(
                        _tr(initial_ui_lang, "Test (explanation)", "Test (explanation)")
                    )
                translate_status = gr.Markdown(
                    label=_tr(
                        initial_ui_lang,
                        "Explanation LLM status",
                        "Explanation LLM status",
                    )
                )

            with gr.Accordion(
                _tr(initial_ui_lang, "Chunk & Cloze", "Chunk & Cloze"), open=False
            ) as chunk_accordion:
                chunk_max_chars_env = gr.Textbox(
                    label="CLAWLINGUA_CHUNK_MAX_CHARS",
                    value=cfg_view.get("CLAWLINGUA_CHUNK_MAX_CHARS", "1800"),
                    info=_tr(
                        initial_ui_lang,
                        "Default max chars per chunk.",
                        "Default max chars per chunk.",
                    ),
                )
                chunk_min_chars_env = gr.Textbox(
                    label="CLAWLINGUA_CHUNK_MIN_CHARS",
                    value=cfg_view.get("CLAWLINGUA_CHUNK_MIN_CHARS", "120"),
                )
                cloze_min_chars_env = gr.Textbox(
                    label="CLAWLINGUA_CLOZE_MIN_CHARS",
                    value=cfg_view.get("CLAWLINGUA_CLOZE_MIN_CHARS", "0"),
                    info=_tr(
                        initial_ui_lang,
                        "Minimum chars required for cloze text.",
                        "Minimum chars required for cloze text.",
                    ),
                )
                cloze_max_per_chunk_env = gr.Textbox(
                    label="CLAWLINGUA_CLOZE_MAX_PER_CHUNK",
                    value=cfg_view.get("CLAWLINGUA_CLOZE_MAX_PER_CHUNK", ""),
                    info=_tr(
                        initial_ui_lang,
                        "Max cards per chunk after dedupe. Empty/0 means unlimited.",
                        "Max cards per chunk after dedupe. Empty/0 means unlimited.",
                    ),
                )
                validate_retry_enable_env = gr.Textbox(
                    label="CLAWLINGUA_VALIDATE_FORMAT_RETRY_ENABLE",
                    value=cfg_view.get(
                        "CLAWLINGUA_VALIDATE_FORMAT_RETRY_ENABLE", "true"
                    ),
                )
                validate_retry_max_env = gr.Textbox(
                    label="CLAWLINGUA_VALIDATE_FORMAT_RETRY_MAX",
                    value=cfg_view.get("CLAWLINGUA_VALIDATE_FORMAT_RETRY_MAX", "3"),
                )
                validate_retry_llm_enable_env = gr.Textbox(
                    label="CLAWLINGUA_VALIDATE_FORMAT_RETRY_LLM_ENABLE",
                    value=cfg_view.get(
                        "CLAWLINGUA_VALIDATE_FORMAT_RETRY_LLM_ENABLE", "true"
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
                    info=_tr(
                        initial_ui_lang,
                        "Prompt language for multi-lingual prompts (en/zh).",
                        "Prompt language for multi-lingual prompts (en/zh).",
                    ),
                )

            with gr.Accordion(
                _tr(initial_ui_lang, "Paths & defaults", "Paths & defaults"), open=False
            ) as paths_accordion:
                output_dir_env = gr.Textbox(
                    label="CLAWLINGUA_OUTPUT_DIR",
                    value=cfg_view.get("CLAWLINGUA_OUTPUT_DIR", "./runs"),
                    info=_tr(
                        initial_ui_lang,
                        "Directory for intermediate run data (JSONL, media).",
                        "Directory for intermediate run data (JSONL, media).",
                    ),
                )
                export_dir_env = gr.Textbox(
                    label="CLAWLINGUA_EXPORT_DIR",
                    value=cfg_view.get("CLAWLINGUA_EXPORT_DIR", "./outputs"),
                    info=_tr(
                        initial_ui_lang,
                        "Default directory for exported decks.",
                        "Default directory for exported decks.",
                    ),
                )
                log_dir_env = gr.Textbox(
                    label="CLAWLINGUA_LOG_DIR",
                    value=cfg_view.get("CLAWLINGUA_LOG_DIR", "./logs"),
                    info=_tr(
                        initial_ui_lang,
                        "Directory for log files.",
                        "Directory for log files.",
                    ),
                )
                default_deck_name_env = gr.Textbox(
                    label="CLAWLINGUA_DEFAULT_DECK_NAME",
                    value=cfg_view.get(
                        "CLAWLINGUA_DEFAULT_DECK_NAME", cfg.default_deck_name
                    ),
                )

            with gr.Accordion(
                _tr(initial_ui_lang, "TTS voices (Edge)", "TTS voices (Edge)"),
                open=False,
            ) as tts_accordion:
                tts_hint_md = gr.Markdown(
                    _tr(
                        initial_ui_lang,
                        "Voice reference: [Edge TTS Voice Samples](https://tts.travisvn.com/)",
                        "鍏蜂綋鐨勯煶鑹插彲浠ュ弬鑰僛Edge TTS Voice Samples](https://tts.travisvn.com/)",
                    )
                )
                tts_voice1_env = gr.Textbox(
                    label="CLAWLINGUA_TTS_EDGE_VOICE1",
                    value=cfg_view.get("CLAWLINGUA_TTS_EDGE_VOICE1", ""),
                    info=_tr(
                        initial_ui_lang,
                        "Configure 4 voice slots used for random selection.",
                        "Configure 4 voice slots used for random selection.",
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
                load_defaults_btn = gr.Button(
                    _tr(
                        initial_ui_lang,
                        "Load defaults from ENV_EXAMPLE.md",
                        "Load defaults from ENV_EXAMPLE.md",
                    )
                )
                save_config_btn = gr.Button(
                    _tr(initial_ui_lang, "Save config", "淇濆瓨閰嶇疆")
                )
            save_config_status = gr.Markdown()

            def _on_list_models(
                base_url: str, api_key: str, timeout_raw: Any, ui_lang_val: str
            ) -> str:
                return _list_models_markdown(
                    base_url=base_url,
                    api_key=api_key,
                    timeout_seconds=_to_timeout_seconds(timeout_raw),
                    lang=_normalize_ui_lang(ui_lang_val),
                )

            def _on_test_models(
                base_url: str, api_key: str, timeout_raw: Any, ui_lang_val: str
            ) -> str:
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
                    f"✅ {_tr(lang, 'Loaded defaults from ENV_EXAMPLE.md (not yet saved).', 'Loaded defaults from ENV_EXAMPLE.md (not yet saved).')}",
                )

            load_defaults_btn.click(
                _on_load_defaults,
                inputs=[
                    llm_base_url,
                    llm_api_key,
                    llm_model,
                    llm_timeout,
                    llm_temperature_env,
                    llm_chunk_batch_size_env,
                    translate_base_url,
                    translate_api_key,
                    translate_model,
                    translate_temperature,
                    chunk_max_chars_env,
                    chunk_min_chars_env,
                    cloze_min_chars_env,
                    cloze_max_per_chunk_env,
                    validate_retry_enable_env,
                    validate_retry_max_env,
                    validate_retry_llm_enable_env,
                    content_profile_env,
                    cloze_difficulty_env,
                    prompt_lang_env,
                    extract_prompt_env,
                    explain_prompt_env,
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
                    llm_chunk_batch_size_env,
                    translate_base_url,
                    translate_api_key,
                    translate_model,
                    translate_temperature,
                    chunk_max_chars_env,
                    chunk_min_chars_env,
                    cloze_min_chars_env,
                    cloze_max_per_chunk_env,
                    validate_retry_enable_env,
                    validate_retry_max_env,
                    validate_retry_llm_enable_env,
                    content_profile_env,
                    cloze_difficulty_env,
                    prompt_lang_env,
                    extract_prompt_env,
                    explain_prompt_env,
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
                llm_chunk_batch_size_val,
                translate_base_url_val,
                translate_api_key_val,
                translate_model_val,
                translate_temperature_val,
                chunk_max_chars_val,
                chunk_min_chars_val,
                cloze_min_chars_val,
                cloze_max_per_chunk_val,
                validate_retry_enable_val,
                validate_retry_max_val,
                validate_retry_llm_enable_val,
                content_profile_val,
                cloze_difficulty_val,
                prompt_lang_val,
                extract_prompt_env_val,
                explain_prompt_env_val,
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
                    "CLAWLINGUA_LLM_CHUNK_BATCH_SIZE": llm_chunk_batch_size_val or "",
                    "CLAWLINGUA_TRANSLATE_LLM_BASE_URL": translate_base_url_val or "",
                    "CLAWLINGUA_TRANSLATE_LLM_API_KEY": translate_api_key_val or "",
                    "CLAWLINGUA_TRANSLATE_LLM_MODEL": translate_model_val or "",
                    "CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE": translate_temperature_val
                    or "",
                    "CLAWLINGUA_CHUNK_MAX_CHARS": chunk_max_chars_val or "",
                    "CLAWLINGUA_CHUNK_MIN_CHARS": chunk_min_chars_val or "",
                    "CLAWLINGUA_CLOZE_MIN_CHARS": cloze_min_chars_val or "",
                    "CLAWLINGUA_CLOZE_MAX_PER_CHUNK": cloze_max_per_chunk_val or "",
                    "CLAWLINGUA_VALIDATE_FORMAT_RETRY_ENABLE": validate_retry_enable_val
                    or "",
                    "CLAWLINGUA_VALIDATE_FORMAT_RETRY_MAX": validate_retry_max_val
                    or "",
                    "CLAWLINGUA_VALIDATE_FORMAT_RETRY_LLM_ENABLE": validate_retry_llm_enable_val
                    or "",
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
                    llm_chunk_batch_size_env,
                    translate_base_url,
                    translate_api_key,
                    translate_model,
                    translate_temperature,
                    chunk_max_chars_env,
                    chunk_min_chars_env,
                    cloze_min_chars_env,
                    cloze_max_per_chunk_env,
                    validate_retry_enable_env,
                    validate_retry_max_env,
                    validate_retry_llm_enable_env,
                    content_profile_env,
                    cloze_difficulty_env,
                    prompt_lang_env,
                    extract_prompt_env,
                    explain_prompt_env,
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

        with gr.Tab(_tr(initial_ui_lang, "Prompt", "Prompt")) as prompt_tab:
            prompt_heading = gr.Markdown(
                _tr(
                    initial_ui_lang,
                    "### Prompt template editor",
                    "### Prompt template editor",
                )
            )
            with gr.Row():
                prompt_content_type_selector = gr.Dropdown(
                    choices=_PROMPT_CONTENT_TYPE_OPTIONS,
                    value=initial_prompt_content_type,
                    label=_tr(
                        initial_ui_lang,
                        "Prompt content type",
                        "Prompt content type",
                    ),
                    scale=1,
                )
                prompt_learning_mode_selector = gr.Dropdown(
                    choices=_PROMPT_LEARNING_MODE_OPTIONS,
                    value=initial_prompt_learning_mode,
                    label=_tr(
                        initial_ui_lang,
                        "Prompt learning mode",
                        "Prompt learning mode",
                    ),
                    scale=1,
                )
                prompt_difficulty_selector = gr.Dropdown(
                    choices=_PROMPT_DIFFICULTY_OPTIONS,
                    value=initial_prompt_difficulty,
                    label=_tr(
                        initial_ui_lang,
                        "Prompt difficulty",
                        "Prompt difficulty",
                    ),
                    scale=1,
                )
            with gr.Row():
                prompt_file_selector = gr.Dropdown(
                    choices=_prompt_choices(
                        initial_ui_lang,
                        mode_filter=initial_prompt_mode or "extraction",
                        content_type_filter=initial_prompt_content_type,
                        learning_mode_filter=initial_prompt_learning_mode,
                        difficulty_filter=initial_prompt_difficulty,
                    ),
                    value=initial_prompt_key,
                    label=_tr(initial_ui_lang, "Prompt file", "Prompt 鏂囦欢"),
                    scale=2,
                )
                prompt_mode_selector = gr.Dropdown(
                    choices=[
                        (
                            _prompt_mode_label("extraction", lang=initial_ui_lang),
                            "extraction",
                        ),
                        (
                            _prompt_mode_label("explanation", lang=initial_ui_lang),
                            "explanation",
                        ),
                    ],
                    value=initial_prompt_mode or "extraction",
                    label=_tr(initial_ui_lang, "Prompt type", "Prompt type"),
                    scale=1,
                )
            with gr.Row():
                prompt_new_name = gr.Textbox(
                    label=_tr(
                        initial_ui_lang, "New prompt file name", "New prompt file name"
                    ),
                    placeholder="my_prompt.json",
                )
                prompt_rename_name = gr.Textbox(
                    label=_tr(initial_ui_lang, "Rename to", "閲嶅懡鍚嶄负"),
                    placeholder="renamed_prompt.json",
                )
            prompt_editor = gr.Textbox(
                label=_tr(initial_ui_lang, "Prompt template", "Prompt 妯℃澘"),
                value=initial_prompt_text,
                lines=24,
            )
            with gr.Row():
                prompt_new_btn = gr.Button(_tr(initial_ui_lang, "New", "鏂板缓"))
                prompt_save_btn = gr.Button(_tr(initial_ui_lang, "Save", "淇濆瓨"))
                prompt_rename_btn = gr.Button(_tr(initial_ui_lang, "Rename", "Rename"))
                prompt_load_default_btn = gr.Button(
                    _tr(initial_ui_lang, "Delete", "鍒犻櫎"), variant="stop"
                )
            prompt_save_confirm = gr.Checkbox(value=False, visible=False)
            prompt_delete_confirm = gr.Checkbox(value=False, visible=False)
            prompt_status = gr.Markdown(
                label=_tr(initial_ui_lang, "Prompt status", "Prompt status"),
                value=initial_prompt_status,
            )

            def _prompt_mode_choices_for_ui(lang: str) -> list[tuple[str, str]]:
                return [
                    (_prompt_mode_label("extraction", lang=lang), "extraction"),
                    (_prompt_mode_label("explanation", lang=lang), "explanation"),
                ]

            def _normalize_dropdown_value(
                current: str, choices: list[Any]
            ) -> str:
                valid_values: set[str] = set()
                for item in choices:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        valid_values.add(_as_str(item[1]))
                    else:
                        valid_values.add(_as_str(item))
                current_value = _as_str(current)
                return current_value if current_value in valid_values else ""

            def _pick_prompt_key(
                prompt_files_now: dict[str, Path],
                *,
                lang: str,
                preferred_key: str = "",
                preferred_mode: str = "",
            ) -> str:
                if preferred_key in prompt_files_now:
                    return preferred_key
                mode_value = _normalize_prompt_mode(preferred_mode)
                if mode_value:
                    for key in prompt_files_now:
                        if (
                            _load_prompt_mode(key, prompt_files_now, lang=lang)
                            == mode_value
                        ):
                            return key
                if prompt_files_now:
                    return next(iter(prompt_files_now))
                return ""

            def _refresh_prompt_controls(
                *,
                lang: str,
                prompt_key: str,
                preferred_mode: str,
                preferred_content_type: str = "all",
                preferred_learning_mode: str = "all",
                preferred_difficulty: str = "all",
                status: str,
                editor_override: str | None,
                run_content_type: str = "all",
                run_learning_mode: str = "all",
                run_difficulty: str = "all",
                run_extract_current: str = "",
                run_explain_current: str = "",
                config_extract_current: str = "",
                config_explain_current: str = "",
            ) -> tuple[Any, ...]:
                cfg_now = _load_app_config()
                prompt_files_now = _prompt_file_map(cfg_now)
                selected_mode = _normalize_prompt_mode(preferred_mode)
                if not selected_mode and prompt_key in prompt_files_now:
                    selected_mode = _load_prompt_mode(
                        prompt_key, prompt_files_now, lang=lang
                    )
                if not selected_mode:
                    selected_mode = "extraction"
                selected_content_type = _normalize_prompt_content_type(
                    preferred_content_type
                )
                selected_learning_mode = _normalize_prompt_learning_mode(
                    preferred_learning_mode
                )
                selected_difficulty = _normalize_prompt_difficulty(preferred_difficulty)
                has_explicit_filters = any(
                    value != "all"
                    for value in (
                        selected_content_type,
                        selected_learning_mode,
                        selected_difficulty,
                    )
                )
                prompt_files_for_mode = _prompt_file_map(
                    cfg_now,
                    mode_filter=selected_mode,
                    content_type_filter=selected_content_type,
                    learning_mode_filter=selected_learning_mode,
                    difficulty_filter=selected_difficulty,
                )
                selected_key = _pick_prompt_key(
                    prompt_files_for_mode,
                    lang=lang,
                    preferred_key=prompt_key,
                    preferred_mode=selected_mode,
                )
                prompt_text = ""
                load_msg = ""
                if selected_key:
                    prompt_text, load_msg = _load_prompt_template(
                        selected_key, prompt_files_now, lang=lang
                    )
                    file_mode = _load_prompt_mode(
                        selected_key, prompt_files_now, lang=lang
                    )
                    if file_mode:
                        selected_mode = file_mode
                        prompt_files_for_mode = _prompt_file_map(
                            cfg_now,
                            mode_filter=selected_mode,
                            content_type_filter=selected_content_type,
                            learning_mode_filter=selected_learning_mode,
                            difficulty_filter=selected_difficulty,
                        )
                    if not has_explicit_filters:
                        (
                            selected_content_type,
                            selected_learning_mode,
                            selected_difficulty,
                        ) = _load_prompt_filter_metadata(
                            selected_key, prompt_files_now, lang=lang
                        )
                        prompt_files_for_mode = _prompt_file_map(
                            cfg_now,
                            mode_filter=selected_mode,
                            content_type_filter=selected_content_type,
                            learning_mode_filter=selected_learning_mode,
                            difficulty_filter=selected_difficulty,
                        )
                if editor_override is not None:
                    prompt_text = editor_override

                prompt_choices_now = _prompt_choices_from_map(
                    prompt_files_for_mode, lang=lang
                )
                selected_key = _normalize_dropdown_value(
                    selected_key, prompt_choices_now
                )
                if not selected_key and editor_override is None:
                    prompt_text = ""
                mode_choices_now = _prompt_mode_choices_for_ui(lang)
                run_content_type_value = _normalize_prompt_content_type(run_content_type)
                run_learning_mode_value = _normalize_prompt_learning_mode(
                    run_learning_mode
                )
                run_difficulty_value = _normalize_prompt_difficulty(run_difficulty)
                run_extract_choices_now = _prompt_path_choices(
                    cfg_now,
                    lang=lang,
                    mode_filter="extraction",
                    content_type_filter=run_content_type_value,
                    learning_mode_filter=run_learning_mode_value,
                    difficulty_filter=run_difficulty_value,
                    include_auto=True,
                )
                run_explain_choices_now = _prompt_path_choices(
                    cfg_now,
                    lang=lang,
                    mode_filter="explanation",
                    content_type_filter=run_content_type_value,
                    learning_mode_filter=run_learning_mode_value,
                    difficulty_filter=run_difficulty_value,
                    include_auto=True,
                )
                config_extract_choices_now = _prompt_path_choices(
                    cfg_now,
                    lang=lang,
                    mode_filter="extraction",
                    include_auto=True,
                )
                config_explain_choices_now = _prompt_path_choices(
                    cfg_now,
                    lang=lang,
                    mode_filter="explanation",
                    include_auto=True,
                )
                status_text = status or load_msg
                return (
                    gr.update(choices=prompt_choices_now, value=selected_key),
                    gr.update(choices=mode_choices_now, value=selected_mode),
                    gr.update(
                        choices=_PROMPT_CONTENT_TYPE_OPTIONS,
                        value=selected_content_type,
                    ),
                    gr.update(
                        choices=_PROMPT_LEARNING_MODE_OPTIONS,
                        value=selected_learning_mode,
                    ),
                    gr.update(
                        choices=_PROMPT_DIFFICULTY_OPTIONS,
                        value=selected_difficulty,
                    ),
                    gr.update(value=prompt_text),
                    gr.update(value=status_text),
                    gr.update(
                        choices=run_extract_choices_now,
                        value=_normalize_dropdown_value(
                            run_extract_current, run_extract_choices_now
                        ),
                    ),
                    gr.update(
                        choices=run_explain_choices_now,
                        value=_normalize_dropdown_value(
                            run_explain_current, run_explain_choices_now
                        ),
                    ),
                    gr.update(
                        choices=config_extract_choices_now,
                        value=_normalize_dropdown_value(
                            config_extract_current, config_extract_choices_now
                        ),
                    ),
                    gr.update(
                        choices=config_explain_choices_now,
                        value=_normalize_dropdown_value(
                            config_explain_current, config_explain_choices_now
                        ),
                    ),
                )

            def _append_prompt_aux_updates(
                updates: tuple[Any, ...],
                *,
                new_name_value: str = "",
                rename_name_value: str = "",
            ) -> tuple[Any, ...]:
                return (
                    *updates,
                    gr.update(value=new_name_value),
                    gr.update(value=rename_name_value),
                    gr.update(value=False),
                    gr.update(value=False),
                )

            def _on_prompt_file_change(
                prompt_key: str,
                prompt_mode: str,
                prompt_content_type: str,
                prompt_learning_mode: str,
                prompt_difficulty: str,
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                config_extract_val: str,
                config_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, ...]:
                lang = _normalize_ui_lang(ui_lang_val)
                return _refresh_prompt_controls(
                    lang=lang,
                    prompt_key=prompt_key,
                    preferred_mode=prompt_mode,
                    preferred_content_type=prompt_content_type,
                    preferred_learning_mode=prompt_learning_mode,
                    preferred_difficulty=prompt_difficulty,
                    status="",
                    editor_override=None,
                    run_content_type=run_content_type,
                    run_learning_mode=run_learning_mode,
                    run_difficulty=run_difficulty,
                    run_extract_current=run_extract_val,
                    run_explain_current=run_explain_val,
                    config_extract_current=config_extract_val,
                    config_explain_current=config_explain_val,
                )

            def _on_prompt_mode_change(
                prompt_mode: str,
                prompt_key: str,
                prompt_content_type: str,
                prompt_learning_mode: str,
                prompt_difficulty: str,
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                config_extract_val: str,
                config_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, ...]:
                lang = _normalize_ui_lang(ui_lang_val)
                return _refresh_prompt_controls(
                    lang=lang,
                    prompt_key=prompt_key,
                    preferred_mode=prompt_mode,
                    preferred_content_type=prompt_content_type,
                    preferred_learning_mode=prompt_learning_mode,
                    preferred_difficulty=prompt_difficulty,
                    status="",
                    editor_override=None,
                    run_content_type=run_content_type,
                    run_learning_mode=run_learning_mode,
                    run_difficulty=run_difficulty,
                    run_extract_current=run_extract_val,
                    run_explain_current=run_explain_val,
                    config_extract_current=config_extract_val,
                    config_explain_current=config_explain_val,
                )

            def _on_prompt_filter_change(
                prompt_content_type: str,
                prompt_learning_mode: str,
                prompt_difficulty: str,
                prompt_mode: str,
                prompt_key: str,
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                config_extract_val: str,
                config_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, ...]:
                lang = _normalize_ui_lang(ui_lang_val)
                return _refresh_prompt_controls(
                    lang=lang,
                    prompt_key=prompt_key,
                    preferred_mode=prompt_mode,
                    preferred_content_type=prompt_content_type,
                    preferred_learning_mode=prompt_learning_mode,
                    preferred_difficulty=prompt_difficulty,
                    status="",
                    editor_override=None,
                    run_content_type=run_content_type,
                    run_learning_mode=run_learning_mode,
                    run_difficulty=run_difficulty,
                    run_extract_current=run_extract_val,
                    run_explain_current=run_explain_val,
                    config_extract_current=config_extract_val,
                    config_explain_current=config_explain_val,
                )

            def _on_prompt_new(
                prompt_key: str,
                new_name: str,
                prompt_mode: str,
                prompt_content_type: str,
                prompt_learning_mode: str,
                prompt_difficulty: str,
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                config_extract_val: str,
                config_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, ...]:
                lang = _normalize_ui_lang(ui_lang_val)
                mode = _normalize_prompt_mode(prompt_mode) or "extraction"
                file_name = _sanitize_prompt_filename(new_name)
                if not file_name:
                    file_name = f"{mode}_prompt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
                if (
                    file_name in _PROMPT_META_FILENAMES
                    or file_name in _PROMPT_TEMPLATE_FILENAMES
                ):
                    status = f"❌ {_tr(lang, 'Prompt file already exists.', 'Prompt file already exists.')}: `{file_name}`"
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode=mode,
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=status,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(updates, new_name_value=file_name)

                cfg_now = _load_app_config()
                prompts_dir = cfg_now.resolve_path(_PROMPT_DIR)
                prompts_dir.mkdir(parents=True, exist_ok=True)
                target_path = prompts_dir / file_name
                if target_path.exists():
                    status = f"❌ {_tr(lang, 'Prompt file already exists.', 'Prompt file already exists.')}: `{target_path}`"
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode=mode,
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=status,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(updates, new_name_value=file_name)

                template_path = _prompt_template_path(cfg_now, mode)
                if template_path is None or not template_path.exists():
                    status = f"❌ {_tr(lang, 'Template prompt file missing.', 'Template prompt file missing.')}: `{template_path}`"
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode=mode,
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=status,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(updates, new_name_value=file_name)

                try:
                    template_payload = json.loads(
                        template_path.read_text(encoding="utf-8")
                    )
                    template_spec = PromptSpec.model_validate(template_payload)
                    payload = template_spec.model_dump(mode="json")
                    payload["name"] = Path(file_name).stem
                    payload["mode"] = mode
                    target_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                except (
                    OSError,
                    json.JSONDecodeError,
                    ValidationError,
                    ValueError,
                ) as exc:
                    status = f"❌ {_tr(lang, 'Failed to save prompt file', 'Failed to save prompt file')}: `{exc}`"
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode=mode,
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=status,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(updates, new_name_value=file_name)

                status = (
                    f"✅ {_tr(lang, 'Prompt file created.', 'Prompt file created.')}\n\n"
                    f"- file: `{target_path}`"
                )
                updates = _refresh_prompt_controls(
                    lang=lang,
                    prompt_key=file_name,
                    preferred_mode=mode,
                    preferred_content_type=prompt_content_type,
                    preferred_learning_mode=prompt_learning_mode,
                    preferred_difficulty=prompt_difficulty,
                    status=status,
                    editor_override=None,
                    run_content_type=run_content_type,
                    run_learning_mode=run_learning_mode,
                    run_difficulty=run_difficulty,
                    run_extract_current=run_extract_val,
                    run_explain_current=run_explain_val,
                    config_extract_current=config_extract_val,
                    config_explain_current=config_explain_val,
                )
                return _append_prompt_aux_updates(updates)

            def _on_prompt_save(
                prompt_key: str,
                prompt_mode: str,
                prompt_content_type: str,
                prompt_learning_mode: str,
                prompt_difficulty: str,
                prompt_template: str,
                save_confirmed: bool,
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                config_extract_val: str,
                config_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, ...]:
                lang = _normalize_ui_lang(ui_lang_val)
                mode = _normalize_prompt_mode(prompt_mode) or "extraction"
                template = (prompt_template or "").rstrip()
                if not save_confirmed:
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode=mode,
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status="",
                        editor_override=template,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(updates)
                if not template.strip():
                    status = f"❌ {_tr(lang, 'Prompt template is empty.', 'Prompt template is empty.')}"
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode=mode,
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=status,
                        editor_override=template,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(updates)

                prompt_files_now = _prompt_file_map(_load_app_config())
                path, payload, msg = _read_prompt_payload(
                    prompt_key, prompt_files_now, lang=lang
                )
                if payload is None or path is None:
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode=mode,
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=msg,
                        editor_override=template,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(updates)

                payload["mode"] = mode
                _set_user_prompt_template(payload, lang=lang, template=template)
                ok, details = _write_prompt_payload(path, payload, lang=lang)
                if not ok:
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode=mode,
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=details,
                        editor_override=template,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(updates)

                status = (
                    f"✅ {_tr(lang, 'Prompt template saved.', 'Prompt template saved.')}\n\n"
                    f"{details}"
                )
                updates = _refresh_prompt_controls(
                    lang=lang,
                    prompt_key=prompt_key,
                    preferred_mode=mode,
                    preferred_content_type=prompt_content_type,
                    preferred_learning_mode=prompt_learning_mode,
                    preferred_difficulty=prompt_difficulty,
                    status=status,
                    editor_override=template,
                    run_content_type=run_content_type,
                    run_learning_mode=run_learning_mode,
                    run_difficulty=run_difficulty,
                    run_extract_current=run_extract_val,
                    run_explain_current=run_explain_val,
                    config_extract_current=config_extract_val,
                    config_explain_current=config_explain_val,
                )
                return _append_prompt_aux_updates(updates)

            def _on_prompt_rename(
                prompt_key: str,
                rename_name: str,
                prompt_mode: str,
                prompt_content_type: str,
                prompt_learning_mode: str,
                prompt_difficulty: str,
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                config_extract_val: str,
                config_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, ...]:
                lang = _normalize_ui_lang(ui_lang_val)
                prompt_files_now = _prompt_file_map(_load_app_config())
                current_path = prompt_files_now.get(prompt_key)
                if current_path is None:
                    status = f"❌ {_tr(lang, 'Failed to load prompt file', 'Failed to load prompt file')}: `{prompt_key}`"
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode="",
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=status,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(
                        updates, rename_name_value=rename_name
                    )

                target_name = _sanitize_prompt_filename(rename_name)
                if not target_name:
                    status = f"❌ {_tr(lang, 'Prompt file name is empty.', 'Prompt file name is empty.')}"
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode="",
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=status,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(
                        updates, rename_name_value=rename_name
                    )

                if (
                    target_name in _PROMPT_META_FILENAMES
                    or target_name in _PROMPT_TEMPLATE_FILENAMES
                ):
                    status = f"❌ {_tr(lang, 'Prompt file already exists.', 'Prompt file already exists.')}: `{target_name}`"
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode="",
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=status,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(
                        updates, rename_name_value=rename_name
                    )

                target_path = current_path.with_name(target_name)
                if target_path.exists():
                    status = f"❌ {_tr(lang, 'Prompt file already exists.', 'Prompt file already exists.')}: `{target_path}`"
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode="",
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=status,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(
                        updates, rename_name_value=rename_name
                    )

                path, payload, msg = _read_prompt_payload(
                    prompt_key, prompt_files_now, lang=lang
                )
                if payload is None or path is None:
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode="",
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=msg,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(
                        updates, rename_name_value=rename_name
                    )

                payload["name"] = target_path.stem
                ok, details = _write_prompt_payload(target_path, payload, lang=lang)
                if not ok:
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode=_normalize_prompt_mode(payload.get("mode")),
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=details,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(
                        updates, rename_name_value=rename_name
                    )

                try:
                    path.unlink()
                except OSError as exc:
                    status = f"⚠️ {_tr(lang, 'Prompt file renamed.', 'Prompt file renamed.')} `{target_path}`; old file cleanup failed: `{exc}`"
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=target_name,
                        preferred_mode=_normalize_prompt_mode(payload.get("mode")),
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=status,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(updates)

                status = (
                    f"✅ {_tr(lang, 'Prompt file renamed.', 'Prompt file renamed.')}\n\n"
                    f"- from: `{path}`\n"
                    f"- to: `{target_path}`\n"
                    f"{details}"
                )
                updates = _refresh_prompt_controls(
                    lang=lang,
                    prompt_key=target_name,
                    preferred_mode=_normalize_prompt_mode(payload.get("mode")),
                    preferred_content_type=prompt_content_type,
                    preferred_learning_mode=prompt_learning_mode,
                    preferred_difficulty=prompt_difficulty,
                    status=status,
                    editor_override=None,
                    run_content_type=run_content_type,
                    run_learning_mode=run_learning_mode,
                    run_difficulty=run_difficulty,
                    run_extract_current=run_extract_val,
                    run_explain_current=run_explain_val,
                    config_extract_current=config_extract_val,
                    config_explain_current=config_explain_val,
                )
                return _append_prompt_aux_updates(updates)

            def _on_prompt_delete(
                prompt_key: str,
                delete_confirmed: bool,
                prompt_mode: str,
                prompt_content_type: str,
                prompt_learning_mode: str,
                prompt_difficulty: str,
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                config_extract_val: str,
                config_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, ...]:
                lang = _normalize_ui_lang(ui_lang_val)
                if not delete_confirmed:
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode="",
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status="",
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(updates)

                prompt_files_now = _prompt_file_map(_load_app_config())
                path = prompt_files_now.get(prompt_key)
                if path is None:
                    status = f"❌ {_tr(lang, 'Failed to load prompt file', 'Failed to load prompt file')}: `{prompt_key}`"
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode="",
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=status,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(updates)

                mode = _load_prompt_mode(prompt_key, prompt_files_now, lang=lang)
                mode_counts = {"extraction": 0, "explanation": 0}
                for key in prompt_files_now:
                    key_mode = _load_prompt_mode(key, prompt_files_now, lang=lang)
                    if key_mode in mode_counts:
                        mode_counts[key_mode] += 1
                if mode == "extraction" and mode_counts["extraction"] <= 1:
                    status = f"❌ {_tr(lang, 'Cannot delete the last Extraction prompt.', 'Cannot delete the last Extraction prompt.')}"
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode=mode,
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=status,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(updates)
                if mode == "explanation" and mode_counts["explanation"] <= 1:
                    status = f"❌ {_tr(lang, 'Cannot delete the last Explanation prompt.', 'Cannot delete the last Explanation prompt.')}"
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode=mode,
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=status,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(updates)

                try:
                    path.unlink()
                except OSError as exc:
                    status = f"❌ {_tr(lang, 'Failed to save prompt file', 'Failed to save prompt file')}: `{exc}`"
                    updates = _refresh_prompt_controls(
                        lang=lang,
                        prompt_key=prompt_key,
                        preferred_mode=mode,
                        preferred_content_type=prompt_content_type,
                        preferred_learning_mode=prompt_learning_mode,
                        preferred_difficulty=prompt_difficulty,
                        status=status,
                        editor_override=None,
                        run_content_type=run_content_type,
                        run_learning_mode=run_learning_mode,
                        run_difficulty=run_difficulty,
                        run_extract_current=run_extract_val,
                        run_explain_current=run_explain_val,
                        config_extract_current=config_extract_val,
                        config_explain_current=config_explain_val,
                    )
                    return _append_prompt_aux_updates(updates)

                status = (
                    f"✅ {_tr(lang, 'Prompt file deleted.', 'Prompt file deleted.')}\n\n"
                    f"- file: `{path}`"
                )
                updates = _refresh_prompt_controls(
                    lang=lang,
                    prompt_key="",
                    preferred_mode=mode,
                    preferred_content_type=prompt_content_type,
                    preferred_learning_mode=prompt_learning_mode,
                    preferred_difficulty=prompt_difficulty,
                    status=status,
                    editor_override=None,
                    run_content_type=run_content_type,
                    run_learning_mode=run_learning_mode,
                    run_difficulty=run_difficulty,
                    run_extract_current=run_extract_val,
                    run_explain_current=run_explain_val,
                    config_extract_current=config_extract_val,
                    config_explain_current=config_explain_val,
                )
                return _append_prompt_aux_updates(updates)

            def _on_run_prompt_filters_change(
                run_content_type: str,
                run_learning_mode: str,
                run_difficulty: str,
                run_extract_val: str,
                run_explain_val: str,
                ui_lang_val: str,
            ) -> tuple[Any, Any]:
                cfg_now = _load_app_config()
                lang = _normalize_ui_lang(ui_lang_val)
                run_extract_choices_now = _prompt_path_choices(
                    cfg_now,
                    lang=lang,
                    mode_filter="extraction",
                    content_type_filter=run_content_type,
                    learning_mode_filter=run_learning_mode,
                    difficulty_filter=run_difficulty,
                    include_auto=True,
                )
                run_explain_choices_now = _prompt_path_choices(
                    cfg_now,
                    lang=lang,
                    mode_filter="explanation",
                    content_type_filter=run_content_type,
                    learning_mode_filter=run_learning_mode,
                    difficulty_filter=run_difficulty,
                    include_auto=True,
                )
                return (
                    gr.update(
                        choices=run_extract_choices_now,
                        value=_normalize_dropdown_value(
                            run_extract_val, run_extract_choices_now
                        ),
                    ),
                    gr.update(
                        choices=run_explain_choices_now,
                        value=_normalize_dropdown_value(
                            run_explain_val, run_explain_choices_now
                        ),
                    ),
                )

            prompt_file_selector.change(
                _on_prompt_file_change,
                inputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                ],
            )
            prompt_mode_selector.change(
                _on_prompt_mode_change,
                inputs=[
                    prompt_mode_selector,
                    prompt_file_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                ],
            )
            prompt_content_type_selector.change(
                _on_prompt_filter_change,
                inputs=[
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_mode_selector,
                    prompt_file_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                ],
            )
            prompt_learning_mode_selector.change(
                _on_prompt_filter_change,
                inputs=[
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_mode_selector,
                    prompt_file_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                ],
            )
            prompt_difficulty_selector.change(
                _on_prompt_filter_change,
                inputs=[
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_mode_selector,
                    prompt_file_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                ],
            )
            content_profile.change(
                _on_run_prompt_filters_change,
                inputs=[
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    ui_lang,
                ],
                outputs=[run_extract_prompt, run_explain_prompt],
            )
            learning_mode.change(
                _on_run_prompt_filters_change,
                inputs=[
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    ui_lang,
                ],
                outputs=[run_extract_prompt, run_explain_prompt],
            )
            difficulty.change(
                _on_run_prompt_filters_change,
                inputs=[
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    ui_lang,
                ],
                outputs=[run_extract_prompt, run_explain_prompt],
            )
            prompt_new_btn.click(
                _on_prompt_new,
                inputs=[
                    prompt_file_selector,
                    prompt_new_name,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    prompt_new_name,
                    prompt_rename_name,
                    prompt_save_confirm,
                    prompt_delete_confirm,
                ],
            )
            prompt_save_btn.click(
                _on_prompt_save,
                inputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_save_confirm,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    prompt_new_name,
                    prompt_rename_name,
                    prompt_save_confirm,
                    prompt_delete_confirm,
                ],
                js="""
(prompt_key, prompt_mode, prompt_content_type, prompt_learning_mode, prompt_difficulty, prompt_template, _save_confirmed, run_content_type, run_learning_mode, run_difficulty, run_extract_val, run_explain_val, config_extract_val, config_explain_val, ui_lang_val) => {
    const message = ui_lang_val === "zh" ? "确认保存当前提示词文件？" : "Confirm saving the current prompt file?";
    return [
        prompt_key,
        prompt_mode,
        prompt_content_type,
        prompt_learning_mode,
        prompt_difficulty,
        prompt_template,
        window.confirm(message),
        run_content_type,
        run_learning_mode,
        run_difficulty,
        run_extract_val,
        run_explain_val,
        config_extract_val,
        config_explain_val,
        ui_lang_val,
    ];
}
""",
            )
            prompt_rename_btn.click(
                _on_prompt_rename,
                inputs=[
                    prompt_file_selector,
                    prompt_rename_name,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    prompt_new_name,
                    prompt_rename_name,
                    prompt_save_confirm,
                    prompt_delete_confirm,
                ],
            )
            prompt_load_default_btn.click(
                _on_prompt_delete,
                inputs=[
                    prompt_file_selector,
                    prompt_delete_confirm,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    content_profile,
                    learning_mode,
                    difficulty,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    ui_lang,
                ],
                outputs=[
                    prompt_file_selector,
                    prompt_mode_selector,
                    prompt_content_type_selector,
                    prompt_learning_mode_selector,
                    prompt_difficulty_selector,
                    prompt_editor,
                    prompt_status,
                    run_extract_prompt,
                    run_explain_prompt,
                    extract_prompt_env,
                    explain_prompt_env,
                    prompt_new_name,
                    prompt_rename_name,
                    prompt_save_confirm,
                    prompt_delete_confirm,
                ],
                js="""
(prompt_key, _delete_confirmed, prompt_mode, prompt_content_type, prompt_learning_mode, prompt_difficulty, run_content_type, run_learning_mode, run_difficulty, run_extract_val, run_explain_val, config_extract_val, config_explain_val, ui_lang_val) => {
    const message = ui_lang_val === "zh" ? "确认删除当前提示词文件？" : "Confirm deleting the current prompt file?";
    return [
        prompt_key,
        window.confirm(message),
        prompt_mode,
        prompt_content_type,
        prompt_learning_mode,
        prompt_difficulty,
        run_content_type,
        run_learning_mode,
        run_difficulty,
        run_extract_val,
        run_explain_val,
        config_extract_val,
        config_explain_val,
        ui_lang_val,
    ];
}
""",
            )

        with gr.Tab(
            _tr(initial_ui_lang, "Run analytics", "杩愯缁熻鍒嗘瀽")
        ) as analytics_tab:
            analytics_heading.render()
            with gr.Row():
                taxonomy_filter.render()
                transfer_filter.render()
                rejection_filter.render()
                chunk_filter.render()
            apply_analysis_filter_btn.render()
            run_analysis.render()
            run_samples.render()

        def _on_ui_lang_change(
            lang_value: str,
            prompt_lang_current: str,
            prompt_key_current: str,
            prompt_mode_current: str,
            prompt_content_type_current: str,
            prompt_learning_mode_current: str,
            prompt_difficulty_current: str,
            run_content_type_current: str,
            run_learning_mode_current: str,
            run_difficulty_current: str,
            run_extract_prompt_current: str,
            run_explain_prompt_current: str,
            run_id_current: str | None,
        ) -> tuple[Any, ...]:
            lang = _normalize_ui_lang(lang_value)
            _ = prompt_lang_current
            prompt_lang_next = lang
            cfg_now = _load_app_config()
            prompt_files_now = _prompt_file_map(cfg_now)
            prompt_mode_pref = _normalize_prompt_mode(prompt_mode_current)
            prompt_content_type_pref = _normalize_prompt_content_type(
                prompt_content_type_current
            )
            prompt_learning_mode_pref = _normalize_prompt_learning_mode(
                prompt_learning_mode_current
            )
            prompt_difficulty_pref = _normalize_prompt_difficulty(
                prompt_difficulty_current
            )
            if not prompt_mode_pref and prompt_key_current in prompt_files_now:
                prompt_mode_pref = _load_prompt_mode(
                    prompt_key_current, prompt_files_now, lang=lang
                )
            if not prompt_mode_pref:
                prompt_mode_pref = "extraction"
            prompt_files_filtered_next = _prompt_file_map(
                cfg_now,
                mode_filter=prompt_mode_pref,
                content_type_filter=prompt_content_type_pref,
                learning_mode_filter=prompt_learning_mode_pref,
                difficulty_filter=prompt_difficulty_pref,
            )
            prompt_key_next = _pick_prompt_key(
                prompt_files_filtered_next,
                lang=lang,
                preferred_key=prompt_key_current,
                preferred_mode=prompt_mode_pref,
            )
            if not prompt_key_next and prompt_files_now:
                prompt_key_next = _pick_prompt_key(
                    prompt_files_now,
                    lang=lang,
                    preferred_key=prompt_key_current,
                    preferred_mode=prompt_mode_pref,
                )
            prompt_mode_next = (
                _load_prompt_mode(prompt_key_next, prompt_files_now, lang=lang)
                or prompt_mode_pref
            )
            prompt_files_filtered_next = _prompt_file_map(
                cfg_now,
                mode_filter=prompt_mode_next,
                content_type_filter=prompt_content_type_pref,
                learning_mode_filter=prompt_learning_mode_pref,
                difficulty_filter=prompt_difficulty_pref,
            )
            if prompt_key_next:
                prompt_template_next, prompt_status_next = _load_prompt_template(
                    prompt_key_next,
                    prompt_files_now,
                    lang=lang,
                )
                if (
                    prompt_content_type_pref == "all"
                    and prompt_learning_mode_pref == "all"
                    and prompt_difficulty_pref == "all"
                ):
                    (
                        prompt_content_type_pref,
                        prompt_learning_mode_pref,
                        prompt_difficulty_pref,
                    ) = _load_prompt_filter_metadata(
                        prompt_key_next, prompt_files_now, lang=lang
                    )
                    prompt_files_filtered_next = _prompt_file_map(
                        cfg_now,
                        mode_filter=prompt_mode_next,
                        content_type_filter=prompt_content_type_pref,
                        learning_mode_filter=prompt_learning_mode_pref,
                        difficulty_filter=prompt_difficulty_pref,
                    )
            else:
                prompt_template_next, prompt_status_next = "", ""
            prompt_mode_choices_next = _prompt_mode_choices_for_ui(lang)
            run_content_type_pref = _normalize_prompt_content_type(
                run_content_type_current
            )
            run_learning_mode_pref = _normalize_prompt_learning_mode(
                run_learning_mode_current
            )
            run_difficulty_pref = _normalize_prompt_difficulty(run_difficulty_current)
            run_extract_prompt_choices_next = _prompt_path_choices(
                cfg_now,
                lang=lang,
                mode_filter="extraction",
                content_type_filter=run_content_type_pref,
                learning_mode_filter=run_learning_mode_pref,
                difficulty_filter=run_difficulty_pref,
                include_auto=True,
            )
            run_explain_prompt_choices_next = _prompt_path_choices(
                cfg_now,
                lang=lang,
                mode_filter="explanation",
                content_type_filter=run_content_type_pref,
                learning_mode_filter=run_learning_mode_pref,
                difficulty_filter=run_difficulty_pref,
                include_auto=True,
            )
            config_extract_prompt_choices_next = _prompt_path_choices(
                cfg_now,
                lang=lang,
                mode_filter="extraction",
                include_auto=True,
            )
            config_explain_prompt_choices_next = _prompt_path_choices(
                cfg_now,
                lang=lang,
                mode_filter="explanation",
                include_auto=True,
            )
            run_selector_next, run_detail_next, run_download_next = (
                _refresh_recent_runs(
                    cfg_now,
                    lang=lang,
                    preferred_run_id=run_id_current,
                )
            )
            selector_choices = run_selector_next.get("choices", [])
            selector_value = run_selector_next.get("value")
            return (
                gr.update(label=_tr(lang, "UI language", "UI language")),
                gr.update(
                    value=_tr(
                        lang,
                        "# ClawLingua Web UI\nLocal deck builder for text learning.",
                        "# ClawLingua Web UI\nLocal deck builder for text learning.",
                    )
                ),
                gr.update(label=_tr(lang, "Run", "Run")),
                gr.update(label=_tr(lang, "Config", "Config")),
                gr.update(label=_tr(lang, "Prompt", "Prompt")),
                gr.update(label=_tr(lang, "Input file", "Input file")),
                gr.update(
                    label=_tr(lang, "Deck title (optional)", "Deck title (optional)")
                ),
                gr.update(label=_tr(lang, "Source language", "Source language")),
                gr.update(label=_tr(lang, "Target language", "Target language")),
                gr.update(label=_tr(lang, "Content profile", "Content profile")),
                gr.update(label=_tr(lang, "Difficulty", "Difficulty")),
                gr.update(
                    label=_tr(
                        lang,
                        "Extraction prompt (run override)",
                        "Extraction prompt (run override)",
                    ),
                    info=_tr(
                        lang,
                        "Equivalent to CLI --extract-prompt.",
                        "Equivalent to CLI --extract-prompt.",
                    ),
                    choices=run_extract_prompt_choices_next,
                    value=_normalize_dropdown_value(
                        run_extract_prompt_current, run_extract_prompt_choices_next
                    ),
                ),
                gr.update(
                    label=_tr(
                        lang,
                        "Explanation prompt (run override)",
                        "Explanation prompt (run override)",
                    ),
                    info=_tr(
                        lang,
                        "Equivalent to CLI --explain-prompt.",
                        "Equivalent to CLI --explain-prompt.",
                    ),
                    choices=run_explain_prompt_choices_next,
                    value=_normalize_dropdown_value(
                        run_explain_prompt_current, run_explain_prompt_choices_next
                    ),
                ),
                gr.update(
                    label=_tr(
                        lang, "Max notes (0 = no limit)", "Max notes (0 = no limit)"
                    ),
                    info=_tr(
                        lang,
                        "Maximum notes after dedupe. Empty/0 means no limit.",
                        "Maximum notes after dedupe. Empty/0 means no limit.",
                    ),
                ),
                gr.update(
                    label=_tr(lang, "Input char limit", "Input char limit"),
                    info=_tr(
                        lang,
                        "Only process the first N chars of input. Empty means no limit.",
                        "Only process the first N chars of input. Empty means no limit.",
                    ),
                ),
                gr.update(label=_tr(lang, "Advanced", "Advanced")),
                gr.update(
                    label=_tr(
                        lang,
                        "Cloze min chars (override env)",
                        "Cloze min chars (override env)",
                    ),
                    info=_tr(
                        lang,
                        "One-run override for CLAWLINGUA_CLOZE_MIN_CHARS.",
                        "One-run override for CLAWLINGUA_CLOZE_MIN_CHARS.",
                    ),
                ),
                gr.update(
                    label=_tr(
                        lang,
                        "Chunk max chars (override env)",
                        "Chunk max chars (override env)",
                    ),
                    info=_tr(
                        lang,
                        "One-run override for CLAWLINGUA_CHUNK_MAX_CHARS.",
                        "One-run override for CLAWLINGUA_CHUNK_MAX_CHARS.",
                    ),
                ),
                gr.update(
                    label=_tr(
                        lang, "Temperature (override env)", "Temperature (override env)"
                    ),
                    info=_tr(
                        lang,
                        "0 is more deterministic; higher values are more random.",
                        "0 is more deterministic; higher values are more random.",
                    ),
                ),
                gr.update(
                    label=_tr(
                        lang, "Save intermediate files", "Save intermediate files"
                    ),
                    info=_tr(
                        lang,
                        "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.",
                        "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.",
                    ),
                ),
                gr.update(
                    label=_tr(lang, "Continue on error", "Continue on error"),
                    info=_tr(
                        lang,
                        "If enabled, continue processing after per-item failures.",
                        "If enabled, continue processing after per-item failures.",
                    ),
                ),
                gr.update(value=_tr(lang, "Run", "Run")),
                gr.update(label=_tr(lang, "Status", "Status")),
                gr.update(label=_tr(lang, "Download .apkg", "Download .apkg")),
                gr.update(value=_tr(lang, "### Recent runs", "### Recent runs")),
                gr.update(value=_tr(lang, "Refresh runs", "Refresh runs")),
                gr.update(
                    label=_tr(lang, "Run ID", "Run ID"),
                    choices=selector_choices,
                    value=selector_value,
                ),
                gr.update(value=run_detail_next),
                gr.update(
                    label=_tr(lang, "Download .apkg", "Download .apkg"),
                    value=run_download_next,
                ),
                gr.update(
                    value=_tr(
                        lang, "### Config (.env editor)", "### Config (.env editor)"
                    )
                ),
                gr.update(label=_tr(lang, "Extraction LLM", "Extraction LLM")),
                gr.update(
                    info=_tr(
                        lang,
                        "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).",
                        "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "API key for extraction LLM, when required.",
                        "API key for extraction LLM, when required.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Model name for extraction LLM.",
                        "Model name for extraction LLM.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Request timeout in seconds.",
                        "Request timeout in seconds.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Default temperature for extraction LLM.",
                        "Default temperature for extraction LLM.",
                    )
                ),
                gr.update(value=_tr(lang, "List models", "List models")),
                gr.update(value=_tr(lang, "Test", "Test")),
                gr.update(
                    label=_tr(lang, "Extraction LLM status", "Extraction LLM status")
                ),
                gr.update(label=_tr(lang, "Explanation LLM", "Explanation LLM")),
                gr.update(
                    info=_tr(
                        lang,
                        "Optional base URL for explanation model.",
                        "Optional base URL for explanation model.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "API key for explanation LLM, when required.",
                        "API key for explanation LLM, when required.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Model name for explanation LLM.",
                        "Model name for explanation LLM.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Default temperature for explanation LLM.",
                        "Default temperature for explanation LLM.",
                    )
                ),
                gr.update(
                    value=_tr(
                        lang, "List models (explanation)", "List models (explanation)"
                    )
                ),
                gr.update(value=_tr(lang, "Test (explanation)", "Test (explanation)")),
                gr.update(
                    label=_tr(lang, "Explanation LLM status", "Explanation LLM status")
                ),
                gr.update(label=_tr(lang, "Chunk & Cloze", "Chunk & Cloze")),
                gr.update(
                    info=_tr(
                        lang,
                        "Default max chars per chunk.",
                        "Default max chars per chunk.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Minimum chars required for cloze text.",
                        "Minimum chars required for cloze text.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Max cards per chunk after dedupe. Empty/0 means unlimited.",
                        "Max cards per chunk after dedupe. Empty/0 means unlimited.",
                    )
                ),
                gr.update(
                    label="CLAWLINGUA_PROMPT_LANG",
                    info=_tr(
                        lang,
                        "Prompt language for multi-lingual prompts (en/zh).",
                        "Prompt language for multi-lingual prompts (en/zh).",
                    ),
                    value=prompt_lang_next,
                ),
                gr.update(
                    label="CLAWLINGUA_EXTRACT_PROMPT",
                    info=_tr(
                        lang,
                        "Default extraction prompt path.",
                        "Default extraction prompt path.",
                    ),
                    choices=config_extract_prompt_choices_next,
                ),
                gr.update(
                    label="CLAWLINGUA_EXPLAIN_PROMPT",
                    info=_tr(
                        lang,
                        "Default explanation prompt path.",
                        "Default explanation prompt path.",
                    ),
                    choices=config_explain_prompt_choices_next,
                ),
                gr.update(label=_tr(lang, "Paths & defaults", "Paths & defaults")),
                gr.update(
                    info=_tr(
                        lang,
                        "Directory for intermediate run data (JSONL, media).",
                        "Directory for intermediate run data (JSONL, media).",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Default directory for exported decks.",
                        "Default directory for exported decks.",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang, "Directory for log files.", "Directory for log files."
                    )
                ),
                gr.update(
                    value=_tr(
                        lang,
                        "Load defaults from ENV_EXAMPLE.md",
                        "Load defaults from ENV_EXAMPLE.md",
                    )
                ),
                gr.update(value=_tr(lang, "Save config", "Save config")),
                gr.update(label=_tr(lang, "TTS voices (Edge)", "TTS voices (Edge)")),
                gr.update(
                    value=_tr(
                        lang,
                        "Voice reference: [Edge TTS Voice Samples](https://tts.travisvn.com/)",
                        "Voice reference: [Edge TTS Voice Samples](https://tts.travisvn.com/)",
                    )
                ),
                gr.update(
                    info=_tr(
                        lang,
                        "Configure 4 voice slots used for random selection.",
                        "Configure 4 voice slots used for random selection.",
                    )
                ),
                gr.update(
                    value=_tr(
                        lang, "### Prompt template editor", "### Prompt template editor"
                    )
                ),
                gr.update(
                    label=_tr(lang, "Prompt file", "Prompt file"),
                    choices=_prompt_choices_from_map(
                        prompt_files_filtered_next, lang=lang
                    ),
                    value=prompt_key_next,
                ),
                gr.update(
                    label=_tr(lang, "Prompt type", "Prompt type"),
                    choices=prompt_mode_choices_next,
                    value=prompt_mode_next,
                ),
                gr.update(
                    label=_tr(lang, "Prompt content type", "Prompt content type"),
                    choices=_PROMPT_CONTENT_TYPE_OPTIONS,
                    value=prompt_content_type_pref,
                ),
                gr.update(
                    label=_tr(lang, "Prompt learning mode", "Prompt learning mode"),
                    choices=_PROMPT_LEARNING_MODE_OPTIONS,
                    value=prompt_learning_mode_pref,
                ),
                gr.update(
                    label=_tr(lang, "Prompt difficulty", "Prompt difficulty"),
                    choices=_PROMPT_DIFFICULTY_OPTIONS,
                    value=prompt_difficulty_pref,
                ),
                gr.update(
                    label=_tr(lang, "New prompt file name", "New prompt file name")
                ),
                gr.update(label=_tr(lang, "Rename to", "Rename to")),
                gr.update(
                    label=_tr(lang, "Prompt template", "Prompt template"),
                    value=prompt_template_next,
                ),
                gr.update(value=_tr(lang, "New", "New")),
                gr.update(value=_tr(lang, "Save", "Save")),
                gr.update(value=_tr(lang, "Rename", "Rename")),
                gr.update(value=_tr(lang, "Delete", "Delete")),
                gr.update(value=False, visible=False),
                gr.update(value=False, visible=False),
                gr.update(
                    label=_tr(lang, "Prompt status", "Prompt status"),
                    value=prompt_status_next,
                ),
                gr.update(label=_tr(lang, "Run analytics", "杩愯缁熻鍒嗘瀽")),
            )

        ui_lang.change(
            _on_ui_lang_change,
            inputs=[
                ui_lang,
                prompt_lang_env,
                prompt_file_selector,
                prompt_mode_selector,
                prompt_content_type_selector,
                prompt_learning_mode_selector,
                prompt_difficulty_selector,
                content_profile,
                learning_mode,
                difficulty,
                run_extract_prompt,
                run_explain_prompt,
                run_selector,
            ],
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
                run_extract_prompt,
                run_explain_prompt,
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
                extract_prompt_env,
                explain_prompt_env,
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
                prompt_mode_selector,
                prompt_content_type_selector,
                prompt_learning_mode_selector,
                prompt_difficulty_selector,
                prompt_new_name,
                prompt_rename_name,
                prompt_editor,
                prompt_new_btn,
                prompt_save_btn,
                prompt_rename_btn,
                prompt_load_default_btn,
                prompt_save_confirm,
                prompt_delete_confirm,
                prompt_status,
                analytics_tab,
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
    host_value = (
        server_host or os.getenv("CLAWLINGUA_WEB_HOST") or "0.0.0.0"
    ).strip() or "0.0.0.0"

    logger.info("starting ClawLingua web UI | host=%s port=%d", host_value, port_value)
    demo = build_interface()
    demo.queue().launch(server_name=host_value, server_port=port_value)
    logger.info("ClawLingua web UI stopped")


if __name__ == "__main__":  # pragma: no cover
    launch()
