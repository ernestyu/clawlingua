"""Load and validate prompt files."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from ..errors import build_error
from ..exit_codes import ExitCode
from ..models.prompt_schema import PromptSpec
from ..utils.jsonx import load_json


def _resolve_lang_map(value: dict[str, str], lang: str, default_lang: str = "zh") -> str:
    """Resolve a language-keyed prompt field.

    - Prefer ``value[lang]`` when present and non-empty
    - Fallback to ``value[default_lang]``
    - Otherwise, fallback to the first non-empty value
    """

    if not value:
        return ""
    if lang in value and str(value[lang]).strip():
        return str(value[lang]).strip()
    if default_lang in value and str(value[default_lang]).strip():
        return str(value[default_lang]).strip()
    for v in value.values():
        if str(v).strip():
            return str(v).strip()
    return ""


def _apply_prompt_lang(spec: PromptSpec, prompt_lang: str) -> PromptSpec:
    """Return a copy of PromptSpec with language-resolved prompt fields.

    This keeps the underlying PromptSpec model flexible (accepting either
    strings or {"en": ..., "zh": ...}) while exposing only concrete
    strings to downstream code.
    """

    lang = (prompt_lang or "zh").strip().lower()
    data = spec.model_dump()
    system_prompt = data.get("system_prompt")
    user_prompt_template = data.get("user_prompt_template")

    if isinstance(system_prompt, dict):
        data["system_prompt"] = _resolve_lang_map(system_prompt, lang)
    if isinstance(user_prompt_template, dict):
        data["user_prompt_template"] = _resolve_lang_map(user_prompt_template, lang)

    return PromptSpec.model_validate(data)


def load_prompt(path: Path, *, prompt_lang: str | None = None) -> PromptSpec:
    if not path.exists():
        raise build_error(
            error_code="PROMPT_NOT_FOUND",
            cause="Prompt 文件不存在。",
            detail=f"path={path}",
            next_steps=["检查路径是否正确"],
            exit_code=ExitCode.SCHEMA_ERROR,
        )
    try:
        data = load_json(path)
        spec = PromptSpec.model_validate(data)
        if prompt_lang:
            spec = _apply_prompt_lang(spec, prompt_lang)
        return spec
    except ValidationError as exc:
        raise build_error(
            error_code="PROMPT_SCHEMA_INVALID",
            cause="Prompt JSON 文件不符合要求。",
            detail=str(exc),
            next_steps=[
                "补齐 prompt 必需字段",
                "运行 clawlearn prompt validate <path> 复查",
            ],
            exit_code=ExitCode.SCHEMA_ERROR,
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise build_error(
            error_code="PROMPT_LOAD_FAILED",
            cause="Prompt 文件读取失败。",
            detail=str(exc),
            next_steps=["检查 JSON 格式是否有效"],
            exit_code=ExitCode.SCHEMA_ERROR,
        ) from exc
