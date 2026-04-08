"""Load and validate prompt files."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from ..errors import build_error
from ..exit_codes import ExitCode
from ..models.prompt_schema import PromptSpec
from ..utils.jsonx import load_json


def load_prompt(path: Path) -> PromptSpec:
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
        return PromptSpec.model_validate(data)
    except ValidationError as exc:
        raise build_error(
            error_code="PROMPT_SCHEMA_INVALID",
            cause="Prompt JSON 文件不符合要求。",
            detail=str(exc),
            next_steps=[
                "补齐 prompt 必需字段",
                "运行 clawlingua prompt validate <path> 复查",
            ],
            exit_code=ExitCode.SCHEMA_ERROR,
        ) from exc
    except Exception as exc:
        raise build_error(
            error_code="PROMPT_LOAD_FAILED",
            cause="Prompt 文件读取失败。",
            detail=str(exc),
            next_steps=["检查 JSON 格式是否有效"],
            exit_code=ExitCode.SCHEMA_ERROR,
        ) from exc

