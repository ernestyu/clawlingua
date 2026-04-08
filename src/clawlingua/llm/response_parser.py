"""Parse raw LLM response content."""

from __future__ import annotations

import re
from typing import Any

from ..errors import build_error
from ..exit_codes import ExitCode
from ..utils.jsonx import loads

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = _FENCE_RE.sub("", cleaned)
    return cleaned.strip()


def parse_json_content(content: str, *, expect_array: bool) -> Any:
    text = strip_code_fences(content)
    try:
        data = loads(text)
    except Exception as exc:
        raise build_error(
            error_code="LLM_RESPONSE_PARSE_FAILED",
            cause="模型输出无法解析为 JSON。",
            detail=f"raw_prefix={content[:120]!r}",
            next_steps=[
                "检查 prompt 是否要求严格 JSON 输出",
                "尝试降低 temperature，例如 --temperature 0.1",
                "使用 --save-intermediate 检查原始返回内容",
            ],
            exit_code=ExitCode.LLM_PARSE_ERROR,
        ) from exc
    if expect_array and not isinstance(data, list):
        raise build_error(
            error_code="LLM_RESPONSE_SHAPE_INVALID",
            cause="模型输出 JSON 结构不符合要求。",
            detail=f"expect=list, got={type(data).__name__}",
            next_steps=["检查 prompt output_format 与 parser 配置"],
            exit_code=ExitCode.LLM_PARSE_ERROR,
        )
    return data

