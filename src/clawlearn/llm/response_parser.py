"""Parse raw LLM response content."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..errors import build_error
from ..exit_codes import ExitCode
from ..utils.jsonx import loads

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = _FENCE_RE.sub("", cleaned)
    return cleaned.strip()


@dataclass
class ExtractionParseReport:
    control_char_cleaned: bool = False
    json_fragment_extracted: bool = False
    partial_salvaged: bool = False
    salvaged_count: int = 0
    salvage_reason: str = ""


def _raise_parse_error(*, content: str) -> None:
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
    )


def _raise_shape_error(*, data: Any) -> None:
    raise build_error(
        error_code="LLM_RESPONSE_SHAPE_INVALID",
        cause="模型输出 JSON 结构不符合要求。",
        detail=f"expect=list, got={type(data).__name__}",
        next_steps=["检查 prompt output_format 与 parser 配置"],
        exit_code=ExitCode.LLM_PARSE_ERROR,
    )


def _sanitize_json_control_chars(text: str) -> tuple[str, bool]:
    if not text:
        return "", False
    out: list[str] = []
    in_string = False
    escaped = False
    changed = False
    for ch in text:
        code = ord(ch)
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if code < 0x20:
                out.append(" ")
                changed = True
                continue
            out.append(ch)
            continue

        if ch == '"':
            out.append(ch)
            in_string = True
            continue
        if code < 0x20 and ch not in {"\n", "\r", "\t"}:
            changed = True
            continue
        out.append(ch)
    return "".join(out), changed


def _extract_json_fragment(text: str) -> tuple[str, bool]:
    if not text:
        return text, False
    first_list = text.find("[")
    first_obj = text.find("{")
    start_candidates = [idx for idx in (first_list, first_obj) if idx >= 0]
    if not start_candidates:
        return text, False
    start = min(start_candidates)
    root = text[start]
    closer = "]" if root == "[" else "}"
    stack: list[str] = [closer]
    in_string = False
    escaped = False
    for idx in range(start + 1, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "[{":
            stack.append("]" if ch == "[" else "}")
            continue
        if ch in "]}":
            if stack and ch == stack[-1]:
                stack.pop()
                if not stack:
                    fragment = text[start : idx + 1]
                    extracted = start > 0 or (idx + 1) < len(text)
                    return fragment, extracted
            else:
                break
    # No full closure found; keep from root start so partial salvage can continue.
    return text[start:], True


def _salvage_partial_json_array(text: str) -> tuple[list[Any], str]:
    if not text:
        return [], "empty content"
    start = text.find("[")
    if start < 0:
        return [], "missing opening bracket"
    decoder = json.JSONDecoder()
    items: list[Any] = []
    idx = start + 1
    reason = "truncated last item"
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            reason = "missing closing bracket"
            break
        if text[idx] == "]":
            return items, "complete"
        if text[idx] == ",":
            idx += 1
            continue
        try:
            item, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            if "]" not in text[idx:]:
                reason = "missing closing bracket"
            break
        items.append(item)
        idx = end
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx < len(text) and text[idx] == ",":
            idx += 1
            continue
        if idx < len(text) and text[idx] == "]":
            return items, "complete"
        if idx >= len(text):
            reason = "missing closing bracket"
        break
    return items, reason


def parse_extraction_json_content(content: str, *, expect_array: bool) -> tuple[Any, ExtractionParseReport]:
    report = ExtractionParseReport()
    stripped = strip_code_fences(content)
    sanitized, sanitized_changed = _sanitize_json_control_chars(stripped)
    if sanitized_changed:
        report.control_char_cleaned = True
    fragment, extracted = _extract_json_fragment(sanitized)
    if extracted:
        report.json_fragment_extracted = True
    try:
        data = loads(fragment)
    except Exception:
        if expect_array:
            salvaged, reason = _salvage_partial_json_array(fragment)
            if salvaged:
                report.partial_salvaged = True
                report.salvaged_count = len(salvaged)
                report.salvage_reason = reason
                return salvaged, report
        _raise_parse_error(content=content)
    if expect_array and not isinstance(data, list):
        _raise_shape_error(data=data)
    return data, report


def parse_json_content(content: str, *, expect_array: bool) -> Any:
    text = strip_code_fences(content)
    try:
        data = loads(text)
    except Exception:
        _raise_parse_error(content=content)
    if expect_array and not isinstance(data, list):
        _raise_shape_error(data=data)
    return data
