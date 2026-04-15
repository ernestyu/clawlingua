"""Prompt template rendering utilities."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ..errors import build_error
from ..exit_codes import ExitCode

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render_prompt_template(template: str, values: Mapping[str, Any]) -> str:
    """Render prompt template without treating JSON braces as format fields.

    We intentionally replace only ``{identifier}`` placeholders. This keeps
    literal JSON examples like ``{"chunk_id": "..."}`` intact and avoids
    ``str.format`` KeyError crashes on prompt templates that include braces.
    """

    text = str(template or "")
    normalized_values = {str(k): str(v) for k, v in values.items()}

    referenced = set(_PLACEHOLDER_RE.findall(text))
    missing = sorted([name for name in referenced if name not in normalized_values])
    if missing:
        allowed = ", ".join(sorted(normalized_values.keys()))
        raise build_error(
            error_code="PROMPT_TEMPLATE_PLACEHOLDER_MISSING",
            cause="Prompt template references unknown placeholders.",
            detail=f"missing={', '.join(missing)}",
            next_steps=[
                "Check placeholder names in the prompt template.",
                f"Allowed placeholders: {allowed}",
            ],
            exit_code=ExitCode.SCHEMA_ERROR,
        )

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return normalized_values.get(key, match.group(0))

    return _PLACEHOLDER_RE.sub(_replace, text)

