"""Prompt JSON schema model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PromptParserSpec(BaseModel):
    strip_code_fences: bool = True
    expect_json_array: bool = True


class PromptOutputFormatSpec(BaseModel):
    type: Literal["json"]
    schema_name: str


class PromptSpec(BaseModel):
    name: str
    version: str
    description: str
    mode: Literal["cloze", "translate"]
    system_prompt: str
    user_prompt_template: str
    placeholders: list[str] = Field(default_factory=list)
    output_format: PromptOutputFormatSpec
    parser: PromptParserSpec = Field(default_factory=PromptParserSpec)

    @field_validator("placeholders")
    @classmethod
    def _ensure_unique_placeholders(cls, value: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in value:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

