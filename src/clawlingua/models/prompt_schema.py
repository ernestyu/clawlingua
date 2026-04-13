"""Prompt JSON schema model."""

from __future__ import annotations

from typing import Literal, Union

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
    mode: Literal["extraction", "explanation"]

    # Metadata for UI filtering.
    content_type: Literal[
        "prose_article",
        "transcript_dialogue",
        "textbook_examples",
        "all",
    ] = "all"
    learning_mode: Literal["expression_mining", "reading_support", "all"] = "all"
    difficulty_level: Literal["beginner", "intermediate", "advanced", "all"] = "all"

    # Support either raw string prompts or language-keyed prompt maps.
    system_prompt: Union[str, dict[str, str]]
    user_prompt_template: Union[str, dict[str, str]]
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

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "cloze":
                return "extraction"
            if normalized == "translate":
                return "explanation"
            return normalized
        return value

    @field_validator("content_type", mode="before")
    @classmethod
    def _normalize_content_type(cls, value: object) -> object:
        if not isinstance(value, str):
            return "all"
        normalized = value.strip().lower()
        if normalized == "general":
            return "prose_article"
        if normalized in {"prose", "article"}:
            return "prose_article"
        if normalized in {"transcript", "dialogue"}:
            return "transcript_dialogue"
        if normalized in {"textbook", "example"}:
            return "textbook_examples"
        if not normalized:
            return "all"
        return normalized

    @field_validator("learning_mode", mode="before")
    @classmethod
    def _normalize_learning_mode(cls, value: object) -> object:
        if not isinstance(value, str):
            return "all"
        normalized = value.strip().lower()
        if not normalized:
            return "all"
        return normalized

    @field_validator("difficulty_level", mode="before")
    @classmethod
    def _normalize_difficulty_level(cls, value: object) -> object:
        if not isinstance(value, str):
            return "all"
        normalized = value.strip().lower()
        if not normalized:
            return "all"
        return normalized
