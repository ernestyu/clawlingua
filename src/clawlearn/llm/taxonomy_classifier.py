"""Classify phrase taxonomy labels for existing cloze candidates."""

from __future__ import annotations

import json
from typing import Any

from ..errors import build_error
from ..exit_codes import ExitCode
from .client import OpenAICompatibleClient
from .response_parser import parse_json_content


def _normalize_classifier_output_item(
    item: object,
    *,
    normalize_phrase_types_fn: Any,
) -> list[str]:
    if isinstance(item, dict):
        return normalize_phrase_types_fn(item.get("phrase_types"), max_items=2)
    if isinstance(item, list):
        return normalize_phrase_types_fn(item, max_items=2)
    if isinstance(item, str):
        return normalize_phrase_types_fn(item, max_items=2)
    return []


def classify_phrase_types_batch(
    *,
    client: OpenAICompatibleClient,
    items: list[dict[str, Any]],
    temperature: float | None = None,
) -> list[list[str]]:
    if not items:
        return []
    from ..pipeline.taxonomy import PHRASE_TAXONOMY, normalize_phrase_types as normalize_phrase_types_fn

    payload_items = [
        {
            "text_cloze": str(item.get("text_cloze", "")).strip(),
            "text_original": str(item.get("text_original", "")).strip(),
            "target_phrases": [str(x).strip() for x in (item.get("target_phrases") or []) if str(x).strip()],
            "difficulty": str(item.get("difficulty", "")).strip(),
            "learning_mode": str(item.get("learning_mode", "")).strip(),
        }
        for item in items
    ]
    taxonomy_list = ", ".join(PHRASE_TAXONOMY)
    content = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict taxonomy classifier for language-learning candidates. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Classify each input item using only these labels:\n"
                    f"{taxonomy_list}\n\n"
                    "Rules:\n"
                    "1) Return a JSON array with exactly the same length as input.\n"
                    "2) Each output item must be: {\"phrase_types\": [\"label1\", \"label2\"]}.\n"
                    "3) phrase_types must contain 0-2 labels from the given taxonomy only.\n"
                    "4) If uncertain, return an empty array for that item.\n"
                    "5) Do not translate or rewrite the text.\n\n"
                    "input_json:\n"
                    f"{json.dumps(payload_items, ensure_ascii=False)}"
                ),
            },
        ],
        temperature=temperature,
        max_retries=1,
    )
    data = parse_json_content(content, expect_array=True)
    if len(data) != len(items):
        raise build_error(
            error_code="LLM_RESPONSE_SHAPE_INVALID",
            cause="Taxonomy classifier response length mismatch.",
            detail=f"expected={len(items)}, got={len(data)}",
            next_steps=["Strengthen taxonomy classification prompt contract", "Retry with lower temperature"],
            exit_code=ExitCode.LLM_PARSE_ERROR,
        )
    return [
        _normalize_classifier_output_item(item, normalize_phrase_types_fn=normalize_phrase_types_fn)
        for item in data
    ]
