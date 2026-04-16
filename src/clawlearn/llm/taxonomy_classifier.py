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
    max_items: int = 2,
) -> list[str]:
    if isinstance(item, dict):
        return normalize_phrase_types_fn(item.get("phrase_types"), max_items=max_items)
    if isinstance(item, list):
        return normalize_phrase_types_fn(item, max_items=max_items)
    if isinstance(item, str):
        return normalize_phrase_types_fn(item, max_items=max_items)
    return []


def classify_phrase_types_batch(
    *,
    client: OpenAICompatibleClient,
    items: list[dict[str, Any]],
    temperature: float | None = None,
    allow_partial: bool = False,
    source_lang: str | None = None,
) -> list[list[str]]:
    if not items:
        return []
    from ..pipeline.taxonomy import get_allowed_taxonomy, normalize_phrase_types as normalize_phrase_types_fn

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
    allowed_taxonomy, _ = get_allowed_taxonomy(source_lang)
    taxonomy_list = ", ".join(allowed_taxonomy)
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
    if len(data) != len(items) and not allow_partial:
        raise build_error(
            error_code="LLM_RESPONSE_SHAPE_INVALID",
            cause="Taxonomy classifier response length mismatch.",
            detail=f"expected={len(items)}, got={len(data)}",
            next_steps=["Strengthen taxonomy classification prompt contract", "Retry with lower temperature"],
            exit_code=ExitCode.LLM_PARSE_ERROR,
        )
    if allow_partial and len(data) > len(items):
        data = data[: len(items)]
    return [
        _normalize_classifier_output_item(item, normalize_phrase_types_fn=normalize_phrase_types_fn, max_items=2)
        for item in data
    ]


def classify_lingua_prerank_taxonomy_batch(
    *,
    client: OpenAICompatibleClient,
    items: list[dict[str, Any]],
    temperature: float | None = None,
    allow_partial: bool = False,
    source_lang: str | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible candidate-level pre-rank taxonomy classifier.

    New pipeline should prefer `classify_lingua_prerank_phrases_batch`.
    """
    if not items:
        return []
    from ..pipeline.taxonomy import get_allowed_taxonomy, normalize_phrase_types as normalize_phrase_types_fn

    payload_items = [
        {
            "id": str(item.get("id") or "").strip(),
            "text_cloze": str(item.get("text_cloze", "")).strip(),
            "text_original": str(item.get("text_original", "")).strip(),
            "target_phrases": [str(x).strip() for x in (item.get("target_phrases") or []) if str(x).strip()],
            "difficulty": str(item.get("difficulty", "")).strip(),
            "learning_mode": str(item.get("learning_mode", "")).strip(),
        }
        for item in items
    ]
    allowed_taxonomy, _ = get_allowed_taxonomy(source_lang)
    taxonomy_list = ", ".join(allowed_taxonomy)
    content = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict taxonomy classifier for language-learning pre-rank candidates. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Classify each input item using only these labels:\n"
                    f"{taxonomy_list}\n\n"
                    "Rules:\n"
                    "1) Return a JSON array. Each item must be: {\"id\":\"...\",\"phrase_types\":[\"label1\"]}.\n"
                    "2) Use only input ids; do not invent ids.\n"
                    "3) phrase_types must contain 0-1 labels from the given taxonomy only.\n"
                    "4) Do not output reason, selection_reason, learning_value_score, or expression_transfer.\n"
                    "5) If uncertain, return phrase_types as empty array.\n"
                    "6) Do not translate or rewrite the text.\n\n"
                    "input_json:\n"
                    f"{json.dumps(payload_items, ensure_ascii=False)}"
                ),
            },
        ],
        temperature=temperature,
        max_retries=1,
    )
    data = parse_json_content(content, expect_array=True)
    if len(data) != len(items) and not allow_partial:
        raise build_error(
            error_code="LLM_RESPONSE_SHAPE_INVALID",
            cause="Lingua pre-rank taxonomy response length mismatch.",
            detail=f"expected={len(items)}, got={len(data)}",
            next_steps=["Strengthen pre-rank taxonomy output contract", "Retry with lower temperature"],
            exit_code=ExitCode.LLM_PARSE_ERROR,
        )
    if allow_partial and len(data) > len(items):
        data = data[: len(items)]

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        payload = item if isinstance(item, dict) else {}
        item_id = str(payload.get("id") or "").strip()
        if not item_id and idx < len(payload_items):
            item_id = str(payload_items[idx].get("id") or "").strip()
        phrase_types = _normalize_classifier_output_item(
            item,
            normalize_phrase_types_fn=normalize_phrase_types_fn,
            max_items=1,
        )
        normalized.append(
            {
                "id": item_id,
                "phrase_types": phrase_types,
            }
        )
    return normalized


def classify_lingua_prerank_phrases_batch(
    *,
    client: OpenAICompatibleClient,
    items: list[dict[str, Any]],
    temperature: float | None = None,
    allow_partial: bool = False,
    source_lang: str | None = None,
) -> list[dict[str, Any]]:
    if not items:
        return []
    from ..pipeline.taxonomy import get_prerank_taxonomy, normalize_prerank_phrase_label

    payload_items = [
        {
            "id": str(item.get("id") or "").strip(),
            "phrase_text": str(item.get("phrase_text", "")).strip(),
            "text_original": str(item.get("text_original", "")).strip(),
            "text_cloze": str(item.get("text_cloze", "")).strip(),
            "local_context": str(item.get("local_context", "")).strip(),
            "difficulty": str(item.get("difficulty", "")).strip(),
            "learning_mode": str(item.get("learning_mode", "")).strip(),
        }
        for item in items
    ]
    allowed_taxonomy, _ = get_prerank_taxonomy(source_lang)
    taxonomy_list = ", ".join(allowed_taxonomy)
    content = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict phrase-level taxonomy classifier for language-learning pre-rank filtering. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Classify each phrase using only these labels:\n"
                    f"{taxonomy_list}\n\n"
                    "Rules:\n"
                    "1) Return a JSON array.\n"
                    "2) Each output item must be: "
                    "{\"id\":\"...\",\"label\":\"label_or_none\",\"keep\":true_or_false,\"confidence\":0_to_1}.\n"
                    "3) id must come from input ids only; do not invent ids.\n"
                    "4) label can be one of allowed labels or \"none\".\n"
                    "5) If label is \"none\", keep must be false.\n"
                    "6) If uncertain, return label=\"none\" and keep=false.\n"
                    "7) Do not output reason/selection_reason or any extra fields.\n"
                    "8) Do not translate or rewrite text.\n\n"
                    "input_json:\n"
                    f"{json.dumps(payload_items, ensure_ascii=False)}"
                ),
            },
        ],
        temperature=temperature,
        max_retries=1,
    )
    data = parse_json_content(content, expect_array=True)
    if len(data) != len(items) and not allow_partial:
        raise build_error(
            error_code="LLM_RESPONSE_SHAPE_INVALID",
            cause="Lingua pre-rank phrase response length mismatch.",
            detail=f"expected={len(items)}, got={len(data)}",
            next_steps=["Strengthen pre-rank phrase output contract", "Retry with lower temperature"],
            exit_code=ExitCode.LLM_PARSE_ERROR,
        )
    if allow_partial and len(data) > len(items):
        data = data[: len(items)]

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        payload = item if isinstance(item, dict) else {}
        item_id = str(payload.get("id") or "").strip()
        if not item_id and idx < len(payload_items):
            item_id = str(payload_items[idx].get("id") or "").strip()
        label = normalize_prerank_phrase_label(payload.get("label"))
        keep_raw = payload.get("keep")
        keep = bool(keep_raw) if isinstance(keep_raw, bool) else bool(label)
        if not label:
            keep = False
        confidence_value = payload.get("confidence")
        confidence = 0.0
        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0
        normalized.append(
            {
                "id": item_id,
                "label": label or "none",
                "keep": keep,
                "confidence": round(confidence, 4),
            }
        )
    return normalized


def classify_phrase_annotations_batch(
    *,
    client: OpenAICompatibleClient,
    items: list[dict[str, Any]],
    temperature: float | None = None,
    allow_partial: bool = False,
) -> list[dict[str, Any]]:
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
                    "You are a strict taxonomy and reason annotator for language-learning candidates. "
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
                    "2) Each output item must be: {\"phrase_types\": [\"label1\"], \"reason\": \"...\"}.\n"
                    "3) phrase_types must contain 0-1 labels from the given taxonomy only.\n"
                    "4) reason must be short (<=120 chars), plain text, and in the same language style as input.\n"
                    "5) If uncertain, return {\"phrase_types\": [], \"reason\": \"\"} for that item.\n"
                    "6) Do not translate or rewrite the text.\n\n"
                    "input_json:\n"
                    f"{json.dumps(payload_items, ensure_ascii=False)}"
                ),
            },
        ],
        temperature=temperature,
        max_retries=1,
    )
    data = parse_json_content(content, expect_array=True)
    if len(data) != len(items) and not allow_partial:
        raise build_error(
            error_code="LLM_RESPONSE_SHAPE_INVALID",
            cause="Taxonomy classifier response length mismatch.",
            detail=f"expected={len(items)}, got={len(data)}",
            next_steps=["Strengthen taxonomy classification prompt contract", "Retry with lower temperature"],
            exit_code=ExitCode.LLM_PARSE_ERROR,
        )
    if allow_partial and len(data) > len(items):
        data = data[: len(items)]
    normalized: list[dict[str, Any]] = []
    for item in data:
        phrase_types = _normalize_classifier_output_item(
            item,
            normalize_phrase_types_fn=normalize_phrase_types_fn,
            max_items=1,
        )
        reason = ""
        if isinstance(item, dict):
            reason = str(item.get("reason") or item.get("selection_reason") or "").strip()
        if len(reason) > 120:
            reason = reason[:120].strip()
        normalized.append(
            {
                "phrase_types": phrase_types,
                "reason": reason if phrase_types else "",
            }
        )
    return normalized
