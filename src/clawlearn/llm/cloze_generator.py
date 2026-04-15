"""Generate cloze candidates from chunks."""

from __future__ import annotations

from typing import Any

from ..models.chunk import ChunkRecord
from ..models.document import DocumentRecord
from ..models.prompt_schema import PromptSpec
from .client import OpenAICompatibleClient
from .response_parser import parse_json_content
from .template_renderer import render_prompt_template


def _effective_max_per_chunk(client: OpenAICompatibleClient) -> int:
    max_per_chunk = client.config.cloze_max_per_chunk
    return max_per_chunk if max_per_chunk and max_per_chunk > 0 else 4


def _build_extraction_placeholders(
    *,
    client: OpenAICompatibleClient,
    document: DocumentRecord,
    chunk_text: str,
) -> dict[str, str]:
    return {
        "source_lang": document.source_lang,
        "target_lang": document.target_lang,
        "document_title": document.title or "",
        "source_url": document.source_url or "",
        "chunk_text": chunk_text,
        "learning_mode": getattr(client.config, "learning_mode", "lingua_expression"),
        "difficulty": client.config.cloze_difficulty,
        "cloze_max_sentences": str(client.config.cloze_max_sentences),
        "cloze_min_chars": str(client.config.cloze_min_chars),
        "cloze_max_per_chunk": str(_effective_max_per_chunk(client)),
    }


def _normalize_phrase_types(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _extract_phrase_candidates_from_items(
    *,
    data: list[Any],
    forced_chunk_id: str | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or forced_chunk_id or "").strip()
        sentence_text = str(
            item.get("sentence")
            or item.get("sentence_text")
            or item.get("original")
            or item.get("text_original")
            or ""
        ).strip()
        if not chunk_id or not sentence_text:
            continue
        phrases = item.get("phrases")
        # Accept a degraded flat shape:
        # {"sentence": "...", "text": "...", ...}
        if not isinstance(phrases, list):
            phrase_text = str(
                item.get("phrase_text")
                or item.get("text")
                or item.get("target_phrase")
                or ""
            ).strip()
            if not phrase_text:
                continue
            candidate: dict[str, Any] = {
                "chunk_id": chunk_id,
                "sentence_text": sentence_text,
                "phrase_text": phrase_text,
                "reason": str(item.get("reason") or item.get("selection_reason") or "").strip(),
                "phrase_types": _normalize_phrase_types(item.get("phrase_types")),
            }
            if "learning_value_score" in item:
                candidate["learning_value_score"] = item.get("learning_value_score")
            if "expression_transfer" in item:
                candidate["expression_transfer"] = str(item.get("expression_transfer") or "").strip()
            candidates.append(candidate)
            continue

        for phrase in phrases:
            if isinstance(phrase, str):
                phrase_text = phrase.strip()
                reason = ""
                phrase_types: list[str] = []
                learning_value_score: object | None = None
                expression_transfer = ""
            elif isinstance(phrase, dict):
                phrase_text = str(phrase.get("text") or phrase.get("phrase_text") or "").strip()
                reason = str(phrase.get("reason") or phrase.get("selection_reason") or "").strip()
                phrase_types = _normalize_phrase_types(phrase.get("phrase_types"))
                learning_value_score = phrase.get("learning_value_score")
                expression_transfer = str(phrase.get("expression_transfer") or "").strip()
            else:
                continue
            if not phrase_text:
                continue
            candidate = {
                "chunk_id": chunk_id,
                "sentence_text": sentence_text,
                "phrase_text": phrase_text,
                "reason": reason,
                "phrase_types": phrase_types,
            }
            if learning_value_score is not None:
                candidate["learning_value_score"] = learning_value_score
            if expression_transfer:
                candidate["expression_transfer"] = expression_transfer
            candidates.append(candidate)
    return candidates


def generate_cloze_candidates_for_chunk(
    *,
    client: OpenAICompatibleClient,
    prompt: PromptSpec,
    document: DocumentRecord,
    chunk: ChunkRecord,
    temperature: float | None = None,
) -> list[dict]:
    placeholders = _build_extraction_placeholders(
        client=client,
        document=document,
        chunk_text=chunk.source_text,
    )
    placeholders["chunk_id"] = chunk.chunk_id
    user_prompt = render_prompt_template(prompt.user_prompt_template, placeholders)
    content = client.chat(
        [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    data = parse_json_content(content, expect_array=prompt.parser.expect_json_array)
    candidates: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("text_cloze") or ""
        original = item.get("original") or item.get("text_original") or ""
        target_phrases = item.get("target_phrases") or []
        note_hint = item.get("note_hint") or item.get("note") or ""
        # Single-chunk generation must always be attributed to that chunk.
        # Model-supplied chunk_id values are treated as untrusted.
        chunk_id = chunk.chunk_id
        candidate = {
            "chunk_id": str(chunk_id).strip(),
            "text": str(text).strip(),
            "original": str(original).strip(),
            "target_phrases": [str(x).strip() for x in target_phrases if str(x).strip()],
            "note_hint": str(note_hint).strip(),
        }
        if "selection_reason" in item:
            candidate["selection_reason"] = str(item.get("selection_reason", "")).strip()
        if "learning_value_score" in item:
            candidate["learning_value_score"] = item.get("learning_value_score")
        if "phrase_types" in item:
            raw_types = item.get("phrase_types")
            if isinstance(raw_types, list):
                candidate["phrase_types"] = [str(x).strip() for x in raw_types if str(x).strip()]
            elif str(raw_types).strip():
                candidate["phrase_types"] = [str(raw_types).strip()]
        if "expression_transfer" in item:
            candidate["expression_transfer"] = str(item.get("expression_transfer", "")).strip()
        candidates.append(candidate)
    return candidates


def generate_cloze_candidates_for_batch(
    *,
    client: OpenAICompatibleClient,
    prompt: PromptSpec,
    document: DocumentRecord,
    chunks: list[ChunkRecord],
    temperature: float | None = None,
) -> list[dict]:
    # 构造一个包含多个 chunk 的 prompt
    chunk_blocks = []
    for chunk in chunks:
        chunk_blocks.append(f"chunk_id={chunk.chunk_id}\nchunk_text=\n{chunk.source_text}")
    merged_chunk_text = "\n\n".join(chunk_blocks)

    placeholders = _build_extraction_placeholders(
        client=client,
        document=document,
        chunk_text=merged_chunk_text,
    )
    user_prompt = render_prompt_template(prompt.user_prompt_template, placeholders)
    content = client.chat(
        [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    data = parse_json_content(content, expect_array=prompt.parser.expect_json_array)
    candidates: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("text_cloze") or ""
        original = item.get("original") or item.get("text_original") or ""
        target_phrases = item.get("target_phrases") or []
        note_hint = item.get("note_hint") or item.get("note") or ""
        chunk_id = item.get("chunk_id") or ""
        candidate = {
            "chunk_id": str(chunk_id).strip(),
            "text": str(text).strip(),
            "original": str(original).strip(),
            "target_phrases": [str(x).strip() for x in target_phrases if str(x).strip()],
            "note_hint": str(note_hint).strip(),
        }
        if "selection_reason" in item:
            candidate["selection_reason"] = str(item.get("selection_reason", "")).strip()
        if "learning_value_score" in item:
            candidate["learning_value_score"] = item.get("learning_value_score")
        if "phrase_types" in item:
            raw_types = item.get("phrase_types")
            if isinstance(raw_types, list):
                candidate["phrase_types"] = [str(x).strip() for x in raw_types if str(x).strip()]
            elif str(raw_types).strip():
                candidate["phrase_types"] = [str(raw_types).strip()]
        if "expression_transfer" in item:
            candidate["expression_transfer"] = str(item.get("expression_transfer", "")).strip()
        candidates.append(candidate)
    return candidates


def generate_phrase_candidates_for_chunk(
    *,
    client: OpenAICompatibleClient,
    prompt: PromptSpec,
    document: DocumentRecord,
    chunk: ChunkRecord,
    temperature: float | None = None,
) -> list[dict[str, Any]]:
    placeholders = _build_extraction_placeholders(
        client=client,
        document=document,
        chunk_text=chunk.source_text,
    )
    placeholders["chunk_id"] = chunk.chunk_id
    user_prompt = render_prompt_template(prompt.user_prompt_template, placeholders)
    phrase_contract = (
        "Return only a JSON array.\n"
        "Each item must contain keys: chunk_id, sentence, phrases.\n"
        "phrases must be an array of objects with key text, optional reason and phrase_types.\n"
        "Each phrase text must be an exact substring in sentence.\n"
        "Do not output cloze markers, numbering, or hints."
    )
    content = client.chat(
        [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": f"{phrase_contract}\n\n{user_prompt}"},
        ],
        temperature=temperature,
    )
    data = parse_json_content(content, expect_array=prompt.parser.expect_json_array)
    return _extract_phrase_candidates_from_items(data=data, forced_chunk_id=chunk.chunk_id)


def generate_phrase_candidates_for_batch(
    *,
    client: OpenAICompatibleClient,
    prompt: PromptSpec,
    document: DocumentRecord,
    chunks: list[ChunkRecord],
    temperature: float | None = None,
) -> list[dict[str, Any]]:
    chunk_blocks = []
    for chunk in chunks:
        chunk_blocks.append(f"chunk_id={chunk.chunk_id}\nchunk_text=\n{chunk.source_text}")
    merged_chunk_text = "\n\n".join(chunk_blocks)

    placeholders = _build_extraction_placeholders(
        client=client,
        document=document,
        chunk_text=merged_chunk_text,
    )
    user_prompt = render_prompt_template(prompt.user_prompt_template, placeholders)
    phrase_contract = (
        "Return only a JSON array.\n"
        "Each item must contain keys: chunk_id, sentence, phrases.\n"
        "phrases must be an array of objects with key text, optional reason and phrase_types.\n"
        "Each phrase text must be an exact substring in sentence.\n"
        "Do not output cloze markers, numbering, or hints."
    )
    content = client.chat(
        [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": f"{phrase_contract}\n\n{user_prompt}"},
        ],
        temperature=temperature,
    )
    data = parse_json_content(content, expect_array=prompt.parser.expect_json_array)
    return _extract_phrase_candidates_from_items(data=data)
