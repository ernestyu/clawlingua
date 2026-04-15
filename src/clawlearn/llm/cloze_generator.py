"""Generate cloze candidates from chunks."""

from __future__ import annotations

import re
from typing import Any

from ..errors import build_error
from ..exit_codes import ExitCode
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


_PHRASE_TOKEN_RE = re.compile(r"[A-Za-z']+")
_FORBIDDEN_PHRASE_LEADS = {"that", "which", "whereas"}


def _normalize_context_text(item: dict[str, Any]) -> str:
    sentence_text = str(
        item.get("sentence")
        or item.get("sentence_text")
        or item.get("original")
        or item.get("text_original")
        or ""
    ).strip()
    if sentence_text:
        return sentence_text
    context_sentences = item.get("context_sentences")
    if isinstance(context_sentences, list):
        parts = [str(x).strip() for x in context_sentences if str(x).strip()]
        if parts:
            return " ".join(parts)
    return ""


def _is_stage1_phrase_valid(*, phrase_text: str, context_text: str) -> bool:
    phrase = str(phrase_text or "").strip()
    context = str(context_text or "").strip()
    if not phrase or not context:
        return False
    if any(ch in phrase for ch in ",;:"):
        return False
    tokens = _PHRASE_TOKEN_RE.findall(phrase)
    if len(tokens) < 2 or len(tokens) > 6:
        return False
    if tokens and tokens[0].lower() in _FORBIDDEN_PHRASE_LEADS:
        return False
    if phrase in context:
        return True
    return phrase.lower() in context.lower()


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
        sentence_text = _normalize_context_text(item)
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
            if not _is_stage1_phrase_valid(phrase_text=phrase_text, context_text=sentence_text):
                continue
            candidate: dict[str, Any] = {
                "chunk_id": chunk_id,
                "sentence_text": sentence_text,
                "phrase_text": phrase_text,
            }
            candidates.append(candidate)
            continue

        for phrase in phrases:
            if isinstance(phrase, str):
                phrase_text = phrase.strip()
            elif isinstance(phrase, dict):
                phrase_text = str(phrase.get("text") or phrase.get("phrase_text") or "").strip()
            else:
                continue
            if not phrase_text:
                continue
            if not _is_stage1_phrase_valid(phrase_text=phrase_text, context_text=sentence_text):
                continue
            candidate = {
                "chunk_id": chunk_id,
                "sentence_text": sentence_text,
                "phrase_text": phrase_text,
            }
            candidates.append(candidate)
    return candidates


def _looks_like_phrase_response_item(item: dict[str, Any]) -> bool:
    return isinstance(item.get("context_sentences"), list) or isinstance(item.get("phrases"), list)


def _extract_cloze_candidates_from_items(
    *,
    data: list[Any],
    forced_chunk_id: str | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text") or item.get("text_cloze") or "").strip()
        original = str(item.get("original") or item.get("text_original") or "").strip()
        target_phrases_raw = item.get("target_phrases")
        target_phrases = (
            [str(x).strip() for x in target_phrases_raw if str(x).strip()]
            if isinstance(target_phrases_raw, list)
            else []
        )

        if _looks_like_phrase_response_item(item) and not (text and original and target_phrases):
            raise build_error(
                error_code="LLM_RESPONSE_SHAPE_INVALID",
                cause="Extraction response schema mismatch.",
                detail="Received phrase-candidate shape while cloze parser expected cloze-card fields.",
                next_steps=[
                    "Use a phrase_candidates schema with the phrase pipeline route",
                    "Or return cloze-card fields: text/original/target_phrases",
                ],
                exit_code=ExitCode.LLM_PARSE_ERROR,
            )

        if not text or not original or not target_phrases:
            continue

        note_hint = item.get("note_hint") or item.get("note") or ""
        if forced_chunk_id is not None:
            chunk_id = forced_chunk_id
        else:
            chunk_id = str(item.get("chunk_id") or "").strip()
        candidate: dict[str, Any] = {
            "chunk_id": str(chunk_id).strip(),
            "text": text,
            "original": original,
            "target_phrases": target_phrases,
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
    # Single-chunk generation must always be attributed to this chunk_id.
    return _extract_cloze_candidates_from_items(data=data, forced_chunk_id=chunk.chunk_id)


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
    return _extract_cloze_candidates_from_items(data=data)


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
        "Each item must contain keys: chunk_id, context_sentences, phrases.\n"
        "context_sentences must be an array of 2-3 verbatim sentences when possible.\n"
        "phrases must be an array where each phrase is either a string or {\"text\":\"...\"}.\n"
        "Each phrase must be a 2-6 word substring of the context and must not contain ',', ';', or ':'.\n"
        "Each phrase must not start with that/which/whereas.\n"
        "Do not output reason, selection_reason, phrase_types, learning_value_score, or expression_transfer.\n"
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
        "Each item must contain keys: chunk_id, context_sentences, phrases.\n"
        "context_sentences must be an array of 2-3 verbatim sentences when possible.\n"
        "phrases must be an array where each phrase is either a string or {\"text\":\"...\"}.\n"
        "Each phrase must be a 2-6 word substring of the context and must not contain ',', ';', or ':'.\n"
        "Each phrase must not start with that/which/whereas.\n"
        "Do not output reason, selection_reason, phrase_types, learning_value_score, or expression_transfer.\n"
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
