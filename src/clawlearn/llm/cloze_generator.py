"""Generate cloze candidates from chunks."""

from __future__ import annotations

import re
import time
from typing import Any

from ..errors import ClawLearnError, build_error
from ..exit_codes import ExitCode
from ..models.chunk import ChunkRecord
from ..models.document import DocumentRecord
from ..models.prompt_schema import PromptSpec
from .client import OpenAICompatibleClient
from .response_parser import parse_extraction_json_content
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
_EXTRACTION_PARSE_MAX_RETRIES = 3
_EXTRACTION_PARSE_BACKOFF_SECONDS = (0.5, 1.0, 2.0)


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


def _append_extraction_event(
    *,
    events: list[dict[str, Any]] | None,
    payload: dict[str, Any],
) -> None:
    if events is None:
        return
    events.append(payload)


def _extraction_retry_delay_seconds(attempt: int) -> float:
    if attempt <= 0:
        return 0.0
    index = min(attempt - 1, len(_EXTRACTION_PARSE_BACKOFF_SECONDS) - 1)
    return float(_EXTRACTION_PARSE_BACKOFF_SECONDS[index])


def _is_retryable_extraction_error(exc: ClawLearnError) -> bool:
    return exc.error_code in {
        "LLM_RESPONSE_PARSE_FAILED",
        "LLM_RESPONSE_SHAPE_INVALID",
    }


def _build_chunk_blocks(chunks: list[ChunkRecord]) -> str:
    chunk_blocks = []
    for chunk in chunks:
        chunk_blocks.append(f"chunk_id={chunk.chunk_id}\nchunk_text=\n{chunk.source_text}")
    return "\n\n".join(chunk_blocks)


def _build_messages(
    *,
    prompt: PromptSpec,
    user_prompt: str,
    phrase_contract: str = "",
) -> list[dict[str, str]]:
    if phrase_contract:
        content = f"{phrase_contract}\n\n{user_prompt}"
    else:
        content = user_prompt
    return [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": content},
    ]


def _collect_done_chunk_ids(data: Any, expected_chunk_ids: set[str]) -> set[str]:
    if not isinstance(data, list):
        return set()
    done: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        if chunk_id and chunk_id in expected_chunk_ids:
            done.add(chunk_id)
    return done


def _call_and_extract_with_retries(
    *,
    client: OpenAICompatibleClient,
    prompt: PromptSpec,
    messages: list[dict[str, str]],
    temperature: float | None,
    chunk_ids: list[str],
    extract_fn: Any,
    events: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], set[str], bool]:
    expected_chunk_ids = {str(cid).strip() for cid in chunk_ids if str(cid).strip()}
    for attempt in range(1, _EXTRACTION_PARSE_MAX_RETRIES + 1):
        content = client.chat(messages, temperature=temperature)
        try:
            data, report = parse_extraction_json_content(
                content,
                expect_array=prompt.parser.expect_json_array,
            )
            if report.control_char_cleaned or report.json_fragment_extracted:
                _append_extraction_event(
                    events=events,
                    payload={
                        "stage": "extraction_parse_repair",
                        "attempt": attempt,
                        "chunk_ids": chunk_ids,
                        "control_char_cleaned": report.control_char_cleaned,
                        "json_fragment_extracted": report.json_fragment_extracted,
                    },
                )
            if report.partial_salvaged:
                salvaged_chunk_ids = sorted(_collect_done_chunk_ids(data, expected_chunk_ids))
                _append_extraction_event(
                    events=events,
                    payload={
                        "stage": "extraction_parse_partial_salvage",
                        "attempt": attempt,
                        "salvaged_count": int(report.salvaged_count),
                        "expected_chunks": len(expected_chunk_ids),
                        "salvaged_chunk_ids": salvaged_chunk_ids,
                        "reason": report.salvage_reason or "truncated last item",
                    },
                )
            candidates = extract_fn(data)
            done_chunk_ids = _collect_done_chunk_ids(data, expected_chunk_ids)
            return candidates, done_chunk_ids, False
        except ClawLearnError as exc:
            if not _is_retryable_extraction_error(exc):
                raise
            if attempt < _EXTRACTION_PARSE_MAX_RETRIES:
                _append_extraction_event(
                    events=events,
                    payload={
                        "stage": "extraction_parse_retry",
                        "attempt": attempt,
                        "chunk_ids": chunk_ids,
                        "error_code": exc.error_code,
                        "error": exc.to_lines(),
                    },
                )
                delay = _extraction_retry_delay_seconds(attempt)
                if delay > 0:
                    time.sleep(delay)
                continue
            _append_extraction_event(
                events=events,
                payload={
                    "stage": "extraction_parse_exhausted",
                    "attempt": attempt,
                    "chunk_ids": chunk_ids,
                    "error_code": exc.error_code,
                    "error": exc.to_lines(),
                    "raw_prefix": content[:120],
                },
            )
            return [], set(), True
    return [], set(), True


def _phrase_contract_text() -> str:
    return (
        "Return only a JSON array.\n"
        "Each item must contain keys: chunk_id, context_sentences, phrases.\n"
        "context_sentences must be an array of 2-3 verbatim sentences when possible.\n"
        "phrases must be an array where each phrase is either a string or {\"text\":\"...\"}.\n"
        "Each phrase must be a 2-6 word substring of the context and must not contain ',', ';', or ':'.\n"
        "Each phrase must not start with that/which/whereas.\n"
        "Do not output reason, selection_reason, phrase_types, learning_value_score, or expression_transfer.\n"
        "Do not output cloze markers, numbering, or hints."
    )


def generate_cloze_candidates_for_chunk(
    *,
    client: OpenAICompatibleClient,
    prompt: PromptSpec,
    document: DocumentRecord,
    chunk: ChunkRecord,
    temperature: float | None = None,
    events: list[dict[str, Any]] | None = None,
) -> list[dict]:
    placeholders = _build_extraction_placeholders(
        client=client,
        document=document,
        chunk_text=chunk.source_text,
    )
    placeholders["chunk_id"] = chunk.chunk_id
    user_prompt = render_prompt_template(prompt.user_prompt_template, placeholders)
    messages = _build_messages(prompt=prompt, user_prompt=user_prompt)
    candidates, _done_chunk_ids, exhausted = _call_and_extract_with_retries(
        client=client,
        prompt=prompt,
        messages=messages,
        temperature=temperature,
        chunk_ids=[chunk.chunk_id],
        extract_fn=lambda data: _extract_cloze_candidates_from_items(
            data=data,
            forced_chunk_id=chunk.chunk_id,
        ),
        events=events,
    )
    if exhausted:
        _append_extraction_event(
            events=events,
            payload={
                "stage": "extraction_chunk_skipped_after_parse_retries",
                "chunk_id": chunk.chunk_id,
                "attempts": _EXTRACTION_PARSE_MAX_RETRIES,
            },
        )
        return []
    return candidates


def generate_cloze_candidates_for_batch(
    *,
    client: OpenAICompatibleClient,
    prompt: PromptSpec,
    document: DocumentRecord,
    chunks: list[ChunkRecord],
    temperature: float | None = None,
    events: list[dict[str, Any]] | None = None,
) -> list[dict]:
    if not chunks:
        return []
    remaining = list(chunks)
    aggregated: list[dict[str, Any]] = []

    for batch_attempt in range(1, _EXTRACTION_PARSE_MAX_RETRIES + 1):
        merged_chunk_text = _build_chunk_blocks(remaining)
        placeholders = _build_extraction_placeholders(
            client=client,
            document=document,
            chunk_text=merged_chunk_text,
        )
        user_prompt = render_prompt_template(prompt.user_prompt_template, placeholders)
        messages = _build_messages(prompt=prompt, user_prompt=user_prompt)
        chunk_ids = [chunk.chunk_id for chunk in remaining]
        candidates, done_chunk_ids, exhausted = _call_and_extract_with_retries(
            client=client,
            prompt=prompt,
            messages=messages,
            temperature=temperature,
            chunk_ids=chunk_ids,
            extract_fn=lambda data: _extract_cloze_candidates_from_items(data=data),
            events=events,
        )
        aggregated.extend(candidates)
        if exhausted:
            break
        missing_chunks = [chunk for chunk in remaining if chunk.chunk_id not in done_chunk_ids]
        if not missing_chunks:
            remaining = []
            break
        if batch_attempt < _EXTRACTION_PARSE_MAX_RETRIES:
            _append_extraction_event(
                events=events,
                payload={
                    "stage": "extraction_batch_partial_retry",
                    "attempt": batch_attempt,
                    "expected_chunks": len(chunk_ids),
                    "received_chunks": len(done_chunk_ids),
                    "missing_chunk_ids": [chunk.chunk_id for chunk in missing_chunks],
                },
            )
            remaining = missing_chunks
            continue
        remaining = missing_chunks
        break

    if remaining:
        _append_extraction_event(
            events=events,
            payload={
                "stage": "extraction_batch_missing_exhausted",
                "attempts": _EXTRACTION_PARSE_MAX_RETRIES,
                "missing_chunk_ids": [chunk.chunk_id for chunk in remaining],
            },
        )
        for chunk in remaining:
            _append_extraction_event(
                events=events,
                payload={
                    "stage": "extraction_chunk_skipped_after_parse_retries",
                    "chunk_id": chunk.chunk_id,
                    "attempts": _EXTRACTION_PARSE_MAX_RETRIES,
                },
            )
    return aggregated


def generate_phrase_candidates_for_chunk(
    *,
    client: OpenAICompatibleClient,
    prompt: PromptSpec,
    document: DocumentRecord,
    chunk: ChunkRecord,
    temperature: float | None = None,
    events: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    placeholders = _build_extraction_placeholders(
        client=client,
        document=document,
        chunk_text=chunk.source_text,
    )
    placeholders["chunk_id"] = chunk.chunk_id
    user_prompt = render_prompt_template(prompt.user_prompt_template, placeholders)
    messages = _build_messages(
        prompt=prompt,
        user_prompt=user_prompt,
        phrase_contract=_phrase_contract_text(),
    )
    candidates, _done_chunk_ids, exhausted = _call_and_extract_with_retries(
        client=client,
        prompt=prompt,
        messages=messages,
        temperature=temperature,
        chunk_ids=[chunk.chunk_id],
        extract_fn=lambda data: _extract_phrase_candidates_from_items(
            data=data,
            forced_chunk_id=chunk.chunk_id,
        ),
        events=events,
    )
    if exhausted:
        _append_extraction_event(
            events=events,
            payload={
                "stage": "extraction_chunk_skipped_after_parse_retries",
                "chunk_id": chunk.chunk_id,
                "attempts": _EXTRACTION_PARSE_MAX_RETRIES,
            },
        )
        return []
    return candidates


def generate_phrase_candidates_for_batch(
    *,
    client: OpenAICompatibleClient,
    prompt: PromptSpec,
    document: DocumentRecord,
    chunks: list[ChunkRecord],
    temperature: float | None = None,
    events: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not chunks:
        return []
    remaining = list(chunks)
    aggregated: list[dict[str, Any]] = []

    for batch_attempt in range(1, _EXTRACTION_PARSE_MAX_RETRIES + 1):
        merged_chunk_text = _build_chunk_blocks(remaining)
        placeholders = _build_extraction_placeholders(
            client=client,
            document=document,
            chunk_text=merged_chunk_text,
        )
        user_prompt = render_prompt_template(prompt.user_prompt_template, placeholders)
        messages = _build_messages(
            prompt=prompt,
            user_prompt=user_prompt,
            phrase_contract=_phrase_contract_text(),
        )
        chunk_ids = [chunk.chunk_id for chunk in remaining]
        candidates, done_chunk_ids, exhausted = _call_and_extract_with_retries(
            client=client,
            prompt=prompt,
            messages=messages,
            temperature=temperature,
            chunk_ids=chunk_ids,
            extract_fn=lambda data: _extract_phrase_candidates_from_items(data=data),
            events=events,
        )
        aggregated.extend(candidates)
        if exhausted:
            break
        missing_chunks = [chunk for chunk in remaining if chunk.chunk_id not in done_chunk_ids]
        if not missing_chunks:
            remaining = []
            break
        if batch_attempt < _EXTRACTION_PARSE_MAX_RETRIES:
            _append_extraction_event(
                events=events,
                payload={
                    "stage": "extraction_batch_partial_retry",
                    "attempt": batch_attempt,
                    "expected_chunks": len(chunk_ids),
                    "received_chunks": len(done_chunk_ids),
                    "missing_chunk_ids": [chunk.chunk_id for chunk in missing_chunks],
                },
            )
            remaining = missing_chunks
            continue
        remaining = missing_chunks
        break

    if remaining:
        _append_extraction_event(
            events=events,
            payload={
                "stage": "extraction_batch_missing_exhausted",
                "attempts": _EXTRACTION_PARSE_MAX_RETRIES,
                "missing_chunk_ids": [chunk.chunk_id for chunk in remaining],
            },
        )
        for chunk in remaining:
            _append_extraction_event(
                events=events,
                payload={
                    "stage": "extraction_chunk_skipped_after_parse_retries",
                    "chunk_id": chunk.chunk_id,
                    "attempts": _EXTRACTION_PARSE_MAX_RETRIES,
                },
            )
    return aggregated
