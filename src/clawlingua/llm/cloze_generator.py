"""Generate cloze candidates from chunks."""

from __future__ import annotations

from ..models.chunk import ChunkRecord
from ..models.document import DocumentRecord
from ..models.prompt_schema import PromptSpec
from .client import OpenAICompatibleClient
from .response_parser import parse_json_content


def _render(template: str, values: dict[str, str]) -> str:
    return template.format(**values)


def generate_cloze_candidates_for_chunk(
    *,
    client: OpenAICompatibleClient,
    prompt: PromptSpec,
    document: DocumentRecord,
    chunk: ChunkRecord,
    temperature: float | None = None,
) -> list[dict]:
    max_per_chunk = client.config.cloze_max_per_chunk
    effective_max_per_chunk = max_per_chunk if max_per_chunk and max_per_chunk > 0 else 4
    placeholders = {
        "source_lang": document.source_lang,
        "target_lang": document.target_lang,
        "document_title": document.title or "",
        "source_url": document.source_url or "",
        "chunk_id": chunk.chunk_id,
        "chunk_text": chunk.source_text,
        "difficulty": client.config.cloze_difficulty,
        "cloze_max_sentences": str(client.config.cloze_max_sentences),
        "cloze_min_chars": str(client.config.cloze_min_chars),
        "cloze_max_per_chunk": str(effective_max_per_chunk),
    }
    user_prompt = _render(prompt.user_prompt_template, placeholders)
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
        chunk_id = item.get("chunk_id") or chunk.chunk_id
        candidates.append(
            {
                "chunk_id": str(chunk_id).strip(),
                "text": str(text).strip(),
                "original": str(original).strip(),
                "target_phrases": [str(x).strip() for x in target_phrases if str(x).strip()],
                "note_hint": str(note_hint).strip(),
            }
        )
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

    max_per_chunk = client.config.cloze_max_per_chunk
    effective_max_per_chunk = max_per_chunk if max_per_chunk and max_per_chunk > 0 else 4
    placeholders = {
        "source_lang": document.source_lang,
        "target_lang": document.target_lang,
        "document_title": document.title or "",
        "source_url": document.source_url or "",
        "chunk_text": merged_chunk_text,
        "difficulty": client.config.cloze_difficulty,
        "cloze_max_sentences": str(client.config.cloze_max_sentences),
        "cloze_min_chars": str(client.config.cloze_min_chars),
        "cloze_max_per_chunk": str(effective_max_per_chunk),
    }
    user_prompt = _render(prompt.user_prompt_template, placeholders)
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
        candidates.append(
            {
                "chunk_id": str(chunk_id).strip(),
                "text": str(text).strip(),
                "original": str(original).strip(),
                "target_phrases": [str(x).strip() for x in target_phrases if str(x).strip()],
                "note_hint": str(note_hint).strip(),
            }
        )
    return candidates
