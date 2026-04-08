"""Text chunk splitter."""

from __future__ import annotations

from typing import Iterable

from ..models.chunk import ChunkRecord
from ..utils.hash import stable_hash
from ..utils.text import count_sentences, split_sentences


def _split_long_paragraph(
    paragraph: str, *, max_chars: int, max_sentences: int, overlap_sentences: int
) -> list[str]:
    sentences = split_sentences(paragraph)
    if not sentences:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(sentences):
        end = min(i + max_sentences, len(sentences))
        while end > i + 1:
            candidate = " ".join(sentences[i:end]).strip()
            if len(candidate) <= max_chars:
                break
            end -= 1
        candidate = " ".join(sentences[i:end]).strip()
        if candidate:
            chunks.append(candidate)
        if end >= len(sentences):
            break
        i = max(i + 1, end - max(0, overlap_sentences))
    return chunks


def _merge_short_paragraphs(paragraphs: Iterable[str], *, min_chars: int) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if not buffer:
            buffer = para
            continue
        if len(buffer) < min_chars:
            buffer = f"{buffer}\n\n{para}"
        else:
            merged.append(buffer)
            buffer = para
    if buffer:
        merged.append(buffer)
    return merged


def split_into_chunks(
    *,
    run_id: str,
    text: str,
    max_chars: int,
    max_sentences: int,
    min_chars: int,
    overlap_sentences: int,
) -> list[ChunkRecord]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged_paragraphs = _merge_short_paragraphs(paragraphs, min_chars=min_chars)

    chunk_texts: list[str] = []
    for para in merged_paragraphs:
        if len(para) <= max_chars and count_sentences(para) <= max_sentences:
            chunk_texts.append(para)
            continue
        chunk_texts.extend(
            _split_long_paragraph(
                para,
                max_chars=max_chars,
                max_sentences=max_sentences,
                overlap_sentences=overlap_sentences,
            )
        )

    records: list[ChunkRecord] = []
    for idx, chunk_text in enumerate(chunk_texts, start=1):
        cleaned = chunk_text.strip()
        if not cleaned:
            continue
        chunk_id = f"chunk_{idx:04d}_{stable_hash(cleaned, length=6)}"
        records.append(
            ChunkRecord(
                run_id=run_id,
                chunk_id=chunk_id,
                chunk_index=idx,
                source_text=cleaned,
                char_count=len(cleaned),
                sentence_count=count_sentences(cleaned),
            )
        )
    return records

