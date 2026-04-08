"""Text chunk splitter."""

from __future__ import annotations

from typing import Iterable

from ..models.chunk import ChunkRecord
from ..utils.hash import stable_hash
from ..utils.text import count_sentences, split_sentences


def _split_long_paragraph(
    paragraph: str, *, max_chars: int, overlap_sentences: int
) -> list[str]:
    sentences = split_sentences(paragraph)
    if not sentences:
        return []
    chunks: list[str] = []
    i = 0
    # 以字符数为主，从当前位置开始尽量向后扩展句子，直到接近 max_chars。
    while i < len(sentences):
        end = i + 1
        while end <= len(sentences):
            candidate = " ".join(sentences[i:end]).strip()
            if len(candidate) > max_chars:
                # 超出上限，则回退到上一句作为 chunk。
                end -= 1
                break
            end += 1
        if end <= i:
            # 单句已经超过 max_chars，强行截断到这一句。
            end = i + 1
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
    min_chars: int,
    overlap_sentences: int,
) -> list[ChunkRecord]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged_paragraphs = _merge_short_paragraphs(paragraphs, min_chars=min_chars)

    chunk_texts: list[str] = []
    for para in merged_paragraphs:
        if len(para) <= max_chars:
            chunk_texts.append(para)
            continue
        chunk_texts.extend(
            _split_long_paragraph(
                para,
                max_chars=max_chars,
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

