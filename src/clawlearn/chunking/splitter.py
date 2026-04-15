"""Text chunk splitter."""

from __future__ import annotations

from typing import Iterable

from ..models.chunk import ChunkRecord
from ..utils.hash import stable_hash
from ..utils.text import count_sentences, split_sentences


def _split_long_paragraph(
    paragraph: str,
    *,
    max_chars: int,
    overlap_sentences: int,
) -> list[str]:
    sentences = split_sentences(paragraph)
    if not sentences:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(sentences):
        end = i + 1
        while end <= len(sentences):
            candidate = " ".join(sentences[i:end]).strip()
            if len(candidate) > max_chars:
                end -= 1
                break
            end += 1
        if end <= i:
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


def _build_chunks_from_units(
    units: list[str],
    *,
    max_chars: int,
    overlap_units: int,
) -> list[str]:
    if not units:
        return []
    chunk_texts: list[str] = []
    i = 0
    while i < len(units):
        end = i
        acc: list[str] = []
        char_count = 0
        while end < len(units):
            unit = units[end].strip()
            if not unit:
                end += 1
                continue
            next_count = char_count + len(unit) + (1 if acc else 0)
            if acc and next_count > max_chars:
                break
            acc.append(unit)
            char_count = next_count
            end += 1
        if not acc:
            i += 1
            continue
        chunk_texts.append(" ".join(acc).strip())
        if end >= len(units):
            break
        i = max(i + 1, end - max(0, overlap_units))
    return chunk_texts


def split_into_chunks(
    *,
    run_id: str,
    text: str,
    max_chars: int,
    min_chars: int,
    overlap_sentences: int,
    material_profile: str = "prose_article",
    difficulty: str = "intermediate",
) -> list[ChunkRecord]:
    profile = (material_profile or "prose_article").strip().lower()
    if profile == "general":
        profile = "prose_article"
    diff = (difficulty or "intermediate").strip().lower()
    effective_max_chars = max(120, int(max_chars))
    if diff == "advanced":
        # Advanced profile prefers tighter context windows for more precise mining.
        effective_max_chars = max(120, int(effective_max_chars * 0.85))

    chunk_texts: list[str]
    if profile == "transcript_dialogue":
        # transcript-like inputs: treat each non-empty line as one unit and avoid
        # aggressive cross-segment merging.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        overlap_units = max(0, min(1, overlap_sentences))
        chunk_texts = _build_chunks_from_units(
            lines,
            max_chars=effective_max_chars,
            overlap_units=overlap_units,
        )
    else:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        merged_paragraphs = _merge_short_paragraphs(paragraphs, min_chars=min_chars)
        chunk_texts = []
        for para in merged_paragraphs:
            if len(para) <= effective_max_chars:
                chunk_texts.append(para)
                continue
            chunk_texts.extend(
                _split_long_paragraph(
                    para,
                    max_chars=effective_max_chars,
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
