"""End-to-end textbook deck building pipeline (minimal demo)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..anki.deck_exporter import export_apkg
from ..anki.template_loader import load_anki_template
from ..config import AppConfig, validate_base_config, validate_runtime_config
from ..constants import SUPPORTED_FILE_SUFFIXES
from ..errors import build_error
from ..exit_codes import ExitCode
from ..ingest.epub_reader import read_epub_file
from ..ingest.file_reader import read_text_file
from ..ingest.normalizer import NormalizeOptions, normalize_text, strip_markdown_to_text
from ..models.card import CardRecord
from ..models.document import DocumentRecord
from ..runtime import create_run_context
from ..utils.hash import stable_hash
from ..utils.jsonx import dump_json, dump_jsonl
from ..utils.time import utc_now_iso
from .core_chunking import chunk_document
from .core_export import resolve_output_path
from .core_io import resolve_input_path


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


@dataclass
class BuildTextbookDeckOptions:
    input_value: str
    run_id: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    input_char_limit: int | None = None
    output: Path | None = None
    deck_name: str | None = None
    max_chars: int | None = None
    max_notes: int | None = None
    max_concepts_per_chunk: int | None = None
    keep_source_excerpt: bool = True
    save_intermediate: bool | None = None
    continue_on_error: bool = False


@dataclass
class BuildDeckResult:
    run_id: str
    run_dir: Path
    output_path: Path
    cards_count: int
    errors_count: int


def _split_sentences(text: str) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    parts = re.split(r"(?<=[.!?])\s+", value)
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return cleaned if cleaned else [value]


def _extract_concept_candidate(chunk_text: str) -> tuple[str, str, str] | None:
    sentences = _split_sentences(chunk_text)
    if not sentences:
        return None
    sentence = sentences[0]
    tokens = _TOKEN_RE.findall(sentence)
    if len(tokens) < 3:
        return None
    concept_title = " ".join(tokens[: min(6, len(tokens))]).strip()
    explanation = f"Key idea: {concept_title}"
    return concept_title, sentence, explanation


def _extract_concept_candidates(
    chunk_text: str,
    *,
    max_concepts_per_chunk: int,
) -> list[tuple[str, str, str]]:
    max_per_chunk = max(1, int(max_concepts_per_chunk))
    # Preserve legacy behavior for the default path: only inspect the first sentence.
    if max_per_chunk <= 1:
        concept = _extract_concept_candidate(chunk_text)
        return [concept] if concept is not None else []

    concepts: list[tuple[str, str, str]] = []
    seen_titles: set[str] = set()
    for sentence in _split_sentences(chunk_text):
        tokens = _TOKEN_RE.findall(sentence)
        if len(tokens) < 3:
            continue
        concept_title = " ".join(tokens[: min(6, len(tokens))]).strip()
        if not concept_title:
            continue
        key = concept_title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        concepts.append((concept_title, sentence, f"Key idea: {concept_title}"))
        if len(concepts) >= max_per_chunk:
            break
    return concepts


def _build_textbook_cloze(sentence: str, concept_title: str) -> str:
    target = str(concept_title or "").strip()
    if not sentence or not target:
        return ""
    idx = sentence.lower().find(target.lower())
    if idx < 0:
        return f"{{{{c1::<b>{target}</b>}}}}(concept): {sentence}"
    matched = sentence[idx : idx + len(target)]
    return f"{sentence[:idx]}{{{{c1::<b>{matched}</b>}}}}(concept){sentence[idx + len(matched):]}"


def _build_document(cfg: AppConfig, run_id: str, options: BuildTextbookDeckOptions) -> DocumentRecord:
    source_lang = options.source_lang or cfg.default_source_lang
    target_lang = options.target_lang or cfg.default_target_lang

    file_path = resolve_input_path(workspace_root=cfg.workspace_root, input_value=options.input_value)
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_FILE_SUFFIXES:
        raise build_error(
            error_code="INPUT_FILE_TYPE_UNSUPPORTED",
            cause="Unsupported input file type.",
            detail=f"suffix={suffix or '<none>'}",
            next_steps=["Use one of: .txt, .md, .markdown, .epub"],
            exit_code=ExitCode.INPUT_ERROR,
        )

    if suffix == ".epub":
        epub_result = read_epub_file(file_path)
        raw_text = epub_result.text
        title = epub_result.title or file_path.stem
    else:
        raw_text = read_text_file(file_path)
        if suffix in {".md", ".markdown"}:
            raw_text = strip_markdown_to_text(raw_text)
        title = file_path.stem

    if options.input_char_limit and options.input_char_limit > 0:
        raw_text = raw_text[: options.input_char_limit]

    cleaned = normalize_text(
        raw_text,
        options=NormalizeOptions(
            short_line_max_words=cfg.ingest_short_line_max_words,
            material_profile="textbook_examples",
        ),
    )
    if not cleaned:
        raise build_error(
            error_code="INPUT_EMPTY_TEXT",
            cause="Input text is empty after normalization.",
            detail="No usable text produced.",
            next_steps=["Check input content"],
            exit_code=ExitCode.INPUT_ERROR,
        )

    options.input_value = str(file_path)
    return DocumentRecord(
        run_id=run_id,
        source_type="file",
        source_value=options.input_value,
        source_lang=source_lang,
        target_lang=target_lang,
        title=title,
        source_url=None,
        raw_text=raw_text,
        cleaned_text=cleaned,
        cleaned_markdown=None,
        fetched_at=utc_now_iso(),
        metadata={
            "material_profile": "textbook_examples",
            "learning_mode": "textbook_concept",
        },
    )


def run_build_textbook_deck(cfg: AppConfig, options: BuildTextbookDeckOptions) -> BuildDeckResult:
    validate_base_config(cfg)
    validate_runtime_config(cfg)

    run_ctx = create_run_context(cfg, name="build_textbook", run_id=options.run_id)
    template = load_anki_template(cfg.resolve_path(cfg.anki_template))
    document = _build_document(cfg, run_ctx.run_id, options)
    output_path = resolve_output_path(
        workspace_root=cfg.workspace_root,
        export_dir=cfg.export_dir,
        run_id=document.run_id,
        explicit_output=options.output,
    )

    chunks = chunk_document(
        run_id=run_ctx.run_id,
        text=document.cleaned_text,
        max_chars=options.max_chars or cfg.chunk_max_chars,
        min_chars=cfg.chunk_min_chars,
        overlap_sentences=cfg.chunk_overlap_sentences,
        material_profile="textbook_examples",
        difficulty=cfg.cloze_difficulty,
    )
    if not chunks:
        raise build_error(
            error_code="CHUNKING_EMPTY",
            cause="Text chunking failed.",
            detail="No valid chunks were produced.",
            next_steps=["Lower min chars or check input text structure"],
            exit_code=ExitCode.CHUNKING_ERROR,
        )

    errors: list[dict[str, Any]] = []
    cards: list[CardRecord] = []
    candidates: list[dict[str, Any]] = []
    max_concepts_per_chunk = max(1, int(options.max_concepts_per_chunk or 1))
    keep_source_excerpt = bool(options.keep_source_excerpt)
    for idx, chunk in enumerate(chunks, start=1):
        concepts = _extract_concept_candidates(
            str(getattr(chunk, "source_text", "")),
            max_concepts_per_chunk=max_concepts_per_chunk,
        )
        for concept_idx, (concept_title, sentence, explanation) in enumerate(concepts, start=1):
            cloze_text = _build_textbook_cloze(sentence, concept_title)
            if not cloze_text:
                continue
            excerpt = sentence if keep_source_excerpt else ""
            card_id = (
                f"textbook_{idx:06d}_{concept_idx:02d}_{stable_hash(sentence, length=6)}"
            )
            candidates.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "concept_title": concept_title,
                    "excerpt": excerpt,
                    "explanation": explanation,
                    "text": cloze_text,
                }
            )
            cards.append(
                CardRecord(
                    run_id=run_ctx.run_id,
                    card_id=card_id,
                    chunk_id=chunk.chunk_id,
                    source_lang=document.source_lang,
                    target_lang=document.target_lang,
                    title=document.title,
                    source_url=document.source_url,
                    text=cloze_text,
                    original=excerpt,
                    translation=explanation,
                    note=f"concept: {concept_title}\nchunk: {chunk.chunk_id}",
                    target_phrases=[concept_title],
                    phrase_types=[],
                    expression_transfer="",
                )
            )

    if options.max_notes and options.max_notes > 0:
        cards = cards[: options.max_notes]
        candidates = candidates[: options.max_notes]

    if not cards and not cfg.allow_empty_deck:
        raise build_error(
            error_code="CARD_EMPTY",
            cause="No exportable cards were produced.",
            detail="No textbook concept candidates passed minimal extraction.",
            next_steps=["Check input quality", "Try larger input text"],
            exit_code=ExitCode.CARD_VALIDATION_ERROR,
        )

    export_apkg(
        cards=cards,
        template=template,
        output_path=output_path,
        media_files=[],
        deck_name_override=options.deck_name or document.title or cfg.default_deck_name,
    )

    save_intermediate = cfg.save_intermediate if options.save_intermediate is None else options.save_intermediate
    if save_intermediate:
        dump_json(run_ctx.run_dir / "document.json", document.model_dump(mode="json"))
        dump_jsonl(run_ctx.run_dir / "chunks.jsonl", [chunk.model_dump(mode="json") for chunk in chunks])
        dump_jsonl(run_ctx.run_dir / "textbook_candidates.jsonl", candidates)
        dump_jsonl(run_ctx.run_dir / "cards.final.jsonl", [card.model_dump(mode="json") for card in cards])
        if errors:
            dump_jsonl(run_ctx.run_dir / "errors.jsonl", errors)
        dump_json(
            run_ctx.run_dir / "run_summary.json",
            {
                "run_id": run_ctx.run_id,
                "domain": "textbook",
                "cards": len(cards),
                "errors": len(errors),
                "output_path": str(output_path),
            },
        )

    return BuildDeckResult(
        run_id=run_ctx.run_id,
        run_dir=run_ctx.run_dir,
        output_path=output_path,
        cards_count=len(cards),
        errors_count=len(errors),
    )
