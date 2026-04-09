"""End-to-end deck building pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..anki.deck_exporter import export_apkg
from ..anki.media_manager import MediaManager
from ..anki.template_loader import load_anki_template
from ..chunking.splitter import split_into_chunks
from ..config import AppConfig, validate_base_config, validate_runtime_config
from ..constants import SUPPORTED_FILE_SUFFIXES
from ..errors import ClawLinguaError, build_error
from ..exit_codes import ExitCode
from ..ingest.epub_reader import read_epub_file
from ..ingest.file_reader import read_text_file
from ..ingest.pdf_reader import read_pdf_file
from ..ingest.normalizer import NormalizeOptions, normalize_text, strip_markdown_to_text
from ..llm.client import OpenAICompatibleClient
from ..llm.cloze_generator import (
    generate_cloze_candidates_for_batch,
    generate_cloze_candidates_for_chunk,
)
from ..llm.prompt_loader import load_prompt
from ..llm.translation_generator import generate_translation
from ..models.card import CardRecord
from ..models.document import DocumentRecord
from ..runtime import create_run_context
from ..tts.provider_registry import get_tts_provider
from ..tts.voice_selector import UniformVoiceSelector
from ..utils.hash import stable_hash
from ..utils.jsonx import dump_json, dump_jsonl
from ..utils.time import utc_now_iso
from .dedupe import dedupe_candidates
from .validators import validate_text_candidate, validate_translation_text

logger = logging.getLogger(__name__)


@dataclass
class BuildDeckOptions:
    input_value: str
    source_lang: str | None = None
    target_lang: str | None = None
    input_char_limit: int | None = None
    output: Path | None = None
    deck_name: str | None = None
    max_chars: int | None = None
    max_sentences: int | None = None
    max_notes: int | None = None
    temperature: float | None = None
    cloze_difficulty: str | None = None
    save_intermediate: bool | None = None
    continue_on_error: bool = False


@dataclass
class BuildDeckResult:
    run_id: str
    run_dir: Path
    output_path: Path
    cards_count: int
    errors_count: int


def _build_note(
    *,
    title: str | None,
    source_url: str | None,
    target_phrases: list[str],
    chunk_id: str,
    source_lang: str,
    target_lang: str,
) -> str:
    lines = [
        f"phrases: {' | '.join(target_phrases)}",
        f"title: {title or ''}",
        f"source: {source_url or ''}",
        f"chunk: {chunk_id}",
        f"source_lang: {source_lang}",
        f"target_lang: {target_lang}",
    ]
    return "\n".join(lines)


def _save_intermediate(
    *,
    run_dir: Path,
    document: DocumentRecord,
    chunks: list,
    raw_candidates: list[dict],
    valid_candidates: list[dict],
    cards: list[CardRecord],
    errors: list[dict],
    output_path: Path,
) -> None:
    dump_json(run_dir / "document.json", document.model_dump(mode="json"))
    (run_dir / "document.md").write_text(document.cleaned_text, encoding="utf-8")
    dump_jsonl(run_dir / "chunks.jsonl", [chunk.model_dump(mode="json") for chunk in chunks])
    dump_jsonl(run_dir / "text_candidates.raw.jsonl", raw_candidates)
    dump_jsonl(run_dir / "text_candidates.validated.jsonl", valid_candidates)
    dump_jsonl(run_dir / "translations.jsonl", [{"original": c.original, "translation": c.translation} for c in cards])
    dump_jsonl(run_dir / "cards.final.jsonl", [card.model_dump(mode="json") for card in cards])
    if errors:
        dump_jsonl(run_dir / "errors.jsonl", errors)
    dump_json(
        run_dir / "run_summary.json",
        {
            "run_id": document.run_id,
            "cards": len(cards),
            "errors": len(errors),
            "output_path": str(output_path),
        },
    )


def _build_document(cfg: AppConfig, run_id: str, options: BuildDeckOptions) -> DocumentRecord:
    source_lang = options.source_lang or cfg.default_source_lang
    target_lang = options.target_lang or cfg.default_target_lang

    file_path = Path(options.input_value)
    if not file_path.is_absolute():
        file_path = (cfg.workspace_root / file_path).resolve()
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_FILE_SUFFIXES:
        raise build_error(
            error_code="INPUT_FILE_TYPE_UNSUPPORTED",
            cause="Unsupported input file type.",
            detail=f"suffix={suffix or '<none>'}",
            next_steps=["Use one of: .txt, .md, .markdown, .epub, .pdf"],
            exit_code=ExitCode.INPUT_ERROR,
        )
    if suffix == ".epub":
        epub_result = read_epub_file(file_path)
        raw_text = epub_result.text
        title = epub_result.title or file_path.stem
    elif suffix == ".pdf":
        pdf_result = read_pdf_file(file_path)
        raw_text = pdf_result.text
        title = pdf_result.title or file_path.stem
    else:
        raw_text = read_text_file(file_path)
        if suffix in {".md", ".markdown"}:
            raw_text = strip_markdown_to_text(raw_text)
        title = file_path.stem

    if options.input_char_limit and options.input_char_limit > 0:
        raw_text = raw_text[: options.input_char_limit]

    source_url = None
    cleaned_markdown = None
    options.input_value = str(file_path)

    cleaned = normalize_text(
        raw_text,
        options=NormalizeOptions(
            short_line_max_words=cfg.ingest_short_line_max_words,
        ),
    )
    if not cleaned:
        raise build_error(
            error_code="INPUT_EMPTY_TEXT",
            cause="输入文本为空。",
            detail="清洗后没有可用文本。",
            next_steps=["检查输入内容是否有效"],
            exit_code=ExitCode.INPUT_ERROR,
        )

    return DocumentRecord(
        run_id=run_id,
        source_type="file",
        source_value=options.input_value,
        source_lang=source_lang,
        target_lang=target_lang,
        title=title,
        source_url=source_url,
        raw_text=raw_text,
        cleaned_text=cleaned,
        cleaned_markdown=cleaned_markdown,
        fetched_at=utc_now_iso(),
    )


def run_build_deck(cfg: AppConfig, options: BuildDeckOptions) -> BuildDeckResult:
    validate_base_config(cfg)
    validate_runtime_config(cfg)

    # CLI 的 --difficulty 优先级高于 env；如提供则覆盖 cfg.cloze_difficulty
    if options.cloze_difficulty:
        cfg.cloze_difficulty = options.cloze_difficulty

    run_ctx = create_run_context(cfg, name="build_deck")
    template = load_anki_template(cfg.resolve_path(cfg.anki_template))
    cloze_prompt = load_prompt(cfg.resolve_path(cfg.prompt_cloze))
    translate_prompt = load_prompt(cfg.resolve_path(cfg.prompt_translate))

    document = _build_document(cfg, run_ctx.run_id, options)
    logger.info('ingest complete | title="%s"', document.title or "")

    save_intermediate = cfg.save_intermediate if options.save_intermediate is None else options.save_intermediate
    output_path = options.output or (run_ctx.run_dir / "output.apkg")
    if not output_path.is_absolute():
        output_path = (cfg.workspace_root / output_path).resolve()

    errors: list[dict] = []
    raw_candidates: list[dict] = []
    valid_candidates: list[dict] = []
    deduped: list[dict] = []
    cards: list[CardRecord] = []
    chunks: list = []

    def _save_failure_snapshot(exc: ClawLinguaError) -> None:
        if not save_intermediate:
            return
        validated = deduped if deduped else valid_candidates
        snapshot_errors = [*errors, {"stage": "fatal", "error": exc.to_lines()}]
        try:
            _save_intermediate(
                run_dir=run_ctx.run_dir,
                document=document,
                chunks=chunks,
                raw_candidates=raw_candidates,
                valid_candidates=validated,
                cards=cards,
                errors=snapshot_errors,
                output_path=output_path,
            )
        except Exception:
            logger.exception("failed to persist failure snapshot")

    chunks = split_into_chunks(
        run_id=run_ctx.run_id,
        text=document.cleaned_text,
        max_chars=options.max_chars or cfg.chunk_max_chars,
        min_chars=cfg.chunk_min_chars,
        overlap_sentences=cfg.chunk_overlap_sentences,
    )
    if not chunks:
        err = build_error(
            error_code="CHUNKING_EMPTY",
            cause="Text chunking failed.",
            detail="No valid chunks were produced.",
            next_steps=["Lower min_chars or check input text structure"],
            exit_code=ExitCode.CHUNKING_ERROR,
        )
        _save_failure_snapshot(err)
        raise err
    logger.info("chunking complete | chunks=%d", len(chunks))

    client = OpenAICompatibleClient(cfg)
    translate_client = OpenAICompatibleClient(cfg, for_translation=True)

    # LLM chunk batch：一次可以处理多个 chunk。
    batch_size = max(1, int(cfg.llm_chunk_batch_size or 1))

    def _iter_batches(items: list[ChunkRecord], size: int) -> list[list[ChunkRecord]]:
        if size <= 1:
            return [[c] for c in items]
        return [items[i : i + size] for i in range(0, len(items), size)]

    for batch in _iter_batches(chunks, batch_size):
        try:
            if len(batch) == 1:
                # 保持单 chunk 行为，便于 debug。
                chunk = batch[0]
                items = generate_cloze_candidates_for_chunk(
                    client=client,
                    prompt=cloze_prompt,
                    document=document,
                    chunk=chunk,
                    temperature=options.temperature,
                )
            else:
                items = generate_cloze_candidates_for_batch(
                    client=client,
                    prompt=cloze_prompt,
                    document=document,
                    chunks=batch,
                    temperature=options.temperature,
                )

            # 补充 chunk_id/chunk_text（有些模型可能没带 chunk_text）
            chunk_map = {c.chunk_id: c for c in batch}
            for item in items:
                cid = str(item.get("chunk_id") or "").strip()
                chunk = chunk_map.get(cid) if cid else None
                if len(batch) > 1 and chunk is None:
                    # batch 模式下必须有可识别的 chunk_id，避免候选挂错 chunk。
                    errors.append(
                        {
                            "stage": "cloze_batch_mapping",
                            "reason": "missing_or_unknown_chunk_id",
                            "chunk_id": cid,
                            "item": item,
                        }
                    )
                    if options.continue_on_error:
                        continue
                    err = build_error(
                        error_code="CLOZE_BATCH_CHUNK_ID_MISSING",
                        cause="Batch cloze output is missing a valid chunk_id.",
                        detail=f"chunk_id={cid!r}",
                        next_steps=["Force model output to include a valid chunk_id"],
                        exit_code=ExitCode.LLM_PARSE_ERROR,
                    )
                    _save_failure_snapshot(err)
                    raise err
                if chunk is not None:
                    item["chunk_id"] = chunk.chunk_id
                    item["chunk_text"] = chunk.source_text
                raw_candidates.append(item)
        except ClawLinguaError as exc:
            if not options.continue_on_error:
                _save_failure_snapshot(exc)
                raise
            # 记录整个 batch 的错误，但保留每个 chunk_id 方便排查。
            errors.append(
                {
                    "stage": "cloze",
                    "chunk_ids": [c.chunk_id for c in batch],
                    "error": exc.to_lines(),
                }
            )

    validation_rejects = 0
    first_validation_reason: str | None = None
    for item in raw_candidates:
        ok, reason = validate_text_candidate(
            item,
            max_sentences=cfg.cloze_max_sentences,
            min_chars=cfg.cloze_min_chars,
            difficulty=cfg.cloze_difficulty,
        )
        if not ok:
            validation_rejects += 1
            if first_validation_reason is None:
                first_validation_reason = reason
            errors.append({"stage": "validate_text", "reason": reason, "item": item})
            continue
        valid_candidates.append(item)

    if validation_rejects:
        logger.warning(
            "validation filtered candidates | rejected=%d raw=%d first_reason=%s",
            validation_rejects,
            len(raw_candidates),
            first_validation_reason or "",
        )
    if not valid_candidates:
        err = build_error(
            error_code="CARD_VALIDATION_FAILED",
            cause="All candidates failed validation.",
            detail=f"raw={len(raw_candidates)}, first_reason={first_validation_reason or 'unknown'}",
            next_steps=["Adjust prompt constraints", "Lower CLAWLINGUA_CLOZE_MIN_CHARS if needed"],
            exit_code=ExitCode.CARD_VALIDATION_ERROR,
        )
        _save_failure_snapshot(err)
        raise err

    deduped = dedupe_candidates(valid_candidates)
    if cfg.cloze_max_per_chunk and cfg.cloze_max_per_chunk > 0:
        # Per-chunk cap to avoid generating too many cards from a single segment.
        by_chunk: dict[str, list[dict]] = {}
        for item in deduped:
            cid = str(item.get("chunk_id", ""))
            bucket = by_chunk.setdefault(cid, [])
            if len(bucket) < cfg.cloze_max_per_chunk:
                bucket.append(item)
        deduped = [c for bucket in by_chunk.values() for c in bucket]

    if options.max_notes and options.max_notes > 0:
        deduped = deduped[: options.max_notes]
    logger.info("text generation complete | raw=%d valid=%d", len(raw_candidates), len(deduped))

    for idx, item in enumerate(deduped, start=1):
        try:
            translation = generate_translation(
                client=translate_client,
                prompt=translate_prompt,
                document=document,
                chunk_text="",
                text_original=str(item["original"]),
                temperature=options.temperature,
            )
            ok, reason = validate_translation_text(translation)
            if not ok:
                raise build_error(
                    error_code="TRANSLATION_VALIDATION_FAILED",
                    cause="Translation 校验失败。",
                    detail=reason,
                    next_steps=["检查 translate prompt 输出"],
                    exit_code=ExitCode.CARD_VALIDATION_ERROR,
                )
        except ClawLinguaError as exc:
            if not options.continue_on_error:
                _save_failure_snapshot(exc)
                raise
            errors.append({"stage": "translation", "index": idx, "error": exc.to_lines()})
            continue

        card_id = f"card_{idx:06d}_{stable_hash(str(item['original']), length=6)}"
        target_phrases = list(item.get("target_phrases") or [])
        note = _build_note(
            title=document.title,
            source_url=document.source_url,
            target_phrases=[str(x) for x in target_phrases],
            chunk_id=str(item.get("chunk_id", "")),
            source_lang=document.source_lang,
            target_lang=document.target_lang,
        )
        cards.append(
            CardRecord(
                run_id=run_ctx.run_id,
                card_id=card_id,
                chunk_id=str(item.get("chunk_id", "")),
                source_lang=document.source_lang,
                target_lang=document.target_lang,
                title=document.title,
                source_url=document.source_url,
                text=str(item["text"]),
                original=str(item["original"]),
                translation=translation,
                note=note,
                target_phrases=[str(x) for x in target_phrases],
            )
        )

    if not cards:
        err = build_error(
            error_code="CARD_EMPTY",
            cause="No exportable cards were produced.",
            detail="Candidates became empty after validation/translation.",
            next_steps=["Check input quality", "Enable --save-intermediate and inspect outputs"],
            exit_code=ExitCode.CARD_VALIDATION_ERROR,
        )
        _save_failure_snapshot(err)
        raise err
    logger.info("translation generation complete | translated=%d", len(cards))

    voices = cfg.get_source_voices(document.source_lang)
    media_manager = MediaManager(run_ctx.media_dir, ext=cfg.tts_output_format)
    media_files: list[Path] = []

    if not voices:
        logger.info("tts skipped | reason=no voices configured")
    else:
        tts_provider = get_tts_provider(cfg)
        if len(voices) < 3:
            logger.warning("tts voice list has less than 3 voices | voices=%d", len(voices))
        selector = UniformVoiceSelector(seed=cfg.tts_random_seed)
        for card in cards:
            media = media_manager.next_audio_file()
            voice = selector.select(voices)
            try:
                tts_provider.synthesize(
                    text=card.original,
                    voice=voice,
                    output_path=media.path,
                    lang=card.source_lang,
                )
            except ClawLinguaError as exc:
                if not options.continue_on_error:
                    _save_failure_snapshot(exc)
                    raise
                errors.append({"stage": "tts", "card_id": card.card_id, "error": exc.to_lines()})
                continue
            card.audio_file = media.filename
            card.audio_field = media_manager.to_anki_sound_field(media.filename)
            media_files.append(media.path)
        logger.info("tts generation complete | audio=%d", len(media_files))

    try:
        export_apkg(
            cards=cards,
            template=template,
            output_path=output_path,
            media_files=media_files,
            deck_name_override=options.deck_name or cfg.default_deck_name,
        )
    except ClawLinguaError as exc:
        _save_failure_snapshot(exc)
        raise
    logger.info("deck export complete | file=%s", output_path)

    if save_intermediate:
        _save_intermediate(
            run_dir=run_ctx.run_dir,
            document=document,
            chunks=chunks,
            raw_candidates=raw_candidates,
            valid_candidates=deduped,
            cards=cards,
            errors=errors,
            output_path=output_path,
        )

    return BuildDeckResult(
        run_id=run_ctx.run_id,
        run_dir=run_ctx.run_dir,
        output_path=output_path,
        cards_count=len(cards),
        errors_count=len(errors),
    )
