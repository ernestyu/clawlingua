"""End-to-end lingua deck building pipeline."""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

from ..anki.deck_exporter import export_apkg
from ..anki.media_manager import MediaManager
from ..anki.template_loader import load_anki_template
from ..config import AppConfig, validate_base_config, validate_runtime_config
from ..constants import SUPPORTED_CONTENT_PROFILES, SUPPORTED_FILE_SUFFIXES, SUPPORTED_LEARNING_MODES
from ..errors import ClawLearnError, build_error
from ..exit_codes import ExitCode
from ..ingest.epub_reader import read_epub_file
from ..ingest.file_reader import read_text_file
from ..ingest.normalizer import NormalizeOptions, normalize_text, strip_markdown_to_text
from ..llm.client import OpenAICompatibleClient
from ..llm.cloze_generator import (
    generate_cloze_candidates_for_batch,
    generate_cloze_candidates_for_chunk,
    generate_phrase_candidates_for_batch,
    generate_phrase_candidates_for_chunk,
)
from ..llm.taxonomy_classifier import classify_phrase_types_batch
from ..llm.prompt_loader import load_prompt
from ..llm.response_parser import parse_json_content
from ..llm.translation_generator import (
    TranslationBatchResult,
    generate_phrase_translations_batch,
    generate_translation_batch,
)
from ..models.card import CardRecord
from ..models.document import DocumentRecord
from ..runtime import create_run_context
from ..tts.provider_registry import get_tts_provider
from ..tts.voice_selector import UniformVoiceSelector
from ..utils.hash import stable_hash
from ..utils.jsonx import dump_json, dump_jsonl, load_json
from ..utils.text import normalize_for_dedupe
from ..utils.time import utc_now_iso
from .core_chunking import chunk_document
from .core_export import resolve_output_path
from .core_io import resolve_input_path
from .core_llm import iter_batches
from .dedupe import dedupe_candidates
from .ranking import rank_candidates
from .taxonomy import normalize_phrase_types
from .validators import classify_rejection_reason, validate_text_candidate, validate_translation_text

logger = logging.getLogger(__name__)
TEXTBOOK_PROFILE_MAX_RECOMMENDED_CLOZE_MIN_CHARS = 120
TRANSLATION_BATCH_MAX_RETRIES = 3
CLOZE_MARK_RE = re.compile(r"\{\{c\d+::")
_FORMAT_RETRY_TEXT_EXCEEDS_RE = re.compile(r"^format:text exceeds (\d+) sentences$")
_FORMAT_RETRY_MIN_CHARS_RE = re.compile(r"^format:text chars < (\d+)$")
_CLOZE_BLOCK_GENERIC_RE = re.compile(r"\{\{c(\d+)::(.*?)\}\}", re.DOTALL)
_CLOZE_WITH_HINT_RE = re.compile(r"(\{\{c\d+::\s*<b>(.*?)</b>\s*\}\})\(([^)]*)\)", re.DOTALL)
_CHUNK_ID_INDEX_RE = re.compile(r"^chunk_(\d+)(?:_|$)", re.IGNORECASE)
_FORMAT_RETRYABLE_REASONS = (
    "format:missing cloze marker",
    "format:cloze style must be {{cN::<b>...</b>}}(hint)",
    "format:text exceeds ",
    "format:text chars < ",
    "format:target_phrases is empty or invalid",
)
_FORMAT_RETRYABLE_REASONS_LOWER = tuple(prefix.lower() for prefix in _FORMAT_RETRYABLE_REASONS)
_TAXONOMY_RETRYABLE_REASONS = (
    "taxonomy:invalid_labels:",
    "taxonomy:too_many_labels:",
)
_TAXONOMY_RETRYABLE_REASONS_LOWER = tuple(prefix.lower() for prefix in _TAXONOMY_RETRYABLE_REASONS)


@dataclass
class _FormatRetryStats:
    attempt1_fixed_count: int = 0
    llm_repair_success_count: int = 0
    llm_regen_success_count: int = 0
    total_extra_llm_calls: int = 0

    @property
    def recovered_candidates(self) -> int:
        return (
            self.attempt1_fixed_count
            + self.llm_repair_success_count
            + self.llm_regen_success_count
        )

    def absorb(self, other: "_FormatRetryStats") -> None:
        self.attempt1_fixed_count += int(other.attempt1_fixed_count)
        self.llm_repair_success_count += int(other.llm_repair_success_count)
        self.llm_regen_success_count += int(other.llm_regen_success_count)
        self.total_extra_llm_calls += int(other.total_extra_llm_calls)


@dataclass
class _TaxonomyRepairStats:
    taxonomy_reject_count: int = 0
    taxonomy_repair_attempted: int = 0
    taxonomy_repair_success: int = 0
    taxonomy_repair_failed: int = 0

    def absorb(self, other: "_TaxonomyRepairStats") -> None:
        self.taxonomy_reject_count += int(other.taxonomy_reject_count)
        self.taxonomy_repair_attempted += int(other.taxonomy_repair_attempted)
        self.taxonomy_repair_success += int(other.taxonomy_repair_success)
        self.taxonomy_repair_failed += int(other.taxonomy_repair_failed)


@dataclass
class PhraseCandidate:
    chunk_id: str
    sentence_text: str
    phrase_text: str
    reason: str | None = None
    phrase_types: list[str] | None = None
    learning_value_score: float | None = None
    expression_transfer: str | None = None
    chunk_text: str | None = None


@dataclass
class ClozeUnit:
    chunk_id: str
    text_cloze: str
    text_original: str
    target_phrases: list[str]
    phrase_types: list[str]
    expression_transfer: str | None = None
    learning_value_score: float | None = None
    chunk_text: str | None = None


@dataclass
class BuildDeckOptions:
    input_value: str
    run_id: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    material_profile: str | None = None
    learning_mode: str | None = None
    input_char_limit: int | None = None
    output: Path | None = None
    deck_name: str | None = None
    max_chars: int | None = None
    max_sentences: int | None = None
    max_notes: int | None = None
    temperature: float | None = None
    cloze_difficulty: str | None = None
    cloze_min_chars: int | None = None
    extract_prompt: Path | None = None
    explain_prompt: Path | None = None
    # Legacy alias of material_profile.
    content_profile: str | None = None
    save_intermediate: bool | None = None
    continue_on_error: bool = False


@dataclass
class BuildDeckResult:
    run_id: str
    run_dir: Path
    output_path: Path
    cards_count: int
    errors_count: int


def _use_phrase_extraction_pipeline(*, learning_mode: str, schema_name: str | None) -> bool:
    mode = (learning_mode or "").strip().lower()
    schema = (schema_name or "").strip().lower()
    return mode == "expression_mining_v2" or schema == "phrase_candidates_v1"


def _coerce_learning_value_score(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_phrase_candidate_item(item: dict[str, Any]) -> PhraseCandidate | None:
    chunk_id = str(item.get("chunk_id") or "").strip()
    sentence_text = str(item.get("sentence_text") or item.get("sentence") or "").strip()
    phrase_text = str(item.get("phrase_text") or item.get("text") or "").strip()
    if not chunk_id or not sentence_text or not phrase_text:
        return None
    reason = str(item.get("reason") or item.get("selection_reason") or "").strip() or None
    phrase_types = normalize_phrase_types(item.get("phrase_types"), max_items=2)
    expression_transfer = str(item.get("expression_transfer") or "").strip() or None
    chunk_text = str(item.get("chunk_text") or "").strip() or None
    return PhraseCandidate(
        chunk_id=chunk_id,
        sentence_text=sentence_text,
        phrase_text=phrase_text,
        reason=reason,
        phrase_types=phrase_types,
        learning_value_score=_coerce_learning_value_score(item.get("learning_value_score")),
        expression_transfer=expression_transfer,
        chunk_text=chunk_text,
    )


def _candidate_span(sentence_text: str, phrase_text: str) -> tuple[int, int, str] | None:
    sentence = str(sentence_text or "")
    phrase = str(phrase_text or "").strip()
    if not sentence or not phrase:
        return None
    start = sentence.find(phrase)
    if start >= 0:
        end = start + len(phrase)
        return start, end, sentence[start:end]
    lowered_sentence = sentence.lower()
    lowered_phrase = phrase.lower()
    start = lowered_sentence.find(lowered_phrase)
    if start < 0:
        return None
    end = start + len(phrase)
    if end > len(sentence):
        end = start + len(lowered_phrase)
    return start, end, sentence[start:end]


def _select_non_overlapping_spans(
    sentence_text: str,
    candidates: list[PhraseCandidate],
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    seen_phrase_keys: set[str] = set()
    for candidate in candidates:
        key = normalize_for_dedupe(candidate.phrase_text)
        if not key or key in seen_phrase_keys:
            continue
        found = _candidate_span(sentence_text, candidate.phrase_text)
        if found is None:
            continue
        start, end, matched_phrase = found
        seen_phrase_keys.add(key)
        spans.append(
            {
                "start": start,
                "end": end,
                "phrase": matched_phrase,
                "phrase_types": list(candidate.phrase_types or []),
                "reason": candidate.reason or "",
                "learning_value_score": candidate.learning_value_score,
                "expression_transfer": candidate.expression_transfer or "",
            }
        )
    if not spans:
        return []

    spans.sort(key=lambda item: (-(item["end"] - item["start"]), item["start"]))
    selected: list[dict[str, Any]] = []
    for span in spans:
        overlaps = any(
            not (span["end"] <= existing["start"] or span["start"] >= existing["end"])
            for existing in selected
        )
        if overlaps:
            continue
        selected.append(span)
    selected.sort(key=lambda item: item["start"])
    return selected


def _build_cloze_text_from_spans(sentence_text: str, spans: list[dict[str, Any]]) -> tuple[str, list[str]]:
    if not spans:
        return "", []
    pieces: list[str] = []
    phrases: list[str] = []
    cursor = 0
    cloze_index = 1
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        if start < cursor:
            continue
        phrase = str(span["phrase"]).strip()
        if not phrase:
            continue
        pieces.append(sentence_text[cursor:start])
        pieces.append(f"{{{{c{cloze_index}::<b>{phrase}</b>}}}}(hint)")
        phrases.append(phrase)
        cursor = end
        cloze_index += 1
    pieces.append(sentence_text[cursor:])
    return "".join(pieces).strip(), phrases


def _collect_phrase_types_from_spans(spans: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for span in spans:
        for label in normalize_phrase_types(span.get("phrase_types"), max_items=2):
            if label not in labels:
                labels.append(label)
            if len(labels) >= 2:
                return labels
    return labels


def _build_cloze_units_from_phrases(phrase_candidates: list[PhraseCandidate]) -> list[ClozeUnit]:
    grouped: dict[tuple[str, str], list[PhraseCandidate]] = {}
    for candidate in phrase_candidates:
        key = (candidate.chunk_id, candidate.sentence_text)
        grouped.setdefault(key, []).append(candidate)

    units: list[ClozeUnit] = []
    for (chunk_id, sentence_text), group in grouped.items():
        spans = _select_non_overlapping_spans(sentence_text, group)
        if not spans:
            continue
        text_cloze, target_phrases = _build_cloze_text_from_spans(sentence_text, spans)
        if not text_cloze or not target_phrases:
            continue
        phrase_types = _collect_phrase_types_from_spans(spans)
        expression_transfer = next(
            (
                str(span.get("expression_transfer") or "").strip()
                for span in spans
                if str(span.get("expression_transfer") or "").strip()
            ),
            "",
        )
        scores = [span.get("learning_value_score") for span in spans if isinstance(span.get("learning_value_score"), float)]
        learning_value_score = (sum(scores) / len(scores)) if scores else None
        chunk_text = next((c.chunk_text for c in group if c.chunk_text), None)
        units.append(
            ClozeUnit(
                chunk_id=chunk_id,
                text_cloze=text_cloze,
                text_original=sentence_text,
                target_phrases=target_phrases,
                phrase_types=phrase_types,
                expression_transfer=expression_transfer or None,
                learning_value_score=learning_value_score,
                chunk_text=chunk_text,
            )
        )
    return units


def _cloze_unit_to_candidate(unit: ClozeUnit) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "chunk_id": unit.chunk_id,
        "text": unit.text_cloze,
        "original": unit.text_original,
        "target_phrases": list(unit.target_phrases),
        "phrase_types": list(unit.phrase_types),
        "note_hint": "",
    }
    if unit.expression_transfer:
        candidate["expression_transfer"] = unit.expression_transfer
    if unit.learning_value_score is not None:
        candidate["learning_value_score"] = unit.learning_value_score
    if unit.chunk_text:
        candidate["chunk_text"] = unit.chunk_text
    return candidate


def _materialize_cloze_candidates_from_phrase_items(
    phrase_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    phrase_candidates = [
        candidate
        for item in phrase_items
        if (candidate := _normalize_phrase_candidate_item(item)) is not None
    ]
    units = _build_cloze_units_from_phrases(phrase_candidates)
    return [_cloze_unit_to_candidate(unit) for unit in units]


def _sanitize_cloze_hint_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value)
    return value.replace("(", "（").replace(")", "）")


def _inject_phrase_hints(
    *,
    text: str,
    phrase_to_hint: dict[str, str],
) -> str:
    if not text or not phrase_to_hint:
        return text

    def _replace(match: re.Match[str]) -> str:
        block = match.group(1)
        phrase = re.sub(r"</?b>", "", match.group(2), flags=re.IGNORECASE).strip()
        hint = match.group(3)
        mapped = phrase_to_hint.get(normalize_for_dedupe(phrase), "")
        final_hint = _sanitize_cloze_hint_text(mapped) if mapped else hint
        return f"{block}({final_hint})"

    return _CLOZE_WITH_HINT_RE.sub(_replace, text)


def _build_note(
    *,
    title: str | None,
    source_url: str | None,
    target_phrases: list[str],
    phrase_types: list[str],
    expression_transfer: str | None,
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
    if phrase_types:
        lines.append(f"phrase_types: {' | '.join(phrase_types)}")
    if expression_transfer:
        lines.append(f"transfer: {expression_transfer}")
    return "\n".join(lines)


def _resolve_deck_name(*, cfg: AppConfig, options: BuildDeckOptions, document: DocumentRecord) -> str:
    if options.deck_name:
        return options.deck_name
    source_name = Path(document.source_value).stem.strip()
    if source_name:
        return source_name
    return cfg.default_deck_name


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
    material_profile: str,
    learning_mode: str,
    difficulty: str,
    metrics: dict[str, Any] | None = None,
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
    summary_path = run_dir / "run_summary.json"
    summary_payload: dict[str, Any] = {}
    if summary_path.exists():
        try:
            existing = load_json(summary_path)
            if isinstance(existing, dict):
                summary_payload.update(existing)
        except Exception:
            logger.warning("failed to load existing run summary | path=%s", summary_path)
    summary_payload.update(
        {
            "run_id": document.run_id,
            "cards": len(cards),
            "errors": len(errors),
            "material_profile": material_profile,
            "content_profile": material_profile,  # backward-compatible summary key
            "learning_mode": learning_mode,
            "difficulty": difficulty,
            "output_path": str(output_path),
        }
    )
    if metrics:
        summary_payload["metrics"] = metrics
    dump_json(summary_path, summary_payload)


def _resolve_material_profile(cfg: AppConfig, options: BuildDeckOptions) -> str:
    # Priority: explicit material_profile > legacy content_profile > cfg.material_profile > cfg.content_profile.
    profile = (
        options.material_profile
        or options.content_profile
        or cfg.material_profile
        or cfg.content_profile
        or "prose_article"
    ).strip().lower()
    if profile == "general":
        profile = "prose_article"
    if profile not in SUPPORTED_CONTENT_PROFILES:
        allowed = ", ".join(sorted(SUPPORTED_CONTENT_PROFILES))
        raise build_error(
            error_code="ARG_MATERIAL_PROFILE_INVALID",
            cause="Invalid material profile.",
            detail=f"material_profile={profile!r}",
            next_steps=[f"Use one of: {allowed}"],
            exit_code=ExitCode.ARGUMENT_ERROR,
        )
    return profile


def _resolve_learning_mode(cfg: AppConfig, options: BuildDeckOptions) -> str:
    mode = (options.learning_mode or cfg.learning_mode or "expression_mining").strip().lower()
    if mode not in SUPPORTED_LEARNING_MODES:
        allowed = ", ".join(sorted(SUPPORTED_LEARNING_MODES))
        raise build_error(
            error_code="ARG_LEARNING_MODE_INVALID",
            cause="Unsupported learning mode in current build.",
            detail=f"learning_mode={mode!r}",
            next_steps=[f"Use one of: {allowed}"],
            exit_code=ExitCode.ARGUMENT_ERROR,
        )
    return mode


def _check_textbook_profile_settings(*, cfg: AppConfig, options: BuildDeckOptions, material_profile: str) -> None:
    if material_profile != "textbook_examples":
        return
    if options.cloze_min_chars is None and cfg.cloze_min_chars > TEXTBOOK_PROFILE_MAX_RECOMMENDED_CLOZE_MIN_CHARS:
        raise build_error(
            error_code="TEXTBOOK_PROFILE_MIN_CHARS_TOO_HIGH",
            cause="textbook_examples profile requires shorter minimum cloze length.",
            detail=(
                "CLAWLEARN_CLOZE_MIN_CHARS="
                f"{cfg.cloze_min_chars} exceeds recommended max "
                f"{TEXTBOOK_PROFILE_MAX_RECOMMENDED_CLOZE_MIN_CHARS}"
            ),
            next_steps=[
                "Lower `CLAWLEARN_CLOZE_MIN_CHARS` in .env (recommended 40-80)",
                "Or override this run with `--cloze-min-chars <value>`",
            ],
            exit_code=ExitCode.ARGUMENT_ERROR,
        )


def _translation_retry_delay_seconds(cfg: AppConfig, attempt: int) -> float:
    base = float(cfg.llm_retry_backoff_seconds or 0.0)
    if base <= 0:
        return 0.0
    return base * (2 ** max(0, attempt - 1))


def _is_retryable_translation_batch_error(exc: ClawLearnError) -> bool:
    return exc.error_code in {
        "LLM_REQUEST_FAILED",
        "LLM_RESPONSE_PARSE_FAILED",
        "LLM_RESPONSE_SHAPE_INVALID",
    }


def _is_retryable_translation_item_error(result: TranslationBatchResult) -> bool:
    return bool(result.error and result.error.startswith("retryable:"))


def _materialize_translation_fallback_cards(
    *,
    remaining: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    stage: str,
    reason: str,
    append_card_from_translation: Callable[[int, dict[str, Any], str], None],
    attempt: int | None = None,
) -> None:
    for entry in remaining:
        idx = int(entry["index"])
        item = entry["item"]
        append_card_from_translation(idx, item, "")
        payload: dict[str, Any] = {
            "stage": stage,
            "index": idx,
            "reason": reason,
            "translation_fallback": "empty",
        }
        if attempt is not None:
            payload["attempt"] = attempt
        errors.append(payload)


def _apply_per_chunk_cap(items: list[dict[str, Any]], *, max_per_chunk: int | None) -> list[dict[str, Any]]:
    if not max_per_chunk or max_per_chunk <= 0:
        return list(items)
    limited: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for item in items:
        cid = str(item.get("chunk_id", "")).strip()
        current = counts.get(cid, 0)
        if current >= max_per_chunk:
            continue
        counts[cid] = current + 1
        limited.append(item)
    return limited


def _count_cloze_marks(text: str) -> int:
    return len(CLOZE_MARK_RE.findall(str(text or "")))


def _extract_chunk_index(chunk_id: str | None) -> int | None:
    value = str(chunk_id or "").strip()
    if not value:
        return None
    match = _CHUNK_ID_INDEX_RE.match(value)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _build_pipeline_metrics(
    *,
    chunks: list[Any],
    raw_candidates: list[dict[str, Any]],
    valid_candidates: list[dict[str, Any]],
    ranked_candidates: list[dict[str, Any]],
    deduped_candidates: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    learning_mode: str = "expression_mining",
    format_retry_stats: _FormatRetryStats | None = None,
) -> dict[str, Any]:
    chunk_ids = {str(getattr(chunk, "chunk_id", "")) for chunk in chunks if str(getattr(chunk, "chunk_id", ""))}
    raw_by_chunk: dict[str, int] = {}
    raw_candidates_unattributed = 0
    raw_candidates_unknown_chunk_id = 0
    raw_candidates_missing_chunk_id = 0
    for item in raw_candidates:
        cid = str(item.get("chunk_id", "")).strip()
        if not cid:
            raw_candidates_unattributed += 1
            raw_candidates_missing_chunk_id += 1
            continue
        if cid not in chunk_ids:
            raw_candidates_unattributed += 1
            raw_candidates_unknown_chunk_id += 1
            continue
        raw_by_chunk[cid] = raw_by_chunk.get(cid, 0) + 1
    empty_chunk_count = len([cid for cid in chunk_ids if raw_by_chunk.get(cid, 0) == 0])

    reason_hist: dict[str, int] = {}
    for err in errors:
        if str(err.get("stage", "")) != "validate_text":
            continue
        reason = str(err.get("reason", ""))
        category = classify_rejection_reason(reason)
        reason_hist[category] = reason_hist.get(category, 0) + 1
    top_rejection_categories = [
        {"category": category, "count": count}
        for category, count in sorted(reason_hist.items(), key=lambda kv: kv[1], reverse=True)[:5]
    ]

    model_taxonomy_hist: dict[str, int] = {}
    for item in raw_candidates:
        for ptype in normalize_phrase_types(item.get("phrase_types"), max_items=2):
            model_taxonomy_hist[ptype] = model_taxonomy_hist.get(ptype, 0) + 1

    candidate_phrase_type_hist: dict[str, int] = {}
    candidate_phrase_type_score_sum: dict[str, float] = {}
    candidate_phrase_type_score_count: dict[str, int] = {}
    for item in ranked_candidates:
        score = float(item.get("learning_value_score", 0.0))
        for ptype in item.get("phrase_types", []) or []:
            key = str(ptype).strip()
            if not key:
                continue
            candidate_phrase_type_hist[key] = candidate_phrase_type_hist.get(key, 0) + 1
            candidate_phrase_type_score_sum[key] = candidate_phrase_type_score_sum.get(key, 0.0) + score
            candidate_phrase_type_score_count[key] = candidate_phrase_type_score_count.get(key, 0) + 1

    selected_phrase_type_hist: dict[str, int] = {}
    for item in deduped_candidates:
        for ptype in item.get("phrase_types", []) or []:
            key = str(ptype).strip()
            if not key:
                continue
            selected_phrase_type_hist[key] = selected_phrase_type_hist.get(key, 0) + 1

    phrase_type_avg_score = {
        key: round(candidate_phrase_type_score_sum[key] / candidate_phrase_type_score_count[key], 4)
        for key in candidate_phrase_type_score_sum
        if candidate_phrase_type_score_count.get(key, 0) > 0
    }
    transfer_non_empty_count = len(
        [
            item
            for item in deduped_candidates
            if str(item.get("expression_transfer", "")).strip()
        ]
    )
    transfer_total_by_taxonomy: dict[str, int] = {}
    transfer_non_empty_by_taxonomy: dict[str, int] = {}
    for item in deduped_candidates:
        has_transfer = bool(str(item.get("expression_transfer", "")).strip())
        for ptype in item.get("phrase_types", []) or []:
            key = str(ptype).strip()
            if not key:
                continue
            transfer_total_by_taxonomy[key] = transfer_total_by_taxonomy.get(key, 0) + 1
            if has_transfer:
                transfer_non_empty_by_taxonomy[key] = transfer_non_empty_by_taxonomy.get(key, 0) + 1
    transfer_non_empty_ratio_by_taxonomy = {
        key: round(transfer_non_empty_by_taxonomy.get(key, 0) / total, 4)
        for key, total in transfer_total_by_taxonomy.items()
        if total > 0
    }

    avg_clozes_per_candidate = (
        sum(_count_cloze_marks(str(item.get("text", ""))) for item in deduped_candidates) / len(deduped_candidates)
        if deduped_candidates
        else 0.0
    )
    avg_target_phrases_per_candidate = (
        sum(len(item.get("target_phrases", []) or []) for item in deduped_candidates) / len(deduped_candidates)
        if deduped_candidates
        else 0.0
    )
    selected_by_chunk: dict[str, int] = {}
    selected_candidates_unattributed = 0
    selected_candidates_unknown_chunk_id = 0
    selected_candidates_missing_chunk_id = 0
    for item in deduped_candidates:
        cid = str(item.get("chunk_id", "")).strip()
        if not cid:
            selected_candidates_unattributed += 1
            selected_candidates_missing_chunk_id += 1
            continue
        if cid not in chunk_ids:
            selected_candidates_unattributed += 1
            selected_candidates_unknown_chunk_id += 1
            continue
        selected_by_chunk[cid] = selected_by_chunk.get(cid, 0) + 1
    raw_candidates_attributed = sum(raw_by_chunk.values())
    selected_candidates_attributed = sum(selected_by_chunk.values())
    avg_selected_per_chunk = (selected_candidates_attributed / len(chunks)) if chunks else 0.0

    avg_raw_per_chunk = (raw_candidates_attributed / len(chunks)) if chunks else 0.0
    pass_rate = (len(valid_candidates) / len(raw_candidates)) if raw_candidates else 0.0
    stats = format_retry_stats or _FormatRetryStats()
    return {
        "learning_mode": learning_mode,
        "chunks_total": len(chunks),
        "raw_candidates": len(raw_candidates),
        "raw_candidates_attributed": raw_candidates_attributed,
        "raw_candidates_unattributed": raw_candidates_unattributed,
        "raw_candidates_unknown_chunk_id": raw_candidates_unknown_chunk_id,
        "raw_candidates_missing_chunk_id": raw_candidates_missing_chunk_id,
        "validated_candidates": len(valid_candidates),
        "deduped_candidates": len(deduped_candidates),
        "selected_candidates_attributed": selected_candidates_attributed,
        "selected_candidates_unattributed": selected_candidates_unattributed,
        "selected_candidates_unknown_chunk_id": selected_candidates_unknown_chunk_id,
        "selected_candidates_missing_chunk_id": selected_candidates_missing_chunk_id,
        "avg_raw_candidates_per_chunk": round(avg_raw_per_chunk, 4),
        "avg_selected_candidates_per_chunk": round(avg_selected_per_chunk, 4),
        "selected_candidates_per_chunk": selected_by_chunk,
        "avg_clozes_per_candidate": round(avg_clozes_per_candidate, 4),
        "avg_target_phrases_per_candidate": round(avg_target_phrases_per_candidate, 4),
        "validation_pass_rate": round(pass_rate, 4),
        "empty_chunk_count": empty_chunk_count,
        "empty_chunk_ratio": round((empty_chunk_count / len(chunks)) if chunks else 0.0, 4),
        "rejection_reason_histogram": reason_hist,
        "rejection_reason_top": top_rejection_categories,
        # Backward-compatible key: selected/final histogram.
        "phrase_type_histogram": selected_phrase_type_hist,
        "taxonomy_model_histogram": model_taxonomy_hist,
        "taxonomy_candidate_histogram": candidate_phrase_type_hist,
        "taxonomy_selected_histogram": selected_phrase_type_hist,
        "taxonomy_average_score": phrase_type_avg_score,
        "expression_transfer_non_empty_count": transfer_non_empty_count,
        "expression_transfer_non_empty_ratio": round(
            transfer_non_empty_count / len(deduped_candidates), 4
        )
        if deduped_candidates
        else 0.0,
        "expression_transfer_non_empty_ratio_by_taxonomy": transfer_non_empty_ratio_by_taxonomy,
        "format_retry": {
            "attempt1_fixed_count": stats.attempt1_fixed_count,
            "llm_repair_success_count": stats.llm_repair_success_count,
            "llm_regen_success_count": stats.llm_regen_success_count,
            "total_extra_llm_calls": stats.total_extra_llm_calls,
            "recovered_candidates": stats.recovered_candidates,
        },
    }


def _is_retryable_format_reason(reason: str) -> bool:
    value = str(reason or "").strip().lower()
    if not value:
        return False
    if value.startswith("format:"):
        return any(value.startswith(prefix) for prefix in _FORMAT_RETRYABLE_REASONS_LOWER)
    if value.startswith("taxonomy:"):
        return any(value.startswith(prefix) for prefix in _TAXONOMY_RETRYABLE_REASONS_LOWER)
    return False


def _is_taxonomy_reason(reason: str) -> bool:
    return str(reason or "").strip().lower().startswith("taxonomy:")


def _split_sentences(text: str) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    # Keep this intentionally lightweight; validator remains the source of truth.
    parts = re.split(r"(?<=[.!?])\s+", value)
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return cleaned if cleaned else [value]


def _trim_text_to_sentence_cap(text: str, max_sentences: int) -> str:
    if max_sentences <= 0:
        return str(text or "").strip()
    sentences = _split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences).strip()
    cloze_idx = 0
    for idx, sentence in enumerate(sentences):
        if "{{c" in sentence.lower():
            cloze_idx = idx
            break
    start = max(0, cloze_idx - (max_sentences - 1))
    end = min(len(sentences), start + max_sentences)
    start = max(0, end - max_sentences)
    return " ".join(sentences[start:end]).strip()


def _normalize_cloze_markup(text: str) -> str:
    value = str(text or "")
    if not value:
        return ""
    normalized = value
    normalized = re.sub(r"\{\{cN::", "{{c1::", normalized)
    normalized = re.sub(r"\{\{cN:", "{{c1::", normalized)
    normalized = re.sub(r"\{cN::", "{{c1::", normalized)
    normalized = re.sub(r"\{c(\d+)::", r"{{c\1::", normalized)
    normalized = re.sub(r"\{\{c(\d+)::([^}]+)\}", r"{{c\1::\2}}", normalized)

    def _fmt(match: re.Match[str]) -> str:
        idx = match.group(1)
        body = re.sub(r"</?b>", "", match.group(2), flags=re.IGNORECASE).strip()
        return f"{{{{c{idx}::<b>{body}</b>}}}}(hint)" if body else ""

    normalized = _CLOZE_BLOCK_GENERIC_RE.sub(_fmt, normalized)
    return normalized.strip()


def _extract_cloze_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for match in _CLOZE_BLOCK_GENERIC_RE.finditer(str(text or "")):
        body = re.sub(r"</?b>", "", match.group(2), flags=re.IGNORECASE).strip()
        if body:
            phrases.append(body)
    return phrases


def _normalize_candidate_shape(item: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(item)
    target_phrases = normalized.get("target_phrases")
    if isinstance(target_phrases, list):
        normalized["target_phrases"] = [str(x).strip() for x in target_phrases if str(x).strip()]
    elif isinstance(target_phrases, str):
        normalized["target_phrases"] = [part.strip() for part in target_phrases.split(",") if part.strip()]
    else:
        normalized["target_phrases"] = []

    phrase_types = normalized.get("phrase_types")
    if isinstance(phrase_types, list):
        normalized["phrase_types"] = [str(x).strip() for x in phrase_types if str(x).strip()]
    elif isinstance(phrase_types, str):
        normalized["phrase_types"] = [part.strip() for part in phrase_types.split(",") if part.strip()]
    elif phrase_types is None:
        normalized["phrase_types"] = []
    else:
        value = str(phrase_types).strip()
        normalized["phrase_types"] = [value] if value else []
    return normalized


def _attempt_format_canonicalize(
    *,
    item: dict[str, Any],
    reason: str,
    max_sentences: int,
) -> dict[str, Any]:
    candidate = _normalize_candidate_shape(item)
    reason_value = str(reason or "")

    text = str(candidate.get("text", "")).strip()
    original = str(candidate.get("original", "")).strip()
    target_phrases = list(candidate.get("target_phrases") or [])

    if reason_value.startswith("format:phrase_types must be in taxonomy:") or reason_value.startswith(
        "taxonomy:invalid_labels:"
    ):
        candidate["phrase_types"] = []
    if reason_value.startswith("format:phrase_types has too many labels") or reason_value.startswith(
        "taxonomy:too_many_labels:"
    ):
        phrase_types = candidate.get("phrase_types")
        if isinstance(phrase_types, list):
            candidate["phrase_types"] = phrase_types[:2]
        else:
            candidate["phrase_types"] = []

    if reason_value.startswith("format:target_phrases is empty or invalid"):
        parsed = _extract_cloze_phrases(text)
        if parsed:
            candidate["target_phrases"] = parsed

    if reason_value.startswith("format:missing cloze marker"):
        if "{{cn:" in text.lower() or "{cn::" in text.lower():
            text = _normalize_cloze_markup(text)
        if "{{c" not in text.lower() and original:
            text = original
        candidate["text"] = text

    if reason_value.startswith("format:cloze style must be {{cN::<b>...</b>}}(hint)"):
        candidate["text"] = _normalize_cloze_markup(text)

    match_sent = _FORMAT_RETRY_TEXT_EXCEEDS_RE.match(reason_value)
    if match_sent:
        cap = max(1, int(match_sent.group(1)))
        candidate["text"] = _trim_text_to_sentence_cap(str(candidate.get("text", "")), cap)

    match_min = _FORMAT_RETRY_MIN_CHARS_RE.match(reason_value)
    if match_min:
        min_chars = max(1, int(match_min.group(1)))
        current = str(candidate.get("text", ""))
        if len(current) < min_chars and original:
            candidate["text"] = original

    if "{{c" in str(candidate.get("text", "")).lower():
        candidate["text"] = _normalize_cloze_markup(str(candidate.get("text", "")))

    # Keep text compact if canonicalization expanded context too much.
    candidate["text"] = _trim_text_to_sentence_cap(str(candidate.get("text", "")), max_sentences)

    if not candidate.get("target_phrases"):
        parsed = _extract_cloze_phrases(str(candidate.get("text", "")))
        if parsed:
            candidate["target_phrases"] = parsed

    return candidate


def _build_format_retry_messages(
    *,
    mode: str,
    item: dict[str, Any],
    reason: str,
    cfg: AppConfig,
    material_profile: str,
    learning_mode: str,
) -> list[dict[str, str]]:
    taxonomy = ", ".join(normalize_phrase_types(list(item.get("phrase_types") or []), max_items=8) or [])
    allowed_taxonomy = (
        "metaphor_imagery, stance_positioning, concession_contrast, discourse_organizer, "
        "abstraction_bridge, reusable_high_frequency_chunk, phrasal_verb, strong_collocation"
    )
    contract = (
        "Return ONLY one JSON object. "
        "Required keys: text, original, target_phrases, note_hint, phrase_types, expression_transfer. "
        "text must include at least one cloze with style {{cN::<b>...</b>}}(hint). "
        f"text must be <= {cfg.cloze_max_sentences} sentences. "
        f"phrase_types must be 0-2 labels from: {allowed_taxonomy}. "
        "target_phrases must be a non-empty JSON array of strings."
    )
    if mode == "regen":
        task = "Regenerate a compliant candidate from original/chunk context."
    else:
        task = "Repair this candidate with minimal semantic change."
    payload = {
        "reason": reason,
        "material_profile": material_profile,
        "learning_mode": learning_mode,
        "difficulty": cfg.cloze_difficulty,
        "candidate": item,
        "candidate_phrase_types_normalized": taxonomy,
    }
    return [
        {
            "role": "system",
            "content": "You are a strict JSON repair assistant for language-learning cloze cards.",
        },
        {
            "role": "user",
            "content": f"{task}\n{contract}\nInput JSON:\n{json.dumps(payload, ensure_ascii=False)}",
        },
    ]


def _attempt_llm_format_retry(
    *,
    client: OpenAICompatibleClient,
    item: dict[str, Any],
    reason: str,
    cfg: AppConfig,
    material_profile: str,
    learning_mode: str,
    mode: str,
) -> dict[str, Any] | None:
    messages = _build_format_retry_messages(
        mode=mode,
        item=item,
        reason=reason,
        cfg=cfg,
        material_profile=material_profile,
        learning_mode=learning_mode,
    )
    content = client.chat(messages, temperature=0.0, max_retries=1)
    data = parse_json_content(content, expect_array=False)
    if isinstance(data, list):
        if not data:
            return None
        data = data[0]
    if not isinstance(data, dict):
        return None
    repaired = _normalize_candidate_shape(data)

    # Preserve pipeline identity/context fields from the source candidate.
    for key in ("chunk_id", "chunk_text"):
        if key in item and key not in repaired:
            repaired[key] = item[key]
    if not str(repaired.get("original", "")).strip():
        repaired["original"] = str(item.get("original", "")).strip()
    if not str(repaired.get("text", "")).strip():
        repaired["text"] = str(item.get("text", "")).strip()
    return repaired


def _collect_valid_candidates(
    *,
    raw_candidates: list[dict[str, Any]],
    cfg: AppConfig,
    material_profile: str,
    learning_mode: str,
    errors: list[dict[str, Any]],
    retry_client: OpenAICompatibleClient | None = None,
    taxonomy_repair_enable: bool = False,
) -> tuple[
    list[dict[str, Any]],
    int,
    str | None,
    str | None,
    _FormatRetryStats,
    list[dict[str, Any]],
    int,
]:
    valid_candidates: list[dict[str, Any]] = []
    validation_rejects = 0
    first_validation_reason: str | None = None
    first_final_reject_reason: str | None = None
    retry_stats = _FormatRetryStats()
    taxonomy_repair_queue: list[dict[str, Any]] = []
    taxonomy_reject_count = 0

    for item in raw_candidates:
        current_item = item
        ok, reason = validate_text_candidate(
            current_item,
            max_sentences=cfg.cloze_max_sentences,
            min_chars=cfg.cloze_min_chars,
            difficulty=cfg.cloze_difficulty,
            material_profile=material_profile,
            learning_mode=learning_mode,
        )
        if ok:
            valid_candidates.append(current_item)
            continue

        if first_validation_reason is None:
            first_validation_reason = reason
        last_reason = reason
        retry_recovered = False
        retry_candidate = current_item
        retry_max = max(0, int(cfg.validate_format_retry_max or 0))

        if cfg.validate_format_retry_enable and _is_retryable_format_reason(reason) and retry_max > 0:
            for attempt in range(1, retry_max + 1):
                retry_action = "canonicalize"
                try:
                    if attempt == 1:
                        retry_candidate = _attempt_format_canonicalize(
                            item=retry_candidate,
                            reason=last_reason,
                            max_sentences=cfg.cloze_max_sentences,
                        )
                    elif attempt == 2 and cfg.validate_format_retry_llm_enable:
                        retry_action = "llm_repair"
                        retry_stats.total_extra_llm_calls += 1
                        retry_candidate = _attempt_llm_format_retry(
                            client=retry_client or OpenAICompatibleClient(cfg),
                            item=retry_candidate,
                            reason=last_reason,
                            cfg=cfg,
                            material_profile=material_profile,
                            learning_mode=learning_mode,
                            mode="repair",
                        ) or retry_candidate
                    elif attempt >= 3 and cfg.validate_format_retry_llm_enable:
                        retry_action = "llm_regen"
                        retry_stats.total_extra_llm_calls += 1
                        retry_candidate = _attempt_llm_format_retry(
                            client=retry_client or OpenAICompatibleClient(cfg),
                            item=retry_candidate,
                            reason=last_reason,
                            cfg=cfg,
                            material_profile=material_profile,
                            learning_mode=learning_mode,
                            mode="regen",
                        ) or retry_candidate
                    else:
                        break
                except ClawLearnError as exc:
                    errors.append(
                        {
                            "stage": "validate_text_retry_attempt",
                            "attempt": attempt,
                            "retry_action": retry_action,
                            "original_reason": reason,
                            "error": exc.to_lines(),
                            "item": copy.deepcopy(retry_candidate),
                        }
                    )
                    continue

                ok_retry, retry_reason = validate_text_candidate(
                    retry_candidate,
                    max_sentences=cfg.cloze_max_sentences,
                    min_chars=cfg.cloze_min_chars,
                    difficulty=cfg.cloze_difficulty,
                    material_profile=material_profile,
                    learning_mode=learning_mode,
                )
                if ok_retry:
                    if attempt == 1:
                        retry_stats.attempt1_fixed_count += 1
                    elif retry_action == "llm_repair":
                        retry_stats.llm_repair_success_count += 1
                    else:
                        retry_stats.llm_regen_success_count += 1
                    errors.append(
                        {
                            "stage": "validate_text_retry_recovered",
                            "attempt": attempt,
                            "retry_action": retry_action,
                            "original_reason": reason,
                            "item": copy.deepcopy(retry_candidate),
                        }
                    )
                    valid_candidates.append(retry_candidate)
                    retry_recovered = True
                    break
                last_reason = retry_reason
                errors.append(
                    {
                        "stage": "validate_text_retry_attempt",
                        "attempt": attempt,
                        "retry_action": retry_action,
                        "original_reason": reason,
                        "reason": retry_reason,
                        "item": copy.deepcopy(retry_candidate),
                    }
                )

        if retry_recovered:
            continue
        is_taxonomy_reject = _is_taxonomy_reason(last_reason)
        if is_taxonomy_reject:
            taxonomy_reject_count += 1
        if taxonomy_repair_enable and is_taxonomy_reject:
            taxonomy_repair_queue.append(
                {
                    "item": copy.deepcopy(retry_candidate),
                    "reason": last_reason,
                    "original_reason": reason,
                }
            )
            errors.append(
                {
                    "stage": "taxonomy_repair_queued",
                    "reason": last_reason,
                    "original_reason": reason,
                    "item": copy.deepcopy(retry_candidate),
                }
            )
            continue
        validation_rejects += 1
        if first_final_reject_reason is None:
            first_final_reject_reason = last_reason
        errors.append(
            {
                "stage": "validate_text",
                "reason": last_reason,
                "original_reason": reason,
                "item": copy.deepcopy(retry_candidate),
            }
        )
    return (
        valid_candidates,
        validation_rejects,
        first_validation_reason,
        first_final_reject_reason,
        retry_stats,
        taxonomy_repair_queue,
        taxonomy_reject_count,
    )


def _collect_fallback_raw_candidates(
    *,
    client: OpenAICompatibleClient,
    prompt: Any,
    document: DocumentRecord,
    chunks: list[Any],
    temperature: float | None,
    errors: list[dict[str, Any]],
    generate_fn: Callable[..., list[dict[str, Any]]] = generate_cloze_candidates_for_chunk,
) -> list[dict[str, Any]]:
    """Generate fallback raw candidates chunk-by-chunk.

    This helper keeps fallback accumulation explicit and isolated so candidates
    are appended exactly once per generated item.
    """

    fallback_raw: list[dict[str, Any]] = []
    for chunk in chunks:
        try:
            items = generate_fn(
                client=client,
                prompt=prompt,
                document=document,
                chunk=chunk,
                temperature=temperature,
            )
        except ClawLearnError as exc:
            errors.append(
                {
                    "stage": "cloze_fallback_single_chunk_error",
                    "chunk_id": chunk.chunk_id,
                    "error": exc.to_lines(),
                }
            )
            continue

        for item in items:
            item["chunk_id"] = chunk.chunk_id
            item["chunk_text"] = chunk.source_text
            fallback_raw.append(item)
    return fallback_raw


def _repair_taxonomy_candidates(
    *,
    queued_candidates: list[dict[str, Any]],
    cfg: AppConfig,
    material_profile: str,
    learning_mode: str,
    client: OpenAICompatibleClient,
    errors: list[dict[str, Any]],
    temperature: float | None,
    classify_fn: Callable[..., list[list[str]]] = classify_phrase_types_batch,
) -> tuple[list[dict[str, Any]], _TaxonomyRepairStats]:
    stats = _TaxonomyRepairStats(
        taxonomy_reject_count=len(queued_candidates),
        taxonomy_repair_attempted=len(queued_candidates),
    )
    if not cfg.taxonomy_repair_enable or not queued_candidates:
        return [], stats

    repaired_valid: list[dict[str, Any]] = []
    batch_size = max(1, int(cfg.translate_batch_size or 1))

    def _record_batch_failure(batch: list[dict[str, Any]], reason: str, *, stage: str, attempt: int) -> None:
        for entry in batch:
            stats.taxonomy_repair_failed += 1
            errors.append(
                {
                    "stage": stage,
                    "attempt": attempt,
                    "reason": reason,
                    "original_reason": entry.get("reason", ""),
                    "item": copy.deepcopy(entry.get("item", {})),
                }
            )

    for batch in iter_batches(queued_candidates, batch_size):
        remaining = list(batch)
        attempt = 0
        while remaining:
            attempt += 1
            classifier_input = [
                {
                    "text_cloze": str(entry.get("item", {}).get("text", "")).strip(),
                    "text_original": str(entry.get("item", {}).get("original", "")).strip(),
                    "target_phrases": list(entry.get("item", {}).get("target_phrases") or []),
                    "difficulty": cfg.cloze_difficulty,
                    "learning_mode": learning_mode,
                }
                for entry in remaining
            ]
            try:
                repaired_phrase_types = classify_fn(
                    client=client,
                    items=classifier_input,
                    temperature=temperature,
                )
                if len(repaired_phrase_types) != len(remaining):
                    raise build_error(
                        error_code="LLM_RESPONSE_SHAPE_INVALID",
                        cause="Taxonomy repair response length mismatch.",
                        detail=f"expected={len(remaining)}, got={len(repaired_phrase_types)}",
                        next_steps=["Fix taxonomy classifier output contract"],
                        exit_code=ExitCode.LLM_PARSE_ERROR,
                    )
            except ClawLearnError as exc:
                if _is_retryable_translation_batch_error(exc) and attempt < TRANSLATION_BATCH_MAX_RETRIES:
                    errors.append(
                        {
                            "stage": "taxonomy_repair_batch_retry",
                            "attempt": attempt,
                            "remaining": len(remaining),
                            "error": exc.to_lines(),
                        }
                    )
                    delay = _translation_retry_delay_seconds(cfg, attempt)
                    if delay > 0:
                        time.sleep(delay)
                    continue
                _record_batch_failure(
                    remaining,
                    "taxonomy repair batch failed",
                    stage="taxonomy_repair_failed",
                    attempt=attempt,
                )
                errors.append(
                    {
                        "stage": "taxonomy_repair_batch_error",
                        "attempt": attempt,
                        "remaining": len(remaining),
                        "error": exc.to_lines(),
                    }
                )
                break

            for entry, labels in zip(remaining, repaired_phrase_types, strict=False):
                candidate = copy.deepcopy(entry.get("item", {}))
                candidate["phrase_types"] = normalize_phrase_types(labels, max_items=2)
                ok, reason = validate_text_candidate(
                    candidate,
                    max_sentences=cfg.cloze_max_sentences,
                    min_chars=cfg.cloze_min_chars,
                    difficulty=cfg.cloze_difficulty,
                    material_profile=material_profile,
                    learning_mode=learning_mode,
                )
                if ok:
                    stats.taxonomy_repair_success += 1
                    repaired_valid.append(candidate)
                    errors.append(
                        {
                            "stage": "taxonomy_repair_recovered",
                            "original_reason": entry.get("reason", ""),
                            "item": copy.deepcopy(candidate),
                        }
                    )
                    continue
                stats.taxonomy_repair_failed += 1
                errors.append(
                    {
                        "stage": "taxonomy_repair_failed",
                        "reason": reason,
                        "original_reason": entry.get("reason", ""),
                        "item": copy.deepcopy(candidate),
                    }
                )
            break

    return repaired_valid, stats


def _build_document(cfg: AppConfig, run_id: str, options: BuildDeckOptions) -> DocumentRecord:
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

    source_url = None
    cleaned_markdown = None
    options.input_value = str(file_path)

    material_profile = _resolve_material_profile(cfg, options)
    cleaned = normalize_text(
        raw_text,
        options=NormalizeOptions(
            short_line_max_words=cfg.ingest_short_line_max_words,
            material_profile=material_profile,
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
        metadata={
            "material_profile": material_profile,
            "learning_mode": options.learning_mode or cfg.learning_mode,
        },
    )


def run_build_lingua_deck(cfg: AppConfig, options: BuildDeckOptions) -> BuildDeckResult:
    validate_base_config(cfg)
    validate_runtime_config(cfg)

    # CLI 的 --difficulty 优先级高于 env；如提供则覆盖 cfg.cloze_difficulty
    if options.cloze_difficulty:
        cfg.cloze_difficulty = options.cloze_difficulty
    if options.cloze_min_chars is not None:
        cfg.cloze_min_chars = options.cloze_min_chars
    material_profile = _resolve_material_profile(cfg, options)
    learning_mode = _resolve_learning_mode(cfg, options)
    options.learning_mode = learning_mode
    _check_textbook_profile_settings(cfg=cfg, options=options, material_profile=material_profile)

    run_ctx = create_run_context(cfg, name="build_deck", run_id=options.run_id)
    template = load_anki_template(cfg.resolve_path(cfg.anki_template))
    extract_prompt_path = (
        options.extract_prompt
        if options.extract_prompt is not None
        else cfg.resolve_extract_prompt_path(
            material_profile=material_profile,
            difficulty=cfg.cloze_difficulty,
            learning_mode=learning_mode,
        )
    )
    explain_prompt_path = (
        options.explain_prompt
        if options.explain_prompt is not None
        else cfg.resolve_explain_prompt_path(
            material_profile=material_profile,
            difficulty=cfg.cloze_difficulty,
            learning_mode=learning_mode,
        )
    )
    extract_prompt = load_prompt(cfg.resolve_path(extract_prompt_path), prompt_lang=cfg.prompt_lang)
    explain_prompt = load_prompt(cfg.resolve_path(explain_prompt_path), prompt_lang=cfg.prompt_lang)
    use_phrase_pipeline = _use_phrase_extraction_pipeline(
        learning_mode=learning_mode,
        schema_name=extract_prompt.output_format.schema_name,
    )
    logger.info(
        "prompt selection resolved | extraction=%s explanation=%s material_profile=%s learning_mode=%s difficulty=%s",
        cfg.resolve_path(extract_prompt_path),
        cfg.resolve_path(explain_prompt_path),
        material_profile,
        learning_mode,
        cfg.cloze_difficulty,
    )
    logger.info(
        "extraction pipeline selected | mode=%s schema=%s phrase_pipeline=%s",
        learning_mode,
        extract_prompt.output_format.schema_name,
        use_phrase_pipeline,
    )

    document = _build_document(cfg, run_ctx.run_id, options)
    logger.info('ingest complete | title="%s"', document.title or "")

    save_intermediate = cfg.save_intermediate if options.save_intermediate is None else options.save_intermediate

    output_path = resolve_output_path(
        workspace_root=cfg.workspace_root,
        export_dir=cfg.export_dir,
        run_id=document.run_id,
        explicit_output=options.output,
    )

    errors: list[dict] = []
    raw_extraction_candidates: list[dict[str, Any]] = []
    raw_candidates: list[dict] = []
    valid_candidates: list[dict] = []
    deduped: list[dict] = []
    ranked_candidates: list[dict[str, Any]] = []
    cards: list[CardRecord] = []
    chunks: list = []
    format_retry_stats = _FormatRetryStats()
    taxonomy_repair_stats = _TaxonomyRepairStats()

    def _save_failure_snapshot(exc: ClawLearnError) -> None:
        if not save_intermediate:
            return
        validated = deduped if deduped else valid_candidates
        snapshot_errors = [*errors, {"stage": "fatal", "error": exc.to_lines()}]
        snapshot_raw = raw_candidates if raw_candidates else raw_extraction_candidates
        try:
            _save_intermediate(
                run_dir=run_ctx.run_dir,
                document=document,
                chunks=chunks,
                raw_candidates=snapshot_raw,
                valid_candidates=validated,
                cards=cards,
                errors=snapshot_errors,
                output_path=output_path,
                material_profile=material_profile,
                learning_mode=learning_mode,
                difficulty=cfg.cloze_difficulty,
                metrics=None,
            )
        except Exception:
            logger.exception("failed to persist failure snapshot")

    chunks = chunk_document(
        run_id=run_ctx.run_id,
        text=document.cleaned_text,
        max_chars=options.max_chars or cfg.chunk_max_chars,
        min_chars=cfg.chunk_min_chars,
        overlap_sentences=cfg.chunk_overlap_sentences,
        material_profile=material_profile,
        difficulty=cfg.cloze_difficulty,
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

    def _append_raw_items(batch: list[Any], items: list[dict[str, Any]]) -> None:
        chunk_map = {c.chunk_id: c for c in batch}
        chunk_index_map: dict[int, Any | None] = {}
        for chunk in batch:
            idx = _extract_chunk_index(chunk.chunk_id)
            if idx is None:
                continue
            if idx in chunk_index_map:
                chunk_index_map[idx] = None
            else:
                chunk_index_map[idx] = chunk
        for item in items:
            cid = str(item.get("chunk_id") or "").strip()
            chunk = chunk_map.get(cid) if cid else None
            if chunk is None and cid:
                idx = _extract_chunk_index(cid)
                if idx is not None:
                    mapped = chunk_index_map.get(idx)
                    if mapped is not None:
                        chunk = mapped
            if chunk is None and len(batch) == 1:
                chunk = batch[0]
            if chunk is None:
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
            item["chunk_id"] = chunk.chunk_id
            item["chunk_text"] = chunk.source_text
            raw_extraction_candidates.append(item)

    for batch in iter_batches(chunks, batch_size):
        try:
            if len(batch) == 1:
                chunk = batch[0]
                if use_phrase_pipeline:
                    items = generate_phrase_candidates_for_chunk(
                        client=client,
                        prompt=extract_prompt,
                        document=document,
                        chunk=chunk,
                        temperature=options.temperature,
                    )
                else:
                    items = generate_cloze_candidates_for_chunk(
                        client=client,
                        prompt=extract_prompt,
                        document=document,
                        chunk=chunk,
                        temperature=options.temperature,
                    )
            else:
                if use_phrase_pipeline:
                    items = generate_phrase_candidates_for_batch(
                        client=client,
                        prompt=extract_prompt,
                        document=document,
                        chunks=batch,
                        temperature=options.temperature,
                    )
                else:
                    items = generate_cloze_candidates_for_batch(
                        client=client,
                        prompt=extract_prompt,
                        document=document,
                        chunks=batch,
                        temperature=options.temperature,
                    )
            _append_raw_items(batch, items)
        except ClawLearnError as exc:
            if not options.continue_on_error:
                _save_failure_snapshot(exc)
                raise
            errors.append(
                {
                    "stage": "cloze",
                    "chunk_ids": [c.chunk_id for c in batch],
                    "error": exc.to_lines(),
                }
            )

    if use_phrase_pipeline:
        raw_candidates = _materialize_cloze_candidates_from_phrase_items(raw_extraction_candidates)
        if raw_extraction_candidates and not raw_candidates:
            errors.append(
                {
                    "stage": "phrase_to_cloze",
                    "reason": "phrase candidates could not be converted into cloze units",
                    "raw_phrase_candidates": len(raw_extraction_candidates),
                }
            )
    else:
        raw_candidates = list(raw_extraction_candidates)

    (
        valid_candidates,
        validation_rejects,
        first_validation_reason,
        first_final_reject_reason,
        retry_stats_initial,
        taxonomy_repair_queue,
        taxonomy_reject_count_initial,
    ) = _collect_valid_candidates(
        raw_candidates=raw_candidates,
        cfg=cfg,
        material_profile=material_profile,
        learning_mode=learning_mode,
        errors=errors,
        retry_client=client,
        taxonomy_repair_enable=cfg.taxonomy_repair_enable,
    )
    format_retry_stats.absorb(retry_stats_initial)
    if cfg.taxonomy_repair_enable and taxonomy_repair_queue:
        repaired_valid, repair_stats = _repair_taxonomy_candidates(
            queued_candidates=taxonomy_repair_queue,
            cfg=cfg,
            material_profile=material_profile,
            learning_mode=learning_mode,
            client=translate_client,
            errors=errors,
            temperature=options.temperature,
        )
        taxonomy_repair_stats.absorb(repair_stats)
        valid_candidates.extend(repaired_valid)
    else:
        taxonomy_repair_stats.taxonomy_reject_count += taxonomy_reject_count_initial

    if validation_rejects:
        logger.warning(
            "validation filtered candidates | rejected=%d raw=%d first_final_reject_reason=%s",
            validation_rejects,
            len(raw_candidates),
            first_final_reject_reason or "",
        )

    if not valid_candidates:
        errors.append(
            {
                "stage": "cloze_fallback_single_chunk",
                "reason": "no valid candidates after first pass",
            }
        )
        fallback_generate_fn: Callable[..., list[dict[str, Any]]] = (
            generate_phrase_candidates_for_chunk if use_phrase_pipeline else generate_cloze_candidates_for_chunk
        )
        fallback_raw_extraction = _collect_fallback_raw_candidates(
            client=client,
            prompt=extract_prompt,
            document=document,
            chunks=chunks,
            temperature=0.1 if options.temperature is None else options.temperature,
            errors=errors,
            generate_fn=fallback_generate_fn,
        )
        fallback_raw = (
            _materialize_cloze_candidates_from_phrase_items(fallback_raw_extraction)
            if use_phrase_pipeline
            else fallback_raw_extraction
        )
        if use_phrase_pipeline and fallback_raw_extraction and not fallback_raw:
            errors.append(
                {
                    "stage": "phrase_to_cloze_fallback",
                    "reason": "fallback phrase candidates could not be converted into cloze units",
                    "raw_phrase_candidates": len(fallback_raw_extraction),
                }
            )

        raw_candidates.extend(fallback_raw)
        (
            valid_candidates,
            _,
            first_validation_reason,
            first_final_reject_reason,
            retry_stats_fallback,
            taxonomy_repair_queue_fallback,
            taxonomy_reject_count_fallback,
        ) = _collect_valid_candidates(
            raw_candidates=fallback_raw,
            cfg=cfg,
            material_profile=material_profile,
            learning_mode=learning_mode,
            errors=errors,
            retry_client=client,
            taxonomy_repair_enable=cfg.taxonomy_repair_enable,
        )
        format_retry_stats.absorb(retry_stats_fallback)
        if cfg.taxonomy_repair_enable and taxonomy_repair_queue_fallback:
            repaired_valid_fallback, repair_stats_fallback = _repair_taxonomy_candidates(
                queued_candidates=taxonomy_repair_queue_fallback,
                cfg=cfg,
                material_profile=material_profile,
                learning_mode=learning_mode,
                client=translate_client,
                errors=errors,
                temperature=options.temperature,
            )
            taxonomy_repair_stats.absorb(repair_stats_fallback)
            valid_candidates.extend(repaired_valid_fallback)
        else:
            taxonomy_repair_stats.taxonomy_reject_count += taxonomy_reject_count_fallback

    if not valid_candidates and not (cfg.allow_empty_deck or options.continue_on_error):
        err = build_error(
            error_code="CARD_VALIDATION_FAILED",
            cause="All candidates failed validation.",
            detail=f"raw={len(raw_candidates)}, first_final_reject_reason={first_final_reject_reason or 'unknown'}",
            next_steps=[
                "Adjust prompt constraints",
                "Lower CLAWLEARN_CLOZE_MIN_CHARS if needed",
                "Try `--difficulty intermediate` for difficult inputs",
            ],
            exit_code=ExitCode.CARD_VALIDATION_ERROR,
        )
        _save_failure_snapshot(err)
        raise err

    if valid_candidates:
        ranked_candidates = rank_candidates(
            valid_candidates,
            difficulty=cfg.cloze_difficulty,
            material_profile=material_profile,
            learning_mode=learning_mode,
        )
        deduped = dedupe_candidates(ranked_candidates)
        deduped = _apply_per_chunk_cap(deduped, max_per_chunk=cfg.cloze_max_per_chunk)
    else:
        deduped = []
        errors.append(
            {
                "stage": "cloze_empty_result",
                "reason": "no valid candidates after fallback",
            }
        )

    if options.max_notes and options.max_notes > 0:
        deduped = deduped[: options.max_notes]
    logger.info(
        "text generation complete | raw=%d valid=%d deduped=%d",
        len(raw_candidates),
        len(valid_candidates),
        len(deduped),
    )

    translate_batch_size = max(1, int(cfg.translate_batch_size or 1))
    phrase_hint_map: dict[str, str] = {}
    phrase_translation_total = 0

    if use_phrase_pipeline and deduped:
        ordered_unique_phrases: list[str] = []
        seen_phrase_keys: set[str] = set()
        for item in deduped:
            for phrase in item.get("target_phrases", []) or []:
                value = str(phrase).strip()
                key = normalize_for_dedupe(value)
                if not value or not key or key in seen_phrase_keys:
                    continue
                seen_phrase_keys.add(key)
                ordered_unique_phrases.append(value)

        phrase_translation_total = len(ordered_unique_phrases)
        for batch in iter_batches(ordered_unique_phrases, translate_batch_size):
            remaining = list(batch)
            attempt = 0
            while remaining:
                attempt += 1
                try:
                    batch_results = generate_phrase_translations_batch(
                        client=translate_client,
                        prompt=explain_prompt,
                        document=document,
                        phrases=remaining,
                        temperature=options.temperature,
                    )
                except ClawLearnError as exc:
                    if _is_retryable_translation_batch_error(exc) and attempt < TRANSLATION_BATCH_MAX_RETRIES:
                        errors.append(
                            {
                                "stage": "phrase_translation_batch_retry",
                                "attempt": attempt,
                                "remaining": len(remaining),
                                "error": exc.to_lines(),
                            }
                        )
                        delay = _translation_retry_delay_seconds(cfg, attempt)
                        if delay > 0:
                            time.sleep(delay)
                        continue

                    errors.append(
                        {
                            "stage": "phrase_translation_batch_error",
                            "attempt": attempt,
                            "remaining": len(remaining),
                            "error": exc.to_lines(),
                        }
                    )
                    break

                expected_count = len(remaining)
                returned_count = len(batch_results)
                if returned_count != expected_count:
                    errors.append(
                        {
                            "stage": "phrase_translation_batch_incomplete_response",
                            "attempt": attempt,
                            "expected": expected_count,
                            "returned": returned_count,
                        }
                    )

                retry_remaining: list[str] = []
                matched_count = min(expected_count, returned_count)
                for pos in range(matched_count):
                    phrase = remaining[pos]
                    result = batch_results[pos]
                    if result.ok:
                        hint = str(result.translation or "").strip()
                        ok, reason = validate_translation_text(hint)
                        if ok and hint:
                            phrase_hint_map[normalize_for_dedupe(phrase)] = _sanitize_cloze_hint_text(hint)
                        else:
                            errors.append(
                                {
                                    "stage": "phrase_translation_validation",
                                    "phrase": phrase,
                                    "reason": reason or "empty_phrase_translation",
                                }
                            )
                        continue
                    if _is_retryable_translation_item_error(result):
                        retry_remaining.append(phrase)
                        continue
                    errors.append(
                        {
                            "stage": "phrase_translation_item_error",
                            "phrase": phrase,
                            "reason": result.error or "phrase translation failed",
                        }
                    )

                if returned_count < expected_count:
                    retry_remaining.extend(remaining[returned_count:])
                elif returned_count > expected_count:
                    errors.append(
                        {
                            "stage": "phrase_translation_batch_extra_items",
                            "attempt": attempt,
                            "expected": expected_count,
                            "returned": returned_count,
                        }
                    )

                if not retry_remaining:
                    break
                if attempt < TRANSLATION_BATCH_MAX_RETRIES:
                    errors.append(
                        {
                            "stage": "phrase_translation_batch_retry_remaining",
                            "attempt": attempt,
                            "remaining": len(retry_remaining),
                        }
                    )
                    delay = _translation_retry_delay_seconds(cfg, attempt)
                    if delay > 0:
                        time.sleep(delay)
                    remaining = retry_remaining
                    continue

                errors.append(
                    {
                        "stage": "phrase_translation_batch_retries_exhausted",
                        "attempt": attempt,
                        "remaining": len(retry_remaining),
                    }
                )
                break

        for item in deduped:
            item["text"] = _inject_phrase_hints(
                text=str(item.get("text", "")),
                phrase_to_hint=phrase_hint_map,
            )

    def _append_card_from_translation(idx: int, item: dict[str, Any], translation: str) -> None:
        card_id = f"card_{idx:06d}_{stable_hash(str(item['original']), length=6)}"
        target_phrases = list(item.get("target_phrases") or [])
        expression_transfer = str(item.get("expression_transfer", "")).strip()
        if expression_transfer and normalize_for_dedupe(expression_transfer) == normalize_for_dedupe(translation):
            expression_transfer = ""
        note = _build_note(
            title=document.title,
            source_url=document.source_url,
            target_phrases=[str(x) for x in target_phrases],
            phrase_types=[str(x) for x in (item.get("phrase_types") or []) if str(x).strip()],
            expression_transfer=expression_transfer or None,
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
                phrase_types=[str(x) for x in (item.get("phrase_types") or []) if str(x).strip()],
                expression_transfer=expression_transfer,
            )
        )

    pending_translations: list[dict[str, Any]] = [
        {"index": idx, "item": item}
        for idx, item in enumerate(deduped, start=1)
    ]

    for batch in iter_batches(pending_translations, translate_batch_size):
        remaining = list(batch)
        attempt = 0

        while remaining:
            attempt += 1
            originals = [str(entry["item"].get("original", "")) for entry in remaining]
            try:
                batch_results = generate_translation_batch(
                    client=translate_client,
                    prompt=explain_prompt,
                    document=document,
                    chunk_text="",
                    text_originals=originals,
                    temperature=options.temperature,
                )
            except ClawLearnError as exc:
                if _is_retryable_translation_batch_error(exc) and attempt < TRANSLATION_BATCH_MAX_RETRIES:
                    errors.append(
                        {
                            "stage": "translation_batch_retry",
                            "attempt": attempt,
                            "remaining": len(remaining),
                            "error": exc.to_lines(),
                        }
                    )
                    delay = _translation_retry_delay_seconds(cfg, attempt)
                    if delay > 0:
                        time.sleep(delay)
                    continue

                if _is_retryable_translation_batch_error(exc):
                    exhausted_msg = f"translation batch retries exhausted after {TRANSLATION_BATCH_MAX_RETRIES} attempts"
                    _materialize_translation_fallback_cards(
                        remaining=remaining,
                        errors=errors,
                        stage="translation_batch_retries_exhausted",
                        reason=exhausted_msg,
                        append_card_from_translation=_append_card_from_translation,
                        attempt=attempt,
                    )
                    break

                errors.append(
                    {
                        "stage": "translation_batch_error",
                        "attempt": attempt,
                        "remaining": len(remaining),
                        "error": exc.to_lines(),
                        "translation_fallback": "empty",
                    }
                )
                _materialize_translation_fallback_cards(
                    remaining=remaining,
                    errors=errors,
                    stage="translation_item_fallback",
                    reason="translation batch failed; fallback to empty translation",
                    append_card_from_translation=_append_card_from_translation,
                    attempt=attempt,
                )
                break

            expected_count = len(remaining)
            returned_count = len(batch_results)
            if returned_count != expected_count:
                errors.append(
                    {
                        "stage": "translation_batch_incomplete_response",
                        "attempt": attempt,
                        "expected": expected_count,
                        "returned": returned_count,
                    }
                )

            retry_remaining: list[dict[str, Any]] = []
            matched_count = min(expected_count, returned_count)
            for pos in range(matched_count):
                entry = remaining[pos]
                idx = int(entry["index"])
                item = entry["item"]
                result = batch_results[pos]

                if result.ok:
                    translation = str(result.translation or "").strip()
                    ok, reason = validate_translation_text(translation)
                    if not ok:
                        errors.append(
                            {
                                "stage": "translation_validation_fallback",
                                "index": idx,
                                "reason": reason,
                                "original": str(item.get("original", "")),
                                "translation_fallback": "empty",
                            }
                        )
                        _append_card_from_translation(idx, item, "")
                        continue

                    _append_card_from_translation(idx, item, translation)
                    continue

                if _is_retryable_translation_item_error(result):
                    retry_remaining.append(entry)
                    continue

                item_error = result.error or "translation item failed"
                errors.append(
                    {
                        "stage": "translation_item_fallback",
                        "index": idx,
                        "reason": item_error,
                        "original": str(item.get("original", "")),
                        "translation_fallback": "empty",
                    }
                )
                _append_card_from_translation(idx, item, "")

            if returned_count < expected_count:
                retry_remaining.extend(remaining[returned_count:])
            elif returned_count > expected_count:
                errors.append(
                    {
                        "stage": "translation_batch_extra_items",
                        "attempt": attempt,
                        "expected": expected_count,
                        "returned": returned_count,
                    }
                )

            if not retry_remaining:
                break

            if attempt < TRANSLATION_BATCH_MAX_RETRIES:
                errors.append(
                    {
                        "stage": "translation_batch_retry_remaining",
                        "attempt": attempt,
                        "remaining": len(retry_remaining),
                    }
                )
                delay = _translation_retry_delay_seconds(cfg, attempt)
                if delay > 0:
                    time.sleep(delay)
                remaining = retry_remaining
                continue

            exhausted_msg = f"translation batch retries exhausted after {TRANSLATION_BATCH_MAX_RETRIES} attempts"
            _materialize_translation_fallback_cards(
                remaining=retry_remaining,
                errors=errors,
                stage="translation_batch_retries_exhausted",
                reason=exhausted_msg,
                append_card_from_translation=_append_card_from_translation,
                attempt=attempt,
            )
            break

    if not cards and not cfg.allow_empty_deck:
        err = build_error(
            error_code="CARD_EMPTY",
            cause="No exportable cards were produced.",
            detail="Candidates became empty after validation/translation.",
            next_steps=["Check input quality", "Enable --save-intermediate and inspect outputs"],
            exit_code=ExitCode.CARD_VALIDATION_ERROR,
        )
        _save_failure_snapshot(err)
        raise err
    if not cards:
        errors.append(
            {
                "stage": "empty_output",
                "reason": "no cards produced; exporting empty deck",
            }
        )
    translated_non_empty_count = len(
        [card for card in cards if str(card.translation or "").strip()]
    )
    translation_empty_count = len(cards) - translated_non_empty_count
    logger.info(
        "translation generation complete | cards=%d translated_non_empty=%d translation_empty=%d",
        len(cards),
        translated_non_empty_count,
        translation_empty_count,
    )
    if use_phrase_pipeline:
        logger.info(
            "phrase hint translation complete | unique_phrases=%d hints_mapped=%d",
            phrase_translation_total,
            len(phrase_hint_map),
        )

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
            except ClawLearnError as exc:
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
        deck_name = _resolve_deck_name(cfg=cfg, options=options, document=document)
        export_apkg(
            cards=cards,
            template=template,
            output_path=output_path,
            media_files=media_files,
            deck_name_override=deck_name,
        )
    except ClawLearnError as exc:
        _save_failure_snapshot(exc)
        raise
    logger.info("deck export complete | file=%s", output_path)

    pipeline_metrics = _build_pipeline_metrics(
        chunks=chunks,
        raw_candidates=raw_candidates,
        valid_candidates=valid_candidates,
        ranked_candidates=ranked_candidates,
        deduped_candidates=deduped,
        errors=errors,
        learning_mode=learning_mode,
        format_retry_stats=format_retry_stats,
    )
    pipeline_metrics["taxonomy_reject_count"] = taxonomy_repair_stats.taxonomy_reject_count
    pipeline_metrics["taxonomy_repair_attempted"] = taxonomy_repair_stats.taxonomy_repair_attempted
    pipeline_metrics["taxonomy_repair_success"] = taxonomy_repair_stats.taxonomy_repair_success
    pipeline_metrics["taxonomy_repair_failed"] = taxonomy_repair_stats.taxonomy_repair_failed
    pipeline_metrics["translation"] = {
        "cards_total": len(cards),
        "translated_non_empty": translated_non_empty_count,
        "translation_empty": translation_empty_count,
        "phrase_hints_requested": phrase_translation_total,
        "phrase_hints_mapped": len(phrase_hint_map),
    }
    pipeline_metrics["prompt_info"] = {
        "extraction_prompt_name": extract_prompt.name,
        "extraction_prompt_version": extract_prompt.version,
        "extraction_prompt_path": str(cfg.resolve_path(extract_prompt_path)),
        "explanation_prompt_name": explain_prompt.name,
        "explanation_prompt_version": explain_prompt.version,
        "explanation_prompt_path": str(cfg.resolve_path(explain_prompt_path)),
        # Backward-compatible aliases.
        "cloze_prompt_name": extract_prompt.name,
        "cloze_prompt_version": extract_prompt.version,
        "translate_prompt_name": explain_prompt.name,
        "translate_prompt_version": explain_prompt.version,
    }
    pipeline_metrics["difficulty"] = cfg.cloze_difficulty

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
            material_profile=material_profile,
            learning_mode=learning_mode,
            difficulty=cfg.cloze_difficulty,
            metrics=pipeline_metrics,
        )

    return BuildDeckResult(
        run_id=run_ctx.run_id,
        run_dir=run_ctx.run_dir,
        output_path=output_path,
        cards_count=len(cards),
        errors_count=len(errors),
    )
