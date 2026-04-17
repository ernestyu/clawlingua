"""End-to-end lingua deck building pipeline."""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any

from ..anki.deck_exporter import export_apkg
from ..anki.media_manager import MediaManager
from ..anki.template_loader import load_anki_template
from ..config import AppConfig, validate_base_config, validate_runtime_config
from ..constants import (
    SUPPORTED_CONTENT_PROFILES,
    SUPPORTED_FILE_SUFFIXES,
    SUPPORTED_LINGUA_LEARNING_MODES,
)
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
from ..llm.taxonomy_classifier import classify_lingua_prerank_phrases_batch
from ..llm.prompt_loader import load_prompt
from ..llm.response_parser import parse_json_content
from ..llm.translation_generator import (
    TranslationBatchResult,
    generate_phrase_translations_batch,
    generate_translation_batch,
)
from ..models.card import CardRecord
from ..models.document import DocumentRecord
from ..phrase_filters.en import phrase_quality_score
from ..phrase_filters import filter_phrases
from ..runtime import create_run_context
from ..tts.provider_registry import get_tts_provider
from ..tts.voice_selector import UniformVoiceSelector
from ..utils.hash import stable_hash
from ..utils.jsonx import dump_json, dump_jsonl, load_json
from ..utils.text import normalize_for_dedupe, split_sentences
from ..utils.time import utc_now_iso
from .core_chunking import chunk_document
from .core_candidates import dump_candidate_artifacts
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
_EXPORT_FORBIDDEN_TRIPLE_CLOZE_RE = re.compile(r"\{\{\{c\d+::", re.IGNORECASE)
_EXPORT_FORBIDDEN_SINGLE_BRACE_HINT_RE = re.compile(r"\{[^{}]+\}\(hint\)", re.IGNORECASE)
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
_CONTEXT_EXPAND_MAX_SENTENCE_CHARS = 360
_CONTEXT_SUBSTRING_MATCH_MIN_CHARS = 24
_CONTEXT_LEADING_CONNECTOR_RE = re.compile(r"^(?:and|but|so|then)\s+", re.IGNORECASE)
_EMPTY_RAW_CANDIDATE_RATIO_GUARD = 0.95
_SECONDARY_CONTEXT_OVERLAP_MIN = 0.6
_SECONDARY_CONTEXT_TOKEN_RE = re.compile(r"[a-z0-9']+")
_LINGUA_PRE_RANK_FALLBACK_MIN_SCORE = 1.0
_SECONDARY_CONTEXT_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "that",
    "this",
    "it",
    "is",
    "are",
    "was",
    "were",
}


def _empty_phrase_filter_stats(*, enabled: bool, source_lang: str, language: str = "") -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "source_lang": str(source_lang or "").strip().lower(),
        "language": str(language or "").strip().lower(),
        "dropped_count": 0,
        "dropped_by_rule": {},
        "examples_by_rule": {},
    }


def _merge_phrase_filter_stats(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged["enabled"] = bool(merged.get("enabled", False) or incoming.get("enabled", False))
    merged["source_lang"] = str(incoming.get("source_lang") or merged.get("source_lang") or "").strip().lower()
    merged["language"] = str(incoming.get("language") or merged.get("language") or "").strip().lower()
    merged["dropped_count"] = int(merged.get("dropped_count", 0)) + int(incoming.get("dropped_count", 0))

    dropped_by_rule: dict[str, int] = {}
    for mapping in (merged.get("dropped_by_rule", {}), incoming.get("dropped_by_rule", {})):
        if not isinstance(mapping, dict):
            continue
        for key, value in mapping.items():
            rule = str(key).strip()
            if not rule:
                continue
            try:
                count = int(value)
            except (TypeError, ValueError):
                count = 0
            if count <= 0:
                continue
            dropped_by_rule[rule] = dropped_by_rule.get(rule, 0) + count
    merged["dropped_by_rule"] = dropped_by_rule

    examples_by_rule: dict[str, list[str]] = {}
    for mapping in (merged.get("examples_by_rule", {}), incoming.get("examples_by_rule", {})):
        if not isinstance(mapping, dict):
            continue
        for key, value in mapping.items():
            rule = str(key).strip()
            if not rule or not isinstance(value, list):
                continue
            bucket = examples_by_rule.setdefault(rule, [])
            for item in value:
                text = str(item).strip()
                if not text or text in bucket:
                    continue
                if len(bucket) >= 3:
                    break
                bucket.append(text)
    merged["examples_by_rule"] = examples_by_rule
    return merged


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
class _LinguaPreRankStats:
    candidates_input: int = 0
    candidates_output: int = 0
    candidates_context_only: int = 0
    candidates_empty_label_dropped: int = 0
    phrases_total: int = 0
    phrases_annotated: int = 0
    phrases_kept: int = 0
    phrases_none: int = 0
    phrases_dropped: int = 0
    partial_retry_count: int = 0
    exhausted_count: int = 0
    batch_error_count: int = 0


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
    extract_sources: list[str] = field(default_factory=list)


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
    extract_sources: list[str] = field(default_factory=list)


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
    secondary_extract_enable: bool | None = None


@dataclass
class BuildDeckResult:
    run_id: str
    run_dir: Path
    output_path: Path
    cards_count: int
    errors_count: int


def _use_phrase_extraction_pipeline(*, learning_mode: str, schema_name: str | None) -> bool:
    schema = (schema_name or "").strip().lower()
    return schema.startswith("phrase_candidates_")


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
    extract_sources = [
        str(source).strip()
        for source in (item.get("extract_sources") or [])
        if str(source).strip()
    ]
    return PhraseCandidate(
        chunk_id=chunk_id,
        sentence_text=sentence_text,
        phrase_text=phrase_text,
        reason=reason,
        phrase_types=phrase_types,
        learning_value_score=_coerce_learning_value_score(item.get("learning_value_score")),
        expression_transfer=expression_transfer,
        chunk_text=chunk_text,
        extract_sources=extract_sources,
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
                "extract_sources": list(candidate.extract_sources or []),
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


def _build_cloze_units_from_phrases(
    phrase_candidates: list[PhraseCandidate],
    *,
    source_lang: str,
    learning_mode: str,
    difficulty: str,
) -> tuple[list[ClozeUnit], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[PhraseCandidate]] = {}
    for candidate in phrase_candidates:
        key = (candidate.chunk_id, candidate.sentence_text)
        grouped.setdefault(key, []).append(candidate)

    filter_enabled = learning_mode in {"lingua_expression", "lingua_reading"}
    phrase_filter_stats = _empty_phrase_filter_stats(
        enabled=filter_enabled,
        source_lang=source_lang,
        language=(source_lang or "").strip().lower(),
    )
    units: list[ClozeUnit] = []
    for (chunk_id, sentence_text), group in grouped.items():
        filtered_group = group
        if filter_enabled:
            kept_phrases, stats = filter_phrases(
                source_lang=source_lang,
                phrases=[candidate.phrase_text for candidate in group],
                context=sentence_text,
                difficulty=difficulty,
            )
            phrase_filter_stats = _merge_phrase_filter_stats(
                phrase_filter_stats,
                {
                    **stats,
                    "enabled": True,
                    "source_lang": source_lang,
                },
            )
            kept_keys = {normalize_for_dedupe(phrase) for phrase in kept_phrases}
            filtered_group = []
            seen_keys: set[str] = set()
            for candidate in group:
                key = normalize_for_dedupe(candidate.phrase_text)
                if not key or key not in kept_keys or key in seen_keys:
                    continue
                seen_keys.add(key)
                filtered_group.append(candidate)

        spans = _select_non_overlapping_spans(sentence_text, filtered_group)
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
        extract_sources: list[str] = []
        for span in spans:
            for source in (span.get("extract_sources") or []):
                source_text = str(source).strip()
                if not source_text or source_text in extract_sources:
                    continue
                extract_sources.append(source_text)
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
                extract_sources=extract_sources,
            )
        )
    return units, phrase_filter_stats


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
    if unit.extract_sources:
        candidate["extract_sources"] = list(unit.extract_sources)
    return candidate


def _materialize_cloze_candidates_from_phrase_items(
    phrase_items: list[dict[str, Any]],
    *,
    source_lang: str,
    learning_mode: str,
    difficulty: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phrase_candidates = [
        candidate
        for item in phrase_items
        if (candidate := _normalize_phrase_candidate_item(item)) is not None
    ]
    units, phrase_filter_stats = _build_cloze_units_from_phrases(
        phrase_candidates,
        source_lang=source_lang,
        learning_mode=learning_mode,
        difficulty=difficulty,
    )
    return [_cloze_unit_to_candidate(unit) for unit in units], phrase_filter_stats


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


def _filter_cards_for_export_cloze_format(
    *,
    cards: list[CardRecord],
    errors: list[dict[str, Any]],
) -> list[CardRecord]:
    filtered: list[CardRecord] = []
    for card in cards:
        text = str(card.text or "")
        reason = ""
        if _EXPORT_FORBIDDEN_TRIPLE_CLOZE_RE.search(text):
            reason = "contains triple-brace cloze marker"
        elif _EXPORT_FORBIDDEN_SINGLE_BRACE_HINT_RE.search(text):
            reason = "contains single-brace hint block"
        elif not _CLOZE_WITH_HINT_RE.search(text):
            reason = "missing canonical cloze block"
        if reason:
            errors.append(
                {
                    "stage": "export_cloze_format_guard",
                    "card_id": card.card_id,
                    "chunk_id": card.chunk_id,
                    "reason": reason,
                    "text_prefix": text[:120],
                }
            )
            continue
        filtered.append(card)
    return filtered


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
    dump_candidate_artifacts(
        run_dir=run_dir,
        raw_candidates=raw_candidates,
        validated_candidates=valid_candidates,
        write_legacy_text_candidates=True,
    )
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
    mode = (options.learning_mode or cfg.learning_mode or "lingua_expression").strip().lower()
    if mode not in SUPPORTED_LINGUA_LEARNING_MODES:
        allowed = ", ".join(sorted(SUPPORTED_LINGUA_LEARNING_MODES))
        raise build_error(
            error_code="ARG_LEARNING_MODE_INVALID",
            cause="Unsupported learning mode for lingua domain.",
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


def _apply_contrastive_rerank(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return []
    primary_label_freq: dict[str, int] = {}
    for item in items:
        labels = [str(x).strip() for x in (item.get("phrase_types") or []) if str(x).strip()]
        primary = labels[0] if labels else ""
        if not primary:
            continue
        primary_label_freq[primary] = primary_label_freq.get(primary, 0) + 1

    reranked: list[dict[str, Any]] = []
    for item in items:
        row = dict(item)
        base = float(row.get("learning_value_score", 0.0))
        labels = [str(x).strip() for x in (row.get("phrase_types") or []) if str(x).strip()]
        primary = labels[0] if labels else ""
        freq = primary_label_freq.get(primary, 0)
        bonus = (0.2 / freq) if freq > 0 else 0.0
        row["learning_value_score"] = round(base + bonus, 4)
        reranked.append(row)
    reranked.sort(
        key=lambda row: (
            float(row.get("learning_value_score", 0.0)),
            len(str(row.get("original", ""))),
            len(str(row.get("text", ""))),
        ),
        reverse=True,
    )
    return reranked


def _apply_phrase_diversity_cap(
    items: list[dict[str, Any]],
    *,
    max_per_primary_phrase: int,
) -> list[dict[str, Any]]:
    if max_per_primary_phrase <= 0:
        return list(items)
    limited: list[dict[str, Any]] = []
    primary_phrase_counts: dict[str, int] = {}
    for item in items:
        phrases = [str(x).strip() for x in (item.get("target_phrases") or []) if str(x).strip()]
        primary_key = normalize_for_dedupe(phrases[0]) if phrases else ""
        if not primary_key:
            limited.append(item)
            continue
        current = primary_phrase_counts.get(primary_key, 0)
        if current >= max_per_primary_phrase:
            continue
        primary_phrase_counts[primary_key] = current + 1
        limited.append(item)
    return limited


def _expand_candidate_context_in_chunk(
    *,
    text: str,
    original: str,
    chunk_text: str,
    min_sentences: int,
) -> tuple[str, str, str]:
    required_sentences = 2 if int(min_sentences or 0) >= 2 else 1
    base_text = str(text or "").strip()
    base_original = str(original or "").strip()
    if required_sentences <= 1 or not base_text or not base_original:
        return base_text, base_original, ""
    if len(split_sentences(base_original)) >= required_sentences:
        return base_text, base_original, ""

    chunk_sentences = split_sentences(chunk_text)
    if len(chunk_sentences) < 2:
        return base_text, base_original, ""

    def _normalized_context_key(value: str) -> str:
        key = normalize_for_dedupe(value)
        key = _CONTEXT_LEADING_CONNECTOR_RE.sub("", key).strip()
        return key

    idx = -1
    for i, sentence in enumerate(chunk_sentences):
        if sentence == base_original:
            idx = i
            break
    if idx < 0:
        original_key = _normalized_context_key(base_original)
        for i, sentence in enumerate(chunk_sentences):
            if _normalized_context_key(sentence) == original_key:
                idx = i
                break
    if idx < 0:
        original_key = _normalized_context_key(base_original)
        if len(original_key) >= _CONTEXT_SUBSTRING_MATCH_MIN_CHARS:
            for i, sentence in enumerate(chunk_sentences):
                sentence_key = _normalized_context_key(sentence)
                if not sentence_key:
                    continue
                if original_key in sentence_key or sentence_key in original_key:
                    idx = i
                    break
    if idx < 0:
        return base_text, base_original, ""

    expanded_sentence = ""
    expanded_with = ""
    if idx + 1 < len(chunk_sentences):
        candidate = chunk_sentences[idx + 1]
        if len(candidate) > _CONTEXT_EXPAND_MAX_SENTENCE_CHARS:
            return base_text, base_original, ""
        expanded_sentence = candidate
        expanded_with = "next_sentence"
    elif idx > 0:
        candidate = chunk_sentences[idx - 1]
        if len(candidate) > _CONTEXT_EXPAND_MAX_SENTENCE_CHARS:
            return base_text, base_original, ""
        expanded_sentence = candidate
        expanded_with = "previous_sentence"

    if not expanded_sentence:
        return base_text, base_original, ""
    if expanded_with == "next_sentence":
        return (
            f"{base_text} {expanded_sentence}".strip(),
            f"{base_original} {expanded_sentence}".strip(),
            expanded_with,
        )
    return (
        f"{expanded_sentence} {base_text}".strip(),
        f"{expanded_sentence} {base_original}".strip(),
        expanded_with,
    )


def _apply_transcript_context_fallback(
    *,
    items: list[dict[str, Any]],
    material_profile: str,
    learning_mode: str,
    min_sentences: int,
) -> int:
    if material_profile != "transcript_dialogue" or learning_mode != "lingua_expression":
        return 0
    required_sentences = 2 if int(min_sentences or 0) >= 2 else 1
    if required_sentences <= 1:
        return 0

    expanded_count = 0
    for item in items:
        chunk_text = str(item.get("chunk_text") or "").strip()
        text = str(item.get("text") or "").strip()
        original = str(item.get("original") or "").strip()
        if not chunk_text or not text or not original:
            continue
        expanded_text, expanded_original, expanded_with = _expand_candidate_context_in_chunk(
            text=text,
            original=original,
            chunk_text=chunk_text,
            min_sentences=required_sentences,
        )
        if not expanded_with:
            continue
        item["text"] = expanded_text
        item["original"] = expanded_original
        item["context_expanded"] = True
        item["context_expanded_with"] = expanded_with
        expanded_count += 1
    return expanded_count


def _count_cloze_marks(text: str) -> int:
    return len(CLOZE_MARK_RE.findall(str(text or "")))


def _is_structurally_empty_raw_candidate(item: dict[str, Any]) -> bool:
    text = str(item.get("text") or "").strip()
    original = str(item.get("original") or "").strip()
    target_phrases = item.get("target_phrases")
    if not isinstance(target_phrases, list):
        target_count = 0
    else:
        target_count = len([str(x).strip() for x in target_phrases if str(x).strip()])
    return (not text) and (not original) and target_count == 0


def _should_fail_for_structurally_empty_raw_candidates(raw_candidates: list[dict[str, Any]]) -> tuple[bool, int, float]:
    total = len(raw_candidates)
    if total <= 0:
        return False, 0, 0.0
    empty_count = len([item for item in raw_candidates if _is_structurally_empty_raw_candidate(item)])
    ratio = float(empty_count) / float(total)
    if empty_count == total:
        return True, empty_count, ratio
    if total >= 20 and ratio >= _EMPTY_RAW_CANDIDATE_RATIO_GUARD:
        return True, empty_count, ratio
    return False, empty_count, ratio


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


def _empty_secondary_extract_stats(
    *,
    requested: bool,
    model: str,
    parallel_requested: bool = False,
) -> dict[str, Any]:
    return {
        "requested": bool(requested),
        "enabled": False,
        "configured": False,
        "parallel": bool(parallel_requested),
        "execution_mode": "disabled",
        "secondary_model": str(model or "").strip(),
        "candidates_primary_count": 0,
        "candidates_secondary_count": 0,
        "candidates_merged_count": 0,
        "dedup_removed_count": 0,
        "unique_phrase_gain_from_secondary": 0,
        "secondary_error_type": "",
        "secondary_error_message": "",
        "fallback_to_primary": False,
    }


def _secondary_error_type(exc: Exception) -> str:
    if isinstance(exc, ClawLearnError):
        code = str(exc.error_code or "").strip().lower()
        if "timeout" in code:
            return "timeout"
        if "parse" in code or "shape" in code:
            return "parse"
    text = str(exc).strip().lower()
    if "timeout" in text:
        return "timeout"
    return "other"


def _context_token_set(value: str) -> set[str]:
    normalized = normalize_for_dedupe(value)
    if not normalized:
        return set()
    tokens = [
        token
        for token in _SECONDARY_CONTEXT_TOKEN_RE.findall(normalized)
        if len(token) >= 3 and token not in _SECONDARY_CONTEXT_STOPWORDS
    ]
    return set(tokens)


def _context_overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    common = len(left.intersection(right))
    baseline = float(max(1, min(len(left), len(right))))
    return common / baseline


def _merge_phrase_extraction_candidates(
    *,
    primary_items: list[dict[str, Any]],
    secondary_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    normalized_primary = [
        candidate
        for item in primary_items
        if isinstance(item, dict)
        if (candidate := _normalize_phrase_candidate_item(item)) is not None
    ]
    normalized_secondary = [
        candidate
        for item in secondary_items
        if isinstance(item, dict)
        if (candidate := _normalize_phrase_candidate_item(item)) is not None
    ]
    if not normalized_secondary:
        merged_primary: list[dict[str, Any]] = []
        for candidate in normalized_primary:
            merged_primary.append(
                {
                    "chunk_id": candidate.chunk_id,
                    "sentence_text": candidate.sentence_text,
                    "phrase_text": candidate.phrase_text,
                    "extract_sources": ["primary"],
                }
            )
        return merged_primary, 0

    groups: list[dict[str, Any]] = []

    def _add_candidate(candidate: PhraseCandidate, source: str) -> None:
        context_text = str(candidate.sentence_text or "").strip()
        if not context_text:
            return
        context_tokens = _context_token_set(context_text)
        normalized_context = normalize_for_dedupe(context_text)
        phrase_key = normalize_for_dedupe(candidate.phrase_text)
        if not phrase_key:
            return

        best_idx = -1
        best_score = 0.0
        for idx, group in enumerate(groups):
            if normalized_context and normalized_context == group["context_norm"]:
                best_idx = idx
                best_score = 1.0
                break
            overlap = _context_overlap_ratio(context_tokens, group["context_tokens"])
            if overlap >= _SECONDARY_CONTEXT_OVERLAP_MIN and overlap > best_score:
                best_idx = idx
                best_score = overlap

        if best_idx < 0:
            groups.append(
                {
                    "chunk_id": candidate.chunk_id,
                    "context_text": context_text,
                    "context_norm": normalized_context,
                    "context_tokens": context_tokens,
                    "phrases": {},
                }
            )
            best_idx = len(groups) - 1

        group = groups[best_idx]
        if source == "primary":
            group["chunk_id"] = candidate.chunk_id
            group["context_text"] = context_text
            group["context_norm"] = normalized_context
            group["context_tokens"] = context_tokens

        phrase_bucket = group["phrases"].setdefault(
            phrase_key,
            {"candidate": candidate, "sources": set()},
        )
        if source == "primary":
            phrase_bucket["candidate"] = candidate
        phrase_bucket["sources"].add(source)

    for item in normalized_primary:
        _add_candidate(item, "primary")
    for item in normalized_secondary:
        _add_candidate(item, "secondary")

    primary_phrase_keys = {
        normalize_for_dedupe(candidate.phrase_text)
        for candidate in normalized_primary
        if normalize_for_dedupe(candidate.phrase_text)
    }
    secondary_phrase_keys = {
        normalize_for_dedupe(candidate.phrase_text)
        for candidate in normalized_secondary
        if normalize_for_dedupe(candidate.phrase_text)
    }
    unique_gain = len([key for key in secondary_phrase_keys if key not in primary_phrase_keys])

    merged: list[dict[str, Any]] = []
    for group in groups:
        context_text = str(group.get("context_text") or "").strip()
        context_key = f"ctx_{stable_hash(context_text, length=8)}" if context_text else ""
        for phrase_key, payload in group.get("phrases", {}).items():
            candidate = payload.get("candidate")
            if not isinstance(candidate, PhraseCandidate):
                continue
            merged.append(
                {
                    "chunk_id": str(group.get("chunk_id") or candidate.chunk_id).strip(),
                    "sentence_text": context_text or candidate.sentence_text,
                    "phrase_text": candidate.phrase_text,
                    "extract_sources": sorted(
                        str(source) for source in payload.get("sources", set()) if str(source).strip()
                    ),
                    "extract_context_key": context_key or phrase_key,
                }
            )
    return merged, unique_gain


def _merge_cloze_extraction_candidates(
    *,
    primary_items: list[dict[str, Any]],
    secondary_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    merged: dict[str, dict[str, Any]] = {}

    def _candidate_key(item: dict[str, Any]) -> str:
        chunk_id = str(item.get("chunk_id") or "").strip().lower()
        original = normalize_for_dedupe(str(item.get("original") or item.get("text") or ""))
        if not chunk_id or not original:
            return ""
        return f"{chunk_id}|{original}"

    def _append(items: list[dict[str, Any]], source: str) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            key = _candidate_key(item)
            if not key:
                continue
            if key not in merged:
                created = dict(item)
                created["extract_sources"] = [source]
                merged[key] = created
                continue
            existing = merged[key]
            existing_sources = {
                str(value).strip()
                for value in (existing.get("extract_sources") or [])
                if str(value).strip()
            }
            existing_sources.add(source)
            existing["extract_sources"] = sorted(existing_sources)

            existing_phrases = [
                str(value).strip()
                for value in (existing.get("target_phrases") or [])
                if str(value).strip()
            ]
            incoming_phrases = [
                str(value).strip()
                for value in (item.get("target_phrases") or [])
                if str(value).strip()
            ]
            seen_keys = {normalize_for_dedupe(value) for value in existing_phrases}
            for phrase in incoming_phrases:
                key_norm = normalize_for_dedupe(phrase)
                if not key_norm or key_norm in seen_keys:
                    continue
                seen_keys.add(key_norm)
                existing_phrases.append(phrase)
            if existing_phrases:
                existing["target_phrases"] = existing_phrases

    _append(primary_items, "primary")
    _append(secondary_items, "secondary")

    primary_phrase_keys = {
        normalize_for_dedupe(str(phrase))
        for item in primary_items
        if isinstance(item, dict)
        for phrase in (item.get("target_phrases") or [])
        if normalize_for_dedupe(str(phrase))
    }
    secondary_phrase_keys = {
        normalize_for_dedupe(str(phrase))
        for item in secondary_items
        if isinstance(item, dict)
        for phrase in (item.get("target_phrases") or [])
        if normalize_for_dedupe(str(phrase))
    }
    unique_gain = len([key for key in secondary_phrase_keys if key not in primary_phrase_keys])
    return list(merged.values()), unique_gain


def _merge_secondary_extraction_candidates(
    *,
    use_phrase_pipeline: bool,
    primary_items: list[dict[str, Any]],
    secondary_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    if not secondary_items:
        return list(primary_items), 0
    if use_phrase_pipeline:
        return _merge_phrase_extraction_candidates(
            primary_items=primary_items,
            secondary_items=secondary_items,
        )
    return _merge_cloze_extraction_candidates(
        primary_items=primary_items,
        secondary_items=secondary_items,
    )


def _run_extraction_passes(
    *,
    run_primary: Callable[[], Exception | None],
    run_secondary: Callable[[], Exception | None] | None = None,
    secondary_parallel: bool = False,
) -> tuple[Exception | None, Exception | None]:
    if run_secondary is None:
        return run_primary(), None
    if not secondary_parallel:
        return run_primary(), run_secondary()
    with ThreadPoolExecutor(max_workers=2) as pool:
        primary_future = pool.submit(run_primary)
        secondary_future = pool.submit(run_secondary)
        return primary_future.result(), secondary_future.result()


def _build_pipeline_metrics(
    *,
    chunks: list[Any],
    raw_candidates: list[dict[str, Any]],
    valid_candidates: list[dict[str, Any]],
    ranked_candidates: list[dict[str, Any]],
    deduped_candidates: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    learning_mode: str = "lingua_expression",
    format_retry_stats: _FormatRetryStats | None = None,
    phrase_filter_stats: dict[str, Any] | None = None,
    secondary_extract_stats: dict[str, Any] | None = None,
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
    filter_stats = phrase_filter_stats or _empty_phrase_filter_stats(enabled=False, source_lang="", language="")
    secondary_stats = secondary_extract_stats or _empty_secondary_extract_stats(
        requested=False,
        model="",
        parallel_requested=False,
    )
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
        "phrase_filter": {
            "enabled": bool(filter_stats.get("enabled", False)),
            "source_lang": str(filter_stats.get("source_lang", "")).strip(),
            "language": str(filter_stats.get("language", "")).strip(),
            "dropped_count": int(filter_stats.get("dropped_count", 0)),
            "dropped_by_rule": dict(filter_stats.get("dropped_by_rule", {})),
            "examples_by_rule": dict(filter_stats.get("examples_by_rule", {})),
        },
        "secondary_extraction": {
            "requested": bool(secondary_stats.get("requested", False)),
            "enabled": bool(secondary_stats.get("enabled", False)),
            "configured": bool(secondary_stats.get("configured", False)),
            "parallel": bool(secondary_stats.get("parallel", False)),
            "execution_mode": str(secondary_stats.get("execution_mode", "")).strip(),
            "secondary_model": str(secondary_stats.get("secondary_model", "")).strip(),
            "candidates_primary_count": int(secondary_stats.get("candidates_primary_count", 0)),
            "candidates_secondary_count": int(secondary_stats.get("candidates_secondary_count", 0)),
            "candidates_merged_count": int(secondary_stats.get("candidates_merged_count", 0)),
            "dedup_removed_count": int(secondary_stats.get("dedup_removed_count", 0)),
            "unique_phrase_gain_from_secondary": int(
                secondary_stats.get("unique_phrase_gain_from_secondary", 0)
            ),
            "secondary_error_type": str(secondary_stats.get("secondary_error_type", "")).strip(),
            "secondary_error_message": str(secondary_stats.get("secondary_error_message", "")).strip(),
            "fallback_to_primary": bool(secondary_stats.get("fallback_to_primary", False)),
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
    normalized = re.sub(r"(?<!\{)\{cN::", "{{c1::", normalized)
    normalized = re.sub(r"(?<!\{)\{c(\d+)::", r"{{c\1::", normalized)
    normalized = re.sub(r"\{\{c(\d+)::([^}]+)\}(?!\})", r"{{c\1::\2}}", normalized)

    cloze_with_optional_hint_re = re.compile(
        r"\{\{c(\d+)::(.*?)\}\}(?:\(([^)]*)\))?",
        re.DOTALL,
    )

    def _fmt(match: re.Match[str]) -> str:
        idx = match.group(1)
        body = re.sub(r"</?b>", "", match.group(2), flags=re.IGNORECASE).strip()
        hint = str(match.group(3) or "").strip() or "hint"
        return f"{{{{c{idx}::<b>{body}</b>}}}}({hint})" if body else ""

    normalized = cloze_with_optional_hint_re.sub(_fmt, normalized)
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
                    allow_partial=True,
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

            usable_count = min(len(repaired_phrase_types), len(remaining))
            if usable_count <= 0:
                if attempt < TRANSLATION_BATCH_MAX_RETRIES:
                    errors.append(
                        {
                            "stage": "taxonomy_repair_batch_partial_retry",
                            "attempt": attempt,
                            "received": 0,
                            "remaining": len(remaining),
                        }
                    )
                    delay = _translation_retry_delay_seconds(cfg, attempt)
                    if delay > 0:
                        time.sleep(delay)
                    continue
                _record_batch_failure(
                    remaining,
                    "taxonomy repair batch partial response exhausted",
                    stage="taxonomy_repair_failed",
                    attempt=attempt,
                )
                errors.append(
                    {
                        "stage": "taxonomy_repair_batch_error",
                        "attempt": attempt,
                        "remaining": len(remaining),
                        "error": [
                            "taxonomy repair partial response exhausted",
                            f"attempts={TRANSLATION_BATCH_MAX_RETRIES}",
                        ],
                    }
                )
                break

            current_entries = remaining[:usable_count]
            current_labels = repaired_phrase_types[:usable_count]
            for entry, labels in zip(current_entries, current_labels, strict=False):
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
            remaining = remaining[usable_count:]
            if not remaining:
                break
            if attempt < TRANSLATION_BATCH_MAX_RETRIES:
                errors.append(
                    {
                        "stage": "taxonomy_repair_batch_partial_retry",
                        "attempt": attempt,
                        "received": usable_count,
                        "remaining": len(remaining),
                    }
                )
                delay = _translation_retry_delay_seconds(cfg, attempt)
                if delay > 0:
                    time.sleep(delay)
                continue
            _record_batch_failure(
                remaining,
                "taxonomy repair batch partial response exhausted",
                stage="taxonomy_repair_failed",
                attempt=attempt,
            )
            errors.append(
                {
                    "stage": "taxonomy_repair_batch_error",
                    "attempt": attempt,
                    "remaining": len(remaining),
                    "error": [
                        "taxonomy repair partial response exhausted",
                        f"attempts={TRANSLATION_BATCH_MAX_RETRIES}",
                    ],
                }
            )
            break

    return repaired_valid, stats


def _extract_candidate_cloze_occurrences(text: str) -> list[dict[str, Any]]:
    value = str(text or "")
    if not value:
        return []
    occurrences: list[dict[str, Any]] = []
    for match in _CLOZE_WITH_HINT_RE.finditer(value):
        full = match.group(0)
        phrase = str(match.group(2) or "").strip()
        if not full:
            continue
        occurrences.append(
            {
                "full": full,
                "phrase": phrase,
            }
        )
    if occurrences:
        return occurrences
    for match in _CLOZE_BLOCK_GENERIC_RE.finditer(value):
        full = str(match.group(0) or "")
        body = str(match.group(2) or "")
        phrase = re.sub(r"</?b>", "", body, flags=re.IGNORECASE).strip()
        if not full:
            continue
        occurrences.append(
            {
                "full": full,
                "phrase": phrase,
            }
        )
    return occurrences


def _reindex_candidate_cloze_numbers(text: str) -> str:
    counter = {"n": 0}

    def _repl(match: re.Match[str]) -> str:
        counter["n"] += 1
        body = match.group(2)
        return f"{{{{c{counter['n']}::{body}}}}}"

    return _CLOZE_BLOCK_GENERIC_RE.sub(_repl, str(text or ""))


def _map_phrase_indices_to_cloze_occurrences(
    *,
    target_phrases: list[str],
    occurrences: list[dict[str, Any]],
) -> dict[int, int]:
    if not target_phrases or not occurrences:
        return {}
    if len(target_phrases) == len(occurrences):
        return {idx: idx for idx in range(len(target_phrases))}
    mapping: dict[int, int] = {}
    unused = set(range(len(occurrences)))
    for phrase_idx, phrase in enumerate(target_phrases):
        key = normalize_for_dedupe(phrase)
        matched_occ: int | None = None
        for occ_idx in sorted(unused):
            occ_key = normalize_for_dedupe(str(occurrences[occ_idx].get("phrase", "")))
            if key and occ_key and key == occ_key:
                matched_occ = occ_idx
                break
        if matched_occ is None and phrase_idx in unused:
            matched_occ = phrase_idx
        if matched_occ is None and unused:
            matched_occ = min(unused)
        if matched_occ is None:
            continue
        mapping[phrase_idx] = matched_occ
        unused.discard(matched_occ)
    return mapping


def _rewrite_candidate_by_kept_phrase_indices(
    *,
    item: dict[str, Any],
    kept_phrase_indices: set[int],
) -> dict[str, Any] | None:
    text = str(item.get("text", "")).strip()
    target_phrases = [str(x).strip() for x in (item.get("target_phrases") or []) if str(x).strip()]
    if not target_phrases:
        return None

    occurrences = _extract_candidate_cloze_occurrences(text)
    mapping = _map_phrase_indices_to_cloze_occurrences(
        target_phrases=target_phrases,
        occurrences=occurrences,
    )
    rewritten = text
    for phrase_idx in range(len(target_phrases)):
        if phrase_idx in kept_phrase_indices:
            continue
        occ_idx = mapping.get(phrase_idx)
        if occ_idx is None or occ_idx >= len(occurrences):
            continue
        full = str(occurrences[occ_idx].get("full", ""))
        phrase_text = str(occurrences[occ_idx].get("phrase", "")).strip() or target_phrases[phrase_idx]
        if not full:
            continue
        rewritten = rewritten.replace(full, phrase_text, 1)

    kept_phrases = [target_phrases[idx] for idx in range(len(target_phrases)) if idx in kept_phrase_indices]
    rewritten = re.sub(r"\s+", " ", _reindex_candidate_cloze_numbers(rewritten)).strip()
    if not kept_phrases or not CLOZE_MARK_RE.search(rewritten):
        return None
    updated = dict(item)
    updated["text"] = rewritten
    updated["target_phrases"] = kept_phrases
    return updated


def _apply_advanced_phrase_quality_gate(
    *,
    item: dict[str, Any],
    source_lang: str,
    difficulty: str,
) -> tuple[dict[str, Any] | None, int]:
    lang = str(source_lang or "").strip().lower()
    diff = str(difficulty or "").strip().lower()
    if lang != "en" or diff != "advanced":
        return item, 0

    target_phrases = [str(x).strip() for x in (item.get("target_phrases") or []) if str(x).strip()]
    if not target_phrases:
        return None, 0
    kept_indices: set[int] = set()
    for idx, phrase in enumerate(target_phrases):
        if phrase_quality_score(phrase) >= 1.0:
            kept_indices.add(idx)
    if len(kept_indices) == len(target_phrases):
        return item, 0
    rewritten = _rewrite_candidate_by_kept_phrase_indices(
        item=item,
        kept_phrase_indices=kept_indices,
    )
    dropped_count = len(target_phrases) - len(kept_indices)
    return rewritten, max(0, dropped_count)


def _drop_candidates_without_taxonomy_labels(
    *,
    items: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    filtered: list[dict[str, Any]] = []
    dropped = 0
    for item in items:
        labels = normalize_phrase_types(item.get("phrase_types"), max_items=2)
        if not labels:
            dropped += 1
            continue
        updated = dict(item)
        updated["phrase_types"] = labels
        filtered.append(updated)
    if dropped > 0:
        errors.append(
            {
                "stage": "lingua_taxonomy_pre_rank_empty_label_drop",
                "dropped_count": dropped,
            }
        )
    return filtered, dropped


def _annotate_lingua_candidates_pre_rank(
    *,
    items: list[dict[str, Any]],
    cfg: AppConfig,
    learning_mode: str,
    source_lang: str,
    client: OpenAICompatibleClient,
    errors: list[dict[str, Any]],
    temperature: float | None = None,
    classify_fn: Callable[..., list[dict[str, Any]]] = classify_lingua_prerank_phrases_batch,
) -> tuple[list[dict[str, Any]], _LinguaPreRankStats]:
    stats = _LinguaPreRankStats(candidates_input=len(items))
    if not items:
        return [], stats
    if not cfg.lingua_annotate_enable or learning_mode not in SUPPORTED_LINGUA_LEARNING_MODES:
        stats.candidates_output = len(items)
        return list(items), stats

    max_items = int(cfg.lingua_annotate_max_items or 0)
    target_items = items[:max_items] if max_items > 0 else items
    untouched_tail = items[max_items:] if max_items > 0 else []
    if not target_items:
        stats.candidates_output = len(items)
        return list(items), stats

    candidate_entries = [
        {"id": f"candidate_{idx:06d}", "item": item}
        for idx, item in enumerate(target_items)
    ]
    phrase_units: list[dict[str, Any]] = []
    for entry in candidate_entries:
        candidate_id = str(entry.get("id") or "").strip()
        item = entry.get("item", {})
        phrase_values = [str(x).strip() for x in (item.get("target_phrases") or []) if str(x).strip()]
        for phrase_idx, phrase in enumerate(phrase_values):
            phrase_units.append(
                {
                    "id": f"{candidate_id}_phrase_{phrase_idx:02d}",
                    "candidate_id": candidate_id,
                    "phrase_index": phrase_idx,
                    "phrase_text": phrase,
                    "text_original": str(item.get("original", "")).strip(),
                    "text_cloze": str(item.get("text", "")).strip(),
                    "local_context": str(item.get("original", "")).strip(),
                    "difficulty": cfg.cloze_difficulty,
                    "learning_mode": learning_mode,
                }
            )
    stats.phrases_total = len(phrase_units)
    if not phrase_units:
        stats.candidates_output = len(items)
        return list(items), stats

    phrase_decisions: dict[str, dict[str, Any]] = {}
    batch_size = max(1, int(cfg.lingua_annotate_batch_size or 50))
    for batch in iter_batches(phrase_units, batch_size):
        remaining = list(batch)
        attempt = 0
        while remaining:
            attempt += 1
            classifier_input = [
                {
                    "id": str(entry.get("id") or "").strip(),
                    "phrase_text": str(entry.get("phrase_text", "")).strip(),
                    "text_original": str(entry.get("text_original", "")).strip(),
                    "text_cloze": str(entry.get("text_cloze", "")).strip(),
                    "local_context": str(entry.get("local_context", "")).strip(),
                    "difficulty": cfg.cloze_difficulty,
                    "learning_mode": learning_mode,
                }
                for entry in remaining
            ]
            try:
                annotations = classify_fn(
                    client=client,
                    items=classifier_input,
                    temperature=temperature,
                    allow_partial=True,
                    source_lang=source_lang,
                )
            except ClawLearnError as exc:
                if _is_retryable_translation_batch_error(exc) and attempt < TRANSLATION_BATCH_MAX_RETRIES:
                    stats.partial_retry_count += 1
                    errors.append(
                        {
                            "stage": "lingua_taxonomy_pre_rank_batch_retry",
                            "attempt": attempt,
                            "remaining": len(remaining),
                            "error": exc.to_lines(),
                        }
                    )
                    delay = _translation_retry_delay_seconds(cfg, attempt)
                    if delay > 0:
                        time.sleep(delay)
                    continue
                stats.batch_error_count += 1
                stats.exhausted_count += len(remaining)
                errors.append(
                    {
                        "stage": "lingua_taxonomy_pre_rank_batch_error",
                        "batch_size": len(batch),
                        "remaining": len(remaining),
                        "attempt": attempt,
                        "error": exc.to_lines(),
                    }
                )
                for entry in remaining:
                    pid = str(entry.get("id") or "").strip()
                    if not pid:
                        continue
                    phrase_decisions[pid] = {
                        "label": "none",
                        "keep": False,
                        "confidence": 0.0,
                    }
                break

            remaining_by_id = {
                str(entry.get("id") or "").strip(): entry
                for entry in remaining
                if str(entry.get("id") or "").strip()
            }
            processed_ids: set[str] = set()
            for payload in annotations:
                payload_id = str((payload or {}).get("id") or "").strip()
                if not payload_id or payload_id in processed_ids:
                    continue
                if payload_id not in remaining_by_id:
                    continue
                label = str(payload.get("label") or "").strip().lower()
                keep = bool(payload.get("keep", False))
                confidence = payload.get("confidence", 0.0)
                try:
                    confidence_f = float(confidence)
                except (TypeError, ValueError):
                    confidence_f = 0.0
                if label in {"", "none", "null"}:
                    label = "none"
                    keep = False
                phrase_decisions[payload_id] = {
                    "label": label,
                    "keep": keep and label != "none",
                    "confidence": max(0.0, min(1.0, confidence_f)),
                }
                processed_ids.add(payload_id)

            usable_count = len(processed_ids)
            if usable_count == 0:
                if attempt < TRANSLATION_BATCH_MAX_RETRIES:
                    stats.partial_retry_count += 1
                    errors.append(
                        {
                            "stage": "lingua_taxonomy_pre_rank_batch_partial_retry",
                            "attempt": attempt,
                            "received": 0,
                            "remaining": len(remaining),
                        }
                    )
                    delay = _translation_retry_delay_seconds(cfg, attempt)
                    if delay > 0:
                        time.sleep(delay)
                    continue
                stats.batch_error_count += 1
                stats.exhausted_count += len(remaining)
                errors.append(
                    {
                        "stage": "lingua_taxonomy_pre_rank_batch_error",
                        "batch_size": len(batch),
                        "remaining": len(remaining),
                        "attempt": attempt,
                        "error": [
                            "lingua taxonomy pre-rank partial response exhausted",
                            f"attempts={TRANSLATION_BATCH_MAX_RETRIES}",
                        ],
                    }
                )
                for entry in remaining:
                    pid = str(entry.get("id") or "").strip()
                    if not pid:
                        continue
                    phrase_decisions[pid] = {
                        "label": "none",
                        "keep": False,
                        "confidence": 0.0,
                    }
                break

            remaining = [entry for entry in remaining if str(entry.get("id") or "").strip() not in processed_ids]
            if not remaining:
                break
            if attempt < TRANSLATION_BATCH_MAX_RETRIES:
                stats.partial_retry_count += 1
                errors.append(
                    {
                        "stage": "lingua_taxonomy_pre_rank_batch_partial_retry",
                        "attempt": attempt,
                        "received": usable_count,
                        "remaining": len(remaining),
                    }
                )
                delay = _translation_retry_delay_seconds(cfg, attempt)
                if delay > 0:
                    time.sleep(delay)
                continue
            stats.batch_error_count += 1
            stats.exhausted_count += len(remaining)
            errors.append(
                {
                    "stage": "lingua_taxonomy_pre_rank_batch_error",
                    "batch_size": len(batch),
                    "remaining": len(remaining),
                    "attempt": attempt,
                    "error": [
                        "lingua taxonomy pre-rank partial response exhausted",
                        f"attempts={TRANSLATION_BATCH_MAX_RETRIES}",
                    ],
                }
            )
            for entry in remaining:
                pid = str(entry.get("id") or "").strip()
                if not pid:
                    continue
                phrase_decisions[pid] = {
                    "label": "none",
                    "keep": False,
                    "confidence": 0.0,
                }
            break

    output: list[dict[str, Any]] = []
    for entry in candidate_entries:
        candidate_id = str(entry.get("id") or "").strip()
        item = dict(entry.get("item", {}))
        phrases = [str(x).strip() for x in (item.get("target_phrases") or []) if str(x).strip()]
        keep_indices: set[int] = set()
        labels: list[str] = []
        used_quality_fallback = False
        phrase_debug_rows: list[dict[str, Any]] = []
        for phrase_idx in range(len(phrases)):
            pid = f"{candidate_id}_phrase_{phrase_idx:02d}"
            decision = phrase_decisions.get(pid)
            if decision is None:
                decision = {"label": "none", "keep": False, "confidence": 0.0}
            label = str(decision.get("label") or "none").strip().lower() or "none"
            keep = bool(decision.get("keep", False))
            confidence_raw = decision.get("confidence", 0.0)
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            stats.phrases_annotated += 1
            phrase_debug_rows.append(
                {
                    "phrase_index": phrase_idx,
                    "phrase_text": phrases[phrase_idx],
                    "label": label,
                    "keep": keep,
                    "confidence": confidence,
                }
            )
            if keep:
                keep_indices.add(phrase_idx)
                if label and label not in labels:
                    labels.append(label)
                stats.phrases_kept += 1
            else:
                stats.phrases_none += 1

        if not keep_indices:
            fallback_indices = {
                idx
                for idx, phrase in enumerate(phrases)
                if phrase_quality_score(phrase) >= _LINGUA_PRE_RANK_FALLBACK_MIN_SCORE
            }
            if fallback_indices:
                keep_indices = fallback_indices
                used_quality_fallback = True
                rescued_count = len(fallback_indices)
                stats.phrases_kept += rescued_count
                stats.phrases_none = max(0, stats.phrases_none - rescued_count)
                errors.append(
                    {
                        "stage": "lingua_taxonomy_pre_rank_all_none_fallback",
                        "candidate_id": candidate_id,
                        "rescued_count": rescued_count,
                        "phrase_count": len(phrases),
                        "score_threshold": _LINGUA_PRE_RANK_FALLBACK_MIN_SCORE,
                    }
                )

        rewritten = _rewrite_candidate_by_kept_phrase_indices(
            item=item,
            kept_phrase_indices=keep_indices,
        )
        if rewritten is None:
            stats.candidates_context_only += 1
            if not keep_indices:
                errors.append(
                    {
                        "stage": "lingua_taxonomy_pre_rank_context_only_drop",
                        "candidate_id": candidate_id,
                        "chunk_id": str(item.get("chunk_id") or "").strip(),
                        "original": str(item.get("original") or "").strip(),
                        "phrases": phrase_debug_rows,
                    }
                )
            continue
        if labels:
            rewritten["phrase_types"] = labels[:2]
        elif used_quality_fallback:
            rewritten["phrase_types"] = normalize_phrase_types(item.get("phrase_types"), max_items=2)
        else:
            rewritten["phrase_types"] = []
        rewritten_after_quality, dropped_by_quality = _apply_advanced_phrase_quality_gate(
            item=rewritten,
            source_lang=source_lang,
            difficulty=cfg.cloze_difficulty,
        )
        if dropped_by_quality > 0:
            stats.phrases_none += dropped_by_quality
            stats.phrases_kept = max(0, stats.phrases_kept - dropped_by_quality)
            errors.append(
                {
                    "stage": "lingua_taxonomy_pre_rank_phrase_quality_drop",
                    "candidate_id": candidate_id,
                    "dropped_count": dropped_by_quality,
                }
            )
        if rewritten_after_quality is None:
            stats.candidates_context_only += 1
            continue
        output.append(rewritten_after_quality)

    output.extend(untouched_tail)
    stats.candidates_output = len(output)
    stats.phrases_dropped = max(0, stats.phrases_total - stats.phrases_kept)
    return output, stats


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
    lingua_pre_rank_stats = _LinguaPreRankStats()
    phrase_filter_stats = _empty_phrase_filter_stats(
        enabled=False,
        source_lang=document.source_lang,
    )
    secondary_requested = bool(
        cfg.secondary_extract_enable
        if options.secondary_extract_enable is None
        else options.secondary_extract_enable
    )
    secondary_extract_stats = _empty_secondary_extract_stats(
        requested=secondary_requested,
        model=str(cfg.secondary_extract_llm_model or "").strip(),
        parallel_requested=bool(cfg.secondary_extract_parallel),
    )

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

    def _append_raw_items(
        batch: list[Any],
        items: list[dict[str, Any]],
        *,
        sink: list[dict[str, Any]],
        stage_name: str,
        fail_hard: bool,
    ) -> None:
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
                        "stage": f"{stage_name}_batch_mapping",
                        "reason": "missing_or_unknown_chunk_id",
                        "chunk_id": cid,
                        "item": item,
                    }
                )
                if not fail_hard:
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
            sink.append(item)

    def _extract_batch_items(
        *,
        client_for_pass: OpenAICompatibleClient,
        batch: list[Any],
    ) -> list[dict[str, Any]]:
        if len(batch) == 1:
            chunk = batch[0]
            if use_phrase_pipeline:
                return generate_phrase_candidates_for_chunk(
                    client=client_for_pass,
                    prompt=extract_prompt,
                    document=document,
                    chunk=chunk,
                    temperature=options.temperature,
                    events=errors,
                )
            return generate_cloze_candidates_for_chunk(
                client=client_for_pass,
                prompt=extract_prompt,
                document=document,
                chunk=chunk,
                temperature=options.temperature,
                events=errors,
            )

        if use_phrase_pipeline:
            return generate_phrase_candidates_for_batch(
                client=client_for_pass,
                prompt=extract_prompt,
                document=document,
                chunks=batch,
                temperature=options.temperature,
                events=errors,
            )
        return generate_cloze_candidates_for_batch(
            client=client_for_pass,
            prompt=extract_prompt,
            document=document,
            chunks=batch,
            temperature=options.temperature,
            events=errors,
        )

    def _run_extraction_pass(
        *,
        stage_name: str,
        client_for_pass: OpenAICompatibleClient,
        batch_size_for_pass: int,
        sink: list[dict[str, Any]],
        fail_hard: bool,
    ) -> Exception | None:
        last_error: Exception | None = None
        for batch in iter_batches(chunks, batch_size_for_pass):
            try:
                items = _extract_batch_items(
                    client_for_pass=client_for_pass,
                    batch=batch,
                )
                _append_raw_items(
                    batch,
                    items,
                    sink=sink,
                    stage_name=stage_name,
                    fail_hard=fail_hard,
                )
            except ClawLearnError as exc:
                last_error = exc
                if fail_hard:
                    _save_failure_snapshot(exc)
                    raise
                errors.append(
                    {
                        "stage": stage_name,
                        "chunk_ids": [c.chunk_id for c in batch],
                        "error": exc.to_lines(),
                    }
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                last_error = exc
                if fail_hard:
                    raise
                errors.append(
                    {
                        "stage": stage_name,
                        "chunk_ids": [c.chunk_id for c in batch],
                        "error": [str(exc)],
                    }
                )
        return last_error

    primary_raw_extraction_candidates: list[dict[str, Any]] = []
    secondary_raw_extraction_candidates: list[dict[str, Any]] = []
    raw_extraction_candidates = list(primary_raw_extraction_candidates)

    secondary_parallel_enabled = False
    secondary_model_name = str(cfg.secondary_extract_llm_model or "").strip()
    secondary_batch_size = max(1, int(batch_size))
    secondary_client: OpenAICompatibleClient | None = None
    if secondary_requested:
        if not secondary_model_name:
            secondary_extract_stats["fallback_to_primary"] = True
            errors.append(
                {
                    "stage": "secondary_extract_disabled_missing_config",
                    "reason": "secondary model is missing",
                }
            )
            logger.warning("secondary extraction requested but secondary model is not configured")
        else:
            secondary_extract_stats["configured"] = True
            secondary_extract_stats["enabled"] = True
            secondary_parallel_enabled = bool(cfg.secondary_extract_parallel)
            secondary_extract_stats["parallel"] = secondary_parallel_enabled
            secondary_extract_stats["execution_mode"] = (
                "parallel" if secondary_parallel_enabled else "serial"
            )
            secondary_cfg = cfg.model_copy(deep=True)
            secondary_cfg.llm_model = secondary_model_name
            if cfg.secondary_extract_llm_base_url:
                secondary_cfg.llm_base_url = cfg.secondary_extract_llm_base_url
            if cfg.secondary_extract_llm_api_key:
                secondary_cfg.llm_api_key = cfg.secondary_extract_llm_api_key
            if cfg.secondary_extract_llm_timeout_seconds is not None:
                secondary_cfg.llm_timeout_seconds = cfg.secondary_extract_llm_timeout_seconds
            if cfg.secondary_extract_llm_temperature is not None:
                secondary_cfg.llm_temperature = cfg.secondary_extract_llm_temperature
            if cfg.secondary_extract_llm_max_retries is not None:
                secondary_cfg.llm_max_retries = cfg.secondary_extract_llm_max_retries
            if cfg.secondary_extract_llm_retry_backoff_seconds is not None:
                secondary_cfg.llm_retry_backoff_seconds = (
                    cfg.secondary_extract_llm_retry_backoff_seconds
                )
            if cfg.secondary_extract_llm_chunk_batch_size is not None:
                secondary_cfg.llm_chunk_batch_size = (
                    cfg.secondary_extract_llm_chunk_batch_size
                )
            secondary_client = OpenAICompatibleClient(secondary_cfg)
            secondary_batch_size = max(1, int(secondary_cfg.llm_chunk_batch_size or 1))
            logger.info(
                "secondary extraction mode | parallel=%s primary_batch=%d secondary_batch=%d",
                secondary_parallel_enabled,
                batch_size,
                secondary_batch_size,
            )

    def _run_primary_pass() -> Exception | None:
        return _run_extraction_pass(
            stage_name="cloze",
            client_for_pass=client,
            batch_size_for_pass=batch_size,
            sink=primary_raw_extraction_candidates,
            fail_hard=not options.continue_on_error,
        )

    def _run_secondary_pass() -> Exception | None:
        if secondary_client is None:
            return None
        return _run_extraction_pass(
            stage_name="secondary_extract",
            client_for_pass=secondary_client,
            batch_size_for_pass=secondary_batch_size,
            sink=secondary_raw_extraction_candidates,
            fail_hard=False,
        )

    _primary_error, secondary_error = _run_extraction_passes(
        run_primary=_run_primary_pass,
        run_secondary=_run_secondary_pass if secondary_client is not None else None,
        secondary_parallel=secondary_parallel_enabled,
    )
    secondary_extract_stats["candidates_primary_count"] = len(primary_raw_extraction_candidates)
    secondary_extract_stats["candidates_secondary_count"] = len(secondary_raw_extraction_candidates)
    raw_extraction_candidates, unique_gain = _merge_secondary_extraction_candidates(
        use_phrase_pipeline=use_phrase_pipeline,
        primary_items=primary_raw_extraction_candidates,
        secondary_items=secondary_raw_extraction_candidates,
    )
    secondary_extract_stats["candidates_merged_count"] = len(raw_extraction_candidates)
    secondary_extract_stats["dedup_removed_count"] = max(
        0,
        len(primary_raw_extraction_candidates)
        + len(secondary_raw_extraction_candidates)
        - len(raw_extraction_candidates),
    )
    secondary_extract_stats["unique_phrase_gain_from_secondary"] = unique_gain
    if secondary_error is not None:
        secondary_extract_stats["secondary_error_type"] = _secondary_error_type(
            secondary_error
        )
        secondary_extract_stats["secondary_error_message"] = str(secondary_error).strip()
        if not secondary_raw_extraction_candidates:
            secondary_extract_stats["fallback_to_primary"] = True
        logger.warning(
            "secondary extraction had errors | type=%s msg=%s",
            secondary_extract_stats["secondary_error_type"],
            secondary_extract_stats["secondary_error_message"],
        )

    if use_phrase_pipeline:
        raw_candidates, initial_phrase_filter_stats = _materialize_cloze_candidates_from_phrase_items(
            raw_extraction_candidates,
            source_lang=document.source_lang,
            learning_mode=learning_mode,
            difficulty=cfg.cloze_difficulty,
        )
        phrase_filter_stats = _merge_phrase_filter_stats(phrase_filter_stats, initial_phrase_filter_stats)
        if initial_phrase_filter_stats.get("dropped_count", 0):
            errors.append(
                {
                    "stage": "phrase_filter",
                    "dropped_count": int(initial_phrase_filter_stats.get("dropped_count", 0)),
                    "dropped_by_rule": dict(initial_phrase_filter_stats.get("dropped_by_rule", {})),
                    "examples_by_rule": dict(initial_phrase_filter_stats.get("examples_by_rule", {})),
                }
            )
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

    should_fail_for_empty_raw, empty_raw_count, empty_raw_ratio = _should_fail_for_structurally_empty_raw_candidates(
        raw_candidates
    )
    if should_fail_for_empty_raw:
        errors.append(
            {
                "stage": "extract_shape_guard",
                "reason": "structurally_empty_raw_candidates",
                "empty_raw_candidates": empty_raw_count,
                "raw_candidates": len(raw_candidates),
                "empty_ratio": round(empty_raw_ratio, 4),
                "schema_name": extract_prompt.output_format.schema_name,
            }
        )
        err = build_error(
            error_code="LLM_RESPONSE_SHAPE_INVALID",
            cause="Extraction output shape is incompatible with the active parser.",
            detail=(
                f"raw={len(raw_candidates)} empty={empty_raw_count} ratio={empty_raw_ratio:.2%} "
                f"schema={extract_prompt.output_format.schema_name}"
            ),
            next_steps=[
                "Verify extraction prompt schema_name matches the pipeline parser path",
                "Ensure extraction outputs include text/original/target_phrases for cloze schemas",
            ],
            exit_code=ExitCode.LLM_PARSE_ERROR,
        )
        _save_failure_snapshot(err)
        raise err

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
            _materialize_cloze_candidates_from_phrase_items(
                fallback_raw_extraction,
                source_lang=document.source_lang,
                learning_mode=learning_mode,
                difficulty=cfg.cloze_difficulty,
            )
            if use_phrase_pipeline
            else fallback_raw_extraction
        )
        if use_phrase_pipeline:
            fallback_raw, fallback_phrase_filter_stats = fallback_raw
            phrase_filter_stats = _merge_phrase_filter_stats(phrase_filter_stats, fallback_phrase_filter_stats)
            if fallback_phrase_filter_stats.get("dropped_count", 0):
                errors.append(
                    {
                        "stage": "phrase_filter_fallback",
                        "dropped_count": int(fallback_phrase_filter_stats.get("dropped_count", 0)),
                        "dropped_by_rule": dict(fallback_phrase_filter_stats.get("dropped_by_rule", {})),
                        "examples_by_rule": dict(fallback_phrase_filter_stats.get("examples_by_rule", {})),
                    }
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

    pre_rank_annotated_count = 0
    if valid_candidates:
        valid_candidates, lingua_pre_rank_stats = _annotate_lingua_candidates_pre_rank(
            items=valid_candidates,
            cfg=cfg,
            learning_mode=learning_mode,
            source_lang=document.source_lang,
            client=translate_client,
            errors=errors,
            temperature=options.temperature,
        )
        pre_rank_annotated_count = int(lingua_pre_rank_stats.phrases_annotated)
        if pre_rank_annotated_count > 0:
            logger.info(
                "lingua taxonomy pre-rank complete | annotated=%d batch_size=%d",
                pre_rank_annotated_count,
                max(1, int(cfg.lingua_annotate_batch_size or 50)),
            )
        if not valid_candidates:
            errors.append(
                {
                    "stage": "lingua_taxonomy_pre_rank_empty",
                    "reason": "all candidates degraded to context-only after phrase-level taxonomy",
                }
            )
        elif cfg.lingua_annotate_enable and learning_mode in SUPPORTED_LINGUA_LEARNING_MODES:
            valid_candidates, empty_label_dropped = _drop_candidates_without_taxonomy_labels(
                items=valid_candidates,
                errors=errors,
            )
            lingua_pre_rank_stats.candidates_empty_label_dropped = int(empty_label_dropped)
            if empty_label_dropped > 0:
                logger.info(
                    "lingua taxonomy hard gate dropped empty-label candidates | dropped=%d",
                    empty_label_dropped,
                )
            if not valid_candidates:
                errors.append(
                    {
                        "stage": "lingua_taxonomy_pre_rank_empty",
                        "reason": "all candidates dropped by taxonomy hard gate",
                    }
                )
        ranked_candidates = rank_candidates(
            valid_candidates,
            difficulty=cfg.cloze_difficulty,
            material_profile=material_profile,
            learning_mode=learning_mode,
        )
        if (
            (cfg.cloze_difficulty or "").strip().lower() == "advanced"
            and learning_mode == "lingua_expression"
        ):
            ranked_candidates = _apply_contrastive_rerank(ranked_candidates)
        deduped = dedupe_candidates(ranked_candidates)
        if (
            (cfg.cloze_difficulty or "").strip().lower() == "advanced"
            and learning_mode == "lingua_expression"
        ):
            deduped = _apply_phrase_diversity_cap(deduped, max_per_primary_phrase=2)
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
    context_expanded_count = _apply_transcript_context_fallback(
        items=deduped,
        material_profile=material_profile,
        learning_mode=learning_mode,
        min_sentences=cfg.lingua_transcript_min_context_sentences,
    )
    if context_expanded_count > 0:
        logger.info(
            "transcript context fallback expanded | count=%d min_sentences=%d",
            context_expanded_count,
            max(2, int(cfg.lingua_transcript_min_context_sentences or 2)),
        )
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

    cards_before_export_guard = len(cards)
    if cards:
        cards = _filter_cards_for_export_cloze_format(
            cards=cards,
            errors=errors,
        )
    dropped_by_export_guard = max(0, cards_before_export_guard - len(cards))
    if dropped_by_export_guard > 0:
        logger.warning(
            "export cloze format guard dropped cards | dropped=%d",
            dropped_by_export_guard,
        )

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
        phrase_filter_stats=phrase_filter_stats,
        secondary_extract_stats=secondary_extract_stats,
    )
    pipeline_metrics["taxonomy_reject_count"] = taxonomy_repair_stats.taxonomy_reject_count
    pipeline_metrics["taxonomy_repair_attempted"] = taxonomy_repair_stats.taxonomy_repair_attempted
    pipeline_metrics["taxonomy_repair_success"] = taxonomy_repair_stats.taxonomy_repair_success
    pipeline_metrics["taxonomy_repair_failed"] = taxonomy_repair_stats.taxonomy_repair_failed
    pipeline_metrics["taxonomy_pre_rank"] = {
        "schema_version": "taxonomy_v2_phrase_level",
        "candidates_input": lingua_pre_rank_stats.candidates_input,
        "candidates_output": lingua_pre_rank_stats.candidates_output,
        "candidates_context_only": lingua_pre_rank_stats.candidates_context_only,
        "candidates_empty_label_dropped": lingua_pre_rank_stats.candidates_empty_label_dropped,
        "phrases_total": lingua_pre_rank_stats.phrases_total,
        "phrases_annotated": lingua_pre_rank_stats.phrases_annotated,
        "phrases_kept": lingua_pre_rank_stats.phrases_kept,
        "phrases_none": lingua_pre_rank_stats.phrases_none,
        "phrases_dropped": lingua_pre_rank_stats.phrases_dropped,
        "phrase_hit_rate": round(
            lingua_pre_rank_stats.phrases_kept / lingua_pre_rank_stats.phrases_total,
            4,
        )
        if lingua_pre_rank_stats.phrases_total
        else 0.0,
        "phrase_none_rate": round(
            lingua_pre_rank_stats.phrases_none / lingua_pre_rank_stats.phrases_total,
            4,
        )
        if lingua_pre_rank_stats.phrases_total
        else 0.0,
        "partial_retry_count": lingua_pre_rank_stats.partial_retry_count,
        "exhausted_count": lingua_pre_rank_stats.exhausted_count,
        "batch_error_count": lingua_pre_rank_stats.batch_error_count,
    }
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
