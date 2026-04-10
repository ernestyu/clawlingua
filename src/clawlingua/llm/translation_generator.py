"""Generate translation text from the translation prompt."""

from __future__ import annotations

from dataclasses import dataclass
import json

from ..models.document import DocumentRecord
from ..models.prompt_schema import PromptSpec
from .client import OpenAICompatibleClient
from .response_parser import parse_json_content


@dataclass
class TranslationBatchResult:
    ok: bool
    translation: str | None = None
    error: str | None = None


def _render(template: str, values: dict[str, str]) -> str:
    return template.format(**values)


def _normalize_translation_item(item: object) -> TranslationBatchResult:
    if isinstance(item, dict):
        if "translation" not in item and "translated_text" not in item:
            return TranslationBatchResult(ok=False, error="retryable:missing_translation_field")
        value = str(item.get("translation") or item.get("translated_text") or "").strip()
        if not value:
            return TranslationBatchResult(ok=False, error="validation:empty_translation")
        return TranslationBatchResult(ok=True, translation=value)

    if isinstance(item, str):
        value = item.strip()
        if not value:
            return TranslationBatchResult(ok=False, error="validation:empty_translation")
        return TranslationBatchResult(ok=True, translation=value)

    return TranslationBatchResult(ok=False, error="retryable:invalid_item_type")


def generate_translation_batch(
    *,
    client: OpenAICompatibleClient,
    prompt: PromptSpec,
    document: DocumentRecord,
    chunk_text: str,
    text_originals: list[str],
    temperature: float | None = None,
) -> list[TranslationBatchResult]:
    if not text_originals:
        return []

    originals_json = json.dumps(text_originals, ensure_ascii=False, indent=2)
    placeholders = {
        "source_lang": document.source_lang,
        "target_lang": document.target_lang,
        "document_title": document.title or "",
        "source_url": document.source_url or "",
        "chunk_text": chunk_text,
        # Backward-compatible placeholder; many templates still reference this key.
        "text_original": originals_json,
        "text_originals_json": originals_json,
        "batch_size": str(len(text_originals)),
    }

    user_prompt = _render(prompt.user_prompt_template, placeholders)
    batch_contract = (
        "Input is a JSON array named text_originals_json.\n"
        f"Translate each item to {document.target_lang} and return a JSON array with exactly {len(text_originals)} items.\n"
        "Position must match the input order.\n"
        "Each output item must be an object with key \"translation\" only.\n"
        "Do not include explanations or markdown."
    )
    content = client.chat(
        [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": f"{batch_contract}\n\n{user_prompt}"},
        ],
        temperature=temperature,
        # Batch-level retries are controlled in pipeline (max 3 attempts).
        max_retries=1,
    )
    # Batch translation always uses array-in/array-out protocol, regardless of
    # prompt parser flags, to keep retry/error handling deterministic.
    data = parse_json_content(content, expect_array=True)
    return [_normalize_translation_item(item) for item in data]


def generate_translation(
    *,
    client: OpenAICompatibleClient,
    prompt: PromptSpec,
    document: DocumentRecord,
    chunk_text: str,
    text_original: str,
    temperature: float | None = None,
) -> str:
    results = generate_translation_batch(
        client=client,
        prompt=prompt,
        document=document,
        chunk_text=chunk_text,
        text_originals=[text_original],
        temperature=temperature,
    )
    if not results:
        return ""
    first = results[0]
    return (first.translation or "").strip() if first.ok else ""
