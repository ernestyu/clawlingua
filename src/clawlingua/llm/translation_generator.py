"""Generate translation for a single card original text."""

from __future__ import annotations

from ..models.document import DocumentRecord
from ..models.prompt_schema import PromptSpec
from .client import OpenAICompatibleClient
from .response_parser import parse_json_content


def _render(template: str, values: dict[str, str]) -> str:
    return template.format(**values)


def generate_translation(
    *,
    client: OpenAICompatibleClient,
    prompt: PromptSpec,
    document: DocumentRecord,
    chunk_text: str,
    text_original: str,
    temperature: float | None = None,
) -> str:
    placeholders = {
        "source_lang": document.source_lang,
        "target_lang": document.target_lang,
        "document_title": document.title or "",
        "source_url": document.source_url or "",
        "chunk_text": chunk_text,
        "text_original": text_original,
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
    if not data:
        return ""
    first = data[0]
    if isinstance(first, dict):
        value = first.get("translation") or first.get("translated_text") or ""
        return str(value).strip()
    return str(first).strip()

