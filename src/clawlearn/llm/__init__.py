"""LLM helpers."""

from .client import OpenAICompatibleClient
from .cloze_generator import (
    generate_cloze_candidates_for_batch,
    generate_cloze_candidates_for_chunk,
)
from .prompt_loader import load_prompt
from .translation_generator import (
    TranslationBatchResult,
    generate_translation,
    generate_translation_batch,
)

__all__ = [
    "OpenAICompatibleClient",
    "load_prompt",
    "generate_cloze_candidates_for_batch",
    "generate_cloze_candidates_for_chunk",
    "TranslationBatchResult",
    "generate_translation",
    "generate_translation_batch",
]
