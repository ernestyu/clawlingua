"""LLM helpers."""

from .client import OpenAICompatibleClient
from .cloze_generator import (
    generate_cloze_candidates_for_batch,
    generate_cloze_candidates_for_chunk,
)
from .prompt_loader import load_prompt
from .translation_generator import generate_translation

__all__ = [
    "OpenAICompatibleClient",
    "load_prompt",
    "generate_cloze_candidates_for_batch",
    "generate_cloze_candidates_for_chunk",
    "generate_translation",
]

