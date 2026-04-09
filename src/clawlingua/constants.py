"""Project constants."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "clawlingua"

DEFAULT_PROMPT_CLOZE = Path("./prompts/cloze_contextual.json")
DEFAULT_PROMPT_TRANSLATE = Path("./prompts/translate_rewrite.json")
DEFAULT_ANKI_TEMPLATE = Path("./templates/anki_cloze_default.json")
DEFAULT_OUTPUT_DIR = Path("./runs")
DEFAULT_DECK_NAME = "ClawLingua Default Deck"

ANKI_FIELDS_ORDER = ["Text", "Original", "Translation", "Note", "Audio"]
SUPPORTED_FILE_SUFFIXES = {".txt", ".md", ".markdown", ".epub", ".pdf"}
SUPPORTED_INPUT_TYPES = {"file"}
