"""Project constants."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "clawlearn"

DEFAULT_ANKI_TEMPLATE = Path("./templates/anki_cloze_default.json")
# Where intermediate run directories (JSONL, snapshots) are stored.
DEFAULT_OUTPUT_DIR = Path("./runs")
# Where final exported decks are stored when no --output is provided.
DEFAULT_EXPORT_DIR = Path("./outputs")
# Where log files are written.
DEFAULT_LOG_DIR = Path("./logs")
DEFAULT_DECK_NAME = "ClawLearn Default Deck"

ANKI_FIELDS_ORDER = ["Text", "Original", "Translation", "ExpressionTransfer", "Note", "Audio"]
SUPPORTED_FILE_SUFFIXES = {".txt", ".md", ".markdown", ".epub"}
SUPPORTED_INPUT_TYPES = {"file"}
# Legacy alias "general" maps to "prose_article".
SUPPORTED_MATERIAL_PROFILES = {"general", "prose_article", "transcript_dialogue", "textbook_examples"}
SUPPORTED_CONTENT_PROFILES = SUPPORTED_MATERIAL_PROFILES
SUPPORTED_LINGUA_LEARNING_MODES = {"lingua_expression", "lingua_reading"}
SUPPORTED_TEXTBOOK_LEARNING_MODES = {"textbook_focus", "textbook_review"}
SUPPORTED_LEARNING_MODES = SUPPORTED_LINGUA_LEARNING_MODES | SUPPORTED_TEXTBOOK_LEARNING_MODES
