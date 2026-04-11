"""Project constants."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "clawlingua"

DEFAULT_PROMPT_CLOZE = Path("./prompts/cloze_contextual.json")
DEFAULT_PROMPT_CLOZE_TEXTBOOK = Path("./prompts/cloze_textbook_examples.json")
DEFAULT_PROMPT_CLOZE_PROSE_BEGINNER = Path("./prompts/cloze_prose_beginner.json")
DEFAULT_PROMPT_CLOZE_PROSE_INTERMEDIATE = Path("./prompts/cloze_prose_intermediate.json")
DEFAULT_PROMPT_CLOZE_PROSE_ADVANCED = Path("./prompts/cloze_prose_advanced.json")
DEFAULT_PROMPT_CLOZE_TRANSCRIPT_BEGINNER = Path("./prompts/cloze_transcript_beginner.json")
DEFAULT_PROMPT_CLOZE_TRANSCRIPT_INTERMEDIATE = Path("./prompts/cloze_transcript_intermediate.json")
DEFAULT_PROMPT_CLOZE_TRANSCRIPT_ADVANCED = Path("./prompts/cloze_transcript_advanced.json")
DEFAULT_PROMPT_CLOZE_PROSE_READING_SUPPORT_BEGINNER = Path(
    "./prompts/cloze_prose_reading_support_beginner.json"
)
DEFAULT_PROMPT_CLOZE_PROSE_READING_SUPPORT_INTERMEDIATE = Path(
    "./prompts/cloze_prose_reading_support_intermediate.json"
)
DEFAULT_PROMPT_CLOZE_PROSE_READING_SUPPORT_ADVANCED = Path(
    "./prompts/cloze_prose_reading_support_advanced.json"
)
DEFAULT_PROMPT_CLOZE_TRANSCRIPT_READING_SUPPORT_BEGINNER = Path(
    "./prompts/cloze_transcript_reading_support_beginner.json"
)
DEFAULT_PROMPT_CLOZE_TRANSCRIPT_READING_SUPPORT_INTERMEDIATE = Path(
    "./prompts/cloze_transcript_reading_support_intermediate.json"
)
DEFAULT_PROMPT_CLOZE_TRANSCRIPT_READING_SUPPORT_ADVANCED = Path(
    "./prompts/cloze_transcript_reading_support_advanced.json"
)
DEFAULT_PROMPT_TRANSLATE = Path("./prompts/translate_rewrite.json")
DEFAULT_ANKI_TEMPLATE = Path("./templates/anki_cloze_default.json")
# Where intermediate run directories (JSONL, snapshots) are stored.
DEFAULT_OUTPUT_DIR = Path("./runs")
# Where final exported decks are stored when no --output is provided.
DEFAULT_EXPORT_DIR = Path("./outputs")
# Where log files are written.
DEFAULT_LOG_DIR = Path("./logs")
DEFAULT_DECK_NAME = "ClawLingua Default Deck"

ANKI_FIELDS_ORDER = ["Text", "Original", "Translation", "ExpressionTransfer", "Note", "Audio"]
SUPPORTED_FILE_SUFFIXES = {".txt", ".md", ".markdown", ".epub"}
SUPPORTED_INPUT_TYPES = {"file"}
# Legacy alias "general" maps to "prose_article".
SUPPORTED_MATERIAL_PROFILES = {"general", "prose_article", "transcript_dialogue", "textbook_examples"}
SUPPORTED_CONTENT_PROFILES = SUPPORTED_MATERIAL_PROFILES
SUPPORTED_LEARNING_MODES = {"expression_mining", "reading_support"}
