"""Long-form help text constants."""

from __future__ import annotations

BUILD_DECK_HELP = (
    "Build an Anki .apkg from local text/markdown/EPUB file input.\n\n"
    "Input: .txt, .md, .epub\n"
    "Profiles: general, textbook_examples\n"
    "Output: .apkg deck with fields Text/Original/Translation/Note/Audio\n"
    "Text prompt: ./prompts/cloze_contextual.json\n"
    "Textbook prompt: ./prompts/cloze_textbook_examples.json\n"
    "Translation prompt: ./prompts/translate_rewrite.json\n"
    "Template: ./templates/anki_cloze_default.json"
)

DOCTOR_HELP = "Check environment, config, prompt/template schemas, LLM connectivity, TTS and output path."
INIT_HELP = "Initialize runtime files and verify required prompt/template assets."
CONFIG_VALIDATE_HELP = "Validate merged runtime configuration."
CONFIG_SHOW_HELP = "Show merged runtime configuration with secret masking."
PROMPT_VALIDATE_HELP = "Validate prompt JSON schema file."
