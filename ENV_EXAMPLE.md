CLAWLEARN_DEFAULT_SOURCE_LANG=en
CLAWLEARN_DEFAULT_TARGET_LANG=zh

CLAWLEARN_LLM_PROVIDER=openai_compatible
CLAWLEARN_LLM_BASE_URL=http://127.0.0.1:8000/v1
CLAWLEARN_LLM_API_KEY=YOUR_API_KEY
CLAWLEARN_LLM_MODEL=qwen3-30b
CLAWLEARN_LLM_TIMEOUT_SECONDS=120
CLAWLEARN_LLM_MAX_RETRIES=3
CLAWLEARN_LLM_RETRY_BACKOFF_SECONDS=2.0
CLAWLEARN_LLM_REQUEST_SLEEP_SECONDS=0
CLAWLEARN_LLM_TEMPERATURE=0.2

# Pre-LLM line filter: drop lines with <= N words. Set 0 to disable.
CLAWLEARN_INGEST_SHORT_LINE_MAX_WORDS=3

CLAWLEARN_CHUNK_MAX_CHARS=1800
CLAWLEARN_CHUNK_MIN_CHARS=120
CLAWLEARN_CHUNK_OVERLAP_SENTENCES=1

CLAWLEARN_CLOZE_MAX_SENTENCES=3
CLAWLEARN_CLOZE_MIN_CHARS=200
CLAWLEARN_CLOZE_DIFFICULTY=intermediate
CLAWLEARN_CLOZE_MAX_PER_CHUNK=4
CLAWLEARN_LLM_CHUNK_BATCH_SIZE=1

# Format-only validation retry controls.
CLAWLEARN_VALIDATE_FORMAT_RETRY_ENABLE=true
# Retry attempts after initial validation failure (0-3).
CLAWLEARN_VALIDATE_FORMAT_RETRY_MAX=3
# If true, attempts >=2 may call LLM repair/regenerate.
CLAWLEARN_VALIDATE_FORMAT_RETRY_LLM_ENABLE=true
# If true, taxonomy-related rejects get one small-model repair pass.
CLAWLEARN_TAXONOMY_REPAIR_ENABLE=false

# Prompt language: en | zh
CLAWLEARN_PROMPT_LANG=zh

# Optional translation LLM. Leave empty to reuse main LLM.
CLAWLEARN_TRANSLATE_LLM_BASE_URL=
CLAWLEARN_TRANSLATE_LLM_API_KEY=
CLAWLEARN_TRANSLATE_LLM_MODEL=
CLAWLEARN_TRANSLATE_LLM_TEMPERATURE=
# Number of originals translated in one request (recommended: 4-8).
CLAWLEARN_TRANSLATE_BATCH_SIZE=4

# Secondary extraction (dual-LLM phrase extraction). If enabled, a secondary LLM
# runs an additional extraction pass and candidates are merged/deduped.
CLAWLEARN_SECONDARY_EXTRACT_ENABLE=false
CLAWLEARN_SECONDARY_EXTRACT_PARALLEL=false
CLAWLEARN_SECONDARY_EXTRACT_LLM_BASE_URL=
CLAWLEARN_SECONDARY_EXTRACT_LLM_API_KEY=
CLAWLEARN_SECONDARY_EXTRACT_LLM_MODEL=
CLAWLEARN_SECONDARY_EXTRACT_LLM_TIMEOUT_SECONDS=
CLAWLEARN_SECONDARY_EXTRACT_LLM_TEMPERATURE=
CLAWLEARN_SECONDARY_EXTRACT_LLM_MAX_RETRIES=
CLAWLEARN_SECONDARY_EXTRACT_LLM_RETRY_BACKOFF_SECONDS=
CLAWLEARN_SECONDARY_EXTRACT_LLM_CHUNK_BATCH_SIZE=

# Legacy alias (kept for backward compatibility): general|textbook_examples
# NOTE: legacy "general" is an alias of "prose_article".
CLAWLEARN_CONTENT_PROFILE=prose_article
# V2 selectors
CLAWLEARN_MATERIAL_PROFILE=prose_article
# learning_mode controls which pipeline + prompt family is used.
# Supported: lingua_expression|lingua_reading (lingua); textbook_focus|textbook_review (textbook).
CLAWLEARN_LEARNING_MODE=lingua_expression

# Legacy generic cloze prompt path (still supported)
CLAWLEARN_PROMPT_CLOZE=./prompts/cloze_contextual.json
CLAWLEARN_PROMPT_CLOZE_TEXTBOOK=./prompts/cloze_textbook_examples.json
CLAWLEARN_PROMPT_CLOZE_PROSE_BEGINNER=./prompts/cloze_prose_beginner.json
CLAWLEARN_PROMPT_CLOZE_PROSE_INTERMEDIATE=./prompts/cloze_prose_intermediate.json
CLAWLEARN_PROMPT_CLOZE_PROSE_ADVANCED=./prompts/cloze_prose_advanced.json
CLAWLEARN_PROMPT_CLOZE_TRANSCRIPT_BEGINNER=./prompts/cloze_transcript_beginner.json
CLAWLEARN_PROMPT_CLOZE_TRANSCRIPT_INTERMEDIATE=./prompts/cloze_transcript_intermediate.json
CLAWLEARN_PROMPT_CLOZE_TRANSCRIPT_ADVANCED=./prompts/cloze_transcript_advanced.json
CLAWLEARN_PROMPT_TRANSLATE=./prompts/translate_rewrite.json
# Preferred default prompt files by role (used when no CLI override).
CLAWLEARN_EXTRACT_PROMPT=
CLAWLEARN_EXPLAIN_PROMPT=
CLAWLEARN_ANKI_TEMPLATE=./templates/anki_cloze_default.json

# Intermediate run data (JSONL, media snapshots)
CLAWLEARN_OUTPUT_DIR=./runs
# Final exported decks (when --output is not provided)
CLAWLEARN_EXPORT_DIR=./outputs
CLAWLEARN_LOG_DIR=./logs
CLAWLEARN_LOG_LEVEL=INFO
CLAWLEARN_SAVE_INTERMEDIATE=true
CLAWLEARN_ALLOW_EMPTY_DECK=true
CLAWLEARN_DEFAULT_DECK_NAME=ClawLearn Default Deck

CLAWLEARN_TTS_PROVIDER=edge_tts
CLAWLEARN_TTS_OUTPUT_FORMAT=mp3
CLAWLEARN_TTS_RATE=+0%
CLAWLEARN_TTS_VOLUME=+0%
CLAWLEARN_TTS_RANDOM_SEED=

CLAWLEARN_TTS_EDGE_VOICE1=en-US-AnaNeural
CLAWLEARN_TTS_EDGE_VOICE2=en-GB-SoniaNeural
CLAWLEARN_TTS_EDGE_VOICE3=en-US-AndrewNeural
CLAWLEARN_TTS_EDGE_VOICE4=en-US-GuyNeural
