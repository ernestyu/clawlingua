# ClawLingua

ClawLingua is a Python CLI that turns real-world text (podcast transcripts, articles, etc.)
into Anki cloze decks (`.apkg`) for language learning.

It:

- ingests content from local `.txt`/`.md`/`.epub` files
- cleans and chunks the text into context blocks
- uses an OpenAI-compatible LLM to generate contextual cloze sentences
- uses a separate (usually cheaper) LLM for translations
- uses `edge_tts` to generate audio for each card
- exports a complete Anki deck via `genanki`
- applies taxonomy-aware candidate ranking in advanced mode
- combines model-proposed labels with programmatic re-ranking corrections
- adds `expression_transfer` hints to capture cross-context reuse intent
- writes taxonomy/validation/transfer metrics into `run_summary.json` for tuning

This README describes the current **V2-oriented CLI**. For an overview in Chinese, see
[`README_zh.md`](./README_zh.md).

---

## 1. Installation

### 1.1 Create a virtualenv

```bash
python -m venv .venv
# Windows (PowerShell)
. .venv/Scripts/activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 1.2 Initialize the project

```bash
# Copy env example and check prompts/templates
python -m clawlingua.cli init
```

`init` will:

- create `.env` from `.env.example` (if missing)
- verify that the default prompts and template exist:
  - `./prompts/cloze_contextual.json`
  - `./prompts/cloze_prose_beginner.json`
  - `./prompts/cloze_prose_intermediate.json`
  - `./prompts/cloze_prose_advanced.json`
  - `./prompts/cloze_transcript_beginner.json`
  - `./prompts/cloze_transcript_intermediate.json`
  - `./prompts/cloze_transcript_advanced.json`
  - `./prompts/cloze_textbook_examples.json`
  - `./prompts/translate_rewrite.json`
  - `./prompts/template_extraction.json`
  - `./prompts/template_explanation.json`
  - `./templates/anki_cloze_default.json`

Edit `.env` (or copy `ENV_EXAMPLE.md` content into your own env file) to point at
your LLM endpoints and TTS voices.

---

## 2. Configuration (ENV_EXAMPLE)

All configuration is done via environment variables. The repository includes
[`ENV_EXAMPLE.md`](./ENV_EXAMPLE.md) describing defaults. Key groups:

### 2.1 LLM (cloze)

```env
CLAWLINGUA_LLM_PROVIDER=openai_compatible
CLAWLINGUA_LLM_BASE_URL=http://127.0.0.1:8000/v1
CLAWLINGUA_LLM_API_KEY=YOUR_API_KEY
CLAWLINGUA_LLM_MODEL=qwen3-30b
CLAWLINGUA_LLM_TIMEOUT_SECONDS=120
CLAWLINGUA_LLM_MAX_RETRIES=3
CLAWLINGUA_LLM_RETRY_BACKOFF_SECONDS=2.0
# Base sleep between successful LLM calls (seconds);
# actual sleep is random in [N, 3N], 0 means no sleep.
CLAWLINGUA_LLM_REQUEST_SLEEP_SECONDS=0
CLAWLINGUA_LLM_TEMPERATURE=0.2
```

The cloze LLM is responsible for generating contextual cloze sentences.
It expects an **OpenAI-compatible** `/chat/completions` endpoint.

### 2.2 Ingest cleaning

```env
CLAWLINGUA_INGEST_SHORT_LINE_MAX_WORDS=3
```

This controls pre-LLM line filtering:

- lines with very few words (for example one-word interjections) are dropped;
- set `CLAWLINGUA_INGEST_SHORT_LINE_MAX_WORDS=0` to disable this filter.
- `.md` input is converted to plain text before filtering.
- `.epub` input is unpacked and chapter HTML is converted to plain text before filtering.

### 2.3 Chunking

Chunking controls how input text is split into context blocks:

```env
# Character-based chunking, with soft sentence boundaries.
CLAWLINGUA_CHUNK_MAX_CHARS=1800
CLAWLINGUA_CHUNK_MIN_CHARS=120
CLAWLINGUA_CHUNK_OVERLAP_SENTENCES=1
```

Behaviour:

- text is split into paragraphs, then short paragraphs are merged until
  `CHUNK_MIN_CHARS` is reached
- each merged paragraph is split into chunks based on `CHUNK_MAX_CHARS`:
  - extend by whole sentences until just below the char limit
  - never split inside a sentence
  - adjacent chunks may share `CHUNK_OVERLAP_SENTENCES` of overlap

### 2.4 Cloze-level controls

These control **per-card** behaviour:

```env
# Max sentences per cloze text (validator + prompt docs)
CLAWLINGUA_CLOZE_MAX_SENTENCES=3

# Min characters per cloze text. Too-short candidates are discarded (0 = no limit).
CLAWLINGUA_CLOZE_MIN_CHARS=200

# Difficulty: beginner | intermediate | advanced
# Can be overridden by CLI --difficulty
CLAWLINGUA_CLOZE_DIFFICULTY=intermediate

# Max number of cards per chunk (after dedupe); empty/0 = no per-chunk cap
CLAWLINGUA_CLOZE_MAX_PER_CHUNK=4

# LLM chunk batch size: how many chunks to process per LLM call; 1 = per-chunk
CLAWLINGUA_LLM_CHUNK_BATCH_SIZE=1

# Retry format-only validation failures (recover candidates rejected for format issues).
CLAWLINGUA_VALIDATE_FORMAT_RETRY_ENABLE=true
# Retry attempts after initial validation failure (0-3).
CLAWLINGUA_VALIDATE_FORMAT_RETRY_MAX=3
# Allow attempts >=2 to call LLM repair/regenerate.
CLAWLINGUA_VALIDATE_FORMAT_RETRY_LLM_ENABLE=true
```

- The LLM decides how many candidates to return per chunk (0-N).
- `CLOZE_MAX_PER_CHUNK` is a **safety cap** applied *after* validation and
  dedupe; set to `0` or empty to disable.
- Difficulty is now a first-class strategy selector (not only a prompt hint):
  it affects prompt family variant, validation, and ranking.
- `CLAWLINGUA_LLM_CHUNK_BATCH_SIZE` is always user-respected (no profile/difficulty hard override).

For `material_profile=textbook_examples`, if `CLOZE_MIN_CHARS` is above 120
and you do not override with `--cloze-min-chars`, the run is rejected.

### 2.5 Translation LLM (small LLM)

```env
# Optional small LLM for translation; falls back to main LLM if empty.
CLAWLINGUA_TRANSLATE_LLM_BASE_URL=
CLAWLINGUA_TRANSLATE_LLM_API_KEY=
CLAWLINGUA_TRANSLATE_LLM_MODEL=
CLAWLINGUA_TRANSLATE_LLM_TEMPERATURE=
# Number of originals translated in one request (recommended: 4-8).
CLAWLINGUA_TRANSLATE_BATCH_SIZE=4
```

- If `TRANSLATE_LLM_BASE_URL`/`MODEL` are set, translations are generated by
  this small LLM (cheaper backend).
- Otherwise, translations use the main cloze LLM.
- `CLAWLINGUA_TRANSLATE_BATCH_SIZE` controls batch size per request (recommended range: 4-8; start from 4).
- Translation batches use request-level retries (max 3 attempts). For partial
  responses, successful items are consumed first and only remaining items are retried.

### 2.6 Prompts, templates, output

```env
CLAWLINGUA_CONTENT_PROFILE=prose_article
CLAWLINGUA_MATERIAL_PROFILE=prose_article
CLAWLINGUA_LEARNING_MODE=expression_mining
CLAWLINGUA_PROMPT_CLOZE=./prompts/cloze_contextual.json
CLAWLINGUA_PROMPT_CLOZE_TEXTBOOK=./prompts/cloze_textbook_examples.json
CLAWLINGUA_PROMPT_CLOZE_PROSE_BEGINNER=./prompts/cloze_prose_beginner.json
CLAWLINGUA_PROMPT_CLOZE_PROSE_INTERMEDIATE=./prompts/cloze_prose_intermediate.json
CLAWLINGUA_PROMPT_CLOZE_PROSE_ADVANCED=./prompts/cloze_prose_advanced.json
CLAWLINGUA_PROMPT_CLOZE_TRANSCRIPT_BEGINNER=./prompts/cloze_transcript_beginner.json
CLAWLINGUA_PROMPT_CLOZE_TRANSCRIPT_INTERMEDIATE=./prompts/cloze_transcript_intermediate.json
CLAWLINGUA_PROMPT_CLOZE_TRANSCRIPT_ADVANCED=./prompts/cloze_transcript_advanced.json
CLAWLINGUA_PROMPT_TRANSLATE=./prompts/translate_rewrite.json
# Preferred default prompt files by role (used when no CLI override)
CLAWLINGUA_EXTRACT_PROMPT=
CLAWLINGUA_EXPLAIN_PROMPT=
CLAWLINGUA_PROMPT_LANG=zh
CLAWLINGUA_ANKI_TEMPLATE=./templates/anki_cloze_default.json

# Intermediate run data (JSONL, media snapshots)
CLAWLINGUA_OUTPUT_DIR=./runs
# Final exported decks (when --output is not provided)
CLAWLINGUA_EXPORT_DIR=./outputs
CLAWLINGUA_LOG_DIR=./logs
CLAWLINGUA_LOG_LEVEL=INFO
CLAWLINGUA_SAVE_INTERMEDIATE=true
CLAWLINGUA_ALLOW_EMPTY_DECK=true
CLAWLINGUA_DEFAULT_DECK_NAME=ClawLingua Default Deck
```

- `CLAWLINGUA_MATERIAL_PROFILE` chooses material strategy: `prose_article`,
  `transcript_dialogue`, `textbook_examples`.
- `CLAWLINGUA_LEARNING_MODE` is currently `expression_mining` (explicit in V2 model).
- Prompt selection is now **profile + difficulty** driven:
  - prose: `cloze_prose_{beginner|intermediate|advanced}.json`
  - transcript: `cloze_transcript_{beginner|intermediate|advanced}.json`
  - textbook_examples: `cloze_textbook_examples.json`
- `CLAWLINGUA_CONTENT_PROFILE` is kept as a backward-compatible alias.
- `CLAWLINGUA_EXTRACT_PROMPT` / `CLAWLINGUA_EXPLAIN_PROMPT` can pin default
  prompt files directly (higher priority than profile-chain defaults, lower
  priority than CLI `--extract-prompt` / `--explain-prompt`).
- `CLAWLINGUA_PROMPT_LANG` controls which language variant is used for multi-lingual prompts (`en` or `zh`), and can be overridden by `--prompt-lang`.

### 2.7 TTS (edge_tts)

```env
CLAWLINGUA_TTS_PROVIDER=edge_tts
CLAWLINGUA_TTS_OUTPUT_FORMAT=mp3
CLAWLINGUA_TTS_RATE=+0%
CLAWLINGUA_TTS_VOLUME=+0%
CLAWLINGUA_TTS_RANDOM_SEED=

# Voice lists per language; at least 3 voices per source language is recommended.
CLAWLINGUA_TTS_EDGE_EN_VOICES=en-US-AnaNeural,en-US-AndrewNeural,en-GB-SoniaNeural
CLAWLINGUA_TTS_EDGE_ZH_VOICES=zh-CN-XiaoxiaoNeural,zh-CN-YunxiNeural,zh-CN-liaoning-XiaobeiNeural
CLAWLINGUA_TTS_EDGE_JA_VOICES=ja-JP-NanamiNeural,ja-JP-KeitaNeural,ja-JP-AoiNeural
```

ClawLingua uses `edge_tts` to synthesize audio for the `Original` sentence of
each card, selecting a voice from the configured list based on `source_lang`.

---

## 3. Cloze & translation prompts

### 3.1 Cloze prompt families

Cloze generation now uses **prompt family + difficulty variant**:

- `prose_article`: `cloze_prose_beginner.json`, `cloze_prose_intermediate.json`, `cloze_prose_advanced.json`
- `transcript_dialogue`: `cloze_transcript_beginner.json`, `cloze_transcript_intermediate.json`, `cloze_transcript_advanced.json`
- `textbook_examples`: `cloze_textbook_examples.json`

Legacy `cloze_contextual.json` is still supported for backward compatibility.

- Prompt fields support both legacy and multi-lingual formats:
  - Legacy string format: `"system_prompt": "..."`.
  - Multi-lingual map: `"system_prompt": { "en": "...", "zh": "..." }`.
- At runtime, the language variant is selected based on
  `CLAWLINGUA_PROMPT_LANG` (or the `--prompt-lang` CLI override).

It uses `source_lang`, `target_lang`, `learning_mode`, `difficulty`, `cloze_max_sentences`, and
a merged `chunk_text` (possibly containing multiple chunk blocks) as
placeholders.

The expected output is **JSON array**, each element roughly:

```json
{
  "chunk_id": "chunk_0001_abcd12",
  "text": "The more {{c1::<b>whimsical explanation</b>}}(target-lang hint) is that maybe RL training makes the models a little too {{c2::<b>single-minded</b>}}(target-lang hint) and narrowly focused.",
  "original": "The more whimsical explanation is that maybe RL training makes the models a little too single-minded and narrowly focused.",
  "target_phrases": ["whimsical explanation", "single-minded"],
  "note_hint": "optional short hint"
}
```

Key rules (enforced via prompt + validator):

- `text` must contain at least one cloze:
  - cloze syntax: `{{cN::...}}` where N = 1, 2, 3...
  - cloze **inside** uses `<b>...</b>` to emphasize the phrase.
  - cloze is immediately followed by a short explanation in parentheses:
    - `{{c1::<b>whimsical explanation</b>}}(target-lang hint)`.
- `original` must not contain any cloze markers or HTML.
- Each chunk may produce 0-4 high-quality cloze candidates.
- Each candidate may contain multiple clozes (e.g. c1 and c2 in the same sentence).

The validator further:

- normalizes single-brace clozes (`{c1::...}` -> `{{c1::...}}`);
- auto-injects a `{{c1::...}}` cloze from `target_phrases` when text has no clozes
  but phrases are present (fallback only);
- rejects candidates that are too short (`len(text) < CLOZE_MIN_CHARS`) or exceed
  `CLOZE_MAX_SENTENCES`.

> NOTE: future work may include renumbering multiple `c1` occurrences to `c1`,
> `c2`, `c3` in order of appearance.

### 3.2 Textbook prompt: `prompts/cloze_textbook_examples.json`

Use this with `--content-profile textbook_examples` for textbook-style entries
that mix headwords, definitions, and example sentences. The prompt is tuned to:

- ignore standalone headword/title lines;
- ignore dictionary-style definition lines;
- extract cloze candidates only from natural example sentences.

### 3.3 Translation prompt: `prompts/translate_rewrite.json`

The translation prompt follows the same multi-lingual structure support as the
cloze prompts: `system_prompt` and `user_prompt_template` may be plain strings
or `{ "en": "...", "zh": "..." }` maps, selected via
`CLAWLINGUA_PROMPT_LANG` / `--prompt-lang`.

The translation prompt runs in batch mode. Input is an array of originals, and
the LLM should return a JSON array in the same order:

```json
[
  { "translation": "..." },
  { "translation": "..." }
]
```

Validator ensures:

- non-empty `translation`;
- no translation-prefix artifacts like `"Translation:"`;
- no Markdown `**` (HTML `<b>` is allowed).

Error/retry semantics:

- Network/request-level failures (timeout, HTTP, full JSON parse failure) are
  retried up to 3 times for the current remaining batch.
- When a batch response is incomplete, successful entries are accepted and only
  the remaining entries are retried (up to 3 attempts total).
- Content/validation failures are not retried; they follow `--continue-on-error`
  semantics.

---

## 4. CLI commands

The entrypoint module is `src/clawlingua/cli.py`, exposing a Typer-based CLI.
You can run it either as a module or via an installed entrypoint (if configured
in your environment).

### 4.1 `init`

```bash
python -m clawlingua.cli init
```

- Creates `.env` from `.env.example` if needed.
- Verifies required prompt/template files.
- Optionally prepares an output directory.

### 4.2 `doctor`

```bash
python -m clawlingua.cli doctor --env-file .env
```

Performs a series of checks:

- Python dependencies (`edge_tts`, `genanki`, `httpx`, `typer`)
- base config (paths, prompt/template files)
- runtime config (LLM, TTS voices)
- cloze/translate prompt schema
- primary LLM connectivity (`CLAWLINGUA_LLM_*`)
- translation LLM config & connectivity (`CLAWLINGUA_TRANSLATE_LLM_*`)
- output directory writability
- cloze control summary (max_sentences / min_chars / difficulty / max_per_chunk / material_profile / learning_mode)
- TTS voices for `default_source_lang`

### 4.3 `build deck`

Core command:

```bash
python -m clawlingua.cli build deck INPUT \
  --source-lang en \
  --target-lang zh \
  --material-profile prose_article|transcript_dialogue|textbook_examples \
  --learning-mode expression_mining \
  --input-char-limit 4000 \
  --env-file .env \
  --output deck.apkg \
  --deck-name "My Cloze Deck" \
  --max-chars 1500 \
  --cloze-min-chars 60 \
  --max-notes 200 \
  --temperature 0.2 \
  --difficulty beginner|intermediate|advanced \
  --prompt-lang en|zh \
  --extract-prompt ./prompts/cloze_prose_intermediate.json \
  --explain-prompt ./prompts/translate_rewrite.json \
  --save-intermediate \
  --continue-on-error \
  --debug
```

Where:

- `INPUT`: path to `.txt`/`.md`/`.epub` file.
- `--source-lang` / `--target-lang` override defaults from env.
- `--material-profile` selects material strategy and cloze prompt family.
- `--learning-mode` is explicit in V2 model (`expression_mining` for now).
- `--content-profile` is kept as a deprecated alias of `--material-profile`.
- `--input-char-limit` lets you process only the first N characters for quick tests.
- `--difficulty` overrides `CLAWLINGUA_CLOZE_DIFFICULTY`.
- `--prompt-lang` overrides `CLAWLINGUA_PROMPT_LANG` for multi-lingual prompts.
- `--extract-prompt` overrides extraction prompt file for this run.
- `--explain-prompt` overrides explanation prompt file for this run.
- `--max-chars` overrides `CLAWLINGUA_CHUNK_MAX_CHARS` for this run.
- `--cloze-min-chars` overrides `CLAWLINGUA_CLOZE_MIN_CHARS` for this run.
- In `textbook_examples` profile, runs are rejected when env `CLOZE_MIN_CHARS > 120`
  unless you explicitly provide `--cloze-min-chars`.
- `--max-notes` imposes a global cap on number of notes.
- `--save-intermediate` dumps intermediates under `CLAWLINGUA_OUTPUT_DIR/<run_id>`.
- When `--output` is not provided, the final deck is written to
  `CLAWLINGUA_EXPORT_DIR/<run_id>/output.apkg`.
- `--continue-on-error` logs and skips individual failures instead of aborting.
- `--debug` makes `_run_guard` re-raise exceptions with tracebacks.
- By default, deck name uses the input file name (without extension); `--deck-name` overrides it.

### 4.4 `prompt validate`

```bash
python -m clawlingua.cli prompt validate ./prompts/cloze_prose_intermediate.json
python -m clawlingua.cli prompt validate ./prompts/cloze_textbook_examples.json
```

Validates a prompt file against the expected JSON schema (see
`src/clawlingua/llm/prompt_loader.py`).

### 4.5 `config show` / `config validate`

```bash
python -m clawlingua.cli config show --env-file .env
python -m clawlingua.cli config validate --env-file .env
```

- `config show` prints resolved `AppConfig` as JSON.
- `config validate` runs config validation without building a deck.

---

## 5. Output format

Using the default Anki template (`templates/anki_cloze_default.json`), each card
has at least the following fields:

- **Text**: the cloze sentence with `{{cN::...}}` markers, HTML `<b>` for emphasis,
  and optional translations in parentheses.
- **Original**: the original sentence(s) without cloze markers or HTML.
- **Translation**: target-language translation of `Original`.
- **Note**: metadata (source title, chunk id, target phrases).
- **Audio**: `edge_tts`-generated audio for `Original` (via `[sound:xxx.mp3]`).

The exact Anki field mapping is defined in the JSON template; you can customize
it if you want different field names or card faces.

---

## 6. Typical workflow

1. **Set up `.env`**

   - Configure LLM endpoints (primary + optional translation LLM).
   - Configure chunking & cloze controls.
   - Configure TTS voices for your source language.

2. **Run doctor**

   ```bash
   python -m clawlingua.cli doctor --env-file .env
   ```

3. **Build a deck from a podcast transcript**

   ```bash
   python -m clawlingua.cli build deck ./podcast_transcript.md \
     --source-lang en --target-lang zh --env-file .env \
     --difficulty intermediate --max-chars 1500 \
     --save-intermediate --continue-on-error
   ```

4. **Import the generated `.apkg` into Anki** and review cards.

5. **Inspect intermediates** (optional) under `./runs/<run_id>`:

   - `chunks.jsonl`: chunked text
   - `text_candidates.raw.jsonl`: raw cloze candidates from LLM
   - `text_candidates.validated.jsonl`: candidates that passed validation
   - `translations.jsonl`: translations per card
   - `cards.final.jsonl`: final card data before export

---

## 7. Optional web UI

For users who prefer a browser-based interface, ClawLingua ships an optional
local-only web UI built with Gradio. This does **not** change the core CLI
behaviour and is only started when explicitly invoked.

### 7.1 Installation

Install the `web` extra (in addition to the core dependencies):

```bash
pip install .[web]
```

### 7.2 Launching the web UI

From the project root:

```bash
clawlingua-web
# or
python -m clawlingua_web.app
```

This starts a Gradio app bound to `127.0.0.1:7860`. Open
<http://127.0.0.1:7860> in your browser.

The web UI has three tabs:

- **Run** - upload a `.txt`/`.md`/`.epub` file, select source/target language,
  content profile, difficulty, and per-run overrides (max notes, input char
  limit, cloze min chars, chunk max chars, temperature). The backend calls the
  same `run_build_deck` pipeline and writes intermediate data to
  `CLAWLINGUA_OUTPUT_DIR/<run_id>` and the final deck to
  `CLAWLINGUA_EXPORT_DIR/<run_id>/output.apkg`.
- **Config** - a `.env` editor for common `CLAWLINGUA_*` settings (LLM
  endpoints, chunk/cloze defaults, prompt language, output/log directories,
  default deck name, TTS, etc.). Saving changes writes a new `.env`, validates
  it via `clawlingua.config.validate_base_config` + `validate_runtime_config`,
  and rolls back on failure. The "Load defaults" button loads values from
  `ENV_EXAMPLE.md` into the form without writing to disk. The Config tab also
  provides "List models" / "Test connectivity" helpers for both the Extraction
  LLM and the Explanation LLM using their `/models` endpoints, and includes
  `CLAWLINGUA_EXTRACT_PROMPT` / `CLAWLINGUA_EXPLAIN_PROMPT` dropdowns.
- **Prompt** - manage prompt files in `./prompts` with `New / Save / Rename / Delete`.
  New prompt creation loads role templates (`template_extraction.json` or
  `template_explanation.json`) based on selected Prompt type
  (Extraction/Explanation). Save/Delete require explicit confirmation in the UI.
  Delete is guarded: the app refuses to remove the last Extraction prompt or the
  last Explanation prompt.
The web UI is optional; OpenClaw skills and automated usage should continue
calling the CLI directly.

## 8. Known limitations / future work

- Cloze numbering: when multiple clozes appear in a single `text`, we may need
  to renumber them deterministically (`c1`, `c2`, `c3`) in order of appearance.
- Cloze formatting: current prompt encourages the
  `{{cN::<b>phrase</b>}}(translation)` style, but behaviour still depends on the
  chosen LLM and may require further prompt tuning.
- Tests: the original `tests/` directory has been removed from `main`; if you
  extend the project, consider reintroducing a focused test suite.

For a Chinese overview and usage guide, see [`README_zh.md`](./README_zh.md).

