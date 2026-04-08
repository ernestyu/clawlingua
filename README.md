# ClawLingua

ClawLingua is a Python CLI that converts web/file content into an Anki cloze deck (`.apkg`).

## Features (V1)

- Input: URL, `.txt`, `.md`
- Text cleaning and chunking
- OpenAI-compatible LLM generation
  - `Text` via `./prompts/cloze_contextual.json`
  - `Translation` via `./prompts/translate_rewrite.json`
- TTS for `Original` (default provider: `edge_tts`)
- Anki export via `genanki`

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and edit values.

```bash
clawlingua init
clawlingua doctor
clawlingua build deck ./example.md --source-lang en --target-lang zh
```

## Commands

```bash
clawlingua init
clawlingua doctor
clawlingua build deck <input>
clawlingua prompt validate <path>
clawlingua config show
clawlingua config validate
```

## Output Fields

- `Text`
- `Original`
- `Translation`
- `Note`
- `Audio`
