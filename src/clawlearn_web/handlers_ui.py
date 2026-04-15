"""Top-level UI event handlers extracted from app.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import gradio as gr


@dataclass(frozen=True)
class UiDeps:
    normalize_ui_lang: Callable[[str | None], str]
    load_app_config: Callable[[], Any]
    prompt_file_map: Callable[..., dict[str, Any]]
    normalize_prompt_mode: Callable[[Any], str]
    normalize_prompt_content_type: Callable[[Any], str]
    normalize_prompt_learning_mode: Callable[[Any], str]
    normalize_prompt_difficulty: Callable[[Any], str]
    load_prompt_mode: Callable[..., str]
    pick_prompt_key: Callable[..., str]
    load_prompt_template: Callable[..., tuple[str, str]]
    load_prompt_filter_metadata: Callable[..., tuple[str, str, str]]
    prompt_mode_choices_for_ui: Callable[[str], list[tuple[str, str]]]
    prompt_path_choices: Callable[..., list[tuple[str, str]]]
    refresh_recent_runs: Callable[..., tuple[Any, str, str | None]]
    normalize_dropdown_value: Callable[[str, list[Any]], str]
    tr: Callable[[str, str, str], str]
    prompt_choices_from_map: Callable[..., list[tuple[str, str]]]
    prompt_content_type_options: Sequence[str]
    prompt_learning_mode_options: Sequence[str]
    prompt_difficulty_options: Sequence[str]


def on_ui_lang_change(
    lang_value: str,
    prompt_lang_current: str,
    prompt_key_current: str,
    prompt_mode_current: str,
    prompt_content_type_current: str,
    prompt_learning_mode_current: str,
    prompt_difficulty_current: str,
    run_content_type_current: str,
    run_learning_mode_current: str,
    run_difficulty_current: str,
    run_extract_prompt_current: str,
    run_explain_prompt_current: str,
    run_id_current: str | None,
    *,
    deps: UiDeps,
) -> tuple[Any, ...]:
    lang = deps.normalize_ui_lang(lang_value)
    _ = prompt_lang_current
    prompt_lang_next = lang
    cfg_now = deps.load_app_config()
    prompt_files_now = deps.prompt_file_map(cfg_now)
    prompt_mode_pref = deps.normalize_prompt_mode(prompt_mode_current)
    prompt_content_type_pref = deps.normalize_prompt_content_type(
        prompt_content_type_current
    )
    prompt_learning_mode_pref = deps.normalize_prompt_learning_mode(
        prompt_learning_mode_current
    )
    prompt_difficulty_pref = deps.normalize_prompt_difficulty(prompt_difficulty_current)
    if not prompt_mode_pref and prompt_key_current in prompt_files_now:
        prompt_mode_pref = deps.load_prompt_mode(
            prompt_key_current, prompt_files_now, lang=lang
        )
    if not prompt_mode_pref:
        prompt_mode_pref = "extraction"
    prompt_files_filtered_next = deps.prompt_file_map(
        cfg_now,
        mode_filter=prompt_mode_pref,
        content_type_filter=prompt_content_type_pref,
        learning_mode_filter=prompt_learning_mode_pref,
        difficulty_filter=prompt_difficulty_pref,
    )
    prompt_key_next = deps.pick_prompt_key(
        prompt_files_filtered_next,
        lang=lang,
        preferred_key=prompt_key_current,
        preferred_mode=prompt_mode_pref,
    )
    if not prompt_key_next and prompt_files_now:
        prompt_key_next = deps.pick_prompt_key(
            prompt_files_now,
            lang=lang,
            preferred_key=prompt_key_current,
            preferred_mode=prompt_mode_pref,
        )
    prompt_mode_next = (
        deps.load_prompt_mode(prompt_key_next, prompt_files_now, lang=lang)
        or prompt_mode_pref
    )
    prompt_files_filtered_next = deps.prompt_file_map(
        cfg_now,
        mode_filter=prompt_mode_next,
        content_type_filter=prompt_content_type_pref,
        learning_mode_filter=prompt_learning_mode_pref,
        difficulty_filter=prompt_difficulty_pref,
    )
    if prompt_key_next:
        prompt_template_next, prompt_status_next = deps.load_prompt_template(
            prompt_key_next,
            prompt_files_now,
            lang=lang,
        )
        if (
            prompt_content_type_pref == "all"
            and prompt_learning_mode_pref == "all"
            and prompt_difficulty_pref == "all"
        ):
            (
                prompt_content_type_pref,
                prompt_learning_mode_pref,
                prompt_difficulty_pref,
            ) = deps.load_prompt_filter_metadata(
                prompt_key_next, prompt_files_now, lang=lang
            )
            prompt_files_filtered_next = deps.prompt_file_map(
                cfg_now,
                mode_filter=prompt_mode_next,
                content_type_filter=prompt_content_type_pref,
                learning_mode_filter=prompt_learning_mode_pref,
                difficulty_filter=prompt_difficulty_pref,
            )
    else:
        prompt_template_next, prompt_status_next = "", ""
    prompt_mode_choices_next = deps.prompt_mode_choices_for_ui(lang)
    run_content_type_pref = deps.normalize_prompt_content_type(run_content_type_current)
    run_learning_mode_pref = deps.normalize_prompt_learning_mode(
        run_learning_mode_current
    )
    run_difficulty_pref = deps.normalize_prompt_difficulty(run_difficulty_current)
    run_extract_prompt_choices_next = deps.prompt_path_choices(
        cfg_now,
        lang=lang,
        mode_filter="extraction",
        content_type_filter=run_content_type_pref,
        learning_mode_filter=run_learning_mode_pref,
        difficulty_filter=run_difficulty_pref,
        include_auto=True,
    )
    run_explain_prompt_choices_next = deps.prompt_path_choices(
        cfg_now,
        lang=lang,
        mode_filter="explanation",
        content_type_filter=run_content_type_pref,
        learning_mode_filter=run_learning_mode_pref,
        difficulty_filter=run_difficulty_pref,
        include_auto=True,
    )
    config_extract_prompt_choices_next = deps.prompt_path_choices(
        cfg_now,
        lang=lang,
        mode_filter="extraction",
        include_auto=True,
    )
    config_explain_prompt_choices_next = deps.prompt_path_choices(
        cfg_now,
        lang=lang,
        mode_filter="explanation",
        include_auto=True,
    )
    run_selector_next, run_detail_next, run_download_next = deps.refresh_recent_runs(
        cfg_now,
        lang=lang,
        preferred_run_id=run_id_current,
    )
    selector_choices = run_selector_next.get("choices", [])
    selector_value = run_selector_next.get("value")
    prompt_choices_next = deps.prompt_choices_from_map(
        prompt_files_filtered_next, lang=lang
    )
    prompt_key_next = deps.normalize_dropdown_value(
        prompt_key_next, prompt_choices_next
    )
    if not prompt_key_next:
        prompt_template_next = ""
    return (
        gr.update(label=deps.tr(lang, "UI language", "UI language")),
        gr.update(
            value=deps.tr(
                lang,
                "# ClawLearn Web UI\nLocal deck builder for text learning.",
                "# ClawLearn Web UI\nLocal deck builder for text learning.",
            )
        ),
        gr.update(label=deps.tr(lang, "Run", "Run")),
        gr.update(label=deps.tr(lang, "Config", "Config")),
        gr.update(label=deps.tr(lang, "Prompt", "Prompt")),
        gr.update(label=deps.tr(lang, "Input file", "Input file")),
        gr.update(
            label=deps.tr(lang, "Deck title (optional)", "Deck title (optional)")
        ),
        gr.update(label=deps.tr(lang, "Source language", "Source language")),
        gr.update(label=deps.tr(lang, "Target language", "Target language")),
        gr.update(label=deps.tr(lang, "Content profile", "Content profile")),
        gr.update(label=deps.tr(lang, "Difficulty", "Difficulty")),
        gr.update(
            label=deps.tr(
                lang,
                "Extraction prompt (run override)",
                "Extraction prompt (run override)",
            ),
            info=deps.tr(
                lang,
                "Equivalent to CLI --extract-prompt.",
                "Equivalent to CLI --extract-prompt.",
            ),
            choices=run_extract_prompt_choices_next,
            value=deps.normalize_dropdown_value(
                run_extract_prompt_current, run_extract_prompt_choices_next
            ),
        ),
        gr.update(
            label=deps.tr(
                lang,
                "Explanation prompt (run override)",
                "Explanation prompt (run override)",
            ),
            info=deps.tr(
                lang,
                "Equivalent to CLI --explain-prompt.",
                "Equivalent to CLI --explain-prompt.",
            ),
            choices=run_explain_prompt_choices_next,
            value=deps.normalize_dropdown_value(
                run_explain_prompt_current, run_explain_prompt_choices_next
            ),
        ),
        gr.update(
            label=deps.tr(lang, "Max notes (0 = no limit)", "Max notes (0 = no limit)"),
            info=deps.tr(
                lang,
                "Maximum notes after dedupe. Empty/0 means no limit.",
                "Maximum notes after dedupe. Empty/0 means no limit.",
            ),
        ),
        gr.update(
            label=deps.tr(lang, "Input char limit", "Input char limit"),
            info=deps.tr(
                lang,
                "Only process the first N chars of input. Empty means no limit.",
                "Only process the first N chars of input. Empty means no limit.",
            ),
        ),
        gr.update(label=deps.tr(lang, "Advanced", "Advanced")),
        gr.update(
            label=deps.tr(
                lang,
                "Cloze min chars (override env)",
                "Cloze min chars (override env)",
            ),
            info=deps.tr(
                lang,
                "One-run override for CLAWLEARN_CLOZE_MIN_CHARS.",
                "One-run override for CLAWLEARN_CLOZE_MIN_CHARS.",
            ),
        ),
        gr.update(
            label=deps.tr(
                lang,
                "Chunk max chars (override env)",
                "Chunk max chars (override env)",
            ),
            info=deps.tr(
                lang,
                "One-run override for CLAWLEARN_CHUNK_MAX_CHARS.",
                "One-run override for CLAWLEARN_CHUNK_MAX_CHARS.",
            ),
        ),
        gr.update(
            label=deps.tr(
                lang, "Temperature (override env)", "Temperature (override env)"
            ),
            info=deps.tr(
                lang,
                "0 is more deterministic; higher values are more random.",
                "0 is more deterministic; higher values are more random.",
            ),
        ),
        gr.update(
            label=deps.tr(lang, "Save intermediate files", "Save intermediate files"),
            info=deps.tr(
                lang,
                "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.",
                "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.",
            ),
        ),
        gr.update(
            label=deps.tr(lang, "Continue on error", "Continue on error"),
            info=deps.tr(
                lang,
                "If enabled, continue processing after per-item failures.",
                "If enabled, continue processing after per-item failures.",
            ),
        ),
        gr.update(value=deps.tr(lang, "Run", "Run")),
        gr.update(label=deps.tr(lang, "Status", "Status")),
        gr.update(label=deps.tr(lang, "Output file", "Output file")),
        gr.update(value=deps.tr(lang, "### Recent runs", "### Recent runs")),
        gr.update(value=deps.tr(lang, "Refresh runs", "Refresh runs")),
        gr.update(
            label=deps.tr(lang, "Run ID", "Run ID"),
            choices=selector_choices,
            value=selector_value,
        ),
        gr.update(value=run_detail_next),
        gr.update(
            label=deps.tr(lang, "Download .apkg", "Download .apkg"),
            value=run_download_next,
        ),
        gr.update(
            value=deps.tr(
                lang,
                "### Config (.env editor)\nWeb UI reads `.env` from the current working directory of the web process.",
                "### Config (.env editor)\nWeb UI reads `.env` from the current working directory of the web process.",
            )
        ),
        gr.update(label=deps.tr(lang, "Extraction LLM", "Extraction LLM")),
        gr.update(
            info=deps.tr(
                lang,
                "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).",
                "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).",
            )
        ),
        gr.update(
            info=deps.tr(
                lang,
                "API key for extraction LLM, when required.",
                "API key for extraction LLM, when required.",
            )
        ),
        gr.update(
            info=deps.tr(
                lang,
                "Model name for extraction LLM.",
                "Model name for extraction LLM.",
            )
        ),
        gr.update(
            info=deps.tr(
                lang, "Request timeout in seconds.", "Request timeout in seconds."
            ),
        ),
        gr.update(
            info=deps.tr(
                lang,
                "Default temperature for extraction LLM.",
                "Default temperature for extraction LLM.",
            )
        ),
        gr.update(value=deps.tr(lang, "List models", "List models")),
        gr.update(value=deps.tr(lang, "Test", "Test")),
        gr.update(
            label=deps.tr(lang, "Extraction LLM status", "Extraction LLM status")
        ),
        gr.update(label=deps.tr(lang, "Explanation LLM", "Explanation LLM")),
        gr.update(
            info=deps.tr(
                lang,
                "Optional base URL for explanation model.",
                "Optional base URL for explanation model.",
            )
        ),
        gr.update(
            info=deps.tr(
                lang,
                "API key for explanation LLM, when required.",
                "API key for explanation LLM, when required.",
            )
        ),
        gr.update(
            info=deps.tr(
                lang,
                "Model name for explanation LLM.",
                "Model name for explanation LLM.",
            )
        ),
        gr.update(
            info=deps.tr(
                lang,
                "Default temperature for explanation LLM.",
                "Default temperature for explanation LLM.",
            )
        ),
        gr.update(
            value=deps.tr(
                lang, "List models (explanation)", "List models (explanation)"
            )
        ),
        gr.update(value=deps.tr(lang, "Test (explanation)", "Test (explanation)")),
        gr.update(
            label=deps.tr(lang, "Explanation LLM status", "Explanation LLM status")
        ),
        gr.update(label=deps.tr(lang, "Chunk & Cloze", "Chunk & Cloze")),
        gr.update(
            info=deps.tr(
                lang,
                "Default max chars per chunk.",
                "Default max chars per chunk.",
            )
        ),
        gr.update(
            info=deps.tr(
                lang,
                "Minimum chars required for cloze text.",
                "Minimum chars required for cloze text.",
            )
        ),
        gr.update(
            info=deps.tr(
                lang,
                "Max cards per chunk after dedupe. Empty/0 means unlimited.",
                "Max cards per chunk after dedupe. Empty/0 means unlimited.",
            )
        ),
        gr.update(
            label="CLAWLEARN_PROMPT_LANG",
            info=deps.tr(
                lang,
                "Prompt language for multi-lingual prompts (en/zh).",
                "Prompt language for multi-lingual prompts (en/zh).",
            ),
            value=prompt_lang_next,
        ),
        gr.update(
            label="CLAWLEARN_EXTRACT_PROMPT",
            info=deps.tr(
                lang,
                "Default extraction prompt path.",
                "Default extraction prompt path.",
            ),
            choices=config_extract_prompt_choices_next,
        ),
        gr.update(
            label="CLAWLEARN_EXPLAIN_PROMPT",
            info=deps.tr(
                lang,
                "Default explanation prompt path.",
                "Default explanation prompt path.",
            ),
            choices=config_explain_prompt_choices_next,
        ),
        gr.update(label=deps.tr(lang, "Paths & defaults", "Paths & defaults")),
        gr.update(
            info=deps.tr(
                lang,
                "Directory for intermediate run data (JSONL, media).",
                "Directory for intermediate run data (JSONL, media).",
            )
        ),
        gr.update(
            info=deps.tr(
                lang,
                "Default directory for exported decks.",
                "Default directory for exported decks.",
            )
        ),
        gr.update(
            info=deps.tr(lang, "Directory for log files.", "Directory for log files."),
        ),
        gr.update(
            value=deps.tr(
                lang,
                "Load defaults from ENV_EXAMPLE.md",
                "Load defaults from ENV_EXAMPLE.md",
            )
        ),
        gr.update(value=deps.tr(lang, "Save config", "Save config")),
        gr.update(label=deps.tr(lang, "TTS voices (Edge)", "TTS voices (Edge)")),
        gr.update(
            value=deps.tr(
                lang,
                "Voice reference: [Edge TTS Voice Samples](https://tts.travisvn.com/)",
                "Voice reference: [Edge TTS Voice Samples](https://tts.travisvn.com/)",
            )
        ),
        gr.update(
            info=deps.tr(
                lang,
                "Configure 4 voice slots used for random selection.",
                "Configure 4 voice slots used for random selection.",
            )
        ),
        gr.update(
            value=deps.tr(
                lang, "### Prompt template editor", "### Prompt template editor"
            )
        ),
        gr.update(
            label=deps.tr(lang, "Prompt file", "Prompt file"),
            choices=prompt_choices_next,
            value=prompt_key_next,
        ),
        gr.update(
            label=deps.tr(lang, "Prompt type", "Prompt type"),
            choices=prompt_mode_choices_next,
            value=prompt_mode_next,
        ),
        gr.update(
            label=deps.tr(lang, "Prompt content type", "Prompt content type"),
            choices=list(deps.prompt_content_type_options),
            value=prompt_content_type_pref,
        ),
        gr.update(
            label=deps.tr(lang, "Prompt learning mode", "Prompt learning mode"),
            choices=list(deps.prompt_learning_mode_options),
            value=prompt_learning_mode_pref,
        ),
        gr.update(
            label=deps.tr(lang, "Prompt difficulty", "Prompt difficulty"),
            choices=list(deps.prompt_difficulty_options),
            value=prompt_difficulty_pref,
        ),
        gr.update(label=deps.tr(lang, "New prompt file name", "New prompt file name")),
        gr.update(label=deps.tr(lang, "Rename to", "Rename to")),
        gr.update(
            label=deps.tr(lang, "Prompt template", "Prompt template"),
            value=prompt_template_next,
        ),
        gr.update(value=deps.tr(lang, "New", "New")),
        gr.update(value=deps.tr(lang, "Save", "Save")),
        gr.update(value=deps.tr(lang, "Rename", "Rename")),
        gr.update(value=deps.tr(lang, "Delete", "Delete")),
        gr.update(value=False, visible=False),
        gr.update(value=False, visible=False),
        gr.update(
            label=deps.tr(lang, "Prompt status", "Prompt status"),
            value=prompt_status_next,
        ),
        gr.update(label=deps.tr(lang, "Run analytics", "Run analytics")),
    )
