"""Run tab component builder and event wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

import gradio as gr

from clawlearn_web import handlers_run
from clawlearn_web.ui.tab_analytics import AnalyticsTabComponents


def learning_mode_to_domain(learning_mode_val: Any) -> str:
    normalized = str(learning_mode_val or "").strip().lower()
    if normalized.startswith("textbook_"):
        return "textbook"
    return "lingua"


def learning_mode_visibility_flags(learning_mode_val: Any) -> tuple[bool, bool]:
    domain = learning_mode_to_domain(learning_mode_val)
    is_textbook = domain == "textbook"
    return (not is_textbook, is_textbook)


@dataclass(frozen=True)
class RunTabComponents:
    run_tab: Any
    input_file: Any
    deck_title: Any
    lingua_options_group: Any
    textbook_options_group: Any
    source_lang: Any
    target_lang: Any
    content_profile: Any
    learning_mode: Any
    difficulty: Any
    run_extract_prompt: Any
    run_explain_prompt: Any
    max_notes: Any
    input_char_limit: Any
    textbook_max_concepts_per_chunk: Any
    textbook_keep_source_excerpt: Any
    run_advanced: Any
    cloze_min_chars: Any
    chunk_max_chars: Any
    temperature: Any
    save_intermediate: Any
    continue_on_error: Any
    run_button: Any
    run_status: Any
    output_file: Any
    recent_runs_heading: Any
    refresh_runs_button: Any
    run_selector: Any
    run_detail: Any
    run_download_file: Any


def build_tab(
    *,
    initial_ui_lang: str,
    cfg: Any,
    initial_run_choices: list[tuple[str, str]],
    initial_run_selected: str | None,
    initial_run_detail: str,
    initial_run_download: str | None,
    run_extract_prompt_choices: list[tuple[str, str]],
    run_explain_prompt_choices: list[tuple[str, str]],
    tr: Callable[[str, str, str], str],
) -> RunTabComponents:
    with gr.Tab(tr(initial_ui_lang, "Run", "杩愯")) as run_tab:
        with gr.Row():
            input_file = gr.File(
                label=tr(initial_ui_lang, "Input file", "杈撳叆鏂囦欢"),
                file_types=[".txt", ".md", ".markdown", ".epub"],
                file_count="single",
            )
            deck_title = gr.Textbox(
                label=tr(
                    initial_ui_lang,
                    "Deck title (optional)",
                    "鐗岀粍鍚嶇О锛堝彲閫夛級",
                )
            )

        initial_learning_mode = str(
            getattr(cfg, "learning_mode", "lingua_expression") or "lingua_expression"
        ).strip().lower()
        if initial_learning_mode not in {
            "lingua_expression",
            "lingua_reading",
            "textbook_focus",
            "textbook_review",
        }:
            initial_learning_mode = "lingua_expression"
        show_lingua_options, show_textbook_options = learning_mode_visibility_flags(
            initial_learning_mode
        )

        with gr.Row():
            source_lang = gr.Dropdown(
                choices=["en", "zh", "ja", "de", "fr"],
                value=cfg.default_source_lang,
                label=tr(initial_ui_lang, "Source language", "婧愯瑷€"),
            )
            target_lang = gr.Dropdown(
                choices=["zh", "en", "ja", "de", "fr"],
                value=cfg.default_target_lang,
                label=tr(initial_ui_lang, "Target language", "鐩爣璇█"),
            )
            learning_mode = gr.Dropdown(
                choices=[
                    "lingua_expression",
                    "lingua_reading",
                    "textbook_focus",
                    "textbook_review",
                ],
                value=initial_learning_mode,
                label=tr(initial_ui_lang, "Learning mode", "瀛︿範妯″紡"),
            )

        with gr.Group(visible=show_lingua_options) as lingua_options_group:
            with gr.Row():
                content_profile = gr.Dropdown(
                    choices=[
                        "prose_article",
                        "transcript_dialogue",
                        "textbook_examples",
                    ],
                    value=cfg.content_profile,
                    label=tr(initial_ui_lang, "Content profile", "鍐呭绫诲瀷"),
                )
                difficulty = gr.Dropdown(
                    choices=["beginner", "intermediate", "advanced"],
                    value=cfg.cloze_difficulty,
                    label=tr(initial_ui_lang, "Difficulty", "闅惧害"),
                )

            with gr.Row():
                run_extract_prompt = gr.Dropdown(
                    choices=run_extract_prompt_choices,
                    value="",
                    label=tr(
                        initial_ui_lang,
                        "Extraction prompt (run override)",
                        "Extraction prompt (run override)",
                    ),
                    info=tr(
                        initial_ui_lang,
                        "Equivalent to CLI --extract-prompt.",
                        "Equivalent to CLI --extract-prompt.",
                    ),
                )
                run_explain_prompt = gr.Dropdown(
                    choices=run_explain_prompt_choices,
                    value="",
                    label=tr(
                        initial_ui_lang,
                        "Explanation prompt (run override)",
                        "Explanation prompt (run override)",
                    ),
                    info=tr(
                        initial_ui_lang,
                        "Equivalent to CLI --explain-prompt.",
                        "Equivalent to CLI --explain-prompt.",
                    ),
                )

        with gr.Group(visible=show_textbook_options) as textbook_options_group:
            with gr.Row():
                textbook_max_concepts_per_chunk = gr.Number(
                    label=tr(
                        initial_ui_lang,
                        "Max concepts per chunk",
                        "Max concepts per chunk",
                    ),
                    info=tr(
                        initial_ui_lang,
                        "Textbook domain only. 1 keeps current behavior.",
                        "Textbook domain only. 1 keeps current behavior.",
                    ),
                    value=1,
                    precision=0,
                )
                textbook_keep_source_excerpt = gr.Checkbox(
                    label=tr(
                        initial_ui_lang,
                        "Keep source excerpt",
                        "Keep source excerpt",
                    ),
                    info=tr(
                        initial_ui_lang,
                        "Textbook domain only. Store original excerpt in cards.",
                        "Textbook domain only. Store original excerpt in cards.",
                    ),
                    value=True,
                )

        with gr.Row():
            max_notes = gr.Number(
                label=tr(
                    initial_ui_lang,
                    "Max notes (0 = no limit)",
                    "Max notes (0 = no limit)",
                ),
                info=tr(
                    initial_ui_lang,
                    "Maximum notes after dedupe. Empty/0 means no limit.",
                    "Maximum notes after dedupe. Empty/0 means no limit.",
                ),
                value=None,
                precision=0,
            )
            input_char_limit = gr.Number(
                label=tr(initial_ui_lang, "Input char limit", "Input char limit"),
                info=tr(
                    initial_ui_lang,
                    "Only process the first N chars of input. Empty means no limit.",
                    "Only process the first N chars of input. Empty means no limit.",
                ),
                value=None,
                precision=0,
            )

        with gr.Accordion(tr(initial_ui_lang, "Advanced", "Advanced"), open=False) as run_advanced:
            cloze_min_chars = gr.Number(
                label=tr(
                    initial_ui_lang,
                    "Cloze min chars (override env)",
                    "Cloze min chars (override env)",
                ),
                info=tr(
                    initial_ui_lang,
                    "One-run override for CLAWLEARN_CLOZE_MIN_CHARS.",
                    "One-run override for CLAWLEARN_CLOZE_MIN_CHARS.",
                ),
                value=cfg.cloze_min_chars,
                precision=0,
                visible=show_lingua_options,
            )
            chunk_max_chars = gr.Number(
                label=tr(
                    initial_ui_lang,
                    "Chunk max chars (override env)",
                    "Chunk max chars (override env)",
                ),
                info=tr(
                    initial_ui_lang,
                    "One-run override for CLAWLEARN_CHUNK_MAX_CHARS.",
                    "One-run override for CLAWLEARN_CHUNK_MAX_CHARS.",
                ),
                value=cfg.chunk_max_chars,
                precision=0,
            )
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=cfg.llm_temperature,
                step=0.05,
                label=tr(
                    initial_ui_lang,
                    "Temperature (override env)",
                    "Temperature (override env)",
                ),
                info=tr(
                    initial_ui_lang,
                    "0 is more deterministic; higher values are more random.",
                    "0 is more deterministic; higher values are more random.",
                ),
            )
            save_intermediate = gr.Checkbox(
                label=tr(
                    initial_ui_lang,
                    "Save intermediate files",
                    "Save intermediate files",
                ),
                info=tr(
                    initial_ui_lang,
                    "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.",
                    "Write intermediate JSONL/media into OUTPUT_DIR/<run_id>.",
                ),
                value=cfg.save_intermediate,
            )
            continue_on_error = gr.Checkbox(
                label=tr(initial_ui_lang, "Continue on error", "Continue on error"),
                info=tr(
                    initial_ui_lang,
                    "If enabled, continue processing after per-item failures.",
                    "If enabled, continue processing after per-item failures.",
                ),
                value=False,
            )

        run_button = gr.Button(tr(initial_ui_lang, "Run", "Run"))
        run_status = gr.Markdown(label=tr(initial_ui_lang, "Status", "Status"))
        output_file = gr.File(
            label=tr(initial_ui_lang, "Download .apkg", "Download .apkg"),
            interactive=False,
        )
        recent_runs_heading = gr.Markdown(
            tr(initial_ui_lang, "### Recent runs", "### Recent runs")
        )
        with gr.Row():
            refresh_runs_button = gr.Button(
                tr(initial_ui_lang, "Refresh runs", "Refresh runs")
            )
            run_selector = gr.Dropdown(
                choices=initial_run_choices,
                value=initial_run_selected,
                label=tr(initial_ui_lang, "Run ID", "Run ID"),
            )
        run_detail = gr.Markdown(value=initial_run_detail)
        run_download_file = gr.File(
            label=tr(initial_ui_lang, "Download .apkg", "Download .apkg"),
            interactive=False,
            value=initial_run_download,
        )

    return RunTabComponents(
        run_tab=run_tab,
        input_file=input_file,
        deck_title=deck_title,
        lingua_options_group=lingua_options_group,
        textbook_options_group=textbook_options_group,
        source_lang=source_lang,
        target_lang=target_lang,
        content_profile=content_profile,
        learning_mode=learning_mode,
        difficulty=difficulty,
        run_extract_prompt=run_extract_prompt,
        run_explain_prompt=run_explain_prompt,
        max_notes=max_notes,
        input_char_limit=input_char_limit,
        textbook_max_concepts_per_chunk=textbook_max_concepts_per_chunk,
        textbook_keep_source_excerpt=textbook_keep_source_excerpt,
        run_advanced=run_advanced,
        cloze_min_chars=cloze_min_chars,
        chunk_max_chars=chunk_max_chars,
        temperature=temperature,
        save_intermediate=save_intermediate,
        continue_on_error=continue_on_error,
        run_button=run_button,
        run_status=run_status,
        output_file=output_file,
        recent_runs_heading=recent_runs_heading,
        refresh_runs_button=refresh_runs_button,
        run_selector=run_selector,
        run_detail=run_detail,
        run_download_file=run_download_file,
    )


def bind_events(
    *,
    components: RunTabComponents,
    analytics: AnalyticsTabComponents,
    ui_lang: Any,
    deps: handlers_run.RunDeps,
) -> None:
    def _on_learning_mode_change(learning_mode_val: Any) -> tuple[Any, Any, Any]:
        show_lingua, show_textbook = learning_mode_visibility_flags(learning_mode_val)
        return (
            gr.update(visible=show_lingua),
            gr.update(visible=show_textbook),
            gr.update(visible=show_lingua),
        )

    def _on_run_start(ui_lang_val: str) -> tuple[str, None]:
        return handlers_run.on_run_start(ui_lang_val, deps=deps)

    def _on_run(
        file_obj: Any,
        deck_title_val: Any,
        src: Any,
        tgt: Any,
        profile: Any,
        mode: Any,
        diff: Any,
        extract_prompt_val: Any,
        explain_prompt_val: Any,
        max_notes_val: Any,
        input_limit_val: Any,
        cloze_min_val: Any,
        textbook_max_concepts_val: Any,
        textbook_keep_excerpt_val: Any,
        chunk_max_val: Any,
        temperature_val: Any,
        save_inter_val: Any,
        continue_on_error_val: Any,
        ui_lang_val: Any,
    ) -> Iterator[tuple[Any, ...]]:
        yield from handlers_run.on_run(
            file_obj,
            deck_title_val,
            src,
            tgt,
            profile,
            mode,
            diff,
            extract_prompt_val,
            explain_prompt_val,
            max_notes_val,
            input_limit_val,
            cloze_min_val,
            textbook_max_concepts_val,
            textbook_keep_excerpt_val,
            chunk_max_val,
            temperature_val,
            save_inter_val,
            continue_on_error_val,
            ui_lang_val,
            deps=deps,
        )

    components.learning_mode.change(
        _on_learning_mode_change,
        inputs=[components.learning_mode],
        outputs=[
            components.lingua_options_group,
            components.textbook_options_group,
            components.cloze_min_chars,
        ],
        queue=False,
    )

    def _on_refresh_runs(
        ui_lang_val: str,
        selected_run_id: str | None,
    ) -> tuple[Any, str, str | None, str, list[list[Any]], Any, Any, Any, Any]:
        return handlers_run.on_refresh_runs(
            ui_lang_val,
            selected_run_id,
            deps=deps,
        )

    def _on_run_selected(
        run_id_val: str | None,
        ui_lang_val: str,
    ) -> tuple[str, str | None, str, list[list[Any]], Any, Any, Any, Any]:
        return handlers_run.on_run_selected(
            run_id_val,
            ui_lang_val,
            deps=deps,
        )

    components.run_button.click(
        _on_run_start,
        inputs=[ui_lang],
        outputs=[components.run_status, components.output_file],
        queue=False,
    ).then(
        _on_run,
        inputs=[
            components.input_file,
            components.deck_title,
            components.source_lang,
            components.target_lang,
            components.content_profile,
            components.learning_mode,
            components.difficulty,
            components.run_extract_prompt,
            components.run_explain_prompt,
            components.max_notes,
            components.input_char_limit,
            components.cloze_min_chars,
            components.textbook_max_concepts_per_chunk,
            components.textbook_keep_source_excerpt,
            components.chunk_max_chars,
            components.temperature,
            components.save_intermediate,
            components.continue_on_error,
            ui_lang,
        ],
        outputs=[
            components.run_status,
            components.output_file,
            components.run_selector,
            components.run_detail,
            components.run_download_file,
            analytics.run_analysis,
            analytics.run_samples,
            analytics.taxonomy_filter,
            analytics.transfer_filter,
            analytics.rejection_filter,
            analytics.chunk_filter,
        ],
        queue=True,
    )

    components.refresh_runs_button.click(
        _on_refresh_runs,
        inputs=[ui_lang, components.run_selector],
        outputs=[
            components.run_selector,
            components.run_detail,
            components.run_download_file,
            analytics.run_analysis,
            analytics.run_samples,
            analytics.taxonomy_filter,
            analytics.transfer_filter,
            analytics.rejection_filter,
            analytics.chunk_filter,
        ],
        queue=False,
    )

    components.run_selector.change(
        _on_run_selected,
        inputs=[components.run_selector, ui_lang],
        outputs=[
            components.run_detail,
            components.run_download_file,
            analytics.run_analysis,
            analytics.run_samples,
            analytics.taxonomy_filter,
            analytics.transfer_filter,
            analytics.rejection_filter,
            analytics.chunk_filter,
        ],
        queue=False,
    )
