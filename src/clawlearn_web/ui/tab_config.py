"""Config tab component builder and event wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gradio as gr

from clawlearn_web import handlers_config


@dataclass(frozen=True)
class ConfigTabComponents:
    config_tab: Any
    config_heading: Any
    llm_accordion: Any
    llm_base_url: Any
    llm_api_key: Any
    llm_model: Any
    llm_timeout: Any
    llm_temperature_env: Any
    llm_chunk_batch_size_env: Any
    extract_prompt_env: Any
    llm_list_models_btn: Any
    llm_test_btn: Any
    llm_status: Any
    translate_accordion: Any
    translate_base_url: Any
    translate_api_key: Any
    translate_model: Any
    translate_temperature: Any
    explain_prompt_env: Any
    translate_list_models_btn: Any
    translate_test_btn: Any
    translate_status: Any
    chunk_accordion: Any
    chunk_max_chars_env: Any
    chunk_min_chars_env: Any
    cloze_min_chars_env: Any
    cloze_max_per_chunk_env: Any
    validate_retry_enable_env: Any
    validate_retry_max_env: Any
    validate_retry_llm_enable_env: Any
    lingua_annotate_enable_env: Any
    lingua_annotate_batch_size_env: Any
    lingua_annotate_max_items_env: Any
    content_profile_env: Any
    cloze_difficulty_env: Any
    prompt_lang_env: Any
    paths_accordion: Any
    output_dir_env: Any
    export_dir_env: Any
    log_dir_env: Any
    default_deck_name_env: Any
    tts_accordion: Any
    tts_hint_md: Any
    tts_voice1_env: Any
    tts_voice2_env: Any
    tts_voice3_env: Any
    tts_voice4_env: Any
    load_defaults_btn: Any
    save_config_btn: Any
    save_config_status: Any


def build_tab(
    *,
    initial_ui_lang: str,
    cfg: Any,
    cfg_view: dict[str, str],
    config_extract_prompt_choices: list[tuple[str, str]],
    config_explain_prompt_choices: list[tuple[str, str]],
    tr: Callable[[str, str, str], str],
) -> ConfigTabComponents:
    with gr.Tab(tr(initial_ui_lang, "Config", "閰嶇疆")) as config_tab:
        config_heading = gr.Markdown(
            tr(
                initial_ui_lang,
                "### Config (.env editor)\nWeb UI reads `.env` from the current working directory of the web process.",
                "### 閰嶇疆锛?env 缂栬緫鍣級\nWeb UI reads `.env` from the current working directory of the web process.",
            )
        )

        with gr.Accordion(
            tr(initial_ui_lang, "Extraction LLM", "Extraction LLM"), open=True
        ) as llm_accordion:
            llm_base_url = gr.Textbox(
                label="CLAWLEARN_LLM_BASE_URL",
                value=cfg_view.get("CLAWLEARN_LLM_BASE_URL", ""),
                info=tr(
                    initial_ui_lang,
                    "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).",
                    "OpenAI-compatible base URL before /chat/completions (e.g. .../v1).",
                ),
            )
            llm_api_key = gr.Textbox(
                label="CLAWLEARN_LLM_API_KEY",
                value=cfg_view.get("CLAWLEARN_LLM_API_KEY", ""),
                type="password",
                info=tr(
                    initial_ui_lang,
                    "API key for extraction LLM, when required.",
                    "API key for extraction LLM, when required.",
                ),
            )
            llm_model = gr.Textbox(
                label="CLAWLEARN_LLM_MODEL",
                value=cfg_view.get("CLAWLEARN_LLM_MODEL", ""),
                info=tr(
                    initial_ui_lang,
                    "Model name for extraction LLM.",
                    "Model name for extraction LLM.",
                ),
            )
            llm_timeout = gr.Textbox(
                label="CLAWLEARN_LLM_TIMEOUT_SECONDS",
                value=cfg_view.get("CLAWLEARN_LLM_TIMEOUT_SECONDS", "120"),
                info=tr(
                    initial_ui_lang,
                    "Request timeout in seconds.",
                    "Request timeout in seconds.",
                ),
            )
            llm_temperature_env = gr.Textbox(
                label="CLAWLEARN_LLM_TEMPERATURE",
                value=cfg_view.get("CLAWLEARN_LLM_TEMPERATURE", "0.2"),
                info=tr(
                    initial_ui_lang,
                    "Default temperature for extraction LLM.",
                    "Default temperature for extraction LLM.",
                ),
            )
            llm_chunk_batch_size_env = gr.Textbox(
                label="CLAWLEARN_LLM_CHUNK_BATCH_SIZE",
                value=cfg_view.get("CLAWLEARN_LLM_CHUNK_BATCH_SIZE", "1"),
                info="Chunk batch size for cloze LLM calls; 1 means per-chunk requests.",
            )
            extract_prompt_env = gr.Dropdown(
                choices=config_extract_prompt_choices,
                value=cfg_view.get("CLAWLEARN_EXTRACT_PROMPT", ""),
                label="CLAWLEARN_EXTRACT_PROMPT",
                info=tr(
                    initial_ui_lang,
                    "Default extraction prompt path.",
                    "Default extraction prompt path.",
                ),
            )
            with gr.Row():
                llm_list_models_btn = gr.Button(
                    tr(initial_ui_lang, "List models", "鍒楀嚭妯″瀷")
                )
                llm_test_btn = gr.Button(tr(initial_ui_lang, "Test", "Test"))
            llm_status = gr.Markdown(
                label=tr(
                    initial_ui_lang,
                    "Extraction LLM status",
                    "Extraction LLM status",
                )
            )

        with gr.Accordion(
            tr(initial_ui_lang, "Explanation LLM", "Explanation LLM"), open=False
        ) as translate_accordion:
            translate_base_url = gr.Textbox(
                label="CLAWLEARN_TRANSLATE_LLM_BASE_URL",
                value=cfg_view.get("CLAWLEARN_TRANSLATE_LLM_BASE_URL", ""),
                info=tr(
                    initial_ui_lang,
                    "Optional base URL for explanation model.",
                    "Optional base URL for explanation model.",
                ),
            )
            translate_api_key = gr.Textbox(
                label="CLAWLEARN_TRANSLATE_LLM_API_KEY",
                value=cfg_view.get("CLAWLEARN_TRANSLATE_LLM_API_KEY", ""),
                type="password",
                info=tr(
                    initial_ui_lang,
                    "API key for explanation LLM, when required.",
                    "API key for explanation LLM, when required.",
                ),
            )
            translate_model = gr.Textbox(
                label="CLAWLEARN_TRANSLATE_LLM_MODEL",
                value=cfg_view.get("CLAWLEARN_TRANSLATE_LLM_MODEL", ""),
                info=tr(
                    initial_ui_lang,
                    "Model name for explanation LLM.",
                    "Model name for explanation LLM.",
                ),
            )
            translate_temperature = gr.Textbox(
                label="CLAWLEARN_TRANSLATE_LLM_TEMPERATURE",
                value=cfg_view.get("CLAWLEARN_TRANSLATE_LLM_TEMPERATURE", ""),
                info=tr(
                    initial_ui_lang,
                    "Default temperature for explanation LLM.",
                    "Default temperature for explanation LLM.",
                ),
            )
            explain_prompt_env = gr.Dropdown(
                choices=config_explain_prompt_choices,
                value=cfg_view.get("CLAWLEARN_EXPLAIN_PROMPT", ""),
                label="CLAWLEARN_EXPLAIN_PROMPT",
                info=tr(
                    initial_ui_lang,
                    "Default explanation prompt path.",
                    "Default explanation prompt path.",
                ),
            )
            with gr.Row():
                translate_list_models_btn = gr.Button(
                    tr(
                        initial_ui_lang,
                        "List models (explanation)",
                        "鍒楀嚭瑙ｉ噴妯″瀷",
                    )
                )
                translate_test_btn = gr.Button(
                    tr(initial_ui_lang, "Test (explanation)", "Test (explanation)")
                )
            translate_status = gr.Markdown(
                label=tr(
                    initial_ui_lang,
                    "Explanation LLM status",
                    "Explanation LLM status",
                )
            )

        with gr.Accordion(
            tr(initial_ui_lang, "Chunk & Cloze", "Chunk & Cloze"), open=False
        ) as chunk_accordion:
            chunk_max_chars_env = gr.Textbox(
                label="CLAWLEARN_CHUNK_MAX_CHARS",
                value=cfg_view.get("CLAWLEARN_CHUNK_MAX_CHARS", "1800"),
                info=tr(
                    initial_ui_lang,
                    "Default max chars per chunk.",
                    "Default max chars per chunk.",
                ),
            )
            chunk_min_chars_env = gr.Textbox(
                label="CLAWLEARN_CHUNK_MIN_CHARS",
                value=cfg_view.get("CLAWLEARN_CHUNK_MIN_CHARS", "120"),
            )
            cloze_min_chars_env = gr.Textbox(
                label="CLAWLEARN_CLOZE_MIN_CHARS",
                value=cfg_view.get("CLAWLEARN_CLOZE_MIN_CHARS", "0"),
                info=tr(
                    initial_ui_lang,
                    "Minimum chars required for cloze text.",
                    "Minimum chars required for cloze text.",
                ),
            )
            cloze_max_per_chunk_env = gr.Textbox(
                label="CLAWLEARN_CLOZE_MAX_PER_CHUNK",
                value=cfg_view.get("CLAWLEARN_CLOZE_MAX_PER_CHUNK", ""),
                info=tr(
                    initial_ui_lang,
                    "Max cards per chunk after dedupe. Empty/0 means unlimited.",
                    "Max cards per chunk after dedupe. Empty/0 means unlimited.",
                ),
            )
            validate_retry_enable_env = gr.Textbox(
                label="CLAWLEARN_VALIDATE_FORMAT_RETRY_ENABLE",
                value=cfg_view.get("CLAWLEARN_VALIDATE_FORMAT_RETRY_ENABLE", "true"),
            )
            validate_retry_max_env = gr.Textbox(
                label="CLAWLEARN_VALIDATE_FORMAT_RETRY_MAX",
                value=cfg_view.get("CLAWLEARN_VALIDATE_FORMAT_RETRY_MAX", "3"),
            )
            validate_retry_llm_enable_env = gr.Textbox(
                label="CLAWLEARN_VALIDATE_FORMAT_RETRY_LLM_ENABLE",
                value=cfg_view.get("CLAWLEARN_VALIDATE_FORMAT_RETRY_LLM_ENABLE", "true"),
            )
            lingua_annotate_enable_env = gr.Textbox(
                label="CLAWLEARN_LINGUA_ANNOTATE_ENABLE",
                value=cfg_view.get("CLAWLEARN_LINGUA_ANNOTATE_ENABLE", "false"),
                info=tr(
                    initial_ui_lang,
                    "Enable Lingua taxonomy pre-rank annotation.",
                    "Enable Lingua taxonomy pre-rank annotation.",
                ),
            )
            lingua_annotate_batch_size_env = gr.Textbox(
                label="CLAWLEARN_LINGUA_ANNOTATE_BATCH_SIZE",
                value=cfg_view.get("CLAWLEARN_LINGUA_ANNOTATE_BATCH_SIZE", "50"),
                info=tr(
                    initial_ui_lang,
                    "Batch size for Lingua taxonomy pre-rank requests.",
                    "Batch size for Lingua taxonomy pre-rank requests.",
                ),
            )
            lingua_annotate_max_items_env = gr.Textbox(
                label="CLAWLEARN_LINGUA_ANNOTATE_MAX_ITEMS",
                value=cfg_view.get("CLAWLEARN_LINGUA_ANNOTATE_MAX_ITEMS", ""),
                info=tr(
                    initial_ui_lang,
                    "Optional max candidates to annotate per run. Empty means all.",
                    "Optional max candidates to annotate per run. Empty means all.",
                ),
            )
            content_profile_env = gr.Textbox(
                label="CLAWLEARN_CONTENT_PROFILE",
                value=cfg_view.get("CLAWLEARN_CONTENT_PROFILE", "prose_article"),
            )
            cloze_difficulty_env = gr.Textbox(
                label="CLAWLEARN_CLOZE_DIFFICULTY",
                value=cfg_view.get("CLAWLEARN_CLOZE_DIFFICULTY", "intermediate"),
            )
            prompt_lang_env = gr.Textbox(
                label="CLAWLEARN_PROMPT_LANG",
                value=cfg_view.get("CLAWLEARN_PROMPT_LANG", "zh"),
                info=tr(
                    initial_ui_lang,
                    "Prompt language for multi-lingual prompts (en/zh).",
                    "Prompt language for multi-lingual prompts (en/zh).",
                ),
            )

        with gr.Accordion(
            tr(initial_ui_lang, "Paths & defaults", "Paths & defaults"), open=False
        ) as paths_accordion:
            output_dir_env = gr.Textbox(
                label="CLAWLEARN_OUTPUT_DIR",
                value=cfg_view.get("CLAWLEARN_OUTPUT_DIR", "./runs"),
                info=tr(
                    initial_ui_lang,
                    "Directory for intermediate run data (JSONL, media).",
                    "Directory for intermediate run data (JSONL, media).",
                ),
            )
            export_dir_env = gr.Textbox(
                label="CLAWLEARN_EXPORT_DIR",
                value=cfg_view.get("CLAWLEARN_EXPORT_DIR", "./outputs"),
                info=tr(
                    initial_ui_lang,
                    "Default directory for exported decks.",
                    "Default directory for exported decks.",
                ),
            )
            log_dir_env = gr.Textbox(
                label="CLAWLEARN_LOG_DIR",
                value=cfg_view.get("CLAWLEARN_LOG_DIR", "./logs"),
                info=tr(
                    initial_ui_lang,
                    "Directory for log files.",
                    "Directory for log files.",
                ),
            )
            default_deck_name_env = gr.Textbox(
                label="CLAWLEARN_DEFAULT_DECK_NAME",
                value=cfg_view.get("CLAWLEARN_DEFAULT_DECK_NAME", cfg.default_deck_name),
            )

        with gr.Accordion(
            tr(initial_ui_lang, "TTS voices (Edge)", "TTS voices (Edge)"),
            open=False,
        ) as tts_accordion:
            tts_hint_md = gr.Markdown(
                tr(
                    initial_ui_lang,
                    "Voice reference: [Edge TTS Voice Samples](https://tts.travisvn.com/)",
                    "鍏蜂綋鐨勯煶鑹插彲浠ュ弬鑰僛Edge TTS Voice Samples](https://tts.travisvn.com/)",
                )
            )
            tts_voice1_env = gr.Textbox(
                label="CLAWLEARN_TTS_EDGE_VOICE1",
                value=cfg_view.get("CLAWLEARN_TTS_EDGE_VOICE1", ""),
                info=tr(
                    initial_ui_lang,
                    "Configure 4 voice slots used for random selection.",
                    "Configure 4 voice slots used for random selection.",
                ),
            )
            tts_voice2_env = gr.Textbox(
                label="CLAWLEARN_TTS_EDGE_VOICE2",
                value=cfg_view.get("CLAWLEARN_TTS_EDGE_VOICE2", ""),
            )
            tts_voice3_env = gr.Textbox(
                label="CLAWLEARN_TTS_EDGE_VOICE3",
                value=cfg_view.get("CLAWLEARN_TTS_EDGE_VOICE3", ""),
            )
            tts_voice4_env = gr.Textbox(
                label="CLAWLEARN_TTS_EDGE_VOICE4",
                value=cfg_view.get("CLAWLEARN_TTS_EDGE_VOICE4", ""),
            )

        with gr.Row():
            load_defaults_btn = gr.Button(
                tr(
                    initial_ui_lang,
                    "Load defaults from ENV_EXAMPLE.md",
                    "Load defaults from ENV_EXAMPLE.md",
                )
            )
            save_config_btn = gr.Button(tr(initial_ui_lang, "Save config", "淇濆瓨閰嶇疆"))
        save_config_status = gr.Markdown()

    return ConfigTabComponents(
        config_tab=config_tab,
        config_heading=config_heading,
        llm_accordion=llm_accordion,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_timeout=llm_timeout,
        llm_temperature_env=llm_temperature_env,
        llm_chunk_batch_size_env=llm_chunk_batch_size_env,
        extract_prompt_env=extract_prompt_env,
        llm_list_models_btn=llm_list_models_btn,
        llm_test_btn=llm_test_btn,
        llm_status=llm_status,
        translate_accordion=translate_accordion,
        translate_base_url=translate_base_url,
        translate_api_key=translate_api_key,
        translate_model=translate_model,
        translate_temperature=translate_temperature,
        explain_prompt_env=explain_prompt_env,
        translate_list_models_btn=translate_list_models_btn,
        translate_test_btn=translate_test_btn,
        translate_status=translate_status,
        chunk_accordion=chunk_accordion,
        chunk_max_chars_env=chunk_max_chars_env,
        chunk_min_chars_env=chunk_min_chars_env,
        cloze_min_chars_env=cloze_min_chars_env,
        cloze_max_per_chunk_env=cloze_max_per_chunk_env,
        validate_retry_enable_env=validate_retry_enable_env,
        validate_retry_max_env=validate_retry_max_env,
        validate_retry_llm_enable_env=validate_retry_llm_enable_env,
        lingua_annotate_enable_env=lingua_annotate_enable_env,
        lingua_annotate_batch_size_env=lingua_annotate_batch_size_env,
        lingua_annotate_max_items_env=lingua_annotate_max_items_env,
        content_profile_env=content_profile_env,
        cloze_difficulty_env=cloze_difficulty_env,
        prompt_lang_env=prompt_lang_env,
        paths_accordion=paths_accordion,
        output_dir_env=output_dir_env,
        export_dir_env=export_dir_env,
        log_dir_env=log_dir_env,
        default_deck_name_env=default_deck_name_env,
        tts_accordion=tts_accordion,
        tts_hint_md=tts_hint_md,
        tts_voice1_env=tts_voice1_env,
        tts_voice2_env=tts_voice2_env,
        tts_voice3_env=tts_voice3_env,
        tts_voice4_env=tts_voice4_env,
        load_defaults_btn=load_defaults_btn,
        save_config_btn=save_config_btn,
        save_config_status=save_config_status,
    )


def bind_events(
    *,
    components: ConfigTabComponents,
    ui_lang: Any,
    deps: handlers_config.ConfigDeps,
) -> None:
    config_value_outputs = [
        components.llm_base_url,
        components.llm_api_key,
        components.llm_model,
        components.llm_timeout,
        components.llm_temperature_env,
        components.llm_chunk_batch_size_env,
        components.translate_base_url,
        components.translate_api_key,
        components.translate_model,
        components.translate_temperature,
        components.chunk_max_chars_env,
        components.chunk_min_chars_env,
        components.cloze_min_chars_env,
        components.cloze_max_per_chunk_env,
        components.validate_retry_enable_env,
        components.validate_retry_max_env,
        components.validate_retry_llm_enable_env,
        components.lingua_annotate_enable_env,
        components.lingua_annotate_batch_size_env,
        components.lingua_annotate_max_items_env,
        components.content_profile_env,
        components.cloze_difficulty_env,
        components.prompt_lang_env,
        components.extract_prompt_env,
        components.explain_prompt_env,
        components.output_dir_env,
        components.export_dir_env,
        components.log_dir_env,
        components.default_deck_name_env,
        components.tts_voice1_env,
        components.tts_voice2_env,
        components.tts_voice3_env,
        components.tts_voice4_env,
    ]

    def _on_list_models(base_url: str, api_key: str, timeout_raw: Any, ui_lang_val: str) -> str:
        return handlers_config.on_list_models(
            base_url,
            api_key,
            timeout_raw,
            ui_lang_val,
            deps=deps,
        )

    def _on_test_models(base_url: str, api_key: str, timeout_raw: Any, ui_lang_val: str) -> str:
        return handlers_config.on_test_models(
            base_url,
            api_key,
            timeout_raw,
            ui_lang_val,
            deps=deps,
        )

    components.llm_list_models_btn.click(
        _on_list_models,
        inputs=[components.llm_base_url, components.llm_api_key, components.llm_timeout, ui_lang],
        outputs=[components.llm_status],
    )
    components.llm_test_btn.click(
        _on_test_models,
        inputs=[components.llm_base_url, components.llm_api_key, components.llm_timeout, ui_lang],
        outputs=[components.llm_status],
    )
    components.translate_list_models_btn.click(
        _on_list_models,
        inputs=[
            components.translate_base_url,
            components.translate_api_key,
            components.llm_timeout,
            ui_lang,
        ],
        outputs=[components.translate_status],
    )
    components.translate_test_btn.click(
        _on_test_models,
        inputs=[
            components.translate_base_url,
            components.translate_api_key,
            components.llm_timeout,
            ui_lang,
        ],
        outputs=[components.translate_status],
    )

    def _on_load_defaults(
        llm_base_url_val: str,
        llm_api_key_val: str,
        llm_model_val: str,
        llm_timeout_val: str,
        llm_temperature_val: str,
        llm_chunk_batch_size_val: str,
        translate_base_url_val: str,
        translate_api_key_val: str,
        translate_model_val: str,
        translate_temperature_val: str,
        chunk_max_chars_val: str,
        chunk_min_chars_val: str,
        cloze_min_chars_val: str,
        cloze_max_per_chunk_val: str,
        validate_retry_enable_val: str,
        validate_retry_max_val: str,
        validate_retry_llm_enable_val: str,
        lingua_annotate_enable_val: str,
        lingua_annotate_batch_size_val: str,
        lingua_annotate_max_items_val: str,
        content_profile_val: str,
        cloze_difficulty_val: str,
        prompt_lang_val: str,
        extract_prompt_env_val: str,
        explain_prompt_env_val: str,
        output_dir_val: str,
        export_dir_val: str,
        log_dir_val: str,
        default_deck_name_val: str,
        tts_voice1_val: str,
        tts_voice2_val: str,
        tts_voice3_val: str,
        tts_voice4_val: str,
        ui_lang_val: str,
    ) -> tuple[str, ...]:
        return handlers_config.on_load_defaults(
            llm_base_url_val,
            llm_api_key_val,
            llm_model_val,
            llm_timeout_val,
            llm_temperature_val,
            llm_chunk_batch_size_val,
            translate_base_url_val,
            translate_api_key_val,
            translate_model_val,
            translate_temperature_val,
            chunk_max_chars_val,
            chunk_min_chars_val,
            cloze_min_chars_val,
            cloze_max_per_chunk_val,
            validate_retry_enable_val,
            validate_retry_max_val,
            validate_retry_llm_enable_val,
            lingua_annotate_enable_val,
            lingua_annotate_batch_size_val,
            lingua_annotate_max_items_val,
            content_profile_val,
            cloze_difficulty_val,
            prompt_lang_val,
            extract_prompt_env_val,
            explain_prompt_env_val,
            output_dir_val,
            export_dir_val,
            log_dir_val,
            default_deck_name_val,
            tts_voice1_val,
            tts_voice2_val,
            tts_voice3_val,
            tts_voice4_val,
            ui_lang_val,
            deps=deps,
        )

    components.load_defaults_btn.click(
        _on_load_defaults,
        inputs=[
            components.llm_base_url,
            components.llm_api_key,
            components.llm_model,
            components.llm_timeout,
            components.llm_temperature_env,
            components.llm_chunk_batch_size_env,
            components.translate_base_url,
            components.translate_api_key,
            components.translate_model,
            components.translate_temperature,
            components.chunk_max_chars_env,
            components.chunk_min_chars_env,
            components.cloze_min_chars_env,
            components.cloze_max_per_chunk_env,
            components.validate_retry_enable_env,
            components.validate_retry_max_env,
            components.validate_retry_llm_enable_env,
            components.lingua_annotate_enable_env,
            components.lingua_annotate_batch_size_env,
            components.lingua_annotate_max_items_env,
            components.content_profile_env,
            components.cloze_difficulty_env,
            components.prompt_lang_env,
            components.extract_prompt_env,
            components.explain_prompt_env,
            components.output_dir_env,
            components.export_dir_env,
            components.log_dir_env,
            components.default_deck_name_env,
            components.tts_voice1_env,
            components.tts_voice2_env,
            components.tts_voice3_env,
            components.tts_voice4_env,
            ui_lang,
        ],
        outputs=config_value_outputs + [components.save_config_status],
    )

    def _on_save_config(
        llm_base_url_val: Any,
        llm_api_key_val: Any,
        llm_model_val: Any,
        llm_timeout_val: Any,
        llm_temperature_val: Any,
        llm_chunk_batch_size_val: Any,
        translate_base_url_val: Any,
        translate_api_key_val: Any,
        translate_model_val: Any,
        translate_temperature_val: Any,
        chunk_max_chars_val: Any,
        chunk_min_chars_val: Any,
        cloze_min_chars_val: Any,
        cloze_max_per_chunk_val: Any,
        validate_retry_enable_val: Any,
        validate_retry_max_val: Any,
        validate_retry_llm_enable_val: Any,
        lingua_annotate_enable_val: Any,
        lingua_annotate_batch_size_val: Any,
        lingua_annotate_max_items_val: Any,
        content_profile_val: Any,
        cloze_difficulty_val: Any,
        prompt_lang_val: Any,
        extract_prompt_env_val: Any,
        explain_prompt_env_val: Any,
        output_dir_val: Any,
        export_dir_val: Any,
        log_dir_val: Any,
        default_deck_name_val: Any,
        tts_voice1_val: Any,
        tts_voice2_val: Any,
        tts_voice3_val: Any,
        tts_voice4_val: Any,
        ui_lang_val: Any,
    ) -> tuple[str, ...]:
        return handlers_config.on_save_config(
            llm_base_url_val,
            llm_api_key_val,
            llm_model_val,
            llm_timeout_val,
            llm_temperature_val,
            llm_chunk_batch_size_val,
            translate_base_url_val,
            translate_api_key_val,
            translate_model_val,
            translate_temperature_val,
            chunk_max_chars_val,
            chunk_min_chars_val,
            cloze_min_chars_val,
            cloze_max_per_chunk_val,
            validate_retry_enable_val,
            validate_retry_max_val,
            validate_retry_llm_enable_val,
            lingua_annotate_enable_val,
            lingua_annotate_batch_size_val,
            lingua_annotate_max_items_val,
            content_profile_val,
            cloze_difficulty_val,
            prompt_lang_val,
            extract_prompt_env_val,
            explain_prompt_env_val,
            output_dir_val,
            export_dir_val,
            log_dir_val,
            default_deck_name_val,
            tts_voice1_val,
            tts_voice2_val,
            tts_voice3_val,
            tts_voice4_val,
            ui_lang_val,
            deps=deps,
        )

    def _on_reload_env(ui_lang_val: Any) -> tuple[str, ...]:
        return handlers_config.on_reload_env(ui_lang_val, deps=deps)

    components.config_tab.select(
        _on_reload_env,
        inputs=[ui_lang],
        outputs=config_value_outputs + [components.save_config_status],
        queue=False,
    )

    components.save_config_btn.click(
        _on_save_config,
        inputs=[
            components.llm_base_url,
            components.llm_api_key,
            components.llm_model,
            components.llm_timeout,
            components.llm_temperature_env,
            components.llm_chunk_batch_size_env,
            components.translate_base_url,
            components.translate_api_key,
            components.translate_model,
            components.translate_temperature,
            components.chunk_max_chars_env,
            components.chunk_min_chars_env,
            components.cloze_min_chars_env,
            components.cloze_max_per_chunk_env,
            components.validate_retry_enable_env,
            components.validate_retry_max_env,
            components.validate_retry_llm_enable_env,
            components.lingua_annotate_enable_env,
            components.lingua_annotate_batch_size_env,
            components.lingua_annotate_max_items_env,
            components.content_profile_env,
            components.cloze_difficulty_env,
            components.prompt_lang_env,
            components.extract_prompt_env,
            components.explain_prompt_env,
            components.output_dir_env,
            components.export_dir_env,
            components.log_dir_env,
            components.default_deck_name_env,
            components.tts_voice1_env,
            components.tts_voice2_env,
            components.tts_voice3_env,
            components.tts_voice4_env,
            ui_lang,
        ],
        outputs=config_value_outputs + [components.save_config_status],
    )
