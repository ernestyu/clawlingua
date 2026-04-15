"""Prompt tab component builder and event wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

import gradio as gr

from clawlearn_web import handlers_prompt

if TYPE_CHECKING:
    from clawlearn_web.ui.tab_config import ConfigTabComponents
    from clawlearn_web.ui.tab_run import RunTabComponents


@dataclass(frozen=True)
class PromptTabComponents:
    prompt_tab: Any
    prompt_heading: Any
    prompt_content_type_selector: Any
    prompt_learning_mode_selector: Any
    prompt_difficulty_selector: Any
    prompt_file_selector: Any
    prompt_mode_selector: Any
    prompt_new_name: Any
    prompt_rename_name: Any
    prompt_editor: Any
    prompt_new_btn: Any
    prompt_save_btn: Any
    prompt_rename_btn: Any
    prompt_load_default_btn: Any
    prompt_save_confirm: Any
    prompt_delete_confirm: Any
    prompt_status: Any


def build_tab(
    *,
    initial_ui_lang: str,
    initial_prompt_content_type: str,
    initial_prompt_learning_mode: str,
    initial_prompt_difficulty: str,
    initial_prompt_choices: list[tuple[str, str]],
    initial_prompt_key: str,
    initial_prompt_mode: str,
    initial_prompt_text: str,
    initial_prompt_status: str,
    prompt_content_type_options: list[str],
    prompt_learning_mode_options: list[str],
    prompt_difficulty_options: list[str],
    prompt_mode_label: Callable[..., str],
    tr: Callable[[str, str, str], str],
) -> PromptTabComponents:
    with gr.Tab(tr(initial_ui_lang, "Prompt", "Prompt")) as prompt_tab:
        prompt_heading = gr.Markdown(
            tr(
                initial_ui_lang,
                "### Prompt template editor",
                "### Prompt template editor",
            )
        )
        with gr.Row():
            prompt_content_type_selector = gr.Dropdown(
                choices=prompt_content_type_options,
                value=initial_prompt_content_type,
                label=tr(initial_ui_lang, "Prompt content type", "Prompt content type"),
                scale=1,
            )
            prompt_learning_mode_selector = gr.Dropdown(
                choices=prompt_learning_mode_options,
                value=initial_prompt_learning_mode,
                label=tr(initial_ui_lang, "Prompt learning mode", "Prompt learning mode"),
                scale=1,
            )
            prompt_difficulty_selector = gr.Dropdown(
                choices=prompt_difficulty_options,
                value=initial_prompt_difficulty,
                label=tr(initial_ui_lang, "Prompt difficulty", "Prompt difficulty"),
                scale=1,
            )
        with gr.Row():
            prompt_file_selector = gr.Dropdown(
                choices=initial_prompt_choices,
                value=initial_prompt_key,
                label=tr(initial_ui_lang, "Prompt file", "Prompt 鏂囦欢"),
                scale=2,
            )
            prompt_mode_selector = gr.Dropdown(
                choices=[
                    (prompt_mode_label("extraction", lang=initial_ui_lang), "extraction"),
                    (prompt_mode_label("explanation", lang=initial_ui_lang), "explanation"),
                ],
                value=initial_prompt_mode or "extraction",
                label=tr(initial_ui_lang, "Prompt type", "Prompt type"),
                scale=1,
            )
        with gr.Row():
            prompt_new_name = gr.Textbox(
                label=tr(initial_ui_lang, "New prompt file name", "New prompt file name"),
                placeholder="my_prompt.json",
            )
            prompt_rename_name = gr.Textbox(
                label=tr(initial_ui_lang, "Rename to", "閲嶅懡鍚嶄负"),
                placeholder="renamed_prompt.json",
            )
        prompt_editor = gr.Textbox(
            label=tr(initial_ui_lang, "Prompt template", "Prompt 妯℃澘"),
            value=initial_prompt_text,
            lines=24,
        )
        with gr.Row():
            prompt_new_btn = gr.Button(tr(initial_ui_lang, "New", "鏂板缓"))
            prompt_save_btn = gr.Button(tr(initial_ui_lang, "Save", "淇濆瓨"))
            prompt_rename_btn = gr.Button(tr(initial_ui_lang, "Rename", "Rename"))
            prompt_load_default_btn = gr.Button(
                tr(initial_ui_lang, "Delete", "鍒犻櫎"), variant="stop"
            )
        prompt_save_confirm = gr.Checkbox(value=False, visible=False)
        prompt_delete_confirm = gr.Checkbox(value=False, visible=False)
        prompt_status = gr.Markdown(
            label=tr(initial_ui_lang, "Prompt status", "Prompt status"),
            value=initial_prompt_status,
        )

    return PromptTabComponents(
        prompt_tab=prompt_tab,
        prompt_heading=prompt_heading,
        prompt_content_type_selector=prompt_content_type_selector,
        prompt_learning_mode_selector=prompt_learning_mode_selector,
        prompt_difficulty_selector=prompt_difficulty_selector,
        prompt_file_selector=prompt_file_selector,
        prompt_mode_selector=prompt_mode_selector,
        prompt_new_name=prompt_new_name,
        prompt_rename_name=prompt_rename_name,
        prompt_editor=prompt_editor,
        prompt_new_btn=prompt_new_btn,
        prompt_save_btn=prompt_save_btn,
        prompt_rename_btn=prompt_rename_btn,
        prompt_load_default_btn=prompt_load_default_btn,
        prompt_save_confirm=prompt_save_confirm,
        prompt_delete_confirm=prompt_delete_confirm,
        prompt_status=prompt_status,
    )


def bind_events(
    *,
    components: PromptTabComponents,
    run_tab: "RunTabComponents",
    config_tab: "ConfigTabComponents",
    ui_lang: Any,
    deps: handlers_prompt.PromptDeps,
) -> None:
    def _on_prompt_file_change(
        prompt_key: str,
        prompt_mode: str,
        prompt_content_type: str,
        prompt_learning_mode: str,
        prompt_difficulty: str,
        run_content_type: str,
        run_learning_mode: str,
        run_difficulty: str,
        run_extract_val: str,
        run_explain_val: str,
        config_extract_val: str,
        config_explain_val: str,
        ui_lang_val: str,
    ) -> tuple[Any, ...]:
        return handlers_prompt.on_prompt_file_change(
            prompt_key,
            prompt_mode,
            prompt_content_type,
            prompt_learning_mode,
            prompt_difficulty,
            run_content_type,
            run_learning_mode,
            run_difficulty,
            run_extract_val,
            run_explain_val,
            config_extract_val,
            config_explain_val,
            ui_lang_val,
            deps=deps,
        )

    def _on_prompt_mode_change(
        prompt_mode: str,
        prompt_key: str,
        prompt_content_type: str,
        prompt_learning_mode: str,
        prompt_difficulty: str,
        run_content_type: str,
        run_learning_mode: str,
        run_difficulty: str,
        run_extract_val: str,
        run_explain_val: str,
        config_extract_val: str,
        config_explain_val: str,
        ui_lang_val: str,
    ) -> tuple[Any, ...]:
        return handlers_prompt.on_prompt_mode_change(
            prompt_mode,
            prompt_key,
            prompt_content_type,
            prompt_learning_mode,
            prompt_difficulty,
            run_content_type,
            run_learning_mode,
            run_difficulty,
            run_extract_val,
            run_explain_val,
            config_extract_val,
            config_explain_val,
            ui_lang_val,
            deps=deps,
        )

    def _on_prompt_filter_change(
        prompt_content_type: str,
        prompt_learning_mode: str,
        prompt_difficulty: str,
        prompt_mode: str,
        prompt_key: str,
        run_content_type: str,
        run_learning_mode: str,
        run_difficulty: str,
        run_extract_val: str,
        run_explain_val: str,
        config_extract_val: str,
        config_explain_val: str,
        ui_lang_val: str,
    ) -> tuple[Any, ...]:
        return handlers_prompt.on_prompt_filter_change(
            prompt_content_type,
            prompt_learning_mode,
            prompt_difficulty,
            prompt_mode,
            prompt_key,
            run_content_type,
            run_learning_mode,
            run_difficulty,
            run_extract_val,
            run_explain_val,
            config_extract_val,
            config_explain_val,
            ui_lang_val,
            deps=deps,
        )

    def _on_prompt_new(
        prompt_key: str,
        new_name: str,
        prompt_mode: str,
        prompt_content_type: str,
        prompt_learning_mode: str,
        prompt_difficulty: str,
        run_content_type: str,
        run_learning_mode: str,
        run_difficulty: str,
        run_extract_val: str,
        run_explain_val: str,
        config_extract_val: str,
        config_explain_val: str,
        ui_lang_val: str,
    ) -> tuple[Any, ...]:
        return handlers_prompt.on_prompt_new(
            prompt_key,
            new_name,
            prompt_mode,
            prompt_content_type,
            prompt_learning_mode,
            prompt_difficulty,
            run_content_type,
            run_learning_mode,
            run_difficulty,
            run_extract_val,
            run_explain_val,
            config_extract_val,
            config_explain_val,
            ui_lang_val,
            deps=deps,
        )

    def _on_prompt_save(
        prompt_key: str,
        prompt_mode: str,
        prompt_content_type: str,
        prompt_learning_mode: str,
        prompt_difficulty: str,
        prompt_template: str,
        save_confirmed: bool,
        run_content_type: str,
        run_learning_mode: str,
        run_difficulty: str,
        run_extract_val: str,
        run_explain_val: str,
        config_extract_val: str,
        config_explain_val: str,
        ui_lang_val: str,
    ) -> tuple[Any, ...]:
        return handlers_prompt.on_prompt_save(
            prompt_key,
            prompt_mode,
            prompt_content_type,
            prompt_learning_mode,
            prompt_difficulty,
            prompt_template,
            save_confirmed,
            run_content_type,
            run_learning_mode,
            run_difficulty,
            run_extract_val,
            run_explain_val,
            config_extract_val,
            config_explain_val,
            ui_lang_val,
            deps=deps,
        )

    def _on_prompt_rename(
        prompt_key: str,
        rename_name: str,
        prompt_mode: str,
        prompt_content_type: str,
        prompt_learning_mode: str,
        prompt_difficulty: str,
        run_content_type: str,
        run_learning_mode: str,
        run_difficulty: str,
        run_extract_val: str,
        run_explain_val: str,
        config_extract_val: str,
        config_explain_val: str,
        ui_lang_val: str,
    ) -> tuple[Any, ...]:
        return handlers_prompt.on_prompt_rename(
            prompt_key,
            rename_name,
            prompt_mode,
            prompt_content_type,
            prompt_learning_mode,
            prompt_difficulty,
            run_content_type,
            run_learning_mode,
            run_difficulty,
            run_extract_val,
            run_explain_val,
            config_extract_val,
            config_explain_val,
            ui_lang_val,
            deps=deps,
        )

    def _on_prompt_delete(
        prompt_key: str,
        delete_confirmed: bool,
        prompt_mode: str,
        prompt_content_type: str,
        prompt_learning_mode: str,
        prompt_difficulty: str,
        run_content_type: str,
        run_learning_mode: str,
        run_difficulty: str,
        run_extract_val: str,
        run_explain_val: str,
        config_extract_val: str,
        config_explain_val: str,
        ui_lang_val: str,
    ) -> tuple[Any, ...]:
        return handlers_prompt.on_prompt_delete(
            prompt_key,
            delete_confirmed,
            prompt_mode,
            prompt_content_type,
            prompt_learning_mode,
            prompt_difficulty,
            run_content_type,
            run_learning_mode,
            run_difficulty,
            run_extract_val,
            run_explain_val,
            config_extract_val,
            config_explain_val,
            ui_lang_val,
            deps=deps,
        )

    def _on_run_prompt_filters_change(
        run_content_type: str,
        run_learning_mode: str,
        run_difficulty: str,
        run_extract_val: str,
        run_explain_val: str,
        ui_lang_val: str,
    ) -> tuple[Any, Any]:
        return handlers_prompt.on_run_prompt_filters_change(
            run_content_type,
            run_learning_mode,
            run_difficulty,
            run_extract_val,
            run_explain_val,
            ui_lang_val,
            deps=deps,
        )

    shared_prompt_inputs = [
        run_tab.content_profile,
        run_tab.learning_mode,
        run_tab.difficulty,
        run_tab.run_extract_prompt,
        run_tab.run_explain_prompt,
        config_tab.extract_prompt_env,
        config_tab.explain_prompt_env,
        ui_lang,
    ]

    shared_prompt_outputs = [
        components.prompt_file_selector,
        components.prompt_mode_selector,
        components.prompt_content_type_selector,
        components.prompt_learning_mode_selector,
        components.prompt_difficulty_selector,
        components.prompt_editor,
        components.prompt_status,
        run_tab.run_extract_prompt,
        run_tab.run_explain_prompt,
        config_tab.extract_prompt_env,
        config_tab.explain_prompt_env,
    ]

    components.prompt_file_selector.change(
        _on_prompt_file_change,
        inputs=[
            components.prompt_file_selector,
            components.prompt_mode_selector,
            components.prompt_content_type_selector,
            components.prompt_learning_mode_selector,
            components.prompt_difficulty_selector,
            *shared_prompt_inputs,
        ],
        outputs=shared_prompt_outputs,
    )
    components.prompt_mode_selector.change(
        _on_prompt_mode_change,
        inputs=[
            components.prompt_mode_selector,
            components.prompt_file_selector,
            components.prompt_content_type_selector,
            components.prompt_learning_mode_selector,
            components.prompt_difficulty_selector,
            *shared_prompt_inputs,
        ],
        outputs=shared_prompt_outputs,
    )
    components.prompt_content_type_selector.change(
        _on_prompt_filter_change,
        inputs=[
            components.prompt_content_type_selector,
            components.prompt_learning_mode_selector,
            components.prompt_difficulty_selector,
            components.prompt_mode_selector,
            components.prompt_file_selector,
            *shared_prompt_inputs,
        ],
        outputs=shared_prompt_outputs,
    )
    components.prompt_learning_mode_selector.change(
        _on_prompt_filter_change,
        inputs=[
            components.prompt_content_type_selector,
            components.prompt_learning_mode_selector,
            components.prompt_difficulty_selector,
            components.prompt_mode_selector,
            components.prompt_file_selector,
            *shared_prompt_inputs,
        ],
        outputs=shared_prompt_outputs,
    )
    components.prompt_difficulty_selector.change(
        _on_prompt_filter_change,
        inputs=[
            components.prompt_content_type_selector,
            components.prompt_learning_mode_selector,
            components.prompt_difficulty_selector,
            components.prompt_mode_selector,
            components.prompt_file_selector,
            *shared_prompt_inputs,
        ],
        outputs=shared_prompt_outputs,
    )
    run_tab.content_profile.change(
        _on_run_prompt_filters_change,
        inputs=[
            run_tab.content_profile,
            run_tab.learning_mode,
            run_tab.difficulty,
            run_tab.run_extract_prompt,
            run_tab.run_explain_prompt,
            ui_lang,
        ],
        outputs=[run_tab.run_extract_prompt, run_tab.run_explain_prompt],
    )
    run_tab.learning_mode.change(
        _on_run_prompt_filters_change,
        inputs=[
            run_tab.content_profile,
            run_tab.learning_mode,
            run_tab.difficulty,
            run_tab.run_extract_prompt,
            run_tab.run_explain_prompt,
            ui_lang,
        ],
        outputs=[run_tab.run_extract_prompt, run_tab.run_explain_prompt],
    )
    run_tab.difficulty.change(
        _on_run_prompt_filters_change,
        inputs=[
            run_tab.content_profile,
            run_tab.learning_mode,
            run_tab.difficulty,
            run_tab.run_extract_prompt,
            run_tab.run_explain_prompt,
            ui_lang,
        ],
        outputs=[run_tab.run_extract_prompt, run_tab.run_explain_prompt],
    )
    components.prompt_new_btn.click(
        _on_prompt_new,
        inputs=[
            components.prompt_file_selector,
            components.prompt_new_name,
            components.prompt_mode_selector,
            components.prompt_content_type_selector,
            components.prompt_learning_mode_selector,
            components.prompt_difficulty_selector,
            *shared_prompt_inputs,
        ],
        outputs=[
            *shared_prompt_outputs,
            components.prompt_new_name,
            components.prompt_rename_name,
            components.prompt_save_confirm,
            components.prompt_delete_confirm,
        ],
    )
    components.prompt_save_btn.click(
        _on_prompt_save,
        inputs=[
            components.prompt_file_selector,
            components.prompt_mode_selector,
            components.prompt_content_type_selector,
            components.prompt_learning_mode_selector,
            components.prompt_difficulty_selector,
            components.prompt_editor,
            components.prompt_save_confirm,
            *shared_prompt_inputs,
        ],
        outputs=[
            *shared_prompt_outputs,
            components.prompt_new_name,
            components.prompt_rename_name,
            components.prompt_save_confirm,
            components.prompt_delete_confirm,
        ],
        js="""
(prompt_key, prompt_mode, prompt_content_type, prompt_learning_mode, prompt_difficulty, prompt_template, _save_confirmed, run_content_type, run_learning_mode, run_difficulty, run_extract_val, run_explain_val, config_extract_val, config_explain_val, ui_lang_val) => {
    const message = ui_lang_val === "zh" ? "确认保存当前提示词文件？" : "Confirm saving the current prompt file?";
    return [
        prompt_key,
        prompt_mode,
        prompt_content_type,
        prompt_learning_mode,
        prompt_difficulty,
        prompt_template,
        window.confirm(message),
        run_content_type,
        run_learning_mode,
        run_difficulty,
        run_extract_val,
        run_explain_val,
        config_extract_val,
        config_explain_val,
        ui_lang_val,
    ];
}
""",
    )
    components.prompt_rename_btn.click(
        _on_prompt_rename,
        inputs=[
            components.prompt_file_selector,
            components.prompt_rename_name,
            components.prompt_mode_selector,
            components.prompt_content_type_selector,
            components.prompt_learning_mode_selector,
            components.prompt_difficulty_selector,
            *shared_prompt_inputs,
        ],
        outputs=[
            *shared_prompt_outputs,
            components.prompt_new_name,
            components.prompt_rename_name,
            components.prompt_save_confirm,
            components.prompt_delete_confirm,
        ],
    )
    components.prompt_load_default_btn.click(
        _on_prompt_delete,
        inputs=[
            components.prompt_file_selector,
            components.prompt_delete_confirm,
            components.prompt_mode_selector,
            components.prompt_content_type_selector,
            components.prompt_learning_mode_selector,
            components.prompt_difficulty_selector,
            *shared_prompt_inputs,
        ],
        outputs=[
            *shared_prompt_outputs,
            components.prompt_new_name,
            components.prompt_rename_name,
            components.prompt_save_confirm,
            components.prompt_delete_confirm,
        ],
        js="""
(prompt_key, _delete_confirmed, prompt_mode, prompt_content_type, prompt_learning_mode, prompt_difficulty, run_content_type, run_learning_mode, run_difficulty, run_extract_val, run_explain_val, config_extract_val, config_explain_val, ui_lang_val) => {
    const message = ui_lang_val === "zh" ? "确认删除当前提示词文件？" : "Confirm deleting the current prompt file?";
    return [
        prompt_key,
        window.confirm(message),
        prompt_mode,
        prompt_content_type,
        prompt_learning_mode,
        prompt_difficulty,
        run_content_type,
        run_learning_mode,
        run_difficulty,
        run_extract_val,
        run_explain_val,
        config_extract_val,
        config_explain_val,
        ui_lang_val,
    ];
}
""",
    )
