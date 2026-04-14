"""Prompt-tab event handlers extracted from app.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gradio as gr


@dataclass(frozen=True)
class PromptDeps:
    normalize_ui_lang: Callable[[str | None], str]
    tr: Callable[[str, str, str], str]
    load_app_config: Callable[[], Any]
    normalize_prompt_mode: Callable[[Any], str]
    normalize_prompt_content_type: Callable[[Any], str]
    normalize_prompt_learning_mode: Callable[[Any], str]
    normalize_prompt_difficulty: Callable[[Any], str]
    prompt_mode_label: Callable[..., str]
    prompt_file_map: Callable[..., dict[str, Path]]
    prompt_choices_from_map: Callable[..., list[tuple[str, str]]]
    prompt_path_choices: Callable[..., list[tuple[str, str]]]
    load_prompt_template: Callable[..., tuple[str, str]]
    load_prompt_mode: Callable[..., str]
    load_prompt_filter_metadata: Callable[..., tuple[str, str, str]]
    sanitize_prompt_filename: Callable[[str], str]
    prompt_content_type_options: list[str]
    prompt_learning_mode_options: list[str]
    prompt_difficulty_options: list[str]
    prompt_io: Any


def prompt_mode_choices_for_ui(lang: str, *, deps: PromptDeps) -> list[tuple[str, str]]:
    return [
        (deps.prompt_mode_label("extraction", lang=lang), "extraction"),
        (deps.prompt_mode_label("explanation", lang=lang), "explanation"),
    ]


def normalize_dropdown_value(current: str, choices: list[Any]) -> str:
    valid_values: set[str] = set()
    for item in choices:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            valid_values.add(as_str(item[1]))
        else:
            valid_values.add(as_str(item))
    current_value = as_str(current)
    return current_value if current_value in valid_values else ""


def pick_prompt_key(
    prompt_files_now: dict[str, Path],
    *,
    lang: str,
    preferred_key: str = "",
    preferred_mode: str = "",
    deps: PromptDeps,
) -> str:
    if preferred_key in prompt_files_now:
        return preferred_key
    mode_value = deps.normalize_prompt_mode(preferred_mode)
    if mode_value:
        for key in prompt_files_now:
            if deps.load_prompt_mode(key, prompt_files_now, lang=lang) == mode_value:
                return key
    if prompt_files_now:
        return next(iter(prompt_files_now))
    return ""


def refresh_prompt_controls(
    *,
    lang: str,
    prompt_key: str,
    preferred_mode: str,
    preferred_content_type: str = "all",
    preferred_learning_mode: str = "all",
    preferred_difficulty: str = "all",
    status: str,
    editor_override: str | None,
    run_content_type: str = "all",
    run_learning_mode: str = "all",
    run_difficulty: str = "all",
    run_extract_current: str = "",
    run_explain_current: str = "",
    config_extract_current: str = "",
    config_explain_current: str = "",
    deps: PromptDeps,
) -> tuple[Any, ...]:
    cfg_now = deps.load_app_config()
    prompt_files_now = deps.prompt_file_map(cfg_now)
    selected_mode = deps.normalize_prompt_mode(preferred_mode)
    if not selected_mode and prompt_key in prompt_files_now:
        selected_mode = deps.load_prompt_mode(prompt_key, prompt_files_now, lang=lang)
    if not selected_mode:
        selected_mode = "extraction"
    selected_content_type = deps.normalize_prompt_content_type(preferred_content_type)
    selected_learning_mode = deps.normalize_prompt_learning_mode(preferred_learning_mode)
    selected_difficulty = deps.normalize_prompt_difficulty(preferred_difficulty)
    has_explicit_filters = any(
        value != "all"
        for value in (
            selected_content_type,
            selected_learning_mode,
            selected_difficulty,
        )
    )
    prompt_files_for_mode = deps.prompt_file_map(
        cfg_now,
        mode_filter=selected_mode,
        content_type_filter=selected_content_type,
        learning_mode_filter=selected_learning_mode,
        difficulty_filter=selected_difficulty,
    )
    selected_key = pick_prompt_key(
        prompt_files_for_mode,
        lang=lang,
        preferred_key=prompt_key,
        preferred_mode=selected_mode,
        deps=deps,
    )
    prompt_text = ""
    load_msg = ""
    if selected_key:
        prompt_text, load_msg = deps.load_prompt_template(
            selected_key, prompt_files_now, lang=lang
        )
        file_mode = deps.load_prompt_mode(selected_key, prompt_files_now, lang=lang)
        if file_mode:
            selected_mode = file_mode
            prompt_files_for_mode = deps.prompt_file_map(
                cfg_now,
                mode_filter=selected_mode,
                content_type_filter=selected_content_type,
                learning_mode_filter=selected_learning_mode,
                difficulty_filter=selected_difficulty,
            )
        if not has_explicit_filters:
            (
                selected_content_type,
                selected_learning_mode,
                selected_difficulty,
            ) = deps.load_prompt_filter_metadata(selected_key, prompt_files_now, lang=lang)
            prompt_files_for_mode = deps.prompt_file_map(
                cfg_now,
                mode_filter=selected_mode,
                content_type_filter=selected_content_type,
                learning_mode_filter=selected_learning_mode,
                difficulty_filter=selected_difficulty,
            )
    if editor_override is not None:
        prompt_text = editor_override

    prompt_choices_now = deps.prompt_choices_from_map(prompt_files_for_mode, lang=lang)
    selected_key = normalize_dropdown_value(selected_key, prompt_choices_now)
    if not selected_key and editor_override is None:
        prompt_text = ""
    mode_choices_now = prompt_mode_choices_for_ui(lang, deps=deps)
    run_content_type_value = deps.normalize_prompt_content_type(run_content_type)
    run_learning_mode_value = deps.normalize_prompt_learning_mode(run_learning_mode)
    run_difficulty_value = deps.normalize_prompt_difficulty(run_difficulty)
    run_extract_choices_now = deps.prompt_path_choices(
        cfg_now,
        lang=lang,
        mode_filter="extraction",
        content_type_filter=run_content_type_value,
        learning_mode_filter=run_learning_mode_value,
        difficulty_filter=run_difficulty_value,
        include_auto=True,
    )
    run_explain_choices_now = deps.prompt_path_choices(
        cfg_now,
        lang=lang,
        mode_filter="explanation",
        content_type_filter=run_content_type_value,
        learning_mode_filter=run_learning_mode_value,
        difficulty_filter=run_difficulty_value,
        include_auto=True,
    )
    config_extract_choices_now = deps.prompt_path_choices(
        cfg_now,
        lang=lang,
        mode_filter="extraction",
        include_auto=True,
    )
    config_explain_choices_now = deps.prompt_path_choices(
        cfg_now,
        lang=lang,
        mode_filter="explanation",
        include_auto=True,
    )
    status_text = status or load_msg
    return (
        gr.update(choices=prompt_choices_now, value=selected_key),
        gr.update(choices=mode_choices_now, value=selected_mode),
        gr.update(
            choices=deps.prompt_content_type_options,
            value=selected_content_type,
        ),
        gr.update(
            choices=deps.prompt_learning_mode_options,
            value=selected_learning_mode,
        ),
        gr.update(
            choices=deps.prompt_difficulty_options,
            value=selected_difficulty,
        ),
        gr.update(value=prompt_text),
        gr.update(value=status_text),
        gr.update(
            choices=run_extract_choices_now,
            value=normalize_dropdown_value(run_extract_current, run_extract_choices_now),
        ),
        gr.update(
            choices=run_explain_choices_now,
            value=normalize_dropdown_value(run_explain_current, run_explain_choices_now),
        ),
        gr.update(
            choices=config_extract_choices_now,
            value=normalize_dropdown_value(
                config_extract_current, config_extract_choices_now
            ),
        ),
        gr.update(
            choices=config_explain_choices_now,
            value=normalize_dropdown_value(
                config_explain_current, config_explain_choices_now
            ),
        ),
    )


def append_prompt_aux_updates(
    updates: tuple[Any, ...],
    *,
    new_name_value: str = "",
    rename_name_value: str = "",
) -> tuple[Any, ...]:
    return (
        *updates,
        gr.update(value=new_name_value),
        gr.update(value=rename_name_value),
        gr.update(value=False),
        gr.update(value=False),
    )


def on_prompt_file_change(
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
    *,
    deps: PromptDeps,
) -> tuple[Any, ...]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    return refresh_prompt_controls(
        lang=lang,
        prompt_key=prompt_key,
        preferred_mode=prompt_mode,
        preferred_content_type=prompt_content_type,
        preferred_learning_mode=prompt_learning_mode,
        preferred_difficulty=prompt_difficulty,
        status="",
        editor_override=None,
        run_content_type=run_content_type,
        run_learning_mode=run_learning_mode,
        run_difficulty=run_difficulty,
        run_extract_current=run_extract_val,
        run_explain_current=run_explain_val,
        config_extract_current=config_extract_val,
        config_explain_current=config_explain_val,
        deps=deps,
    )


def on_prompt_mode_change(
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
    *,
    deps: PromptDeps,
) -> tuple[Any, ...]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    return refresh_prompt_controls(
        lang=lang,
        prompt_key=prompt_key,
        preferred_mode=prompt_mode,
        preferred_content_type=prompt_content_type,
        preferred_learning_mode=prompt_learning_mode,
        preferred_difficulty=prompt_difficulty,
        status="",
        editor_override=None,
        run_content_type=run_content_type,
        run_learning_mode=run_learning_mode,
        run_difficulty=run_difficulty,
        run_extract_current=run_extract_val,
        run_explain_current=run_explain_val,
        config_extract_current=config_extract_val,
        config_explain_current=config_explain_val,
        deps=deps,
    )


def on_prompt_filter_change(
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
    *,
    deps: PromptDeps,
) -> tuple[Any, ...]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    return refresh_prompt_controls(
        lang=lang,
        prompt_key=prompt_key,
        preferred_mode=prompt_mode,
        preferred_content_type=prompt_content_type,
        preferred_learning_mode=prompt_learning_mode,
        preferred_difficulty=prompt_difficulty,
        status="",
        editor_override=None,
        run_content_type=run_content_type,
        run_learning_mode=run_learning_mode,
        run_difficulty=run_difficulty,
        run_extract_current=run_extract_val,
        run_explain_current=run_explain_val,
        config_extract_current=config_extract_val,
        config_explain_current=config_explain_val,
        deps=deps,
    )


def on_prompt_new(
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
    *,
    deps: PromptDeps,
) -> tuple[Any, ...]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    mode = deps.normalize_prompt_mode(prompt_mode) or "extraction"
    cfg_now = deps.load_app_config()
    ok, new_key, status = deps.prompt_io.create_prompt_file(
        cfg_now,
        raw_name=new_name,
        mode=mode,
        lang=lang,
        tr=deps.tr,
    )
    new_name_value = ""
    selected_key = prompt_key
    if ok:
        selected_key = new_key
    else:
        new_name_value = deps.sanitize_prompt_filename(new_name) or (new_name or "")

    updates = refresh_prompt_controls(
        lang=lang,
        prompt_key=selected_key,
        preferred_mode=mode,
        preferred_content_type=prompt_content_type,
        preferred_learning_mode=prompt_learning_mode,
        preferred_difficulty=prompt_difficulty,
        status=status,
        editor_override=None,
        run_content_type=run_content_type,
        run_learning_mode=run_learning_mode,
        run_difficulty=run_difficulty,
        run_extract_current=run_extract_val,
        run_explain_current=run_explain_val,
        config_extract_current=config_extract_val,
        config_explain_current=config_explain_val,
        deps=deps,
    )
    return append_prompt_aux_updates(updates, new_name_value=new_name_value)


def on_prompt_save(
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
    *,
    deps: PromptDeps,
) -> tuple[Any, ...]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    mode = deps.normalize_prompt_mode(prompt_mode) or "extraction"
    template = (prompt_template or "").rstrip()
    if not save_confirmed:
        updates = refresh_prompt_controls(
            lang=lang,
            prompt_key=prompt_key,
            preferred_mode=mode,
            preferred_content_type=prompt_content_type,
            preferred_learning_mode=prompt_learning_mode,
            preferred_difficulty=prompt_difficulty,
            status="",
            editor_override=template,
            run_content_type=run_content_type,
            run_learning_mode=run_learning_mode,
            run_difficulty=run_difficulty,
            run_extract_current=run_extract_val,
            run_explain_current=run_explain_val,
            config_extract_current=config_extract_val,
            config_explain_current=config_explain_val,
            deps=deps,
        )
        return append_prompt_aux_updates(updates)
    _ok, status = deps.prompt_io.save_prompt_file(
        deps.load_app_config(),
        prompt_key=prompt_key,
        mode=mode,
        template=template,
        lang=lang,
        tr=deps.tr,
    )
    updates = refresh_prompt_controls(
        lang=lang,
        prompt_key=prompt_key,
        preferred_mode=mode,
        preferred_content_type=prompt_content_type,
        preferred_learning_mode=prompt_learning_mode,
        preferred_difficulty=prompt_difficulty,
        status=status,
        editor_override=template,
        run_content_type=run_content_type,
        run_learning_mode=run_learning_mode,
        run_difficulty=run_difficulty,
        run_extract_current=run_extract_val,
        run_explain_current=run_explain_val,
        config_extract_current=config_extract_val,
        config_explain_current=config_explain_val,
        deps=deps,
    )
    return append_prompt_aux_updates(updates)


def on_prompt_rename(
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
    *,
    deps: PromptDeps,
) -> tuple[Any, ...]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    ok, target_name, status = deps.prompt_io.rename_prompt_file(
        deps.load_app_config(),
        prompt_key=prompt_key,
        target_raw_name=rename_name,
        lang=lang,
        tr=deps.tr,
    )
    selected_key = target_name if ok else prompt_key
    updates = refresh_prompt_controls(
        lang=lang,
        prompt_key=selected_key,
        preferred_mode=deps.normalize_prompt_mode(prompt_mode),
        preferred_content_type=prompt_content_type,
        preferred_learning_mode=prompt_learning_mode,
        preferred_difficulty=prompt_difficulty,
        status=status,
        editor_override=None,
        run_content_type=run_content_type,
        run_learning_mode=run_learning_mode,
        run_difficulty=run_difficulty,
        run_extract_current=run_extract_val,
        run_explain_current=run_explain_val,
        config_extract_current=config_extract_val,
        config_explain_current=config_explain_val,
        deps=deps,
    )
    return append_prompt_aux_updates(
        updates,
        rename_name_value="" if ok else rename_name,
    )


def on_prompt_delete(
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
    *,
    deps: PromptDeps,
) -> tuple[Any, ...]:
    lang = deps.normalize_ui_lang(ui_lang_val)
    if not delete_confirmed:
        updates = refresh_prompt_controls(
            lang=lang,
            prompt_key=prompt_key,
            preferred_mode="",
            preferred_content_type=prompt_content_type,
            preferred_learning_mode=prompt_learning_mode,
            preferred_difficulty=prompt_difficulty,
            status="",
            editor_override=None,
            run_content_type=run_content_type,
            run_learning_mode=run_learning_mode,
            run_difficulty=run_difficulty,
            run_extract_current=run_extract_val,
            run_explain_current=run_explain_val,
            config_extract_current=config_extract_val,
            config_explain_current=config_explain_val,
            deps=deps,
        )
        return append_prompt_aux_updates(updates)

    ok, mode, status = deps.prompt_io.delete_prompt_file(
        deps.load_app_config(),
        prompt_key=prompt_key,
        lang=lang,
        tr=deps.tr,
    )
    updates = refresh_prompt_controls(
        lang=lang,
        prompt_key="" if ok else prompt_key,
        preferred_mode=mode or deps.normalize_prompt_mode(prompt_mode),
        preferred_content_type=prompt_content_type,
        preferred_learning_mode=prompt_learning_mode,
        preferred_difficulty=prompt_difficulty,
        status=status,
        editor_override=None,
        run_content_type=run_content_type,
        run_learning_mode=run_learning_mode,
        run_difficulty=run_difficulty,
        run_extract_current=run_extract_val,
        run_explain_current=run_explain_val,
        config_extract_current=config_extract_val,
        config_explain_current=config_explain_val,
        deps=deps,
    )
    return append_prompt_aux_updates(updates)


def on_run_prompt_filters_change(
    run_content_type: str,
    run_learning_mode: str,
    run_difficulty: str,
    run_extract_val: str,
    run_explain_val: str,
    ui_lang_val: str,
    *,
    deps: PromptDeps,
) -> tuple[Any, Any]:
    cfg_now = deps.load_app_config()
    lang = deps.normalize_ui_lang(ui_lang_val)
    run_extract_choices_now = deps.prompt_path_choices(
        cfg_now,
        lang=lang,
        mode_filter="extraction",
        content_type_filter=run_content_type,
        learning_mode_filter=run_learning_mode,
        difficulty_filter=run_difficulty,
        include_auto=True,
    )
    run_explain_choices_now = deps.prompt_path_choices(
        cfg_now,
        lang=lang,
        mode_filter="explanation",
        content_type_filter=run_content_type,
        learning_mode_filter=run_learning_mode,
        difficulty_filter=run_difficulty,
        include_auto=True,
    )
    return (
        gr.update(
            choices=run_extract_choices_now,
            value=normalize_dropdown_value(run_extract_val, run_extract_choices_now),
        ),
        gr.update(
            choices=run_explain_choices_now,
            value=normalize_dropdown_value(run_explain_val, run_explain_choices_now),
        ),
    )


def as_str(value: Any, *, default: str = "") -> str:
    if isinstance(value, str):
        text = value.strip()
        return text if text else default
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default
