from __future__ import annotations

from pathlib import Path
from typing import Any

from clawlearn.config import load_config
from clawlearn_web import config_io, handlers_config


def _make_deps(
    *,
    resolve_env_file: Any,
    save_env_v2: Any,
) -> handlers_config.ConfigDeps:
    return handlers_config.ConfigDeps(
        to_timeout_seconds=lambda _value: 20.0,
        normalize_ui_lang=lambda value: str(value or "en"),
        read_env_example=lambda: {},
        resolve_env_file=resolve_env_file,
        load_config=load_config,
        load_env_view=config_io.load_env_view,
        tr=lambda _lang, en, _zh: en,
        save_env_v2=save_env_v2,
    )


def _save_config_kwargs(**overrides: str) -> dict[str, str]:
    payload = {
        "llm_base_url_val": "",
        "llm_api_key_val": "",
        "llm_model_val": "",
        "llm_timeout_val": "",
        "llm_temperature_val": "",
        "llm_chunk_batch_size_val": "",
        "translate_base_url_val": "",
        "translate_api_key_val": "",
        "translate_model_val": "",
        "translate_temperature_val": "",
        "chunk_max_chars_val": "",
        "chunk_min_chars_val": "",
        "cloze_min_chars_val": "",
        "cloze_max_per_chunk_val": "",
        "validate_retry_enable_val": "",
        "validate_retry_max_val": "",
        "validate_retry_llm_enable_val": "",
        "lingua_annotate_enable_val": "",
        "lingua_annotate_batch_size_val": "",
        "lingua_annotate_max_items_val": "",
        "content_profile_val": "",
        "cloze_difficulty_val": "",
        "prompt_lang_val": "",
        "extract_prompt_env_val": "",
        "explain_prompt_env_val": "",
        "output_dir_val": "",
        "export_dir_val": "",
        "log_dir_val": "",
        "default_deck_name_val": "",
        "tts_voice1_val": "",
        "tts_voice2_val": "",
        "tts_voice3_val": "",
        "tts_voice4_val": "",
        "secondary_extract_enable_val": "",
        "secondary_extract_parallel_val": "",
        "secondary_extract_base_url_val": "",
        "secondary_extract_api_key_val": "",
        "secondary_extract_model_val": "",
        "secondary_extract_timeout_val": "",
        "secondary_extract_temperature_val": "",
        "secondary_extract_max_retries_val": "",
        "secondary_extract_retry_backoff_val": "",
        "secondary_extract_chunk_batch_size_val": "",
        "ui_lang_val": "en",
    }
    payload.update(overrides)
    return payload


def test_on_reload_env_reads_existing_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CLAWLEARN_LLM_BASE_URL=http://env.local/v1\n"
        "CLAWLEARN_DEFAULT_DECK_NAME=Deck From Env\n",
        encoding="utf-8",
    )
    deps = _make_deps(
        resolve_env_file=lambda: env_file,
        save_env_v2=lambda _updated, *, lang: f"saved:{lang}",
    )

    values = handlers_config.on_reload_env("en", deps=deps)

    assert len(values) == 44
    assert values[0] == "http://env.local/v1"
    assert values[28] == "Deck From Env"
    assert values[33].lower() == "false"
    assert values[34].lower() == "false"
    assert values[37] == ""
    assert "Reloaded config from .env" in values[-1]
    assert str(env_file) in values[-1]


def test_on_reload_env_reports_missing_env_file(tmp_path: Path) -> None:
    missing_env_file = tmp_path / "missing.env"
    deps = _make_deps(
        resolve_env_file=lambda: missing_env_file,
        save_env_v2=lambda _updated, *, lang: f"saved:{lang}",
    )

    values = handlers_config.on_reload_env("en", deps=deps)

    assert len(values) == 44
    assert ".env not found" in values[-1]
    assert str(missing_env_file) in values[-1]


def test_on_save_config_returns_reloaded_values_and_combined_status(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"

    def _save_env(updated: dict[str, str], *, lang: str) -> str:
        lines = []
        for key in sorted(updated):
            value = str(updated[key]).strip()
            if value:
                lines.append(f"{key}={value}")
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"save-ok:{lang}"

    deps = _make_deps(
        resolve_env_file=lambda: env_file,
        save_env_v2=_save_env,
    )
    kwargs = _save_config_kwargs(
        llm_base_url_val="http://after-save.local/v1",
        default_deck_name_val="After Save Deck",
    )

    values = handlers_config.on_save_config(**kwargs, deps=deps)

    assert len(values) == 44
    assert values[0] == "http://after-save.local/v1"
    assert values[28] == "After Save Deck"
    assert "save-ok:en" in values[-1]
    assert "Reloaded config from .env" in values[-1]


def test_resolve_env_file_uses_override_env_var(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "custom.env"
    monkeypatch.setenv("CLAWLEARN_ENV_FILE", str(override))

    resolved = config_io.resolve_env_file()

    assert resolved == override.resolve()
