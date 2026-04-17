from __future__ import annotations

import pytest
from pydantic import ValidationError

from clawlearn.config import AppConfig
from clawlearn.errors import ClawLearnError
from clawlearn.models.prompt_schema import PromptSpec
from clawlearn.pipeline.build_lingua_deck import (
    BuildDeckOptions,
    _resolve_learning_mode,
    _use_phrase_extraction_pipeline,
)
from clawlearn_web import prompt_io


def test_app_config_default_learning_mode_is_lingua_expression() -> None:
    cfg = AppConfig()
    assert cfg.learning_mode == "lingua_expression"


def test_app_config_rejects_legacy_learning_mode() -> None:
    with pytest.raises(ValidationError):
        AppConfig(learning_mode="expression_mining")


def test_prompt_spec_rejects_legacy_learning_mode_literal() -> None:
    payload = {
        "name": "sample",
        "version": "1",
        "description": "sample",
        "mode": "extraction",
        "learning_mode": "expression_mining_v2",
        "system_prompt": "s",
        "user_prompt_template": "u",
        "placeholders": [],
        "output_format": {"type": "json", "schema_name": "x"},
    }
    with pytest.raises(ValidationError):
        PromptSpec.model_validate(payload)


def test_prompt_learning_mode_options_expose_only_new_values() -> None:
    assert prompt_io.PROMPT_LEARNING_MODE_OPTIONS == [
        "all",
        "lingua_expression",
        "lingua_reading",
        "textbook_focus",
        "textbook_review",
    ]


def test_lingua_pipeline_mode_resolver_accepts_only_lingua_modes() -> None:
    cfg = AppConfig(learning_mode="lingua_expression")
    assert _resolve_learning_mode(cfg, BuildDeckOptions(input_value="x")) == "lingua_expression"
    assert (
        _resolve_learning_mode(
            cfg,
            BuildDeckOptions(input_value="x", learning_mode="lingua_reading"),
        )
        == "lingua_reading"
    )
    with pytest.raises(ClawLearnError) as exc_info:
        _resolve_learning_mode(
            cfg,
            BuildDeckOptions(input_value="x", learning_mode="textbook_focus"),
        )
    assert exc_info.value.error_code == "ARG_LEARNING_MODE_INVALID"


def test_phrase_extraction_toggle_no_longer_depends_on_v2_mode() -> None:
    assert _use_phrase_extraction_pipeline(learning_mode="lingua_expression", schema_name=None) is False
    assert (
        _use_phrase_extraction_pipeline(
            learning_mode="lingua_expression",
            schema_name="phrase_candidates_v1",
        )
        is True
    )
    assert (
        _use_phrase_extraction_pipeline(
            learning_mode="lingua_expression",
            schema_name="phrase_candidates_v2",
        )
        is True
    )
    assert (
        _use_phrase_extraction_pipeline(
            learning_mode="lingua_expression",
            schema_name="cloze_cards_v1",
        )
        is False
    )
