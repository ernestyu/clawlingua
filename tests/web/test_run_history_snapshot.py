from __future__ import annotations

import json
from pathlib import Path

from clawlearn.config import AppConfig
from clawlearn_web.run_history import build_env_snapshot, record_run_start


def test_build_env_snapshot_includes_lingua_two_phase_fields() -> None:
    cfg = AppConfig(
        llm_model="m1",
        secondary_extract_enable=True,
        secondary_extract_parallel=True,
        secondary_extract_llm_base_url="http://secondary.local/v1",
        secondary_extract_llm_model="m1-secondary",
        secondary_extract_llm_timeout_seconds=90,
        secondary_extract_llm_temperature=0.1,
        secondary_extract_llm_max_retries=2,
        secondary_extract_llm_retry_backoff_seconds=1.5,
        secondary_extract_llm_chunk_batch_size=3,
        translate_llm_model="m2",
        prompt_lang="zh",
        extract_prompt="p1.json",
        explain_prompt="p2.json",
        material_profile="transcript_dialogue",
        learning_mode="lingua_expression",
        lingua_transcript_min_context_sentences=2,
        lingua_annotate_enable=True,
        lingua_annotate_batch_size=16,
        lingua_annotate_max_items=40,
    )
    snapshot = build_env_snapshot(cfg)
    assert snapshot["CLAWLEARN_LINGUA_TRANSCRIPT_MIN_CONTEXT_SENTENCES"] == "2"
    assert snapshot["CLAWLEARN_LINGUA_ANNOTATE_ENABLE"] == "True"
    assert snapshot["CLAWLEARN_LINGUA_ANNOTATE_BATCH_SIZE"] == "16"
    assert snapshot["CLAWLEARN_LINGUA_ANNOTATE_MAX_ITEMS"] == "40"
    assert snapshot["CLAWLEARN_SECONDARY_EXTRACT_ENABLE"] == "True"
    assert snapshot["CLAWLEARN_SECONDARY_EXTRACT_PARALLEL"] == "True"
    assert snapshot["CLAWLEARN_SECONDARY_EXTRACT_LLM_BASE_URL"] == "http://secondary.local/v1"
    assert snapshot["CLAWLEARN_SECONDARY_EXTRACT_LLM_MODEL"] == "m1-secondary"
    assert snapshot["CLAWLEARN_SECONDARY_EXTRACT_LLM_TIMEOUT_SECONDS"] == "90"
    assert snapshot["CLAWLEARN_SECONDARY_EXTRACT_LLM_TEMPERATURE"] == "0.1"
    assert snapshot["CLAWLEARN_SECONDARY_EXTRACT_LLM_MAX_RETRIES"] == "2"
    assert snapshot["CLAWLEARN_SECONDARY_EXTRACT_LLM_RETRY_BACKOFF_SECONDS"] == "1.5"
    assert snapshot["CLAWLEARN_SECONDARY_EXTRACT_LLM_CHUNK_BATCH_SIZE"] == "3"


def test_build_env_snapshot_allows_runtime_override_values() -> None:
    cfg = AppConfig(
        material_profile="prose_article",
        learning_mode="lingua_expression",
        extract_prompt=None,
        explain_prompt=None,
    )
    snapshot = build_env_snapshot(
        cfg,
        overrides={
            "CLAWLEARN_MATERIAL_PROFILE": "transcript_dialogue",
            "CLAWLEARN_LEARNING_MODE": "lingua_reading",
            "CLAWLEARN_EXTRACT_PROMPT": "./prompts/custom_extract.json",
        },
    )
    assert snapshot["CLAWLEARN_MATERIAL_PROFILE"] == "transcript_dialogue"
    assert snapshot["CLAWLEARN_LEARNING_MODE"] == "lingua_reading"
    assert snapshot["CLAWLEARN_EXTRACT_PROMPT"] == "./prompts/custom_extract.json"


def test_record_run_start_persists_env_snapshot_overrides(tmp_path: Path) -> None:
    summary_path = tmp_path / "run_summary.json"
    cfg = AppConfig(material_profile="prose_article", learning_mode="lingua_expression")
    record_run_start(
        summary_path,
        run_id="run_x",
        started_at="2026-04-16T00:00:00+00:00",
        title="t",
        source_lang="en",
        target_lang="zh",
        domain="lingua",
        content_profile="transcript_dialogue",
        learning_mode="lingua_expression",
        difficulty="advanced",
        extract_prompt_override="./prompts/a.json",
        explain_prompt_override="./prompts/b.json",
        output_path="/tmp/out.apkg",
        cfg=cfg,
        env_snapshot_overrides={"CLAWLEARN_MATERIAL_PROFILE": "transcript_dialogue"},
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["env_snapshot"]["CLAWLEARN_MATERIAL_PROFILE"] == "transcript_dialogue"
