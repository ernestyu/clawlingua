from __future__ import annotations

from clawlearn.config import AppConfig, load_config


def test_lingua_2phase_defaults() -> None:
    cfg = AppConfig()
    assert cfg.lingua_transcript_min_context_sentences == 2
    assert cfg.lingua_annotate_enable is False
    assert cfg.lingua_annotate_batch_size == 50
    assert cfg.lingua_annotate_max_items is None
    assert cfg.secondary_extract_enable is False
    assert cfg.secondary_extract_parallel is False
    assert cfg.secondary_extract_llm_model is None
    assert cfg.secondary_extract_llm_chunk_batch_size is None


def test_load_config_reads_lingua_2phase_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CLAWLEARN_LINGUA_TRANSCRIPT_MIN_CONTEXT_SENTENCES", "2")
    monkeypatch.setenv("CLAWLEARN_LINGUA_ANNOTATE_ENABLE", "true")
    monkeypatch.setenv("CLAWLEARN_LINGUA_ANNOTATE_BATCH_SIZE", "12")
    monkeypatch.setenv("CLAWLEARN_LINGUA_ANNOTATE_MAX_ITEMS", "34")
    cfg = load_config(workspace_root=tmp_path)
    assert cfg.lingua_transcript_min_context_sentences == 2
    assert cfg.lingua_annotate_enable is True
    assert cfg.lingua_annotate_batch_size == 12
    assert cfg.lingua_annotate_max_items == 34


def test_load_config_override_beats_lingua_2phase_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CLAWLEARN_LINGUA_ANNOTATE_ENABLE", "false")
    cfg = load_config(
        workspace_root=tmp_path,
        overrides={
            "lingua_annotate_enable": True,
            "lingua_annotate_batch_size": 7,
        },
    )
    assert cfg.lingua_annotate_enable is True
    assert cfg.lingua_annotate_batch_size == 7


def test_load_config_reads_secondary_extract_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CLAWLEARN_SECONDARY_EXTRACT_ENABLE", "true")
    monkeypatch.setenv("CLAWLEARN_SECONDARY_EXTRACT_PARALLEL", "true")
    monkeypatch.setenv("CLAWLEARN_SECONDARY_EXTRACT_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("CLAWLEARN_SECONDARY_EXTRACT_LLM_API_KEY", "none")
    monkeypatch.setenv("CLAWLEARN_SECONDARY_EXTRACT_LLM_MODEL", "mistral-small3.2:latest")
    monkeypatch.setenv("CLAWLEARN_SECONDARY_EXTRACT_LLM_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("CLAWLEARN_SECONDARY_EXTRACT_LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("CLAWLEARN_SECONDARY_EXTRACT_LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("CLAWLEARN_SECONDARY_EXTRACT_LLM_RETRY_BACKOFF_SECONDS", "1.5")
    monkeypatch.setenv("CLAWLEARN_SECONDARY_EXTRACT_LLM_CHUNK_BATCH_SIZE", "3")
    cfg = load_config(workspace_root=tmp_path)
    assert cfg.secondary_extract_enable is True
    assert cfg.secondary_extract_parallel is True
    assert cfg.secondary_extract_llm_base_url == "http://127.0.0.1:11434/v1"
    assert cfg.secondary_extract_llm_api_key == "none"
    assert cfg.secondary_extract_llm_model == "mistral-small3.2:latest"
    assert cfg.secondary_extract_llm_timeout_seconds == 90
    assert cfg.secondary_extract_llm_temperature == 0.1
    assert cfg.secondary_extract_llm_max_retries == 2
    assert cfg.secondary_extract_llm_retry_backoff_seconds == 1.5
    assert cfg.secondary_extract_llm_chunk_batch_size == 3


def test_load_config_override_beats_secondary_extract_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CLAWLEARN_SECONDARY_EXTRACT_ENABLE", "false")
    monkeypatch.setenv("CLAWLEARN_SECONDARY_EXTRACT_PARALLEL", "false")
    monkeypatch.setenv("CLAWLEARN_SECONDARY_EXTRACT_LLM_MODEL", "env-model")
    cfg = load_config(
        workspace_root=tmp_path,
        overrides={
            "secondary_extract_enable": True,
            "secondary_extract_parallel": True,
            "secondary_extract_llm_model": "override-model",
            "secondary_extract_llm_chunk_batch_size": 4,
        },
    )
    assert cfg.secondary_extract_enable is True
    assert cfg.secondary_extract_parallel is True
    assert cfg.secondary_extract_llm_model == "override-model"
    assert cfg.secondary_extract_llm_chunk_batch_size == 4
