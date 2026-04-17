from __future__ import annotations

from clawlearn.config import AppConfig, load_config


def test_taxonomy_repair_default_disabled() -> None:
    cfg = AppConfig()
    assert cfg.taxonomy_repair_enable is False


def test_load_config_reads_taxonomy_repair_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CLAWLEARN_TAXONOMY_REPAIR_ENABLE", "true")
    cfg = load_config(workspace_root=tmp_path)
    assert cfg.taxonomy_repair_enable is True


def test_load_config_override_beats_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CLAWLEARN_TAXONOMY_REPAIR_ENABLE", "false")
    cfg = load_config(
        workspace_root=tmp_path,
        overrides={"taxonomy_repair_enable": True},
    )
    assert cfg.taxonomy_repair_enable is True
