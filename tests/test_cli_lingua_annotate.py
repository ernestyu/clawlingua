from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import clawlearn.cli as cli_module
from clawlearn.config import AppConfig
from clawlearn.cli import app


runner = CliRunner()


def test_lingua_build_help_includes_annotate_options() -> None:
    result = runner.invoke(app, ["lingua", "build", "deck", "--help"], env={"COLUMNS": "220"})
    assert result.exit_code == 0
    assert "--lingua-annotate" in result.stdout
    assert "--lingua-annotate-batch-size" in result.stdout
    assert "--lingua-annotate-max-items" in result.stdout


def test_lingua_build_cli_overrides_annotate_env(monkeypatch) -> None:
    captured: dict[str, AppConfig] = {}

    def _fake_load_config(*, env_file=None):  # noqa: ANN001, ANN202
        return AppConfig(
            workspace_root=Path.cwd(),
            lingua_annotate_enable=False,
            lingua_annotate_batch_size=50,
            lingua_annotate_max_items=None,
        )

    def _fake_run_build(cfg, options):  # noqa: ANN001, ANN202
        captured["cfg"] = cfg
        return SimpleNamespace(run_id="r", run_dir=Path("runs/r"), output_path=Path("out.apkg"), cards_count=0, errors_count=0)

    monkeypatch.setattr(cli_module, "load_config", _fake_load_config)
    monkeypatch.setattr(cli_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module, "run_build_lingua_deck", _fake_run_build)

    result = runner.invoke(
        app,
        [
            "lingua",
            "build",
            "deck",
            "input.txt",
            "--lingua-annotate",
            "--lingua-annotate-batch-size",
            "12",
            "--lingua-annotate-max-items",
            "34",
        ],
    )
    assert result.exit_code == 0
    assert captured["cfg"].lingua_annotate_enable is True
    assert captured["cfg"].lingua_annotate_batch_size == 12
    assert captured["cfg"].lingua_annotate_max_items == 34


def test_lingua_build_cli_keeps_env_when_no_annotate_override(monkeypatch) -> None:
    captured: dict[str, AppConfig] = {}

    def _fake_load_config(*, env_file=None):  # noqa: ANN001, ANN202
        return AppConfig(
            workspace_root=Path.cwd(),
            lingua_annotate_enable=True,
            lingua_annotate_batch_size=16,
            lingua_annotate_max_items=40,
        )

    def _fake_run_build(cfg, options):  # noqa: ANN001, ANN202
        captured["cfg"] = cfg
        return SimpleNamespace(run_id="r", run_dir=Path("runs/r"), output_path=Path("out.apkg"), cards_count=0, errors_count=0)

    monkeypatch.setattr(cli_module, "load_config", _fake_load_config)
    monkeypatch.setattr(cli_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module, "run_build_lingua_deck", _fake_run_build)

    result = runner.invoke(app, ["lingua", "build", "deck", "input.txt"])
    assert result.exit_code == 0
    assert captured["cfg"].lingua_annotate_enable is True
    assert captured["cfg"].lingua_annotate_batch_size == 16
    assert captured["cfg"].lingua_annotate_max_items == 40
