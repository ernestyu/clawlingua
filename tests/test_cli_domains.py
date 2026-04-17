from __future__ import annotations

from typer.testing import CliRunner

from clawlearn.cli import app


runner = CliRunner()


def test_root_help_lists_domain_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "lingua" in result.stdout
    assert "textbook" in result.stdout


def test_lingua_build_deck_help_works() -> None:
    result = runner.invoke(app, ["lingua", "build", "deck", "--help"], env={"COLUMNS": "220"})
    assert result.exit_code == 0
    assert "--difficulty" in result.stdout
    assert "--extract-prompt" in result.stdout
    assert "lingua_expression" in result.stdout
    assert "lingua_reading" in result.stdout
    assert "expression_mining" not in result.stdout


def test_textbook_build_deck_help_works() -> None:
    result = runner.invoke(app, ["textbook", "build", "deck", "--help"], env={"COLUMNS": "220"})
    assert result.exit_code == 0
    assert "--max-notes" in result.stdout
    assert "--save-intermediate" in result.stdout
    assert "--learning-mode" in result.stdout
    assert "textbook_focus" in result.stdout
    assert "textbook_review" in result.stdout


def test_lingua_build_deck_rejects_legacy_mode() -> None:
    result = runner.invoke(
        app,
        [
            "lingua",
            "build",
            "deck",
            "input.txt",
            "--learning-mode",
            "expression_mining",
        ],
    )
    assert result.exit_code != 0
    assert "ARG_LEARNING_MODE_INVALID" in result.output


def test_textbook_build_deck_rejects_legacy_mode() -> None:
    result = runner.invoke(
        app,
        [
            "textbook",
            "build",
            "deck",
            "input.txt",
            "--learning-mode",
            "reading_support",
        ],
    )
    assert result.exit_code != 0
    assert "ARG_LEARNING_MODE_INVALID" in result.output
