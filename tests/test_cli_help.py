from typer.testing import CliRunner

from clawlingua.cli import app

runner = CliRunner()


def test_cli_help_available() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "build" in result.stdout


def test_build_deck_help_available() -> None:
    result = runner.invoke(app, ["build", "deck", "--help"])
    assert result.exit_code == 0
    assert ".apkg" in result.stdout

