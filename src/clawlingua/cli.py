"""Typer CLI entrypoint."""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import typer

from .config import load_config, validate_base_config, validate_runtime_config
from .errors import ClawLinguaError, build_error, format_error
from .exit_codes import ExitCode
from .helptext import (
    BUILD_DECK_HELP,
    CONFIG_SHOW_HELP,
    CONFIG_VALIDATE_HELP,
    DOCTOR_HELP,
    INIT_HELP,
    PROMPT_VALIDATE_HELP,
)
from .llm.prompt_loader import load_prompt
from .logger import setup_logging
from .pipeline.build_deck import BuildDeckOptions, run_build_deck
from .anki.template_loader import load_anki_template

app = typer.Typer(
    name="clawlingua",
    add_completion=False,
    no_args_is_help=True,
    help="Build Anki cloze decks from URL or text files.",
)
build_app = typer.Typer(no_args_is_help=True, help="Build commands.")
prompt_app = typer.Typer(no_args_is_help=True, help="Prompt file commands.")
config_app = typer.Typer(no_args_is_help=True, help="Config commands.")


def _echo_error(error: ClawLinguaError) -> None:
    typer.echo(format_error(error), err=True)


def _run_guard(func, *, debug: bool = False):  # type: ignore[no-untyped-def]
    try:
        return func()
    except ClawLinguaError as exc:
        _echo_error(exc)
        raise typer.Exit(code=exc.exit_code)
    except Exception as exc:
        if debug:
            raise
        err = build_error(
            error_code="INTERNAL_ERROR",
            cause="未预期内部错误。",
            detail=str(exc),
            next_steps=["使用 --debug 复现并查看 traceback"],
            exit_code=ExitCode.INTERNAL_ERROR,
        )
        _echo_error(err)
        raise typer.Exit(code=err.exit_code)


@app.command(help=INIT_HELP)
def init(
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Optional output directory to create."),
) -> None:
    def _impl() -> None:
        env_example = Path(".env.example")
        env_file = Path(".env")
        if env_example.exists() and not env_file.exists():
            env_file.write_text(env_example.read_text(encoding="utf-8"), encoding="utf-8")
            typer.echo("INFO | created .env from .env.example")

        required = [
            Path("./prompts/cloze_contextual.json"),
            Path("./prompts/translate_rewrite.json"),
            Path("./templates/anki_cloze_default.json"),
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise build_error(
                error_code="INIT_REQUIRED_FILE_MISSING",
                cause="初始化检查失败。",
                detail=f"missing={missing}",
                next_steps=["补齐 prompts/templates 文件后重试"],
                exit_code=ExitCode.SCHEMA_ERROR,
            )

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            typer.echo(f"INFO | output dir ready | path={output_dir}")
        typer.echo("INFO | init complete")

    _run_guard(_impl)


@app.command(help=DOCTOR_HELP)
def doctor(
    env_file: Path | None = typer.Option(None, "--env-file", help="Path to .env file."),
) -> None:
    def _impl() -> None:
        cfg = load_config(env_file=env_file)
        setup_logging(cfg.log_level)

        checks: list[tuple[str, bool, str]] = []

        for module_name in ["edge_tts", "genanki", "httpx", "typer"]:
            try:
                importlib.import_module(module_name)
                checks.append((f"dependency:{module_name}", True, "ok"))
            except Exception as exc:
                checks.append((f"dependency:{module_name}", False, str(exc)))

        try:
            validate_base_config(cfg)
            checks.append(("config:paths", True, "ok"))
        except ClawLinguaError as exc:
            checks.append(("config:paths", False, "; ".join(exc.to_lines())))

        try:
            validate_runtime_config(cfg)
            checks.append(("config:runtime", True, "ok"))
        except ClawLinguaError as exc:
            checks.append(("config:runtime", False, "; ".join(exc.to_lines())))

        try:
            load_prompt(cfg.resolve_path(cfg.prompt_cloze))
            load_prompt(cfg.resolve_path(cfg.prompt_translate))
            checks.append(("prompt:schema", True, "ok"))
        except ClawLinguaError as exc:
            checks.append(("prompt:schema", False, "; ".join(exc.to_lines())))

        try:
            load_anki_template(cfg.resolve_path(cfg.anki_template))
            checks.append(("template:schema", True, "ok"))
        except ClawLinguaError as exc:
            checks.append(("template:schema", False, "; ".join(exc.to_lines())))

        try:
            endpoint = cfg.llm_base_url.rstrip("/") + "/models"
            headers = {"Authorization": f"Bearer {cfg.llm_api_key}"} if cfg.llm_api_key else {}
            with httpx.Client(timeout=cfg.llm_timeout_seconds) as client:
                response = client.get(endpoint, headers=headers)
            ok = response.status_code < 500
            checks.append(("llm:connectivity", ok, f"status={response.status_code}"))
        except Exception as exc:
            checks.append(("llm:connectivity", False, str(exc)))

        out_dir = cfg.resolve_path(cfg.output_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            test_file = out_dir / ".doctor_write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            checks.append(("output:writable", True, str(out_dir)))
        except Exception as exc:
            checks.append(("output:writable", False, str(exc)))

        failed = False
        for name, ok, detail in checks:
            status = "OK" if ok else "FAIL"
            typer.echo(f"{status} | {name} | {detail}")
            if not ok:
                failed = True

        if failed:
            raise build_error(
                error_code="DOCTOR_CHECK_FAILED",
                cause="doctor 检查未通过。",
                detail="至少一个检查项失败。",
                next_steps=["根据 FAIL 项修复后重试"],
                exit_code=ExitCode.CONFIG_ERROR,
            )
        typer.echo("INFO | doctor complete")

    _run_guard(_impl)


@build_app.command("deck", help=BUILD_DECK_HELP)
def build_deck(
    input_value: str = typer.Argument(..., help="URL or path to .txt/.md input."),
    input_type: str = typer.Option("auto", "--input-type", help="auto|url|file"),
    source_lang: str | None = typer.Option(None, "--source-lang", help="Source language code."),
    target_lang: str | None = typer.Option(None, "--target-lang", help="Target language code."),
    env_file: Path | None = typer.Option(None, "--env-file", help="Path to .env file."),
    output: Path | None = typer.Option(None, "--output", help="Output .apkg path."),
    deck_name: str | None = typer.Option(None, "--deck-name", help="Deck name override."),
    max_chars: int | None = typer.Option(None, "--max-chars", help="Chunk max chars."),
    max_sentences: int | None = typer.Option(None, "--max-sentences", help="Chunk max sentences."),
    max_notes: int | None = typer.Option(None, "--max-notes", help="Max notes to export."),
    temperature: float | None = typer.Option(None, "--temperature", help="LLM temperature."),
    save_intermediate: bool | None = typer.Option(None, "--save-intermediate/--no-save-intermediate"),
    continue_on_error: bool = typer.Option(False, "--continue-on-error"),
    verbose: bool = typer.Option(False, "--verbose"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    def _impl() -> None:
        cfg = load_config(env_file=env_file)
        setup_logging(cfg.log_level if not verbose else "DEBUG")

        if input_type not in {"auto", "url", "file"}:
            raise build_error(
                error_code="ARG_INPUT_TYPE_INVALID",
                cause="input-type 参数非法。",
                detail=f"input_type={input_type}",
                next_steps=["使用 auto/url/file 之一"],
                exit_code=ExitCode.ARGUMENT_ERROR,
            )

        result = run_build_deck(
            cfg,
            BuildDeckOptions(
                input_value=input_value,
                input_type=input_type,
                source_lang=source_lang,
                target_lang=target_lang,
                output=output,
                deck_name=deck_name,
                max_chars=max_chars,
                max_sentences=max_sentences,
                max_notes=max_notes,
                temperature=temperature,
                save_intermediate=save_intermediate,
                continue_on_error=continue_on_error,
            ),
        )
        typer.echo(f"INFO | build complete | cards={result.cards_count} errors={result.errors_count}")
        typer.echo(f"INFO | output={result.output_path}")

    _run_guard(_impl, debug=debug)


@prompt_app.command("validate", help=PROMPT_VALIDATE_HELP)
def prompt_validate(path: Path = typer.Argument(..., help="Prompt JSON path.")) -> None:
    def _impl() -> None:
        load_prompt(path)
        typer.echo(f"INFO | prompt valid | path={path}")

    _run_guard(_impl)


@config_app.command("show", help=CONFIG_SHOW_HELP)
def config_show(
    env_file: Path | None = typer.Option(None, "--env-file", help="Path to .env file."),
) -> None:
    def _impl() -> None:
        cfg = load_config(env_file=env_file)
        typer.echo(json.dumps(cfg.masked_dump(), ensure_ascii=False, indent=2, default=str))

    _run_guard(_impl)


@config_app.command("validate", help=CONFIG_VALIDATE_HELP)
def config_validate(
    env_file: Path | None = typer.Option(None, "--env-file", help="Path to .env file."),
) -> None:
    def _impl() -> None:
        cfg = load_config(env_file=env_file)
        validate_base_config(cfg)
        validate_runtime_config(cfg)
        typer.echo("INFO | config validate complete")

    _run_guard(_impl)


app.add_typer(build_app, name="build")
app.add_typer(prompt_app, name="prompt")
app.add_typer(config_app, name="config")


def main() -> Any:
    return app()


if __name__ == "__main__":
    main()

