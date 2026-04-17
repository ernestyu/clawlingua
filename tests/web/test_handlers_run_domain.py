from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clawlearn.config import AppConfig
from clawlearn_web import handlers_run


@dataclass
class _FakeResult:
    run_id: str
    run_dir: Path
    output_path: Path
    cards_count: int
    errors_count: int


def _make_cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        workspace_root=tmp_path,
        output_dir=Path("runs"),
        export_dir=Path("outputs"),
    )


def test_run_single_build_routes_lingua(monkeypatch, tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    input_path = tmp_path / "input.txt"
    input_path.write_text("sample input", encoding="utf-8")

    called = {"lingua": 0, "textbook": 0}

    def _fake_lingua(_cfg, _opts):  # noqa: ANN001, ANN202
        called["lingua"] += 1
        return _FakeResult("run_lingua", tmp_path / "runs/run_lingua", tmp_path / "outputs/deck.apkg", 3, 0)

    def _fake_textbook(_cfg, _opts):  # noqa: ANN001, ANN202
        called["textbook"] += 1
        return _FakeResult("run_textbook", tmp_path / "runs/run_textbook", tmp_path / "outputs/deck.apkg", 2, 0)

    monkeypatch.setattr(handlers_run.upload_io, "materialize_uploaded_file", lambda *_a, **_k: input_path)
    monkeypatch.setattr(handlers_run.run_history, "record_run_start", lambda *_a, **_k: None)
    monkeypatch.setattr(handlers_run.run_history, "record_run_completed", lambda *_a, **_k: None)
    monkeypatch.setattr(handlers_run.run_history, "record_run_failed", lambda *_a, **_k: None)
    monkeypatch.setattr(handlers_run, "run_build_lingua_deck", _fake_lingua)
    monkeypatch.setattr(handlers_run, "run_build_textbook_deck", _fake_textbook)

    result = handlers_run.run_single_build(
        uploaded_file=object(),
        deck_title="deck",
        source_lang="en",
        target_lang="zh",
        content_profile="prose_article",
        learning_mode="lingua_expression",
        difficulty="intermediate",
        max_notes=5,
        input_char_limit=None,
        cloze_min_chars=None,
        textbook_max_concepts_per_chunk=None,
        textbook_keep_source_excerpt=True,
        chunk_max_chars=None,
        temperature=None,
        save_intermediate=True,
        continue_on_error=False,
        deps=handlers_run.RunServiceDeps(
            load_app_config=lambda: cfg,
            normalize_ui_lang=lambda value: value or "en",
            as_str=lambda value, default="": (str(value).strip() if value is not None else default),
            logger=None,
        ),
    )

    assert result["status"] == "ok"
    assert called["lingua"] == 1
    assert called["textbook"] == 0


def test_run_single_build_routes_textbook(monkeypatch, tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    input_path = tmp_path / "input.txt"
    input_path.write_text("sample input", encoding="utf-8")

    called = {"lingua": 0, "textbook": 0}

    def _fake_lingua(_cfg, _opts):  # noqa: ANN001, ANN202
        called["lingua"] += 1
        return _FakeResult("run_lingua", tmp_path / "runs/run_lingua", tmp_path / "outputs/deck.apkg", 3, 0)

    captured: dict[str, object] = {}

    def _fake_textbook(_cfg, _opts):  # noqa: ANN001, ANN202
        called["textbook"] += 1
        captured["max_concepts_per_chunk"] = _opts.max_concepts_per_chunk
        captured["keep_source_excerpt"] = _opts.keep_source_excerpt
        return _FakeResult("run_textbook", tmp_path / "runs/run_textbook", tmp_path / "outputs/deck.apkg", 2, 0)

    monkeypatch.setattr(handlers_run.upload_io, "materialize_uploaded_file", lambda *_a, **_k: input_path)
    monkeypatch.setattr(handlers_run.run_history, "record_run_start", lambda *_a, **_k: None)
    monkeypatch.setattr(handlers_run.run_history, "record_run_completed", lambda *_a, **_k: None)
    monkeypatch.setattr(handlers_run.run_history, "record_run_failed", lambda *_a, **_k: None)
    monkeypatch.setattr(handlers_run, "run_build_lingua_deck", _fake_lingua)
    monkeypatch.setattr(handlers_run, "run_build_textbook_deck", _fake_textbook)

    result = handlers_run.run_single_build(
        uploaded_file=object(),
        deck_title="deck",
        source_lang="en",
        target_lang="zh",
        content_profile="textbook_examples",
        learning_mode="textbook_review",
        difficulty="intermediate",
        max_notes=5,
        input_char_limit=None,
        cloze_min_chars=None,
        textbook_max_concepts_per_chunk=3,
        textbook_keep_source_excerpt=False,
        chunk_max_chars=None,
        temperature=None,
        save_intermediate=True,
        continue_on_error=False,
        deps=handlers_run.RunServiceDeps(
            load_app_config=lambda: cfg,
            normalize_ui_lang=lambda value: value or "en",
            as_str=lambda value, default="": (str(value).strip() if value is not None else default),
            logger=None,
        ),
    )

    assert result["status"] == "ok"
    assert called["lingua"] == 0
    assert called["textbook"] == 1
    assert captured["max_concepts_per_chunk"] == 3
    assert captured["keep_source_excerpt"] is False


def test_run_single_build_invalid_learning_mode_falls_back_to_lingua(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = _make_cfg(tmp_path)
    input_path = tmp_path / "input.txt"
    input_path.write_text("sample input", encoding="utf-8")

    called = {"lingua": 0}

    def _fake_lingua(_cfg, _opts):  # noqa: ANN001, ANN202
        called["lingua"] += 1
        return _FakeResult("run_lingua", tmp_path / "runs/run_lingua", tmp_path / "outputs/deck.apkg", 1, 0)

    monkeypatch.setattr(handlers_run.upload_io, "materialize_uploaded_file", lambda *_a, **_k: input_path)
    monkeypatch.setattr(handlers_run.run_history, "record_run_start", lambda *_a, **_k: None)
    monkeypatch.setattr(handlers_run.run_history, "record_run_completed", lambda *_a, **_k: None)
    monkeypatch.setattr(handlers_run.run_history, "record_run_failed", lambda *_a, **_k: None)
    monkeypatch.setattr(handlers_run, "run_build_lingua_deck", _fake_lingua)
    monkeypatch.setattr(handlers_run, "run_build_textbook_deck", lambda *_a, **_k: None)

    result = handlers_run.run_single_build(
        uploaded_file=object(),
        deck_title="deck",
        source_lang="en",
        target_lang="zh",
        content_profile="prose_article",
        learning_mode="not-a-supported-mode",
        difficulty="intermediate",
        max_notes=5,
        input_char_limit=None,
        cloze_min_chars=None,
        textbook_max_concepts_per_chunk=None,
        textbook_keep_source_excerpt=True,
        chunk_max_chars=None,
        temperature=None,
        save_intermediate=True,
        continue_on_error=False,
        deps=handlers_run.RunServiceDeps(
            load_app_config=lambda: cfg,
            normalize_ui_lang=lambda value: value or "en",
            as_str=lambda value, default="": (str(value).strip() if value is not None else default),
            logger=None,
        ),
    )

    assert result["status"] == "ok"
    assert called["lingua"] == 1


def test_run_single_build_records_effective_snapshot_overrides(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = _make_cfg(tmp_path)
    input_path = tmp_path / "input.txt"
    input_path.write_text("sample input", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_record_start(*_a, **kwargs):  # noqa: ANN001, ANN202
        captured.update(kwargs)

    def _fake_lingua(_cfg, _opts):  # noqa: ANN001, ANN202
        return _FakeResult("run_lingua", tmp_path / "runs/run_lingua", tmp_path / "outputs/deck.apkg", 1, 0)

    monkeypatch.setattr(handlers_run.upload_io, "materialize_uploaded_file", lambda *_a, **_k: input_path)
    monkeypatch.setattr(handlers_run.run_history, "record_run_start", _fake_record_start)
    monkeypatch.setattr(handlers_run.run_history, "record_run_completed", lambda *_a, **_k: None)
    monkeypatch.setattr(handlers_run.run_history, "record_run_failed", lambda *_a, **_k: None)
    monkeypatch.setattr(handlers_run, "run_build_lingua_deck", _fake_lingua)
    monkeypatch.setattr(handlers_run, "run_build_textbook_deck", lambda *_a, **_k: None)

    result = handlers_run.run_single_build(
        uploaded_file=object(),
        deck_title="deck",
        source_lang="en",
        target_lang="zh",
        content_profile="transcript_dialogue",
        learning_mode="lingua_expression",
        difficulty="advanced",
        max_notes=5,
        input_char_limit=None,
        cloze_min_chars=None,
        textbook_max_concepts_per_chunk=None,
        textbook_keep_source_excerpt=True,
        chunk_max_chars=None,
        temperature=None,
        save_intermediate=True,
        continue_on_error=False,
        prompt_lang="zh",
        extract_prompt="./prompts/cloze_transcript_advanced.json",
        explain_prompt="./prompts/translate_rewrite.json",
        deps=handlers_run.RunServiceDeps(
            load_app_config=lambda: cfg,
            normalize_ui_lang=lambda value: value or "en",
            as_str=lambda value, default="": (str(value).strip() if value is not None else default),
            logger=None,
        ),
    )

    assert result["status"] == "ok"
    overrides = captured["env_snapshot_overrides"]
    assert isinstance(overrides, dict)
    assert overrides["CLAWLEARN_MATERIAL_PROFILE"] == "transcript_dialogue"
    assert overrides["CLAWLEARN_LEARNING_MODE"] == "lingua_expression"
    assert overrides["CLAWLEARN_EXTRACT_PROMPT"] == "./prompts/cloze_transcript_advanced.json"
    assert overrides["CLAWLEARN_EXPLAIN_PROMPT"] == "./prompts/translate_rewrite.json"
