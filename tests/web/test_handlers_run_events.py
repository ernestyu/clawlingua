from __future__ import annotations

import time
from typing import Any

from clawlearn_web import handlers_run


def _deps(
    *,
    run_single_build: Any,
    build_run_analysis: Any,
) -> handlers_run.RunDeps:
    return handlers_run.RunDeps(
        normalize_ui_lang=lambda value: str(value or "en"),
        tr=lambda _lang, en, _zh: en,
        run_single_build=run_single_build,
        to_optional_int=lambda value, min_value=None: int(value) if value not in (None, "") else None,
        to_optional_float=lambda value: float(value) if value not in (None, "") else None,
        as_str=lambda value, default="": (str(value).strip() if value is not None else default),
        load_app_config=lambda: object(),
        refresh_recent_runs=lambda _cfg, *, lang, preferred_run_id=None: (
            {"value": preferred_run_id},
            f"detail:{lang}",
            None,
        ),
        load_run_detail=lambda _run_id, _cfg, *, lang: (f"detail:{lang}", None),
        build_run_analysis=build_run_analysis,
    )


def _run_args() -> tuple[Any, ...]:
    return (
        object(),  # file_obj
        "deck",  # deck_title
        "en",  # source
        "zh",  # target
        "transcript_dialogue",  # profile
        "lingua_expression",  # mode
        "advanced",  # difficulty
        "",  # extract prompt
        "",  # explain prompt
        "",  # max notes
        "",  # input limit
        "",  # cloze min
        "",  # textbook max
        True,  # textbook keep excerpt
        "",  # chunk max
        "",  # temperature
        False,  # secondary extract enable
        True,  # save_intermediate
        False,  # continue_on_error
        "en",  # ui lang
    )


def test_on_run_analysis_failure_is_degraded() -> None:
    def _run_single_build(**kwargs):  # noqa: ANN003, ANN202
        return {
            "status": "ok",
            "run_id": "run_001",
            "output_path": "/tmp/output.apkg",
            "cards_count": 1,
            "errors_count": 0,
        }

    def _raise_analysis(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("analysis boom")

    deps = _deps(
        run_single_build=_run_single_build,
        build_run_analysis=_raise_analysis,
    )
    events = list(handlers_run.on_run(*_run_args(), deps=deps))
    assert len(events) >= 1
    final = events[-1]
    assert "Completed" in final[0]
    assert "Run analytics unavailable" in final[5]


def test_on_run_streams_elapsed_status_while_running(monkeypatch) -> None:
    monkeypatch.setattr(handlers_run, "_RUN_PROGRESS_POLL_SECONDS", 0.01)

    def _run_single_build(**kwargs):  # noqa: ANN003, ANN202
        time.sleep(0.03)
        return {
            "status": "ok",
            "run_id": "run_002",
            "output_path": "/tmp/output.apkg",
            "cards_count": 2,
            "errors_count": 0,
        }

    def _analysis(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return ("analysis", [], [("all", "all")], [("all", "all")], [("all", "all")])

    deps = _deps(
        run_single_build=_run_single_build,
        build_run_analysis=_analysis,
    )
    events = list(handlers_run.on_run(*_run_args(), deps=deps))
    assert len(events) >= 2
    assert any("Running" in event[0] for event in events[:-1])
    assert "Completed" in events[-1][0]


def test_on_run_passes_secondary_extract_toggle_to_service() -> None:
    seen: dict[str, Any] = {}

    def _run_single_build(**kwargs):  # noqa: ANN003, ANN202
        seen["secondary_extract_enable"] = kwargs.get("secondary_extract_enable")
        return {
            "status": "ok",
            "run_id": "run_002b",
            "output_path": "/tmp/output.apkg",
            "cards_count": 2,
            "errors_count": 0,
        }

    def _analysis(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return ("analysis", [], [("all", "all")], [("all", "all")], [("all", "all")])

    args = list(_run_args())
    args[16] = True
    deps = _deps(
        run_single_build=_run_single_build,
        build_run_analysis=_analysis,
    )
    events = list(handlers_run.on_run(*tuple(args), deps=deps))
    assert events
    assert seen["secondary_extract_enable"] is True


def test_on_run_analysis_invalid_payload_is_degraded() -> None:
    def _run_single_build(**kwargs):  # noqa: ANN003, ANN202
        return {
            "status": "ok",
            "run_id": "run_003",
            "output_path": "/tmp/output.apkg",
            "cards_count": 1,
            "errors_count": 0,
        }

    def _invalid_analysis(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return ("analysis",)

    deps = _deps(
        run_single_build=_run_single_build,
        build_run_analysis=_invalid_analysis,
    )
    events = list(handlers_run.on_run(*_run_args(), deps=deps))
    assert len(events) >= 1
    final = events[-1]
    assert "Completed" in final[0]
    assert "Run analytics unavailable" in final[5]


def test_on_run_selected_analysis_invalid_payload_is_degraded() -> None:
    def _run_single_build(**kwargs):  # noqa: ANN003, ANN202
        return {
            "status": "ok",
            "run_id": "run_004",
            "output_path": "/tmp/output.apkg",
            "cards_count": 1,
            "errors_count": 0,
        }

    def _invalid_analysis(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return ("analysis", [], [("all", "all")])

    deps = _deps(
        run_single_build=_run_single_build,
        build_run_analysis=_invalid_analysis,
    )
    detail_md, _download, analysis_md, sample_rows, _tax, _transfer, _rej, _chunk = (
        handlers_run.on_run_selected("run_004", "en", deps=deps)
    )
    assert "detail:en" in detail_md
    assert "Run analytics unavailable" in analysis_md
    assert sample_rows == []


def test_on_refresh_runs_invalid_payload_is_degraded() -> None:
    def _run_single_build(**kwargs):  # noqa: ANN003, ANN202
        return {
            "status": "ok",
            "run_id": "run_005",
            "output_path": "/tmp/output.apkg",
            "cards_count": 1,
            "errors_count": 0,
        }

    def _invalid_refresh(_cfg, *, lang, preferred_run_id=None):  # noqa: ANN001, ANN202
        return ({"value": preferred_run_id}, f"detail:{lang}")

    def _analysis(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return ("analysis", [], [("all", "all")], [("all", "all")], [("all", "all")])

    deps = handlers_run.RunDeps(
        normalize_ui_lang=lambda value: str(value or "en"),
        tr=lambda _lang, en, _zh: en,
        run_single_build=_run_single_build,
        to_optional_int=lambda value, min_value=None: int(value) if value not in (None, "") else None,
        to_optional_float=lambda value: float(value) if value not in (None, "") else None,
        as_str=lambda value, default="": (str(value).strip() if value is not None else default),
        load_app_config=lambda: object(),
        refresh_recent_runs=_invalid_refresh,
        load_run_detail=lambda _run_id, _cfg, *, lang: (f"detail:{lang}", None),
        build_run_analysis=_analysis,
    )
    selector_update, detail_md, _download, analysis_md, _rows, _tax, _transfer, _rej, _chunk = (
        handlers_run.on_refresh_runs("en", "run_005", deps=deps)
    )
    assert isinstance(selector_update, dict)
    assert selector_update.get("value") == "run_005"
    assert "Run details unavailable" in detail_md
    assert "analysis" in analysis_md


def test_on_apply_analysis_filters_failure_is_degraded() -> None:
    def _run_single_build(**kwargs):  # noqa: ANN003, ANN202
        return {
            "status": "ok",
            "run_id": "run_006",
            "output_path": "/tmp/output.apkg",
            "cards_count": 1,
            "errors_count": 0,
        }

    def _raise_analysis(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("analysis filter boom")

    deps = _deps(
        run_single_build=_run_single_build,
        build_run_analysis=_raise_analysis,
    )
    analysis_md, sample_rows = handlers_run.on_apply_analysis_filters(
        "run_006",
        "en",
        "all",
        "all",
        "all",
        "all",
        deps=deps,
    )
    assert "Run analytics unavailable" in analysis_md
    assert "analysis filter boom" in analysis_md
    assert sample_rows == []
