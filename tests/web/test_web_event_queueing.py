from __future__ import annotations

from clawlearn_web.app import build_interface


def test_analysis_related_events_use_queue() -> None:
    demo = build_interface()
    try:
        dependencies = demo.config.get("dependencies", [])
        queue_by_name = {
            str(dep.get("api_name")): bool(dep.get("queue"))
            for dep in dependencies
            if isinstance(dep, dict)
        }
        assert queue_by_name.get("_on_refresh_runs") is True
        assert queue_by_name.get("_on_run_selected") is True
        assert queue_by_name.get("_on_apply_analysis_filters") is True
    finally:
        if hasattr(demo, "close"):
            demo.close()
