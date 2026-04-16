"""Analytics tab component builder and event wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gradio as gr

from clawlearn_web import handlers_run


@dataclass(frozen=True)
class AnalyticsTabComponents:
    analytics_tab: Any
    analytics_heading: Any
    taxonomy_filter: Any
    transfer_filter: Any
    rejection_filter: Any
    chunk_filter: Any
    apply_analysis_filter_btn: Any
    run_analysis: Any
    run_samples: Any


def build_tab(
    *,
    initial_ui_lang: str,
    initial_analysis_md: str,
    initial_samples_rows: list[list[Any]],
    initial_taxonomy_choices: list[tuple[str, str]],
    initial_rejection_choices: list[tuple[str, str]],
    initial_chunk_choices: list[tuple[str, str]],
    tr: Callable[[str, str, str], str],
) -> AnalyticsTabComponents:
    with gr.Tab(
        tr(initial_ui_lang, "Run analytics", "杩愯缁熻鍒嗘瀽")
    ) as analytics_tab:
        analytics_heading = gr.Markdown(
            tr(initial_ui_lang, "### Run analytics", "### 杩愯缁熻鍒嗘瀽"),
        )
        with gr.Row():
            taxonomy_filter = gr.Dropdown(
                choices=initial_taxonomy_choices,
                value="all",
                label=tr(initial_ui_lang, "Taxonomy filter", "taxonomy 杩囨护"),
            )
            transfer_filter = gr.Dropdown(
                choices=[
                    ("all", "all"),
                    ("with_transfer", "with_transfer"),
                    ("without_transfer", "without_transfer"),
                ],
                value="all",
                label=tr(initial_ui_lang, "Transfer filter", "transfer 杩囨护"),
            )
            rejection_filter = gr.Dropdown(
                choices=initial_rejection_choices,
                value="all",
                label=tr(initial_ui_lang, "Rejection filter", "鎷掔粷鍘熷洜杩囨护"),
            )
            chunk_filter = gr.Dropdown(
                choices=initial_chunk_choices,
                value="all",
                label=tr(initial_ui_lang, "Chunk filter", "chunk 杩囨护"),
            )
        apply_analysis_filter_btn = gr.Button(
            tr(initial_ui_lang, "Apply filters", "搴旂敤杩囨护"),
        )
        run_analysis = gr.Markdown(value=initial_analysis_md)
        run_samples = gr.Dataframe(
            headers=[
                "kind",
                "chunk_id",
                "phrase_types",
                "score",
                "text",
                "expression_transfer",
                "reason",
            ],
            datatype=["str", "str", "str", "number", "str", "str", "str"],
            interactive=False,
            wrap=True,
            value=initial_samples_rows,
            label=tr(initial_ui_lang, "Representative samples", "Representative samples"),
        )

    return AnalyticsTabComponents(
        analytics_tab=analytics_tab,
        analytics_heading=analytics_heading,
        taxonomy_filter=taxonomy_filter,
        transfer_filter=transfer_filter,
        rejection_filter=rejection_filter,
        chunk_filter=chunk_filter,
        apply_analysis_filter_btn=apply_analysis_filter_btn,
        run_analysis=run_analysis,
        run_samples=run_samples,
    )


def bind_events(
    *,
    components: AnalyticsTabComponents,
    run_selector: Any,
    ui_lang: Any,
    deps: handlers_run.RunDeps,
) -> None:
    def _on_apply_analysis_filters(
        run_id_val: str | None,
        ui_lang_val: str,
        taxonomy_val: str,
        transfer_val: str,
        rejection_val: str,
        chunk_val: str,
    ) -> tuple[str, list[list[Any]]]:
        return handlers_run.on_apply_analysis_filters(
            run_id_val,
            ui_lang_val,
            taxonomy_val,
            transfer_val,
            rejection_val,
            chunk_val,
            deps=deps,
        )

    components.apply_analysis_filter_btn.click(
        _on_apply_analysis_filters,
        inputs=[
            run_selector,
            ui_lang,
            components.taxonomy_filter,
            components.transfer_filter,
            components.rejection_filter,
            components.chunk_filter,
        ],
        outputs=[components.run_analysis, components.run_samples],
        queue=False,
    )
