from __future__ import annotations

from dataclasses import dataclass

import clawlearn.llm.cloze_generator as cloze_generator
from clawlearn.config import AppConfig
from clawlearn.llm.cloze_generator import (
    generate_cloze_candidates_for_batch,
    generate_cloze_candidates_for_chunk,
    generate_phrase_candidates_for_batch,
    generate_phrase_candidates_for_chunk,
)
from clawlearn.models.chunk import ChunkRecord
from clawlearn.models.document import DocumentRecord
from clawlearn.models.prompt_schema import PromptOutputFormatSpec, PromptParserSpec, PromptSpec


@dataclass
class _FakeClient:
    responses: list[str]

    def __post_init__(self) -> None:
        self.config = AppConfig()
        self.calls: list[str] = []

    def chat(self, messages, *, temperature=None, max_retries=None):  # noqa: ANN001, ANN202
        user_content = ""
        for message in messages:
            if message.get("role") == "user":
                user_content = str(message.get("content") or "")
                break
        self.calls.append(user_content)
        if not self.responses:
            raise AssertionError("unexpected extra chat call")
        return self.responses.pop(0)


def _prompt() -> PromptSpec:
    return PromptSpec(
        name="test_extract",
        version="1",
        description="test",
        mode="extraction",
        system_prompt="system",
        user_prompt_template="{chunk_text}",
        placeholders=["chunk_text"],
        output_format=PromptOutputFormatSpec(type="json", schema_name="phrase_candidates_v1"),
        parser=PromptParserSpec(strip_code_fences=True, expect_json_array=True),
    )


def _document() -> DocumentRecord:
    return DocumentRecord(
        run_id="run_1",
        source_type="file",
        source_value="/tmp/input.txt",
        source_lang="en",
        target_lang="zh",
        title="t",
        raw_text="raw",
        cleaned_text="clean",
    )


def _chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        run_id="run_1",
        chunk_id=chunk_id,
        chunk_index=1,
        source_text=text,
        char_count=len(text),
        sentence_count=1,
    )


def test_generate_phrase_candidates_for_chunk_retries_parse_errors(monkeypatch) -> None:
    monkeypatch.setattr(cloze_generator.time, "sleep", lambda _seconds: None)
    client = _FakeClient(
        responses=[
            "not-json",
            '[{"chunk_id":"chunk_0001","context_sentences":["We made progress today."],"phrases":[{"text":"made progress"}]}]',
        ]
    )
    errors: list[dict] = []
    out = generate_phrase_candidates_for_chunk(
        client=client,
        prompt=_prompt(),
        document=_document(),
        chunk=_chunk("chunk_0001", "We made progress today."),
        events=errors,
    )
    assert len(out) == 1
    assert out[0]["phrase_text"] == "made progress"
    assert len(client.calls) == 2
    assert any(err.get("stage") == "extraction_parse_retry" for err in errors)


def test_generate_phrase_candidates_for_batch_salvages_and_refills_missing_chunks(monkeypatch) -> None:
    monkeypatch.setattr(cloze_generator.time, "sleep", lambda _seconds: None)
    client = _FakeClient(
        responses=[
            (
                '[{"chunk_id":"chunk_0001","context_sentences":["We made progress today."],'
                '"phrases":[{"text":"made progress"}]},'
                '{"chunk_id":"chunk_0002","context_sentences":["It turned out well."],"phrases":[{"text":"turn'
            ),
            '[{"chunk_id":"chunk_0002","context_sentences":["It turned out well."],"phrases":[{"text":"turned out"}]}]',
        ]
    )
    errors: list[dict] = []
    out = generate_phrase_candidates_for_batch(
        client=client,
        prompt=_prompt(),
        document=_document(),
        chunks=[
            _chunk("chunk_0001", "We made progress today."),
            _chunk("chunk_0002", "It turned out well."),
        ],
        events=errors,
    )
    assert {row["chunk_id"] for row in out} == {"chunk_0001", "chunk_0002"}
    assert len(client.calls) == 2
    assert "chunk_id=chunk_0001" in client.calls[0]
    assert "chunk_id=chunk_0002" in client.calls[0]
    assert "chunk_id=chunk_0001" not in client.calls[1]
    assert "chunk_id=chunk_0002" in client.calls[1]
    assert any(err.get("stage") == "extraction_parse_partial_salvage" for err in errors)
    assert any(err.get("stage") == "extraction_batch_partial_retry" for err in errors)


def test_generate_cloze_candidates_for_chunk_retries_shape_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(cloze_generator.time, "sleep", lambda _seconds: None)
    client = _FakeClient(
        responses=[
            '[{"chunk_id":"chunk_0001","context_sentences":["We made progress today."],"phrases":[{"text":"made progress"}]}]',
            '[{"chunk_id":"chunk_0001","text":"We {{c1::<b>made progress</b>}}(hint).","original":"We made progress.","target_phrases":["made progress"]}]',
        ]
    )
    errors: list[dict] = []
    out = generate_cloze_candidates_for_chunk(
        client=client,
        prompt=_prompt(),
        document=_document(),
        chunk=_chunk("chunk_0001", "We made progress today."),
        events=errors,
    )
    assert len(out) == 1
    assert out[0]["target_phrases"] == ["made progress"]
    retry_rows = [err for err in errors if err.get("stage") == "extraction_parse_retry"]
    assert retry_rows
    assert retry_rows[0].get("error_code") == "LLM_RESPONSE_SHAPE_INVALID"


def test_generate_cloze_candidates_for_batch_refills_missing_chunks(monkeypatch) -> None:
    monkeypatch.setattr(cloze_generator.time, "sleep", lambda _seconds: None)
    client = _FakeClient(
        responses=[
            '[{"chunk_id":"chunk_0001","text":"We {{c1::<b>made progress</b>}}(hint).","original":"We made progress.","target_phrases":["made progress"]}]',
            '[{"chunk_id":"chunk_0002","text":"It {{c1::<b>turned out</b>}}(hint) well.","original":"It turned out well.","target_phrases":["turned out"]}]',
        ]
    )
    errors: list[dict] = []
    out = generate_cloze_candidates_for_batch(
        client=client,
        prompt=_prompt(),
        document=_document(),
        chunks=[
            _chunk("chunk_0001", "We made progress."),
            _chunk("chunk_0002", "It turned out well."),
        ],
        events=errors,
    )
    assert {row["chunk_id"] for row in out} == {"chunk_0001", "chunk_0002"}
    assert len(client.calls) == 2
    assert "chunk_id=chunk_0001" not in client.calls[1]
    assert "chunk_id=chunk_0002" in client.calls[1]
    assert any(err.get("stage") == "extraction_batch_partial_retry" for err in errors)


def test_generate_phrase_candidates_for_chunk_skips_after_retry_exhausted(monkeypatch) -> None:
    monkeypatch.setattr(cloze_generator.time, "sleep", lambda _seconds: None)
    client = _FakeClient(responses=["bad-json-1", "bad-json-2", "bad-json-3"])
    errors: list[dict] = []
    out = generate_phrase_candidates_for_chunk(
        client=client,
        prompt=_prompt(),
        document=_document(),
        chunk=_chunk("chunk_0009", "No valid model output."),
        events=errors,
    )
    assert out == []
    assert len(client.calls) == 3
    assert any(err.get("stage") == "extraction_parse_exhausted" for err in errors)
    assert any(
        err.get("stage") == "extraction_chunk_skipped_after_parse_retries"
        and err.get("chunk_id") == "chunk_0009"
        for err in errors
    )
