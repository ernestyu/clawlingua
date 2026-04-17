from __future__ import annotations

import pytest

from clawlearn.errors import ClawLearnError
from clawlearn.llm.response_parser import parse_extraction_json_content


def test_parse_extraction_json_content_cleans_control_chars_and_extracts_fragment() -> None:
    content = (
        "assistant note before json\n"
        "```json\n"
        '[{"chunk_id":"chunk_0001","phrase_text":"made\x00progress","sentence_text":"We made progress."}]\n'
        "```\n"
        "assistant note after json"
    )
    data, report = parse_extraction_json_content(content, expect_array=True)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["chunk_id"] == "chunk_0001"
    assert data[0]["phrase_text"] == "made progress"
    assert report.control_char_cleaned is True
    assert report.json_fragment_extracted is True
    assert report.partial_salvaged is False


def test_parse_extraction_json_content_salvages_truncated_array_prefix() -> None:
    content = (
        '[{"chunk_id":"chunk_0001","phrases":["made progress"]},'
        '{"chunk_id":"chunk_0002","phrases":["turn'
    )
    data, report = parse_extraction_json_content(content, expect_array=True)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["chunk_id"] == "chunk_0001"
    assert report.partial_salvaged is True
    assert report.salvaged_count == 1
    assert report.salvage_reason in {"truncated last item", "missing closing bracket"}


def test_parse_extraction_json_content_shape_invalid_raises() -> None:
    with pytest.raises(ClawLearnError) as exc_info:
        parse_extraction_json_content('{"chunk_id":"chunk_0001"}', expect_array=True)
    assert exc_info.value.error_code == "LLM_RESPONSE_SHAPE_INVALID"
