import pytest

from clawlingua.errors import ClawLinguaError
from clawlingua.llm.response_parser import parse_json_content


def test_parse_cloze_response_json_array() -> None:
    content = "```json\n[{\"text\":\"{{c1::x}}\", \"original\":\"x\", \"target_phrases\":[\"x\"]}]\n```"
    data = parse_json_content(content, expect_array=True)
    assert isinstance(data, list)
    assert data[0]["original"] == "x"


def test_parse_translation_response_invalid_json() -> None:
    with pytest.raises(ClawLinguaError):
        parse_json_content("not-json", expect_array=True)

