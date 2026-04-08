from pathlib import Path

import pytest

from clawlingua.errors import ClawLinguaError
from clawlingua.ingest.file_reader import read_text_file
from clawlingua.ingest.url_fetcher import fetch_url


def test_file_input_reading(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello", encoding="utf-8")
    assert read_text_file(path) == "hello"


def test_url_fetch_failure_prompt(monkeypatch) -> None:
    import httpx

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "Client", FakeClient)
    with pytest.raises(ClawLinguaError):
        fetch_url("https://example.com", timeout_seconds=5, user_agent="x", verify_ssl=True)

