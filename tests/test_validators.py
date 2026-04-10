import zipfile

import pytest

from clawlingua.errors import ClawLinguaError
from clawlingua.ingest.epub_reader import read_epub_file
from clawlingua.ingest.normalizer import NormalizeOptions, normalize_text, strip_markdown_to_text
from clawlingua.ingest.pdf_reader import read_pdf_file
from clawlingua.pipeline.validators import validate_text_candidate


def test_validate_text_candidate_reindexes_duplicate_c1() -> None:
    item = {
        "text": "A {{c1::<b>first phrase</b>}}(hint) and {{c1::<b>second phrase</b>}}(hint).",
        "original": "A first phrase and second phrase.",
        "target_phrases": ["first phrase", "second phrase"],
    }
    ok, reason = validate_text_candidate(
        item,
        max_sentences=3,
        min_chars=0,
        difficulty="intermediate",
    )
    assert ok, reason
    assert "{{c1::" in item["text"]
    assert "{{c2::" in item["text"]


def test_validate_text_candidate_rejects_easy_advanced_phrase() -> None:
    item = {
        "text": "You can {{c1::<b>think if</b>}}(hint) it is {{c2::<b>this data or that data</b>}}(hint).",
        "original": "You can think if it is this data or that data.",
        "target_phrases": ["think if", "this data or that data"],
    }
    ok, _ = validate_text_candidate(
        item,
        max_sentences=3,
        min_chars=0,
        difficulty="advanced",
    )
    assert not ok


def test_validate_text_candidate_rejects_duplicate_cloze_phrase() -> None:
    item = {
        "text": "It is {{c1::<b>such and such</b>}}(hint) company and {{c2::<b>such and such</b>}}(hint) amount.",
        "original": "It is such and such company and such and such amount.",
        "target_phrases": ["such and such"],
    }
    ok, _ = validate_text_candidate(
        item,
        max_sentences=3,
        min_chars=0,
        difficulty="intermediate",
    )
    assert not ok


def test_normalize_text_filters_low_value_lines() -> None:
    raw = """Transcript
00:00:00 - Explaining model jaggedness
Ilya Sutskever 00:00:00

You know what's crazy? That all of this is real.

Dwarkesh Patel 00:00:04

Meaning what?

Ilya Sutskever 00:00:47

Sure.

Should we actually begin here? I think this is an interesting discussion.
"""
    cleaned = normalize_text(raw)
    assert "Transcript" not in cleaned
    assert "Ilya Sutskever 00:00:00" not in cleaned
    assert "Dwarkesh Patel 00:00:04" not in cleaned
    assert "Meaning what?" not in cleaned
    assert "Sure." not in cleaned
    assert "Should we actually begin here? I think this is an interesting discussion." in cleaned
    assert "You know what's crazy? That all of this is real." in cleaned


def test_normalize_text_filters_short_lines_globally() -> None:
    raw = """Chapter One

Sure.

This paragraph should remain because this document is not transcript-like.
"""
    cleaned = normalize_text(raw)
    assert "Chapter One" not in cleaned
    assert "Sure." not in cleaned
    assert "This paragraph should remain because this document is not transcript-like." in cleaned


def test_normalize_text_can_disable_short_line_filter() -> None:
    raw = """Transcript
Dwarkesh Patel 00:00:04

Sure.
"""
    cleaned = normalize_text(
        raw,
        options=NormalizeOptions(short_line_max_words=0),
    )
    assert "Dwarkesh Patel 00:00:04" not in cleaned
    assert "Sure." in cleaned


def test_strip_markdown_to_text_removes_markup() -> None:
    raw = """# Title

This is **bold** and *italic* with [a link](https://example.com).

- one
- two

```python
print("x")
```
"""
    cleaned = strip_markdown_to_text(raw)
    assert "# Title" not in cleaned
    assert "**bold**" not in cleaned
    assert "[a link](https://example.com)" not in cleaned
    assert "a link" in cleaned
    assert "print(\"x\")" not in cleaned


def test_read_epub_file_extracts_plain_text(tmp_path) -> None:  # type: ignore[no-untyped-def]
    epub_path = tmp_path / "sample.epub"
    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""",
        )
        zf.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="BookId">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Sample Book</dc:title>
  </metadata>
  <manifest>
    <item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="c1"/>
    <itemref idref="c2"/>
  </spine>
</package>""",
        )
        zf.writestr("OEBPS/chapter1.xhtml", "<html><body><h1>One</h1><p>Hello <b>EPUB</b>.</p></body></html>")
        zf.writestr("OEBPS/chapter2.xhtml", "<html><body><p>Second chapter text.</p></body></html>")

    result = read_epub_file(epub_path)
    assert result.title == "Sample Book"
    assert "Hello EPUB." in result.text
    assert "Second chapter text." in result.text


def test_read_pdf_file_extracts_plain_text(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import clawlingua.ingest.pdf_reader as pdf_reader_module

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _Metadata:
        title = "Sample PDF"

    class _Page:
        def __init__(self, content: str) -> None:
            self._content = content

        def extract_text(self) -> str:
            return self._content

    class _Reader:
        def __init__(self, _: str) -> None:
            self.is_encrypted = False
            self.pages = [_Page("First page text."), _Page("Second page text.")]
            self.metadata = _Metadata()

    monkeypatch.setattr(pdf_reader_module, "_import_pdf_reader", lambda: _Reader)

    result = read_pdf_file(pdf_path)
    assert result.title == "Sample PDF"
    assert "First page text." in result.text
    assert "Second page text." in result.text


def test_read_pdf_file_rejects_empty_text(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import clawlingua.ingest.pdf_reader as pdf_reader_module

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _Page:
        def extract_text(self) -> str:
            return "   "

    class _Reader:
        def __init__(self, _: str) -> None:
            self.is_encrypted = False
            self.pages = [_Page()]
            self.metadata = {}

    monkeypatch.setattr(pdf_reader_module, "_import_pdf_reader", lambda: _Reader)

    with pytest.raises(ClawLinguaError) as exc_info:
        read_pdf_file(pdf_path)
    assert exc_info.value.error_code == "INPUT_PDF_EMPTY"
