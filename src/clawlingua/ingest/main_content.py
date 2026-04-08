"""Extract main content from HTML."""

from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from readability import Document

from ..errors import build_error
from ..exit_codes import ExitCode


@dataclass
class MainContentResult:
    title: str | None
    text: str
    markdown: str | None


def extract_main_content(html: str) -> MainContentResult:
    try:
        readable = Document(html)
        title = readable.short_title()
        article_html = readable.summary(html_partial=True)
    except Exception as exc:  # pragma: no cover
        raise build_error(
            error_code="INPUT_HTML_EXTRACT_FAILED",
            cause="网页正文提取失败。",
            detail=str(exc),
            next_steps=["检查输入网页是否可解析"],
            exit_code=ExitCode.INPUT_ERROR,
        ) from exc

    soup = BeautifulSoup(article_html, "html.parser")
    for bad in soup.select("script,style,noscript,nav,footer,header,aside"):
        bad.decompose()
    text = soup.get_text("\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    markdown = md(str(soup), heading_style="ATX")

    if not text.strip():
        raise build_error(
            error_code="INPUT_EMPTY_MAIN_CONTENT",
            cause="网页正文为空。",
            detail="正文提取后没有可用文本内容。",
            next_steps=["更换 URL 或手动保存文本后使用文件输入"],
            exit_code=ExitCode.INPUT_ERROR,
        )

    return MainContentResult(title=title, text=text, markdown=markdown)

