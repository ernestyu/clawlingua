"""Fetch URL content."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from ..errors import build_error
from ..exit_codes import ExitCode


@dataclass
class URLFetchResult:
    url: str
    final_url: str
    status_code: int
    html: str
    fetched_at: str | None = None


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise build_error(
            error_code="INPUT_URL_INVALID",
            cause="输入 URL 非法。",
            detail=f"url={url}",
            next_steps=["提供带 http/https 的完整 URL"],
            exit_code=ExitCode.INPUT_ERROR,
        )


def fetch_url(
    url: str,
    *,
    timeout_seconds: float,
    user_agent: str,
    verify_ssl: bool,
) -> URLFetchResult:
    _validate_url(url)

    headers = {"User-Agent": user_agent}
    try:
        with httpx.Client(timeout=timeout_seconds, verify=verify_ssl, headers=headers) as client:
            response = client.get(url, follow_redirects=True)
    except httpx.RequestError as exc:
        raise build_error(
            error_code="INPUT_URL_FETCH_FAILED",
            cause="URL 抓取失败。",
            detail=f"url={url}, reason={exc}",
            next_steps=["检查网络连接或提高 HTTP 超时配置"],
            exit_code=ExitCode.INPUT_ERROR,
        ) from exc

    if response.status_code >= 400:
        raise build_error(
            error_code="INPUT_URL_HTTP_ERROR",
            cause="URL 返回非成功状态码。",
            detail=f"url={url}, status={response.status_code}",
            next_steps=["检查 URL 是否可访问", "确认目标站点未屏蔽请求"],
            exit_code=ExitCode.INPUT_ERROR,
        )

    return URLFetchResult(
        url=url,
        final_url=str(response.url),
        status_code=response.status_code,
        html=response.text,
    )

