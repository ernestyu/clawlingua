"""OpenAI-compatible chat client."""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from ..config import AppConfig
from ..errors import build_error
from ..exit_codes import ExitCode
from ..utils.jsonx import loads


class OpenAICompatibleClient:
    def __init__(self, cfg: AppConfig, *, for_translation: bool = False) -> None:
        self._cfg = cfg
        self._for_translation = for_translation
        # Choose endpoint by usage: translation LLM can have an isolated backend.
        if for_translation and cfg.translate_llm_base_url:
            base = cfg.translate_llm_base_url
        else:
            base = cfg.llm_base_url
        self._endpoint = base.rstrip("/") + "/chat/completions"

    @property
    def config(self) -> AppConfig:
        return self._cfg

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_retries: int | None = None,
    ) -> str:
        # Pick model/key/temperature by usage.
        if self._for_translation and self._cfg.translate_llm_model:
            model = self._cfg.translate_llm_model
            api_key = self._cfg.translate_llm_api_key or self._cfg.llm_api_key
            temp_default = (
                self._cfg.translate_llm_temperature
                if self._cfg.translate_llm_temperature is not None
                else self._cfg.llm_temperature
            )
        else:
            model = self._cfg.llm_model
            api_key = self._cfg.llm_api_key
            temp_default = self._cfg.llm_temperature

        temp = temp_default if temperature is None else temperature
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temp,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        retries = self._cfg.llm_max_retries if max_retries is None else max(1, int(max_retries))
        last_err: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                with httpx.Client(timeout=self._cfg.llm_timeout_seconds) as client:
                    response = client.post(self._endpoint, json=payload, headers=headers)

                # Small jitter after successful request to reduce bursty traffic.
                if self._cfg.llm_request_sleep_seconds and self._cfg.llm_request_sleep_seconds > 0:
                    base = float(self._cfg.llm_request_sleep_seconds)
                    time.sleep(random.uniform(base, 3 * base))

                if response.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        message=f"status={response.status_code}",
                        request=response.request,
                        response=response,
                    )

                data = loads(response.text)
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(str(item["text"]))
                    return "\n".join(parts).strip()
                return str(content)
            except Exception as exc:
                last_err = exc
                if attempt >= retries:
                    break

                delay = self._cfg.llm_retry_backoff_seconds * (2 ** (attempt - 1))
                if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                    if exc.response.status_code == 429:
                        retry_after = _parse_retry_after_seconds(exc.response.headers)
                        if retry_after is not None:
                            delay = max(delay, retry_after)
                time.sleep(delay)

        raise build_error(
            error_code="LLM_REQUEST_FAILED",
            cause="LLM request failed.",
            detail=_format_request_error_detail(last_err),
            next_steps=[
                "Check CLAWLINGUA_LLM_BASE_URL and API key.",
                "Confirm the model endpoint is reachable.",
                "Increase timeout/retries or reduce request rate.",
            ],
            exit_code=ExitCode.LLM_REQUEST_ERROR,
        )


def _parse_retry_after_seconds(headers: httpx.Headers) -> float | None:
    value = str(headers.get("Retry-After", "")).strip()
    if not value:
        return None
    try:
        seconds = float(value)
        return max(0.0, seconds)
    except ValueError:
        pass

    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max(0.0, (when - now).total_seconds())


def _format_request_error_detail(exc: Exception | None) -> str:
    if exc is None:
        return "unknown request failure"
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        status = exc.response.status_code
        body = (exc.response.text or "").strip()
        body_preview = body[:400]
        return f"status={status}, body={body_preview!r}"
    if isinstance(exc, httpx.TimeoutException):
        return f"timeout: {exc}"
    return str(exc)
