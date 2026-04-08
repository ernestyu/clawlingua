"""OpenAI-compatible chat client."""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..config import AppConfig
from ..errors import build_error
from ..exit_codes import ExitCode
from ..utils.jsonx import loads


class OpenAICompatibleClient:
    def __init__(self, cfg: AppConfig, *, for_translation: bool = False) -> None:
        self._cfg = cfg
        self._for_translation = for_translation
        # 根据用途选择 base_url
        if for_translation and cfg.translate_llm_base_url:
            base = cfg.translate_llm_base_url
        else:
            base = cfg.llm_base_url
        self._endpoint = base.rstrip("/") + "/chat/completions"

    @property
    def config(self) -> AppConfig:
        return self._cfg

    def chat(self, messages: list[dict[str, str]], *, temperature: float | None = None) -> str:
        # 根据用途选择模型/温度/API key
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
        last_err: Exception | None = None
        for attempt in range(1, self._cfg.llm_max_retries + 1):
            try:
                with httpx.Client(timeout=self._cfg.llm_timeout_seconds) as client:
                    response = client.post(self._endpoint, json=payload, headers=headers)
                if response.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        message=f"status={response.status_code}",
                        request=response.request,
                        response=response,
                    )
                data = loads(response.text)
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(str(item["text"]))
                    return "\n".join(parts).strip()
                return str(content)
            except Exception as exc:
                last_err = exc
                if attempt >= self._cfg.llm_max_retries:
                    break
                time.sleep(self._cfg.llm_retry_backoff_seconds * attempt)

        raise build_error(
            error_code="LLM_REQUEST_FAILED",
            cause="LLM 请求失败。",
            detail=str(last_err),
            next_steps=[
                "检查 CLAWLINGUA_LLM_BASE_URL 与 API key",
                "确认模型服务可访问",
                "可尝试增加超时或重试次数",
            ],
            exit_code=ExitCode.LLM_REQUEST_ERROR,
        )

