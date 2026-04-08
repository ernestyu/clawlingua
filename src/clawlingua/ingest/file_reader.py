"""Read local text or markdown files."""

from __future__ import annotations

from pathlib import Path

import chardet

from ..errors import build_error
from ..exit_codes import ExitCode


def read_text_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise build_error(
            error_code="INPUT_FILE_NOT_FOUND",
            cause="输入文件不存在。",
            detail=f"path={path}",
            next_steps=["确认路径存在并重试"],
            exit_code=ExitCode.INPUT_ERROR,
        )

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_bytes()
        guess = chardet.detect(raw)
        encoding = guess.get("encoding") or "utf-8"
        try:
            return raw.decode(encoding)
        except Exception as exc:  # pragma: no cover
            raise build_error(
                error_code="INPUT_FILE_DECODE_FAILED",
                cause="输入文件无法解码。",
                detail=f"path={path}, guessed_encoding={encoding}, reason={exc}",
                next_steps=["将文件转换为 UTF-8 编码后重试"],
                exit_code=ExitCode.INPUT_ERROR,
            ) from exc

