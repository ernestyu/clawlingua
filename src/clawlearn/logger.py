"""Logging setup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from .constants import APP_NAME
from .utils.fs import ensure_dir


_LOGGERS_CONFIGURED: set[str] = set()


def setup_logging(
    level: str = "INFO",
    *,
    log_dir: Path | None = None,
    mode: Literal["append", "rotate"] = "append",
) -> None:
    """Configure logging.

    - Always logs to stdout (for interactive runs).
    - When ``log_dir`` is provided, also logs to ``<log_dir>/<APP_NAME>.log``.
    - ``mode`` currently only supports ``append``; rotation can be added later.
    """

    global _LOGGERS_CONFIGURED

    root = logging.getLogger()
    if root.name in _LOGGERS_CONFIGURED:
        # Avoid re-adding handlers when called multiple times.
        return

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Stdout handler for foreground inspection
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    if log_dir is not None:
        log_dir = ensure_dir(log_dir)
        log_path = log_dir / f"{APP_NAME}.log"
        file_handler = logging.FileHandler(log_path, mode="a" if mode == "append" else "w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _LOGGERS_CONFIGURED.add(root.name)

