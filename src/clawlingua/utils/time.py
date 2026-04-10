"""Time helpers."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def make_run_id(prefix: str | None = None) -> str:
    """Generate a run id in `YYYYMMDDTHHMMSSZ_<short_hash>` format."""

    _ = prefix  # Backward compatible signature.
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_hash = secrets.token_hex(4)
    return f"{ts}_{short_hash}"
