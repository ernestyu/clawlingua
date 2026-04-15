"""Hash helpers."""

from __future__ import annotations

import hashlib


def stable_hash(value: str, *, length: int = 12) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return digest[:length]


def stable_int_id(value: str, *, max_value: int = 2_147_483_647) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max_value

