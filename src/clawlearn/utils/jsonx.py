"""JSON helpers with optional orjson acceleration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None


def loads(text: str) -> Any:
    if orjson is not None:
        return orjson.loads(text)
    return json.loads(text)


def dumps(value: Any, *, indent: int = 2) -> str:
    if orjson is not None:
        option = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(value, option=option).decode("utf-8")
    return json.dumps(value, ensure_ascii=False, indent=indent)


def load_json(path: Path) -> Any:
    return loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(value, indent=indent), encoding="utf-8")


def dump_jsonl(path: Path, values: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in values:
            f.write(dumps(item, indent=0))
            f.write("\n")

