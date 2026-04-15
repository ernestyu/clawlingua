"""Error primitives for structured CLI output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .exit_codes import ExitCode


@dataclass
class ClawLearnError(Exception):
    error_code: str
    cause: str
    detail: str
    next_steps: list[str] = field(default_factory=list)
    exit_code: int = ExitCode.INTERNAL_ERROR

    def to_lines(self) -> list[str]:
        lines = [
            f"ERROR | {self.error_code}",
            f"CAUSE | {self.cause}",
            f"DETAIL | {self.detail}",
        ]
        lines.extend(f"NEXT | {step}" for step in self.next_steps)
        return lines


def format_error(error: ClawLearnError) -> str:
    return "\n".join(error.to_lines())


def build_error(
    *,
    error_code: str,
    cause: str,
    detail: str,
    next_steps: Iterable[str],
    exit_code: int,
) -> ClawLearnError:
    return ClawLearnError(
        error_code=error_code,
        cause=cause,
        detail=detail,
        next_steps=list(next_steps),
        exit_code=exit_code,
    )

