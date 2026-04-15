"""CLI exit codes."""

from __future__ import annotations


class ExitCode:
    SUCCESS = 0
    ARGUMENT_ERROR = 2
    CONFIG_ERROR = 3
    INPUT_ERROR = 4
    CHUNKING_ERROR = 5
    SCHEMA_ERROR = 6
    LLM_REQUEST_ERROR = 7
    LLM_PARSE_ERROR = 8
    CARD_VALIDATION_ERROR = 9
    TTS_ERROR = 10
    ANKI_EXPORT_ERROR = 11
    INTERNAL_ERROR = 12

