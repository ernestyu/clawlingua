"""Load and validate Anki template."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from ..errors import build_error
from ..exit_codes import ExitCode
from ..models.template_schema import AnkiTemplateSpec
from ..utils.jsonx import load_json


def _normalize_legacy_template(data: dict) -> dict:
    normalized = dict(data)
    if "model_name" not in normalized:
        normalized["model_name"] = normalized.get("note_type_name_default", "ClawLearn Cloze")
    if "deck_name" not in normalized:
        normalized["deck_name"] = normalized.get("deck_name_default", "ClawLearn Default Deck")
    if "card_templates" not in normalized and "templates" in normalized:
        templates = normalized.get("templates", {})
        normalized["card_templates"] = [
            {
                "name": "Card 1",
                "qfmt": templates.get("front", "{{cloze:Text}}"),
                "afmt": templates.get("back", "{{cloze:Text}}"),
            }
        ]
    return normalized


def load_anki_template(path: Path) -> AnkiTemplateSpec:
    if not path.exists():
        raise build_error(
            error_code="TEMPLATE_NOT_FOUND",
            cause="Anki 模板文件不存在。",
            detail=f"path={path}",
            next_steps=["检查 CLAWLEARN_ANKI_TEMPLATE 路径"],
            exit_code=ExitCode.SCHEMA_ERROR,
        )
    try:
        data = load_json(path)
        normalized = _normalize_legacy_template(data)
        spec = AnkiTemplateSpec.model_validate(normalized)
        spec.validate_field_order()
        return spec
    except ValidationError as exc:
        raise build_error(
            error_code="TEMPLATE_SCHEMA_INVALID",
            cause="Anki 模板 JSON 不符合要求。",
            detail=str(exc),
            next_steps=["修复模板字段", "重新运行 clawlearn doctor 检查模板"],
            exit_code=ExitCode.SCHEMA_ERROR,
        ) from exc
    except ValueError as exc:
        raise build_error(
            error_code="TEMPLATE_FIELDS_INVALID",
            cause="模板字段顺序不符合规范。",
            detail=str(exc),
            next_steps=["将字段顺序调整为 Text, Original, Translation, ExpressionTransfer, Note, Audio"],
            exit_code=ExitCode.SCHEMA_ERROR,
        ) from exc
    except Exception as exc:
        raise build_error(
            error_code="TEMPLATE_LOAD_FAILED",
            cause="模板读取失败。",
            detail=str(exc),
            next_steps=["检查模板 JSON 格式"],
            exit_code=ExitCode.SCHEMA_ERROR,
        ) from exc
