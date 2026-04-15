"""Prompt file discovery, validation, and persistence helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any, Callable

from pydantic import ValidationError

from clawlearn.models.prompt_schema import PromptSpec

PROMPT_DIR = Path("./prompts")
PROMPT_META_FILENAMES = {"user_prompt_overrides.json"}
PROMPT_TEMPLATE_FILENAMES = {"template_extraction.json", "template_explanation.json"}
PROMPT_TEMPLATE_BY_MODE = {
    "extraction": Path("./prompts/template_extraction.json"),
    "explanation": Path("./prompts/template_explanation.json"),
}
PROMPT_CONTENT_TYPE_OPTIONS = [
    "all",
    "prose_article",
    "transcript_dialogue",
    "textbook_examples",
]
PROMPT_LEARNING_MODE_OPTIONS = [
    "all",
    "lingua_expression",
    "lingua_reading",
    "textbook_focus",
    "textbook_review",
]
PROMPT_DIFFICULTY_OPTIONS = ["all", "beginner", "intermediate", "advanced"]


def as_str(value: Any, *, default: str = "") -> str:
    if isinstance(value, str):
        text = value.strip()
        return text if text else default
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def normalize_prompt_mode(value: Any) -> str:
    mode = as_str(value).lower()
    if mode == "cloze":
        return "extraction"
    if mode == "translate":
        return "explanation"
    if mode in {"extraction", "explanation"}:
        return mode
    return ""


def normalize_prompt_content_type(value: Any) -> str:
    content_type = as_str(value).lower()
    if content_type in {"", "auto"}:
        return "all"
    if content_type == "general":
        return "prose_article"
    if content_type in {"prose", "article"}:
        return "prose_article"
    if content_type in {"transcript", "dialogue"}:
        return "transcript_dialogue"
    if content_type in {"textbook", "example"}:
        return "textbook_examples"
    if content_type in PROMPT_CONTENT_TYPE_OPTIONS:
        return content_type
    return "all"


def normalize_prompt_learning_mode(value: Any) -> str:
    learning_mode = as_str(value).lower()
    if learning_mode in {"", "auto"}:
        return "all"
    if learning_mode in PROMPT_LEARNING_MODE_OPTIONS:
        return learning_mode
    return "all"


def normalize_prompt_difficulty(value: Any) -> str:
    difficulty = as_str(value).lower()
    if difficulty in {"", "auto"}:
        return "all"
    if difficulty in PROMPT_DIFFICULTY_OPTIONS:
        return difficulty
    return "all"


def normalize_prompt_metadata_from_payload(
    payload: dict[str, Any],
) -> tuple[str, str, str]:
    content_type = normalize_prompt_content_type(
        payload.get("content_type")
        or payload.get("material_profile")
        or payload.get("content_profile")
    )
    learning_mode = normalize_prompt_learning_mode(payload.get("learning_mode"))
    difficulty = normalize_prompt_difficulty(
        payload.get("difficulty_level")
        or payload.get("difficulty")
        or payload.get("cloze_difficulty")
    )
    return content_type, learning_mode, difficulty


def prompt_meta_matches(filter_value: str, prompt_value: str) -> bool:
    if filter_value == "all":
        return True
    if prompt_value == "all":
        return True
    return filter_value == prompt_value


def prompt_mode_label(
    mode: str, *, lang: str, tr: Callable[[str, str, str], str]
) -> str:
    if normalize_prompt_mode(mode) == "explanation":
        return tr(lang, "Explanation", "Explanation")
    return tr(lang, "Extraction", "Extraction")


def prompt_file_map(
    cfg: Any,
    *,
    mode_filter: str | None = None,
    content_type_filter: str | None = None,
    learning_mode_filter: str | None = None,
    difficulty_filter: str | None = None,
    include_templates: bool = False,
) -> dict[str, Path]:
    prompts_dir = cfg.resolve_path(PROMPT_DIR)
    mode_value = normalize_prompt_mode(mode_filter) if mode_filter else ""
    content_type_value = (
        normalize_prompt_content_type(content_type_filter)
        if content_type_filter is not None
        else "all"
    )
    learning_mode_value = (
        normalize_prompt_learning_mode(learning_mode_filter)
        if learning_mode_filter is not None
        else "all"
    )
    difficulty_value = (
        normalize_prompt_difficulty(difficulty_filter)
        if difficulty_filter is not None
        else "all"
    )
    if not prompts_dir.exists():
        return {}
    result: dict[str, Path] = {}
    for path in sorted(prompts_dir.glob("*.json")):
        if path.name in PROMPT_META_FILENAMES:
            continue
        if not include_templates and path.name in PROMPT_TEMPLATE_FILENAMES:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            spec = PromptSpec.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError, ValueError):
            continue
        mode = normalize_prompt_mode(spec.mode)
        if mode_value and mode != mode_value:
            continue
        prompt_content_type, prompt_learning_mode, prompt_difficulty = (
            normalize_prompt_metadata_from_payload(payload)
        )
        if not prompt_meta_matches(content_type_value, prompt_content_type):
            continue
        if not prompt_meta_matches(learning_mode_value, prompt_learning_mode):
            continue
        if not prompt_meta_matches(difficulty_value, prompt_difficulty):
            continue
        result[path.name] = path
    return result


def prompt_choices_from_map(
    prompt_files: dict[str, Path],
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for key, path in prompt_files.items():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            spec = PromptSpec.model_validate(payload)
            mode = normalize_prompt_mode(spec.mode)
        except (OSError, json.JSONDecodeError, ValidationError, ValueError):
            continue
        mode_text = prompt_mode_label(mode, lang=lang, tr=tr)
        choices.append((f"{path.name} ({mode_text})", key))
    return choices


def prompt_choices(
    cfg: Any,
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
    mode_filter: str | None = None,
    content_type_filter: str | None = None,
    learning_mode_filter: str | None = None,
    difficulty_filter: str | None = None,
    include_templates: bool = False,
) -> list[tuple[str, str]]:
    prompt_files = prompt_file_map(
        cfg,
        mode_filter=mode_filter,
        content_type_filter=content_type_filter,
        learning_mode_filter=learning_mode_filter,
        difficulty_filter=difficulty_filter,
        include_templates=include_templates,
    )
    return prompt_choices_from_map(prompt_files, lang=lang, tr=tr)


def prompt_path_value(cfg: Any, path: Path) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(cfg.workspace_root.resolve())
        rel_text = rel.as_posix()
        if rel_text.startswith("./"):
            return rel_text
        return f"./{rel_text}"
    except ValueError:
        return str(resolved)


def prompt_path_choices(
    cfg: Any,
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
    mode_filter: str,
    content_type_filter: str | None = None,
    learning_mode_filter: str | None = None,
    difficulty_filter: str | None = None,
    include_auto: bool = False,
) -> list[tuple[str, str]]:
    prompt_files = prompt_file_map(
        cfg,
        mode_filter=mode_filter,
        content_type_filter=content_type_filter,
        learning_mode_filter=learning_mode_filter,
        difficulty_filter=difficulty_filter,
        include_templates=False,
    )
    mode_text = prompt_mode_label(mode_filter, lang=lang, tr=tr)
    choices: list[tuple[str, str]] = []
    if include_auto:
        choices.append((tr(lang, "Auto (default chain)", "Auto (default chain)"), ""))
    for path in prompt_files.values():
        choices.append((f"{path.name} ({mode_text})", prompt_path_value(cfg, path)))
    return choices


def prompt_template_path(cfg: Any, mode: str) -> Path | None:
    normalized_mode = normalize_prompt_mode(mode)
    if not normalized_mode:
        return None
    template_rel = PROMPT_TEMPLATE_BY_MODE.get(normalized_mode)
    if template_rel is None:
        return None
    return cfg.resolve_path(template_rel)


def sanitize_prompt_filename(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", as_str(raw))
    if not safe:
        return ""
    if not safe.lower().endswith(".json"):
        safe += ".json"
    return safe


def read_prompt_payload(
    prompt_key: str,
    prompt_files: dict[str, Path],
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
) -> tuple[Path | None, dict[str, Any] | None, str]:
    path = prompt_files.get(prompt_key)
    if path is None:
        return (
            None,
            None,
            f"❌ {tr(lang, 'Failed to load prompt file', 'Failed to load prompt file')}: `{prompt_key}`",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (
            path,
            None,
            (
                f"❌ {tr(lang, 'Prompt file JSON parse error', 'Prompt file JSON parse error')}: "
                f"{exc.msg} (line {exc.lineno}, col {exc.colno})"
            ),
        )
    except Exception as exc:
        return (
            None,
            None,
            f"❌ {tr(lang, 'Failed to load prompt file', 'Failed to load prompt file')}: `{exc}`",
        )
    if not isinstance(payload, dict):
        return (
            path,
            None,
            (
                f"❌ {tr(lang, 'Prompt file JSON parse error', 'Prompt file JSON parse error')}: "
                "root must be a JSON object."
            ),
        )
    return path, payload, ""


def resolve_template_for_lang(value: Any, *, lang: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""

    order = [lang, "zh", "en"]
    seen: set[str] = set()
    for key in order:
        if key in seen:
            continue
        seen.add(key)
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            return text
    for text in value.values():
        if isinstance(text, str) and text.strip():
            return text
    return ""


def load_prompt_template(
    prompt_key: str,
    prompt_files: dict[str, Path],
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
) -> tuple[str, str]:
    _path, payload, msg = read_prompt_payload(prompt_key, prompt_files, lang=lang, tr=tr)
    if payload is None:
        return "", msg
    ok, validation_msg = validate_prompt_payload(payload, lang=lang, tr=tr)
    if not ok:
        return "", validation_msg
    template = resolve_template_for_lang(payload.get("user_prompt_template"), lang=lang)
    return template, ""


def load_prompt_mode(
    prompt_key: str,
    prompt_files: dict[str, Path],
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
) -> str:
    _path, payload, _msg = read_prompt_payload(prompt_key, prompt_files, lang=lang, tr=tr)
    if payload is None:
        return ""
    return normalize_prompt_mode(payload.get("mode"))


def load_prompt_filter_metadata(
    prompt_key: str,
    prompt_files: dict[str, Path],
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
) -> tuple[str, str, str]:
    _path, payload, _msg = read_prompt_payload(prompt_key, prompt_files, lang=lang, tr=tr)
    if payload is None:
        return ("all", "all", "all")
    return normalize_prompt_metadata_from_payload(payload)


def format_prompt_validation_error(exc: ValidationError) -> str:
    errors = exc.errors()
    lines: list[str] = []
    for err in errors[:5]:
        loc = ".".join(str(piece) for piece in err.get("loc", []))
        msg = str(err.get("msg", "invalid"))
        lines.append(f"- `{loc}`: {msg}" if loc else f"- {msg}")
    if len(errors) > 5:
        lines.append(f"- ... ({len(errors) - 5} more)")
    return "\n".join(lines)


def validate_prompt_payload(
    payload: dict[str, Any],
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
) -> tuple[bool, str]:
    try:
        PromptSpec.model_validate(payload)
    except ValidationError as exc:
        return (
            False,
            (
                f"❌ {tr(lang, 'Schema validation failed', 'Schema validation failed')}\n\n"
                f"{format_prompt_validation_error(exc)}"
            ),
        )
    return True, ""


def set_user_prompt_template(payload: dict[str, Any], *, template: str) -> None:
    payload["user_prompt_template"] = template


def write_prompt_payload(
    path: Path,
    payload: dict[str, Any],
    *,
    lang: str,
    tr: Callable[[str, str, str], str],
) -> tuple[bool, str]:
    ok, msg = validate_prompt_payload(payload, lang=lang, tr=tr)
    if not ok:
        return (
            False,
            (
                f"{msg}\n\n"
                f"⚠️ {tr(lang, 'Not saved because validation failed.', 'Not saved because validation failed.')}"
            ),
        )
    backup = path.with_suffix(path.suffix + ".bak")
    try:
        if path.exists():
            shutil.copyfile(path, backup)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        return (
            False,
            f"❌ {tr(lang, 'Failed to save prompt file', 'Failed to save prompt file')}: `{exc}`",
        )
    return (
        True,
        f"- file: `{path}`\n- {tr(lang, 'Backup created', 'Backup created')}: `{backup}`",
    )


def create_prompt_file(
    cfg: Any,
    *,
    raw_name: str,
    mode: str,
    lang: str,
    tr: Callable[[str, str, str], str],
) -> tuple[bool, str, str]:
    mode_value = normalize_prompt_mode(mode) or "extraction"
    file_name = sanitize_prompt_filename(raw_name)
    if not file_name:
        file_name = (
            f"{mode_value}_prompt_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )
    if file_name in PROMPT_META_FILENAMES or file_name in PROMPT_TEMPLATE_FILENAMES:
        return (
            False,
            "",
            f"❌ {tr(lang, 'Prompt file already exists.', 'Prompt file already exists.')}: `{file_name}`",
        )

    prompts_dir = cfg.resolve_path(PROMPT_DIR)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    target_path = prompts_dir / file_name
    if target_path.exists():
        return (
            False,
            "",
            f"❌ {tr(lang, 'Prompt file already exists.', 'Prompt file already exists.')}: `{target_path}`",
        )

    template_path = prompt_template_path(cfg, mode_value)
    if template_path is None or not template_path.exists():
        return (
            False,
            "",
            f"❌ {tr(lang, 'Template prompt file missing.', 'Template prompt file missing.')}: `{template_path}`",
        )

    try:
        template_payload = json.loads(template_path.read_text(encoding="utf-8"))
        template_spec = PromptSpec.model_validate(template_payload)
        payload = template_spec.model_dump(mode="json")
        payload["name"] = Path(file_name).stem
        payload["mode"] = mode_value
        target_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        return (
            False,
            "",
            f"❌ {tr(lang, 'Failed to save prompt file', 'Failed to save prompt file')}: `{exc}`",
        )

    status = (
        f"✅ {tr(lang, 'Prompt file created.', 'Prompt file created.')}\n\n"
        f"- file: `{target_path}`"
    )
    return True, file_name, status


def save_prompt_file(
    cfg: Any,
    *,
    prompt_key: str,
    mode: str,
    template: str,
    lang: str,
    tr: Callable[[str, str, str], str],
) -> tuple[bool, str]:
    template_value = (template or "").rstrip()
    if not template_value.strip():
        return (
            False,
            f"❌ {tr(lang, 'Prompt template is empty.', 'Prompt template is empty.')}",
        )

    prompt_files_now = prompt_file_map(cfg)
    path, payload, msg = read_prompt_payload(prompt_key, prompt_files_now, lang=lang, tr=tr)
    if payload is None or path is None:
        return False, msg

    payload["mode"] = normalize_prompt_mode(mode) or "extraction"
    set_user_prompt_template(payload, template=template_value)
    ok, details = write_prompt_payload(path, payload, lang=lang, tr=tr)
    if not ok:
        return False, details

    return (
        True,
        f"✅ {tr(lang, 'Prompt template saved.', 'Prompt template saved.')}\n\n{details}",
    )


def rename_prompt_file(
    cfg: Any,
    *,
    prompt_key: str,
    target_raw_name: str,
    lang: str,
    tr: Callable[[str, str, str], str],
) -> tuple[bool, str, str]:
    prompt_files_now = prompt_file_map(cfg)
    current_path = prompt_files_now.get(prompt_key)
    if current_path is None:
        return (
            False,
            prompt_key,
            f"❌ {tr(lang, 'Failed to load prompt file', 'Failed to load prompt file')}: `{prompt_key}`",
        )

    target_name = sanitize_prompt_filename(target_raw_name)
    if not target_name:
        return (
            False,
            prompt_key,
            f"❌ {tr(lang, 'Prompt file name is empty.', 'Prompt file name is empty.')}",
        )
    if target_name in PROMPT_META_FILENAMES or target_name in PROMPT_TEMPLATE_FILENAMES:
        return (
            False,
            prompt_key,
            f"❌ {tr(lang, 'Prompt file already exists.', 'Prompt file already exists.')}: `{target_name}`",
        )

    target_path = current_path.with_name(target_name)
    if target_path.exists():
        return (
            False,
            prompt_key,
            f"❌ {tr(lang, 'Prompt file already exists.', 'Prompt file already exists.')}: `{target_path}`",
        )

    path, payload, msg = read_prompt_payload(prompt_key, prompt_files_now, lang=lang, tr=tr)
    if payload is None or path is None:
        return False, prompt_key, msg

    payload["name"] = target_path.stem
    ok, details = write_prompt_payload(target_path, payload, lang=lang, tr=tr)
    if not ok:
        return False, prompt_key, details

    try:
        path.unlink()
    except OSError as exc:
        return (
            True,
            target_name,
            (
                f"⚠️ {tr(lang, 'Prompt file renamed.', 'Prompt file renamed.')} "
                f"`{target_path}`; old file cleanup failed: `{exc}`"
            ),
        )

    status = (
        f"✅ {tr(lang, 'Prompt file renamed.', 'Prompt file renamed.')}\n\n"
        f"- from: `{path}`\n"
        f"- to: `{target_path}`\n"
        f"{details}"
    )
    return True, target_name, status


def delete_prompt_file(
    cfg: Any,
    *,
    prompt_key: str,
    lang: str,
    tr: Callable[[str, str, str], str],
) -> tuple[bool, str, str]:
    prompt_files_now = prompt_file_map(cfg)
    path = prompt_files_now.get(prompt_key)
    if path is None:
        return (
            False,
            "",
            f"❌ {tr(lang, 'Failed to load prompt file', 'Failed to load prompt file')}: `{prompt_key}`",
        )

    mode = load_prompt_mode(prompt_key, prompt_files_now, lang=lang, tr=tr)
    mode_counts = {"extraction": 0, "explanation": 0}
    for key in prompt_files_now:
        key_mode = load_prompt_mode(key, prompt_files_now, lang=lang, tr=tr)
        if key_mode in mode_counts:
            mode_counts[key_mode] += 1
    if mode == "extraction" and mode_counts["extraction"] <= 1:
        return (
            False,
            mode,
            (
                "❌ "
                f"{tr(lang, 'Cannot delete the last Extraction prompt.', 'Cannot delete the last Extraction prompt.')}"
            ),
        )
    if mode == "explanation" and mode_counts["explanation"] <= 1:
        return (
            False,
            mode,
            (
                "❌ "
                f"{tr(lang, 'Cannot delete the last Explanation prompt.', 'Cannot delete the last Explanation prompt.')}"
            ),
        )

    try:
        path.unlink()
    except OSError as exc:
        return (
            False,
            mode,
            f"❌ {tr(lang, 'Failed to save prompt file', 'Failed to save prompt file')}: `{exc}`",
        )

    status = (
        f"✅ {tr(lang, 'Prompt file deleted.', 'Prompt file deleted.')}\n\n"
        f"- file: `{path}`"
    )
    return True, mode, status
