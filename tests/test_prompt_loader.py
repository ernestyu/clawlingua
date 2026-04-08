from pathlib import Path

from clawlingua.llm.prompt_loader import load_prompt


def test_load_prompt_schema() -> None:
    prompt = load_prompt(Path("prompts/cloze_contextual.json"))
    assert prompt.mode == "cloze"
    assert "chunk_text" in prompt.placeholders

