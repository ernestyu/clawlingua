from pathlib import Path

from clawlingua.config import load_config


def test_load_config_from_env_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "CLAWLINGUA_DEFAULT_SOURCE_LANG=ja",
                "CLAWLINGUA_DEFAULT_TARGET_LANG=en",
                "CLAWLINGUA_LLM_API_KEY=test_key_123",
                "CLAWLINGUA_LLM_MODEL=test-model",
                "CLAWLINGUA_TTS_EDGE_JA_VOICES=a,b,c",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(env_file=env, workspace_root=tmp_path)
    assert cfg.default_source_lang == "ja"
    assert cfg.default_target_lang == "en"
    assert cfg.llm_api_key == "test_key_123"
    assert cfg.llm_model == "test-model"
    assert cfg.get_source_voices("ja") == ["a", "b", "c"]

