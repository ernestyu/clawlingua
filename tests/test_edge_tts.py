from pathlib import Path

from clawlingua.tts.edge_tts_provider import EdgeTTSProvider
from clawlingua.tts.voice_selector import UniformVoiceSelector


def test_voice_selector_uniformity_basic() -> None:
    selector = UniformVoiceSelector(seed=42)
    voices = ["a", "b", "c"]
    counts = {v: 0 for v in voices}
    for _ in range(600):
        counts[selector.select(voices)] += 1
    assert all(v > 150 for v in counts.values())


def test_edge_tts_synthesize_success(monkeypatch, tmp_path: Path) -> None:
    async def fake_synthesize_async(self, *, text: str, voice: str, output_path: Path) -> None:
        output_path.write_bytes(b"fake-mp3")

    monkeypatch.setattr(EdgeTTSProvider, "_synthesize_async", fake_synthesize_async)
    provider = EdgeTTSProvider()
    output = tmp_path / "a.mp3"
    provider.synthesize(text="hello", voice="en-US-AnaNeural", output_path=output)
    assert output.exists()
    assert output.read_bytes() == b"fake-mp3"

