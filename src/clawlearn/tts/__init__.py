"""TTS package."""

from .provider_registry import get_tts_provider
from .voice_selector import UniformVoiceSelector

__all__ = ["get_tts_provider", "UniformVoiceSelector"]

