"""
Whisper API STT Provider — OpenAI Whisper API wrapper.

Reference: Open-LLM-VTuber src/open_llm_vtuber/asr/ (whisper implementation)
"""

from __future__ import annotations

import logging
from io import BytesIO

from openai import AsyncOpenAI

from core.plugin_base import STTProvider, STTResult
from config import get_settings

logger = logging.getLogger("stt.whisper")


class WhisperAPIProvider(STTProvider):
    """Speech-to-text via OpenAI Whisper API."""

    config_schema = {
        "model": {"type": "string", "default": "whisper-1", "description": "Whisper model ID"},
        "language": {"type": "string", "default": "zh", "description": "Recognition language"},
        "prompt": {"type": "string", "default": "", "description": "Optional context prompt"},
    }

    def __init__(self, model: str = "whisper-1", language: str = "zh", prompt: str = "", **kwargs):
        settings = get_settings()
        self._model = model
        self._language = language
        self._prompt = prompt
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    @property
    def provider_name(self) -> str:
        return "whisper_api"

    @property
    def display_name(self) -> str:
        return f"OpenAI Whisper ({self._model})"

    async def transcribe(self, audio: bytes, audio_format: str = "opus") -> STTResult:
        """Send audio to Whisper API for transcription.

        Note: Whisper API expects specific formats. Opus/PCM are converted
        to a compatible WAV container in the dialogue service before calling.
        """
        import time
        start = time.monotonic()

        # Create a file-like object from audio bytes
        audio_file = BytesIO(audio)
        audio_file.name = f"audio.{'ogg' if audio_format == 'opus' else 'wav'}"

        response = await self._client.audio.transcriptions.create(
            model=self._model,
            file=audio_file,
            language=self._language,
            prompt=self._prompt if self._prompt else None,
            response_format="verbose_json",
        )

        duration_ms = int((time.monotonic() - start) * 1000)

        logger.debug(
            f"Whisper transcribed: '{response.text}' "
            f"(lang={response.language}, confidence={getattr(response, 'confidence', 0):.2f})"
        )

        return STTResult(
            text=response.text.strip(),
            language=response.language or self._language,
            confidence=getattr(response, 'confidence', 0.8),
            duration_ms=duration_ms,
        )

    async def validate_config(self) -> bool:
        settings = get_settings()
        return bool(settings.openai_api_key)
