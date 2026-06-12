"""
GPT-SoVITS TTS Provider — GPT-SoVITS v2/v3 via HTTP API.

GPT-SoVITS supports few-shot voice cloning, emotion control, and reference audio.
Typically deployed with api_v2.py from the GPT-SoVITS project.
Reference: https://github.com/RVC-Boss/GPT-SoVITS

Expected API:
  GET  /health → {"status": "ok"}
  POST /tts   → streaming audio
    Body: {"text": "...", "voice": "...", "emotion": "...", "speed": 1.0}
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from core.plugin_base import TTSProvider, TTSChunk, VoiceModelMeta
from config import get_settings

logger = logging.getLogger("tts.gpt_sovits")


class GPTSovitsProvider(TTSProvider):
    """GPT-SoVITS TTS via HTTP API."""

    EMOTION_SPEED: dict[str, float] = {
        "happy": 1.10,
        "gentle": 0.88,
        "sad": 0.78,
        "excited": 1.25,
        "neutral": 1.00,
        "comforting": 0.82,
        "curious": 1.05,
        "shy": 0.90,
        "playful": 1.15,
        "angry": 1.20,
    }

    BUILTIN_VOICES = [
        VoiceModelMeta(
            id="gptsovits_default",
            name="GPT-SoVITS 默认",
            engine="gpt_sovits",
            engine_config={"voice": "default"},
            emotions=["happy", "gentle", "sad", "neutral", "excited", "angry", "comforting"],
            languages=["zh-CN", "en", "ja"],
            is_builtin=True,
        ),
    ]

    def __init__(self, endpoint: str = "", **kwargs):
        settings = get_settings()
        self._endpoint = (endpoint or getattr(settings, 'gpt_sovits_endpoint', 'http://localhost:9880')).rstrip("/")

    @property
    def provider_name(self) -> str:
        return "gpt_sovits"

    @property
    def display_name(self) -> str:
        return "GPT-SoVITS"

    async def validate_config(self) -> bool:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._endpoint}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def synthesize(
        self,
        text: str,
        voice: VoiceModelMeta,
        emotion: str = "neutral",
        emotion_intensity: float = 0.5,
        speed: float = 1.0,
    ) -> AsyncIterator[TTSChunk]:
        """Stream TTS audio from GPT-SoVITS API."""
        import aiohttp
        import math

        emotion_speed = self.EMOTION_SPEED.get(emotion, 1.0)
        final_speed = emotion_speed * 0.7 + speed * 0.3

        voice_name = voice.engine_config.get("voice", "default")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._endpoint}/tts",
                    json={
                        "text": text,
                        "voice": voice_name,
                        "emotion": emotion,
                        "speed": final_speed,
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"GPT-SoVITS API error: {resp.status}")
                        return

                    # Stream chunks — assume raw PCM 16-bit 16kHz mono
                    chunk_size = 640  # 20ms at 16kHz 16-bit mono
                    buffer = b""
                    async for data, end_of_http_chunk in resp.content.iter_chunks():
                        buffer += data
                        while len(buffer) >= chunk_size:
                            chunk = buffer[:chunk_size]
                            buffer = buffer[chunk_size:]

                            rms = 0.0
                            if len(chunk) >= 2:
                                import struct
                                samples = [
                                    struct.unpack('<h', chunk[i:i+2])[0]
                                    for i in range(0, len(chunk) - 1, 2)
                                ]
                                if samples:
                                    rms = math.sqrt(
                                        sum(s * s for s in samples) / len(samples)
                                    ) / 32768.0
                                    rms = min(rms * 3.0, 1.0)

                            yield TTSChunk(audio=chunk, rms=rms, is_final=False)

                    if buffer:
                        yield TTSChunk(audio=buffer, rms=0.0, is_final=True)
                    else:
                        yield TTSChunk(audio=b"", rms=0.0, is_final=True)

        except Exception as e:
            logger.error(f"GPT-SoVITS synthesis error: {e}")
            yield TTSChunk(audio=b"", rms=0.0, is_final=True)

    async def list_voices(self) -> list[VoiceModelMeta]:
        """Return built-in voice as fallback."""
        return list(self.BUILTIN_VOICES)
