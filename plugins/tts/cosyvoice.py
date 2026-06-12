"""
CosyVoice 2 TTS Provider — Alibaba's CosyVoice 2 via HTTP API.

CosyVoice 2 supports zero-shot voice cloning, emotion control, and streaming.
Typically deployed as a separate service via the CosyVoice project.
Reference: https://github.com/FunAudioLLM/CosyVoice

Expected API:
  GET  /health → {"status": "ok"}
  POST /tts   → streaming PCM audio
    Body: {"text": "...", "voice": "...", "emotion": "...", "speed": 1.0, "stream": true}
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from core.plugin_base import TTSProvider, TTSChunk, VoiceModelMeta
from config import get_settings

logger = logging.getLogger("tts.cosyvoice")


class CosyVoice2Provider(TTSProvider):
    """CosyVoice 2 TTS via HTTP API."""

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
            id="cosyvoice_default",
            name="CosyVoice 默认女声",
            engine="cosyvoice",
            engine_config={"voice": "default"},
            emotions=["happy", "gentle", "sad", "neutral", "excited", "angry", "comforting"],
            languages=["zh-CN", "en", "ja"],
            is_builtin=True,
        ),
    ]

    def __init__(self, endpoint: str = "", **kwargs):
        settings = get_settings()
        self._endpoint = (endpoint or getattr(settings, 'cosyvoice_endpoint', 'http://localhost:5001')).rstrip("/")

    @property
    def provider_name(self) -> str:
        return "cosyvoice2"

    @property
    def display_name(self) -> str:
        return "CosyVoice 2"

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
        """Stream TTS audio from CosyVoice 2 API."""
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
                        "stream": True,
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"CosyVoice API error: {resp.status}")
                        return

                    # Stream chunks — assume raw PCM 16-bit 16kHz mono
                    chunk_size = 640  # 20ms at 16kHz 16-bit mono
                    buffer = b""
                    async for data, end_of_http_chunk in resp.content.iter_chunks():
                        buffer += data
                        while len(buffer) >= chunk_size:
                            chunk = buffer[:chunk_size]
                            buffer = buffer[chunk_size:]

                            # Compute RMS for lip-sync
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
                                    rms = min(rms * 3.0, 1.0)  # Normalize

                            yield TTSChunk(audio=chunk, rms=rms, is_final=False)

                    # Flush remaining buffer
                    if buffer:
                        yield TTSChunk(audio=buffer, rms=0.0, is_final=True)
                    else:
                        yield TTSChunk(audio=b"", rms=0.0, is_final=True)

        except Exception as e:
            logger.error(f"CosyVoice synthesis error: {e}")
            yield TTSChunk(audio=b"", rms=0.0, is_final=True)

    async def list_voices(self) -> list[VoiceModelMeta]:
        """Return built-in voices + try to fetch from API."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._endpoint}/voices",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        voices = []
                        for v in data.get("voices", []):
                            voices.append(VoiceModelMeta(
                                id=f"cosyvoice_{v['id']}",
                                name=v.get("name", v["id"]),
                                engine="cosyvoice",
                                engine_config={"voice": v["id"]},
                                emotions=v.get("emotions", ["neutral"]),
                                languages=v.get("languages", ["zh-CN"]),
                                is_builtin=False,
                            ))
                        return voices
        except Exception:
            pass
        return list(self.BUILTIN_VOICES)
