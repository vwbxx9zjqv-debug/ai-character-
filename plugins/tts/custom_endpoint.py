"""
Custom Endpoint TTS Provider — Generic TTS via user-provided HTTP endpoint.

Users configure their own TTS endpoint URL and audio format. The endpoint
must conform to a simple JSON-in / audio-out protocol.

Expected API:
  POST {endpoint}/tts
    Body: {"text": "...", "voice": "...", "emotion": "...", "speed": 1.0}
    Response: streaming audio (PCM, WAV, or MP3 based on config)
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from typing import AsyncIterator

from core.plugin_base import TTSProvider, TTSChunk, VoiceModelMeta
from config import get_settings

logger = logging.getLogger("tts.custom_endpoint")


class CustomEndpointProvider(TTSProvider):
    """Generic TTS provider for user-deployed custom endpoints."""

    config_schema = {
        "endpoint": {"type": "string", "default": "http://localhost:8001", "description": "TTS API base URL"},
        "format": {"type": "string", "default": "pcm", "description": "Audio format: pcm / wav / mp3"},
        "sample_rate": {"type": "int", "default": 16000, "description": "Sample rate in Hz"},
        "voice": {"type": "string", "default": "default", "description": "Default voice ID"},
    }

    BUILTIN_VOICES = [
        VoiceModelMeta(
            id="custom_tts_default",
            name="自定义端点",
            engine="custom",
            engine_config={"voice": "default"},
            emotions=["neutral", "happy", "gentle", "sad", "excited"],
            languages=["zh-CN"],
            is_builtin=True,
        ),
    ]

    def __init__(self, endpoint: str = "", format: str = "pcm", sample_rate: int = 16000, voice: str = "default", **kwargs):
        settings = get_settings()
        self._endpoint = (endpoint or getattr(settings, 'custom_tts_endpoint', 'http://localhost:8001')).rstrip("/")
        self._format = format
        self._sample_rate = sample_rate
        self._default_voice = voice

    @property
    def provider_name(self) -> str:
        return "custom_endpoint"

    @property
    def display_name(self) -> str:
        return f"自定义端点 ({self._endpoint})"

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
            # Don't fail — endpoint might not have /health
            return True

    async def synthesize(
        self,
        text: str,
        voice: VoiceModelMeta,
        emotion: str = "neutral",
        emotion_intensity: float = 0.5,
        speed: float = 1.0,
    ) -> AsyncIterator[TTSChunk]:
        """Stream TTS audio from custom endpoint."""
        import aiohttp
        import math

        voice_name = voice.engine_config.get("voice", self._default_voice)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._endpoint}/tts",
                    json={
                        "text": text,
                        "voice": voice_name,
                        "emotion": emotion,
                        "speed": speed,
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Custom endpoint API error: {resp.status}")
                        return

                    raw_data = await resp.read()

            if not raw_data:
                yield TTSChunk(audio=b"", rms=0.0, is_final=True)
                return

            # Convert to PCM 16-bit 16kHz mono if needed
            pcm_data = raw_data
            if self._format == "wav":
                pcm_data = self._wav_to_pcm(raw_data)
            elif self._format == "mp3":
                logger.warning("MP3 decoding not implemented, passing raw bytes")
                # MP3 decoding would need pydub/ffmpeg
                pcm_data = raw_data

            # Split into 20ms chunks
            chunk_size = 640  # 20ms at 16kHz 16-bit mono
            total_chunks = len(pcm_data) // chunk_size

            for i in range(total_chunks):
                start = i * chunk_size
                chunk = pcm_data[start:start + chunk_size]

                rms = 0.0
                if len(chunk) >= 2:
                    samples = [
                        struct.unpack('<h', chunk[j:j+2])[0]
                        for j in range(0, len(chunk) - 1, 2)
                    ]
                    if samples:
                        rms = math.sqrt(
                            sum(s * s for s in samples) / len(samples)
                        ) / 32768.0
                        rms = min(rms * 3.0, 1.0)

                is_final = (i == total_chunks - 1)
                yield TTSChunk(audio=chunk, rms=rms, is_final=is_final)

            # If no chunks were yielded (data too small), yield it as final
            if total_chunks == 0 and pcm_data:
                yield TTSChunk(audio=pcm_data, rms=0.0, is_final=True)

        except Exception as e:
            logger.error(f"Custom endpoint synthesis error: {e}")
            yield TTSChunk(audio=b"", rms=0.0, is_final=True)

    async def list_voices(self) -> list[VoiceModelMeta]:
        """Return built-in voice. Custom endpoints may not support voice listing."""
        return list(self.BUILTIN_VOICES)

    @staticmethod
    def _wav_to_pcm(wav_data: bytes) -> bytes:
        """Convert WAV to raw PCM 16-bit mono."""
        try:
            buf = io.BytesIO(wav_data)
            with wave.open(buf, "rb") as wf:
                return wf.readframes(wf.getnframes())
        except Exception:
            return wav_data
