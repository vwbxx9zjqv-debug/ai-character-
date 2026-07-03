"""
MiniMax TTS Provider — High-quality Chinese speech synthesis.

MiniMax API: https://platform.minimaxi.com
Reference: MiniMax T2A HTTP API (/v1/t2a_v2)
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

import numpy as np
import httpx

from core.plugin_base import TTSProvider, TTSChunk, VoiceModelMeta
from config import get_settings

logger = logging.getLogger("tts.minimax")


class MiniMaxTTSProvider(TTSProvider):
    """High-quality Chinese TTS using MiniMax API.

    Supports emotion-driven speed/pitch modulation and 20+ Chinese voices.
    """

    config_schema = {
        "api_key": {"type": "string", "default": "", "description": "MiniMax API key"},
        "model": {"type": "string", "default": "speech-2.6-turbo", "description": "TTS model"},
        "voice_id": {"type": "string", "default": "female-shaonv", "description": "Default voice ID"},
        "base_url": {"type": "string", "default": "https://api.minimaxi.com/v1", "description": "API base URL"},
    }

    # Emotion → speed multiplier (MiniMax speed range: 0.5–2.0)
    EMOTION_SPEED = {
        "happy": 1.15,
        "gentle": 0.85,
        "sad": 0.75,
        "excited": 1.30,
        "neutral": 1.0,
        "comforting": 0.80,
        "curious": 1.05,
        "shy": 0.90,
        "playful": 1.20,
    }

    # Emotion → pitch adjustment (MiniMax pitch range: -12 to +12)
    EMOTION_PITCH = {
        "happy": 2,
        "gentle": 0,
        "sad": -3,
        "excited": 4,
        "neutral": 0,
        "comforting": -1,
        "curious": 1,
        "shy": -2,
        "playful": 3,
    }

    # Built-in Chinese voices (subset of MiniMax's catalog)
    BUILTIN_VOICES = [
        VoiceModelMeta(
            id="minimax_shaonv", name="少女 (活泼可爱)", engine="minimax",
            engine_config={"voice_id": "female-shaonv"},
            emotions=["happy", "gentle", "excited", "neutral", "shy", "curious", "playful"],
            languages=["zh-CN"], is_builtin=True,
        ),
        VoiceModelMeta(
            id="minimax_qingse", name="青涩少年 (元气少年)", engine="minimax",
            engine_config={"voice_id": "male-qn-qingse"},
            emotions=["happy", "excited", "neutral", "playful"],
            languages=["zh-CN"], is_builtin=True,
        ),
        VoiceModelMeta(
            id="minimax_wenrou", name="温柔女性 (知性优雅)", engine="minimax",
            engine_config={"voice_id": "gentle-female"},
            emotions=["gentle", "neutral", "sad", "comforting", "shy"],
            languages=["zh-CN"], is_builtin=True,
        ),
        VoiceModelMeta(
            id="minimax_nanzhu", name="男主播 (沉稳大气)", engine="minimax",
            engine_config={"voice_id": "presenter_male"},
            emotions=["neutral", "gentle", "comforting"],
            languages=["zh-CN"], is_builtin=True,
        ),
        VoiceModelMeta(
            id="minimax_nvzhu", name="女主播 (专业播报)", engine="minimax",
            engine_config={"voice_id": "presenter_female"},
            emotions=["neutral", "gentle", "excited"],
            languages=["zh-CN"], is_builtin=True,
        ),
        VoiceModelMeta(
            id="minimax_congming", name="聪明女孩 (机智灵动)", engine="minimax",
            engine_config={"voice_id": "female_smartgirl"},
            emotions=["happy", "curious", "excited", "playful", "neutral"],
            languages=["zh-CN"], is_builtin=True,
        ),
    ]

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        voice_id: str = "",
        base_url: str = "",
        **kwargs,
    ):
        settings = get_settings()
        self._api_key = api_key or settings.minimax_api_key or settings.openai_api_key
        self._model = model or settings.minimax_tts_model
        self._default_voice_id = voice_id or settings.minimax_tts_voice_id
        self._base_url = (base_url or settings.minimax_base_url).rstrip("/")
        self._voice_map = {v.id: v for v in self.BUILTIN_VOICES}

    @property
    def provider_name(self) -> str:
        return "minimax_tts"

    @property
    def display_name(self) -> str:
        return f"MiniMax TTS ({self._model})"

    async def synthesize(
        self,
        text: str,
        voice: VoiceModelMeta,
        emotion: str = "neutral",
        emotion_intensity: float = 0.5,
        speed: float = 1.0,
    ) -> AsyncIterator[TTSChunk]:
        """Call MiniMax T2A API and stream PCM chunks with RMS."""
        voice_id = voice.engine_config.get("voice_id", self._default_voice_id)

        # Blend emotion-driven speed with explicit speed parameter
        base_speed = self.EMOTION_SPEED.get(emotion, 1.0)
        effective_speed = base_speed * 0.6 + speed * 0.4
        effective_speed = max(0.5, min(2.0, effective_speed))

        # Pitch from emotion + intensity
        base_pitch = self.EMOTION_PITCH.get(emotion, 0)
        effective_pitch = int(base_pitch + (emotion_intensity - 0.5) * 4)
        effective_pitch = max(-12, min(12, effective_pitch))

        url = f"{self._base_url}/t2a_v2"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model,
            "text": text,
            "stream": False,
            "output_format": "hex",
            "voice_setting": {
                "voice_id": voice_id,
                "speed": effective_speed,
                "vol": 1.0,
                "pitch": effective_pitch,
            },
            "audio_setting": {
                "sample_rate": 16000,
                "bitrate": 64000,
                "format": "pcm",
                "channel": 1,
            },
            "language_boost": "auto",
        }

        logger.debug(
            f"MiniMax TTS: text='{text[:40]}...', voice={voice_id}, "
            f"emotion={emotion}, speed={effective_speed:.2f}, pitch={effective_pitch}"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        base_resp = data.get("base_resp", {})
        if base_resp.get("status_code") != 0:
            error_msg = base_resp.get("status_msg", "Unknown error")
            logger.error(f"MiniMax TTS error: {error_msg}")
            yield TTSChunk(audio=b"", rms=0.0, is_final=True)
            return

        # Decode hex audio to PCM bytes
        hex_audio = data.get("data", {}).get("audio", "")
        if not hex_audio:
            logger.warning("MiniMax TTS returned empty audio")
            yield TTSChunk(audio=b"", rms=0.0, is_final=True)
            return

        audio_bytes = bytes.fromhex(hex_audio)

        # Stream in ~20ms chunks (640 bytes at 16kHz 16bit mono)
        chunk_size = 640
        total = len(audio_bytes)
        for offset in range(0, total, chunk_size):
            chunk = audio_bytes[offset:offset + chunk_size]
            rms = self._compute_rms(chunk)
            is_final = (offset + chunk_size >= total)
            yield TTSChunk(audio=chunk, rms=rms, is_final=is_final)

        # If audio length was exactly multiple of chunk_size, send final marker
        if total % chunk_size == 0 and total > 0:
            yield TTSChunk(audio=b"", rms=0.0, is_final=True)

    @staticmethod
    def _compute_rms(samples: bytes) -> float:
        """Compute RMS of PCM 16bit mono samples."""
        if len(samples) < 2:
            return 0.0
        arr = np.frombuffer(samples, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(arr ** 2))
        return float(rms / 20000.0)

    async def list_voices(self) -> list[VoiceModelMeta]:
        return self.BUILTIN_VOICES

    async def validate_config(self) -> bool:
        if not self._api_key:
            logger.error("MiniMax API key not set (MINIMAX_API_KEY or OPENAI_API_KEY)")
            return False
        return True
