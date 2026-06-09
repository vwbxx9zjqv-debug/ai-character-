"""
Edge-TTS Provider — Free Microsoft Edge TTS.

Reference: Open-LLM-VTuber src/open_llm_vtuber/tts/edge_tts.py
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
from typing import AsyncIterator

import numpy as np

from core.plugin_base import TTSProvider, TTSChunk, VoiceModelMeta
from config import get_settings

logger = logging.getLogger("tts.edge")


class EdgeTTSProvider(TTSProvider):
    """Free TTS using Microsoft Edge's built-in voices."""

    # Pre-defined emotion voice mappings
    # Edge-TTS doesn't natively support emotion, but different voices
    # have different styles. We'll adjust speed as a proxy.
    EMOTION_SPEED = {
        "happy": 1.1,
        "gentle": 0.85,
        "sad": 0.75,
        "excited": 1.3,
        "neutral": 1.0,
        "comforting": 0.8,
        "curious": 1.05,
        "shy": 0.9,
        "playful": 1.15,
    }

    BUILTIN_VOICES = [
        VoiceModelMeta(
            id="edge_xiaoxiao", name="小晓 (活泼少女)", engine="edge",
            engine_config={"voice": "zh-CN-XiaoxiaoNeural"},
            emotions=["happy", "gentle", "excited", "neutral", "shy", "curious"],
            languages=["zh-CN"], is_builtin=True,
        ),
        VoiceModelMeta(
            id="edge_yunxi", name="云希 (元气少年)", engine="edge",
            engine_config={"voice": "zh-CN-YunxiNeural"},
            emotions=["happy", "excited", "neutral", "comforting", "playful"],
            languages=["zh-CN"], is_builtin=True,
        ),
        VoiceModelMeta(
            id="edge_xiaoyi", name="小依 (温柔知性)", engine="edge",
            engine_config={"voice": "zh-CN-XiaoyiNeural"},
            emotions=["gentle", "neutral", "sad", "comforting", "shy"],
            languages=["zh-CN"], is_builtin=True,
        ),
        VoiceModelMeta(
            id="edge_yunjian", name="云健 (成熟男声)", engine="edge",
            engine_config={"voice": "zh-CN-YunjianNeural"},
            emotions=["neutral", "gentle", "comforting", "sad"],
            languages=["zh-CN"], is_builtin=True,
        ),
        VoiceModelMeta(
            id="edge_xiaochen", name="小晨 (治愈软萌)", engine="edge",
            engine_config={"voice": "zh-CN-XiaochenNeural"},
            emotions=["gentle", "happy", "shy", "comforting"],
            languages=["zh-CN"], is_builtin=True,
        ),
    ]

    def __init__(self, **kwargs):
        self._voice_map = {v.id: v for v in self.BUILTIN_VOICES}

    @property
    def provider_name(self) -> str:
        return "edge_tts"

    @property
    def display_name(self) -> str:
        return "Microsoft Edge TTS (Free)"

    async def synthesize(
        self,
        text: str,
        voice: VoiceModelMeta,
        emotion: str = "neutral",
        emotion_intensity: float = 0.5,
        speed: float = 1.0,
    ) -> AsyncIterator[TTSChunk]:
        """Stream audio from Edge-TTS, computing RMS per chunk."""
        import edge_tts

        voice_id = voice.engine_config.get("voice", "zh-CN-XiaoxiaoNeural")

        # Adjust rate based on emotion
        base_speed = self.EMOTION_SPEED.get(emotion, 1.0)
        # Blend: 70% emotion speed + 30% explicit speed
        effective_speed = base_speed * 0.7 + speed * 0.3

        rate_str = f"{int((effective_speed - 1.0) * 100):+d}%"
        # Intensity affects pitch slightly
        pitch_offset = int((emotion_intensity - 0.5) * 20)
        pitch_str = f"{pitch_offset:+d}Hz"

        logger.debug(
            f"Edge-TTS: text='{text[:30]}...', voice={voice_id}, "
            f"emotion={emotion}, rate={rate_str}, pitch={pitch_str}"
        )

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice_id,
            rate=rate_str,
            pitch=pitch_str,
        )

        # Stream audio chunks and compute RMS
        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
                # Yield chunks of ~20ms (640 bytes at 16kHz 16bit mono)
                audio_buffer.seek(0)
                data = audio_buffer.read()
                if len(data) >= 640:
                    rms = self._compute_rms(data[:640])
                    yield TTSChunk(audio=data[:640], rms=rms, is_final=False)
                    audio_buffer = io.BytesIO(data[640:])
                audio_buffer.seek(0, io.SEEK_END)

        # Flush remaining audio
        audio_buffer.seek(0)
        remaining = audio_buffer.read()
        if remaining:
            rms = self._compute_rms(remaining)
            yield TTSChunk(audio=remaining, rms=rms, is_final=True)
        else:
            # Signal end
            yield TTSChunk(audio=b"", rms=0.0, is_final=True)

    @staticmethod
    def _compute_rms(samples: bytes) -> float:
        """Compute RMS of PCM 16bit mono samples."""
        if len(samples) < 2:
            return 0.0
        arr = np.frombuffer(samples, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(arr ** 2))
        # Normalize to ~0-1 range (32768 is max for int16)
        return float(rms / 20000.0)  # Clamp typical speech to ~0-1

    async def list_voices(self) -> list[VoiceModelMeta]:
        return self.BUILTIN_VOICES

    async def validate_config(self) -> bool:
        try:
            import edge_tts
            return True
        except ImportError:
            logger.error("edge-tts not installed. Run: pip install edge-tts")
            return False
