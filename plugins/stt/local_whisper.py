"""
Local Whisper STT Provider — Offline speech recognition via faster-whisper (CTranslate2).

Runs entirely on-device: no API key, no network, no cloud.
Hardware: CPU (int8) or GPU (CUDA). Model sizes: tiny / base / small / medium / large-v3.

Dependencies: pip install faster-whisper
Reference: https://github.com/SYSTRAN/faster-whisper
"""

from __future__ import annotations

import logging
import time
from io import BytesIO

from core.plugin_base import STTProvider, STTResult

logger = logging.getLogger("stt.local_whisper")


class LocalWhisperProvider(STTProvider):
    """Offline STT via faster-whisper (CTranslate2 Whisper).

    Models are auto-downloaded on first use (~150MB for base, ~500MB for small).
    Runs on CPU with int8 quantization by default — fast enough for real-time on
    any modern laptop or Raspberry Pi 5.
    """

    config_schema = {
        "model_size": {
            "type": "string",
            "default": "base",
            "description": "Whisper model size: tiny / base / small / medium / large-v3",
        },
        "language": {
            "type": "string",
            "default": "zh",
            "description": "Default recognition language (set to '' for auto-detect)",
        },
        "device": {
            "type": "string",
            "default": "cpu",
            "description": "Device: cpu or cuda",
        },
        "compute_type": {
            "type": "string",
            "default": "int8",
            "description": "Quantization: int8 / int8_float16 / float16 / float32",
        },
    }

    def __init__(
        self,
        model_size: str = "base",
        language: str = "zh",
        device: str = "cpu",
        compute_type: str = "int8",
        **kwargs,
    ):
        self._model_size = model_size
        self._language = language
        self._device = device
        self._compute_type = compute_type
        self._model = None  # Lazy-loaded on first use

    @property
    def provider_name(self) -> str:
        return "local_whisper"

    @property
    def display_name(self) -> str:
        return f"本地 Whisper ({self._model_size}, {self._device})"

    async def transcribe(self, audio: bytes, audio_format: str = "opus") -> STTResult:
        """Transcribe audio using local Whisper model.

        Accepts raw PCM 16kHz 16bit mono, or opus (will be decoded if ffmpeg available).
        """
        start = time.monotonic()

        # Lazy-load model on first call
        if self._model is None:
            await self._load_model()

        # faster-whisper expects raw PCM 16kHz mono as numpy array
        import numpy as np

        if audio_format == "opus":
            # Try to decode opus → PCM via ffmpeg
            audio = await self._decode_opus(audio)

        # Convert PCM bytes → numpy float32 array (normalized to [-1, 1])
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Run transcription in a thread (faster-whisper is synchronous)
        import asyncio
        loop = asyncio.get_running_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(
                samples,
                language=self._language or None,
                beam_size=5,
                vad_filter=True,           # Filter out silence
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                ),
            ),
        )

        text = " ".join(seg.text.strip() for seg in segments)
        duration_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            f"Local Whisper: '{text[:60]}...' "
            f"(lang={info.language}, prob={info.language_probability:.2f}, {duration_ms}ms)"
        )

        return STTResult(
            text=text.strip(),
            language=info.language or self._language,
            confidence=info.language_probability,
            duration_ms=duration_ms,
        )

    async def validate_config(self) -> bool:
        """Check that faster-whisper is installed."""
        try:
            import faster_whisper
            return True
        except ImportError:
            logger.warning("faster-whisper not installed. Run: pip install faster-whisper")
            return False

    async def _load_model(self) -> None:
        """Load Whisper model (downloads on first use)."""
        import asyncio
        from faster_whisper import WhisperModel

        logger.info(f"Loading local Whisper model: {self._model_size} on {self._device}...")
        loop = asyncio.get_running_loop()

        self._model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            ),
        )
        logger.info(f"Local Whisper model loaded: {self._model_size}")

    async def _decode_opus(self, opus_data: bytes) -> bytes:
        """Decode opus audio to raw PCM 16kHz 16bit mono via ffmpeg subprocess."""
        import asyncio
        import subprocess

        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-i", "pipe:0",
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate(input=opus_data)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg exited with {proc.returncode}")
            return stdout
        except FileNotFoundError:
            logger.warning("ffmpeg not found, assuming raw PCM input")
            return opus_data
