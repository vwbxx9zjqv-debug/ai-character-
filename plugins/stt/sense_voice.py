"""
SenseVoice STT Provider — Offline speech recognition via sherpa-onnx + SenseVoice.

Fully offline: no API key, no network, no cloud. Optimized for Chinese (zh/en/ja/ko/yue).
Uses the int8-quantized ONNX model (~229MB) with SenseVoice's CTC decoder.

Dependencies: pip install sherpa-onnx numpy
Model: sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
Reference: https://k2-fsa.github.io/sherpa/onnx/sense-voice/index.html
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from core.plugin_base import STTProvider, STTResult

logger = logging.getLogger("stt.sense_voice")

# Default model directory relative to this file
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = BACKEND_DIR / "models" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"


class SenseVoiceProvider(STTProvider):
    """Offline STT via sherpa-onnx SenseVoice.

    Uses SenseVoice ONNX model with CTC greedy search decoding.
    Supports Chinese, English, Japanese, Korean, and Cantonese.
    Runs on CPU by default — fast enough for real-time on any modern machine.
    """

    config_schema = {
        "model_dir": {
            "type": "string",
            "default": str(DEFAULT_MODEL_DIR),
            "description": "Path to extracted SenseVoice model directory",
        },
        "model_file": {
            "type": "string",
            "default": "model.int8.onnx",
            "description": "Model filename: model.int8.onnx (229MB, fast) or model.onnx (895MB)",
        },
        "language": {
            "type": "string",
            "default": "zh",
            "description": "Recognition language: zh / en / ja / ko / yue / auto",
        },
        "use_itn": {
            "type": "boolean",
            "default": True,
            "description": "Inverse Text Normalization (numbers, dates, etc.)",
        },
        "num_threads": {
            "type": "integer",
            "default": 4,
            "description": "Number of CPU threads for inference",
        },
    }

    def __init__(
        self,
        model_dir: str = str(DEFAULT_MODEL_DIR),
        model_file: str = "model.int8.onnx",
        language: str = "zh",
        use_itn: bool = True,
        num_threads: int = 4,
        **kwargs,
    ):
        self._model_dir = Path(model_dir)
        self._model_file = model_file
        self._language = language
        self._use_itn = use_itn
        self._num_threads = num_threads
        self._recognizer = None  # Lazy-loaded on first use

    @property
    def provider_name(self) -> str:
        return "sense_voice"

    @property
    def display_name(self) -> str:
        return f"SenseVoice (sherpa-onnx, {self._language}, itn={self._use_itn})"

    @property
    def _model_path(self) -> Path:
        return self._model_dir / self._model_file

    @property
    def _tokens_path(self) -> Path:
        return self._model_dir / "tokens.txt"

    async def transcribe(self, audio: bytes, audio_format: str = "opus") -> STTResult:
        """Transcribe audio using SenseVoice ONNX model.

        Accepts raw PCM 16kHz 16bit mono, or opus (decoded via ffmpeg).
        """
        start = time.monotonic()

        # Lazy-load model on first call
        if self._recognizer is None:
            await self._load_model()

        # Decode opus → PCM if needed
        if audio_format == "opus":
            audio = await self._decode_opus(audio)

        # Convert PCM bytes → numpy float32 array (normalized to [-1, 1])
        import numpy as np
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        if len(samples) == 0:
            logger.warning("Empty audio buffer received")
            return STTResult(text="", language=self._language, confidence=0.0, duration_ms=0)

        # Run transcription in a thread (sherpa-onnx is synchronous)
        loop = asyncio.get_running_loop()
        result_text = await loop.run_in_executor(
            None,
            self._run_transcribe,
            samples,
        )

        duration_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            f"SenseVoice: '{result_text[:60]}...' "
            f"(lang={self._language}, {duration_ms}ms)"
        )

        return STTResult(
            text=result_text.strip(),
            language=self._language,
            confidence=0.95,  # SenseVoice doesn't expose per-frame confidence
            duration_ms=duration_ms,
        )

    def _run_transcribe(self, samples: "np.ndarray") -> str:
        """Run sherpa-onnx transcription on the given PCM samples. Runs in thread executor."""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        self._recognizer.decode_stream(stream)
        return stream.result.text

    async def validate_config(self) -> bool:
        """Check that sherpa-onnx is installed and model files exist."""
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            logger.warning("sherpa-onnx not installed. Run: pip install sherpa-onnx")
            return False

        if not self._model_path.exists():
            logger.warning(f"SenseVoice model not found: {self._model_path}")
            return False

        if not self._tokens_path.exists():
            logger.warning(f"SenseVoice tokens not found: {self._tokens_path}")
            return False

        return True

    async def _load_model(self) -> None:
        """Load SenseVoice model via sherpa-onnx OfflineRecognizer."""
        import sherpa_onnx

        logger.info(
            f"Loading SenseVoice model: {self._model_path} "
            f"(language={self._language}, itn={self._use_itn}, threads={self._num_threads})"
        )

        loop = asyncio.get_running_loop()

        self._recognizer = await loop.run_in_executor(
            None,
            lambda: sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(self._model_path),
                tokens=str(self._tokens_path),
                num_threads=self._num_threads,
                use_itn=self._use_itn,
                language=self._language,
            ),
        )

        logger.info("SenseVoice model loaded successfully")

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
