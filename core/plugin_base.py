"""
Plugin Base Interfaces — Abstract contracts for all pluggable components.

Every STT, LLM, TTS, and capability module implements these interfaces.
New engines = new files in plugins/, zero changes to core.

Design inspired by Open-LLM-VTuber's modular provider architecture
(ASRInterface / TTSInterface / StatelessLLMInterface pattern).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable


# ── STT (Speech-to-Text) Interface ────────────────────────────────

@dataclass
class STTResult:
    text: str
    language: str = "zh"
    confidence: float = 0.0
    duration_ms: int = 0


class STTProvider(ABC):
    """Speech-to-Text plugin interface.

    Implementations: WhisperAPI, FasterWhisper, FunASR, SherpaONNX, etc.
    Reference: Open-LLM-VTuber src/open_llm_vtuber/asr/asr_interface.py
    """

    # Each provider declares its config schema for the admin panel
    config_schema: dict[str, Any] = {}

    @abstractmethod
    async def transcribe(self, audio: bytes, audio_format: str = "opus") -> STTResult:
        """Convert audio bytes to text."""
        ...

    @abstractmethod
    async def validate_config(self) -> bool:
        """Check if provider is properly configured (API keys, model files, etc.)."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique identifier, e.g. 'whisper_api', 'faster_whisper'."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name, e.g. 'OpenAI Whisper'."""
        ...


# ── LLM Interface ─────────────────────────────────────────────────

@dataclass
class Message:
    role: str                      # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResult:
    text: str
    emotion: str = "neutral"
    emotion_intensity: float = 0.5
    tool_calls: list[dict[str, Any]] | None = None
    tokens_used: int = 0
    model: str = ""


class LLMProvider(ABC):
    """Language Model plugin interface.

    Implementations: ClaudeAPI, OpenAIAPI, Ollama, DeepSeek, etc.
    Reference: Open-LLM-VTuber src/open_llm_vtuber/agent/stateless_llm/
    """

    config_schema: dict[str, Any] = {}

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        system_prompt: str = "",
        emotion_hint: str | None = None,
        temperature: float = 0.8,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> LLMResult:
        """Send messages to LLM and get response with emotion tag."""
        ...

    @abstractmethod
    async def validate_config(self) -> bool:
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        ...


# ── TTS Interface ─────────────────────────────────────────────────

@dataclass
class TTSChunk:
    audio: bytes                    # PCM 16bit 16kHz mono
    rms: float = 0.0               # Current frame RMS for lip-sync
    is_final: bool = False


@dataclass
class VoiceModelMeta:
    """Metadata for a voice model (built-in or user-uploaded)."""
    id: str
    name: str                       # User-given name, e.g. "妈妈的声音"
    engine: str                     # 'cosyvoice' | 'gpt_sovits' | 'edge' | 'azure' | 'custom'
    engine_config: dict[str, Any] = field(default_factory=dict)
    emotions: list[str] = field(default_factory=lambda: ["neutral"])
    languages: list[str] = field(default_factory=lambda: ["zh-CN"])
    sample_url: str = ""
    is_builtin: bool = False


class TTSProvider(ABC):
    """Text-to-Speech plugin interface.

    Emotion is a FIRST-CLASS parameter — all TTS engines must support it.
    Implementations: CosyVoice2, GPT-SoVITS, EdgeTTS, AzureTTS, CustomEndpoint.

    Reference: Open-LLM-VTuber src/open_llm_vtuber/tts/tts_interface.py
    """

    config_schema: dict[str, Any] = {}

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: VoiceModelMeta,
        emotion: str = "neutral",
        emotion_intensity: float = 0.5,
        speed: float = 1.0,
    ) -> AsyncIterator[TTSChunk]:
        """Stream audio chunks. Each chunk carries RMS for lip-sync."""
        ...

    @abstractmethod
    async def validate_config(self) -> bool:
        ...

    @abstractmethod
    async def list_voices(self) -> list[VoiceModelMeta]:
        """List available voices for this engine."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        ...


# ── Capability Plugin Interface ────────────────────────────────────

@runtime_checkable
class CapabilityPlugin(Protocol):
    """Phase 2 capability plugins (Agent, Music, Smart Home, etc.).

    These subscribe to events on the event bus. They don't need to
    implement a specific method — they just need to wire themselves up.

    A capability plugin is any object that:
    1. Takes the EventBus in __init__
    2. Subscribes to relevant events
    3. Can be enabled/disabled
    """

    name: str
    is_enabled: bool

    async def on_enable(self) -> None: ...
    async def on_disable(self) -> None: ...


# ── Convenience: All provider types ────────────────────────────────

ProviderType = STTProvider | LLMProvider | TTSProvider | CapabilityPlugin
