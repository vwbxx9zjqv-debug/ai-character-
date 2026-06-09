"""
Event Bus — Central nervous system of the virtual character companion.

All dialogue lifecycle events flow through here. Plugins subscribe to events
they care about. New features (Phase 2: Agent, Music, Smart Home) just add
new subscribers — zero changes to existing code.

Usage:
    bus = EventBus()

    @bus.on(SpeechEnded)
    async def handle_speech(event: SpeechEnded):
        ...

    await bus.emit(SpeechEnded(device_id="esp32-01", audio_segment=...))
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Type, TypeVar

# ── Event Definitions ──────────────────────────────────────────────

E = TypeVar("E", bound="PetEvent")


@dataclass(kw_only=True)
class PetEvent:
    """Base class for all events."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    device_id: str
    timestamp: float = field(default_factory=time.time)


# ── Dialogue lifecycle events ──

@dataclass(kw_only=True)
class SpeechStarted(PetEvent):
    """VAD detected user started speaking."""
    ambient_noise_level: float = 0.0


@dataclass(kw_only=True)
class SpeechEnded(PetEvent):
    """User finished speaking, audio segment ready."""
    audio_segment: bytes
    audio_format: str = "opus"        # "opus" | "pcm"
    duration_ms: int = 0
    vad_confidence: float = 0.0


@dataclass(kw_only=True)
class TranscriptionReady(PetEvent):
    """STT completed, text ready."""
    text: str
    language: str = "zh"
    confidence: float = 0.0
    duration_ms: int = 0


@dataclass(kw_only=True)
class LLMResponseReady(PetEvent):
    """LLM generated a response."""
    text: str
    emotion: str = "neutral"           # happy/gentle/sad/excited/neutral/...
    emotion_intensity: float = 0.5     # 0.0 - 1.0
    tool_calls: list[dict] | None = None  # Phase 2: Agent tool calls
    tokens_used: int = 0


@dataclass(kw_only=True)
class TTSStarted(PetEvent):
    """TTS began generating audio."""
    voice_id: str
    voice_name: str = ""
    emotion: str = "neutral"


@dataclass(kw_only=True)
class TTSChunkReady(PetEvent):
    """A chunk of TTS audio is ready."""
    audio_chunk: bytes
    rms: float = 0.0
    is_final: bool = False


@dataclass(kw_only=True)
class DialogueCompleted(PetEvent):
    """Full dialogue round trip completed."""
    user_text: str
    assistant_text: str
    emotion: str = "neutral"
    total_latency_ms: int = 0
    stt_latency_ms: int = 0
    llm_latency_ms: int = 0
    tts_latency_ms: int = 0


# ── Character & System events ──

@dataclass(kw_only=True)
class CharacterSwitched(PetEvent):
    """Character model was changed."""
    previous_character_id: str
    new_character_id: str
    new_character_name: str


@dataclass(kw_only=True)
class VoiceSwitched(PetEvent):
    """Voice model was changed."""
    previous_voice_id: str
    new_voice_id: str
    new_voice_name: str


@dataclass(kw_only=True)
class MoodChanged(PetEvent):
    """Character's internal mood evolved."""
    previous_mood: str
    new_mood: str
    reason: str = ""


@dataclass(kw_only=True)
class CharacterInitiative(PetEvent):
    """Character decided to speak proactively (idle-triggered)."""
    text: str
    emotion: str = "neutral"
    trigger: str = ""                 # "idle_300s" | "time_of_day" | "user_returned"


# ── Phase 2 events (defined now, used later) ──

@dataclass(kw_only=True)
class MusicPlaybackStarted(PetEvent):
    """Music started playing (Phase 2)."""
    song_title: str
    source: str = "spotify"


@dataclass(kw_only=True)
class AgentTaskStarted(PetEvent):
    """Claude Agent began a task (Phase 2)."""
    task_description: str
    tool_calls: list[dict]


@dataclass(kw_only=True)
class AgentTaskCompleted(PetEvent):
    """Claude Agent completed a task (Phase 2)."""
    task_description: str
    result: str
    success: bool = True


# ── Event Bus Implementation ──────────────────────────────────────

EventHandler = Callable[[Any], Coroutine[Any, Any, None]]


class EventBus:
    """
    Simple, fast, in-process event bus.

    Features:
    - Type-based subscription (subscribe to SpeechEnded, not string "speech.ended")
    - Async handlers run concurrently via asyncio.gather
    - Error isolation: one handler failing doesn't crash others
    - Wildcard subscription: subscribe to all PetEvent subclasses
    """

    def __init__(self):
        self._handlers: dict[Type[PetEvent], list[EventHandler]] = defaultdict(list)
        self._wildcard_handlers: list[EventHandler] = []
        self._event_count: int = 0

    def on(self, event_type: Type[E]):
        """Decorator: register a handler for a specific event type.

        Usage:
            @bus.on(SpeechEnded)
            async def handle(event: SpeechEnded): ...
        """
        def decorator(handler: EventHandler) -> EventHandler:
            self._handlers[event_type].append(handler)
            return handler
        return decorator

    def on_any(self, handler: EventHandler) -> EventHandler:
        """Register a handler that receives ALL events (useful for logging/debug)."""
        self._wildcard_handlers.append(handler)
        return handler

    def subscribe(self, event_type: Type[E], handler: EventHandler) -> None:
        """Programmatic subscription."""
        self._handlers[event_type].append(handler)

    async def emit(self, event: PetEvent) -> None:
        """Emit an event. All matching handlers run concurrently."""
        self._event_count += 1

        # Collect handlers: type-specific + wildcard
        handlers: list[EventHandler] = []
        handlers.extend(self._wildcard_handlers)

        # Match exact type AND parent types (so SpeechEnded also triggers PetEvent handlers)
        for event_type, type_handlers in self._handlers.items():
            if isinstance(event, event_type):
                handlers.extend(type_handlers)

        if not handlers:
            return

        # Run all handlers concurrently, with error isolation
        results = await asyncio.gather(
            *[self._safe_call(h, event) for h in handlers],
            return_exceptions=True
        )

        # Log errors (don't crash the bus)
        for result in results:
            if isinstance(result, Exception):
                import logging
                logging.getLogger("event_bus").error(
                    f"Handler error for {type(event).__name__}: {result}"
                )

    @staticmethod
    async def _safe_call(handler: EventHandler, event: PetEvent) -> None:
        """Call handler with proper error handling for non-coroutine results."""
        result = handler(event)
        if asyncio.iscoroutine(result):
            await result

    @property
    def event_count(self) -> int:
        return self._event_count

    def clear(self) -> None:
        """Remove all handlers (useful for testing)."""
        self._handlers.clear()
        self._wildcard_handlers.clear()


# ── Global singleton ──────────────────────────────────────────────

_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus singleton."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
