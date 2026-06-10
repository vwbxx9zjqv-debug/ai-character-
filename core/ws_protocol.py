"""
WebSocket Protocol v3 — Communication contract between ESP32 and backend.

All messages carry a version field for forward compatibility.
Binary audio frames are preceded by a JSON header frame.

Protocol evolution:
  v1: Basic text chat + audio
  v2: Added emotion, RMS lip-sync, model switching
  v3: Added character performance instructions, initiative, BT audio toggle
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


PROTOCOL_VERSION = 3


# ── Message Types ─────────────────────────────────────────────────

class MessageType(StrEnum):
    """All valid WebSocket message types."""

    # ESP32 → Backend
    SPEECH_SEGMENT = "speech_segment"     # VAD-cut audio segment (binary follows)
    HEARTBEAT = "heartbeat"              # Periodic ping with device state
    LIST_MODELS = "list_models"          # Query available characters/voices
    SWITCH_REQUEST = "switch_request"    # Request model switch (voice cmd)

    # Backend → ESP32
    STATE = "state"                      # Device state change (drives screen animation)
    SPEECH = "speech"                    # TTS audio reply (binary follows)
    SUBTITLE = "subtitle"               # Text subtitle
    MODELS_LIST = "models_list"         # Response to list_models
    SWITCH_MODEL = "switch_model"       # Trigger model switch on device
    DOWNLOAD_SPRITE = "download_sprite" # URL to download sprite pack
    CHARACTER_SYNC = "character_sync"   # Full character state sync
    INITIATIVE = "initiative"           # Character-initiated speech (idle trigger)
    ERROR = "error"                     # Error message

    # Bidirectional
    PING = "ping"
    PONG = "pong"


class DeviceState(StrEnum):
    """States the ESP32 device can be in (drives animation state machine)."""
    IDLE = "idle"                # Sleeping / waiting
    LISTENING = "listening"      # User is speaking (VAD active)
    THINKING = "thinking"        # Processing (STT → LLM → TTS)
    SPEAKING = "speaking"        # Playing TTS audio
    ERROR = "error"              # Something went wrong


# ── Message Data Classes ──────────────────────────────────────────

@dataclass
class WSMessage:
    """Base WebSocket message (JSON frame)."""
    v: int = PROTOCOL_VERSION
    type: str = ""
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(self._to_dict(), ensure_ascii=False)

    def _to_dict(self) -> dict:
        result = {"v": self.v, "type": self.type, "ts": self.ts}
        for key, value in self.__dict__.items():
            if key not in ("v", "type", "ts") and value is not None:
                result[key] = value
        return result


# ── ESP32 → Backend Messages ──────────────────────────────────────

@dataclass
class SpeechSegmentMessage(WSMessage):
    """ESP32 sends a VAD-cut audio segment."""
    type: str = MessageType.SPEECH_SEGMENT
    duration_ms: int = 0
    vad_confidence: float = 0.0
    audio_format: str = "opus"


@dataclass
class HeartbeatMessage(WSMessage):
    """ESP32 periodic heartbeat with device state."""
    type: str = MessageType.HEARTBEAT
    state: str = "idle"
    battery: int | None = None        # Battery percentage (if portable)
    wifi_rssi: int = 0                # WiFi signal strength
    current_pet_id: str = ""
    current_voice_id: str = ""


@dataclass
class ListModelsMessage(WSMessage):
    """ESP32 requests available models."""
    type: str = MessageType.LIST_MODELS
    model_type: str = "pet"            # "pet" | "voice"


@dataclass
class SwitchRequestMessage(WSMessage):
    """ESP32 requests model switch (from voice command)."""
    type: str = MessageType.SWITCH_REQUEST
    model_type: str = "pet"            # "pet" | "voice"
    model_id: str = ""


# ── Backend → ESP32 Messages ──────────────────────────────────────

@dataclass
class StateMessage(WSMessage):
    """Backend tells ESP32 to change device state (animation trigger)."""
    type: str = MessageType.STATE
    state: str = "idle"                # DeviceState value
    emotion: str | None = None         # Current emotion for expression
    text: str | None = None            # Optional subtitle


@dataclass
class SpeechMessage(WSMessage):
    """Backend sends TTS audio reply (header before binary audio frames)."""
    type: str = MessageType.SPEECH
    text: str = ""                     # Full response text (subtitle)
    emotion: str = "neutral"
    emotion_intensity: float = 0.5
    voice_name: str = ""
    character_name: str = ""
    rms_frames: list[float] = field(default_factory=list)  # Pre-computed RMS for lip-sync
    audio_chunks: int = 0              # Number of binary audio frames following


@dataclass
class CharacterSyncMessage(WSMessage):
    """Full character state sync (on connect or switch)."""
    type: str = MessageType.CHARACTER_SYNC
    character: dict = field(default_factory=dict)
    # character: {id, name, current_mood, current_outfit, background, persona_summary}
    sprites_url: str = ""
    voice_id: str = ""


@dataclass
class SwitchModelMessage(WSMessage):
    """Backend instructs ESP32 to switch model."""
    type: str = MessageType.SWITCH_MODEL
    model_type: str = "pet"            # "pet" | "voice"
    model_id: str = ""
    model_name: str = ""
    sprite_manifest: dict | None = None  # Animation metadata for character
    download_url: str | None = None      # URL to download sprite pack


@dataclass
class InitiativeMessage(WSMessage):
    """Character proactively speaks (idle-triggered)."""
    type: str = MessageType.INITIATIVE
    text: str = ""
    emotion: str = "neutral"
    performance: dict = field(default_factory=dict)
    # performance: {expression, gesture, eye_movement, special_effect, background_change}


@dataclass
class ErrorMessage(WSMessage):
    type: str = MessageType.ERROR
    code: str = ""
    message: str = ""


# ── Binary Frame Format ───────────────────────────────────────────

# Binary audio frames follow a JSON header:
#
#   JSON Header:  { "v": 3, "type": "speech_segment", "duration_ms": 3200, ... }
#   Binary Frame: [4-byte LE length][Opus/PCM audio data]
#
# For multi-chunk TTS replies:
#   { "v": 3, "type": "speech", "audio_chunks": 45, ... }
#   [4-byte LE length][PCM chunk 0]
#   [4-byte LE length][PCM chunk 1]
#   ...
#   [4-byte LE length=0]  ← terminator

BINARY_FRAME_HEADER_SIZE = 4  # uint32 little-endian length prefix


def encode_audio_frame(audio_data: bytes) -> bytes:
    """Wrap audio data in a length-prefixed binary frame."""
    return struct.pack("<I", len(audio_data)) + audio_data


def decode_audio_frames(data: bytes) -> list[bytes]:
    """Decode a stream of length-prefixed audio frames."""
    frames = []
    offset = 0
    while offset + BINARY_FRAME_HEADER_SIZE <= len(data):
        length = struct.unpack("<I", data[offset:offset + BINARY_FRAME_HEADER_SIZE])[0]
        offset += BINARY_FRAME_HEADER_SIZE
        if length == 0:
            break  # Terminator
        frames.append(data[offset:offset + length])
        offset += length
    return frames


# ── Message Parsing ───────────────────────────────────────────────

MESSAGE_CLASSES: dict[str, type[WSMessage]] = {
    MessageType.SPEECH_SEGMENT: SpeechSegmentMessage,
    MessageType.HEARTBEAT: HeartbeatMessage,
    MessageType.LIST_MODELS: ListModelsMessage,
    MessageType.SWITCH_REQUEST: SwitchRequestMessage,
    MessageType.STATE: StateMessage,
    MessageType.SPEECH: SpeechMessage,
    MessageType.CHARACTER_SYNC: CharacterSyncMessage,
    MessageType.SWITCH_MODEL: SwitchModelMessage,
    MessageType.INITIATIVE: InitiativeMessage,
    MessageType.ERROR: ErrorMessage,
}


def parse_message(raw: str | bytes) -> WSMessage:
    """Parse a JSON string into the appropriate WSMessage subclass."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    msg_type = data.get("type", "")
    cls = MESSAGE_CLASSES.get(msg_type, WSMessage)
    return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})  # type: ignore
