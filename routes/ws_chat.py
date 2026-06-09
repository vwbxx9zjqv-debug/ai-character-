"""
WebSocket Chat Route — The real-time communication channel between ESP32 and backend.

Protocol v3: JSON headers + binary audio frames.
Handles: speech segments, heartbeats, model switching, state management.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.ws_protocol import (
    PROTOCOL_VERSION,
    MessageType, DeviceState,
    SpeechMessage, StateMessage, CharacterSyncMessage,
    SwitchModelMessage, ErrorMessage,
    encode_audio_frame, parse_message,
    SpeechSegmentMessage, HeartbeatMessage,
)
from core.plugin_base import Message
from services.dialogue_service import get_dialogue_service
from db.database import AsyncSessionLocal
from db import queries

logger = logging.getLogger("ws_chat")

router = APIRouter()

# Track connected devices
_connections: dict[str, WebSocket] = {}
_connection_states: dict[str, str] = {}  # device_id → current state


@router.websocket("/ws/chat/{device_id}")
async def ws_chat(websocket: WebSocket, device_id: str):
    """Main WebSocket endpoint for ESP32 devices."""
    await websocket.accept()
    _connections[device_id] = websocket
    _connection_states[device_id] = DeviceState.IDLE

    dialogue = get_dialogue_service()

    logger.info(f"Device connected: {device_id}")

    # Send character sync on connect
    await _send_character_sync(websocket, device_id)

    try:
        while True:
            # Receive message (text JSON or binary)
            raw = await websocket.receive()

            if "text" in raw:
                # JSON message
                msg = parse_message(raw["text"])
                await _handle_json_message(websocket, device_id, msg, dialogue)

            elif "bytes" in raw:
                # Binary audio frame
                await _handle_binary_audio(websocket, device_id, raw["bytes"], dialogue)

    except WebSocketDisconnect:
        logger.info(f"Device disconnected: {device_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {device_id}: {e}")
        try:
            await websocket.send_json({"v": PROTOCOL_VERSION, "type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        _connections.pop(device_id, None)
        _connection_states.pop(device_id, None)


async def _handle_json_message(
    ws: WebSocket,
    device_id: str,
    msg,
    dialogue,
) -> None:
    """Handle JSON control messages."""

    if isinstance(msg, HeartbeatMessage):
        # Update device heartbeat
        _connection_states[device_id] = msg.state
        await ws.send_json({"v": PROTOCOL_VERSION, "type": "pong", "ts": time.time()})

    elif isinstance(msg, SpeechSegmentMessage):
        # Audio segment header — actual audio comes as next binary frame
        # (handled in _handle_binary_audio)
        pass

    elif msg.type == MessageType.LIST_MODELS:
        model_type = getattr(msg, "model_type", "pet")
        async with AsyncSessionLocal() as session:
            if model_type == "pet":
                models = await queries.list_characters(session)
                await ws.send_json({
                    "v": PROTOCOL_VERSION,
                    "type": "models_list",
                    "model_type": "pet",
                    "models": [{"id": m.id, "name": m.name, "preview_url": m.preview_url} for m in models],
                })
            elif model_type == "voice":
                voices = await queries.list_voices(session)
                await ws.send_json({
                    "v": PROTOCOL_VERSION,
                    "type": "models_list",
                    "model_type": "voice",
                    "models": [{"id": v.id, "name": v.name, "engine": v.engine} for v in voices],
                })

    elif msg.type == MessageType.SWITCH_REQUEST:
        model_type = getattr(msg, "model_type", "pet")
        model_id = getattr(msg, "model_id", "")
        await _handle_switch_request(ws, device_id, model_type, model_id)

    else:
        logger.debug(f"Unknown message type: {msg.type}")


async def _handle_binary_audio(
    ws: WebSocket,
    device_id: str,
    audio_data: bytes,
    dialogue,
) -> None:
    """Process incoming audio segment from ESP32."""

    if len(audio_data) < 100:  # Too small to be meaningful speech
        return

    # Update state
    _connection_states[device_id] = DeviceState.THINKING
    await ws.send_json(StateMessage(
        state=DeviceState.THINKING,
        emotion="neutral",
    ).to_json())

    try:
        # Load conversation history
        async with AsyncSessionLocal() as session:
            history_rows = await queries.get_recent_history(session, device_id, limit=10)
        history = [
            Message(role=r["role"], content=r["content"])
            for r in history_rows
        ]

        # Run dialogue pipeline
        chunks = await dialogue.process_speech(
            device_id=device_id,
            audio=audio_data,
            conversation_history=history,
        )

        if not chunks:
            # Empty response (e.g., silence)
            _connection_states[device_id] = DeviceState.IDLE
            await ws.send_json(StateMessage(state=DeviceState.IDLE).to_json())
            return

        # Get the LLM response text from last turn
        async with AsyncSessionLocal() as session:
            hist = await queries.get_recent_history(session, device_id, limit=1)
        assistant_text = hist[-1]["content"] if hist and hist[-1]["role"] == "assistant" else ""
        assistant_emotion = hist[-1].get("emotion", "neutral") if hist else "neutral"

        # Send speech header
        rms_frames = [c.rms for c in chunks if c.audio]
        await ws.send_json(SpeechMessage(
            text=assistant_text,
            emotion=assistant_emotion or "neutral",
            rms_frames=rms_frames,
            audio_chunks=len(chunks),
        ).to_json())

        # Send state → speaking
        _connection_states[device_id] = DeviceState.SPEAKING
        await ws.send_json(StateMessage(
            state=DeviceState.SPEAKING,
            emotion=assistant_emotion or "neutral",
            text=assistant_text,
        ).to_json())

        # Stream audio chunks as binary frames
        for chunk in chunks:
            if chunk.audio:
                await ws.send_bytes(encode_audio_frame(chunk.audio))

        # Send terminator (zero-length frame)
        await ws.send_bytes(encode_audio_frame(b""))

        # Back to idle
        _connection_states[device_id] = DeviceState.IDLE
        await ws.send_json(StateMessage(state=DeviceState.IDLE).to_json())

    except Exception as e:
        logger.error(f"Dialogue error for {device_id}: {e}", exc_info=True)
        _connection_states[device_id] = DeviceState.ERROR
        await ws.send_json(ErrorMessage(
            code="DIALOGUE_ERROR",
            message=str(e),
        ).to_json())
        # Recover after error
        _connection_states[device_id] = DeviceState.IDLE
        await ws.send_json(StateMessage(state=DeviceState.IDLE).to_json())


async def _handle_switch_request(
    ws: WebSocket,
    device_id: str,
    model_type: str,
    model_id: str,
) -> None:
    """Handle model switch requests from ESP32 or web admin."""
    async with AsyncSessionLocal() as session:
        if model_type == "pet":
            character = await queries.get_character(session, model_id)
            if character:
                await queries.update_device_character(session, device_id, model_id)
                await session.commit()
                await ws.send_json(SwitchModelMessage(
                    model_type="pet",
                    model_id=model_id,
                    model_name=character.name,
                    sprite_manifest=json.loads(character.sprite_manifest) if character.sprite_manifest else None,
                    download_url=character.sprite_pack_url or None,
                ).to_json())
                await _send_character_sync(ws, device_id)

        elif model_type == "voice":
            voice = await queries.get_voice(session, model_id)
            if voice:
                await queries.update_device_voice(session, device_id, model_id)
                await session.commit()
                await ws.send_json(SwitchModelMessage(
                    model_type="voice",
                    model_id=model_id,
                    model_name=voice.name,
                ).to_json())


async def _send_character_sync(ws: WebSocket, device_id: str) -> None:
    """Send full character state to device on connect or switch."""
    async with AsyncSessionLocal() as session:
        character = await queries.get_device_character(session, device_id)
        voice_id = await queries.get_device_voice(session, device_id)

    if character:
        await ws.send_json(CharacterSyncMessage(
            character={
                "id": character.id,
                "name": character.name,
                "personality_summary": character.personality[:200],
                "default_mood": character.default_mood,
                "default_outfit": character.default_outfit,
                "name_call": character.name_call,
            },
            sprites_url=character.sprite_pack_url,
            voice_id=voice_id or character.default_voice_id,
        ).to_json())
