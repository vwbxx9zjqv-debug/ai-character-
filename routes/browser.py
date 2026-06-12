"""
Browser Chat API — HTTP endpoints for the PC browser companion prototype.

The browser uses Web Speech API for STT (free, no Whisper needed),
so this endpoint accepts TEXT directly and returns LLM response + TTS audio.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import struct
import time
import wave

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.plugin_base import Message
from core.plugin_manager import get_plugin_registry
from db.database import AsyncSessionLocal
from db import queries
from services.prompt_builder import build_system_prompt, get_current_time_str

logger = logging.getLogger("browser")

router = APIRouter(prefix="/api/browser", tags=["browser"])


# ── Request/Response Models ──────────────────────────────────────

class ChatRequest(BaseModel):
    text: str
    device_id: str = "browser"


class ChatResponse(BaseModel):
    user_text: str
    assistant_text: str
    emotion: str
    audio_base64: str | None = None  # base64-encoded WAV
    audio_duration_ms: int = 0


class CharacterInfo(BaseModel):
    id: str
    name: str
    default_mood: str
    personality_preview: str


# ── Chat Endpoint ─────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def browser_chat(req: ChatRequest) -> ChatResponse:
    """Process a text message from the browser companion.

    Flow: text → LLM → TTS → return text + audio
    (STT is done client-side via browser Web Speech API)
    """
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    registry = get_plugin_registry()
    llm = registry.active_llm
    tts = registry.active_tts

    # ── 1. Load character & context ──────────────────────────────
    async with AsyncSessionLocal() as session:
        character = await queries.get_device_character(session, req.device_id)
        history_rows = await queries.get_recent_history(session, req.device_id, limit=10)
        facts = await queries.get_recent_facts(session, req.device_id, limit=10)

    # ── 2. Build system prompt ───────────────────────────────────
    if character:
        user_name = character.name_call if character.name_call and character.name_call not in ("你", "") else ""
        system_prompt = build_system_prompt(
            character,
            user_name=user_name,
            current_time=get_current_time_str(),
            recent_facts=facts,
        )
    else:
        system_prompt = "你是一个友善的虚拟伙伴。保持回复简短自然，2-4句话以内。"

    # ── 3. Build conversation messages ───────────────────────────
    messages = [
        Message(role=r["role"], content=r["content"])
        for r in history_rows
    ]
    messages.append(Message(role="user", content=req.text))

    # ── 4. Call LLM ──────────────────────────────────────────────
    llm_result = await llm.chat(messages, system_prompt=system_prompt)

    logger.info(
        f"[browser] LLM response: emotion={llm_result.emotion}, "
        f"text='{llm_result.text[:60]}...'"
    )

    # ── 5. Save conversation ─────────────────────────────────────
    async with AsyncSessionLocal() as session:
        await queries.save_conversation(
            session, req.device_id, "user", req.text,
        )
        await queries.save_conversation(
            session, req.device_id, "assistant", llm_result.text,
            emotion=llm_result.emotion,
        )
        await session.commit()

    # ── 6. Call TTS ──────────────────────────────────────────────
    voice_id = "edge_xiaoxiao"  # Default, can be made configurable
    voice = registry.get_voice(voice_id)
    if voice is None:
        voices = registry.list_voices()
        voice = voices[0] if voices else None

    audio_base64 = None
    audio_duration_ms = 0

    if voice:
        try:
            # Collect all PCM chunks
            pcm_chunks: list[bytes] = []
            async for chunk in tts.synthesize(
                text=llm_result.text,
                voice=voice,
                emotion=llm_result.emotion,
            ):
                if chunk.audio:
                    pcm_chunks.append(chunk.audio)

            if pcm_chunks:
                pcm_data = b"".join(pcm_chunks)
                audio_duration_ms = int(len(pcm_data) / 32)  # 16kHz 16bit mono = 32 bytes/ms
                wav_data = _pcm_to_wav(pcm_data)
                audio_base64 = base64.b64encode(wav_data).decode("utf-8")
                logger.info(f"[browser] TTS: {len(pcm_data)} bytes PCM → {len(wav_data)} bytes WAV")
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)

    return ChatResponse(
        user_text=req.text,
        assistant_text=llm_result.text,
        emotion=llm_result.emotion,
        audio_base64=audio_base64,
        audio_duration_ms=audio_duration_ms,
    )


# ── Character List ────────────────────────────────────────────────

@router.get("/characters")
async def list_characters():
    """List available characters for the browser companion."""
    async with AsyncSessionLocal() as session:
        chars = await queries.list_characters(session)
    return {
        "characters": [
            {
                "id": c.id,
                "name": c.name,
                "default_mood": c.default_mood,
                "personality_preview": c.personality[:100] + "..." if len(c.personality) > 100 else c.personality,
                "voice_id": c.default_voice_id,
            }
            for c in chars
        ]
    }


# ── Switch Character ──────────────────────────────────────────────

class SwitchRequest(BaseModel):
    device_id: str = "browser"
    character_id: str


@router.post("/switch")
async def switch_character(req: SwitchRequest):
    """Switch the active character for a browser session."""
    async with AsyncSessionLocal() as session:
        character = await queries.get_character(session, req.character_id)
        if not character:
            raise HTTPException(404, "Character not found")
        await queries.update_device_character(session, req.device_id, req.character_id)
        await session.commit()

    return {
        "ok": True,
        "character": {
            "id": character.id,
            "name": character.name,
            "default_mood": character.default_mood,
        }
    }


# ── Helpers ───────────────────────────────────────────────────────

def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Convert raw PCM audio to WAV format for browser playback."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(bits_per_sample // 8)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()
