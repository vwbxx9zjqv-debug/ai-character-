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
from services.prompt_builder import build_system_prompt, get_current_time_str, strip_emotion_tags

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
    t_total = time.time()

    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    registry = get_plugin_registry()
    llm = registry.active_llm
    tts = registry.active_tts

    # ── 1. Load character & context ──────────────────────────────
    t_db = time.time()
    from datetime import datetime as dt

    async with AsyncSessionLocal() as session:
        character = await queries.get_device_character(session, req.device_id)
        history_rows = await queries.get_recent_history(session, req.device_id, limit=5)
        facts = await queries.get_recent_facts(session, req.device_id, limit=5)

        # Load L3 milestones and L4 character memories
        milestones = await queries.get_milestones(session, req.device_id, limit=5)
        char_memories = await queries.get_character_memories(
            session, req.device_id, character.id if character else "", limit=3
        )
        first_date_str = await queries.get_first_interaction_date(session, req.device_id)

    # Compute days_passed
    days_passed = 0
    if first_date_str:
        try:
            first_date = dt.fromisoformat(first_date_str.replace("Z", "+00:00"))
            days_passed = (dt.now() - first_date.replace(tzinfo=None)).days
        except (ValueError, TypeError):
            pass

    # Build relationship context from milestones
    relationship_context = ""
    if milestones:
        recent_ms = [m.milestone for m in milestones[:3]]
        relationship_context = "你们的关系里程碑：" + "；".join(recent_ms)

    # Get current mood from mood engine (if enabled)
    current_mood = ""
    try:
        from services.mood_engine import get_mood_engine
        mood_engine = get_mood_engine()
        if mood_engine.is_enabled:
            current_mood = mood_engine.get_mood(req.device_id)
    except Exception:
        pass

    # Build character memory context
    memory_context = ""
    if char_memories:
        memory_context = "你内心的想法：" + "；".join(
            m.memory for m in char_memories
        )

    t_db = int((time.time() - t_db) * 1000)

    # ── 2. Build system prompt ───────────────────────────────────
    t_prompt = time.time()
    if character:
        user_name = character.name_call if character.name_call and character.name_call not in ("你", "") else ""
        system_prompt = build_system_prompt(
            character,
            user_name=user_name,
            current_time=get_current_time_str(),
            current_mood=current_mood,
            days_passed=days_passed,
            recent_facts=facts,
            relationship_context=relationship_context,
        )
        if memory_context:
            system_prompt += f"\n\n{memory_context}\n（这些是你内心的想法，不要直接说出来，但它们会影响你的回应方式。）"
    else:
        system_prompt = "你是一个友善的虚拟伙伴。保持回复简短自然，2-4句话以内。"

    # ── 3. Build conversation messages ───────────────────────────
    messages = [
        Message(role=r["role"], content=r["content"])
        for r in history_rows
    ]
    messages.append(Message(role="user", content=req.text))

    t_prompt = int((time.time() - t_prompt) * 1000)

    # ── 4. Call LLM ──────────────────────────────────────────────
    t_llm = time.time()
    llm_result = await llm.chat(messages, system_prompt=system_prompt)
    t_llm = int((time.time() - t_llm) * 1000)
    logger.info(
        f"[browser] LLM: emotion={llm_result.emotion} "
        f"text='{llm_result.text[:50]}...' ({t_llm}ms)"
    )

    # ── 5. Save conversation ─────────────────────────────────────
    t_save = time.time()
    async with AsyncSessionLocal() as session:
        await queries.save_conversation(
            session, req.device_id, "user", req.text,
        )
        await queries.save_conversation(
            session, req.device_id, "assistant", llm_result.text,
            emotion=llm_result.emotion,
        )
        await session.commit()

    t_save = int((time.time() - t_save) * 1000)

    # ── 6. Clean text & Call TTS ─────────────────────────────────
    t_tts = time.time()
    # Strip any leaked emotion tags before both TTS and display
    clean_text = strip_emotion_tags(llm_result.text)

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
                text=clean_text,
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

    t_tts = int((time.time() - t_tts) * 1000)
    t_total = int((time.time() - t_total) * 1000)
    logger.info(
        f"[browser] Turn complete: {t_total}ms "
        f"(DB={t_db}ms prompt={t_prompt}ms LLM={t_llm}ms save={t_save}ms TTS={t_tts}ms)"
    )

    return ChatResponse(
        user_text=req.text,
        assistant_text=clean_text,
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
