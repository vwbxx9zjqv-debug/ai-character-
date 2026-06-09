"""
Dialogue Service — Orchestrates the STT → LLM → TTS pipeline.

This is the brain of the companion: it receives audio, runs the full
dialogue pipeline, and emits events at each stage so plugins can react.
"""

from __future__ import annotations

import logging
import time

from core.event_bus import (
    get_event_bus,
    SpeechEnded, TranscriptionReady, LLMResponseReady,
    TTSStarted, TTSChunkReady, DialogueCompleted,
)
from core.plugin_base import Message, TTSChunk
from core.plugin_manager import get_plugin_registry
from db.database import AsyncSessionLocal
from db import queries

logger = logging.getLogger("dialogue")


class DialogueService:
    """Coordinates a single conversation turn: audio → text → LLM → TTS → audio."""

    def __init__(self):
        self.registry = get_plugin_registry()
        self.bus = get_event_bus()

    async def process_speech(
        self,
        device_id: str,
        audio: bytes,
        audio_format: str = "opus",
        conversation_history: list[Message] | None = None,
    ) -> list[TTSChunk]:
        """
        Process a full conversation turn.

        Returns TTS audio chunks ready for streaming to ESP32.
        All intermediate events are emitted on the bus.
        """
        turn_start = time.monotonic()

        # ── Step 1: STT ──────────────────────────────────────────
        stt_start = time.monotonic()
        stt = self.registry.active_stt
        stt_result = await stt.transcribe(audio, audio_format)
        stt_latency = int((time.monotonic() - stt_start) * 1000)

        logger.info(f"[{device_id}] STT: '{stt_result.text[:50]}...' ({stt_latency}ms)")

        await self.bus.emit(TranscriptionReady(
            device_id=device_id,
            text=stt_result.text,
            language=stt_result.language,
            confidence=stt_result.confidence,
            duration_ms=stt_latency,
        ))

        if not stt_result.text.strip():
            logger.info(f"[{device_id}] Empty transcription, skipping")
            return []

        # ── Step 2: LLM ──────────────────────────────────────────
        llm_start = time.monotonic()

        # Build messages from history
        messages = conversation_history or []
        messages.append(Message(role="user", content=stt_result.text))

        # Load character config for system prompt
        system_prompt = await self._load_system_prompt(device_id)

        llm = self.registry.active_llm
        llm_result = await llm.chat(messages, system_prompt=system_prompt)
        llm_latency = int((time.monotonic() - llm_start) * 1000)

        logger.info(
            f"[{device_id}] LLM: emotion={llm_result.emotion}, "
            f"text='{llm_result.text[:50]}...' ({llm_latency}ms)"
        )

        await self.bus.emit(LLMResponseReady(
            device_id=device_id,
            text=llm_result.text,
            emotion=llm_result.emotion,
            emotion_intensity=llm_result.emotion_intensity,
            tool_calls=llm_result.tool_calls,
            tokens_used=llm_result.tokens_used,
        ))

        # ── Step 3: TTS ──────────────────────────────────────────
        tts_start = time.monotonic()
        tts = self.registry.active_tts

        # Resolve voice model
        voice_id = await self._get_device_voice(device_id)
        voice = self.registry.get_voice(voice_id)
        if voice is None:
            # Fallback to first available
            voices = self.registry.list_voices()
            voice = voices[0] if voices else None
            if voice is None:
                raise RuntimeError("No voice models available")

        await self.bus.emit(TTSStarted(
            device_id=device_id,
            voice_id=voice.id,
            voice_name=voice.name,
            emotion=llm_result.emotion,
        ))

        # Stream TTS chunks
        chunks: list[TTSChunk] = []
        async for chunk in tts.synthesize(
            text=llm_result.text,
            voice=voice,
            emotion=llm_result.emotion,
            emotion_intensity=llm_result.emotion_intensity,
        ):
            chunks.append(chunk)
            await self.bus.emit(TTSChunkReady(
                device_id=device_id,
                audio_chunk=chunk.audio,
                rms=chunk.rms,
                is_final=chunk.is_final,
            ))

        tts_latency = int((time.monotonic() - tts_start) * 1000)
        total_latency = int((time.monotonic() - turn_start) * 1000)

        # ── Step 4: Persist & Complete ────────────────────────────
        async with AsyncSessionLocal() as session:
            await queries.save_conversation(
                session, device_id, "user", stt_result.text,
                stt_latency=stt_latency,
            )
            await queries.save_conversation(
                session, device_id, "assistant", llm_result.text,
                emotion=llm_result.emotion,
                stt_latency=stt_latency, llm_latency=llm_latency,
                tts_latency=tts_latency, total_latency=total_latency,
            )
            await session.commit()

        await self.bus.emit(DialogueCompleted(
            device_id=device_id,
            user_text=stt_result.text,
            assistant_text=llm_result.text,
            emotion=llm_result.emotion,
            total_latency_ms=total_latency,
            stt_latency_ms=stt_latency,
            llm_latency_ms=llm_latency,
            tts_latency_ms=tts_latency,
        ))

        logger.info(
            f"[{device_id}] Turn complete: {total_latency}ms "
            f"(STT={stt_latency}, LLM={llm_latency}, TTS={tts_latency})"
        )

        return chunks

    async def _load_system_prompt(self, device_id: str) -> str:
        """Build the character system prompt with context."""
        async with AsyncSessionLocal() as session:
            character = await queries.get_device_character(session, device_id)
            facts = await queries.get_recent_facts(session, device_id, limit=10)

        if character is None:
            return "You are a friendly AI companion. Keep responses short and natural."

        # Build from character data
        prompt = f"""你是{character.name}，一个存在于虚拟世界中的二次元角色。

## 你的性格
{character.personality}

## 你的说话方式
{character.speaking_style}

## 你的喜好
喜欢：{character.likes}
讨厌：{character.dislikes}

## 关于你面前的人
你把对方称呼为「{character.name_call}」。
{chr(10).join(f"- {fact.fact}" for fact in facts) if facts else ""}

## 重要规则
1. 始终保持在角色内。你不是AI助手，你是{character.name}本人。
2. 每次回复控制在1-3句话，保持聊天节奏。
3. 你对世界有自己的看法和态度，不需要迎合。
4. 偶尔会主动问对方问题，发起新话题。

## 情感输出
每条回复标注情感状态，格式: {{emotion: "xxx", text: "回复内容"}}
可用情感: happy / gentle / sad / excited / neutral / comforting / curious / shy / playful
"""
        return prompt

    async def _get_device_voice(self, device_id: str) -> str:
        """Get the current voice ID for a device."""
        async with AsyncSessionLocal() as session:
            voice_id = await queries.get_device_voice(session, device_id)
        return voice_id or "edge_xiaoxiao"


# ── Global singleton ──────────────────────────────────────────────

_dialogue_service: DialogueService | None = None


def get_dialogue_service() -> DialogueService:
    global _dialogue_service
    if _dialogue_service is None:
        _dialogue_service = DialogueService()
    return _dialogue_service
