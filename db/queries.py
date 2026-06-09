"""
Database queries — Async SQLAlchemy operations.

Thin data access layer. All queries go through here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class CharacterRow:
    id: str
    name: str
    personality: str
    personality_tags: str
    speaking_style: str
    verbal_tics: str
    backstory: str
    likes: str
    dislikes: str
    hobbies: str
    default_mood: str
    name_call: str
    default_voice_id: str
    default_outfit: str
    outfits: str
    sprite_pack_url: str
    sprite_manifest: str
    preview_url: str


@dataclass
class VoiceRow:
    id: str
    name: str
    engine: str
    engine_config: str
    emotions: str
    languages: str
    sample_url: str
    is_builtin: bool


@dataclass
class FactRow:
    id: int
    fact: str
    category: str
    importance: int


# ── Device Queries ────────────────────────────────────────────────

async def get_device(session: AsyncSession, device_id: str) -> dict | None:
    result = await session.execute(
        text("SELECT * FROM devices WHERE id = :id"), {"id": device_id}
    )
    row = result.fetchone()
    return dict(row._mapping) if row else None


async def upsert_device(session: AsyncSession, device_id: str, **kwargs) -> None:
    """Insert or update device record."""
    existing = await get_device(session, device_id)
    if existing:
        set_clause = ", ".join(f"{k} = :{k}" for k in kwargs)
        await session.execute(
            text(f"UPDATE devices SET {set_clause}, updated_at = datetime('now') WHERE id = :id"),
            {"id": device_id, **kwargs}
        )
    else:
        columns = ["id"] + list(kwargs.keys())
        values = [f":{c}" for c in columns]
        await session.execute(
            text(f"INSERT INTO devices ({', '.join(columns)}) VALUES ({', '.join(values)})"),
            {"id": device_id, **kwargs}
        )


async def get_device_character(session: AsyncSession, device_id: str) -> CharacterRow | None:
    """Get the currently active character for a device."""
    result = await session.execute(
        text("""
            SELECT cm.* FROM character_models cm
            JOIN devices d ON d.current_character_id = cm.id
            WHERE d.id = :device_id
        """),
        {"device_id": device_id}
    )
    row = result.fetchone()
    if row is None:
        # Fallback to default
        result = await session.execute(
            text("SELECT * FROM character_models WHERE is_builtin = 1 LIMIT 1")
        )
        row = result.fetchone()
    return _row_to_character(row) if row else None


async def get_device_voice(session: AsyncSession, device_id: str) -> str | None:
    """Get the current voice ID for a device."""
    result = await session.execute(
        text("SELECT current_voice_id FROM devices WHERE id = :id"),
        {"id": device_id}
    )
    row = result.fetchone()
    return row[0] if row and row[0] else None


async def update_device_character(session: AsyncSession, device_id: str, character_id: str) -> None:
    await upsert_device(session, device_id, current_character_id=character_id)


async def update_device_voice(session: AsyncSession, device_id: str, voice_id: str) -> None:
    await upsert_device(session, device_id, current_voice_id=voice_id)


# ── Character Queries ─────────────────────────────────────────────

async def list_characters(session: AsyncSession, user_id: str | None = None) -> list[CharacterRow]:
    """List all available characters (built-in + user's custom ones)."""
    if user_id:
        result = await session.execute(
            text("SELECT * FROM character_models WHERE is_builtin = 1 OR user_id = :uid ORDER BY is_builtin DESC, name"),
            {"uid": user_id}
        )
    else:
        result = await session.execute(
            text("SELECT * FROM character_models ORDER BY is_builtin DESC, name")
        )
    return [_row_to_character(row) for row in result.fetchall()]


async def get_character(session: AsyncSession, character_id: str) -> CharacterRow | None:
    result = await session.execute(
        text("SELECT * FROM character_models WHERE id = :id"), {"id": character_id}
    )
    row = result.fetchone()
    return _row_to_character(row) if row else None


async def create_character(session: AsyncSession, **kwargs) -> str:
    """Create a new character. Returns the character ID."""
    await session.execute(
        text("""INSERT INTO character_models (id, name, personality, user_id, is_builtin)
                VALUES (:id, :name, :personality, :user_id, 0)"""),
        kwargs
    )
    return kwargs["id"]


# ── Voice Queries ─────────────────────────────────────────────────

async def list_voices(session: AsyncSession, user_id: str | None = None) -> list[VoiceRow]:
    if user_id:
        result = await session.execute(
            text("SELECT * FROM voice_models WHERE is_builtin = 1 OR user_id = :uid"),
            {"uid": user_id}
        )
    else:
        result = await session.execute(text("SELECT * FROM voice_models"))
    return [_row_to_voice(row) for row in result.fetchall()]


async def get_voice(session: AsyncSession, voice_id: str) -> VoiceRow | None:
    result = await session.execute(
        text("SELECT * FROM voice_models WHERE id = :id"), {"id": voice_id}
    )
    row = result.fetchone()
    return _row_to_voice(row) if row else None


async def create_voice(session: AsyncSession, **kwargs) -> str:
    await session.execute(
        text("""INSERT INTO voice_models (id, name, engine, engine_config, emotions, languages, user_id, is_builtin)
                VALUES (:id, :name, :engine, :engine_config, :emotions, :languages, :user_id, 0)"""),
        kwargs
    )
    return kwargs["id"]


# ── Conversation Queries ──────────────────────────────────────────

async def save_conversation(
    session: AsyncSession,
    device_id: str, role: str, content: str,
    emotion: str | None = None,
    character_id: str | None = None,
    voice_id: str | None = None,
    stt_latency: int = 0, llm_latency: int = 0,
    tts_latency: int = 0, total_latency: int = 0,
) -> None:
    await session.execute(
        text("""INSERT INTO conversations
                (device_id, role, content, emotion, character_id, voice_id,
                 stt_latency_ms, llm_latency_ms, tts_latency_ms, total_latency_ms)
                VALUES (:device_id, :role, :content, :emotion, :character_id, :voice_id,
                        :stt_latency, :llm_latency, :tts_latency, :total_latency)"""),
        {
            "device_id": device_id, "role": role, "content": content,
            "emotion": emotion, "character_id": character_id,
            "voice_id": voice_id, "stt_latency": stt_latency,
            "llm_latency": llm_latency, "tts_latency": tts_latency,
            "total_latency": total_latency,
        }
    )


async def get_recent_history(
    session: AsyncSession, device_id: str, limit: int = 10
) -> list[dict]:
    result = await session.execute(
        text("""SELECT role, content FROM conversations
                WHERE device_id = :did ORDER BY created_at DESC LIMIT :lim"""),
        {"did": device_id, "lim": limit}
    )
    return [dict(row._mapping) for row in reversed(result.fetchall())]


# ── Memory Queries ────────────────────────────────────────────────

async def save_fact(session: AsyncSession, device_id: str, fact: str, category: str = "personal", importance: int = 1) -> None:
    await session.execute(
        text("INSERT INTO user_facts (device_id, fact, category, importance) VALUES (:did, :fact, :cat, :imp)"),
        {"did": device_id, "fact": fact, "cat": category, "imp": importance}
    )


async def get_recent_facts(session: AsyncSession, device_id: str, limit: int = 10) -> list[FactRow]:
    result = await session.execute(
        text("SELECT id, fact, category, importance FROM user_facts WHERE device_id = :did ORDER BY importance DESC, created_at DESC LIMIT :lim"),
        {"did": device_id, "lim": limit}
    )
    return [FactRow(id=row[0], fact=row[1], category=row[2], importance=row[3]) for row in result.fetchall()]


# ── Helpers ───────────────────────────────────────────────────────

def _row_to_character(row) -> CharacterRow:
    m = row._mapping if hasattr(row, '_mapping') else dict(zip(
        ["id", "user_id", "name", "personality", "personality_tags", "speaking_style",
         "verbal_tics", "backstory", "likes", "dislikes", "hobbies", "default_mood",
         "mood_volatility", "attachment_speed", "name_call", "default_voice_id",
         "default_outfit", "outfits", "sprite_pack_url", "sprite_manifest",
         "preview_url", "preview_anim_url", "is_builtin", "is_public",
         "download_count", "created_at", "updated_at"],
        row
    ))
    return CharacterRow(
        id=m["id"], name=m["name"], personality=m["personality"],
        personality_tags=m.get("personality_tags", "[]"),
        speaking_style=m.get("speaking_style", ""), verbal_tics=m.get("verbal_tics", "[]"),
        backstory=m.get("backstory", ""), likes=m.get("likes", "[]"),
        dislikes=m.get("dislikes", "[]"), hobbies=m.get("hobbies", "[]"),
        default_mood=m.get("default_mood", "neutral"),
        name_call=m.get("name_call", "主人"),
        default_voice_id=m.get("default_voice_id", ""),
        default_outfit=m.get("default_outfit", "默认服装"),
        outfits=m.get("outfits", "[]"),
        sprite_pack_url=m.get("sprite_pack_url", ""),
        sprite_manifest=m.get("sprite_manifest", "{}"),
        preview_url=m.get("preview_url", ""),
    )


def _row_to_voice(row) -> VoiceRow:
    m = row._mapping if hasattr(row, '_mapping') else dict(zip(
        ["id", "user_id", "name", "engine", "engine_config", "emotions",
         "languages", "sample_url", "is_builtin", "is_public", "usage_count", "created_at"],
        row
    ))
    return VoiceRow(
        id=m["id"], name=m["name"], engine=m["engine"],
        engine_config=m.get("engine_config", "{}"),
        emotions=m.get("emotions", "[]"),
        languages=m.get("languages", '["zh-CN"]'),
        sample_url=m.get("sample_url", ""),
        is_builtin=bool(m.get("is_builtin", 0)),
    )
