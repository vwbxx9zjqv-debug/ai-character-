"""
Admin API Routes — Character/voice management, model upload, system status.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db import queries
from core.plugin_manager import get_plugin_registry

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ── Character Management ──────────────────────────────────────────

@router.get("/characters")
async def list_characters(user_id: Optional[str] = None, session: AsyncSession = Depends(get_db)):
    """List all available character models."""
    chars = await queries.list_characters(session, user_id)
    return {"characters": [_character_to_dict(c) for c in chars]}


@router.get("/characters/{character_id}")
async def get_character(character_id: str, session: AsyncSession = Depends(get_db)):
    char = await queries.get_character(session, character_id)
    if not char:
        raise HTTPException(404, "Character not found")
    return _character_to_dict(char)


@router.post("/characters")
async def create_character(
    name: str = Form(...),
    personality: str = Form(...),
    personality_tags: str = Form('[]'),
    speaking_style: str = Form(''),
    verbal_tics: str = Form('[]'),
    backstory: str = Form(''),
    likes: str = Form('[]'),
    dislikes: str = Form('[]'),
    hobbies: str = Form('[]'),
    default_mood: str = Form('neutral'),
    name_call: str = Form('主人'),
    default_voice_id: str = Form('edge_xiaoxiao'),
    default_outfit: str = Form('默认服装'),
    outfits: str = Form('[]'),
    session: AsyncSession = Depends(get_db),
):
    """Create a new user character."""
    char_id = f"user_{uuid.uuid4().hex[:12]}"
    await queries.create_character(
        session,
        id=char_id, name=name, personality=personality,
        personality_tags=personality_tags, speaking_style=speaking_style,
        verbal_tics=verbal_tics, backstory=backstory,
        likes=likes, dislikes=dislikes, hobbies=hobbies,
        default_mood=default_mood, name_call=name_call,
        default_voice_id=default_voice_id, default_outfit=default_outfit,
        outfits=outfits, user_id="user",
    )
    await session.commit()
    return {"id": char_id, "name": name}


@router.put("/characters/{character_id}")
async def update_character(
    character_id: str,
    name: str = Form(None),
    personality: str = Form(None),
    session: AsyncSession = Depends(get_db),
):
    """Update character fields."""
    char = await queries.get_character(session, character_id)
    if not char:
        raise HTTPException(404, "Character not found")
    updates = {}
    if name is not None:
        updates["name"] = name
    if personality is not None:
        updates["personality"] = personality
    if updates:
        from sqlalchemy import text
        set_clause = ", ".join(f"{k} = :{k}" for k in updates)
        await session.execute(
            text(f"UPDATE character_models SET {set_clause}, updated_at = datetime('now') WHERE id = :id"),
            {"id": character_id, **updates}
        )
        await session.commit()
    return {"ok": True}


@router.delete("/characters/{character_id}")
async def delete_character(character_id: str, session: AsyncSession = Depends(get_db)):
    """Delete a character (only user-created, non-builtin)."""
    char = await queries.get_character(session, character_id)
    if not char:
        raise HTTPException(404, "Character not found")
    if char.id.startswith("hoshino") or char.id.startswith("nanase") or char.id.startswith("kujo"):
        raise HTTPException(400, "Cannot delete built-in characters")
    from sqlalchemy import text
    await session.execute(text("DELETE FROM character_models WHERE id = :id"), {"id": character_id})
    await session.commit()
    return {"ok": True}


# ── Voice Management ──────────────────────────────────────────────

@router.get("/voices")
async def list_voices(user_id: Optional[str] = None, session: AsyncSession = Depends(get_db)):
    voices = await queries.list_voices(session, user_id)
    return {"voices": [_voice_to_dict(v) for v in voices]}


@router.post("/voices")
async def create_voice(
    name: str = Form(...),
    engine: str = Form(...),
    engine_config: str = Form('{}'),
    emotions: str = Form('["neutral"]'),
    languages: str = Form('["zh-CN"]'),
    session: AsyncSession = Depends(get_db),
):
    """Create a new voice model entry."""
    voice_id = f"voice_{uuid.uuid4().hex[:12]}"
    await queries.create_voice(
        session,
        id=voice_id, name=name, engine=engine,
        engine_config=engine_config, emotions=emotions, languages=languages,
        user_id="user",
    )
    await session.commit()
    return {"id": voice_id, "name": name}


@router.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str, session: AsyncSession = Depends(get_db)):
    voice = await queries.get_voice(session, voice_id)
    if not voice:
        raise HTTPException(404, "Voice not found")
    if voice.is_builtin:
        raise HTTPException(400, "Cannot delete built-in voices")
    from sqlalchemy import text
    await session.execute(text("DELETE FROM voice_models WHERE id = :id"), {"id": voice_id})
    await session.commit()
    return {"ok": True}


# ── Device Management ─────────────────────────────────────────────

@router.get("/devices")
async def list_devices(session: AsyncSession = Depends(get_db)):
    from sqlalchemy import text
    result = await session.execute(text("SELECT * FROM devices ORDER BY last_seen_at DESC"))
    return {"devices": [dict(row._mapping) for row in result.fetchall()]}


@router.put("/devices/{device_id}/character")
async def set_device_character(
    device_id: str,
    character_id: str = Form(...),
    session: AsyncSession = Depends(get_db),
):
    await queries.update_device_character(session, device_id, character_id)
    await session.commit()
    return {"ok": True}


@router.put("/devices/{device_id}/voice")
async def set_device_voice(
    device_id: str,
    voice_id: str = Form(...),
    session: AsyncSession = Depends(get_db),
):
    await queries.update_device_voice(session, device_id, voice_id)
    await session.commit()
    return {"ok": True}


# ── Provider Info ─────────────────────────────────────────────────

@router.get("/providers")
async def list_providers():
    """List all registered STT/LLM/TTS providers."""
    registry = get_plugin_registry()
    return {
        "stt": registry.list_stt_providers(),
        "llm": registry.list_llm_providers(),
        "tts": registry.list_tts_providers(),
    }


# ── Upload ────────────────────────────────────────────────────────

@router.post("/upload/sprite")
async def upload_sprite(
    character_id: str = Form(...),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db),
):
    """Upload a sprite pack ZIP for a character."""
    from config import get_settings
    settings = get_settings()
    upload_dir = Path(settings.upload_dir) / "sprites" / character_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / (file.filename or "sprite_pack.zip")
    content = await file.read()
    file_path.write_bytes(content)

    # Update character's sprite_pack_url
    from sqlalchemy import text
    await session.execute(
        text("UPDATE character_models SET sprite_pack_url = :url, updated_at = datetime('now') WHERE id = :id"),
        {"url": f"/uploads/sprites/{character_id}/{file.filename}", "id": character_id}
    )
    await session.commit()

    return {"ok": True, "url": f"/uploads/sprites/{character_id}/{file.filename}", "size": len(content)}


# ── Helpers ───────────────────────────────────────────────────────

def _character_to_dict(c) -> dict:
    return {
        "id": c.id, "name": c.name,
        "personality": c.personality,
        "personality_tags": json.loads(c.personality_tags) if isinstance(c.personality_tags, str) else c.personality_tags,
        "speaking_style": c.speaking_style,
        "verbal_tics": json.loads(c.verbal_tics) if isinstance(c.verbal_tics, str) else c.verbal_tics,
        "backstory": c.backstory,
        "likes": json.loads(c.likes) if isinstance(c.likes, str) else c.likes,
        "dislikes": json.loads(c.dislikes) if isinstance(c.dislikes, str) else c.dislikes,
        "hobbies": json.loads(c.hobbies) if isinstance(c.hobbies, str) else c.hobbies,
        "default_mood": c.default_mood,
        "name_call": c.name_call,
        "default_voice_id": c.default_voice_id,
        "default_outfit": c.default_outfit,
        "outfits": json.loads(c.outfits) if isinstance(c.outfits, str) else c.outfits,
        "sprite_pack_url": c.sprite_pack_url,
        "sprite_manifest": json.loads(c.sprite_manifest) if isinstance(c.sprite_manifest, str) else c.sprite_manifest,
        "preview_url": c.preview_url,
    }


def _voice_to_dict(v) -> dict:
    return {
        "id": v.id, "name": v.name, "engine": v.engine,
        "engine_config": json.loads(v.engine_config) if isinstance(v.engine_config, str) else v.engine_config,
        "emotions": json.loads(v.emotions) if isinstance(v.emotions, str) else v.emotions,
        "languages": json.loads(v.languages) if isinstance(v.languages, str) else v.languages,
        "sample_url": v.sample_url,
        "is_builtin": v.is_builtin,
    }
