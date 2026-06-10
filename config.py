"""
Application configuration — loaded from environment variables and YAML files.

Pattern: pydantic-settings for env vars + YAML for character configs.
Reference: Open-LLM-VTuber src/open_llm_vtuber/config_manager/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Project paths ──────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
CHARACTERS_DIR = BASE_DIR / "characters"
VOICES_DIR = BASE_DIR / "voices"
STATIC_DIR = BASE_DIR / "static"
DB_PATH = BASE_DIR / "data" / "companion.db"


class Settings(BaseSettings):
    """Global application settings. Loaded from env / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Server ──────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ── Database ────────────────────────────────────────────────
    database_url: str = f"sqlite+aiosqlite:///{DB_PATH}"

    # ── Default Providers ───────────────────────────────────────
    default_stt: str = "WhisperAPIProvider"
    default_llm: str = "ClaudeProvider"
    default_tts: str = "EdgeTTSProvider"

    # ── API Keys ────────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── STT Config ──────────────────────────────────────────────
    whisper_model: str = "whisper-1"       # or "whisper-large-v3" for local
    whisper_language: str = "zh"           # Default recognition language

    # ── LLM Config ──────────────────────────────────────────────
    claude_model: str = "claude-haiku-4-5-20251001"
    llm_temperature: float = 0.8
    llm_max_tokens: int = 256
    # DeepSeek (used by OpenAICompatibleProvider)
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    # ── TTS Config ──────────────────────────────────────────────
    edge_tts_voice: str = "zh-CN-XiaoxiaoNeural"

    # ── Default Character ───────────────────────────────────────
    default_character: str = "hoshino_ruri"   # Default character config file

    # ── Storage ─────────────────────────────────────────────────
    upload_dir: str = str(BASE_DIR / "uploads")
    max_upload_size_mb: int = 100             # Max model file upload size

    # ── ESP32 / Device ──────────────────────────────────────────
    ws_heartbeat_interval: int = 30           # Seconds
    ws_heartbeat_timeout: int = 90            # Disconnect if no heartbeat

    # ── Conversation ────────────────────────────────────────────
    max_history_turns: int = 10               # Recent turns in LLM context
    vad_silence_timeout_ms: int = 1200        # Matches ESP32 config
    max_recording_ms: int = 15000

    # ── Memory System ───────────────────────────────────────────
    enable_long_term_memory: bool = False     # Enable L2/L3/L4 memory (Phase 3)
    memory_extraction_interval: int = 5       # Extract facts every N conversations

    # ── Mood Engine ─────────────────────────────────────────────
    enable_mood_engine: bool = False          # Enable mood evolution (Phase 3)
    initiative_enabled: bool = False          # Enable proactive speech (Phase 3)
    initiative_idle_seconds: int = 300        # Speak proactively after idle


# ── Global singleton ──────────────────────────────────────────────

_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
