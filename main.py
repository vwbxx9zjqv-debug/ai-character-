"""
Virtual Character Companion — FastAPI Application Entry Point.

Startup sequence:
1. Initialize database schema
2. Discover and register built-in providers
3. Activate default STT/LLM/TTS providers
4. Start event bus subscribers
5. Serve API + WebSocket + Admin panel

Usage:
    uv run uvicorn main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import get_settings
from db.database import init_db, close_db
from core.plugin_manager import get_plugin_registry
from core.event_bus import get_event_bus
from routes import ws_chat, admin, browser

# ── Logging ───────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# ── App Lifecycle ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    settings = get_settings()

    # ── Startup ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Virtual Character Companion starting...")
    logger.info(f"  DB: {settings.database_url}")

    # 1. Initialize database
    await init_db()

    # 2. Discover providers
    registry = get_plugin_registry()
    registry.discover_builtin_providers()

    # 3. Activate default providers
    try:
        await registry.activate_stt(
            settings.default_stt,
            {"model": settings.whisper_model, "language": settings.whisper_language}
        )
        logger.info(f"  STT: {registry.active_stt.display_name}")
    except Exception as e:
        logger.warning(f"  STT activation skipped: {e}")

    try:
        # Build LLM config based on provider type
        if settings.default_llm == "OpenAICompatibleProvider":
            llm_config = {
                "model": settings.deepseek_model,
                "base_url": settings.deepseek_base_url,
                "temperature": settings.llm_temperature,
                "api_key": settings.openai_api_key,
            }
        else:
            llm_config = {
                "model": settings.claude_model,
                "temperature": settings.llm_temperature,
            }
        await registry.activate_llm(settings.default_llm, llm_config)
        logger.info(f"  LLM: {registry.active_llm.display_name}")
    except Exception as e:
        logger.warning(f"  LLM activation skipped: {e}")

    try:
        await registry.activate_tts(settings.default_tts, {})
        logger.info(f"  TTS: {registry.active_tts.display_name}")
    except Exception as e:
        logger.warning(f"  TTS activation skipped: {e}")

    # 4. Start capabilities (conversation logger)
    bus = get_event_bus()
    from plugins.capabilities.conversation_logger import ConversationLogger
    logger_plugin = ConversationLogger()
    await logger_plugin.on_enable()

    # 5. Start long-term memory system (L2/L3/L4) — if enabled
    if settings.enable_long_term_memory:
        try:
            from services.memory_service import MemoryExtractionService
            from services.milestone_service import MilestoneDetectionService
            from services.reflection_service import ReflectionService

            mem_service = MemoryExtractionService()
            await mem_service.on_enable()
            logger.info("  Memory: L2 fact extraction active")

            milestone_service = MilestoneDetectionService()
            await milestone_service.on_enable()
            logger.info("  Memory: L3 milestone detection active")

            reflection_service = ReflectionService()
            await reflection_service.on_enable()
            logger.info("  Memory: L4 character reflection active")
        except Exception as e:
            logger.warning(f"  Memory services skipped: {e}")

    # 6. Start mood engine — if enabled
    if settings.enable_mood_engine:
        try:
            from services.mood_engine import get_mood_engine
            mood_engine = get_mood_engine()
            await mood_engine.on_enable()
            logger.info("  Mood: emotion engine active")
        except Exception as e:
            logger.warning(f"  Mood engine skipped: {e}")

    # 7. Start proactive initiative engine + WebSocket subscriber — if enabled
    if settings.initiative_enabled:
        try:
            from services.initiative_engine import get_initiative_engine
            initiative_engine = get_initiative_engine()
            await initiative_engine.on_enable()
            logger.info("  Initiative: proactive speech engine active")

            # Wire initiative events to WebSocket push
            from routes.ws_chat import setup_initiative_subscriber
            setup_initiative_subscriber()
            logger.info("  Initiative: WebSocket subscriber active")
        except Exception as e:
            logger.warning(f"  Initiative engine skipped: {e}")

    logger.info(f"  Event bus: {bus.event_count} events processed")
    logger.info("Server ready! Listening on port " + str(settings.port))
    logger.info("=" * 60)

    yield  # ← App runs here

    # ── Shutdown ─────────────────────────────────────────────────
    logger.info("Shutting down...")
    await close_db()


# ── Application ───────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title="Virtual Character Companion",
    description="AI Virtual Character Companion — Extensible backend for ESP32 companion device",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────

app.include_router(ws_chat.router)
app.include_router(admin.router)
app.include_router(browser.router)

# Static files (admin panel + uploads)
static_dir = Path(__file__).parent / "static"
uploads_dir = Path(settings.upload_dir)
uploads_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")


# ── Health Check ──────────────────────────────────────────────────

@app.get("/chat")
async def chat_page():
    """Serve the browser companion page."""
    from fastapi.responses import HTMLResponse
    chat_html = Path(__file__).parent / "static" / "companion.html"
    return HTMLResponse(chat_html.read_text(encoding="utf-8"))


@app.get("/")
async def root():
    return {"name": "Virtual Character Companion", "version": "0.1.0", "status": "running"}


@app.get("/health")
async def health():
    registry = get_plugin_registry()
    return {
        "status": "ok",
        "stt": registry._active_stt.display_name if registry._active_stt else "not configured",
        "llm": registry._active_llm.display_name if registry._active_llm else "not configured",
        "tts": registry._active_tts.display_name if registry._active_tts else "not configured",
    }
