"""
Database setup — SQLite (dev) / Turso (production).

Uses SQLAlchemy 2.0 async with aiosqlite.
Schema is loaded from schema.sql on first run.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text

from config import get_settings

logger = logging.getLogger("db")

# ── Engine & Session ───────────────────────────────────────────────

_settings = get_settings()

engine = create_async_engine(
    _settings.database_url,
    echo=_settings.debug,
    connect_args={"check_same_thread": False} if "sqlite" in _settings.database_url else {},
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    """FastAPI dependency: yields an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ── Schema Initialization ─────────────────────────────────────────

async def init_db() -> None:
    """Initialize database schema on first run (idempotent)."""
    schema_path = Path(__file__).parent / "schema.sql"

    if not schema_path.exists():
        logger.warning(f"Schema file not found: {schema_path}")
        return

    schema_sql = schema_path.read_text(encoding="utf-8")

    # Ensure data directory exists
    db_dir = Path(_settings.database_url.replace("sqlite+aiosqlite:///", ""))
    db_dir.parent.mkdir(parents=True, exist_ok=True)

    async with engine.begin() as conn:
        # Split by statement and execute each (SQLite doesn't support multi-statement)
        statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
        for stmt in statements:
            try:
                await conn.execute(text(stmt))
            except Exception as e:
                logger.debug(f"Schema statement skipped (likely already exists): {e}")

    logger.info("Database schema initialized")


async def close_db() -> None:
    """Close database connections on shutdown."""
    await engine.dispose()
    logger.info("Database connections closed")
