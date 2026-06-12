"""
Mood Engine — Tracks character emotional state with exponential smoothing.

Uses a multi-factor weighted model to evolve mood naturally over time,
avoiding abrupt changes. Mood is a float 0.0–1.0 mapped to discrete labels.
Emits MoodChanged events when the discrete label changes.

Lightweight, in-process, zero external dependencies.

Design:
- Exponential smoothing: new = old + α * (target - old), α = 0.3
- Four weighted input factors:
  1. Conversation sentiment (0.50) — LLM emotion output
  2. Time of day (0.15) — late night lowers, morning raises
  3. User engagement (0.20) — recent interactions raise mood
  4. Relationship duration (0.15) — longer relationship adds comfort

This is a CapabilityPlugin — gated behind enable_mood_engine config.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from core.event_bus import get_event_bus, DialogueCompleted, MoodChanged

logger = logging.getLogger("mood_engine")


class MoodEngine:
    """Per-device emotional state machine with exponential smoothing."""

    name = "mood_engine"
    is_enabled = False

    # Mood float range → discrete label
    MOOD_MAP: list[tuple[float, float, str]] = [
        (0.00, 0.10, "depressed"),
        (0.10, 0.25, "sad"),
        (0.25, 0.40, "melancholy"),
        (0.40, 0.60, "neutral"),
        (0.60, 0.75, "content"),
        (0.75, 0.90, "happy"),
        (0.90, 1.00, "excited"),
    ]

    # Emotion → float mapping (center of mood range for computing deltas)
    EMOTION_VALUE: dict[str, float] = {
        "happy": 0.83,
        "excited": 0.95,
        "playful": 0.80,
        "gentle": 0.60,
        "comforting": 0.58,
        "curious": 0.62,
        "neutral": 0.50,
        "shy": 0.45,
        "sad": 0.17,
        "helpless": 0.12,
        "passionate": 0.72,
    }

    # Default mood label → float
    DEFAULT_MOOD_VALUE: dict[str, float] = {
        "冷静": 0.55,
        "元气": 0.75,
        "平静": 0.50,
        "happy": 0.80,
        "neutral": 0.50,
        "sad": 0.20,
        "gentle": 0.60,
        "shy": 0.45,
    }

    def __init__(self):
        self.bus = get_event_bus()
        # Per-device state
        self._moods: dict[str, float] = {}  # device_id → current mood value (0-1)
        self._default_moods: dict[str, float] = {}  # device_id → character's default mood value
        self._last_interaction: dict[str, float] = {}  # device_id → timestamp
        self._alpha: float = 0.3  # Smoothing factor (higher = faster response)

    async def on_enable(self) -> None:
        self.is_enabled = True
        self.bus.subscribe(DialogueCompleted, self._on_dialogue_completed)
        logger.info("Mood Engine enabled (α=%.2f)", self._alpha)

    async def on_disable(self) -> None:
        self.is_enabled = False
        logger.info("Mood Engine disabled")

    def get_mood(self, device_id: str) -> str:
        """Get the current mood label for a device."""
        value = self._moods.get(device_id, 0.5)
        for lo, hi, label in self.MOOD_MAP:
            if lo <= value < hi:
                return label
        return "neutral"

    def get_mood_value(self, device_id: str) -> float:
        """Get the raw mood float value for a device."""
        return self._moods.get(device_id, 0.5)

    def set_default_mood(self, device_id: str, default_mood_label: str) -> None:
        """Initialize or reset a device's mood to its character's default."""
        default_value = self.DEFAULT_MOOD_VALUE.get(default_mood_label, 0.5)
        self._default_moods[device_id] = default_value
        if device_id not in self._moods:
            self._moods[device_id] = default_value
            logger.debug(
                f"[{device_id}] Mood initialized: {default_mood_label} ({default_value:.2f})"
            )

    async def _on_dialogue_completed(self, event: DialogueCompleted) -> None:
        if not self.is_enabled:
            return

        device_id = event.device_id
        now = time.time()
        self._last_interaction[device_id] = now

        # Ensure device has a default mood set
        if device_id not in self._default_moods:
            self._default_moods[device_id] = 0.5
        if device_id not in self._moods:
            self._moods[device_id] = self._default_moods[device_id]

        current = self._moods[device_id]
        previous_label = self.get_mood(device_id)

        # ── Factor 1: Conversation Sentiment (weight 0.50) ────────
        emotion_value = self.EMOTION_VALUE.get(event.emotion, 0.5)
        sentiment_delta = emotion_value - 0.5  # -0.5 to +0.5
        sentiment_factor = sentiment_delta * 0.50

        # ── Factor 2: Time of Day (weight 0.15) ───────────────────
        hour = datetime.now().hour
        if 22 <= hour or hour < 4:
            time_factor = -0.15  # Late night
        elif 4 <= hour < 6:
            time_factor = -0.05  # Early dawn
        elif 6 <= hour < 9:
            time_factor = 0.10  # Morning energy
        elif 9 <= hour < 12:
            time_factor = 0.05  # Productive morning
        elif 12 <= hour < 14:
            time_factor = 0.02  # Afternoon lull
        elif 14 <= hour < 18:
            time_factor = 0.00  # Neutral afternoon
        elif 18 <= hour < 22:
            time_factor = 0.08  # Evening relaxation
        else:
            time_factor = 0.0
        time_factor *= 0.15

        # ── Factor 3: User Engagement (weight 0.20) ───────────────
        # If user interacted within last 30 min, small positive bump
        engagement_factor = 0.0
        last_interaction = self._last_interaction.get(device_id, 0)
        if now - last_interaction < 1800:  # < 30 minutes
            engagement_factor = 0.10 * 0.20
        else:
            engagement_factor = -0.05 * 0.20  # Slight loneliness

        # ── Factor 4: Relationship Duration (weight 0.15) ──────────
        # Small positive bias from days_passed (loaded via event context)
        # We use a gentle asymptotic curve: 0.0 → 0.15 over time
        duration_factor = 0.0
        try:
            from db.database import AsyncSessionLocal
            from db import queries
            async with AsyncSessionLocal() as session:
                first_date_str = await queries.get_first_interaction_date(
                    session, device_id
                )
            if first_date_str:
                delta = datetime.now() - datetime.fromisoformat(
                    first_date_str.replace("Z", "+00:00")
                )
                days = max(0, delta.days)
                # Asymptotic: approaches 0.15 as days→∞, half at day 30
                duration_factor = (0.15 * days / (days + 30)) * 0.15
        except Exception:
            pass

        # ── Composite: Exponential Smoothing ──────────────────────
        target = 0.5 + sentiment_factor + time_factor + engagement_factor + duration_factor
        target = max(0.0, min(1.0, target))  # Clamp to [0, 1]

        new_mood = current + self._alpha * (target - current)
        new_mood = max(0.0, min(1.0, new_mood))
        self._moods[device_id] = new_mood

        # Check if discrete label changed
        new_label = self.get_mood(device_id)
        if new_label != previous_label:
            logger.info(
                f"[{device_id}] Mood changed: {previous_label} → {new_label} "
                f"(value: {current:.2f} → {new_mood:.2f}, target: {target:.2f})"
            )
            await self.bus.emit(MoodChanged(
                device_id=device_id,
                previous_mood=previous_label,
                new_mood=new_label,
                reason=f"conversation_sentiment={event.emotion}",
            ))


# ── Global singleton ──────────────────────────────────────────────

_mood_engine: MoodEngine | None = None


def get_mood_engine() -> MoodEngine:
    """Get the global mood engine singleton."""
    global _mood_engine
    if _mood_engine is None:
        _mood_engine = MoodEngine()
    return _mood_engine
