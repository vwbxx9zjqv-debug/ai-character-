"""
Proactive Topic Engine — Character initiates conversation when user is idle.

Uses asyncio.sleep-based idle timers (no CPU polling). When the idle threshold
is exceeded, generates a natural proactive message via LLM and emits a
CharacterInitiative event.

Trigger conditions:
  1. idle_Ns — timer expiry (configurable, default 300s)
  2. time_of_day — morning greeting, evening check-in (optional)

This is a CapabilityPlugin — gated behind initiative_enabled config.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from core.event_bus import get_event_bus, DialogueCompleted, CharacterInitiative, PetEvent
from core.plugin_base import Message
from core.plugin_manager import get_plugin_registry
from db.database import AsyncSessionLocal
from db import queries
from config import get_settings

logger = logging.getLogger("initiative_engine")


def _build_initiative_prompt(
    character_name: str,
    character_personality: str,
    character_speaking_style: str,
    mood: str,
    facts: list[str],
    trigger_desc: str,
    user_name: str = "",
) -> str:
    """Build a system prompt for generating proactive character speech."""
    user_ref = user_name if user_name else "对方"

    prompt = f"""你是{character_name}，一个真实存在的人。

{character_personality}

你的说话方式：
{character_speaking_style}

你现在想要主动和{user_ref}说话，因为你注意到对方已经有{trigger_desc}了。
根据你的性格、你们的关系、以及当前的心情，自然地说出你想说的话（2-3句话）。

不要显得刻意——就像真人一样自然地开启话题。不要用"我注意到"、"看起来"这类观察者口吻。

你的心情：{mood}
"""

    if facts:
        prompt += f"\n你了解到的关于{user_ref}的事：\n"
        for fact in facts[:5]:
            prompt += f"- {fact}\n"

    prompt += """
直接输出带情感的回复，使用格式：{emotion: "xxx", text: "你的发言"}
可用情感标签：gentle / shy / passionate / comforting / curious / neutral / helpless / happy / sad / excited / playful
"""
    return prompt.strip()


class InitiativeEngine:
    """Manages idle timers and triggers proactive character speech."""

    name = "initiative_engine"
    is_enabled = False

    def __init__(self):
        self.bus = get_event_bus()
        self._last_interaction: dict[str, float] = {}  # device_id → timestamp
        self._idle_tasks: dict[str, asyncio.Task] = {}
        self._time_tasks: dict[str, asyncio.Task] = {}  # Time-of-day watchers

    async def on_enable(self) -> None:
        self.is_enabled = True
        # Track interaction resets
        self.bus.subscribe(DialogueCompleted, self._on_interaction)
        # Track all device events for liveness
        self.bus.on_any(self._on_any_device_event)
        logger.info("Initiative Engine enabled")

    async def on_disable(self) -> None:
        self.is_enabled = False
        for task in list(self._idle_tasks.values()):
            task.cancel()
        for task in list(self._time_tasks.values()):
            task.cancel()
        self._idle_tasks.clear()
        self._time_tasks.clear()
        logger.info("Initiative Engine disabled")

    def _get_idle_seconds(self) -> int:
        """Get configured idle timeout."""
        settings = get_settings()
        return getattr(settings, 'initiative_idle_seconds', 300)

    async def _on_interaction(self, event: DialogueCompleted) -> None:
        """Reset idle timer on every completed dialogue turn."""
        if not self.is_enabled:
            return
        device_id = event.device_id
        self._last_interaction[device_id] = time.time()

        # Cancel old timer, start new one
        if device_id in self._idle_tasks:
            self._idle_tasks[device_id].cancel()
        self._idle_tasks[device_id] = asyncio.create_task(
            self._idle_watcher(device_id)
        )

        # Ensure time-of-day watcher exists
        if device_id not in self._time_tasks:
            self._time_tasks[device_id] = asyncio.create_task(
                self._time_of_day_watcher(device_id)
            )

    async def _on_any_device_event(self, event: PetEvent) -> None:
        """Reset timer on any device activity."""
        if not self.is_enabled:
            return
        device_id = getattr(event, "device_id", None)
        if device_id:
            self._last_interaction[device_id] = time.time()

    async def _idle_watcher(self, device_id: str) -> None:
        """Sleep for idle timeout, then trigger if still idle."""
        idle_seconds = self._get_idle_seconds()
        try:
            await asyncio.sleep(idle_seconds)
        except asyncio.CancelledError:
            return

        last = self._last_interaction.get(device_id, 0)
        if time.time() - last >= idle_seconds - 2:  # Small tolerance
            await self._trigger_initiative(device_id, f"idle_{idle_seconds}s")

    async def _time_of_day_watcher(self, device_id: str) -> None:
        """Check hourly for time-of-day-based initiative triggers."""
        triggers = [
            (7, 8, "早上好"),  # Morning greeting window
            (21, 23, "夜深了"),  # Evening check-in window
        ]

        while self.is_enabled:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                return

            from datetime import datetime
            hour = datetime.now().hour

            for start_h, end_h, trigger_label in triggers:
                if start_h <= hour < end_h:
                    last = self._last_interaction.get(device_id, 0)
                    # Only trigger if user hasn't interacted in 30+ minutes
                    # and we haven't already triggered in this window
                    idle_seconds = self._get_idle_seconds()
                    if time.time() - last >= min(idle_seconds, 1800):
                        await self._trigger_initiative(device_id, trigger_label)
                    break  # Only one time-based trigger per check

    async def _trigger_initiative(self, device_id: str, trigger: str) -> None:
        """Generate and emit a proactive character message."""
        try:
            registry = get_plugin_registry()
            llm = registry.active_llm
        except RuntimeError:
            logger.warning("No active LLM, skipping initiative")
            return

        # Load context
        async with AsyncSessionLocal() as session:
            character = await queries.get_device_character(session, device_id)
            facts = await queries.get_recent_facts(session, device_id, limit=5)

        if character is None:
            return

        # Get current mood
        mood = character.default_mood
        try:
            from services.mood_engine import get_mood_engine
            me = get_mood_engine()
            if me.is_enabled:
                mood = me.get_mood(device_id)
        except Exception:
            pass

        # Build user name
        user_name = ""
        if character.name_call and character.name_call not in ("你", ""):
            user_name = character.name_call

        # Map trigger code to Chinese description
        trigger_desc_map = {
            "idle_300s": "一阵子没有动静了",
            "idle_600s": "好一阵子没有动静了",
            "idle_900s": "很久没有动静了",
            "早上好": "早上了",
            "夜深了": "夜深了",
        }
        trigger_desc = trigger_desc_map.get(trigger, trigger)
        if trigger.startswith("idle_"):
            seconds = int(trigger.replace("idle_", "").replace("s", ""))
            minutes = seconds // 60
            trigger_desc = f"{minutes}分钟没有动静了"

        prompt = _build_initiative_prompt(
            character_name=character.name,
            character_personality=character.personality,
            character_speaking_style=character.speaking_style,
            mood=mood,
            facts=[f.fact for f in facts],
            trigger_desc=trigger_desc,
            user_name=user_name,
        )

        try:
            result = await llm.chat(
                [],
                system_prompt=prompt,
                temperature=0.85,
                max_tokens=256,
            )
        except Exception as e:
            logger.error(f"LLM initiative generation failed: {e}")
            return

        logger.info(
            f"[{device_id}] Initiative triggered ({trigger}): "
            f"'{result.text[:60]}...' emotion={result.emotion}"
        )

        await self.bus.emit(CharacterInitiative(
            device_id=device_id,
            text=result.text,
            emotion=result.emotion,
            trigger=trigger,
        ))


# ── Global singleton ──────────────────────────────────────────────

_initiative_engine: InitiativeEngine | None = None


def get_initiative_engine() -> InitiativeEngine:
    """Get the global initiative engine singleton."""
    global _initiative_engine
    if _initiative_engine is None:
        _initiative_engine = InitiativeEngine()
    return _initiative_engine
