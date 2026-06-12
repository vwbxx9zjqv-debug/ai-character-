"""
L4 Character Reflection Service — Character privately reflects after turns.

After every few conversation turns, the character generates a brief private
thought (1-2 sentences) about the interaction. These thoughts are stored in
character_memories and subtly influence future behavior via the system prompt.

This is a CapabilityPlugin — gated behind enable_long_term_memory.
"""

from __future__ import annotations

import logging

from core.event_bus import get_event_bus, DialogueCompleted
from core.plugin_manager import get_plugin_registry
from db.database import AsyncSessionLocal
from db import queries

logger = logging.getLogger("reflection_service")


class ReflectionService:
    """Character generates private reflections periodically (L4 memory)."""

    name = "character_reflection"
    is_enabled = False

    def __init__(self):
        self.bus = get_event_bus()
        self._turn_counters: dict[str, int] = {}

    async def on_enable(self) -> None:
        self.is_enabled = True
        self.bus.subscribe(DialogueCompleted, self._on_dialogue_completed)
        logger.info("Reflection Service enabled")

    async def on_disable(self) -> None:
        self.is_enabled = False
        logger.info("Reflection Service disabled")

    async def _on_dialogue_completed(self, event: DialogueCompleted) -> None:
        if not self.is_enabled:
            return

        device_id = event.device_id
        self._turn_counters[device_id] = self._turn_counters.get(device_id, 0) + 1

        # Reflect every 3 turns (not every turn — balances cost vs richness)
        if self._turn_counters[device_id] % 3 != 0:
            return

        await self._reflect(event)

    async def _reflect(self, event: DialogueCompleted) -> None:
        """Generate and store a private character reflection."""
        try:
            registry = get_plugin_registry()
            llm = registry.active_llm
        except RuntimeError:
            logger.warning("No active LLM, skipping reflection")
            return

        device_id = event.device_id

        async with AsyncSessionLocal() as session:
            character = await queries.get_device_character(session, device_id)

        if character is None:
            return

        # Build a reflection prompt from the character's perspective
        reflection_prompt = f"""你现在是以{character.name}的身份进行内心反思。这是你私人的想法，不会被对方看到。

{character.personality}

刚才的对话：
对方说：{event.user_text}
你回答：{event.assistant_text}

请从{character.name}的视角，用1-2句话简短记录你此刻的真实感受或想法。不要评价对话质量，只需记录你内心的真实反应。
像写私人日记一样自然，不要用"我反思"、"我感觉"这种刻意的开头——直接写下你的想法。

输出格式：只输出反思内容本身，不要加任何前缀或引号。"""

        try:
            result = await llm.chat(
                [],
                system_prompt=reflection_prompt,
                temperature=0.7,
                max_tokens=128,
            )
        except Exception as e:
            logger.error(f"LLM reflection failed: {e}")
            return

        thought = result.text.strip()
        # Clean up common artifacts
        thought = thought.strip('"\'""'' \n')
        if len(thought) < 6:
            return  # Too short to be meaningful

        async with AsyncSessionLocal() as session:
            await queries.save_character_memory(
                session, device_id, character.id, thought
            )
            await session.commit()

        logger.info(
            f"[{device_id}] {character.name} reflection: {thought[:80]}..."
        )
