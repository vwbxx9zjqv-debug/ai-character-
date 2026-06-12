"""
L2 Memory Extraction Service — Auto-extracts user facts from dialogue.

Subscribes to DialogueCompleted events and periodically calls the LLM
to extract structured facts about the user. Deduplicates against existing
facts using Jaccard character-level similarity (no embedding model needed).

This is a CapabilityPlugin — gated behind enable_long_term_memory config.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from core.event_bus import get_event_bus, DialogueCompleted
from core.plugin_manager import get_plugin_registry
from db.database import AsyncSessionLocal
from db import queries
from config import get_settings

logger = logging.getLogger("memory_service")


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute character bigram Jaccard similarity between two Chinese strings."""
    def bigrams(s: str) -> set[str]:
        return {s[i:i+2] for i in range(len(s)-1)} if len(s) >= 2 else {s}

    set_a = bigrams(a)
    set_b = bigrams(b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _parse_llm_facts(text: str) -> list[dict[str, Any]]:
    """Parse the LLM's JSON fact extraction output. Robust to format variations."""
    try:
        # Try direct JSON parse
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "facts" in data:
            return data["facts"]
        return []
    except json.JSONDecodeError:
        pass

    # Try to find JSON block
    json_match = re.search(r'\{[^{}]*"facts"[^{}]*\[.*?\][^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get("facts", [])
        except json.JSONDecodeError:
            pass

    # Try to find JSON array in the text
    arr_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if arr_match:
        try:
            return json.loads(arr_match.group())
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not parse fact extraction output: {text[:200]}")
    return []


class MemoryExtractionService:
    """Extracts user facts (L2) from conversation turns.

    Wires itself to the event bus as a DialogueCompleted subscriber.
    Only extracts every N turns to reduce LLM cost.
    """

    name = "memory_extraction"
    is_enabled = False

    def __init__(self):
        self.bus = get_event_bus()
        self._turn_counters: dict[str, int] = {}

    async def on_enable(self) -> None:
        self.is_enabled = True
        self.bus.subscribe(DialogueCompleted, self._on_dialogue_completed)
        logger.info("Memory Extraction Service enabled")

    async def on_disable(self) -> None:
        self.is_enabled = False
        logger.info("Memory Extraction Service disabled")

    async def _on_dialogue_completed(self, event: DialogueCompleted) -> None:
        if not self.is_enabled:
            return

        settings = get_settings()
        interval = getattr(settings, 'memory_extraction_interval', 5)

        device_id = event.device_id
        self._turn_counters[device_id] = self._turn_counters.get(device_id, 0) + 1

        if self._turn_counters[device_id] % interval != 0:
            return

        await self._extract_and_store(
            device_id, event.user_text, event.assistant_text
        )

    async def _extract_and_store(
        self, device_id: str, user_text: str, assistant_text: str
    ) -> None:
        """Extract facts via LLM, deduplicate, and store."""
        try:
            registry = get_plugin_registry()
            llm = registry.active_llm
        except RuntimeError:
            logger.warning("No active LLM, skipping fact extraction")
            return

        # Build extraction prompt
        extraction_prompt = f"""从以下对话中提取关于用户的新事实。只提取这次对话中显露的**新信息**，不要重复已知内容。

返回严格的JSON格式，不要加任何解释或markdown：
{{"facts": [{{"fact": "事实描述（中文短句）", "category": "personal/preference/event/milestone", "importance": 1-5}}]}}

如果没有值得记录的新事实，返回 {{"facts": []}}。

对话内容：
用户：{user_text}
角色：{assistant_text}"""

        try:
            result = await llm.chat(
                [],
                system_prompt=extraction_prompt,
                temperature=0.3,
                max_tokens=512,
            )
        except Exception as e:
            logger.error(f"LLM fact extraction failed: {e}")
            return

        facts = _parse_llm_facts(result.text)
        if not facts:
            return

        # Deduplicate against existing facts
        async with AsyncSessionLocal() as session:
            new_count = 0
            for fact_data in facts:
                fact_text = fact_data.get("fact", "").strip()
                if not fact_text or len(fact_text) < 4:
                    continue

                category = fact_data.get("category", "personal")
                importance = min(max(int(fact_data.get("importance", 1)), 1), 5)

                # Check similarity against existing facts
                existing = await queries.get_facts_by_category(
                    session, device_id, category
                )
                is_duplicate = any(
                    _jaccard_similarity(fact_text, f.fact) > 0.65
                    for f in existing
                )

                if is_duplicate:
                    continue

                await queries.save_fact(
                    session, device_id, fact_text,
                    category=category, importance=importance
                )
                new_count += 1
                logger.info(f"[{device_id}] New L2 fact: {fact_text}")

            if new_count > 0:
                await session.commit()
                logger.info(
                    f"[{device_id}] Extracted {new_count} new facts "
                    f"(turn #{self._turn_counters[device_id]})"
                )
