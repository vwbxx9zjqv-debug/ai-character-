"""
L3 Milestone Detection Service — Detects relationship milestones.

Two detection modes:
  A) Heuristic (fast, local): count-based milestones (day N, conversation N)
  B) LLM-based (deeper): semantic milestones (first goodnight, first secret shared)

Subscribes to DialogueCompleted events. Gated behind enable_long_term_memory.
"""

from __future__ import annotations

import logging
from datetime import datetime

from core.event_bus import get_event_bus, DialogueCompleted
from core.plugin_manager import get_plugin_registry
from db.database import AsyncSessionLocal
from db import queries
from config import get_settings

logger = logging.getLogger("milestone_service")

# ── Heuristic milestone definitions ──────────────────────────────
# (check function, milestone template pattern for dedup)

HEURISTIC_MILESTONES: list[tuple[str, int]] = [
    ("第1天", 1),
    ("第7天", 7),
    ("第14天", 14),
    ("第30天", 30),
    ("第60天", 60),
    ("第100天", 100),
    ("第365天", 365),
]


class MilestoneDetectionService:
    """Detects relationship milestones (L3) from dialogue and interaction patterns."""

    name = "milestone_detection"
    is_enabled = False

    def __init__(self):
        self.bus = get_event_bus()
        self._extraction_counter: dict[str, int] = {}

    async def on_enable(self) -> None:
        self.is_enabled = True
        self.bus.subscribe(DialogueCompleted, self._on_dialogue_completed)
        logger.info("Milestone Detection Service enabled")

    async def on_disable(self) -> None:
        self.is_enabled = False
        logger.info("Milestone Detection Service disabled")

    async def _on_dialogue_completed(self, event: DialogueCompleted) -> None:
        if not self.is_enabled:
            return

        settings = get_settings()
        interval = getattr(settings, 'memory_extraction_interval', 5)

        self._extraction_counter[event.device_id] = (
            self._extraction_counter.get(event.device_id, 0) + 1
        )

        # Run heuristic checks every turn (cheap)
        await self._check_heuristic_milestones(event)

        # Run LLM-based detection every N turns
        if self._extraction_counter[event.device_id] % interval == 0:
            await self._check_llm_milestones(event)

    async def _check_heuristic_milestones(self, event: DialogueCompleted) -> None:
        """Check count-based milestones (days together, conversation count)."""
        device_id = event.device_id

        async with AsyncSessionLocal() as session:
            first_date_str = await queries.get_first_interaction_date(session, device_id)
            conv_count = await queries.count_conversations(session, device_id)

            if first_date_str:
                try:
                    first_date = datetime.fromisoformat(first_date_str)
                    days_passed = (datetime.now() - first_date).days
                except (ValueError, TypeError):
                    days_passed = 0
            else:
                days_passed = 0

            # Day-based milestones
            for milestone_text, target_day in HEURISTIC_MILESTONES:
                if days_passed == target_day:
                    already = await queries.has_milestone(session, device_id, milestone_text)
                    if not already:
                        await queries.save_milestone(
                            session, device_id,
                            f"相处的{milestone_text}",
                            day_count=days_passed,
                        )
                        await session.commit()
                        logger.info(f"[{device_id}] Milestone: {milestone_text}")

            # Conversation count milestones
            conv_milestones = [10, 50, 100, 500, 1000]
            for threshold in conv_milestones:
                if conv_count == threshold:
                    text = f"第{threshold}次对话"
                    already = await queries.has_milestone(session, device_id, text)
                    if not already:
                        await queries.save_milestone(
                            session, device_id, text, day_count=days_passed
                        )
                        await session.commit()
                        logger.info(f"[{device_id}] Milestone: {text}")

    async def _check_llm_milestones(self, event: DialogueCompleted) -> None:
        """Use LLM to detect semantic milestones (first personal sharing, etc.)."""
        try:
            registry = get_plugin_registry()
            llm = registry.active_llm
        except RuntimeError:
            return

        device_id = event.device_id

        async with AsyncSessionLocal() as session:
            existing = await queries.get_milestones(session, device_id, limit=10)
            existing_texts = [m.milestone for m in existing]

            # Load character for context
            character = await queries.get_device_character(session, device_id)

        character_name = character.name if character else "角色"

        detection_prompt = f"""你正在分析一段对话，判断是否出现了值得记录的「关系里程碑」。

已有的里程碑（不要重复）：
{chr(10).join(f"- {t}" for t in existing_texts) if existing_texts else "(尚无)"}

这次对话：
用户：{event.user_text}
{character_name}：{event.assistant_text}

判断标准：
- "第一次说晚安" — 用户在睡前道别
- "第一次分享秘密" — 用户分享了非常私人的事
- "第一次表达关心" — 用户主动关心角色的状态
- "第一次约定" — 用户和角色做了某个约定
- "第一次夸奖" — 用户真诚地夸奖/感谢角色
- "第一次倾诉" — 用户向角色倾诉烦恼
- 其他你认为值得记录的「第一次」

返回严格JSON：
{{"milestones": [{{"milestone": "里程碑描述（中文）"}}]}}

如果没有新里程碑，返回 {{"milestones": []}}。只返回真正的新里程碑。"""

        try:
            result = await llm.chat(
                [],
                system_prompt=detection_prompt,
                temperature=0.3,
                max_tokens=512,
            )
        except Exception as e:
            logger.error(f"LLM milestone detection failed: {e}")
            return

        import json
        import re

        text = result.text

        # Parse JSON from LLM output
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\{[^{}]*"milestones"[^{}]*\[.*?\][^{}]*\}', text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    return
            else:
                return

        milestones = data.get("milestones", [])
        if not milestones:
            return

        async with AsyncSessionLocal() as session:
            new_count = 0
            for ms in milestones:
                ms_text = ms.get("milestone", "").strip()
                if not ms_text:
                    continue
                already = await queries.has_milestone(session, device_id, ms_text)
                if not already:
                    await queries.save_milestone(session, device_id, ms_text)
                    new_count += 1
                    logger.info(f"[{device_id}] LLM Milestone: {ms_text}")

            if new_count > 0:
                await session.commit()
