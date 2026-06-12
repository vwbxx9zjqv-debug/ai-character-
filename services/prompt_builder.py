"""
System Prompt Builder — Compiles character DB data into LLM system prompts.

Takes a CharacterRow + device context and produces a compiled system prompt
with all template variables resolved. This is Person B's B1 deliverable.

Design:
- Core character identity is compiled from DB fields (personality, speaking_style, etc.)
- Dynamic variables ({user_name}, {days_passed}, {current_mood}, etc.) are injected at runtime
- Output is an 800-1500 token system prompt ready for LLM consumption
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db.queries import CharacterRow, FactRow


def strip_emotion_tags(text: str) -> str:
    """Remove emotion tag formatting and parenthetical stage directions.

    Handles:
      - {emotion: "xxx", text: "..."}  →  ...
      - {emotion: "xxx"} text          →  text
      - {emotion: xxx} text            →  text
      - {"emotion": "xxx", "text": "..."} → ...
      - （动作描写）说话内容              →  说话内容
    """
    # Strip Chinese parenthetical stage directions: （...） or ( ... )
    text = re.sub(r'[（(]\s*[^）)]*[）)]\s*', '', text)
    # Full structured emotion format
    text = re.sub(r'\{emotion:\s*"\w+"\s*,\s*text:\s*"(.+?)"\s*\}', r'\1', text)
    # Tag prefix format with quotes
    text = re.sub(r'\{emotion:\s*"\w+"\}\s*', '', text)
    # Tag prefix without quotes
    text = re.sub(r'\{emotion:\s*\w+\}\s*', '', text)
    # JSON format
    text = re.sub(r'\{"emotion":\s*"\w+"\s*,\s*"text":\s*"(.+?)"\s*\}', r'\1', text)
    return text.strip()


def _parse_json_list(value: str) -> str:
    """Parse a JSON array string into a human-readable list."""
    if not value:
        return ""
    try:
        items = json.loads(value)
        if isinstance(items, list):
            return "、".join(str(i) for i in items)
    except (json.JSONDecodeError, TypeError):
        pass
    return value


def build_system_prompt(
    character: CharacterRow,
    *,
    user_name: str = "",
    current_mood: str = "",
    days_passed: int = 0,
    recent_facts: list[FactRow] | None = None,
    relationship_context: str = "",
    current_time: str | None = None,
) -> str:
    """Build a complete system prompt from character data and runtime context.

    Args:
        character: Character DB row with personality, speaking_style, etc.
        user_name: The user's name (from memory/L2 facts)
        current_mood: Character's current emotional state
        days_passed: Days since first interaction
        recent_facts: User facts from long-term memory
        relationship_context: Relationship milestone summary
        current_time: Current time string (e.g., "晚上 11:23")
    """

    # ── Section 1: Core Identity ──────────────────────────────────
    prompt = f"""你是{character.name}。{character.personality}
说话方式：{character.speaking_style} 口头禅：{_parse_json_list(character.verbal_tics)}
"""

    # ── Section 2: Backstory ──────────────────────────────────────
    if character.backstory:
        prompt += f"""
## 你的故事
{character.backstory}
"""

    # ── Section 3: Daily Life ─────────────────────────────────────
    likes_str = _parse_json_list(character.likes)
    dislikes_str = _parse_json_list(character.dislikes)
    hobbies_str = _parse_json_list(character.hobbies) if character.hobbies else ""

    daily_parts = []
    if likes_str:
        daily_parts.append(f"喜欢{likes_str}")
    if dislikes_str:
        daily_parts.append(f"不太喜欢{dislikes_str}")
    if hobbies_str:
        daily_parts.append(f"空闲时{hobbies_str}")
    if daily_parts:
        prompt += f"\n你{'，'.join(daily_parts)}。\n"

    # ── Section 4: Relationship Context ───────────────────────────
    prompt += f"""
## 关于你面前的人
"""

    if user_name:
        prompt += f"对方的名字是{user_name}。"

    name_call = character.name_call.strip()
    if name_call and name_call not in ("你", ""):
        prompt += f"你称呼对方为「{name_call}」。"
    else:
        prompt += "你自然地称呼对方，根据语境使用「你」或对方的名字。"

    if days_passed > 0:
        prompt += f"这是你们相处的第{days_passed}天。\n"
    if relationship_context:
        prompt += f"{relationship_context}\n"

    # ── Section 5: Runtime Context ────────────────────────────────
    if current_time:
        prompt += f"\n现在的时间是{current_time}。\n"

    if current_mood:
        prompt += f"你现在的心情：{current_mood}。\n"

    # ── Section 6: User Facts (Memory) ────────────────────────────
    if recent_facts:
        prompt += "\n## 你了解到的关于对方的事\n"
        for fact in recent_facts:
            prompt += f"- {fact.fact}\n"

    # ── Section 7: Conversation Format ────────────────────────────
    prompt += f"""

## 格式
像发语音消息一样自然回复2-4句话。禁止括号动作描写如（笑）（轻声说）。
输出格式：{{emotion: \"xxx\", text: \"你的回复\"}}
可选情绪：gentle / shy / passionate / comforting / curious / neutral / helpless / happy / sad / excited / playful
"""
    return prompt.strip()


def build_system_prompt_simple(character: CharacterRow) -> str:
    """Simplified version without runtime context — used for admin previews."""
    return build_system_prompt(character)


def get_current_time_str() -> str:
    """Get a human-readable current time string in Chinese."""
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    if hour < 6:
        period = "凌晨"
    elif hour < 9:
        period = "早上"
    elif hour < 12:
        period = "上午"
    elif hour < 14:
        period = "中午"
    elif hour < 18:
        period = "下午"
    elif hour < 22:
        period = "晚上"
    else:
        period = "深夜"
    return f"{period}{hour}点{minute:02d}分"
