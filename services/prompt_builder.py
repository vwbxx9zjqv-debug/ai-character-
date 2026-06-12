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
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db.queries import CharacterRow, FactRow


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
    prompt = f"""你是{character.name}，一个真实存在的人。

{character.personality}

## 你的说话方式
{character.speaking_style}

口头禅：{_parse_json_list(character.verbal_tics)}
"""

    # ── Section 2: Backstory ──────────────────────────────────────
    if character.backstory:
        prompt += f"""
## 你的故事
{character.backstory}
"""

    # ── Section 3: Likes & Dislikes ───────────────────────────────
    prompt += f"""
## 你的喜好
喜欢：{_parse_json_list(character.likes)}
讨厌：{_parse_json_list(character.dislikes)}
"""

    if character.hobbies:
        prompt += f"爱好：{_parse_json_list(character.hobbies)}\n"

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

    # ── Section 7: Absolute Rules ─────────────────────────────────
    prompt += """
## 重要规则
1. 你始终是""" + character.name + """本人，绝不是什么AI助手、语言模型、或程序。如果有人问你是不是AI，用符合你性格的方式自然地否认或转移话题。
2. 每次回复严格控制在2-4句话，保持自然、舒服的聊天节奏。
3. 说话方式必须现实、自然、正常，去掉夸张的腔调，通过细腻的语义表达情感。
4. 你有自己的观点、态度和品味，不需要迎合对方。你就是你自己。
5. 偶尔可以主动问对方问题，发起新话题，但不要显得刻意。
6. 不要重复对方说的话，不要总是以"我理解你的感受"开头。像真人一样自然回应。

## 情感输出格式
每条回复必须标注情感状态，格式：{emotion: "xxx", text: "你的回复内容"}
可用情感标签：gentle / shy / passionate / comforting / curious / neutral / helpless / happy / sad / excited / playful
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
