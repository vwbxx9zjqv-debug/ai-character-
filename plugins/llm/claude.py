"""
Claude LLM Provider — Anthropic Claude API.

Reference: Open-LLM-VTuber src/open_llm_vtuber/agent/stateless_llm/claude_llm.py
"""

from __future__ import annotations

import json
import logging
import re
import time

from anthropic import AsyncAnthropic

from core.plugin_base import LLMProvider, LLMResult, Message
from config import get_settings

logger = logging.getLogger("llm.claude")


class ClaudeProvider(LLMProvider):
    """LLM provider using Anthropic Claude API.

    Supports emotion extraction: the LLM response is parsed for
    {emotion: "xxx", text: "xxx"} format, or falls back to
    emotion keyword detection.
    """

    config_schema = {
        "model": {"type": "string", "default": "claude-haiku-4-5-20251001"},
        "temperature": {"type": "float", "default": 0.8},
        "max_tokens": {"type": "int", "default": 256},
    }

    # Emotion keywords for fallback detection
    EMOTION_KEYWORDS = {
        "happy": ["开心", "太好了", "哈哈", "真好", "棒", "高兴", "嘻嘻", "耶"],
        "gentle": ["温柔", "慢慢来", "没关系", "没事", "陪", "安心"],
        "sad": ["难过", "伤心", "对不起", "遗憾", "唉", "可惜"],
        "excited": ["哇", "太棒了", "厉害", "太强了", "！！！", "天啊"],
        "comforting": ["别担心", "会好的", "加油", "支持你", "没关系"],
        "curious": ["为什么", "怎么", "真的吗", "说说", "讲讲", "有意思"],
        "shy": ["不好意思", "害羞", "脸红", "那个……"],
        "playful": ["略略", "骗你的", "开玩笑", "逗你"],
    }

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.8,
        max_tokens: int = 256,
        **kwargs,
    ):
        settings = get_settings()
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    @property
    def provider_name(self) -> str:
        return "claude"

    @property
    def display_name(self) -> str:
        return f"Claude ({self._model})"

    async def chat(
        self,
        messages: list[Message],
        system_prompt: str = "",
        emotion_hint: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResult:
        start = time.monotonic()

        # Build system prompt with emotion guidance
        full_system = system_prompt
        if emotion_hint:
            full_system += f"\n\n当前情绪倾向：{emotion_hint}"

        # Convert messages to Claude format
        claude_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]

        response = await self._client.messages.create(
            model=self._model,
            system=full_system,
            messages=claude_messages,
            max_tokens=max_tokens or self._max_tokens,
            temperature=temperature if temperature is not None else self._temperature,
        )

        # Extract text
        raw_text = ""
        for block in response.content:
            if block.type == "text":
                raw_text += block.text

        raw_text = raw_text.strip()

        # Parse emotion from response
        emotion, text = self._parse_emotion(raw_text)

        duration_ms = int((time.monotonic() - start) * 1000)
        tokens = response.usage.input_tokens + response.usage.output_tokens if response.usage else 0

        logger.debug(f"Claude response: emotion={emotion}, text='{text[:50]}...', tokens={tokens}")

        return LLMResult(
            text=text,
            emotion=emotion,
            emotion_intensity=0.6,
            tokens_used=tokens,
            model=self._model,
        )

    def _parse_emotion(self, raw: str) -> tuple[str, str]:
        """Parse emotion-tagged responses. Supports multiple formats."""
        # Format 1: {emotion: "xxx", text: "..."}
        pattern1 = r'\{emotion:\s*"(\w+)"\s*,\s*text:\s*"(.+?)"\s*\}'
        match = re.search(pattern1, raw, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()

        # Format 2: {emotion: "xxx"} text
        pattern2 = r'\{emotion:\s*"(\w+)"\}\s*(.+)'
        match = re.search(pattern2, raw, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()

        # Format 3: {emotion: xxx} text (no quotes)
        pattern3 = r'\{emotion:\s*(\w+)\}\s*(.+)'
        match = re.search(pattern3, raw, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()

        # Format 4: JSON {"emotion": "xxx", "text": "xxx"}
        try:
            if raw.startswith("{"):
                data = json.loads(raw)
                if "emotion" in data and "text" in data:
                    return data["emotion"], data["text"].strip()
        except json.JSONDecodeError:
            pass

        # Format 5: Generic strip — remove any {emotion:...} wrapper, keep the text
        emo_match = re.search(r'\{emotion:\s*"?(\w+)"?', raw)
        emo = emo_match.group(1) if emo_match else "neutral"
        clean = re.sub(
            r'\s*\{emotion:\s*"?\w+"?\s*(?:,\s*text:\s*"?(.*?)"?)?\s*\}\s*',
            r'\1', raw, flags=re.DOTALL
        ).strip()
        if clean and clean != raw:
            return emo, clean

        # Fallback: keyword detection
        logger.debug(f"Emotion parse fallback (keyword): {raw[:100]}")
        detected = self._detect_emotion(raw)
        return detected, raw

    def _detect_emotion(self, text: str) -> str:
        """Simple keyword-based emotion detection."""
        scores: dict[str, int] = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            scores[emotion] = sum(1 for kw in keywords if kw in text)
        if max(scores.values(), default=0) > 0:
            return max(scores, key=scores.get)  # type: ignore
        return "neutral"

    async def validate_config(self) -> bool:
        settings = get_settings()
        return bool(settings.anthropic_api_key)
