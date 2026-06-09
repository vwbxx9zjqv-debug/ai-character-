"""
OpenAI-Compatible LLM Provider — Works with OpenAI, Ollama, DeepSeek, Groq, etc.

Reference: Open-LLM-VTuber src/open_llm_vtuber/agent/stateless_llm/openai_llm.py
"""

from __future__ import annotations

import json
import logging
import re
import time

from openai import AsyncOpenAI

from core.plugin_base import LLMProvider, LLMResult, Message
from config import get_settings

logger = logging.getLogger("llm.openai_compat")


class OpenAICompatibleProvider(LLMProvider):
    """Generic OpenAI-compatible API provider.

    Set base_url to use with Ollama, LM Studio, DeepSeek, Groq, etc.
    """

    config_schema = {
        "model": {"type": "string", "default": "gpt-4o-mini"},
        "base_url": {"type": "string", "default": "https://api.openai.com/v1", "description": "API base URL for Ollama/DeepSeek etc."},
        "temperature": {"type": "float", "default": 0.8},
        "max_tokens": {"type": "int", "default": 256},
    }

    EMOTION_KEYWORDS = {
        "happy": ["开心", "太好了", "哈哈"],
        "gentle": ["温柔", "慢慢来", "没关系"],
        "sad": ["难过", "伤心", "对不起"],
        "excited": ["哇", "太棒了", "厉害"],
        "comforting": ["别担心", "会好的", "加油"],
        "neutral": [],
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.8,
        max_tokens: int = 256,
        api_key: str = "",
        **kwargs,
    ):
        settings = get_settings()
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key or "not-needed",
            base_url=base_url,
        )

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    @property
    def display_name(self) -> str:
        return f"OpenAI Compatible ({self._model})"

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

        full_system = system_prompt
        if emotion_hint:
            full_system += f"\n\nCurrent emotional direction: {emotion_hint}"

        api_messages = [{"role": "system", "content": full_system}]
        api_messages += [{"role": m.role, "content": m.content} for m in messages]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=api_messages,
            max_tokens=max_tokens or self._max_tokens,
            temperature=temperature if temperature is not None else self._temperature,
        )

        raw_text = response.choices[0].message.content or ""

        # Parse emotion (same format as Claude)
        emotion, text = self._parse_emotion(raw_text)
        duration_ms = int((time.monotonic() - start) * 1000)
        tokens = response.usage.total_tokens if response.usage else 0

        return LLMResult(
            text=text,
            emotion=emotion,
            tokens_used=tokens,
            model=self._model,
        )

    def _parse_emotion(self, raw: str) -> tuple[str, str]:
        # Structured format
        pattern = r'\{emotion:\s*"(\w+)"\s*,\s*text:\s*"(.+?)"\s*\}'
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        # JSON format
        try:
            if raw.startswith("{"):
                data = json.loads(raw)
                if "emotion" in data and "text" in data:
                    return data["emotion"], data["text"].strip()
        except json.JSONDecodeError:
            pass
        # Fallback
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            if any(kw in raw for kw in keywords):
                return emotion, raw
        return "neutral", raw

    async def validate_config(self) -> bool:
        return True  # Always valid — user provides endpoint
