"""
Conversation Logger — Capability plugin that persists conversations.

Subscribes to dialogue events and logs them. This is the simplest
example of a capability plugin. Phase 2 plugins follow the same pattern.
"""

from __future__ import annotations

import logging

from core.event_bus import (
    get_event_bus,
    SpeechEnded, TranscriptionReady, LLMResponseReady, DialogueCompleted,
)

logger = logging.getLogger("capability.conversation_logger")


class ConversationLogger:
    """Logs all dialogue events for debugging and analytics."""

    name = "conversation_logger"
    is_enabled = True

    def __init__(self):
        self.bus = get_event_bus()

    async def on_enable(self) -> None:
        self.bus.on_any(self._log_event)
        logger.info("ConversationLogger enabled")

    async def on_disable(self) -> None:
        # EventBus doesn't support unsubscribing individual handlers in v1.
        # For now, we just mark disabled.
        self.is_enabled = False
        logger.info("ConversationLogger disabled")

    async def _log_event(self, event) -> None:
        if not self.is_enabled:
            return
        event_name = type(event).__name__
        device = getattr(event, "device_id", "unknown")
        logger.debug(f"[{device}] Event: {event_name}")
