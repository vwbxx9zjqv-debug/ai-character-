"""
Plugin Manager — Registry and lifecycle for all plugins.

Plugins are discovered, registered, and configured here.
The dialogue service queries the manager for the currently active
STT/LLM/TTS provider at runtime.

Pattern: Service Locator + Registry
Reference: Open-LLM-VTuber's factory pattern (asr_factory.py, tts_factory.py)
"""

from __future__ import annotations

import importlib
import logging
from typing import Type

from .plugin_base import (
    STTProvider,
    LLMProvider,
    TTSProvider,
    VoiceModelMeta,
    ProviderType,
)

logger = logging.getLogger("plugin_manager")


class PluginRegistry:
    """Central registry for all provider plugins."""

    def __init__(self):
        # Registered provider CLASSES (not instances)
        self._stt_providers: dict[str, Type[STTProvider]] = {}
        self._llm_providers: dict[str, Type[LLMProvider]] = {}
        self._tts_providers: dict[str, Type[TTSProvider]] = {}

        # Active provider INSTANCES (initialized with config)
        self._active_stt: STTProvider | None = None
        self._active_llm: LLMProvider | None = None
        self._active_tts: TTSProvider | None = None

        # Registered voice models (from all TTS engines + user uploads)
        self._voice_models: dict[str, VoiceModelMeta] = {}

        # Built-in voice discovery (populated when TTS providers register)
        self._builtin_voices_discovered: bool = False

    # ── Registration ──────────────────────────────────────────────

    def register_stt(self, provider_class: Type[STTProvider]) -> None:
        """Register an STT provider class."""
        # Instantiate temporarily to get metadata
        name = provider_class.__name__
        self._stt_providers[name] = provider_class
        logger.info(f"Registered STT provider: {name}")

    def register_llm(self, provider_class: Type[LLMProvider]) -> None:
        """Register an LLM provider class."""
        name = provider_class.__name__
        self._llm_providers[name] = provider_class
        logger.info(f"Registered LLM provider: {name}")

    def register_tts(self, provider_class: Type[TTSProvider]) -> None:
        """Register a TTS provider class."""
        name = provider_class.__name__
        self._tts_providers[name] = provider_class
        logger.info(f"Registered TTS provider: {name}")

    def register_voice_model(self, voice: VoiceModelMeta) -> None:
        """Register a voice model (from any engine or user upload)."""
        self._voice_models[voice.id] = voice

    # ── Provider Listing ──────────────────────────────────────────

    def list_stt_providers(self) -> list[dict]:
        """List all available STT providers for the admin panel."""
        return [
            {"name": name, "display_name": cls.__name__}
            for name, cls in self._stt_providers.items()
        ]

    def list_llm_providers(self) -> list[dict]:
        """List all available LLM providers."""
        return [
            {"name": name, "display_name": cls.__name__}
            for name, cls in self._llm_providers.items()
        ]

    def list_tts_providers(self) -> list[dict]:
        """List all available TTS providers."""
        return [
            {"name": name, "display_name": cls.__name__}
            for name, cls in self._tts_providers.items()
        ]

    def list_voices(self) -> list[VoiceModelMeta]:
        """List all registered voice models."""
        return list(self._voice_models.values())

    # ── Activation ────────────────────────────────────────────────

    async def activate_stt(self, provider_name: str, config: dict) -> STTProvider:
        """Instantiate and activate an STT provider with config."""
        cls = self._stt_providers.get(provider_name)
        if cls is None:
            raise ValueError(f"Unknown STT provider: {provider_name}")

        provider = cls(**config)  # type: ignore
        if not await provider.validate_config():
            raise RuntimeError(f"STT provider '{provider_name}' config validation failed")

        self._active_stt = provider
        logger.info(f"Activated STT: {provider.display_name}")
        return provider

    async def activate_llm(self, provider_name: str, config: dict) -> LLMProvider:
        """Instantiate and activate an LLM provider with config."""
        cls = self._llm_providers.get(provider_name)
        if cls is None:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

        provider = cls(**config)
        if not await provider.validate_config():
            raise RuntimeError(f"LLM provider '{provider_name}' config validation failed")

        self._active_llm = provider
        logger.info(f"Activated LLM: {provider.display_name}")
        return provider

    async def activate_tts(self, provider_name: str, config: dict) -> TTSProvider:
        """Instantiate and activate a TTS provider with config."""
        cls = self._tts_providers.get(provider_name)
        if cls is None:
            raise ValueError(f"Unknown TTS provider: {provider_name}")

        provider = cls(**config)
        if not await provider.validate_config():
            raise RuntimeError(f"TTS provider '{provider_name}' config validation failed")

        # Discover built-in voices from this engine
        try:
            voices = await provider.list_voices()
            for v in voices:
                self.register_voice_model(v)
        except Exception:
            pass  # Some engines don't support listing

        self._active_tts = provider
        logger.info(f"Activated TTS: {provider.display_name}")
        return provider

    # ── Current Provider Access ───────────────────────────────────

    @property
    def active_stt(self) -> STTProvider:
        if self._active_stt is None:
            raise RuntimeError("No STT provider activated")
        return self._active_stt

    @property
    def active_llm(self) -> LLMProvider:
        if self._active_llm is None:
            raise RuntimeError("No LLM provider activated")
        return self._active_llm

    @property
    def active_tts(self) -> TTSProvider:
        if self._active_tts is None:
            raise RuntimeError("No TTS provider activated")
        return self._active_tts

    def get_voice(self, voice_id: str) -> VoiceModelMeta | None:
        return self._voice_models.get(voice_id)

    # ── Auto-discovery ────────────────────────────────────────────

    def discover_builtin_providers(self) -> None:
        """Auto-discover and register all built-in providers from plugins/."""
        # STT providers
        from plugins.stt.whisper_api import WhisperAPIProvider
        self.register_stt(WhisperAPIProvider)

        # LLM providers
        from plugins.llm.claude import ClaudeProvider
        self.register_llm(ClaudeProvider)

        from plugins.llm.openai_compatible import OpenAICompatibleProvider
        self.register_llm(OpenAICompatibleProvider)

        # TTS providers
        from plugins.tts.edge_tts import EdgeTTSProvider
        self.register_tts(EdgeTTSProvider)

        from plugins.tts.cosyvoice import CosyVoice2Provider
        self.register_tts(CosyVoice2Provider)

        from plugins.tts.gpt_sovits import GPTSovitsProvider
        self.register_tts(GPTSovitsProvider)

        from plugins.tts.custom_endpoint import CustomEndpointProvider
        self.register_tts(CustomEndpointProvider)

        # User-upload voice models are loaded from DB at startup
        logger.info(
            f"Discovered providers — "
            f"STT: {list(self._stt_providers.keys())}, "
            f"LLM: {list(self._llm_providers.keys())}, "
            f"TTS: {list(self._tts_providers.keys())}"
        )


# ── Global singleton ──────────────────────────────────────────────

_registry: PluginRegistry | None = None


def get_plugin_registry() -> PluginRegistry:
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry
