from __future__ import annotations

import httpx

from ..config import Provider, Settings
from .base import LLMProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider


def build_providers(settings: Settings, client: httpx.AsyncClient) -> dict[Provider, LLMProvider]:
    providers: dict[Provider, LLMProvider] = {}
    if settings.openai_api_key:
        providers[Provider.OPENAI] = OpenAIProvider(
            settings.openai_api_key, settings.openai_base_url, client, settings.request_timeout_s
        )
    if settings.gemini_api_key:
        providers[Provider.GEMINI] = GeminiProvider(
            settings.gemini_api_key, settings.gemini_base_url, client, settings.request_timeout_s
        )
    return providers
