"""Settings and the model ladder.

Models are arranged in tiers: 0 = fast/cheap, 1 = balanced, 2 = strongest.
Prices are USD per 1M tokens and are approximate -- verify against the
providers' pricing pages and override via the MODELS env var if needed.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


class ModelSpec(BaseModel):
    model_config = {"frozen": True}

    name: str
    provider: Provider
    tier: int = Field(ge=0, le=2)
    input_per_m: float
    output_per_m: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_m + output_tokens * self.output_per_m) / 1_000_000


# Order within a tier = default preference order.
DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec(name="gemini-2.5-flash-lite", provider=Provider.GEMINI, tier=0, input_per_m=0.10, output_per_m=0.40),
    ModelSpec(name="gpt-4o-mini", provider=Provider.OPENAI, tier=0, input_per_m=0.15, output_per_m=0.60),
    ModelSpec(name="gemini-2.5-flash", provider=Provider.GEMINI, tier=1, input_per_m=0.30, output_per_m=2.50),
    ModelSpec(name="gpt-4.1-mini", provider=Provider.OPENAI, tier=1, input_per_m=0.40, output_per_m=1.60),
    ModelSpec(name="gemini-2.5-pro", provider=Provider.GEMINI, tier=2, input_per_m=1.25, output_per_m=10.00),
    ModelSpec(name="gpt-4.1", provider=Provider.OPENAI, tier=2, input_per_m=2.00, output_per_m=8.00),
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    classifier_mode: Literal["heuristic", "llm", "hybrid"] = "hybrid"
    confidence_threshold: float = 0.7

    request_timeout_s: float = 30.0
    max_retries_per_model: int = 1
    retry_backoff_s: float = 0.5
    breaker_failure_threshold: int = 3
    breaker_cooldown_s: float = 30.0
    default_max_tokens: int = 1024

    models: list[ModelSpec] = Field(default_factory=lambda: list(DEFAULT_MODELS))

    def key_for(self, provider: Provider) -> str | None:
        return self.openai_api_key if provider == Provider.OPENAI else self.gemini_api_key

    def available_models(self) -> list[ModelSpec]:
        """Models whose provider has an API key configured (config order preserved)."""
        return [m for m in self.models if self.key_for(m.provider)]


@lru_cache
def get_settings() -> Settings:
    return Settings()
