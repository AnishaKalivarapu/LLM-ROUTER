from __future__ import annotations

import pytest

from app.config import ModelSpec, Provider, Settings
from app.providers.base import Completion, ProviderError


class FakeProvider:
    """Scripted provider. `script[model]` is a list of outcomes consumed in order;
    an exception instance is raised, a str is returned as the answer. The last
    outcome repeats once the list is exhausted."""

    def __init__(self, name: str, script: dict[str, list] | None = None):
        self.name = name
        self.script = script or {}
        self.calls: list[str] = []

    async def complete(self, model, prompt, *, system, max_tokens, temperature):
        self.calls.append(model)
        outcomes = self.script.get(model, [f"answer from {model}"])
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return Completion(text=outcome, input_tokens=100, output_tokens=50)


def transient() -> ProviderError:
    return ProviderError("HTTP 503", retryable=True, status=503)


def permanent() -> ProviderError:
    return ProviderError("HTTP 400", retryable=False, status=400)


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key="sk-test", gemini_api_key="g-test", retry_backoff_s=0, _env_file=None)


@pytest.fixture
def no_sleep():
    async def _sleep(_):  # keep retry tests instant
        return None
    return _sleep
