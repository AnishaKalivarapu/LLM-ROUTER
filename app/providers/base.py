from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float = 0.0


class ProviderError(Exception):
    """Provider call failed. `retryable` marks transient failures (429/5xx/timeouts)."""

    def __init__(self, message: str, *, retryable: bool, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class LLMProvider(Protocol):
    name: str

    async def complete(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> Completion: ...


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    provider: str,
    timeout: float,
) -> dict[str, Any]:
    """POST and map every failure mode to a ProviderError with a retryable flag."""
    try:
        resp = await client.post(url, headers=headers, json=payload, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise ProviderError(f"{provider} request timed out", retryable=True) from exc
    except httpx.TransportError as exc:
        raise ProviderError(f"{provider} network error: {type(exc).__name__}", retryable=True) from exc

    if resp.status_code >= 400:
        retryable = resp.status_code in (408, 409, 425, 429) or resp.status_code >= 500
        raise ProviderError(
            f"{provider} HTTP {resp.status_code}: {resp.text[:200]}",
            retryable=retryable,
            status=resp.status_code,
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise ProviderError(f"{provider} returned invalid JSON", retryable=True) from exc
