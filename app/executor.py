"""Executes a prompt against an ordered chain of models with retries,
a circuit breaker, and escalation to the next (more capable) model on failure."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from .config import ModelSpec, Provider, Settings
from .providers.base import Completion, LLMProvider, ProviderError
from .schemas import Attempt


class AllModelsFailed(Exception):
    def __init__(self, attempts: list[Attempt]):
        super().__init__("all candidate models failed")
        self.attempts = attempts
        self.classification = None  # filled in by the router


@dataclass
class ExecutionResult:
    completion: Completion
    model: ModelSpec
    attempts: list[Attempt]

    @property
    def fallback_used(self) -> bool:
        """True if the answering model differs from the first one we tried/considered."""
        return self.model.name != self.attempts[0].model


class CircuitBreaker:
    """Per-model breaker: N consecutive failures -> skip the model for a cooldown."""

    def __init__(self, threshold: int, cooldown_s: float, clock: Callable[[], float] = time.monotonic):
        self._threshold = threshold
        self._cooldown = cooldown_s
        self._clock = clock
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def allow(self, model: str) -> bool:
        opened = self._opened_at.get(model)
        if opened is None:
            return True
        if self._clock() - opened >= self._cooldown:  # half-open: let one probe through
            del self._opened_at[model]
            self._failures[model] = self._threshold - 1
            return True
        return False

    def record_success(self, model: str) -> None:
        self._failures.pop(model, None)
        self._opened_at.pop(model, None)

    def record_failure(self, model: str) -> None:
        n = self._failures.get(model, 0) + 1
        self._failures[model] = n
        if n >= self._threshold:
            self._opened_at[model] = self._clock()

    def open_models(self) -> list[str]:
        return [m for m in list(self._opened_at) if not self.allow(m)]


class Executor:
    def __init__(
        self,
        settings: Settings,
        providers: dict[Provider, LLMProvider],
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.settings = settings
        self.providers = providers
        self.breaker = breaker or CircuitBreaker(settings.breaker_failure_threshold, settings.breaker_cooldown_s)
        self._sleep = sleep

    async def run(
        self,
        chain: list[ModelSpec],
        prompt: str,
        *,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> ExecutionResult:
        attempts: list[Attempt] = []
        # Pass 1 honours the circuit breaker. If it would skip *everything*, pass 2 tries anyway.
        for ignore_breaker in (False, True):
            tried_any = False
            for spec in chain:
                if not ignore_breaker and not self.breaker.allow(spec.name):
                    attempts.append(
                        Attempt(model=spec.name, provider=spec.provider.value, status="skipped", error="circuit open")
                    )
                    continue
                tried_any = True
                completion = await self._try_model(spec, attempts, prompt, system, max_tokens, temperature)
                if completion is not None:
                    return ExecutionResult(completion=completion, model=spec, attempts=attempts)
            if tried_any:
                break
        raise AllModelsFailed(attempts)

    async def _try_model(self, spec, attempts, prompt, system, max_tokens, temperature) -> Completion | None:
        provider = self.providers.get(spec.provider)
        if provider is None:
            attempts.append(
                Attempt(model=spec.name, provider=spec.provider.value, status="error", error="provider not configured")
            )
            return None

        for i in range(self.settings.max_retries_per_model + 1):
            t0 = time.perf_counter()
            try:
                completion = await provider.complete(
                    spec.name, prompt, system=system, max_tokens=max_tokens, temperature=temperature
                )
                if not completion.text.strip():
                    raise ProviderError("empty response", retryable=False)
            except ProviderError as exc:
                err = exc
            except Exception as exc:  # defensive: never let an adapter bug take down the request
                err = ProviderError(f"unexpected {type(exc).__name__}: {exc}", retryable=False)
            else:
                completion.latency_ms = (time.perf_counter() - t0) * 1000
                attempts.append(
                    Attempt(
                        model=spec.name,
                        provider=spec.provider.value,
                        status="success",
                        latency_ms=round(completion.latency_ms, 1),
                    )
                )
                self.breaker.record_success(spec.name)
                return completion

            attempts.append(
                Attempt(
                    model=spec.name,
                    provider=spec.provider.value,
                    status="error",
                    error=str(err),
                    retryable=err.retryable,
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
            )
            if err.retryable and i < self.settings.max_retries_per_model:
                await self._sleep(self.settings.retry_backoff_s * (2**i))
                continue
            break

        self.breaker.record_failure(spec.name)
        return None
