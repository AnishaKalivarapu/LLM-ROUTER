from __future__ import annotations

import time

from .classifiers.heuristic import HeuristicClassifier
from .config import ModelSpec, Settings
from .executor import AllModelsFailed, Executor
from .metrics import MetricsStore, RouteRecord
from .schemas import (
    ClassificationInfo,
    ClassificationResult,
    Complexity,
    Preference,
    RouteRequest,
    RouteResponse,
    TokenUsage,
)


class NoProvidersConfigured(Exception):
    pass


class Router:
    """classify -> pick starting tier -> build escalation chain -> execute -> record."""

    def __init__(self, settings: Settings, executor: Executor, classifiers: dict, metrics: MetricsStore):
        self.settings = settings
        self.executor = executor
        self.classifiers = classifiers
        self.metrics = metrics
        self._heuristic = HeuristicClassifier()

    # ---- model selection -------------------------------------------------
    def select_chain(self, level: Complexity, preference: Preference, available: list[ModelSpec] | None = None) -> list[ModelSpec]:
        available = available if available is not None else self.settings.available_models()
        if not available:
            raise NoProvidersConfigured("set OPENAI_API_KEY and/or GEMINI_API_KEY")

        tier = int(level) + (1 if preference == Preference.QUALITY else 0)
        tier = min(tier, max(m.tier for m in available))

        def order_key(item: tuple[int, ModelSpec]):
            idx, m = item
            if preference == Preference.COST:
                return (m.tier, m.input_per_m + m.output_per_m, idx)
            if preference == Preference.LATENCY:
                lat = self.metrics.avg_latency(m.name)
                return (m.tier, lat if lat is not None else 0.0, idx)  # unknown latency -> explore first
            return (m.tier, idx)

        candidates = [(i, m) for i, m in enumerate(available) if m.tier >= tier]
        return [m for _, m in sorted(candidates, key=order_key)]

    # ---- classification --------------------------------------------------
    async def classify(self, query: str, name: str | None = None) -> ClassificationResult:
        name = name or self.settings.classifier_mode
        t0 = time.perf_counter()
        try:
            result = await self.classifiers[name].classify(query)
        except Exception as exc:  # classifier problems must never fail the user's request
            result = self._heuristic.classify_sync(query)
            result.classifier = "heuristic-fallback"
            result.reason = f"{name} classifier failed ({type(exc).__name__}); used heuristic"
        if not result.latency_ms:
            result.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return result

    # ---- main entrypoint -------------------------------------------------
    async def route(self, req: RouteRequest) -> RouteResponse:
        t0 = time.perf_counter()
        available = self.settings.available_models()
        if not available:
            raise NoProvidersConfigured("set OPENAI_API_KEY and/or GEMINI_API_KEY")

        cls = await self.classify(req.query, req.classifier)
        chain = self.select_chain(cls.level, req.preference, available)

        try:
            result = await self.executor.run(
                chain,
                req.query,
                system=req.system_prompt,
                max_tokens=req.max_tokens or self.settings.default_max_tokens,
                temperature=req.temperature,
            )
        except AllModelsFailed as exc:
            self.metrics.record_failure()
            exc.classification = cls
            raise

        c = result.completion
        answer_cost = result.model.cost(c.input_tokens, c.output_tokens)
        total_cost = answer_cost + cls.cost_usd
        self.metrics.record(
            RouteRecord(
                model=result.model.name,
                provider=result.model.provider.value,
                complexity=cls.level.label,
                classifier=cls.classifier,
                preference=req.preference.value,
                latency_ms=(time.perf_counter() - t0) * 1000,
                input_tokens=c.input_tokens,
                output_tokens=c.output_tokens,
                cost_usd=total_cost,
                fallback_used=result.fallback_used,
            )
        )
        return RouteResponse(
            answer=c.text,
            model_used=result.model.name,
            provider=result.model.provider.value,
            tier=result.model.tier,
            complexity=cls.level.label,
            classification=self._info(cls),
            fallback_used=result.fallback_used,
            attempts=result.attempts,
            usage=TokenUsage(input_tokens=c.input_tokens, output_tokens=c.output_tokens, total_tokens=c.input_tokens + c.output_tokens),
            estimated_cost_usd=round(total_cost, 8),
            generation_latency_ms=round(c.latency_ms, 1),
            total_latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    @staticmethod
    def _info(cls: ClassificationResult) -> ClassificationInfo:
        return ClassificationInfo(
            complexity=cls.level.label,
            confidence=cls.confidence,
            classifier=cls.classifier,
            heuristic_score=cls.score,
            signals=cls.signals,
            reason=cls.reason,
            latency_ms=cls.latency_ms,
            cost_usd=round(cls.cost_usd, 8),
        )
