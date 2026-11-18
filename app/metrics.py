from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass

from .config import ModelSpec


@dataclass
class RouteRecord:
    model: str
    provider: str
    complexity: str
    classifier: str
    preference: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    fallback_used: bool


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1))))
    return round(sorted_vals[idx], 1)


class MetricsStore:
    """Thread-safe in-memory metrics. `baseline` is the model every request would have
    hit without routing (the strongest configured model); it powers the savings estimate."""

    def __init__(self, baseline: ModelSpec | None = None, window: int = 1000):
        self._lock = threading.Lock()
        self._baseline = baseline
        self._latencies: deque[float] = deque(maxlen=window)
        self._ewma: dict[str, float] = {}
        self._by_model: dict[str, dict] = defaultdict(lambda: {"requests": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0})
        self._by_complexity: dict[str, int] = defaultdict(int)
        self._by_classifier: dict[str, int] = defaultdict(int)
        self.requests = 0
        self.failures = 0
        self.fallbacks = 0
        self.total_cost = 0.0
        self.baseline_cost = 0.0

    def record(self, r: RouteRecord) -> None:
        with self._lock:
            self.requests += 1
            self.fallbacks += int(r.fallback_used)
            self.total_cost += r.cost_usd
            if self._baseline:
                self.baseline_cost += self._baseline.cost(r.input_tokens, r.output_tokens)
            m = self._by_model[r.model]
            m["requests"] += 1
            m["cost_usd"] += r.cost_usd
            m["input_tokens"] += r.input_tokens
            m["output_tokens"] += r.output_tokens
            self._by_complexity[r.complexity] += 1
            self._by_classifier[r.classifier] += 1
            self._latencies.append(r.latency_ms)
            prev = self._ewma.get(r.model)
            self._ewma[r.model] = r.latency_ms if prev is None else 0.7 * prev + 0.3 * r.latency_ms

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1

    def avg_latency(self, model: str) -> float | None:
        with self._lock:
            return self._ewma.get(model)

    def summary(self) -> dict:
        with self._lock:
            lats = sorted(self._latencies)
            savings = None
            if self._baseline and self.baseline_cost > 0:
                savings = round(100 * (1 - self.total_cost / self.baseline_cost), 1)
            return {
                "requests": self.requests,
                "failures": self.failures,
                "fallback_requests": self.fallbacks,
                "fallback_rate": round(self.fallbacks / self.requests, 4) if self.requests else 0.0,
                "latency_ms": {"p50": _pct(lats, 0.5), "p95": _pct(lats, 0.95)},
                "total_cost_usd": round(self.total_cost, 6),
                "baseline_cost_usd": round(self.baseline_cost, 6),
                "baseline_model": self._baseline.name if self._baseline else None,
                "estimated_savings_pct": savings,
                "by_complexity": dict(self._by_complexity),
                "by_classifier": dict(self._by_classifier),
                "by_model": {k: {**v, "cost_usd": round(v["cost_usd"], 6), "ewma_latency_ms": round(self._ewma.get(k, 0), 1)} for k, v in self._by_model.items()},
            }
