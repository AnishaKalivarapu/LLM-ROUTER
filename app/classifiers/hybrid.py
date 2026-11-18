from __future__ import annotations

from ..schemas import ClassificationResult
from .heuristic import HeuristicClassifier
from .llm import LLMClassifier


class HybridClassifier:
    """Heuristic first (free). Only if its confidence is below the threshold do we pay
    for an LLM call. If the LLM path fails we keep the heuristic answer."""

    name = "hybrid"

    def __init__(self, heuristic: HeuristicClassifier, llm: LLMClassifier, threshold: float):
        self.heuristic = heuristic
        self.llm = llm
        self.threshold = threshold

    async def classify(self, query: str) -> ClassificationResult:
        h = self.heuristic.classify_sync(query)
        if h.confidence >= self.threshold:
            h.classifier = "hybrid:heuristic"
            return h
        try:
            result = await self.llm.classify(query)
        except Exception as exc:
            h.classifier = "hybrid:heuristic-fallback"
            h.reason = f"LLM classifier unavailable ({type(exc).__name__}); used heuristic"
            return h
        result.classifier = "hybrid:llm"
        result.score = h.score        # keep heuristic evidence for debugging
        result.signals = h.signals
        return result
