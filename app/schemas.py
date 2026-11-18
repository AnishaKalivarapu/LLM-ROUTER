from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Literal

from pydantic import BaseModel, Field


class Complexity(IntEnum):
    SIMPLE = 0
    MODERATE = 1
    COMPLEX = 2

    @property
    def label(self) -> str:
        return self.name.lower()


class Preference(str, Enum):
    BALANCED = "balanced"  # follow the classifier
    COST = "cost"          # cheapest model within the chosen tier
    LATENCY = "latency"    # fastest observed model within the chosen tier
    QUALITY = "quality"    # bump up one tier


ClassifierName = Literal["heuristic", "llm", "hybrid"]


@dataclass
class ClassificationResult:
    level: Complexity
    score: float
    confidence: float
    classifier: str
    signals: dict[str, float] = field(default_factory=dict)
    reason: str | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0


class ClassificationInfo(BaseModel):
    complexity: str
    confidence: float
    classifier: str
    heuristic_score: float
    signals: dict[str, float]
    reason: str | None = None
    latency_ms: float
    cost_usd: float


class RouteRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    preference: Preference = Preference.BALANCED
    classifier: ClassifierName | None = None
    system_prompt: str | None = Field(default=None, max_length=4_000)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class ClassifyRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    classifier: ClassifierName | None = None


class Attempt(BaseModel):
    model: str
    provider: str
    status: Literal["success", "error", "skipped"]
    error: str | None = None
    retryable: bool = False
    latency_ms: float = 0.0


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class RouteResponse(BaseModel):
    answer: str
    model_used: str
    provider: str
    tier: int
    complexity: str
    classification: ClassificationInfo
    fallback_used: bool
    attempts: list[Attempt]
    usage: TokenUsage
    estimated_cost_usd: float
    generation_latency_ms: float
    total_latency_ms: float
