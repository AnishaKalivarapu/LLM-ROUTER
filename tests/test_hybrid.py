from app.classifiers.heuristic import HeuristicClassifier
from app.classifiers.hybrid import HybridClassifier
from app.classifiers.llm import LLMClassifier, parse_llm_label
from app.executor import Executor
from app.providers.base import ProviderError
from app.schemas import Complexity

from .conftest import FakeProvider, permanent
from app.config import Provider


def make(settings, gem_script=None, openai_script=None, threshold=0.7):
    gem, oai = FakeProvider("gemini", gem_script), FakeProvider("openai", openai_script)
    ex = Executor(settings, {Provider.GEMINI: gem, Provider.OPENAI: oai})
    llm = LLMClassifier(settings, ex)
    return HybridClassifier(HeuristicClassifier(), llm, threshold), gem, oai


async def test_confident_heuristic_skips_llm(settings):
    hybrid, gem, oai = make(settings)
    r = await hybrid.classify("Hello!")
    assert r.classifier == "hybrid:heuristic"
    assert gem.calls == [] and oai.calls == []
    assert r.cost_usd == 0


async def test_low_confidence_calls_llm_on_tier0_only(settings):
    hybrid, gem, oai = make(settings, gem_script={"gemini-2.5-flash-lite": ['{"complexity": "moderate", "reason": "explanation"}']})
    r = await hybrid.classify("Explain how photosynthesis works")
    assert r.classifier == "hybrid:llm"
    assert r.level == Complexity.MODERATE
    assert r.cost_usd > 0
    assert gem.calls == ["gemini-2.5-flash-lite"]  # cheapest model only


async def test_llm_failure_falls_back_to_heuristic(settings):
    hybrid, gem, oai = make(
        settings,
        gem_script={"gemini-2.5-flash-lite": [permanent()]},
        openai_script={"gpt-4o-mini": [permanent()]},
    )
    r = await hybrid.classify("Explain how photosynthesis works")
    assert r.classifier == "hybrid:heuristic-fallback"
    assert "unavailable" in r.reason


async def test_garbage_llm_output_falls_back(settings):
    hybrid, *_ = make(settings, gem_script={"gemini-2.5-flash-lite": ["no idea"]})
    r = await hybrid.classify("Explain how photosynthesis works")
    assert r.classifier == "hybrid:heuristic-fallback"


def test_parse_variants():
    assert parse_llm_label('{"complexity": "complex", "reason": "x"}')[0] == Complexity.COMPLEX
    assert parse_llm_label('```json\n{"complexity":"simple"}\n```')[0] == Complexity.SIMPLE
    assert parse_llm_label("I think this is moderate.")[0] == Complexity.MODERATE
