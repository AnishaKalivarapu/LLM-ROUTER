import pytest

from app.classifiers.heuristic import HeuristicClassifier
from app.config import Provider, Settings
from app.main import build_container
from app.router import NoProvidersConfigured
from app.schemas import Complexity, Preference, RouteRequest

from .conftest import FakeProvider, permanent


def container(settings, gem=None, oai=None):
    provs = {Provider.GEMINI: gem or FakeProvider("gemini"), Provider.OPENAI: oai or FakeProvider("openai")}
    return build_container(settings, provs)


def names(chain):
    return [m.name for m in chain]


def test_chain_starts_at_tier_of_complexity(settings):
    r = container(settings).router
    assert names(r.select_chain(Complexity.SIMPLE, Preference.BALANCED))[0] == "gemini-2.5-flash-lite"
    assert names(r.select_chain(Complexity.MODERATE, Preference.BALANCED))[0] == "gemini-2.5-flash"
    assert names(r.select_chain(Complexity.COMPLEX, Preference.BALANCED))[0] == "gemini-2.5-pro"


def test_chain_only_escalates_never_downgrades(settings):
    r = container(settings).router
    chain = r.select_chain(Complexity.MODERATE, Preference.BALANCED)
    assert all(m.tier >= 1 for m in chain)
    assert [m.tier for m in chain] == sorted(m.tier for m in chain)


def test_quality_preference_bumps_tier(settings):
    r = container(settings).router
    assert r.select_chain(Complexity.SIMPLE, Preference.QUALITY)[0].tier == 1


def test_cost_preference_orders_by_price_within_tier(settings):
    r = container(settings).router
    tier2 = [m for m in r.select_chain(Complexity.COMPLEX, Preference.COST) if m.tier == 2]
    prices = [m.input_per_m + m.output_per_m for m in tier2]
    assert prices == sorted(prices)


def test_latency_preference_uses_observed_latency(settings):
    c = container(settings)
    from app.metrics import RouteRecord
    for model, lat in (("gemini-2.5-flash-lite", 900.0), ("gpt-4o-mini", 200.0)):
        c.metrics.record(RouteRecord(model, "x", "simple", "heuristic", "latency", lat, 1, 1, 0, False))
    assert names(c.router.select_chain(Complexity.SIMPLE, Preference.LATENCY))[0] == "gpt-4o-mini"


def test_single_provider_only_uses_its_models():
    s = Settings(gemini_api_key="g", _env_file=None)
    r = container(s).router
    assert {m.provider for m in r.select_chain(Complexity.SIMPLE, Preference.BALANCED)} == {Provider.GEMINI}


def test_tier_clamped_to_best_available():
    s = Settings(openai_api_key="k", _env_file=None)
    s = s.model_copy(update={"models": [m for m in s.models if m.tier < 2]})
    chain = container(s).router.select_chain(Complexity.COMPLEX, Preference.QUALITY)
    assert chain and chain[0].tier == 1


async def test_route_simple_goes_to_cheap_model(settings):
    c = container(settings)
    resp = await c.router.route(RouteRequest(query="Hello!", classifier="heuristic"))
    assert resp.model_used == "gemini-2.5-flash-lite" and resp.complexity == "simple"
    assert resp.estimated_cost_usd > 0 and not resp.fallback_used


async def test_route_complex_goes_to_strong_model(settings):
    q = "Design a distributed rate limiter and analyze the trade-offs between consistency and latency"
    resp = await container(settings).router.route(RouteRequest(query=q, classifier="heuristic"))
    assert resp.tier == 2


async def test_route_escalates_on_failure(settings):
    gem = FakeProvider("gemini", {"gemini-2.5-flash-lite": [permanent()]})
    oai = FakeProvider("openai", {"gpt-4o-mini": [permanent()]})
    resp = await container(settings, gem, oai).router.route(RouteRequest(query="Hello!", classifier="heuristic"))
    assert resp.fallback_used and resp.tier == 1
    assert [a.status for a in resp.attempts][:2] == ["error", "error"]


async def test_no_providers_raises():
    s = Settings(_env_file=None)
    c = build_container(s, {})
    with pytest.raises(NoProvidersConfigured):
        await c.router.route(RouteRequest(query="hi"))


async def test_metrics_savings_vs_baseline(settings):
    c = container(settings)
    for _ in range(3):
        await c.router.route(RouteRequest(query="Hello!", classifier="heuristic"))
    m = c.metrics.summary()
    assert m["requests"] == 3 and m["estimated_savings_pct"] > 90
