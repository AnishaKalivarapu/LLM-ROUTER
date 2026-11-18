import pytest

from app.config import Provider
from app.executor import AllModelsFailed, CircuitBreaker, Executor

from .conftest import FakeProvider, permanent, transient


def chain(settings, *names):
    by_name = {m.name: m for m in settings.models}
    return [by_name[n] for n in names]


def executor(settings, gem=None, oai=None, no_sleep=None, breaker=None):
    return Executor(
        settings,
        {Provider.GEMINI: gem or FakeProvider("gemini"), Provider.OPENAI: oai or FakeProvider("openai")},
        breaker=breaker,
        sleep=no_sleep,
    )


async def run(ex, c):
    return await ex.run(c, "hi", system=None, max_tokens=50, temperature=0)


async def test_first_model_succeeds(settings, no_sleep):
    ex = executor(settings, no_sleep=no_sleep)
    r = await run(ex, chain(settings, "gemini-2.5-flash-lite", "gemini-2.5-flash"))
    assert r.model.name == "gemini-2.5-flash-lite"
    assert not r.fallback_used


async def test_transient_error_retried_same_model(settings, no_sleep):
    gem = FakeProvider("gemini", {"gemini-2.5-flash-lite": [transient(), "ok after retry"]})
    ex = executor(settings, gem=gem, no_sleep=no_sleep)
    r = await run(ex, chain(settings, "gemini-2.5-flash-lite", "gemini-2.5-flash"))
    assert r.completion.text == "ok after retry"
    assert r.model.name == "gemini-2.5-flash-lite"
    assert not r.fallback_used
    assert [a.status for a in r.attempts] == ["error", "success"]


async def test_permanent_error_escalates_immediately(settings, no_sleep):
    gem = FakeProvider("gemini", {"gemini-2.5-flash-lite": [permanent()]})
    ex = executor(settings, gem=gem, no_sleep=no_sleep)
    r = await run(ex, chain(settings, "gemini-2.5-flash-lite", "gemini-2.5-flash"))
    assert r.model.name == "gemini-2.5-flash"
    assert r.fallback_used
    assert gem.calls == ["gemini-2.5-flash-lite", "gemini-2.5-flash"]  # no retry on 400


async def test_exhausted_retries_then_escalate(settings, no_sleep):
    gem = FakeProvider("gemini", {"gemini-2.5-flash-lite": [transient()]})  # always fails
    ex = executor(settings, gem=gem, no_sleep=no_sleep)
    r = await run(ex, chain(settings, "gemini-2.5-flash-lite", "gemini-2.5-flash"))
    assert r.fallback_used
    assert gem.calls.count("gemini-2.5-flash-lite") == 2  # 1 try + 1 retry


async def test_cross_provider_fallback(settings, no_sleep):
    gem = FakeProvider("gemini", {"gemini-2.5-flash-lite": [permanent()]})
    ex = executor(settings, gem=gem, no_sleep=no_sleep)
    r = await run(ex, chain(settings, "gemini-2.5-flash-lite", "gpt-4o-mini"))
    assert r.model.provider == Provider.OPENAI and r.fallback_used


async def test_empty_response_escalates(settings, no_sleep):
    gem = FakeProvider("gemini", {"gemini-2.5-flash-lite": ["   "]})
    ex = executor(settings, gem=gem, no_sleep=no_sleep)
    r = await run(ex, chain(settings, "gemini-2.5-flash-lite", "gemini-2.5-flash"))
    assert r.model.name == "gemini-2.5-flash"


async def test_all_fail_raises_with_trace(settings, no_sleep):
    gem = FakeProvider("gemini", {"gemini-2.5-flash-lite": [permanent()], "gemini-2.5-flash": [permanent()]})
    ex = executor(settings, gem=gem, no_sleep=no_sleep)
    with pytest.raises(AllModelsFailed) as e:
        await run(ex, chain(settings, "gemini-2.5-flash-lite", "gemini-2.5-flash"))
    assert [a.model for a in e.value.attempts] == ["gemini-2.5-flash-lite", "gemini-2.5-flash"]


async def test_circuit_breaker_skips_then_recovers(settings, no_sleep):
    now = [0.0]
    breaker = CircuitBreaker(threshold=2, cooldown_s=30, clock=lambda: now[0])
    gem = FakeProvider("gemini", {"gemini-2.5-flash-lite": [permanent()]})
    ex = executor(settings, gem=gem, no_sleep=no_sleep, breaker=breaker)
    c = chain(settings, "gemini-2.5-flash-lite", "gemini-2.5-flash")

    await run(ex, c)
    await run(ex, c)  # 2nd failure opens the breaker
    calls_before = gem.calls.count("gemini-2.5-flash-lite")
    r = await run(ex, c)
    assert gem.calls.count("gemini-2.5-flash-lite") == calls_before  # skipped
    assert r.attempts[0].status == "skipped" and r.fallback_used

    now[0] = 31  # cooldown elapsed -> half-open probe
    await run(ex, c)
    assert gem.calls.count("gemini-2.5-flash-lite") == calls_before + 1


async def test_if_everything_is_circuit_open_we_still_try(settings, no_sleep):
    breaker = CircuitBreaker(threshold=1, cooldown_s=999)
    ex = executor(settings, no_sleep=no_sleep, breaker=breaker)
    c = chain(settings, "gemini-2.5-flash-lite")
    breaker.record_failure("gemini-2.5-flash-lite")
    r = await run(ex, c)
    assert r.model.name == "gemini-2.5-flash-lite"
