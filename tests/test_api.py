from fastapi.testclient import TestClient

from app.config import Provider, Settings
from app.main import create_app

from .conftest import FakeProvider, permanent


def client(settings, gem=None, oai=None):
    provs = {Provider.GEMINI: gem or FakeProvider("gemini"), Provider.OPENAI: oai or FakeProvider("openai")}
    return TestClient(create_app(settings, provs))


def test_route_and_metrics(settings):
    with client(settings) as c:
        r = c.post("/v1/route", json={"query": "Hello!", "classifier": "heuristic"})
        assert r.status_code == 200
        body = r.json()
        assert body["model_used"] == "gemini-2.5-flash-lite" and body["complexity"] == "simple"
        assert body["classification"]["classifier"] == "heuristic"
        m = c.get("/v1/metrics").json()
        assert m["requests"] == 1 and m["by_model"]["gemini-2.5-flash-lite"]["requests"] == 1


def test_classify_endpoint(settings):
    with client(settings) as c:
        r = c.post("/v1/classify", json={"query": "Prove that sqrt(2) is irrational, step by step", "classifier": "heuristic"})
        assert r.json()["complexity"] == "complex"


def test_fallback_visible_in_response(settings):
    gem = FakeProvider("gemini", {"gemini-2.5-flash-lite": [permanent()]})
    with client(settings, gem=gem) as c:
        body = c.post("/v1/route", json={"query": "Hello!", "classifier": "heuristic"}).json()
        assert body["fallback_used"] is True and body["model_used"] == "gpt-4o-mini"
        assert c.get("/v1/metrics").json()["fallback_requests"] == 1


def test_502_when_everything_fails(settings):
    gem = FakeProvider("gemini", {m: [permanent()] for m in ("gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro")})
    oai = FakeProvider("openai", {m: [permanent()] for m in ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1")})
    with client(settings, gem, oai) as c:
        r = c.post("/v1/route", json={"query": "Hello!", "classifier": "heuristic"})
        assert r.status_code == 502 and len(r.json()["detail"]["attempts"]) >= 6
        assert c.get("/v1/metrics").json()["failures"] == 1


def test_503_without_keys():
    s = Settings(_env_file=None)
    with TestClient(create_app(s, {})) as c:
        assert c.post("/v1/route", json={"query": "hi"}).status_code == 503


def test_validation(settings):
    with client(settings) as c:
        assert c.post("/v1/route", json={"query": ""}).status_code == 422
        assert c.get("/health").json()["status"] == "ok"
        assert len(c.get("/v1/models").json()) == 6
