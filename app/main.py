from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI

from .api import router as api_router
from .classifiers.heuristic import HeuristicClassifier
from .classifiers.hybrid import HybridClassifier
from .classifiers.llm import LLMClassifier
from .config import Provider, Settings, get_settings
from .executor import Executor
from .metrics import MetricsStore
from .providers.base import LLMProvider
from .providers.registry import build_providers
from .router import Router


@dataclass
class Container:
    settings: Settings
    executor: Executor
    metrics: MetricsStore
    router: Router


def build_container(settings: Settings, providers: dict[Provider, LLMProvider]) -> Container:
    executor = Executor(settings, providers)
    available = settings.available_models()
    # Baseline = what you'd pay with no router: always the strongest configured model.
    baseline = max(available, key=lambda m: (m.tier, m.input_per_m + m.output_per_m)) if available else None
    metrics = MetricsStore(baseline=baseline)

    heuristic = HeuristicClassifier()
    llm = LLMClassifier(settings, executor)
    hybrid = HybridClassifier(heuristic, llm, settings.confidence_threshold)
    router = Router(settings, executor, {"heuristic": heuristic, "llm": llm, "hybrid": hybrid}, metrics)
    return Container(settings, executor, metrics, router)


def create_app(settings: Settings | None = None, providers: dict[Provider, LLMProvider] | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = None
        provs = providers
        if provs is None:
            client = httpx.AsyncClient(timeout=settings.request_timeout_s)
            provs = build_providers(settings, client)
        app.state.container = build_container(settings, provs)
        yield
        if client is not None:
            await client.aclose()

    app = FastAPI(title="Smart LLM Router", version="0.1.0", lifespan=lifespan)
    app.include_router(api_router)
    return app


app = create_app()
