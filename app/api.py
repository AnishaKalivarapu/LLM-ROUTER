from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from .executor import AllModelsFailed
from .router import NoProvidersConfigured, Router
from .schemas import ClassifyRequest, RouteRequest, RouteResponse

router = APIRouter()


def _container(request: Request):
    return request.app.state.container


@router.post("/v1/route", response_model=RouteResponse)
async def route(req: RouteRequest, request: Request):
    r: Router = _container(request).router
    try:
        return await r.route(req)
    except NoProvidersConfigured as exc:
        raise HTTPException(503, detail=f"No LLM providers configured: {exc}")
    except AllModelsFailed as exc:
        raise HTTPException(
            502,
            detail={
                "message": "All candidate models failed",
                "attempts": [a.model_dump() for a in exc.attempts],
            },
        )


@router.post("/v1/classify")
async def classify(req: ClassifyRequest, request: Request):
    r: Router = _container(request).router
    result = await r.classify(req.query, req.classifier)
    return r._info(result)


@router.get("/v1/metrics")
async def metrics(request: Request):
    c = _container(request)
    return {**c.metrics.summary(), "open_circuits": c.executor.breaker.open_models()}


@router.get("/v1/models")
async def models(request: Request):
    s = _container(request).settings
    live = {m.name for m in s.available_models()}
    return [{**m.model_dump(mode="json"), "available": m.name in live} for m in s.models]


@router.get("/health")
async def health(request: Request):
    s = _container(request).settings
    return {
        "status": "ok",
        "classifier_mode": s.classifier_mode,
        "providers_configured": sorted({m.provider.value for m in s.available_models()}),
    }
