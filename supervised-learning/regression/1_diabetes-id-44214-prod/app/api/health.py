"""
app/api/health.py
Kubernetes-style health endpoints + Prometheus scrape endpoint.

GET /health      — liveness probe  (is the process alive?)
GET /ready       — readiness probe (is the model loaded and ready?)
GET /metrics     — Prometheus text format
GET /monitoring  — JSON metrics snapshot
GET /drift       — per-feature PSI drift report
GET /model-info  — model metadata
"""

import time
from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import PlainTextResponse

from app.api.schemas import HealthResponse, MetricsSnapshot, DriftReport
from app.ml.model_loader import get_registry
from app.monitoring.metrics import (
    prometheus_output, get_snapshot, get_drift_detector,
)
from app.core.settings import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Liveness probe — always returns 200 if the process is running."""
    settings = get_settings()
    try:
        arts = get_registry().get()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_id=arts.metadata.get("model_id"),
            model_hash=arts.model_hash,
            environment=settings.environment,
            version=settings.app_version,
        )
    except RuntimeError:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_id=None,
            model_hash=None,
            environment=settings.environment,
            version=settings.app_version,
        )


@router.get("/ready", tags=["ops"])
async def ready():
    """
    Readiness probe — returns 200 only when model is loaded.
    Kubernetes will stop sending traffic until this returns 200.
    """
    try:
        get_registry().get()
        return {"status": "ready"}
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet",
        )


@router.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
async def metrics():
    """Prometheus scrape endpoint."""
    data, content_type = prometheus_output()
    return Response(content=data, media_type=content_type)


@router.get("/monitoring", response_model=MetricsSnapshot, tags=["ops"])
async def monitoring():
    """JSON metrics snapshot — useful for dashboards / alerting."""
    snap = get_snapshot()
    return MetricsSnapshot(**snap)


@router.get("/drift", response_model=list[DriftReport], tags=["ops"])
async def drift():
    """Per-feature PSI drift report based on the rolling prediction window."""
    dd = get_drift_detector()
    if dd is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drift detector not initialised",
        )
    settings = get_settings()
    return [
        DriftReport(
            feature=feat,
            psi_score=dd.psi_scores.get(feat, 0.0),
            threshold=settings.drift_psi_threshold,
            drifted=dd.drift_flags.get(feat, False),
            n_samples=len(dd._buffer),
        )
        for feat in dd.feature_names
    ]


@router.get("/model-info", tags=["ops"])
async def model_info():
    """Returns model metadata from the loaded artifact."""
    try:
        arts = get_registry().get()
        return arts.metadata
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
