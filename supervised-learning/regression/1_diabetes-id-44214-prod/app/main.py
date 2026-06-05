"""
app/main.py — FastAPI application factory.

Middleware stack (outermost first):
  1. CORS
  2. APIKeyMiddleware     (optional, off in dev)
  3. RateLimitMiddleware  (optional, off in dev)
  4. Request-ID + timing (always on)
"""

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import predict, health
from app.core.logging import configure_root_logger, get_logger
from app.core.middleware import RateLimitMiddleware, APIKeyMiddleware
from app.core.settings import get_settings
from app.ml.model_loader import get_registry
from app.monitoring import metrics as mon

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_root_logger(settings.log_level)

    logger.info("Starting up", extra={
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    })

    registry = get_registry()
    arts = registry.load(settings.artifacts_path)

    mon.init_drift_detector(
        feature_names=arts.feature_names,
        train_stats=arts.train_stats,
        window_size=settings.drift_window_size,
        threshold=settings.drift_psi_threshold,
    )
    mon.MODEL_INFO.labels(
        model_id=arts.metadata.get("model_id", "unknown"),
        model_hash=arts.model_hash,
    ).set(1)

    if not settings.debug:
        registry.start_hot_reload(settings.artifacts_path,
                                   settings.model_reload_interval_seconds)

    logger.info("Startup complete — model ready")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production ML API — diabetes disease progression prediction. "
            "GradientBoostingRegressor trained on OpenML 44214."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS
    allowed_origins = (
        ["*"] if settings.environment == "development"
        else ["https://your-frontend.example.com"]
    )
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins,
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    # ── Auth (outermost guarded middleware)
    app.add_middleware(
        APIKeyMiddleware,
        valid_key_hashes=settings.valid_key_hashes,
        enabled=settings.api_keys_enabled,
    )

    # ── Rate limiting
    if settings.rate_limit_enabled:
        app.add_middleware(RateLimitMiddleware,
                           limit_per_minute=settings.rate_limit_per_minute)

    # ── Request-ID + timing (always on)
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
        logger.info("Request", extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 2),
        })
        return response

    # ── Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        mon.increment_errors(type(exc).__name__)
        logger.error("Unhandled exception", extra={
            "request_id": request_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
        )

    app.include_router(predict.router, tags=["inference"])
    app.include_router(health.router, tags=["ops"])

    @app.get("/", tags=["root"])
    async def root():
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "docs": "/docs",
        }

    return app


app = create_app()
