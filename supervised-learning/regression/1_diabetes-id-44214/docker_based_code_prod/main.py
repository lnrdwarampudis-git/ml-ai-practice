"""
app/main.py
FastAPI application factory.

Production patterns wired in here:
- Lifespan: model loaded ONCE at startup, cleaned up on shutdown
- Request ID middleware: every request gets a trace ID
- Timing middleware: latency logged for every request
- CORS (locked down for prod)
- Global exception handler: unhandled errors → structured JSON, never stack traces to client
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
from app.core.settings import get_settings
from app.ml.model_loader import get_registry
from app.monitoring import metrics as mon

logger = get_logger(__name__)


# ── Lifespan: startup + shutdown ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_root_logger(settings.log_level)

    logger.info("Starting up", extra={
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    })

    # Load model artifacts ONCE
    artifacts_dir = settings.artifacts_path
    registry = get_registry()
    arts = registry.load(artifacts_dir)

    # Initialise drift detector with training distribution
    mon.init_drift_detector(
        feature_names=arts.feature_names,
        train_stats=arts.train_stats,
        window_size=settings.drift_window_size,
        threshold=settings.drift_psi_threshold,
    )

    # Register model info in Prometheus
    mon.MODEL_INFO.labels(
        model_id=arts.metadata.get("model_id", "unknown"),
        model_hash=arts.model_hash,
    ).set(1)

    # Start hot-reload watcher
    if not settings.debug:
        registry.start_hot_reload(artifacts_dir, settings.model_reload_interval_seconds)

    logger.info("Startup complete — model ready")
    yield

    # ── Shutdown
    logger.info("Shutting down")


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production ML API for diabetes disease progression prediction. "
            "Powered by GradientBoostingRegressor trained on OpenML 44214."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS — lock down in production
    allowed_origins = (
        ["*"] if settings.environment == "development"
        else ["https://your-frontend.example.com"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Request-ID + timing middleware
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
            "method":     request.method,
            "path":       request.url.path,
            "status":     response.status_code,
            "latency_ms": round(latency_ms, 2),
        })
        return response

    # ── Global exception handler — never leak stack traces to clients
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        mon.increment_errors(type(exc).__name__)
        logger.error("Unhandled exception", extra={
            "request_id": request_id,
            "error_type": type(exc).__name__,
            "error":      str(exc),
        }, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail":     "Internal server error",
                "request_id": request_id,
            },
        )

    # ── Routers
    app.include_router(predict.router, tags=["inference"])
    app.include_router(health.router, tags=["ops"])

    @app.get("/", tags=["root"])
    async def root():
        return {
            "service":     settings.app_name,
            "version":     settings.app_version,
            "environment": settings.environment,
            "docs":        "/docs",
        }

    return app


app = create_app()
