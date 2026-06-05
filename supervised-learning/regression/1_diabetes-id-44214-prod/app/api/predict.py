"""
app/api/predict.py
POST /predict — single or batch inference endpoint.

Key production patterns:
- Input validated by Pydantic before touching the model
- Model loaded from registry (never re-loaded per request)
- Latency measured end-to-end and pushed to Prometheus
- Every request/response logged as structured JSON
- Bootstrap confidence intervals included in response
- Drift detector fed with each batch
"""

import time
import uuid
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.schemas import PredictRequest, PredictResponse, PredictionResult
from app.ml.model_loader import get_registry, ModelRegistry
from app.monitoring import metrics as mon
from app.core.logging import get_logger
from app.core.settings import get_settings, Settings

router = APIRouter()
logger = get_logger(__name__)

# Bootstrap CI width pre-computed from training (±1 std of bootstrap distribution)
# In a full system this would be loaded from artifacts; here we use the
# training-derived value for a 90% PI
_BOOTSTRAP_HALF_WIDTH = 83.6   # matches conformal q̂ from training


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict disease progression",
    description=(
        "Given one or more patient feature vectors, returns predicted "
        "disease progression scores with 90% prediction intervals."
    ),
    status_code=status.HTTP_200_OK,
)
async def predict(
    request: Request,
    body: PredictRequest,
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
):
    t_start = time.perf_counter()
    request_id = body.request_id or str(uuid.uuid4())
    mon.increment_requests()

    try:
        arts = registry.get()
    except RuntimeError as exc:
        mon.increment_errors("model_not_loaded")
        logger.error("Model not loaded", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available. Try again shortly.",
        )

    # ── Build feature matrix
    feature_matrix = [inst.to_list() for inst in body.instances]

    # ── Predict
    try:
        raw_preds = registry.predict(feature_matrix)
    except Exception as exc:
        mon.increment_errors("prediction_error")
        logger.error("Prediction failed", extra={
            "request_id": request_id, "error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        )

    # ── Build response with PI
    predictions = []
    for pred in raw_preds:
        predictions.append(PredictionResult(
            prediction=round(float(pred), 4),
            prediction_lower=round(float(max(25.0, pred - _BOOTSTRAP_HALF_WIDTH)), 4),
            prediction_upper=round(float(min(346.0, pred + _BOOTSTRAP_HALF_WIDTH)), 4),
        ))
        mon.PREDICTION_VALUE.observe(float(pred))

    # ── Monitoring side-effects
    mon.increment_predictions(len(predictions))
    drift_det = mon.get_drift_detector()
    if drift_det:
        for row in feature_matrix:
            drift_det.record(row)

    latency_ms = (time.perf_counter() - t_start) * 1000
    mon.get_latency_tracker().record(latency_ms)
    mon.LATENCY.labels(endpoint="/predict").observe(latency_ms / 1000)
    mon.REQUEST_COUNT.labels(
        method="POST", endpoint="/predict", status_code=200).inc()

    logger.info("Prediction served", extra={
        "request_id":  request_id,
        "n_instances": len(predictions),
        "latency_ms":  round(latency_ms, 2),
        "model_id":    arts.metadata.get("model_id"),
    })

    return PredictResponse(
        request_id=request_id,
        model_id=arts.metadata.get("model_id", "unknown"),
        model_hash=arts.model_hash,
        predictions=predictions,
        latency_ms=round(latency_ms, 3),
    )
