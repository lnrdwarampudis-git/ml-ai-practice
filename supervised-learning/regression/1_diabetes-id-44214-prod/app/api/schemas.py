"""
app/api/schemas.py
Strict Pydantic v2 schemas — bad input is rejected before it ever
touches the model. Field ranges match the data contract from training.
"""

from __future__ import annotations

from typing import Annotated, Optional
from pydantic import BaseModel, Field, model_validator
import time


# ── Numeric bounds derived from the diabetes dataset feature ranges
# Slightly wider than training min/max to allow real-world variation
_FLOAT_FIELD = Annotated[float, Field(ge=-1.0, le=1.0)]


class PatientFeatures(BaseModel):
    """
    10 clinical features for one patient.
    All values are mean-centred and unit-variance scaled
    (same pre-processing as training data).
    """
    age: float = Field(..., ge=-0.15, le=0.15,  description="Age (scaled)")
    sex: float = Field(..., ge=-0.05, le=0.06,  description="Sex (scaled binary)")
    bmi: float = Field(..., ge=-0.10, le=0.20,  description="BMI (scaled)")
    bp:  float = Field(..., ge=-0.15, le=0.15,  description="Blood pressure (scaled)")
    s1:  float = Field(..., ge=-0.15, le=0.20,  description="TC - total serum cholesterol")
    s2:  float = Field(..., ge=-0.15, le=0.20,  description="LDL - low-density lipoproteins")
    s3:  float = Field(..., ge=-0.15, le=0.20,  description="HDL - high-density lipoproteins")
    s4:  float = Field(..., ge=-0.15, le=0.20,  description="TCH - total cholesterol / HDL")
    s5:  float = Field(..., ge=-0.15, le=0.20,  description="LTG - log serum triglycerides")
    s6:  float = Field(..., ge=-0.15, le=0.20,  description="GLU - blood sugar level")

    model_config = {"json_schema_extra": {
        "example": {
            "age":  0.038, "sex":  0.051, "bmi":  0.062,
            "bp":   0.022, "s1":  -0.044, "s2":  -0.035,
            "s3":  -0.043, "s4":  -0.003, "s5":   0.020,
            "s6":  -0.018,
        }
    }}

    def to_list(self) -> list[float]:
        return [self.age, self.sex, self.bmi, self.bp,
                self.s1, self.s2, self.s3, self.s4, self.s5, self.s6]


class PredictRequest(BaseModel):
    """Single or batch prediction request."""
    instances: list[PatientFeatures] = Field(
        ..., min_length=1, max_length=512,
        description="List of patient feature objects (1–512 per request)"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional client-supplied request ID for tracing"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "instances": [{
                "age":  0.038, "sex":  0.051, "bmi":  0.062,
                "bp":   0.022, "s1":  -0.044, "s2":  -0.035,
                "s3":  -0.043, "s4":  -0.003, "s5":   0.020,
                "s6":  -0.018,
            }],
            "request_id": "client-abc-001"
        }
    }}


class PredictionResult(BaseModel):
    prediction:       float  = Field(..., description="Predicted disease progression score")
    prediction_lower: float  = Field(..., description="90% CI lower bound (bootstrap)")
    prediction_upper: float  = Field(..., description="90% CI upper bound (bootstrap)")


class PredictResponse(BaseModel):
    request_id:   str
    model_id:     str
    model_hash:   str
    predictions:  list[PredictionResult]
    latency_ms:   float
    timestamp:    float = Field(default_factory=time.time)


class HealthResponse(BaseModel):
    status:       str   # "ok" | "degraded" | "unavailable"
    model_loaded: bool
    model_id:     Optional[str]
    model_hash:   Optional[str]
    environment:  str
    version:      str


class MetricsSnapshot(BaseModel):
    total_predictions:    int
    total_requests:       int
    error_count:          int
    avg_latency_ms:       float
    p95_latency_ms:       float
    p99_latency_ms:       float
    drift_flags:          dict[str, bool]
    uptime_seconds:       float


class DriftReport(BaseModel):
    feature:    str
    psi_score:  float
    threshold:  float
    drifted:    bool
    n_samples:  int
