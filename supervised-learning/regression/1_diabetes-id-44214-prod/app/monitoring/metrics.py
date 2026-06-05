"""
app/monitoring/metrics.py
Prometheus counters/histograms + in-process drift detection.
Metrics are exposed at GET /metrics in Prometheus text format,
ready to be scraped by a Prometheus server or Grafana Agent.
"""

import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry,
)

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "diabetes_api_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=REGISTRY,
)
PREDICTION_COUNT = Counter(
    "diabetes_predictions_total",
    "Total individual predictions served",
    registry=REGISTRY,
)
ERROR_COUNT = Counter(
    "diabetes_errors_total",
    "Total errors",
    ["error_type"],
    registry=REGISTRY,
)
LATENCY = Histogram(
    "diabetes_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    registry=REGISTRY,
)
PREDICTION_VALUE = Histogram(
    "diabetes_prediction_value",
    "Distribution of predicted disease progression scores",
    buckets=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 346],
    registry=REGISTRY,
)
DRIFT_DETECTED = Gauge(
    "diabetes_drift_detected",
    "1 if drift detected for this feature, 0 otherwise",
    ["feature"],
    registry=REGISTRY,
)
MODEL_INFO = Gauge(
    "diabetes_model_info",
    "Model metadata",
    ["model_id", "model_hash"],
    registry=REGISTRY,
)


def prometheus_output() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


# ── In-process drift detector ─────────────────────────────────────────────────
def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breaks = np.histogram(expected, bins=bins)[1]
    exp_c = np.histogram(expected, bins=breaks)[0] / len(expected) + 1e-8
    act_c = np.histogram(actual,   bins=breaks)[0] / len(actual)   + 1e-8
    return float(np.sum((act_c - exp_c) * np.log(act_c / exp_c)))


class DriftDetector:
    """
    Buffers the last N prediction inputs and computes PSI against
    the training distribution for each feature.
    """

    def __init__(
        self,
        feature_names: list[str],
        train_stats: dict,
        window_size: int = 100,
        psi_threshold: float = 0.2,
    ):
        self.feature_names = feature_names
        self.train_stats   = train_stats
        self.window_size   = window_size
        self.psi_threshold = psi_threshold
        self._lock  = threading.Lock()
        self._buffer: deque[list[float]] = deque(maxlen=window_size)
        self.psi_scores: dict[str, float] = {f: 0.0 for f in feature_names}
        self.drift_flags: dict[str, bool] = {f: False for f in feature_names}

    def record(self, features: list[float]):
        """Add one prediction's features to the rolling window."""
        with self._lock:
            self._buffer.append(features)
            if len(self._buffer) >= self.window_size // 2:
                self._compute_drift()

    def _compute_drift(self):
        arr = np.array(self._buffer)          # (N, n_features)
        for i, feat in enumerate(self.feature_names):
            stats = self.train_stats[feat]
            # Synthesise training reference from stored stats
            train_ref = np.random.normal(
                stats["mean"], max(stats["std"], 1e-8), 500)
            train_ref = np.clip(train_ref, stats["min"], stats["max"])
            try:
                score = _psi(train_ref, arr[:, i])
            except Exception:
                score = 0.0
            self.psi_scores[feat] = round(score, 4)
            self.drift_flags[feat] = score > self.psi_threshold
            DRIFT_DETECTED.labels(feature=feat).set(
                1 if self.drift_flags[feat] else 0)

        drifted = [f for f, v in self.drift_flags.items() if v]
        if drifted:
            logger.warning("Feature drift detected", extra={
                "drifted_features": drifted,
                "psi_scores": self.psi_scores,
            })


# ── Latency tracker (in-memory percentiles) ───────────────────────────────────
class LatencyTracker:
    def __init__(self, maxlen: int = 10_000):
        self._lock    = threading.Lock()
        self._samples = deque(maxlen=maxlen)

    def record(self, ms: float):
        with self._lock:
            self._samples.append(ms)

    @property
    def stats(self) -> dict:
        with self._lock:
            if not self._samples:
                return {"avg": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}
            arr = np.array(self._samples)
            return {
                "avg":   round(float(arr.mean()), 3),
                "p95":   round(float(np.percentile(arr, 95)), 3),
                "p99":   round(float(np.percentile(arr, 99)), 3),
                "count": len(arr),
            }


# ── Module-level singletons ───────────────────────────────────────────────────
_drift_detector: Optional[DriftDetector] = None
_latency_tracker = LatencyTracker()
_start_time = time.time()
_total_predictions = 0
_total_requests    = 0
_error_count       = 0
_lock = threading.Lock()


def init_drift_detector(feature_names, train_stats, window_size, threshold):
    global _drift_detector
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        train_stats=train_stats,
        window_size=window_size,
        psi_threshold=threshold,
    )


def get_drift_detector() -> Optional[DriftDetector]:
    return _drift_detector


def get_latency_tracker() -> LatencyTracker:
    return _latency_tracker


def increment_predictions(n: int = 1):
    global _total_predictions
    with _lock:
        _total_predictions += n
    PREDICTION_COUNT.inc(n)


def increment_requests():
    global _total_requests
    with _lock:
        _total_requests += 1


def increment_errors(error_type: str = "unknown"):
    global _error_count
    with _lock:
        _error_count += 1
    ERROR_COUNT.labels(error_type=error_type).inc()


def get_snapshot() -> dict:
    lat = _latency_tracker.stats
    dd  = _drift_detector
    with _lock:
        return {
            "total_predictions": _total_predictions,
            "total_requests":    _total_requests,
            "error_count":       _error_count,
            "avg_latency_ms":    lat["avg"],
            "p95_latency_ms":    lat["p95"],
            "p99_latency_ms":    lat["p99"],
            "drift_flags":       dd.drift_flags if dd else {},
            "uptime_seconds":    round(time.time() - _start_time, 1),
        }
