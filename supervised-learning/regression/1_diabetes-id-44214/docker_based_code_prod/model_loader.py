"""
app/ml/model_loader.py
Loads model + metadata ONCE at startup.
Supports hot-reload: checks artifact hash periodically and swaps
the model in-place without restarting the server.
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from app.core.logging import get_logger
from app.core.settings import get_settings

logger = get_logger(__name__)


@dataclass
class ModelArtifacts:
    pipeline:      object                       # sklearn Pipeline
    metadata:      dict
    train_stats:   dict                         # per-feature train distribution
    feature_names: list[str]
    model_hash:    str
    loaded_at:     float = field(default_factory=time.time)


class ModelRegistry:
    """Thread-safe singleton that holds the live model."""

    def __init__(self):
        self._lock     = threading.RLock()
        self._artifacts: Optional[ModelArtifacts] = None

    def load(self, artifacts_dir: Path) -> ModelArtifacts:
        model_path = artifacts_dir / "model.pkl"
        meta_path  = artifacts_dir / "metadata.json"
        stats_path = artifacts_dir / "train_stats.json"

        if not model_path.exists():
            raise FileNotFoundError(f"model.pkl not found in {artifacts_dir}")

        pipeline = joblib.load(model_path)
        metadata = json.loads(meta_path.read_text())
        train_stats = json.loads(stats_path.read_text())
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]

        arts = ModelArtifacts(
            pipeline=pipeline,
            metadata=metadata,
            train_stats=train_stats,
            feature_names=metadata["feature_names"],
            model_hash=model_hash,
        )

        with self._lock:
            self._artifacts = arts

        logger.info("Model loaded", extra={
            "model_id":   metadata.get("model_id"),
            "model_hash": model_hash,
            "metrics":    metadata.get("metrics"),
        })
        return arts

    def get(self) -> ModelArtifacts:
        with self._lock:
            if self._artifacts is None:
                raise RuntimeError("Model not loaded yet — call load() first")
            return self._artifacts

    def predict(self, features: list[list[float]]) -> np.ndarray:
        import pandas as pd
        arts = self.get()
        X = pd.DataFrame(features, columns=arts.feature_names)
        return arts.pipeline.predict(X)

    def start_hot_reload(self, artifacts_dir: Path, interval_seconds: int = 3600):
        """Background thread: reload model if artifact hash changes."""
        def _watch():
            while True:
                time.sleep(interval_seconds)
                try:
                    model_path = artifacts_dir / "model.pkl"
                    current_hash = hashlib.sha256(
                        model_path.read_bytes()).hexdigest()[:16]
                    if current_hash != self.get().model_hash:
                        logger.info("New model detected — hot-reloading",
                                    extra={"new_hash": current_hash})
                        self.load(artifacts_dir)
                except Exception as exc:
                    logger.error("Hot-reload failed", extra={"error": str(exc)})

        t = threading.Thread(target=_watch, daemon=True, name="model-hot-reload")
        t.start()
        logger.info("Hot-reload watcher started",
                    extra={"interval_seconds": interval_seconds})


# ── Module-level singleton
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    return _registry
