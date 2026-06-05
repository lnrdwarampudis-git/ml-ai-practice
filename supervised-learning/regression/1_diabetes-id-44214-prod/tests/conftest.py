"""
conftest.py — shared test fixtures.
Loads the model into the registry before any test that needs predictions.
"""
import os
import pytest
from pathlib import Path

os.environ.setdefault("MODEL_ARTIFACTS_DIR", str(Path(__file__).parent.parent / "model_artifacts"))
os.environ.setdefault("ENVIRONMENT", "development")

from app.ml.model_loader import get_registry
from app.monitoring.metrics import init_drift_detector
from app.core.settings import get_settings


@pytest.fixture(scope="session", autouse=True)
def load_model():
    settings = get_settings()
    registry = get_registry()
    arts = registry.load(settings.artifacts_path)
    init_drift_detector(
        feature_names=arts.feature_names,
        train_stats=arts.train_stats,
        window_size=50,
        threshold=0.2,
    )
    return registry
