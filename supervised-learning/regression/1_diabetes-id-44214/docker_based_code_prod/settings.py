"""
app/core/settings.py
All config via environment variables (12-factor).
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── App
    app_name: str = "Diabetes Progression Predictor"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # ── Model
    model_artifacts_dir: str = "./model_artifacts"
    model_reload_interval_seconds: int = 3600

    # ── Drift
    drift_psi_threshold: float = 0.2
    drift_window_size: int = 100

    # ── Rate limiting
    rate_limit_per_minute: int = 60

    # ── Metrics
    metrics_enabled: bool = True

    @property
    def artifacts_path(self) -> Path:
        return Path(self.model_artifacts_dir)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
