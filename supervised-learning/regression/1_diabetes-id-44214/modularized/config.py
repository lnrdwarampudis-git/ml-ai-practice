"""
config.py — Central configuration for the Diabetes ML Pipeline.
All paths, hyperparameters, and constants live here.
Pass --output-dir at runtime to override the base output directory.
"""

import os
import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths (set via CLI or defaults)
    base_dir: Path = Path("./diabetes_output")

    @property
    def figures_dir(self) -> Path:
        return self.base_dir / "figures"

    @property
    def artifacts_dir(self) -> Path:
        return self.base_dir / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    def make_dirs(self):
        for d in [self.figures_dir, self.artifacts_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Experiment
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    # ── Modelling
    ridge_alphas: list = field(default_factory=lambda: [0.01, 0.1, 1, 10, 100, 500])
    gbr_param_grid: dict = field(default_factory=lambda: {
        "n_estimators": [50, 100, 200],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_samples_leaf": [1, 2, 5],
    })
    gbr_n_iter: int = 20

    # ── Uncertainty / Conformal
    n_bootstrap: int = 100
    conformal_alpha: float = 0.1

    # ── Drift / Monitoring
    psi_bins: int = 10
    n_drift_windows: int = 5

    # ── Cost-aware
    cost_per_error_unit: float = 5.0
    asym_penalty: float = 2.0
    cost_bias_factor: float = 1.05


def parse_args() -> Config:
    """Parse CLI arguments and return a populated Config."""
    parser = argparse.ArgumentParser(
        description="Diabetes ML Pipeline — OpenML 44214"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("./diabetes_output")),
        help="Root directory for all outputs (figures, artifacts, models). "
             "Default: ./diabetes_output",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--phases",
        nargs="+",
        type=int,
        default=list(range(1, 51)),
        help="Phases to run, e.g. --phases 1 2 3 (default: all 1-50)",
    )
    args = parser.parse_args()

    cfg = Config()
    cfg.base_dir = Path(args.output_dir)
    cfg.random_seed = args.seed
    cfg.test_size = args.test_size
    cfg.cv_folds = args.cv_folds
    return cfg, args.phases
