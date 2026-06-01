"""
utils/helpers.py — Shared utilities used across all phase modules.
"""

import hashlib
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error,
)


# ── Plot style ────────────────────────────────────────────────────────────────
def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 110,
    })


def save_fig(fig, path: Path, filename: str, bbox_inches="tight"):
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / filename, bbox_inches=bbox_inches)
    plt.close(fig)
    print(f"  📊 Saved → {path / filename}")


# ── Metrics ───────────────────────────────────────────────────────────────────
def regression_metrics(y_true, y_pred, label="") -> dict:
    m = {
        "RMSE":  round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "MAE":   round(float(mean_absolute_error(y_true, y_pred)), 3),
        "MAPE":  round(float(mean_absolute_percentage_error(y_true, y_pred) * 100), 3),
        "R2":    round(float(r2_score(y_true, y_pred)), 4),
        "MedAE": round(float(np.median(np.abs(np.array(y_true) - np.array(y_pred)))), 3),
    }
    if label:
        print(f"  [{label}] " + "  ".join(f"{k}={v}" for k, v in m.items()))
    return m


# ── Artifact I/O ──────────────────────────────────────────────────────────────
def save_json(obj: dict, path: Path, filename: str):
    path.mkdir(parents=True, exist_ok=True)
    fp = path / filename
    with open(fp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  💾 Saved → {fp}")


def data_hash(df: pd.DataFrame, n: int = 12) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:n]


def today() -> str:
    return str(datetime.date.today())


def now() -> str:
    return str(datetime.datetime.now())


# ── PSI ───────────────────────────────────────────────────────────────────────
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breaks = np.histogram(expected, bins=bins)[1]
    exp_cnt = np.histogram(expected, bins=breaks)[0] / len(expected) + 1e-8
    act_cnt = np.histogram(actual, bins=breaks)[0] / len(actual) + 1e-8
    return float(np.sum((act_cnt - exp_cnt) * np.log(act_cnt / exp_cnt)))


# ── Asymmetric loss ───────────────────────────────────────────────────────────
def asymmetric_loss(y_true, y_pred, alpha: float = 2.0) -> float:
    errors = np.array(y_true) - np.array(y_pred)
    losses = np.where(errors > 0, alpha * errors**2, errors**2)
    return float(np.mean(losses))
