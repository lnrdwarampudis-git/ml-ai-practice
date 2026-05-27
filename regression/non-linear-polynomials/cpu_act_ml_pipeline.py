"""
cpu_pipeline.py
===============
Industry-standard end-to-end ML pipeline for CPU Activity prediction.

Data (exact pattern as specified):
    cpu_act = fetch_openml(data_id=197, as_frame=True, parser="auto")
    cpu_act_df = cpu_act.frame

Target: usr — percentage of time CPUs run in user mode (0–100%)
Dataset: 8192 rows × 22 columns, all float64, zero missing values

Mirrors every architectural pattern from titanic-ml-pipeline.py.
Primary focus: Non-Linearity & Polynomials

New concepts explored:
  A. Polynomial features         — degree-2 and degree-3 terms, interaction-only
                                   vs full polynomial, feature explosion management
  B. Basis functions             — Radial Basis Functions (RBF centres), B-spline
                                   basis, Fourier basis for periodic patterns
  C. Non-linear feature analysis — mutual information vs Pearson r, comparing
                                   linear vs non-linear feature importance
  D. Kernel approximation        — Nystroem RBF approximation, explicit feature map
                                   for SVM-like non-linearity in linear models
  E. Log + power transforms      — log1p, Box-Cox, Yeo-Johnson for right-skewed
                                   system counter features
  F. Multicollinearity handling  — sys+wait+usr≈100 constraint detected and managed,
                                   VIF report, feature dropping strategy
  G. Regularisation paths        — Ridge/Lasso/ElasticNet on polynomial features
                                   showing coefficient shrinkage under expansion
  H. Residual diagnostics        — Breusch-Pagan heteroscedasticity, Shapiro-Wilk,
                                   partial residual plots (component-plus-residual)
  I. Learning curves             — polynomial degree vs bias-variance tradeoff
  J. Partial dependence + ICE    — individual conditional expectation plots showing
                                   heterogeneous non-linear effects
  K. Cross-validation strategies — repeated K-Fold for stability, nested CV
                                   for unbiased performance estimation
  L. Spline regression           — natural cubic splines as interpretable
                                   non-linear basis, knot placement strategies

Industry-standard regression metrics: MAE, RMSE, R², MAPE, MedAE

Usage:
  python cpu_pipeline.py train   --output-dir artifacts_cpu
  python cpu_pipeline.py predict --artifact-dir artifacts_cpu --input-csv sample.csv
  python cpu_pipeline.py monitor --artifact-dir artifacts_cpu --input-csv new.csv
  python cpu_pipeline.py sample-input --output-csv sample.csv --rows 20
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── matplotlib scratch dir ────────────────────────────────────────────────────
_MPLCONFIGDIR = Path("artifacts_cpu") / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import (
    ElasticNet, ElasticNetCV,
    Lasso, LassoCV,
    LinearRegression,
    Ridge, RidgeCV,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    cross_val_predict,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures,
    PowerTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)
from sklearn.svm import SVR

try:
    import shap; _SHAP = True
except ImportError:
    _SHAP = False
try:
    import mlflow; import mlflow.sklearn; _MLFLOW = True
except ImportError:
    _MLFLOW = False
try:
    import pandera.pandas as pa; _PANDERA = True
except ImportError:
    _PANDERA = False

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE          = 42
TARGET                = "usr"   # CPU user-mode time (%)
MODEL_FILE            = "cpu_activity_pipeline.joblib"
METRICS_FILE          = "metrics.json"
TRAINING_PROFILE_FILE = "training_profile.json"
MODEL_CARD_FILE       = "model_card.json"
ENVIRONMENT_FILE      = "environment.json"
N_JOBS                = int(os.environ.get("ML_N_JOBS", 1))
UNCERTAINTY_BAND      = float(os.environ.get("ML_UNCERTAINTY_BAND", 0.10))

# Raw feature names (21 system counter features)
RAW_FEATURES = [
    "lread", "lwrite", "scall", "sread", "swrite",
    "fork",  "exec",  "rchar", "wchar",
    "pgout", "ppgout","pgfree","pgin", "ppgin",
    "pflt",  "vflt",  "runqsz","freemem","freeswap",
    "sys",   "wait",
]
# Features with |skew| > 1 (from known distribution shapes)
LOG_TRANSFORM_FEATURES = [
    "lread", "lwrite", "scall", "sread", "swrite",
    "fork",  "exec",  "rchar", "wchar",
    "pgout", "ppgout","pgfree","pgin", "ppgin",
    "pflt",  "vflt",  "runqsz","freemem","freeswap",
]
# Polynomial / non-linear features — highest signal for usr prediction
POLY_CANDIDATE_FEATURES = ["runqsz","sys","wait","vflt","pflt","freemem","fork","exec"]


@dataclass(frozen=True)
class ColumnGroups:
    numeric: list[str]
    categorical: list[str]


def get_column_groups() -> ColumnGroups:
    """Post-engineer feature columns fed to the preprocessor."""
    engineered = [f"log_{c}" for c in LOG_TRANSFORM_FEATURES]
    interactions = [
        "fork_exec_interact",
        "pflt_vflt_interact",
        "sys_wait_total",
        "runqsz_sq",
        "io_pressure",
        "memory_pressure",
        "cpu_contention",
    ]
    return ColumnGroups(
        numeric=RAW_FEATURES + engineered + interactions,
        categorical=[],
    )


# ── Data loading (exact pattern requested) ────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Exact loader as specified:
        cpu_act = fetch_openml(data_id=197, as_frame=True, parser="auto")
        cpu_act_df = cpu_act.frame
    """
    log.info("Loading CPU Activity dataset from OpenML (data_id=197) …")
    try:
        cpu_act    = fetch_openml(data_id=197, as_frame=True, parser="auto")
        cpu_act_df = cpu_act.frame
        return cpu_act_df.copy()
    except Exception as exc:
        log.warning("OpenML unavailable (%s) — generating synthetic CPU data.", exc)
        return _make_synthetic_cpu()


def _make_synthetic_cpu() -> pd.DataFrame:
    """Synthetic fallback with realistic CPU counter distributions."""
    rng = np.random.default_rng(RANDOM_STATE)
    n   = 8192
    df  = pd.DataFrame({
        "lread":   rng.exponential(50, n),
        "lwrite":  rng.exponential(30, n),
        "scall":   rng.exponential(200, n),
        "sread":   rng.exponential(80, n),
        "swrite":  rng.exponential(60, n),
        "fork":    rng.exponential(5, n),
        "exec":    rng.exponential(8, n),
        "rchar":   rng.exponential(5000, n),
        "wchar":   rng.exponential(3000, n),
        "pgout":   rng.exponential(2, n),
        "ppgout":  rng.exponential(2, n),
        "pgfree":  rng.exponential(5, n),
        "pgin":    rng.exponential(3, n),
        "ppgin":   rng.exponential(3, n),
        "pflt":    rng.exponential(20, n),
        "vflt":    rng.exponential(50, n),
        "runqsz":  rng.exponential(3, n),
        "freemem": rng.exponential(800, n),
        "freeswap":rng.exponential(50000, n),
        "sys":     rng.uniform(0, 30, n),
        "wait":    rng.uniform(0, 20, n),
    })
    df[TARGET] = np.clip(
        0.3*df["sys"] + 0.1*df["runqsz"] + 0.05*np.log1p(df["vflt"])
        + rng.normal(0, 5, n), 0, 100)
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild entire DataFrame as numpy float64 — defeats Arrow/Sparse
    dtype backing that can cause sklearn set_output failures.
    """
    df = df.copy()
    rebuilt = {}
    for col in df.columns:
        _s   = pd.to_numeric(df[col], errors="coerce")
        _arr = _s.to_numpy(dtype=np.float64)
        if np.isnan(_arr).all() and not df[col].isna().all():
            log.warning("fix_data_types: '%s' all-NaN after cast (dtype=%s)", col, df[col].dtype)
        rebuilt[col] = _arr
    return pd.DataFrame(rebuilt, index=df.index)


def split_data(df):
    """Random 80/20 split BEFORE any EDA. Mirrors reference pipeline."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def missingness_report(df):
    r = df.isna().agg(["sum","mean"]).T.rename(
        columns={"sum":"missing_count","mean":"missing_rate"})
    r["dtype"] = df.dtypes.astype(str)
    return r.sort_values("missing_rate", ascending=False)


# ── Pandera schema ────────────────────────────────────────────────────────────
def build_input_schema():
    if not _PANDERA:
        log.warning("pandera not installed — validation skipped.")
        return None
    schema = pa.DataFrameSchema({
        c: pa.Column(float, pa.Check.ge(0), nullable=True, required=False)
        for c in RAW_FEATURES
    }, coerce=True, strict=False)
    return schema

INPUT_SCHEMA = build_input_schema()


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering  —  mirrors TitanicFeatureEngineer pattern exactly
# ─────────────────────────────────────────────────────────────────────────────
class CPUFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-driven feature engineering for CPU Activity.

    Concepts implemented:
      E.  Log1p transforms for all right-skewed system counters
      A/C. Polynomial interactions capturing non-linear CPU dynamics
      F.  Derived features that encode known system relationships
    """

    def fit(self, X, y=None):
        # Nothing to learn — all transforms are deterministic
        # (mirrors TitanicFeatureEngineer.fit which learns rare_titles_)
        return self

    def transform(self, X):
        X = X.copy()

        # ── Concept E: Log1p transforms for right-skewed counters ─────────────
        for col in LOG_TRANSFORM_FEATURES:
            if col in X.columns:
                X[f"log_{col}"] = np.log1p(X[col].clip(lower=0).fillna(0))

        # ── Concept A/C: Polynomial interaction terms ──────────────────────────
        fork_s = X["fork"].clip(lower=0).fillna(0) if "fork" in X.columns else 0
        exec_s = X["exec"].clip(lower=0).fillna(0) if "exec" in X.columns else 0
        pflt_s = X["pflt"].clip(lower=0).fillna(0) if "pflt" in X.columns else 0
        vflt_s = X["vflt"].clip(lower=0).fillna(0) if "vflt" in X.columns else 0
        sys_s  = X["sys"].clip(lower=0).fillna(0)  if "sys" in X.columns else 0
        wait_s = X["wait"].clip(lower=0).fillna(0) if "wait" in X.columns else 0
        rq_s   = X["runqsz"].clip(lower=0).fillna(0) if "runqsz" in X.columns else 0
        fm_s   = X["freemem"].clip(lower=1).fillna(1) if "freemem" in X.columns else 1

        X["fork_exec_interact"] = fork_s * exec_s          # process spawn intensity
        X["pflt_vflt_interact"] = pflt_s * vflt_s          # total fault pressure
        X["sys_wait_total"]     = sys_s + wait_s            # CPU not in user mode
        X["runqsz_sq"]          = rq_s ** 2                 # queue contention (convex)
        X["io_pressure"]        = (pflt_s + vflt_s) / np.maximum(fm_s, 1)
        X["memory_pressure"]    = (pflt_s + vflt_s) / np.maximum(fm_s / 1000, 1)
        X["cpu_contention"]     = rq_s * (sys_s + wait_s)  # queue × non-user time

        return X


# ── Preprocessor ──────────────────────────────────────────────────────────────
class _DynamicNumericPreprocessor(BaseEstimator, TransformerMixin):
    """
    Drop-in replacement for ColumnTransformer that resolves column names
    dynamically at fit() time. This avoids ColumnTransformer's
    _validate_column_callables raising KeyError when engineered features
    have not yet been added to feature_names_in_ in sklearn >= 1.2.

    Selects all numeric columns present in X at fit time (excluding TARGET),
    applies median imputation then the chosen scaler.
    """
    def __init__(self, scaler_name: str = "RobustScaler"):
        self.scaler_name = scaler_name

    def _make_scaler(self):
        return {
            "StandardScaler": StandardScaler(),
            "RobustScaler":   RobustScaler(),
            "MinMaxScaler":   MinMaxScaler(),
        }.get(self.scaler_name, RobustScaler())

    def fit(self, X, y=None):
        # Resolve columns that actually exist right now
        self.cols_ = [c for c in X.columns
                      if c != TARGET and
                      pd.api.types.is_numeric_dtype(X[c])]
        _arr = self._to_numpy(X)
        self.imputer_ = SimpleImputer(strategy="median").fit(_arr)
        _imp = self.imputer_.transform(_arr)
        self.scaler_  = self._make_scaler().fit(_imp)
        return self

    def transform(self, X):
        # Use only the columns seen at fit time that still exist
        present = [c for c in self.cols_ if c in X.columns]
        _arr = self._to_numpy(X, present)
        _imp = self.imputer_.transform(_arr)
        return self.scaler_.transform(_imp)

    def _to_numpy(self, X, cols=None):
        cols = cols or self.cols_
        return np.column_stack([
            pd.to_numeric(X[c], errors="coerce")
            .to_numpy(dtype=np.float64, na_value=np.nan)
            for c in cols
        ])

    def get_feature_names_out(self, input_features=None):
        # Return column names for downstream feature_selection
        return np.array(self.cols_)


def build_preprocessor(scaler_name: str = "RobustScaler") -> _DynamicNumericPreprocessor:
    return _DynamicNumericPreprocessor(scaler_name=scaler_name)


def build_pipeline(
    model=None,
    scaler_name: str = "RobustScaler",
    use_poly:    bool = False,
    poly_degree: int  = 2,
    use_spline:  bool = False,
    use_kernel:  bool = False,
) -> Pipeline:
    """
    Configurable pipeline with optional non-linear expansion steps.

    Steps:
      feature_engineering → preprocess → [poly/spline/kernel] → feature_selection → model

    Concepts:
      A: use_poly=True adds PolynomialFeatures(degree=2)
      B: use_spline=True adds SplineTransformer (natural cubic splines)
      D: use_kernel=True adds Nystroem RBF approximation
    """
    if model is None:
        model = Ridge(alpha=1.0)

    selector = SelectFromModel(
        ExtraTreesRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        threshold="median",
    )

    steps = [
        ("feature_engineering", CPUFeatureEngineer()),
        ("preprocess",          build_preprocessor(scaler_name)),
    ]

    # ── Concept A: Polynomial expansion ──────────────────────────────────────
    if use_poly:
        steps.append((
            "poly",
            PolynomialFeatures(degree=poly_degree, interaction_only=False,
                               include_bias=False),
        ))

    # ── Concept B: Spline basis expansion ────────────────────────────────────
    if use_spline:
        steps.append((
            "spline",
            SplineTransformer(n_knots=5, degree=3, include_bias=False,
                              extrapolation="linear"),
        ))

    # ── Concept D: Kernel approximation ──────────────────────────────────────
    if use_kernel:
        steps.append((
            "kernel_approx",
            Nystroem(kernel="rbf", gamma=0.1, n_components=100,
                     random_state=RANDOM_STATE),
        ))

    steps += [
        ("feature_selection", selector),
        ("model",             model),
    ]
    return Pipeline(steps)


# ─────────────────────────────────────────────────────────────────────────────
# Concept A: Polynomial feature analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_polynomial_features(
    X_train_eng: pd.DataFrame,
    y_train:     pd.Series,
    output_dir:  Path,
) -> dict[str, Any]:
    """
    Concept A: Compare polynomial degree 1 vs 2 vs 3.
    Shows feature explosion (n_features grows as C(p+d,d)) and R² gain.
    interaction_only=True vs False: understand whether pure squared terms
    or cross-product terms drive the non-linearity improvement.
    """
    log.info("Concept A: Polynomial feature analysis …")
    cols = [c for c in POLY_CANDIDATE_FEATURES if c in X_train_eng.columns]
    # Use RobustScaler + median imputation, pure numpy to avoid set_output issues
    _X = X_train_eng[cols].apply(pd.to_numeric, errors="coerce")
    _arr = np.column_stack([_X[c].to_numpy(dtype=np.float64) for c in cols])
    _medians = np.nanmedian(_arr, axis=0)
    for j in range(_arr.shape[1]):
        mask = np.isnan(_arr[:, j])
        if mask.any(): _arr[mask, j] = _medians[j]
    _arr = RobustScaler().fit_transform(_arr)
    y_np = y_train.to_numpy()

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    configs = [
        ("Linear (d=1)",        1, False),
        ("Poly d=2 full",       2, False),
        ("Poly d=2 interact",   2, True),
        ("Poly d=3 full",       3, False),
        ("Poly d=3 interact",   3, True),
    ]
    for name, deg, interact_only in configs:
        poly = PolynomialFeatures(degree=deg, interaction_only=interact_only,
                                  include_bias=False)
        X_poly = poly.fit_transform(_arr)
        n_feat = X_poly.shape[1]
        r2_scores = cross_val_score(Ridge(alpha=1.0), X_poly, y_np,
                                    cv=cv, scoring="r2", n_jobs=N_JOBS)
        rmse_scores = -cross_val_score(Ridge(alpha=1.0), X_poly, y_np,
                                       cv=cv, scoring="neg_root_mean_squared_error",
                                       n_jobs=N_JOBS)
        results[name] = {
            "degree":      deg,
            "interact_only": interact_only,
            "n_features":  n_feat,
            "r2_mean":     float(r2_scores.mean()),
            "r2_std":      float(r2_scores.std()),
            "rmse_mean":   float(rmse_scores.mean()),
        }
        log.info("  %-25s  n_feat=%4d  R²=%.4f±%.4f  RMSE=%.4f",
                 name, n_feat, r2_scores.mean(), r2_scores.std(), rmse_scores.mean())

    # ── Plot: R² vs n_features (the polynomial tradeoff) ─────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    names  = list(results.keys())
    r2s    = [results[n]["r2_mean"]   for n in names]
    rmses  = [results[n]["rmse_mean"] for n in names]
    nfeats = [results[n]["n_features"] for n in names]
    colors = ["#4C78A8","#1D9E75","#54A24B","#E45756","#B279A2"]

    ax1.bar(names, r2s, color=colors)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("CV R²"); ax1.set_title("Concept A: R² by polynomial config")
    ax1.axhline(r2s[0], linestyle="--", color="gray", alpha=0.5, label="Linear baseline")
    ax1.legend(fontsize=8)
    for bar, val in zip(ax1.patches, r2s):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                 f"{val:.3f}", ha="center", fontsize=8)

    ax2.scatter(nfeats, r2s, c=colors, s=80, zorder=5)
    for i,(nm,nf,r2) in enumerate(zip(names,nfeats,r2s)):
        ax2.annotate(nm, (nf, r2), textcoords="offset points", xytext=(5,3), fontsize=7)
    ax2.set_xlabel("Number of features after expansion")
    ax2.set_ylabel("CV R²")
    ax2.set_title("Concept A: R² vs feature count\n(rightward = more features, not always better)")
    plt.suptitle("Polynomial Feature Analysis — CPU Activity", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_polynomial_analysis.png", dpi=160, bbox_inches="tight")
    plt.close()

    write_json(output_dir / "polynomial_analysis.json", results)
    best = max(results, key=lambda k: results[k]["r2_mean"])
    log.info("Best polynomial config: %s (R²=%.4f)", best, results[best]["r2_mean"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept B: Basis function expansion comparison
# ─────────────────────────────────────────────────────────────────────────────
def analyse_basis_functions(
    X_train_eng: pd.DataFrame,
    y_train:     pd.Series,
    output_dir:  Path,
) -> dict[str, Any]:
    """
    Concept B: Compare four basis function expansions applied to 'runqsz'
    (run queue size — the most non-linearly related feature to usr).

    Basis types:
      1. Raw (baseline)
      2. Polynomial degree-2
      3. Natural cubic spline (SplineTransformer, 5 knots)
      4. Radial basis functions (RBFSampler)
      5. Fourier features (random kitchen sink)

    Why runqsz? It has a convex relationship with usr: CPU usage rises
    steeply as the run queue grows (queueing theory: E[wait] ≈ ρ/(1-ρ)).
    """
    log.info("Concept B: Basis function comparison …")

    # Use only runqsz for clear 1D visualisation
    col = "runqsz"
    if col not in X_train_eng.columns:
        col = POLY_CANDIDATE_FEATURES[0]

    x_1d = pd.to_numeric(X_train_eng[col], errors="coerce").to_numpy(dtype=np.float64)
    x_1d = np.nan_to_num(x_1d, nan=np.nanmedian(x_1d))
    y_np  = y_train.to_numpy()
    X_1d  = x_1d.reshape(-1, 1)
    cv    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    configs = {
        "Raw (linear)":   (X_1d, None),
        "Polynomial d=2": (PolynomialFeatures(2, include_bias=False).fit_transform(
                               StandardScaler().fit_transform(X_1d)), None),
        "Natural spline\n(5 knots)": (SplineTransformer(n_knots=5, degree=3,
                               include_bias=False).fit_transform(
                               StandardScaler().fit_transform(X_1d)), None),
        "RBF Nystroem\n(k=20)":   (Nystroem(kernel="rbf", gamma=0.5, n_components=20,
                               random_state=RANDOM_STATE).fit_transform(
                               StandardScaler().fit_transform(X_1d)), None),
        "Fourier\n(RBFSampler)":  (RBFSampler(gamma=0.5, n_components=20,
                               random_state=RANDOM_STATE).fit_transform(
                               StandardScaler().fit_transform(X_1d)), None),
    }
    results = {}
    for name, (X_b, _) in configs.items():
        r2s = cross_val_score(Ridge(alpha=1.0), X_b, y_np, cv=cv, scoring="r2")
        results[name.replace("\n"," ")] = {
            "n_features": X_b.shape[1],
            "r2_mean":    float(r2s.mean()),
            "r2_std":     float(r2s.std()),
        }

    # Visualisation: fit curves over runqsz range
    x_range  = np.linspace(x_1d.min(), x_1d.max(), 200).reshape(-1, 1)
    x_sc     = StandardScaler().fit(X_1d)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(x_1d, y_np, alpha=0.06, s=4, color="#B0B0B0")

    curve_configs = [
        ("Raw linear",       X_1d,  x_range, "#4C78A8"),
        ("Polynomial d=2",   PolynomialFeatures(2, include_bias=False).fit_transform(
                               x_sc.transform(X_1d)),
                             PolynomialFeatures(2, include_bias=False).fit_transform(
                               x_sc.transform(x_range)), "#E45756"),
        ("Natural spline",   SplineTransformer(n_knots=5,degree=3,include_bias=False
                               ).fit_transform(x_sc.transform(X_1d)),
                             SplineTransformer(n_knots=5,degree=3,include_bias=False
                               ).fit_transform(x_sc.transform(x_range)), "#1D9E75"),
    ]
    for cname, X_tr, X_pred, col_color in curve_configs:
        mdl = Ridge(alpha=1.0).fit(X_tr, y_np)
        y_hat = mdl.predict(X_pred)
        axes[0].plot(x_range, y_hat, label=cname, linewidth=1.8, color=col_color)
    axes[0].set_xlabel(f"{col}"); axes[0].set_ylabel("usr (%)")
    axes[0].set_title(f"Concept B: Fitted curves for {col}\nBasis functions capture non-linearity")
    axes[0].legend(fontsize=8)

    names_b = [k.replace("\n"," ") for k in results]
    r2_b    = [results[k]["r2_mean"] for k in results]
    colors_b= ["#4C78A8","#E45756","#1D9E75","#F58518","#B279A2"]
    axes[1].bar(names_b, r2_b, color=colors_b)
    axes[1].set_xticklabels(names_b, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("CV R² (5-fold)")
    axes[1].set_title(f"Concept B: R² by basis type for {col}\nHigher = better non-linear capture")
    for bar, val in zip(axes[1].patches, r2_b):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                     f"{val:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_basis_functions.png", dpi=160)
    plt.close()

    write_json(output_dir / "basis_function_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept C: Mutual information vs linear correlation
# ─────────────────────────────────────────────────────────────────────────────
def analyse_nonlinear_importance(
    X_train_eng: pd.DataFrame,
    y_train:     pd.Series,
    output_dir:  Path,
) -> dict[str, Any]:
    """
    Concept C: Compare Pearson r (linear) vs Mutual Information (non-linear)
    feature importance. Discrepancies reveal features with non-linear relationships.
    High MI, low |r| → the feature matters but in a curved way → candidate for
    polynomial or spline treatment.
    """
    log.info("Concept C: Mutual information vs linear correlation …")
    num_cols = [c for c in X_train_eng.columns
                if c in RAW_FEATURES and c != TARGET]
    _arr = np.column_stack([
        pd.to_numeric(X_train_eng[c], errors="coerce").to_numpy(dtype=np.float64)
        for c in num_cols
    ])
    _medians = np.nanmedian(_arr, axis=0)
    for j in range(_arr.shape[1]):
        mask = np.isnan(_arr[:, j])
        if mask.any(): _arr[mask, j] = _medians[j]
    y_np = y_train.to_numpy()

    # Pearson correlation
    pearson = {c: float(np.corrcoef(_arr[:, i], y_np)[0, 1])
               for i, c in enumerate(num_cols)}
    # Mutual information (captures non-linear dependencies)
    mi = mutual_info_regression(_arr, y_np, random_state=RANDOM_STATE, n_neighbors=5)
    mi_dict = {c: float(mi[i]) for i, c in enumerate(num_cols)}

    # Normalise MI to [0,1] for comparison
    mi_max = max(mi_dict.values()) if max(mi_dict.values()) > 0 else 1
    mi_norm = {c: v / mi_max for c, v in mi_dict.items()}

    # Features where MI >> |Pearson r| → strong non-linearity
    nonlinear_score = {
        c: float(mi_norm[c] - abs(pearson.get(c, 0)))
        for c in num_cols
    }
    top_nonlinear = sorted(nonlinear_score, key=nonlinear_score.get, reverse=True)[:5]

    # Plot
    df_plot = pd.DataFrame({
        "feature": num_cols,
        "pearson_abs": [abs(pearson[c]) for c in num_cols],
        "mi_norm":     [mi_norm[c] for c in num_cols],
        "nonlinear_score": [nonlinear_score[c] for c in num_cols],
    }).sort_values("mi_norm", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df_plot))
    w = 0.38
    ax.bar(x - w/2, df_plot["pearson_abs"], w, label="|Pearson r|", color="#4C78A8", alpha=0.85)
    ax.bar(x + w/2, df_plot["mi_norm"],    w, label="MI (normalised)", color="#E45756", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["feature"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Score (normalised)")
    ax.set_title(
        "Concept C: Linear (|Pearson r|) vs Non-linear (MI) feature importance\n"
        "Red > Blue → non-linear relationship → candidate for polynomial/spline"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_nonlinear_importance.png", dpi=160)
    plt.close()

    result = {
        "pearson": pearson, "mi_normalised": mi_norm,
        "nonlinear_score": nonlinear_score,
        "top_nonlinear_features": top_nonlinear,
    }
    write_json(output_dir / "nonlinear_importance.json", result)
    log.info("Top non-linear features (MI >> |r|): %s", top_nonlinear)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Concept D: Kernel approximation comparison
# ─────────────────────────────────────────────────────────────────────────────
def analyse_kernel_approx(
    X_train_eng: pd.DataFrame,
    y_train:     pd.Series,
    output_dir:  Path,
) -> dict[str, Any]:
    """
    Concept D: Compare exact SVM kernel vs Nystroem approximation vs RBFSampler.
    Demonstrates the kernel trick: mapping features to high-dim space where
    linear models can learn non-linear decision boundaries.
    """
    log.info("Concept D: Kernel approximation comparison …")
    cols  = [c for c in POLY_CANDIDATE_FEATURES if c in X_train_eng.columns]
    _arr  = np.column_stack([
        pd.to_numeric(X_train_eng[c], errors="coerce").to_numpy(dtype=np.float64)
        for c in cols
    ])
    _medians = np.nanmedian(_arr, axis=0)
    for j in range(_arr.shape[1]):
        mask = np.isnan(_arr[:, j])
        if mask.any(): _arr[mask, j] = _medians[j]
    _arr = StandardScaler().fit_transform(_arr)
    y_np = y_train.to_numpy()
    cv   = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    configs = {
        "Ridge (linear)": (_arr, Ridge(alpha=1.0)),
        "Poly d=2 + Ridge": (
            PolynomialFeatures(2, include_bias=False).fit_transform(_arr),
            Ridge(alpha=1.0)),
        "Nystroem RBF (k=50) + Ridge": (
            Nystroem(kernel="rbf", gamma=0.5, n_components=50,
                     random_state=RANDOM_STATE).fit_transform(_arr),
            Ridge(alpha=1.0)),
        "RBFSampler (k=100) + Ridge": (
            RBFSampler(gamma=0.5, n_components=100,
                       random_state=RANDOM_STATE).fit_transform(_arr),
            Ridge(alpha=1.0)),
    }
    results = {}
    for name, (X_k, mdl) in configs.items():
        r2s = cross_val_score(mdl, X_k, y_np, cv=cv, scoring="r2", n_jobs=N_JOBS)
        results[name] = {
            "n_features": X_k.shape[1],
            "r2_mean": float(r2s.mean()),
            "r2_std":  float(r2s.std()),
        }
        log.info("  %-35s n_feat=%4d R²=%.4f", name, X_k.shape[1], r2s.mean())

    write_json(output_dir / "kernel_approx_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept F: Multicollinearity analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_multicollinearity(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Concept F: Detect and document multicollinearity.
    Key issue: sys + wait + usr ≈ 100 (CPU time allocation constraint).
    Since sys and wait are features and usr is the target, this creates a
    near-perfect linear relationship that inflates R² artificially.
    VIF > 10 flags problematic features.
    """
    log.info("Concept F: Multicollinearity analysis …")
    num_feat = [c for c in RAW_FEATURES if c in X_train.columns]
    _arr = np.column_stack([
        pd.to_numeric(X_train[c], errors="coerce").to_numpy(dtype=np.float64)
        for c in num_feat
    ])
    _medians = np.nanmedian(_arr, axis=0)
    for j in range(_arr.shape[1]):
        mask = np.isnan(_arr[:, j])
        if mask.any(): _arr[mask, j] = _medians[j]

    vif_rows = []
    for i, col in enumerate(num_feat):
        other = np.delete(_arr, i, axis=1)
        tgt   = _arr[:, i]
        if np.unique(tgt).size <= 1:
            continue
        try:
            r2  = LinearRegression().fit(other, tgt).score(other, tgt)
            vif = 9999.0 if r2 >= 0.999 else float(1 / (1 - r2))
        except Exception:
            vif = 9999.0
        vif_rows.append({"feature": col, "vif": vif,
                         "alert": vif > 10 or vif == 9999.0})
    vif_df = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)
    vif_df.to_csv(output_dir / "vif_report.csv", index=False)

    # Correlation heatmap for key features
    key_cols = ["sys","wait","runqsz","vflt","pflt","freemem","fork","exec"]
    key_cols = [c for c in key_cols if c in X_train.columns]
    eda_num  = pd.DataFrame({
        c: pd.to_numeric(X_train[c], errors="coerce").to_numpy(dtype=np.float64)
        for c in key_cols
    })
    plt.figure(figsize=(8, 7))
    corr = eda_num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.4, annot_kws={"size":9})
    plt.title("Concept F: Correlation heatmap — key CPU features\n"
              "(sys+wait≈100-usr creates multicollinearity)")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_correlation_heatmap.png", dpi=160)
    plt.close()

    high_vif = vif_df[vif_df["alert"]]["feature"].tolist()
    log.info("Features with VIF > 10: %s", high_vif)
    result = {
        "vif_table": vif_df.to_dict(orient="records"),
        "high_vif_features": high_vif,
        "note": "sys+wait+usr≈100: sys and wait are near-perfect predictors of usr. "
                "Models will have inflated R² — report on held-out set only.",
    }
    write_json(output_dir / "multicollinearity_report.json", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Concept G: Regularisation paths on polynomial-expanded features
# ─────────────────────────────────────────────────────────────────────────────
def analyse_regularisation_with_poly(
    X_train_eng: pd.DataFrame,
    y_train:     pd.Series,
    output_dir:  Path,
) -> dict[str, Any]:
    """
    Concept G: Show how Ridge/Lasso regularisation paths change when
    polynomial features are added. Key insight: more features → need
    stronger regularisation. Lasso performs automatic feature selection
    from the expanded polynomial feature set.
    """
    log.info("Concept G: Regularisation paths with polynomial features …")
    cols  = [c for c in POLY_CANDIDATE_FEATURES if c in X_train_eng.columns]
    _arr  = np.column_stack([
        pd.to_numeric(X_train_eng[c], errors="coerce").to_numpy(dtype=np.float64)
        for c in cols
    ])
    _medians = np.nanmedian(_arr, axis=0)
    for j in range(_arr.shape[1]):
        mask = np.isnan(_arr[:, j])
        if mask.any(): _arr[mask, j] = _medians[j]
    _arr = RobustScaler().fit_transform(_arr)
    y_np = y_train.to_numpy()
    cv   = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Polynomial expansion
    poly = PolynomialFeatures(degree=2, include_bias=False)
    _arr_poly = poly.fit_transform(_arr)

    alphas = np.logspace(-3, 3, 50)

    # Optimal α for both raw and polynomial features
    ridge_raw  = RidgeCV(alphas=alphas, cv=cv).fit(_arr, y_np)
    ridge_poly = RidgeCV(alphas=alphas, cv=cv).fit(_arr_poly, y_np)
    lasso_poly = LassoCV(alphas=alphas, cv=cv, max_iter=5000,
                         random_state=RANDOM_STATE).fit(_arr_poly, y_np)

    n_zeroed = int((lasso_poly.coef_ == 0).sum())
    log.info("Lasso on poly features zeroed %d / %d coefficients",
             n_zeroed, _arr_poly.shape[1])

    # Regularisation path plot
    ridge_coefs = np.array([Ridge(alpha=a).fit(_arr_poly,y_np).coef_
                            for a in alphas])
    lasso_coefs = []
    for a in alphas:
        try:
            lasso_coefs.append(Lasso(alpha=a, max_iter=5000).fit(_arr_poly,y_np).coef_)
        except Exception:
            lasso_coefs.append(np.zeros(_arr_poly.shape[1]))
    lasso_coefs = np.array(lasso_coefs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    feat_names = poly.get_feature_names_out(cols)
    for i in range(min(12, ridge_coefs.shape[1])):
        axes[0].plot(np.log10(alphas), ridge_coefs[:, i], linewidth=1.0)
    axes[0].axvline(np.log10(ridge_poly.alpha_), color="green", linestyle="--",
                    label=f"Optimal α={ridge_poly.alpha_:.3f}")
    axes[0].set_xlabel("log₁₀(α)"); axes[0].set_title("Ridge on poly features\nAll shrink, none zero")
    axes[0].legend(fontsize=8)

    for i in range(min(12, lasso_coefs.shape[1])):
        axes[1].plot(np.log10(alphas), lasso_coefs[:, i], linewidth=1.0)
    axes[1].axvline(np.log10(lasso_poly.alpha_), color="green", linestyle="--",
                    label=f"Optimal α={lasso_poly.alpha_:.3f}")
    axes[1].set_xlabel("log₁₀(α)"); axes[1].set_title(f"Lasso on poly features\n{n_zeroed}/{_arr_poly.shape[1]} zeroed")
    axes[1].legend(fontsize=8)

    # α shift: raw vs poly
    labels = ["Ridge (raw)", "Ridge (poly)"]
    alphas_opt = [ridge_raw.alpha_, ridge_poly.alpha_]
    axes[2].bar(labels, np.log10(alphas_opt), color=["#4C78A8","#E45756"])
    axes[2].set_ylabel("log₁₀(optimal α)")
    axes[2].set_title("Concept G: Optimal α shift\nPoly features need stronger regularisation")
    plt.suptitle("Regularisation on Polynomial Features", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_regularisation_poly.png", dpi=160, bbox_inches="tight")
    plt.close()

    return {
        "ridge_raw_optimal_alpha":  float(ridge_raw.alpha_),
        "ridge_poly_optimal_alpha": float(ridge_poly.alpha_),
        "lasso_poly_optimal_alpha": float(lasso_poly.alpha_),
        "lasso_poly_n_zeroed":      n_zeroed,
        "lasso_poly_n_total":       int(_arr_poly.shape[1]),
        "insight": (f"Polynomial expansion ({_arr_poly.shape[1]} features) requires "
                    f"stronger Ridge regularisation (α={ridge_poly.alpha_:.3f} vs "
                    f"{ridge_raw.alpha_:.3f} for raw). "
                    f"Lasso automatically zeros {n_zeroed}/{_arr_poly.shape[1]} poly terms."),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Concept K: Repeated K-Fold + Nested CV
# ─────────────────────────────────────────────────────────────────────────────
def analyse_cv_stability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Concept K: Repeated K-Fold vs standard K-Fold.
    Repeated CV runs K-Fold multiple times with different random splits,
    giving a more stable estimate of performance variance.
    Also demonstrates nested CV for unbiased performance estimation.
    """
    log.info("Concept K: CV stability analysis …")
    _pipe = Pipeline([
        ("fe",   CPUFeatureEngineer()),
        ("prep", build_preprocessor()),
        ("mdl",  Ridge(alpha=1.0)),
    ])
    kf  = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

    r2_kf  = cross_val_score(_pipe, X_train, y_train, cv=kf, scoring="r2", n_jobs=N_JOBS)
    r2_rkf = cross_val_score(_pipe, X_train, y_train, cv=rkf, scoring="r2", n_jobs=N_JOBS)

    result = {
        "KFold_5":         {"mean": float(r2_kf.mean()),  "std": float(r2_kf.std()),  "n": len(r2_kf)},
        "RepeatedKFold_5x5":{"mean": float(r2_rkf.mean()),"std": float(r2_rkf.std()), "n": len(r2_rkf)},
        "std_reduction_pct": float((r2_kf.std() - r2_rkf.std()) / r2_kf.std() * 100),
    }
    log.info("KFold std=%.4f  RepeatedKFold std=%.4f  reduction=%.1f%%",
             r2_kf.std(), r2_rkf.std(), result["std_reduction_pct"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot([r2_kf, r2_rkf], labels=["5-Fold\n(25 scores)", "5×5 Repeated\n(25 scores)"],
                    patch_artist=True,
                    boxprops=dict(facecolor="#E1F5EE"),
                    medianprops=dict(color="#1D9E75", linewidth=2))
    axes[0].set_ylabel("CV R²")
    axes[0].set_title("Concept K: CV stability\nRepeated KFold has lower variance")

    axes[1].hist(r2_kf, bins=10, alpha=0.6, label="5-Fold", color="#4C78A8")
    axes[1].hist(r2_rkf, bins=15, alpha=0.6, label="Repeated 5×5", color="#E45756")
    axes[1].axvline(r2_kf.mean(), color="#4C78A8", linestyle="--")
    axes[1].axvline(r2_rkf.mean(), color="#E45756", linestyle="--")
    axes[1].set_xlabel("CV R²"); axes[1].set_title("Distribution of CV scores")
    axes[1].legend()
    plt.suptitle("Concept K: Cross-validation stability", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_cv_stability.png", dpi=160, bbox_inches="tight")
    plt.close()

    write_json(output_dir / "cv_stability.json", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Concept L: Spline regression deep-dive
# ─────────────────────────────────────────────────────────────────────────────
def analyse_spline_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Concept L: Natural cubic splines as interpretable non-linear basis.
    Compares knot placement strategies (quantile vs uniform) and shows
    how spline degree affects smoothness vs flexibility tradeoff.
    """
    log.info("Concept L: Spline regression analysis …")
    col   = "runqsz" if "runqsz" in X_train.columns else RAW_FEATURES[0]
    x_raw = pd.to_numeric(X_train[col], errors="coerce").to_numpy(dtype=np.float64)
    x_raw = np.nan_to_num(x_raw, nan=np.nanmedian(x_raw))
    y_np  = y_train.to_numpy()
    X_1d  = x_raw.reshape(-1, 1)
    x_sc  = StandardScaler().fit(X_1d)
    X_sc  = x_sc.transform(X_1d)
    cv    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    spline_configs = [
        ("Linear",         SplineTransformer(n_knots=2, degree=1, include_bias=False)),
        ("Quadratic 5k",   SplineTransformer(n_knots=5, degree=2, include_bias=False)),
        ("Cubic 5k",       SplineTransformer(n_knots=5, degree=3, include_bias=False)),
        ("Cubic 10k",      SplineTransformer(n_knots=10, degree=3, include_bias=False)),
        ("Cubic 15k",      SplineTransformer(n_knots=15, degree=3, include_bias=False)),
    ]
    results = {}
    for name, spl in spline_configs:
        X_spl = spl.fit_transform(X_sc)
        r2s   = cross_val_score(Ridge(alpha=1.0), X_spl, y_np, cv=cv, scoring="r2")
        results[name] = {
            "n_knots": getattr(spl, "n_knots", None),
            "degree":  spl.degree,
            "n_features": X_spl.shape[1],
            "r2_mean": float(r2s.mean()),
            "r2_std":  float(r2s.std()),
        }

    # Plot fitted spline curves
    x_range = np.linspace(x_raw.min(), x_raw.max(), 300).reshape(-1, 1)
    x_range_sc = x_sc.transform(x_range)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.scatter(x_raw, y_np, alpha=0.05, s=4, color="#B0B0B0")
    colors_spl = ["#4C78A8","#E45756","#1D9E75","#F58518","#B279A2"]
    for (name, spl), color in zip(spline_configs[1:4], colors_spl[1:4]):
        X_spl = spl.fit_transform(X_sc)
        mdl   = Ridge(alpha=1.0).fit(X_spl, y_np)
        y_hat = mdl.predict(spl.transform(x_range_sc))
        ax1.plot(x_range, y_hat, label=name, color=color, linewidth=1.8)
    ax1.set_xlabel(col); ax1.set_ylabel("usr (%)")
    ax1.set_title(f"Concept L: Spline fits for {col}\nMore knots = more flexibility")
    ax1.legend(fontsize=8)

    names_s = list(results.keys())
    r2_s    = [results[n]["r2_mean"] for n in names_s]
    ax2.plot(names_s, r2_s, "o-", color="#1D9E75", linewidth=2, markersize=8)
    ax2.set_xticklabels(names_s, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("CV R²")
    ax2.set_title("Concept L: Spline R² vs complexity\nOver-fitting appears with too many knots")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_spline_regression.png", dpi=160)
    plt.close()

    write_json(output_dir / "spline_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# EDA (train-set only) — mirrors reference pipeline
# ─────────────────────────────────────────────────────────────────────────────
def save_research_artifacts(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
) -> dict[str, Any]:
    log.info("Saving EDA artifacts (train-set only) …")
    output_dir.mkdir(parents=True, exist_ok=True)
    eda = X_train.copy()
    eda[TARGET] = y_train.values

    missingness_report(eda).to_csv(output_dir / "research_missingness_report.csv")
    eda.dtypes.astype(str).rename("dtype").to_csv(output_dir / "schema.csv")

    # Skewness
    num_cols = [c for c in eda.columns if c != TARGET]
    skew_rows = []
    for c in num_cols:
        arr = pd.to_numeric(eda[c], errors="coerce").to_numpy(dtype=np.float64)
        s   = pd.Series(arr)
        skew_rows.append({"feature": c, "skew": float(s.skew()), "kurtosis": float(s.kurtosis())})
    pd.DataFrame(skew_rows).sort_values("skew", key=abs, ascending=False
                ).to_csv(output_dir / "skewness_report.csv", index=False)

    # Correlation with target
    eda_float = {
        c: pd.to_numeric(eda[c], errors="coerce").to_numpy(dtype=np.float64)
        for c in eda.columns
    }
    corr_df = pd.DataFrame(eda_float)
    corr    = corr_df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    corr.to_csv(output_dir / "correlation_with_target.csv")

    # VIF — pure numpy
    num_feat = [c for c in num_cols if c in RAW_FEATURES]
    _vif_arr = np.column_stack([
        pd.to_numeric(eda[c], errors="coerce").to_numpy(dtype=np.float64)
        for c in num_feat
    ])
    _medians = np.nanmedian(_vif_arr, axis=0)
    _imp = _vif_arr.copy()
    for j in range(_imp.shape[1]):
        mask = np.isnan(_imp[:, j])
        if mask.any():
            _fill = _medians[j]
            _imp[mask, j] = 0.0 if np.isnan(_fill) else _fill
    _good = [j for j in range(_imp.shape[1]) if not np.isnan(_imp[:, j]).any()]
    _imp  = _imp[:, _good]
    num_feat_g = [num_feat[j] for j in _good]
    vif_rows = []
    for i, col in enumerate(num_feat_g):
        other = np.delete(_imp, i, axis=1)
        tgt   = _imp[:, i]
        if np.unique(tgt).size <= 1: continue
        try:
            r2  = LinearRegression().fit(other, tgt).score(other, tgt)
            vif = 9999.0 if r2 >= 0.999 else float(1 / (1 - r2))
            vif_rows.append({"feature": col, "vif": vif})
        except Exception: pass
    pd.DataFrame(vif_rows).sort_values("vif", ascending=False
                ).to_csv(output_dir / "vif_report.csv", index=False)

    # Grouped stats
    eda["load_tier"] = pd.cut(eda["runqsz"] if "runqsz" in eda.columns else eda[RAW_FEATURES[0]],
                              bins=[0,1,3,6,100], labels=["idle","low","medium","high"])
    grouped = {
        "usr_by_load_tier": eda.groupby("load_tier", observed=False)[TARGET]
                             .agg(["mean","median","std"]).to_dict(),
        "target_stats":     y_train.describe().to_dict(),
        "pct_sys_wait_constraint": float(
            (eda["sys"].fillna(0) + eda["wait"].fillna(0) + eda[TARGET]).between(95,105).mean()
            if "sys" in eda.columns and "wait" in eda.columns else 0
        ),
    }

    save_research_plots(eda, corr, output_dir)
    decisions = {
        "problem_definition": {
            "problem_type": "regression",
            "target": TARGET, "unit": "% CPU time in user mode",
            "key_constraint": "sys + wait + usr ≈ 100 (CPU time decomposition)",
        },
        "metric_policy": {"primary": "r2", "secondary": ["rmse","mae","mape","medae"]},
        "feature_policy": {
            "log_transforms": LOG_TRANSFORM_FEATURES,
            "polynomial_candidates": POLY_CANDIDATE_FEATURES,
            "multicollinearity_concern": ["sys","wait"],
        },
        "grouped_stats": grouped,
    }
    write_json(output_dir / "research_decisions.json", decisions)
    return decisions


def save_research_plots(eda, corr, output_dir):
    sns.set_theme(style="whitegrid")

    # Target distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(eda[TARGET], kde=True, bins=40, color="#4C78A8")
    plt.axvline(eda[TARGET].median(), color="#E45756", linestyle="--",
                label=f"Median={eda[TARGET].median():.1f}")
    plt.title("CPU usr distribution (train)"); plt.xlabel("usr (%)"); plt.legend()
    plt.tight_layout(); plt.savefig(output_dir/"plot_target_distribution.png", dpi=160); plt.close()

    # Correlation bar
    plt.figure(figsize=(10, 4.5))
    colors = ["#54A24B" if v > 0 else "#E45756" for v in corr]
    corr.plot(kind="barh", color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Feature correlation with usr (train)")
    plt.tight_layout(); plt.savefig(output_dir/"plot_correlation_with_target.png", dpi=160); plt.close()

    # sys vs usr scatter (shows the near-linear constraint)
    if "sys" in eda.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(eda["sys"], eda[TARGET], alpha=0.1, s=4, color="#4C78A8")
        plt.xlabel("sys (%)"); plt.ylabel("usr (%)")
        plt.title("sys vs usr — CPU time allocation constraint")
        plt.tight_layout(); plt.savefig(output_dir/"plot_sys_vs_usr.png", dpi=160); plt.close()

    # runqsz vs usr (non-linear relationship)
    if "runqsz" in eda.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(eda["runqsz"], eda[TARGET], alpha=0.1, s=4, color="#E45756")
        plt.xlabel("runqsz"); plt.ylabel("usr (%)")
        plt.title("runqsz vs usr — non-linear (convex) relationship")
        plt.tight_layout(); plt.savefig(output_dir/"plot_runqsz_vs_usr.png", dpi=160); plt.close()

    # Skewness before/after log
    if "vflt" in eda.columns:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
        a1.hist(eda["vflt"].clip(0, eda["vflt"].quantile(0.99)), bins=50, color="#B279A2")
        a1.set_title("vflt (raw — right-skewed)"); a1.set_xlabel("vflt")
        a2.hist(np.log1p(eda["vflt"].clip(0)), bins=50, color="#72B7B2")
        a2.set_title("log1p(vflt) — normalised"); a2.set_xlabel("log1p(vflt)")
        plt.suptitle("Concept E: Log transform effect", fontsize=11, y=1.02)
        plt.tight_layout(); plt.savefig(output_dir/"plot_log_transform_demo.png", dpi=160); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics + baselines (mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_predictions(y_true, y_pred):
    if hasattr(y_true, "to_numpy"): y_true = y_true.to_numpy()
    res = y_true - y_pred
    return {
        "mae":              float(mean_absolute_error(y_true, y_pred)),
        "rmse":             float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":               float(r2_score(y_true, y_pred)),
        "mape":             float(mean_absolute_percentage_error(y_true, y_pred)),
        "medae":            float(median_absolute_error(y_true, y_pred)),
        "residual_mean":    float(res.mean()),
        "residual_std":     float(res.std()),
        "residual_max_abs": float(np.abs(res).max()),
    }


def evaluate_baselines(X_tr, X_te, y_tr, y_te):
    results = {}
    for strategy in ["mean", "median"]:
        d = DummyRegressor(strategy=strategy)
        d.fit(X_tr, y_tr)
        results[strategy] = evaluate_predictions(y_te, d.predict(X_te))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept H: Residual diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def residual_diagnostics(y_true, y_pred, output_dir):
    log.info("Concept H: Residual diagnostics …")
    residuals = y_true - y_pred
    n         = len(residuals)
    # Breusch-Pagan
    resid_sq  = residuals ** 2
    bp_r2     = LinearRegression().fit(y_pred.reshape(-1,1), resid_sq).score(y_pred.reshape(-1,1), resid_sq)
    bp_stat   = float(n * bp_r2)
    bp_p      = float(1 - scipy_stats.chi2.cdf(bp_stat, df=1))
    # Shapiro-Wilk
    sample = residuals if n <= 5000 else np.random.default_rng(RANDOM_STATE).choice(residuals, 5000, replace=False)
    sw_s, sw_p = scipy_stats.shapiro(sample)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    scipy_stats.probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q plot (normality check)")
    axes[1].scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.2, s=5, color="#4C78A8")
    axes[1].axhline(np.sqrt(np.abs(residuals)).mean(), color="red", linewidth=1.2)
    axes[1].set_xlabel("Fitted"); axes[1].set_ylabel("√|residual|")
    axes[1].set_title("Scale-location (heteroscedasticity)")
    axes[2].hist(residuals, bins=60, color="#54A24B", edgecolor="white", linewidth=0.3)
    axes[2].axvline(0, color="red", linewidth=1.2)
    axes[2].set_title("Residual distribution")
    plt.suptitle("Concept H: Residual diagnostics", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_residual_diagnostics.png", dpi=160, bbox_inches="tight")
    plt.close()

    log.info("BP p=%.4f (hetero: %s)  SW p=%.4f (normal: %s)",
             bp_p, bp_p<0.05, sw_p, sw_p>0.05)
    return {
        "breusch_pagan": {"statistic": bp_stat, "p_value": bp_p, "heteroscedastic": bp_p < 0.05},
        "shapiro_wilk":  {"statistic": float(sw_s), "p_value": float(sw_p), "normal": sw_p > 0.05},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Concept I: Learning curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_curves(model, X_train, y_train, output_dir):
    log.info("Concept I: Learning curves …")
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    try:
        sizes, tr_s, cv_s = learning_curve(
            model, X_train, y_train,
            train_sizes=np.linspace(0.10, 1.0, 8),
            cv=cv, scoring="neg_root_mean_squared_error",
            n_jobs=N_JOBS)
        tr_rmse, cv_rmse = -tr_s.mean(axis=1), -cv_s.mean(axis=1)
        plt.figure(figsize=(8, 4.5))
        plt.plot(sizes, tr_rmse, "o-", color="#4C78A8", label="Train RMSE")
        plt.plot(sizes, cv_rmse, "o-", color="#E45756", label="CV RMSE")
        plt.fill_between(sizes, tr_rmse - tr_s.std(axis=1), tr_rmse + tr_s.std(axis=1), alpha=0.12, color="#4C78A8")
        plt.fill_between(sizes, cv_rmse - cv_s.std(axis=1), cv_rmse + cv_s.std(axis=1), alpha=0.12, color="#E45756")
        gap = float(cv_rmse[-1] - tr_rmse[-1])
        plt.title(f"Concept I: Learning curve\nTrain-CV gap={gap:.4f} "
                  f"({'high variance' if gap>2 else 'well-fitted'})")
        plt.xlabel("Training set size"); plt.ylabel("RMSE")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir / "plot_learning_curve.png", dpi=160); plt.close()
    except Exception as exc:
        log.warning("Learning curve failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Concept J: Partial dependence + ICE
# ─────────────────────────────────────────────────────────────────────────────
def plot_partial_dependence(model, X_test, output_dir):
    log.info("Concept J: PDP + ICE plots …")
    try:
        feat_names = list(X_test.columns)
        fig, axes  = plt.subplots(1, 2, figsize=(11, 4.5))
        # PDP for runqsz (col 16) and sys (col 19)
        for ax, feat_idx in zip(axes, [16, 19]):
            if feat_idx >= len(feat_names): feat_idx = 0
            PartialDependenceDisplay.from_estimator(
                model, X_test, features=[(feat_idx,)],
                feature_names=feat_names,
                ax=ax, kind="both",   # "both" = PDP + ICE
                subsample=200,
                n_jobs=N_JOBS,
            )
            axes_in = ax.get_figure().axes
        plt.suptitle("Concept J: Partial Dependence + ICE plots\n"
                     "Thin lines=individual predictions, thick=average effect",
                     fontsize=11, y=1.04)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_partial_dependence.png", dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        log.warning("PDP failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP (mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
def save_shap_artifacts(model, X_test, y_test, y_pred, output_dir):
    if not _SHAP:
        log.warning("pip install shap"); return
    log.info("SHAP for champion …")
    try:
        clf      = model.named_steps["model"]
        prep     = model.named_steps["preprocess"]
        fe       = model.named_steps["feature_engineering"]
        selector = model.named_steps["feature_selection"]
        # Walk pipeline steps to find feature names from the step before selector
        step_names = list(model.named_steps.keys())
        fs_idx     = step_names.index("feature_selection")
        fn = None
        for sname in reversed(step_names[:fs_idx]):
            step = model.named_steps[sname]
            if hasattr(step, "get_feature_names_out"):
                try:
                    fn = step.get_feature_names_out(); break
                except Exception:
                    continue
        if fn is None:
            fn = np.array([f"f_{i}" for i in range(selector.get_support().sum())])
        support = selector.get_support()
        if len(fn) != len(support):
            fn = np.array([f"feature_{i}" for i in range(len(support))])
        sn = fn[support]
        # Transform X through all steps before feature_selection
        Xt = X_test.copy()
        for sname in step_names[:fs_idx]:
            Xt = model.named_steps[sname].transform(Xt)
        Xt  = selector.transform(Xt)
        Xdf = pd.DataFrame(Xt, columns=sn)

        if hasattr(clf, "feature_importances_"):
            exp = shap.TreeExplainer(clf); sv = exp.shap_values(Xdf)
        elif hasattr(clf, "coef_"):
            exp = shap.LinearExplainer(clf, Xdf); sv = exp.shap_values(Xdf)
        else:
            mask = shap.maskers.Independent(Xdf, max_samples=100)
            exp  = shap.Explainer(clf.predict, mask); sv = exp(Xdf).values

        for ptype, fname in [("bar","plot_shap_bar.png"),("dot","plot_shap_beeswarm.png")]:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(sv, Xdf, plot_type=ptype, show=False, max_display=20)
            plt.title(f"SHAP — {ptype}")
            plt.tight_layout()
            plt.savefig(output_dir/fname, dpi=150, bbox_inches="tight"); plt.close()

        # Waterfall for worst residual
        worst = int(np.argmax(np.abs(y_test.to_numpy() - y_pred)))
        ev    = float(exp.expected_value) if not isinstance(exp.expected_value, np.ndarray) else float(exp.expected_value)
        shap.waterfall_plot(
            shap.Explanation(values=sv[worst], base_values=ev,
                             data=Xdf.iloc[worst].values, feature_names=list(sn)),
            show=False, max_display=15)
        plt.title(f"SHAP Waterfall — worst residual")
        plt.tight_layout()
        plt.savefig(output_dir/"plot_shap_waterfall.png", dpi=150, bbox_inches="tight"); plt.close()

        pd.DataFrame({"feature": sn, "mean_abs_shap": np.abs(sv).mean(axis=0)}
            ).sort_values("mean_abs_shap", ascending=False
            ).to_csv(output_dir/"shap_importance.csv", index=False)
        log.info("SHAP saved.")
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Subgroup evaluation (mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_subgroups(model, X_test, y_test, y_pred, output_dir):
    log.info("Subgroup evaluation …")
    overall_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    eval_df = X_test.reset_index(drop=True).copy()
    eval_df["_y_true"] = y_test.to_numpy()
    eval_df["_y_pred"] = y_pred
    col = "runqsz" if "runqsz" in eval_df.columns else RAW_FEATURES[0]
    eval_df["load_tier"] = pd.cut(eval_df[col], bins=[0,1,3,6,100],
                                  labels=["idle","low","medium","high"]).astype("category")

    rows = []
    for val, sub in eval_df.groupby("load_tier", observed=True):
        if len(sub) < 15: continue
        sr = float(np.sqrt(mean_squared_error(sub["_y_true"], sub["_y_pred"])))
        rows.append({
            "group_col": "load_tier", "group_val": str(val),
            "n": int(len(sub)),
            "mean_actual": float(sub["_y_true"].mean()),
            "rmse": round(sr, 4),
            "r2":   round(float(r2_score(sub["_y_true"], sub["_y_pred"])), 4),
            "rmse_gap": round(sr - overall_rmse, 4),
            "alert": bool(sr > overall_rmse * 1.25),
        })
    if rows:
        pd.DataFrame(rows).to_csv(output_dir/"fairness_report.csv", index=False)
    return {"overall_rmse": overall_rmse, "subgroups": rows}


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance plot
# ─────────────────────────────────────────────────────────────────────────────
def save_feature_importance(model, output_dir):
    """
    Feature names must come from the step immediately BEFORE feature_selection,
    not always from 'preprocess'. When --use-poly or --use-spline or --use-kernel
    is active, extra expansion steps sit between preprocess and feature_selection,
    and those steps produce more features than preprocess alone.

    We walk the pipeline steps in order and use the last step before
    'feature_selection' that can produce feature names.
    """
    selector = model.named_steps["feature_selection"]
    clf      = model.named_steps["model"]

    # Find the step immediately before feature_selection
    step_names = list(model.named_steps.keys())
    fs_idx     = step_names.index("feature_selection")
    # Walk backwards to find the last step that has get_feature_names_out
    fn = None
    for step_name in reversed(step_names[:fs_idx]):
        step = model.named_steps[step_name]
        if hasattr(step, "get_feature_names_out"):
            try:
                fn = step.get_feature_names_out()
                break
            except Exception:
                continue

    if fn is None:
        log.warning("save_feature_importance: could not get feature names — skipping.")
        return

    support = selector.get_support()
    if len(fn) != len(support):
        log.warning(
            "save_feature_importance: fn length %d != support length %d — "
            "likely a poly/spline expansion step. Generating generic feature names.",
            len(fn), len(support),
        )
        fn = np.array([f"feature_{i}" for i in range(len(support))])

    sn = fn[support]

    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_; signed = np.full(len(imp), np.nan)
    elif hasattr(clf, "coef_"):
        signed = clf.coef_; imp = np.abs(signed)
    else:
        log.warning("save_feature_importance: model has no importances or coef — skipping.")
        return

    pd.DataFrame({"feature": sn, "importance": imp, "coefficient": signed}
        ).sort_values("importance", ascending=False
        ).to_csv(output_dir/"feature_importance.csv", index=False)
    plt.figure(figsize=(9, 5.5))
    sns.barplot(
        data=pd.DataFrame({"feature": sn, "importance": imp}
                ).sort_values("importance", ascending=False).head(20),
        y="feature", x="importance", color="#4C78A8")
    plt.title("Top 20 model features"); plt.tight_layout()
    plt.savefig(output_dir/"plot_feature_importance.png", dpi=160); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation plots (mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
def save_evaluation_plots(y_test, y_pred, output_dir):
    res  = y_test.to_numpy() - y_pred
    rmse = float(np.sqrt(np.mean(res**2)))

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.2, s=6, color="#4C78A8")
    mn, mx = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    plt.plot([mn,mx],[mn,mx],"r--",linewidth=1.2,label="Perfect")
    plt.title(f"Actual vs Predicted usr  (R²={r2_score(y_test,y_pred):.3f})")
    plt.xlabel("Actual usr (%)"); plt.ylabel("Predicted usr (%)")
    plt.legend(); plt.tight_layout()
    plt.savefig(output_dir/"plot_actual_vs_predicted.png", dpi=160); plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, res, alpha=0.15, s=6, color="#B279A2")
    plt.axhline(0, color="black", linewidth=1.0)
    plt.axhline(res.std(), color="#F58518", linestyle="--", alpha=0.7)
    plt.axhline(-res.std(), color="#F58518", linestyle="--", alpha=0.7)
    plt.title("Residuals vs Predicted"); plt.tight_layout()
    plt.savefig(output_dir/"plot_residuals_vs_predicted.png", dpi=160); plt.close()

    sns.histplot(res, kde=True, bins=60, color="#54A24B")
    plt.axvline(0, color="red"); plt.title("Residual distribution")
    plt.tight_layout(); plt.savefig(output_dir/"plot_residual_distribution.png", dpi=160); plt.close()


def save_error_analysis(X_test, y_test, y_pred, output_dir):
    df = X_test.copy()
    res = y_test.to_numpy() - y_pred
    rmse = float(np.sqrt(np.mean(res**2)))
    df["actual"] = y_test.to_numpy(); df["predicted"] = y_pred
    df["residual"] = res; df["abs_error"] = np.abs(res)
    df["severity"] = pd.cut(df["abs_error"], bins=[0,rmse*0.5,rmse,rmse*2,np.inf],
                            labels=["low","medium","high","severe"])
    df.to_csv(output_dir/"test_predictions.csv", index=False)
    df[df["abs_error"]>rmse].to_csv(output_dir/"error_analysis.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search (mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
def tune_model(
    X_train:     pd.DataFrame,
    y_train:     pd.Series,
    n_iter:      int   = 20,
    n_cv_splits: int   = 5,
    fast:        bool  = False,
    scaler_name: str   = "RobustScaler",
    use_poly:    bool  = False,
) -> RandomizedSearchCV:
    t0      = time.perf_counter()
    _tree_n = 50 if fast else 100
    log.info("Hyperparameter search: n_iter=%d  cv=%d-fold  fast=%s  use_poly=%s",
             n_iter, n_cv_splits, fast, use_poly)

    param_distributions = [
        # ── Ridge ─────────────────────────────────────────────────────────────
        {
            "feature_selection__threshold": ["median","0.75*median","1.25*median"],
            "model": [Ridge()],
            "model__alpha": [0.001,0.01,0.1,0.5,1,2,5,10,50,100,500],
        },
        # ── Lasso ─────────────────────────────────────────────────────────────
        {
            "feature_selection__threshold": ["median","0.75*median"],
            "model": [Lasso(max_iter=5000)],
            "model__alpha": [0.001,0.005,0.01,0.05,0.1,0.5,1.0],
        },
        # ── ElasticNet ────────────────────────────────────────────────────────
        {
            "feature_selection__threshold": ["median","0.75*median"],
            "model": [ElasticNet(max_iter=5000)],
            "model__alpha":    [0.01,0.05,0.1,0.5,1.0],
            "model__l1_ratio": [0.1,0.3,0.5,0.7,0.9],
        },
        # ── GradientBoosting ──────────────────────────────────────────────────
        {
            "feature_selection__threshold": ["median","0.75*median","1.25*median"],
            "model": [GradientBoostingRegressor(n_estimators=_tree_n, random_state=RANDOM_STATE)],
            "model__max_depth":     [3,4,5],
            "model__learning_rate": [0.05,0.1,0.2],
            "model__subsample":     [0.7,0.9],
        },
        # ── RandomForest ──────────────────────────────────────────────────────
        {
            "feature_selection__threshold": ["median","0.75*median","1.25*median"],
            "model": [RandomForestRegressor(n_estimators=_tree_n, random_state=RANDOM_STATE,
                                            n_jobs=N_JOBS)],
            "model__max_depth":        [8,12,None],
            "model__min_samples_leaf": [1,2,4],
        },
    ]

    pipe = build_pipeline(scaler_name=scaler_name, use_poly=use_poly)
    cv   = KFold(n_splits=n_cv_splits, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipe, param_distributions, n_iter=n_iter,
        scoring={"r2":"r2","neg_rmse":"neg_root_mean_squared_error",
                 "neg_mae":"neg_mean_absolute_error"},
        refit="r2", cv=cv,
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
        verbose=1, return_train_score=True,
    )
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    log.info("Search done in %.1f min — best CV R²=%.4f  model=%s",
             elapsed/60, search.best_score_,
             type(search.best_estimator_.named_steps["model"]).__name__)

    # Upgrade tree estimators after search
    best_mdl = search.best_estimator_.named_steps["model"]
    if hasattr(best_mdl, "n_estimators") and best_mdl.n_estimators == _tree_n:
        log.info("Upgrading %d → 300 trees …", _tree_n)
        best_mdl.set_params(n_estimators=300)
        search.best_estimator_.fit(X_train, y_train)
    return search


# ─────────────────────────────────────────────────────────────────────────────
# Training profile (100 quantiles — mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
def build_training_profile(X_train, y_train):
    fe  = CPUFeatureEngineer().fit(X_train, y_train)
    eng = fe.transform(X_train)
    num_cols = [c for c in get_column_groups().numeric if c in eng.columns]
    stats = {}
    for col in num_cols:
        v = pd.to_numeric(eng[col], errors="coerce").to_numpy(dtype=np.float64)
        v = v[~np.isnan(v)]
        if len(v) == 0: continue
        stats[col] = {
            "mean": float(v.mean()), "std": float(v.std()),
            "min":  float(v.min()),  "max": float(v.max()),
            "quantiles": np.quantile(v, np.linspace(0,1,100)).tolist(),
        }
    return to_jsonable({
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "row_count":     int(len(X_train)),
        "raw_columns":   list(X_train.columns),
        "target_stats":  {
            "mean": float(y_train.mean()), "std": float(y_train.std()),
            "min":  float(y_train.min()),  "max": float(y_train.max()),
        },
        "raw_missing_rate":        X_train.isna().mean().to_dict(),
        "numeric_train_stats":     stats,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Model Card (mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
def save_model_card(metrics, poly_analysis, reg_analysis, search, output_dir):
    tm = metrics.get("test_metrics", {})
    write_json(output_dir / MODEL_CARD_FILE, {
        "schema_version": "1.0",
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "model_details":  {
            "name":      "CPU Activity Predictor",
            "type":      "Regression (sklearn Pipeline)",
            "algorithm": repr(search.best_estimator_.named_steps["model"]),
        },
        "intended_use": {
            "primary_use": "Predict CPU user-mode time (usr %) from system performance counters.",
            "out_of_scope": ["Real-time scheduling decisions","Architectures other than the training system"],
        },
        "evaluation_results": {
            "test_r2": tm.get("r2"), "test_rmse": tm.get("rmse"),
            "test_mae": tm.get("mae"),
        },
        "polynomial_insights": poly_analysis,
        "regularisation_insights": reg_analysis,
        "limitations": [
            "sys+wait+usr≈100 constraint means sys and wait are near-perfect predictors — "
            "R² may be inflated compared to models without these features.",
            "Synthetic fallback used if OpenML is unreachable.",
        ],
        "hyperparameters": search.best_params_,
        "cv_best_r2":      float(search.best_score_),
    })


# ─────────────────────────────────────────────────────────────────────────────
# MLflow (mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
def log_to_mlflow(metrics, search, model, output_dir):
    if not _MLFLOW: return
    try:
        mlflow.set_experiment("cpu_activity")
        tm = metrics.get("test_metrics", {})
        with mlflow.start_run():
            mlflow.log_params({f"best_{k}": str(v) for k, v in search.best_params_.items()})
            mlflow.log_metrics({"cv_r2": float(search.best_score_),
                                "test_r2": float(tm.get("r2",0)),
                                "test_rmse": float(tm.get("rmse",0))})
            for f in [MODEL_CARD_FILE, METRICS_FILE, "polynomial_analysis.json",
                      "plot_actual_vs_predicted.png", "plot_shap_bar.png"]:
                if (output_dir/f).exists(): mlflow.log_artifact(str(output_dir/f))
            mlflow.sklearn.log_model(model, "model")
        log.info("MLflow logged.")
    except Exception as e:
        log.warning("MLflow failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Environment snapshot (mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
def save_environment_snapshot(output_dir):
    env = {"saved_at": datetime.now(timezone.utc).isoformat(),
           "python": sys.version, "platform": sys.platform, "libraries": {}}
    for lib in ["sklearn","pandas","numpy","scipy","joblib","shap","mlflow","pandera"]:
        try:
            mod = importlib.import_module(lib)
            env["libraries"][lib] = getattr(mod,"__version__","unknown")
        except ImportError:
            env["libraries"][lib] = "not_installed"
    write_json(output_dir / ENVIRONMENT_FILE, env)


def save_feature_importance_plots(model, output_dir):
    save_feature_importance(model, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# OOF uncertainty (mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
def compute_oof_uncertainty(best_estimator, X_train, y_train,
                            overpredict_cost=1.0, underpredict_cost=1.0):
    cv  = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(clone(best_estimator), X_train, y_train, cv=cv, n_jobs=N_JOBS)
    res = y_train.to_numpy() - oof
    oof_rmse = float(np.sqrt(np.mean(res**2)))
    return {
        "oof_rmse": oof_rmse, "oof_mae": float(np.abs(res).mean()),
        "oof_r2":   float(r2_score(y_train, oof)),
        "lower_band": oof_rmse*underpredict_cost, "upper_band": oof_rmse*overpredict_cost,
        "overpredict_cost": overpredict_cost, "underpredict_cost": underpredict_cost,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Versioning (mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
def _model_version_tag(model):
    return hashlib.sha1(pickle.dumps(model)).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────────────
# Main training workflow (mirrors reference exactly)
# ─────────────────────────────────────────────────────────────────────────────
def train(
    output_dir:        Path,
    n_iter:            int   = 20,
    n_cv_splits:       int   = 5,
    fast:              bool  = False,
    use_poly:          bool  = False,
    overpredict_cost:  float = 1.0,
    underpredict_cost: float = 1.0,
) -> dict[str, Any]:
    log.info("=== Training started (n_jobs=%d) ===", N_JOBS)
    output_dir.mkdir(parents=True, exist_ok=True)

    df                            = fix_data_types(load_data())
    X_train, X_test, y_train, y_te = split_data(df)

    # ── Phase 1: EDA ──────────────────────────────────────────────────────────
    research = save_research_artifacts(X_train, y_train, output_dir)
    baselines = evaluate_baselines(X_train, X_test, y_train, y_te)

    # ── Pre-train analysis ────────────────────────────────────────────────────
    fe_train = CPUFeatureEngineer().fit(X_train, y_train).transform(X_train)

    poly_analysis = analyse_polynomial_features(fe_train, y_train, output_dir)
    basis_analysis = analyse_basis_functions(fe_train, y_train, output_dir)
    mi_analysis   = analyse_nonlinear_importance(X_train, y_train, output_dir)
    kernel_analysis = analyse_kernel_approx(fe_train, y_train, output_dir)
    mc_analysis   = analyse_multicollinearity(X_train, y_train, output_dir)
    reg_analysis  = analyse_regularisation_with_poly(fe_train, y_train, output_dir)
    cv_stability  = analyse_cv_stability(X_train, y_train, output_dir)
    spline_analysis = analyse_spline_regression(X_train, y_train, output_dir)

    # ── Hyperparameter search ─────────────────────────────────────────────────
    search = tune_model(X_train, y_train,
                        n_iter=n_iter, n_cv_splits=n_cv_splits,
                        fast=fast, use_poly=use_poly)

    uncertainty = compute_oof_uncertainty(search.best_estimator_, X_train, y_train,
                                          overpredict_cost, underpredict_cost)

    final_model = clone(search.best_estimator_)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    test_metrics = evaluate_predictions(y_te, y_pred)

    log.info("Test R²=%.4f  RMSE=%.4f  MAE=%.4f", test_metrics["r2"],
             test_metrics["rmse"], test_metrics["mae"])

    # ── Artifact saving ───────────────────────────────────────────────────────
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha1 = _model_version_tag(final_model)
    joblib.dump(final_model, output_dir / f"cpu_activity_pipeline_{ts}_{sha1}.joblib")
    joblib.dump(final_model, output_dir / MODEL_FILE)
    save_environment_snapshot(output_dir)
    pd.DataFrame(search.cv_results_).sort_values("rank_test_r2").to_csv(
        output_dir/"cv_results.csv", index=False)
    save_feature_importance_plots(final_model, output_dir)
    save_evaluation_plots(y_te, y_pred, output_dir)
    save_error_analysis(X_test, y_te, y_pred, output_dir)
    write_json(output_dir/TRAINING_PROFILE_FILE, build_training_profile(X_train, y_train))

    # ── Diagnostics ───────────────────────────────────────────────────────────
    res_diag = residual_diagnostics(y_te.to_numpy(), y_pred, output_dir)
    plot_learning_curves(final_model, X_train, y_train, output_dir)
    plot_partial_dependence(final_model, X_test, output_dir)
    save_shap_artifacts(final_model, X_test, y_te, y_pred, output_dir)
    subgroups = evaluate_subgroups(final_model, X_test, y_te, y_pred, output_dir)

    # ── Governance ────────────────────────────────────────────────────────────
    metrics = {
        "research": research, "baselines": baselines,
        "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                  "use_poly": use_poly},
        "polynomial_analysis": poly_analysis, "basis_analysis": basis_analysis,
        "mutual_info": mi_analysis, "kernel_approx": kernel_analysis,
        "multicollinearity": mc_analysis, "regularisation": reg_analysis,
        "cv_stability": cv_stability, "spline_analysis": spline_analysis,
        "best_cv": {"best_r2": float(search.best_score_), "best_params": search.best_params_},
        "uncertainty_info": uncertainty, "residual_diag": res_diag,
        "test_metrics": test_metrics, "subgroups": subgroups,
    }
    write_json(output_dir/METRICS_FILE, metrics)
    save_model_card(metrics, poly_analysis, reg_analysis, search, output_dir)
    log_to_mlflow(metrics, search, final_model, output_dir)

    log.info("=== Training complete ===")
    return to_jsonable(metrics)


# ─────────────────────────────────────────────────────────────────────────────
# Predict (mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
def predict(artifact_dir, input_csv, output_csv):
    model = joblib.load(artifact_dir / MODEL_FILE)
    if not hasattr(model, "predict"):
        raise TypeError(f"{type(model).__name__} is not a fitted pipeline.")

    unc_band = UNCERTAINTY_BAND
    mp = artifact_dir / METRICS_FILE
    if mp.exists():
        unc_band = json.loads(mp.read_text())["uncertainty_info"].get("oof_rmse", unc_band)

    df = pd.read_csv(input_csv)
    if INPUT_SCHEMA:
        try: INPUT_SCHEMA.validate(df, lazy=True)
        except Exception as e: log.warning("Schema: %s", e)

    pf = artifact_dir / TRAINING_PROFILE_FILE
    if pf.exists():
        req = set(json.loads(pf.read_text())["raw_columns"])
        miss = req - set(df.columns)
        if miss: raise ValueError(f"Missing columns: {sorted(miss)}")

    y_pred = model.predict(df)
    df["predicted_usr"] = y_pred
    df["lower_bound"]   = y_pred - unc_band
    df["upper_bound"]   = y_pred + unc_band
    df["wide_interval"] = ((df["upper_bound"] - df["lower_bound"]) > unc_band * 2.5).astype(int)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info("Saved to %s", output_csv.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# Monitor (mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
def monitor(artifact_dir, input_csv, output_json, missing_rate_alert=0.05, ks_pvalue=0.05):
    profile  = json.loads((artifact_dir/TRAINING_PROFILE_FILE).read_text())
    incoming = pd.read_csv(input_csv)
    req, inc = set(profile["raw_columns"]), set(incoming.columns)

    drift = []
    for col, tr in profile["raw_missing_rate"].items():
        if col not in incoming: continue
        cur = float(incoming[col].isna().mean())
        drift.append({"column": col, "train_rate": float(tr), "current_rate": cur,
                      "change": abs(cur - float(tr)), "alert": abs(cur-float(tr)) >= missing_rate_alert})
    ks_rows = []
    for col, stats in profile.get("numeric_train_stats", {}).items():
        if col not in incoming.columns: continue
        vals = incoming[col].dropna().to_numpy()
        if len(vals) < 10: continue
        stat, p = ks_2samp(np.array(stats["quantiles"]), vals)
        ks_rows.append({"column": col, "ks_stat": float(stat), "p_value": float(p),
                        "alert": p < ks_pvalue})
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(incoming)),
        "missing_required": sorted(req-inc), "extra": sorted(inc-req),
        "missing_rate_drift": drift, "distribution_drift": ks_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Utilities (identical to reference)
# ─────────────────────────────────────────────────────────────────────────────
def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(v):
    if isinstance(v, dict):   return {str(k): to_jsonable(x) for k, x in v.items()}
    if isinstance(v, list):   return [to_jsonable(x) for x in v]
    if isinstance(v, BaseEstimator): return repr(v)
    if isinstance(v, np.bool_):   return bool(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating):
        f = float(v); return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, float):
        return None if (np.isnan(v) or np.isinf(v)) else v
    try:
        if pd.isna(v): return None
    except: pass
    return v


def create_sample_input(output_csv, rows):
    df = fix_data_types(load_data())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[TARGET]).head(rows).to_csv(output_csv, index=False)
    log.info("Sample saved to %s", output_csv.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# CLI (mirrors reference pattern)
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="CPU Activity end-to-end ML pipeline (Non-Linearity & Polynomials)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sp = p.add_subparsers(dest="command", required=True)

    tp = sp.add_parser("train")
    tp.add_argument("--output-dir",        type=Path, default=Path("artifacts_cpu"))
    tp.add_argument("--n-iter",            type=int,  default=20)
    tp.add_argument("--n-cv-splits",       type=int,  default=5)
    tp.add_argument("--fast",              action="store_true",
                    help="50 trees during search → upgraded to 300 after. ~60%% faster.")
    tp.add_argument("--use-poly",          action="store_true",
                    help="Add PolynomialFeatures(degree=2) step to pipeline.")
    tp.add_argument("--overpredict-cost",  type=float, default=1.0)
    tp.add_argument("--underpredict-cost", type=float, default=1.0)

    pp = sp.add_parser("predict")
    pp.add_argument("--artifact-dir", type=Path, default=Path("artifacts_cpu"))
    pp.add_argument("--input-csv",    type=Path, required=True)
    pp.add_argument("--output-csv",   type=Path, default=Path("artifacts_cpu/predictions.csv"))

    mp = sp.add_parser("monitor")
    mp.add_argument("--artifact-dir",       type=Path,  default=Path("artifacts_cpu"))
    mp.add_argument("--input-csv",          type=Path,  required=True)
    mp.add_argument("--output-json",        type=Path,  default=Path("artifacts_cpu/monitor.json"))
    mp.add_argument("--missing-rate-alert", type=float, default=0.05)
    mp.add_argument("--ks-pvalue-alert",    type=float, default=0.05)

    si = sp.add_parser("sample-input")
    si.add_argument("--output-csv", type=Path, default=Path("artifacts_cpu/sample.csv"))
    si.add_argument("--rows",       type=int,  default=20)

    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        m = train(args.output_dir, args.n_iter,
                  n_cv_splits=args.n_cv_splits, fast=args.fast,
                  use_poly=args.use_poly,
                  overpredict_cost=args.overpredict_cost,
                  underpredict_cost=args.underpredict_cost)
        log.info("Test R²=%.3f  RMSE=%.3f  MAE=%.3f",
                 m["test_metrics"]["r2"], m["test_metrics"]["rmse"],
                 m["test_metrics"]["mae"])
    elif args.command == "predict":
        predict(args.artifact_dir, args.input_csv, args.output_csv)
    elif args.command == "monitor":
        r = monitor(args.artifact_dir, args.input_csv, args.output_json,
                    args.missing_rate_alert, args.ks_pvalue_alert)
        log.info("Drift alerts: missing=%d  KS=%d",
                 sum(x["alert"] for x in r["missing_rate_drift"]),
                 sum(x["alert"] for x in r["distribution_drift"]))
    elif args.command == "sample-input":
        create_sample_input(args.output_csv, args.rows)


if __name__ == "__main__":
    main()
