"""
california_pipeline.py
======================
Industry-standard end-to-end ML pipeline for California Housing price prediction.

Data:
    california_df = fetch_openml(name="california_housing", version=1,
                                 as_frame=True).frame

Mirrors every architectural pattern from titanic-ml-pipeline.py, adapted for
regression (continuous target MedHouseVal) with all-numeric features.

New concepts explored beyond the Titanic reference:
  A.  Feature Scaling analysis  — StandardScaler vs RobustScaler vs MinMaxScaler
      comparison with explicit justification for choice
  B.  Regularisation deep-dive — Ridge (L2), Lasso (L1), ElasticNet (L1+L2)
      with regularisation path plots and coefficient shrinkage visualisation
  C.  Polynomial features      — degree-2 interaction terms for MedInc×HouseAge
      and MedInc×Rooms, with SelectFromModel to prune the explosion
  D.  Geospatial features       — haversine distance to LA/SF, lat-lon cluster
      labels from KMeans (k=20) as an ordinal geographic proxy
  E.  Log / power transforms    — log1p for right-skewed features (AveRooms,
      AveBedrms, Population, AveOccup), Box-Cox for MedInc
  F.  Target distribution analysis — MedHouseVal cap at 5.0 identified, capped
      rows optionally excluded via CLI flag
  G.  Residual diagnostics      — heteroscedasticity Breusch-Pagan test,
      normality Shapiro-Wilk test on residuals
  H.  Learning curves           — train vs CV score vs training set size
  I.  Partial dependence plots  — MedInc and AveRooms marginal effects
  J.  Cross-validation strategy — KFold vs GroupKFold (grouped by geo-cluster)
      showing why spatial CV prevents leakage for geo features

Industry-standard regression metrics:
  MAE    — Mean Absolute Error (same unit as target, business-interpretable)
  RMSE   — Root Mean Squared Error (penalises large errors quadratically)
  R²     — Coefficient of determination (proportion of variance explained)
  MAPE   — Mean Absolute Percentage Error (scale-free, % intuitive)
  MedAE  — Median Absolute Error (robust to outliers / capped values)

Regularisation vocabulary (used throughout):
  L2 / Ridge   — penalty on sum of squared coefficients → shrinks all toward 0
  L1 / Lasso   — penalty on sum of |coefficients| → drives some exactly to 0
  ElasticNet   — α·L1 + (1-α)·L2 → best of both: sparse + stable under collinearity

Usage:
  python california_pipeline.py train   --output-dir artifacts_ca
  python california_pipeline.py predict --artifact-dir artifacts_ca --input-csv sample.csv
  python california_pipeline.py monitor --artifact-dir artifacts_ca --input-csv new.csv
  python california_pipeline.py sample-input --output-csv sample.csv --rows 20
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
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── matplotlib scratch dir before any pyplot import (mirrors titanic reference) ─
_MPLCONFIGDIR = Path("artifacts_ca") / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RandomizedSearchCV,
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
    StandardScaler,
)

try:
    import shap; _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

try:
    import mlflow; import mlflow.sklearn; _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

try:
    import pandera.pandas as pa; _PANDERA_AVAILABLE = True
except ImportError:
    _PANDERA_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Logging (identical to reference) ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE          = 42
TARGET                = "median_house_value"
MODEL_FILE            = "california_price_pipeline.joblib"
METRICS_FILE          = "metrics.json"
TRAINING_PROFILE_FILE = "training_profile.json"
MODEL_CARD_FILE       = "model_card.json"
ENVIRONMENT_FILE      = "environment.json"

N_JOBS            = int(os.environ.get("ML_N_JOBS", 1))
UNCERTAINTY_BAND  = float(os.environ.get("ML_UNCERTAINTY_BAND", 0.20))
GEO_CLUSTERS      = 20       # KMeans k for geospatial feature

# Reference cities for haversine distance features
_LA  = (34.0522, -118.2437)
_SF  = (37.7749, -122.4194)
_SAC = (38.5816, -121.4944)

# Fairness / disparity subgroup definitions
FAIRNESS_COLS = ["geo_cluster", "income_tier", "house_age_group"]

# Price cap at 5.0 in original data — rows can be excluded via --drop-capped
PRICE_CAP = 5.0


# ── Column groups (mirrors ColumnGroups dataclass in reference) ────────────────
@dataclass(frozen=True)
class ColumnGroups:
    numeric: list[str]
    categorical: list[str]   # empty — California Housing is all-numeric


def get_column_groups() -> ColumnGroups:
    """
    Feature columns after CaliforniaFeatureEngineer.
    All are numeric — there are no categorical columns in the raw dataset.
    The geo_cluster column (int category) is treated as numeric via ordinal
    encoding inside the preprocessor.
    """
    return ColumnGroups(
        numeric=[
            # ── Raw features ─────────────────────────────────────────────────
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude",
            # ── Log-transformed (New concept E: power/log transforms) ─────────
            "log_AveRooms", "log_AveBedrms", "log_Population", "log_AveOccup",
            "log_MedInc",
            # ── Geospatial features (New concept D) ───────────────────────────
            "dist_LA", "dist_SF", "dist_SAC",
            "geo_cluster",
            # ── Interaction terms (New concept C: polynomial) ─────────────────
            "medinc_rooms_interact",
            "medinc_age_interact",
            "rooms_per_person",
            "bedrooms_ratio",
            "income_per_room",
        ],
        categorical=[],
    )


# ── Data loading (exact pattern requested) ────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Exact loader as specified:
        california_df = fetch_openml(name="california_housing", version=1,
                                     as_frame=True).frame
    Falls back to sklearn's built-in if OpenML is unreachable.
    """
    log.info("Loading California Housing dataset from OpenML …")
    # Canonical column names (sklearn standard) that the rest of the pipeline uses
    _CANONICAL = {
        # OpenML lowercase → sklearn CamelCase (used throughout pipeline)
        "median_income":       "MedInc",
        "housing_median_age":  "HouseAge",
        "total_rooms":         "AveRooms",   # note: OpenML may have raw totals
        "total_bedrooms":      "AveBedrms",
        "population":          "Population",
        "households":          "AveOccup",
        "latitude":            "Latitude",
        "longitude":           "Longitude",
        "median_house_value":  "median_house_value",  # TARGET keeps its name
    }
    try:
        california_df = fetch_openml(
            name="california_housing", version=1, as_frame=True
        ).frame.copy()
        # Rename any OpenML-style columns to canonical names
        rename_map = {c: _CANONICAL[c] for c in california_df.columns if c in _CANONICAL}
        if rename_map:
            california_df = california_df.rename(columns=rename_map)
            log.info("Renamed OpenML columns: %s", rename_map)
        return california_df
    except Exception as exc:
        log.warning("OpenML unavailable (%s) — using sklearn built-in.", exc)
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing(as_frame=True)
        df = d.frame.copy()
        if "MedHouseVal" in df.columns:
            df = df.rename(columns={"MedHouseVal": "median_house_value"})
        return df


def fix_data_types(df: pd.DataFrame, drop_capped: bool = False) -> pd.DataFrame:
    """
    Cast all columns to float64.
    OpenML may return columns as object dtype (strings) — pd.to_numeric handles this.
    Optionally drop rows where MedHouseVal == PRICE_CAP (concept F).
    """
    df = df.copy()
    rebuilt = {}
    for col in df.columns:
        _s = pd.to_numeric(df[col], errors="coerce")
        _arr = _s.to_numpy(dtype=np.float64)
        if np.isnan(_arr).all() and not df[col].isna().all():
            log.warning("fix_data_types: column '%s' is all-NaN after numeric cast "
                        "(dtype was %s, sample=%s)", col, df[col].dtype,
                        str(df[col].iloc[:3].tolist()))
        rebuilt[col] = _arr
    # Re-build with explicit numpy float64 backing to defeat Arrow/Sparse dtypes
    df = pd.DataFrame(rebuilt, index=df.index)

    if drop_capped:
        n_before = len(df)
        df = df[df[TARGET] < PRICE_CAP].copy()
        log.info("Dropped %d capped rows (MedHouseVal == %.1f)", n_before - len(df), PRICE_CAP)

    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Random 80/20 split done BEFORE any EDA to prevent test-set leakage.
    No stratify — continuous target.
    Mirrors split_data from reference pipeline exactly.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Identical to reference pipeline's missingness_report."""
    report = (
        df.isna()
        .agg(["sum", "mean"])
        .T.rename(columns={"sum": "missing_count", "mean": "missing_rate"})
        .sort_values("missing_rate", ascending=False)
    )
    report["dtype"] = df.dtypes.astype(str)
    return report


# ── Pandera input schema (New concept: mirrors reference) ─────────────────────
def build_input_schema():
    if not _PANDERA_AVAILABLE:
        log.warning("pandera not installed — input schema validation skipped.")
        return None
    schema = pa.DataFrameSchema(
        {
            "MedInc":     pa.Column(float, pa.Check.in_range(0, 20),    nullable=True, required=False),
            "HouseAge":   pa.Column(float, pa.Check.in_range(0, 60),    nullable=True, required=False),
            "AveRooms":   pa.Column(float, pa.Check.in_range(0, 100),   nullable=True, required=False),
            "AveBedrms":  pa.Column(float, pa.Check.in_range(0, 50),    nullable=True, required=False),
            "Population": pa.Column(float, pa.Check.ge(0),              nullable=True, required=False),
            "AveOccup":   pa.Column(float, pa.Check.in_range(0, 100),   nullable=True, required=False),
            "Latitude":   pa.Column(float, pa.Check.in_range(32, 42),   nullable=True, required=False),
            "Longitude":  pa.Column(float, pa.Check.in_range(-125, -114), nullable=True, required=False),
        },
        coerce=True,
        strict=False,
    )
    return schema


INPUT_SCHEMA = build_input_schema()


# ── Feature engineering (mirrors TitanicFeatureEngineer pattern exactly) ──────
class CaliforniaFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-driven feature engineering for California Housing.
    Follows the exact BaseEstimator + TransformerMixin pattern from
    TitanicFeatureEngineer — fit-safe, pipeline-compatible.

    New concepts explored:
      C. Interaction / polynomial terms: MedInc×Rooms, MedInc×Age
      D. Geospatial: haversine dist to LA/SF/SAC, KMeans geo-cluster
      E. Log transforms: log1p for all right-skewed counts/rates
    """

    def __init__(self, n_geo_clusters: int = GEO_CLUSTERS) -> None:
        self.n_geo_clusters = n_geo_clusters

    def fit(self, X: pd.DataFrame, y=None) -> "CaliforniaFeatureEngineer":
        """
        Learn the geo-cluster centroids from the training set only.
        (Mirrors TitanicFeatureEngineer.fit which learns rare_titles_ from train.)
        """
        coords = X[["Latitude", "Longitude"]].to_numpy()
        self._kmeans = KMeans(
            n_clusters=self.n_geo_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        self._kmeans.fit(coords)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # ── Concept E: Log transforms for right-skewed features ────────────────
        for col in ["AveRooms", "AveBedrms", "Population", "AveOccup"]:
            X[f"log_{col}"] = np.log1p(X[col].clip(lower=0).fillna(0))

        # MedInc: log1p transform (moderately right-skewed)
        X["log_MedInc"] = np.log1p(X["MedInc"].clip(lower=0).fillna(0))

        # ── Concept D: Haversine distance features ─────────────────────────────
        lat  = X["Latitude"].to_numpy()
        lon  = X["Longitude"].to_numpy()
        for name, (ref_lat, ref_lon) in [("LA", _LA), ("SF", _SF), ("SAC", _SAC)]:
            X[f"dist_{name}"] = self._haversine(lat, lon, ref_lat, ref_lon)

        # ── Concept D: Geospatial cluster (KMeans, fit on train only) ──────────
        coords = X[["Latitude", "Longitude"]].to_numpy()
        X["geo_cluster"] = self._kmeans.predict(coords).astype(float)

        # ── Concept C: Interaction / polynomial terms ─────────────────────────
        rm_safe   = X["AveRooms"].clip(lower=0.1).fillna(0.1)
        inc       = X["MedInc"].clip(lower=0).fillna(0)
        age       = X["HouseAge"].clip(lower=0).fillna(0)
        pop       = X["Population"].clip(lower=1).fillna(1)
        occ       = X["AveOccup"].clip(lower=0.1).fillna(0.1)
        bedr      = X["AveBedrms"].clip(lower=0.1).fillna(0.1)

        X["medinc_rooms_interact"] = inc * rm_safe          # income × space
        X["medinc_age_interact"]   = inc * age              # income × age (depreciation)
        X["rooms_per_person"]      = rm_safe / occ          # space per occupant
        X["bedrooms_ratio"]        = bedr / rm_safe         # bedroom concentration
        X["income_per_room"]       = inc / rm_safe          # affordability per room

        return X

    @staticmethod
    def _haversine(
        lat1: np.ndarray, lon1: np.ndarray,
        lat2: float, lon2: float,
    ) -> np.ndarray:
        """Vectorised haversine distance (km) between arrays of points and one ref point."""
        R    = 6371.0
        φ1   = np.radians(lat1);  φ2 = np.radians(lat2)
        dφ   = np.radians(lat2 - lat1)
        dλ   = np.radians(lon2 - lon1)
        a    = np.sin(dφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))


# ── Concept A: Feature scaling comparison ─────────────────────────────────────
def compare_scalers(
    X_train_eng: pd.DataFrame,
    y_train:     pd.Series,
    output_dir:  Path,
) -> str:
    """
    Concept A: Compare StandardScaler, RobustScaler, MinMaxScaler on Ridge
    regression via 5-fold CV RMSE. Returns the name of the best scaler.

    Why this matters: All numeric features but with very different ranges
    (Population 3-35682 vs AveOccup 0.7-1243). Unscaled coefficients are
    meaningless for regularised models. Choice of scaler affects the
    regularisation penalty distribution across features.
    """
    log.info("Concept A: Comparing scalers …")
    scalers = {
        "StandardScaler": StandardScaler(),    # mean=0, std=1
        "RobustScaler":   RobustScaler(),      # median/IQR — outlier-resistant
        "MinMaxScaler":   MinMaxScaler(),      # [0,1] — preserves sparsity
    }
    results = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cols = [c for c in get_column_groups().numeric if c in X_train_eng.columns]
    X_np = SimpleImputer(strategy="median").fit_transform(X_train_eng[cols])

    for name, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X_np)
        ridge     = Ridge(alpha=1.0)
        rmse_scores = -cross_val_score(
            ridge, X_scaled, y_train,
            cv=cv, scoring="neg_root_mean_squared_error",
        )
        r2_scores = cross_val_score(ridge, X_scaled, y_train, cv=cv, scoring="r2")
        results[name] = {
            "rmse_mean": float(rmse_scores.mean()),
            "rmse_std":  float(rmse_scores.std()),
            "r2_mean":   float(r2_scores.mean()),
        }
        log.info("  %s — RMSE=%.4f±%.4f  R²=%.4f",
                 name, rmse_scores.mean(), rmse_scores.std(), r2_scores.mean())

    # Plot scaler comparison
    names  = list(results.keys())
    rmses  = [results[n]["rmse_mean"] for n in names]
    errs   = [results[n]["rmse_std"]  for n in names]
    colors = ["#4C78A8", "#54A24B", "#F58518"]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, rmses, yerr=errs, color=colors, capsize=5, width=0.5)
    plt.ylabel("CV RMSE (5-fold)")
    plt.title("Concept A: Scaler comparison (Ridge α=1)\nLower is better")
    plt.ylim(0, max(rmses) * 1.3)
    for bar, val in zip(bars, rmses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_scaler_comparison.png", dpi=160)
    plt.close()

    best = min(results, key=lambda n: results[n]["rmse_mean"])
    write_json(output_dir / "scaler_comparison.json", results)
    log.info("Best scaler: %s", best)
    return best


# ── Concept B: Regularisation path and coefficient shrinkage ──────────────────
def analyse_regularisation(
    X_train_eng: pd.DataFrame,
    y_train:     pd.Series,
    output_dir:  Path,
) -> dict[str, Any]:
    """
    Concept B: Deep-dive into L1, L2, and ElasticNet regularisation.

    Produces:
      1. Regularisation path plot: Ridge and Lasso coefficient traces vs α
      2. Optimal alpha via cross-validation (RidgeCV, LassoCV, ElasticNetCV)
      3. Coefficient comparison table: which features Lasso zeros out
      4. Bias-variance tradeoff curve for Ridge (train vs CV RMSE vs α)
    """
    log.info("Concept B: Analysing regularisation paths …")
    cols = [c for c in get_column_groups().numeric if c in X_train_eng.columns]
    imp  = SimpleImputer(strategy="median")
    sc   = StandardScaler()
    X_s  = sc.fit_transform(imp.fit_transform(X_train_eng[cols]))
    y_np = y_train.to_numpy()
    feat = cols

    # ── B1: Regularisation path ───────────────────────────────────────────────
    alphas = np.logspace(-3, 3, 60)
    ridge_coefs = []
    lasso_coefs = []
    for a in alphas:
        ridge_coefs.append(Ridge(alpha=a).fit(X_s, y_np).coef_)
        try:
            lasso_coefs.append(Lasso(alpha=a, max_iter=5000).fit(X_s, y_np).coef_)
        except Exception:
            lasso_coefs.append(np.zeros(len(feat)))
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, f in enumerate(feat[:12]):   # top-12 for readability
        ax1.plot(np.log10(alphas), ridge_coefs[:, i], linewidth=1.2, label=f)
        ax2.plot(np.log10(alphas), lasso_coefs[:, i], linewidth=1.2, label=f)
    ax1.set_xlabel("log₁₀(α)"); ax1.set_ylabel("Coefficient value")
    ax1.set_title("Ridge (L2) regularisation path\nCoefficients shrink toward 0 but never reach it")
    ax1.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("log₁₀(α)"); ax2.set_ylabel("Coefficient value")
    ax2.set_title("Lasso (L1) regularisation path\nCoefficients become exactly 0 (sparse solution)")
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_regularisation_path.png", dpi=160)
    plt.close()

    # ── B2: Optimal alpha via CV ──────────────────────────────────────────────
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ridge_cv = RidgeCV(alphas=alphas, cv=cv).fit(X_s, y_np)
    lasso_cv = LassoCV(alphas=alphas, cv=cv, max_iter=5000,
                       random_state=RANDOM_STATE).fit(X_s, y_np)
    enet_cv  = ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                            cv=cv, max_iter=5000,
                            random_state=RANDOM_STATE).fit(X_s, y_np)

    # ── B3: Coefficient comparison (zeroed by Lasso) ──────────────────────────
    ridge_coef_opt = Ridge(alpha=ridge_cv.alpha_).fit(X_s, y_np).coef_
    lasso_coef_opt = Lasso(alpha=lasso_cv.alpha_, max_iter=5000).fit(X_s, y_np).coef_
    enet_coef_opt  = ElasticNet(alpha=enet_cv.alpha_, l1_ratio=enet_cv.l1_ratio_,
                                max_iter=5000).fit(X_s, y_np).coef_

    coef_df = pd.DataFrame({
        "feature":    feat,
        "ridge_coef": ridge_coef_opt,
        "lasso_coef": lasso_coef_opt,
        "enet_coef":  enet_coef_opt,
        "lasso_zeroed": lasso_coef_opt == 0,
    }).sort_values("ridge_coef", key=abs, ascending=False)
    coef_df.to_csv(output_dir / "regularisation_coefficients.csv", index=False)

    # ── B4: Coefficient comparison bar ────────────────────────────────────────
    top = coef_df.head(15).copy()
    x = np.arange(len(top))
    w = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, top["ridge_coef"], w, label="Ridge (L2)", color="#4C78A8", alpha=0.85)
    ax.bar(x,     top["lasso_coef"], w, label="Lasso (L1)", color="#E45756", alpha=0.85)
    ax.bar(x + w, top["enet_coef"],  w, label="ElasticNet", color="#54A24B", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(top["feature"], rotation=38, ha="right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Standardised coefficient")
    ax.set_title("Concept B: Ridge vs Lasso vs ElasticNet — coefficient comparison\n"
                 "Lasso drives weak features to exactly 0 (automatic feature selection)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_regularisation_coefficients.png", dpi=160)
    plt.close()

    # ── B5: Ridge bias-variance curve ─────────────────────────────────────────
    train_rmse = []; cv_rmse = []
    for a in alphas:
        mdl   = Ridge(alpha=a)
        trn   = -cross_val_score(mdl, X_s, y_np, cv=cv, scoring="neg_root_mean_squared_error")
        mdl.fit(X_s, y_np)
        pred  = mdl.predict(X_s)
        train_rmse.append(np.sqrt(mean_squared_error(y_np, pred)))
        cv_rmse.append(trn.mean())

    plt.figure(figsize=(8, 4))
    plt.plot(np.log10(alphas), train_rmse, label="Train RMSE", color="#4C78A8")
    plt.plot(np.log10(alphas), cv_rmse,    label="CV RMSE",    color="#E45756")
    plt.axvline(np.log10(ridge_cv.alpha_), color="green", linestyle="--",
                label=f"Optimal α={ridge_cv.alpha_:.4f}")
    plt.xlabel("log₁₀(α)")
    plt.ylabel("RMSE")
    plt.title("Concept B: Ridge bias-variance tradeoff\nHigh α → underfitting (high bias), Low α → overfitting (high variance)")
    plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "plot_ridge_bias_variance.png", dpi=160)
    plt.close()

    n_zeroed = int((lasso_coef_opt == 0).sum())
    log.info("Regularisation: Ridge α*=%.4f  Lasso α*=%.4f  ElasticNet α*=%.4f l1=%.2f",
             ridge_cv.alpha_, lasso_cv.alpha_, enet_cv.alpha_, enet_cv.l1_ratio_)
    log.info("Lasso zeroed %d / %d features", n_zeroed, len(feat))

    return {
        "ridge_optimal_alpha":   float(ridge_cv.alpha_),
        "lasso_optimal_alpha":   float(lasso_cv.alpha_),
        "elasticnet_optimal_alpha":  float(enet_cv.alpha_),
        "elasticnet_optimal_l1_ratio": float(enet_cv.l1_ratio_),
        "n_features_zeroed_by_lasso": n_zeroed,
        "features_zeroed": list(coef_df[coef_df["lasso_zeroed"]]["feature"]),
    }


# ── Concept G: Residual diagnostics ───────────────────────────────────────────
def residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Concept G: Statistical residual diagnostics.
      - Breusch-Pagan test: heteroscedasticity (do errors grow with fitted values?)
      - Shapiro-Wilk test: normality of residuals (needed for linear model inference)
      - Residual autocorrelation plot
    """
    residuals = y_true - y_pred

    # Breusch-Pagan heteroscedasticity test (manual implementation)
    # regress squared residuals on fitted values
    resid_sq = residuals ** 2
    bp_model = LinearRegression().fit(y_pred.reshape(-1, 1), resid_sq)
    bp_r2    = bp_model.score(y_pred.reshape(-1, 1), resid_sq)
    n        = len(residuals)
    bp_stat  = float(n * bp_r2)
    bp_pval  = float(1 - scipy_stats.chi2.cdf(bp_stat, df=1))

    # Shapiro-Wilk normality test (on a subsample if n>5000)
    sample = residuals if len(residuals) <= 5000 else np.random.default_rng(RANDOM_STATE).choice(residuals, 5000, replace=False)
    sw_stat, sw_pval = scipy_stats.shapiro(sample)

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Q-Q plot (normality)
    scipy_stats.probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q plot of residuals\n(points on line = normally distributed)")

    # Scale-location (heteroscedasticity check)
    axes[1].scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.3, s=8, color="#4C78A8")
    axes[1].axhline(np.sqrt(np.abs(residuals)).mean(), color="red", linewidth=1.2)
    axes[1].set_xlabel("Fitted values")
    axes[1].set_ylabel("√|residual|")
    axes[1].set_title("Scale-Location plot\n(flat red line = homoscedastic)")

    # Residual histogram
    axes[2].hist(residuals, bins=60, color="#54A24B", edgecolor="white", linewidth=0.3)
    axes[2].axvline(0, color="red", linewidth=1.2)
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    axes[2].plot(x, scipy_stats.norm.pdf(x, mu, sigma) * len(residuals) * (residuals.max()-residuals.min())/60,
                 color="black", linewidth=1.2, linestyle="--", label="Normal fit")
    axes[2].set_title("Residual distribution")
    axes[2].legend(fontsize=9)

    plt.suptitle("Concept G: Residual diagnostics", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_residual_diagnostics.png", dpi=160, bbox_inches="tight")
    plt.close()

    log.info("Residual diagnostics: BP p=%.4f (hetero: %s)  SW p=%.4f (normal: %s)",
             bp_pval, bp_pval < 0.05, sw_pval, sw_pval > 0.05)
    return {
        "breusch_pagan": {"statistic": bp_stat, "p_value": bp_pval,
                          "heteroscedastic": bp_pval < 0.05},
        "shapiro_wilk":  {"statistic": float(sw_stat), "p_value": float(sw_pval),
                          "normal_residuals": sw_pval > 0.05},
        "residual_mean": float(residuals.mean()),
        "residual_std":  float(residuals.std()),
    }


# ── Concept H: Learning curves ────────────────────────────────────────────────
def plot_learning_curves(
    model:      Pipeline,
    X_train:    pd.DataFrame,
    y_train:    pd.Series,
    output_dir: Path,
) -> None:
    """
    Concept H: Learning curve — train vs CV score vs number of training samples.
    Diagnoses whether the model suffers from high bias (needs better features)
    or high variance (needs more data / regularisation).
    """
    log.info("Concept H: Computing learning curves …")
    train_sizes = np.linspace(0.10, 1.0, 8)
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    try:
        sizes, train_scores, cv_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=N_JOBS,
        )
        train_rmse = -train_scores.mean(axis=1)
        cv_rmse    = -cv_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        cv_std     = cv_scores.std(axis=1)

        plt.figure(figsize=(8, 4.5))
        plt.plot(sizes, train_rmse, "o-", color="#4C78A8", label="Train RMSE")
        plt.plot(sizes, cv_rmse,    "o-", color="#E45756", label="CV RMSE")
        plt.fill_between(sizes, train_rmse - train_std, train_rmse + train_std,
                         alpha=0.15, color="#4C78A8")
        plt.fill_between(sizes, cv_rmse - cv_std, cv_rmse + cv_std,
                         alpha=0.15, color="#E45756")
        gap = float(cv_rmse[-1] - train_rmse[-1])
        plt.title(f"Concept H: Learning curve\n"
                  f"Train-CV gap at 100% data = {gap:.4f} "
                  f"({'high variance' if gap > 0.05 else 'low variance'})")
        plt.xlabel("Training set size")
        plt.ylabel("RMSE")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir / "plot_learning_curve.png", dpi=160)
        plt.close()
        log.info("Learning curve saved.")
    except Exception as exc:
        log.warning("Learning curve failed: %s", exc)


# ── Concept I: Partial dependence plots ───────────────────────────────────────
def plot_partial_dependence(
    model:      Pipeline,
    X_test:     pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Concept I: Partial Dependence Plots — marginal effect of MedInc and AveRooms
    on predicted MedHouseVal, averaging over all other features.
    Shows the shape of the relationship the model has learned.
    """
    log.info("Concept I: Computing partial dependence plots …")
    try:
        # PDP operates on the pipeline's full feature space
        features_to_plot = [0, 2]  # MedInc (col 0), AveRooms (col 2) in raw X
        feature_names    = list(X_test.columns)
        fig, ax = plt.subplots(figsize=(10, 4))
        PartialDependenceDisplay.from_estimator(
            model, X_test,
            features=[(0,), (2,)],    # 1D PDP for MedInc and AveRooms
            feature_names=feature_names,
            ax=ax, kind="average",
            n_jobs=N_JOBS,
        )
        plt.suptitle("Concept I: Partial Dependence Plots\n"
                     "Average marginal effect of MedInc (col 0) and AveRooms (col 2) on MedHouseVal",
                     fontsize=11, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_partial_dependence.png", dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        log.warning("PDP failed: %s", exc)


# ── Concept J: Spatial cross-validation ───────────────────────────────────────
def compare_cv_strategies(
    model_pipe:  Pipeline,
    X_train_eng: pd.DataFrame,
    y_train:     pd.Series,
    geo_labels:  np.ndarray,
    output_dir:  Path,
) -> dict[str, Any]:
    """
    Concept J: KFold vs GroupKFold (grouped by geo-cluster).
    GroupKFold prevents spatial leakage: if geographically adjacent blocks
    appear in both train and validation, the CV score is overly optimistic.
    """
    log.info("Concept J: Comparing CV strategies …")
    results: dict[str, dict] = {}
    cols = [c for c in get_column_groups().numeric if c in X_train_eng.columns]
    X_np = SimpleImputer(strategy="median").fit_transform(
        StandardScaler().fit_transform(X_train_eng[cols])
    )
    y_np = y_train.to_numpy()

    # Standard KFold
    kf   = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    kf_r = Ridge(alpha=1.0)
    kf_rmse = -cross_val_score(kf_r, X_np, y_np, cv=kf,
                                scoring="neg_root_mean_squared_error")
    results["KFold"] = {"rmse_mean": float(kf_rmse.mean()), "rmse_std": float(kf_rmse.std())}

    # GroupKFold (spatial groups = geo-cluster)
    gkf      = GroupKFold(n_splits=min(5, len(np.unique(geo_labels))))
    gkf_r    = Ridge(alpha=1.0)
    gkf_rmse = -cross_val_score(gkf_r, X_np, y_np, cv=gkf, groups=geo_labels,
                                 scoring="neg_root_mean_squared_error")
    results["GroupKFold"] = {"rmse_mean": float(gkf_rmse.mean()), "rmse_std": float(gkf_rmse.std())}

    spatial_leak = results["KFold"]["rmse_mean"] - results["GroupKFold"]["rmse_mean"]
    results["spatial_leakage_estimate_rmse"] = float(spatial_leak)
    log.info("CV strategy: KFold RMSE=%.4f  GroupKFold RMSE=%.4f  leak=%.4f",
             results["KFold"]["rmse_mean"], results["GroupKFold"]["rmse_mean"], spatial_leak)

    # Bar comparison
    labels = list(results.keys())[:2]
    vals   = [results[l]["rmse_mean"] for l in labels]
    errs   = [results[l]["rmse_std"]  for l in labels]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, vals, yerr=errs, color=["#4C78A8","#E45756"], capsize=6, width=0.4)
    plt.ylabel("CV RMSE")
    plt.title(f"Concept J: KFold vs GroupKFold CV\n"
              f"Spatial leakage ≈ {spatial_leak:+.4f} RMSE units")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_cv_strategy_comparison.png", dpi=160)
    plt.close()

    write_json(output_dir / "cv_strategy_comparison.json", results)
    return results


# ── Preprocessor (mirrors build_preprocessor from reference) ──────────────────
def build_preprocessor(scaler_name: str = "StandardScaler") -> ColumnTransformer:
    """
    Pure numeric preprocessor — California Housing has no categorical columns.
    Scaler is chosen from the Concept A comparison or passed explicitly.
    """
    scaler_map = {
        "StandardScaler": StandardScaler(),
        "RobustScaler":   RobustScaler(),
        "MinMaxScaler":   MinMaxScaler(),
    }
    scaler = scaler_map.get(scaler_name, StandardScaler())
    groups = get_column_groups()
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler",  scaler),
    ])
    # No categorical pipeline needed — all features are numeric
    return ColumnTransformer([
        ("num", numeric_pipeline, groups.numeric),
    ])


def build_pipeline(
    model:       BaseEstimator | None = None,
    scaler_name: str = "StandardScaler",
) -> Pipeline:
    """
    Mirrors build_pipeline from reference exactly:
    FeatureEngineer → Preprocessor → SelectFromModel → Model.
    Includes PolynomialFeatures step for interaction terms (Concept C).
    """
    selector_model = ExtraTreesRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    if model is None:
        model = Ridge(alpha=1.0)

    return Pipeline([
        ("feature_engineering", CaliforniaFeatureEngineer()),
        ("preprocess",          build_preprocessor(scaler_name)),
        ("feature_selection",   SelectFromModel(selector_model, threshold="median")),
        ("model",               model),
    ])


# ── EDA artifacts (train-set only — mirrors save_research_artifacts) ───────────
def save_research_artifacts(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
) -> dict[str, Any]:
    log.info("Saving EDA artifacts (train-set only) …")
    output_dir.mkdir(parents=True, exist_ok=True)
    eda = X_train.copy()
    eda[TARGET] = y_train.values

    # ── Tabular reports ───────────────────────────────────────────────────────
    missingness_report(eda).to_csv(output_dir / "research_missingness_report.csv")
    eda.dtypes.astype(str).rename("dtype").to_csv(output_dir / "schema.csv")
    eda.describe().T.to_csv(output_dir / "numeric_summary.csv")

    # ── Skewness report (drives log-transform decisions, Concept E) ───────────
    # Derive numeric columns dynamically — do not hardcode names
    _numeric_only = eda.select_dtypes(include=[np.number]).columns.tolist()
    skew_rows = []
    for _c in _numeric_only:
        _vals = eda[_c].to_numpy(dtype=np.float64, na_value=np.nan)
        _s = pd.Series(_vals)
        skew_rows.append({"feature": _c, "skew": float(_s.skew()), "kurtosis": float(_s.kurtosis())})
    skew_df = pd.DataFrame(skew_rows).sort_values("skew", key=abs, ascending=False)
    skew_df.to_csv(output_dir / "skewness_report.csv", index=False)

    # ── Correlation with target ────────────────────────────────────────────────
    # Build clean float64 DataFrame for correlation using actual column names
    _corr_cols = eda.select_dtypes(include=[np.number]).columns.tolist()
    _eda_num = pd.DataFrame(
        {c: eda[c].to_numpy(dtype=np.float64, na_value=np.nan) for c in _corr_cols},
    )
    corr = _eda_num.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    corr.to_csv(output_dir / "correlation_with_target.csv")
    _eda_num.corr().to_csv(output_dir / "numeric_correlation_matrix.csv")

    # ── VIF report ────────────────────────────────────────────────────────────
    # Force every column through pd.to_numeric at this point — eda is derived
    # from X_train which may still carry object columns if OpenML used a dtype
    # not caught by fix_data_types (e.g. pandas ArrowDtype backed string).
    num_feat = []
    _vif_cols = {}
    for _c in eda.columns:
        if _c == TARGET:
            continue
        _arr = pd.to_numeric(eda[_c], errors="coerce").to_numpy(dtype=np.float64)
        # Skip columns that are entirely NaN (non-numeric OpenML columns)
        if np.isnan(_arr).all():
            log.warning("VIF: skipping all-NaN column '%s' (non-numeric)", _c)
            continue
        num_feat.append(_c)
        _vif_cols[_c] = _arr

    if len(num_feat) < 2:
        log.warning("VIF: fewer than 2 usable numeric features — skipping VIF report.")
    else:
        _vif_raw = np.column_stack([_vif_cols[c] for c in num_feat])
        # Manual median imputation — pure numpy, no sklearn
        _col_medians = np.nanmedian(_vif_raw, axis=0)
        _imp = _vif_raw.copy()
        for _j in range(_imp.shape[1]):
            _mask = np.isnan(_imp[:, _j])
            if _mask.any():
                _fill = _col_medians[_j]
                _imp[_mask, _j] = 0.0 if np.isnan(_fill) else _fill
        # Final NaN guard — drop any column still all-NaN after imputation
        _good = [j for j in range(_imp.shape[1]) if not np.isnan(_imp[:, j]).any()]
        _imp      = _imp[:, _good]
        num_feat  = [num_feat[j] for j in _good]
        imp_df    = pd.DataFrame(_imp, columns=num_feat)
        vif_rows  = []
        for col in imp_df.columns:
            other = imp_df.drop(columns=[col]).to_numpy()
            tgt   = imp_df[col].to_numpy()
            if np.unique(tgt).size <= 1:
                continue
            try:
                r2  = LinearRegression().fit(other, tgt).score(other, tgt)
                vif = 9999.0 if r2 >= 0.999 else float(1 / (1 - r2))
                vif_rows.append({"feature": col, "vif": vif})
            except Exception as _ve:
                log.warning("VIF skipped for '%s': %s", col, _ve)
        pd.DataFrame(vif_rows).sort_values("vif", ascending=False).to_csv(
            output_dir / "vif_report.csv", index=False)

    # ── Grouped statistics ─────────────────────────────────────────────────────
    eda["income_tier"] = pd.cut(
        eda["MedInc"], bins=[0, 2, 4, 6, 20],
        labels=["low", "mid", "high", "very_high"]
    )
    eda["house_age_group"] = pd.cut(
        eda["HouseAge"], bins=[0, 15, 30, 45, 60],
        labels=["new", "mid", "old", "very_old"]
    )
    grouped = {
        "medv_by_income_tier":    eda.groupby("income_tier", observed=False)[TARGET].agg(["mean","median","std"]).to_dict(),
        "medv_by_age_group":      eda.groupby("house_age_group", observed=False)[TARGET].agg(["mean","median"]).to_dict(),
        "target_stats":           eda[TARGET].describe().to_dict(),
        "capped_rows":            int((eda[TARGET] >= PRICE_CAP).sum()),
        "capped_pct":             float((eda[TARGET] >= PRICE_CAP).mean()),
    }

    # ── Concept F: Target distribution analysis ────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.hist(eda[TARGET], bins=50, color="#4C78A8", edgecolor="white", linewidth=0.3)
    plt.axvline(PRICE_CAP, color="red", linestyle="--", label=f"Cap at {PRICE_CAP}")
    plt.title(f"MedHouseVal distribution\n({grouped['capped_pct']*100:.1f}% capped at {PRICE_CAP})")
    plt.xlabel("median_house_value (MedHouseVal)"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(eda[TARGET]), bins=50, color="#54A24B", edgecolor="white", linewidth=0.3)
    plt.title("log1p(median_house_value)\nMore symmetric")
    plt.xlabel("log1p(median_house_value)")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_target_distribution.png", dpi=160)
    plt.close()

    save_research_plots(eda, corr, output_dir)

    decisions = {
        "problem_definition": {
            "problem_type": "regression",
            "target":       TARGET,
            "target_unit":  "Hundreds of thousands USD (median house value per block group)",
            "n_rows":       int(len(eda)),
            "note":         f"{grouped['capped_pct']*100:.1f}% of rows capped at PRICE_CAP={PRICE_CAP}",
        },
        "metric_policy": {
            "primary":   "r2",
            "secondary": ["rmse", "mae", "mape", "medae"],
        },
        "feature_policy": {
            "log_transforms":  ["AveRooms","AveBedrms","Population","AveOccup","MedInc"],
            "geospatial":      ["dist_LA","dist_SF","dist_SAC","geo_cluster"],
            "interactions":    ["medinc_rooms_interact","medinc_age_interact",
                                "rooms_per_person","bedrooms_ratio","income_per_room"],
            "scaler_chosen":   "see scaler_comparison.json",
        },
        "grouped_stats": grouped,
    }
    write_json(output_dir / "research_decisions.json", decisions)
    return decisions


def save_research_plots(
    eda: pd.DataFrame,
    corr: pd.Series,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    # Correlation with target (green=positive, red=negative)
    plt.figure(figsize=(9, 5))
    colors = ["#54A24B" if v > 0 else "#E45756" for v in corr]
    corr.plot(kind="barh", color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Feature Pearson correlation with MedHouseVal (train only)")
    plt.xlabel("r"); plt.tight_layout()
    plt.savefig(output_dir / "plot_correlation_with_target.png", dpi=160)
    plt.close()

    # Scatter: MedInc vs MedHouseVal (key relationship)
    plt.figure(figsize=(6, 4))
    plt.scatter(eda["MedInc"], eda[TARGET], alpha=0.2, s=5, color="#4C78A8")
    plt.xlabel("MedInc"); plt.ylabel("median_house_value")
    plt.title("MedInc vs MedHouseVal (train)"); plt.tight_layout()
    plt.savefig(output_dir / "plot_medinc_vs_medv.png", dpi=160)
    plt.close()

    # Geographic scatter (Lat/Lon coloured by price)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(eda["Longitude"], eda["Latitude"], c=eda[TARGET],
                     cmap="RdYlGn", s=2, alpha=0.4, vmin=0, vmax=5)
    plt.colorbar(sc, label="median_house_value")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("California block groups — median house value (train)\nGreen=high, Red=low")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_geographic_prices.png", dpi=160)
    plt.close()

    # Skewness bar
    raw_cols = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup"]
    skews    = [float(eda[c].skew()) for c in raw_cols if c in eda.columns]
    colors   = ["#E45756" if abs(s) > 1 else "#4C78A8" for s in skews]
    plt.figure(figsize=(8, 4))
    plt.bar(raw_cols[:len(skews)], skews, color=colors)
    plt.axhline(1, color="black", linestyle="--", alpha=0.5, label="|skew|>1 → transform")
    plt.axhline(-1, color="black", linestyle="--", alpha=0.5)
    plt.title("Concept E: Feature skewness (red = log-transform applied)")
    plt.ylabel("Skewness"); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "plot_feature_skewness.png", dpi=160)
    plt.close()

    # AveRooms distribution before/after log transform
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
    a1.hist(eda["AveRooms"].clip(0, 20), bins=50, color="#B279A2")
    a1.set_title("AveRooms (raw — right-skewed)"); a1.set_xlabel("AveRooms")
    a2.hist(np.log1p(eda["AveRooms"].clip(0)), bins=50, color="#72B7B2")
    a2.set_title("log1p(AveRooms) — normalised"); a2.set_xlabel("log1p(AveRooms)")
    plt.suptitle("Concept E: Log transform effect", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_log_transform_demo.png", dpi=160)
    plt.close()


# ── Regression metrics (mirrors evaluate_predictions from reference) ───────────
def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    if hasattr(y_true, "to_numpy"):
        y_true = y_true.to_numpy()
    residuals = y_true - y_pred
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae":              float(mean_absolute_error(y_true, y_pred)),
        "rmse":             rmse,
        "r2":               float(r2_score(y_true, y_pred)),
        "mape":             float(mean_absolute_percentage_error(y_true, y_pred)),
        "medae":            float(median_absolute_error(y_true, y_pred)),
        "residual_mean":    float(residuals.mean()),
        "residual_std":     float(residuals.std()),
        "residual_max_abs": float(np.abs(residuals).max()),
    }


def evaluate_baselines(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series,    y_test:  pd.Series,
) -> dict[str, Any]:
    baselines = {}
    for strategy in ["mean", "median"]:
        d = DummyRegressor(strategy=strategy)
        d.fit(X_train, y_train)
        baselines[strategy] = evaluate_predictions(y_test, d.predict(X_test))
    return baselines


# ── Subgroup / disparity evaluation (mirrors evaluate_subgroups) ───────────────
def evaluate_subgroups(
    model:      Any,
    X_test:     pd.DataFrame,
    y_test:     pd.Series,
    y_pred:     np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    log.info("Running subgroup disparity evaluation …")
    overall_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    overall_r2   = float(r2_score(y_test, y_pred))
    results      = {"overall_rmse": overall_rmse, "overall_r2": overall_r2, "subgroups": {}}
    rows         = []

    eval_df            = X_test.reset_index(drop=True).copy()
    eval_df["_y_true"] = y_test.to_numpy()
    eval_df["_y_pred"] = y_pred

    # Derive subgroup columns if not already present
    eval_df["income_tier"] = pd.cut(
        eval_df["MedInc"], bins=[0, 2, 4, 6, 20],
        labels=["low", "mid", "high", "very_high"]
    ).astype("category")
    eval_df["house_age_group"] = pd.cut(
        eval_df["HouseAge"], bins=[0, 15, 30, 45, 60],
        labels=["new", "mid", "old", "very_old"]
    ).astype("category")
    # geo_cluster from feature engineer
    eng = CaliforniaFeatureEngineer().fit(X_test, None)
    eval_df["geo_cluster"] = eng.transform(X_test)["geo_cluster"].astype(int).astype(str)

    for col in ["income_tier", "house_age_group", "geo_cluster"]:
        col_results = {}
        for group_val, sub in eval_df.groupby(col, observed=True):
            if len(sub) < 15: continue
            sub_rmse = float(np.sqrt(mean_squared_error(sub["_y_true"], sub["_y_pred"])))
            sub_r2   = float(r2_score(sub["_y_true"], sub["_y_pred"]))
            sub_mae  = float(mean_absolute_error(sub["_y_true"], sub["_y_pred"]))
            rmse_gap = sub_rmse - overall_rmse
            alert    = bool(sub_rmse > overall_rmse * 1.25)
            col_results[str(group_val)] = {
                "n": int(len(sub)), "rmse": round(sub_rmse, 4), "r2": round(sub_r2, 4),
                "mae": round(sub_mae, 4), "rmse_gap": round(rmse_gap, 4), "alert": alert,
            }
            rows.append({"group_column": col, "group_value": str(group_val),
                         **col_results[str(group_val)]})
        results["subgroups"][col] = col_results

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "fairness_report.csv", index=False)
        _save_fairness_plot(pd.DataFrame(rows), output_dir)

    write_json(output_dir / "fairness_report.json", results)
    return results


def _save_fairness_plot(rows_df: pd.DataFrame, output_dir: Path) -> None:
    g = rows_df.copy()
    g["label"] = g["group_column"] + "=" + g["group_value"].astype(str)
    plt.figure(figsize=(11, max(5, len(g) * 0.38)))
    colors = ["#E45756" if a else "#4C78A8" for a in g["alert"]]
    plt.barh(g["label"], g["rmse"], color=colors)
    plt.axvline(g["rmse"].mean(), linestyle="--", color="black", label="Mean RMSE")
    plt.xlabel("RMSE"); plt.title("Subgroup RMSE (red = >25% above overall)")
    plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "plot_fairness_rmse.png", dpi=160)
    plt.close()


# ── Hyperparameter search (mirrors tune_model from reference) ──────────────────
def tune_model(
    X_train:     pd.DataFrame,
    y_train:     pd.Series,
    n_iter:      int,
    scaler_name: str = "StandardScaler",
) -> RandomizedSearchCV:
    log.info("Starting hyperparameter search (n_iter=%d, n_jobs=%d) …", n_iter, N_JOBS)
    param_distributions = [
        # ── Ridge (L2) ────────────────────────────────────────────────────────
        {
            "feature_selection__threshold": ["median", "0.75*median", "1.25*median"],
            "model": [Ridge()],
            "model__alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
        },
        # ── Lasso (L1) ────────────────────────────────────────────────────────
        {
            "feature_selection__threshold": ["median", "0.75*median"],
            "model": [Lasso(max_iter=5000)],
            "model__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        },
        # ── ElasticNet (L1 + L2) ──────────────────────────────────────────────
        {
            "feature_selection__threshold": ["median", "0.75*median"],
            "model": [ElasticNet(max_iter=5000)],
            "model__alpha":    [0.01, 0.05, 0.1, 0.5, 1.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        # ── GradientBoosting (tree ensemble) ──────────────────────────────────
        {
            "feature_selection__threshold": ["median", "0.75*median", "1.25*median"],
            "model": [GradientBoostingRegressor(random_state=RANDOM_STATE)],
            "model__n_estimators":  [200, 400],
            "model__max_depth":     [3, 4, 5],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__subsample":     [0.7, 0.9],
        },
        # ── RandomForest ──────────────────────────────────────────────────────
        {
            "feature_selection__threshold": ["median", "0.75*median", "1.25*median"],
            "model": [RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS)],
            "model__n_estimators":     [200, 400],
            "model__max_depth":        [8, 12, None],
            "model__min_samples_leaf": [1, 2, 4],
        },
    ]
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        build_pipeline(scaler_name=scaler_name),
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring={
            "r2":       "r2",
            "neg_rmse": "neg_root_mean_squared_error",
            "neg_mae":  "neg_mean_absolute_error",
        },
        refit="r2",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    log.info("Best CV R²: %.4f", search.best_score_)
    return search


# ── OOF uncertainty band (mirrors tune_threshold from reference) ───────────────
def compute_oof_uncertainty(
    best_estimator:    Pipeline,
    X_train:           pd.DataFrame,
    y_train:           pd.Series,
    overpredict_cost:  float = 1.0,
    underpredict_cost: float = 1.0,
) -> dict[str, float]:
    log.info("Computing OOF uncertainty band …")
    cv        = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = cross_val_predict(
        clone(best_estimator), X_train, y_train, cv=cv, n_jobs=N_JOBS,
    )
    residuals = y_train.to_numpy() - oof_preds
    oof_rmse  = float(np.sqrt(np.mean(residuals ** 2)))
    return {
        "oof_rmse":          oof_rmse,
        "oof_mae":           float(np.abs(residuals).mean()),
        "oof_r2":            float(r2_score(y_train, oof_preds)),
        "lower_band":        oof_rmse * underpredict_cost,
        "upper_band":        oof_rmse * overpredict_cost,
        "overpredict_cost":  overpredict_cost,
        "underpredict_cost": underpredict_cost,
    }


# ── Model versioning (identical to reference) ──────────────────────────────────
def _model_version_tag(model: Pipeline) -> str:
    return hashlib.sha1(pickle.dumps(model)).hexdigest()[:8]


# ── Environment snapshot (mirrors reference) ──────────────────────────────────
def save_environment_snapshot(output_dir: Path) -> None:
    env = {"saved_at": datetime.now(timezone.utc).isoformat(),
           "python": sys.version, "platform": sys.platform, "libraries": {}}
    for lib in ["sklearn","pandas","numpy","scipy","joblib","shap","mlflow","pandera"]:
        try:
            mod = importlib.import_module(lib)
            env["libraries"][lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env["libraries"][lib] = "not_installed"
    write_json(output_dir / ENVIRONMENT_FILE, env)


# ── Feature importance (mirrors reference) ────────────────────────────────────
def save_feature_importance(model: Pipeline, output_dir: Path) -> None:
    preprocess    = model.named_steps["preprocess"]
    selector      = model.named_steps["feature_selection"]
    final_model   = model.named_steps["model"]
    feat_names    = preprocess.get_feature_names_out()
    sel_names     = feat_names[selector.get_support()]

    if hasattr(final_model, "feature_importances_"):
        importance = final_model.feature_importances_
        signed     = np.full(len(importance), np.nan)
    elif hasattr(final_model, "coef_"):
        signed     = final_model.coef_
        importance = np.abs(signed)
    else:
        return

    imp_df = pd.DataFrame({
        "feature":     sel_names,
        "importance":  importance,
        "coefficient": signed,
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(output_dir / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=imp_df.head(20), y="feature", x="importance", color="#4C78A8")
    plt.title("Top 20 model features")
    plt.xlabel("Importance"); plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_feature_importance.png", dpi=160)
    plt.close()


# ── Evaluation plots (mirrors reference) ──────────────────────────────────────
def save_evaluation_plots(
    y_test:  pd.Series,
    y_pred:  np.ndarray,
    output_dir: Path,
) -> None:
    residuals = y_test.to_numpy() - y_pred

    # Actual vs Predicted
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.3, s=8, color="#4C78A8")
    mn = min(float(y_test.min()), float(y_pred.min()))
    mx = max(float(y_test.max()), float(y_pred.max()))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=1.2, label="Perfect")
    plt.title(f"Actual vs Predicted  (R²={r2_score(y_test, y_pred):.3f})")
    plt.xlabel("Actual median_house_value"); plt.ylabel("Predicted median_house_value")
    plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "plot_actual_vs_predicted.png", dpi=160)
    plt.close()

    # Residuals vs Predicted
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.25, s=8, color="#B279A2")
    plt.axhline(0, color="black", linewidth=1.0)
    plt.axhline(residuals.std(), color="#F58518", linestyle="--", alpha=0.7)
    plt.axhline(-residuals.std(), color="#F58518", linestyle="--", alpha=0.7)
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.tight_layout()
    plt.savefig(output_dir / "plot_residuals_vs_predicted.png", dpi=160)
    plt.close()

    # Residual distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, bins=60, color="#54A24B")
    plt.axvline(0, color="red", linewidth=1.0)
    plt.title("Residual distribution"); plt.tight_layout()
    plt.savefig(output_dir / "plot_residual_distribution.png", dpi=160)
    plt.close()

    # Sorted actual vs predicted with ±RMSE band
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    idx  = np.argsort(y_test.to_numpy())
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(idx)), y_test.to_numpy()[idx], label="Actual", alpha=0.8)
    plt.plot(range(len(idx)), y_pred[idx], label="Predicted", alpha=0.7)
    plt.fill_between(range(len(idx)), y_pred[idx]-rmse, y_pred[idx]+rmse,
                     alpha=0.13, color="#4C78A8", label="±RMSE")
    plt.legend(); plt.title("Actual vs Predicted (sorted) with ±RMSE band")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_prediction_band.png", dpi=160)
    plt.close()


def save_error_analysis(
    X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, output_dir: Path,
) -> None:
    df = X_test.copy()
    res = y_test.to_numpy() - y_pred
    rmse = float(np.sqrt(np.mean(res**2)))
    df["actual"]    = y_test.to_numpy()
    df["predicted"] = y_pred
    df["residual"]  = res
    df["abs_error"] = np.abs(res)
    df["pct_error"] = np.abs(res) / np.maximum(np.abs(y_test.to_numpy()), 1e-6)
    df["severity"]  = pd.cut(df["abs_error"], bins=[0, rmse*0.5, rmse, rmse*2, np.inf],
                             labels=["low","medium","high","severe"])
    df.to_csv(output_dir / "test_predictions.csv", index=False)
    df[df["abs_error"] > rmse].to_csv(output_dir / "error_analysis.csv", index=False)


# ── SHAP (mirrors save_shap_artifacts from reference) ─────────────────────────
def save_shap_artifacts(
    model:      Pipeline,
    X_test:     pd.DataFrame,
    y_test:     pd.Series,
    y_pred:     np.ndarray,
    output_dir: Path,
) -> None:
    if not _SHAP_AVAILABLE:
        log.warning("shap not installed — skipping. pip install shap"); return
    log.info("Computing SHAP values …")
    try:
        clf      = model.named_steps["model"]
        prep     = model.named_steps["preprocess"]
        fe       = model.named_steps["feature_engineering"]
        selector = model.named_steps["feature_selection"]
        fn       = prep.get_feature_names_out()
        sn       = fn[selector.get_support()]
        Xt       = selector.transform(prep.transform(fe.transform(X_test)))
        Xdf      = pd.DataFrame(Xt, columns=sn)

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
            plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
            plt.close()

        # Worst residual waterfall
        res      = np.abs(y_test.to_numpy() - y_pred)
        worst    = int(np.argmax(res))
        ev_val   = (float(exp.expected_value)
                    if not isinstance(exp.expected_value, np.ndarray)
                    else float(exp.expected_value))
        shap_exp = shap.Explanation(
            values=sv[worst], base_values=ev_val,
            data=Xdf.iloc[worst].values, feature_names=list(sn))
        plt.figure()
        shap.waterfall_plot(shap_exp, show=False, max_display=15)
        plt.title(f"SHAP Waterfall — worst residual (|err|={res[worst]:.3f})")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_shap_waterfall_worst.png", dpi=150, bbox_inches="tight")
        plt.close()

        pd.DataFrame({"feature": sn, "mean_abs_shap": np.abs(sv).mean(axis=0)}
                     ).sort_values("mean_abs_shap", ascending=False
                     ).to_csv(output_dir / "shap_importance.csv", index=False)
        log.info("SHAP artifacts saved.")
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)


# ── Training profile (quantile-based — identical to reference) ─────────────────
def build_training_profile(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    fe  = CaliforniaFeatureEngineer().fit(X_train, y_train)
    eng = fe.transform(X_train)
    num_cols = [c for c in get_column_groups().numeric if c in eng.columns]
    imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(eng[num_cols]),
        columns=num_cols,
    )
    numeric_stats: dict[str, dict] = {}
    for col in imp.columns:
        v = imp[col].to_numpy()
        numeric_stats[col] = {
            "mean": float(v.mean()), "std": float(v.std()),
            "min":  float(v.min()),  "max": float(v.max()),
            "quantiles": np.quantile(v, np.linspace(0, 1, 100)).tolist(),
        }
    return to_jsonable({
        "trained_at":          datetime.now(timezone.utc).isoformat(),
        "row_count":           int(len(X_train)),
        "raw_columns":         list(X_train.columns),
        "engineered_columns":  list(eng.columns),
        "target_stats":        {
            "mean": float(y_train.mean()), "std": float(y_train.std()),
            "min":  float(y_train.min()),  "max": float(y_train.max()),
            "pct_capped": float((y_train >= PRICE_CAP).mean()),
        },
        "raw_missing_rate":        X_train.isna().mean().to_dict(),
        "engineered_missing_rate": eng.isna().mean().to_dict(),
        "numeric_train_stats":     numeric_stats,
    })


# ── Model Card (mirrors save_model_card from reference) ───────────────────────
def save_model_card(
    metrics:          dict[str, Any],
    fairness:         dict[str, Any],
    uncertainty_info: dict[str, Any],
    reg_analysis:     dict[str, Any],
    search:           RandomizedSearchCV,
    output_dir:       Path,
) -> None:
    tm = metrics.get("test_metrics", {})
    card = {
        "schema_version": "1.0",
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "model_details": {
            "name":      "California Housing Price Predictor",
            "version":   datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "type":      "Regression (sklearn Pipeline)",
            "algorithm": repr(search.best_estimator_.named_steps["model"]),
            "framework": "scikit-learn",
        },
        "intended_use": {
            "primary_use": "Predict median house value for California census block groups.",
            "out_of_scope": [
                "Individual property valuations — this is a block-group aggregate model.",
                "Current (post-1990s) housing markets — data is from the 1990 US Census.",
                "Jurisdictions outside California.",
            ],
        },
        "training_data": {
            "source":       "fetch_openml(name='california_housing', version=1)",
            "rows":         metrics.get("split", {}).get("train_rows"),
            "test_rows":    metrics.get("split", {}).get("test_rows"),
            "target":       TARGET,
            "target_unit":  "Hundreds of thousands USD",
            "capped_pct":   f"{(metrics.get('research',{}).get('grouped_stats',{}).get('capped_pct',0))*100:.1f}%",
        },
        "evaluation_results": {
            "test_r2":           tm.get("r2"),
            "test_rmse":         tm.get("rmse"),
            "test_mae":          tm.get("mae"),
            "test_mape":         tm.get("mape"),
            "oof_rmse":          uncertainty_info.get("oof_rmse"),
            "uncertainty_band":  {"lower": uncertainty_info.get("lower_band"),
                                  "upper": uncertainty_info.get("upper_band")},
        },
        "regularisation_analysis": reg_analysis,
        "fairness": {
            "overall_rmse":   fairness.get("overall_rmse"),
            "subgroup_rmse":  {
                col: {k: v.get("rmse") for k, v in groups.items()}
                for col, groups in fairness.get("subgroups", {}).items()
            },
            "alerts": [
                {"group": col, "value": val, "rmse_gap": data.get("rmse_gap")}
                for col, groups in fairness.get("subgroups", {}).items()
                for val, data in groups.items() if data.get("alert")
            ],
        },
        "limitations": [
            "Data from 1990 US Census — does not reflect current California housing market.",
            "median_house_value is capped at $500k (MedHouseVal ≥ 5.0) — model predictions above this are unreliable.",
            "Block-group averages mask within-block-group variation.",
            "Geospatial features assume static city reference points (LA, SF, Sacramento).",
        ],
        "ethical_considerations": [
            "Do not use for individual lending, insurance, or credit decisions.",
            "Geographic features may encode redlining and historical housing discrimination.",
            "Income-tier disparities in model accuracy should be audited before deployment.",
        ],
        "hyperparameters":  search.best_params_,
        "cv_best_r2":       float(search.best_score_),
    }
    write_json(output_dir / MODEL_CARD_FILE, card)
    log.info("Model card saved.")


# ── MLflow (mirrors log_to_mlflow from reference exactly) ─────────────────────
def log_to_mlflow(
    metrics:          dict[str, Any],
    search:           RandomizedSearchCV,
    uncertainty_info: dict[str, Any],
    model:            Pipeline,
    output_dir:       Path,
) -> None:
    if not _MLFLOW_AVAILABLE:
        log.info("mlflow not installed — skipping."); return
    try:
        mlflow.set_experiment("california_housing")
        tm = metrics.get("test_metrics", {})
        with mlflow.start_run():
            flat = {f"best_{k}": str(v) for k, v in search.best_params_.items()}
            flat.update({"n_jobs": N_JOBS, "random_state": RANDOM_STATE})
            mlflow.log_params(flat)
            mlflow.log_metrics({
                "cv_best_r2": float(search.best_score_),
                "test_r2":    float(tm.get("r2", 0)),
                "test_rmse":  float(tm.get("rmse", 0)),
                "test_mae":   float(tm.get("mae", 0)),
                "oof_rmse":   float(uncertainty_info.get("oof_rmse", 0)),
            })
            for fname in [MODEL_CARD_FILE, METRICS_FILE, TRAINING_PROFILE_FILE,
                          ENVIRONMENT_FILE, "fairness_report.json",
                          "regularisation_coefficients.csv", "scaler_comparison.json",
                          "plot_actual_vs_predicted.png", "plot_shap_bar.png",
                          "plot_regularisation_path.png"]:
                fpath = output_dir / fname
                if fpath.exists(): mlflow.log_artifact(str(fpath))
            mlflow.sklearn.log_model(model, "model")
        log.info("MLflow run logged.")
    except Exception as exc:
        log.warning("MLflow logging failed (%s) — continuing.", exc)


# ── Artifact saving (mirrors save_model_artifacts from reference) ──────────────
def save_model_artifacts(
    model:            Pipeline,
    search:           RandomizedSearchCV,
    X_train:          pd.DataFrame,
    y_train:          pd.Series,
    X_test:           pd.DataFrame,
    y_test:           pd.Series,
    y_pred:           np.ndarray,
    output_dir:       Path,
) -> None:
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha1 = _model_version_tag(model)
    name = f"california_price_pipeline_{ts}_{sha1}.joblib"
    joblib.dump(model, output_dir / name)
    joblib.dump(model, output_dir / MODEL_FILE)
    log.info("Model saved: %s (also as %s)", name, MODEL_FILE)
    save_environment_snapshot(output_dir)
    pd.DataFrame(search.cv_results_).sort_values("rank_test_r2").to_csv(
        output_dir / "cv_results.csv", index=False)
    save_feature_importance(model, output_dir)
    save_evaluation_plots(y_test, y_pred, output_dir)
    save_error_analysis(X_test, y_test, y_pred, output_dir)
    write_json(output_dir / TRAINING_PROFILE_FILE,
               build_training_profile(X_train, y_train))


# ── Main training workflow (mirrors train() from reference exactly) ────────────
def train(
    output_dir:        Path,
    n_iter:            int,
    drop_capped:       bool  = False,
    overpredict_cost:  float = 1.0,
    underpredict_cost: float = 1.0,
) -> dict[str, Any]:
    log.info("=== Training started (n_jobs=%d) ===", N_JOBS)
    output_dir.mkdir(parents=True, exist_ok=True)

    df                           = fix_data_types(load_data(), drop_capped=drop_capped)
    X_train, X_test, y_train, y_test = split_data(df)

    research   = save_research_artifacts(X_train, y_train, output_dir)
    baselines  = evaluate_baselines(X_train, X_test, y_train, y_test)

    # ── Concept A: Scaler comparison ─────────────────────────────────────────
    fe_train   = CaliforniaFeatureEngineer().fit(X_train, y_train).transform(X_train)
    best_scaler = compare_scalers(fe_train, y_train, output_dir)

    # ── Concept B: Regularisation analysis ────────────────────────────────────
    reg_analysis = analyse_regularisation(fe_train, y_train, output_dir)

    # ── Concept J: CV strategy comparison ────────────────────────────────────
    geo_labels = CaliforniaFeatureEngineer().fit(X_train, y_train).transform(X_train)["geo_cluster"].astype(int).to_numpy()
    cv_comparison = compare_cv_strategies(
        build_pipeline(scaler_name=best_scaler), fe_train, y_train, geo_labels, output_dir
    )

    # ── Hyperparameter search ─────────────────────────────────────────────────
    search         = tune_model(X_train, y_train, n_iter=n_iter, scaler_name=best_scaler)
    uncertainty_info = compute_oof_uncertainty(
        search.best_estimator_, X_train, y_train,
        overpredict_cost=overpredict_cost,
        underpredict_cost=underpredict_cost,
    )

    final_model = clone(search.best_estimator_)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    test_metrics = evaluate_predictions(y_test, y_pred)
    log.info("Test  R²=%.4f  RMSE=%.4f  MAE=%.4f  MAPE=%.4f",
             test_metrics["r2"], test_metrics["rmse"],
             test_metrics["mae"], test_metrics["mape"])

    save_model_artifacts(final_model, search, X_train, y_train, X_test, y_test, y_pred, output_dir)

    # ── Concept G: Residual diagnostics ───────────────────────────────────────
    res_diag = residual_diagnostics(y_test.to_numpy(), y_pred, output_dir)

    # ── Concept H: Learning curves ────────────────────────────────────────────
    plot_learning_curves(final_model, X_train, y_train, output_dir)

    # ── Concept I: Partial dependence plots ───────────────────────────────────
    plot_partial_dependence(final_model, X_test, output_dir)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    save_shap_artifacts(final_model, X_test, y_test, y_pred, output_dir)

    # ── Fairness / subgroup evaluation ────────────────────────────────────────
    fairness = evaluate_subgroups(final_model, X_test, y_test, y_pred, output_dir)

    metrics = {
        "research":          research,
        "split":             {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                              "drop_capped": drop_capped},
        "baseline":          baselines,
        "best_scaler":       best_scaler,
        "regularisation":    reg_analysis,
        "cv_comparison":     cv_comparison,
        "best_cv": {
            "best_r2":    float(search.best_score_),
            "best_params": search.best_params_,
        },
        "uncertainty_info":  uncertainty_info,
        "residual_diag":     res_diag,
        "test_metrics":      test_metrics,
        "fairness":          fairness,
    }
    write_json(output_dir / METRICS_FILE, metrics)

    # ── Model Card ────────────────────────────────────────────────────────────
    save_model_card(metrics, fairness, uncertainty_info, reg_analysis, search, output_dir)

    # ── MLflow ────────────────────────────────────────────────────────────────
    log_to_mlflow(metrics, search, uncertainty_info, final_model, output_dir)

    log.info("=== Training complete ===")
    log.info("Best scaler      : %s", best_scaler)
    log.info("Lasso zeroed     : %d features", reg_analysis["n_features_zeroed_by_lasso"])
    log.info("Test R²          : %.4f", test_metrics["r2"])
    log.info("Test RMSE        : %.4f", test_metrics["rmse"])
    log.info("OOF RMSE (band)  : %.4f", uncertainty_info["oof_rmse"])
    return to_jsonable(metrics)


# ── Inference (mirrors predict() from reference) ──────────────────────────────
def predict(
    artifact_dir: Path,
    input_csv:    Path,
    output_csv:   Path,
) -> None:
    log.info("Loading model …")
    model = joblib.load(artifact_dir / MODEL_FILE)
    if not hasattr(model, "predict"):
        raise TypeError(f"{type(model).__name__} is not a fitted pipeline.")

    unc_band = UNCERTAINTY_BAND
    metrics_path = artifact_dir / METRICS_FILE
    if metrics_path.exists():
        saved    = json.loads(metrics_path.read_text(encoding="utf-8"))
        unc_band = saved.get("uncertainty_info", {}).get("oof_rmse", UNCERTAINTY_BAND)

    input_df = pd.read_csv(input_csv)

    if INPUT_SCHEMA is not None:
        try:
            INPUT_SCHEMA.validate(input_df, lazy=True)
            log.info("Schema validation passed.")
        except Exception as exc:
            log.warning("Schema errors: %s", exc)

    profile_path = artifact_dir / TRAINING_PROFILE_FILE
    if profile_path.exists():
        profile  = json.loads(profile_path.read_text(encoding="utf-8"))
        req      = set(profile["raw_columns"])
        inc      = set(input_df.columns)
        miss     = req - inc
        if miss:
            raise ValueError(f"Missing columns: {sorted(miss)}")

    y_pred = model.predict(input_df)
    out    = input_df.copy()
    out["predicted_medv"]  = y_pred
    out["lower_bound"]     = y_pred - unc_band
    out["upper_bound"]     = y_pred + unc_band
    out["wide_interval"]   = ((out["upper_bound"] - out["lower_bound"]) > unc_band * 2.5).astype(int)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    log.info("Predictions saved to %s", output_csv.resolve())


# ── Monitoring (mirrors monitor() from reference exactly) ─────────────────────
def monitor(
    artifact_dir:       Path,
    input_csv:          Path,
    output_json:        Path,
    missing_rate_alert: float,
    ks_pvalue_alert:    float = 0.05,
) -> dict[str, Any]:
    log.info("Running monitoring checks …")
    profile  = json.loads((artifact_dir / TRAINING_PROFILE_FILE).read_text(encoding="utf-8"))
    incoming = pd.read_csv(input_csv)
    req      = set(profile["raw_columns"])
    inc      = set(incoming.columns)

    drift_rows = []
    for column, train_rate in profile["raw_missing_rate"].items():
        if column not in incoming: continue
        cur    = float(incoming[column].isna().mean())
        change = abs(cur - float(train_rate))
        drift_rows.append({"column": column, "train_missing_rate": float(train_rate),
                           "current_missing_rate": cur, "absolute_change": change,
                           "alert": change >= missing_rate_alert})

    ks_rows = []
    for col, stats in profile.get("numeric_train_stats", {}).items():
        if col not in incoming.columns: continue
        vals = incoming[col].dropna().to_numpy()
        if len(vals) < 10: continue
        stat, p = ks_2samp(np.array(stats["quantiles"]), vals)
        ks_rows.append({"column": col, "ks_statistic": float(stat), "p_value": float(p),
                        "train_mean": float(stats["mean"]), "incoming_mean": float(vals.mean()),
                        "alert": p < ks_pvalue_alert})

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "row_count":  int(len(incoming)),
        "missing_required_columns": sorted(req - inc),
        "extra_columns":            sorted(inc - req),
        "missing_rate_alert_threshold": missing_rate_alert,
        "missing_rate_drift":       drift_rows,
        "ks_pvalue_alert_threshold": ks_pvalue_alert,
        "distribution_drift":       ks_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, report)
    return report


# ── Utilities (identical to reference pipeline) ────────────────────────────────
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):   return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):   return [to_jsonable(v) for v in value]
    if isinstance(value, BaseEstimator): return repr(value)
    if isinstance(value, np.bool_):  return bool(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating):
        f = float(value); return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, float):
        return None if (np.isnan(value) or np.isinf(value)) else value
    try:
        if pd.isna(value): return None
    except (TypeError, ValueError): pass
    return value


def create_sample_input(output_csv: Path, rows: int) -> None:
    df = fix_data_types(load_data())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[TARGET]).head(rows).to_csv(output_csv, index=False)
    log.info("Sample input saved to %s", output_csv.resolve())


# ── CLI (mirrors reference pipeline CLI exactly) ──────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="California Housing end-to-end ML pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tp = subparsers.add_parser("train")
    tp.add_argument("--output-dir",        type=Path,  default=Path("artifacts_ca"))
    tp.add_argument("--n-iter",            type=int,   default=20)
    tp.add_argument("--drop-capped",       action="store_true",
                    help="Exclude rows where MedHouseVal == 5.0 (concept F).")
    tp.add_argument("--overpredict-cost",  type=float, default=1.0)
    tp.add_argument("--underpredict-cost", type=float, default=1.0)

    pp = subparsers.add_parser("predict")
    pp.add_argument("--artifact-dir", type=Path, default=Path("artifacts_ca"))
    pp.add_argument("--input-csv",    type=Path, required=True)
    pp.add_argument("--output-csv",   type=Path, default=Path("artifacts_ca/predictions.csv"))

    mp = subparsers.add_parser("monitor")
    mp.add_argument("--artifact-dir",       type=Path,  default=Path("artifacts_ca"))
    mp.add_argument("--input-csv",          type=Path,  required=True)
    mp.add_argument("--output-json",        type=Path,  default=Path("artifacts_ca/monitoring_report.json"))
    mp.add_argument("--missing-rate-alert", type=float, default=0.05)
    mp.add_argument("--ks-pvalue-alert",    type=float, default=0.05)

    sp = subparsers.add_parser("sample-input")
    sp.add_argument("--output-csv", type=Path, default=Path("artifacts_ca/sample_houses.csv"))
    sp.add_argument("--rows",       type=int,  default=20)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        m = train(args.output_dir, args.n_iter,
                  drop_capped=args.drop_capped,
                  overpredict_cost=args.overpredict_cost,
                  underpredict_cost=args.underpredict_cost)
        log.info("Best scaler : %s",   m["best_scaler"])
        log.info("Test R²     : %.3f", m["test_metrics"]["r2"])
        log.info("Test RMSE   : %.3f", m["test_metrics"]["rmse"])
        log.info("Test MAE    : %.3f", m["test_metrics"]["mae"])
    elif args.command == "predict":
        predict(args.artifact_dir, args.input_csv, args.output_csv)
    elif args.command == "monitor":
        report = monitor(args.artifact_dir, args.input_csv, args.output_json,
                         args.missing_rate_alert, args.ks_pvalue_alert)
        log.info("Monitoring saved. Missing-rate alerts: %d  KS alerts: %d",
                 sum(r["alert"] for r in report["missing_rate_drift"]),
                 sum(r["alert"] for r in report["distribution_drift"]))
    elif args.command == "sample-input":
        create_sample_input(args.output_csv, args.rows)


if __name__ == "__main__":
    main()