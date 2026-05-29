"""
boston_pipeline.py
==================
Industry-standard end-to-end ML pipeline for Boston Housing price prediction.
Mirrors every architectural pattern from titanic-ml-pipeline.py, adapted for
regression (continuous target MEDV) instead of binary classification.

Data source (identical pattern to reference):
    boston = fetch_openml(name="boston", version=1, as_frame=True).frame

Improvements implemented (parallel to titanic pipeline):
  1.  Model versioning with timestamp + SHA1 hash
  2.  KS-test distribution drift + quantile-based storage (no raw value bloat)
  3.  predict guard at inference time (checks fitted pipeline)
  4.  JSON-safe serialisation (inf / nan / pd.NA)
  5.  OOF residual-based threshold + cost-sensitive prediction band
  6.  Input schema validation (pandera) at inference
  7.  Subgroup / fairness evaluation (CHAS, RAD, price_tier)
  8.  SHAP explainability (summary + waterfall for worst residuals)
  9.  MLflow experiment tracking (optional, graceful fallback)
 10.  n_jobs via env-var ML_N_JOBS; environment snapshot saved alongside model
 11.  Prediction confidence / uncertainty interval flags
 12.  Model Card generated at train time

Regression metrics used (industry standard for housing / pricing models):
  MAE   — Mean Absolute Error (same unit as target, interpretable)
  RMSE  — Root Mean Squared Error (penalises large errors more)
  R²    — Coefficient of determination (proportion of variance explained)
  MAPE  — Mean Absolute Percentage Error (scale-free, intuitive for business)
  MedAE — Median Absolute Error (robust to outliers)

Usage:
  python boston_pipeline.py train   --output-dir artifacts_boston
  python boston_pipeline.py predict --artifact-dir artifacts_boston --input-csv sample.csv --output-csv preds.csv
  python boston_pipeline.py monitor --artifact-dir artifacts_boston --input-csv new_data.csv
  python boston_pipeline.py sample-input --output-csv sample.csv --rows 10
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

# ── matplotlib scratch dir must be set before any pyplot import ──────────────
_MPLCONFIGDIR = Path("artifacts_boston") / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    Ridge,
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
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── optional heavy deps (graceful degradation) — identical pattern to reference
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

try:
    import pandera.pandas as pa
    _PANDERA_AVAILABLE = True
except ImportError:
    _PANDERA_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE          = 42
TARGET                = "MEDV"               # Median home value (1000s USD)
MODEL_FILE            = "boston_price_pipeline.joblib"
METRICS_FILE          = "metrics.json"
TRAINING_PROFILE_FILE = "training_profile.json"
MODEL_CARD_FILE       = "model_card.json"
ENVIRONMENT_FILE      = "environment.json"

# Parallelism via env-var; -1 = all cores (mirrors reference pipeline)
N_JOBS = int(os.environ.get("ML_N_JOBS", 1))

# Prediction uncertainty band (fraction of predicted value)
UNCERTAINTY_BAND = float(os.environ.get("ML_UNCERTAINTY_BAND", 0.10))

# Subgroup columns for fairness / disparity evaluation
FAIRNESS_COLS = ["CHAS", "RAD", "price_tier"]


# ── Column groups (mirrors ColumnGroups dataclass in reference) ───────────────
@dataclass(frozen=True)
class ColumnGroups:
    numeric: list[str]
    categorical: list[str]


def get_column_groups() -> ColumnGroups:
    """
    Feature columns after BostonFeatureEngineer.
    RAD and CHAS are both category from OpenML — they go into the categorical
    pipeline (OHE), not the numeric pipeline.
    RAD stays as-is: already category from OpenML, fed directly to OHE pipeline.
    """
    return ColumnGroups(
        numeric=[
            "CRIM", "ZN", "INDUS", "NOX", "RM", "AGE",
            "DIS", "TAX", "PTRATIO", "B", "LSTAT",
            "log_CRIM", "log_LSTAT", "rm_lstat_interact",
            "rooms_per_tax", "crime_per_room",
        ],
        categorical=["CHAS", "RAD"],   # both already category from OpenML
    )


# ── Data loading (identical pattern to reference pipeline) ────────────────────
def load_data() -> pd.DataFrame:
    """
    Load the Boston Housing dataset from OpenML.
    Exact pattern requested:
        boston = fetch_openml(name="boston", version=1, as_frame=True).frame
    """
    log.info("Loading Boston Housing dataset from OpenML …")
    boston = fetch_openml(name="boston", version=1, as_frame=True).frame
    return boston.copy()


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast columns to the correct types, reflecting the actual OpenML schema:
      - CHAS  : category (0/1 binary indicator — already category from OpenML)
      - RAD   : category (ordinal accessibility index — already category from OpenML)
      - MEDV  : float64 (regression target)
      - All other 11 columns: float64
    """
    df = df.copy()
    df[TARGET]  = df[TARGET].astype(float)
    df["CHAS"]  = df["CHAS"].astype("category")
    df["RAD"]   = df["RAD"].astype("category")   # ordinal, not numeric
    num_cols = ["CRIM","ZN","INDUS","NOX","RM","AGE","DIS","TAX","PTRATIO","B","LSTAT"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Random 80/20 split done BEFORE any EDA to prevent test-set leakage.
    Regression uses random split (no stratify — continuous target).
    Mirrors split_data in reference pipeline.
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


# ── Pandera input schema (mirrors reference pipeline) ─────────────────────────
def build_input_schema():
    """Return pandera DataFrameSchema for raw inference inputs, or None."""
    if not _PANDERA_AVAILABLE:
        log.warning("pandera not installed — input schema validation skipped.")
        return None
    schema = pa.DataFrameSchema(
        {
            "CRIM":    pa.Column(float,  pa.Check.ge(0),              nullable=True, required=False),
            "ZN":      pa.Column(float,  pa.Check.in_range(0, 100),   nullable=True, required=False),
            "INDUS":   pa.Column(float,  pa.Check.in_range(0, 100),   nullable=True, required=False),
            "CHAS":    pa.Column(pa.Category,
                                 pa.Check.isin(["0", "1", 0, 1]),
                                 nullable=False, required=False),
            "NOX":     pa.Column(float,  pa.Check.in_range(0, 1),     nullable=True, required=False),
            "RM":      pa.Column(float,  pa.Check.in_range(1, 15),    nullable=True, required=False),
            "AGE":     pa.Column(float,  pa.Check.in_range(0, 100),   nullable=True, required=False),
            "DIS":     pa.Column(float,  pa.Check.ge(0),              nullable=True, required=False),
            "RAD":     pa.Column(pa.Category,
                                 pa.Check.isin(["1","2","3","4","5","6","7","8","24"]),
                                 nullable=False, required=False),
            "TAX":     pa.Column(float,  pa.Check.ge(0),              nullable=True, required=False),
            "PTRATIO": pa.Column(float,  pa.Check.in_range(10, 25),   nullable=True, required=False),
            "B":       pa.Column(float,  pa.Check.in_range(0, 400),   nullable=True, required=False),
            "LSTAT":   pa.Column(float,  pa.Check.in_range(0, 40),    nullable=True, required=False),
        },
        coerce=True,
        strict=False,
    )
    return schema


INPUT_SCHEMA = build_input_schema()


# ── Feature engineering (mirrors TitanicFeatureEngineer pattern) ──────────────
class BostonFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-driven feature engineering for Boston Housing.
    Follows the exact BaseEstimator + TransformerMixin pattern from the reference
    pipeline's TitanicFeatureEngineer — fit-safe, pipeline-compatible.

    Engineered features:
      log_CRIM          — log1p transform: crime is right-skewed
      log_LSTAT         — log1p transform: % lower status is right-skewed
      rm_lstat_interact — RM × (1/LSTAT): rooms matter more in low-LSTAT areas
      rooms_per_tax     — RM / (TAX / 100): value-to-tax efficiency
      crime_per_room    — CRIM / RM: crime burden per room (density proxy)

    Note: CHAS and RAD arrive as category from OpenML (no bucketing needed).
    Both are preserved as category and fed to the OHE categorical pipeline.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "BostonFeatureEngineer":
        # Nothing to learn from data — all transforms are deterministic
        # (mirrors TitanicFeatureEngineer where rare_titles_ is learned)
        self._input_columns = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # ── Log transforms (reduce right skew) ───────────────────────────────
        X["log_CRIM"]  = np.log1p(X["CRIM"].clip(lower=0).fillna(0))
        X["log_LSTAT"] = np.log1p(X["LSTAT"].clip(lower=0).fillna(0))

        # ── Interaction: rooms × inverse lower-status (penalise high LSTAT) ──
        lstat_safe = X["LSTAT"].clip(lower=0.1).fillna(0.1)
        X["rm_lstat_interact"] = X["RM"].fillna(X["RM"].median()) / lstat_safe

        # ── Tax efficiency: rooms relative to tax burden ──────────────────────
        tax_safe = (X["TAX"].clip(lower=1).fillna(1)) / 100.0
        X["rooms_per_tax"] = X["RM"].fillna(X["RM"].median()) / tax_safe

        # ── Crime density per room ────────────────────────────────────────────
        rm_safe = X["RM"].clip(lower=0.1).fillna(0.1)
        X["crime_per_room"] = X["CRIM"].clip(lower=0).fillna(0) / rm_safe

        # ── RAD is already a category from OpenML — no bucketing needed ─────
        # Ensure it stays category after copy() so OHE handles it correctly.
        if "RAD" in X.columns and not hasattr(X["RAD"], "cat"):
            X["RAD"] = X["RAD"].astype("category")
        if "CHAS" in X.columns and not hasattr(X["CHAS"], "cat"):
            X["CHAS"] = X["CHAS"].astype("category")

        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(get_column_groups().numeric + get_column_groups().categorical)


# ── Preprocessor (mirrors build_preprocessor in reference) ───────────────────
def build_preprocessor() -> ColumnTransformer:
    groups = get_column_groups()
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline,     groups.numeric),
        ("cat", categorical_pipeline, groups.categorical),
    ])


# ── Pipeline construction (mirrors build_pipeline in reference) ───────────────
def build_pipeline(model: BaseEstimator | None = None) -> Pipeline:
    """
    Full pipeline: BostonFeatureEngineer → ColumnTransformer →
                   SelectFromModel (ExtraTrees) → Regressor

    Mirrors reference pipeline's build_pipeline exactly — same SelectFromModel
    pattern with ExtraTreesRegressor as the selector estimator.
    """
    selector_model = ExtraTreesRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    if model is None:
        model = Ridge(alpha=1.0)
    return Pipeline([
        ("feature_engineering", BostonFeatureEngineer()),
        ("preprocess",          build_preprocessor()),
        ("feature_selection",   SelectFromModel(selector_model, threshold="median")),
        ("model",               model),
    ])


# ── EDA artifacts (train-set only — mirrors save_research_artifacts) ──────────
def save_research_artifacts(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Saves EDA artifacts computed only on training data.
    Mirrors save_research_artifacts / save_research_plots from reference pipeline.
    """
    log.info("Saving EDA artifacts (train-set only) …")
    output_dir.mkdir(parents=True, exist_ok=True)

    eda = X_train.copy()
    eda[TARGET] = y_train.values

    # ── Tabular reports ───────────────────────────────────────────────────────
    missingness_report(eda).to_csv(output_dir / "research_missingness_report.csv")
    eda.dtypes.astype(str).rename("dtype").to_csv(output_dir / "schema.csv")
    eda.select_dtypes(include=["number"]).describe().T.to_csv(
        output_dir / "numeric_summary.csv"
    )

    # ── Grouped statistics ────────────────────────────────────────────────────
    # RAD is already category from OpenML — group directly
    grouped_reports = {
        "medv_by_chas":  eda.groupby("CHAS", observed=False)[TARGET].agg(["mean","median","std"]).to_dict(),
        "medv_by_rad":   eda.groupby("RAD",  observed=False)[TARGET].agg(["mean","median"]).to_dict(),
        "target_stats":      eda[TARGET].describe().to_dict(),
        "skew_kurtosis": {
            col: {"skew": float(eda[col].skew()), "kurtosis": float(eda[col].kurtosis())}
            for col in ["CRIM","LSTAT","DIS","MEDV"]
            if col in eda.columns
        },
    }

    # ── Correlation with target ───────────────────────────────────────────────
    num_eda = eda.select_dtypes(include=["number"])
    corr_with_target = (
        num_eda.corr()[TARGET]
        .drop(TARGET)
        .sort_values(key=abs, ascending=False)
    )
    corr_with_target.to_csv(output_dir / "correlation_with_target.csv")
    num_eda.corr().to_csv(output_dir / "numeric_correlation_matrix.csv")

    # ── VIF (mirrors reference pipeline exactly) ──────────────────────────────
    # Exclude category columns (CHAS, RAD) from VIF — VIF requires numeric input
    cat_cols = ["CHAS", "RAD"]
    num_feat = [c for c in num_eda.columns if c != TARGET and c not in cat_cols]
    imp_df   = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(num_eda[num_feat]),
        columns=num_feat,
    )
    vif_rows = []
    for col in imp_df.columns:
        other = imp_df.drop(columns=[col])
        tgt   = imp_df[col]
        if tgt.nunique() <= 1:
            continue
        r2  = LinearRegression().fit(other, tgt).score(other, tgt)
        vif = 9999.0 if r2 >= 0.999 else float(1 / (1 - r2))
        vif_rows.append({"feature": col, "vif": vif})
    pd.DataFrame(vif_rows).sort_values("vif", ascending=False).to_csv(
        output_dir / "vif_report.csv", index=False
    )

    decisions = {
        "problem_definition": {
            "problem_type":    "regression",
            "target":          TARGET,
            "target_unit":     "1000s USD (median home value)",
            "prediction_time": "Given neighbourhood attributes, predict median home value.",
        },
        "metric_policy": {
            "primary":   "r2",
            "secondary": ["rmse", "mae", "mape", "medae"],
        },
        "feature_policy": {
            "log_transforms": ["CRIM", "LSTAT"],
            "interactions":   ["rm_lstat_interact", "rooms_per_tax", "crime_per_room"],
            "categorical":    ["CHAS", "RAD"],
            "note":           "CHAS and RAD arrive as category from OpenML; no bucketing needed.",
        },
        "grouped_stats": grouped_reports,
    }
    write_json(output_dir / "research_decisions.json", decisions)

    # ── EDA plots ─────────────────────────────────────────────────────────────
    save_research_plots(eda, corr_with_target, output_dir)
    return decisions


def save_research_plots(
    eda: pd.DataFrame,
    corr_with_target: pd.Series,
    output_dir: Path,
) -> None:
    """EDA visualisations on train data only. Mirrors save_research_plots."""
    sns.set_theme(style="whitegrid")

    # Target distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(eda[TARGET], kde=True, color="#4C78A8", bins=30)
    plt.axvline(eda[TARGET].median(), color="#E45756", linestyle="--",
                label=f"Median = {eda[TARGET].median():.1f}")
    plt.title("MEDV Distribution (train)")
    plt.xlabel("Median Home Value (1000s USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_target_distribution.png", dpi=160)
    plt.close()

    # Correlation bar
    plt.figure(figsize=(8, 5))
    colors = ["#E45756" if v < 0 else "#54A24B" for v in corr_with_target]
    corr_with_target.plot(kind="barh", color=colors)
    plt.title("Feature Correlation with MEDV (train)")
    plt.xlabel("Pearson r")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_correlation_with_target.png", dpi=160)
    plt.close()

    # LSTAT vs MEDV scatter (key relationship)
    plt.figure(figsize=(6, 4))
    plt.scatter(eda["LSTAT"], eda[TARGET], alpha=0.4, s=18, color="#B279A2")
    plt.xlabel("LSTAT (% lower-status population)")
    plt.ylabel("MEDV (1000s USD)")
    plt.title("LSTAT vs Median Home Value (train)")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_lstat_vs_medv.png", dpi=160)
    plt.close()

    # RM vs MEDV scatter (strongest positive correlation)
    plt.figure(figsize=(6, 4))
    plt.scatter(eda["RM"], eda[TARGET], alpha=0.4, s=18, color="#72B7B2")
    plt.xlabel("RM (average rooms per dwelling)")
    plt.ylabel("MEDV (1000s USD)")
    plt.title("Rooms vs Median Home Value (train)")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_rm_vs_medv.png", dpi=160)
    plt.close()

    # MEDV by CHAS (river vs non-river)
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=eda, x="CHAS", y=TARGET, palette=["#F58518","#4C78A8"])
    plt.title("MEDV by Charles River (CHAS)")
    plt.xlabel("CHAS (1 = bounds river)")
    plt.ylabel("MEDV (1000s USD)")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_medv_by_chas.png", dpi=160)
    plt.close()

    # Price tier distribution
    eda_copy = eda.copy()
    eda_copy["price_tier"] = pd.cut(
        eda_copy[TARGET],
        bins=[0, 15, 25, 35, 60],
        labels=["low", "mid", "high", "luxury"],
    )
    plt.figure(figsize=(6, 3.5))
    sns.countplot(data=eda_copy, x="price_tier", palette="Blues_d")
    plt.title("Price Tier Distribution (train)")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_price_tier_distribution.png", dpi=160)
    plt.close()

    # Missingness
    miss = eda.isna().mean().sort_values(ascending=False)
    if (miss > 0).any():
        plt.figure(figsize=(8, 4))
        miss[miss > 0].sort_values().plot(kind="barh", color="#F58518")
        plt.title("Missing Rate by Column (train)")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_missingness.png", dpi=160)
        plt.close()


# ── Regression evaluation helpers ─────────────────────────────────────────────
def evaluate_predictions(
    y_true: pd.Series, y_pred: np.ndarray,
) -> dict[str, Any]:
    """
    Regression equivalent of evaluate_predictions in reference pipeline.
    Returns all industry-standard regression metrics.
    """
    residuals = y_true.to_numpy() - y_pred
    rmse      = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae":    float(mean_absolute_error(y_true, y_pred)),
        "rmse":   rmse,
        "r2":     float(r2_score(y_true, y_pred)),
        "mape":   float(mean_absolute_percentage_error(y_true, y_pred)),
        "medae":  float(median_absolute_error(y_true, y_pred)),
        "residual_mean":  float(residuals.mean()),
        "residual_std":   float(residuals.std()),
        "residual_max_abs": float(np.abs(residuals).max()),
    }


def evaluate_baselines(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series,    y_test:  pd.Series,
) -> dict[str, Any]:
    """
    Baseline comparisons — mirrors evaluate_baselines from reference.
    DummyRegressor strategies: mean, median.
    """
    baselines = {}
    for strategy in ["mean", "median"]:
        dummy = DummyRegressor(strategy=strategy)
        dummy.fit(X_train, y_train)
        preds = dummy.predict(X_test)
        baselines[strategy] = evaluate_predictions(y_test, preds)
    return baselines


# ── Subgroup / fairness / disparity evaluation ────────────────────────────────
def evaluate_subgroups(
    X_test:     pd.DataFrame,
    y_test:     pd.Series,
    y_pred:     np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Disaggregated regression performance across subgroups.
    Mirrors evaluate_subgroups from reference pipeline.
    Flags any subgroup whose RMSE is more than 20% above overall RMSE.
    """
    log.info("Running subgroup disparity evaluation …")
    overall_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    overall_r2   = float(r2_score(y_test, y_pred))

    results: dict[str, Any] = {
        "overall_rmse": overall_rmse,
        "overall_r2":   overall_r2,
        "subgroups":    {},
    }
    rows = []

    eval_df              = X_test.reset_index(drop=True).copy()
    eval_df["_y_true"]   = y_test.to_numpy()
    eval_df["_y_pred"]   = y_pred
    eval_df["_residual"] = y_test.to_numpy() - y_pred

    # Add price tier (derived from actual values — not predicted)
    eval_df["price_tier"] = pd.cut(
        eval_df["_y_true"],
        bins=[0, 15, 25, 35, 60],
        labels=["low", "mid", "high", "luxury"],
        right=True,
    ).astype("category")

    # RAD is already category from OpenML — used directly for subgroup eval
    group_cols = [c for c in FAIRNESS_COLS if c in eval_df.columns]
    for col in group_cols:
        col_results = {}
        for group_val, sub in eval_df.groupby(col, observed=True):
            if len(sub) < 8:
                continue
            sub_rmse = float(np.sqrt(mean_squared_error(sub["_y_true"], sub["_y_pred"])))
            sub_r2   = float(r2_score(sub["_y_true"], sub["_y_pred"]))
            sub_mae  = float(mean_absolute_error(sub["_y_true"], sub["_y_pred"]))
            rmse_gap = float(sub_rmse - overall_rmse)
            alert    = bool(sub_rmse > overall_rmse * 1.20)
            col_results[str(group_val)] = {
                "n":           int(len(sub)),
                "mean_actual": float(sub["_y_true"].mean()),
                "rmse":        round(sub_rmse, 4),
                "r2":          round(sub_r2, 4),
                "mae":         round(sub_mae, 4),
                "rmse_gap":    round(rmse_gap, 4),
                "alert":       alert,
            }
            rows.append({
                "group_column": col,
                "group_value":  str(group_val),
                **col_results[str(group_val)],
            })
        results["subgroups"][col] = col_results

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "fairness_report.csv", index=False)
        _save_fairness_plot(pd.DataFrame(rows), output_dir)
        alerts = [r for r in rows if r.get("alert")]
        if alerts:
            log.warning(
                "Disparity alert: %d subgroup(s) with RMSE >20%% above overall: %s",
                len(alerts),
                [(r["group_column"], r["group_value"], round(r["rmse_gap"], 2))
                 for r in alerts],
            )

    write_json(output_dir / "fairness_report.json", results)
    return results


def _save_fairness_plot(rows_df: pd.DataFrame, output_dir: Path) -> None:
    g = rows_df.copy()
    g["label"] = g["group_column"] + "=" + g["group_value"].astype(str)
    plt.figure(figsize=(10, max(4, len(g) * 0.45)))
    colors = ["#E45756" if a else "#4C78A8" for a in g["alert"]]
    plt.barh(g["label"], g["rmse"], color=colors)
    plt.axvline(g["rmse"].mean(), linestyle="--", color="black", label="Mean RMSE")
    plt.xlabel("RMSE")
    plt.title("Subgroup RMSE (red = alert: >20% above overall)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_fairness_rmse.png", dpi=160)
    plt.close()


# ── Hyperparameter search (mirrors tune_model from reference) ─────────────────
def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter:  int,
) -> RandomizedSearchCV:
    """
    RandomizedSearchCV over multiple regressor families.
    Mirrors the multi-model param_distributions pattern in reference pipeline.
    """
    log.info("Starting hyperparameter search (n_iter=%d, n_jobs=%d) …", n_iter, N_JOBS)

    param_distributions = [
        # ── Ridge Regression ─────────────────────────────────────────────────
        {
            "feature_selection__threshold": ["median", "0.75*median", "1.25*median"],
            "model": [Ridge()],
            "model__alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        },
        # ── Lasso (sparse coefficients) ───────────────────────────────────────
        {
            "feature_selection__threshold": ["median", "0.75*median"],
            "model": [Lasso(max_iter=5000)],
            "model__alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        },
        # ── Gradient Boosting Regressor ───────────────────────────────────────
        {
            "feature_selection__threshold": ["median", "0.75*median", "1.25*median"],
            "model": [GradientBoostingRegressor(random_state=RANDOM_STATE)],
            "model__n_estimators":  [100, 200, 400],
            "model__max_depth":     [3, 4, 5],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__subsample":     [0.7, 0.8, 1.0],
            "model__min_samples_leaf": [1, 2, 4],
        },
        # ── Random Forest Regressor ───────────────────────────────────────────
        {
            "feature_selection__threshold": ["median", "0.75*median", "1.25*median"],
            "model": [RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS)],
            "model__n_estimators":     [200, 400],
            "model__max_depth":        [6, 8, None],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features":     ["sqrt", 0.5, 0.75],
        },
    ]

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        build_pipeline(),
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring={
            "r2":   "r2",
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


# ── OOF uncertainty band (mirrors tune_threshold from reference) ──────────────
def compute_oof_uncertainty(
    best_estimator: Pipeline,
    X_train:        pd.DataFrame,
    y_train:        pd.Series,
    overpredict_cost: float = 1.0,
    underpredict_cost: float = 1.0,
) -> dict[str, float]:
    """
    Compute out-of-fold residual statistics for uncertainty quantification.
    Mirrors tune_threshold from reference — OOF, no leakage.

    The uncertainty band is set as (RMSE × 1.0) — the ±1 RMSE interval.
    Asymmetric cost support: overpredict_cost / underpredict_cost allow
    domain-specific bias adjustment (e.g. underpricing is costlier than overpricing).
    """
    log.info(
        "Computing OOF uncertainty (over_cost=%.2f, under_cost=%.2f) …",
        overpredict_cost, underpredict_cost,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = cross_val_predict(
        clone(best_estimator), X_train, y_train, cv=cv, n_jobs=N_JOBS,
    )
    residuals = y_train.to_numpy() - oof_preds
    oof_rmse  = float(np.sqrt(np.mean(residuals ** 2)))
    oof_mae   = float(np.abs(residuals).mean())
    oof_r2    = float(r2_score(y_train, oof_preds))

    # Asymmetric band: widen toward the costlier direction
    lower_band = oof_rmse * underpredict_cost
    upper_band = oof_rmse * overpredict_cost

    return {
        "oof_rmse":          oof_rmse,
        "oof_mae":           oof_mae,
        "oof_r2":            oof_r2,
        "lower_band":        lower_band,
        "upper_band":        upper_band,
        "overpredict_cost":  overpredict_cost,
        "underpredict_cost": underpredict_cost,
    }


# ── Model versioning (identical to reference) ─────────────────────────────────
def _model_version_tag(model: Pipeline) -> str:
    return hashlib.sha1(pickle.dumps(model)).hexdigest()[:8]  # noqa: S324


# ── Environment snapshot (mirrors reference exactly) ─────────────────────────
def save_environment_snapshot(output_dir: Path) -> None:
    env = {
        "saved_at":  datetime.now(timezone.utc).isoformat(),
        "python":    sys.version,
        "platform":  sys.platform,
        "libraries": {},
    }
    for lib in ["sklearn","pandas","numpy","scipy","joblib","shap","mlflow","pandera"]:
        try:
            mod = importlib.import_module(lib)
            env["libraries"][lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env["libraries"][lib] = "not_installed"
    write_json(output_dir / ENVIRONMENT_FILE, env)
    log.info("Environment snapshot saved to %s", output_dir / ENVIRONMENT_FILE)


# ── Feature importance plot ────────────────────────────────────────────────────
def save_feature_importance(model: Pipeline, output_dir: Path) -> None:
    preprocess   = model.named_steps["preprocess"]
    selector     = model.named_steps["feature_selection"]
    final_model  = model.named_steps["model"]
    feature_names       = preprocess.get_feature_names_out()
    selected_names      = feature_names[selector.get_support()]

    if hasattr(final_model, "feature_importances_"):
        importance = final_model.feature_importances_
        signed     = np.full(len(importance), np.nan)
    elif hasattr(final_model, "coef_"):
        signed     = final_model.coef_
        importance = np.abs(signed)
    else:
        log.warning("Model has no feature_importances_ or coef_; skipping.")
        return

    imp_df = pd.DataFrame({
        "feature":     selected_names,
        "importance":  importance,
        "coefficient": signed,
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(output_dir / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=imp_df.head(20), y="feature", x="importance", color="#4C78A8")
    plt.title("Top 20 Model Features (by importance)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_feature_importance.png", dpi=160)
    plt.close()


# ── Evaluation plots ──────────────────────────────────────────────────────────
def save_evaluation_plots(
    y_test:  pd.Series,
    y_pred:  np.ndarray,
    output_dir: Path,
) -> None:
    """Regression evaluation plots. Mirrors save_evaluation_plots from reference."""
    log.info("Saving evaluation plots …")
    residuals = y_test.to_numpy() - y_pred

    # Actual vs Predicted
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.45, s=20, color="#4C78A8")
    mn, mx = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=1.2, label="Perfect prediction")
    r2  = r2_score(y_test, y_pred)
    plt.title(f"Actual vs Predicted MEDV  (R² = {r2:.3f})")
    plt.xlabel("Actual MEDV")
    plt.ylabel("Predicted MEDV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_actual_vs_predicted.png", dpi=160)
    plt.close()

    # Residuals vs Predicted
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.45, s=20, color="#B279A2")
    plt.axhline(0, color="black", linewidth=1.0)
    plt.axhline(residuals.std(),  color="#F58518", linestyle="--", alpha=0.7, label="+1 std")
    plt.axhline(-residuals.std(), color="#F58518", linestyle="--", alpha=0.7, label="-1 std")
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted MEDV")
    plt.ylabel("Residual (Actual − Predicted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_residuals_vs_predicted.png", dpi=160)
    plt.close()

    # Residual distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, color="#54A24B", bins=30)
    plt.axvline(0, color="black", linewidth=1.0)
    plt.title("Residual Distribution")
    plt.xlabel("Residual (Actual − Predicted)")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_residual_distribution.png", dpi=160)
    plt.close()

    # Learning curve proxy — sorted actual vs predicted
    sorted_idx = np.argsort(y_test.to_numpy())
    plt.figure(figsize=(9, 4))
    plt.plot(range(len(sorted_idx)), y_test.to_numpy()[sorted_idx], label="Actual", alpha=0.8)
    plt.plot(range(len(sorted_idx)), y_pred[sorted_idx], label="Predicted", alpha=0.7)
    plt.fill_between(
        range(len(sorted_idx)),
        y_pred[sorted_idx] - abs(residuals).mean(),
        y_pred[sorted_idx] + abs(residuals).mean(),
        alpha=0.15, color="#4C78A8", label=f"±MAE band",
    )
    plt.legend(); plt.title("Actual vs Predicted (sorted by actual)")
    plt.xlabel("Sample (sorted)"); plt.ylabel("MEDV")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_prediction_band.png", dpi=160)
    plt.close()


# ── Error analysis CSV (mirrors save_error_analysis from reference) ───────────
def save_error_analysis(
    X_test:     pd.DataFrame,
    y_test:     pd.Series,
    y_pred:     np.ndarray,
    output_dir: Path,
) -> None:
    err_df = X_test.copy()
    residuals           = y_test.to_numpy() - y_pred
    err_df["actual_medv"]    = y_test.to_numpy()
    err_df["predicted_medv"] = y_pred
    err_df["residual"]       = residuals
    err_df["abs_error"]      = np.abs(residuals)
    err_df["pct_error"]      = np.abs(residuals) / np.maximum(np.abs(y_test.to_numpy()), 1e-6)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    err_df["error_severity"] = pd.cut(
        err_df["abs_error"],
        bins=[0, rmse * 0.5, rmse, rmse * 2, np.inf],
        labels=["low", "medium", "high", "severe"],
    )
    err_df.to_csv(output_dir / "test_predictions.csv", index=False)
    err_df[err_df["abs_error"] > rmse].to_csv(output_dir / "error_analysis.csv", index=False)


# ── SHAP explainability (mirrors save_shap_artifacts from reference) ──────────
def save_shap_artifacts(
    model:      Pipeline,
    X_test:     pd.DataFrame,
    y_test:     pd.Series,
    y_pred:     np.ndarray,
    output_dir: Path,
) -> None:
    """
    SHAP for regression. Waterfall for worst-predicted sample (largest |residual|).
    Mirrors save_shap_artifacts from reference pipeline.
    """
    if not _SHAP_AVAILABLE:
        log.warning("shap not installed — SHAP artifacts skipped. pip install shap")
        return

    log.info("Computing SHAP values …")
    final_model = model.named_steps["model"]
    preprocess  = model.named_steps["preprocess"]
    feat_eng    = model.named_steps["feature_engineering"]
    selector    = model.named_steps["feature_selection"]

    feature_names   = preprocess.get_feature_names_out()
    selected_names  = feature_names[selector.get_support()]
    X_transformed   = selector.transform(preprocess.transform(feat_eng.transform(X_test)))
    X_transformed_df = pd.DataFrame(X_transformed, columns=selected_names)

    try:
        if hasattr(final_model, "feature_importances_"):
            explainer   = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X_transformed_df)
        elif hasattr(final_model, "coef_"):
            explainer   = shap.LinearExplainer(final_model, X_transformed_df)
            shap_values = explainer.shap_values(X_transformed_df)
        else:
            masker      = shap.maskers.Independent(X_transformed_df, max_samples=100)
            explainer   = shap.Explainer(final_model.predict, masker)
            shap_values = explainer(X_transformed_df).values

        # ── Global summary bar ─────────────────────────────────────────────
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_transformed_df, plot_type="bar",
                          show=False, max_display=20)
        plt.title("SHAP Global Feature Importance")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ── Beeswarm ──────────────────────────────────────────────────────
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_transformed_df, show=False, max_display=20)
        plt.title("SHAP Beeswarm")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ── Waterfall for worst prediction (largest absolute residual) ─────
        residuals = y_test.to_numpy() - y_pred
        worst_idx = int(np.argmax(np.abs(residuals)))
        ev_val    = (explainer.expected_value
                     if not isinstance(explainer.expected_value, np.ndarray)
                     else float(explainer.expected_value))
        shap_exp  = shap.Explanation(
            values        = shap_values[worst_idx],
            base_values   = ev_val,
            data          = X_transformed_df.iloc[worst_idx].values,
            feature_names = list(selected_names),
        )
        plt.figure()
        shap.waterfall_plot(shap_exp, show=False, max_display=15)
        plt.title(f"SHAP Waterfall — Worst Prediction "
                  f"(residual={residuals[worst_idx]:.2f})")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_shap_waterfall_worst.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # ── CSV ─────────────────────────────────────────────────────────────
        pd.DataFrame({
            "feature":       selected_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).to_csv(
            output_dir / "shap_importance.csv", index=False)

        log.info("SHAP artifacts saved.")
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)


# ── Training profile (quantile-based — mirrors reference exactly) ─────────────
def build_training_profile(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Any]:
    """
    Compact training profile for drift monitoring.
    Quantile-based storage (100 quantiles) — no raw-value bloat.
    Mirrors build_training_profile from reference pipeline exactly.
    """
    engineer   = BostonFeatureEngineer().fit(X_train, y_train)
    engineered = engineer.transform(X_train)
    num_cols   = [c for c in get_column_groups().numeric if c in engineered.columns]
    num_eng    = engineered[num_cols]
    imputed    = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(num_eng),
        columns=num_eng.columns,
    )

    numeric_train_stats: dict[str, dict] = {}
    for col in imputed.columns:
        vals = imputed[col].to_numpy()
        numeric_train_stats[col] = {
            "mean":      float(np.mean(vals)),
            "std":       float(np.std(vals)),
            "min":       float(np.min(vals)),
            "max":       float(np.max(vals)),
            "quantiles": np.quantile(vals, np.linspace(0, 1, 100)).tolist(),
        }

    return to_jsonable({
        "trained_at":              datetime.now(timezone.utc).isoformat(),
        "row_count":               int(len(X_train)),
        "raw_columns":             list(X_train.columns),
        "engineered_columns":      list(engineered.columns),
        "target_stats":            {
            "mean":   float(y_train.mean()),
            "std":    float(y_train.std()),
            "min":    float(y_train.min()),
            "max":    float(y_train.max()),
            "median": float(y_train.median()),
        },
        "raw_missing_rate":        X_train.isna().mean().to_dict(),
        "engineered_missing_rate": engineered.isna().mean().to_dict(),
        "numeric_train_stats":     numeric_train_stats,
    })


# ── Model Card (mirrors save_model_card from reference exactly) ───────────────
def save_model_card(
    metrics:          dict[str, Any],
    fairness:         dict[str, Any],
    uncertainty_info: dict[str, Any],
    search:           RandomizedSearchCV,
    output_dir:       Path,
) -> None:
    tuned = metrics.get("test_metrics", {})
    card  = {
        "schema_version":   "1.0",
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "model_details": {
            "name":      "Boston Housing Price Predictor",
            "version":   datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "type":      "Regression (sklearn Pipeline)",
            "algorithm": repr(search.best_estimator_.named_steps["model"]),
            "framework": "scikit-learn",
        },
        "intended_use": {
            "primary_use": "Predict median home value for Boston census tracts.",
            "out_of_scope": [
                "Current real-estate pricing — data is from 1970s census.",
                "Individual property valuation — tract-level model only.",
                "Any jurisdiction outside the original Boston study area.",
            ],
        },
        "training_data": {
            "source":      "OpenML Boston Housing (fetch_openml name='boston', version=1)",
            "rows":        metrics.get("split", {}).get("train_rows"),
            "test_rows":   metrics.get("split", {}).get("test_rows"),
            "stratified":  False,
            "target":      TARGET,
            "target_unit": "Median home value in 1000s USD",
        },
        "evaluation_results": {
            "test_r2":          tuned.get("r2"),
            "test_rmse":        tuned.get("rmse"),
            "test_mae":         tuned.get("mae"),
            "test_mape":        tuned.get("mape"),
            "oof_rmse":         uncertainty_info.get("oof_rmse"),
            "uncertainty_band": {
                "lower": uncertainty_info.get("lower_band"),
                "upper": uncertainty_info.get("upper_band"),
            },
        },
        "fairness": {
            "overall_rmse":   fairness.get("overall_rmse"),
            "subgroup_rmse":  {
                col: {k: v.get("rmse") for k, v in groups.items()}
                for col, groups in fairness.get("subgroups", {}).items()
            },
            "alerts": [
                {"group": col, "value": val, "rmse_gap": data.get("rmse_gap")}
                for col, groups in fairness.get("subgroups", {}).items()
                for val, data in groups.items()
                if data.get("alert")
            ],
        },
        "limitations": [
            "Data from 1970s US Census — not representative of current housing markets.",
            "B (proportion Black residents) column reflects historical segregation data.",
            "CHAS binary variable encodes limited geographic information.",
            "Tract-level predictions — individual variation within a tract is not captured.",
        ],
        "ethical_considerations": [
            "Do not use for individual credit scoring, lending, or insurance pricing.",
            "B column is a proxy for historical racial composition; bias may be encoded.",
            "Model trained on historically discriminatory data — audit before deployment.",
        ],
        "hyperparameters":  search.best_params_,
        "cv_best_r2":       float(search.best_score_),
    }
    write_json(output_dir / MODEL_CARD_FILE, card)
    log.info("Model card saved to %s", output_dir / MODEL_CARD_FILE)


# ── MLflow tracking (mirrors log_to_mlflow from reference exactly) ────────────
def log_to_mlflow(
    metrics:          dict[str, Any],
    search:           RandomizedSearchCV,
    uncertainty_info: dict[str, Any],
    model:            Pipeline,
    output_dir:       Path,
) -> None:
    if not _MLFLOW_AVAILABLE:
        log.info("mlflow not installed — experiment tracking skipped.")
        return
    try:
        mlflow.set_experiment("boston_housing")
        tuned = metrics.get("test_metrics", {})
        with mlflow.start_run():
            flat_params = {f"best_{k}": str(v) for k, v in search.best_params_.items()}
            flat_params["n_jobs"]      = N_JOBS
            flat_params["random_state"] = RANDOM_STATE
            mlflow.log_params(flat_params)
            mlflow.log_metrics({
                "cv_best_r2":   float(search.best_score_),
                "test_r2":      float(tuned.get("r2", 0)),
                "test_rmse":    float(tuned.get("rmse", 0)),
                "test_mae":     float(tuned.get("mae", 0)),
                "test_mape":    float(tuned.get("mape", 0)),
                "oof_rmse":     float(uncertainty_info.get("oof_rmse", 0)),
            })
            for fname in [
                MODEL_CARD_FILE, METRICS_FILE, TRAINING_PROFILE_FILE,
                ENVIRONMENT_FILE, "fairness_report.json",
                "feature_importance.csv", "shap_importance.csv",
                "plot_actual_vs_predicted.png", "plot_shap_bar.png",
            ]:
                fpath = output_dir / fname
                if fpath.exists():
                    mlflow.log_artifact(str(fpath))
            mlflow.sklearn.log_model(model, "model")
        log.info("MLflow run logged successfully.")
    except Exception as exc:
        log.warning("MLflow logging failed (%s) — continuing without it.", exc)


# ── Artifact saving (mirrors save_model_artifacts from reference) ─────────────
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
    # ── Versioned model file (identical pattern to reference) ─────────────────
    timestamp      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_tag    = _model_version_tag(model)
    versioned_name = f"boston_price_pipeline_{timestamp}_{version_tag}.joblib"
    joblib.dump(model, output_dir / versioned_name)
    joblib.dump(model, output_dir / MODEL_FILE)
    log.info("Model saved: %s (also as %s)", versioned_name, MODEL_FILE)

    # ── Environment snapshot ──────────────────────────────────────────────────
    save_environment_snapshot(output_dir)

    pd.DataFrame(search.cv_results_).sort_values("rank_test_r2").to_csv(
        output_dir / "cv_results.csv", index=False
    )
    save_feature_importance(model, output_dir)
    save_evaluation_plots(y_test, y_pred, output_dir)
    save_error_analysis(X_test, y_test, y_pred, output_dir)
    write_json(output_dir / TRAINING_PROFILE_FILE,
               build_training_profile(X_train, y_train))


# ── Main training workflow (mirrors train() from reference exactly) ───────────
def train(
    output_dir:        Path,
    n_iter:            int,
    overpredict_cost:  float = 1.0,
    underpredict_cost: float = 1.0,
) -> dict[str, Any]:
    log.info("=== Training started (n_jobs=%d) ===", N_JOBS)
    output_dir.mkdir(parents=True, exist_ok=True)

    df                          = fix_data_types(load_data())
    X_train, X_test, y_train, y_test = split_data(df)

    research_decisions = save_research_artifacts(X_train, y_train, output_dir)
    baseline_metrics   = evaluate_baselines(X_train, X_test, y_train, y_test)

    search            = tune_model(X_train, y_train, n_iter=n_iter)
    uncertainty_info  = compute_oof_uncertainty(
        search.best_estimator_, X_train, y_train,
        overpredict_cost=overpredict_cost,
        underpredict_cost=underpredict_cost,
    )

    final_model = clone(search.best_estimator_)
    final_model.fit(X_train, y_train)
    y_pred      = final_model.predict(X_test)

    test_metrics = evaluate_predictions(y_test, y_pred)

    save_model_artifacts(
        final_model, search,
        X_train, y_train,
        X_test,  y_test,
        y_pred,
        output_dir,
    )

    # ── SHAP ──────────────────────────────────────────────────────────────────
    save_shap_artifacts(final_model, X_test, y_test, y_pred, output_dir)

    # ── Fairness / subgroup evaluation ────────────────────────────────────────
    fairness = evaluate_subgroups(X_test, y_test, y_pred, output_dir)

    metrics = {
        "research_decisions": research_decisions,
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows":  int(len(X_test)),
            "test_size":  0.2,
        },
        "baseline":          baseline_metrics,
        "best_cv": {
            "best_r2":    float(search.best_score_),
            "best_params": search.best_params_,
        },
        "uncertainty_info":  uncertainty_info,
        "test_metrics":      test_metrics,
        "fairness":          fairness,
    }
    write_json(output_dir / METRICS_FILE, metrics)

    # ── Model Card ────────────────────────────────────────────────────────────
    save_model_card(metrics, fairness, uncertainty_info, search, output_dir)

    # ── MLflow ────────────────────────────────────────────────────────────────
    log_to_mlflow(metrics, search, uncertainty_info, final_model, output_dir)

    log.info("=== Training complete ===")
    log.info("Test R²   : %.4f", test_metrics["r2"])
    log.info("Test RMSE : %.4f", test_metrics["rmse"])
    log.info("Test MAE  : %.4f", test_metrics["mae"])
    log.info("OOF RMSE  : %.4f", uncertainty_info["oof_rmse"])
    return to_jsonable(metrics)


# ── Inference (mirrors predict() from reference exactly) ─────────────────────
def predict(
    artifact_dir: Path,
    input_csv:    Path,
    output_csv:   Path,
) -> None:
    log.info("Loading model from %s …", artifact_dir / MODEL_FILE)
    model = joblib.load(artifact_dir / MODEL_FILE)

    if not hasattr(model, "predict"):
        raise TypeError(
            f"Loaded object ({type(model).__name__}) is not a fitted sklearn pipeline."
        )

    # Load uncertainty info from saved metrics
    metrics_path     = artifact_dir / METRICS_FILE
    uncertainty_band = UNCERTAINTY_BAND  # fallback
    if metrics_path.exists():
        saved            = json.loads(metrics_path.read_text(encoding="utf-8"))
        uncertainty_band = saved.get("uncertainty_info", {}).get("oof_rmse", UNCERTAINTY_BAND)
    log.info("Using uncertainty band: ±%.3f (OOF RMSE)", uncertainty_band)

    input_df = pd.read_csv(input_csv)

    # ── Pandera schema validation (mirrors reference pipeline) ────────────────
    if INPUT_SCHEMA is not None:
        try:
            INPUT_SCHEMA.validate(input_df, lazy=True)
            log.info("Input schema validation passed.")
        except Exception as exc:
            log.warning("Input schema validation errors: %s", exc)

    # ── Column completeness check ─────────────────────────────────────────────
    profile_path = artifact_dir / TRAINING_PROFILE_FILE
    if profile_path.exists():
        profile          = json.loads(profile_path.read_text(encoding="utf-8"))
        required_columns = set(profile["raw_columns"])
        incoming_columns = set(input_df.columns)
        missing_cols     = required_columns - incoming_columns
        if missing_cols:
            raise ValueError(
                f"Input CSV missing {len(missing_cols)} required column(s): "
                + ", ".join(sorted(missing_cols))
            )

    y_pred = model.predict(input_df)

    output_df = input_df.copy()
    output_df["predicted_medv"]   = y_pred
    output_df["lower_bound"]      = y_pred - uncertainty_band
    output_df["upper_bound"]      = y_pred + uncertainty_band

    # ── Confidence flag (mirrors reference's low_confidence flag) ─────────────
    output_df["wide_interval"] = (
        (output_df["upper_bound"] - output_df["lower_bound"]) > uncertainty_band * 2.5
    ).astype(int)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    log.info("Predictions saved to %s", output_csv.resolve())


# ── Monitoring (mirrors monitor() from reference exactly) ─────────────────────
def monitor(
    artifact_dir:       Path,
    input_csv:          Path,
    output_json:        Path,
    missing_rate_alert: float,
    ks_pvalue_alert:    float = 0.05,
) -> dict[str, Any]:
    """
    Data drift monitoring with:
      - Missing-rate drift alerts
      - KS-test distribution drift using stored quantiles
    Mirrors monitor() from reference pipeline exactly.
    """
    log.info("Running monitoring checks …")
    profile  = json.loads((artifact_dir / TRAINING_PROFILE_FILE).read_text(encoding="utf-8"))
    incoming = pd.read_csv(input_csv)

    required_columns = set(profile["raw_columns"])
    incoming_columns = set(incoming.columns)

    # ── Missing-rate drift ────────────────────────────────────────────────────
    drift_rows = []
    for column, train_rate in profile["raw_missing_rate"].items():
        if column not in incoming:
            continue
        current_rate = float(incoming[column].isna().mean())
        change       = abs(current_rate - float(train_rate))
        drift_rows.append({
            "column":               column,
            "train_missing_rate":   float(train_rate),
            "current_missing_rate": current_rate,
            "absolute_change":      change,
            "alert":                change >= missing_rate_alert,
        })

    # ── KS distribution drift (quantile-based — no raw-value bloat) ──────────
    ks_rows       = []
    numeric_stats = profile.get("numeric_train_stats", {})
    for col, stats in numeric_stats.items():
        if col not in incoming.columns:
            continue
        incoming_values = incoming[col].dropna().to_numpy()
        if len(incoming_values) < 10:
            continue
        quantiles        = np.array(stats["quantiles"])
        ks_stat, p_value = ks_2samp(quantiles, incoming_values)
        ks_rows.append({
            "column":        col,
            "ks_statistic":  float(ks_stat),
            "p_value":       float(p_value),
            "train_mean":    float(stats["mean"]),
            "incoming_mean": float(np.mean(incoming_values)),
            "train_std":     float(stats["std"]),
            "incoming_std":  float(np.std(incoming_values)),
            "alert":         p_value < ks_pvalue_alert,
        })

    report = {
        "checked_at":                   datetime.now(timezone.utc).isoformat(),
        "row_count":                    int(len(incoming)),
        "missing_required_columns":     sorted(required_columns - incoming_columns),
        "extra_columns":                sorted(incoming_columns - required_columns),
        "missing_rate_alert_threshold": missing_rate_alert,
        "missing_rate_drift":           drift_rows,
        "ks_pvalue_alert_threshold":    ks_pvalue_alert,
        "distribution_drift":           ks_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, report)
    return report


# ── Utilities (identical to reference pipeline) ───────────────────────────────
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    """Recursively convert to JSON-safe primitive. Mirrors reference pipeline."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, BaseEstimator):
        return repr(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        fv = float(value)
        return None if (np.isnan(fv) or np.isinf(fv)) else fv
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float):
        return None if (np.isnan(value) or np.isinf(value)) else value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def create_sample_input(output_csv: Path, rows: int) -> None:
    df = fix_data_types(load_data())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[TARGET]).head(rows).to_csv(output_csv, index=False)
    log.info("Sample input saved to %s", output_csv.resolve())


# ── CLI (mirrors reference pipeline CLI pattern exactly) ──────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end research, training, inference, and monitoring for Boston Housing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    tp = subparsers.add_parser("train", help="Train and evaluate model.")
    tp.add_argument("--output-dir",        type=Path,  default=Path("artifacts_boston"))
    tp.add_argument("--n-iter",            type=int,   default=20)
    tp.add_argument("--overpredict-cost",  type=float, default=1.0,
                    help="Relative cost of overpredicting (for uncertainty band).")
    tp.add_argument("--underpredict-cost", type=float, default=1.0,
                    help="Relative cost of underpredicting.")

    # predict
    pp = subparsers.add_parser("predict", help="Run inference on a CSV.")
    pp.add_argument("--artifact-dir", type=Path, default=Path("artifacts_boston"))
    pp.add_argument("--input-csv",    type=Path, required=True)
    pp.add_argument("--output-csv",   type=Path,
                    default=Path("artifacts_boston/predictions.csv"))

    # monitor
    mp = subparsers.add_parser("monitor", help="Check for data drift.")
    mp.add_argument("--artifact-dir",       type=Path,  default=Path("artifacts_boston"))
    mp.add_argument("--input-csv",          type=Path,  required=True)
    mp.add_argument("--output-json",        type=Path,
                    default=Path("artifacts_boston/monitoring_report.json"))
    mp.add_argument("--missing-rate-alert", type=float, default=0.15)
    mp.add_argument("--ks-pvalue-alert",    type=float, default=0.05)

    # sample-input
    sp = subparsers.add_parser("sample-input", help="Export sample input rows.")
    sp.add_argument("--output-csv", type=Path,
                    default=Path("artifacts_boston/sample_houses.csv"))
    sp.add_argument("--rows", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        metrics = train(
            args.output_dir, args.n_iter,
            overpredict_cost=args.overpredict_cost,
            underpredict_cost=args.underpredict_cost,
        )
        log.info("Best CV R²  : %.3f", metrics["best_cv"]["best_r2"])
        log.info("Test R²     : %.3f", metrics["test_metrics"]["r2"])
        log.info("Test RMSE   : %.3f", metrics["test_metrics"]["rmse"])
        log.info("OOF RMSE    : %.3f", metrics["uncertainty_info"]["oof_rmse"])

    elif args.command == "predict":
        predict(args.artifact_dir, args.input_csv, args.output_csv)

    elif args.command == "monitor":
        report = monitor(
            args.artifact_dir, args.input_csv, args.output_json,
            args.missing_rate_alert, args.ks_pvalue_alert,
        )
        miss_alerts = sum(r["alert"] for r in report["missing_rate_drift"])
        ks_alerts   = sum(r["alert"] for r in report["distribution_drift"])
        log.info("Monitoring report saved: %s", args.output_json.resolve())
        log.info("Missing-rate alerts: %d", miss_alerts)
        log.info("Distribution-drift (KS) alerts: %d", ks_alerts)

    elif args.command == "sample-input":
        create_sample_input(args.output_csv, args.rows)


if __name__ == "__main__":
    main()