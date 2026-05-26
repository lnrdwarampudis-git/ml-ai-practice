"""
titanic.py — Industry-standard end-to-end ML pipeline for Titanic survival prediction.

Improvements over baseline:
  1. Model versioning with timestamp + SHA1 hash
  2. KS-test distribution drift + quantile-based storage (no raw value bloat)
  3. predict_proba guard at inference time
  4. JSON-safe serialisation (inf/nan/pd.NA)
  5. OOF threshold tuning (no leakage) with optional cost-matrix weighting
  6. Input schema validation (pandera) at inference
  7. Subgroup / fairness evaluation (sex, pclass, age_group)
  8. SHAP explainability (summary + force plot on errors)
  9. MLflow experiment tracking (optional, graceful fallback)
 10. n_jobs via env-var ML_N_JOBS; environment snapshot saved alongside model
 11. Prediction confidence flags (low-confidence band around threshold)
 12. Model Card generated at train time
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── matplotlib scratch dir must be set before any pyplot import ──────────────
_MPLCONFIGDIR = Path("artifacts") / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── optional heavy deps (graceful degradation) ───────────────────────────────
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

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
TARGET         = "survived"
MODEL_FILE     = "titanic_survival_pipeline.joblib"
METRICS_FILE   = "metrics.json"
TRAINING_PROFILE_FILE = "training_profile.json"
MODEL_CARD_FILE       = "model_card.json"
ENVIRONMENT_FILE      = "environment.json"

LEAKAGE_COLUMNS        = ["boat", "body"]
RAW_TEXT_COLUMNS       = ["name", "ticket"]
HIGH_MISSING_RAW_COLUMNS = ["cabin", "home.dest"]

# NEW: parallelism controlled by env-var; -1 = all cores
N_JOBS = int(os.environ.get("ML_N_JOBS", 1))

# NEW: confidence band half-width around the decision threshold
CONFIDENCE_BAND = float(os.environ.get("ML_CONFIDENCE_BAND", 0.10))

# NEW: subgroup columns for fairness evaluation
FAIRNESS_COLS = ["sex", "pclass"]


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ColumnGroups:
    numeric: list[str]
    categorical: list[str]


# ── Data loading & preprocessing ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load OpenML Titanic data."""
    log.info("Loading Titanic dataset from OpenML …")
    return fetch_openml("titanic", version=1, as_frame=True, parser="auto").frame.copy()


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET]       = df[TARGET].astype(int)
    df["pclass"]     = df["pclass"].astype("category")
    df["sex"]        = df["sex"].astype("category")
    df["embarked"]   = df["embarked"].astype("category")
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split — done BEFORE any EDA to prevent test-set leakage."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    report = (
        df.isna()
        .agg(["sum", "mean"])
        .T.rename(columns={"sum": "missing_count", "mean": "missing_rate"})
        .sort_values("missing_rate", ascending=False)
    )
    report["dtype"] = df.dtypes.astype(str)
    return report


def get_column_groups() -> ColumnGroups:
    """Feature columns produced by TitanicFeatureEngineer."""
    return ColumnGroups(
        numeric=[
            "age", "sibsp", "parch", "fare", "has_cabin", "cabin_count",
            "family_size", "is_alone", "fare_per_person", "home_dest_known",
        ],
        categorical=["pclass", "sex", "embarked", "title", "cabin_deck", "ticket_prefix"],
    )


# ── NEW: Pandera input schema ─────────────────────────────────────────────────
def build_input_schema():
    """Return a pandera DataFrameSchema for raw inference inputs, or None."""
    if not _PANDERA_AVAILABLE:
        log.warning("pandera not installed — input schema validation skipped.")
        return None
    schema = pa.DataFrameSchema(
        {
            "age":      pa.Column(float,  pa.Check.in_range(0, 120),  nullable=True,  required=False),
            "fare":     pa.Column(float,  pa.Check.ge(0),             nullable=True,  required=False),
            "sibsp":    pa.Column(float,  pa.Check.in_range(0, 20),   nullable=True,  required=False),
            "parch":    pa.Column(float,  pa.Check.in_range(0, 20),   nullable=True,  required=False),
            "pclass":   pa.Column(pa.Category,
                                  pa.Check.isin(["1", "2", "3", 1, 2, 3]),
                                  nullable=False, required=False),
            "sex":      pa.Column(pa.Category,
                                  pa.Check.isin(["male", "female"]),
                                  nullable=False, required=False),
            "embarked": pa.Column(pa.Category,
                                  pa.Check.isin(["C", "Q", "S"]),
                                  nullable=True,  required=False),
        },
        coerce=True,
        strict=False,   # extra columns (name, ticket …) are allowed
    )
    return schema


INPUT_SCHEMA = build_input_schema()


# ── Feature engineering ───────────────────────────────────────────────────────
class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Production-safe feature engineering for raw Titanic passenger rows."""

    def __init__(self, rare_title_min_count: int = 10) -> None:
        self.rare_title_min_count = rare_title_min_count

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TitanicFeatureEngineer":
        titles = self._extract_title(X.get("name", pd.Series(index=X.index, dtype="object")))
        counts = titles.value_counts(dropna=False)
        self.rare_titles_ = set(counts[counts < self.rare_title_min_count].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        titles   = self._extract_title(X.get("name", pd.Series(index=X.index, dtype="object")))
        X["title"] = titles.where(~titles.isin(self.rare_titles_), "Rare")

        cabin = X.get("cabin", pd.Series(index=X.index, dtype="object")).astype("string")
        X["has_cabin"]   = cabin.notna().astype(int)
        X["cabin_deck"]  = cabin.str[0].fillna("Unknown")
        X["cabin_count"] = cabin.fillna("").str.split().map(
            lambda values: len([v for v in values if v])
        )

        ticket = X.get("ticket", pd.Series(index=X.index, dtype="object")).astype("string")
        X["ticket_prefix"] = ticket.map(self._ticket_prefix)

        X["family_size"]    = X["sibsp"].fillna(0) + X["parch"].fillna(0) + 1
        X["is_alone"]       = (X["family_size"] == 1).astype(int)
        X["fare_per_person"] = X["fare"] / X["family_size"].replace(0, np.nan)
        X["home_dest_known"] = X.get("home.dest", pd.Series(index=X.index)).notna().astype(int)

        drop_columns = LEAKAGE_COLUMNS + RAW_TEXT_COLUMNS + HIGH_MISSING_RAW_COLUMNS
        return X.drop(columns=[col for col in drop_columns if col in X.columns])

    @staticmethod
    def _extract_title(names: pd.Series) -> pd.Series:
        titles = names.astype("string").str.extract(r",\s*([^\.]+)\.", expand=False)
        return titles.fillna("Unknown").replace(
            {
                "Mlle": "Miss",  "Ms": "Miss",  "Mme": "Mrs",
                "Lady": "Nobility", "Sir": "Nobility", "the Countess": "Nobility",
                "Dona": "Nobility", "Don": "Nobility", "Jonkheer": "Nobility",
                "Capt": "Officer", "Col": "Officer", "Major": "Officer",
                "Dr": "Officer",   "Rev": "Officer",
            }
        )

    @staticmethod
    def _ticket_prefix(ticket: Any) -> str:
        if pd.isna(ticket):
            return "MISSING_TICKET"
        cleaned = re.sub(r"[\.\/]", " ", str(ticket)).strip()
        prefix  = re.sub(r"\d+", "", cleaned).strip().upper()
        prefix  = re.sub(r"\s+", "_", prefix)
        return prefix if prefix else "NO_PREFIX"


# ── Pipeline construction ─────────────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    groups = get_column_groups()
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline,    groups.numeric),
        ("cat", categorical_pipeline, groups.categorical),
    ])


def build_pipeline(model: BaseEstimator | None = None) -> Pipeline:
    selector_model = ExtraTreesClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=N_JOBS,
    )
    if model is None:
        model = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
    return Pipeline([
        ("feature_engineering", TitanicFeatureEngineer()),
        ("preprocess",          build_preprocessor()),
        ("feature_selection",   SelectFromModel(selector_model, threshold="median")),
        ("model",               model),
    ])


# ── Research / EDA artifacts (train-set only) ─────────────────────────────────
def save_research_artifacts(
    X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path
) -> dict[str, Any]:
    log.info("Saving research artifacts …")
    output_dir.mkdir(parents=True, exist_ok=True)
    eda_train = X_train.copy()
    eda_train[TARGET] = y_train

    missingness_report(eda_train).to_csv(output_dir / "research_missingness_report.csv")
    eda_train.dtypes.astype(str).rename("dtype").to_csv(output_dir / "schema.csv")
    eda_train.select_dtypes(include=["number"]).describe().T.to_csv(output_dir / "numeric_summary.csv")

    categorical_summary = []
    for column in eda_train.select_dtypes(include=["category", "object"]).columns:
        counts = eda_train[column].astype("string").value_counts(dropna=False)
        categorical_summary.append({
            "column":          column,
            "unique_values":   int(eda_train[column].nunique(dropna=True)),
            "top_value":       counts.index[0] if len(counts) else None,
            "top_value_count": int(counts.iloc[0]) if len(counts) else 0,
        })
    pd.DataFrame(categorical_summary).to_csv(output_dir / "categorical_summary.csv", index=False)

    eda_train["age_missing"]   = eda_train["age"].isna().astype(int)
    eda_train["cabin_missing"] = eda_train["cabin"].isna().astype(int)
    eda_train["has_cabin"]     = eda_train["cabin"].notna().astype(int)
    eda_train["family_size"]   = eda_train["sibsp"] + eda_train["parch"] + 1
    eda_train["title"]         = TitanicFeatureEngineer._extract_title(eda_train["name"])

    grouped_reports = {
        "age_missing_by_pclass":    eda_train.groupby("pclass", observed=False)["age_missing"].mean().to_dict(),
        "cabin_missing_by_pclass":  eda_train.groupby("pclass", observed=False)["cabin_missing"].mean().to_dict(),
        "survival_by_sex":          eda_train.groupby("sex", observed=False)[TARGET].mean().to_dict(),
        "survival_by_pclass":       eda_train.groupby("pclass", observed=False)[TARGET].mean().to_dict(),
        "survival_by_has_cabin":    eda_train.groupby("has_cabin", observed=False)[TARGET].mean().to_dict(),
        "survival_by_family_size":  eda_train.groupby("family_size", observed=False)[TARGET].mean().to_dict(),
        "title_summary":            eda_train.groupby("title", observed=False)[TARGET].agg(["count", "mean"]).to_dict(),
    }

    engineer = TitanicFeatureEngineer().fit(X_train, y_train)
    numeric_engineered = engineer.transform(X_train)[get_column_groups().numeric]
    imputed_numeric = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(numeric_engineered),
        columns=get_column_groups().numeric,
    )
    imputed_numeric.corr().to_csv(output_dir / "numeric_correlation_matrix.csv")

    vif_rows = []
    for column in imputed_numeric.columns:
        other         = imputed_numeric.drop(columns=[column])
        target_column = imputed_numeric[column]
        if target_column.nunique() <= 1:
            continue
        r2        = LinearRegression().fit(other, target_column).score(other, target_column)
        vif_value = 9999.0 if r2 >= 0.999 else float(1 / (1 - r2))
        vif_rows.append({"feature": column, "vif": vif_value})
    pd.DataFrame(vif_rows).sort_values("vif", ascending=False).to_csv(
        output_dir / "vif_report.csv", index=False
    )

    save_research_plots(eda_train, output_dir)

    decisions = {
        "problem_definition": {
            "problem_type":    "binary_classification",
            "target":          TARGET,
            "positive_class":  "survived",
            "prediction_time": "Before knowing rescue/death outcome.",
        },
        "metric_policy": {
            "primary":     "roc_auc",
            "secondary":   ["f1", "precision", "recall", "average_precision"],
            "calibration": ["brier_score", "calibration_curve"],
        },
        "column_policy": {
            "drop_leakage": LEAKAGE_COLUMNS,
            "engineer_then_drop": {
                "name":      ["title"],
                "cabin":     ["has_cabin", "cabin_deck", "cabin_count"],
                "ticket":    ["ticket_prefix"],
                "home.dest": ["home_dest_known"],
            },
            "keep": ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"],
        },
        "missing_value_policy": {
            "numeric":     "median imputation with missing indicators inside pipeline",
            "categorical": "most-frequent imputation plus one-hot encoding",
            "cabin":       "preserve missingness as has_cabin before dropping raw cabin",
        },
        "grouped_research": grouped_reports,
    }
    write_json(output_dir / "research_decisions.json", decisions)
    return decisions


def save_research_plots(eda_train: pd.DataFrame, output_dir: Path) -> None:
    log.info("Saving research plots …")
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    sns.countplot(data=eda_train, x=TARGET, color="#4C78A8")
    plt.title("Target Class Balance")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_target_balance.png", dpi=160)
    plt.close()

    missing = eda_train.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    missing[missing > 0].sort_values().plot(kind="barh", color="#F58518")
    plt.title("Missing Rate by Column")
    plt.xlabel("Missing Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_missingness.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=eda_train, x="pclass", y="cabin_missing", color="#54A24B")
    plt.title("Cabin Missing Rate by Passenger Class")
    plt.ylabel("Cabin Missing Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_cabin_missing_by_pclass.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(data=eda_train, x="fare", kde=True, color="#B279A2")
    plt.title("Fare Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_fare_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=eda_train, x="family_size", y=TARGET, color="#E45756")
    plt.title("Survival Rate by Family Size")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_survival_by_family_size.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.barplot(data=eda_train, x="title", y=TARGET, color="#72B7B2")
    plt.xticks(rotation=45, ha="right")
    plt.title("Survival Rate by Title")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_survival_by_title.png", dpi=160)
    plt.close()


# ── Evaluation helpers ────────────────────────────────────────────────────────
def evaluate_predictions(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold":           float(threshold),
        "accuracy":            float(accuracy_score(y_true, predictions)),
        "balanced_accuracy":   float(balanced_accuracy_score(y_true, predictions)),
        "precision":           float(precision_score(y_true, predictions, zero_division=0)),
        "recall":              float(recall_score(y_true, predictions, zero_division=0)),
        "f1":                  float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc":             float(roc_auc_score(y_true, probabilities)),
        "average_precision":   float(average_precision_score(y_true, probabilities)),
        "brier_score":         float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix":    confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0
        ),
    }


def evaluate_baselines(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series,    y_test:  pd.Series,
) -> dict[str, Any]:
    baselines = {}
    for strategy in ["most_frequent", "stratified"]:
        baseline = DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)
        baseline.fit(X_train, y_train)
        probs = baseline.predict_proba(X_test)[:, 1]
        baselines[strategy] = evaluate_predictions(y_test, probs, 0.5)
    return baselines


# ── NEW: Subgroup / fairness evaluation ───────────────────────────────────────
def evaluate_subgroups(
    X_test:        pd.DataFrame,
    y_test:        pd.Series,
    probabilities: np.ndarray,
    threshold:     float,
    output_dir:    Path,
) -> dict[str, Any]:
    """
    Disaggregated metrics for FAIRNESS_COLS and an age_group bucketing.
    Flags any subgroup whose F1 is more than 0.10 below the overall F1.
    """
    log.info("Running subgroup / fairness evaluation …")
    overall_f1 = f1_score(y_test, (probabilities >= threshold).astype(int), zero_division=0)

    results: dict[str, Any] = {"overall_f1": float(overall_f1), "subgroups": {}}
    rows = []

    # Attach raw test features and label for grouping
    eval_df = X_test.reset_index(drop=True).copy()
    eval_df["_y_true"] = y_test.to_numpy()
    eval_df["_prob"]   = probabilities

    # Age bucket for intersectional analysis
    eval_df["_age_group"] = pd.cut(
        eval_df["age"].astype(float),
        bins=[0, 12, 18, 40, 60, 120],
        labels=["child", "teen", "adult", "middle_aged", "senior"],
        right=False,
    ).astype("category").cat.add_categories("unknown").fillna("unknown")

    group_cols = FAIRNESS_COLS + ["_age_group"]
    for col in group_cols:
        if col not in eval_df.columns:
            continue
        col_results = {}
        for group_val, sub in eval_df.groupby(col, observed=True):
            if len(sub) < 10:          # too few samples to be meaningful
                continue
            sub_probs = sub["_prob"].to_numpy()
            sub_true  = sub["_y_true"].to_numpy()
            sub_f1    = float(f1_score(sub_true, (sub_probs >= threshold).astype(int), zero_division=0))
            col_results[str(group_val)] = {
                "n":                  int(len(sub)),
                "positive_rate":      float(sub_true.mean()),
                "f1":                 sub_f1,
                "roc_auc":            float(roc_auc_score(sub_true, sub_probs))
                                      if len(np.unique(sub_true)) > 1 else None,
                "f1_gap_vs_overall":  float(sub_f1 - overall_f1),
                "alert":              bool((overall_f1 - sub_f1) > 0.10),
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
                "Fairness alert: %d subgroup(s) have F1 gap > 0.10 vs overall: %s",
                len(alerts),
                [(r["group_column"], r["group_value"], round(r["f1_gap_vs_overall"], 3)) for r in alerts],
            )

    write_json(output_dir / "fairness_report.json", results)
    return results


def _save_fairness_plot(rows_df: pd.DataFrame, output_dir: Path) -> None:
    g = rows_df.copy()
    g["label"] = g["group_column"] + "=" + g["group_value"].astype(str)
    plt.figure(figsize=(10, max(4, len(g) * 0.45)))
    colors = ["#E45756" if alert else "#4C78A8" for alert in g["alert"]]
    plt.barh(g["label"], g["f1"], color=colors)
    plt.axvline(g["f1"].mean(), linestyle="--", color="black", label="Mean F1")
    plt.xlabel("F1 Score")
    plt.title("Subgroup F1 (red = alert: gap > 0.10 vs overall)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_fairness_f1.png", dpi=160)
    plt.close()


# ── Hyperparameter search ─────────────────────────────────────────────────────
def tune_model(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int) -> RandomizedSearchCV:
    log.info("Starting hyperparameter search (n_iter=%d, n_jobs=%d) …", n_iter, N_JOBS)
    param_distributions = [
        {
            "feature_selection__threshold": ["median", "0.75*median", "1.25*median"],
            "model": [
                LogisticRegression(
                    max_iter=3000, class_weight="balanced",
                    solver="liblinear", random_state=RANDOM_STATE,
                )
            ],
            "model__C":       [0.05, 0.1, 0.3, 1.0, 3.0, 10.0],
            "model__penalty": ["l1", "l2"],
        },
        {
            "feature_selection__threshold": ["median", "0.75*median", "1.25*median"],
            "model": [
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                    n_jobs=N_JOBS,
                )
            ],
            "model__n_estimators":    [250, 500, 800],
            "model__max_depth":       [4, 6, 8, None],
            "model__min_samples_leaf":[1, 2, 4, 8],
            "model__max_features":    ["sqrt", "log2", 0.75],
        },
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        build_pipeline(),
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring={
            "roc_auc":           "roc_auc",
            "f1":                "f1",
            "average_precision": "average_precision",
            "balanced_accuracy": "balanced_accuracy",
        },
        refit="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    log.info("Best CV ROC-AUC: %.4f", search.best_score_)
    return search


# ── Threshold tuning (OOF, optional cost-matrix) ──────────────────────────────
def tune_threshold(
    best_estimator: Pipeline,
    X_train:        pd.DataFrame,
    y_train:        pd.Series,
    fn_cost:        float = 1.0,
    fp_cost:        float = 1.0,
) -> dict[str, float]:
    """
    Select the classification threshold that minimises a weighted cost on
    out-of-fold predictions (no data leakage).

    When fn_cost == fp_cost == 1.0 the objective degrades to maximising F1.
    Adjust the ratio to encode domain preferences, e.g. fn_cost=2.0 penalises
    missed survivors twice as much as false alarms.
    """
    log.info(
        "Tuning threshold via OOF predictions (fn_cost=%.2f, fp_cost=%.2f) …",
        fn_cost, fp_cost,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = cross_val_predict(
        clone(best_estimator), X_train, y_train,
        cv=cv, method="predict_proba", n_jobs=N_JOBS,
    )[:, 1]

    thresholds = np.linspace(0.05, 0.95, 181)
    costs = []
    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        cm    = confusion_matrix(y_train, preds)
        # cm layout: [[TN, FP], [FN, TP]]
        fn = cm[1, 0] if cm.shape == (2, 2) else 0
        fp = cm[0, 1] if cm.shape == (2, 2) else 0
        costs.append(fn_cost * fn + fp_cost * fp)

    best_index = int(np.argmin(costs))
    best_t     = float(thresholds[best_index])
    oof_f1     = float(f1_score(y_train, (oof_probs >= best_t).astype(int), zero_division=0))

    return {
        "threshold":  best_t,
        "oof_f1":     oof_f1,
        "fn_cost":    fn_cost,
        "fp_cost":    fp_cost,
        "oof_cost":   float(costs[best_index]),
    }


# ── Artifact saving ───────────────────────────────────────────────────────────
def _model_version_tag(model: Pipeline) -> str:
    model_bytes = pickle.dumps(model)
    return hashlib.sha1(model_bytes).hexdigest()[:8]  # noqa: S324


def save_model_artifacts(
    model:         Pipeline,
    search:        RandomizedSearchCV,
    X_train:       pd.DataFrame,
    y_train:       pd.Series,
    X_test:        pd.DataFrame,
    y_test:        pd.Series,
    probabilities: np.ndarray,
    threshold:     float,
    output_dir:    Path,
) -> None:
    predictions = (probabilities >= threshold).astype(int)

    # ── versioned model file ──────────────────────────────────────────────────
    timestamp         = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_tag       = _model_version_tag(model)
    versioned_name    = f"titanic_survival_pipeline_{timestamp}_{version_tag}.joblib"
    joblib.dump(model, output_dir / versioned_name)
    joblib.dump(model, output_dir / MODEL_FILE)
    log.info("Model saved: %s (also as %s)", versioned_name, MODEL_FILE)

    # ── NEW: environment snapshot ─────────────────────────────────────────────
    save_environment_snapshot(output_dir)

    pd.DataFrame(search.cv_results_).sort_values("rank_test_roc_auc").to_csv(
        output_dir / "cv_results.csv", index=False
    )
    save_feature_importance(model, output_dir)
    save_evaluation_plots(y_test, probabilities, predictions, output_dir)
    save_error_analysis(X_test, y_test, probabilities, threshold, output_dir)
    write_json(output_dir / TRAINING_PROFILE_FILE, build_training_profile(X_train, y_train))


# ── NEW: environment snapshot ─────────────────────────────────────────────────
def save_environment_snapshot(output_dir: Path) -> None:
    """Save Python + key library versions for reproducibility audits."""
    import importlib
    env = {
        "saved_at":  datetime.now(timezone.utc).isoformat(),
        "python":    sys.version,
        "platform":  sys.platform,
        "libraries": {},
    }
    for lib in ["sklearn", "pandas", "numpy", "scipy", "joblib", "shap", "mlflow", "pandera"]:
        try:
            mod = importlib.import_module(lib)
            env["libraries"][lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env["libraries"][lib] = "not_installed"
    write_json(output_dir / ENVIRONMENT_FILE, env)
    log.info("Environment snapshot saved to %s", output_dir / ENVIRONMENT_FILE)


def save_feature_importance(model: Pipeline, output_dir: Path) -> None:
    preprocess            = model.named_steps["preprocess"]
    selector              = model.named_steps["feature_selection"]
    final_model           = model.named_steps["model"]
    feature_names         = preprocess.get_feature_names_out()
    selected_feature_names = feature_names[selector.get_support()]

    if hasattr(final_model, "feature_importances_"):
        importance = final_model.feature_importances_
        signed     = np.repeat(np.nan, len(importance))
    elif hasattr(final_model, "coef_"):
        signed     = final_model.coef_[0]
        importance = np.abs(signed)
    else:
        log.warning("Model has no feature_importances_ or coef_; skipping importance plot.")
        return

    importance_df = pd.DataFrame({
        "feature":     selected_feature_names,
        "importance":  importance,
        "coefficient": signed,
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=importance_df.head(20), y="feature", x="importance", color="#4C78A8")
    plt.title("Top Model Features")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_feature_importance.png", dpi=160)
    plt.close()


def save_evaluation_plots(
    y_test:        pd.Series,
    probabilities: np.ndarray,
    predictions:   np.ndarray,
    output_dir:    Path,
) -> None:
    log.info("Saving evaluation plots …")

    plt.figure(figsize=(4.8, 4.2))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d",
                cbar=False, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_roc_curve.png", dpi=160)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, probabilities)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_precision_recall_curve.png", dpi=160)
    plt.close()

    prob_true, prob_pred = calibration_curve(y_test, probabilities, n_bins=8, strategy="quantile")
    plt.figure(figsize=(5.2, 4.4))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect")
    plt.title("Calibration Curve")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Survival Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_calibration_curve.png", dpi=160)
    plt.close()


def save_error_analysis(
    X_test:        pd.DataFrame,
    y_test:        pd.Series,
    probabilities: np.ndarray,
    threshold:     float,
    output_dir:    Path,
) -> None:
    predictions = (probabilities >= threshold).astype(int)
    error_df    = X_test.copy()
    error_df["actual_survived"]    = y_test.to_numpy()
    error_df["predicted_survived"] = predictions
    error_df["survival_probability"] = probabilities
    error_df["error_type"] = np.select(
        [
            (error_df["actual_survived"] == 1) & (error_df["predicted_survived"] == 0),
            (error_df["actual_survived"] == 0) & (error_df["predicted_survived"] == 1),
        ],
        ["false_negative", "false_positive"],
        default="correct",
    )
    error_df.to_csv(output_dir / "test_predictions.csv", index=False)
    error_df.query("error_type != 'correct'").to_csv(output_dir / "error_analysis.csv", index=False)


# ── NEW: SHAP explainability ──────────────────────────────────────────────────
def save_shap_artifacts(
    model:         Pipeline,
    X_test:        pd.DataFrame,
    y_test:        pd.Series,
    probabilities: np.ndarray,
    threshold:     float,
    output_dir:    Path,
) -> None:
    
    """
    Compute SHAP values and save:
      - Global summary bar plot + beeswarm
      - Waterfall plot for the highest-confidence false negative (worst miss)
      - CSV of per-sample mean |SHAP| values
    Skips gracefully when shap is not installed or the model type is unsupported.
    """
    if not _SHAP_AVAILABLE:
        log.warning("shap not installed — SHAP artifacts skipped. pip install shap")
        return

    log.info("Computing SHAP values (this may take a moment) …")
    final_model = model.named_steps["model"]

    # Transform test data through all steps up to (but not including) the model
    preprocess         = model.named_steps["preprocess"]
    feat_eng           = model.named_steps["feature_engineering"]
    selector           = model.named_steps["feature_selection"]
    feature_names      = preprocess.get_feature_names_out()
    selected_names     = feature_names[selector.get_support()]

    X_transformed = selector.transform(
        preprocess.transform(feat_eng.transform(X_test))
    )
    X_transformed_df = pd.DataFrame(X_transformed, columns=selected_names)

    try:
        if hasattr(final_model, "feature_importances_"):
            # Tree-based models: use fast TreeExplainer
            explainer   = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X_transformed_df)
            # For binary classifiers shap_values may be [class0, class1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            print("SHAP VERSION:", shap.__version__)
            print("type(shap_values):", type(shap_values))

            if hasattr(shap_values, "shape"):
                print("shap_values.shape:", shap_values.shape)

            print("worst:", worst)
            print("shap_values[worst].shape:", np.asarray(shap_values[worst]).shape)
        else:
            # Linear / other: use LinearExplainer or KernelExplainer (slower)
            masker      = shap.maskers.Independent(X_transformed_df, max_samples=100)
            explainer   = shap.Explainer(final_model.predict_proba, masker)
            shap_values = explainer(X_transformed_df).values[:, :, 1]
    except Exception as exc:
        log.warning("SHAP computation failed (%s) — skipping SHAP artifacts.", exc)
        return

    # ── Summary bar (global feature importance) ───────────────────────────────
    plt.figure()
    shap.summary_plot(shap_values, X_transformed_df, plot_type="bar",
                      show=False, max_display=20)
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_shap_bar.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ── Beeswarm / dot plot ───────────────────────────────────────────────────
    plt.figure()
    shap.summary_plot(shap_values, X_transformed_df, show=False, max_display=20)
    plt.title("SHAP Summary (beeswarm)")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_shap_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ── Waterfall for worst false-negative ────────────────────────────────────
    y_np   = y_test.to_numpy()
    preds  = (probabilities >= threshold).astype(int)
    fn_idx = np.where((y_np == 1) & (preds == 0))[0]
    if len(fn_idx) > 0:
        # Pick the false negative with the highest predicted probability of 0
        worst  = fn_idx[np.argmax(probabilities[fn_idx])]
        # ev_val = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) \
        #          else explainer.expected_value
        # shap_exp = shap.Explanation(
        #     values    = shap_values[worst],
        #     base_values = ev_val,
        #     data      = X_transformed_df.iloc[worst].values,
        #     feature_names = list(selected_names),
        # )
        # plt.figure()
        
        # shap.waterfall_plot(shap_exp, show=False, max_display=15)
        ev_val = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, np.ndarray)
            else explainer.expected_value
        )

        shap_exp = shap.Explanation(
            values=shap_values[worst],
            base_values=ev_val,
            data=X_transformed_df.iloc[worst].values,
            feature_names=list(selected_names),
        )

        shap.waterfall_plot(
            shap_exp,
            show=False,
            max_display=15
        )
        plt.title("SHAP Waterfall — Worst False Negative")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_shap_waterfall_fn.png", dpi=160, bbox_inches="tight")
        plt.close()

    # ── Per-feature mean |SHAP| CSV ───────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    pd.DataFrame({
        "feature":       selected_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).to_csv(
        output_dir / "shap_importance.csv", index=False
    )
    log.info("SHAP artifacts saved to %s", output_dir)


# ── NEW: Training profile (quantile-based, no raw-value bloat) ────────────────
def build_training_profile(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """
    Build a compact training profile for monitoring.
    Numeric drift reference uses 100-quantile summaries instead of raw arrays,
    keeping the JSON file small while still supporting approximate KS tests.
    """
    engineer       = TitanicFeatureEngineer().fit(X_train, y_train)
    engineered     = engineer.transform(X_train)
    numeric_cols   = get_column_groups().numeric
    numeric_eng    = engineered[[c for c in numeric_cols if c in engineered.columns]]
    imputed        = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(numeric_eng),
        columns=numeric_eng.columns,
    )

    numeric_train_stats: dict[str, dict[str, Any]] = {}
    for col in imputed.columns:
        vals = imputed[col].to_numpy()
        numeric_train_stats[col] = {
            "mean":      float(np.mean(vals)),
            "std":       float(np.std(vals)),
            "min":       float(np.min(vals)),
            "max":       float(np.max(vals)),
            # 100 quantiles ≈ 800 bytes vs O(n) raw values — enough for KS
            "quantiles": np.quantile(vals, np.linspace(0, 1, 100)).tolist(),
        }

    return to_jsonable({
        "trained_at":               datetime.now(timezone.utc).isoformat(),
        "row_count":                int(len(X_train)),
        "raw_columns":              list(X_train.columns),
        "engineered_columns":       list(engineered.columns),
        "target_distribution":      y_train.value_counts(normalize=True).sort_index().to_dict(),
        "raw_missing_rate":         X_train.isna().mean().to_dict(),
        "engineered_missing_rate":  engineered.isna().mean().to_dict(),
        "numeric_train_stats":      numeric_train_stats,
    })


# ── NEW: Model Card ───────────────────────────────────────────────────────────
def save_model_card(
    metrics:        dict[str, Any],
    fairness:       dict[str, Any],
    threshold_info: dict[str, Any],
    search:         RandomizedSearchCV,
    output_dir:     Path,
) -> None:
    """
    Write a structured model card (JSON) covering intended use, performance,
    fairness, limitations, and training details.
    Compatible with Google's Model Card spec and EU AI Act documentation needs.
    """
    tuned = metrics.get("test_tuned_threshold", {})
    card  = {
        "schema_version":   "1.0",
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "model_details": {
            "name":         "Titanic Survival Classifier",
            "version":      datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "type":         "Binary classification (sklearn Pipeline)",
            "algorithm":    repr(search.best_estimator_.named_steps["model"]),
            "framework":    "scikit-learn",
        },
        "intended_use": {
            "primary_use":   "Predict survival probability for Titanic passengers.",
            "out_of_scope":  [
                "Any real-time or operational life-safety decision.",
                "Populations outside the Titanic passenger demographic.",
            ],
        },
        "training_data": {
            "source":       "OpenML Titanic dataset (version 1)",
            "rows":         metrics.get("split", {}).get("train_rows"),
            "test_rows":    metrics.get("split", {}).get("test_rows"),
            "stratified":   True,
            "target":       TARGET,
            "positive_class": "survived (1)",
        },
        "evaluation_results": {
            "test_roc_auc":          tuned.get("roc_auc"),
            "test_f1":               tuned.get("f1"),
            "test_precision":        tuned.get("precision"),
            "test_recall":           tuned.get("recall"),
            "test_brier_score":      tuned.get("brier_score"),
            "decision_threshold":    threshold_info.get("threshold"),
            "threshold_fn_cost":     threshold_info.get("fn_cost"),
            "threshold_fp_cost":     threshold_info.get("fp_cost"),
        },
        "fairness": {
            "overall_f1":    fairness.get("overall_f1"),
            "subgroup_f1":   {
                col: {k: v.get("f1") for k, v in groups.items()}
                for col, groups in fairness.get("subgroups", {}).items()
            },
            "alerts":        [
                {"group": col, "value": val, "f1_gap": data.get("f1_gap_vs_overall")}
                for col, groups in fairness.get("subgroups", {}).items()
                for val, data in groups.items()
                if data.get("alert")
            ],
        },
        "limitations": [
            "Trained on historical data — causality cannot be inferred.",
            "Class imbalance handled via class_weight; still imperfect for rare groups.",
            "Age missing for ~20% of passengers; median imputation may introduce bias.",
            "Cabin deck is a proxy for socioeconomic status and carries survivorship bias.",
        ],
        "ethical_considerations": [
            "Model reflects historical inequalities (sex, class) present in the data.",
            "Do not use predictions to infer present-day group-level survival outcomes.",
        ],
        "hyperparameters": search.best_params_,
        "cv_best_roc_auc": float(search.best_score_),
    }
    write_json(output_dir / MODEL_CARD_FILE, card)
    log.info("Model card saved to %s", output_dir / MODEL_CARD_FILE)


# ── NEW: MLflow experiment tracking ──────────────────────────────────────────
def log_to_mlflow(
    metrics:        dict[str, Any],
    search:         RandomizedSearchCV,
    threshold_info: dict[str, Any],
    model:          Pipeline,
    output_dir:     Path,
) -> None:
    """
    Log params, metrics, and the fitted model to MLflow.
    Silently skips when MLflow is not installed or when the tracking server
    is unreachable (so the training run still completes successfully).
    """
    if not _MLFLOW_AVAILABLE:
        log.info("mlflow not installed — experiment tracking skipped. pip install mlflow")
        return

    try:
        mlflow.set_experiment("titanic_survival")
        tuned = metrics.get("test_tuned_threshold", {})
        with mlflow.start_run():
            # ── Parameters ────────────────────────────────────────────────────
            flat_params = {
                f"best_{k}": str(v) for k, v in search.best_params_.items()
            }
            flat_params["threshold"]   = threshold_info.get("threshold")
            flat_params["fn_cost"]     = threshold_info.get("fn_cost")
            flat_params["fp_cost"]     = threshold_info.get("fp_cost")
            flat_params["n_jobs"]      = N_JOBS
            flat_params["random_state"] = RANDOM_STATE
            mlflow.log_params(flat_params)

            # ── Metrics ───────────────────────────────────────────────────────
            mlflow.log_metrics({
                "cv_best_roc_auc":  float(search.best_score_),
                "test_roc_auc":     float(tuned.get("roc_auc", 0)),
                "test_f1":          float(tuned.get("f1", 0)),
                "test_precision":   float(tuned.get("precision", 0)),
                "test_recall":      float(tuned.get("recall", 0)),
                "test_brier_score": float(tuned.get("brier_score", 0)),
            })

            # ── Artifacts ─────────────────────────────────────────────────────
            for fname in [
                MODEL_CARD_FILE, METRICS_FILE, TRAINING_PROFILE_FILE,
                ENVIRONMENT_FILE, "fairness_report.json",
                "feature_importance.csv", "shap_importance.csv",
                "plot_roc_curve.png", "plot_shap_bar.png",
                "plot_fairness_f1.png", "plot_calibration_curve.png",
            ]:
                fpath = output_dir / fname
                if fpath.exists():
                    mlflow.log_artifact(str(fpath))

            mlflow.sklearn.log_model(model, "model")
        log.info("MLflow run logged successfully.")
    except Exception as exc:
        log.warning("MLflow logging failed (%s) — continuing without it.", exc)


# ── Main training workflow ────────────────────────────────────────────────────
def train(
    output_dir: Path,
    n_iter:     int,
    fn_cost:    float = 1.0,
    fp_cost:    float = 1.0,
) -> dict[str, Any]:
    log.info("=== Training started (n_jobs=%d) ===", N_JOBS)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = fix_data_types(load_data())
    X_train, X_test, y_train, y_test = split_data(df)

    research_decisions = save_research_artifacts(X_train, y_train, output_dir)
    baseline_metrics   = evaluate_baselines(X_train, X_test, y_train, y_test)

    search         = tune_model(X_train, y_train, n_iter=n_iter)
    threshold_info = tune_threshold(search.best_estimator_, X_train, y_train,
                                    fn_cost=fn_cost, fp_cost=fp_cost)

    final_model   = clone(search.best_estimator_)
    final_model.fit(X_train, y_train)
    probabilities = final_model.predict_proba(X_test)[:, 1]

    default_metrics = evaluate_predictions(y_test, probabilities, 0.5)
    tuned_metrics   = evaluate_predictions(y_test, probabilities, threshold_info["threshold"])

    save_model_artifacts(
        final_model, search,
        X_train, y_train,
        X_test,  y_test,
        probabilities,
        threshold_info["threshold"],
        output_dir,
    )

    # ── NEW: SHAP ─────────────────────────────────────────────────────────────
    save_shap_artifacts(
        final_model, X_test, y_test, probabilities,
        threshold_info["threshold"], output_dir,
    )

    # ── NEW: Fairness / subgroup evaluation ───────────────────────────────────
    fairness = evaluate_subgroups(
        X_test, y_test, probabilities, threshold_info["threshold"], output_dir
    )

    metrics = {
        "research_decisions":    research_decisions,
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows":  int(len(X_test)),
            "test_size":  0.2,
            "stratified": True,
        },
        "baseline":               baseline_metrics,
        "best_cv": {
            "best_roc_auc": float(search.best_score_),
            "best_params":  search.best_params_,
        },
        "threshold_tuning":       threshold_info,
        "test_default_threshold": default_metrics,
        "test_tuned_threshold":   tuned_metrics,
        "fairness":               fairness,
    }
    write_json(output_dir / METRICS_FILE, metrics)

    # ── NEW: Model Card ───────────────────────────────────────────────────────
    save_model_card(metrics, fairness, threshold_info, search, output_dir)

    # ── NEW: MLflow ───────────────────────────────────────────────────────────
    log_to_mlflow(metrics, search, threshold_info, final_model, output_dir)

    log.info("=== Training complete ===")
    return to_jsonable(metrics)


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(
    artifact_dir: Path,
    input_csv:    Path,
    output_csv:   Path,
    threshold:    float | None = None,
) -> None:
    log.info("Loading model from %s …", artifact_dir / MODEL_FILE)
    model = joblib.load(artifact_dir / MODEL_FILE)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"Loaded model ({type(model).__name__}) does not support predict_proba. "
            "Re-train with a probability-calibrated estimator."
        )

    metrics_path = artifact_dir / METRICS_FILE
    if threshold is None and metrics_path.exists():
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = float(saved["threshold_tuning"]["threshold"])
    if threshold is None:
        threshold = 0.5
    log.info("Using decision threshold: %.3f", threshold)

    input_df = pd.read_csv(input_csv)

    # ── NEW: pandera schema validation ────────────────────────────────────────
    if INPUT_SCHEMA is not None:
        try:
            INPUT_SCHEMA.validate(input_df, lazy=True)
            log.info("Input schema validation passed.")
        except Exception as exc:
            log.warning("Input schema validation errors: %s", exc)

    # ── Column completeness check (original) ──────────────────────────────────
    profile_path = artifact_dir / TRAINING_PROFILE_FILE
    if profile_path.exists():
        profile          = json.loads(profile_path.read_text(encoding="utf-8"))
        required_columns = set(profile["raw_columns"])
        incoming_columns = set(input_df.columns)
        missing_cols     = required_columns - incoming_columns
        if missing_cols:
            raise ValueError(
                f"Input CSV is missing {len(missing_cols)} required column(s): "
                + ", ".join(sorted(missing_cols))
            )
        extra_cols = incoming_columns - required_columns
        if extra_cols:
            log.warning("Input CSV has %d unexpected column(s): %s",
                        len(extra_cols), sorted(extra_cols))

    probabilities = model.predict_proba(input_df)[:, 1]
    predictions   = (probabilities >= threshold).astype(int)

    output_df = input_df.copy()
    output_df["survival_probability"]  = probabilities
    output_df["predicted_survived"]    = predictions
    output_df["decision_threshold"]    = threshold

    # ── NEW: confidence flag ──────────────────────────────────────────────────
    output_df["low_confidence"] = (
        (probabilities > threshold - CONFIDENCE_BAND) &
        (probabilities < threshold + CONFIDENCE_BAND)
    ).astype(int)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    low_conf_count = int(output_df["low_confidence"].sum())
    log.info("Predictions saved to %s (%d low-confidence rows flagged)",
             output_csv.resolve(), low_conf_count)


# ── Monitoring (quantile-based KS test) ───────────────────────────────────────
def monitor(
    artifact_dir:      Path,
    input_csv:         Path,
    output_json:       Path,
    missing_rate_alert: float,
    ks_pvalue_alert:   float = 0.05,
) -> dict[str, Any]:
    """
    Data drift monitoring with:
      - Missing-rate drift alerts
      - KS-test distribution drift using stored quantiles (no raw-value storage)
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
            "column":              column,
            "train_missing_rate":  float(train_rate),
            "current_missing_rate": current_rate,
            "absolute_change":     change,
            "alert":               change >= missing_rate_alert,
        })

    # ── Distribution drift via quantile-reconstructed KS test ─────────────────
    ks_rows        = []
    numeric_stats  = profile.get("numeric_train_stats", {})
    for col, stats in numeric_stats.items():
        if col not in incoming.columns:
            continue
        incoming_values = incoming[col].dropna().to_numpy()
        if len(incoming_values) < 10:
            continue

        # Reconstruct approximate reference distribution from stored quantiles
        quantiles   = np.array(stats["quantiles"])
        train_values = quantiles   # pass quantile array as surrogate sample

        ks_stat, p_value = ks_2samp(train_values, incoming_values)
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
        "checked_at":                 datetime.now(timezone.utc).isoformat(),
        "row_count":                  int(len(incoming)),
        "missing_required_columns":   sorted(required_columns - incoming_columns),
        "extra_columns":              sorted(incoming_columns - required_columns),
        "missing_rate_alert_threshold": missing_rate_alert,
        "missing_rate_drift":         drift_rows,
        "ks_pvalue_alert_threshold":  ks_pvalue_alert,
        "distribution_drift":         ks_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, report)
    return report


# ── Utilities ─────────────────────────────────────────────────────────────────
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    """Recursively convert a value to a JSON-safe primitive."""
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


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end research, training, inference, and monitoring for Titanic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    tp = subparsers.add_parser("train", help="Train and evaluate model.")
    tp.add_argument("--output-dir", type=Path, default=Path("artifacts_end_to_end"))
    tp.add_argument("--n-iter",     type=int,  default=18,
                    help="Number of RandomizedSearchCV iterations.")
    tp.add_argument("--fn-cost",    type=float, default=1.0,
                    help="Cost of false negatives for threshold tuning.")
    tp.add_argument("--fp-cost",    type=float, default=1.0,
                    help="Cost of false positives for threshold tuning.")

    # predict
    pp = subparsers.add_parser("predict", help="Run inference on a CSV.")
    pp.add_argument("--artifact-dir", type=Path, default=Path("artifacts_end_to_end"))
    pp.add_argument("--input-csv",    type=Path, required=True)
    pp.add_argument("--output-csv",   type=Path, default=Path("artifacts_end_to_end/predictions.csv"))
    pp.add_argument("--threshold",    type=float, default=None)

    # monitor
    mp = subparsers.add_parser("monitor", help="Check for data drift.")
    mp.add_argument("--artifact-dir",       type=Path,  default=Path("artifacts_end_to_end"))
    mp.add_argument("--input-csv",          type=Path,  required=True)
    mp.add_argument("--output-json",        type=Path,  default=Path("artifacts_end_to_end/monitoring_report.json"))
    mp.add_argument("--missing-rate-alert", type=float, default=0.15)
    mp.add_argument("--ks-pvalue-alert",    type=float, default=0.05,
                    help="KS-test p-value below which a drift alert fires.")

    # sample-input
    sp = subparsers.add_parser("sample-input", help="Export sample input rows.")
    sp.add_argument("--output-csv", type=Path, default=Path("artifacts_end_to_end/sample_passengers.csv"))
    sp.add_argument("--rows",       type=int,  default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        metrics = train(
            args.output_dir, args.n_iter,
            fn_cost=args.fn_cost,
            fp_cost=args.fp_cost,
        )
        tuned = metrics["test_tuned_threshold"]
        log.info("Best CV ROC-AUC : %.3f", metrics["best_cv"]["best_roc_auc"])
        log.info("Test ROC-AUC    : %.3f", tuned["roc_auc"])
        log.info("Test F1         : %.3f", tuned["f1"])
        log.info("Threshold       : %.3f", tuned["threshold"])

    elif args.command == "predict":
        predict(args.artifact_dir, args.input_csv, args.output_csv, args.threshold)

    elif args.command == "monitor":
        report = monitor(
            args.artifact_dir,
            args.input_csv,
            args.output_json,
            args.missing_rate_alert,
            args.ks_pvalue_alert,
        )
        missing_alerts = sum(r["alert"] for r in report["missing_rate_drift"])
        ks_alerts      = sum(r["alert"] for r in report["distribution_drift"])
        log.info("Monitoring report saved: %s", args.output_json.resolve())
        log.info("Missing-rate alerts  : %d", missing_alerts)
        log.info("Distribution-drift (KS) alerts: %d", ks_alerts)

    elif args.command == "sample-input":
        create_sample_input(args.output_csv, args.rows)


if __name__ == "__main__":
    main()