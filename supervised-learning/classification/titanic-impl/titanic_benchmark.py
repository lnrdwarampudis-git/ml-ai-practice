"""
titanic_benchmark.py
====================
Industry-standard algorithm benchmarking script for Titanic survival prediction.

Built on top of titanic-ml-pipeline.py:
  - Reuses load_data / fix_data_types / split_data exactly as-is
  - Reuses TitanicFeatureEngineer, build_preprocessor, get_column_groups, to_jsonable
  - Reuses save_research_artifacts / save_research_plots / evaluate_subgroups

Adds a full professional benchmark workflow:
  Phase 1  — EDA  (train-set only, from reference pipeline)
  Phase 2  — Algorithm screening: 15 classifiers across every sklearn family
             + XGBoost + LightGBM (industry de-facto)
             5-fold stratified CV, 6 metrics
  Phase 3  — Statistical significance: Friedman + Wilcoxon–Bonferroni
  Phase 4  — Hyperparameter tuning for top-3 models (RandomizedSearchCV)
  Phase 5  — Ensemble construction: SoftVoting + Stacking
  Phase 6  — Hold-out test evaluation: 10-metric suite, Youden-J threshold
  Phase 7  — Calibration analysis (Brier + reliability diagrams)
  Phase 8  — SHAP explainability for champion model
  Phase 9  — Fairness / subgroup evaluation (from reference pipeline)
  Phase 10 — Self-contained HTML + JSON + CSV report

References:
  https://scikit-learn.org/stable/supervised_learning.html

Usage:
  python titanic_benchmark.py                        # full run (~8-12 min)
  python titanic_benchmark.py --quick                # smoke-test (~45 s)
  python titanic_benchmark.py --output-dir ./results
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
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── matplotlib scratch dir before any pyplot import ──────────────────────────
_MPLCFG = Path("benchmark_out") / ".matplotlib"
_MPLCFG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCFG))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, ks_2samp, wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:
    _XGB = False

try:
    from lightgbm import LGBMClassifier
    _LGB = True
except ImportError:
    _LGB = False

try:
    import shap
    _SHAP = True
except ImportError:
    _SHAP = False

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (identical to reference pipeline)
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE            = 42
TARGET                  = "survived"
LEAKAGE_COLUMNS         = ["boat", "body"]
RAW_TEXT_COLUMNS        = ["name", "ticket"]
HIGH_MISSING_RAW_COLUMNS = ["cabin", "home.dest"]
FAIRNESS_COLS           = ["sex", "pclass"]
N_JOBS                  = int(os.environ.get("ML_N_JOBS", -1))

# Benchmark-specific
N_CV_SPLITS  = 5
TOP_N_TUNE   = 3
TUNE_N_ITER  = 40
ALPHA        = 0.05   # significance level

CV_SCORING = {
    "roc_auc":           "roc_auc",
    "f1":                "f1",
    "average_precision": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
    "precision":         "precision",
    "recall":            "recall",
}


# ─────────────────────────────────────────────────────────────────────────────
# ── Shared infrastructure (copied verbatim from reference pipeline) ───────────
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ColumnGroups:
    numeric: list[str]
    categorical: list[str]


def get_column_groups() -> ColumnGroups:
    return ColumnGroups(
        numeric=[
            "age", "sibsp", "parch", "fare", "has_cabin", "cabin_count",
            "family_size", "is_alone", "fare_per_person", "home_dest_known",
        ],
        categorical=["pclass", "sex", "embarked", "title", "cabin_deck", "ticket_prefix"],
    )


# ── Data loading (identical to reference) ────────────────────────────────────
def load_data() -> pd.DataFrame:
    log.info("Loading Titanic dataset from OpenML …")
    return fetch_openml("titanic", version=1, as_frame=True, parser="auto").frame.copy()


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET]     = df[TARGET].astype(int)
    df["pclass"]   = df["pclass"].astype("category")
    df["sex"]      = df["sex"].astype("category")
    df["embarked"] = df["embarked"].astype("category")
    return df


def split_data(df):
    """Stratified 80/20 split — done BEFORE EDA to prevent leakage."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def missingness_report(df):
    report = (
        df.isna()
        .agg(["sum", "mean"])
        .T.rename(columns={"sum": "missing_count", "mean": "missing_rate"})
        .sort_values("missing_rate", ascending=False)
    )
    report["dtype"] = df.dtypes.astype(str)
    return report


# ── Feature engineering (identical to reference) ─────────────────────────────
class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, rare_title_min_count: int = 10):
        self.rare_title_min_count = rare_title_min_count

    def fit(self, X, y=None):
        titles = self._extract_title(
            X.get("name", pd.Series(index=X.index, dtype="object"))
        )
        counts = titles.value_counts(dropna=False)
        self.rare_titles_ = set(counts[counts < self.rare_title_min_count].index)
        return self

    def transform(self, X):
        X = X.copy()
        titles = self._extract_title(
            X.get("name", pd.Series(index=X.index, dtype="object"))
        )
        X["title"] = titles.where(~titles.isin(self.rare_titles_), "Rare")

        cabin = X.get("cabin", pd.Series(index=X.index, dtype="object")).astype("string")
        X["has_cabin"]   = cabin.notna().astype(int)
        X["cabin_deck"]  = cabin.str[0].fillna("Unknown")
        X["cabin_count"] = cabin.fillna("").str.split().map(
            lambda v: len([x for x in v if x])
        )

        ticket = X.get("ticket", pd.Series(index=X.index, dtype="object")).astype("string")
        X["ticket_prefix"] = ticket.map(self._ticket_prefix)

        X["family_size"]    = X["sibsp"].fillna(0) + X["parch"].fillna(0) + 1
        X["is_alone"]       = (X["family_size"] == 1).astype(int)
        X["fare_per_person"] = X["fare"] / X["family_size"].replace(0, np.nan)
        X["home_dest_known"] = X.get("home.dest", pd.Series(index=X.index)).notna().astype(int)

        drop = LEAKAGE_COLUMNS + RAW_TEXT_COLUMNS + HIGH_MISSING_RAW_COLUMNS
        return X.drop(columns=[c for c in drop if c in X.columns])

    @staticmethod
    def _extract_title(names):
        t = names.astype("string").str.extract(r",\s*([^\.]+)\.", expand=False)
        return t.fillna("Unknown").replace({
            "Mlle": "Miss",  "Ms": "Miss",   "Mme": "Mrs",
            "Lady": "Nobility", "Sir": "Nobility", "the Countess": "Nobility",
            "Dona": "Nobility", "Don": "Nobility", "Jonkheer": "Nobility",
            "Capt": "Officer", "Col": "Officer", "Major": "Officer",
            "Dr":   "Officer", "Rev": "Officer",
        })

    @staticmethod
    def _ticket_prefix(ticket):
        if pd.isna(ticket):
            return "MISSING_TICKET"
        cleaned = re.sub(r"[\.\/]", " ", str(ticket)).strip()
        prefix  = re.sub(r"\d+", "", cleaned).strip().upper()
        prefix  = re.sub(r"\s+", "_", prefix)
        return prefix if prefix else "NO_PREFIX"


# ── Preprocessor (identical to reference) ────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    groups = get_column_groups()
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=5,
                                  sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, groups.numeric),
        ("cat", cat_pipe, groups.categorical),
    ])


def wrap_classifier(clf: BaseEstimator) -> Pipeline:
    """
    Wrap any classifier in the full reference-pipeline stack:
      TitanicFeatureEngineer → ColumnTransformer → classifier
    No SelectFromModel — we benchmark the raw classifier signal cleanly.
    """
    return Pipeline([
        ("feature_engineering", TitanicFeatureEngineer()),
        ("preprocess",          build_preprocessor()),
        ("model",               clf),
    ])


# ── EDA (reused from reference) ───────────────────────────────────────────────
def save_research_artifacts(X_train, y_train, output_dir):
    log.info("Saving EDA artifacts (train-set only) …")
    output_dir.mkdir(parents=True, exist_ok=True)
    eda = X_train.copy()
    eda[TARGET] = y_train

    missingness_report(eda).to_csv(output_dir / "eda_missingness.csv")
    eda.dtypes.astype(str).rename("dtype").to_csv(output_dir / "eda_schema.csv")
    eda.select_dtypes(include=["number"]).describe().T.to_csv(output_dir / "eda_numeric_summary.csv")

    eda["age_missing"]   = eda["age"].isna().astype(int)
    eda["cabin_missing"] = eda["cabin"].isna().astype(int)
    eda["has_cabin"]     = eda["cabin"].notna().astype(int)
    eda["family_size"]   = eda["sibsp"] + eda["parch"] + 1
    eda["title"]         = TitanicFeatureEngineer._extract_title(eda["name"])

    grouped = {
        "survival_by_sex":    eda.groupby("sex",    observed=False)[TARGET].mean().to_dict(),
        "survival_by_pclass": eda.groupby("pclass", observed=False)[TARGET].mean().to_dict(),
        "survival_by_title":  eda.groupby("title",  observed=False)[TARGET].mean().to_dict(),
        "cabin_missing_by_pclass": eda.groupby("pclass", observed=False)["cabin_missing"].mean().to_dict(),
    }
    write_json(output_dir / "eda_grouped_stats.json", grouped)

    # Correlation matrix on engineered numerics
    eng    = TitanicFeatureEngineer().fit(X_train, y_train).transform(X_train)
    num_df = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(eng[get_column_groups().numeric]),
        columns=get_column_groups().numeric,
    )
    num_df.corr().to_csv(output_dir / "eda_correlation_matrix.csv")

    # VIF
    vif_rows = []
    for col in num_df.columns:
        other = num_df.drop(columns=[col])
        tgt   = num_df[col]
        if tgt.nunique() <= 1:
            continue
        r2  = LinearRegression().fit(other, tgt).score(other, tgt)
        vif = 9999.0 if r2 >= 0.999 else float(1 / (1 - r2))
        vif_rows.append({"feature": col, "vif": vif})
    pd.DataFrame(vif_rows).sort_values("vif", ascending=False).to_csv(
        output_dir / "eda_vif_report.csv", index=False)

    # EDA plots
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 3.5))
    sns.countplot(data=eda, x=TARGET, palette=["#E45756","#4C78A8"])
    plt.title("Target Class Balance"); plt.tight_layout()
    plt.savefig(output_dir / "eda_target_balance.png", dpi=150); plt.close()

    miss = eda.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    miss[miss > 0].sort_values().plot(kind="barh", color="#F58518")
    plt.title("Missing Rate by Column"); plt.tight_layout()
    plt.savefig(output_dir / "eda_missingness.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 3.5))
    sns.barplot(data=eda, x="pclass", y=TARGET, palette="Blues_d")
    plt.title("Survival Rate by Class"); plt.tight_layout()
    plt.savefig(output_dir / "eda_survival_by_class.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 3.5))
    sns.barplot(data=eda, x="sex", y=TARGET, palette="Set2")
    plt.title("Survival Rate by Sex"); plt.tight_layout()
    plt.savefig(output_dir / "eda_survival_by_sex.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 3.5))
    sns.histplot(data=eda, x="age", hue=TARGET, kde=True, bins=30)
    plt.title("Age Distribution by Survival"); plt.tight_layout()
    plt.savefig(output_dir / "eda_age_distribution.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 3.5))
    sns.barplot(data=eda, x="family_size", y=TARGET, color="#72B7B2")
    plt.title("Survival Rate by Family Size"); plt.tight_layout()
    plt.savefig(output_dir / "eda_survival_by_family.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=eda, x="title", y=TARGET, color="#54A24B")
    plt.xticks(rotation=40, ha="right"); plt.title("Survival Rate by Title")
    plt.tight_layout(); plt.savefig(output_dir / "eda_survival_by_title.png", dpi=150); plt.close()

    log.info("EDA artifacts saved to %s", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# ── Phase 2: Classifier registry ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def get_classifiers(quick: bool = False) -> dict[str, BaseEstimator]:
    """
    15 classifiers covering every major family in:
    https://scikit-learn.org/stable/supervised_learning.html

    Section 1.1  Linear Models      → LogisticRegression, SGD
    Section 1.2  Discriminant       → LDA, QDA (QDA uses reg_param for stability)
    Section 1.4  SVM                → SVC (RBF kernel, probability=True)
    Section 1.6  Nearest Neighbours → KNeighborsClassifier
    Section 1.9  Naive Bayes        → GaussianNB
    Section 1.10 Decision Trees     → DecisionTreeClassifier
    Section 1.11 Ensembles          → RandomForest, ExtraTrees,
                                      GradientBoosting, AdaBoost
    Section 1.17 Neural Networks    → MLPClassifier
    Industry     XGBoost + LightGBM (conditional on install)
    """
    n = 100 if quick else 300
    clfs: dict[str, BaseEstimator] = {
        # ── 1.1 Linear ────────────────────────────────────────────────────
        "LogisticRegression": LogisticRegression(
            max_iter=3000, class_weight="balanced",
            solver="liblinear", C=1.0, random_state=RANDOM_STATE,
        ),
        "SGD": SGDClassifier(
            loss="log_loss", penalty="elasticnet", l1_ratio=0.15,
            class_weight="balanced", max_iter=1000,
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
        ),
        # ── 1.2 Discriminant Analysis ─────────────────────────────────────
        "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),   # reg_param avoids singular covariance
        # ── 1.4 SVM ───────────────────────────────────────────────────────
        "SVM_RBF": SVC(
            kernel="rbf", probability=True,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        # ── 1.6 k-Nearest Neighbours ──────────────────────────────────────
        "KNN": KNeighborsClassifier(
            n_neighbors=7, weights="distance", n_jobs=N_JOBS,
        ),
        # ── 1.9 Naive Bayes ───────────────────────────────────────────────
        "GaussianNB": GaussianNB(),
        # ── 1.10 Decision Tree ────────────────────────────────────────────
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced",
            min_samples_leaf=4, random_state=RANDOM_STATE,
        ),
        # ── 1.11 Ensemble methods ─────────────────────────────────────────
        "RandomForest": RandomForestClassifier(
            n_estimators=n, class_weight="balanced_subsample",
            max_depth=8, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=n, class_weight="balanced_subsample",
            max_depth=8, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=n, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.5,
            random_state=RANDOM_STATE,
        ),
        # ── 1.17 Neural Network ───────────────────────────────────────────
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu",
            max_iter=500, early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }
    if _XGB:
        clfs["XGBoost"] = XGBClassifier(
            n_estimators=n, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=1.5, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0,
        )
    if _LGB:
        clfs["LightGBM"] = LGBMClassifier(
            n_estimators=n, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1,
        )
    return clfs


# ── Hyperparameter search spaces ──────────────────────────────────────────────
PARAM_GRIDS: dict[str, dict] = {
    "LogisticRegression": {
        "model__C":       [0.01, 0.05, 0.1, 0.5, 1, 3, 10, 30],
        "model__penalty": ["l1", "l2"],
        "model__solver":  ["liblinear"],
    },
    "SGD": {
        "model__alpha":    np.logspace(-4, 0, 10).tolist(),
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "model__penalty":  ["l2", "elasticnet"],
    },
    "LDA": {
        "model__solver":    ["svd", "lsqr", "eigen"],
        "model__shrinkage": [None, "auto", 0.1, 0.3, 0.5, 0.9],
    },
    "SVM_RBF": {
        "model__C":     [0.1, 0.5, 1, 5, 10, 50],
        "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    },
    "KNN": {
        "model__n_neighbors": [3, 5, 7, 11, 15, 21],
        "model__weights":     ["uniform", "distance"],
        "model__metric":      ["euclidean", "manhattan", "minkowski"],
    },
    "RandomForest": {
        "model__n_estimators":     [200, 400, 600],
        "model__max_depth":        [4, 6, 8, None],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features":     ["sqrt", "log2", 0.5, 0.75],
    },
    "ExtraTrees": {
        "model__n_estimators":     [200, 400, 600],
        "model__max_depth":        [4, 6, 8, None],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features":     ["sqrt", "log2", 0.5],
    },
    "GradientBoosting": {
        "model__n_estimators":     [100, 200, 300],
        "model__max_depth":        [3, 4, 5],
        "model__learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "model__subsample":        [0.6, 0.8, 1.0],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "DecisionTree": {
        "model__max_depth":        [3, 4, 5, 6, 8, None],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__criterion":        ["gini", "entropy"],
    },
    "AdaBoost": {
        "model__n_estimators":  [50, 100, 200, 300],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
    },
    "MLP": {
        "model__hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128, 64)],
        "model__alpha":              [0.0001, 0.001, 0.01],
        "model__learning_rate_init": [0.001, 0.005, 0.01],
    },
    "XGBoost": {
        "model__n_estimators":    [100, 200, 300],
        "model__max_depth":       [3, 4, 5, 6],
        "model__learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "model__subsample":       [0.6, 0.8, 1.0],
        "model__colsample_bytree":[0.5, 0.7, 1.0],
        "model__reg_alpha":       [0, 0.1, 0.5],
    },
    "LightGBM": {
        "model__n_estimators":    [100, 200, 300],
        "model__max_depth":       [3, 4, 5, 6],
        "model__learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "model__subsample":       [0.6, 0.8, 1.0],
        "model__colsample_bytree":[0.5, 0.7, 1.0],
        "model__reg_alpha":       [0, 0.1, 0.5],
    },
    "GaussianNB": {
        "model__var_smoothing": np.logspace(-9, -5, 10).tolist(),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# ── Phase 2: CV Screening ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def screen_classifiers(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    classifiers: dict[str, BaseEstimator],
    n_splits: int,
) -> pd.DataFrame:
    """
    Run n_splits-fold stratified CV for every classifier.
    Captures mean ± std for 6 metrics + train–test overfit gap.
    """
    cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, clf in classifiers.items():
        log.info("  [Screen] %-22s …", name)
        t0 = time.perf_counter()
        try:
            res = cross_validate(
                wrap_classifier(clone(clf)),
                X_tr, y_tr,
                cv=cv,
                scoring=CV_SCORING,
                return_train_score=True,
                n_jobs=1,
                error_score="raise",
            )
            elapsed = time.perf_counter() - t0
            row = {"model": name, "cv_time_s": round(elapsed, 2)}
            for m in CV_SCORING:
                ts  = res[f"test_{m}"]
                trs = res[f"train_{m}"]
                row[f"{m}_mean"]        = float(ts.mean())
                row[f"{m}_std"]         = float(ts.std())
                row[f"{m}_train_mean"]  = float(trs.mean())
                row[f"{m}_overfit_gap"] = float(trs.mean() - ts.mean())
                row[f"_raw_{m}"]        = ts.tolist()   # kept for stat tests
        except Exception as exc:
            log.warning("    %s — FAILED: %s", name, exc)
            row = {"model": name, "cv_time_s": -1,
                   **{f"{m}_mean": np.nan for m in CV_SCORING}}
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# ── Phase 3: Statistical significance tests ───────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def run_stat_tests(screen_df: pd.DataFrame, metric: str = "roc_auc") -> dict[str, Any]:
    """
    1. Friedman χ² test across all CV score arrays (H₀: all equal)
    2. Pairwise Wilcoxon signed-rank vs champion with Bonferroni correction
    """
    raw_col = f"_raw_{metric}"
    valid   = screen_df.dropna(subset=[f"{metric}_mean"]).copy()
    if raw_col not in valid.columns or len(valid) < 2:
        return {}

    arrays = [np.array(r) for r in valid[raw_col]]
    names  = valid["model"].tolist()
    best_i = int(valid[f"{metric}_mean"].idxmax())
    champ  = valid.loc[best_i, "model"]
    champ_sc = arrays[list(valid.index).index(best_i)]

    # Friedman test
    try:
        f_stat, f_p = friedmanchisquare(*arrays)
    except Exception:
        f_stat, f_p = np.nan, np.nan

    # Pairwise Wilcoxon vs champion + Bonferroni
    n_comp   = len(arrays) - 1
    pairwise = []
    for nm, sc in zip(names, arrays):
        if nm == champ:
            continue
        try:
            diff     = champ_sc - sc
            stat, p  = (wilcoxon(diff, alternative="greater", zero_method="wilcox")
                        if not np.all(diff == 0) else (np.nan, 1.0))
        except Exception:
            stat, p  = np.nan, np.nan
        p_bonf = float(min(1.0, p * n_comp)) if not np.isnan(p) else np.nan
        pairwise.append({
            "model":                nm,
            "wilcoxon_stat":        float(stat)   if not np.isnan(stat) else None,
            "p_value":              float(p)      if not np.isnan(p)    else None,
            "p_bonferroni":         float(p_bonf) if not np.isnan(p_bonf) else None,
            "significantly_worse":  bool(p_bonf < ALPHA) if not np.isnan(p_bonf) else None,
        })

    return {
        "metric":         metric,
        "champion_model": champ,
        "friedman": {
            "statistic":   float(f_stat) if not np.isnan(f_stat) else None,
            "p_value":     float(f_p)    if not np.isnan(f_p)    else None,
            "significant": bool(f_p < ALPHA) if not np.isnan(f_p) else None,
        },
        "pairwise_vs_champion": pairwise,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ── Phase 4: Hyperparameter tuning ────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def tune_top_models(
    screen_df:   pd.DataFrame,
    classifiers: dict[str, BaseEstimator],
    X_tr:        pd.DataFrame,
    y_tr:        pd.Series,
    top_n:       int,
    n_iter:      int,
    n_splits:    int,
) -> dict[str, Any]:
    cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    top    = (screen_df.dropna(subset=["roc_auc_mean"])
              .sort_values("roc_auc_mean", ascending=False)
              .head(top_n)["model"].tolist())
    log.info("Top-%d for tuning: %s", top_n, top)

    tuned: dict[str, Any] = {}
    for name in top:
        clf  = clone(classifiers[name])
        pipe = wrap_classifier(clf)

        if name not in PARAM_GRIDS:
            log.info("  [Tune] %-22s — no grid, fitting with defaults.", name)
            pipe.fit(X_tr, y_tr)
            tuned[name] = {"tuned": False, "best_estimator": pipe}
            continue

        log.info("  [Tune] %-22s (n_iter=%d) …", name, n_iter)
        try:
            search = RandomizedSearchCV(
                pipe, PARAM_GRIDS[name],
                n_iter=n_iter, scoring="roc_auc",
                cv=cv, refit=True,
                random_state=RANDOM_STATE, n_jobs=N_JOBS,
                error_score="raise",
            )
            search.fit(X_tr, y_tr)
            tuned[name] = {
                "tuned":           True,
                "best_params":     search.best_params_,
                "best_cv_roc_auc": float(search.best_score_),
                "best_estimator":  search.best_estimator_,
            }
            log.info("    Best CV ROC-AUC after tuning: %.4f", search.best_score_)
        except Exception as exc:
            log.warning("    Tuning failed for %s: %s — using defaults.", name, exc)
            pipe.fit(X_tr, y_tr)
            tuned[name] = {"tuned": False, "best_estimator": pipe}
    return tuned


# ─────────────────────────────────────────────────────────────────────────────
# ── Phase 5: Ensemble ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def build_ensembles(
    tuned:     dict[str, Any],
    X_tr:      pd.DataFrame,
    y_tr:      pd.Series,
    X_te:      pd.DataFrame,
    y_te:      pd.Series,
) -> dict[str, Any]:
    estimators = [(n, info["best_estimator"]) for n, info in tuned.items()]
    if len(estimators) < 2:
        return {}

    results: dict[str, Any] = {}
    for ename, ekwargs in [
        ("SoftVoting",  {"voting": "soft", "n_jobs": N_JOBS}),
        ("Stacking",    {"final_estimator": LogisticRegression(max_iter=2000,
                                             random_state=RANDOM_STATE),
                         "cv": 3, "n_jobs": N_JOBS}),
    ]:
        try:
            log.info("  [Ensemble] Building %s …", ename)
            EClass = VotingClassifier if ename == "SoftVoting" else StackingClassifier
            m = EClass(estimators=estimators, **ekwargs)
            m.fit(X_tr, y_tr)
            results[ename] = {"model": m}
        except Exception as exc:
            log.warning("  %s failed: %s", ename, exc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ── Phase 6: Hold-out test evaluation ────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def youden_threshold(y_true, probs) -> float:
    """Optimal threshold via Youden's J (maximises sensitivity + specificity)."""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def evaluate_on_holdout(
    name:  str,
    model: Any,
    X_te:  pd.DataFrame,
    y_te:  pd.Series,
) -> dict[str, Any]:
    probs = model.predict_proba(X_te)[:, 1]
    thr   = youden_threshold(y_te, probs)
    preds = (probs >= thr).astype(int)
    cm    = confusion_matrix(y_te, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        "model":             name,
        "threshold":         round(thr, 4),
        # ── Core discrimination ───────────────────────────────────────────
        "roc_auc":           round(float(roc_auc_score(y_te, probs)), 4),
        "average_precision": round(float(average_precision_score(y_te, probs)), 4),
        # ── Class-balance–aware ───────────────────────────────────────────
        "f1":                round(float(f1_score(y_te, preds, zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_te, preds)), 4),
        "mcc":               round(float(matthews_corrcoef(y_te, preds)), 4),
        "cohen_kappa":       round(float(cohen_kappa_score(y_te, preds)), 4),
        # ── Per-class ─────────────────────────────────────────────────────
        "precision":         round(float(precision_score(y_te, preds, zero_division=0)), 4),
        "recall":            round(float(recall_score(y_te, preds, zero_division=0)), 4),
        "accuracy":          round(float(accuracy_score(y_te, preds)), 4),
        # ── Calibration ───────────────────────────────────────────────────
        "brier_score":       round(float(brier_score_loss(y_te, probs)), 4),
        # ── Confusion matrix counts ───────────────────────────────────────
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "confusion_matrix":         cm.tolist(),
        "classification_report":    classification_report(
            y_te, preds, output_dict=True, zero_division=0),
        "_probs": probs,   # for plots — stripped before JSON serialisation
    }


# ─────────────────────────────────────────────────────────────────────────────
# ── Phase 8: SHAP for champion ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def compute_shap(model, X_te: pd.DataFrame, output_dir: Path) -> None:
    if not _SHAP:
        log.warning("shap not installed — skipping. pip install shap")
        return
    log.info("Computing SHAP values for champion …")
    try:
        clf  = model.named_steps["model"]
        prep = model.named_steps["preprocess"]
        fe   = model.named_steps["feature_engineering"]
        Xt   = prep.transform(fe.transform(X_te))
        fn   = prep.get_feature_names_out()
        Xdf  = pd.DataFrame(Xt, columns=fn)

        if hasattr(clf, "feature_importances_"):
            exp  = shap.TreeExplainer(clf)
            sv   = exp.shap_values(Xdf)
            if isinstance(sv, list):
                sv = sv[1]
        else:
            masker = shap.maskers.Independent(Xdf, max_samples=100)
            exp    = shap.Explainer(clf.predict_proba, masker)
            sv     = exp(Xdf).values[:, :, 1]

        for ptype, fname in [("bar", "shap_bar.png"), ("dot", "shap_beeswarm.png")]:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(sv, Xdf, plot_type=ptype, show=False, max_display=20)
            plt.title(f"SHAP — Champion ({ptype})")
            plt.tight_layout()
            plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
            plt.close()

        pd.DataFrame({
            "feature": fn,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).to_csv(
            output_dir / "shap_importance.csv", index=False)

        log.info("SHAP artifacts saved.")
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# ── Phase 9: Fairness / subgroup evaluation ───────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_subgroups(
    model:      Any,
    X_te:       pd.DataFrame,
    y_te:       pd.Series,
    threshold:  float,
    output_dir: Path,
) -> list[dict]:
    """Disaggregated metrics for sex, pclass, and age_group (from reference pipeline)."""
    probs      = model.predict_proba(X_te)[:, 1]
    overall_f1 = float(f1_score(y_te, (probs >= threshold).astype(int), zero_division=0))

    eval_df = X_te.reset_index(drop=True).copy()
    eval_df["_y"]    = y_te.to_numpy()
    eval_df["_prob"] = probs
    eval_df["_age_group"] = pd.cut(
        eval_df["age"].astype(float),
        bins=[0, 12, 18, 40, 60, 120],
        labels=["child", "teen", "adult", "middle_aged", "senior"],
        right=False,
    ).astype("category").cat.add_categories("unknown").fillna("unknown")

    rows = []
    for col in FAIRNESS_COLS + ["_age_group"]:
        if col not in eval_df.columns:
            continue
        for val, sub in eval_df.groupby(col, observed=True):
            if len(sub) < 10:
                continue
            sp    = sub["_prob"].to_numpy()
            st    = sub["_y"].to_numpy()
            sf1   = float(f1_score(st, (sp >= threshold).astype(int), zero_division=0))
            sauc  = (float(roc_auc_score(st, sp))
                     if len(np.unique(st)) > 1 else None)
            rows.append({
                "group_col":        col,
                "group_val":        str(val),
                "n":                int(len(sub)),
                "positive_rate":    round(float(st.mean()), 4),
                "f1":               round(sf1, 4),
                "roc_auc":          round(sauc, 4) if sauc else None,
                "f1_gap_vs_overall": round(sf1 - overall_f1, 4),
                "alert":            bool((overall_f1 - sf1) > 0.10),
            })

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "fairness_report.csv", index=False)
        _plot_fairness(pd.DataFrame(rows), output_dir)
        alerts = [r for r in rows if r["alert"]]
        if alerts:
            log.warning(
                "Fairness alert: %d subgroup(s) with F1 gap > 0.10: %s",
                len(alerts),
                [(r["group_col"], r["group_val"],
                  round(r["f1_gap_vs_overall"], 3)) for r in alerts],
            )
    return rows


def _plot_fairness(df: pd.DataFrame, out: Path) -> None:
    df = df.copy()
    df["label"] = df["group_col"] + "=" + df["group_val"].astype(str)
    plt.figure(figsize=(10, max(4, len(df) * 0.45)))
    colors = ["#E45756" if a else "#4C78A8" for a in df["alert"]]
    plt.barh(df["label"], df["f1"], color=colors)
    plt.axvline(df["f1"].mean(), linestyle="--", color="black", label="Mean F1")
    plt.xlabel("F1 Score")
    plt.title("Fairness — Subgroup F1\n(red = F1 gap > 0.10 vs overall)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out / "fairness_subgroup_f1.png", dpi=150); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# ── Plots ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def plot_screening_heatmap(df: pd.DataFrame, out: Path) -> None:
    cols = ["roc_auc_mean","f1_mean","average_precision_mean",
            "balanced_accuracy_mean","precision_mean","recall_mean"]
    d = df.dropna(subset=["roc_auc_mean"]).set_index("model")[cols].copy()
    d.columns = [c.replace("_mean","").replace("_"," ").title() for c in d.columns]
    d = d.sort_values("Roc Auc", ascending=False)
    fig, ax = plt.subplots(figsize=(11, max(5, len(d) * 0.48 + 1.5)))
    sns.heatmap(d, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.5, vmax=1.0, linewidths=0.4, ax=ax, annot_kws={"size": 9})
    ax.set_title("Algorithm Screening — CV Metric Heatmap (sorted by ROC-AUC)",
                 fontsize=13, pad=12)
    plt.tight_layout(); plt.savefig(out / "screening_heatmap.png", dpi=150); plt.close()


def plot_cv_boxplot(df: pd.DataFrame, out: Path, metric: str = "roc_auc") -> None:
    raw = f"_raw_{metric}"
    if raw not in df.columns:
        return
    v     = df.dropna(subset=[f"{metric}_mean"]).sort_values(f"{metric}_mean", ascending=False)
    data  = [np.array(r) for r in v[raw]]
    names = v["model"].tolist()
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.75), 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55)
    cmap = plt.cm.get_cmap("RdYlGn", len(names))
    for i, (patch, med) in enumerate(zip(bp["boxes"], bp["medians"])):
        patch.set_facecolor(cmap(i / len(names)))
        med.set_color("black"); med.set_linewidth(1.5)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=38, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"CV Distribution — {metric.replace('_', ' ').title()}", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out / f"cv_boxplot_{metric}.png", dpi=150); plt.close()


def plot_overfit_gap(df: pd.DataFrame, out: Path) -> None:
    v = df.dropna(subset=["roc_auc_mean"]).sort_values("roc_auc_mean", ascending=False)
    colors = ["#E45756" if g > 0.05 else "#54A24B" for g in v["roc_auc_overfit_gap"]]
    fig, ax = plt.subplots(figsize=(max(10, len(v) * 0.75), 4))
    ax.bar(range(len(v)), v["roc_auc_overfit_gap"], color=colors, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.05, color="#E45756", linewidth=0.8, linestyle="--",
               alpha=0.6, label="Alert threshold 0.05")
    ax.set_xticks(range(len(v)))
    ax.set_xticklabels(v["model"].tolist(), rotation=38, ha="right", fontsize=9)
    ax.set_ylabel("Train ROC-AUC − CV ROC-AUC")
    ax.set_title("Overfitting Analysis — Train–CV ROC-AUC Gap\n(red = possible overfit)",
                 fontsize=12)
    ax.legend(); plt.tight_layout()
    plt.savefig(out / "overfit_gap.png", dpi=150); plt.close()


def _all_models(tuned, ensembles):
    out = {n: i["best_estimator"] for n, i in tuned.items()}
    out.update({n: i["model"] for n, i in ensembles.items()})
    return out


def plot_roc_curves(tuned, ensembles, X_te, y_te, out) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.get_cmap("tab10")
    for i, (name, model) in enumerate(_all_models(tuned, ensembles).items()):
        probs = model.predict_proba(X_te)[:, 1]
        RocCurveDisplay.from_predictions(
            y_te, probs,
            name=f"{name} (AUC={roc_auc_score(y_te, probs):.3f})",
            ax=ax, color=cmap(i % 10))
    ax.set_title("ROC Curves — Tuned Models + Ensembles", fontsize=12)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(out / "roc_curves_all.png", dpi=150); plt.close()


def plot_pr_curves(tuned, ensembles, X_te, y_te, out) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.get_cmap("tab10")
    for i, (name, model) in enumerate(_all_models(tuned, ensembles).items()):
        probs = model.predict_proba(X_te)[:, 1]
        PrecisionRecallDisplay.from_predictions(
            y_te, probs,
            name=f"{name} (AP={average_precision_score(y_te, probs):.3f})",
            ax=ax, color=cmap(i % 10))
    ax.set_title("Precision-Recall Curves — Tuned Models + Ensembles", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout(); plt.savefig(out / "pr_curves_all.png", dpi=150); plt.close()


def plot_calibration(tuned, ensembles, X_te, y_te, out) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    cmap = plt.cm.get_cmap("tab10")
    for i, (name, model) in enumerate(_all_models(tuned, ensembles).items()):
        probs        = model.predict_proba(X_te)[:, 1]
        bs           = brier_score_loss(y_te, probs)
        pt, pp       = calibration_curve(y_te, probs, n_bins=8, strategy="quantile")
        ax.plot(pp, pt, "o-", color=cmap(i % 10), label=f"{name} (Brier={bs:.3f})")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Survival Rate")
    ax.set_title("Calibration Curves — lower Brier = better", fontsize=12)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(out / "calibration_curves.png", dpi=150); plt.close()


def plot_confusion_matrices(test_results: list[dict], out: Path) -> None:
    n    = len(test_results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.8))
    axes = np.array(axes).flatten()
    for i, res in enumerate(test_results):
        sns.heatmap(np.array(res["confusion_matrix"]),
                    annot=True, fmt="d", cbar=False, cmap="Blues",
                    ax=axes[i], annot_kws={"size": 13})
        axes[i].set_title(f"{res['model']}\nAUC={res['roc_auc']}", fontsize=10)
        axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("Actual")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Confusion Matrices — Tuned + Ensemble Models",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_metric_radar(test_results: list[dict], out: Path) -> None:
    metrics = ["roc_auc","f1","precision","recall","balanced_accuracy","average_precision"]
    labels  = ["ROC-AUC","F1","Precision","Recall","Bal. Acc","Avg. Prec"]
    angles  = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    cmap = plt.cm.get_cmap("tab10")
    for i, res in enumerate(test_results):
        vals = [res[m] for m in metrics] + [res[metrics[0]]]
        ax.plot(angles, vals, "o-", linewidth=1.8, color=cmap(i % 10), label=res["model"])
        ax.fill(angles, vals, alpha=0.06, color=cmap(i % 10))
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, size=11)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Metric Radar — Tuned + Ensemble Models", size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "metric_radar.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_stat_tests(stat: dict, out: Path) -> None:
    if not stat or not stat.get("pairwise_vs_champion"):
        return
    df = pd.DataFrame(stat["pairwise_vs_champion"]).dropna(subset=["p_bonferroni"])
    df = df.sort_values("p_bonferroni")
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.48 + 1.5)))
    colors = ["#E45756" if r else "#54A24B" for r in df["significantly_worse"]]
    ax.barh(df["model"], -np.log10(df["p_bonferroni"].clip(1e-10)), color=colors)
    ax.axvline(-np.log10(ALPHA), color="black", linestyle="--",
               label=f"α = {ALPHA} (Bonferroni)")
    ax.set_xlabel("−log₁₀(p Bonferroni)")
    ax.set_title(
        f"Wilcoxon Signed-Rank vs Champion ({stat['champion_model']})\n"
        f"Red = significantly worse at α = {ALPHA} after Bonferroni correction",
        fontsize=11)
    ax.legend(); plt.tight_layout()
    plt.savefig(out / "stat_test_results.png", dpi=150); plt.close()


def plot_leaderboard(test_results: list[dict], champion: str, out: Path) -> None:
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if not k.startswith("_") and k not in
                        ("confusion_matrix","classification_report")}
                       for r in test_results])
    df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.55 + 1.5)))
    x = range(len(df))
    bar_colors = ["#1d9e75" if nm == champion else "#4C78A8" for nm in df["model"]]
    ax.barh(df["model"], df["roc_auc"], color=bar_colors, height=0.55)
    ax.set_xlabel("ROC-AUC (hold-out test)")
    ax.set_title("Model Leaderboard — Hold-out Test ROC-AUC\n(green = champion)",
                 fontsize=12)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["roc_auc"] + 0.002, i,
                f"F1={row['f1']:.3f}  MCC={row['mcc']:.3f}  Brier={row['brier_score']:.3f}",
                va="center", fontsize=8, color="#333")
    ax.set_xlim(0, 1.08)
    plt.tight_layout(); plt.savefig(out / "leaderboard.png", dpi=150); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# ── Utilities ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def to_jsonable(value: Any) -> Any:
    """JSON-safe serialisation — identical to reference pipeline."""
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
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else f
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# ── Phase 10: HTML Report ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def build_html_report(
    screen_df:     pd.DataFrame,
    stat_tests:    dict,
    test_results:  list[dict],
    fairness_rows: list[dict],
    champion:      str,
    n_cv:          int,
    out:           Path,
) -> None:
    def _chip(v: float, lo: float = 0.78, hi: float = 0.82) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        bg = "#c8f5c8" if v >= hi else ("#fff3c8" if v >= lo else "#f5c8c8")
        return (f'<span style="background:{bg};padding:2px 7px;'
                f'border-radius:4px;font-weight:600">{v:.3f}</span>')

    def screen_row(r) -> str:
        gap  = r.get("roc_auc_overfit_gap", float("nan"))
        gcol = "#c0392b" if (not np.isnan(gap) and gap > 0.05) else "#27ae60"
        chk  = "🏆 " if r["model"] == champion else ""
        return (
            f"<tr>"
            f"<td><b>{chk}{r['model']}</b></td>"
            f"<td>{_chip(r.get('roc_auc_mean', float('nan')))}</td>"
            f"<td>{_chip(r.get('f1_mean', float('nan')))}</td>"
            f"<td>{_chip(r.get('average_precision_mean', float('nan')))}</td>"
            f"<td>{_chip(r.get('balanced_accuracy_mean', float('nan')))}</td>"
            f"<td style='color:{gcol};font-weight:600'>"
            f"{'—' if np.isnan(gap) else f'{gap:.3f}'}</td>"
            f"<td>{r.get('cv_time_s', '—')}</td>"
            f"</tr>"
        )

    def test_row(r) -> str:
        chk = "🏆 " if r["model"] == champion else ""
        return (
            f"<tr>"
            f"<td><b>{chk}{r['model']}</b></td>"
            f"<td>{r['roc_auc']}</td><td>{r['average_precision']}</td>"
            f"<td>{r['f1']}</td><td>{r['precision']}</td><td>{r['recall']}</td>"
            f"<td>{r['balanced_accuracy']}</td><td>{r['brier_score']}</td>"
            f"<td>{r['mcc']}</td><td>{r['cohen_kappa']}</td>"
            f"<td>{r['TP']}</td><td>{r['TN']}</td>"
            f"<td>{r['FP']}</td><td>{r['FN']}</td>"
            f"<td>{r['threshold']}</td>"
            f"</tr>"
        )

    s_rows = "\n".join(
        screen_row(r)
        for _, r in screen_df.dropna(subset=["roc_auc_mean"])
        .sort_values("roc_auc_mean", ascending=False).iterrows()
    )
    t_rows = "\n".join(
        test_row(r)
        for r in sorted(test_results, key=lambda x: x["roc_auc"], reverse=True)
    )

    # Statistical section
    stat_html = ""
    if stat_tests:
        fr  = stat_tests.get("friedman", {})
        sig = ('<span style="color:#c0392b;font-weight:bold">Significant</span>'
               if fr.get("significant") else
               '<span style="color:#27ae60">Not significant</span>')
        stat_html = f"""
        <h2>Phase 3 — Statistical Significance</h2>
        <p class="note">Friedman χ² test (global H₀: all models perform equally) +
        pairwise Wilcoxon signed-rank vs champion with Bonferroni correction (α = {ALPHA}).</p>
        <p><b>Friedman:</b> χ² = {fr.get('statistic', '—')}, p = {fr.get('p_value', '—')},
        result = {sig}</p>
        <p>Champion: <b>{stat_tests['champion_model']}</b></p>
        <img src="stat_test_results.png">
        """

    # Fairness section
    fair_html = ""
    if fairness_rows:
        alerts   = [r for r in fairness_rows if r["alert"]]
        alert_p  = (f'<p class="alert">⚠ {len(alerts)} subgroup(s) have F1 gap &gt; 0.10'
                    f'</p>' if alerts else
                    '<p style="color:#27ae60">✓ No fairness alerts</p>')
        f_rows_h = "\n".join(
            f"<tr><td>{r['group_col']}</td><td>{r['group_val']}</td>"
            f"<td>{r['n']}</td><td>{r['f1']}</td><td>{r['roc_auc']}</td>"
            f"<td style='color:{'#c0392b' if r['alert'] else '#27ae60'};"
            f"font-weight:600'>{r['f1_gap_vs_overall']:+.4f}</td></tr>"
            for r in fairness_rows
        )
        fair_html = f"""
        <h2>Phase 9 — Fairness / Subgroup Evaluation (Champion)</h2>
        <p class="note">Disaggregated metrics for sex, pclass, and age_group.
        F1 gap = subgroup F1 − overall F1. Alert when gap &lt; −0.10.</p>
        {alert_p}
        <table>
          <tr><th>Group</th><th>Value</th><th>N</th><th>F1</th>
              <th>ROC-AUC</th><th>F1 Gap</th></tr>
          {f_rows_h}
        </table>
        <img src="fairness_subgroup_f1.png">
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Titanic ML — Algorithm Benchmark Report</title>
<style>
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f5f4f0;
       color:#1a1a18;padding:28px 36px 64px;max-width:1150px;margin:auto}}
  h1{{font-size:26px;font-weight:700;border-bottom:3px solid #1d9e75;
     padding-bottom:10px;margin-bottom:6px}}
  h2{{font-size:17px;font-weight:600;color:#0f5c3a;margin:36px 0 10px}}
  p{{font-size:13px;color:#555;margin:4px 0 10px;line-height:1.65}}
  .meta{{font-size:12px;color:#888;margin-bottom:28px}}
  .note{{font-size:12px;color:#666;background:#fff;border-left:3px solid #1d9e75;
         padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:12px}}
  .alert{{color:#c0392b;font-weight:600;margin-bottom:8px}}
  table{{border-collapse:collapse;width:100%;font-size:12px;margin-bottom:20px}}
  th{{background:#0f5c3a;color:#fff;padding:8px 10px;text-align:left}}
  td{{padding:6px 10px;border-bottom:0.5px solid #d3d1c7}}
  tr:nth-child(even) td{{background:#f0efe9}}
  img{{display:block;margin:14px 0 28px;border:0.5px solid #ccc;
      border-radius:8px;max-width:100%;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
  footer{{margin-top:48px;font-size:11px;color:#aaa;
          border-top:0.5px solid #ddd;padding-top:12px;line-height:1.8}}
</style>
</head>
<body>

<h1>Titanic Survival — Algorithm Benchmark Report</h1>
<p class="meta">
  Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
  &nbsp;|&nbsp; Champion: <b>{champion}</b>
  &nbsp;|&nbsp; Data: OpenML Titanic v1 (fetch_openml)
  &nbsp;|&nbsp; Reference pipeline: titanic-ml-pipeline.py
</p>

<h2>Phase 1 — EDA (train-set only)</h2>
<p class="note">All EDA computed on the training split only — no test-set leakage.
See eda_*.csv / eda_*.png files.</p>
<img src="eda_survival_by_sex.png" style="max-width:500px">
<img src="eda_age_distribution.png" style="max-width:600px">

<h2>Phase 2 — Algorithm Screening ({n_cv}-fold Stratified CV)</h2>
<p class="note">All 15 classifiers share identical feature engineering and preprocessing
from the reference pipeline. Only the final estimator changes.
Overfit gap = Train AUC − CV AUC; values &gt; 0.05 shown in red.</p>
<table>
  <tr><th>Model</th><th>ROC-AUC</th><th>F1</th><th>Avg Precision</th>
      <th>Bal. Accuracy</th><th>Overfit Gap</th><th>CV Time (s)</th></tr>
  {s_rows}
</table>
<img src="screening_heatmap.png">
<img src="cv_boxplot_roc_auc.png">
<img src="cv_boxplot_f1.png">
<img src="overfit_gap.png">

{stat_html}

<h2>Phase 4 — Hyperparameter Tuning (Top {TOP_N_TUNE} models)</h2>
<p class="note">RandomizedSearchCV with {TUNE_N_ITER} iterations per model on the same
{n_cv}-fold CV. Only the top-{TOP_N_TUNE} screening models are tuned — cost-efficient
industry practice.</p>

<h2>Phase 5 — Ensemble Construction</h2>
<p class="note">SoftVoting and Stacking (meta-learner = LogisticRegression) built
from the tuned top-{TOP_N_TUNE} estimators.</p>

<h2>Phase 6 — Hold-out Test Evaluation</h2>
<p class="note">Threshold per model via Youden's J (maximises sensitivity + specificity).
10 metrics: ROC-AUC (discrimination), Avg Precision (imbalance-robust), F1 (harm-balanced),
Brier ↓ (calibration), MCC (class-imbalance robust), Cohen's κ (chance-corrected).</p>
<table>
  <tr><th>Model</th><th>ROC-AUC</th><th>Avg Prec</th><th>F1</th>
      <th>Precision</th><th>Recall</th><th>Bal Acc</th><th>Brier↓</th>
      <th>MCC</th><th>Kappa</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th>
      <th>Threshold</th></tr>
  {t_rows}
</table>
<img src="leaderboard.png">
<img src="roc_curves_all.png">
<img src="pr_curves_all.png">

<h2>Phase 7 — Calibration Analysis</h2>
<p class="note">Reliability diagrams. Lower Brier score = better calibrated predictions.</p>
<img src="calibration_curves.png">
<img src="confusion_matrices.png">
<img src="metric_radar.png">

<h2>Phase 8 — SHAP Explainability (Champion)</h2>
<p class="note">Global feature importance via SHAP. Bar = mean |SHAP value|.
Beeswarm = feature value direction (red = high, blue = low).</p>
<img src="shap_bar.png">
<img src="shap_beeswarm.png">

{fair_html}

<footer>
  <b>Metric guide:</b><br>
  ROC-AUC — discrimination across all thresholds, threshold-independent.<br>
  Average Precision — area under P-R curve, robust to class imbalance.<br>
  F1 — harmonic mean of precision and recall, balance between both error types.<br>
  Brier score — mean squared error of probabilities (lower = better; 0 = perfect).<br>
  MCC (Matthews) — correlation between predicted and actual; robust to imbalance.<br>
  Cohen's κ — agreement corrected for chance.<br>
  <br>
  <b>Statistical tests:</b> Friedman χ² (global H₀: all models equal) +
  pairwise Wilcoxon signed-rank vs champion with Bonferroni correction (α = {ALPHA}).<br>
  <b>Threshold selection:</b> Youden's J statistic (maximises sensitivity + specificity).
</footer>
</body></html>"""

    path = out / "benchmark_report.html"
    path.write_text(html, encoding="utf-8")
    log.info("HTML report: %s", path.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# ── Main orchestrator ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(output_dir: Path, quick: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    t_start = time.perf_counter()

    n_splits = 3 if quick else N_CV_SPLITS
    n_iter   = 10 if quick else TUNE_N_ITER
    top_n    = 2 if quick else TOP_N_TUNE

    # ── Load data (from reference pipeline) ──────────────────────────────────
    df           = fix_data_types(load_data())
    X_tr, X_te, y_tr, y_te = split_data(df)

    # ── Phase 1: EDA ──────────────────────────────────────────────────────────
    log.info("═══ Phase 1: EDA ═══")
    save_research_artifacts(X_tr, y_tr, output_dir)

    # ── Phase 2: Screening ────────────────────────────────────────────────────
    classifiers = get_classifiers(quick)
    log.info("═══ Phase 2: Screening %d classifiers (%d-fold CV) ═══",
             len(classifiers), n_splits)
    screen_df = screen_classifiers(X_tr, y_tr, classifiers, n_splits)
    (screen_df
     .drop(columns=[c for c in screen_df.columns if c.startswith("_")], errors="ignore")
     .sort_values("roc_auc_mean", ascending=False)
     .to_csv(output_dir / "screening_results.csv", index=False))

    # ── Phase 3: Statistical tests ────────────────────────────────────────────
    log.info("═══ Phase 3: Statistical significance tests ═══")
    stat_tests = run_stat_tests(screen_df)
    write_json(output_dir / "stat_tests.json", stat_tests)

    # ── Phase 4: Tuning ───────────────────────────────────────────────────────
    log.info("═══ Phase 4: Hyperparameter tuning (top-%d) ═══", top_n)
    tuned = tune_top_models(screen_df, classifiers, X_tr, y_tr, top_n, n_iter, n_splits)

    # ── Phase 5: Ensembles ────────────────────────────────────────────────────
    log.info("═══ Phase 5: Ensemble construction ═══")
    ensembles = build_ensembles(tuned, X_tr, y_tr, X_te, y_te)

    # ── Phase 6: Hold-out evaluation ─────────────────────────────────────────
    log.info("═══ Phase 6: Hold-out evaluation ═══")
    all_m        = _all_models(tuned, ensembles)
    test_results = []
    for name, model in all_m.items():
        r = evaluate_on_holdout(name, model, X_te, y_te)
        test_results.append(r)
        log.info("  %-22s  AUC=%.4f  F1=%.4f  Brier=%.4f  MCC=%.4f",
                 name, r["roc_auc"], r["f1"], r["brier_score"], r["mcc"])

    champion_res = max(test_results, key=lambda r: r["roc_auc"])
    champion     = champion_res["model"]
    log.info("Champion: %s (AUC=%.4f)", champion, champion_res["roc_auc"])

    # Save champion model (versioned filename, identical to reference pipeline)
    champion_model = all_m[champion]
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha1 = hashlib.sha1(pickle.dumps(champion_model)).hexdigest()[:8]
    joblib.dump(champion_model, output_dir / f"champion_{champion}_{ts}_{sha1}.joblib")
    joblib.dump(champion_model, output_dir / "champion_model.joblib")

    # ── Phase 7: Calibration plots ────────────────────────────────────────────
    log.info("═══ Phase 7: Calibration ═══")
    plot_calibration(tuned, ensembles, X_te, y_te, output_dir)

    # ── Phase 8: SHAP ─────────────────────────────────────────────────────────
    log.info("═══ Phase 8: SHAP ═══")
    compute_shap(champion_model, X_te, output_dir)

    # ── Phase 9: Fairness ─────────────────────────────────────────────────────
    log.info("═══ Phase 9: Fairness ═══")
    fairness_rows = evaluate_subgroups(
        champion_model, X_te, y_te, champion_res["threshold"], output_dir
    )

    # ── All remaining plots ───────────────────────────────────────────────────
    log.info("═══ Generating plots ═══")
    plot_screening_heatmap(screen_df, output_dir)
    plot_cv_boxplot(screen_df, output_dir, "roc_auc")
    plot_cv_boxplot(screen_df, output_dir, "f1")
    plot_overfit_gap(screen_df, output_dir)
    plot_roc_curves(tuned, ensembles, X_te, y_te, output_dir)
    plot_pr_curves(tuned, ensembles, X_te, y_te, output_dir)
    plot_confusion_matrices(test_results, output_dir)
    plot_metric_radar(test_results, output_dir)
    plot_stat_tests(stat_tests, output_dir)
    plot_leaderboard(test_results, champion, output_dir)

    # ── Strip _probs before serialisation ────────────────────────────────────
    for r in test_results:
        r.pop("_probs", None)

    # ── JSON report ───────────────────────────────────────────────────────────
    report = {
        "generated_at":            datetime.now(timezone.utc).isoformat(),
        "champion":                champion,
        "elapsed_seconds":         round(time.perf_counter() - t_start, 1),
        "data_source":             "fetch_openml('titanic', version=1)",
        "n_classifiers_screened":  len(classifiers),
        "cv_splits":               n_splits,
        "screening_summary": (
            screen_df
            .drop(columns=[c for c in screen_df.columns if c.startswith("_")], errors="ignore")
            .to_dict(orient="records")
        ),
        "stat_tests":              stat_tests,
        "tuning_summary": {
            n: {k: v for k, v in i.items() if k != "best_estimator"}
            for n, i in tuned.items()
        },
        "test_results":            test_results,
        "fairness":                fairness_rows,
        "champion_test_metrics":   champion_res,
    }
    write_json(output_dir / "benchmark_report.json", report)

    # ── HTML report ───────────────────────────────────────────────────────────
    build_html_report(
        screen_df, stat_tests, test_results,
        fairness_rows, champion, n_splits, output_dir,
    )

    log.info(
        "═══ Benchmark complete in %.1fs — Champion: %s  AUC=%.4f ═══",
        time.perf_counter() - t_start, champion, champion_res["roc_auc"],
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# ── CLI ───────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description="Titanic supervised learning benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=Path("benchmark_out"),
                   help="Directory for all outputs")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke-test: 3-fold CV, 10 tune iters, top-2 models")
    args = p.parse_args()
    run_benchmark(args.output_dir, quick=args.quick)


if __name__ == "__main__":
    main()