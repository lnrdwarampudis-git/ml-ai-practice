"""
titanic_compare.py — Professional supervised-learning algorithm comparison.

How professionals do it:
  1. Define a catalogue of algorithms with realistic search spaces
  2. Run RandomizedSearchCV with StratifiedKFold on TRAIN data only
  3. Collect CV + held-out test metrics for every algorithm
  4. Rank by primary metric (ROC-AUC); flag statistical significance
  5. Run calibration, feature importance, and SHAP for every finalist
  6. Produce a single comparison report (JSON + CSV + plots)
  7. Auto-select the best model; save it as the production artifact

Algorithms compared:
  - Logistic Regression        (linear baseline)
  - Ridge Classifier           (regularised linear)
  - K-Nearest Neighbours       (instance-based)
  - Naive Bayes                (probabilistic)
  - Decision Tree              (single tree)
  - Random Forest              (bagging ensemble)
  - Extra Trees                (randomised bagging)
  - Gradient Boosting (sklearn)(boosting)
  - XGBoost                    (boosting + regularisation)
  - LightGBM                   (fast boosting)
  - SVM (RBF kernel)           (margin-based)
  - MLP Neural Network         (simple deep learning)

Usage:
    python titanic_compare.py compare --output-dir results/comparison
    python titanic_compare.py compare --output-dir results/comparison --quick   # fast demo
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path("artifacts") / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml

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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TARGET       = "survived"
N_JOBS       = int(os.environ.get("ML_N_JOBS", 1))
CV_FOLDS     = 5
PRIMARY_METRIC = "roc_auc"

LEAKAGE_COLUMNS        = ["boat", "body"]
RAW_TEXT_COLUMNS       = ["name", "ticket"]
HIGH_MISSING_RAW_COLS  = ["cabin", "home.dest"]


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering (same as titanic.py — kept self-contained)
# ══════════════════════════════════════════════════════════════════════════════
import re
from sklearn.base import TransformerMixin

class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, rare_title_min_count: int = 10):
        self.rare_title_min_count = rare_title_min_count

    def fit(self, X, y=None):
        titles = self._extract_title(X.get("name", pd.Series(index=X.index, dtype="object")))
        counts = titles.value_counts(dropna=False)
        self.rare_titles_ = set(counts[counts < self.rare_title_min_count].index)
        return self

    def transform(self, X):
        X = X.copy()
        titles   = self._extract_title(X.get("name", pd.Series(index=X.index, dtype="object")))
        X["title"] = titles.where(~titles.isin(self.rare_titles_), "Rare")
        cabin = X.get("cabin", pd.Series(index=X.index, dtype="object")).astype("string")
        X["has_cabin"]    = cabin.notna().astype(int)
        X["cabin_deck"]   = cabin.str[0].fillna("Unknown")
        X["cabin_count"]  = cabin.fillna("").str.split().map(lambda v: len([x for x in v if x]))
        ticket = X.get("ticket", pd.Series(index=X.index, dtype="object")).astype("string")
        X["ticket_prefix"] = ticket.map(self._ticket_prefix)
        X["family_size"]   = X["sibsp"].fillna(0) + X["parch"].fillna(0) + 1
        X["is_alone"]      = (X["family_size"] == 1).astype(int)
        X["fare_per_person"] = X["fare"] / X["family_size"].replace(0, np.nan)
        X["home_dest_known"] = X.get("home.dest", pd.Series(index=X.index)).notna().astype(int)
        drop = LEAKAGE_COLUMNS + RAW_TEXT_COLUMNS + HIGH_MISSING_RAW_COLS
        return X.drop(columns=[c for c in drop if c in X.columns])

    @staticmethod
    def _extract_title(names):
        titles = names.astype("string").str.extract(r",\s*([^\.]+)\.", expand=False)
        return titles.fillna("Unknown").replace({
            "Mlle": "Miss","Ms": "Miss","Mme": "Mrs",
            "Lady": "Nobility","Sir": "Nobility","the Countess": "Nobility",
            "Dona": "Nobility","Don": "Nobility","Jonkheer": "Nobility",
            "Capt": "Officer","Col": "Officer","Major": "Officer",
            "Dr": "Officer","Rev": "Officer",
        })

    @staticmethod
    def _ticket_prefix(ticket):
        if pd.isna(ticket): return "MISSING_TICKET"
        cleaned = re.sub(r"[\.\/]", " ", str(ticket)).strip()
        prefix  = re.sub(r"\d+", "", cleaned).strip().upper()
        prefix  = re.sub(r"\s+", "_", prefix)
        return prefix if prefix else "NO_PREFIX"


NUMERIC_COLS     = ["age","sibsp","parch","fare","has_cabin","cabin_count",
                    "family_size","is_alone","fare_per_person","home_dest_known"]
CATEGORICAL_COLS = ["pclass","sex","embarked","title","cabin_deck","ticket_prefix"]


def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_COLS),
        ("cat", cat_pipe, CATEGORICAL_COLS),
    ])


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Wrap any estimator in the standard feature-engineering + preprocessing pipeline."""
    selector_base = ExtraTreesClassifier(
        n_estimators=200, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=N_JOBS,
    )
    return Pipeline([
        ("feature_engineering", TitanicFeatureEngineer()),
        ("preprocess",          build_preprocessor()),
        ("feature_selection",   SelectFromModel(selector_base, threshold="median")),
        ("model",               model),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm catalogue
# ══════════════════════════════════════════════════════════════════════════════

def get_algorithm_catalogue(quick: bool = False) -> dict[str, dict]:
    """
    Returns {algo_name: {"estimator": ..., "param_dist": {...}, "n_iter": int}}.

    quick=True uses fewer iterations and smaller search spaces for rapid testing.
    """
    n = 5 if quick else 25       # CV iterations per algorithm
    n_est = [100] if quick else [200, 400, 600]

    catalogue: dict[str, dict] = {

        # ── Linear / regularised ─────────────────────────────────────────────
        "LogisticRegression": {
            "estimator": LogisticRegression(
                max_iter=3000, class_weight="balanced",
                solver="liblinear", random_state=RANDOM_STATE,
            ),
            "param_dist": {
                "model__C":       [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
                "model__penalty": ["l1", "l2"],
            },
            "n_iter": n,
            "family": "Linear",
        },

        "RidgeClassifier": {
            "estimator": CalibratedClassifierCV(
                RidgeClassifier(class_weight="balanced"), cv=3, method="sigmoid"
            ),
            "param_dist": {
                "model__estimator__alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
            },
            "n_iter": n,
            "family": "Linear",
        },

        # ── Instance-based ───────────────────────────────────────────────────
        "KNN": {
            "estimator": KNeighborsClassifier(),
            "param_dist": {
                "model__n_neighbors": list(range(3, 31, 2)),
                "model__weights":     ["uniform", "distance"],
                "model__metric":      ["euclidean", "manhattan", "minkowski"],
            },
            "n_iter": n,
            "family": "Instance",
        },

        # ── Probabilistic ────────────────────────────────────────────────────
        "NaiveBayes": {
            "estimator": GaussianNB(),
            "param_dist": {
                "model__var_smoothing": np.logspace(-12, -1, 20).tolist(),
            },
            "n_iter": n,
            "family": "Probabilistic",
        },

        # ── Tree-based ───────────────────────────────────────────────────────
        "DecisionTree": {
            "estimator": DecisionTreeClassifier(
                class_weight="balanced", random_state=RANDOM_STATE,
            ),
            "param_dist": {
                "model__max_depth":        [3, 5, 7, 10, None],
                "model__min_samples_leaf": [1, 2, 5, 10, 20],
                "model__criterion":        ["gini", "entropy"],
            },
            "n_iter": n,
            "family": "Tree",
        },

        "RandomForest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE, n_jobs=N_JOBS,
            ),
            "param_dist": {
                "model__n_estimators":     n_est,
                "model__max_depth":        [4, 6, 8, None],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features":     ["sqrt", "log2", 0.5],
            },
            "n_iter": n,
            "family": "Ensemble",
        },

        "ExtraTrees": {
            "estimator": ExtraTreesClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=N_JOBS,
            ),
            "param_dist": {
                "model__n_estimators":     n_est,
                "model__max_depth":        [4, 6, 8, None],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features":     ["sqrt", "log2"],
            },
            "n_iter": n,
            "family": "Ensemble",
        },

        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_dist": {
                "model__n_estimators":   [100, 200, 300] if not quick else [100],
                "model__learning_rate":  [0.01, 0.05, 0.1, 0.2],
                "model__max_depth":      [2, 3, 4, 5],
                "model__subsample":      [0.6, 0.8, 1.0],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "n_iter": n,
            "family": "Ensemble",
        },

        # ── SVM ──────────────────────────────────────────────────────────────
        "SVM_RBF": {
            "estimator": SVC(
                kernel="rbf", class_weight="balanced",
                probability=True, random_state=RANDOM_STATE,
            ),
            "param_dist": {
                "model__C":     [0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
                "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            },
            "n_iter": n,
            "family": "SVM",
        },

        # ── Neural network ───────────────────────────────────────────────────
        "MLP": {
            "estimator": MLPClassifier(
                max_iter=500, random_state=RANDOM_STATE, early_stopping=True,
            ),
            "param_dist": {
                "model__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (64, 64, 32)],
                "model__alpha":              [1e-4, 1e-3, 1e-2, 0.1],
                "model__learning_rate_init": [1e-4, 1e-3, 5e-3],
                "model__activation":         ["relu", "tanh"],
            },
            "n_iter": n,
            "family": "Neural Net",
        },
    }

    # ── Optional boosting libraries ──────────────────────────────────────────
    if _XGB:
        catalogue["XGBoost"] = {
            "estimator": XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                verbosity=0,
            ),
            "param_dist": {
                "model__n_estimators":  [100, 200, 300] if not quick else [100],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth":     [3, 4, 5, 6],
                "model__subsample":     [0.6, 0.8, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
                "model__scale_pos_weight": [1, 2, 3],
            },
            "n_iter": n,
            "family": "Ensemble",
        }

    if _LGB:
        catalogue["LightGBM"] = {
            "estimator": LGBMClassifier(
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                verbose=-1,
                class_weight="balanced",
            ),
            "param_dist": {
                "model__n_estimators":   [100, 200, 300] if not quick else [100],
                "model__learning_rate":  [0.01, 0.05, 0.1, 0.2],
                "model__max_depth":      [-1, 4, 6, 8],
                "model__num_leaves":     [15, 31, 63, 127],
                "model__subsample":      [0.6, 0.8, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
            },
            "n_iter": n,
            "family": "Ensemble",
        }

    return catalogue


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_and_split():
    """
    Load Titanic data.  Tries seaborn's bundled dataset first (no network needed),
    then falls back to OpenML if seaborn is unavailable.
    Normalises column names so the feature engineer works identically.
    """
    log.info("Loading Titanic dataset …")
    try:
        import seaborn as sns
        raw = sns.load_dataset("titanic")
        # seaborn has different column names — map to the OpenML schema expected
        # by TitanicFeatureEngineer
        df = pd.DataFrame()
        df["survived"] = raw["survived"].astype(int)
        df["pclass"]   = raw["pclass"].astype("category")
        df["sex"]      = raw["sex"].astype("category")
        df["age"]      = raw["age"].astype(float)
        df["sibsp"]    = raw["sibsp"].astype(float)
        df["parch"]    = raw["parch"].astype(float)
        df["fare"]     = raw["fare"].astype(float)
        df["embarked"] = raw["embarked"].astype("category")
        # seaborn has 'deck' — map it as 'cabin' prefix (single letter deck)
        df["cabin"]    = raw["deck"].astype("object")
        # no 'name', 'ticket', 'boat', 'body', 'home.dest' in seaborn version
        # Feature engineer handles missing columns gracefully
        log.info("Loaded via seaborn (%d rows)", len(df))
    except Exception:
        from sklearn.datasets import fetch_openml
        raw = fetch_openml("titanic", version=1, as_frame=True, parser="auto").frame.copy()
        df = raw.copy()
        df["survived"]  = df["survived"].astype(int)
        df["pclass"]    = df["pclass"].astype("category")
        df["sex"]       = df["sex"].astype("category")
        df["embarked"]  = df["embarked"].astype("category")
        log.info("Loaded via OpenML (%d rows)", len(df))

    df = df.dropna(subset=["survived"]).copy()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


# ══════════════════════════════════════════════════════════════════════════════
# Per-algorithm tuning
# ══════════════════════════════════════════════════════════════════════════════

def tune_one(
    name:       str,
    algo_cfg:   dict,
    X_train:    pd.DataFrame,
    y_train:    pd.Series,
    cv:         StratifiedKFold,
) -> tuple[Pipeline, RandomizedSearchCV]:
    log.info("  Tuning %-20s …", name)
    pipeline = build_pipeline(clone(algo_cfg["estimator"]))
    search   = RandomizedSearchCV(
        pipeline,
        param_distributions = algo_cfg["param_dist"],
        n_iter              = algo_cfg["n_iter"],
        scoring             = {
            "roc_auc":           "roc_auc",
            "average_precision": "average_precision",
            "f1":                "f1",
            "balanced_accuracy": "balanced_accuracy",
        },
        refit         = PRIMARY_METRIC,
        cv            = cv,
        random_state  = RANDOM_STATE,
        n_jobs        = N_JOBS,
        verbose       = 0,
        return_train_score = True,
        error_score   = 0.0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search


# ══════════════════════════════════════════════════════════════════════════════
# OOF threshold tuning
# ══════════════════════════════════════════════════════════════════════════════

def tune_threshold_oof(
    estimator: Pipeline,
    X_train:   pd.DataFrame,
    y_train:   pd.Series,
    cv:        StratifiedKFold,
) -> float:
    try:
        oof = cross_val_predict(
            clone(estimator), X_train, y_train,
            cv=cv, method="predict_proba", n_jobs=N_JOBS,
        )[:, 1]
    except Exception:
        return 0.5
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_train, (oof >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(f1s))])


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true, probs, threshold: float) -> dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    return {
        "threshold":          round(float(threshold), 4),
        "roc_auc":            round(float(roc_auc_score(y_true, probs)), 4),
        "average_precision":  round(float(average_precision_score(y_true, probs)), 4),
        "f1":                 round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "precision":          round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall":             round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "balanced_accuracy":  round(float(balanced_accuracy_score(y_true, preds)), 4),
        "accuracy":           round(float(accuracy_score(y_true, preds)), 4),
        "brier_score":        round(float(brier_score_loss(y_true, probs)), 4),
        "confusion_matrix":   confusion_matrix(y_true, preds).tolist(),
        "classification_report": classification_report(y_true, preds, output_dict=True, zero_division=0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Statistical significance: pairwise Wilcoxon on CV fold scores
# ══════════════════════════════════════════════════════════════════════════════

def significance_matrix(
    cv_scores: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank test on per-fold ROC-AUC scores.
    Returns a DataFrame where True = statistically significant difference (p < alpha).
    """
    names  = list(cv_scores.keys())
    n      = len(names)
    pvals  = pd.DataFrame(np.ones((n, n)), index=names, columns=names)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cv_scores[names[i]], cv_scores[names[j]]
            if np.allclose(a, b):
                p = 1.0
            else:
                try:
                    _, p = wilcoxon(a, b)
                except Exception:
                    p = 1.0
            pvals.iloc[i, j] = p
            pvals.iloc[j, i] = p
    return pvals


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def _palette(n: int) -> list[str]:
    return sns.color_palette("tab10", n)


def plot_cv_comparison(results: list[dict], output_dir: Path) -> None:
    """Box plot of per-fold CV ROC-AUC scores for every algorithm."""
    sns.set_theme(style="whitegrid")
    names  = [r["name"] for r in results]
    scores = [r["cv_fold_scores"] for r in results]
    order  = sorted(range(len(names)), key=lambda i: np.median(scores[i]), reverse=True)
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.9), 5))
    bp = ax.boxplot(
        [scores[i] for i in order],
        labels=[names[i] for i in order],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    palette = _palette(len(names))
    for patch, color in zip(bp["boxes"], [palette[i % 10] for i in range(len(names))]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("CV ROC-AUC (5-fold)")
    ax.set_title("Cross-validation ROC-AUC by Algorithm")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_cv_comparison.png", dpi=160)
    plt.close()


def plot_metric_heatmap(results: list[dict], output_dir: Path) -> None:
    """Heatmap of all test metrics across algorithms."""
    metrics = ["roc_auc", "f1", "precision", "recall",
               "average_precision", "balanced_accuracy", "brier_score"]
    rows = []
    for r in results:
        row = {"Algorithm": r["name"]}
        for m in metrics:
            row[m] = r["test"][m]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Algorithm")
    # Brier score: lower is better — invert for heatmap
    df["brier_score"] = 1 - df["brier_score"]

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 1.4), max(5, len(results) * 0.55)))
    sns.heatmap(
        df,
        annot=True, fmt=".3f",
        cmap="RdYlGn",
        vmin=0.55, vmax=0.95,
        linewidths=0.4,
        ax=ax,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Test metrics heatmap (brier_score shown as 1-brier)")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_metric_heatmap.png", dpi=160)
    plt.close()


def plot_roc_all(results: list[dict], X_test, y_test, output_dir: Path) -> None:
    """Overlaid ROC curves for all algorithms."""
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = _palette(len(results))
    for r, color in zip(results, palette):
        fpr, tpr, _ = roc_curve(y_test, r["test_probs"])
        ax.plot(fpr, tpr, lw=1.5, color=color,
                label=f'{r["name"]} ({r["test"]["roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Algorithms")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_roc_all.png", dpi=160)
    plt.close()


def plot_pr_all(results: list[dict], y_test, output_dir: Path) -> None:
    """Overlaid Precision-Recall curves."""
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = _palette(len(results))
    baseline = y_test.mean()
    for r, color in zip(results, palette):
        prec, rec, _ = precision_recall_curve(y_test, r["test_probs"])
        ax.plot(rec, prec, lw=1.5, color=color,
                label=f'{r["name"]} (AP={r["test"]["average_precision"]:.3f})')
    ax.axhline(baseline, color="gray", linestyle="--", lw=0.8, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Algorithms")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_pr_all.png", dpi=160)
    plt.close()


def plot_calibration_all(results: list[dict], y_test, output_dir: Path) -> None:
    """Calibration curves for all algorithms."""
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = _palette(len(results))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
    for r, color in zip(results, palette):
        try:
            prob_true, prob_pred = calibration_curve(
                y_test, r["test_probs"], n_bins=8, strategy="quantile"
            )
            ax.plot(prob_pred, prob_true, marker="o", ms=4, lw=1.5, color=color,
                    label=f'{r["name"]} (brier={r["test"]["brier_score"]:.3f})')
        except Exception:
            pass
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves — All Algorithms")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_calibration_all.png", dpi=160)
    plt.close()


def plot_significance_heatmap(pval_df: pd.DataFrame, output_dir: Path) -> None:
    """Significance matrix: green = significantly different, red = not."""
    fig, ax = plt.subplots(figsize=(max(7, len(pval_df) * 0.75), max(6, len(pval_df) * 0.65)))
    sig = (pval_df < 0.05).astype(float)
    sns.heatmap(sig, annot=pval_df.round(3), fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Significant (green=yes)"})
    ax.set_title("Pairwise Wilcoxon p-values (green = p < 0.05)")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_significance_matrix.png", dpi=160)
    plt.close()


def plot_ranked_bar(results: list[dict], output_dir: Path) -> None:
    """Horizontal bar chart ranked by test ROC-AUC with CV spread."""
    sorted_r = sorted(results, key=lambda r: r["test"]["roc_auc"])
    names    = [r["name"] for r in sorted_r]
    aucs     = [r["test"]["roc_auc"] for r in sorted_r]
    cv_mean  = [np.mean(r["cv_fold_scores"]) for r in sorted_r]
    cv_std   = [np.std(r["cv_fold_scores"]) for r in sorted_r]
    family   = [r["family"] for r in sorted_r]
    fam_map  = {f: c for f, c in zip(sorted(set(family)), _palette(len(set(family))))}

    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.52)))
    colors  = [fam_map[f] for f in family]
    bars    = ax.barh(names, aucs, color=colors, alpha=0.8, height=0.6)
    # CV dots
    ax.scatter(cv_mean, names, color="black", zorder=5, s=25, label="CV mean")
    ax.errorbar(cv_mean, names, xerr=cv_std, fmt="none", ecolor="black",
                elinewidth=1, capsize=3)

    ax.axvline(0.5, color="gray", linestyle="--", lw=0.8, label="Random baseline")
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Algorithm ranking — test ROC-AUC (dot = CV mean ± std)")
    ax.set_xlim(0.45, 1.0)

    # Annotate bars
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{auc:.3f}", va="center", fontsize=9)

    # Family legend
    legend_patches = [mpatches.Patch(color=c, label=f) for f, c in fam_map.items()]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], marker="o", color="black", markersize=5,
                   lw=0, label="CV mean ± std"),
        plt.Line2D([0], [0], color="gray", linestyle="--", label="Random"),
    ], fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_ranked_algorithms.png", dpi=160)
    plt.close()


def plot_shap_winner(best_model: Pipeline, X_test: pd.DataFrame,
                     best_name: str, output_dir: Path) -> None:
    """SHAP summary for the best model only."""
    if not _SHAP:
        return
    try:
        preprocess  = best_model.named_steps["preprocess"]
        feat_eng    = best_model.named_steps["feature_engineering"]
        selector    = best_model.named_steps["feature_selection"]
        final_model = best_model.named_steps["model"]
        feat_names  = preprocess.get_feature_names_out()
        sel_names   = feat_names[selector.get_support()]

        X_tr = selector.transform(preprocess.transform(feat_eng.transform(X_test)))
        X_df = pd.DataFrame(X_tr, columns=sel_names)

        if hasattr(final_model, "feature_importances_"):
            explainer   = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            masker      = shap.maskers.Independent(X_df, max_samples=80)
            explainer   = shap.Explainer(final_model.predict_proba, masker)
            shap_values = explainer(X_df).values[:, :, 1]

        plt.figure()
        shap.summary_plot(shap_values, X_df, plot_type="bar",
                          show=False, max_display=15)
        plt.title(f"SHAP — {best_name} (winner)")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_shap_winner.png", dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as e:
        log.warning("SHAP for winner failed: %s", e)


def plot_confusion_grid(results: list[dict], y_test, output_dir: Path) -> None:
    """Grid of confusion matrices for all algorithms."""
    n  = len(results)
    nc = min(4, n)
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.5, nr * 3.2))
    axes_flat = axes.flatten() if n > 1 else [axes]
    for ax, r in zip(axes_flat, results):
        probs  = r["test_probs"]
        preds  = (probs >= r["test"]["threshold"]).astype(int)
        cm     = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    cbar=False, linewidths=0.5)
        ax.set_title(f'{r["name"]}\nAUC={r["test"]["roc_auc"]:.3f}', fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    plt.suptitle("Confusion Matrices — All Algorithms", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_confusion_grid.png", dpi=160, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def _safe(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.ndarray): return v.tolist()
    return v


def _jsonable(obj):
    if isinstance(obj, dict):   return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [_jsonable(v) for v in obj]
    return _safe(obj)


def save_comparison_report(
    results: list[dict],
    pval_df: pd.DataFrame,
    best_name: str,
    output_dir: Path,
) -> None:
    # ── Ranked summary CSV ────────────────────────────────────────────────────
    rows = []
    for rank, r in enumerate(
        sorted(results, key=lambda x: x["test"]["roc_auc"], reverse=True), 1
    ):
        rows.append({
            "rank":              rank,
            "algorithm":         r["name"],
            "family":            r["family"],
            "test_roc_auc":      r["test"]["roc_auc"],
            "cv_roc_auc_mean":   round(float(np.mean(r["cv_fold_scores"])), 4),
            "cv_roc_auc_std":    round(float(np.std(r["cv_fold_scores"])), 4),
            "test_f1":           r["test"]["f1"],
            "test_precision":    r["test"]["precision"],
            "test_recall":       r["test"]["recall"],
            "test_avg_precision":r["test"]["average_precision"],
            "test_balanced_acc": r["test"]["balanced_accuracy"],
            "test_brier_score":  r["test"]["brier_score"],
            "threshold":         r["test"]["threshold"],
            "best_params":       json.dumps(r["best_params"]),
            "is_winner":         r["name"] == best_name,
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "comparison_summary.csv", index=False)
    log.info("Saved comparison_summary.csv")

    # ── Full JSON report ──────────────────────────────────────────────────────
    report = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "winner":        best_name,
        "primary_metric": PRIMARY_METRIC,
        "n_algorithms":  len(results),
        "cv_folds":      CV_FOLDS,
        "algorithms":    _jsonable([
            {
                "name":           r["name"],
                "family":         r["family"],
                "cv_roc_auc_mean": round(float(np.mean(r["cv_fold_scores"])), 4),
                "cv_roc_auc_std":  round(float(np.std(r["cv_fold_scores"])), 4),
                "test":           r["test"],
                "best_params":    r["best_params"],
            }
            for r in sorted(results, key=lambda x: x["test"]["roc_auc"], reverse=True)
        ]),
        "significance_matrix": _jsonable(pval_df.round(4).to_dict()),
    }
    (output_dir / "comparison_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    log.info("Saved comparison_report.json")

    # ── Significance CSV ──────────────────────────────────────────────────────
    pval_df.round(4).to_csv(output_dir / "significance_matrix.csv")


def print_leaderboard(results: list[dict], best_name: str) -> None:
    ranked = sorted(results, key=lambda r: r["test"]["roc_auc"], reverse=True)
    header = (
        f"\n{'Rank':<5} {'Algorithm':<22} {'Family':<14}"
        f"{'Test AUC':<11} {'CV AUC':<10} {'CV±std':<9}"
        f"{'F1':<8} {'Brier':<8} Winner"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for i, r in enumerate(ranked, 1):
        winner = "  ← WINNER" if r["name"] == best_name else ""
        print(
            f"{i:<5} {r['name']:<22} {r['family']:<14}"
            f"{r['test']['roc_auc']:<11.4f}"
            f"{np.mean(r['cv_fold_scores']):<10.4f}"
            f"{np.std(r['cv_fold_scores']):<9.4f}"
            f"{r['test']['f1']:<8.4f}"
            f"{r['test']['brier_score']:<8.4f}"
            f"{winner}"
        )
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# Main comparison workflow
# ══════════════════════════════════════════════════════════════════════════════

def compare(output_dir: Path, quick: bool = False) -> dict[str, Any]:
    """
    Full professional model comparison workflow:
      1. Load data
      2. For every algorithm: tune hyperparams on train, collect CV fold scores
      3. Final refit on full train set, evaluate on held-out test
      4. OOF threshold tuning per algorithm
      5. Statistical significance matrix
      6. All comparison plots
      7. Save best model as production artifact
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("=== Model comparison started (quick=%s, n_jobs=%d) ===", quick, N_JOBS)

    X_train, X_test, y_train, y_test = load_and_split()
    log.info("Train: %d rows | Test: %d rows | Positive rate: %.2f",
             len(X_train), len(X_test), y_train.mean())

    catalogue = get_algorithm_catalogue(quick=quick)
    cv        = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results   = []
    cv_scores_map: dict[str, np.ndarray] = {}

    log.info("Running %d algorithms …", len(catalogue))
    for name, cfg in catalogue.items():
        try:
            best_model, search = tune_one(name, cfg, X_train, y_train, cv)

            # Per-fold CV scores for the best pipeline
            fold_scores = cross_val_score(
                clone(best_model), X_train, y_train,
                cv=cv, scoring=PRIMARY_METRIC, n_jobs=N_JOBS,
            )
            cv_scores_map[name] = fold_scores

            # Final refit on full train set
            final = clone(best_model)
            final.fit(X_train, y_train)
            probs = final.predict_proba(X_test)[:, 1]

            # OOF threshold
            thr = tune_threshold_oof(best_model, X_train, y_train, cv)

            results.append({
                "name":            name,
                "family":          cfg.get("family", "Other"),
                "model":           final,
                "test_probs":      probs,
                "test":            evaluate(y_test, probs, thr),
                "cv_fold_scores":  fold_scores,
                "best_params":     search.best_params_,
                "cv_best_score":   float(search.best_score_),
            })
            log.info("  ✓ %-22s  CV AUC=%.4f  Test AUC=%.4f  F1=%.4f",
                     name,
                     float(np.mean(fold_scores)),
                     results[-1]["test"]["roc_auc"],
                     results[-1]["test"]["f1"])
        except Exception as exc:
            log.warning("  ✗ %-22s  FAILED: %s", name, exc)

    if not results:
        raise RuntimeError("All algorithms failed. Check your environment.")

    # ── Select winner ─────────────────────────────────────────────────────────
    best = max(results, key=lambda r: r["test"]["roc_auc"])
    best_name = best["name"]
    log.info("Winner: %s (test AUC=%.4f)", best_name, best["test"]["roc_auc"])

    # ── Plots ─────────────────────────────────────────────────────────────────
    log.info("Generating plots …")
    plot_ranked_bar(results, output_dir)
    plot_cv_comparison(results, output_dir)
    plot_metric_heatmap(results, output_dir)
    plot_roc_all(results, X_test, y_test, output_dir)
    plot_pr_all(results, y_test, output_dir)
    plot_calibration_all(results, y_test, output_dir)
    plot_confusion_grid(results, y_test, output_dir)
    plot_shap_winner(best["model"], X_test, best_name, output_dir)

    # ── Statistical significance ──────────────────────────────────────────────
    pval_df = significance_matrix(cv_scores_map)
    plot_significance_heatmap(pval_df, output_dir)

    # ── Save reports ──────────────────────────────────────────────────────────
    save_comparison_report(results, pval_df, best_name, output_dir)

    # ── Save winner as production model ───────────────────────────────────────
    winner_path = output_dir / "best_model.joblib"
    joblib.dump(best["model"], winner_path)
    log.info("Best model saved → %s", winner_path)

    # ── Leaderboard ───────────────────────────────────────────────────────────
    print_leaderboard(results, best_name)

    log.info("=== Comparison complete. Artifacts in: %s ===", output_dir.resolve())
    return {
        "winner":     best_name,
        "winner_auc": best["test"]["roc_auc"],
        "winner_f1":  best["test"]["f1"],
        "n_compared": len(results),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Professional supervised-learning comparison for Titanic survival.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cp = sub.add_parser("compare", help="Run full model comparison.")
    cp.add_argument("--output-dir", type=Path, default=Path("results/comparison"))
    cp.add_argument("--quick", action="store_true",
                    help="Fast mode: fewer CV iterations for quick testing.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "compare":
        summary = compare(args.output_dir, quick=args.quick)
        print(f"\nWinner  : {summary['winner']}")
        print(f"Test AUC: {summary['winner_auc']:.4f}")
        print(f"Test F1 : {summary['winner_f1']:.4f}")


if __name__ == "__main__":
    main()
