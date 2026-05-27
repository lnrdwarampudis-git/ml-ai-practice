"""
cpu_benchmark.py
================
Industry-standard algorithm benchmarking script for CPU Activity prediction.

Built on top of cpu_pipeline.py:
  - Reuses load_data / fix_data_types / split_data exactly as-is
  - Reuses CPUFeatureEngineer, _DynamicNumericPreprocessor, build_pipeline
  - Reuses RAW_FEATURES, LOG_TRANSFORM_FEATURES, POLY_CANDIDATE_FEATURES

Full professional benchmark workflow (parallel to titanic_benchmark.py):
  Phase 1  — EDA (train-set only, from reference pipeline)
  Phase 2  — Algorithm screening: 16 regressors across every sklearn section
             + XGBoost + LightGBM (industry de-facto)
             5-fold KFold CV, 6 regression metrics + overfit gap
  Phase 3  — Friedman χ² + Wilcoxon–Bonferroni statistical significance tests
  Phase 4  — RandomizedSearchCV tuning for top-3 models (refit=r2)
  Phase 5  — Ensemble construction: VotingRegressor + StackingRegressor (Ridge meta)
  Phase 6  — Hold-out test: 8-metric evaluation suite
  Phase 7  — Calibration: residual diagnostics, prediction interval coverage
  Phase 8  — SHAP explainability for champion model
  Phase 9  — Subgroup disparity (load_tier: idle/low/medium/high)
  Phase 10 — Self-contained HTML + JSON + CSV report

References:
  https://scikit-learn.org/stable/supervised_learning.html

Usage:
  python cpu_benchmark.py                    # full run
  python cpu_benchmark.py --quick            # smoke-test (~1 min)
  python cpu_benchmark.py --output-dir ./results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path("benchmark_cpu") / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, ks_2samp, wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
    SGDRegressor,
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
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, RobustScaler, SplineTransformer, StandardScaler,
    PolynomialFeatures,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor; _XGB = True
except ImportError:
    _XGB = False
try:
    from lightgbm import LGBMRegressor; _LGB = True
except ImportError:
    _LGB = False
try:
    import shap; _SHAP = True
except ImportError:
    _SHAP = False

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants (identical to reference) ────────────────────────────────────────
RANDOM_STATE = 42
TARGET       = "usr"
N_JOBS       = int(os.environ.get("ML_N_JOBS", -1))
ALPHA        = 0.05   # significance level

RAW_FEATURES = [
    "lread","lwrite","scall","sread","swrite",
    "fork","exec","rchar","wchar",
    "pgout","ppgout","pgfree","pgin","ppgin",
    "pflt","vflt","runqsz","freemem","freeswap",
    "sys","wait",
]
LOG_TRANSFORM_FEATURES = [
    "lread","lwrite","scall","sread","swrite",
    "fork","exec","rchar","wchar",
    "pgout","ppgout","pgfree","pgin","ppgin",
    "pflt","vflt","runqsz","freemem","freeswap",
]

# Benchmark CV scoring (6 regression metrics)
CV_SCORING = {
    "r2":           "r2",
    "neg_rmse":     "neg_root_mean_squared_error",
    "neg_mae":      "neg_mean_absolute_error",
    "neg_medae":    "neg_median_absolute_error",
    "neg_mape":     "neg_mean_absolute_percentage_error",
    "explained_var":"explained_variance",
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared infrastructure (copied / adapted from reference pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    log.info("Loading CPU Activity dataset (data_id=197) …")
    try:
        cpu_act    = fetch_openml(data_id=197, as_frame=True, parser="auto")
        cpu_act_df = cpu_act.frame
        return cpu_act_df.copy()
    except Exception as exc:
        log.warning("OpenML unavailable (%s) — using synthetic data.", exc)
        return _make_synthetic_cpu()


def _make_synthetic_cpu() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    n   = 8192
    df  = pd.DataFrame({
        "lread":rng.exponential(50,n),"lwrite":rng.exponential(30,n),
        "scall":rng.exponential(200,n),"sread":rng.exponential(80,n),
        "swrite":rng.exponential(60,n),"fork":rng.exponential(5,n),
        "exec":rng.exponential(8,n),"rchar":rng.exponential(5000,n),
        "wchar":rng.exponential(3000,n),"pgout":rng.exponential(2,n),
        "ppgout":rng.exponential(2,n),"pgfree":rng.exponential(5,n),
        "pgin":rng.exponential(3,n),"ppgin":rng.exponential(3,n),
        "pflt":rng.exponential(20,n),"vflt":rng.exponential(50,n),
        "runqsz":rng.exponential(3,n),"freemem":rng.exponential(800,n),
        "freeswap":rng.exponential(50000,n),
        "sys":rng.uniform(0,30,n),"wait":rng.uniform(0,20,n),
    })
    df[TARGET] = np.clip(
        0.3*df["sys"]+0.1*df["runqsz"]+0.05*np.log1p(df["vflt"])
        +rng.normal(0,5,n), 0, 100)
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rebuilt = {}
    for col in df.columns:
        _arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        if np.isnan(_arr).all() and not df[col].isna().all():
            log.warning("fix_data_types: '%s' all-NaN after cast", col)
        rebuilt[col] = _arr
    return pd.DataFrame(rebuilt, index=df.index)


def split_data(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def missingness_report(df):
    r = df.isna().agg(["sum","mean"]).T.rename(
        columns={"sum":"missing_count","mean":"missing_rate"})
    r["dtype"] = df.dtypes.astype(str)
    return r.sort_values("missing_rate", ascending=False)


# ── Feature engineering (copied verbatim from reference) ─────────────────────
class CPUFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for col in LOG_TRANSFORM_FEATURES:
            if col in X.columns:
                X[f"log_{col}"] = np.log1p(X[col].clip(lower=0).fillna(0))
        fork_s = X["fork"].clip(lower=0).fillna(0)  if "fork"  in X.columns else 0
        exec_s = X["exec"].clip(lower=0).fillna(0)  if "exec"  in X.columns else 0
        pflt_s = X["pflt"].clip(lower=0).fillna(0)  if "pflt"  in X.columns else 0
        vflt_s = X["vflt"].clip(lower=0).fillna(0)  if "vflt"  in X.columns else 0
        sys_s  = X["sys"].clip(lower=0).fillna(0)   if "sys"   in X.columns else 0
        wait_s = X["wait"].clip(lower=0).fillna(0)  if "wait"  in X.columns else 0
        rq_s   = X["runqsz"].clip(lower=0).fillna(0) if "runqsz" in X.columns else 0
        fm_s   = X["freemem"].clip(lower=1).fillna(1) if "freemem" in X.columns else 1
        X["fork_exec_interact"] = fork_s * exec_s
        X["pflt_vflt_interact"] = pflt_s * vflt_s
        X["sys_wait_total"]     = sys_s + wait_s
        X["runqsz_sq"]          = rq_s ** 2
        X["io_pressure"]        = (pflt_s + vflt_s) / np.maximum(fm_s, 1)
        X["memory_pressure"]    = (pflt_s + vflt_s) / np.maximum(fm_s / 1000, 1)
        X["cpu_contention"]     = rq_s * (sys_s + wait_s)
        return X


# ── _DynamicNumericPreprocessor (copied verbatim from reference) ──────────────
class _DynamicNumericPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_name: str = "RobustScaler"):
        self.scaler_name = scaler_name
    def _make_scaler(self):
        return {"StandardScaler":StandardScaler(),"RobustScaler":RobustScaler(),
                "MinMaxScaler":MinMaxScaler()}.get(self.scaler_name, RobustScaler())
    def fit(self, X, y=None):
        self.cols_ = [c for c in X.columns
                      if c != TARGET and pd.api.types.is_numeric_dtype(X[c])]
        _arr = self._to_numpy(X)
        self.imputer_ = SimpleImputer(strategy="median").fit(_arr)
        self.scaler_  = self._make_scaler().fit(self.imputer_.transform(_arr))
        return self
    def transform(self, X):
        present = [c for c in self.cols_ if c in X.columns]
        _arr = self._to_numpy(X, present)
        return self.scaler_.transform(self.imputer_.transform(_arr))
    def _to_numpy(self, X, cols=None):
        cols = cols or self.cols_
        return np.column_stack([
            pd.to_numeric(X[c], errors="coerce")
            .to_numpy(dtype=np.float64, na_value=np.nan) for c in cols
        ])
    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols_)


def wrap_regressor(reg: BaseEstimator) -> Pipeline:
    """
    Wrap any regressor in the full reference-pipeline stack:
      CPUFeatureEngineer → _DynamicNumericPreprocessor → regressor
    No SelectFromModel — benchmark measures raw classifier signal.
    """
    return Pipeline([
        ("feature_engineering", CPUFeatureEngineer()),
        ("preprocess",          _DynamicNumericPreprocessor("RobustScaler")),
        ("model",               reg),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: EDA (mirrors reference save_research_artifacts)
# ─────────────────────────────────────────────────────────────────────────────
def save_research_artifacts(X_train, y_train, output_dir):
    log.info("Saving EDA artifacts (train-set only) …")
    output_dir.mkdir(parents=True, exist_ok=True)
    eda = X_train.copy(); eda[TARGET] = y_train.values

    missingness_report(eda).to_csv(output_dir / "eda_missingness.csv")
    eda.dtypes.astype(str).rename("dtype").to_csv(output_dir / "eda_schema.csv")

    num_cols = [c for c in eda.select_dtypes(include=[np.number]).columns if c != TARGET]
    corr_df  = pd.DataFrame({c: pd.to_numeric(eda[c], errors="coerce")
                              .to_numpy(dtype=np.float64) for c in num_cols + [TARGET]})
    corr     = corr_df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    corr.to_csv(output_dir / "eda_correlation_with_target.csv")

    # VIF — pure numpy
    feat_arr = np.column_stack([
        pd.to_numeric(eda[c], errors="coerce").to_numpy(dtype=np.float64)
        for c in num_cols])
    _med = np.nanmedian(feat_arr, axis=0)
    _imp = feat_arr.copy()
    for j in range(_imp.shape[1]):
        mask = np.isnan(_imp[:,j]); _imp[mask,j] = 0.0 if np.isnan(_med[j]) else _med[j]
    _good = [j for j in range(_imp.shape[1]) if not np.isnan(_imp[:,j]).any()]
    _imp  = _imp[:,_good]; num_feat_g = [num_cols[j] for j in _good]
    vif_rows = []
    for i,col in enumerate(num_feat_g):
        other = np.delete(_imp,i,axis=1); tgt = _imp[:,i]
        if np.unique(tgt).size <= 1: continue
        try:
            r2  = LinearRegression().fit(other,tgt).score(other,tgt)
            vif = 9999.0 if r2>=0.999 else float(1/(1-r2))
            vif_rows.append({"feature":col,"vif":vif})
        except Exception: pass
    pd.DataFrame(vif_rows).sort_values("vif",ascending=False).to_csv(
        output_dir/"eda_vif_report.csv", index=False)

    # Grouped stats by load tier
    eda["load_tier"] = pd.cut(
        pd.to_numeric(eda["runqsz"], errors="coerce").fillna(0) if "runqsz" in eda.columns
        else pd.Series(np.zeros(len(eda))),
        bins=[0,1,3,6,1000], labels=["idle","low","medium","high"])
    grouped = {
        "usr_by_load_tier": eda.groupby("load_tier",observed=False)[TARGET]
                             .agg(["mean","median","std"]).to_dict(),
        "target_stats":     y_train.describe().to_dict(),
        "sys_wait_constraint_pct": float(
            ((eda.get("sys",pd.Series(0,index=eda.index)).fillna(0)
             + eda.get("wait",pd.Series(0,index=eda.index)).fillna(0)
             + eda[TARGET]).between(95,105).mean())
            if "sys" in eda.columns and "wait" in eda.columns else 0),
    }
    write_json(output_dir/"eda_grouped_stats.json", grouped)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7,4))
    sns.histplot(y_train, kde=True, bins=40, color="#4C78A8")
    plt.axvline(y_train.median(), color="#E45756", linestyle="--",
                label=f"Median={y_train.median():.1f}")
    plt.title("usr distribution (train)"); plt.xlabel("usr (%)"); plt.legend()
    plt.tight_layout(); plt.savefig(output_dir/"eda_target_distribution.png",dpi=150); plt.close()

    plt.figure(figsize=(10,5))
    col_c = ["#54A24B" if v>0 else "#E45756" for v in corr]
    corr.plot(kind="barh", color=col_c)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Feature correlation with usr (train)")
    plt.tight_layout(); plt.savefig(output_dir/"eda_correlation.png",dpi=150); plt.close()

    if "runqsz" in eda.columns:
        plt.figure(figsize=(6,4))
        plt.scatter(eda["runqsz"], eda[TARGET], alpha=0.12, s=5, color="#E45756")
        plt.xlabel("runqsz"); plt.ylabel("usr (%)")
        plt.title("runqsz vs usr — non-linear convex relationship")
        plt.tight_layout(); plt.savefig(output_dir/"eda_runqsz_vs_usr.png",dpi=150); plt.close()

    if "sys" in eda.columns:
        plt.figure(figsize=(6,4))
        plt.scatter(eda["sys"], eda[TARGET], alpha=0.12, s=5, color="#4C78A8")
        plt.xlabel("sys (%)"); plt.ylabel("usr (%)")
        plt.title("sys vs usr — CPU constraint (sys+wait+usr≈100)")
        plt.tight_layout(); plt.savefig(output_dir/"eda_sys_vs_usr.png",dpi=150); plt.close()

    log.info("EDA artifacts saved to %s", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Regressor registry
# ─────────────────────────────────────────────────────────────────────────────
def get_regressors(quick: bool = False) -> dict[str, BaseEstimator]:
    """
    16 regressors covering every major family in:
    https://scikit-learn.org/stable/supervised_learning.html

    Section 1.1  Linear Models      → Ridge, Lasso, ElasticNet, BayesianRidge,
                                       HuberRegressor, SGD
    Section 1.4  SVM                → SVR (RBF kernel)
    Section 1.6  Nearest Neighbours → KNeighborsRegressor
    Section 1.10 Decision Trees     → DecisionTreeRegressor
    Section 1.11 Ensembles          → RandomForest, ExtraTrees,
                                       GradientBoosting, AdaBoost, Bagging
    Section 1.17 Neural Networks    → MLPRegressor
    Industry     XGBoost + LightGBM (conditional on install)
    """
    n = 50 if quick else 150
    regs: dict[str, BaseEstimator] = {
        # ── 1.1 Linear ────────────────────────────────────────────────────────
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "BayesianRidge": BayesianRidge(),
        "HuberRegressor": HuberRegressor(epsilon=1.35, max_iter=300),
        "SGD": SGDRegressor(
            loss="huber", penalty="elasticnet", l1_ratio=0.15,
            max_iter=1000, random_state=RANDOM_STATE,
        ),
        # ── 1.4 SVM ───────────────────────────────────────────────────────────
        "SVR_RBF": SVR(kernel="rbf", C=10.0, epsilon=0.5),
        # ── 1.6 Neighbours ────────────────────────────────────────────────────
        "KNN": KNeighborsRegressor(n_neighbors=7, weights="distance", n_jobs=N_JOBS),
        # ── 1.10 Decision Tree ────────────────────────────────────────────────
        "DecisionTree": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=4, random_state=RANDOM_STATE),
        # ── 1.11 Ensembles ────────────────────────────────────────────────────
        "RandomForest": RandomForestRegressor(
            n_estimators=n, max_depth=10, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=n, max_depth=10, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=n, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=n, learning_rate=0.5, random_state=RANDOM_STATE),
        "Bagging": BaggingRegressor(
            n_estimators=n, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        # ── 1.17 Neural Network ───────────────────────────────────────────────
        "MLP": MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu",
            max_iter=500, early_stopping=True, random_state=RANDOM_STATE),
    }
    if _XGB:
        regs["XGBoost"] = XGBRegressor(
            n_estimators=n, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="rmse", random_state=RANDOM_STATE,
            n_jobs=N_JOBS, verbosity=0)
    if _LGB:
        regs["LightGBM"] = LGBMRegressor(
            n_estimators=n, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1)
    return regs


# ── Hyperparameter search spaces ──────────────────────────────────────────────
PARAM_GRIDS: dict[str, dict] = {
    "Ridge":         {"model__alpha": [0.001,0.01,0.1,1,5,10,50,100,500]},
    "Lasso":         {"model__alpha": [0.0001,0.001,0.01,0.05,0.1,0.5,1.0]},
    "ElasticNet":    {"model__alpha": [0.001,0.01,0.1,0.5,1.0],
                      "model__l1_ratio": [0.1,0.3,0.5,0.7,0.9]},
    "BayesianRidge": {"model__alpha_1":[1e-6,1e-4,1e-2],
                      "model__lambda_1":[1e-6,1e-4,1e-2]},
    "HuberRegressor":{"model__epsilon":[1.1,1.35,1.5,2.0],
                      "model__alpha":[0.0001,0.001,0.01]},
    "SGD":           {"model__alpha":np.logspace(-4,0,8).tolist(),
                      "model__l1_ratio":[0.1,0.3,0.5,0.7,0.9]},
    "SVR_RBF":       {"model__C":[0.5,1,5,10,50],
                      "model__gamma":["scale","auto",0.01,0.1],
                      "model__epsilon":[0.1,0.5,1.0]},
    "KNN":           {"model__n_neighbors":[3,5,7,11,15],
                      "model__weights":["uniform","distance"]},
    "DecisionTree":  {"model__max_depth":[4,6,8,12,None],
                      "model__min_samples_leaf":[1,2,4,8]},
    "RandomForest":  {"model__n_estimators":[100,200,300],
                      "model__max_depth":[6,8,12,None],
                      "model__min_samples_leaf":[1,2,4]},
    "ExtraTrees":    {"model__n_estimators":[100,200,300],
                      "model__max_depth":[6,8,12,None],
                      "model__min_samples_leaf":[1,2,4]},
    "GradientBoosting":{"model__n_estimators":[100,200,300],
                        "model__max_depth":[3,4,5],
                        "model__learning_rate":[0.02,0.05,0.1,0.2],
                        "model__subsample":[0.7,0.9]},
    "AdaBoost":      {"model__n_estimators":[50,100,200],
                      "model__learning_rate":[0.05,0.1,0.5,1.0]},
    "Bagging":       {"model__n_estimators":[50,100,200]},
    "MLP":           {"model__hidden_layer_sizes":[(64,),(128,),(128,64),(256,128)],
                      "model__alpha":[0.0001,0.001,0.01]},
    "XGBoost":       {"model__n_estimators":[100,200,300],
                      "model__max_depth":[3,4,5,6],
                      "model__learning_rate":[0.02,0.05,0.1,0.2],
                      "model__subsample":[0.7,0.9],
                      "model__colsample_bytree":[0.6,0.8,1.0]},
    "LightGBM":      {"model__n_estimators":[100,200,300],
                      "model__max_depth":[3,4,5,6],
                      "model__learning_rate":[0.02,0.05,0.1,0.2],
                      "model__subsample":[0.7,0.9]},
}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: CV Screening
# ─────────────────────────────────────────────────────────────────────────────
def screen_regressors(X_tr, y_tr, regressors, n_splits):
    cv   = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, reg in regressors.items():
        log.info("  [Screen] %-22s …", name)
        t0 = time.perf_counter()
        try:
            res = cross_validate(
                wrap_regressor(clone(reg)),
                X_tr, y_tr, cv=cv,
                scoring=CV_SCORING,
                return_train_score=True,
                n_jobs=1,
                error_score="raise",
            )
            elapsed = time.perf_counter() - t0
            row = {"model": name, "cv_time_s": round(elapsed,2)}
            for m in CV_SCORING:
                ts  = res[f"test_{m}"]
                trs = res[f"train_{m}"]
                row[f"{m}_mean"]         = float(ts.mean())
                row[f"{m}_std"]          = float(ts.std())
                row[f"{m}_train_mean"]   = float(trs.mean())
                row[f"{m}_overfit_gap"]  = float(trs.mean() - ts.mean())
                row[f"_raw_{m}"]         = ts.tolist()
        except Exception as exc:
            log.warning("    %s — FAILED: %s", name, exc)
            row = {"model": name, "cv_time_s": -1,
                   **{f"{m}_mean": np.nan for m in CV_SCORING}}
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Statistical significance
# ─────────────────────────────────────────────────────────────────────────────
def run_stat_tests(screen_df, metric="r2"):
    raw_col = f"_raw_{metric}"
    valid   = screen_df.dropna(subset=[f"{metric}_mean"]).copy()
    if raw_col not in valid.columns or len(valid) < 2:
        return {}

    arrays = [np.array(r) for r in valid[raw_col]]
    names  = valid["model"].tolist()
    best_i = int(valid[f"{metric}_mean"].idxmax())
    champ  = valid.loc[best_i, "model"]
    champ_sc = arrays[list(valid.index).index(best_i)]

    try:
        f_stat, f_p = friedmanchisquare(*arrays)
    except Exception:
        f_stat, f_p = np.nan, np.nan

    n_comp   = len(arrays) - 1
    pairwise = []
    for nm, sc in zip(names, arrays):
        if nm == champ: continue
        try:
            diff    = champ_sc - sc
            stat, p = (wilcoxon(diff, alternative="greater", zero_method="wilcox")
                       if not np.all(diff==0) else (np.nan, 1.0))
        except Exception:
            stat, p = np.nan, np.nan
        p_bonf = float(min(1.0, p*n_comp)) if not np.isnan(p) else np.nan
        pairwise.append({
            "model": nm,
            "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
            "p_value": float(p) if not np.isnan(p) else None,
            "p_bonferroni": float(p_bonf) if not np.isnan(p_bonf) else None,
            "significantly_worse": bool(p_bonf < ALPHA) if not np.isnan(p_bonf) else None,
        })
    return {
        "metric": metric, "champion_model": champ,
        "friedman": {
            "statistic": float(f_stat) if not np.isnan(f_stat) else None,
            "p_value":   float(f_p)    if not np.isnan(f_p)    else None,
            "significant": bool(f_p < ALPHA) if not np.isnan(f_p) else None,
        },
        "pairwise_vs_champion": pairwise,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Hyperparameter tuning
# ─────────────────────────────────────────────────────────────────────────────
def tune_top_models(screen_df, regressors, X_tr, y_tr, top_n, n_iter, n_splits):
    cv  = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    top = (screen_df.dropna(subset=["r2_mean"])
           .sort_values("r2_mean", ascending=False)
           .head(top_n)["model"].tolist())
    log.info("Top-%d for tuning: %s", top_n, top)

    tuned: dict[str, Any] = {}
    for name in top:
        reg  = clone(regressors[name])
        pipe = wrap_regressor(reg)
        if name not in PARAM_GRIDS:
            log.info("  [Tune] %-22s — no grid, fitting defaults.", name)
            pipe.fit(X_tr, y_tr)
            tuned[name] = {"tuned": False, "best_estimator": pipe}
            continue
        log.info("  [Tune] %-22s (n_iter=%d) …", name, n_iter)
        try:
            search = RandomizedSearchCV(
                pipe, PARAM_GRIDS[name],
                n_iter=n_iter, scoring="r2",
                cv=cv, refit=True,
                random_state=RANDOM_STATE, n_jobs=N_JOBS,
                error_score="raise",
            )
            search.fit(X_tr, y_tr)
            tuned[name] = {
                "tuned": True,
                "best_params":     search.best_params_,
                "best_cv_r2":      float(search.best_score_),
                "best_estimator":  search.best_estimator_,
            }
            log.info("    Best CV R² after tuning: %.4f", search.best_score_)
        except Exception as exc:
            log.warning("    Tuning failed for %s: %s", name, exc)
            pipe.fit(X_tr, y_tr)
            tuned[name] = {"tuned": False, "best_estimator": pipe}
    return tuned


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Ensemble
# ─────────────────────────────────────────────────────────────────────────────
def build_ensembles(tuned, X_tr, y_tr):
    estimators = [(n, info["best_estimator"]) for n, info in tuned.items()]
    if len(estimators) < 2:
        return {}
    results: dict[str, Any] = {}
    for ename, ekwargs in [
        ("AverageVoting",  {"weights": None}),
        ("Stacking",       {"final_estimator": Ridge(alpha=1.0), "cv": 3,
                            "n_jobs": N_JOBS}),
    ]:
        try:
            log.info("  [Ensemble] %s …", ename)
            if ename == "AverageVoting":
                m = VotingRegressor(estimators=estimators, n_jobs=N_JOBS)
            else:
                m = StackingRegressor(estimators=estimators, **ekwargs)
            m.fit(X_tr, y_tr)
            results[ename] = {"model": m}
        except Exception as exc:
            log.warning("  %s failed: %s", ename, exc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: Hold-out test evaluation — 8 regression metrics
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_on_holdout(name, model, X_te, y_te):
    y_pred    = model.predict(X_te)
    residuals = y_te.to_numpy() - y_pred
    rmse      = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    return {
        "model": name,
        # ── Discrimination / fit ─────────────────────────────────────────────
        "r2":            round(float(r2_score(y_te, y_pred)), 4),
        "explained_var": round(float(1 - np.var(residuals)/np.var(y_te)), 4),
        # ── Scale-specific ───────────────────────────────────────────────────
        "rmse":          round(rmse, 4),
        "mae":           round(float(mean_absolute_error(y_te, y_pred)), 4),
        "medae":         round(float(median_absolute_error(y_te, y_pred)), 4),
        # ── Scale-free ───────────────────────────────────────────────────────
        "mape":          round(float(mean_absolute_percentage_error(y_te, y_pred)), 4),
        # ── Residual stats ───────────────────────────────────────────────────
        "residual_mean": round(float(residuals.mean()), 4),
        "residual_std":  round(float(residuals.std()), 4),
        # Store for plots — stripped before JSON
        "_y_pred":    y_pred,
        "_residuals": residuals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7: Residual / calibration diagnostics for all test models
# ─────────────────────────────────────────────────────────────────────────────
def residual_summary_plot(test_results, output_dir):
    """Overlay residual distributions for all tuned+ensemble models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.get_cmap("tab10")
    for i, res in enumerate(test_results):
        r = np.array(res["_residuals"])
        axes[0].hist(r, bins=40, alpha=0.5, label=res["model"],
                     color=cmap(i % 10), density=True)
        axes[1].scatter(np.array(res["_y_pred"]), r,
                        alpha=0.2, s=6, color=cmap(i % 10), label=res["model"])
    axes[0].axvline(0, color="black", linewidth=1.2)
    axes[0].set_xlabel("Residual (actual − predicted)"); axes[0].set_ylabel("Density")
    axes[0].set_title("Residual distributions — tuned models"); axes[0].legend(fontsize=7)
    axes[1].axhline(0, color="black", linewidth=1.0)
    axes[1].set_xlabel("Predicted usr (%)"); axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Predicted"); axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "residual_diagnostics.png", dpi=150); plt.close()


def prediction_interval_coverage(test_results, output_dir):
    """
    Phase 7 calibration: check what % of test residuals fall within
    ±1 RMSE, ±2 RMSE bands. For a well-calibrated model, ±1 RMSE ≈ 68%.
    """
    rows = []
    for res in test_results:
        r    = np.abs(np.array(res["_residuals"]))
        rmse = res["rmse"]
        rows.append({
            "model":          res["model"],
            "within_1_rmse":  round(float((r <= rmse).mean()), 4),
            "within_2_rmse":  round(float((r <= 2*rmse).mean()), 4),
            "within_3_rmse":  round(float((r <= 3*rmse).mean()), 4),
            "ideal_1_rmse":   0.683,
            "ideal_2_rmse":   0.954,
        })
    pd.DataFrame(rows).to_csv(output_dir/"prediction_interval_coverage.csv", index=False)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8: SHAP for champion
# ─────────────────────────────────────────────────────────────────────────────
def compute_shap(model, X_te, y_te, y_pred, output_dir):
    if not _SHAP:
        log.warning("pip install shap"); return
    log.info("Computing SHAP values for champion …")
    try:
        clf  = model.named_steps["model"]
        prep = model.named_steps["preprocess"]
        fe   = model.named_steps["feature_engineering"]
        Xt   = prep.transform(fe.transform(X_te))
        fn   = prep.get_feature_names_out()
        Xdf  = pd.DataFrame(Xt, columns=fn)

        if hasattr(clf, "feature_importances_"):
            exp = shap.TreeExplainer(clf); sv = exp.shap_values(Xdf)
        elif hasattr(clf, "coef_"):
            exp = shap.LinearExplainer(clf, Xdf); sv = exp.shap_values(Xdf)
        else:
            mask = shap.maskers.Independent(Xdf, max_samples=100)
            exp  = shap.Explainer(clf.predict, mask); sv = exp(Xdf).values

        for ptype, fname in [("bar","shap_bar.png"),("dot","shap_beeswarm.png")]:
            plt.figure(figsize=(10,6))
            shap.summary_plot(sv, Xdf, plot_type=ptype, show=False, max_display=20)
            plt.title(f"SHAP — champion ({ptype})")
            plt.tight_layout()
            plt.savefig(output_dir/fname, dpi=150, bbox_inches="tight"); plt.close()

        worst = int(np.argmax(np.abs(y_te.to_numpy() - y_pred)))
        ev    = float(exp.expected_value) if not isinstance(exp.expected_value, np.ndarray) else float(exp.expected_value)
        shap.waterfall_plot(
            shap.Explanation(values=sv[worst], base_values=ev,
                             data=Xdf.iloc[worst].values, feature_names=list(fn)),
            show=False, max_display=15)
        plt.title(f"SHAP Waterfall — worst residual")
        plt.tight_layout()
        plt.savefig(output_dir/"shap_waterfall_worst.png", dpi=150, bbox_inches="tight"); plt.close()

        pd.DataFrame({"feature":fn,"mean_abs_shap":np.abs(sv).mean(axis=0)}
            ).sort_values("mean_abs_shap",ascending=False
            ).to_csv(output_dir/"shap_importance.csv", index=False)
        log.info("SHAP saved.")
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 9: Subgroup disparity
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_subgroups(champion_name, champion_model, X_te, y_te, y_pred, output_dir):
    overall_rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    overall_r2   = float(r2_score(y_te, y_pred))
    eval_df      = X_te.reset_index(drop=True).copy()
    eval_df["_y_true"] = y_te.to_numpy()
    eval_df["_y_pred"] = y_pred
    col = "runqsz" if "runqsz" in eval_df.columns else eval_df.columns[0]
    eval_df["load_tier"] = pd.cut(
        pd.to_numeric(eval_df[col],errors="coerce").fillna(0),
        bins=[0,1,3,6,1000], labels=["idle","low","medium","high"]).astype("category")

    rows = []
    for val, sub in eval_df.groupby("load_tier", observed=True):
        if len(sub) < 15: continue
        sr   = float(np.sqrt(mean_squared_error(sub["_y_true"],sub["_y_pred"])))
        rows.append({
            "group": str(val), "n": int(len(sub)),
            "mean_actual": round(float(sub["_y_true"].mean()),4),
            "rmse":        round(sr,4),
            "r2":          round(float(r2_score(sub["_y_true"],sub["_y_pred"])),4),
            "rmse_gap":    round(sr-overall_rmse,4),
            "alert":       bool(sr > overall_rmse*1.25),
        })

    if rows:
        rdf = pd.DataFrame(rows)
        rdf.to_csv(output_dir/"subgroup_report.csv", index=False)
        plt.figure(figsize=(7,4))
        colors = ["#E45756" if r["alert"] else "#4C78A8" for r in rows]
        plt.bar([r["group"] for r in rows], [r["rmse"] for r in rows], color=colors)
        plt.axhline(overall_rmse, linestyle="--", color="black", label=f"Overall RMSE={overall_rmse:.2f}")
        plt.xlabel("load_tier"); plt.ylabel("RMSE")
        plt.title(f"Phase 9 — Subgroup RMSE by load tier\n(red = alert: >25% above overall)  Champion: {champion_name}")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir/"subgroup_rmse.png", dpi=150); plt.close()
    return {"overall_rmse":overall_rmse,"overall_r2":overall_r2,"subgroups":rows}


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_screening_heatmap(df, out):
    cols = ["r2_mean","neg_rmse_mean","neg_mae_mean","neg_medae_mean","explained_var_mean"]
    d = df.dropna(subset=["r2_mean"]).set_index("model")[cols].copy()
    d.columns = ["R²","RMSE(neg)","MAE(neg)","MedAE(neg)","Explained Var"]
    d = d.sort_values("R²", ascending=False)
    fig, ax = plt.subplots(figsize=(11, max(5, len(d)*0.48+1.5)))
    sns.heatmap(d, annot=True, fmt=".3f", cmap="RdYlGn",
                linewidths=0.4, ax=ax, annot_kws={"size":8})
    ax.set_title("Algorithm Screening — CV Metric Heatmap (sorted by R²)", fontsize=12, pad=10)
    plt.tight_layout(); plt.savefig(out/"screening_heatmap.png", dpi=150); plt.close()


def plot_cv_boxplot(df, out, metric="r2"):
    raw = f"_raw_{metric}"
    if raw not in df.columns: return
    v     = df.dropna(subset=[f"{metric}_mean"]).sort_values(f"{metric}_mean", ascending=False)
    data  = [np.array(r) for r in v[raw]]
    names = v["model"].tolist()
    fig, ax = plt.subplots(figsize=(max(10,len(names)*0.75),5))
    bp  = ax.boxplot(data, patch_artist=True, widths=0.55)
    cmap = plt.cm.get_cmap("RdYlGn", len(names))
    for i,(patch,med) in enumerate(zip(bp["boxes"],bp["medians"])):
        patch.set_facecolor(cmap(i/len(names)))
        med.set_color("black"); med.set_linewidth(1.5)
    ax.set_xticks(range(1,len(names)+1))
    ax.set_xticklabels(names, rotation=38, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_"," ").title())
    ax.set_title(f"CV Distribution — {metric.title()}", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out/f"cv_boxplot_{metric}.png", dpi=150); plt.close()


def plot_overfit_gap(df, out):
    v = df.dropna(subset=["r2_mean"]).sort_values("r2_mean", ascending=False)
    gaps   = v["r2_overfit_gap"]
    colors = ["#E45756" if g>0.05 else "#54A24B" for g in gaps]
    fig, ax = plt.subplots(figsize=(max(10,len(v)*0.75),4))
    ax.bar(range(len(v)), gaps, color=colors, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.05, color="#E45756", linewidth=0.8, linestyle="--", alpha=0.6, label="Alert 0.05")
    ax.set_xticks(range(len(v)))
    ax.set_xticklabels(v["model"].tolist(), rotation=38, ha="right", fontsize=9)
    ax.set_ylabel("Train R² − CV R²")
    ax.set_title("Overfitting Analysis — Train–CV R² Gap\n(red = potential overfit)", fontsize=12)
    ax.legend(); plt.tight_layout()
    plt.savefig(out/"overfit_gap.png", dpi=150); plt.close()


def plot_actual_vs_predicted(test_results, champion, out):
    n    = len(test_results)
    cols = min(3, n); rows = (n+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4.5, rows*4))
    axes = np.array(axes).flatten()
    for i, res in enumerate(test_results):
        axes[i].scatter(np.array(res["_y_pred"]) + np.array(res["_residuals"]),
                        res["_y_pred"], alpha=0.25, s=6)
        mn = min(float(min(np.array(res["_y_pred"])+np.array(res["_residuals"]))),float(min(res["_y_pred"])))
        mx = max(float(max(np.array(res["_y_pred"])+np.array(res["_residuals"]))),float(max(res["_y_pred"])))
        axes[i].plot([mn,mx],[mn,mx],"r--",linewidth=1.2)
        star = "★ " if res["model"]==champion else ""
        axes[i].set_title(f"{star}{res['model']}\nR²={res['r2']}  RMSE={res['rmse']}", fontsize=9)
        axes[i].set_xlabel("Actual usr (%)"); axes[i].set_ylabel("Predicted usr (%)")
    for j in range(i+1,len(axes)): axes[j].set_visible(False)
    plt.suptitle("Actual vs Predicted — Tuned + Ensemble Models", fontsize=12, y=1.01)
    plt.tight_layout(); plt.savefig(out/"actual_vs_predicted.png", dpi=150, bbox_inches="tight"); plt.close()


def plot_leaderboard(test_results, champion, out):
    df = pd.DataFrame([{k:v for k,v in r.items()
                         if not k.startswith("_")} for r in test_results])
    df = df.sort_values("r2", ascending=False).reset_index(drop=True)
    bar_colors = ["#1d9e75" if nm==champion else "#4C78A8" for nm in df["model"]]
    fig, ax = plt.subplots(figsize=(11, max(4,len(df)*0.55+1.5)))
    ax.barh(df["model"], df["r2"], color=bar_colors, height=0.55)
    ax.set_xlabel("Test R²")
    ax.set_title("Model Leaderboard — Hold-out Test R²\n(green = champion)", fontsize=12)
    for i,(_, row) in enumerate(df.iterrows()):
        ax.text(row["r2"]+0.002, i,
                f"RMSE={row['rmse']}  MAE={row['mae']}  MedAE={row['medae']}",
                va="center", fontsize=8, color="#333")
    ax.set_xlim(0, 1.12)
    plt.tight_layout(); plt.savefig(out/"leaderboard.png", dpi=150); plt.close()


def plot_stat_tests(stat, out):
    if not stat or not stat.get("pairwise_vs_champion"): return
    df = pd.DataFrame(stat["pairwise_vs_champion"]).dropna(subset=["p_bonferroni"])
    df = df.sort_values("p_bonferroni")
    fig, ax = plt.subplots(figsize=(9, max(4, len(df)*0.48+1.5)))
    colors = ["#E45756" if r else "#54A24B" for r in df["significantly_worse"]]
    ax.barh(df["model"], -np.log10(df["p_bonferroni"].clip(1e-10)), color=colors)
    ax.axvline(-np.log10(ALPHA), color="black", linestyle="--",
               label=f"α={ALPHA} (Bonferroni)")
    ax.set_xlabel("−log₁₀(p Bonferroni)")
    ax.set_title(f"Wilcoxon Signed-Rank vs Champion ({stat['champion_model']})\n"
                 f"Red = significantly worse (α={ALPHA} Bonferroni)", fontsize=11)
    ax.legend(); plt.tight_layout()
    plt.savefig(out/"stat_test_results.png", dpi=150); plt.close()


def _all_models(tuned, ensembles):
    out = {n: i["best_estimator"] for n, i in tuned.items()}
    out.update({n: i["model"] for n, i in ensembles.items()})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Phase 10: HTML Report
# ─────────────────────────────────────────────────────────────────────────────
def build_html_report(screen_df, stat_tests, test_results, subgroups,
                      interval_coverage, champion, n_cv, out):
    def chip(v, lo=0.7, hi=0.85):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
        bg = "#c8f5c8" if v>=hi else ("#fff3c8" if v>=lo else "#f5c8c8")
        return (f'<span style="background:{bg};padding:2px 7px;border-radius:4px;'
                f'font-weight:600">{v:.3f}</span>')

    def chip_rmse(v, lo=5, hi=2):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
        bg = "#c8f5c8" if v<=hi else ("#fff3c8" if v<=lo else "#f5c8c8")
        return (f'<span style="background:{bg};padding:2px 7px;border-radius:4px;'
                f'font-weight:600">{v:.3f}</span>')

    def s_row(r):
        gap  = r.get("r2_overfit_gap", float("nan"))
        gcol = "#c0392b" if not np.isnan(gap) and gap>0.05 else "#27ae60"
        chk  = "🏆 " if r["model"]==champion else ""
        return (f"<tr><td><b>{chk}{r['model']}</b></td>"
                f"<td>{chip(r.get('r2_mean',float('nan')))}</td>"
                f"<td>{chip_rmse(abs(r.get('neg_rmse_mean',float('nan'))))}</td>"
                f"<td>{chip_rmse(abs(r.get('neg_mae_mean',float('nan'))))}</td>"
                f"<td>{chip_rmse(abs(r.get('neg_medae_mean',float('nan'))))}</td>"
                f"<td style='color:{gcol};font-weight:600'>"
                f"{'—' if np.isnan(gap) else f'{gap:.3f}'}</td>"
                f"<td>{r.get('cv_time_s','—')}</td></tr>")

    def t_row(r):
        chk = "🏆 " if r["model"]==champion else ""
        return (f"<tr><td><b>{chk}{r['model']}</b></td>"
                f"<td>{r['r2']}</td><td>{r['rmse']}</td>"
                f"<td>{r['mae']}</td><td>{r['medae']}</td>"
                f"<td>{r['mape']}</td><td>{r['residual_mean']}</td>"
                f"<td>{r['residual_std']}</td></tr>")

    s_rows = "\n".join(s_row(r) for _,r in
        screen_df.dropna(subset=["r2_mean"]).sort_values("r2_mean",ascending=False).iterrows())
    t_rows = "\n".join(t_row(r) for r in sorted(test_results, key=lambda x:x["r2"], reverse=True))

    stat_html = ""
    if stat_tests:
        fr  = stat_tests.get("friedman",{})
        sig = ('<span style="color:#c0392b;font-weight:bold">Significant</span>'
               if fr.get("significant") else
               '<span style="color:#27ae60">Not significant</span>')
        stat_html = f"""
        <h2>Phase 3 — Statistical significance</h2>
        <p class="note">Friedman χ² (H₀: all models equal) + pairwise Wilcoxon vs champion with Bonferroni correction (α={ALPHA}).</p>
        <p><b>Friedman:</b> χ²={fr.get('statistic','—')}, p={fr.get('p_value','—')}, {sig}</p>
        <p>Champion: <b>{stat_tests['champion_model']}</b></p>
        <img src="stat_test_results.png">"""

    sub_html = ""
    if subgroups.get("subgroups"):
        alerts = [r for r in subgroups["subgroups"] if r["alert"]]
        ap = (f'<p class="alert">⚠ {len(alerts)} load tier(s) with RMSE >25% above overall</p>'
              if alerts else '<p style="color:#27ae60">✓ No disparity alerts</p>')
        sub_rows = "\n".join(
            f"<tr><td>{r['group']}</td><td>{r['n']}</td>"
            f"<td>{r['mean_actual']}</td><td>{r['rmse']}</td>"
            f"<td style='color:{'#c0392b' if r['alert'] else '#27ae60'};font-weight:600'>"
            f"{r['rmse_gap']:+.4f}</td></tr>"
            for r in subgroups["subgroups"])
        sub_html = f"""
        <h2>Phase 9 — Subgroup disparity (load_tier)</h2>
        <p class="note">RMSE per load tier (idle/low/medium/high by runqsz). Alert = >25% above overall.</p>
        {ap}
        <table>
          <tr><th>Tier</th><th>N</th><th>Mean actual usr</th><th>RMSE</th><th>RMSE gap</th></tr>
          {sub_rows}
        </table>
        <img src="subgroup_rmse.png">"""

    ic_html = ""
    if interval_coverage:
        ic_rows = "\n".join(
            f"<tr><td>{'★ '+r['model'] if r['model']==champion else r['model']}</td>"
            f"<td>{r['within_1_rmse']:.3f} <small>(ideal 0.683)</small></td>"
            f"<td>{r['within_2_rmse']:.3f} <small>(ideal 0.954)</small></td>"
            f"<td>{r['within_3_rmse']:.3f}</td></tr>"
            for r in interval_coverage)
        ic_html = f"""
        <h2>Phase 7 — Prediction interval coverage</h2>
        <p class="note">% of test residuals within ±N×RMSE. For normal residuals: ±1 RMSE ≈ 68%, ±2 RMSE ≈ 95%.</p>
        <table>
          <tr><th>Model</th><th>Within ±1 RMSE</th><th>Within ±2 RMSE</th><th>Within ±3 RMSE</th></tr>
          {ic_rows}
        </table>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CPU Activity — Algorithm Benchmark Report</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f5f4f0;color:#1a1a18;padding:28px 36px 64px;max-width:1150px;margin:auto}}
h1{{font-size:26px;font-weight:700;border-bottom:3px solid #1d9e75;padding-bottom:10px;margin-bottom:6px}}
h2{{font-size:17px;font-weight:600;color:#0f5c3a;margin:36px 0 10px}}
p{{font-size:13px;color:#555;margin:4px 0 10px;line-height:1.65}}
.meta{{font-size:12px;color:#888;margin-bottom:28px}}
.note{{font-size:12px;color:#666;background:#fff;border-left:3px solid #1d9e75;padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:12px}}
.alert{{color:#c0392b;font-weight:600;margin-bottom:8px}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin-bottom:20px}}
th{{background:#0f5c3a;color:#fff;padding:8px 10px;text-align:left}}
td{{padding:6px 10px;border-bottom:0.5px solid #d3d1c7}}
tr:nth-child(even) td{{background:#f0efe9}}
img{{display:block;margin:14px 0 28px;border:0.5px solid #ccc;border-radius:8px;max-width:100%;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
footer{{margin-top:48px;font-size:11px;color:#aaa;border-top:0.5px solid #ddd;padding-top:12px;line-height:1.8}}
</style>
</head>
<body>
<h1>CPU Activity — Algorithm Benchmark Report</h1>
<p class="meta">
  Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
  &nbsp;|&nbsp; Champion: <b>{champion}</b>
  &nbsp;|&nbsp; Data: OpenML CPU Activity (data_id=197)
  &nbsp;|&nbsp; Reference: cpu_pipeline.py
</p>

<h2>Phase 1 — EDA (train-set only)</h2>
<p class="note">All EDA computed on the 80% training split only — no test-set leakage.
  Key constraint: sys + wait + usr ≈ 100 (CPU time decomposition).</p>
<img src="eda_target_distribution.png" style="max-width:500px">
<img src="eda_sys_vs_usr.png" style="max-width:500px">

<h2>Phase 2 — Algorithm Screening ({n_cv}-fold KFold CV)</h2>
<p class="note">All regressors share identical CPUFeatureEngineer + _DynamicNumericPreprocessor from the reference pipeline.
  Only the final estimator changes. Colour coding: green = good, yellow = acceptable, red = poor.</p>
<table>
  <tr><th>Model</th><th>R²</th><th>RMSE (neg)</th><th>MAE (neg)</th><th>MedAE (neg)</th>
      <th>Overfit Gap</th><th>CV Time (s)</th></tr>
  {s_rows}
</table>
<img src="screening_heatmap.png">
<img src="cv_boxplot_r2.png">
<img src="cv_boxplot_neg_rmse.png">
<img src="overfit_gap.png">

{stat_html}

<h2>Phase 4 — Hyperparameter Tuning (top-3)</h2>
<p class="note">RandomizedSearchCV with refit=r2 on same {n_cv}-fold CV.
  Only top-3 screening models are tuned — cost-efficient industry practice.</p>

<h2>Phase 5 — Ensemble Construction</h2>
<p class="note">AverageVoting (mean predictions) + Stacking (Ridge meta-learner, 3-fold OOF).</p>

<h2>Phase 6 — Hold-out Test Evaluation (8 metrics)</h2>
<p class="note">Evaluated once on the 20% held-out test set.
  R² = variance explained; RMSE = RMS error in % CPU; MAE = avg absolute % error;
  MedAE = median error (outlier-robust); MAPE = scale-free %.</p>
<table>
  <tr><th>Model</th><th>R²</th><th>RMSE</th><th>MAE</th><th>MedAE</th>
      <th>MAPE</th><th>Residual Mean</th><th>Residual Std</th></tr>
  {t_rows}
</table>
<img src="leaderboard.png">
<img src="actual_vs_predicted.png">

{ic_html}

<h2>Phase 7 — Residual Diagnostics</h2>
<p class="note">Residual distributions and scale-location plots for all tuned + ensemble models.</p>
<img src="residual_diagnostics.png">

<h2>Phase 8 — SHAP Explainability (Champion)</h2>
<p class="note">Global feature importance (bar + beeswarm) and waterfall for the worst-predicted test sample.</p>
<img src="shap_bar.png">
<img src="shap_beeswarm.png">

{sub_html}

<footer>
  <b>Metric guide (regression):</b><br>
  R² — proportion of target variance explained (1.0 = perfect). Scale-invariant, primary selection metric.<br>
  RMSE — root mean squared error in % CPU units. Penalises large errors quadratically.<br>
  MAE — mean absolute error in % CPU. Most interpretable for operations teams.<br>
  MedAE — median absolute error. Robust to outlier predictions during load spikes.<br>
  MAPE — mean absolute percentage error. Scale-free but unreliable near usr≈0.<br>
  <br>
  <b>Statistical tests:</b> Friedman χ² (global H₀) + Wilcoxon pairwise vs champion + Bonferroni (α={ALPHA}).<br>
  <b>Caution:</b> sys + wait + usr ≈ 100. Models that include sys/wait will have inflated R²
  — see model_card for deployment guidance.
</footer>
</body></html>"""

    path = out / "benchmark_report.html"
    path.write_text(html, encoding="utf-8")
    log.info("HTML report: %s", path.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def to_jsonable(v: Any) -> Any:
    if isinstance(v, dict):       return {str(k): to_jsonable(x) for k,x in v.items()}
    if isinstance(v, list):       return [to_jsonable(x) for x in v]
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


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(output_dir: Path, quick: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    t_start = time.perf_counter()

    n_splits = 3 if quick else 5
    n_iter   = 8 if quick else 30
    top_n    = 2 if quick else 3

    # ── Data ──────────────────────────────────────────────────────────────────
    df                            = fix_data_types(load_data())
    X_tr, X_te, y_tr, y_te       = split_data(df)

    # ── Phase 1: EDA ──────────────────────────────────────────────────────────
    log.info("═══ Phase 1: EDA ═══")
    save_research_artifacts(X_tr, y_tr, output_dir)

    # ── Phase 2: Screening ────────────────────────────────────────────────────
    regressors = get_regressors(quick)
    log.info("═══ Phase 2: Screening %d regressors (%d-fold KFold) ═══",
             len(regressors), n_splits)
    screen_df = screen_regressors(X_tr, y_tr, regressors, n_splits)
    (screen_df
     .drop(columns=[c for c in screen_df.columns if c.startswith("_")], errors="ignore")
     .sort_values("r2_mean", ascending=False)
     .to_csv(output_dir/"screening_results.csv", index=False))

    # ── Phase 3: Statistical tests ────────────────────────────────────────────
    log.info("═══ Phase 3: Statistical significance tests ═══")
    stat_tests = run_stat_tests(screen_df)
    write_json(output_dir/"stat_tests.json", stat_tests)

    # ── Phase 4: Tuning ───────────────────────────────────────────────────────
    log.info("═══ Phase 4: Tuning top-%d ═══", top_n)
    tuned = tune_top_models(screen_df, regressors, X_tr, y_tr, top_n, n_iter, n_splits)

    # ── Phase 5: Ensembles ────────────────────────────────────────────────────
    log.info("═══ Phase 5: Ensembles ═══")
    ensembles = build_ensembles(tuned, X_tr, y_tr)

    # ── Phase 6: Hold-out evaluation ─────────────────────────────────────────
    log.info("═══ Phase 6: Hold-out evaluation ═══")
    all_m        = _all_models(tuned, ensembles)
    test_results = []
    for name, model in all_m.items():
        r = evaluate_on_holdout(name, model, X_te, y_te)
        test_results.append(r)
        log.info("  %-22s  R²=%.4f  RMSE=%.4f  MAE=%.4f  MedAE=%.4f",
                 name, r["r2"], r["rmse"], r["mae"], r["medae"])

    champion_res = max(test_results, key=lambda r: r["r2"])
    champion     = champion_res["model"]
    log.info("Champion: %s (R²=%.4f, RMSE=%.4f)", champion, champion_res["r2"], champion_res["rmse"])

    # Save champion
    champion_model = all_m[champion]
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha1 = hashlib.sha1(pickle.dumps(champion_model)).hexdigest()[:8]
    joblib.dump(champion_model, output_dir/f"champion_{champion}_{ts}_{sha1}.joblib")
    joblib.dump(champion_model, output_dir/"champion_model.joblib")

    # ── Phase 7: Diagnostics ──────────────────────────────────────────────────
    log.info("═══ Phase 7: Diagnostics ═══")
    residual_summary_plot(test_results, output_dir)
    interval_coverage = prediction_interval_coverage(test_results, output_dir)

    # ── Phase 8: SHAP ─────────────────────────────────────────────────────────
    log.info("═══ Phase 8: SHAP ═══")
    compute_shap(champion_model, X_te, y_te,
                 champion_res["_y_pred"], output_dir)

    # ── Phase 9: Subgroups ────────────────────────────────────────────────────
    log.info("═══ Phase 9: Subgroup disparity ═══")
    subgroups = evaluate_subgroups(
        champion, champion_model, X_te, y_te,
        champion_res["_y_pred"], output_dir)

    # ── Remaining plots ───────────────────────────────────────────────────────
    log.info("═══ Generating plots ═══")
    plot_screening_heatmap(screen_df, output_dir)
    plot_cv_boxplot(screen_df, output_dir, "r2")
    plot_cv_boxplot(screen_df, output_dir, "neg_rmse")
    plot_overfit_gap(screen_df, output_dir)
    plot_actual_vs_predicted(test_results, champion, output_dir)
    plot_leaderboard(test_results, champion, output_dir)
    plot_stat_tests(stat_tests, output_dir)

    # ── Strip internal arrays before JSON ─────────────────────────────────────
    for r in test_results:
        r.pop("_y_pred", None); r.pop("_residuals", None)

    # ── JSON report ───────────────────────────────────────────────────────────
    report = {
        "generated_at":           datetime.now(timezone.utc).isoformat(),
        "champion":               champion,
        "elapsed_seconds":        round(time.perf_counter()-t_start,1),
        "data_source":            "fetch_openml(data_id=197)",
        "n_regressors_screened":  len(regressors),
        "cv_splits":              n_splits,
        "screening_summary": (
            screen_df
            .drop(columns=[c for c in screen_df.columns if c.startswith("_")], errors="ignore")
            .to_dict(orient="records")),
        "stat_tests":             stat_tests,
        "tuning_summary":         {n:{k:v for k,v in i.items() if k!="best_estimator"}
                                   for n,i in tuned.items()},
        "test_results":           test_results,
        "interval_coverage":      interval_coverage,
        "subgroups":              subgroups,
        "champion_metrics":       champion_res,
    }
    write_json(output_dir/"benchmark_report.json", report)

    # ── HTML report ───────────────────────────────────────────────────────────
    build_html_report(screen_df, stat_tests, test_results, subgroups,
                      interval_coverage, champion, n_splits, output_dir)

    log.info("═══ Benchmark complete in %.1fs — Champion: %s  R²=%.4f  RMSE=%.4f ═══",
             time.perf_counter()-t_start, champion,
             champion_res["r2"], champion_res["rmse"])
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="CPU Activity supervised learning benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path, default=Path("benchmark_cpu"),
                   help="Directory for all outputs")
    p.add_argument("--quick", action="store_true",
                   help="Smoke-test: 3-fold CV, 8 tune iters, top-2 models (~2 min)")
    args = p.parse_args()
    run_benchmark(args.output_dir, quick=args.quick)


if __name__ == "__main__":
    main()
