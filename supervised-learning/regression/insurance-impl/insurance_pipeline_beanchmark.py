"""
insurance_benchmark.py
======================
Industry-standard algorithm benchmarking script for Medical Insurance
charges prediction.

Built directly on insurance_pipeline.py:
  Reuses verbatim: load_data · fix_data_types · split_data
                   InsuranceFeatureEngineer · KFoldTargetEncoder
                   NUMERIC_RAW · CATEGORICAL_COLS · BMI_BINS · AGE_BINS
                   to_jsonable · write_json

10-phase professional benchmark workflow:
  Phase 1  — EDA (train-set only, from reference pipeline)
  Phase 2  — Screening: 16 regressors across every sklearn family
             + XGBoost + LightGBM  |  5-fold KFold · 6 regression metrics
  Phase 3  — Friedman χ² + Wilcoxon–Bonferroni statistical significance
  Phase 4  — RandomizedSearchCV tuning for top-3 (refit=r2)
  Phase 5  — Ensemble: VotingRegressor + StackingRegressor (Ridge meta)
  Phase 6  — Hold-out test: 8 metrics (R², Expl.Var, RMSE, MAE, MedAE,
             MAPE, Bias, Residual Std)
  Phase 7  — Prediction interval coverage calibration
  Phase 8  — SHAP explainability for champion
  Phase 9  — Subgroup disparity: smoker / region / sex + 25% RMSE alert
  Phase 10 — Self-contained HTML + JSON + CSV reports

sklearn families (https://scikit-learn.org/stable/supervised_learning.html):
  §1.1  Ridge · Lasso · ElasticNet · BayesianRidge · HuberRegressor · SGD
  §1.4  SVR (RBF)
  §1.6  KNeighborsRegressor
  §1.10 DecisionTreeRegressor
  §1.11 RandomForest · ExtraTrees · GradientBoosting · AdaBoost · Bagging
  §1.17 MLPRegressor
  Industry: XGBoost · LightGBM (conditional)

Insurance-specific notes:
  - wrap_regressor uses InsuranceFeatureEngineer + OHE ColumnTransformer
    (no SelectFromModel — clean head-to-head comparison)
  - Bias metric added: smokers systematically undercosted → important
  - Subgroups disaggregated by smoker (most important), region, sex

Usage:
  python insurance_benchmark.py                    # full run
  python insurance_benchmark.py --quick            # ~2 min
  python insurance_benchmark.py --output-dir ./results
  ML_N_JOBS=-1 python insurance_benchmark.py --quick
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

_MPLCONFIGDIR = Path("benchmark_insurance") / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, BaggingRegressor,
    ExtraTreesRegressor, GradientBoostingRegressor,
    RandomForestRegressor, StackingRegressor, VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    BayesianRidge, ElasticNet, HuberRegressor,
    Lasso, LinearRegression, Ridge, SGDRegressor,
)
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, median_absolute_error, r2_score,
)
from sklearn.model_selection import (
    KFold, RandomizedSearchCV, StratifiedKFold,
    cross_val_predict, cross_validate, train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor;   _XGB = True
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

# ── Constants (identical to reference pipeline) ───────────────────────────────
RANDOM_STATE     = 42
TARGET           = "charges"
N_JOBS           = int(os.environ.get("ML_N_JOBS", -1))
ALPHA            = 0.05

NUMERIC_RAW      = ["age", "bmi", "children"]
CATEGORICAL_COLS = ["sex", "smoker", "region"]
BINARY_CATS      = ["sex", "smoker"]
MULTICLASS_CATS  = ["region"]

BMI_BINS   = [0, 18.5, 25.0, 30.0, 100]
BMI_LABELS = ["underweight", "normal", "overweight", "obese"]
AGE_BINS   = [17, 25, 35, 50, 65, 100]
AGE_LABELS = ["young_adult", "adult", "middle_aged", "senior", "elderly"]

CV_SCORING = {
    "r2":            "r2",
    "neg_rmse":      "neg_root_mean_squared_error",
    "neg_mae":       "neg_mean_absolute_error",
    "neg_medae":     "neg_median_absolute_error",
    "neg_mape":      "neg_mean_absolute_percentage_error",
    "explained_var": "explained_variance",
}


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure — copied verbatim from insurance_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    _INSURANCE_COLS = {"age","bmi","children","smoker","sex","region","charges"}
    _CANDIDATE_IDS  = [44047, 44, 42477]
    _NAME_FALLBACKS = ["insurance","medical_insurance"]

    def _normalise(df):
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        for alt in ["target","class","label"]:
            if alt in df.columns and "charges" not in df.columns:
                df = df.rename(columns={alt:"charges"})
        for alt in ["bmi_score","body_mass_index"]:
            if alt in df.columns and "bmi" not in df.columns:
                df = df.rename(columns={alt:"bmi"})
        return df

    def _is_insurance(df):
        return len(_INSURANCE_COLS & set(df.columns)) >= 5

    for data_id in _CANDIDATE_IDS:
        try:
            log.info("Trying fetch_openml(data_id=%d) …", data_id)
            raw = fetch_openml(data_id=data_id, as_frame=True, parser="auto").frame
            df  = _normalise(raw)
            if _is_insurance(df):
                log.info("✓ data_id=%d  cols=%s", data_id, df.columns.tolist())
                for col in ["sex","smoker","region"]:
                    if col in df.columns:
                        df[col] = df[col].astype("category")
                return df
            log.warning("data_id=%d wrong dataset — trying next.", data_id)
        except Exception as exc:
            log.warning("data_id=%d failed: %s", data_id, exc)

    for name in _NAME_FALLBACKS:
        try:
            log.info("Trying fetch_openml(name='%s') …", name)
            raw = fetch_openml(name=name, as_frame=True, parser="auto").frame
            df  = _normalise(raw)
            if _is_insurance(df):
                for col in ["sex","smoker","region"]:
                    if col in df.columns:
                        df[col] = df[col].astype("category")
                return df
        except Exception as exc:
            log.warning("name='%s' failed: %s", name, exc)

    log.warning("All OpenML sources failed — using synthetic insurance data.")
    return _make_synthetic_insurance()


def _make_synthetic_insurance() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE); n = 1338
    age  = rng.integers(18,65,n).astype(float)
    bmi  = rng.normal(30.6,6.1,n).clip(15,55)
    children = rng.choice([0,1,2,3,4,5],n,p=[0.43,0.24,0.18,0.10,0.04,0.01]).astype(float)
    smoker = rng.choice(["yes","no"],n,p=[0.204,0.796])
    sex    = rng.choice(["male","female"],n)
    region = rng.choice(["northeast","northwest","southeast","southwest"],n)
    sf = (smoker=="yes").astype(float)
    charges = (2000+250*age+100*bmi+500*children+15000*sf
               +1000*(bmi>30)*sf*bmi+rng.exponential(2000,n)).clip(1000)
    df = pd.DataFrame({"age":age,"sex":sex,"bmi":bmi,"children":children,
                       "smoker":smoker,"region":region,"charges":charges})
    for col in ["sex","smoker","region"]:
        df[col] = df[col].astype("category")
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_RAW + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def split_data(df):
    X = df.drop(columns=[TARGET]); y = df[TARGET]
    q_bins = pd.qcut(y, q=4, labels=False, duplicates="drop")
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=q_bins)


def missingness_report(df):
    r = df.isna().agg(["sum","mean"]).T.rename(
        columns={"sum":"missing_count","mean":"missing_rate"})
    r["dtype"] = df.dtypes.astype(str)
    return r.sort_values("missing_rate",ascending=False)


class InsuranceFeatureEngineer(BaseEstimator, TransformerMixin):
    """Copied verbatim from insurance_pipeline.py (category dtype-safe)."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        rebuilt = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                rebuilt[col] = pd.to_numeric(X[col],errors="coerce").astype(float).values
            else:
                rebuilt[col] = X[col].astype(str).values
        X = pd.DataFrame(rebuilt, index=X.index)

        def _flt(col):
            if col not in X.columns: return np.zeros(len(X),dtype=np.float64)
            arr = pd.to_numeric(X[col],errors="coerce").astype(float).values
            med = float(np.nanmedian(arr))
            return np.where(np.isnan(arr),med,arr)

        age = _flt("age"); bmi = _flt("bmi")
        smoker_flag = (np.where(X["smoker"].str.lower().str.strip()=="yes",1.0,0.0)
                       .astype(np.float64) if "smoker" in X.columns
                       else np.zeros(len(X),dtype=np.float64))

        X["bmi_sq"]              = bmi**2
        X["age_sq"]              = age**2
        X["age_bmi_interact"]    = age*bmi
        X["bmi_smoker_interact"] = bmi*smoker_flag
        X["age_smoker_interact"] = age*smoker_flag
        X["bmi_age_smoker_triple"] = bmi*age*smoker_flag
        X["bmi_category_num"]    = pd.cut(bmi,bins=BMI_BINS,labels=range(len(BMI_LABELS)),right=False).astype(float)
        X["age_group_num"]       = pd.cut(age,bins=AGE_BINS,labels=range(len(AGE_LABELS)),right=True).astype(float)
        X["high_bmi_smoker"]     = ((bmi>=30)&(smoker_flag==1)).astype(float)
        X["senior_flag"]         = (age>=50).astype(float)
        return X


def build_ohe_preprocessor() -> ColumnTransformer:
    """OHE ColumnTransformer from insurance_pipeline.py."""
    engineered_num = [
        "age","bmi","children","bmi_sq","age_sq","age_bmi_interact",
        "bmi_smoker_interact","age_smoker_interact","bmi_age_smoker_triple",
        "bmi_category_num","age_group_num","high_bmi_smoker","senior_flag",
    ]
    num_pipe    = Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),
                             ("sc", RobustScaler())])
    binary_pipe = Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                             ("ohe",OneHotEncoder(drop="if_binary",handle_unknown="ignore",sparse_output=False))])
    multi_pipe  = Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                             ("ohe",OneHotEncoder(drop="first",handle_unknown="ignore",sparse_output=False))])
    return ColumnTransformer([
        ("num",    num_pipe,    engineered_num),
        ("binary", binary_pipe, BINARY_CATS),
        ("multi",  multi_pipe,  MULTICLASS_CATS),
    ], remainder="drop")


def wrap_regressor(reg: BaseEstimator) -> Pipeline:
    """
    Full reference-stack wrapper:
      InsuranceFeatureEngineer → OHE ColumnTransformer → regressor
    No SelectFromModel — all models see same features for clean comparison.
    """
    return Pipeline([
        ("feature_engineering", InsuranceFeatureEngineer()),
        ("preprocess",          build_ohe_preprocessor()),
        ("model",               reg),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: EDA
# ─────────────────────────────────────────────────────────────────────────────
def save_research_artifacts(X_train, y_train, output_dir):
    log.info("EDA artifacts (train-set only) …")
    output_dir.mkdir(parents=True, exist_ok=True)
    eda = X_train.copy(); eda[TARGET] = y_train.values

    missingness_report(eda).to_csv(output_dir/"eda_missingness.csv")

    # Numeric correlation
    num_df = eda.select_dtypes(include=[np.number])
    corr   = num_df.corr()[TARGET].drop(TARGET).sort_values(key=abs,ascending=False)
    corr.to_csv(output_dir/"eda_correlation_with_target.csv")

    # Grouped stats
    grouped = {}
    for col in CATEGORICAL_COLS:
        if col in eda.columns:
            grouped[f"charges_by_{col}"] = (
                eda.groupby(col,observed=False)[TARGET]
                .agg(["mean","median","std"]).to_dict())
    grouped["target_stats"] = y_train.describe().to_dict()
    write_json(output_dir/"eda_grouped_stats.json", grouped)

    sns.set_theme(style="whitegrid")

    # Target distribution
    plt.figure(figsize=(7,4))
    sns.histplot(y_train,kde=True,bins=40,color="#4C78A8")
    plt.axvline(y_train.median(),color="#E45756",linestyle="--",
                label=f"Median=${y_train.median():,.0f}")
    plt.title("charges distribution — bimodal (smoker vs non-smoker)")
    plt.xlabel("charges ($)"); plt.legend()
    plt.tight_layout(); plt.savefig(output_dir/"eda_target_distribution.png",dpi=150); plt.close()

    # Smoker boxplot
    if "smoker" in eda.columns:
        plt.figure(figsize=(6,4))
        sns.boxplot(data=eda,x="smoker",y=TARGET,palette=["#4C78A8","#E45756"])
        plt.title("charges by smoker status — key cost driver")
        plt.tight_layout(); plt.savefig(output_dir/"eda_charges_by_smoker.png",dpi=150); plt.close()

    # BMI × smoker scatter
    if "bmi" in eda.columns and "smoker" in eda.columns:
        plt.figure(figsize=(7,5))
        for sval,color in [("yes","#E45756"),("no","#4C78A8")]:
            sub = eda[eda["smoker"].astype(str)==sval]
            plt.scatter(sub["bmi"],sub[TARGET],alpha=0.35,s=10,color=color,label=f"smoker={sval}")
        plt.xlabel("BMI"); plt.ylabel("charges ($)")
        plt.title("BMI × smoker interaction — obese smokers cost ~4× non-smokers")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir/"eda_bmi_smoker.png",dpi=150); plt.close()

    # Correlation bar
    plt.figure(figsize=(6,4))
    corr.plot(kind="barh",color=["#54A24B" if v>0 else "#E45756" for v in corr])
    plt.axvline(0,color="black",linewidth=0.8)
    plt.title("Feature Pearson r with charges (train)")
    plt.tight_layout(); plt.savefig(output_dir/"eda_correlation.png",dpi=150); plt.close()

    log.info("EDA saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Regressor registry
# ─────────────────────────────────────────────────────────────────────────────
def get_regressors(quick=False) -> dict[str,BaseEstimator]:
    """
    16 regressors covering every sklearn supervised learning family.
    https://scikit-learn.org/stable/supervised_learning.html

    §1.1  Linear: Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, SGD
    §1.4  SVM:    SVR (RBF kernel)
    §1.6  KNN:    KNeighborsRegressor (distance-weighted)
    §1.10 Tree:   DecisionTreeRegressor
    §1.11 Ensemble: RandomForest, ExtraTrees, GradientBoosting, AdaBoost, Bagging
    §1.17 Neural: MLPRegressor (128→64 ReLU, early stopping)
    Industry: XGBoost, LightGBM (conditional on install)
    """
    n = 50 if quick else 150
    regs: dict[str,BaseEstimator] = {
        # ── §1.1 Linear ──────────────────────────────────────────────────────
        "Ridge":          Ridge(alpha=1.0),
        "Lasso":          Lasso(alpha=10.0, max_iter=5000),
        "ElasticNet":     ElasticNet(alpha=10.0, l1_ratio=0.5, max_iter=5000),
        "BayesianRidge":  BayesianRidge(),
        "HuberRegressor": HuberRegressor(epsilon=1.35, max_iter=300),
        "SGD":            SGDRegressor(loss="huber", penalty="elasticnet",
                              l1_ratio=0.15, max_iter=2000, random_state=RANDOM_STATE),
        # ── §1.4 SVM ─────────────────────────────────────────────────────────
        "SVR_RBF":        SVR(kernel="rbf", C=1000.0, epsilon=500.0),
        # ── §1.6 KNN ─────────────────────────────────────────────────────────
        "KNN":            KNeighborsRegressor(n_neighbors=7, weights="distance",
                              n_jobs=N_JOBS),
        # ── §1.10 Tree ───────────────────────────────────────────────────────
        "DecisionTree":   DecisionTreeRegressor(max_depth=6, min_samples_leaf=4,
                              random_state=RANDOM_STATE),
        # ── §1.11 Ensembles ──────────────────────────────────────────────────
        "RandomForest":   RandomForestRegressor(n_estimators=n, max_depth=10,
                              min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "ExtraTrees":     ExtraTreesRegressor(n_estimators=n, max_depth=10,
                              min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=n, max_depth=4,
                              learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE),
        "AdaBoost":       AdaBoostRegressor(n_estimators=n, learning_rate=0.5,
                              random_state=RANDOM_STATE),
        "Bagging":        BaggingRegressor(n_estimators=n, random_state=RANDOM_STATE,
                              n_jobs=N_JOBS),
        # ── §1.17 Neural Network ─────────────────────────────────────────────
        "MLP":            MLPRegressor(hidden_layer_sizes=(128,64), activation="relu",
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


PARAM_GRIDS: dict[str,Any] = {
    "Ridge":            {"model__alpha":[0.01,0.1,1,10,50,100,500,1000,5000]},
    "Lasso":            {"model__alpha":[1,5,10,50,100,500,1000]},
    "ElasticNet":       {"model__alpha":[1,10,50,100,500],
                         "model__l1_ratio":[0.1,0.3,0.5,0.7,0.9]},
    "BayesianRidge":    {"model__alpha_1":[1e-6,1e-4,1e-2],
                         "model__lambda_1":[1e-6,1e-4,1e-2]},
    "HuberRegressor":   {"model__epsilon":[1.1,1.35,1.5,2.0],
                         "model__alpha":[0.0001,0.001,0.01]},
    "SGD":              {"model__alpha":np.logspace(-4,1,8).tolist(),
                         "model__l1_ratio":[0.1,0.3,0.5,0.7,0.9]},
    "SVR_RBF":          {"model__C":[100,500,1000,5000,10000],
                         "model__gamma":["scale","auto",0.001,0.01],
                         "model__epsilon":[100,300,500,1000]},
    "KNN":              {"model__n_neighbors":[3,5,7,11,15],
                         "model__weights":["uniform","distance"]},
    "DecisionTree":     {"model__max_depth":[4,6,8,12,None],
                         "model__min_samples_leaf":[1,2,4,8]},
    "RandomForest":     {"model__n_estimators":[100,200,300],
                         "model__max_depth":[6,8,12,None],
                         "model__min_samples_leaf":[1,2,4]},
    "ExtraTrees":       {"model__n_estimators":[100,200,300],
                         "model__max_depth":[6,8,12,None],
                         "model__min_samples_leaf":[1,2,4]},
    "GradientBoosting": {"model__n_estimators":[100,200,300],
                         "model__max_depth":[3,4,5],
                         "model__learning_rate":[0.02,0.05,0.1,0.2],
                         "model__subsample":[0.7,0.9]},
    "AdaBoost":         {"model__n_estimators":[50,100,200],
                         "model__learning_rate":[0.05,0.1,0.5,1.0]},
    "Bagging":          {"model__n_estimators":[50,100,200]},
    "MLP":              {"model__hidden_layer_sizes":[(64,),(128,),(128,64),(256,128)],
                         "model__alpha":[0.0001,0.001,0.01]},
    "XGBoost":          {"model__n_estimators":[100,200,300],
                         "model__max_depth":[3,4,5,6],
                         "model__learning_rate":[0.02,0.05,0.1,0.2],
                         "model__subsample":[0.7,0.9],
                         "model__colsample_bytree":[0.6,0.8,1.0]},
    "LightGBM":         {"model__n_estimators":[100,200,300],
                         "model__max_depth":[3,4,5,6],
                         "model__learning_rate":[0.02,0.05,0.1,0.2],
                         "model__subsample":[0.7,0.9]},
}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: CV Screening
# ─────────────────────────────────────────────────────────────────────────────
def screen_regressors(X_tr, y_tr, regressors, n_splits):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, reg in regressors.items():
        log.info("  [Screen] %-22s …", name)
        t0 = time.perf_counter()
        try:
            res = cross_validate(
                wrap_regressor(clone(reg)), X_tr, y_tr,
                cv=cv, scoring=CV_SCORING,
                return_train_score=True, n_jobs=1, error_score="raise")
            elapsed = time.perf_counter()-t0
            row = {"model":name,"cv_time_s":round(elapsed,2)}
            for m in CV_SCORING:
                ts=res[f"test_{m}"]; trs=res[f"train_{m}"]
                row[f"{m}_mean"]        = float(ts.mean())
                row[f"{m}_std"]         = float(ts.std())
                row[f"{m}_train_mean"]  = float(trs.mean())
                row[f"{m}_overfit_gap"] = float(trs.mean()-ts.mean())
                row[f"_raw_{m}"]        = ts.tolist()
        except Exception as exc:
            log.warning("    %s FAILED: %s", name, exc)
            row = {"model":name,"cv_time_s":-1,
                   **{f"{m}_mean":np.nan for m in CV_SCORING}}
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Statistical significance
# ─────────────────────────────────────────────────────────────────────────────
def run_stat_tests(screen_df, metric="r2"):
    raw_col = f"_raw_{metric}"
    valid   = screen_df.dropna(subset=[f"{metric}_mean"]).copy()
    if raw_col not in valid.columns or len(valid)<2: return {}
    arrays  = [np.array(r) for r in valid[raw_col]]
    names   = valid["model"].tolist()
    best_i  = int(valid[f"{metric}_mean"].idxmax())
    champ   = valid.loc[best_i,"model"]
    champ_sc= arrays[list(valid.index).index(best_i)]
    try:
        f_stat,f_p = friedmanchisquare(*arrays)
    except Exception:
        f_stat,f_p = np.nan,np.nan
    n_comp = len(arrays)-1; pairwise=[]
    for nm,sc in zip(names,arrays):
        if nm==champ: continue
        try:
            diff=champ_sc-sc
            stat,p=(wilcoxon(diff,alternative="greater",zero_method="wilcox")
                    if not np.all(diff==0) else (np.nan,1.0))
        except Exception:
            stat,p=np.nan,np.nan
        p_bonf=float(min(1.0,p*n_comp)) if not np.isnan(p) else np.nan
        pairwise.append({
            "model":nm,
            "wilcoxon_stat":float(stat) if not np.isnan(stat) else None,
            "p_value":float(p) if not np.isnan(p) else None,
            "p_bonferroni":float(p_bonf) if not np.isnan(p_bonf) else None,
            "significantly_worse":bool(p_bonf<ALPHA) if not np.isnan(p_bonf) else None,
        })
    return {
        "metric":metric,"champion_model":champ,
        "friedman":{"statistic":float(f_stat) if not np.isnan(f_stat) else None,
                    "p_value":float(f_p) if not np.isnan(f_p) else None,
                    "significant":bool(f_p<ALPHA) if not np.isnan(f_p) else None},
        "pairwise_vs_champion":pairwise,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Hyperparameter tuning
# ─────────────────────────────────────────────────────────────────────────────
def tune_top_models(screen_df, regressors, X_tr, y_tr, top_n, n_iter, n_splits):
    cv  = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    top = (screen_df.dropna(subset=["r2_mean"])
           .sort_values("r2_mean",ascending=False)
           .head(top_n)["model"].tolist())
    log.info("Top-%d for tuning: %s", top_n, top)
    tuned: dict[str,Any] = {}
    for name in top:
        pipe = wrap_regressor(clone(regressors[name]))
        if name not in PARAM_GRIDS:
            log.info("  [Tune] %-22s — no grid, defaults.", name)
            pipe.fit(X_tr,y_tr)
            tuned[name]={"tuned":False,"best_estimator":pipe}; continue
        log.info("  [Tune] %-22s (n_iter=%d) …", name, n_iter)
        try:
            search = RandomizedSearchCV(
                pipe, PARAM_GRIDS[name], n_iter=n_iter,
                scoring="r2", cv=cv, refit=True,
                random_state=RANDOM_STATE, n_jobs=N_JOBS, error_score="raise")
            search.fit(X_tr,y_tr)
            tuned[name]={"tuned":True,"best_params":search.best_params_,
                         "best_cv_r2":float(search.best_score_),
                         "best_estimator":search.best_estimator_}
            log.info("    Best CV R²: %.4f", search.best_score_)
        except Exception as exc:
            log.warning("    Tuning failed: %s", exc)
            pipe.fit(X_tr,y_tr)
            tuned[name]={"tuned":False,"best_estimator":pipe}
    return tuned


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Ensembles
# ─────────────────────────────────────────────────────────────────────────────
def build_ensembles(tuned, X_tr, y_tr):
    estimators = [(n,info["best_estimator"]) for n,info in tuned.items()]
    if len(estimators)<2: return {}
    results: dict[str,Any] = {}
    for ename,Cls,kwargs in [
        ("AverageVoting", VotingRegressor, {"n_jobs":N_JOBS}),
        ("Stacking", StackingRegressor,
         {"final_estimator":Ridge(alpha=1.0),"cv":3,"n_jobs":N_JOBS}),
    ]:
        try:
            log.info("  [Ensemble] %s …", ename)
            m = Cls(estimators=estimators,**kwargs)
            m.fit(X_tr,y_tr); results[ename]={"model":m}
        except Exception as exc:
            log.warning("  %s failed: %s", ename, exc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: Hold-out evaluation — 8 metrics
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_on_holdout(name, model, X_te, y_te):
    y_pred    = model.predict(X_te)
    residuals = y_te.to_numpy()-y_pred
    rmse      = float(np.sqrt(mean_squared_error(y_te,y_pred)))
    bias      = float(residuals.mean())   # positive = underpredicts, negative = overpredicts
    return {
        "model":name,
        "r2":            round(float(r2_score(y_te,y_pred)),4),
        "explained_var": round(float(1-np.var(residuals)/np.var(y_te)),4),
        "rmse":          round(rmse,2),
        "mae":           round(float(mean_absolute_error(y_te,y_pred)),2),
        "medae":         round(float(median_absolute_error(y_te,y_pred)),2),
        "mape":          round(float(mean_absolute_percentage_error(y_te,y_pred)),4),
        "bias":          round(bias,2),
        "residual_std":  round(float(residuals.std()),2),
        "_y_pred":    y_pred,
        "_residuals": residuals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7: Prediction interval calibration
# ─────────────────────────────────────────────────────────────────────────────
def prediction_interval_coverage(test_results, output_dir):
    """
    For well-calibrated normally distributed residuals:
      ±1 RMSE ≈ 68%   ±2 RMSE ≈ 95%
    Insurance residuals are right-skewed (smoker/non-smoker modes) →
    expect lower ±1 RMSE coverage than normal. Documents how much.
    """
    rows = []
    for res in test_results:
        r=np.abs(np.array(res["_residuals"])); rmse=res["rmse"]
        rows.append({
            "model":res["model"],
            "within_1_rmse":round(float((r<=rmse).mean()),4),
            "within_2_rmse":round(float((r<=2*rmse).mean()),4),
            "within_3_rmse":round(float((r<=3*rmse).mean()),4),
            "ideal_1_rmse":0.683,"ideal_2_rmse":0.954,
        })
    pd.DataFrame(rows).to_csv(output_dir/"prediction_interval_coverage.csv",index=False)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8: SHAP for champion
# ─────────────────────────────────────────────────────────────────────────────
def compute_shap(model, X_te, y_te, y_pred, output_dir):
    if not _SHAP: log.warning("pip install shap"); return
    log.info("SHAP for champion …")
    try:
        clf  = model.named_steps["model"]
        prep = model.named_steps["preprocess"]
        fe   = model.named_steps["feature_engineering"]
        Xt   = prep.transform(fe.transform(X_te))
        try:
            fn = prep.get_feature_names_out()
        except Exception:
            fn = np.array([f"f{i}" for i in range(Xt.shape[1])])
        Xdf  = pd.DataFrame(Xt,columns=fn)
        if hasattr(clf,"feature_importances_"):
            exp=shap.TreeExplainer(clf); sv=exp.shap_values(Xdf)
        elif hasattr(clf,"coef_"):
            exp=shap.LinearExplainer(clf,Xdf); sv=exp.shap_values(Xdf)
        else:
            mask=shap.maskers.Independent(Xdf,max_samples=100)
            exp=shap.Explainer(clf.predict,mask); sv=exp(Xdf).values
        for ptype,fname in [("bar","shap_bar.png"),("dot","shap_beeswarm.png")]:
            plt.figure(figsize=(10,6))
            shap.summary_plot(sv,Xdf,plot_type=ptype,show=False,max_display=20)
            plt.title(f"SHAP champion — {ptype}")
            plt.tight_layout()
            plt.savefig(output_dir/fname,dpi=150,bbox_inches="tight"); plt.close()
        worst=int(np.argmax(np.abs(y_te.to_numpy()-y_pred)))
        ev=(float(exp.expected_value) if not isinstance(exp.expected_value,np.ndarray)
            else float(exp.expected_value))
        shap.waterfall_plot(
            shap.Explanation(values=sv[worst],base_values=ev,
                             data=Xdf.iloc[worst].values,feature_names=list(fn)),
            show=False,max_display=15)
        plt.title("SHAP Waterfall — highest charges residual")
        plt.tight_layout()
        plt.savefig(output_dir/"shap_waterfall.png",dpi=150,bbox_inches="tight"); plt.close()
        pd.DataFrame({"feature":fn,"mean_abs_shap":np.abs(sv).mean(axis=0)}
            ).sort_values("mean_abs_shap",ascending=False
            ).to_csv(output_dir/"shap_importance.csv",index=False)
        log.info("SHAP saved.")
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 9: Subgroup disparity
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_subgroups(champion_name, champion_model, X_te, y_te, y_pred, output_dir):
    overall_rmse = float(np.sqrt(mean_squared_error(y_te,y_pred)))
    overall_r2   = float(r2_score(y_te,y_pred))
    eval_df      = X_te.reset_index(drop=True).copy()
    eval_df["_y_true"]=y_te.to_numpy(); eval_df["_y_pred"]=y_pred
    rows = []
    for col in ["smoker","region","sex"]:
        if col not in eval_df.columns: continue
        for val,sub in eval_df.groupby(col,observed=True):
            if len(sub)<10: continue
            sr=float(np.sqrt(mean_squared_error(sub["_y_true"],sub["_y_pred"])))
            rows.append({
                "group_col":col,"group_val":str(val),"n":int(len(sub)),
                "mean_actual":round(float(sub["_y_true"].mean()),2),
                "rmse":round(sr,2),
                "r2":round(float(r2_score(sub["_y_true"],sub["_y_pred"])),4),
                "rmse_gap":round(sr-overall_rmse,2),
                "alert":bool(sr>overall_rmse*1.25),
            })
    if rows:
        rdf=pd.DataFrame(rows)
        rdf.to_csv(output_dir/"subgroup_report.csv",index=False)
        g=rdf.copy(); g["label"]=g["group_col"]+"="+g["group_val"].astype(str)
        plt.figure(figsize=(10,max(4,len(g)*0.46)))
        colors=["#E45756" if a else "#4C78A8" for a in g["alert"]]
        plt.barh(g["label"],g["rmse"],color=colors)
        plt.axvline(overall_rmse,linestyle="--",color="black",label=f"Overall RMSE=${overall_rmse:,.0f}")
        plt.xlabel("RMSE ($)"); plt.title(f"Subgroup RMSE — Champion: {champion_name}\n(red = >25% above overall)")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir/"subgroup_rmse.png",dpi=150); plt.close()
    return {"overall_rmse":overall_rmse,"overall_r2":overall_r2,"subgroups":rows}


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_screening_heatmap(df,out):
    cols=["r2_mean","neg_rmse_mean","neg_mae_mean","neg_medae_mean","explained_var_mean"]
    d=df.dropna(subset=["r2_mean"]).set_index("model")[cols].copy()
    d.columns=["R²","RMSE(neg)","MAE(neg)","MedAE(neg)","Expl.Var"]
    d=d.sort_values("R²",ascending=False)
    fig,ax=plt.subplots(figsize=(11,max(5,len(d)*0.48+1.5)))
    sns.heatmap(d,annot=True,fmt=".3f",cmap="RdYlGn",linewidths=0.4,ax=ax,annot_kws={"size":8})
    ax.set_title("Algorithm Screening — CV Metric Heatmap (sorted by R²)",fontsize=12,pad=10)
    plt.tight_layout(); plt.savefig(out/"screening_heatmap.png",dpi=150); plt.close()


def plot_cv_boxplot(df,out,metric="r2"):
    raw=f"_raw_{metric}"
    if raw not in df.columns: return
    v=df.dropna(subset=[f"{metric}_mean"]).sort_values(f"{metric}_mean",ascending=False)
    data=[np.array(r) for r in v[raw]]; names=v["model"].tolist()
    fig,ax=plt.subplots(figsize=(max(10,len(names)*0.75),5))
    bp=ax.boxplot(data,patch_artist=True,widths=0.55)
    cmap=plt.cm.get_cmap("RdYlGn",len(names))
    for i,(patch,med) in enumerate(zip(bp["boxes"],bp["medians"])):
        patch.set_facecolor(cmap(i/len(names))); med.set_color("black"); med.set_linewidth(1.5)
    ax.set_xticks(range(1,len(names)+1))
    ax.set_xticklabels(names,rotation=38,ha="right",fontsize=9)
    ax.set_ylabel(metric.replace("_"," ").title())
    ax.set_title(f"CV Distribution — {metric.upper()}",fontsize=12)
    ax.grid(axis="y",alpha=0.3)
    plt.tight_layout(); plt.savefig(out/f"cv_boxplot_{metric}.png",dpi=150); plt.close()


def plot_overfit_gap(df,out):
    v=df.dropna(subset=["r2_mean"]).sort_values("r2_mean",ascending=False)
    gaps=v["r2_overfit_gap"]
    colors=["#E45756" if g>0.05 else "#54A24B" for g in gaps]
    fig,ax=plt.subplots(figsize=(max(10,len(v)*0.75),4))
    ax.bar(range(len(v)),gaps,color=colors,width=0.6)
    ax.axhline(0,color="black",linewidth=0.8)
    ax.axhline(0.05,color="#E45756",linewidth=0.8,linestyle="--",alpha=0.6,label="Alert 0.05")
    ax.set_xticks(range(len(v)))
    ax.set_xticklabels(v["model"].tolist(),rotation=38,ha="right",fontsize=9)
    ax.set_ylabel("Train R² − CV R²")
    ax.set_title("Overfitting Analysis — Train–CV R² Gap\n(red = gap > 0.05)",fontsize=12)
    ax.legend(); plt.tight_layout()
    plt.savefig(out/"overfit_gap.png",dpi=150); plt.close()


def plot_actual_vs_predicted(test_results,champion,out):
    n=len(test_results); cols=min(3,n); rows=(n+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(cols*4.5,rows*4.2))
    axes=np.array(axes).flatten()
    for i,res in enumerate(test_results):
        actual=np.array(res["_y_pred"])+np.array(res["_residuals"])
        pred=np.array(res["_y_pred"])
        axes[i].scatter(actual,pred,alpha=0.3,s=8)
        mn,mx=float(min(actual.min(),pred.min())),float(max(actual.max(),pred.max()))
        axes[i].plot([mn,mx],[mn,mx],"r--",linewidth=1.2)
        star="★ " if res["model"]==champion else ""
        axes[i].set_title(f"{star}{res['model']}\nR²={res['r2']}  RMSE=${res['rmse']:,.0f}",fontsize=9)
        axes[i].set_xlabel("Actual ($)"); axes[i].set_ylabel("Predicted ($)")
    for j in range(i+1,len(axes)): axes[j].set_visible(False)
    plt.suptitle("Actual vs Predicted — Tuned + Ensemble Models",fontsize=12,y=1.01)
    plt.tight_layout()
    plt.savefig(out/"actual_vs_predicted.png",dpi=150,bbox_inches="tight"); plt.close()


def plot_residual_summary(test_results,out):
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    cmap=plt.cm.get_cmap("tab10")
    for i,res in enumerate(test_results):
        r=np.array(res["_residuals"])
        axes[0].hist(r,bins=40,alpha=0.5,label=res["model"],color=cmap(i%10),density=True)
        axes[1].scatter(np.array(res["_y_pred"]),r,alpha=0.2,s=6,color=cmap(i%10),label=res["model"])
    axes[0].axvline(0,color="black",linewidth=1.2)
    axes[0].set_xlabel("Residual ($)"); axes[0].set_ylabel("Density")
    axes[0].set_title("Residual distributions — tuned models"); axes[0].legend(fontsize=7)
    axes[1].axhline(0,color="black",linewidth=1.0)
    axes[1].set_xlabel("Predicted ($)"); axes[1].set_ylabel("Residual ($)")
    axes[1].set_title("Residuals vs Predicted"); axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out/"residual_diagnostics.png",dpi=150); plt.close()


def plot_leaderboard(test_results,champion,out):
    df=pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in test_results])
    df=df.sort_values("r2",ascending=False).reset_index(drop=True)
    colors=["#1d9e75" if nm==champion else "#4C78A8" for nm in df["model"]]
    fig,ax=plt.subplots(figsize=(11,max(4,len(df)*0.55+1.5)))
    ax.barh(df["model"],df["r2"],color=colors,height=0.55)
    ax.set_xlabel("Test R²")
    ax.set_title("Model Leaderboard — Hold-out Test R²\n(green = champion)",fontsize=12)
    for i,(_,row) in enumerate(df.iterrows()):
        ax.text(row["r2"]+0.002,i,
                f"RMSE=${row['rmse']:,.0f}  MAE=${row['mae']:,.0f}  Bias=${row['bias']:+,.0f}",
                va="center",fontsize=8,color="#333")
    ax.set_xlim(0,1.18); plt.tight_layout()
    plt.savefig(out/"leaderboard.png",dpi=150); plt.close()


def plot_stat_tests(stat,out):
    if not stat or not stat.get("pairwise_vs_champion"): return
    df=pd.DataFrame(stat["pairwise_vs_champion"]).dropna(subset=["p_bonferroni"])
    df=df.sort_values("p_bonferroni")
    fig,ax=plt.subplots(figsize=(9,max(4,len(df)*0.48+1.5)))
    colors=["#E45756" if r else "#54A24B" for r in df["significantly_worse"]]
    ax.barh(df["model"],-np.log10(df["p_bonferroni"].clip(1e-10)),color=colors)
    ax.axvline(-np.log10(ALPHA),color="black",linestyle="--",label=f"α={ALPHA} Bonferroni")
    ax.set_xlabel("−log₁₀(p Bonferroni)")
    ax.set_title(f"Wilcoxon vs Champion ({stat['champion_model']})\nRed = significantly worse",fontsize=11)
    ax.legend(); plt.tight_layout()
    plt.savefig(out/"stat_test_results.png",dpi=150); plt.close()


def _all_models(tuned,ensembles):
    out={n:i["best_estimator"] for n,i in tuned.items()}
    out.update({n:i["model"] for n,i in ensembles.items()})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Phase 10: HTML Report
# ─────────────────────────────────────────────────────────────────────────────
def build_html_report(screen_df,stat_tests,test_results,subgroups,
                      interval_coverage,champion,n_cv,out):
    def chip_r2(v):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
        bg="#c8f5c8" if v>=0.85 else ("#fff3c8" if v>=0.70 else "#f5c8c8")
        return f'<span style="background:{bg};padding:2px 8px;border-radius:4px;font-weight:600">{v:.4f}</span>'

    def chip_err(v,good=1500,ok=3000):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
        v2=abs(v)
        bg="#c8f5c8" if v2<=good else ("#fff3c8" if v2<=ok else "#f5c8c8")
        return f'<span style="background:{bg};padding:2px 8px;border-radius:4px;font-weight:600">${v2:,.0f}</span>'

    def s_row(r):
        gap=r.get("r2_overfit_gap",float("nan"))
        gcol="#c0392b" if not np.isnan(gap) and gap>0.05 else "#27ae60"
        chk="🏆 " if r["model"]==champion else ""
        return (f"<tr><td><b>{chk}{r['model']}</b></td>"
                f"<td>{chip_r2(r.get('r2_mean',float('nan')))}</td>"
                f"<td>{chip_err(abs(r.get('neg_rmse_mean',float('nan'))))}</td>"
                f"<td>{chip_err(abs(r.get('neg_mae_mean',float('nan'))))}</td>"
                f"<td>{chip_err(abs(r.get('neg_medae_mean',float('nan'))))}</td>"
                f"<td style='color:{gcol};font-weight:600'>{'—' if np.isnan(gap) else f'{gap:.4f}'}</td>"
                f"<td>{r.get('cv_time_s','—')}</td></tr>")

    def t_row(r):
        chk="🏆 " if r["model"]==champion else ""
        bias_col="#c0392b" if abs(r["bias"])>500 else "#27ae60"
        return (f"<tr><td><b>{chk}{r['model']}</b></td>"
                f"<td>{chip_r2(r['r2'])}</td>"
                f"<td>{chip_err(r['rmse'])}</td><td>{chip_err(r['mae'])}</td>"
                f"<td>{chip_err(r['medae'])}</td>"
                f"<td>{r['mape']:.4f}</td>"
                f"<td style='color:{bias_col}'>${r['bias']:+,.0f}</td>"
                f"<td>${r['residual_std']:,.0f}</td>"
                f"<td>{r['explained_var']:.4f}</td></tr>")

    s_rows="\n".join(s_row(r) for _,r in
        screen_df.dropna(subset=["r2_mean"]).sort_values("r2_mean",ascending=False).iterrows())
    t_rows="\n".join(t_row(r) for r in sorted(test_results,key=lambda x:x["r2"],reverse=True))

    stat_html=""
    if stat_tests:
        fr=stat_tests.get("friedman",{})
        sig=('<span style="color:#c0392b;font-weight:bold">Significant</span>'
             if fr.get("significant") else
             '<span style="color:#27ae60">Not significant</span>')
        fstat=fr.get('statistic')
        fp=fr.get('p_value')
        stat_html=f"""
        <h2>Phase 3 — Statistical Significance</h2>
        <p class="note">Friedman χ² (H₀: all models equal) + pairwise Wilcoxon vs champion
        + Bonferroni correction (α={ALPHA}).</p>
        <p><b>Friedman:</b> χ²={f'{fstat:.3f}' if fstat else '—'}, p={f'{fp:.4e}' if fp else '—'}, {sig}</p>
        <p>Champion: <b>{stat_tests['champion_model']}</b></p>
        <img src="stat_test_results.png">"""

    sub_html=""
    if subgroups.get("subgroups"):
        alerts=[r for r in subgroups["subgroups"] if r["alert"]]
        ap=(f'<p class="alert">⚠ {len(alerts)} subgroup(s) with RMSE >25% above overall</p>'
            if alerts else '<p style="color:#27ae60">✓ No disparity alerts</p>')
        sub_rows="\n".join(
            f"<tr><td>{r['group_col']}={r['group_val']}</td><td>{r['n']}</td>"
            f"<td>${r['mean_actual']:,.0f}</td><td>${r['rmse']:,.0f}</td>"
            f"<td style='color:{'#c0392b' if r['alert'] else '#27ae60'};font-weight:600'>"
            f"${r['rmse_gap']:+,.0f}</td><td>{r['r2']:.4f}</td></tr>"
            for r in subgroups["subgroups"])
        sub_html=f"""
        <h2>Phase 9 — Subgroup Disparity (smoker / region / sex)</h2>
        <p class="note">RMSE per group. Alert = >25% above overall.
        Smoker flag is the largest cost driver in insurance pricing.</p>
        {ap}
        <table>
          <tr><th>Group</th><th>N</th><th>Mean charges</th><th>RMSE</th>
              <th>RMSE gap</th><th>R²</th></tr>
          {sub_rows}
        </table>
        <img src="subgroup_rmse.png">"""

    ic_html=""
    if interval_coverage:
        ic_rows="\n".join(
            f"<tr><td>{'★ '+r['model'] if r['model']==champion else r['model']}</td>"
            f"<td>{r['within_1_rmse']:.3f} <small>(ideal {r['ideal_1_rmse']})</small></td>"
            f"<td>{r['within_2_rmse']:.3f} <small>(ideal {r['ideal_2_rmse']})</small></td>"
            f"<td>{r['within_3_rmse']:.3f}</td></tr>"
            for r in interval_coverage)
        ic_html=f"""
        <h2>Phase 7 — Prediction Interval Calibration</h2>
        <p class="note">% of test residuals within ±N×RMSE. Ideal (Gaussian): ±1≈68%, ±2≈95%.
        Insurance residuals are right-skewed (bimodal smoker distribution) so expect lower ±1 coverage.</p>
        <table>
          <tr><th>Model</th><th>Within ±1 RMSE</th><th>Within ±2 RMSE</th><th>Within ±3 RMSE</th></tr>
          {ic_rows}
        </table>"""

    html=f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Insurance Charges — Algorithm Benchmark Report</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f5f4f0;color:#1a1a18;
     padding:28px 36px 64px;max-width:1150px;margin:auto}}
h1{{font-size:26px;font-weight:700;border-bottom:3px solid #1d9e75;padding-bottom:10px;margin-bottom:6px}}
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
img{{display:block;margin:14px 0 28px;border:0.5px solid #ccc;border-radius:8px;
     max-width:100%;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
footer{{margin-top:48px;font-size:11px;color:#aaa;border-top:0.5px solid #ddd;padding-top:12px;line-height:1.8}}
</style>
</head>
<body>
<h1>Medical Insurance Charges — Algorithm Benchmark Report</h1>
<p class="meta">
  Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
  &nbsp;|&nbsp; Champion: <b>{champion}</b>
  &nbsp;|&nbsp; Data: OpenML Insurance (data_id=44047)
  &nbsp;|&nbsp; Reference: insurance_pipeline.py
</p>

<h2>Phase 1 — EDA (train-set only)</h2>
<p class="note">All EDA on the 80% training split — zero test-set leakage.
  Key driver: smoker status (3.5× average cost difference).</p>
<img src="eda_target_distribution.png" style="max-width:500px">
<img src="eda_bmi_smoker.png" style="max-width:500px">

<h2>Phase 2 — Algorithm Screening ({n_cv}-fold KFold)</h2>
<p class="note">All regressors share identical InsuranceFeatureEngineer + OHE ColumnTransformer
  from insurance_pipeline.py. Only the final estimator changes.
  Green = strong (R²≥0.85), Yellow = acceptable, Red = weak.</p>
<table>
  <tr><th>Model</th><th>R²</th><th>RMSE</th><th>MAE</th><th>MedAE</th>
      <th>Overfit gap</th><th>CV time (s)</th></tr>
  {s_rows}
</table>
<img src="screening_heatmap.png">
<img src="cv_boxplot_r2.png">
<img src="cv_boxplot_neg_rmse.png">
<img src="overfit_gap.png">

{stat_html}

<h2>Phase 4 — Hyperparameter Tuning (top-3)</h2>
<p class="note">RandomizedSearchCV with refit=r² on same {n_cv}-fold CV.
  Only top-3 screened models are tuned — cost-efficient industry practice.</p>

<h2>Phase 5 — Ensemble Construction</h2>
<p class="note">AverageVoting (mean predictions) + Stacking (Ridge meta, 3-fold OOF)
  from the tuned top-3 estimators.</p>

<h2>Phase 6 — Hold-out Test Evaluation (8 metrics)</h2>
<p class="note">
  <b>R²</b> = variance explained &nbsp;|&nbsp;
  <b>RMSE</b> = RMS error in USD &nbsp;|&nbsp;
  <b>MAE</b> = avg absolute USD error &nbsp;|&nbsp;
  <b>MedAE</b> = median error (robust to outlier smoker predictions) &nbsp;|&nbsp;
  <b>MAPE</b> = scale-free % (unreliable near low charges) &nbsp;|&nbsp;
  <b>Bias</b> = systematic over/under-prediction — red if |bias|>$500 &nbsp;|&nbsp;
  <b>Residual Std</b> = random error spread
</p>
<table>
  <tr><th>Model</th><th>R²</th><th>RMSE</th><th>MAE</th><th>MedAE</th>
      <th>MAPE</th><th>Bias</th><th>Residual Std</th><th>Expl.Var</th></tr>
  {t_rows}
</table>
<img src="leaderboard.png">
<img src="actual_vs_predicted.png">
<img src="residual_diagnostics.png">

{ic_html}

<h2>Phase 8 — SHAP Explainability (Champion)</h2>
<p class="note">Global feature importance + individual prediction waterfall for worst-predicted case.</p>
<img src="shap_bar.png">
<img src="shap_beeswarm.png">

{sub_html}

<footer>
  <b>Metric guide:</b><br>
  R² — proportion of target variance explained. Primary selection metric. Scale-invariant.<br>
  RMSE — root mean squared error in USD. Penalises large errors (e.g. catastrophic misses on smokers).<br>
  MAE — mean absolute error in USD. Most interpretable for actuaries and product managers.<br>
  MedAE — median absolute error. Robust to the bimodal distribution (smoker/non-smoker cost gap).<br>
  Bias (residual mean) — positive = model systematically underestimates charges; negative = overestimates.<br>
  Interval coverage — ±1×RMSE should contain ≈68% of errors for normally distributed residuals.<br>
  <br>
  <b>Statistical tests:</b> Friedman χ² + pairwise Wilcoxon vs champion + Bonferroni correction (α={ALPHA}).<br>
  <b>Ethics note:</b> Do not use sex or region as individual pricing factors. Smoker status
  should comply with applicable insurance regulations.
</footer>
</body></html>"""

    path=out/"benchmark_report.html"
    path.write_text(html,encoding="utf-8")
    log.info("HTML report: %s", path.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# Utilities — identical to reference pipeline
# ─────────────────────────────────────────────────────────────────────────────
def to_jsonable(v):
    if isinstance(v,dict):       return {str(k):to_jsonable(x) for k,x in v.items()}
    if isinstance(v,list):       return [to_jsonable(x) for x in v]
    if isinstance(v,BaseEstimator): return repr(v)
    if isinstance(v,np.bool_):   return bool(v)
    if isinstance(v,np.integer): return int(v)
    if isinstance(v,np.floating):
        f=float(v); return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,float):
        return None if (np.isnan(v) or np.isinf(v)) else v
    try:
        if pd.isna(v): return None
    except: pass
    return v


def write_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload),indent=2),encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(output_dir: Path, quick: bool = False) -> dict[str,Any]:
    output_dir.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style="whitegrid")
    t_start = time.perf_counter()

    n_splits = 3 if quick else 5
    n_iter   = 8 if quick else 30
    top_n    = 2 if quick else 3

    # ── Data ──────────────────────────────────────────────────────────────────
    df                      = fix_data_types(load_data())
    X_tr, X_te, y_tr, y_te = split_data(df)

    # ── Phase 1: EDA ──────────────────────────────────────────────────────────
    log.info("═══ Phase 1: EDA ═══")
    save_research_artifacts(X_tr, y_tr, output_dir)

    # ── Phase 2: Screening ────────────────────────────────────────────────────
    regressors = get_regressors(quick)
    log.info("═══ Phase 2: Screening %d regressors (%d-fold KFold) ═══",
             len(regressors), n_splits)
    screen_df = screen_regressors(X_tr, y_tr, regressors, n_splits)
    (screen_df
     .drop(columns=[c for c in screen_df.columns if c.startswith("_")],errors="ignore")
     .sort_values("r2_mean",ascending=False)
     .to_csv(output_dir/"screening_results.csv",index=False))

    # ── Phase 3: Stat tests ───────────────────────────────────────────────────
    log.info("═══ Phase 3: Statistical tests ═══")
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
    for name,model in all_m.items():
        r = evaluate_on_holdout(name,model,X_te,y_te)
        test_results.append(r)
        log.info("  %-22s  R²=%.4f  RMSE=%s  MAE=%s  Bias=%s",
                 name, r["r2"],
                 f"${r['rmse']:,.0f}", f"${r['mae']:,.0f}",
                 f"${r['bias']:+,.0f}")

    champion_res = max(test_results,key=lambda r:r["r2"])
    champion     = champion_res["model"]
    log.info("Champion: %s  R²=%.4f  RMSE=$%s", champion,
             champion_res["r2"], f"{champion_res['rmse']:,.0f}")

    champion_model = all_m[champion]
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha1 = hashlib.sha1(pickle.dumps(champion_model)).hexdigest()[:8]
    joblib.dump(champion_model, output_dir/f"champion_{champion}_{ts}_{sha1}.joblib")
    joblib.dump(champion_model, output_dir/"champion_model.joblib")

    # ── Phase 7: Calibration ─────────────────────────────────────────────────
    log.info("═══ Phase 7: Interval calibration ═══")
    interval_coverage = prediction_interval_coverage(test_results,output_dir)

    # ── Phase 8: SHAP ─────────────────────────────────────────────────────────
    log.info("═══ Phase 8: SHAP ═══")
    compute_shap(champion_model,X_te,y_te,champion_res["_y_pred"],output_dir)

    # ── Phase 9: Subgroups ────────────────────────────────────────────────────
    log.info("═══ Phase 9: Subgroup disparity ═══")
    subgroups = evaluate_subgroups(
        champion,champion_model,X_te,y_te,champion_res["_y_pred"],output_dir)

    # ── Plots ─────────────────────────────────────────────────────────────────
    log.info("═══ Generating plots ═══")
    plot_screening_heatmap(screen_df,output_dir)
    plot_cv_boxplot(screen_df,output_dir,"r2")
    plot_cv_boxplot(screen_df,output_dir,"neg_rmse")
    plot_overfit_gap(screen_df,output_dir)
    plot_residual_summary(test_results,output_dir)
    plot_actual_vs_predicted(test_results,champion,output_dir)
    plot_leaderboard(test_results,champion,output_dir)
    plot_stat_tests(stat_tests,output_dir)

    # ── Strip internal arrays ─────────────────────────────────────────────────
    for r in test_results:
        r.pop("_y_pred",None); r.pop("_residuals",None)

    # ── JSON report ───────────────────────────────────────────────────────────
    report = {
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "champion":              champion,
        "elapsed_seconds":       round(time.perf_counter()-t_start,1),
        "data_source":           "fetch_openml(data_id=44047) / insurance",
        "n_regressors_screened": len(regressors),
        "cv_splits":             n_splits,
        "screening_summary": (
            screen_df
            .drop(columns=[c for c in screen_df.columns if c.startswith("_")],errors="ignore")
            .to_dict(orient="records")),
        "stat_tests":            stat_tests,
        "tuning_summary":        {n:{k:v for k,v in i.items() if k!="best_estimator"}
                                  for n,i in tuned.items()},
        "test_results":          test_results,
        "interval_coverage":     interval_coverage,
        "subgroups":             subgroups,
        "champion_metrics":      champion_res,
    }
    write_json(output_dir/"benchmark_report.json",report)
    build_html_report(screen_df,stat_tests,test_results,subgroups,
                      interval_coverage,champion,n_splits,output_dir)

    log.info("═══ Done in %.1fs — Champion: %s  R²=%.4f  RMSE=$%s ═══",
             time.perf_counter()-t_start, champion,
             champion_res["r2"], f"{champion_res['rmse']:,.0f}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p=argparse.ArgumentParser(
        description="Medical Insurance charges supervised learning benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir",type=Path,default=Path("benchmark_insurance"))
    p.add_argument("--quick",action="store_true",
                   help="Smoke-test: 3-fold, 8 tune iters, top-2 (~2 min)")
    args=p.parse_args()
    run_benchmark(args.output_dir,quick=args.quick)

if __name__=="__main__":
    main()
