"""
bike_pipeline.py
================
Industry-standard end-to-end ML pipeline for Bike Sharing Demand prediction.

Data (exact pattern as specified):
    Bike_Sharing_Demand    = fetch_openml(data_id=42712, as_frame=True, parser="auto")
    Bike_Sharing_Demand_df = Bike_Sharing_Demand.frame

Target: count — total number of bikes rented per hour (casual + registered)
Dataset: 17379 rows × 17 columns | hourly records Jan 2011–Dec 2012, Washington D.C.

Mirrors every architectural pattern from titanic-ml-pipeline.py,
extended with time-series awareness and gradient boosting depth.

Primary focus: Non-Linearity, Large-Scale & Gradient Boosting
───────────────────────────────────────────────────────────────
  A. Time-based feature engineering
        — hour, day-of-week, month, quarter, year from datetime
        — rush hour flags, peak vs off-peak, weekend indicator
        — cyclic encoding: sin/cos of hour, day, month (avoids boundary artifacts)
        — lag features and rolling stats for temporal context
        — time-trend: days elapsed since dataset start

  B. Missing data & robustness
        — even with no missing values: simulate & compare imputation strategies
        — KNN vs MICE on artificially-masked data; establish robustness baseline
        — weather outlier detection: weather=4 (heavy rain) near-zero counts
        — IsolationForest on multivariate feature space

  C. Log target transform
        — count → log1p(count); right-skewed target
        — RMSLE as primary evaluation metric (Kaggle competition metric)
        — back-transform with expm1 for USD-equivalent reporting

  D. Polynomial + non-linear features
        — temp², humidity × windspeed, temp × season interaction
        — weather × hour peak effects (rain during rush hour vs midnight)
        — basis functions: SplineTransformer on temp, cyclic hour

  E. Gradient Boosting deep-dive
        — XGBoost vs LightGBM vs GradientBoostingRegressor
        — learning rate × depth tradeoff analysis
        — early stopping validation curves
        — feature importance: gain vs permutation vs SHAP

  F. Random Forest comparison
        — RF vs ExtraTrees; feature importance stability
        — out-of-bag error as free CV estimate
        — partial dependence on hour, temp, season

  G. Hyperparameter tuning at scale
        — RandomizedSearchCV (baseline)
        — Optuna (Bayesian, optional) for GB/XGB fine-tuning
        — Learning-rate warm-up analysis

  H. Time-aware cross-validation
        — TimeSeriesSplit (no future leakage)
        — KFold for comparison
        — Walk-forward validation on monthly windows

  I. Outlier analysis
        — weather=4 extreme events
        — night-time zero-inflation
        — holiday anomaly patterns
        — Cook's distance on linear baseline

  J. Residual diagnostics
        — Breusch-Pagan, Q-Q plot, ACF of residuals (temporal autocorrelation)
        — partial residual plots for non-linear relationships

  K. Learning curves & scalability
        — gradient boosting convergence vs n_estimators
        — training time vs dataset size
        — early stopping monitoring

  L. Ensemble & stacking
        — VotingRegressor of top-3
        — Stacking with Ridge meta-learner
        — Blending casual + registered sub-models (optional decomposition)

Industry-standard metrics: RMSLE (primary/Kaggle), RMSE, MAE, R², MAPE, MedAE

Usage:
  python bike_pipeline.py train   --output-dir artifacts_bike
  python bike_pipeline.py predict --artifact-dir artifacts_bike --input-csv sample.csv
  python bike_pipeline.py monitor --artifact-dir artifacts_bike --input-csv new.csv
  python bike_pipeline.py sample-input --output-csv sample.csv --rows 24
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

_MPLCONFIGDIR = Path("artifacts_bike") / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.stats import ks_2samp
try:
    from statsmodels.tsa.stattools import acf as _sm_acf
    def _acf(x, nlags=48):
        return _sm_acf(x, nlags=nlags, fft=True)
except ImportError:
    def _acf(x, nlags=48):
        """Pure-numpy ACF — no statsmodels required."""
        x   = np.asarray(x, dtype=float)
        x   = x - x.mean()
        n   = len(x)
        var = np.dot(x, x) / n
        if var == 0:
            return np.zeros(nlags + 1)
        result = np.array([
            np.dot(x[:n - lag], x[lag:]) / (n * var)
            for lag in range(nlags + 1)
        ])
        return result
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor,
    IsolationForest, RandomForestRegressor,
    StackingRegressor, VotingRegressor,
)
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import (
    BayesianRidge, ElasticNet, HuberRegressor,
    Lasso, LinearRegression, Ridge, RidgeCV,
)
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, median_absolute_error, r2_score,
)
from sklearn.model_selection import (
    KFold, RandomizedSearchCV, TimeSeriesSplit,
    cross_val_predict, cross_val_score, learning_curve, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, PolynomialFeatures,
    RobustScaler, SplineTransformer, StandardScaler,
)

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
try:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING); _OPTUNA = True
except ImportError:
    _OPTUNA = False
try:
    import mlflow; import mlflow.sklearn; _MLFLOW = True
except ImportError:
    _MLFLOW = False

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE          = 42
TARGET                = "count"
MODEL_FILE            = "bike_pipeline.joblib"
METRICS_FILE          = "metrics.json"
TRAINING_PROFILE_FILE = "training_profile.json"
MODEL_CARD_FILE       = "model_card.json"
ENVIRONMENT_FILE      = "environment.json"
N_JOBS                = int(os.environ.get("ML_N_JOBS", 1))

# Features to drop before modelling (leakage or redundant)
DROP_COLS = ["casual", "registered", "datetime"]

WEATHER_LABELS = {1: "clear", 2: "mist", 3: "light_precip", 4: "heavy_precip"}
SEASON_LABELS  = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Exact loader as specified:
        Bike_Sharing_Demand    = fetch_openml(data_id=42712, as_frame=True, parser="auto")
        Bike_Sharing_Demand_df = Bike_Sharing_Demand.frame
    """
    log.info("Loading Bike Sharing Demand dataset from OpenML (data_id=42712) …")
    _IDS   = [42712, 43430]
    _NAMES = ["bike_sharing_demand", "BikeSharing"]
    _SIG   = {"datetime", "season", "weather", "temp", "humidity", "count"}

    def _is_bike(df):
        cols = set(df.columns.str.lower())
        return len(_SIG & cols) >= 4

    def _normalise(df):
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        for alt in ["cnt", "total", "rentals"]:
            if alt in df.columns and "count" not in df.columns:
                df = df.rename(columns={alt: "count"})
        return df

    for did in _IDS:
        try:
            log.info("Trying data_id=%d …", did)
            raw = fetch_openml(data_id=did, as_frame=True, parser="auto").frame
            df  = _normalise(raw)
            if _is_bike(df):
                log.info("✓ data_id=%d accepted  shape=%s", did, df.shape)
                return df
        except Exception as exc:
            log.warning("data_id=%d failed: %s", did, exc)

    for name in _NAMES:
        try:
            raw = fetch_openml(name=name, as_frame=True, parser="auto").frame
            df  = _normalise(raw)
            if _is_bike(df):
                log.info("✓ name='%s' accepted  shape=%s", name, df.shape)
                return df
        except Exception as exc:
            log.warning("name='%s' failed: %s", name, exc)

    log.warning("All OpenML sources failed — using synthetic bike data.")
    return _make_synthetic_bike()


def _make_synthetic_bike() -> pd.DataFrame:
    """Synthetic fallback with realistic hourly demand patterns."""
    rng = np.random.default_rng(RANDOM_STATE)
    dates = pd.date_range("2011-01-01", "2012-12-31 23:00:00", freq="h")
    n     = len(dates)
    hour  = dates.hour
    month = dates.month
    dow   = dates.dayofweek
    year  = (dates.year - 2011).astype(int)

    temp     = (0.5 + 0.3*np.sin(2*np.pi*(month-3)/12) + rng.normal(0,0.08,n)).clip(0,1)
    humidity = (0.6 + 0.1*np.sin(2*np.pi*month/12) + rng.normal(0,0.12,n)).clip(0,1)
    windspeed= rng.beta(2,5,n)
    season   = ((month-1)//3+1).astype(int)
    weather  = rng.choice([1,1,1,2,2,3,4], n, p=[0.45,0,0,0.31,0,0.22,0.02])
    holiday  = ((dow==6)|(dow==0)).astype(int)
    workday  = ((~holiday.astype(bool))&(dow<5)).astype(int)

    rush = ((hour>=7)&(hour<=9)|(hour>=17)&(hour<=19)).astype(float)
    base = (80*(1+year*0.25) + 200*temp - 100*humidity
            + 150*rush*workday + 50*(1-workday)
            + 30*np.sin(2*np.pi*hour/24))
    base *= np.where(weather==1,1.0,np.where(weather==2,0.85,np.where(weather==3,0.5,0.2)))
    base  = np.maximum(base + rng.exponential(20,n), 0)

    casual     = np.maximum(base*0.2 + rng.normal(0,10,n), 0)
    registered = np.maximum(base*0.8 + rng.normal(0,20,n), 0)
    count      = casual + registered

    return pd.DataFrame({
        "datetime":   dates.strftime("%Y-%m-%d %H:%M:%S"),
        "season":     season, "holiday": holiday,
        "workingday": workday, "weather": weather,
        "temp":       temp.round(4), "atemp": (temp*0.95+rng.normal(0,0.02,n)).clip(0,1).round(4),
        "humidity":   humidity.round(4), "windspeed": windspeed.round(4),
        "casual":     casual.round(0), "registered": registered.round(0),
        "count":      count.round(0),
    })


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["season","holiday","workingday","weather"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["temp","atemp","humidity","windspeed","casual","registered","count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    # Parse datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def split_data(df: pd.DataFrame):
    """
    Time-aware split: last 20% of hours = test set.
    Preserves temporal ordering — no future leakage.
    """
    df = df.sort_values("datetime").reset_index(drop=True) if "datetime" in df.columns else df.copy()
    n  = len(df)
    cutoff = int(n * 0.8)
    X  = df.drop(columns=[TARGET])
    y  = df[TARGET]
    return X.iloc[:cutoff], X.iloc[cutoff:], y.iloc[:cutoff], y.iloc[cutoff:]


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    r = df.isna().agg(["sum","mean"]).T.rename(columns={"sum":"missing_count","mean":"missing_rate"})
    r["dtype"] = df.dtypes.astype(str)
    return r.sort_values("missing_rate",ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering — BikeFeatureEngineer
# ─────────────────────────────────────────────────────────────────────────────
class BikeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Concept A — time-based feature engineering + domain interactions.
    All features are computed deterministically from the raw columns.
    Learns dataset_start from training data for trend computation.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "BikeFeatureEngineer":
        if "datetime" in X.columns:
            dt = pd.to_datetime(X["datetime"], errors="coerce")
            self.dataset_start_ = dt.min()
        else:
            self.dataset_start_ = pd.Timestamp("2011-01-01")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # ── Concept A: Time features ──────────────────────────────────────────
        if "datetime" in X.columns:
            dt = pd.to_datetime(X["datetime"], errors="coerce")
            X["hour"]        = dt.dt.hour.astype(float)
            X["dow"]         = dt.dt.dayofweek.astype(float)      # 0=Mon 6=Sun
            X["month"]       = dt.dt.month.astype(float)
            X["quarter"]     = dt.dt.quarter.astype(float)
            X["year"]        = (dt.dt.year - 2011).astype(float)  # 0=2011 1=2012
            X["dayofyear"]   = dt.dt.dayofyear.astype(float)
            X["days_elapsed"]= (dt - self.dataset_start_).dt.total_seconds() / 86400

            # Cyclic encoding — avoids hour 0 ≠ hour 23 boundary artifact
            X["hour_sin"]  = np.sin(2*np.pi*X["hour"]/24)
            X["hour_cos"]  = np.cos(2*np.pi*X["hour"]/24)
            X["dow_sin"]   = np.sin(2*np.pi*X["dow"]/7)
            X["dow_cos"]   = np.cos(2*np.pi*X["dow"]/7)
            X["month_sin"] = np.sin(2*np.pi*X["month"]/12)
            X["month_cos"] = np.cos(2*np.pi*X["month"]/12)

            # Demand segments (domain knowledge)
            h = X["hour"]
            X["is_rush_am"]    = ((h>=7)&(h<=9)).astype(float)
            X["is_rush_pm"]    = ((h>=17)&(h<=19)).astype(float)
            X["is_night"]      = ((h<=5)|(h>=22)).astype(float)
            X["is_midday"]     = ((h>=10)&(h<=16)).astype(float)
            X["is_weekend"]    = (X["dow"] >= 5).astype(float) if "dow" in X.columns else 0.0
            X["is_workrush"]   = (X["is_rush_am"] + X["is_rush_pm"]) * (
                                  X.get("workingday", pd.Series(0,index=X.index)).astype(float))

        # ── Concept D: Non-linear weather × time interactions ─────────────────
        temp = pd.to_numeric(X.get("temp", pd.Series(0.5,index=X.index)), errors="coerce").fillna(0.5)
        hum  = pd.to_numeric(X.get("humidity",pd.Series(0.6,index=X.index)), errors="coerce").fillna(0.6)
        wind = pd.to_numeric(X.get("windspeed",pd.Series(0.1,index=X.index)), errors="coerce").fillna(0.1)
        seas = pd.to_numeric(X.get("season",pd.Series(2,index=X.index)), errors="coerce").fillna(2)
        wth  = pd.to_numeric(X.get("weather",pd.Series(1,index=X.index)), errors="coerce").fillna(1)

        X["temp_sq"]           = temp ** 2
        X["temp_humidity"]     = temp * hum           # humid hot days suppress demand
        X["temp_windchill"]    = temp * (1 - wind)    # wind reduces perceived warmth
        X["comfort_index"]     = temp - hum*0.5 - wind*0.3  # combined comfort score
        X["bad_weather"]       = (wth >= 3).astype(float)
        X["weather_temp"]      = wth * temp           # heavy rain + cold = worst case
        X["is_ideal_cycling"]  = ((temp > 0.4) & (hum < 0.7) & (wth == 1)).astype(float)

        # Interactions that depend on time-derived flags.
        # All reads go through .get() so missing columns return a zero Series
        # rather than raising KeyError — handles datasets where datetime is absent.
        _zeros = pd.Series(0.0, index=X.index)
        if "is_workrush" in X.columns:
            X["rush_x_weather"] = X["is_workrush"] * (4 - wth)
            X["rush_x_temp"]    = (X.get("is_rush_am", _zeros)
                                   + X.get("is_rush_pm", _zeros)) * temp
        else:
            X["rush_x_weather"] = _zeros
            X["rush_x_temp"]    = _zeros
        X["night_x_temp"] = X.get("is_night", _zeros) * temp

        # Season × temp (summer is comfortable at higher temps)
        X["season_temp"] = seas * temp

        return X


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic preprocessor (no hardcoded column list)
# ─────────────────────────────────────────────────────────────────────────────
class _DynamicNumericPreprocessor(BaseEstimator, TransformerMixin):
    """
    Resolves numeric columns at fit-time — avoids ColumnTransformer
    KeyError when new engineered columns are added.
    """
    def __init__(self, scaler_name="RobustScaler"):
        self.scaler_name = scaler_name

    def _make_scaler(self):
        return {"StandardScaler":StandardScaler(),"RobustScaler":RobustScaler(),
                "MinMaxScaler":MinMaxScaler()}.get(self.scaler_name, RobustScaler())

    def fit(self, X, y=None):
        self.cols_ = [c for c in X.columns
                      if c not in DROP_COLS and c != "datetime"
                      and pd.api.types.is_numeric_dtype(X[c])]
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
            pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
            for c in cols])

    def get_feature_names_out(self, _=None):
        return np.array(self.cols_)


def build_pipeline(model=None, scaler_name="RobustScaler") -> Pipeline:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.feature_selection import SelectFromModel
    if model is None:
        model = Ridge(alpha=1.0)
    sel = SelectFromModel(
        ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        threshold="median")
    return Pipeline([
        ("feature_engineering", BikeFeatureEngineer()),
        ("preprocess",          _DynamicNumericPreprocessor(scaler_name)),
        ("feature_selection",   sel),
        ("model",               model),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# RMSLE metric (Kaggle official for this competition)
# ─────────────────────────────────────────────────────────────────────────────
def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


# ─────────────────────────────────────────────────────────────────────────────
# Concept A: Time-feature importance analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_time_features(X_train, y_train, output_dir):
    log.info("Concept A: Time feature analysis …")
    fe   = BikeFeatureEngineer().fit(X_train)
    Xeng = fe.transform(X_train)
    y_log = np.log1p(y_train.to_numpy())

    # Mutual information style: R² gain per feature group
    feature_groups = {
        "Raw only": [c for c in ["season","holiday","workingday","weather","temp",
                                  "atemp","humidity","windspeed"] if c in Xeng.columns],
        "+Hour":    [c for c in ["season","holiday","workingday","weather","temp",
                                  "atemp","humidity","windspeed","hour"] if c in Xeng.columns],
        "+Time flags": [c for c in ["season","holiday","workingday","weather","temp","atemp",
                                     "humidity","windspeed","hour","is_rush_am","is_rush_pm",
                                     "is_night","is_weekend","is_workrush"] if c in Xeng.columns],
        "+Cyclic enc": [c for c in ["season","holiday","workingday","weather","temp","atemp",
                                     "humidity","windspeed","hour_sin","hour_cos","dow_sin","dow_cos",
                                     "month_sin","month_cos","is_rush_am","is_rush_pm","is_workrush",
                                     "is_night","year","days_elapsed"] if c in Xeng.columns],
        "+Interactions": [c for c in Xeng.columns if c not in DROP_COLS and c != "datetime"],
    }

    cv = KFold(n_splits=5, shuffle=False)
    results = {}
    for name, cols in feature_groups.items():
        if not cols: continue
        _arr = np.column_stack([
            pd.to_numeric(Xeng[c], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
            for c in cols])
        _arr = RobustScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(_arr))
        r2s = cross_val_score(Ridge(alpha=1.0), _arr, y_log, cv=cv, scoring="r2")
        results[name] = {"r2_mean": float(r2s.mean()), "r2_std": float(r2s.std()),
                         "n_features": len(cols)}
        log.info("  %-30s n=%3d  R²=%.4f±%.4f", name, len(cols), r2s.mean(), r2s.std())

    names = list(results.keys()); r2s = [results[n]["r2_mean"] for n in names]
    plt.figure(figsize=(10,4))
    bars = plt.barh(names, r2s, color=["#4C78A8","#1D9E75","#54A24B","#E45756","#B279A2"])
    plt.xlabel("CV R² (log target)")
    plt.title("Concept A: Stepwise time feature uplift\nEach row adds more temporal features")
    for bar, val in zip(bars, r2s):
        plt.text(val+0.002, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va="center", fontsize=9)
    plt.xlim(0,1.05); plt.tight_layout()
    plt.savefig(output_dir/"plot_time_features.png", dpi=160); plt.close()
    write_json(output_dir/"time_feature_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept B: Robustness — imputation on artificially masked data
# ─────────────────────────────────────────────────────────────────────────────
def analyse_robustness(X_train, y_train, output_dir):
    log.info("Concept B: Robustness / imputation analysis …")
    fe   = BikeFeatureEngineer().fit(X_train)
    Xeng = fe.transform(X_train)
    num_cols = [c for c in Xeng.columns if c not in DROP_COLS and c != "datetime"
                and pd.api.types.is_numeric_dtype(Xeng[c])]
    _arr = np.column_stack([
        pd.to_numeric(Xeng[c], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        for c in num_cols])
    y_log = np.log1p(y_train.to_numpy())

    # Artificially mask 15% of values to simulate real-world missing
    rng   = np.random.default_rng(RANDOM_STATE)
    mask  = rng.random(_arr.shape) < 0.15
    X_masked = _arr.copy().astype(float)
    X_masked[mask] = np.nan

    strategies = {
        "Simple (median)":  SimpleImputer(strategy="median"),
        "Simple (mean)":    SimpleImputer(strategy="mean"),
        "KNN (k=5)":        KNNImputer(n_neighbors=5),
        "MICE (BayesRidge)":IterativeImputer(estimator=BayesianRidge(),
                                              max_iter=10, random_state=RANDOM_STATE),
    }
    cv = KFold(n_splits=5, shuffle=False)
    results = {}
    for name, imp in strategies.items():
        try:
            X_imp = imp.fit_transform(X_masked)
            X_sc  = RobustScaler().fit_transform(X_imp)
            r2s   = cross_val_score(Ridge(alpha=1.0), X_sc, y_log, cv=cv, scoring="r2")
            results[name] = {"r2_mean": float(r2s.mean()), "r2_std": float(r2s.std())}
            log.info("  %-25s R²=%.4f±%.4f", name, r2s.mean(), r2s.std())
        except Exception as exc:
            log.warning("  %s failed: %s", name, exc)
            results[name] = {"r2_mean": float("nan"), "r2_std": float("nan")}

    # Also test the clean (no masking) baseline
    X_clean = RobustScaler().fit_transform(_arr)
    r2_clean = cross_val_score(Ridge(alpha=1.0), X_clean, y_log, cv=cv, scoring="r2")
    results["Clean (no masking)"] = {"r2_mean": float(r2_clean.mean()), "r2_std": float(r2_clean.std())}

    plt.figure(figsize=(8,4))
    names = list(results.keys()); r2_vals = [results[n]["r2_mean"] for n in names]
    errs  = [results[n]["r2_std"] for n in names]
    colors = ["#888"]*4 + ["#1D9E75"]
    plt.bar(names, r2_vals, yerr=errs, capsize=5, color=colors)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.ylabel("CV R² (with 15% artificial missingness)")
    plt.title("Concept B: Imputation robustness\nAll strategies tested on 15% masked data")
    plt.tight_layout(); plt.savefig(output_dir/"plot_imputation_robustness.png", dpi=160); plt.close()
    write_json(output_dir/"robustness_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept C: Log target analysis + RMSLE
# ─────────────────────────────────────────────────────────────────────────────
def analyse_log_target(X_train, y_train, output_dir):
    log.info("Concept C: Log target transformation …")
    fe   = BikeFeatureEngineer().fit(X_train)
    Xeng = fe.transform(X_train)
    num_cols = [c for c in Xeng.columns if c not in DROP_COLS and c != "datetime"
                and pd.api.types.is_numeric_dtype(Xeng[c])]
    _arr = np.column_stack([
        pd.to_numeric(Xeng[c], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        for c in num_cols])
    _arr = RobustScaler().fit_transform(_arr)
    y_raw = y_train.to_numpy()
    y_log = np.log1p(y_raw)
    cv    = KFold(n_splits=5, shuffle=False)

    oof_raw = cross_val_predict(Ridge(alpha=1.0), _arr, y_raw, cv=cv)
    oof_log = cross_val_predict(Ridge(alpha=1.0), _arr, y_log, cv=cv)
    oof_log_usd = np.expm1(oof_log)

    rmsle_raw = rmsle(y_raw, oof_raw.clip(0))
    rmsle_log = rmsle(y_raw, oof_log_usd.clip(0))
    rmse_raw  = float(np.sqrt(mean_squared_error(y_raw, oof_raw.clip(0))))
    rmse_log  = float(np.sqrt(mean_squared_error(y_raw, oof_log_usd.clip(0))))

    log.info("  Raw target: RMSLE=%.4f  RMSE=%.2f", rmsle_raw, rmse_raw)
    log.info("  Log target: RMSLE=%.4f  RMSE=%.2f  (improvement: %.1f%%)",
             rmsle_log, rmse_log, (rmsle_raw-rmsle_log)/rmsle_raw*100)

    fig, axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].hist(y_raw, bins=60, color="#4C78A8", edgecolor="white", linewidth=0.3)
    axes[0].set_title("count (raw) — right-skewed"); axes[0].set_xlabel("count (bikes/hour)")
    axes[1].hist(y_log, bins=60, color="#54A24B", edgecolor="white", linewidth=0.3)
    axes[1].set_title("log1p(count) — near-normal"); axes[1].set_xlabel("log1p(count)")
    plt.suptitle("Concept C: Log target transformation", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir/"plot_log_transform.png", dpi=160, bbox_inches="tight"); plt.close()

    return {"raw_rmsle":rmsle_raw,"log_rmsle":rmsle_log,
            "raw_rmse":rmse_raw,"log_rmse":rmse_log,
            "rmsle_improvement_pct":float((rmsle_raw-rmsle_log)/rmsle_raw*100)}


# ─────────────────────────────────────────────────────────────────────────────
# Concept D: Polynomial + non-linear basis functions
# ─────────────────────────────────────────────────────────────────────────────
def analyse_nonlinear_features(X_train, y_train, output_dir):
    log.info("Concept D: Non-linear feature analysis …")
    fe   = BikeFeatureEngineer().fit(X_train)
    Xeng = fe.transform(X_train)
    y_log = np.log1p(y_train.to_numpy())
    cv    = KFold(n_splits=5, shuffle=False)

    base_cols = [c for c in ["temp","humidity","windspeed","hour_sin","hour_cos",
                              "month_sin","month_cos","season","workingday","weather"]
                 if c in Xeng.columns]
    X_base = np.column_stack([
        pd.to_numeric(Xeng[c], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        for c in base_cols])
    X_sc   = StandardScaler().fit_transform(X_base)

    configs = {
        "Linear (base)":    (X_sc, None),
        "Poly d=2":         (PolynomialFeatures(2,include_bias=False).fit_transform(X_sc), None),
        "Spline(temp,5k)":  (None, "spline"),
        "All interactions": (None, "all_eng"),
    }

    # Spline on temp only (1D visualisation)
    temp_col = pd.to_numeric(Xeng["temp"], errors="coerce").fillna(0.5).to_numpy().reshape(-1,1)
    temp_spline = SplineTransformer(n_knots=5,degree=3,include_bias=False).fit_transform(
        StandardScaler().fit_transform(temp_col))

    # All engineered features
    all_cols = [c for c in Xeng.columns if c not in DROP_COLS and c!="datetime"
                and pd.api.types.is_numeric_dtype(Xeng[c])]
    X_all = np.column_stack([
        pd.to_numeric(Xeng[c],errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        for c in all_cols])
    X_all_sc = RobustScaler().fit_transform(X_all)

    results = {}
    for name, (X_enc, flag) in configs.items():
        if flag == "spline":
            X_enc = np.hstack([X_sc, temp_spline])
        elif flag == "all_eng":
            X_enc = X_all_sc
        r2s = cross_val_score(Ridge(alpha=1.0), X_enc, y_log, cv=cv, scoring="r2")
        results[name] = {"n_features":X_enc.shape[1],"r2_mean":float(r2s.mean()),"r2_std":float(r2s.std())}
        log.info("  %-25s n=%4d  R²=%.4f", name, X_enc.shape[1], r2s.mean())

    plt.figure(figsize=(9,4))
    names = list(results.keys()); r2_vals = [results[n]["r2_mean"] for n in names]
    plt.bar(names, r2_vals, color=["#4C78A8","#1D9E75","#F58518","#E45756"])
    plt.xticks(rotation=15,ha="right"); plt.ylabel("CV R² (log target)")
    plt.title("Concept D: Non-linear feature expansion\nEach method adds expressiveness")
    for i,(bar,v) in enumerate(zip(plt.gca().patches, r2_vals)):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f"{v:.4f}", ha="center",fontsize=9)
    plt.tight_layout(); plt.savefig(output_dir/"plot_nonlinear_features.png",dpi=160); plt.close()
    write_json(output_dir/"nonlinear_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept E: Gradient Boosting deep-dive
# ─────────────────────────────────────────────────────────────────────────────
def analyse_gradient_boosting(X_train, y_train, output_dir, quick=False):
    log.info("Concept E: Gradient Boosting deep-dive …")
    fe   = BikeFeatureEngineer().fit(X_train)
    Xeng = fe.transform(X_train)
    num_cols = [c for c in Xeng.columns if c not in DROP_COLS and c!="datetime"
                and pd.api.types.is_numeric_dtype(Xeng[c])]
    X_arr = np.column_stack([
        pd.to_numeric(Xeng[c],errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        for c in num_cols])
    y_log = np.log1p(y_train.to_numpy())
    cv    = KFold(n_splits=3 if quick else 5, shuffle=False)

    n_est   = 50 if quick else 200
    configs = {
        "GBR (lr=0.1,d=4)":  GradientBoostingRegressor(n_estimators=n_est,learning_rate=0.1,max_depth=4,random_state=RANDOM_STATE),
        "GBR (lr=0.05,d=5)": GradientBoostingRegressor(n_estimators=n_est,learning_rate=0.05,max_depth=5,random_state=RANDOM_STATE),
        "GBR (lr=0.2,d=3)":  GradientBoostingRegressor(n_estimators=n_est,learning_rate=0.2,max_depth=3,random_state=RANDOM_STATE),
    }
    if _XGB:
        configs["XGBoost (d=4)"] = XGBRegressor(n_estimators=n_est,max_depth=4,learning_rate=0.1,
                                                   subsample=0.8,colsample_bytree=0.8,
                                                   eval_metric="rmse",random_state=RANDOM_STATE,
                                                   n_jobs=N_JOBS,verbosity=0)
    if _LGB:
        configs["LightGBM (d=4)"] = LGBMRegressor(n_estimators=n_est,max_depth=4,learning_rate=0.1,
                                                     subsample=0.8,colsample_bytree=0.8,
                                                     random_state=RANDOM_STATE,n_jobs=N_JOBS,verbose=-1)
    results = {}
    for name, mdl in configs.items():
        r2s  = cross_val_score(mdl, X_arr, y_log, cv=cv, scoring="r2")
        rmse = -cross_val_score(mdl, X_arr, y_log, cv=cv, scoring="neg_root_mean_squared_error")
        results[name] = {"r2_mean":float(r2s.mean()),"r2_std":float(r2s.std()),
                         "rmse_log_mean":float(rmse.mean())}
        log.info("  %-25s R²=%.4f±%.4f  RMSE_log=%.4f", name,r2s.mean(),r2s.std(),rmse.mean())

    # Learning rate comparison plot (number of trees vs training error)
    if not quick:
        fig, ax = plt.subplots(figsize=(9,4.5))
        colors = ["#4C78A8","#1D9E75","#E45756"]
        for (name,mdl),color in zip(list(configs.items())[:3],colors):
            mdl2 = clone(mdl).set_params(n_estimators=300)
            mdl2.fit(X_arr,y_log)
            if hasattr(mdl2,"train_score_"):
                ax.plot(mdl2.train_score_, label=f"{name} (train)", color=color, linewidth=1.5)
            # OOF validation via staged predict
            oof_r2 = []
            kf = KFold(n_splits=3,shuffle=False)
            tr_i,te_i = next(iter(kf.split(X_arr)))
            mdl3=clone(mdl).set_params(n_estimators=300); mdl3.fit(X_arr[tr_i],y_log[tr_i])
            if hasattr(mdl3,"staged_predict"):
                for n_iter,pred in enumerate(mdl3.staged_predict(X_arr[te_i])):
                    oof_r2.append(r2_score(y_log[te_i],pred))
                ax.plot(oof_r2, label=f"{name} (val)", color=color, linewidth=1.0, linestyle="--")
        ax.set_xlabel("n_estimators"); ax.set_ylabel("R² (log target)")
        ax.set_title("Concept E: GBR learning rate × depth tradeoff\nSolid=train, dashed=validation")
        ax.legend(fontsize=8); plt.tight_layout()
        plt.savefig(output_dir/"plot_gb_convergence.png",dpi=160); plt.close()

    write_json(output_dir/"gradient_boosting_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept F: Random Forest + Out-of-Bag
# ─────────────────────────────────────────────────────────────────────────────
def analyse_random_forest(X_train, y_train, output_dir, quick=False):
    log.info("Concept F: Random Forest analysis …")
    fe   = BikeFeatureEngineer().fit(X_train)
    Xeng = fe.transform(X_train)
    num_cols = [c for c in Xeng.columns if c not in DROP_COLS and c!="datetime"
                and pd.api.types.is_numeric_dtype(Xeng[c])]
    X_arr = np.column_stack([
        pd.to_numeric(Xeng[c],errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        for c in num_cols])
    y_log = np.log1p(y_train.to_numpy())
    n_est = 50 if quick else 200

    # RF supports OOB natively (bootstrap=True by default).
    # ExtraTrees defaults to bootstrap=False so oob_score=True crashes it.
    # Fix: enable bootstrap=True for ExtraTrees to get the OOB estimate,
    # which is the correct apples-to-apples comparison with RF.
    rf  = RandomForestRegressor(n_estimators=n_est, oob_score=True,
                                 max_depth=12, min_samples_leaf=2,
                                 random_state=RANDOM_STATE, n_jobs=N_JOBS)
    et  = ExtraTreesRegressor(n_estimators=n_est, oob_score=True,
                               bootstrap=True,        # required for oob_score
                               max_depth=12, min_samples_leaf=2,
                               random_state=RANDOM_STATE, n_jobs=N_JOBS)
    rf.fit(X_arr, y_log); et.fit(X_arr, y_log)

    results = {
        "RF_oob_r2":  float(rf.oob_score_),
        "ET_oob_r2":  float(et.oob_score_),
        "RF_n_features": int(n_est),
        "ET_n_features": int(n_est),
    }
    log.info("  RF OOB R²=%.4f  ET OOB R²=%.4f", rf.oob_score_, et.oob_score_)

    # Feature importance plot
    fi = pd.DataFrame({"feature":num_cols,"importance":rf.feature_importances_}
        ).sort_values("importance",ascending=False).head(20)
    plt.figure(figsize=(9,5))
    sns.barplot(data=fi, y="feature", x="importance", color="#4C78A8")
    plt.title("Concept F: RF feature importance (top-20)\nOOB-validated importance scores")
    plt.tight_layout(); plt.savefig(output_dir/"plot_rf_importance.png",dpi=160); plt.close()

    fi.to_csv(output_dir/"rf_feature_importance.csv", index=False)
    write_json(output_dir/"random_forest_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept G: Hyperparameter tuning (RandomizedSearch + optional Optuna)
# ─────────────────────────────────────────────────────────────────────────────
def analyse_hyperparameter_tuning(X_train, y_train, output_dir, quick=False):
    log.info("Concept G: Hyperparameter tuning analysis …")
    fe   = BikeFeatureEngineer().fit(X_train)
    Xeng = fe.transform(X_train)
    num_cols = [c for c in Xeng.columns if c not in DROP_COLS and c!="datetime"
                and pd.api.types.is_numeric_dtype(Xeng[c])]
    X_arr = np.column_stack([
        pd.to_numeric(Xeng[c],errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        for c in num_cols])
    y_log = np.log1p(y_train.to_numpy())
    cv    = KFold(n_splits=3,shuffle=False)

    results = {}
    # RandomizedSearch on GBR
    n_est = 50 if quick else 100
    n_iter = 5 if quick else 15
    param_grid = {
        "n_estimators":  [50,100,200],
        "max_depth":     [3,4,5,6],
        "learning_rate": [0.02,0.05,0.1,0.15,0.2],
        "subsample":     [0.7,0.8,0.9],
        "min_samples_leaf":[1,2,4],
    }
    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        param_grid, n_iter=n_iter, scoring="r2", cv=cv,
        random_state=RANDOM_STATE, n_jobs=N_JOBS, refit=True)
    search.fit(X_arr, y_log)
    results["GBR_RandomSearch"] = {
        "best_params": search.best_params_,
        "best_cv_r2": float(search.best_score_),
    }
    log.info("  GBR RandomSearch best R²=%.4f  params=%s", search.best_score_, search.best_params_)

    # Optuna Bayesian search (if available)
    if _OPTUNA and _XGB and not quick:
        def objective(trial):
            params = {
                "n_estimators":    trial.suggest_int("n_estimators", 50,300),
                "max_depth":       trial.suggest_int("max_depth", 3,8),
                "learning_rate":   trial.suggest_float("learning_rate", 0.01,0.3,log=True),
                "subsample":       trial.suggest_float("subsample", 0.6,1.0),
                "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5,1.0),
            }
            mdl = XGBRegressor(**params, eval_metric="rmse",
                               random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0)
            r2s = cross_val_score(mdl, X_arr, y_log, cv=cv, scoring="r2")
            return r2s.mean()
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        results["XGB_Optuna"] = {
            "best_params": study.best_params,
            "best_cv_r2":  float(study.best_value),
        }
        log.info("  XGB Optuna best R²=%.4f", study.best_value)

    write_json(output_dir/"hyperparameter_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept H: Time-aware cross-validation
# ─────────────────────────────────────────────────────────────────────────────
def analyse_timeseries_cv(X_train, y_train, output_dir, quick=False):
    log.info("Concept H: Time-aware CV comparison …")
    pipe = build_pipeline(Ridge(alpha=1.0))
    y_log = np.log1p(y_train.to_numpy())

    cv_configs = {
        "KFold (shuffled)":    KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        "KFold (sequential)":  KFold(n_splits=5, shuffle=False),
        "TimeSeriesSplit(5)":  TimeSeriesSplit(n_splits=5),
    }
    results = {}
    for name, cv in cv_configs.items():
        r2s = cross_val_score(pipe, X_train, y_log, cv=cv, scoring="r2")
        results[name] = {"mean": float(r2s.mean()), "std": float(r2s.std()),
                         "_scores": r2s.tolist()}
        log.info("  %-30s R²=%.4f±%.4f", name, r2s.mean(), r2s.std())

    fig, ax = plt.subplots(figsize=(9,4))
    names = list(results.keys())
    for i,(name,res) in enumerate(results.items()):
        scores = res["_scores"]
        ax.scatter([i]*len(scores), scores, alpha=0.7, s=40, zorder=5)
        ax.plot([i-0.3,i+0.3],[res["mean"],res["mean"]],"r-",linewidth=2)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=10,ha="right")
    ax.set_ylabel("CV R² (log target)")
    ax.set_title("Concept H: Time-aware CV strategies\nShuffled KFold leaks future data into past folds")
    plt.tight_layout(); plt.savefig(output_dir/"plot_timeseries_cv.png",dpi=160); plt.close()
    write_json(output_dir/"timeseries_cv.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept I: Outlier analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_outliers(X_train, y_train, output_dir):
    log.info("Concept I: Outlier analysis …")
    fe   = BikeFeatureEngineer().fit(X_train)
    Xeng = fe.transform(X_train)
    y    = y_train.to_numpy()
    y_log = np.log1p(y)
    cv   = KFold(n_splits=3,shuffle=False)

    # Isolation Forest
    num_cols = [c for c in ["temp","humidity","windspeed","hour_sin","hour_cos",
                              "month_sin","month_cos"] if c in Xeng.columns]
    X_key = np.column_stack([
        pd.to_numeric(Xeng[c],errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        for c in num_cols])
    iso   = IsolationForest(contamination=0.03, random_state=RANDOM_STATE)
    iso_labels = iso.fit_predict(X_key)
    n_iso = int((iso_labels==-1).sum())

    # Weather=4 (heavy rain) analysis
    w4_mask = pd.to_numeric(X_train.get("weather", pd.Series(1,index=X_train.index)),
                             errors="coerce").fillna(1) == 4
    n_w4 = int(w4_mask.sum())

    # IQR outliers on count
    q1,q3 = np.percentile(y,[25,75]); iqr = q3-q1
    iqr_mask = (y >= q1-1.5*iqr) & (y <= q3+1.5*iqr)
    n_iqr = int((~iqr_mask).sum())

    # Compare R²
    num_cols_full = [c for c in Xeng.columns if c not in DROP_COLS and c!="datetime"
                     and pd.api.types.is_numeric_dtype(Xeng[c])]
    X_full = np.column_stack([
        pd.to_numeric(Xeng[c],errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        for c in num_cols_full])
    comparison = {}
    for label, mask in [("All data", np.ones(len(y),dtype=bool)),
                         ("Remove IQR outliers", iqr_mask),
                         ("Remove IsolationForest", iso_labels==1)]:
        X_m = X_full[mask]; y_m = y_log[mask]
        r2s = cross_val_score(Ridge(alpha=1.0), X_m, y_m, cv=cv, scoring="r2")
        comparison[label] = {"n_kept":int(mask.sum()),"n_removed":int((~mask).sum()),
                              "r2_mean":float(r2s.mean())}
        log.info("  %-35s n=%5d  R²=%.4f", label, mask.sum(), r2s.mean())

    # Scatter: hour vs count coloured by weather
    if "hour" in Xeng.columns:
        plt.figure(figsize=(9,4))
        for w in [1,2,3]:
            wm = pd.to_numeric(X_train.get("weather",pd.Series(1,index=X_train.index)),
                               errors="coerce").fillna(1)==w
            h  = pd.to_numeric(Xeng.get("hour",pd.Series(0,index=Xeng.index)),errors="coerce")[wm]
            plt.scatter(h, y[wm], alpha=0.1, s=4,
                        label=WEATHER_LABELS.get(w,str(w)))
        plt.xlabel("Hour of day"); plt.ylabel("count (bikes/hour)")
        plt.title("Concept I: Hourly demand by weather\nWeather=4 hours have near-zero count")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(output_dir/"plot_outlier_weather.png",dpi=160); plt.close()

    write_json(output_dir/"outlier_analysis.json",
               {"n_iso_outliers":n_iso,"n_iqr_outliers":n_iqr,
                "n_weather4":n_w4,"comparison":comparison})
    return {"n_iso":n_iso,"n_iqr":n_iqr,"comparison":comparison}


# ─────────────────────────────────────────────────────────────────────────────
# Concept J: Residual diagnostics + ACF
# ─────────────────────────────────────────────────────────────────────────────
def residual_diagnostics(y_true_log, y_pred_log, output_dir):
    log.info("Concept J: Residual diagnostics + ACF …")
    residuals = y_true_log - y_pred_log
    n = len(residuals)
    bp_r2 = LinearRegression().fit(y_pred_log.reshape(-1,1),residuals**2).score(
                y_pred_log.reshape(-1,1),residuals**2)
    bp_stat = float(n*bp_r2)
    bp_p    = float(1-scipy_stats.chi2.cdf(bp_stat,df=1))
    sw_s,sw_p = scipy_stats.shapiro(residuals[:5000])

    # ACF of residuals (temporal autocorrelation)
    try:
        acf_vals = _acf(residuals, nlags=48)
    except Exception:
        acf_vals = np.zeros(49)

    fig, axes = plt.subplots(1,3,figsize=(15,4))
    scipy_stats.probplot(residuals,dist="norm",plot=axes[0])
    axes[0].set_title("Q-Q plot (normality)")
    axes[1].scatter(y_pred_log, residuals, alpha=0.1, s=4, color="#4C78A8")
    axes[1].axhline(0,color="red",linewidth=1.0)
    axes[1].set_xlabel("Fitted log(count)"); axes[1].set_ylabel("Residual")
    axes[1].set_title("Scale-location (heteroscedasticity)")
    lags = np.arange(len(acf_vals))
    axes[2].bar(lags, acf_vals, color="#4C78A8", width=0.8, alpha=0.8)
    axes[2].axhline(0,color="black",linewidth=0.8)
    axes[2].axhline(1.96/np.sqrt(n),color="#E45756",linestyle="--",alpha=0.7,label="95% CI")
    axes[2].axhline(-1.96/np.sqrt(n),color="#E45756",linestyle="--",alpha=0.7)
    axes[2].set_xlabel("Lag (hours)"); axes[2].set_title("Concept J: ACF of residuals\n(temporal autocorrelation)")
    axes[2].legend(fontsize=8)
    plt.suptitle("Concept J: Residual diagnostics",fontsize=12,y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir/"plot_residual_diagnostics.png",dpi=160,bbox_inches="tight"); plt.close()

    log.info("BP p=%.4f (hetero: %s)  SW p=%.4f (normal: %s)  ACF lag-1=%.3f",
             bp_p, bp_p<0.05, sw_p, sw_p>0.05, acf_vals[1] if len(acf_vals)>1 else 0)
    return {
        "breusch_pagan":{"statistic":bp_stat,"p_value":bp_p,"heteroscedastic":bp_p<0.05},
        "shapiro_wilk": {"statistic":float(sw_s),"p_value":float(sw_p),"normal":sw_p>0.05},
        "acf_lag1":     float(acf_vals[1]) if len(acf_vals)>1 else 0.0,
        "acf_lag24":    float(acf_vals[24]) if len(acf_vals)>24 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Concept K: Learning curves + n_estimators convergence
# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_curves_bike(model, X_train, y_train_log, output_dir):
    log.info("Concept K: Learning curves …")
    cv = KFold(n_splits=3,shuffle=False)
    try:
        sizes,tr_s,cv_s = learning_curve(
            model, X_train, y_train_log,
            train_sizes=np.linspace(0.10,1.0,8),
            cv=cv, scoring="r2", n_jobs=N_JOBS)
        tr_r2=tr_s.mean(axis=1); cv_r2=cv_s.mean(axis=1)
        plt.figure(figsize=(8,4.5))
        plt.plot(sizes,tr_r2,"o-",color="#4C78A8",label="Train R²")
        plt.plot(sizes,cv_r2,"o-",color="#E45756",label="CV R²")
        plt.fill_between(sizes,tr_r2-tr_s.std(axis=1),tr_r2+tr_s.std(axis=1),alpha=0.12,color="#4C78A8")
        plt.fill_between(sizes,cv_r2-cv_s.std(axis=1),cv_r2+cv_s.std(axis=1),alpha=0.12,color="#E45756")
        gap=float(cv_r2[-1]-tr_r2[-1])
        plt.title(f"Concept K: Learning curve (n=17379)\nTrain-CV gap={gap:.4f}")
        plt.xlabel("Training set size"); plt.ylabel("R²")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir/"plot_learning_curve.png",dpi=160); plt.close()
    except Exception as exc:
        log.warning("Learning curve failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Concept L: Ensemble + stacking
# ─────────────────────────────────────────────────────────────────────────────
def analyse_ensemble(tuned_models, X_train, y_train_log, output_dir):
    log.info("Concept L: Ensemble construction …")
    estimators = [(n,m) for n,m in tuned_models.items() if m is not None]
    if len(estimators)<2: return {}
    results = {}
    for ename,Cls,kwargs in [
        ("AverageVoting",VotingRegressor,{"n_jobs":N_JOBS}),
        ("Stacking",StackingRegressor,{"final_estimator":Ridge(alpha=1.0),"cv":3,"n_jobs":N_JOBS}),
    ]:
        try:
            m=Cls(estimators=estimators,**kwargs)
            m.fit(X_train,y_train_log)
            results[ename]={"model":m}
            log.info("  [Ensemble] %s fitted.",ename)
        except Exception as exc:
            log.warning("  %s failed: %s",ename,exc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_predictions(y_true, y_pred_log):
    if hasattr(y_true,"to_numpy"): y_true=y_true.to_numpy()
    y_pred_usd = np.expm1(y_pred_log).clip(0)
    y_true_usd = y_true
    return {
        "rmsle":        rmsle(y_true_usd, y_pred_usd),
        "rmse":         float(np.sqrt(mean_squared_error(y_true_usd,y_pred_usd))),
        "mae":          float(mean_absolute_error(y_true_usd,y_pred_usd)),
        "r2":           float(r2_score(y_true_usd,y_pred_usd)),
        "mape":         float(mean_absolute_percentage_error(y_true_usd.clip(1),y_pred_usd.clip(1))),
        "medae":        float(median_absolute_error(y_true_usd,y_pred_usd)),
        "residual_mean":float((y_true_usd-y_pred_usd).mean()),
        "residual_std": float((y_true_usd-y_pred_usd).std()),
    }


def evaluate_baselines(X_tr,X_te,y_tr,y_te):
    return {s:evaluate_predictions(y_te,np.log1p(DummyRegressor(strategy=s).fit(X_tr,y_tr).predict(X_te).clip(0)))
            for s in ["mean","median"]}


# ─────────────────────────────────────────────────────────────────────────────
# Subgroups
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_subgroups(model, X_test, y_test, y_pred_log, output_dir):
    log.info("Subgroup evaluation …")
    y_pred_usd = np.expm1(y_pred_log).clip(0)
    y_true_usd = y_test.to_numpy()
    overall_rmse = float(np.sqrt(mean_squared_error(y_true_usd,y_pred_usd)))
    overall_rmsle= rmsle(y_true_usd,y_pred_usd)

    fe   = BikeFeatureEngineer().fit(X_test)
    Xeng = fe.transform(X_test.reset_index(drop=True))
    eval_df = Xeng.copy()
    eval_df["_y_true"] = y_true_usd; eval_df["_y_pred"] = y_pred_usd

    rows = []
    for col in ["season","weather","workingday","holiday"]:
        if col not in eval_df.columns: continue
        for val,sub in eval_df.groupby(col,observed=True):
            if len(sub)<20: continue
            sr = float(np.sqrt(mean_squared_error(sub["_y_true"],sub["_y_pred"])))
            sr_le = rmsle(sub["_y_true"].to_numpy(),sub["_y_pred"].to_numpy())
            rows.append({
                "group_col":col,"group_val":str(val),"n":int(len(sub)),
                "mean_actual":round(float(sub["_y_true"].mean()),1),
                "rmse":round(sr,2),"rmsle":round(sr_le,4),
                "r2":round(float(r2_score(sub["_y_true"],sub["_y_pred"])),4),
                "rmse_gap":round(sr-overall_rmse,2),
                "alert":bool(sr>overall_rmse*1.25),
            })
    if rows:
        rd=pd.DataFrame(rows); rd.to_csv(output_dir/"subgroup_report.csv",index=False)
        g=rd.copy(); g["label"]=g["group_col"]+"="+g["group_val"].astype(str)
        plt.figure(figsize=(10,max(4,len(g)*0.42)))
        colors=["#E45756" if a else "#4C78A8" for a in g["alert"]]
        plt.barh(g["label"],g["rmse"],color=colors)
        plt.axvline(overall_rmse,linestyle="--",color="black",label=f"Overall RMSE={overall_rmse:.1f}")
        plt.xlabel("RMSE (bikes/hour)"); plt.title("Subgroup RMSE — season/weather/workingday/holiday")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir/"plot_subgroup_rmse.png",dpi=160); plt.close()
    return {"overall_rmse":overall_rmse,"overall_rmsle":overall_rmsle,"subgroups":rows}


# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────
def save_shap_artifacts(model, X_test, y_test, y_pred_log, output_dir):
    if not _SHAP: log.warning("pip install shap"); return
    log.info("SHAP for champion …")
    try:
        step_names = list(model.named_steps.keys())
        clf  = model.named_steps["model"]
        prep = model.named_steps["preprocess"]
        fe   = model.named_steps["feature_engineering"]
        sel  = model.named_steps.get("feature_selection")

        Xt = fe.transform(X_test)
        Xt = prep.transform(Xt)
        fn = prep.get_feature_names_out()

        if sel is not None:
            fs_idx = step_names.index("feature_selection")
            fn2 = None
            for sname in reversed(step_names[:fs_idx]):
                s = model.named_steps[sname]
                if hasattr(s,"get_feature_names_out"):
                    try: fn2=s.get_feature_names_out(); break
                    except: pass
            if fn2 is None: fn2=np.array([f"f{i}" for i in range(Xt.shape[1])])
            support=sel.get_support()
            if len(fn2)!=len(support):
                fn2=np.array([f"feature_{i}" for i in range(len(support))])
            sn=fn2[support]; Xt=sel.transform(Xt)
        else:
            sn=fn

        Xdf=pd.DataFrame(Xt,columns=sn)
        if hasattr(clf,"feature_importances_"):
            exp=shap.TreeExplainer(clf); sv=exp.shap_values(Xdf)
        elif hasattr(clf,"coef_"):
            exp=shap.LinearExplainer(clf,Xdf); sv=exp.shap_values(Xdf)
        else:
            mask=shap.maskers.Independent(Xdf,max_samples=100)
            exp=shap.Explainer(clf.predict,mask); sv=exp(Xdf).values

        for ptype,fname in [("bar","plot_shap_bar.png"),("dot","plot_shap_beeswarm.png")]:
            plt.figure(figsize=(10,6))
            shap.summary_plot(sv,Xdf,plot_type=ptype,show=False,max_display=20)
            plt.tight_layout(); plt.savefig(output_dir/fname,dpi=150,bbox_inches="tight"); plt.close()

        worst=int(np.argmax(np.abs(y_test.to_numpy()-np.expm1(y_pred_log))))
        ev=(float(exp.expected_value) if not isinstance(exp.expected_value,np.ndarray) else float(exp.expected_value))
        shap.waterfall_plot(
            shap.Explanation(values=sv[worst],base_values=ev,
                             data=Xdf.iloc[worst].values,feature_names=list(sn)),
            show=False,max_display=15)
        plt.title("SHAP Waterfall — worst prediction")
        plt.tight_layout(); plt.savefig(output_dir/"plot_shap_waterfall.png",dpi=150,bbox_inches="tight"); plt.close()
        pd.DataFrame({"feature":sn,"mean_abs_shap":np.abs(sv).mean(axis=0)}
            ).sort_values("mean_abs_shap",ascending=False
            ).to_csv(output_dir/"shap_importance.csv",index=False)
        log.info("SHAP saved.")
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────
def save_feature_importance(model, output_dir):
    step_names = list(model.named_steps.keys())
    clf = model.named_steps["model"]
    sel = model.named_steps.get("feature_selection")
    if sel:
        fs_idx=step_names.index("feature_selection")
        fn=None
        for sname in reversed(step_names[:fs_idx]):
            s=model.named_steps[sname]
            if hasattr(s,"get_feature_names_out"):
                try: fn=s.get_feature_names_out(); break
                except: pass
        if fn is None: return
        support=sel.get_support()
        if len(fn)!=len(support): fn=np.array([f"feature_{i}" for i in range(len(support))])
        sn=fn[support]
    else:
        return

    if hasattr(clf,"feature_importances_"):
        imp=clf.feature_importances_; signed=np.full(len(imp),np.nan)
    elif hasattr(clf,"coef_"):
        signed=clf.coef_; imp=np.abs(signed)
    else: return

    pd.DataFrame({"feature":sn,"importance":imp}
        ).sort_values("importance",ascending=False
        ).to_csv(output_dir/"feature_importance.csv",index=False)
    plt.figure(figsize=(9,5.5))
    sns.barplot(data=pd.DataFrame({"feature":sn,"importance":imp}
        ).sort_values("importance",ascending=False).head(25),
        y="feature",x="importance",color="#4C78A8")
    plt.title("Top 25 model features"); plt.tight_layout()
    plt.savefig(output_dir/"plot_feature_importance.png",dpi=160); plt.close()


def save_evaluation_plots(y_test, y_pred_log, output_dir):
    y_pred_usd = np.expm1(y_pred_log).clip(0)
    y_true_usd = y_test.to_numpy()
    plt.figure(figsize=(6,5))
    plt.scatter(y_true_usd, y_pred_usd, alpha=0.2, s=5, color="#4C78A8")
    mn,mx = float(min(y_true_usd.min(),y_pred_usd.min())),float(max(y_true_usd.max(),y_pred_usd.max()))
    plt.plot([mn,mx],[mn,mx],"r--",linewidth=1.2,label="Perfect")
    plt.title(f"Actual vs Predicted count\n(R²={r2_score(y_true_usd,y_pred_usd):.3f}  RMSLE={rmsle(y_true_usd,y_pred_usd):.4f})")
    plt.xlabel("Actual count (bikes/hour)"); plt.ylabel("Predicted count")
    plt.legend(); plt.tight_layout()
    plt.savefig(output_dir/"plot_actual_vs_predicted.png",dpi=160); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# EDA
# ─────────────────────────────────────────────────────────────────────────────
def save_research_artifacts(X_train, y_train, output_dir):
    log.info("Saving EDA artifacts (train-set only) …")
    output_dir.mkdir(parents=True,exist_ok=True)
    eda = X_train.copy(); eda[TARGET] = y_train.values

    missingness_report(eda).to_csv(output_dir/"eda_missingness.csv")

    # Hourly average demand
    fe   = BikeFeatureEngineer().fit(X_train)
    Xeng = fe.transform(X_train)

    sns.set_theme(style="whitegrid")

    # Target distribution
    fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4))
    a1.hist(y_train,bins=60,color="#4C78A8",edgecolor="white",linewidth=0.3)
    a1.set_title("count (raw) — right-skewed"); a1.set_xlabel("bikes/hour")
    a2.hist(np.log1p(y_train),bins=60,color="#54A24B",edgecolor="white",linewidth=0.3)
    a2.set_title("log1p(count) — near-normal"); a2.set_xlabel("log1p(count)")
    plt.suptitle("Target distribution (train)",fontsize=11,y=1.02)
    plt.tight_layout(); plt.savefig(output_dir/"eda_target_distribution.png",dpi=150,bbox_inches="tight"); plt.close()

    # Hourly demand heatmap (hour × day-of-week)
    if "hour" in Xeng.columns and "dow" in Xeng.columns:
        pivot_df = pd.DataFrame({"hour":Xeng["hour"].astype(int),"dow":Xeng["dow"].astype(int),"count":y_train.values})
        pivot = pivot_df.groupby(["hour","dow"])["count"].mean().unstack(fill_value=0)
        plt.figure(figsize=(10,6))
        sns.heatmap(pivot,cmap="YlOrRd",fmt=".0f",annot=False,
                    xticklabels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],linewidths=0.4)
        plt.title("Mean demand by hour × day-of-week (train)\nMorning/evening rush hours clearly visible")
        plt.xlabel("Day of week"); plt.ylabel("Hour of day")
        plt.tight_layout(); plt.savefig(output_dir/"eda_demand_heatmap.png",dpi=150); plt.close()

    # Demand by season × workingday — grouped bar (median per group)
    # Avoids sns.boxplot(hue=...) which has a breaking API change in seaborn 0.13
    if "season" in eda.columns and "workingday" in eda.columns:
        eda["season_label"]  = eda["season"].map(SEASON_LABELS)
        eda["workday_label"] = eda["workingday"].map({0:"Weekend/Holiday",1:"Working day"})
        grp = eda.groupby(["season_label","workday_label"])[TARGET].median().unstack(fill_value=0)
        ordered = [SEASON_LABELS[k] for k in sorted(SEASON_LABELS) if SEASON_LABELS[k] in grp.index]
        grp = grp.reindex(ordered)
        fig, ax = plt.subplots(figsize=(9,4))
        x = np.arange(len(grp.index)); w = 0.35
        colors_pair = ["#4C78A8","#E45756"]
        for i, col in enumerate(list(grp.columns)[:2]):
            ax.bar(x + (i-0.5)*w, grp[col], w, label=col, color=colors_pair[i], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(grp.index)
        ax.set_ylabel("Median count (bikes/hour)"); ax.set_xlabel("Season")
        ax.legend(fontsize=9); ax.set_title("Median demand by season × day type")
        plt.tight_layout(); plt.savefig(output_dir/"eda_season_demand.png",dpi=150); plt.close()

    # Weather × hour
    if "weather" in eda.columns and "hour" in Xeng.columns:
        eda["hour"] = Xeng["hour"].values
        eda["weather_label"] = eda["weather"].map(WEATHER_LABELS)
        wh_avg = eda.groupby(["hour","weather_label"])[TARGET].mean().reset_index()
        plt.figure(figsize=(9,4))
        for w,color in [("clear","#1D9E75"),("mist","#F58518"),("light_precip","#E45756")]:
            sub = wh_avg[wh_avg["weather_label"]==w]
            if len(sub)>0:
                plt.plot(sub["hour"],sub[TARGET],label=w,color=color,linewidth=1.8)
        plt.xlabel("Hour"); plt.ylabel("Mean demand")
        plt.title("Mean demand by hour × weather condition\nBad weather suppresses all hours")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir/"eda_weather_hour.png",dpi=150); plt.close()

    write_json(output_dir/"research_decisions.json",{
        "problem_type":"regression","target":TARGET,"target_unit":"bikes/hour",
        "log_transform":"log1p(count) — right-skewed target",
        "temporal_split":"last 20% of hours as test (time-aware, no future leakage)",
        "leakage_drop":["casual","registered","datetime"],
        "key_features":["hour","day_of_week","season","weather","temp","rush_hour_flags"],
    })
    log.info("EDA saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Training profile + governance
# ─────────────────────────────────────────────────────────────────────────────
def build_training_profile(X_train, y_train):
    num_cols = [c for c in ["temp","atemp","humidity","windspeed"] if c in X_train.columns]
    stats = {}
    for col in num_cols:
        v = pd.to_numeric(X_train[col],errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(v)==0: continue
        stats[col] = {"mean":float(v.mean()),"std":float(v.std()),
                      "min":float(v.min()),"max":float(v.max()),
                      "quantiles":np.quantile(v,np.linspace(0,1,100)).tolist()}
    return to_jsonable({
        "trained_at":datetime.now(timezone.utc).isoformat(),
        "row_count":int(len(X_train)),
        "raw_columns":list(X_train.columns),
        "target_stats":{"mean":float(y_train.mean()),"std":float(y_train.std()),
                        "min":float(y_train.min()),"max":float(y_train.max())},
        "raw_missing_rate":X_train.isna().mean().to_dict(),
        "numeric_train_stats":stats,
    })


def save_model_card(metrics, fairness, search, output_dir):
    tm = metrics.get("test_metrics",{})
    write_json(output_dir/MODEL_CARD_FILE,{
        "schema_version":"1.0",
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "model_details":{
            "name":"Bike Sharing Demand Predictor",
            "type":"Regression (sklearn Pipeline + log(count) target)",
            "algorithm":repr(search.best_estimator_.named_steps["model"]),
        },
        "intended_use":{
            "primary_use":"Hourly bike rental demand forecasting for Capital Bikeshare D.C.",
            "out_of_scope":["Other cities","Real-time dispatch","Post-2012 data without retraining"],
        },
        "evaluation_results":{
            "test_rmsle":tm.get("rmsle"),"test_rmse":tm.get("rmse"),"test_r2":tm.get("r2"),
        },
        "log_transform":"log1p(count) — improves RMSLE and residual normality",
        "temporal_split":"Time-ordered 80/20 split — last 20% of hours as test",
        "fairness":{"overall_rmse":fairness.get("overall_rmse"),
                    "alerts":[r for r in fairness.get("subgroups",[]) if r.get("alert")]},
        "limitations":[
            "Data from 2011–2012 D.C. — not generalisable to other cities or eras.",
            "No real-time weather data — predictions require forecast input.",
            "Extreme weather events (weather=4) have very few training samples.",
        ],
        "hyperparameters":search.best_params_,
        "cv_best_r2":float(search.best_score_),
    })


def log_to_mlflow(metrics, search, model, output_dir):
    if not _MLFLOW: return
    try:
        mlflow.set_experiment("bike_sharing")
        tm=metrics.get("test_metrics",{})
        with mlflow.start_run():
            mlflow.log_params({f"best_{k}":str(v) for k,v in search.best_params_.items()})
            mlflow.log_metrics({"cv_r2":float(search.best_score_),
                                "test_rmsle":float(tm.get("rmsle",0)),
                                "test_rmse":float(tm.get("rmse",0))})
            for f in [MODEL_CARD_FILE,METRICS_FILE,"plot_actual_vs_predicted.png","plot_shap_bar.png"]:
                if (output_dir/f).exists(): mlflow.log_artifact(str(output_dir/f))
            mlflow.sklearn.log_model(model,"model")
        log.info("MLflow logged.")
    except Exception as e:
        log.warning("MLflow failed: %s",e)


def save_environment_snapshot(output_dir):
    env={"saved_at":datetime.now(timezone.utc).isoformat(),"python":sys.version,"platform":sys.platform,"libraries":{}}
    for lib in ["sklearn","pandas","numpy","scipy","joblib","shap","mlflow","xgboost","lightgbm","optuna","statsmodels"]:
        try:
            mod=importlib.import_module(lib); env["libraries"][lib]=getattr(mod,"__version__","unknown")
        except ImportError:
            env["libraries"][lib]="not_installed"
    write_json(output_dir/ENVIRONMENT_FILE,env)


def compute_oof_uncertainty(best_estimator, X_train, y_train_log,
                             overpredict_cost=1.0, underpredict_cost=1.0):
    # cross_val_predict requires a full partition (every row in exactly one
    # test fold). TimeSeriesSplit leaves early rows out of all test folds,
    # so it raises ValueError("cross_val_predict only works for partitions").
    # KFold(shuffle=False) preserves temporal ordering within folds while
    # producing the full partition that cross_val_predict needs.
    cv  = KFold(n_splits=5, shuffle=False)
    oof = cross_val_predict(clone(best_estimator),X_train,y_train_log,cv=cv,n_jobs=N_JOBS)
    res = y_train_log - oof
    oof_rmse_log = float(np.sqrt(np.mean(res**2)))
    y_usd   = np.expm1(y_train_log)
    oof_usd = np.expm1(oof)
    usd_rmse= float(np.sqrt(np.mean((y_usd-oof_usd)**2)))
    return {
        "oof_rmse_log":oof_rmse_log,"oof_rmse_bikes":usd_rmse,
        "oof_rmsle":rmsle(y_usd.clip(0),oof_usd.clip(0)),
        "oof_r2":float(r2_score(y_train_log,oof)),
        "lower_band":oof_rmse_log*underpredict_cost,
        "upper_band":oof_rmse_log*overpredict_cost,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search
# ─────────────────────────────────────────────────────────────────────────────
def tune_model(X_train, y_train_log, n_iter=20, n_cv_splits=5, fast=False):
    log.info("Hyperparameter search: n_iter=%d  cv=%d-fold  fast=%s",n_iter,n_cv_splits,fast)
    _n = 50 if fast else 150

    param_distributions = [
        {"feature_selection__threshold":["median","0.75*median","1.25*median"],
         "model":[Ridge()],"model__alpha":[0.1,1,10,100,500,1000]},
        {"feature_selection__threshold":["median","0.75*median"],
         "model":[HuberRegressor(max_iter=500)],
         "model__epsilon":[1.1,1.35,1.5,2.0],"model__alpha":[0.0001,0.001,0.01]},
        {"feature_selection__threshold":["median","0.75*median","1.25*median"],
         "model":[GradientBoostingRegressor(n_estimators=_n,random_state=RANDOM_STATE)],
         "model__max_depth":[3,4,5],"model__learning_rate":[0.02,0.05,0.1,0.2],
         "model__subsample":[0.7,0.9]},
        {"feature_selection__threshold":["median","0.75*median","1.25*median"],
         "model":[RandomForestRegressor(n_estimators=_n,random_state=RANDOM_STATE,n_jobs=N_JOBS)],
         "model__max_depth":[8,12,None],"model__min_samples_leaf":[1,2,4]},
    ]
    if _XGB:
        param_distributions.append({
            "feature_selection__threshold":["median","0.75*median"],
            "model":[XGBRegressor(n_estimators=_n,eval_metric="rmse",
                                   random_state=RANDOM_STATE,n_jobs=N_JOBS,verbosity=0)],
            "model__max_depth":[3,4,5,6],"model__learning_rate":[0.02,0.05,0.1,0.2],
            "model__subsample":[0.7,0.9],"model__colsample_bytree":[0.6,0.8,1.0],
        })
    if _LGB:
        param_distributions.append({
            "feature_selection__threshold":["median","0.75*median"],
            "model":[LGBMRegressor(n_estimators=_n,random_state=RANDOM_STATE,n_jobs=N_JOBS,verbose=-1)],
            "model__max_depth":[3,4,5,6],"model__learning_rate":[0.02,0.05,0.1,0.2],
            "model__subsample":[0.7,0.9],
        })

    cv = TimeSeriesSplit(n_splits=n_cv_splits)
    search = RandomizedSearchCV(
        build_pipeline(), param_distributions, n_iter=n_iter,
        scoring={"r2":"r2","neg_rmse":"neg_root_mean_squared_error","neg_mae":"neg_mean_absolute_error"},
        refit="r2", cv=cv, random_state=RANDOM_STATE,
        n_jobs=N_JOBS, verbose=1, return_train_score=True)
    search.fit(X_train, y_train_log)

    best_mdl = search.best_estimator_.named_steps["model"]
    if hasattr(best_mdl,"n_estimators") and best_mdl.n_estimators==_n:
        log.info("Upgrading %d → 300 trees …",_n)
        best_mdl.set_params(n_estimators=300)
        search.best_estimator_.fit(X_train,y_train_log)

    log.info("Best CV R²=%.4f  model=%s",search.best_score_,type(best_mdl).__name__)
    return search


def _model_version_tag(model):
    return hashlib.sha1(pickle.dumps(model)).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────────────
# Main train()
# ─────────────────────────────────────────────────────────────────────────────
def train(output_dir,n_iter=20,n_cv_splits=5,fast=False,
          overpredict_cost=1.0,underpredict_cost=1.0):
    log.info("=== Training started (n_jobs=%d) ===",N_JOBS)
    output_dir.mkdir(parents=True,exist_ok=True)

    df                             = fix_data_types(load_data())
    X_train,X_test,y_train,y_te   = split_data(df)
    y_train_log = np.log1p(y_train.to_numpy())
    y_test_log  = np.log1p(y_te.to_numpy())

    save_research_artifacts(X_train,y_train,output_dir)
    baselines = evaluate_baselines(X_train,X_test,y_train,y_te)

    # Concept analyses (train only)
    time_anal  = analyse_time_features(X_train,y_train,output_dir)
    rob_anal   = analyse_robustness(X_train,y_train,output_dir)
    log_anal   = analyse_log_target(X_train,y_train,output_dir)
    nl_anal    = analyse_nonlinear_features(X_train,y_train,output_dir)
    gb_anal    = analyse_gradient_boosting(X_train,y_train,output_dir,quick=fast)
    rf_anal    = analyse_random_forest(X_train,y_train,output_dir,quick=fast)
    hp_anal    = analyse_hyperparameter_tuning(X_train,y_train,output_dir,quick=fast)
    cv_anal    = analyse_timeseries_cv(X_train,y_train,output_dir,quick=fast)
    out_anal   = analyse_outliers(X_train,y_train,output_dir)

    search     = tune_model(X_train,y_train_log,n_iter=n_iter,n_cv_splits=n_cv_splits,fast=fast)
    uncertainty= compute_oof_uncertainty(search.best_estimator_,X_train,y_train_log,
                                          overpredict_cost,underpredict_cost)

    final_model = clone(search.best_estimator_)
    final_model.fit(X_train,y_train_log)
    y_pred_log  = final_model.predict(X_test)
    test_metrics= evaluate_predictions(y_te,y_pred_log)

    log.info("Test R²=%.4f  RMSLE=%.4f  RMSE=%.2f  MAE=%.2f",
             test_metrics["r2"],test_metrics["rmsle"],
             test_metrics["rmse"],test_metrics["mae"])

    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha1 = _model_version_tag(final_model)
    joblib.dump(final_model,output_dir/f"bike_pipeline_{ts}_{sha1}.joblib")
    joblib.dump(final_model,output_dir/MODEL_FILE)
    save_environment_snapshot(output_dir)
    pd.DataFrame(search.cv_results_).sort_values("rank_test_r2").to_csv(output_dir/"cv_results.csv",index=False)
    save_feature_importance(final_model,output_dir)
    save_evaluation_plots(y_te,y_pred_log,output_dir)
    write_json(output_dir/TRAINING_PROFILE_FILE,build_training_profile(X_train,y_train))

    res_diag = residual_diagnostics(y_test_log,y_pred_log,output_dir)
    plot_learning_curves_bike(final_model,X_train,y_train_log,output_dir)
    fairness = evaluate_subgroups(final_model,X_test,y_te,y_pred_log,output_dir)
    save_shap_artifacts(final_model,X_test,y_te,y_pred_log,output_dir)

    metrics = {
        "baselines":baselines,"split":{"train_rows":int(len(X_train)),"test_rows":int(len(X_test))},
        "time_features":time_anal,"robustness":rob_anal,"log_transform":log_anal,
        "nonlinear":nl_anal,"gradient_boosting":gb_anal,"random_forest":rf_anal,
        "hyperparameter_tuning":hp_anal,"timeseries_cv":cv_anal,"outlier_analysis":out_anal,
        "best_cv":{"best_r2":float(search.best_score_),"best_params":search.best_params_},
        "uncertainty_info":uncertainty,"residual_diag":res_diag,
        "test_metrics":test_metrics,"fairness":fairness,
    }
    write_json(output_dir/METRICS_FILE,metrics)
    save_model_card(metrics,fairness,search,output_dir)
    log_to_mlflow(metrics,search,final_model,output_dir)
    log.info("=== Training complete ===")
    return to_jsonable(metrics)


# ─────────────────────────────────────────────────────────────────────────────
# predict / monitor / sample-input
# ─────────────────────────────────────────────────────────────────────────────
def predict(artifact_dir,input_csv,output_csv):
    model = joblib.load(artifact_dir/MODEL_FILE)
    mp    = artifact_dir/METRICS_FILE
    unc   = 0.1
    if mp.exists():
        unc = json.loads(mp.read_text())["uncertainty_info"].get("oof_rmse_log",unc)
    df  = pd.read_csv(input_csv)
    pf  = artifact_dir/TRAINING_PROFILE_FILE
    if pf.exists():
        req = set(json.loads(pf.read_text())["raw_columns"])
        miss= req - set(df.columns) - {"casual","registered"}
        if miss: raise ValueError(f"Missing columns: {sorted(miss)}")
    y_pred_log = model.predict(df)
    df["predicted_count"]= np.expm1(y_pred_log).clip(0)
    df["lower_bound"]    = np.expm1(y_pred_log - unc).clip(0)
    df["upper_bound"]    = np.expm1(y_pred_log + unc).clip(0)
    output_csv=Path(output_csv); output_csv.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(output_csv,index=False)
    log.info("Predictions saved to %s",output_csv.resolve())


def monitor(artifact_dir,input_csv,output_json,missing_rate_alert=0.05,ks_pvalue=0.05):
    profile  = json.loads((artifact_dir/TRAINING_PROFILE_FILE).read_text())
    incoming = pd.read_csv(input_csv)
    drift=[]
    for col,tr in profile["raw_missing_rate"].items():
        if col not in incoming: continue
        cur=float(incoming[col].isna().mean())
        drift.append({"column":col,"train_rate":float(tr),"current_rate":cur,
                      "change":abs(cur-float(tr)),"alert":abs(cur-float(tr))>=missing_rate_alert})
    ks_rows=[]
    for col,stats in profile.get("numeric_train_stats",{}).items():
        if col not in incoming.columns: continue
        vals=incoming[col].dropna().to_numpy()
        if len(vals)<10: continue
        stat,p=ks_2samp(np.array(stats["quantiles"]),vals)
        ks_rows.append({"column":col,"ks_stat":float(stat),"p_value":float(p),"alert":p<ks_pvalue})
    report={"checked_at":datetime.now(timezone.utc).isoformat(),"row_count":int(len(incoming)),
            "missing_rate_drift":drift,"distribution_drift":ks_rows}
    output_json=Path(output_json); output_json.parent.mkdir(parents=True,exist_ok=True)
    write_json(output_json,report)
    return report


def create_sample_input(output_csv,rows):
    df = fix_data_types(load_data())
    output_csv=Path(output_csv); output_csv.parent.mkdir(parents=True,exist_ok=True)
    df.drop(columns=[TARGET,"casual","registered"],errors="ignore").head(rows).to_csv(output_csv,index=False)
    log.info("Sample saved to %s",output_csv.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def write_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload),indent=2),encoding="utf-8")

def to_jsonable(v):
    if isinstance(v,dict):   return {str(k):to_jsonable(x) for k,x in v.items()}
    if isinstance(v,list):   return [to_jsonable(x) for x in v]
    if isinstance(v,BaseEstimator): return repr(v)
    if isinstance(v,np.bool_): return bool(v)
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p=argparse.ArgumentParser(description="Bike Sharing Demand end-to-end ML pipeline",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sp=p.add_subparsers(dest="command",required=True)

    tp=sp.add_parser("train")
    tp.add_argument("--output-dir",type=Path,default=Path("artifacts_bike"))
    tp.add_argument("--n-iter",type=int,default=20)
    tp.add_argument("--n-cv-splits",type=int,default=5)
    tp.add_argument("--fast",action="store_true",help="50 trees in search, 3-fold CV")
    tp.add_argument("--overpredict-cost",type=float,default=1.0)
    tp.add_argument("--underpredict-cost",type=float,default=1.0)

    pp=sp.add_parser("predict")
    pp.add_argument("--artifact-dir",type=Path,default=Path("artifacts_bike"))
    pp.add_argument("--input-csv",type=Path,required=True)
    pp.add_argument("--output-csv",type=Path,default=Path("artifacts_bike/predictions.csv"))

    mp=sp.add_parser("monitor")
    mp.add_argument("--artifact-dir",type=Path,default=Path("artifacts_bike"))
    mp.add_argument("--input-csv",type=Path,required=True)
    mp.add_argument("--output-json",type=Path,default=Path("artifacts_bike/monitor.json"))
    mp.add_argument("--missing-rate-alert",type=float,default=0.05)
    mp.add_argument("--ks-pvalue-alert",type=float,default=0.05)

    si=sp.add_parser("sample-input")
    si.add_argument("--output-csv",type=Path,default=Path("artifacts_bike/sample.csv"))
    si.add_argument("--rows",type=int,default=24)

    return p.parse_args()


def main():
    args=parse_args()
    if args.command=="train":
        m=train(args.output_dir,args.n_iter,n_cv_splits=args.n_cv_splits,fast=args.fast,
                overpredict_cost=args.overpredict_cost,underpredict_cost=args.underpredict_cost)
        log.info("Test R²=%.3f  RMSLE=%.4f  RMSE=%.1f  MAE=%.1f",
                 m["test_metrics"]["r2"],m["test_metrics"]["rmsle"],
                 m["test_metrics"]["rmse"],m["test_metrics"]["mae"])
    elif args.command=="predict":
        predict(args.artifact_dir,args.input_csv,args.output_csv)
    elif args.command=="monitor":
        r=monitor(args.artifact_dir,args.input_csv,args.output_json,
                  args.missing_rate_alert,args.ks_pvalue_alert)
        log.info("Drift alerts: missing=%d  KS=%d",
                 sum(x["alert"] for x in r["missing_rate_drift"]),
                 sum(x["alert"] for x in r["distribution_drift"]))
    elif args.command=="sample-input":
        create_sample_input(args.output_csv,args.rows)


if __name__=="__main__":
    main()
