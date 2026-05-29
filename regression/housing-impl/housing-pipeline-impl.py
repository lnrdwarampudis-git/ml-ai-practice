"""
housing_pipeline.py
===================
Industry-standard end-to-end ML pipeline for Ames Housing price prediction.

Data (exact pattern as specified):
    housing    = fetch_openml(data_id=42165, as_frame=True, parser="auto")
    housing_df = housing.frame

Target: SalePrice — residential home sale price (USD)
Dataset: 2930 rows × 82 columns  |  80 features, all types, 19 with missing data

Mirrors every architectural pattern from titanic-ml-pipeline.py.
Primary focus: Missing Data & Robustness

New concepts explored (A–L):
  A. Missing data taxonomy      — MCAR / MAR / MNAR classification for each
                                   of the 19 columns with missing values.
                                   Pool/Alley/Fence missing = "feature absent"
                                   (MNAR), LotFrontage missing ≈ MAR
  B. Advanced imputation        — Simple (median/mode), KNN imputation,
                                   Iterative (MICE) imputation; comparison of
                                   imputation strategies by downstream R²
  C. Outlier detection          — IQR method, Isolation Forest, Cook's distance;
                                   comparison of outlier-removal strategies
  D. Robust regression          — HuberRegressor vs OLS Ridge: show how Huber
                                   loss dampens large residuals from outlier sales
  E. Log target transform       — SalePrice → log1p(SalePrice) and back;
                                   residual normality improvement
  F. Ordinal encoding           — Quality features Ex>Gd>TA>Fa>Po → 5>4>3>2>1;
                                   comparison with naive OHE
  G. Polynomial + interactions  — GrLivArea², OverallQual×GrLivArea, TotalSF,
                                   YearBuilt×OverallQual; domain-knowledge basis
  H. Spatial imputation         — LotFrontage imputed per Neighborhood median
                                   (spatial proxy), vs global median baseline
  I. Regularisation on wide     — Ridge/Lasso/ElasticNet path on 80+ OHE-expanded
     feature sets               columns; Lasso's role in sparse selection
  J. Residual diagnostics       — Breusch-Pagan, Q-Q plot, leverage-residual plot;
                                   heteroscedasticity on SalePrice vs log(SalePrice)
  K. Learning curves            — n=2930 vs n=1460 (Kaggle split); train-CV gap
  L. Subgroup disparity         — RMSE by Neighborhood, BldgType, HouseStyle;
                                   pricing disparities in physical housing markets

Industry-standard regression metrics: MAE, RMSE, R², MAPE, MedAE

Usage:
  python housing_pipeline.py train   --output-dir artifacts_housing
  python housing_pipeline.py predict --artifact-dir artifacts_housing --input-csv sample.csv
  python housing_pipeline.py monitor --artifact-dir artifacts_housing --input-csv new.csv
  python housing_pipeline.py sample-input --output-csv sample.csv --rows 10
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

_MPLCONFIGDIR = Path("artifacts_housing") / ".matplotlib"
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
    ExtraTreesRegressor, GradientBoostingRegressor,
    IsolationForest, RandomForestRegressor,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import (
    BayesianRidge, ElasticNet, ElasticNetCV,
    HuberRegressor, Lasso, LassoCV,
    LinearRegression, Ridge, RidgeCV,
)
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, median_absolute_error, r2_score,
)
from sklearn.model_selection import (
    KFold, RandomizedSearchCV, cross_val_predict,
    cross_val_score, learning_curve, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, PowerTransformer, RobustScaler,
    SplineTransformer, StandardScaler,
)

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

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE          = 42
TARGET                = "SalePrice"
MODEL_FILE            = "housing_pipeline.joblib"
METRICS_FILE          = "metrics.json"
TRAINING_PROFILE_FILE = "training_profile.json"
MODEL_CARD_FILE       = "model_card.json"
ENVIRONMENT_FILE      = "environment.json"
N_JOBS                = int(os.environ.get("ML_N_JOBS", 1))

# Ordinal quality mappings (domain knowledge: Ex=5 > Gd=4 > TA=3 > Fa=2 > Po=1 > NA=0)
_QUAL_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "nan": 0, "None": 0}
_QUAL_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]
_FENCE_MAP = {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "nan": 0, "None": 0}

# Columns where NaN means "feature absent", not "unknown"
ABSENT_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]
# Columns best imputed by spatial (Neighborhood) median
SPATIAL_IMPUTE_COLS = ["LotFrontage"]

# Nominal categorical (for OHE)
NOMINAL_COLS = [
    "MSZoning", "Street", "Alley", "LandContour", "LotConfig",
    "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",
    "Foundation", "Heating", "CentralAir", "Electrical", "GarageType",
    "GarageFinish", "PavedDrive", "MiscFeature", "SaleType", "SaleCondition",
    "Functional", "LandSlope",
]
# Numeric features kept as-is (after imputation)
NUMERIC_COLS = [
    "LotFrontage", "LotArea", "OverallQual", "OverallCond",
    "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
    "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
    "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
    "MoSold", "YrSold",
]


def get_ordinal_encoded_cols() -> list[str]:
    return [c + "_ord" for c in _QUAL_COLS] + ["Fence_ord"]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Exact loader as specified:
        housing    = fetch_openml(data_id=42165, as_frame=True, parser="auto")
        housing_df = housing.frame
    """
    log.info("Loading Ames Housing dataset from OpenML (data_id=42165) …")
    _HOUSING_COLS = {"SalePrice", "GrLivArea", "OverallQual", "Neighborhood",
                     "YearBuilt", "LotArea"}
    _CANDIDATE_IDS  = [42165, 41211, 42731]
    _NAME_FALLBACKS = ["house_prices_advanced_regression", "ames", "AmesHousing"]

    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Drop Id column if present
        for id_col in ["Id", "id", "ID"]:
            if id_col in df.columns:
                df = df.drop(columns=[id_col])
        return df

    def _is_housing(df: pd.DataFrame) -> bool:
        return len(_HOUSING_COLS & set(df.columns)) >= 4

    for data_id in _CANDIDATE_IDS:
        try:
            log.info("Trying fetch_openml(data_id=%d) …", data_id)
            raw = fetch_openml(data_id=data_id, as_frame=True, parser="auto").frame
            df  = _normalise(raw)
            if _is_housing(df):
                log.info("✓ data_id=%d accepted  shape=%s", data_id, df.shape)
                return df
            log.warning("data_id=%d wrong dataset — trying next.", data_id)
        except Exception as exc:
            log.warning("data_id=%d failed: %s", data_id, exc)

    for name in _NAME_FALLBACKS:
        try:
            raw = fetch_openml(name=name, as_frame=True, parser="auto").frame
            df  = _normalise(raw)
            if _is_housing(df):
                log.info("✓ name='%s' accepted  shape=%s", name, df.shape)
                return df
        except Exception as exc:
            log.warning("name='%s' failed: %s", name, exc)

    log.warning("All OpenML sources failed — using synthetic Ames data.")
    return _make_synthetic_ames()


def _make_synthetic_ames() -> pd.DataFrame:
    """Synthetic fallback with realistic Ames Housing distributions."""
    rng = np.random.default_rng(RANDOM_STATE); n = 2930
    qual  = rng.integers(3, 11, n)
    area  = rng.integers(500, 5000, n)
    year  = rng.integers(1872, 2011, n)
    price = (qual * 15000 + area * 50 + (year - 1872) * 500
             + rng.exponential(10000, n)).clip(50000, 800000)
    df = pd.DataFrame({
        "MSZoning":     rng.choice(["RL","RM","C (all)","FV","RH"], n, p=[0.79,0.11,0.04,0.03,0.03]),
        "LotFrontage":  np.where(rng.random(n)<0.17, np.nan, rng.integers(21,313,n).astype(float)),
        "LotArea":      rng.integers(1300, 215245, n).astype(float),
        "Neighborhood": rng.choice(["NAmes","CollgCr","OldTown","Edwards","Somerst",
                                    "NridgHt","Gilbert","Sawyer","NWAmes","SawyerW"], n),
        "BldgType":     rng.choice(["1Fam","TwnhsE","Duplex","Twnhs","2fmCon"], n,
                                    p=[0.83,0.09,0.04,0.02,0.02]),
        "HouseStyle":   rng.choice(["1Story","2Story","1.5Fin","SLvl","SFoyer","2.5Unf"], n,
                                    p=[0.50,0.31,0.11,0.05,0.02,0.01]),
        "OverallQual":  qual.astype(float),
        "OverallCond":  rng.integers(1, 10, n).astype(float),
        "YearBuilt":    year.astype(float),
        "YearRemodAdd": np.clip(year + rng.integers(0, 30, n), year, 2010).astype(float),
        "ExterQual":    rng.choice(["Ex","Gd","TA","Fa"], n, p=[0.08,0.41,0.48,0.03]),
        "ExterCond":    rng.choice(["Ex","Gd","TA","Fa","Po"], n, p=[0.01,0.10,0.87,0.02,0.00]),
        "BsmtQual":     rng.choice(["Ex","Gd","TA","Fa",None], n, p=[0.09,0.43,0.41,0.04,0.03]),
        "KitchenQual":  rng.choice(["Ex","Gd","TA","Fa"], n, p=[0.07,0.40,0.50,0.03]),
        "GrLivArea":    area.astype(float),
        "TotalBsmtSF":  (area * rng.uniform(0.3, 0.7, n)).astype(int).astype(float),
        "1stFlrSF":     (area * rng.uniform(0.3, 0.6, n)).astype(int).astype(float),
        "2ndFlrSF":     np.where(rng.random(n)<0.47, 0,
                                  (area * rng.uniform(0.2, 0.5, n)).astype(int)).astype(float),
        "GarageArea":   rng.integers(0, 1418, n).astype(float),
        "GarageCars":   rng.integers(0, 5, n).astype(float),
        "GarageYrBlt":  np.where(rng.random(n)<0.05, np.nan,
                                  rng.integers(1895, 2010, n).astype(float)),
        "Fireplaces":   rng.integers(0, 4, n).astype(float),
        "FireplaceQu":  rng.choice(["Ex","Gd","TA","Fa","Po",None], n,
                                    p=[0.03,0.25,0.21,0.02,0.01,0.48]),
        "PoolArea":     np.where(rng.random(n)<0.006, rng.integers(80,800,n).astype(float), 0.0),
        "PoolQC":       np.where(rng.random(n)<0.006, rng.choice(["Ex","Gd","TA"], n), None),
        "Alley":        np.where(rng.random(n)<0.07, rng.choice(["Grvl","Pave"], n), None),
        "Fence":        np.where(rng.random(n)<0.18,
                                  rng.choice(["GdPrv","MnPrv","GdWo","MnWw"], n), None),
        "OpenPorchSF":  rng.integers(0, 742, n).astype(float),
        "WoodDeckSF":   np.where(rng.random(n)<0.53, rng.integers(1, 857, n).astype(float), 0.0),
        "MoSold":       rng.integers(1, 13, n).astype(float),
        "YrSold":       rng.choice([2006,2007,2008,2009,2010], n).astype(float),
        "SalePrice":    price,
    })
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric columns and handle known string-encoded NaN values."""
    df = df.copy()
    # Some OpenML parsers return NaN-like strings
    df = df.replace({"nan": np.nan, "NA": np.nan, "": np.nan})
    for col in df.columns:
        if col == TARGET: continue
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            # Only use numeric conversion if most values are actually numeric
            non_null = df[col].dropna()
            if len(non_null) > 0:
                numeric_rate = converted.notna().sum() / len(non_null)
                if numeric_rate > 0.8:
                    df[col] = converted
        except Exception:
            pass
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype(float)
    return df


def split_data(df: pd.DataFrame):
    """Stratified 80/20 on SalePrice quartile bins."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    q_bins = pd.qcut(y, q=4, labels=False, duplicates="drop")
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=q_bins)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    r = df.isna().agg(["sum", "mean"]).T.rename(
        columns={"sum": "missing_count", "mean": "missing_rate"})
    r["dtype"] = df.dtypes.astype(str)
    return r.sort_values("missing_rate", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Concept A: Missing data taxonomy
# ─────────────────────────────────────────────────────────────────────────────
def analyse_missing_taxonomy(X_train: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """
    Concept A: Classify each missing column as MCAR / MAR / MNAR.
    MNAR (Missing Not At Random):  PoolQC, Fence, Alley, FireplaceQu —
                                    missing because the feature is absent.
    MAR  (Missing At Random):      LotFrontage — missing rate correlates
                                    with Neighborhood, not SalePrice directly.
    MCAR (Missing Completely):     Electrical — single missing value, random.

    Appropriate strategies:
      MNAR absent → fill with "None" / 0 (feature absent, not unknown)
      MAR spatial → impute by Neighborhood median (Concept H)
      MCAR few   → simple median imputation is fine
    """
    log.info("Concept A: Missing data taxonomy …")
    miss_df = missingness_report(X_train)
    miss_df = miss_df[miss_df["missing_count"] > 0].copy()

    taxonomy = []
    for col in miss_df.index:
        rate = float(miss_df.loc[col, "missing_rate"])
        if col in ABSENT_COLS:
            kind = "MNAR_absent"
            strategy = "fill_None_or_0"
            reason = "NaN means feature does not exist for this property"
        elif col in SPATIAL_IMPUTE_COLS:
            kind = "MAR_spatial"
            strategy = "neighborhood_median"
            reason = "Missing rate varies by Neighborhood — spatially structured"
        elif col in ["GarageYrBlt"]:
            kind = "MAR_conditional"
            strategy = "garage_yearbuilt_or_0"
            reason = "Missing when no garage — use YearBuilt as proxy"
        elif rate < 0.01:
            kind = "MCAR"
            strategy = "median_mode"
            reason = "Very low rate — likely random data entry issue"
        else:
            kind = "MAR_general"
            strategy = "iterative_mice"
            reason = "Moderate rate, conditional on other features"
        taxonomy.append({"column": col, "missing_rate": round(rate, 4),
                         "type": kind, "strategy": strategy, "reason": reason})

    tax_df = pd.DataFrame(taxonomy).sort_values("missing_rate", ascending=False)
    tax_df.to_csv(output_dir / "missing_taxonomy.csv", index=False)

    # Bar chart: missing rates coloured by type
    plt.figure(figsize=(11, max(4, len(tax_df) * 0.4 + 1.5)))
    colors = {"MNAR_absent": "#E45756", "MAR_spatial": "#F58518",
              "MAR_conditional": "#FDBE6A", "MCAR": "#54A24B", "MAR_general": "#4C78A8"}
    bar_colors = [colors.get(t, "#888") for t in tax_df["type"]]
    plt.barh(tax_df["column"], tax_df["missing_rate"], color=bar_colors)
    plt.xlabel("Missing rate"); plt.title("Concept A: Missing data taxonomy\nRed=MNAR(absent) Orange=MAR(spatial) Green=MCAR Blue=MAR(general)")
    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=k) for k, c in colors.items()]
    plt.legend(handles=legend, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_missing_taxonomy.png", dpi=160); plt.close()

    write_json(output_dir / "missing_taxonomy.json", taxonomy)
    log.info("Missing taxonomy: %d columns, %d MNAR_absent, %d MAR, %d MCAR",
             len(taxonomy),
             sum(1 for r in taxonomy if "MNAR" in r["type"]),
             sum(1 for r in taxonomy if "MAR" in r["type"]),
             sum(1 for r in taxonomy if r["type"] == "MCAR"))
    return {"taxonomy": taxonomy}


# ─────────────────────────────────────────────────────────────────────────────
# Concept H: Spatial imputation for LotFrontage
# ─────────────────────────────────────────────────────────────────────────────
class SpatialImputer(BaseEstimator, TransformerMixin):
    """
    Concept H: Impute LotFrontage using Neighborhood-level median.
    LotFrontage (linear feet of street connected to property) is MAR —
    its missingness correlates with Neighborhood. Houses on cul-de-sacs
    and corner lots in the same Neighborhood have similar frontage.
    Falls back to global median for Neighborhoods with insufficient data.
    """
    def __init__(self, target_col: str = "LotFrontage",
                 group_col: str = "Neighborhood"):
        self.target_col = target_col
        self.group_col  = group_col

    def fit(self, X: pd.DataFrame, y=None) -> "SpatialImputer":
        if self.target_col not in X.columns:
            self.medians_ = {}; self.global_median_ = 0.0; return self
        tmp = pd.DataFrame({
            "target": pd.to_numeric(X[self.target_col], errors="coerce"),
            "group":  X[self.group_col].astype(str) if self.group_col in X.columns else "all",
        })
        self.medians_       = tmp.groupby("group")["target"].median().to_dict()
        self.global_median_ = float(tmp["target"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.target_col not in X.columns: return X
        grp = X[self.group_col].astype(str) if self.group_col in X.columns else pd.Series("all", index=X.index)
        for i in X.index:
            if pd.isna(X.loc[i, self.target_col]):
                X.loc[i, self.target_col] = self.medians_.get(
                    str(grp.loc[i]), self.global_median_)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Concept B: Imputation strategy comparison
# ─────────────────────────────────────────────────────────────────────────────
def analyse_imputation_strategies(X_train: pd.DataFrame, y_train: pd.Series,
                                   output_dir: Path) -> dict[str, Any]:
    """
    Concept B: Compare Simple median, KNN (k=5), and Iterative (MICE) imputation.
    Applied to the numeric columns with missing values. Evaluated by downstream
    Ridge R² so imputation quality is measured by its effect on prediction.
    """
    log.info("Concept B: Imputation strategy comparison …")
    num_cols_with_na = [c for c in NUMERIC_COLS
                        if c in X_train.columns and X_train[c].isna().any()]
    if not num_cols_with_na:
        log.info("  No numeric columns with NA — skipping imputation comparison.")
        return {}

    X_sub = X_train[num_cols_with_na].apply(pd.to_numeric, errors="coerce")
    y_np  = np.log1p(y_train.to_numpy())  # log-transform target for comparison
    cv    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    strategies = {
        "Simple (median)": SimpleImputer(strategy="median"),
        "KNN (k=5)":       KNNImputer(n_neighbors=5),
        "Iterative (MICE)":IterativeImputer(estimator=BayesianRidge(),
                                             max_iter=10, random_state=RANDOM_STATE),
    }
    results = {}
    for name, imp in strategies.items():
        try:
            X_imp = imp.fit_transform(X_sub)
            r2s   = cross_val_score(Ridge(alpha=1.0), X_imp, y_np, cv=cv, scoring="r2")
            results[name] = {"r2_mean": float(r2s.mean()), "r2_std": float(r2s.std())}
            log.info("  %-25s R²=%.4f±%.4f", name, r2s.mean(), r2s.std())
        except Exception as exc:
            log.warning("  %s failed: %s", name, exc)
            results[name] = {"r2_mean": float("nan"), "r2_std": float("nan")}

    # Plot
    names  = list(results.keys())
    r2s    = [results[n]["r2_mean"] for n in names]
    errs   = [results[n]["r2_std"]  for n in names]
    plt.figure(figsize=(7, 4))
    plt.bar(names, r2s, yerr=errs, capsize=5, color=["#4C78A8","#1D9E75","#E45756"])
    plt.ylabel("CV R² (downstream Ridge on numeric features only)")
    plt.title("Concept B: Imputation strategy comparison\nHigher = better imputation quality")
    plt.tight_layout(); plt.savefig(output_dir/"plot_imputation_comparison.png", dpi=160); plt.close()
    write_json(output_dir/"imputation_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept C: Outlier detection
# ─────────────────────────────────────────────────────────────────────────────
def analyse_outliers(X_train: pd.DataFrame, y_train: pd.Series,
                     output_dir: Path) -> dict[str, Any]:
    """
    Concept C: Detect outliers using IQR and Isolation Forest.
    The Ames Housing paper (De Cock 2011) explicitly flags 5 houses
    with GrLivArea > 4000 and low SalePrice as outliers to remove.
    Shows how removing them improves RMSE on the remaining data.
    """
    log.info("Concept C: Outlier detection …")
    y_log = np.log1p(y_train.to_numpy())
    n     = len(X_train)

    # IQR on log(SalePrice)
    q1, q3  = np.percentile(y_log, [25, 75])
    iqr     = q3 - q1
    iqr_mask= (y_log >= q1 - 1.5*iqr) & (y_log <= q3 + 1.5*iqr)
    n_iqr   = int((~iqr_mask).sum())

    # Isolation Forest on key features
    key_cols = [c for c in ["GrLivArea","TotalBsmtSF","LotArea","OverallQual"]
                if c in X_train.columns]
    if key_cols:
        X_key = X_train[key_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        iso   = IsolationForest(contamination=0.02, random_state=RANDOM_STATE)
        iso_labels = iso.fit_predict(X_key)
        iso_mask   = iso_labels == 1
        n_iso = int((iso_labels == -1).sum())
    else:
        iso_mask = np.ones(n, dtype=bool)
        n_iso    = 0

    # Known hard outliers: GrLivArea > 4000 sqft with low price
    if "GrLivArea" in X_train.columns:
        gr = pd.to_numeric(X_train["GrLivArea"], errors="coerce").fillna(0)
        area_mask = ~((gr > 4000) & (y_train < 200000))
        n_area = int((~area_mask).sum())
    else:
        area_mask = np.ones(n, dtype=bool)
        n_area    = 0

    # Compare R² with/without outliers
    num_cols = [c for c in NUMERIC_COLS if c in X_train.columns]
    X_num = X_train[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    cv    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = {}
    for label, mask in [("All data", np.ones(n, dtype=bool)),
                         ("IQR removal", iqr_mask),
                         ("Area outlier removal", area_mask),
                         ("Isolation Forest", iso_mask)]:
        if mask.sum() < 50: continue
        X_m = X_num[mask]; y_m = y_log[mask] if isinstance(y_log, np.ndarray) else y_log.values[mask]
        r2s = cross_val_score(Ridge(alpha=1.0), X_m, y_m, cv=cv, scoring="r2")
        comparison[label] = {"n_kept": int(mask.sum()), "n_removed": int((~mask).sum()),
                              "r2_mean": float(r2s.mean()), "r2_std": float(r2s.std())}
        log.info("  %-30s  n=%4d  R²=%.4f", label, mask.sum(), r2s.mean())

    # Scatter: GrLivArea vs SalePrice highlighting outliers
    if "GrLivArea" in X_train.columns:
        plt.figure(figsize=(7, 5))
        gr = pd.to_numeric(X_train["GrLivArea"], errors="coerce").fillna(0)
        is_outlier = ~area_mask
        plt.scatter(gr[~is_outlier], y_train.values[~is_outlier],
                    alpha=0.3, s=8, color="#4C78A8", label="Normal")
        plt.scatter(gr[is_outlier], y_train.values[is_outlier],
                    alpha=0.9, s=40, color="#E45756", label="Outlier (GrLivArea>4000 & Price<200k)", zorder=5)
        plt.xlabel("GrLivArea (sqft)"); plt.ylabel("SalePrice ($)")
        plt.title("Concept C: Outlier detection\nRed = known outliers from De Cock 2011 paper")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(output_dir/"plot_outlier_detection.png", dpi=160); plt.close()

    write_json(output_dir/"outlier_analysis.json", comparison)
    return {"n_iqr_outliers": n_iqr, "n_iso_outliers": n_iso,
            "n_area_outliers": n_area, "comparison": comparison}


# ─────────────────────────────────────────────────────────────────────────────
# Concept D: Robust regression (Huber loss)
# ─────────────────────────────────────────────────────────────────────────────
def analyse_robust_regression(X_train: pd.DataFrame, y_train: pd.Series,
                               output_dir: Path) -> dict[str, Any]:
    """
    Concept D: Compare OLS Ridge vs HuberRegressor.
    Huber loss = L2 for |residual| < delta, L1 beyond — resists extreme SalePrice outliers.
    Shows coefficient stability: Huber coefficients are more consistent across folds.
    """
    log.info("Concept D: Robust regression (Huber vs OLS) …")
    num_cols = [c for c in NUMERIC_COLS if c in X_train.columns]
    X_num = X_train[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_sc  = StandardScaler().fit_transform(X_num)
    y_log = np.log1p(y_train.to_numpy())
    cv    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Ridge (OLS-like)":     Ridge(alpha=1.0),
        "HuberRegressor ε=1.35":HuberRegressor(epsilon=1.35, max_iter=500),
        "HuberRegressor ε=2.0": HuberRegressor(epsilon=2.0, max_iter=500),
    }
    results = {}
    for name, mdl in models.items():
        r2s  = cross_val_score(mdl, X_sc, y_log, cv=cv, scoring="r2")
        rmse = -cross_val_score(mdl, X_sc, y_log, cv=cv,
                                scoring="neg_root_mean_squared_error")
        results[name] = {"r2_mean": float(r2s.mean()), "r2_std": float(r2s.std()),
                         "rmse_mean": float(rmse.mean())}
        log.info("  %-30s  R²=%.4f±%.4f  RMSE=%.4f", name,
                 r2s.mean(), r2s.std(), rmse.mean())

    # Residual comparison plot
    ridge_oof = cross_val_predict(Ridge(alpha=1.0), X_sc, y_log, cv=cv)
    huber_oof = cross_val_predict(HuberRegressor(epsilon=1.35, max_iter=500),
                                  X_sc, y_log, cv=cv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, oof, name, color in [(axes[0], ridge_oof, "Ridge (OLS)", "#4C78A8"),
                                   (axes[1], huber_oof, "Huber (ε=1.35)", "#E45756")]:
        res = y_log - oof
        ax.scatter(oof, res, alpha=0.2, s=6, color=color)
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_title(f"{name}\nR²={r2_score(y_log, oof):.4f}  std(res)={res.std():.4f}")
        ax.set_xlabel("Predicted log(SalePrice)"); ax.set_ylabel("Residual")
    plt.suptitle("Concept D: OLS Ridge vs Huber — residual comparison", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir/"plot_robust_regression.png", dpi=160, bbox_inches="tight")
    plt.close()

    write_json(output_dir/"robust_regression.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept E: Log target transformation
# ─────────────────────────────────────────────────────────────────────────────
def analyse_log_transform(X_train: pd.DataFrame, y_train: pd.Series,
                           output_dir: Path) -> dict[str, Any]:
    """
    Concept E: Compare raw SalePrice vs log1p(SalePrice) as target.
    Right-skewed targets cause heteroscedastic residuals — errors are larger
    for expensive houses. log1p normalises the distribution, homogenises residual
    variance, and often improves RMSE on expensive houses.
    """
    log.info("Concept E: Log target transformation …")
    num_cols = [c for c in NUMERIC_COLS if c in X_train.columns]
    X_num = X_train[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_sc  = StandardScaler().fit_transform(X_num)
    y_raw = y_train.to_numpy()
    y_log = np.log1p(y_raw)
    cv    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    r2_raw = cross_val_score(Ridge(alpha=1.0), X_sc, y_raw, cv=cv, scoring="r2")
    r2_log = cross_val_score(Ridge(alpha=1.0), X_sc, y_log, cv=cv, scoring="r2")

    # Back-transform log predictions to compare RMSE in USD
    oof_log = cross_val_predict(Ridge(alpha=1.0), X_sc, y_log, cv=cv)
    oof_raw = cross_val_predict(Ridge(alpha=1.0), X_sc, y_raw, cv=cv)
    rmse_log_usd = float(np.sqrt(mean_squared_error(y_raw, np.expm1(oof_log))))
    rmse_raw_usd = float(np.sqrt(mean_squared_error(y_raw, oof_raw)))

    log.info("  Raw target: R²=%.4f  RMSE_USD=$%s",
             r2_raw.mean(), f"{rmse_raw_usd:,.0f}")
    log.info("  Log target: R²=%.4f  RMSE_USD=$%s (back-transformed)",
             r2_log.mean(), f"{rmse_log_usd:,.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(y_raw, bins=50, color="#4C78A8", edgecolor="white", linewidth=0.3)
    axes[0].set_title("SalePrice (raw) — right-skewed"); axes[0].set_xlabel("SalePrice ($)")
    axes[1].hist(y_log, bins=50, color="#54A24B", edgecolor="white", linewidth=0.3)
    axes[1].set_title("log1p(SalePrice) — near-normal"); axes[1].set_xlabel("log1p(SalePrice)")
    plt.suptitle("Concept E: Log target transformation", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir/"plot_log_transform.png", dpi=160, bbox_inches="tight"); plt.close()

    return {
        "raw_r2": float(r2_raw.mean()), "log_r2": float(r2_log.mean()),
        "raw_rmse_usd": rmse_raw_usd, "log_rmse_usd": rmse_log_usd,
        "improvement_pct": float((rmse_raw_usd - rmse_log_usd) / rmse_raw_usd * 100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Concept F: Ordinal encoding for quality features
# ─────────────────────────────────────────────────────────────────────────────
def analyse_ordinal_encoding(X_train: pd.DataFrame, y_train: pd.Series,
                              output_dir: Path) -> dict[str, Any]:
    """
    Concept F: Compare ordinal (domain-knowledge) vs OHE for quality features.
    Quality features are inherently ordered: Ex > Gd > TA > Fa > Po.
    OHE treats them as unordered categories (5 binary columns per feature).
    Ordinal encoding preserves the order with 1 column and is often better.
    """
    log.info("Concept F: Ordinal encoding comparison …")
    qual_cols_present = [c for c in _QUAL_COLS if c in X_train.columns]
    if not qual_cols_present:
        return {}

    y_log = np.log1p(y_train.to_numpy())
    cv    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def get_ordinal_arr(X_tr):
        rows = []
        for col in qual_cols_present:
            col_arr = X_tr[col].astype(str).map(
                lambda v: _QUAL_MAP.get(v, _QUAL_MAP.get("nan", 0))).fillna(0)
            rows.append(col_arr.to_numpy())
        return np.column_stack(rows)

    def get_ohe_arr(X_tr):
        cat_df = X_tr[qual_cols_present].astype(str).fillna("None")
        enc    = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return enc.fit_transform(cat_df)

    results = {}
    for name, fn in [("Ordinal (domain)", get_ordinal_arr), ("OHE (naive)", get_ohe_arr)]:
        X_enc = fn(X_train)
        r2s   = cross_val_score(Ridge(alpha=1.0), X_enc, y_log, cv=cv, scoring="r2")
        results[name] = {"n_features": X_enc.shape[1],
                         "r2_mean": float(r2s.mean()),
                         "r2_std":  float(r2s.std())}
        log.info("  %-25s  n_feat=%3d  R²=%.4f", name, X_enc.shape[1], r2s.mean())

    # Bar comparison
    names = list(results.keys())
    r2s   = [results[n]["r2_mean"] for n in names]
    plt.figure(figsize=(6, 4))
    plt.bar(names, r2s, color=["#4C78A8", "#E45756"])
    plt.ylabel("CV R²"); plt.title("Concept F: Ordinal vs OHE for quality features\nOrdinal respects Ex>Gd>TA>Fa>Po ordering")
    for i, (bar, val) in enumerate(zip(plt.gca().patches, r2s)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", fontsize=10)
    plt.tight_layout(); plt.savefig(output_dir/"plot_ordinal_encoding.png", dpi=160); plt.close()
    write_json(output_dir/"ordinal_encoding.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept G: Polynomial + interaction analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_polynomial_interactions(X_train: pd.DataFrame, y_train: pd.Series,
                                     output_dir: Path) -> dict[str, Any]:
    """
    Concept G: Domain-knowledge polynomial and interaction terms.
    GrLivArea² — convex price-area curve (diminishing marginal return)
    OverallQual × GrLivArea — quality amplifies area value
    TotalSF — aggregate of all floor spaces (engineered total)
    YearBuilt × OverallQual — newer + quality = premium
    Age at sale — YrSold - YearBuilt (depreciation proxy)
    """
    log.info("Concept G: Polynomial + interaction analysis …")
    y_log = np.log1p(y_train.to_numpy())
    cv    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def safe_col(X, col, fill=0.0):
        if col in X.columns:
            return pd.to_numeric(X[col], errors="coerce").fillna(fill).to_numpy()
        return np.full(len(X), fill)

    base_cols    = [c for c in ["GrLivArea","OverallQual","TotalBsmtSF","YearBuilt","GarageArea"]
                    if c in X_train.columns]
    X_base       = StandardScaler().fit_transform(
        X_train[base_cols].apply(pd.to_numeric, errors="coerce").fillna(0))

    # Build interaction features
    gr   = safe_col(X_train, "GrLivArea")
    qual = safe_col(X_train, "OverallQual")
    bsmt = safe_col(X_train, "TotalBsmtSF")
    yr   = safe_col(X_train, "YearBuilt")
    yr_s = safe_col(X_train, "YrSold", 2010)
    f1   = safe_col(X_train, "1stFlrSF")
    f2   = safe_col(X_train, "2ndFlrSF")

    interact = np.column_stack([
        gr ** 2,                   # GrLivArea²
        qual * gr,                 # OverallQual × GrLivArea
        f1 + f2 + bsmt,            # TotalSF
        yr * qual,                 # YearBuilt × OverallQual
        yr_s - yr,                 # Age at sale
        (gr > 2000).astype(float), # Large house flag
        (qual >= 8).astype(float), # High quality flag
    ])
    X_with_interact = np.hstack([X_base, StandardScaler().fit_transform(interact)])

    r2_base   = cross_val_score(Ridge(alpha=1.0), X_base,           y_log, cv=cv, scoring="r2")
    r2_inter  = cross_val_score(Ridge(alpha=1.0), X_with_interact,  y_log, cv=cv, scoring="r2")

    results = {
        "baseline_r2":    float(r2_base.mean()),
        "with_interact_r2": float(r2_inter.mean()),
        "uplift":         float(r2_inter.mean() - r2_base.mean()),
    }
    log.info("  Baseline R²=%.4f → With interactions R²=%.4f (uplift=%.4f)",
             r2_base.mean(), r2_inter.mean(), results["uplift"])

    # Scatter: OverallQual × GrLivArea vs log(SalePrice)
    plt.figure(figsize=(7, 5))
    interact_score = qual * gr / 1000
    plt.scatter(interact_score, y_log, alpha=0.2, s=6, color="#4C78A8")
    from scipy.stats import pearsonr
    r, _ = pearsonr(interact_score, y_log)
    plt.xlabel("OverallQual × GrLivArea / 1000"); plt.ylabel("log(SalePrice)")
    plt.title(f"Concept G: OverallQual × GrLivArea interaction\nPearson r={r:.3f} — key non-linear driver")
    plt.tight_layout(); plt.savefig(output_dir/"plot_interaction_analysis.png", dpi=160); plt.close()
    write_json(output_dir/"interaction_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept I: Regularisation paths on wide feature set
# ─────────────────────────────────────────────────────────────────────────────
def analyse_regularisation_wide(X_train: pd.DataFrame, y_train: pd.Series,
                                  output_dir: Path) -> dict[str, Any]:
    """
    Concept I: Ridge / Lasso / ElasticNet on full OHE-expanded feature set.
    The Ames dataset with OHE has 200+ features. Lasso's sparsity is critical —
    it zeros out redundant OHE columns while Ridge shrinks all coefficients.
    """
    log.info("Concept I: Regularisation on wide feature set …")
    # Quick numeric-only version for speed
    num_cols = [c for c in NUMERIC_COLS if c in X_train.columns]
    X_num = X_train[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_sc  = StandardScaler().fit_transform(X_num)
    y_log = np.log1p(y_train.to_numpy())
    cv    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    alphas = np.logspace(-3, 4, 50)

    ridge_cv = RidgeCV(alphas=alphas, cv=cv).fit(X_sc, y_log)
    lasso_cv = LassoCV(alphas=alphas, cv=cv, max_iter=5000,
                       random_state=RANDOM_STATE).fit(X_sc, y_log)
    enet_cv  = ElasticNetCV(alphas=alphas, l1_ratio=[0.1,0.3,0.5,0.7,0.9],
                            cv=cv, max_iter=5000,
                            random_state=RANDOM_STATE).fit(X_sc, y_log)
    lasso_coef = lasso_cv.coef_
    n_zeroed   = int((lasso_coef == 0).sum())

    # Regularisation path plot
    ridge_coefs = np.array([Ridge(alpha=a).fit(X_sc,y_log).coef_ for a in alphas])
    lasso_coefs = []
    for a in alphas:
        try:
            lasso_coefs.append(Lasso(alpha=a, max_iter=5000).fit(X_sc,y_log).coef_)
        except Exception:
            lasso_coefs.append(np.zeros(X_sc.shape[1]))
    lasso_coefs = np.array(lasso_coefs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i in range(min(10, ridge_coefs.shape[1])):
        axes[0].plot(np.log10(alphas), ridge_coefs[:, i], linewidth=1.0, alpha=0.7)
    axes[0].axvline(np.log10(ridge_cv.alpha_), color="green", linestyle="--",
                    label=f"α*={ridge_cv.alpha_:.3f}")
    axes[0].set_xlabel("log₁₀(α)"); axes[0].set_ylabel("Coefficient")
    axes[0].set_title("Ridge path — coefficients shrink but stay nonzero"); axes[0].legend()

    for i in range(min(10, lasso_coefs.shape[1])):
        axes[1].plot(np.log10(alphas), lasso_coefs[:, i], linewidth=1.0, alpha=0.7)
    axes[1].axvline(np.log10(lasso_cv.alpha_), color="green", linestyle="--",
                    label=f"α*={lasso_cv.alpha_:.4f}")
    axes[1].set_xlabel("log₁₀(α)"); axes[1].set_ylabel("Coefficient")
    axes[1].set_title(f"Lasso path — {n_zeroed}/{X_sc.shape[1]} features zeroed")
    axes[1].legend()
    plt.suptitle("Concept I: Regularisation paths on Ames Housing features", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir/"plot_regularisation_paths.png", dpi=160, bbox_inches="tight"); plt.close()

    log.info("Ridge α*=%.4f  Lasso α*=%.4f  ElasticNet α*=%.4f  Lasso zeroed: %d/%d",
             ridge_cv.alpha_, lasso_cv.alpha_, enet_cv.alpha_,
             n_zeroed, X_sc.shape[1])
    return {
        "ridge_optimal_alpha":  float(ridge_cv.alpha_),
        "lasso_optimal_alpha":  float(lasso_cv.alpha_),
        "elasticnet_alpha":     float(enet_cv.alpha_),
        "elasticnet_l1_ratio":  float(enet_cv.l1_ratio_),
        "lasso_n_zeroed":       n_zeroed,
        "lasso_n_total":        int(X_sc.shape[1]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — InsuranceFeatureEngineer pattern for Ames
# ─────────────────────────────────────────────────────────────────────────────
class AmesFeatEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-driven feature engineering for Ames Housing.
    Handles: MNAR missing → fill absent, spatial LotFrontage, ordinal quality,
    polynomial + interaction terms (Concept G).
    Fit-safe: learns spatial medians from training data only.
    """
    def __init__(self, n_neighbors_knn: int = 5):
        self.n_neighbors_knn = n_neighbors_knn

    def fit(self, X: pd.DataFrame, y=None) -> "AmesFeatEngineer":
        self._spatial_imp = SpatialImputer("LotFrontage", "Neighborhood")
        self._spatial_imp.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # ── Concept A: Fill MNAR "absent" columns ─────────────────────────────
        for col in ABSENT_COLS:
            if col in X.columns:
                X[col] = X[col].fillna("None")

        # GarageYrBlt: missing = no garage → fill with YearBuilt
        if "GarageYrBlt" in X.columns and "YearBuilt" in X.columns:
            mask = X["GarageYrBlt"].isna()
            X.loc[mask, "GarageYrBlt"] = pd.to_numeric(
                X.loc[mask, "YearBuilt"], errors="coerce")

        # ── Concept H: Spatial imputation for LotFrontage ─────────────────────
        X = self._spatial_imp.transform(X)

        # ── Concept F: Ordinal encoding for quality features ──────────────────
        for col in _QUAL_COLS:
            if col in X.columns:
                X[f"{col}_ord"] = X[col].astype(str).map(
                    lambda v: _QUAL_MAP.get(v, 0)).fillna(0).astype(float)
        if "Fence" in X.columns:
            X["Fence_ord"] = X["Fence"].astype(str).map(
                lambda v: _FENCE_MAP.get(v, 0)).fillna(0).astype(float)

        # ── Concept G: Polynomial + interaction terms ─────────────────────────
        def _flt(col, fill=0.0):
            if col not in X.columns: return pd.Series(fill, index=X.index)
            return pd.to_numeric(X[col], errors="coerce").fillna(fill)

        gr   = _flt("GrLivArea"); qual = _flt("OverallQual")
        bsmt = _flt("TotalBsmtSF"); yr = _flt("YearBuilt")
        yr_s = _flt("YrSold", 2010)
        f1   = _flt("1stFlrSF"); f2 = _flt("2ndFlrSF")
        gar  = _flt("GarageArea"); open_porch = _flt("OpenPorchSF")
        deck = _flt("WoodDeckSF")

        X["GrLivArea_sq"]        = gr ** 2
        X["TotalSF"]             = f1 + f2 + bsmt
        X["Qual_x_Area"]         = qual * gr
        X["Qual_x_Year"]         = qual * yr
        X["AgeAtSale"]           = yr_s - yr
        X["TotalPorchSF"]        = open_porch + deck
        X["TotalBaths"]          = (_flt("FullBath") + 0.5*_flt("HalfBath") +
                                    _flt("BsmtFullBath") + 0.5*_flt("BsmtHalfBath"))
        X["HasGarage"]           = (gar > 0).astype(float)
        X["HasPool"]             = (_flt("PoolArea") > 0).astype(float)
        X["Has2ndFloor"]         = (f2 > 0).astype(float)
        X["IsNewHouse"]          = (yr >= 2000).astype(float)
        X["HighQuality"]         = (qual >= 8).astype(float)

        return X


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor
# ─────────────────────────────────────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    """
    Full ColumnTransformer for Ames Housing after AmesFeatEngineer:
      numeric features → KNN impute → RobustScaler
      nominal categorical → mode impute → OHE
      ordinal quality → already encoded in FE → passthrough via numeric
    """
    numeric_features = (NUMERIC_COLS +
                        get_ordinal_encoded_cols() +
                        ["GrLivArea_sq","TotalSF","Qual_x_Area","Qual_x_Year",
                         "AgeAtSale","TotalPorchSF","TotalBaths",
                         "HasGarage","HasPool","Has2ndFloor","IsNewHouse","HighQuality"])

    nominal_features = NOMINAL_COLS

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler",  RobustScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                  min_frequency=5)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, nominal_features),
    ], remainder="drop", verbose_feature_names_out=False)


def build_pipeline(model=None) -> Pipeline:
    if model is None:
        model = Ridge(alpha=100.0)  # higher alpha for large feature count
    sel = SelectFromModel(
        ExtraTreesRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        threshold="median",
    )
    return Pipeline([
        ("feature_engineering", AmesFeatEngineer()),
        ("preprocess",          build_preprocessor()),
        ("feature_selection",   sel),
        ("model",               model),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# EDA (train-set only)
# ─────────────────────────────────────────────────────────────────────────────
def save_research_artifacts(X_train: pd.DataFrame, y_train: pd.Series,
                             output_dir: Path) -> dict[str, Any]:
    log.info("Saving EDA artifacts (train-set only) …")
    output_dir.mkdir(parents=True, exist_ok=True)
    eda = X_train.copy(); eda[TARGET] = y_train.values
    missingness_report(eda).to_csv(output_dir/"research_missingness_report.csv")

    # Numeric correlation with target
    num_df = eda.select_dtypes(include=[np.number])
    corr   = num_df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    corr.head(20).to_csv(output_dir/"correlation_with_target.csv")

    # Grouped stats
    grouped = {}
    for col in ["Neighborhood", "BldgType", "HouseStyle"]:
        if col in eda.columns:
            grouped[f"price_by_{col}"] = (
                eda.groupby(col)[TARGET].agg(["mean","median","std"]).to_dict())
    grouped["target_stats"] = y_train.describe().to_dict()
    write_json(output_dir/"research_decisions.json", {
        "problem_type": "regression",
        "target": TARGET, "target_unit": "USD",
        "log_transform": "log1p(SalePrice) — see Concept E",
        "missing_strategy": "MNAR→None/0, LotFrontage→neighborhood median, rest→SimpleImputer",
        "outlier_strategy": "Remove GrLivArea>4000 with SalePrice<200k (De Cock 2011)",
        "grouped_stats": grouped,
    })

    sns.set_theme(style="whitegrid")
    # Target
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.hist(y_train, bins=50, color="#4C78A8", edgecolor="white", linewidth=0.3)
    a1.set_title("SalePrice (raw — right-skewed)"); a1.set_xlabel("SalePrice ($)")
    a2.hist(np.log1p(y_train), bins=50, color="#54A24B", edgecolor="white", linewidth=0.3)
    a2.set_title("log1p(SalePrice) — near-normal"); a2.set_xlabel("log1p(SalePrice)")
    plt.suptitle("Concept E: Log target transformation (train)", fontsize=11, y=1.02)
    plt.tight_layout(); plt.savefig(output_dir/"plot_target_distribution.png", dpi=160, bbox_inches="tight"); plt.close()

    # Correlation top-20
    plt.figure(figsize=(9, 5))
    corr.head(20).plot(kind="barh", color=["#54A24B" if v > 0 else "#E45756" for v in corr.head(20)])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Top-20 feature correlations with SalePrice (train)")
    plt.tight_layout(); plt.savefig(output_dir/"plot_correlation.png", dpi=160); plt.close()

    # OverallQual vs SalePrice
    if "OverallQual" in eda.columns:
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=eda, x="OverallQual", y=TARGET, palette="RdYlGn")
        plt.title("SalePrice by OverallQual — strong monotone relationship")
        plt.tight_layout(); plt.savefig(output_dir/"plot_qual_vs_price.png", dpi=160); plt.close()

    # Neighborhood median price
    if "Neighborhood" in eda.columns:
        nb_med = eda.groupby("Neighborhood")[TARGET].median().sort_values(ascending=False)
        plt.figure(figsize=(11, 5))
        nb_med.plot(kind="bar", color="#4C78A8")
        plt.title("Median SalePrice by Neighborhood (train)")
        plt.xlabel(""); plt.ylabel("Median SalePrice ($)")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout(); plt.savefig(output_dir/"plot_neighborhood_price.png", dpi=160); plt.close()

    log.info("EDA saved.")
    return {"grouped_stats": grouped, "top_correlations": corr.head(10).to_dict()}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics + baselines
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_predictions(y_true, y_pred, log_target=True):
    """Evaluate on original USD scale (back-transform if log_target)."""
    if hasattr(y_true, "to_numpy"): y_true = y_true.to_numpy()
    if log_target:
        y_pred_usd = np.expm1(y_pred)
        y_true_usd = np.expm1(y_true) if y_true.max() < 20 else y_true
    else:
        y_pred_usd = y_pred; y_true_usd = y_true
    res = y_true_usd - y_pred_usd
    return {
        "mae":              float(mean_absolute_error(y_true_usd, y_pred_usd)),
        "rmse":             float(np.sqrt(mean_squared_error(y_true_usd, y_pred_usd))),
        "r2":               float(r2_score(y_true_usd, y_pred_usd)),
        "mape":             float(mean_absolute_percentage_error(y_true_usd, y_pred_usd)),
        "medae":            float(median_absolute_error(y_true_usd, y_pred_usd)),
        "rmsle":            float(np.sqrt(mean_squared_error(np.log1p(y_true_usd.clip(0)),
                                                             np.log1p(y_pred_usd.clip(0))))),
        "residual_mean":    float(res.mean()),
        "residual_std":     float(res.std()),
    }


def evaluate_baselines(X_tr, X_te, y_tr, y_te):
    return {s: evaluate_predictions(y_te, DummyRegressor(strategy=s).fit(X_tr,y_tr).predict(X_te),
                                    log_target=False)
            for s in ["mean", "median"]}


# ─────────────────────────────────────────────────────────────────────────────
# Concept J: Residual diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def residual_diagnostics(y_true, y_pred_log, output_dir, log_target=True):
    log.info("Concept J: Residual diagnostics …")
    if log_target:
        residuals = y_true - y_pred_log
    else:
        residuals = y_true - y_pred_log

    n       = len(residuals)
    resid_sq= residuals ** 2
    bp_r2   = LinearRegression().fit(y_pred_log.reshape(-1,1),resid_sq).score(
                y_pred_log.reshape(-1,1), resid_sq)
    bp_stat = float(n * bp_r2)
    bp_p    = float(1 - scipy_stats.chi2.cdf(bp_stat, df=1))
    sample  = residuals if n <= 5000 else np.random.default_rng(RANDOM_STATE).choice(
                residuals, 5000, replace=False)
    sw_s, sw_p = scipy_stats.shapiro(sample)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    scipy_stats.probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q plot (normality)")
    axes[1].scatter(y_pred_log, np.sqrt(np.abs(residuals)), alpha=0.2, s=6, color="#4C78A8")
    axes[1].axhline(np.sqrt(np.abs(residuals)).mean(), color="red", linewidth=1.2)
    axes[1].set_xlabel("Fitted log(SalePrice)"); axes[1].set_ylabel("√|residual|")
    axes[1].set_title("Scale-location (heteroscedasticity)")
    axes[2].hist(residuals, bins=60, color="#54A24B", edgecolor="white", linewidth=0.3)
    axes[2].axvline(0, color="red", linewidth=1.2)
    axes[2].set_title("Residual distribution")
    plt.suptitle("Concept J: Residual diagnostics", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir/"plot_residual_diagnostics.png", dpi=160, bbox_inches="tight"); plt.close()

    log.info("BP p=%.4f (hetero: %s)  SW p=%.4f (normal: %s)",
             bp_p, bp_p<0.05, sw_p, sw_p>0.05)
    return {
        "breusch_pagan": {"statistic": bp_stat, "p_value": bp_p, "heteroscedastic": bp_p<0.05},
        "shapiro_wilk":  {"statistic": float(sw_s), "p_value": float(sw_p), "normal": sw_p>0.05},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Concept K: Learning curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_curves_housing(model, X_train, y_train_log, output_dir):
    log.info("Concept K: Learning curves …")
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    try:
        sizes,tr_s,cv_s = learning_curve(
            model, X_train, y_train_log,
            train_sizes=np.linspace(0.10, 1.0, 8),
            cv=cv, scoring="r2", n_jobs=N_JOBS)
        plt.figure(figsize=(8, 4.5))
        plt.plot(sizes, tr_s.mean(axis=1), "o-", color="#4C78A8", label="Train R²")
        plt.plot(sizes, cv_s.mean(axis=1), "o-", color="#E45756", label="CV R²")
        plt.fill_between(sizes, tr_s.mean(axis=1)-tr_s.std(axis=1),
                         tr_s.mean(axis=1)+tr_s.std(axis=1), alpha=0.12, color="#4C78A8")
        plt.fill_between(sizes, cv_s.mean(axis=1)-cv_s.std(axis=1),
                         cv_s.mean(axis=1)+cv_s.std(axis=1), alpha=0.12, color="#E45756")
        gap = float(cv_s[-1].mean() - tr_s[-1].mean())
        plt.title(f"Concept K: Learning curve (n=2930)\nTrain-CV gap={gap:.4f}")
        plt.xlabel("Training set size"); plt.ylabel("R²")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir/"plot_learning_curve.png", dpi=160); plt.close()
    except Exception as exc:
        log.warning("Learning curve failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Concept L: Subgroup disparity
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_subgroups(model, X_test, y_test, y_pred_log, output_dir):
    log.info("Concept L: Subgroup disparity …")
    y_pred_usd = np.expm1(y_pred_log)
    y_true_usd = y_test.to_numpy()
    overall_rmse = float(np.sqrt(mean_squared_error(y_true_usd, y_pred_usd)))
    overall_r2   = float(r2_score(y_true_usd, y_pred_usd))

    eval_df = X_test.reset_index(drop=True).copy()
    eval_df["_y_true"] = y_true_usd; eval_df["_y_pred"] = y_pred_usd
    rows = []
    for col in ["Neighborhood", "BldgType", "HouseStyle"]:
        if col not in eval_df.columns: continue
        for val, sub in eval_df.groupby(col, observed=True):
            if len(sub) < 8: continue
            sr = float(np.sqrt(mean_squared_error(sub["_y_true"], sub["_y_pred"])))
            rows.append({
                "group_col": col, "group_val": str(val), "n": int(len(sub)),
                "mean_actual": round(float(sub["_y_true"].mean()), 0),
                "rmse": round(sr, 0),
                "r2":   round(float(r2_score(sub["_y_true"], sub["_y_pred"])), 4),
                "rmse_gap": round(sr - overall_rmse, 0),
                "alert": bool(sr > overall_rmse * 1.25),
            })
    if rows:
        rd = pd.DataFrame(rows); rd.to_csv(output_dir/"fairness_report.csv", index=False)
        g  = rd.copy(); g["label"] = g["group_col"] + "=" + g["group_val"].astype(str)
        plt.figure(figsize=(12, max(5, len(g)*0.38)))
        colors = ["#E45756" if a else "#4C78A8" for a in g["alert"]]
        plt.barh(g["label"], g["rmse"], color=colors)
        plt.axvline(overall_rmse, linestyle="--", color="black",
                    label=f"Overall RMSE=${overall_rmse:,.0f}")
        plt.xlabel("RMSE ($)"); plt.title("Concept L: Subgroup RMSE by Neighborhood/BldgType/HouseStyle")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir/"plot_subgroup_rmse.png", dpi=160); plt.close()
    return {"overall_rmse": overall_rmse, "overall_r2": overall_r2, "subgroups": rows}


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────
def save_feature_importance(model, output_dir):
    step_names = list(model.named_steps.keys())
    clf        = model.named_steps["model"]
    if "feature_selection" in step_names:
        sel    = model.named_steps["feature_selection"]
        fs_idx = step_names.index("feature_selection")
        fn = None
        for sname in reversed(step_names[:fs_idx]):
            s = model.named_steps[sname]
            if hasattr(s, "get_feature_names_out"):
                try: fn = s.get_feature_names_out(); break
                except: pass
        if fn is None: return
        support = sel.get_support()
        if len(fn) != len(support):
            fn = np.array([f"feature_{i}" for i in range(len(support))])
        sn = fn[support]
    else:
        return

    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_; signed = np.full(len(imp), np.nan)
    elif hasattr(clf, "coef_"):
        signed = clf.coef_; imp = np.abs(signed)
    else: return

    pd.DataFrame({"feature": sn, "importance": imp, "coefficient": signed}
        ).sort_values("importance", ascending=False
        ).to_csv(output_dir/"feature_importance.csv", index=False)

    plt.figure(figsize=(9, 5.5))
    sns.barplot(
        data=pd.DataFrame({"feature": sn, "importance": imp}
            ).sort_values("importance", ascending=False).head(25),
        y="feature", x="importance", color="#4C78A8")
    plt.title("Top 25 model features"); plt.tight_layout()
    plt.savefig(output_dir/"plot_feature_importance.png", dpi=160); plt.close()


def save_evaluation_plots(y_test, y_pred_log, output_dir):
    y_pred_usd = np.expm1(y_pred_log)
    y_true_usd = y_test.to_numpy()
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true_usd, y_pred_usd, alpha=0.3, s=8, color="#4C78A8")
    mn, mx = min(y_true_usd.min(), y_pred_usd.min()), max(y_true_usd.max(), y_pred_usd.max())
    plt.plot([mn,mx],[mn,mx],"r--",linewidth=1.2,label="Perfect")
    plt.title(f"Actual vs Predicted  (R²={r2_score(y_true_usd,y_pred_usd):.3f})")
    plt.xlabel("Actual SalePrice ($)"); plt.ylabel("Predicted SalePrice ($)")
    plt.legend(); plt.tight_layout()
    plt.savefig(output_dir/"plot_actual_vs_predicted.png", dpi=160); plt.close()


def save_error_analysis(X_test, y_test, y_pred_log, output_dir):
    y_pred_usd = np.expm1(y_pred_log); y_true_usd = y_test.to_numpy()
    res = y_true_usd - y_pred_usd
    rmse = float(np.sqrt(np.mean(res**2)))
    df = X_test.copy()
    df["actual"] = y_true_usd; df["predicted"] = y_pred_usd
    df["residual"] = res; df["abs_error"] = np.abs(res)
    df["pct_error"] = np.abs(res) / np.maximum(np.abs(y_true_usd), 1)
    df["severity"] = pd.cut(df["abs_error"],
                            bins=[0, rmse*0.5, rmse, rmse*2, np.inf],
                            labels=["low","medium","high","severe"])
    df.to_csv(output_dir/"test_predictions.csv", index=False)
    df[df["abs_error"] > rmse].to_csv(output_dir/"error_analysis.csv", index=False)


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
        fn_arr = None
        if sel:
            fs_idx = step_names.index("feature_selection")
            for sname in reversed(step_names[:fs_idx]):
                s = model.named_steps[sname]
                if hasattr(s, "get_feature_names_out"):
                    try: fn_arr = s.get_feature_names_out(); break
                    except: pass
            support = sel.get_support()
            if fn_arr is None: fn_arr = np.array([f"f{i}" for i in range(len(support))])
            if len(fn_arr) != len(support):
                fn_arr = np.array([f"feature_{i}" for i in range(len(support))])
            sn = fn_arr[support]; Xt = sel.transform(Xt)
        else:
            try: sn = prep.get_feature_names_out()
            except: sn = np.array([f"f{i}" for i in range(Xt.shape[1])])

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
            plt.title(f"SHAP — champion ({ptype})")
            plt.tight_layout(); plt.savefig(output_dir/fname, dpi=150, bbox_inches="tight"); plt.close()

        worst = int(np.argmax(np.abs(y_test.to_numpy() - np.expm1(y_pred_log))))
        ev = (float(exp.expected_value) if not isinstance(exp.expected_value, np.ndarray)
              else float(exp.expected_value))
        shap.waterfall_plot(
            shap.Explanation(values=sv[worst], base_values=ev,
                             data=Xdf.iloc[worst].values, feature_names=list(sn)),
            show=False, max_display=15)
        plt.title("SHAP Waterfall — worst prediction residual")
        plt.tight_layout()
        plt.savefig(output_dir/"plot_shap_waterfall.png", dpi=150, bbox_inches="tight"); plt.close()
        pd.DataFrame({"feature": sn, "mean_abs_shap": np.abs(sv).mean(axis=0)}
            ).sort_values("mean_abs_shap", ascending=False
            ).to_csv(output_dir/"shap_importance.csv", index=False)
        log.info("SHAP saved.")
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Training profile (100 quantiles — mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
def build_training_profile(X_train, y_train):
    num_cols = [c for c in NUMERIC_COLS if c in X_train.columns]
    stats = {}
    for col in num_cols:
        v = pd.to_numeric(X_train[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(v) == 0: continue
        stats[col] = {
            "mean": float(v.mean()), "std": float(v.std()),
            "min":  float(v.min()),  "max": float(v.max()),
            "quantiles": np.quantile(v, np.linspace(0, 1, 100)).tolist(),
        }
    cat_stats = {}
    for col in NOMINAL_COLS:
        if col in X_train.columns:
            cat_stats[col] = X_train[col].astype(str).value_counts().head(20).to_dict()
    return to_jsonable({
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "row_count":     int(len(X_train)),
        "raw_columns":   list(X_train.columns),
        "target_stats":  {"mean": float(y_train.mean()), "std": float(y_train.std()),
                          "min": float(y_train.min()), "max": float(y_train.max())},
        "raw_missing_rate": X_train.isna().mean().to_dict(),
        "numeric_train_stats": stats,
        "categorical_stats":   cat_stats,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search — mirrors reference tune_model
# ─────────────────────────────────────────────────────────────────────────────
def tune_model(X_train, y_train_log, n_iter=20, n_cv_splits=5, fast=False):
    log.info("Hyperparameter search: n_iter=%d  cv=%d-fold  fast=%s",
             n_iter, n_cv_splits, fast)
    _n = 50 if fast else 150

    param_distributions = [
        # Ridge
        {"feature_selection__threshold": ["median","0.75*median","1.25*median"],
         "model": [Ridge()],
         "model__alpha": [0.1, 1, 10, 50, 100, 500, 1000, 5000]},
        # Lasso
        {"feature_selection__threshold": ["median","0.75*median"],
         "model": [Lasso(max_iter=5000)],
         "model__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]},
        # ElasticNet
        {"feature_selection__threshold": ["median","0.75*median"],
         "model": [ElasticNet(max_iter=5000)],
         "model__alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
         "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        # HuberRegressor (Concept D)
        {"feature_selection__threshold": ["median","0.75*median"],
         "model": [HuberRegressor(max_iter=500)],
         "model__epsilon": [1.1, 1.35, 1.5, 2.0],
         "model__alpha": [0.001, 0.01, 0.1]},
        # GBR
        {"feature_selection__threshold": ["median","0.75*median","1.25*median"],
         "model": [GradientBoostingRegressor(n_estimators=_n, random_state=RANDOM_STATE)],
         "model__max_depth": [3, 4, 5],
         "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
         "model__subsample": [0.7, 0.9]},
        # RandomForest
        {"feature_selection__threshold": ["median","0.75*median","1.25*median"],
         "model": [RandomForestRegressor(n_estimators=_n, random_state=RANDOM_STATE,
                                         n_jobs=N_JOBS)],
         "model__max_depth": [8, 12, None],
         "model__min_samples_leaf": [1, 2, 4]},
    ]

    cv     = KFold(n_splits=n_cv_splits, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        build_pipeline(), param_distributions, n_iter=n_iter,
        scoring={"r2":"r2","neg_rmse":"neg_root_mean_squared_error",
                 "neg_mae":"neg_mean_absolute_error"},
        refit="r2", cv=cv, random_state=RANDOM_STATE,
        n_jobs=N_JOBS, verbose=1, return_train_score=True)
    search.fit(X_train, y_train_log)

    best_mdl = search.best_estimator_.named_steps["model"]
    if hasattr(best_mdl, "n_estimators") and best_mdl.n_estimators == _n:
        log.info("Upgrading %d → 300 trees …", _n)
        best_mdl.set_params(n_estimators=300)
        search.best_estimator_.fit(X_train, y_train_log)

    log.info("Best CV R²=%.4f  model=%s", search.best_score_,
             type(best_mdl).__name__)
    return search


# ─────────────────────────────────────────────────────────────────────────────
# Model Card + MLflow + versioning — mirrors reference exactly
# ─────────────────────────────────────────────────────────────────────────────
def save_model_card(metrics, fairness, uncertainty, reg_analysis, search, output_dir):
    tm = metrics.get("test_metrics", {})
    write_json(output_dir/MODEL_CARD_FILE, {
        "schema_version": "1.0",
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "model_details":  {
            "name":      "Ames Housing Price Predictor",
            "type":      "Regression (sklearn Pipeline + log(SalePrice) target)",
            "algorithm": repr(search.best_estimator_.named_steps["model"]),
        },
        "intended_use":  {
            "primary_use": "Predict residential home sale prices in Ames, Iowa.",
            "out_of_scope": ["Real-time bidding","Other cities","Post-2010 market conditions"],
        },
        "evaluation_results": {
            "test_r2": tm.get("r2"), "test_rmse_usd": tm.get("rmse"),
            "test_mae_usd": tm.get("mae"), "rmsle": tm.get("rmsle"),
        },
        "missing_data_strategy":  "MNAR→None/0, LotFrontage→Neighborhood median, rest→SimpleImputer",
        "outlier_strategy":       "Remove GrLivArea>4000 & SalePrice<200k (5 houses)",
        "log_transform":          "log1p(SalePrice) — improves RMSE on high-value houses",
        "regularisation":         reg_analysis,
        "fairness":               {"overall_rmse": fairness.get("overall_rmse"),
                                   "alerts": [r for r in fairness.get("subgroups",[]) if r.get("alert")]},
        "limitations": [
            "Data from 2006–2010 Ames Iowa — not generalisable to other cities/eras.",
            "5 influential outliers removed — model may underperform on luxury market.",
            "SaleCondition=Normal assumed — unusual sales may have higher error.",
        ],
        "ethical_considerations": [
            "Neighborhood as a feature may encode historical redlining patterns.",
            "Do not use for automated lending decisions without regulatory review.",
        ],
        "hyperparameters": search.best_params_,
        "cv_best_r2":      float(search.best_score_),
    })


def log_to_mlflow(metrics, search, model, output_dir):
    if not _MLFLOW: return
    try:
        mlflow.set_experiment("ames_housing")
        tm = metrics.get("test_metrics", {})
        with mlflow.start_run():
            mlflow.log_params({f"best_{k}": str(v) for k, v in search.best_params_.items()})
            mlflow.log_metrics({"cv_r2": float(search.best_score_),
                                "test_r2": float(tm.get("r2", 0)),
                                "test_rmse_usd": float(tm.get("rmse", 0))})
            for f in [MODEL_CARD_FILE, METRICS_FILE, "plot_actual_vs_predicted.png",
                      "plot_shap_bar.png", "missing_taxonomy.json"]:
                if (output_dir/f).exists(): mlflow.log_artifact(str(output_dir/f))
            mlflow.sklearn.log_model(model, "model")
        log.info("MLflow logged.")
    except Exception as e:
        log.warning("MLflow failed: %s", e)


def save_environment_snapshot(output_dir):
    env = {"saved_at": datetime.now(timezone.utc).isoformat(),
           "python": sys.version, "platform": sys.platform, "libraries": {}}
    for lib in ["sklearn","pandas","numpy","scipy","joblib","shap","mlflow","pandera"]:
        try:
            mod = importlib.import_module(lib)
            env["libraries"][lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env["libraries"][lib] = "not_installed"
    write_json(output_dir/ENVIRONMENT_FILE, env)


def compute_oof_uncertainty(best_estimator, X_train, y_train_log,
                             overpredict_cost=1.0, underpredict_cost=1.0):
    log.info("Computing OOF uncertainty …")
    cv  = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(clone(best_estimator), X_train, y_train_log,
                            cv=cv, n_jobs=N_JOBS)
    res = y_train_log - oof
    oof_rmse = float(np.sqrt(np.mean(res**2)))
    # Convert back to USD RMSE
    y_usd     = np.expm1(y_train_log)
    oof_usd   = np.expm1(oof)
    usd_rmse  = float(np.sqrt(np.mean((y_usd - oof_usd)**2)))
    return {
        "oof_rmse_log":    oof_rmse,
        "oof_rmse_usd":    usd_rmse,
        "oof_r2":          float(r2_score(y_train_log, oof)),
        "lower_band_log":  oof_rmse * underpredict_cost,
        "upper_band_log":  oof_rmse * overpredict_cost,
        "lower_band_usd":  usd_rmse * underpredict_cost,
        "upper_band_usd":  usd_rmse * overpredict_cost,
    }


def _model_version_tag(model):
    return hashlib.sha1(pickle.dumps(model)).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────────────
# Main train() — mirrors reference exactly
# ─────────────────────────────────────────────────────────────────────────────
def train(output_dir, n_iter=20, n_cv_splits=5, fast=False,
          overpredict_cost=1.0, underpredict_cost=1.0):
    log.info("=== Training started (n_jobs=%d) ===", N_JOBS)
    output_dir.mkdir(parents=True, exist_ok=True)

    df                              = fix_data_types(load_data())
    X_train, X_test, y_train, y_te  = split_data(df)

    # Log-transform target (Concept E)
    y_train_log = np.log1p(y_train.to_numpy())
    y_test_log  = np.log1p(y_te.to_numpy())

    # Phase 1: EDA
    save_research_artifacts(X_train, y_train, output_dir)
    baselines = evaluate_baselines(X_train, X_test, y_train, y_te)

    # Concept analyses (train set only)
    miss_tax      = analyse_missing_taxonomy(X_train, output_dir)
    imp_analysis  = analyse_imputation_strategies(X_train, y_train, output_dir)
    out_analysis  = analyse_outliers(X_train, y_train, output_dir)
    robust_anal   = analyse_robust_regression(X_train, y_train, output_dir)
    log_anal      = analyse_log_transform(X_train, y_train, output_dir)
    ord_anal      = analyse_ordinal_encoding(X_train, y_train, output_dir)
    poly_anal     = analyse_polynomial_interactions(X_train, y_train, output_dir)
    reg_analysis  = analyse_regularisation_wide(X_train, y_train, output_dir)

    # Hyperparameter search (on log target)
    search        = tune_model(X_train, y_train_log, n_iter=n_iter,
                               n_cv_splits=n_cv_splits, fast=fast)
    uncertainty   = compute_oof_uncertainty(search.best_estimator_, X_train, y_train_log,
                                            overpredict_cost, underpredict_cost)

    final_model   = clone(search.best_estimator_)
    final_model.fit(X_train, y_train_log)
    y_pred_log    = final_model.predict(X_test)

    test_metrics  = evaluate_predictions(y_test_log, y_pred_log, log_target=True)
    log.info("Test R²=%.4f  RMSE=$%s  MAE=$%s  RMSLE=%.4f",
             test_metrics["r2"], f"{test_metrics['rmse']:,.0f}",
             f"{test_metrics['mae']:,.0f}", test_metrics["rmsle"])

    # Artifact saving
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha1 = _model_version_tag(final_model)
    joblib.dump(final_model, output_dir/f"housing_pipeline_{ts}_{sha1}.joblib")
    joblib.dump(final_model, output_dir/MODEL_FILE)
    save_environment_snapshot(output_dir)
    pd.DataFrame(search.cv_results_).sort_values("rank_test_r2").to_csv(
        output_dir/"cv_results.csv", index=False)
    save_feature_importance(final_model, output_dir)
    save_evaluation_plots(y_te, y_pred_log, output_dir)
    save_error_analysis(X_test, y_te, y_pred_log, output_dir)
    write_json(output_dir/TRAINING_PROFILE_FILE, build_training_profile(X_train, y_train))

    # Diagnostics
    res_diag  = residual_diagnostics(y_test_log, y_pred_log, output_dir)
    plot_learning_curves_housing(final_model, X_train, y_train_log, output_dir)
    fairness  = evaluate_subgroups(final_model, X_test, y_te, y_pred_log, output_dir)
    save_shap_artifacts(final_model, X_test, y_te, y_pred_log, output_dir)

    metrics = {
        "baselines": baselines,
        "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "missing_taxonomy":   miss_tax,
        "imputation":         imp_analysis,
        "outlier_analysis":   out_analysis,
        "robust_regression":  robust_anal,
        "log_transform":      log_anal,
        "ordinal_encoding":   ord_anal,
        "polynomial":         poly_anal,
        "regularisation":     reg_analysis,
        "best_cv": {"best_r2": float(search.best_score_), "best_params": search.best_params_},
        "uncertainty_info":   uncertainty,
        "residual_diag":      res_diag,
        "test_metrics":       test_metrics,
        "fairness":           fairness,
    }
    write_json(output_dir/METRICS_FILE, metrics)
    save_model_card(metrics, fairness, uncertainty, reg_analysis, search, output_dir)
    log_to_mlflow(metrics, search, final_model, output_dir)

    log.info("=== Training complete ===")
    return to_jsonable(metrics)


# ─────────────────────────────────────────────────────────────────────────────
# predict / monitor / sample-input — mirrors reference CLI
# ─────────────────────────────────────────────────────────────────────────────
def predict(artifact_dir, input_csv, output_csv):
    model = joblib.load(artifact_dir/MODEL_FILE)
    unc_band_log = 0.05  # default ±5% log scale
    mp = artifact_dir/METRICS_FILE
    if mp.exists():
        unc_band_log = json.loads(mp.read_text())["uncertainty_info"].get(
            "oof_rmse_log", unc_band_log)
    df = pd.read_csv(input_csv)
    pf = artifact_dir/TRAINING_PROFILE_FILE
    if pf.exists():
        req  = set(json.loads(pf.read_text())["raw_columns"])
        miss = req - set(df.columns)
        if miss: raise ValueError(f"Missing columns: {sorted(miss)}")
    y_pred_log = model.predict(df)
    df["predicted_saleprice"] = np.expm1(y_pred_log)
    df["lower_bound"]         = np.expm1(y_pred_log - unc_band_log)
    df["upper_bound"]         = np.expm1(y_pred_log + unc_band_log)
    df["wide_interval"]       = ((df["upper_bound"]-df["lower_bound"])
                                  > np.expm1(y_pred_log)*0.5).astype(int)
    output_csv = Path(output_csv); output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info("Predictions saved to %s", output_csv.resolve())


def monitor(artifact_dir, input_csv, output_json, missing_rate_alert=0.05, ks_pvalue=0.05):
    profile  = json.loads((artifact_dir/TRAINING_PROFILE_FILE).read_text())
    incoming = pd.read_csv(input_csv)
    req, inc = set(profile["raw_columns"]), set(incoming.columns)
    drift = []
    for col, tr in profile["raw_missing_rate"].items():
        if col not in incoming: continue
        cur = float(incoming[col].isna().mean())
        drift.append({"column":col,"train_rate":float(tr),"current_rate":cur,
                      "change":abs(cur-float(tr)),"alert":abs(cur-float(tr))>=missing_rate_alert})
    ks_rows = []
    for col, stats in profile.get("numeric_train_stats", {}).items():
        if col not in incoming.columns: continue
        vals = incoming[col].dropna().to_numpy()
        if len(vals) < 10: continue
        stat, p = ks_2samp(np.array(stats["quantiles"]), vals)
        ks_rows.append({"column":col,"ks_stat":float(stat),"p_value":float(p),"alert":p<ks_pvalue})
    report = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "row_count": int(len(incoming)),
              "missing_required": sorted(req-inc), "extra": sorted(inc-req),
              "missing_rate_drift": drift, "distribution_drift": ks_rows}
    output_json = Path(output_json); output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, report)
    return report


def create_sample_input(output_csv, rows):
    df = fix_data_types(load_data())
    output_csv = Path(output_csv); output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[TARGET]).head(rows).to_csv(output_csv, index=False)
    log.info("Sample saved to %s", output_csv.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# Utilities — identical to reference pipeline
# ─────────────────────────────────────────────────────────────────────────────
def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI — mirrors reference pipeline
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Ames Housing end-to-end ML pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sp = p.add_subparsers(dest="command", required=True)

    tp = sp.add_parser("train")
    tp.add_argument("--output-dir",        type=Path,  default=Path("artifacts_housing"))
    tp.add_argument("--n-iter",            type=int,   default=20)
    tp.add_argument("--n-cv-splits",       type=int,   default=5)
    tp.add_argument("--fast",              action="store_true",
                    help="50 trees during search → 300 after. ~60% faster.")
    tp.add_argument("--overpredict-cost",  type=float, default=1.0)
    tp.add_argument("--underpredict-cost", type=float, default=1.0)

    pp = sp.add_parser("predict")
    pp.add_argument("--artifact-dir", type=Path, default=Path("artifacts_housing"))
    pp.add_argument("--input-csv",    type=Path, required=True)
    pp.add_argument("--output-csv",   type=Path, default=Path("artifacts_housing/predictions.csv"))

    mp = sp.add_parser("monitor")
    mp.add_argument("--artifact-dir",       type=Path,  default=Path("artifacts_housing"))
    mp.add_argument("--input-csv",          type=Path,  required=True)
    mp.add_argument("--output-json",        type=Path,  default=Path("artifacts_housing/monitor.json"))
    mp.add_argument("--missing-rate-alert", type=float, default=0.05)
    mp.add_argument("--ks-pvalue-alert",    type=float, default=0.05)

    si = sp.add_parser("sample-input")
    si.add_argument("--output-csv", type=Path, default=Path("artifacts_housing/sample.csv"))
    si.add_argument("--rows",       type=int,  default=10)

    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        m = train(args.output_dir, args.n_iter,
                  n_cv_splits=args.n_cv_splits, fast=args.fast,
                  overpredict_cost=args.overpredict_cost,
                  underpredict_cost=args.underpredict_cost)
        log.info("Test R²=%.3f  RMSE=$%s  MAE=$%s",
                 m["test_metrics"]["r2"],
                 f"{m['test_metrics']['rmse']:,.0f}",
                 f"{m['test_metrics']['mae']:,.0f}")
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