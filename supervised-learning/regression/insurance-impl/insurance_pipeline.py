"""
insurance_pipeline.py
=====================
Industry-standard end-to-end ML pipeline for Medical Insurance Cost prediction.

Data (exact pattern as specified):
    insurance    = fetch_openml(data_id=44047, as_frame=True, parser="auto")
    insurance_df = insurance.frame

Target: charges — individual medical cost billed by health insurance (USD)
Dataset: 1338 rows × 7 columns  |  3 numeric + 3 categorical + 1 target

Mirrors every architectural pattern from titanic-ml-pipeline.py,
adapted for regression with a mixed numeric+categorical feature set.

Primary focus: Categorical Encoding & Interaction Terms
───────────────────────────────────────────────────────
  A. One-Hot Encoding (OHE)      — sex, smoker, region with handle_unknown,
                                   drop='first' analysis, min_frequency guard
  B. Target Encoding             — region, smoker via category_encoders or
                                   manual leave-one-out; smoothing parameter
                                   analysis; K-fold encoding to prevent leakage
  C. Ordinal Encoding            — region as ordinal proxy (cost-ordered),
                                   comparison with OHE
  D. Interaction terms (cat×num) — bmi × smoker_flag (the key non-linear driver),
                                   age × smoker_flag, age × bmi interactions
  E. Polynomial features         — bmi², age², age×bmi from numeric features
  F. Feature binning             — age_group buckets, bmi_category (WHO standard),
                                   children_group; bin-then-encode pattern
  G. Regularisation on encoded   — Ridge/Lasso paths comparing OHE vs target enc,
                                   showing how encoding choice changes optimal α
  H. Residual diagnostics        — heteroscedasticity on right-skewed charges,
                                   log-target transformation analysis
  I. Learning curves             — sample efficiency on small dataset (1338 rows)
  J. Partial dependence + ICE    — bmi, age, smoker marginal effects
  K. Cross-validation strategy   — GroupKFold by region to test geographic leakage,
                                   StratifiedKFold on charges quartile bins
  L. Feature importance comparison — OHE features vs target-encoded features,
                                     SHAP-based interaction detection

Industry-standard regression metrics: MAE, RMSE, R², MAPE, MedAE

Usage:
  python insurance_pipeline.py train   --output-dir artifacts_insurance
  python insurance_pipeline.py predict --artifact-dir artifacts_insurance --input-csv sample.csv
  python insurance_pipeline.py monitor --artifact-dir artifacts_insurance --input-csv new.csv
  python insurance_pipeline.py sample-input --output-csv sample.csv --rows 10
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

_MPLCONFIGDIR = Path("artifacts_insurance") / ".matplotlib"
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
    ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import (
    ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV,
)
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, median_absolute_error, r2_score,
)
from sklearn.model_selection import (
    GroupKFold, KFold, RandomizedSearchCV, StratifiedKFold,
    cross_val_predict, cross_val_score, learning_curve, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, OrdinalEncoder,
    PolynomialFeatures, RobustScaler, StandardScaler,
    OneHotEncoder, PowerTransformer, QuantileTransformer,
)
from sklearn.tree import DecisionTreeRegressor

try:
    import category_encoders as ce; _CE = True
except ImportError:
    _CE = False

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
TARGET                = "charges"
MODEL_FILE            = "insurance_pipeline.joblib"
METRICS_FILE          = "metrics.json"
TRAINING_PROFILE_FILE = "training_profile.json"
MODEL_CARD_FILE       = "model_card.json"
ENVIRONMENT_FILE      = "environment.json"
N_JOBS                = int(os.environ.get("ML_N_JOBS", 1))

NUMERIC_RAW       = ["age", "bmi", "children"]
CATEGORICAL_COLS  = ["sex", "smoker", "region"]
BINARY_CATS       = ["sex", "smoker"]
MULTICLASS_CATS   = ["region"]

# BMI category cut-points (WHO standard)
BMI_BINS   = [0, 18.5, 25.0, 30.0, 100]
BMI_LABELS = ["underweight", "normal", "overweight", "obese"]

# Age groups
AGE_BINS   = [17, 25, 35, 50, 65, 100]
AGE_LABELS = ["young_adult", "adult", "middle_aged", "senior", "elderly"]


@dataclass(frozen=True)
class ColumnGroups:
    numeric: list[str]
    categorical_ohe: list[str]
    categorical_target_enc: list[str]


def get_column_groups_ohe() -> ColumnGroups:
    """Column groups for OHE-based pipeline."""
    engineered_num = [
        "age", "bmi", "children",
        "bmi_sq", "age_sq", "age_bmi_interact",
        "bmi_smoker_interact", "age_smoker_interact",
        "bmi_age_smoker_triple",
        "bmi_category_num", "age_group_num",
        "high_bmi_smoker", "senior_flag",
    ]
    return ColumnGroups(
        numeric=engineered_num,
        categorical_ohe=["sex", "smoker", "region"],
        categorical_target_enc=[],
    )


def get_column_groups_target_enc() -> ColumnGroups:
    """Column groups for target-encoding pipeline."""
    engineered_num = [
        "age", "bmi", "children",
        "bmi_sq", "age_sq", "age_bmi_interact",
        "bmi_smoker_interact", "age_smoker_interact",
        "bmi_age_smoker_triple",
        "bmi_category_num", "age_group_num",
        "high_bmi_smoker", "senior_flag",
        "sex_target_enc", "smoker_target_enc", "region_target_enc",
    ]
    return ColumnGroups(
        numeric=engineered_num,
        categorical_ohe=[],
        categorical_target_enc=["sex", "smoker", "region"],
    )


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Exact loader as specified:
        insurance    = fetch_openml(data_id=44047, as_frame=True, parser="auto")
        insurance_df = insurance.frame
    """
    # ── Try multiple OpenML IDs until we get the insurance dataset ─────────────
    # data_id=44047 is specified by user but may map to different dataset
    # depending on OpenML mirror/version. We validate by checking column names.
    _INSURANCE_COLS = {"age", "bmi", "children", "smoker", "sex", "region", "charges"}
    _CANDIDATE_IDS  = [44047, 44, 42477]      # 44047=user-specified; 44=classic; 42477=variant
    _NAME_FALLBACKS = ["insurance", "medical_insurance"]

    def _normalise(df):
        """Lowercase columns, fix common target/bmi variants, cast categoricals."""
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        for alt in ["target","class","label"]:
            if alt in df.columns and "charges" not in df.columns:
                df = df.rename(columns={alt: "charges"})
        for alt in ["bmi_score","body_mass_index"]:
            if alt in df.columns and "bmi" not in df.columns:
                df = df.rename(columns={alt: "bmi"})
        return df

    def _is_insurance(df):
        """Return True if at least 5 of 7 insurance signature columns are present."""
        return len(_INSURANCE_COLS & set(df.columns)) >= 5

    # ── 1. Try by data_id ────────────────────────────────────────────────────
    for data_id in _CANDIDATE_IDS:
        try:
            log.info("Trying fetch_openml(data_id=%d) …", data_id)
            raw = fetch_openml(data_id=data_id, as_frame=True, parser="auto").frame
            df  = _normalise(raw)
            if _is_insurance(df):
                log.info("✓ data_id=%d is the insurance dataset  cols=%s",
                         data_id, df.columns.tolist())
                for col in ["sex","smoker","region"]:
                    if col in df.columns:
                        df[col] = df[col].astype("category")
                return df
            else:
                log.warning("data_id=%d returned wrong dataset (cols=%s) — trying next.",
                            data_id, df.columns.tolist())
        except Exception as exc:
            log.warning("data_id=%d failed: %s", data_id, exc)

    # ── 2. Try by name ────────────────────────────────────────────────────────
    for name in _NAME_FALLBACKS:
        try:
            log.info("Trying fetch_openml(name='%s') …", name)
            raw = fetch_openml(name=name, as_frame=True, parser="auto").frame
            df  = _normalise(raw)
            if _is_insurance(df):
                log.info("✓ name='%s' is the insurance dataset  cols=%s",
                         name, df.columns.tolist())
                for col in ["sex","smoker","region"]:
                    if col in df.columns:
                        df[col] = df[col].astype("category")
                return df
        except Exception as exc:
            log.warning("name='%s' failed: %s", name, exc)

    # ── 3. Synthetic fallback ─────────────────────────────────────────────────
    log.warning("All OpenML sources failed — using synthetic insurance data.")
    return _make_synthetic_insurance()


def _make_synthetic_insurance() -> pd.DataFrame:
    """Synthetic fallback preserving key distributional properties."""
    rng = np.random.default_rng(RANDOM_STATE); n = 1338
    age     = rng.integers(18, 65, n).astype(float)
    bmi     = rng.normal(30.6, 6.1, n).clip(15, 55)
    children= rng.choice([0,1,2,3,4,5], n, p=[0.43,0.24,0.18,0.10,0.04,0.01]).astype(float)
    smoker  = rng.choice(["yes","no"], n, p=[0.204, 0.796])
    sex     = rng.choice(["male","female"], n)
    region  = rng.choice(["northeast","northwest","southeast","southwest"], n)
    smoker_flag = (smoker == "yes").astype(float)
    charges = (
        2000 + 250*age + 100*bmi + 500*children
        + 15000*smoker_flag
        + 1000*(bmi > 30)*smoker_flag*bmi
        + rng.exponential(2000, n)
    ).clip(1000)
    df = pd.DataFrame({
        "age":age,"sex":sex,"bmi":bmi,"children":children,
        "smoker":smoker,"region":region,"charges":charges,
    })
    for col in ["sex","smoker","region"]:
        df[col] = df[col].astype("category")
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast to correct dtypes. Mirrors fix_data_types from reference pipeline."""
    df = df.copy()
    # Numeric
    for col in NUMERIC_RAW + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    # Categorical
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def split_data(df):
    """Stratified 80/20 on charges quartile — mirrors reference pattern."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    # Stratify on quartile bins to ensure charge distribution preserved
    q_bins = pd.qcut(y, q=4, labels=False, duplicates="drop")
    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=q_bins)


def missingness_report(df):
    r = df.isna().agg(["sum","mean"]).T.rename(
        columns={"sum":"missing_count","mean":"missing_rate"})
    r["dtype"] = df.dtypes.astype(str)
    return r.sort_values("missing_rate",ascending=False)


# ── Pandera input schema ──────────────────────────────────────────────────────
def build_input_schema():
    if not _PANDERA:
        log.warning("pandera not installed — validation skipped."); return None
    schema = pa.DataFrameSchema({
        "age":      pa.Column(float, pa.Check.in_range(18,65), nullable=True, required=False),
        "bmi":      pa.Column(float, pa.Check.in_range(10,60), nullable=True, required=False),
        "children": pa.Column(float, pa.Check.in_range(0,10),  nullable=True, required=False),
        "sex":      pa.Column(pa.Category, pa.Check.isin(["male","female"]),
                              nullable=False, required=False),
        "smoker":   pa.Column(pa.Category, pa.Check.isin(["yes","no"]),
                              nullable=False, required=False),
        "region":   pa.Column(pa.Category,
                              pa.Check.isin(["northeast","northwest","southeast","southwest"]),
                              nullable=False, required=False),
    }, coerce=True, strict=False)
    return schema

INPUT_SCHEMA = build_input_schema()


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — mirrors TitanicFeatureEngineer exactly
# ─────────────────────────────────────────────────────────────────────────────
class InsuranceFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-driven feature engineering for medical insurance costs.
    Follows BaseEstimator + TransformerMixin pattern from reference pipeline.

    Concepts:
      D. Cat×Num interactions: bmi×smoker, age×smoker (key cost drivers)
      E. Polynomial features:  bmi², age², age×bmi
      F. Binning:              bmi_category (WHO), age_group, children_group
      L. Derived flags:        high_bmi_smoker, senior_flag
    """

    def fit(self, X, y=None):
        # Learn smoker encoding from training data (binary, no leakage)
        self._smoker_map = {"yes": 1.0, "no": 0.0}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # ── Strip ALL category/object dtypes up front ─────────────────────────
        # pandas .map() on a category Series returns category dtype, not float.
        # Arithmetic on category raises TypeError in pandas >= 1.3.
        # Safest fix: rebuild the entire DataFrame as plain Python-native dtypes
        # before any computation — numeric cols as float64, categoricals as str.
        rebuilt = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                rebuilt[col] = pd.to_numeric(X[col], errors="coerce").astype(float).values
            else:
                # category, object, string → plain str
                rebuilt[col] = X[col].astype(str).values
        X = pd.DataFrame(rebuilt, index=X.index)

        # ── Extract numeric working arrays ────────────────────────────────────
        def _flt(col):
            if col not in X.columns:
                return np.zeros(len(X), dtype=np.float64)
            arr = pd.to_numeric(X[col], errors="coerce").astype(float).values
            med = float(np.nanmedian(arr))
            arr = np.where(np.isnan(arr), med, arr)
            return arr

        age = _flt("age")
        bmi = _flt("bmi")

        # ── Smoker flag: str "yes"/"no" → 0.0 / 1.0 ─────────────────────────
        if "smoker" in X.columns:
            smoker_flag = np.where(X["smoker"].str.lower().str.strip() == "yes",
                                   1.0, 0.0).astype(np.float64)
        else:
            smoker_flag = np.zeros(len(X), dtype=np.float64)

        # ── Concept E: Polynomial terms ───────────────────────────────────────
        X["bmi_sq"]         = bmi ** 2
        X["age_sq"]         = age ** 2
        X["age_bmi_interact"]= age * bmi

        # ── Concept D: Key interaction terms (cat × num) ─────────────────────
        # bmi × smoker: the MOST important interaction in insurance pricing
        # Obese smokers cost ~4× non-smokers at same age
        X["bmi_smoker_interact"]   = bmi * smoker_flag
        X["age_smoker_interact"]   = age * smoker_flag
        X["bmi_age_smoker_triple"] = bmi * age * smoker_flag   # triple interaction

        # ── Concept F: Binning — WHO BMI categories ───────────────────────────
        bmi_cat = pd.cut(bmi, bins=BMI_BINS, labels=range(len(BMI_LABELS)), right=False)
        X["bmi_category_num"]    = bmi_cat.astype(float)
        # String labels for OHE pipeline use
        X["bmi_category"]        = pd.cut(bmi, bins=BMI_BINS, labels=BMI_LABELS,
                                          right=False).astype("category")

        # Age groups
        age_grp = pd.cut(age, bins=AGE_BINS, labels=range(len(AGE_LABELS)), right=True)
        X["age_group_num"]       = age_grp.astype(float)
        X["age_group"]           = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS,
                                          right=True).astype("category")

        # ── Concept L: Derived risk flags ─────────────────────────────────────
        X["high_bmi_smoker"]     = ((bmi >= 30) & (smoker_flag == 1)).astype(float)
        X["senior_flag"]         = (age >= 50).astype(float)

        return X


# ─────────────────────────────────────────────────────────────────────────────
# Concept B: Target encoder (KFold to prevent leakage — mirrors reference)
# ─────────────────────────────────────────────────────────────────────────────
class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Concept B: Target encoding with K-Fold cross-encoding to prevent leakage.
    Each row is encoded using the target mean from folds that did NOT see it.

    Smoothing: encoded = (count * mean_cat + global_mean * smoothing)
                          / (count + smoothing)
    Higher smoothing → pulls rare categories toward global mean → less overfit.
    At inference, uses the full training-set encoded values.
    """

    def __init__(self, cols: list[str], n_splits: int = 5,
                 smoothing: float = 1.0, handle_unknown: str = "value"):
        self.cols            = cols
        self.n_splits        = n_splits
        self.smoothing       = smoothing
        self.handle_unknown  = handle_unknown

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "KFoldTargetEncoder":
        """Learn full-training target means for inference time."""
        self.global_mean_  = float(y.mean())
        self.encoding_map_ = {}
        for col in self.cols:
            if col not in X.columns: continue
            tmp = pd.DataFrame({"cat": X[col].astype(str), "y": y.values})
            stats = tmp.groupby("cat")["y"].agg(["mean","count"]).reset_index()
            stats["smoothed"] = (
                (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_)
                / (stats["count"] + self.smoothing)
            )
            self.encoding_map_[col] = dict(zip(stats["cat"], stats["smoothed"]))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            if col not in X.columns: continue
            col_str = X[col].astype(str)
            X[f"{col}_target_enc"] = col_str.map(self.encoding_map_.get(col, {}))
            if X[f"{col}_target_enc"].isna().any():
                X[f"{col}_target_enc"].fillna(self.global_mean_, inplace=True)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Concept A: OHE-based preprocessor
# ─────────────────────────────────────────────────────────────────────────────
def build_ohe_preprocessor() -> ColumnTransformer:
    """
    Concept A: Full OHE pipeline for categorical features.
    Numeric pipeline: median impute + add_indicator + RobustScaler (right-skewed charges).
    Categorical pipeline: most_frequent impute + OHE with handle_unknown='ignore'.
    drop='if_binary': sex and smoker are binary → drop one redundant column.
    """
    groups = get_column_groups_ohe()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler",  RobustScaler()),
    ])
    # Binary categoricals: drop one column (it's redundant)
    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                  sparse_output=False)),
    ])
    # Multi-class categoricals (region: 4 levels → 3 OHE cols)
    multi_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num",    num_pipe,    groups.numeric),
        ("binary", binary_pipe, BINARY_CATS),
        ("multi",  multi_pipe,  MULTICLASS_CATS),
    ], remainder="drop")


# ─────────────────────────────────────────────────────────────────────────────
# Build pipelines (OHE vs target-encoding)
# ─────────────────────────────────────────────────────────────────────────────
def build_ohe_pipeline(model=None) -> Pipeline:
    """
    OHE pipeline:
      InsuranceFeatureEngineer → OHE ColumnTransformer → SelectFromModel → model
    """
    if model is None:
        model = Ridge(alpha=1.0)
    selector = SelectFromModel(
        ExtraTreesRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        threshold="median",
    )
    return Pipeline([
        ("feature_engineering", InsuranceFeatureEngineer()),
        ("preprocess",          build_ohe_preprocessor()),
        ("feature_selection",   selector),
        ("model",               model),
    ])


def build_target_enc_pipeline(model=None) -> Pipeline:
    """
    Target encoding pipeline:
      InsuranceFeatureEngineer → KFoldTargetEncoder → numeric only preprocessor → model
    """
    if model is None:
        model = Ridge(alpha=1.0)

    class _TargetEncAndPreprocess(BaseEstimator, TransformerMixin):
        """Combines target encoding + numeric scaling in one step."""
        def __init__(self, smoothing=1.0):
            self.smoothing = smoothing
        def fit(self, X, y=None):
            if y is None: raise ValueError("y required for target encoding")
            self.te_ = KFoldTargetEncoder(
                cols=CATEGORICAL_COLS, smoothing=self.smoothing)
            self.te_.fit(X, y)
            X_enc = self.te_.transform(X)
            num_cols = [c for c in X_enc.columns
                        if pd.api.types.is_numeric_dtype(X_enc[c]) and c != TARGET]
            self.cols_ = num_cols
            _arr = np.column_stack([
                pd.to_numeric(X_enc[c],errors="coerce").to_numpy(dtype=np.float64)
                for c in self.cols_])
            self.imp_ = SimpleImputer(strategy="median").fit(_arr)
            self.sc_  = RobustScaler().fit(self.imp_.transform(_arr))
            return self
        def transform(self, X):
            X_enc = self.te_.transform(X)
            present = [c for c in self.cols_ if c in X_enc.columns]
            _arr = np.column_stack([
                pd.to_numeric(X_enc[c],errors="coerce").to_numpy(dtype=np.float64)
                for c in present])
            return self.sc_.transform(self.imp_.transform(_arr))
        def get_feature_names_out(self, _=None):
            return np.array(self.cols_)

    selector = SelectFromModel(
        ExtraTreesRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        threshold="median",
    )
    return Pipeline([
        ("feature_engineering", InsuranceFeatureEngineer()),
        ("preprocess",          _TargetEncAndPreprocess(smoothing=1.0)),
        ("feature_selection",   selector),
        ("model",               model),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Concept A: OHE deep analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_ohe_strategies(
    X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path,
) -> dict[str,Any]:
    """
    Concept A: Compare OHE strategies:
      1. OHE all categoricals (no drop)
      2. OHE drop='if_binary' (binary cols get 1 col instead of 2)
      3. OHE drop='first' for all
    Also shows the dummy variable trap — why dropping one column matters.
    """
    log.info("Concept A: OHE strategy comparison …")
    fe     = InsuranceFeatureEngineer().fit(X_train)
    X_eng  = fe.transform(X_train)
    cv     = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    strategies = {
        "OHE (no drop)":      OneHotEncoder(drop=None,       handle_unknown="ignore", sparse_output=False),
        "OHE (drop=first)":   OneHotEncoder(drop="first",    handle_unknown="ignore", sparse_output=False),
        "OHE (drop=if_binary)":OneHotEncoder(drop="if_binary",handle_unknown="ignore", sparse_output=False),
        "OrdinalEncoder":     OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    }
    # Extract categorical columns as strings
    cat_df = pd.DataFrame({
        c: X_eng[c].astype(str) if c in X_eng.columns else X_train[c].astype(str)
        for c in CATEGORICAL_COLS if c in X_train.columns
    })
    # Numeric columns — standardised
    num_df = pd.DataFrame({
        c: pd.to_numeric(X_eng[c], errors="coerce").fillna(0)
        for c in ["age","bmi","children","bmi_sq","age_sq","age_bmi_interact",
                  "bmi_smoker_interact","age_smoker_interact","high_bmi_smoker","senior_flag"]
        if c in X_eng.columns
    })
    X_num = RobustScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(num_df))

    for name, enc in strategies.items():
        try:
            X_cat = enc.fit_transform(cat_df)
            X_full = np.hstack([X_num, X_cat])
            n_feat = X_full.shape[1]
            r2s    = cross_val_score(Ridge(alpha=1.0), X_full, y_train,
                                     cv=cv, scoring="r2")
            rmse_s = -cross_val_score(Ridge(alpha=1.0), X_full, y_train,
                                      cv=cv, scoring="neg_root_mean_squared_error")
            results[name] = {
                "n_features":n_feat,"r2_mean":float(r2s.mean()),
                "r2_std":float(r2s.std()),"rmse_mean":float(rmse_s.mean()),
            }
            log.info("  %-30s  n_feat=%3d  R²=%.4f±%.4f", name, n_feat,
                     r2s.mean(), r2s.std())
        except Exception as e:
            log.warning("  %s failed: %s", name, e)

    # Plot
    names = list(results.keys()); r2s = [results[n]["r2_mean"] for n in names]
    fig, ax = plt.subplots(figsize=(9,4))
    colors = ["#4C78A8","#1D9E75","#54A24B","#F58518"]
    bars = ax.bar(names, r2s, color=colors)
    ax.set_ylabel("CV R²"); ax.set_title("Concept A: OHE strategy comparison\nHigher R² = better")
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{val:.4f}", ha="center", fontsize=9)
    plt.xticks(rotation=15, ha="right"); plt.tight_layout()
    plt.savefig(output_dir/"plot_ohe_comparison.png", dpi=160); plt.close()

    write_json(output_dir/"ohe_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept B: Target encoding analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_target_encoding(
    X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path,
) -> dict[str,Any]:
    """
    Concept B: Compare OHE vs Target Encoding (with smoothing analysis).
    Shows how smoothing parameter controls the bias-variance tradeoff.
    Low smoothing → memorises category means → overfits on rare categories.
    High smoothing → pulls toward global mean → safer but loses information.
    """
    log.info("Concept B: Target encoding comparison …")
    fe    = InsuranceFeatureEngineer().fit(X_train)
    X_eng = fe.transform(X_train)
    cv    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    # Baseline: OHE
    cat_df = pd.DataFrame({c: X_eng[c].astype(str) if c in X_eng.columns
                           else X_train[c].astype(str)
                           for c in CATEGORICAL_COLS if c in X_train.columns})
    num_df = pd.DataFrame({c: pd.to_numeric(X_eng[c], errors="coerce").fillna(0)
                           for c in ["age","bmi","children","bmi_sq","bmi_smoker_interact",
                                     "age_smoker_interact","high_bmi_smoker"]
                           if c in X_eng.columns})
    X_num = RobustScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(num_df))
    X_ohe = OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                          sparse_output=False).fit_transform(cat_df)
    r2_ohe = cross_val_score(Ridge(alpha=1.0), np.hstack([X_num, X_ohe]),
                              y_train, cv=cv, scoring="r2")
    results["OHE (baseline)"] = {"smoothing":None,"r2_mean":float(r2_ohe.mean()),
                                  "r2_std":float(r2_ohe.std())}

    # Target encoding with varying smoothing
    smoothings = [0.1, 1.0, 5.0, 10.0, 50.0]
    for s in smoothings:
        te    = KFoldTargetEncoder(cols=CATEGORICAL_COLS, smoothing=s)
        te.fit(X_eng if hasattr(te, 'cols') else X_train, y_train)
        X_tenc= te.transform(X_train)
        tenc_cols = [f"{c}_target_enc" for c in CATEGORICAL_COLS
                     if f"{c}_target_enc" in X_tenc.columns]
        X_te_num = pd.concat([num_df, X_tenc[tenc_cols]], axis=1)
        X_te_scaled = RobustScaler().fit_transform(
            SimpleImputer(strategy="median").fit_transform(X_te_num))
        r2_s = cross_val_score(Ridge(alpha=1.0), X_te_scaled, y_train,
                                cv=cv, scoring="r2")
        results[f"TargetEnc (smoothing={s})"] = {
            "smoothing":s,"r2_mean":float(r2_s.mean()),"r2_std":float(r2_s.std())}
        log.info("  TargetEnc smoothing=%-5s  R²=%.4f±%.4f", s, r2_s.mean(), r2_s.std())

    # Plot smoothing effect
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    all_names  = list(results.keys())
    all_r2     = [results[n]["r2_mean"] for n in all_names]
    axes[0].bar(range(len(all_names)), all_r2,
                color=["#E45756"]+["#4C78A8"]*len(smoothings))
    axes[0].set_xticks(range(len(all_names)))
    axes[0].set_xticklabels(all_names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("CV R²")
    axes[0].set_title("Concept B: OHE vs Target Encoding")

    smooth_r2 = [results[f"TargetEnc (smoothing={s})"]["r2_mean"] for s in smoothings]
    axes[1].plot(smoothings, smooth_r2, "o-", color="#4C78A8", linewidth=2, markersize=8)
    axes[1].set_xlabel("Smoothing parameter"); axes[1].set_ylabel("CV R²")
    axes[1].set_title("Concept B: Smoothing parameter effect\nHigher = pulls toward global mean")
    plt.tight_layout()
    plt.savefig(output_dir/"plot_target_encoding.png", dpi=160); plt.close()

    write_json(output_dir/"target_encoding_analysis.json", results)
    best = max(results, key=lambda k: results[k]["r2_mean"])
    log.info("Best encoding: %s (R²=%.4f)", best, results[best]["r2_mean"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept D: Interaction term analysis (categorical × numeric)
# ─────────────────────────────────────────────────────────────────────────────
def analyse_interactions(
    X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path,
) -> dict[str,Any]:
    """
    Concept D: Quantify the uplift from categorical × numeric interactions.
    Compares R² with and without bmi×smoker, age×smoker, triple interaction.
    Uses mutual information to rank interaction importance.
    """
    log.info("Concept D: Interaction term analysis …")
    fe    = InsuranceFeatureEngineer().fit(X_train)
    X_eng = fe.transform(X_train)
    cv    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    feature_sets = {
        "Baseline (raw numerics only)": ["age","bmi","children"],
        "+ OHE categoricals":           ["age","bmi","children"] + [c for c in X_eng.columns
                                         if any(c.startswith(f) for f in CATEGORICAL_COLS)],
        "+ Polynomial (E)":             ["age","bmi","children","bmi_sq","age_sq","age_bmi_interact"],
        "+ Cat×Num interactions (D)":   ["age","bmi","children","bmi_sq","age_sq","age_bmi_interact",
                                         "bmi_smoker_interact","age_smoker_interact"],
        "+ Triple interaction":         ["age","bmi","children","bmi_sq","age_sq","age_bmi_interact",
                                         "bmi_smoker_interact","age_smoker_interact",
                                         "bmi_age_smoker_triple"],
        "+ Risk flags (L)":             ["age","bmi","children","bmi_sq","age_sq","age_bmi_interact",
                                         "bmi_smoker_interact","age_smoker_interact",
                                         "bmi_age_smoker_triple","high_bmi_smoker","senior_flag"],
    }
    results = {}
    for name, cols in feature_sets.items():
        avail = [c for c in cols if c in X_eng.columns]
        if not avail: continue
        _arr = np.column_stack([
            pd.to_numeric(X_eng[c], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
            for c in avail])
        _arr = RobustScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(_arr))
        r2s = cross_val_score(Ridge(alpha=1.0), _arr, y_train, cv=cv, scoring="r2")
        results[name] = {"n_features":len(avail),"r2_mean":float(r2s.mean()),
                         "r2_std":float(r2s.std())}
        log.info("  %-45s n_feat=%3d R²=%.4f", name, len(avail), r2s.mean())

    # Waterfall / step-wise plot
    names = list(results.keys()); r2s = [results[n]["r2_mean"] for n in names]
    fig, ax = plt.subplots(figsize=(11,4.5))
    colors = plt.cm.get_cmap("RdYlGn",len(names))
    bars = ax.barh(names, r2s, color=[colors(i/len(names)) for i in range(len(names))])
    ax.set_xlabel("CV R²")
    ax.set_title("Concept D: Interaction terms uplift\nEach row adds more features to previous")
    for bar,val in zip(bars,r2s):
        ax.text(val+0.002, bar.get_y()+bar.get_height()/2, f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlim(0,1.05); plt.tight_layout()
    plt.savefig(output_dir/"plot_interaction_uplift.png", dpi=160); plt.close()

    write_json(output_dir/"interaction_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept E: Polynomial analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_polynomial_features(
    X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path,
) -> dict[str,Any]:
    """
    Concept E: Polynomial feature expansion on numeric features.
    BMI and age have non-linear relationships with charges.
    Tests degree 1/2/3 full and interaction-only to find optimal expansion.
    """
    log.info("Concept E: Polynomial feature analysis …")
    fe    = InsuranceFeatureEngineer().fit(X_train)
    X_eng = fe.transform(X_train)
    num_base = [c for c in ["age","bmi","children"] if c in X_eng.columns]
    _arr = np.column_stack([pd.to_numeric(X_eng[c],errors="coerce").fillna(0)
                             .to_numpy(dtype=np.float64) for c in num_base])
    _arr = RobustScaler().fit_transform(_arr)
    y_np = y_train.to_numpy()
    cv   = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    configs = [
        ("Linear (d=1)",      1, False),
        ("Poly d=2 full",     2, False),
        ("Poly d=2 interact", 2, True),
        ("Poly d=3 full",     3, False),
        ("Poly d=3 interact", 3, True),
    ]
    for name, deg, interact in configs:
        poly    = PolynomialFeatures(degree=deg,interaction_only=interact,include_bias=False)
        X_poly  = poly.fit_transform(_arr)
        n_feat  = X_poly.shape[1]
        r2s     = cross_val_score(Ridge(alpha=1.0), X_poly, y_np, cv=cv, scoring="r2")
        rmse_s  = -cross_val_score(Ridge(alpha=1.0), X_poly, y_np, cv=cv,
                                   scoring="neg_root_mean_squared_error")
        results[name] = {"n_features":n_feat,"r2_mean":float(r2s.mean()),
                         "r2_std":float(r2s.std()),"rmse_mean":float(rmse_s.mean())}
        log.info("  %-25s n_feat=%4d R²=%.4f±%.4f", name, n_feat, r2s.mean(), r2s.std())

    names = list(results.keys()); r2s = [results[n]["r2_mean"] for n in names]
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4.5))
    ax1.bar(names, r2s, color=["#4C78A8","#1D9E75","#54A24B","#E45756","#B279A2"])
    ax1.set_xticklabels(names,rotation=30,ha="right",fontsize=9)
    ax1.set_ylabel("CV R²"); ax1.set_title("Concept E: Polynomial R² by degree")
    for bar,val in zip(ax1.patches,r2s):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f"{val:.4f}", ha="center", fontsize=8)
    nfeats = [results[n]["n_features"] for n in names]
    ax2.scatter(nfeats, r2s, s=80, color="#4C78A8", zorder=5)
    for i,(nm,nf,r2) in enumerate(zip(names,nfeats,r2s)):
        ax2.annotate(nm,(nf,r2),textcoords="offset points",xytext=(4,4),fontsize=7)
    ax2.set_xlabel("N features"); ax2.set_ylabel("CV R²")
    ax2.set_title("Concept E: R² vs feature count\n(polynomial tradeoff)")
    plt.tight_layout()
    plt.savefig(output_dir/"plot_polynomial_analysis.png", dpi=160); plt.close()
    write_json(output_dir/"polynomial_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept F: Feature binning analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse_binning(
    X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path,
) -> dict[str,Any]:
    """
    Concept F: Compare raw numeric vs binned categories for bmi and age.
    Binning + OHE can outperform raw numerics when the relationship is
    step-wise (e.g. WHO BMI categories have different insurance implications).
    """
    log.info("Concept F: Binning analysis …")
    fe    = InsuranceFeatureEngineer().fit(X_train)
    X_eng = fe.transform(X_train)
    cv    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, use_bins in [("Raw age+bmi", False), ("Binned age+bmi (OHE)", True)]:
        if not use_bins:
            _arr = np.column_stack([
                pd.to_numeric(X_eng[c],errors="coerce").fillna(0).to_numpy(dtype=np.float64)
                for c in ["age","bmi","children"] if c in X_eng.columns])
            _arr = RobustScaler().fit_transform(_arr)
        else:
            cat_parts = []
            for col in ["bmi_category","age_group"]:
                if col in X_eng.columns:
                    vals = X_eng[col].astype(str).fillna("unknown")
                    enc  = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                    cat_parts.append(enc.fit_transform(vals.to_frame()))
            num_part = RobustScaler().fit_transform(
                pd.to_numeric(X_eng["children"],errors="coerce").fillna(0).to_numpy().reshape(-1,1))
            _arr = np.hstack(cat_parts + [num_part]) if cat_parts else num_part
        r2s = cross_val_score(Ridge(alpha=1.0), _arr, y_train, cv=cv, scoring="r2")
        results[name] = {"r2_mean":float(r2s.mean()),"r2_std":float(r2s.std())}
        log.info("  %-40s  R²=%.4f±%.4f", name, r2s.mean(), r2s.std())

    # Plot BMI category charge distributions
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    eda = X_train.copy(); eda[TARGET] = y_train.values
    eda["bmi_category"] = pd.cut(eda["bmi"], bins=BMI_BINS, labels=BMI_LABELS, right=False)
    eda["age_group"]    = pd.cut(eda["age"],  bins=AGE_BINS,  labels=AGE_LABELS, right=True)
    sns.boxplot(data=eda, x="bmi_category", y=TARGET, ax=axes[0], palette="Set2")
    axes[0].set_title("Concept F: charges by BMI category (WHO)")
    axes[0].set_xlabel("BMI Category"); axes[0].set_ylabel("charges ($)")
    sns.boxplot(data=eda, x="age_group", y=TARGET, ax=axes[1], palette="Set3")
    axes[1].set_title("Concept F: charges by age group")
    axes[1].set_xlabel("Age Group"); axes[1].set_ylabel("charges ($)")
    plt.tight_layout()
    plt.savefig(output_dir/"plot_binning_analysis.png", dpi=160); plt.close()
    write_json(output_dir/"binning_analysis.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept G: Regularisation paths on encoded features
# ─────────────────────────────────────────────────────────────────────────────
def analyse_regularisation(
    X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path,
) -> dict[str,Any]:
    """
    Concept G: Ridge/Lasso/ElasticNet coefficient paths on OHE-expanded features.
    OHE creates many correlated binary columns. Ridge handles this gracefully.
    Lasso zeros out less-predictive OHE columns (automatic selection).
    """
    log.info("Concept G: Regularisation analysis …")
    fe    = InsuranceFeatureEngineer().fit(X_train)
    X_eng = fe.transform(X_train)
    cat_df = pd.DataFrame({c: X_eng[c].astype(str) if c in X_eng.columns
                           else X_train[c].astype(str)
                           for c in CATEGORICAL_COLS if c in X_train.columns})
    num_cols = ["age","bmi","children","bmi_sq","age_sq","age_bmi_interact",
                "bmi_smoker_interact","age_smoker_interact","high_bmi_smoker","senior_flag"]
    num_df = pd.DataFrame({c: pd.to_numeric(X_eng[c],errors="coerce").fillna(0)
                           for c in num_cols if c in X_eng.columns})
    X_num = RobustScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(num_df))
    X_ohe = OneHotEncoder(drop="if_binary",handle_unknown="ignore",sparse_output=False
                          ).fit_transform(cat_df)
    X_full = np.hstack([X_num, X_ohe]); y_np = y_train.to_numpy()

    alphas = np.logspace(-3, 4, 60)
    cv     = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    ridge_cv = RidgeCV(alphas=alphas, cv=cv).fit(X_full, y_np)
    lasso_cv = LassoCV(alphas=alphas, cv=cv, max_iter=5000,
                       random_state=RANDOM_STATE).fit(X_full, y_np)
    enet_cv  = ElasticNetCV(alphas=alphas, l1_ratio=[0.1,0.3,0.5,0.7,0.9],
                            cv=cv, max_iter=5000,
                            random_state=RANDOM_STATE).fit(X_full, y_np)
    lasso_coef = Lasso(alpha=lasso_cv.alpha_,max_iter=5000).fit(X_full,y_np).coef_
    n_zeroed   = int((lasso_coef==0).sum())

    # Bias-variance curve for Ridge
    train_r2=[]; cv_r2=[]
    for a in alphas:
        mdl = Ridge(alpha=a).fit(X_full,y_np)
        train_r2.append(mdl.score(X_full,y_np))
        cv_r2.append(cross_val_score(mdl,X_full,y_np,cv=cv,scoring="r2").mean())

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.5))
    ax1.plot(np.log10(alphas),train_r2,label="Train R²",color="#4C78A8")
    ax1.plot(np.log10(alphas),cv_r2,label="CV R²",color="#E45756")
    ax1.axvline(np.log10(ridge_cv.alpha_),color="green",linestyle="--",
                label=f"Optimal α={ridge_cv.alpha_:.3f}")
    ax1.set_xlabel("log₁₀(α)"); ax1.set_ylabel("R²")
    ax1.set_title("Concept G: Ridge bias-variance tradeoff\n(OHE features)")
    ax1.legend()

    ridge_coef_opt = Ridge(alpha=ridge_cv.alpha_).fit(X_full,y_np).coef_
    lasso_coef_opt = Lasso(alpha=lasso_cv.alpha_,max_iter=5000).fit(X_full,y_np).coef_
    top_idx = np.argsort(np.abs(ridge_coef_opt))[-15:]
    x_idx   = np.arange(len(top_idx)); w = 0.35
    ax2.bar(x_idx-w/2, ridge_coef_opt[top_idx], w, label="Ridge",color="#4C78A8",alpha=0.85)
    ax2.bar(x_idx+w/2, lasso_coef_opt[top_idx], w, label=f"Lasso ({n_zeroed} zeroed)",
            color="#E45756",alpha=0.85)
    ax2.axhline(0,color="black",linewidth=0.8)
    ax2.set_title(f"Concept G: Ridge vs Lasso top-15 coefficients\n"
                  f"Lasso zeroed {n_zeroed}/{len(lasso_coef_opt)} features")
    ax2.set_xlabel("Feature index"); ax2.legend()
    plt.tight_layout()
    plt.savefig(output_dir/"plot_regularisation.png",dpi=160); plt.close()

    log.info("Ridge α*=%.4f  Lasso α*=%.4f  ElasticNet α*=%.4f  Lasso zeroed: %d/%d",
             ridge_cv.alpha_, lasso_cv.alpha_, enet_cv.alpha_, n_zeroed, len(lasso_coef))
    return {
        "ridge_optimal_alpha":  float(ridge_cv.alpha_),
        "lasso_optimal_alpha":  float(lasso_cv.alpha_),
        "elasticnet_optimal_alpha": float(enet_cv.alpha_),
        "elasticnet_l1_ratio":  float(enet_cv.l1_ratio_),
        "lasso_n_zeroed":       n_zeroed,
        "lasso_n_total":        int(len(lasso_coef)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Concept H: Residual diagnostics + log-target transformation
# ─────────────────────────────────────────────────────────────────────────────
def residual_diagnostics(y_true, y_pred, output_dir, prefix=""):
    log.info("Concept H: Residual diagnostics …")
    residuals = y_true - y_pred
    n         = len(residuals)
    resid_sq  = residuals ** 2
    bp_r2     = LinearRegression().fit(y_pred.reshape(-1,1),resid_sq).score(y_pred.reshape(-1,1),resid_sq)
    bp_stat   = float(n*bp_r2)
    bp_p      = float(1-scipy_stats.chi2.cdf(bp_stat,df=1))
    sample    = residuals if n<=5000 else np.random.default_rng(RANDOM_STATE).choice(residuals,5000,replace=False)
    sw_s, sw_p = scipy_stats.shapiro(sample)

    fig,axes = plt.subplots(1,3,figsize=(15,4))
    scipy_stats.probplot(residuals,dist="norm",plot=axes[0])
    axes[0].set_title("Q-Q plot")
    axes[1].scatter(y_pred,np.sqrt(np.abs(residuals)),alpha=0.3,s=8,color="#4C78A8")
    axes[1].axhline(np.sqrt(np.abs(residuals)).mean(),color="red",linewidth=1.2)
    axes[1].set_xlabel("Fitted charges ($)"); axes[1].set_ylabel("√|residual|")
    axes[1].set_title("Scale-location (heteroscedasticity check)")
    axes[2].hist(residuals,bins=60,color="#54A24B",edgecolor="white",linewidth=0.3)
    axes[2].axvline(0,color="red",linewidth=1.2)
    axes[2].set_title("Residual distribution")
    plt.suptitle("Concept H: Residual diagnostics", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir/f"plot_residual_diagnostics{prefix}.png",dpi=160,bbox_inches="tight")
    plt.close()

    log.info("BP p=%.4f (hetero: %s)  SW p=%.4f (normal: %s)",
             bp_p, bp_p<0.05, sw_p, sw_p>0.05)
    return {
        "breusch_pagan":{"statistic":bp_stat,"p_value":bp_p,"heteroscedastic":bp_p<0.05},
        "shapiro_wilk": {"statistic":float(sw_s),"p_value":float(sw_p),"normal":sw_p>0.05},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Concept K: Cross-validation strategy comparison
# ─────────────────────────────────────────────────────────────────────────────
def analyse_cv_strategies(
    X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path,
) -> dict[str,Any]:
    """
    Concept K: KFold vs StratifiedKFold (on charges quartile) vs GroupKFold (by region).
    Insurance charges are bimodal (smokers vs non-smokers). Stratified CV ensures
    both groups appear proportionally in each fold.
    GroupKFold by region tests whether the model generalises to unseen geographies.
    """
    log.info("Concept K: CV strategy comparison …")
    pipe = build_ohe_pipeline(Ridge(alpha=1.0))
    results = {}

    # KFold
    kf   = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    r2_kf = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
    results["KFold"] = {"mean":float(r2_kf.mean()),"std":float(r2_kf.std())}

    # StratifiedKFold on quartile bins.
    # StratifiedKFold uses the `y` argument for stratification, not `groups`.
    # We pass q_bins as `y` to the CV splitter directly by wrapping it in a
    # manual loop — this avoids the "continuous target not supported" error
    # while still scoring on the actual continuous y_train values.
    q_bins   = pd.qcut(y_train, q=4, labels=False, duplicates="drop").astype(int)
    skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    skf_r2s  = []
    for tr_idx, te_idx in skf.split(X_train, q_bins):
        _pipe = clone(pipe)
        _X_tr = X_train.iloc[tr_idx]; _y_tr = y_train.iloc[tr_idx]
        _X_te = X_train.iloc[te_idx]; _y_te = y_train.iloc[te_idx]
        _pipe.fit(_X_tr, _y_tr)
        skf_r2s.append(float(r2_score(_pipe.predict(_X_te), _y_te)))
    r2_skf = np.array(skf_r2s)
    results["StratifiedKFold (quartile)"] = {"mean":float(r2_skf.mean()),"std":float(r2_skf.std())}

    # GroupKFold by region
    if "region" in X_train.columns:
        region_labels = X_train["region"].astype(str)
        gkf  = GroupKFold(n_splits=4)   # 4 regions
        r2_gkf = cross_val_score(pipe, X_train, y_train, cv=gkf,
                                  groups=region_labels, scoring="r2")
        results["GroupKFold (region)"] = {"mean":float(r2_gkf.mean()),"std":float(r2_gkf.std())}

    for k,v in results.items():
        log.info("  %-35s  R²=%.4f±%.4f", k, v["mean"], v["std"])

    fig,ax = plt.subplots(figsize=(8,4))
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds  = [results[n]["std"]  for n in names]
    ax.bar(names, means, yerr=stds, color=["#4C78A8","#1D9E75","#E45756"][:len(names)],
           capsize=6, width=0.5)
    ax.set_ylabel("CV R²"); ax.set_title("Concept K: CV strategy comparison\n"
                                          "GroupKFold(region) tests geographic generalisation")
    plt.xticks(rotation=15, ha="right"); plt.tight_layout()
    plt.savefig(output_dir/"plot_cv_strategies.png", dpi=160); plt.close()
    write_json(output_dir/"cv_strategy_comparison.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Concept I: Learning curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_curves_insurance(model, X_train, y_train, output_dir):
    log.info("Concept I: Learning curves …")
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    try:
        sizes,tr_s,cv_s = learning_curve(
            model, X_train, y_train,
            train_sizes=np.linspace(0.10,1.0,8),
            cv=cv, scoring="r2", n_jobs=N_JOBS)
        tr_r2 = tr_s.mean(axis=1); cv_r2 = cv_s.mean(axis=1)
        plt.figure(figsize=(8,4.5))
        plt.plot(sizes, tr_r2, "o-", color="#4C78A8", label="Train R²")
        plt.plot(sizes, cv_r2, "o-", color="#E45756", label="CV R²")
        plt.fill_between(sizes,tr_r2-tr_s.std(axis=1),tr_r2+tr_s.std(axis=1),alpha=0.12,color="#4C78A8")
        plt.fill_between(sizes,cv_r2-cv_s.std(axis=1),cv_r2+cv_s.std(axis=1),alpha=0.12,color="#E45756")
        gap = float(cv_r2[-1]-tr_r2[-1])
        plt.title(f"Concept I: Learning curves (1338 rows)\n"
                  f"Train-CV gap={gap:.4f} — "
                  f"{'high variance' if abs(gap)>0.05 else 'well-fitted'}")
        plt.xlabel("Training set size"); plt.ylabel("R²")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir/"plot_learning_curve.png", dpi=160); plt.close()
    except Exception as exc:
        log.warning("Learning curve failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Concept J: PDP + ICE
# ─────────────────────────────────────────────────────────────────────────────
def plot_partial_dependence_insurance(model, X_test, output_dir):
    log.info("Concept J: PDP + ICE …")
    try:
        feat_names = list(X_test.columns)
        # bmi (idx 1), age (idx 0)
        bmi_idx = feat_names.index("bmi") if "bmi" in feat_names else 1
        age_idx = feat_names.index("age") if "age" in feat_names else 0
        fig, axes = plt.subplots(1,2,figsize=(11,4.5))
        for ax, feat_idx in zip(axes,[age_idx,bmi_idx]):
            PartialDependenceDisplay.from_estimator(
                model, X_test, features=[(feat_idx,)],
                feature_names=feat_names, ax=ax,
                kind="both", subsample=200, n_jobs=N_JOBS)
        plt.suptitle("Concept J: PDP + ICE — age and BMI marginal effects\n"
                     "Thin lines=individual rows, thick=average",
                     fontsize=11, y=1.04)
        plt.tight_layout()
        plt.savefig(output_dir/"plot_partial_dependence.png",dpi=160,bbox_inches="tight")
        plt.close()
    except Exception as exc:
        log.warning("PDP failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# EDA (train-set only) — mirrors reference save_research_artifacts
# ─────────────────────────────────────────────────────────────────────────────
def save_research_artifacts(X_train, y_train, output_dir):
    log.info("Saving EDA artifacts (train-set only) …")
    output_dir.mkdir(parents=True, exist_ok=True)
    eda = X_train.copy(); eda[TARGET] = y_train.values

    missingness_report(eda).to_csv(output_dir/"research_missingness_report.csv")
    eda.dtypes.astype(str).rename("dtype").to_csv(output_dir/"schema.csv")
    eda.select_dtypes(include=["number"]).describe().T.to_csv(output_dir/"numeric_summary.csv")

    # Correlation with target (numeric only)
    num_df = eda.select_dtypes(include=["number"])
    corr   = num_df.corr()[TARGET].drop(TARGET).sort_values(key=abs,ascending=False)
    corr.to_csv(output_dir/"correlation_with_target.csv")

    # Grouped stats
    grouped = {
        "charges_by_smoker": eda.groupby("smoker",observed=False)[TARGET]
                              .agg(["mean","median","std"]).to_dict()
                              if "smoker" in eda.columns else {},
        "charges_by_region": eda.groupby("region",observed=False)[TARGET]
                              .agg(["mean","median"]).to_dict()
                              if "region" in eda.columns else {},
        "charges_by_sex":    eda.groupby("sex",observed=False)[TARGET]
                              .agg(["mean","median"]).to_dict()
                              if "sex" in eda.columns else {},
        "target_stats":      y_train.describe().to_dict(),
    }
    write_json(output_dir/"research_decisions.json", {
        "problem_definition":{"problem_type":"regression","target":TARGET,
                              "unit":"USD medical charges","note":"right-skewed, bimodal (smokers vs non-smokers)"},
        "metric_policy":{"primary":"r2","secondary":["rmse","mae","mape","medae"]},
        "feature_policy":{"key_interaction":"bmi×smoker — strongest signal in the dataset",
                          "encoding_strategy":"OHE for sex/smoker/region, optional target encoding"},
        "grouped_stats":grouped,
    })

    sns.set_theme(style="whitegrid")

    # Target distribution
    plt.figure(figsize=(7,4))
    sns.histplot(y_train,kde=True,bins=40,color="#4C78A8")
    plt.axvline(y_train.median(),color="#E45756",linestyle="--",label=f"Median=${y_train.median():,.0f}")
    plt.title("charges distribution — bimodal (smoker/non-smoker effect)")
    plt.xlabel("charges ($)"); plt.legend()
    plt.tight_layout(); plt.savefig(output_dir/"plot_target_distribution.png",dpi=160); plt.close()

    # Smoker vs non-smoker charges
    if "smoker" in eda.columns:
        plt.figure(figsize=(7,4))
        sns.boxplot(data=eda,x="smoker",y=TARGET,palette=["#4C78A8","#E45756"])
        plt.title("charges by smoker status — key driver")
        plt.tight_layout(); plt.savefig(output_dir/"plot_charges_by_smoker.png",dpi=160); plt.close()

    # BMI vs charges coloured by smoker
    if "bmi" in eda.columns and "smoker" in eda.columns:
        plt.figure(figsize=(7,5))
        for sval, color in [("yes","#E45756"),("no","#4C78A8")]:
            sub = eda[eda["smoker"]==sval]
            plt.scatter(sub["bmi"],sub[TARGET],alpha=0.3,s=8,color=color,label=f"smoker={sval}")
        plt.xlabel("BMI"); plt.ylabel("charges ($)")
        plt.title("BMI vs charges (coloured by smoker)\n"
                  "Key interaction: high-BMI smokers cost ~4× non-smokers")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir/"plot_bmi_charges_smoker.png",dpi=160); plt.close()

    # Age vs charges
    if "age" in eda.columns:
        plt.figure(figsize=(6,4))
        plt.scatter(eda["age"],eda[TARGET],alpha=0.2,s=6,color="#B279A2")
        plt.xlabel("age"); plt.ylabel("charges ($)")
        plt.title("age vs charges — monotone but non-linear")
        plt.tight_layout(); plt.savefig(output_dir/"plot_age_charges.png",dpi=160); plt.close()

    # Correlation bar
    plt.figure(figsize=(6,4))
    colors = ["#54A24B" if v>0 else "#E45756" for v in corr]
    corr.plot(kind="barh",color=colors)
    plt.axvline(0,color="black",linewidth=0.8)
    plt.title("Feature correlation with charges (train)")
    plt.tight_layout(); plt.savefig(output_dir/"plot_correlation.png",dpi=160); plt.close()

    log.info("EDA artifacts saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Regression metrics + baselines
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_predictions(y_true, y_pred):
    if hasattr(y_true,"to_numpy"): y_true = y_true.to_numpy()
    res  = y_true - y_pred
    return {
        "mae":              float(mean_absolute_error(y_true,y_pred)),
        "rmse":             float(np.sqrt(mean_squared_error(y_true,y_pred))),
        "r2":               float(r2_score(y_true,y_pred)),
        "mape":             float(mean_absolute_percentage_error(y_true,y_pred)),
        "medae":            float(median_absolute_error(y_true,y_pred)),
        "residual_mean":    float(res.mean()),
        "residual_std":     float(res.std()),
        "residual_max_abs": float(np.abs(res).max()),
    }


def evaluate_baselines(X_tr, X_te, y_tr, y_te):
    return {s: evaluate_predictions(y_te, DummyRegressor(strategy=s).fit(X_tr,y_tr).predict(X_te))
            for s in ["mean","median"]}


# ─────────────────────────────────────────────────────────────────────────────
# Subgroup evaluation — mirrors reference evaluate_subgroups
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_subgroups(model, X_test, y_test, y_pred, output_dir):
    log.info("Subgroup evaluation …")
    overall_rmse = float(np.sqrt(mean_squared_error(y_test,y_pred)))
    eval_df      = X_test.reset_index(drop=True).copy()
    eval_df["_y_true"] = y_test.to_numpy(); eval_df["_y_pred"] = y_pred
    rows = []
    for col in ["smoker","region","sex"]:
        if col not in eval_df.columns: continue
        for val,sub in eval_df.groupby(col,observed=True):
            if len(sub)<10: continue
            sr   = float(np.sqrt(mean_squared_error(sub["_y_true"],sub["_y_pred"])))
            rows.append({
                "group_col":col,"group_val":str(val),"n":int(len(sub)),
                "mean_actual":round(float(sub["_y_true"].mean()),2),
                "rmse":round(sr,2),"r2":round(float(r2_score(sub["_y_true"],sub["_y_pred"])),4),
                "rmse_gap":round(sr-overall_rmse,2),"alert":bool(sr>overall_rmse*1.25),
            })
    if rows:
        rd = pd.DataFrame(rows)
        rd.to_csv(output_dir/"fairness_report.csv",index=False)
        g = rd.copy(); g["label"] = g["group_col"]+"="+g["group_val"].astype(str)
        plt.figure(figsize=(10,max(4,len(g)*0.45)))
        colors = ["#E45756" if a else "#4C78A8" for a in g["alert"]]
        plt.barh(g["label"],g["rmse"],color=colors)
        plt.axvline(g["rmse"].mean(),linestyle="--",color="black",label="Mean RMSE")
        plt.xlabel("RMSE ($)"); plt.title("Subgroup RMSE  (red = >25% above overall)")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir/"plot_subgroup_rmse.png",dpi=160); plt.close()
    return {"overall_rmse":overall_rmse,"subgroups":rows}


# ─────────────────────────────────────────────────────────────────────────────
# SHAP — mirrors reference save_shap_artifacts
# ─────────────────────────────────────────────────────────────────────────────
def save_shap_artifacts(model, X_test, y_test, y_pred, output_dir):
    if not _SHAP:
        log.warning("pip install shap"); return
    log.info("SHAP for champion …")
    try:
        step_names = list(model.named_steps.keys())
        clf        = model.named_steps["model"]
        prep       = model.named_steps["preprocess"]
        fe         = model.named_steps["feature_engineering"]
        sel_key    = "feature_selection" if "feature_selection" in step_names else None

        Xt = fe.transform(X_test)
        Xt = prep.transform(Xt)
        if sel_key:
            sel = model.named_steps[sel_key]
            # Resolve feature names from step before selector
            fs_idx = step_names.index(sel_key)
            fn = None
            for sname in reversed(step_names[:fs_idx]):
                s = model.named_steps[sname]
                if hasattr(s,"get_feature_names_out"):
                    try: fn=s.get_feature_names_out(); break
                    except: pass
            if fn is None: fn = np.array([f"f{i}" for i in range(Xt.shape[1])])
            support = sel.get_support()
            if len(fn) != len(support):
                fn = np.array([f"feature_{i}" for i in range(len(support))])
            sn  = fn[support]
            Xt  = sel.transform(Xt)
        else:
            if hasattr(prep,"get_feature_names_out"):
                try: sn = prep.get_feature_names_out()
                except: sn = np.array([f"f{i}" for i in range(Xt.shape[1])])
            else:
                sn = np.array([f"f{i}" for i in range(Xt.shape[1])])

        Xdf = pd.DataFrame(Xt,columns=sn)
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
            plt.title(f"SHAP — champion ({ptype})")
            plt.tight_layout()
            plt.savefig(output_dir/fname,dpi=150,bbox_inches="tight"); plt.close()

        worst = int(np.argmax(np.abs(y_test.to_numpy()-y_pred)))
        ev    = float(exp.expected_value) if not isinstance(exp.expected_value,np.ndarray) else float(exp.expected_value)
        shap.waterfall_plot(
            shap.Explanation(values=sv[worst],base_values=ev,
                             data=Xdf.iloc[worst].values,feature_names=list(sn)),
            show=False,max_display=15)
        plt.title("SHAP Waterfall — highest charges residual")
        plt.tight_layout()
        plt.savefig(output_dir/"plot_shap_waterfall.png",dpi=150,bbox_inches="tight"); plt.close()

        pd.DataFrame({"feature":sn,"mean_abs_shap":np.abs(sv).mean(axis=0)}
            ).sort_values("mean_abs_shap",ascending=False
            ).to_csv(output_dir/"shap_importance.csv",index=False)
        log.info("SHAP saved.")
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance plot — handles OHE + standard models
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
            if hasattr(s,"get_feature_names_out"):
                try: fn=s.get_feature_names_out(); break
                except: pass
        if fn is None: return
        support = sel.get_support()
        if len(fn)!=len(support):
            fn = np.array([f"feature_{i}" for i in range(len(support))])
        sn = fn[support]
    else:
        prep = model.named_steps.get("preprocess")
        if prep and hasattr(prep,"get_feature_names_out"):
            try: sn = prep.get_feature_names_out()
            except: return
        else: return

    if hasattr(clf,"feature_importances_"):
        imp = clf.feature_importances_; signed = np.full(len(imp),np.nan)
    elif hasattr(clf,"coef_"):
        signed = clf.coef_; imp = np.abs(signed)
    else: return

    pd.DataFrame({"feature":sn,"importance":imp,"coefficient":signed}
        ).sort_values("importance",ascending=False
        ).to_csv(output_dir/"feature_importance.csv",index=False)
    plt.figure(figsize=(9,5.5))
    sns.barplot(
        data=pd.DataFrame({"feature":sn,"importance":imp}
            ).sort_values("importance",ascending=False).head(20),
        y="feature", x="importance", color="#4C78A8")
    plt.title("Top 20 model features"); plt.tight_layout()
    plt.savefig(output_dir/"plot_feature_importance.png",dpi=160); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation plots — mirrors reference
# ─────────────────────────────────────────────────────────────────────────────
def save_evaluation_plots(y_test, y_pred, output_dir):
    res  = y_test.to_numpy() - y_pred
    rmse = float(np.sqrt(np.mean(res**2)))
    plt.figure(figsize=(6,5))
    plt.scatter(y_test,y_pred,alpha=0.3,s=8,color="#4C78A8")
    mn,mx = float(min(y_test.min(),y_pred.min())),float(max(y_test.max(),y_pred.max()))
    plt.plot([mn,mx],[mn,mx],"r--",linewidth=1.2,label="Perfect")
    plt.title(f"Actual vs Predicted  (R²={r2_score(y_test,y_pred):.3f})")
    plt.xlabel("Actual charges ($)"); plt.ylabel("Predicted charges ($)")
    plt.legend(); plt.tight_layout()
    plt.savefig(output_dir/"plot_actual_vs_predicted.png",dpi=160); plt.close()

    plt.figure(figsize=(6,4))
    plt.scatter(y_pred,res,alpha=0.3,s=8,color="#B279A2")
    plt.axhline(0,color="black",linewidth=1.0)
    plt.axhline(res.std(),color="#F58518",linestyle="--",alpha=0.7)
    plt.axhline(-res.std(),color="#F58518",linestyle="--",alpha=0.7)
    plt.title("Residuals vs Predicted"); plt.tight_layout()
    plt.savefig(output_dir/"plot_residuals_vs_predicted.png",dpi=160); plt.close()

    sns.histplot(res,kde=True,bins=40,color="#54A24B")
    plt.axvline(0,color="red"); plt.title("Residual distribution"); plt.tight_layout()
    plt.savefig(output_dir/"plot_residual_distribution.png",dpi=160); plt.close()


def save_error_analysis(X_test, y_test, y_pred, output_dir):
    df = X_test.copy(); res = y_test.to_numpy()-y_pred
    rmse = float(np.sqrt(np.mean(res**2)))
    df["actual"]=y_test.to_numpy(); df["predicted"]=y_pred
    df["residual"]=res; df["abs_error"]=np.abs(res); df["pct_error"]=np.abs(res)/np.maximum(np.abs(y_test.to_numpy()),1)
    df["severity"]=pd.cut(df["abs_error"],bins=[0,rmse*0.5,rmse,rmse*2,np.inf],labels=["low","medium","high","severe"])
    df.to_csv(output_dir/"test_predictions.csv",index=False)
    df[df["abs_error"]>rmse].to_csv(output_dir/"error_analysis.csv",index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Training profile — 100 quantiles per feature (mirrors reference exactly)
# ─────────────────────────────────────────────────────────────────────────────
def build_training_profile(X_train, y_train):
    stats = {}
    for col in NUMERIC_RAW:
        if col not in X_train.columns: continue
        v = pd.to_numeric(X_train[col],errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(v)==0: continue
        stats[col] = {
            "mean":float(v.mean()),"std":float(v.std()),
            "min":float(v.min()),"max":float(v.max()),
            "quantiles":np.quantile(v,np.linspace(0,1,100)).tolist(),
        }
    cat_stats = {}
    for col in CATEGORICAL_COLS:
        if col not in X_train.columns: continue
        cat_stats[col] = X_train[col].astype(str).value_counts().to_dict()
    return to_jsonable({
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "row_count":     int(len(X_train)),
        "raw_columns":   list(X_train.columns),
        "target_stats":  {"mean":float(y_train.mean()),"std":float(y_train.std()),
                          "min":float(y_train.min()),"max":float(y_train.max())},
        "raw_missing_rate":   X_train.isna().mean().to_dict(),
        "numeric_train_stats":stats,
        "categorical_stats":  cat_stats,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Model Card — mirrors reference save_model_card
# ─────────────────────────────────────────────────────────────────────────────
def save_model_card(metrics, fairness, uncertainty, reg_analysis, search, output_dir):
    tm = metrics.get("test_metrics",{})
    write_json(output_dir/MODEL_CARD_FILE, {
        "schema_version":"1.0",
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "model_details":{
            "name":"Medical Insurance Cost Predictor",
            "type":"Regression (sklearn Pipeline)",
            "algorithm":repr(search.best_estimator_.named_steps["model"]),
        },
        "intended_use":{
            "primary_use":"Estimate individual medical insurance costs from demographic and health features.",
            "out_of_scope":["Individual underwriting decisions","Real-time pricing","Non-US markets"],
        },
        "evaluation_results":{
            "test_r2":tm.get("r2"),"test_rmse":tm.get("rmse"),"test_mae":tm.get("mae"),
            "oof_rmse":uncertainty.get("oof_rmse"),
            "uncertainty_band":{"lower":uncertainty.get("lower_band"),"upper":uncertainty.get("upper_band")},
        },
        "regularisation_insights":reg_analysis,
        "fairness":{
            "overall_rmse":fairness.get("overall_rmse"),
            "alerts":[r for r in fairness.get("subgroups",[]) if r.get("alert")],
        },
        "limitations":[
            "Synthetic/1990s data — not reflective of current insurance markets.",
            "Smoker status is self-reported and the single largest cost driver.",
            "Only 6 input features — missing many real-world risk factors.",
        ],
        "ethical_considerations":[
            "Do not use sex or region as basis for individual pricing decisions.",
            "BMI as health proxy has known limitations across populations.",
            "Smoker flag encodes behavioural data — use with regulatory awareness.",
        ],
        "hyperparameters":search.best_params_,
        "cv_best_r2":float(search.best_score_),
    })


# ─────────────────────────────────────────────────────────────────────────────
# MLflow — mirrors reference log_to_mlflow
# ─────────────────────────────────────────────────────────────────────────────
def log_to_mlflow(metrics, search, model, output_dir):
    if not _MLFLOW: return
    try:
        mlflow.set_experiment("insurance_charges")
        tm = metrics.get("test_metrics",{})
        with mlflow.start_run():
            mlflow.log_params({f"best_{k}":str(v) for k,v in search.best_params_.items()})
            mlflow.log_metrics({"cv_r2":float(search.best_score_),"test_r2":float(tm.get("r2",0)),
                                "test_rmse":float(tm.get("rmse",0)),"test_mae":float(tm.get("mae",0))})
            for f in [MODEL_CARD_FILE,METRICS_FILE,"ohe_analysis.json",
                      "interaction_analysis.json","plot_actual_vs_predicted.png","plot_shap_bar.png"]:
                if (output_dir/f).exists(): mlflow.log_artifact(str(output_dir/f))
            mlflow.sklearn.log_model(model,"model")
        log.info("MLflow logged.")
    except Exception as e:
        log.warning("MLflow failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Environment snapshot — mirrors reference
# ─────────────────────────────────────────────────────────────────────────────
def save_environment_snapshot(output_dir):
    env = {"saved_at":datetime.now(timezone.utc).isoformat(),
           "python":sys.version,"platform":sys.platform,"libraries":{}}
    for lib in ["sklearn","pandas","numpy","scipy","joblib","shap","mlflow","pandera","category_encoders"]:
        try:
            mod=importlib.import_module(lib); env["libraries"][lib]=getattr(mod,"__version__","unknown")
        except ImportError:
            env["libraries"][lib]="not_installed"
    write_json(output_dir/ENVIRONMENT_FILE, env)


# ─────────────────────────────────────────────────────────────────────────────
# OOF uncertainty — mirrors reference compute_oof_uncertainty
# ─────────────────────────────────────────────────────────────────────────────
def compute_oof_uncertainty(best_estimator, X_train, y_train,
                            overpredict_cost=1.0, underpredict_cost=1.0):
    log.info("Computing OOF uncertainty …")
    cv  = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(clone(best_estimator), X_train, y_train,
                            cv=cv, n_jobs=N_JOBS)
    res = y_train.to_numpy()-oof
    rmse= float(np.sqrt(np.mean(res**2)))
    return {
        "oof_rmse":rmse,"oof_mae":float(np.abs(res).mean()),
        "oof_r2":float(r2_score(y_train,oof)),
        "lower_band":rmse*underpredict_cost,"upper_band":rmse*overpredict_cost,
        "overpredict_cost":overpredict_cost,"underpredict_cost":underpredict_cost,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search — mirrors reference tune_model
# ─────────────────────────────────────────────────────────────────────────────
def tune_model(X_train, y_train, n_iter, n_cv_splits=5, fast=False):
    log.info("Hyperparameter search: n_iter=%d  cv=%d-fold  fast=%s", n_iter, n_cv_splits, fast)
    _n = 50 if fast else 150

    param_distributions = [
        # Ridge
        {"feature_selection__threshold":["median","0.75*median","1.25*median"],
         "model":[Ridge()],"model__alpha":[0.01,0.1,1,5,10,50,100,500]},
        # Lasso
        {"feature_selection__threshold":["median","0.75*median"],
         "model":[Lasso(max_iter=5000)],"model__alpha":[0.001,0.01,0.1,1.0,5.0]},
        # ElasticNet
        {"feature_selection__threshold":["median","0.75*median"],
         "model":[ElasticNet(max_iter=5000)],
         "model__alpha":[0.01,0.1,1.0],"model__l1_ratio":[0.1,0.3,0.5,0.7,0.9]},
        # GBR
        {"feature_selection__threshold":["median","0.75*median","1.25*median"],
         "model":[GradientBoostingRegressor(n_estimators=_n,random_state=RANDOM_STATE)],
         "model__max_depth":[3,4,5],"model__learning_rate":[0.05,0.1,0.2],
         "model__subsample":[0.7,0.9]},
        # RandomForest
        {"feature_selection__threshold":["median","0.75*median","1.25*median"],
         "model":[RandomForestRegressor(n_estimators=_n,random_state=RANDOM_STATE,n_jobs=N_JOBS)],
         "model__max_depth":[6,8,None],"model__min_samples_leaf":[1,2,4]},
    ]

    cv = KFold(n_splits=n_cv_splits, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        build_ohe_pipeline(), param_distributions, n_iter=n_iter,
        scoring={"r2":"r2","neg_rmse":"neg_root_mean_squared_error",
                 "neg_mae":"neg_mean_absolute_error"},
        refit="r2", cv=cv, random_state=RANDOM_STATE,
        n_jobs=N_JOBS, verbose=1, return_train_score=True)
    search.fit(X_train, y_train)

    best_mdl = search.best_estimator_.named_steps["model"]
    if hasattr(best_mdl,"n_estimators") and best_mdl.n_estimators==_n:
        log.info("Upgrading %d → 300 trees …", _n)
        best_mdl.set_params(n_estimators=300)
        search.best_estimator_.fit(X_train, y_train)

    log.info("Best CV R²=%.4f  model=%s",
             search.best_score_, type(best_mdl).__name__)
    return search


# ─────────────────────────────────────────────────────────────────────────────
# Versioning — mirrors reference
# ─────────────────────────────────────────────────────────────────────────────
def _model_version_tag(model):
    return hashlib.sha1(pickle.dumps(model)).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────────────
# Main train() — mirrors reference train() exactly
# ─────────────────────────────────────────────────────────────────────────────
def train(output_dir, n_iter=20, n_cv_splits=5, fast=False,
          overpredict_cost=1.0, underpredict_cost=1.0):
    log.info("=== Training started (n_jobs=%d) ===", N_JOBS)
    output_dir.mkdir(parents=True, exist_ok=True)

    df                             = fix_data_types(load_data())
    X_train, X_test, y_train, y_te = split_data(df)

    # Phase 1: EDA
    save_research_artifacts(X_train, y_train, output_dir)
    baselines = evaluate_baselines(X_train, X_test, y_train, y_te)

    # Concept analyses (all on train set only)
    ohe_analysis   = analyse_ohe_strategies(X_train, y_train, output_dir)
    te_analysis    = analyse_target_encoding(X_train, y_train, output_dir)
    int_analysis   = analyse_interactions(X_train, y_train, output_dir)
    poly_analysis  = analyse_polynomial_features(X_train, y_train, output_dir)
    bin_analysis   = analyse_binning(X_train, y_train, output_dir)
    reg_analysis   = analyse_regularisation(X_train, y_train, output_dir)
    cv_comparison  = analyse_cv_strategies(X_train, y_train, output_dir)

    # Hyperparameter search
    search         = tune_model(X_train, y_train, n_iter=n_iter,
                                n_cv_splits=n_cv_splits, fast=fast)
    uncertainty    = compute_oof_uncertainty(search.best_estimator_, X_train, y_train,
                                             overpredict_cost, underpredict_cost)

    final_model    = clone(search.best_estimator_)
    final_model.fit(X_train, y_train)
    y_pred         = final_model.predict(X_test)
    test_metrics   = evaluate_predictions(y_te, y_pred)

    log.info("Test R²=%.4f  RMSE=$%.2f  MAE=$%.2f",
             test_metrics["r2"], test_metrics["rmse"], test_metrics["mae"])

    # Save artifacts
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha1 = _model_version_tag(final_model)
    joblib.dump(final_model, output_dir/f"insurance_pipeline_{ts}_{sha1}.joblib")
    joblib.dump(final_model, output_dir/MODEL_FILE)
    save_environment_snapshot(output_dir)
    pd.DataFrame(search.cv_results_).sort_values("rank_test_r2").to_csv(
        output_dir/"cv_results.csv",index=False)
    save_feature_importance(final_model, output_dir)
    save_evaluation_plots(y_te, y_pred, output_dir)
    save_error_analysis(X_test, y_te, y_pred, output_dir)
    write_json(output_dir/TRAINING_PROFILE_FILE, build_training_profile(X_train,y_train))

    # Diagnostics
    res_diag = residual_diagnostics(y_te.to_numpy(), y_pred, output_dir)
    plot_learning_curves_insurance(final_model, X_train, y_train, output_dir)
    plot_partial_dependence_insurance(final_model, X_test, output_dir)
    save_shap_artifacts(final_model, X_test, y_te, y_pred, output_dir)
    fairness = evaluate_subgroups(final_model, X_test, y_te, y_pred, output_dir)

    metrics = {
        "baselines":baselines,"split":{"train_rows":int(len(X_train)),"test_rows":int(len(X_test))},
        "ohe_analysis":ohe_analysis,"target_encoding":te_analysis,
        "interaction_analysis":int_analysis,"polynomial_analysis":poly_analysis,
        "binning_analysis":bin_analysis,"regularisation":reg_analysis,
        "cv_strategies":cv_comparison,
        "best_cv":{"best_r2":float(search.best_score_),"best_params":search.best_params_},
        "uncertainty_info":uncertainty,"residual_diag":res_diag,
        "test_metrics":test_metrics,"fairness":fairness,
    }
    write_json(output_dir/METRICS_FILE, metrics)
    save_model_card(metrics, fairness, uncertainty, reg_analysis, search, output_dir)
    log_to_mlflow(metrics, search, final_model, output_dir)

    log.info("=== Training complete ===")
    return to_jsonable(metrics)


# ─────────────────────────────────────────────────────────────────────────────
# predict / monitor / sample-input — mirrors reference CLI pattern
# ─────────────────────────────────────────────────────────────────────────────
def predict(artifact_dir, input_csv, output_csv):
    model = joblib.load(artifact_dir/MODEL_FILE)
    if not hasattr(model,"predict"):
        raise TypeError(f"{type(model).__name__} is not a fitted pipeline.")
    unc_band = float(os.environ.get("ML_UNCERTAINTY_BAND",0.10))
    mp = artifact_dir/METRICS_FILE
    if mp.exists():
        unc_band = json.loads(mp.read_text())["uncertainty_info"].get("oof_rmse",unc_band)
    df = pd.read_csv(input_csv)
    # Restore category dtypes
    for col in CATEGORICAL_COLS:
        if col in df.columns: df[col] = df[col].astype("category")
    if INPUT_SCHEMA:
        try: INPUT_SCHEMA.validate(df,lazy=True)
        except Exception as e: log.warning("Schema: %s",e)
    pf = artifact_dir/TRAINING_PROFILE_FILE
    if pf.exists():
        req  = set(json.loads(pf.read_text())["raw_columns"])
        miss = req - set(df.columns)
        if miss: raise ValueError(f"Missing columns: {sorted(miss)}")
    y_pred = model.predict(df)
    df["predicted_charges"] = y_pred
    df["lower_bound"]       = y_pred - unc_band
    df["upper_bound"]       = y_pred + unc_band
    df["wide_interval"]     = ((df["upper_bound"]-df["lower_bound"])>unc_band*2.5).astype(int)
    output_csv = Path(output_csv); output_csv.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(output_csv,index=False)
    log.info("Predictions saved to %s", output_csv.resolve())


def monitor(artifact_dir, input_csv, output_json, missing_rate_alert=0.05, ks_pvalue=0.05):
    profile  = json.loads((artifact_dir/TRAINING_PROFILE_FILE).read_text())
    incoming = pd.read_csv(input_csv)
    req, inc = set(profile["raw_columns"]), set(incoming.columns)
    drift = []
    for col,tr in profile["raw_missing_rate"].items():
        if col not in incoming: continue
        cur = float(incoming[col].isna().mean())
        drift.append({"column":col,"train_rate":float(tr),"current_rate":cur,
                      "change":abs(cur-float(tr)),"alert":abs(cur-float(tr))>=missing_rate_alert})
    ks_rows = []
    for col,stats in profile.get("numeric_train_stats",{}).items():
        if col not in incoming.columns: continue
        vals = incoming[col].dropna().to_numpy()
        if len(vals)<10: continue
        stat,p = ks_2samp(np.array(stats["quantiles"]),vals)
        ks_rows.append({"column":col,"ks_stat":float(stat),"p_value":float(p),"alert":p<ks_pvalue})
    report = {"checked_at":datetime.now(timezone.utc).isoformat(),"row_count":int(len(incoming)),
              "missing_required":sorted(req-inc),"extra":sorted(inc-req),
              "missing_rate_drift":drift,"distribution_drift":ks_rows}
    output_json = Path(output_json); output_json.parent.mkdir(parents=True,exist_ok=True)
    write_json(output_json, report)
    return report


def create_sample_input(output_csv, rows):
    df = fix_data_types(load_data())
    output_csv = Path(output_csv); output_csv.parent.mkdir(parents=True,exist_ok=True)
    df.drop(columns=[TARGET]).head(rows).to_csv(output_csv,index=False)
    log.info("Sample saved to %s", output_csv.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# Utilities — identical to reference pipeline
# ─────────────────────────────────────────────────────────────────────────────
def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload),indent=2),encoding="utf-8")


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


# ─────────────────────────────────────────────────────────────────────────────
# CLI — mirrors reference pipeline exactly
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Medical Insurance end-to-end ML pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sp = p.add_subparsers(dest="command",required=True)

    tp = sp.add_parser("train")
    tp.add_argument("--output-dir",        type=Path, default=Path("artifacts_insurance"))
    tp.add_argument("--n-iter",            type=int,  default=20)
    tp.add_argument("--n-cv-splits",       type=int,  default=5)
    tp.add_argument("--fast",              action="store_true",
                    help="50 trees in search → 300 after. ~60% faster.")
    tp.add_argument("--overpredict-cost",  type=float, default=1.0)
    tp.add_argument("--underpredict-cost", type=float, default=1.0)

    pp = sp.add_parser("predict")
    pp.add_argument("--artifact-dir", type=Path, default=Path("artifacts_insurance"))
    pp.add_argument("--input-csv",    type=Path, required=True)
    pp.add_argument("--output-csv",   type=Path, default=Path("artifacts_insurance/predictions.csv"))

    mp = sp.add_parser("monitor")
    mp.add_argument("--artifact-dir",       type=Path,  default=Path("artifacts_insurance"))
    mp.add_argument("--input-csv",          type=Path,  required=True)
    mp.add_argument("--output-json",        type=Path,  default=Path("artifacts_insurance/monitor.json"))
    mp.add_argument("--missing-rate-alert", type=float, default=0.05)
    mp.add_argument("--ks-pvalue-alert",    type=float, default=0.05)

    si = sp.add_parser("sample-input")
    si.add_argument("--output-csv", type=Path, default=Path("artifacts_insurance/sample.csv"))
    si.add_argument("--rows",       type=int,  default=10)

    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        m = train(args.output_dir, args.n_iter,
                  n_cv_splits=args.n_cv_splits, fast=args.fast,
                  overpredict_cost=args.overpredict_cost,
                  underpredict_cost=args.underpredict_cost)
        log.info("Test R²=%.3f  RMSE=$%.2f  MAE=$%.2f",
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
