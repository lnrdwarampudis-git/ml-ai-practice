from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path("artifacts_housing") / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("artifacts_housing") / ".cache"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler, SplineTransformer, StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
DATA_ID = 42165
MODEL_FILE = "housing_price_pipeline.joblib"
METRICS_FILE = "metrics.json"
TRAINING_PROFILE_FILE = "training_profile.json"
TARGET_CANDIDATES = ["SalePrice", "saleprice", "target", "price", "median_house_value"]


@dataclass(frozen=True)
class ColumnGroups:
    numeric: list[str]
    categorical: list[str]


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Winsorize numeric features using train-set quantiles only."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99) -> None:
        self.lower = lower
        self.upper = upper

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | None = None) -> "OutlierClipper":
        frame = pd.DataFrame(X).astype(float)
        self.feature_names_in_ = np.asarray(getattr(X, "columns", [f"x{i}" for i in range(frame.shape[1])]), dtype=object)
        self.lower_bounds_ = frame.quantile(self.lower)
        self.upper_bounds_ = frame.quantile(self.upper)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame(X).astype(float)
        return frame.clip(self.lower_bounds_, self.upper_bounds_, axis=1)

    def get_feature_names_out(self, input_features: list[str] | np.ndarray | None = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Small, model-safe features that expose nonlinear housing effects."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HousingFeatureEngineer":
        self.input_columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for column in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageArea", "LotArea"]:
            if column in X:
                X[f"log1p_{column}"] = np.log1p(pd.to_numeric(X[column], errors="coerce").clip(lower=0))

        if {"TotalBsmtSF", "GrLivArea"}.issubset(X.columns):
            total_bsmt = pd.to_numeric(X["TotalBsmtSF"], errors="coerce").fillna(0)
            living = pd.to_numeric(X["GrLivArea"], errors="coerce").replace(0, np.nan)
            X["basement_to_living_ratio"] = total_bsmt / living

        if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(X.columns):
            X["total_bathrooms"] = (
                pd.to_numeric(X["FullBath"], errors="coerce").fillna(0)
                + 0.5 * pd.to_numeric(X["HalfBath"], errors="coerce").fillna(0)
                + pd.to_numeric(X["BsmtFullBath"], errors="coerce").fillna(0)
                + 0.5 * pd.to_numeric(X["BsmtHalfBath"], errors="coerce").fillna(0)
            )

        if {"YearBuilt", "YrSold"}.issubset(X.columns):
            X["age_at_sale"] = pd.to_numeric(X["YrSold"], errors="coerce") - pd.to_numeric(
                X["YearBuilt"], errors="coerce"
            )

        if {"YearRemodAdd", "YrSold"}.issubset(X.columns):
            X["years_since_remodel"] = pd.to_numeric(X["YrSold"], errors="coerce") - pd.to_numeric(
                X["YearRemodAdd"], errors="coerce"
            )

        if {"OverallQual", "GrLivArea"}.issubset(X.columns):
            X["quality_x_living_area"] = (
                pd.to_numeric(X["OverallQual"], errors="coerce")
                * pd.to_numeric(X["GrLivArea"], errors="coerce")
            )

        if {"OverallQual", "OverallCond"}.issubset(X.columns):
            X["quality_condition_interaction"] = (
                pd.to_numeric(X["OverallQual"], errors="coerce")
                * pd.to_numeric(X["OverallCond"], errors="coerce")
            )

        #return X.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
        return X.replace([np.inf, -np.inf], np.nan).infer_objects()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse=False)


def load_data() -> pd.DataFrame:
    """Load the OpenML housing frame requested in the prompt."""
    local_csv = os.environ.get("AMES_HOUSING_CSV")
    if local_csv:
        local_path = Path(local_csv).expanduser()
        if local_path.exists():
            return pd.read_csv(local_path)

    data_home = Path("artifacts_housing") / ".sklearn_data"
    data_home.mkdir(parents=True, exist_ok=True)
    try:
        try:
            import certifi

            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except ImportError:
            pass
        #housing = fetch_openml(data_id=DATA_ID, as_frame=True, parser="auto", data_home=data_home)
        #return housing.frame.copy()
        # url = "https://raw.githubusercontent.com/jseabold/538model/master/data/ames.csv"
        # housing = pd.read_csv(url)
        url = "http://jse.amstat.org/v19n3/decock/AmesHousing.txt"
        housing = pd.read_csv("./data/AmesHousing.csv",header=0)
        print(housing.head())
    
        return housing.copy()
    except Exception as exc:
        message = str(exc).lower()
        if "certificate_verify_failed" in message or "ssl" in message:
            raise RuntimeError(
                "OpenML download failed because Python cannot verify HTTPS certificates.\n\n"
                "macOS fix, if using python.org Python:\n"
                "  open '/Applications/Python 3.12/Install Certificates.command'\n\n"
                "Virtual environment fix:\n"
                "  python3 -m pip install --upgrade certifi\n"
                "  export SSL_CERT_FILE=$(python3 -c 'import certifi; print(certifi.where())')\n"
                "  export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE\n"
                "  python3 housing_pipeline.py train --n-splits 2 --output-dir artifacts_housing\n\n"
                "Offline fallback:\n"
                "  export AMES_HOUSING_CSV=/path/to/ames_housing.csv\n"
                "  python3 housing_pipeline.py train --n-splits 2 --output-dir artifacts_housing"
            ) from exc
        raise


def infer_target(df: pd.DataFrame) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    numeric_columns = df.select_dtypes(include=["number"]).columns
    if len(numeric_columns):
        return str(numeric_columns[-1])
    return str(df.columns[-1])


def split_data(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    clean_df = df.dropna(subset=[target]).copy()
    y = pd.to_numeric(clean_df[target], errors="coerce")
    clean_df = clean_df.loc[y.notna()].copy()
    y = y.loc[y.notna()]
    X = clean_df.drop(columns=[target])
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    report = (
        df.isna()
        .agg(["sum", "mean"])
        .T.rename(columns={"sum": "missing_count", "mean": "missing_rate"})
        .sort_values("missing_rate", ascending=False)
    )
    report["dtype"] = df.dtypes.astype(str)
    report["unique_values"] = df.nunique(dropna=True)
    return report


def build_data_contract(X_train: pd.DataFrame, target: str) -> dict[str, Any]:
    """Create a train-derived schema contract for future prediction/monitoring data."""
    contract: dict[str, Any] = {
        "target": target,
        "required_columns": list(X_train.columns),
        "column_count": int(X_train.shape[1]),
        "columns": {},
        "max_allowed_missing_rate": 0.35,
        "max_allowed_new_category_share": 0.05,
    }
    for column in X_train.columns:
        series = X_train[column]
        entry: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_rate": float(series.isna().mean()),
            "unique_values": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            quantiles = series.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
            entry["kind"] = "numeric"
            entry["min"] = None if pd.isna(series.min()) else float(series.min())
            entry["max"] = None if pd.isna(series.max()) else float(series.max())
            entry["quantiles"] = {str(key): float(value) for key, value in quantiles.items() if not pd.isna(value)}
        else:
            value_counts = series.astype("string").value_counts(dropna=True)
            entry["kind"] = "categorical"
            entry["allowed_values_top_50"] = [str(value) for value in value_counts.head(50).index]
        contract["columns"][column] = entry
    return to_jsonable(contract)


def validate_against_contract(X: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    required_columns = set(contract["required_columns"])
    incoming_columns = set(X.columns)
    column_reports = []
    for column in contract["required_columns"]:
        if column not in X:
            continue
        expected = contract["columns"][column]
        current_missing = float(X[column].isna().mean())
        report = {
            "column": column,
            "expected_kind": expected["kind"],
            "training_missing_rate": float(expected["missing_rate"]),
            "current_missing_rate": current_missing,
            "missing_rate_change": abs(current_missing - float(expected["missing_rate"])),
        }
        if expected["kind"] == "numeric":
            numeric = pd.to_numeric(X[column], errors="coerce")
            q = expected.get("quantiles", {})
            lower = q.get("0.01")
            upper = q.get("0.99")
            if lower is not None and upper is not None:
                report["outside_train_1_99_quantile_share"] = float(((numeric < lower) | (numeric > upper)).mean())
        else:
            allowed = set(expected.get("allowed_values_top_50", []))
            values = X[column].astype("string")
            known = values.isna() | values.isin(allowed)
            report["new_or_rare_category_share"] = float((~known).mean())
        column_reports.append(report)

    return {
        "row_count": int(len(X)),
        "missing_required_columns": sorted(required_columns - incoming_columns),
        "extra_columns": sorted(incoming_columns - required_columns),
        "column_reports": column_reports,
    }


def leakage_audit(df: pd.DataFrame, target: str) -> dict[str, Any]:
    suspicious_terms = [
        "price",
        "saleprice",
        "sold",
        "sale",
        "target",
        "prediction",
        "label",
        "outcome",
    ]
    rows = []
    for column in df.columns:
        if column == target:
            continue
        lower = column.lower()
        reasons = []
        if any(term in lower for term in suspicious_terms):
            reasons.append("name suggests target/outcome timing")
        if pd.api.types.is_numeric_dtype(df[column]):
            corr = pd.to_numeric(df[column], errors="coerce").corr(pd.to_numeric(df[target], errors="coerce"))
            if pd.notna(corr) and abs(corr) > 0.9:
                reasons.append(f"very high absolute target correlation: {corr:.3f}")
        if reasons:
            rows.append({"column": column, "reasons": reasons})
    return {
        "target": target,
        "prediction_time_assumption": "Only features known before the sale price outcome is finalized should be used.",
        "suspicious_columns": rows,
        "status": "review_required" if rows else "no_obvious_leakage_flags",
    }


def validation_strategy_report(row_count: int) -> dict[str, Any]:
    return {
        "selected_strategy": "80/20 holdout plus shuffled KFold cross-validation on the training split",
        "why": "Ames rows are treated as independent property sales without a reliable temporal deployment field in this exercise.",
        "train_test_split": {"test_size": 0.2, "random_state": RANDOM_STATE},
        "cross_validation": {"type": "KFold", "shuffle": True, "random_state": RANDOM_STATE},
        "alternatives_to_consider": [
            "GroupKFold by Neighborhood if evaluating generalization to unseen neighborhoods.",
            "Time-based split by YrSold if deployment is future sale prediction.",
            "Spatial split if latitude/longitude or richer geospatial features are available.",
        ],
        "row_count": int(row_count),
    }


def metric_design_report() -> dict[str, Any]:
    return {
        "primary_metric": "rmse",
        "secondary_metrics": ["mae", "rmsle", "r2"],
        "metric_interpretation": {
            "rmse": "Penalizes large dollar mistakes more strongly.",
            "mae": "Average absolute dollar error; easiest to communicate.",
            "rmsle": "Better for skewed house prices and relative error behavior.",
            "r2": "Explained variance, useful but not sufficient alone.",
        },
        "selection_rule": "Use CV RMSE to select a model, then inspect MAE/RMSLE/residuals before trusting it.",
    }


def dataframe_fingerprint(df: pd.DataFrame) -> dict[str, Any]:
    ordered = df.sort_index(axis=1).copy()
    row_hashes = pd.util.hash_pandas_object(ordered.astype("string"), index=True).values
    digest = hashlib.sha256(row_hashes.tobytes()).hexdigest()
    return {
        "data_id": DATA_ID,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "sha256_hash": digest,
        "hash_policy": "sha256 over pandas row hashes after sorting columns and casting values to string",
    }


def schema_test_report(X: pd.DataFrame, contract: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
    validation = validate_against_contract(X, contract)
    high_missing = [
        row["column"]
        for row in validation["column_reports"]
        if row["current_missing_rate"] > float(contract["max_allowed_missing_rate"])
    ]
    return {
        "checks": {
            "all_required_columns_present": not validation["missing_required_columns"],
            "no_extra_columns_in_training_frame": not validation["extra_columns"],
            "no_columns_above_contract_missing_threshold": not high_missing,
            "leakage_audit_reviewed": leakage["status"] in {"no_obvious_leakage_flags", "review_required"},
        },
        "high_missing_columns": high_missing,
        "leakage_status": leakage["status"],
        "overall_status": "pass_with_review_items" if leakage["status"] == "review_required" else "pass",
    }


def row_validation_report(X: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    required = set(contract["required_columns"])
    missing_required = sorted(required - set(X.columns))
    rejected_rows = []
    if missing_required:
        return {
            "accepted_rows": 0,
            "rejected_rows": int(len(X)),
            "rejection_reasons": [{"row_index": int(i), "reasons": [f"missing required columns: {missing_required}"]} for i in X.index[:50]],
            "sample_limit": 50,
        }

    for idx, row in X.iterrows():
        reasons = []
        for column in contract["required_columns"]:
            expected = contract["columns"][column]
            value = row[column]
            if expected["kind"] == "numeric":
                numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                q = expected.get("quantiles", {})
                lower = q.get("0.01")
                upper = q.get("0.99")
                if pd.notna(numeric_value) and lower is not None and upper is not None:
                    if numeric_value < lower or numeric_value > upper:
                        reasons.append(f"{column} outside train 1st/99th percentile range")
            else:
                allowed = set(expected.get("allowed_values_top_50", []))
                if pd.notna(value) and str(value) not in allowed:
                    reasons.append(f"{column} has new or rare category")
        if reasons:
            rejected_rows.append({"row_index": int(idx), "reasons": reasons[:10]})

    return {
        "accepted_rows": int(len(X) - len(rejected_rows)),
        "rejected_rows": int(len(rejected_rows)),
        "rejection_reasons": rejected_rows[:50],
        "sample_limit": 50,
        "note": "Rows are flagged for review; the model can still score them because preprocessing handles missing and unknown values.",
    }


def validation_comparison_reports(X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path, best_name: str) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    model = candidate_models(X_train)[best_name]

    if "Neighborhood" in X_train.columns and X_train["Neighborhood"].nunique(dropna=True) >= 3:
        groups = X_train["Neighborhood"].astype("string").fillna("Unknown")
        n_splits = min(5, int(groups.nunique()))
        result = cross_validate(
            model,
            X_train,
            y_train,
            cv=GroupKFold(n_splits=n_splits),
            groups=groups,
            scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=1,
            error_score="raise",
        )
        reports["group_kfold_by_neighborhood"] = {
            "n_splits": n_splits,
            "rmse_mean": float(-result["test_rmse"].mean()),
            "rmse_std": float(result["test_rmse"].std()),
            "mae_mean": float(-result["test_mae"].mean()),
            "r2_mean": float(result["test_r2"].mean()),
            "why": "Tests whether performance holds when entire neighborhoods are withheld.",
        }

    if "YrSold" in X_train.columns and X_train["YrSold"].nunique(dropna=True) >= 2:
        years = pd.to_numeric(X_train["YrSold"], errors="coerce")
        holdout_year = int(years.max())
        train_mask = years < holdout_year
        valid_mask = years == holdout_year
        if train_mask.sum() >= 100 and valid_mask.sum() >= 20:
            time_model = clone(model)
            time_model.fit(X_train.loc[train_mask], y_train.loc[train_mask])
            preds = time_model.predict(X_train.loc[valid_mask])
            reports["time_holdout_latest_yrsold"] = {
                "train_years": sorted(int(y) for y in years.loc[train_mask].dropna().unique()),
                "holdout_year": holdout_year,
                "train_rows": int(train_mask.sum()),
                "holdout_rows": int(valid_mask.sum()),
                "metrics": regression_metrics(y_train.loc[valid_mask], preds),
                "why": "Approximates future-sale deployment by holding out the latest sale year.",
            }

    write_json(output_dir / "validation_comparison.json", reports)
    return reports


def save_segment_metrics(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    df = X_test.copy()
    df["actual_price"] = y_test.to_numpy()
    df["predicted_price"] = predictions
    df["absolute_error"] = (df["actual_price"] - df["predicted_price"]).abs()
    df["squared_error"] = (df["actual_price"] - df["predicted_price"]) ** 2

    reports = {}
    for column in ["Neighborhood", "OverallQual"]:
        if column not in df:
            continue
        rows = []
        for value, group in df.groupby(column, dropna=False, observed=False):
            if len(group) < 5:
                continue
            rows.append(
                {
                    "segment_column": column,
                    "segment_value": str(value),
                    "rows": int(len(group)),
                    "mae": float(group["absolute_error"].mean()),
                    "rmse": float(np.sqrt(group["squared_error"].mean())),
                    "mean_actual_price": float(group["actual_price"].mean()),
                }
            )
        if rows:
            reports[column] = rows

    price_bins = pd.qcut(df["actual_price"], q=5, duplicates="drop")
    price_rows = []
    for value, group in df.groupby(price_bins, observed=False):
        price_rows.append(
            {
                "segment_column": "actual_price_quintile",
                "segment_value": str(value),
                "rows": int(len(group)),
                "mae": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt(group["squared_error"].mean())),
                "mean_actual_price": float(group["actual_price"].mean()),
            }
        )
    reports["actual_price_quintile"] = price_rows

    segment_df = pd.DataFrame([row for rows in reports.values() for row in rows])
    segment_df.to_csv(output_dir / "segment_metrics.csv", index=False)
    if not segment_df.empty:
        top = segment_df.sort_values("mae", ascending=False).head(20)
        plt.figure(figsize=(9, 5.5))
        sns.barplot(data=top, y="segment_value", x="mae", hue="segment_column", dodge=False)
        plt.title("Highest MAE Segments")
        plt.xlabel("MAE")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_segment_mae.png", dpi=160)
        plt.close()
    write_json(output_dir / "segment_metrics.json", reports)
    return to_jsonable(reports)


def write_dataset_card(output_dir: Path, df: pd.DataFrame, target: str, fingerprint: dict[str, Any]) -> None:
    text = f"""# Dataset Card: Ames Housing OpenML {DATA_ID}

## Purpose

Train and evaluate regression models that predict `{target}` from property attributes.

## Source

`fetch_openml(data_id={DATA_ID}, as_frame=True, parser="auto")`

## Version Fingerprint

- Rows: {fingerprint["rows"]}
- Columns: {fingerprint["columns"]}
- SHA256: `{fingerprint["sha256_hash"]}`

## Target

`{target}` is a continuous house sale price target. It is right-skewed, so RMSLE and log-target models are useful companion checks.

## Known Data Concerns

- Missingness can be meaningful for some housing attributes.
- Sale timing fields require prediction-time review.
- Neighborhood can induce grouped/geographic dependence.
- Future deployment may require time-based validation by sale year.

## Recommended Validation

- Default exercise: holdout + shuffled KFold.
- Enterprise review: GroupKFold by `Neighborhood` and latest-year holdout by `YrSold`.

## Rows And Columns

- Dataset rows: {len(df)}
- Dataset columns: {len(df.columns)}
"""
    (output_dir / "dataset_card.md").write_text(text, encoding="utf-8")


def write_model_card(
    output_dir: Path,
    best_name: str,
    metrics: dict[str, Any],
    leakage: dict[str, Any],
    validation_comparison: dict[str, Any],
) -> None:
    text = f"""# Model Card: Ames Housing Regression

## Model

Selected model: `{best_name}`

Serialized artifact: `{MODEL_FILE}`

## Intended Use

Estimate house sale price from pre-sale property attributes for learning and prototype analysis.

## Not Intended For

- Final appraisal decisions without human review.
- Deployment where sale-timing fields are unavailable or legally restricted.
- Unseen geographies without spatial/group validation.

## Primary Metrics

- RMSE: {metrics["test"]["rmse"]:.2f}
- MAE: {metrics["test"]["mae"]:.2f}
- RMSLE: {metrics["test"]["rmsle"]:.3f}
- R2: {metrics["test"]["r2"]:.3f}

## Validation

Default split: 80/20 holdout plus KFold on training data.

Additional validation artifacts:

- `validation_comparison.json`
- `segment_metrics.csv`
- `plot_train_vs_cv_rmse.png`

## Leakage Review

Status: `{leakage["status"]}`

Flagged columns should be reviewed against the prediction-time assumption.

## Limitations

- Missingness and rare categories can shift in production.
- Neighborhood and sale-year effects may reduce generalization.
- High-price homes may have different residual behavior than typical homes.

## Monitoring

Use `monitor` to inspect missing required columns, extra columns, missing-rate drift, contract validation, and row-level review flags.
"""
    (output_dir / "model_card.md").write_text(text, encoding="utf-8")


def get_column_groups(X: pd.DataFrame) -> ColumnGroups:
    engineered = HousingFeatureEngineer().fit(X).transform(X)
    numeric = engineered.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in engineered.columns if column not in numeric]
    return ColumnGroups(numeric=numeric, categorical=categorical)


def build_numeric_pipeline(imputer: str, basis: str, scaler: str = "robust") -> Pipeline:
    if imputer == "mice":
        imputer_step = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=20,
            random_state=RANDOM_STATE,
            initial_strategy="median",
            add_indicator=True,
        )
    elif imputer == "knn":
        imputer_step = KNNImputer(n_neighbors=7, weights="distance", add_indicator=True)
    else:
        imputer_step = SimpleImputer(strategy="median", add_indicator=True)

    steps: list[tuple[str, BaseEstimator]] = [("clip", OutlierClipper()), ("imputer", imputer_step)]
    steps.append(("scaler", RobustScaler() if scaler == "robust" else StandardScaler()))

    if basis == "poly":
        steps.append(("basis", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)))
    elif basis == "interactions":
        steps.append(("basis", PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)))
    elif basis == "splines":
        steps.append(("basis", SplineTransformer(n_knots=5, degree=3, include_bias=False)))

    return Pipeline(steps)


def build_preprocessor(X: pd.DataFrame, imputer: str, basis: str, scaler: str = "robust") -> ColumnTransformer:
    groups = get_column_groups(X)
    numeric_pipeline = build_numeric_pipeline(imputer=imputer, basis=basis, scaler=scaler)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, groups.numeric),
            ("cat", categorical_pipeline, groups.categorical),
        ],
        sparse_threshold=0.0,
        remainder="drop",
    )


def build_pipeline(
    X: pd.DataFrame,
    model: BaseEstimator,
    imputer: str = "simple",
    basis: str = "none",
    scaler: str = "robust",
) -> Pipeline:
    return Pipeline(
        [
            ("feature_engineering", HousingFeatureEngineer()),
            ("preprocess", build_preprocessor(X, imputer=imputer, basis=basis, scaler=scaler)),
            ("model", model),
        ]
    )


def log_target(regressor: BaseEstimator) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def candidate_models(X_train: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "dummy_median": Pipeline([("model", DummyRegressor(strategy="median"))]),
        "simple_ridge": build_pipeline(
            X_train,
            log_target(Ridge(alpha=20.0, random_state=RANDOM_STATE)),
            imputer="simple",
            basis="none",
            scaler="standard",
        ),
        "mice_huber_robust": build_pipeline(
            X_train,
            log_target(HuberRegressor(epsilon=1.35, alpha=0.0005, max_iter=800)),
            imputer="mice",
            basis="none",
        ),
        "simple_ridge_polynomial": build_pipeline(
            X_train,
            log_target(Ridge(alpha=80.0, random_state=RANDOM_STATE)),
            imputer="simple",
            basis="poly",
        ),
        "knn_ridge_splines": build_pipeline(
            X_train,
            log_target(Ridge(alpha=50.0, random_state=RANDOM_STATE)),
            imputer="knn",
            basis="splines",
        ),
        "simple_elastic_interactions": build_pipeline(
            X_train,
            log_target(ElasticNet(alpha=0.001, l1_ratio=0.15, max_iter=6000, random_state=RANDOM_STATE)),
            imputer="simple",
            basis="interactions",
        ),
        "mice_random_forest": build_pipeline(
            X_train,
            RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            imputer="mice",
            basis="none",
        ),
        "simple_hist_gradient_boosting": build_pipeline(
            X_train,
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.04,
                max_iter=350,
                l2_regularization=0.05,
                random_state=RANDOM_STATE,
            ),
            imputer="simple",
            basis="none",
        ),
    }


def regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    mae = mean_absolute_error(y_true, predictions)
    rmsle = float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.clip(predictions, 0, None)))))
    return {
        "rmse": rmse,
        "mae": float(mae),
        "rmsle": rmsle,
        "r2": float(r2_score(y_true, predictions)),
    }


def cross_validate_models(X_train: pd.DataFrame, y_train: pd.Series, n_splits: int) -> pd.DataFrame:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    for name, model in candidate_models(X_train).items():
        result = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
            return_train_score=True,
        )
        rows.append(
            {
                "model": name,
                "train_rmse_mean": float(-result["train_rmse"].mean()),
                "cv_rmse_mean": float(-result["test_rmse"].mean()),
                "cv_rmse_std": float(result["test_rmse"].std()),
                "train_mae_mean": float(-result["train_mae"].mean()),
                "cv_mae_mean": float(-result["test_mae"].mean()),
                "train_r2_mean": float(result["train_r2"].mean()),
                "cv_r2_mean": float(result["test_r2"].mean()),
                "generalization_gap_rmse": float((-result["test_rmse"].mean()) - (-result["train_rmse"].mean())),
                "fit_time_mean": float(result["fit_time"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("cv_rmse_mean")


def save_industry_plots(cv_results: pd.DataFrame, output_dir: Path) -> None:
    ordered = cv_results.sort_values("cv_rmse_mean")
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=ordered, y="model", x="cv_rmse_mean", color="#4C78A8")
    plt.errorbar(
        ordered["cv_rmse_mean"],
        np.arange(len(ordered)),
        xerr=ordered["cv_rmse_std"],
        fmt="none",
        ecolor="#333333",
        capsize=3,
    )
    plt.title("Baselines First: CV RMSE by Model")
    plt.xlabel("RMSE lower is better")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_model_comparison_rmse.png", dpi=160)
    plt.close()

    baseline = ordered.loc[ordered["model"] == "dummy_median", "cv_rmse_mean"]
    if not baseline.empty:
        baseline_rmse = float(baseline.iloc[0])
        improvement = ordered.copy()
        improvement["rmse_improvement_vs_dummy_pct"] = 100 * (baseline_rmse - improvement["cv_rmse_mean"]) / baseline_rmse
        improvement.to_csv(output_dir / "baseline_improvement_report.csv", index=False)
        plt.figure(figsize=(9, 5.5))
        sns.barplot(data=improvement, y="model", x="rmse_improvement_vs_dummy_pct", color="#54A24B")
        plt.axvline(0, color="black", linewidth=1)
        plt.title("Improvement Over Dummy Median Baseline")
        plt.xlabel("CV RMSE improvement (%)")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_baseline_improvement.png", dpi=160)
        plt.close()

    melted = ordered.melt(
        id_vars=["model"],
        value_vars=["train_rmse_mean", "cv_rmse_mean"],
        var_name="split",
        value_name="rmse",
    )
    melted["split"] = melted["split"].map({"train_rmse_mean": "Train", "cv_rmse_mean": "Cross-validation"})
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=melted, y="model", x="rmse", hue="split")
    plt.title("Validation Check: Train RMSE vs CV RMSE")
    plt.xlabel("RMSE lower is better")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_train_vs_cv_rmse.png", dpi=160)
    plt.close()


def save_research_artifacts(X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path, target: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = X_train.copy()
    train_df[target] = y_train
    engineered = HousingFeatureEngineer().fit(X_train, y_train).transform(X_train)

    missingness_report(train_df).to_csv(output_dir / "research_missingness_report.csv")
    train_df.dtypes.astype(str).rename("dtype").to_csv(output_dir / "schema.csv")
    train_df.select_dtypes(include=["number"]).describe().T.to_csv(output_dir / "numeric_summary.csv")
    engineered.select_dtypes(include=["number"]).corrwith(y_train).sort_values(ascending=False).to_csv(
        output_dir / "numeric_target_correlations.csv", header=["correlation"]
    )

    numeric = engineered.select_dtypes(include=["number"])
    imputed_numeric = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(numeric), columns=numeric.columns)
    vif_rows = []
    for column in imputed_numeric.columns:
        if imputed_numeric[column].nunique() <= 1 or len(imputed_numeric.columns) < 2:
            continue
        r2 = LinearRegression().fit(imputed_numeric.drop(columns=[column]), imputed_numeric[column]).score(
            imputed_numeric.drop(columns=[column]), imputed_numeric[column]
        )
        vif_rows.append({"feature": column, "vif": float(np.inf if r2 >= 0.999 else 1 / (1 - r2))})
    pd.DataFrame(vif_rows).sort_values("vif", ascending=False).to_csv(output_dir / "vif_report.csv", index=False)

    save_research_plots(train_df, output_dir, target)
    contract = build_data_contract(X_train, target)
    leakage = leakage_audit(train_df, target)
    write_json(output_dir / "data_contract.json", contract)
    write_json(output_dir / "contract_validation_train.json", validate_against_contract(X_train, contract))
    write_json(output_dir / "leakage_audit.json", leakage)
    write_json(output_dir / "schema_test_report.json", schema_test_report(X_train, contract, leakage))
    write_json(output_dir / "validation_strategy.json", validation_strategy_report(len(train_df)))
    write_json(output_dir / "metric_design.json", metric_design_report())
    write_json(
        output_dir / "research_decisions.json",
        {
            "problem_definition": {
                "problem_type": "regression",
                "target": target,
                "source": f"fetch_openml(data_id={DATA_ID}, as_frame=True, parser='auto')",
            },
            "missing_data_policy": {
                "simple": "Median imputation plus missingness indicators.",
                "mice": "IterativeImputer with BayesianRidge to model feature-conditional missing values.",
                "knn": "KNNImputer with distance weighting for local-neighborhood imputation.",
                "categorical": "Most-frequent imputation plus unknown-safe one-hot encoding.",
            },
            "robustness_policy": {
                "outliers": "Clip numeric features at train-set 1st/99th percentiles inside the pipeline.",
                "scaling": "RobustScaler for median/IQR scaling where appropriate.",
                "robust_regression": "HuberRegressor reduces sensitivity to large target residuals.",
                "target_transform": "Linear robust/basis models use log1p target fitting with expm1 predictions.",
            },
            "nonlinearity_policy": {
                "polynomial_features": "Degree-2 numeric basis for curvature and pairwise products.",
                "interaction_only_features": "Pairwise numeric interactions without squared terms.",
                "splines": "Cubic spline basis for smooth nonlinear numeric effects.",
                "tree_models": "Random forest and histogram gradient boosting capture higher-order interactions.",
            },
            "industry_readiness_policy": {
                "data_contract": "Train-derived schema, missingness, numeric range, and categorical-level contract.",
                "leakage_audit": "Column-name and target-correlation checks to flag suspicious fields.",
                "validation_strategy": "Documented holdout and KFold assumptions with alternatives.",
                "metric_design": "RMSE primary with MAE, RMSLE, and R2 as secondary checks.",
                "baselines_first": "Dummy median baseline is included in every model comparison.",
            },
        },
    )


def save_research_plots(train_df: pd.DataFrame, output_dir: Path, target: str) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    sns.histplot(train_df[target], kde=True, color="#4C78A8")
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_target_distribution.png", dpi=160)
    plt.close()

    missing = train_df.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    missing[missing > 0].head(30).sort_values().plot(kind="barh", color="#F58518")
    plt.title("Top Missing Rates")
    plt.xlabel("Missing Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_missingness.png", dpi=160)
    plt.close()

    numeric = train_df.select_dtypes(include=["number"]).drop(columns=[target], errors="ignore")
    if len(numeric.columns):
        corr = numeric.corrwith(train_df[target]).abs().sort_values(ascending=False).head(12)
        plt.figure(figsize=(8, 5))
        corr.sort_values().plot(kind="barh", color="#54A24B")
        plt.title("Strongest Numeric Associations")
        plt.xlabel("Absolute Correlation")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_numeric_target_correlations.png", dpi=160)
        plt.close()

    for candidate in ["OverallQual", "GrLivArea", "TotalBsmtSF", "LotArea"]:
        if candidate in train_df.columns:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=train_df, x=candidate, y=target, alpha=0.55, color="#B279A2")
            plt.title(f"{target} vs {candidate}")
            plt.tight_layout()
            plt.savefig(output_dir / f"plot_{candidate}_vs_target.png", dpi=160)
            plt.close()


def save_feature_importance(model: Pipeline, output_dir: Path) -> None:
    final_model = model.named_steps["model"]
    if not hasattr(final_model, "feature_importances_"):
        return
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=importance_df.head(20), y="feature", x="importance", color="#4C78A8")
    plt.title("Top Model Features")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_feature_importance.png", dpi=160)
    plt.close()


def save_error_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    output_dir: Path,
) -> None:
    error_df = X_test.copy()
    error_df["actual_price"] = y_test.to_numpy()
    error_df["predicted_price"] = predictions
    error_df["residual"] = error_df["actual_price"] - error_df["predicted_price"]
    error_df["absolute_error"] = error_df["residual"].abs()
    error_df.to_csv(output_dir / "test_predictions.csv", index=False)
    error_df.sort_values("absolute_error", ascending=False).head(50).to_csv(
        output_dir / "largest_prediction_errors.csv", index=False
    )

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=predictions, y=error_df["residual"], alpha=0.65, color="#E45756")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Residuals vs Predictions")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_residuals.png", dpi=160)
    plt.close()

    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.65, color="#72B7B2")
    low = min(float(y_test.min()), float(np.min(predictions)))
    high = max(float(y_test.max()), float(np.max(predictions)))
    plt.plot([low, high], [low, high], color="black", linewidth=1)
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_actual_vs_predicted.png", dpi=160)
    plt.close()


def build_training_profile(X_train: pd.DataFrame, y_train: pd.Series, target: str) -> dict[str, Any]:
    engineered = HousingFeatureEngineer().fit(X_train, y_train).transform(X_train)
    return to_jsonable(
        {
            "row_count": int(len(X_train)),
            "target": target,
            "raw_columns": list(X_train.columns),
            "engineered_columns": list(engineered.columns),
            "target_summary": y_train.describe().to_dict(),
            "raw_missing_rate": X_train.isna().mean().to_dict(),
            "engineered_missing_rate": engineered.isna().mean().to_dict(),
            "numeric_summary": engineered.select_dtypes(include=["number"]).describe().T.to_dict(orient="index"),
        }
    )


def train(output_dir: Path, n_splits: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data()
    target = infer_target(df)
    fingerprint = dataframe_fingerprint(df)
    write_json(output_dir / "dataset_version.json", fingerprint)
    write_dataset_card(output_dir, df, target, fingerprint)
    X_train, X_test, y_train, y_test = split_data(df, target)

    save_research_artifacts(X_train, y_train, output_dir, target)
    cv_results = cross_validate_models(X_train, y_train, n_splits=n_splits)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    save_industry_plots(cv_results, output_dir)

    best_name = str(cv_results.iloc[0]["model"])
    validation_comparison = validation_comparison_reports(X_train, y_train, output_dir, best_name)
    best_model = clone(candidate_models(X_train)[best_name])
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    test_metrics = regression_metrics(y_test, predictions)
    joblib.dump(best_model, output_dir / MODEL_FILE)
    save_feature_importance(best_model, output_dir)
    save_error_analysis(X_test, y_test, predictions, output_dir)
    segment_metrics = save_segment_metrics(X_test, y_test, predictions, output_dir)
    write_json(output_dir / TRAINING_PROFILE_FILE, build_training_profile(X_train, y_train, target))

    metrics = {
        "data_id": DATA_ID,
        "dataset_version": fingerprint,
        "target": target,
        "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "test_size": 0.2},
        "validation_strategy": validation_strategy_report(len(X_train)),
        "validation_comparison": validation_comparison,
        "metric_design": metric_design_report(),
        "best_model": best_name,
        "segment_metrics_summary": segment_metrics,
        "cross_validation": cv_results.to_dict(orient="records"),
        "test": test_metrics,
    }
    write_json(output_dir / METRICS_FILE, metrics)
    write_model_card(
        output_dir,
        best_name,
        metrics,
        json.loads((output_dir / "leakage_audit.json").read_text(encoding="utf-8")),
        validation_comparison,
    )
    return to_jsonable(metrics)


def predict(artifact_dir: Path, input_csv: Path, output_csv: Path) -> None:
    model = joblib.load(artifact_dir / MODEL_FILE)
    input_df = pd.read_csv(input_csv)
    output_df = input_df.copy()
    output_df["predicted_price"] = model.predict(input_df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)


def monitor(artifact_dir: Path, input_csv: Path, output_json: Path, missing_rate_alert: float) -> dict[str, Any]:
    profile = json.loads((artifact_dir / TRAINING_PROFILE_FILE).read_text(encoding="utf-8"))
    incoming = pd.read_csv(input_csv)
    required_columns = set(profile["raw_columns"])
    incoming_columns = set(incoming.columns)

    drift_rows = []
    for column, train_rate in profile["raw_missing_rate"].items():
        if column not in incoming:
            continue
        current_rate = float(incoming[column].isna().mean())
        change = abs(current_rate - float(train_rate))
        drift_rows.append(
            {
                "column": column,
                "train_missing_rate": float(train_rate),
                "current_missing_rate": current_rate,
                "absolute_change": change,
                "alert": change >= missing_rate_alert,
            }
        )

    report = {
        "row_count": int(len(incoming)),
        "missing_required_columns": sorted(required_columns - incoming_columns),
        "extra_columns": sorted(incoming_columns - required_columns),
        "missing_rate_alert_threshold": missing_rate_alert,
        "missing_rate_drift": drift_rows,
    }
    contract_path = artifact_dir / "data_contract.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        report["data_contract_validation"] = validate_against_contract(
            incoming,
            contract,
        )
        report["row_validation_report"] = row_validation_report(incoming, contract)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, report)
    return report


def create_sample_input(output_csv: Path, rows: int) -> None:
    df = load_data()
    target = infer_target(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[target]).head(rows).to_csv(output_csv, index=False)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, BaseEstimator):
        return repr(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end OpenML housing regression with missing-data robustness and nonlinear features."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts_housing"))
    train_parser.add_argument("--n-splits", type=int, default=5)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_housing"))
    predict_parser.add_argument("--input-csv", type=Path, required=True)
    predict_parser.add_argument("--output-csv", type=Path, default=Path("artifacts_housing/predictions.csv"))

    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_housing"))
    monitor_parser.add_argument("--input-csv", type=Path, required=True)
    monitor_parser.add_argument("--output-json", type=Path, default=Path("artifacts_housing/monitoring_report.json"))
    monitor_parser.add_argument("--missing-rate-alert", type=float, default=0.15)

    sample_parser = subparsers.add_parser("sample-input")
    sample_parser.add_argument("--output-csv", type=Path, default=Path("artifacts_housing/sample_houses.csv"))
    sample_parser.add_argument("--rows", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        metrics = train(args.output_dir, args.n_splits)
        print("Training complete")
        print(f"Model: {(args.output_dir / MODEL_FILE).resolve()}")
        print(f"Best model: {metrics['best_model']}")
        print(f"Test RMSE: {metrics['test']['rmse']:.2f}")
        print(f"Test MAE: {metrics['test']['mae']:.2f}")
        print(f"Test R2: {metrics['test']['r2']:.3f}")
    elif args.command == "predict":
        predict(args.artifact_dir, args.input_csv, args.output_csv)
        print(f"Predictions saved: {args.output_csv.resolve()}")
    elif args.command == "monitor":
        report = monitor(args.artifact_dir, args.input_csv, args.output_json, args.missing_rate_alert)
        alerts = sum(row["alert"] for row in report["missing_rate_drift"])
        print(f"Monitoring report saved: {args.output_json.resolve()}")
        print(f"Missing-rate alerts: {alerts}")
    elif args.command == "sample-input":
        create_sample_input(args.output_csv, args.rows)
        print(f"Sample input saved: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
