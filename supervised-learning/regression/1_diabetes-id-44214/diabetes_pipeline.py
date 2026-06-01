from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ARTIFACT_ROOT = Path("artifacts_diabetes")
_MPLCONFIGDIR = _ARTIFACT_ROOT / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_ARTIFACT_ROOT / ".cache"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, learning_curve, train_test_split, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, SplineTransformer, StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
DATA_ID = 44214
MODEL_FILE = "diabetes_regression_pipeline.joblib"
METRICS_FILE = "metrics.json"
TRAINING_PROFILE_FILE = "training_profile.json"
TARGET_CANDIDATES = ["target", "disease_progression", "class", "response", "y"]


@dataclass(frozen=True)
class ColumnGroups:
    numeric: list[str]
    categorical: list[str]


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip numeric predictors using train-fold quantiles to reduce leverage points."""

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


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Fit-safe numeric interaction features for the diabetes regression dataset."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DiabetesFeatureEngineer":
        self.input_columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for column in X.columns:
            X[column] = pd.to_numeric(X[column], errors="coerce")

        if {"bmi", "bp"}.issubset(X.columns):
            X["bmi_x_bp"] = X["bmi"] * X["bp"]
        if {"bmi", "s5"}.issubset(X.columns):
            X["bmi_x_s5"] = X["bmi"] * X["s5"]
        if {"s1", "s2"}.issubset(X.columns):
            X["s1_minus_s2"] = X["s1"] - X["s2"]
        if {"s3", "s4"}.issubset(X.columns):
            X["s3_x_s4"] = X["s3"] * X["s4"]

        # Generic interaction fallback if OpenML column names differ.
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        if "bmi_x_bp" not in X.columns and len(numeric_cols) >= 2:
            X[f"{numeric_cols[0]}_x_{numeric_cols[1]}"] = X[numeric_cols[0]] * X[numeric_cols[1]]

        return X.replace([np.inf, -np.inf], np.nan)


def load_data(prefer_openml: bool = True) -> pd.DataFrame:
    """Load OpenML diabetes data_id=44214, with a local sklearn fallback for offline use."""
    if prefer_openml:
        data_home = _ARTIFACT_ROOT / ".sklearn_data"
        data_home.mkdir(parents=True, exist_ok=True)
        try:
            diabetes = fetch_openml(data_id=DATA_ID, as_frame=True, parser="auto", data_home=data_home)
            return diabetes.frame.copy()
        except Exception as exc:
            print(f"OpenML fetch failed for data_id={DATA_ID}; using sklearn built-in diabetes fallback. Reason: {exc}")

    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame.copy()
    df["target"] = diabetes.target
    return df


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


def dataframe_fingerprint(df: pd.DataFrame, source: str) -> dict[str, Any]:
    ordered = df.sort_index(axis=1).copy()
    row_hashes = pd.util.hash_pandas_object(ordered.astype("string"), index=True).values
    digest = hashlib.sha256(row_hashes.tobytes()).hexdigest()
    return {
        "data_id": DATA_ID,
        "source": source,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "sha256_hash": digest,
        "hash_policy": "sha256 over pandas row hashes after sorting columns and casting values to string",
    }


def write_dataset_card(output_dir: Path, df: pd.DataFrame, target: str, fingerprint: dict[str, Any]) -> None:
    text = f"""# Dataset Card: Diabetes Regression

## Purpose

Teach Phase 1 regression foundations: EDA, baselines, loss functions, residual analysis, and bias vs variance.

## Source

Preferred source: `fetch_openml(data_id={DATA_ID}, as_frame=True, parser="auto")`

Fallback source: `sklearn.datasets.load_diabetes(as_frame=True)`

Actual source for this run: `{fingerprint["source"]}`

## Version Fingerprint

- Rows: {fingerprint["rows"]}
- Columns: {fingerprint["columns"]}
- SHA256: `{fingerprint["sha256_hash"]}`

## Target

`{target}` is a continuous disease progression target.

## Known Data Notes

- Dataset is small, so validation variance matters.
- Features are numeric and mostly clean.
- A modest R2 is normal for this dataset; residual analysis matters more than chasing a high score.
"""
    (output_dir / "dataset_card.md").write_text(text, encoding="utf-8")


def write_model_card(output_dir: Path, best_name: str, metrics: dict[str, Any]) -> None:
    text = f"""# Model Card: Diabetes Phase 1 Regression

## Model

Selected model: `{best_name}`

Serialized artifact: `{MODEL_FILE}`

## Intended Use

Educational regression foundation project. Use it to study baseline comparison, regularization, loss functions, residuals, and bias vs variance.

## Not Intended For

- Medical decision-making.
- Patient diagnosis.
- Deployment without clinical validation.

## Metrics

- RMSE: {metrics["test"]["rmse"]:.2f}
- MAE: {metrics["test"]["mae"]:.2f}
- Median absolute error: {metrics["test"]["median_absolute_error"]:.2f}
- R2: {metrics["test"]["r2"]:.3f}

## Validation

The pipeline uses an 80/20 holdout and KFold cross-validation on the training split.

## Learning Artifacts

- `plot_loss_functions.png`
- `plot_learning_curve.png`
- `plot_validation_curve_ridge_alpha.png`
- `plot_ridge_coefficient_paths.png`
- `plot_lasso_coefficient_paths.png`
- `plot_residuals.png`
- `plot_residual_qq.png`

## Limitations

This is a small tabular dataset. Complex models can overfit easily, so simple regularized and robust linear models are often competitive.
"""
    (output_dir / "model_card.md").write_text(text, encoding="utf-8")


def artifact_check(output_dir: Path) -> dict[str, Any]:
    required = [
        MODEL_FILE,
        METRICS_FILE,
        TRAINING_PROFILE_FILE,
        "dataset_version.json",
        "dataset_card.md",
        "model_card.md",
        "cv_results.csv",
        "research_decisions.json",
        "test_predictions.csv",
        "largest_prediction_errors.csv",
        "plot_loss_functions.png",
        "plot_learning_curve.png",
        "plot_validation_curve_ridge_alpha.png",
        "plot_ridge_coefficient_paths.png",
        "plot_lasso_coefficient_paths.png",
        "plot_actual_vs_predicted.png",
        "plot_residuals.png",
        "plot_residual_distribution.png",
        "plot_residual_qq.png",
    ]
    rows = []
    for name in required:
        path = output_dir / name
        rows.append({"artifact": name, "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0})
    report = {"all_required_present": all(row["exists"] for row in rows), "artifacts": rows}
    write_json(output_dir / "artifact_check_report.json", report)
    return report


def get_column_groups(X: pd.DataFrame) -> ColumnGroups:
    engineered = DiabetesFeatureEngineer().fit(X).transform(X)
    numeric = engineered.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in engineered.columns if column not in numeric]
    return ColumnGroups(numeric=numeric, categorical=categorical)


def build_numeric_pipeline(basis: str = "none", scaler: str = "standard", clip: bool = False) -> Pipeline:
    steps: list[tuple[str, BaseEstimator]] = []
    if clip:
        steps.append(("clip", OutlierClipper()))
    steps.append(("imputer", SimpleImputer(strategy="median", add_indicator=True)))
    steps.append(("scaler", RobustScaler() if scaler == "robust" else StandardScaler()))
    if basis == "poly":
        steps.append(("basis", PolynomialFeatures(degree=2, include_bias=False)))
    elif basis == "interactions":
        steps.append(("basis", PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)))
    elif basis == "splines":
        steps.append(("basis", SplineTransformer(n_knots=5, degree=3, include_bias=False)))
    return Pipeline(steps)


def build_preprocessor(X: pd.DataFrame, basis: str = "none", scaler: str = "standard", clip: bool = False) -> ColumnTransformer:
    groups = get_column_groups(X)
    return ColumnTransformer(
        [("num", build_numeric_pipeline(basis=basis, scaler=scaler, clip=clip), groups.numeric)],
        sparse_threshold=0.0,
        remainder="drop",
    )


def build_pipeline(
    X: pd.DataFrame,
    model: BaseEstimator,
    basis: str = "none",
    scaler: str = "standard",
    clip: bool = False,
) -> Pipeline:
    return Pipeline(
        [
            ("feature_engineering", DiabetesFeatureEngineer()),
            ("preprocess", build_preprocessor(X, basis=basis, scaler=scaler, clip=clip)),
            ("model", model),
        ]
    )


def candidate_models(X_train: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "dummy_mean": Pipeline([("model", DummyRegressor(strategy="mean"))]),
        "linear_regression": build_pipeline(X_train, LinearRegression()),
        "ridge": build_pipeline(X_train, Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        "lasso": build_pipeline(X_train, Lasso(alpha=0.05, max_iter=5000, random_state=RANDOM_STATE)),
        "elastic_net": build_pipeline(
            X_train,
            ElasticNet(alpha=0.03, l1_ratio=0.2, max_iter=5000, random_state=RANDOM_STATE),
        ),
        "huber_robust": build_pipeline(
            X_train,
            HuberRegressor(epsilon=1.35, alpha=0.001, max_iter=1000),
            scaler="robust",
            clip=True,
        ),
        "ridge_interactions": build_pipeline(
            X_train,
            Ridge(alpha=30.0, random_state=RANDOM_STATE),
            basis="interactions",
        ),
        "ridge_splines": build_pipeline(
            X_train,
            Ridge(alpha=50.0, random_state=RANDOM_STATE),
            basis="splines",
        ),
        "random_forest": build_pipeline(
            X_train,
            RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            scaler="robust",
            clip=True,
        ),
        "extra_trees": build_pipeline(
            X_train,
            ExtraTreesRegressor(
                n_estimators=400,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            scaler="robust",
            clip=True,
        ),
        "hist_gradient_boosting": build_pipeline(
            X_train,
            HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.04,
                max_iter=250,
                l2_regularization=0.05,
                random_state=RANDOM_STATE,
            ),
            scaler="robust",
            clip=True,
        ),
    }


def regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "median_absolute_error": float(np.median(np.abs(np.asarray(y_true) - predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def cross_validate_models(X_train: pd.DataFrame, y_train: pd.Series, n_splits: int) -> pd.DataFrame:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
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


def save_research_artifacts(X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path, target: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = X_train.copy()
    train_df[target] = y_train
    engineered = DiabetesFeatureEngineer().fit(X_train, y_train).transform(X_train)

    missingness_report(train_df).to_csv(output_dir / "research_missingness_report.csv")
    train_df.dtypes.astype(str).rename("dtype").to_csv(output_dir / "schema.csv")
    train_df.select_dtypes(include=["number"]).describe().T.to_csv(output_dir / "numeric_summary.csv")
    engineered.corrwith(y_train).sort_values(ascending=False).to_csv(
        output_dir / "numeric_target_correlations.csv", header=["correlation"]
    )
    save_research_plots(train_df, output_dir, target)
    write_json(
        output_dir / "research_decisions.json",
        {
            "problem_definition": {
                "problem_type": "regression",
                "target": target,
                "source": f"fetch_openml(data_id={DATA_ID}, as_frame=True, parser='auto')",
                "fallback": "sklearn.datasets.load_diabetes(as_frame=True) when OpenML is unavailable",
            },
            "metric_policy": {
                "primary": "rmse",
                "secondary": ["mae", "median_absolute_error", "r2"],
                "why": "RMSE penalizes larger clinical progression errors; MAE is easier to explain.",
            },
            "modeling_policy": {
                "baselines": ["DummyRegressor", "LinearRegression"],
                "regularization": ["Ridge", "Lasso", "ElasticNet"],
                "robustness": ["OutlierClipper", "RobustScaler", "HuberRegressor"],
                "nonlinearity": ["interaction features", "PolynomialFeatures interaction_only", "SplineTransformer"],
                "ensembles": ["RandomForestRegressor", "ExtraTreesRegressor", "HistGradientBoostingRegressor"],
            },
            "validation_policy": {
                "split": "80/20 holdout for final reporting",
                "cross_validation": f"{RANDOM_STATE}-seeded shuffled KFold on training data",
            },
        },
    )


def save_research_plots(train_df: pd.DataFrame, output_dir: Path, target: str) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    sns.histplot(train_df[target], kde=True, color="#4C78A8")
    plt.title("Diabetes Target Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_target_distribution.png", dpi=160)
    plt.close()

    missing = train_df.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(7, 4))
    missing.sort_values().plot(kind="barh", color="#F58518")
    plt.title("Missing Rate by Column")
    plt.xlabel("Missing Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_missingness.png", dpi=160)
    plt.close()

    numeric = train_df.select_dtypes(include=["number"]).drop(columns=[target], errors="ignore")
    if len(numeric.columns):
        corr = numeric.corrwith(train_df[target]).abs().sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        corr.sort_values().plot(kind="barh", color="#54A24B")
        plt.title("Numeric Feature Association With Target")
        plt.xlabel("Absolute Correlation")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_numeric_target_correlations.png", dpi=160)
        plt.close()

    for column in numeric.corrwith(train_df[target]).abs().sort_values(ascending=False).head(4).index:
        plt.figure(figsize=(5.7, 4))
        sns.regplot(data=train_df, x=column, y=target, lowess=True, scatter_kws={"alpha": 0.6}, color="#B279A2")
        plt.title(f"{target} vs {column}")
        plt.tight_layout()
        plt.savefig(output_dir / f"plot_{column}_vs_target.png", dpi=160)
        plt.close()


def save_model_comparison(cv_results: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=cv_results, y="model", x="cv_rmse_mean", color="#4C78A8")
    plt.errorbar(
        cv_results["cv_rmse_mean"],
        np.arange(len(cv_results)),
        xerr=cv_results["cv_rmse_std"],
        fmt="none",
        ecolor="#333333",
        capsize=3,
    )
    plt.title("Cross-Validated RMSE by Model")
    plt.xlabel("RMSE lower is better")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_model_comparison_rmse.png", dpi=160)
    plt.close()

    melted = cv_results.melt(
        id_vars=["model"],
        value_vars=["train_rmse_mean", "cv_rmse_mean"],
        var_name="split",
        value_name="rmse",
    )
    melted["split"] = melted["split"].map({"train_rmse_mean": "Train", "cv_rmse_mean": "Cross-validation"})
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=melted, y="model", x="rmse", hue="split")
    plt.title("Bias vs Variance Check: Train RMSE vs CV RMSE")
    plt.xlabel("RMSE lower is better")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_bias_variance_train_vs_cv.png", dpi=160)
    plt.close()

    gap_df = cv_results.sort_values("generalization_gap_rmse", ascending=False)
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=gap_df, y="model", x="generalization_gap_rmse", color="#E45756")
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Generalization Gap by Model")
    plt.xlabel("CV RMSE - Train RMSE")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_generalization_gap.png", dpi=160)
    plt.close()


def save_loss_function_plot(output_dir: Path) -> None:
    residuals = np.linspace(-120, 120, 401)
    squared = residuals**2
    absolute = np.abs(residuals)
    delta = 35
    huber = np.where(np.abs(residuals) <= delta, 0.5 * residuals**2, delta * (np.abs(residuals) - 0.5 * delta))

    scaled = pd.DataFrame(
        {
            "Residual": residuals,
            "Squared loss / 120": squared / 120,
            "Absolute loss": absolute,
            "Huber loss / 35": huber / 35,
        }
    ).melt(id_vars="Residual", var_name="Loss", value_name="Penalty")

    plt.figure(figsize=(7.5, 4.8))
    sns.lineplot(data=scaled, x="Residual", y="Penalty", hue="Loss")
    plt.title("Loss Function Behavior")
    plt.xlabel("Prediction error: y - y_hat")
    plt.ylabel("Scaled penalty")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_loss_functions.png", dpi=160)
    plt.close()


def save_learning_curve_plot(X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path) -> None:
    model = build_pipeline(X_train, Ridge(alpha=10.0, random_state=RANDOM_STATE))
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, valid_scores = learning_curve(
        model,
        X_train,
        y_train,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    curve = pd.DataFrame(
        {
            "train_size": train_sizes,
            "train_rmse": -train_scores.mean(axis=1),
            "train_rmse_std": train_scores.std(axis=1),
            "cv_rmse": -valid_scores.mean(axis=1),
            "cv_rmse_std": valid_scores.std(axis=1),
        }
    )
    curve.to_csv(output_dir / "learning_curve.csv", index=False)

    plt.figure(figsize=(7, 4.8))
    plt.plot(curve["train_size"], curve["train_rmse"], marker="o", label="Train RMSE")
    plt.plot(curve["train_size"], curve["cv_rmse"], marker="o", label="CV RMSE")
    plt.fill_between(
        curve["train_size"],
        curve["cv_rmse"] - curve["cv_rmse_std"],
        curve["cv_rmse"] + curve["cv_rmse_std"],
        alpha=0.2,
    )
    plt.title("Learning Curve: Does More Data Help?")
    plt.xlabel("Training examples")
    plt.ylabel("RMSE lower is better")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_learning_curve.png", dpi=160)
    plt.close()


def save_validation_curve_plot(X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path) -> None:
    alphas = np.logspace(-3, 3, 13)
    model = build_pipeline(X_train, Ridge(random_state=RANDOM_STATE))
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_scores, valid_scores = validation_curve(
        model,
        X_train,
        y_train,
        param_name="model__alpha",
        param_range=alphas,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    curve = pd.DataFrame(
        {
            "alpha": alphas,
            "train_rmse": -train_scores.mean(axis=1),
            "cv_rmse": -valid_scores.mean(axis=1),
            "cv_rmse_std": valid_scores.std(axis=1),
        }
    )
    curve.to_csv(output_dir / "validation_curve_ridge_alpha.csv", index=False)

    plt.figure(figsize=(7, 4.8))
    plt.semilogx(curve["alpha"], curve["train_rmse"], marker="o", label="Train RMSE")
    plt.semilogx(curve["alpha"], curve["cv_rmse"], marker="o", label="CV RMSE")
    plt.title("Validation Curve: Ridge Alpha")
    plt.xlabel("alpha")
    plt.ylabel("RMSE lower is better")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_validation_curve_ridge_alpha.png", dpi=160)
    plt.close()


def save_coefficient_path_plots(X_train: pd.DataFrame, y_train: pd.Series, output_dir: Path) -> None:
    engineered = DiabetesFeatureEngineer().fit(X_train, y_train).transform(X_train)
    numeric_cols = engineered.select_dtypes(include=["number"]).columns.tolist()
    values = SimpleImputer(strategy="median").fit_transform(engineered[numeric_cols])
    scaled = StandardScaler().fit_transform(values)
    alphas = np.logspace(-4, 3, 60)

    for model_name, model_factory, filename in [
        ("Ridge", lambda alpha: Ridge(alpha=alpha, random_state=RANDOM_STATE), "ridge"),
        ("Lasso", lambda alpha: Lasso(alpha=alpha, max_iter=10000, random_state=RANDOM_STATE), "lasso"),
    ]:
        rows = []
        for alpha in alphas:
            model = model_factory(alpha)
            model.fit(scaled, y_train)
            for feature, coef in zip(numeric_cols, np.ravel(model.coef_)):
                rows.append({"alpha": alpha, "feature": feature, "coefficient": float(coef)})
        path_df = pd.DataFrame(rows)
        path_df.to_csv(output_dir / f"{filename}_coefficient_paths.csv", index=False)

        plt.figure(figsize=(8, 5.2))
        for feature, group in path_df.groupby("feature"):
            plt.semilogx(group["alpha"], group["coefficient"], linewidth=1, alpha=0.85, label=feature)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title(f"{model_name} Coefficient Paths")
        plt.xlabel("alpha")
        plt.ylabel("coefficient")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(output_dir / f"plot_{filename}_coefficient_paths.png", dpi=160)
        plt.close()


def save_feature_importance(model: Pipeline, output_dir: Path) -> None:
    final_model = model.named_steps["model"]
    preprocess = model.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()

    if hasattr(final_model, "feature_importances_"):
        values = final_model.feature_importances_
        importance = values
        coefficient = np.repeat(np.nan, len(values))
    elif hasattr(final_model, "coef_"):
        coefficient = np.ravel(final_model.coef_)
        importance = np.abs(coefficient)
    else:
        return

    if len(feature_names) != len(importance):
        feature_names = [f"feature_{i}" for i in range(len(importance))]

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance, "coefficient": coefficient}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    plt.figure(figsize=(8.5, 5.2))
    sns.barplot(data=importance_df.head(20), y="feature", x="importance", color="#4C78A8")
    plt.title("Top Model Features")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_feature_importance.png", dpi=160)
    plt.close()


def save_error_analysis(X_test: pd.DataFrame, y_test: pd.Series, predictions: np.ndarray, output_dir: Path) -> None:
    error_df = X_test.copy()
    error_df["actual_target"] = y_test.to_numpy()
    error_df["predicted_target"] = predictions
    error_df["residual"] = error_df["actual_target"] - error_df["predicted_target"]
    error_df["absolute_error"] = error_df["residual"].abs()
    error_df.to_csv(output_dir / "test_predictions.csv", index=False)
    error_df.sort_values("absolute_error", ascending=False).head(40).to_csv(
        output_dir / "largest_prediction_errors.csv", index=False
    )

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=predictions, y=error_df["residual"], alpha=0.7, color="#E45756")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Residuals vs Predictions")
    plt.xlabel("Predicted Target")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_residuals.png", dpi=160)
    plt.close()

    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.7, color="#72B7B2")
    low = min(float(y_test.min()), float(np.min(predictions)))
    high = max(float(y_test.max()), float(np.max(predictions)))
    plt.plot([low, high], [low, high], color="black", linewidth=1)
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Target")
    plt.ylabel("Predicted Target")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_actual_vs_predicted.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(error_df["residual"], kde=True, color="#F58518")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_residual_distribution.png", dpi=160)
    plt.close()

    residuals = np.sort(error_df["residual"].to_numpy())
    theoretical = np.quantile(np.random.default_rng(RANDOM_STATE).normal(size=200_000), np.linspace(0.01, 0.99, len(residuals)))
    plt.figure(figsize=(5.2, 5.2))
    sns.scatterplot(x=theoretical, y=residuals, alpha=0.75, color="#4C78A8")
    low = min(float(theoretical.min()), float(residuals.min()))
    high = max(float(theoretical.max()), float(residuals.max()))
    plt.plot([low, high], [low, high], color="black", linewidth=1)
    plt.title("Residual Q-Q Style Plot")
    plt.xlabel("Theoretical normal quantiles")
    plt.ylabel("Observed residual quantiles")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_residual_qq.png", dpi=160)
    plt.close()


def build_training_profile(X_train: pd.DataFrame, y_train: pd.Series, target: str) -> dict[str, Any]:
    engineered = DiabetesFeatureEngineer().fit(X_train, y_train).transform(X_train)
    return to_jsonable(
        {
            "row_count": int(len(X_train)),
            "target": target,
            "raw_columns": list(X_train.columns),
            "engineered_columns": list(engineered.columns),
            "target_summary": y_train.describe().to_dict(),
            "raw_missing_rate": X_train.isna().mean().to_dict(),
            "engineered_missing_rate": engineered.isna().mean().to_dict(),
            "numeric_summary": engineered.describe().T.to_dict(orient="index"),
        }
    )


def train(output_dir: Path, n_splits: int, no_openml: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(prefer_openml=not no_openml)
    target = infer_target(df)
    data_source = "sklearn_builtin_diabetes" if no_openml else f"openml_{DATA_ID}_or_sklearn_fallback"
    fingerprint = dataframe_fingerprint(df, data_source)
    write_json(output_dir / "dataset_version.json", fingerprint)
    write_dataset_card(output_dir, df, target, fingerprint)
    X_train, X_test, y_train, y_test = split_data(df, target)

    save_research_artifacts(X_train, y_train, output_dir, target)
    cv_results = cross_validate_models(X_train, y_train, n_splits=n_splits)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    save_model_comparison(cv_results, output_dir)
    save_loss_function_plot(output_dir)
    save_learning_curve_plot(X_train, y_train, output_dir)
    save_validation_curve_plot(X_train, y_train, output_dir)
    save_coefficient_path_plots(X_train, y_train, output_dir)

    best_name = str(cv_results.iloc[0]["model"])
    best_model = clone(candidate_models(X_train)[best_name])
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)
    test_metrics = regression_metrics(y_test, predictions)

    joblib.dump(best_model, output_dir / MODEL_FILE)
    save_feature_importance(best_model, output_dir)
    save_error_analysis(X_test, y_test, predictions, output_dir)
    write_json(output_dir / TRAINING_PROFILE_FILE, build_training_profile(X_train, y_train, target))

    metrics = {
        "data_id": DATA_ID,
        "dataset_version": fingerprint,
        "target": target,
        "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "test_size": 0.2},
        "best_model": best_name,
        "cross_validation": cv_results.to_dict(orient="records"),
        "test": test_metrics,
    }
    write_json(output_dir / METRICS_FILE, metrics)
    write_model_card(output_dir, best_name, metrics)
    artifact_check(output_dir)
    return to_jsonable(metrics)


def predict(artifact_dir: Path, input_csv: Path, output_csv: Path) -> None:
    model = joblib.load(artifact_dir / MODEL_FILE)
    input_df = pd.read_csv(input_csv)
    output_df = input_df.copy()
    output_df["predicted_target"] = model.predict(input_df)
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
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, report)
    return report


def create_sample_input(output_csv: Path, rows: int, no_openml: bool = False) -> None:
    df = load_data(prefer_openml=not no_openml)
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
    parser = argparse.ArgumentParser(description="End-to-end regression pipeline for OpenML Diabetes data_id=44214.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts_diabetes"))
    train_parser.add_argument("--n-splits", type=int, default=5)
    train_parser.add_argument("--no-openml", action="store_true", help="Use sklearn built-in diabetes fallback.")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_diabetes"))
    predict_parser.add_argument("--input-csv", type=Path, required=True)
    predict_parser.add_argument("--output-csv", type=Path, default=Path("artifacts_diabetes/predictions.csv"))

    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_diabetes"))
    monitor_parser.add_argument("--input-csv", type=Path, required=True)
    monitor_parser.add_argument("--output-json", type=Path, default=Path("artifacts_diabetes/monitoring_report.json"))
    monitor_parser.add_argument("--missing-rate-alert", type=float, default=0.1)

    sample_parser = subparsers.add_parser("sample-input")
    sample_parser.add_argument("--output-csv", type=Path, default=Path("artifacts_diabetes/sample_diabetes.csv"))
    sample_parser.add_argument("--rows", type=int, default=10)
    sample_parser.add_argument("--no-openml", action="store_true", help="Use sklearn built-in diabetes fallback.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        metrics = train(args.output_dir, args.n_splits, no_openml=args.no_openml)
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
        create_sample_input(args.output_csv, args.rows, no_openml=args.no_openml)
        print(f"Sample input saved: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
