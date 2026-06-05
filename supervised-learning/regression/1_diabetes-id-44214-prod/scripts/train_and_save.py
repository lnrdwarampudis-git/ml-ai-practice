"""
scripts/train_and_save.py
Run ONCE to train the model and save all artifacts the API will load at startup.

Usage:
    python scripts/train_and_save.py --output-dir ./model_artifacts
"""

import argparse
import hashlib
import json
import warnings
import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


def train(output_dir: str, seed: int = 42):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    feature_names = list(data.feature_names)
    TARGET = "target"

    X = df[feature_names]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)

    # ── Build pipeline: scaler → GBR
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(random_state=seed)),
    ])

    param_grid = {
        "model__n_estimators":    [50, 100, 200],
        "model__max_depth":       [2, 3, 4],
        "model__learning_rate":   [0.05, 0.1, 0.2],
        "model__min_samples_leaf":[1, 2, 5],
    }
    search = RandomizedSearchCV(
        pipeline, param_grid, n_iter=20, cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=seed, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    # ── Evaluate
    preds = best.predict(X_test)
    metrics = {
        "rmse":  round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "mae":   round(float(mean_absolute_error(y_test, preds)), 4),
        "r2":    round(float(r2_score(y_test, preds)), 4),
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
    }
    print(f"\n  Best params : {search.best_params_}")
    print(f"  Test RMSE   : {metrics['rmse']}")
    print(f"  Test R²     : {metrics['r2']}")

    # ── Compute training distribution (for drift detection)
    train_stats = {
        col: {
            "mean": float(X_train[col].mean()),
            "std":  float(X_train[col].std()),
            "min":  float(X_train[col].min()),
            "max":  float(X_train[col].max()),
            "p5":   float(X_train[col].quantile(0.05)),
            "p95":  float(X_train[col].quantile(0.95)),
        }
        for col in feature_names
    }

    # ── Save artifacts
    model_path = out / "model.pkl"
    joblib.dump(best, model_path)
    print(f"  Saved model → {model_path}")

    metadata = {
        "model_id":      "gbr_pipeline_v1",
        "model_type":    "Pipeline(StandardScaler + GradientBoostingRegressor)",
        "trained_at":    str(datetime.datetime.utcnow()),
        "seed":          seed,
        "feature_names": feature_names,
        "target":        TARGET,
        "best_params":   search.best_params_,
        "metrics":       metrics,
        "model_hash":    hashlib.sha256(model_path.read_bytes()).hexdigest()[:16],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    (out / "train_stats.json").write_text(json.dumps(train_stats, indent=2))

    # Save training data sample (for reference / baseline drift)
    X_train.describe().to_csv(out / "train_summary.csv")

    print(f"\n  ✅ Artifacts saved to: {out.resolve()}")
    print(f"     model.pkl | metadata.json | train_stats.json | train_summary.csv")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./model_artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.output_dir, args.seed)
