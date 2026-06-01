"""
phases/phase4_production.py
Phases 31–40: Uncertainty → Robustness → Deployment → Batch/Real-Time
              → Monitoring & Drift → Retraining Strategy → Model Registry
              → Governance → Model Cards → Business Impact Tracking
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import joblib
from pathlib import Path
from sklearn.utils import resample
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyRegressor

from utils.helpers import save_fig, save_json, regression_metrics, psi, today, now, set_plot_style


def run(cfg, ctx: dict) -> dict:
    set_plot_style()
    fp = cfg.figures_dir
    ap = cfg.artifacts_dir
    mp = cfg.models_dir
    seed = cfg.random_seed

    feature_names = ctx["feature_names"]
    X_train = ctx["X_train"]; X_test = ctx["X_test"]
    y_train = ctx["y_train"]; y_test  = ctx["y_test"]
    X       = ctx["X"];       y       = ctx["y"]
    ridge_cv   = ctx["ridge_cv"]
    gbr_cv     = ctx["gbr_cv"]
    gbr_preds  = ctx["gbr_preds"]

    print("\n" + "=" * 60)
    print("  PHASE 4 — PRODUCTION")
    print("=" * 60)

    # ── 31. Uncertainty (Bootstrap CI) ───────────────────────────────────────
    print("\n[31] Uncertainty Estimation — Bootstrap + Bayesian Ridge")
    n_boot = cfg.n_bootstrap
    boot_preds = np.zeros((n_boot, len(X_test)))
    for i in range(n_boot):
        Xb, yb = resample(X_train, y_train, random_state=i)
        m = Ridge(alpha=ridge_cv.best_params_["alpha"])
        m.fit(Xb, yb)
        boot_preds[i] = m.predict(X_test)

    lower = np.percentile(boot_preds, 5, axis=0)
    upper = np.percentile(boot_preds, 95, axis=0)
    mean_pred = boot_preds.mean(axis=0)

    idx_sorted = np.argsort(y_test.values)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(range(len(y_test)), y_test.values[idx_sorted],
               color="black", s=15, label="Actual", zorder=5)
    ax.plot(range(len(mean_pred)), mean_pred[idx_sorted],
            color="steelblue", label="Mean Pred", linewidth=1.5)
    ax.fill_between(range(len(y_test)), lower[idx_sorted], upper[idx_sorted],
                    alpha=0.25, color="steelblue", label="90% Bootstrap CI")
    ax.set_title("Phase 4 — Prediction Uncertainty (Bootstrap CI)", fontweight="bold")
    ax.set_xlabel("Sample (sorted by actual)"); ax.set_ylabel("Target"); ax.legend()
    plt.tight_layout()
    save_fig(fig, fp, "22_uncertainty.png")

    br = BayesianRidge().fit(X_train, y_train)
    br_mean, br_std = br.predict(X_test, return_std=True)
    print(f"  Bayesian Ridge prediction: mean={br_mean.mean():.1f}, std={br_std.mean():.1f}")

    # ── 32. Robustness Testing ────────────────────────────────────────────────
    print("\n[32] Robustness Testing — Input Noise")
    noise_levels = [0, 0.01, 0.05, 0.1]
    robustness = {}
    for sigma in noise_levels:
        X_noisy = X_test + np.random.default_rng(seed).normal(0, sigma, X_test.shape)
        preds = gbr_cv.best_estimator_.predict(X_noisy)
        robustness[sigma] = round(float(np.sqrt(mean_squared_error(y_test, preds))), 3)
    print(f"  Noise σ → RMSE: {robustness}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(robustness.keys()), list(robustness.values()),
            "o-", color="coral", linewidth=2, markersize=8)
    ax.set_title("Phase 4 — Robustness: RMSE under Input Noise", fontweight="bold")
    ax.set_xlabel("Noise Std"); ax.set_ylabel("RMSE")
    plt.tight_layout()
    save_fig(fig, fp, "23_robustness.png")

    # ── 33. Deployment ────────────────────────────────────────────────────────
    print("\n[33] Deployment — Serialise & Load")
    model_path = mp / "model_gbr.pkl"
    joblib.dump(gbr_cv.best_estimator_, model_path)
    loaded = joblib.load(model_path)
    check = loaded.predict(X_test.iloc[[0]])[0]
    print(f"  Saved → {model_path}")
    print(f"  Load + single predict: {check:.2f}")

    # ── 34. Batch vs Real-Time Inference ──────────────────────────────────────
    print("\n[34] Batch vs Real-Time Inference")
    start = time.perf_counter()
    _ = loaded.predict(X_test)
    batch_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    _ = loaded.predict(X_test.iloc[[0]])
    single_ms = (time.perf_counter() - start) * 1000

    print(f"  Batch ({len(X_test)} rows): {batch_ms:.2f}ms "
          f"({batch_ms/len(X_test):.4f}ms/row)")
    print(f"  Single row: {single_ms:.4f}ms")

    # ── 35. Monitoring & Drift ────────────────────────────────────────────────
    print("\n[35] Monitoring & Drift Detection (PSI)")
    psi_scores = {col: round(psi(X_train[col].values, X_test[col].values, cfg.psi_bins), 4)
                  for col in feature_names}
    print(f"  PSI scores: {psi_scores}")
    drifted = [k for k, v in psi_scores.items() if v > 0.2]
    print(f"  Features with PSI>0.2 (drift warning): {drifted if drifted else 'None'}")

    # Temporal mean drift plot
    splits_t = np.array_split(X, cfg.n_drift_windows)
    means_df = pd.DataFrame([s.mean() for s in splits_t])
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Phase 4 — Feature Mean Drift across Temporal Windows", fontweight="bold")
    for i, col in enumerate(feature_names):
        ax = axes[i // 5][i % 5]
        ax.plot(means_df[col], "o-", color="steelblue", linewidth=2)
        ax.set_title(col); ax.set_xlabel("Window"); ax.set_ylabel("Mean")
    plt.tight_layout()
    save_fig(fig, fp, "24_drift_monitoring.png")

    # ── 36. Retraining Strategy ───────────────────────────────────────────────
    print("\n[36] Retraining Strategy")
    print("""
  Trigger criteria:
    - PSI > 0.2 on any feature
    - Live RMSE exceeds threshold by +10%
    - ≥1000 new labelled samples accumulated
  Strategy:
    - Benchmark full-retrain vs warm_start incremental
    - Shadow-mode deployment for A/B gating
    - Automated rollback on regression
""")

    # ── 37. Model Registry ────────────────────────────────────────────────────
    print("\n[37] Model Registry")
    registry = {
        "model_id":   "gbr_v1.0",
        "model_type": "GradientBoostingRegressor",
        "params":     str(gbr_cv.best_params_),
        "metrics":    regression_metrics(y_test, gbr_preds),
        "trained_on": today(),
        "artifact_path": str(model_path),
        "status": "champion",
    }
    save_json(registry, ap, "model_registry.json")

    # ── 38. Governance ────────────────────────────────────────────────────────
    print("\n[38] Governance")
    governance = {
        "data_source":      "OpenML 44214 / sklearn diabetes",
        "data_sensitivity": "Low (no PII — all features anonymised)",
        "intended_use":     "Research — disease progression prediction",
        "prohibited_use":   ["Real clinical decisions without physician oversight"],
        "bias_assessment":  "Sex feature present — sex-stratified eval required",
        "approval_status":  "pending review",
        "owner":            "ML Team",
        "review_cadence":   "quarterly",
    }
    save_json(governance, ap, "governance.json")

    # ── 39. Model Card ────────────────────────────────────────────────────────
    print("\n[39] Model Card")
    m = regression_metrics(y_test, gbr_preds)
    card = f"""# Model Card — Diabetes Progression Predictor
**Date**: {today()}
**Model**: GradientBoostingRegressor (best of 12+ evaluated)
**Dataset**: Diabetes (OpenML 44214 / sklearn, n=442)

## Performance
| Metric | Value |
|--------|-------|
| Test RMSE | {m['RMSE']} |
| Test MAE  | {m['MAE']}  |
| Test R²   | {m['R2']}   |

## Features
{', '.join(feature_names)}

## Intended Use
Quantitative prediction of disease progression for **research purposes only**.

## Limitations
- Small dataset (n=442) — high prediction uncertainty
- Features are anonymised — limited clinical context
- Not validated for clinical deployment
- Sex-based disparities not fully characterised

## Fairness
Sex-stratified evaluation recommended before any deployment.

## Reproducibility
SEED={seed}, 80/20 train-test split, 5-fold CV during tuning.
"""
    card_path = ap / "model_card.md"
    card_path.write_text(card)
    print(f"  Saved → {card_path}")

    # ── 40. Business Impact Tracking ──────────────────────────────────────────
    print("\n[40] Business Impact Tracking")
    dummy_preds = DummyRegressor(strategy="mean").fit(X_train, y_train).predict(X_test)
    cost_base  = cfg.cost_per_error_unit * mean_absolute_error(y_test, dummy_preds)
    cost_model = cfg.cost_per_error_unit * mean_absolute_error(y_test, gbr_preds)
    savings    = cost_base - cost_model
    print(f"  Baseline cost/patient: ${cost_base:.2f}")
    print(f"  GBR model cost/patient: ${cost_model:.2f}")
    print(f"  Savings/patient: ${savings:.2f} ({savings/cost_base*100:.1f}% reduction)")

    print("\n  ✅ Phase 4 complete.")
    ctx.update(dict(
        model_path=model_path, boot_lower=lower, boot_upper=upper,
        boot_mean=mean_pred, registry=registry,
    ))
    return ctx
