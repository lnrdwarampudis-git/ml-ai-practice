"""
phases/phase5_advanced.py
Phases 41–50: Quantile Regression → Conformal Prediction → Causal Regression
              → Multi-Output → Online Learning → Fairness → Privacy
              → Feature Store → A/B Testing → Cost-Aware Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

from sklearn.linear_model import (LinearRegression, Ridge, QuantileRegressor, SGDRegressor)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.helpers import save_fig, save_json, regression_metrics, asymmetric_loss, today, set_plot_style


def run(cfg, ctx: dict) -> dict:
    set_plot_style()
    fp = cfg.figures_dir
    ap = cfg.artifacts_dir
    seed = cfg.random_seed

    feature_names = ctx["feature_names"]
    X_train = ctx["X_train"]; X_test = ctx["X_test"]
    y_train = ctx["y_train"]; y_test  = ctx["y_test"]
    X = ctx["X"]; y = ctx["y"]
    ridge_cv  = ctx["ridge_cv"]
    gbr_cv    = ctx["gbr_cv"]
    gbr_preds = ctx["gbr_preds"]

    print("\n" + "=" * 60)
    print("  PHASE 5 — ADVANCED")
    print("=" * 60)

    # ── 41. Quantile Regression ───────────────────────────────────────────────
    print("\n[41] Quantile Regression")
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    qr_models = {}
    for q in quantiles:
        qr = QuantileRegressor(quantile=q, alpha=0.1, solver="highs")
        qr.fit(X_train, y_train)
        qr_models[q] = qr

    idx_sorted = np.argsort(y_test.values)
    colors_q = ["#d7191c", "#fdae61", "#2b83ba", "#abdda4", "#1a9641"]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(range(len(y_test)), y_test.values[idx_sorted],
               color="black", s=15, label="Actual", zorder=5)
    for (q, model), col in zip(qr_models.items(), colors_q):
        preds = model.predict(X_test)
        ax.plot(range(len(preds)), preds[idx_sorted], alpha=0.75,
                label=f"Q{int(q*100)}", color=col, linewidth=1.5)
    ax.set_title("Phase 5 — Quantile Regression (Multiple Quantiles)", fontweight="bold")
    ax.set_xlabel("Sample (sorted by actual)"); ax.set_ylabel("Target"); ax.legend()
    plt.tight_layout()
    save_fig(fig, fp, "25_quantile_regression.png")

    # ── 42. Conformal Prediction ──────────────────────────────────────────────
    print("\n[42] Conformal Prediction")
    X_tr2, X_cal, y_tr2, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed)
    cal_model = Ridge(alpha=ridge_cv.best_params_["alpha"])
    cal_model.fit(X_tr2, y_tr2)
    cal_scores = np.abs(y_cal.values - cal_model.predict(X_cal))
    q_hat = np.quantile(cal_scores, 1 - cfg.conformal_alpha)
    test_preds = cal_model.predict(X_test)
    cp_lower = test_preds - q_hat
    cp_upper = test_preds + q_hat
    coverage = float(np.mean((y_test.values >= cp_lower) & (y_test.values <= cp_upper)))
    print(f"  q̂={q_hat:.2f}, Empirical coverage={coverage:.3f} "
          f"(target={1-cfg.conformal_alpha:.1f})")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(range(len(y_test)), y_test.values[idx_sorted],
               color="black", s=15, zorder=5, label="Actual")
    ax.plot(range(len(test_preds)), test_preds[idx_sorted],
            color="steelblue", linewidth=1.5, label="Predicted")
    ax.fill_between(range(len(y_test)), cp_lower[idx_sorted], cp_upper[idx_sorted],
                    alpha=0.2, color="steelblue",
                    label=f"{int((1-cfg.conformal_alpha)*100)}% PI (cov={coverage:.2f})")
    ax.set_title("Phase 5 — Conformal Prediction Intervals", fontweight="bold")
    ax.set_xlabel("Sample"); ax.set_ylabel("Target"); ax.legend()
    plt.tight_layout()
    save_fig(fig, fp, "26_conformal_prediction.png")

    # ── 43. Causal Regression (Backdoor Adjustment) ───────────────────────────
    print("\n[43] Causal Regression — Structural / Backdoor Adjustment")
    unadj = LinearRegression().fit(X_train[["bmi"]], y_train)
    adj   = LinearRegression().fit(X_train[["bmi", "age", "sex"]], y_train)
    print(f"  Causal graph: BMI → target, controlled for age & sex")
    print(f"  Unadjusted BMI coeff : {unadj.coef_[0]:.3f}")
    print(f"  Adjusted BMI coeff   : {adj.coef_[0]:.3f}  (backdoor adjusted)")

    # ── 44. Multi-Output Regression ───────────────────────────────────────────
    print("\n[44] Multi-Output Regression")
    rng = np.random.default_rng(seed)
    y_multi = np.column_stack([
        y.values,
        y.values * 0.8 + rng.normal(0, 10, len(y))
    ])
    Xtr_m, Xte_m, ytr_m, yte_m = train_test_split(
        X, y_multi, test_size=0.2, random_state=seed)
    mo = MultiOutputRegressor(Ridge(alpha=1.0))
    mo.fit(Xtr_m, ytr_m)
    mo_preds = mo.predict(Xte_m)
    for i in range(2):
        r2 = r2_score(yte_m[:, i], mo_preds[:, i])
        print(f"  Output {i+1}: R²={r2:.3f}")

    # ── 45. Online / Incremental Learning ─────────────────────────────────────
    print("\n[45] Online / Incremental Regression (SGDRegressor)")
    sgd = SGDRegressor(loss="squared_error", random_state=seed, max_iter=1)
    batch_size = 20
    online_rmse = []
    for start in range(0, len(X_train) - batch_size, batch_size):
        Xb = X_train.iloc[start: start + batch_size]
        yb = y_train.iloc[start: start + batch_size]
        sgd.partial_fit(Xb, yb)
        online_rmse.append(
            float(np.sqrt(mean_squared_error(y_test, sgd.predict(X_test)))))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(online_rmse, "o-", color="steelblue", linewidth=2)
    ax.set_title("Phase 5 — Online Learning: RMSE over Mini-Batches", fontweight="bold")
    ax.set_xlabel("Mini-Batch #"); ax.set_ylabel("RMSE")
    plt.tight_layout()
    save_fig(fig, fp, "27_online_learning.png")

    # ── 46. Fairness in Regression ────────────────────────────────────────────
    print("\n[46] Fairness Analysis — Sex-Stratified Error")
    fair_df = X_test.copy()
    fair_df["y_true"]    = y_test.values
    fair_df["y_pred"]    = gbr_preds
    fair_df["abs_error"] = (fair_df["y_true"] - fair_df["y_pred"]).abs()
    fair_df["sex_group"] = (X_test["sex"] > 0).map({True: "Female", False: "Male"})

    fairness_stats = fair_df.groupby("sex_group").agg(
        count=("y_true", "count"),
        mean_mae=("abs_error", "mean"),
        mean_target=("y_true", "mean"),
        mean_pred=("y_pred", "mean"),
    ).round(2)
    print(fairness_stats)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 5 — Fairness Analysis by Sex", fontweight="bold")
    for ax, col, title in zip(axes,
                               ["mean_mae", "mean_target", "mean_pred"],
                               ["MAE by Group", "Actual Mean Target", "Predicted Mean Target"]):
        fairness_stats[col].plot.bar(ax=ax, color=["steelblue", "coral"],
                                      edgecolor="white", rot=0)
        ax.set_title(title)
    plt.tight_layout()
    save_fig(fig, fp, "28_fairness.png")

    # ── 47. Privacy & Security ────────────────────────────────────────────────
    print("\n[47] Privacy & Security — Membership Inference Resilience")
    tr_mse = float(mean_squared_error(y_train, gbr_cv.best_estimator_.predict(X_train)))
    te_mse = float(mean_squared_error(y_test, gbr_preds))
    gap    = (te_mse - tr_mse) / te_mse
    status = "⚠ Overfit risk" if gap > 0.3 else "✓ Acceptable"
    print(f"  Train MSE={tr_mse:.1f}, Test MSE={te_mse:.1f}, "
          f"Generalization Gap={gap:.3f}  {status}")
    print("  Data: all features anonymised — no direct PII present")

    # ── 48. Feature Store ─────────────────────────────────────────────────────
    print("\n[48] Feature Store")
    feature_store = {
        col: {
            "name": col,
            "dtype": str(X[col].dtype),
            "mean": round(float(X[col].mean()), 6),
            "std":  round(float(X[col].std()),  6),
            "min":  round(float(X[col].min()),  6),
            "max":  round(float(X[col].max()),  6),
            "description": f"Preprocessed {col} from Diabetes dataset",
            "version": "1.0",
            "created": today(),
        }
        for col in feature_names
    }
    save_json(feature_store, ap, "feature_store.json")

    # ── 49. A/B Testing ───────────────────────────────────────────────────────
    print("\n[49] A/B Testing — Ridge vs GBR")
    rng2 = np.random.default_rng(seed)
    assignment = rng2.choice(["A", "B"], size=len(X_test))
    idx_A = assignment == "A"
    idx_B = assignment == "B"
    errors_A = np.abs(y_test.values[idx_A] -
                      ridge_cv.best_estimator_.predict(X_test.iloc[idx_A]))
    errors_B = np.abs(y_test.values[idx_B] -
                      gbr_cv.best_estimator_.predict(X_test.iloc[idx_B]))
    t_stat, p_val = stats.ttest_ind(errors_A, errors_B)
    sig = "→ Significant difference" if p_val < 0.05 else "→ No significant difference"
    print(f"  Model A (Ridge): MAE={errors_A.mean():.2f} (n={idx_A.sum()})")
    print(f"  Model B (GBR)  : MAE={errors_B.mean():.2f} (n={idx_B.sum()})")
    print(f"  t={t_stat:.3f}, p={p_val:.4f}  {sig}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors_A, bins=20, alpha=0.6, color="steelblue",
            label=f"A (Ridge) MAE={errors_A.mean():.1f}")
    ax.hist(errors_B, bins=20, alpha=0.6, color="coral",
            label=f"B (GBR)   MAE={errors_B.mean():.1f}")
    ax.axvline(errors_A.mean(), color="steelblue", linestyle="--", linewidth=2)
    ax.axvline(errors_B.mean(), color="coral",     linestyle="--", linewidth=2)
    ax.set_title(f"Phase 5 — A/B Test Error Distributions (p={p_val:.4f})",
                 fontweight="bold")
    ax.set_xlabel("Absolute Error"); ax.set_ylabel("Count"); ax.legend()
    plt.tight_layout()
    save_fig(fig, fp, "29_ab_testing.png")

    # ── 50. Cost-Aware Regression ─────────────────────────────────────────────
    print("\n[50] Cost-Aware Regression")
    sym_mse   = float(mean_squared_error(y_test, gbr_preds))
    asym_cost = asymmetric_loss(y_test.values, gbr_preds, alpha=cfg.asym_penalty)
    biased    = gbr_preds * cfg.cost_bias_factor
    biased_asym = asymmetric_loss(y_test.values, biased, alpha=cfg.asym_penalty)
    biased_mse  = float(mean_squared_error(y_test, biased))
    print(f"  Standard   — MSE={sym_mse:.2f}, Asym={asym_cost:.2f}")
    print(f"  Cost-aware — MSE={biased_mse:.2f}, Asym={biased_asym:.2f} "
          f"(bias={cfg.cost_bias_factor}x)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 5 — Cost-Aware Regression", fontweight="bold")
    axes[0].hist(y_test.values - gbr_preds, bins=25,
                 color="steelblue", edgecolor="white", alpha=0.85,
                 label=f"MSE={sym_mse:.1f}")
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_title("Standard GBR Residuals"); axes[0].legend()
    axes[1].hist(y_test.values - biased, bins=25,
                 color="coral", edgecolor="white", alpha=0.85,
                 label=f"Asym={biased_asym:.1f}")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title(f"Cost-Aware ({cfg.cost_bias_factor}x upward bias)"); axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Residual"); ax.set_ylabel("Count")
    plt.tight_layout()
    save_fig(fig, fp, "30_cost_aware.png")

    print("\n  ✅ Phase 5 complete.")
    ctx.update(dict(
        qr_models=qr_models, coverage=coverage, test_preds=test_preds,
        cp_lower=cp_lower, cp_upper=cp_upper, fairness_stats=fairness_stats,
        errors_A=errors_A, errors_B=errors_B, p_val=p_val,
    ))
    return ctx
