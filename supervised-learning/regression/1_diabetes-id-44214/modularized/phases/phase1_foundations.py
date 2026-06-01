"""
phases/phase1_foundations.py
Phases 1–10: Regression Basics → EDA → Feature Types → Target Distribution
             → Train/Test Split → Baseline Models → Loss Functions
             → Residual Analysis → Bias-Variance → Overfitting/Underfitting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from pathlib import Path

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score

from utils.helpers import save_fig, regression_metrics, set_plot_style


def run(cfg) -> dict:
    """
    Run all Phase 1 steps.
    Returns a context dict passed forward to subsequent phases.
    """
    set_plot_style()
    fp = cfg.figures_dir
    seed = cfg.random_seed

    print("\n" + "=" * 60)
    print("  PHASE 1 — FOUNDATIONS")
    print("=" * 60)

    # ── 1. Load & problem framing ─────────────────────────────────────────────
    print("\n[1] Regression Basics")
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    feature_names = list(data.feature_names)
    TARGET = "target"

    print(f"  Dataset : Diabetes (OpenML 44214 / sklearn)")
    print(f"  Samples : {df.shape[0]}  |  Features : {len(feature_names)}")
    print(f"  Target  : {TARGET} — quantitative measure of disease progression")
    print(f"  Type    : Supervised Regression")

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    print("\n[2] EDA — Feature Distributions")
    print(df[feature_names + [TARGET]].describe().round(4))

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle("Phase 1 — EDA: Feature Distributions", fontsize=16,
                 fontweight="bold", y=1.01)
    for i, col in enumerate(feature_names + [TARGET]):
        ax = axes[i // 4][i % 4]
        ax.hist(df[col], bins=25, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(col)
    axes[2][3].set_visible(False)
    plt.tight_layout()
    save_fig(fig, fp, "01_eda_distributions.png")

    # Correlation heatmap
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 9})
    ax.set_title("Phase 1 — Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, fp, "02_correlation_heatmap.png")

    # ── 3. Feature types ─────────────────────────────────────────────────────
    print("\n[3] Feature Types & Problem Framing")
    print("  All 10 features are continuous (mean-centred, unit-variance scaled)")
    print("  'sex' is binary (±0.05) — treated as categorical")
    print("  Problem: Regression | Target: continuous positive integer [25-346]")

    # ── 4. Target distribution ───────────────────────────────────────────────
    print("\n[4] Target Distribution")
    print(f"  Mean={df[TARGET].mean():.1f}, Median={df[TARGET].median():.1f}, "
          f"Std={df[TARGET].std():.1f}, Skew={df[TARGET].skew():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 1 — Target Distribution Analysis", fontweight="bold")
    axes[0].hist(df[TARGET], bins=30, color="coral", edgecolor="white", alpha=0.85)
    axes[0].set_title("Histogram"); axes[0].set_xlabel("Target")
    stats.probplot(df[TARGET], dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (Normal)")
    axes[2].boxplot(df[TARGET], patch_artist=True,
                    boxprops=dict(facecolor="lightblue", color="steelblue"))
    axes[2].set_title("Boxplot"); axes[2].set_ylabel("Target")
    plt.tight_layout()
    save_fig(fig, fp, "03_target_distribution.png")

    # Log transform comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Phase 1 — Original vs Log-Transformed Target", fontweight="bold")
    axes[0].hist(df[TARGET], bins=30, color="coral", edgecolor="white", alpha=0.85)
    axes[0].set_title("Original Target")
    axes[1].hist(np.log(df[TARGET]), bins=30, color="mediumseagreen",
                 edgecolor="white", alpha=0.85)
    axes[1].set_title("Log(Target)")
    plt.tight_layout()
    save_fig(fig, fp, "04_target_log_transform.png")

    # ── 5. Train/test split ───────────────────────────────────────────────────
    print("\n[5] Train/Test Split")
    X = df[feature_names]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=seed)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 1 — Train/Test Split", fontweight="bold")
    axes[0].bar(["Train", "Test"], [len(y_train), len(y_test)],
                color=["steelblue", "coral"], edgecolor="white", width=0.5)
    axes[0].set_title("Split Sizes"); axes[0].set_ylabel("Samples")
    axes[1].hist(y_train, bins=25, alpha=0.6, label="Train", color="steelblue")
    axes[1].hist(y_test, bins=25, alpha=0.6, label="Test", color="coral")
    axes[1].legend(); axes[1].set_title("Target Distributions after Split")
    plt.tight_layout()
    save_fig(fig, fp, "05_train_test_split.png")

    # ── 6. Baseline models ────────────────────────────────────────────────────
    print("\n[6] Baseline Models")
    baselines = {
        "Mean Predictor": DummyRegressor(strategy="mean"),
        "Median Predictor": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
    }
    bl_results = {}
    for name, model in baselines.items():
        model.fit(X_train, y_train)
        bl_results[name] = regression_metrics(y_test, model.predict(X_test), name)

    # ── 7. Loss functions ─────────────────────────────────────────────────────
    print("\n[7] Loss Functions")
    y_range = np.linspace(-200, 200, 400)
    delta = 50
    huber = np.where(np.abs(y_range) <= delta,
                     0.5 * y_range ** 2, delta * (np.abs(y_range) - 0.5 * delta))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 1 — Loss Function Shapes", fontweight="bold")
    axes[0].plot(y_range, y_range ** 2, color="steelblue", linewidth=2)
    axes[0].set_title("MSE (L2)"); axes[0].set_xlabel("Residual"); axes[0].set_ylabel("Loss")
    axes[1].plot(y_range, np.abs(y_range), color="coral", linewidth=2)
    axes[1].set_title("MAE (L1)"); axes[1].set_xlabel("Residual")
    axes[2].plot(y_range, huber, color="mediumseagreen", linewidth=2)
    axes[2].set_title(f"Huber (δ={delta})"); axes[2].set_xlabel("Residual")
    plt.tight_layout()
    save_fig(fig, fp, "06_loss_functions.png")

    # ── 8. Residual analysis ──────────────────────────────────────────────────
    print("\n[8] Residual Analysis")
    lr = baselines["Linear Regression"]
    lr_preds = lr.predict(X_test)
    residuals = y_test.values - lr_preds

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Phase 1 — Residual Analysis (Linear Regression)", fontweight="bold")
    axes[0][0].scatter(lr_preds, residuals, alpha=0.5, color="steelblue", edgecolors="none")
    axes[0][0].axhline(0, color="red", linestyle="--")
    axes[0][0].set_title("Residuals vs Fitted"); axes[0][0].set_xlabel("Fitted")
    stats.probplot(residuals, dist="norm", plot=axes[0][1])
    axes[0][1].set_title("Q-Q Plot of Residuals")
    axes[0][2].hist(residuals, bins=25, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0][2].axvline(0, color="red", linestyle="--")
    axes[0][2].set_title("Residual Distribution")
    axes[1][0].scatter(lr_preds, np.sqrt(np.abs(residuals)), alpha=0.5, color="coral", edgecolors="none")
    axes[1][0].set_title("Scale-Location"); axes[1][0].set_xlabel("Fitted")
    axes[1][1].scatter(y_test, lr_preds, alpha=0.5, color="mediumseagreen", edgecolors="none")
    axes[1][1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    axes[1][1].set_title("Predicted vs Actual")
    axes[1][2].plot(residuals, "o-", alpha=0.4, color="purple", markersize=3)
    axes[1][2].axhline(0, color="red", linestyle="--")
    axes[1][2].set_title("Residuals vs Index")
    plt.tight_layout()
    save_fig(fig, fp, "07_residual_analysis.png")

    # ── 9. Bias vs Variance ───────────────────────────────────────────────────
    print("\n[9] Bias vs Variance — Learning Curves")
    models_bv = {
        "High Bias (Linear)": LinearRegression(),
        "Balanced (Ridge)": Ridge(alpha=1.0),
        "High Variance (Deep Tree)": DecisionTreeRegressor(max_depth=None, random_state=seed),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Phase 1 — Bias-Variance via Learning Curves", fontweight="bold")
    for ax, (name, model) in zip(axes, models_bv.items()):
        sizes, tr_sc, val_sc = learning_curve(
            model, X, y, cv=5, scoring="neg_mean_squared_error",
            train_sizes=np.linspace(0.1, 1.0, 10), random_state=seed)
        ax.plot(sizes, np.sqrt(-tr_sc.mean(1)), "o-", label="Train", color="steelblue")
        ax.plot(sizes, np.sqrt(-val_sc.mean(1)), "o-", label="Val", color="coral")
        ax.fill_between(sizes,
                        np.sqrt(-tr_sc.mean(1)) - np.sqrt(-tr_sc).std(1),
                        np.sqrt(-tr_sc.mean(1)) + np.sqrt(-tr_sc).std(1),
                        alpha=0.12, color="steelblue")
        ax.set_title(name); ax.set_xlabel("Training Samples"); ax.set_ylabel("RMSE"); ax.legend()
    plt.tight_layout()
    save_fig(fig, fp, "08_bias_variance.png")

    # ── 10. Overfitting vs Underfitting ───────────────────────────────────────
    print("\n[10] Overfitting vs Underfitting — Depth Sweep")
    depths = range(1, 20)
    tr_rmse, te_rmse = [], []
    for d in depths:
        m = DecisionTreeRegressor(max_depth=d, random_state=seed)
        m.fit(X_train, y_train)
        tr_rmse.append(np.sqrt(mean_squared_error(y_train, m.predict(X_train))))
        te_rmse.append(np.sqrt(mean_squared_error(y_test, m.predict(X_test))))
    best_d = list(depths)[int(np.argmin(te_rmse))]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(depths, tr_rmse, "o-", label="Train RMSE", color="steelblue")
    ax.plot(depths, te_rmse, "o-", label="Test RMSE", color="coral")
    ax.axvline(best_d, color="green", linestyle="--", label=f"Best depth={best_d}")
    ax.set_title("Phase 1 — Overfitting vs Underfitting (Tree Depth Sweep)",
                 fontweight="bold")
    ax.set_xlabel("Max Depth"); ax.set_ylabel("RMSE"); ax.legend()
    plt.tight_layout()
    save_fig(fig, fp, "09_overfit_underfit.png")

    print("\n  ✅ Phase 1 complete.")
    return dict(df=df, feature_names=feature_names, TARGET=TARGET,
                X=X, y=y, X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                baselines=baselines, bl_results=bl_results)
