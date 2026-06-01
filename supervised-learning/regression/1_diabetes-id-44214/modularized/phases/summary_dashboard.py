"""
phases/summary_dashboard.py
Generates the final multi-panel summary dashboard.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score

from utils.helpers import save_fig, set_plot_style


def run(cfg, ctx: dict):
    set_plot_style()
    fp = cfg.figures_dir
    seed = cfg.random_seed

    feature_names  = ctx["feature_names"]
    X = ctx["X"]; y = ctx["y"]
    X_test = ctx["X_test"]; y_test = ctx["y_test"]
    gbr_cv    = ctx["gbr_cv"]
    gbr_preds = ctx["gbr_preds"]
    zoo_df    = ctx["zoo_df"]
    best_name = ctx["best_name"]
    coverage  = ctx["coverage"]
    test_preds = ctx["test_preds"]
    cp_lower   = ctx["cp_lower"]
    cp_upper   = ctx["cp_upper"]
    qr_models  = ctx["qr_models"]
    errors_A   = ctx["errors_A"]
    errors_B   = ctx["errors_B"]
    p_val      = ctx["p_val"]
    cv_scores  = ctx["cv_scores"]
    ridge_cv   = ctx["ridge_cv"]

    print("\n" + "=" * 60)
    print("  SUMMARY DASHBOARD")
    print("=" * 60)

    fig = plt.figure(figsize=(24, 16))
    fig.suptitle("OpenML Diabetes 44214 — Complete ML Pipeline Summary",
                 fontsize=18, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)
    idx_sorted = np.argsort(y_test.values)

    # 1. Model RMSE ranking
    ax1 = fig.add_subplot(gs[0, 0])
    zoo_top = zoo_df.head(8)
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(zoo_top))]
    zoo_top["RMSE"].plot.barh(ax=ax1, color=colors, edgecolor="white")
    ax1.set_title("Model RMSE Ranking", fontweight="bold"); ax1.set_xlabel("RMSE")

    # 2. Predicted vs Actual
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(y_test, gbr_preds, alpha=0.5,
                     c=np.abs(y_test - gbr_preds), cmap="RdYlGn_r", edgecolors="none")
    ax2.plot([25, 350], [25, 350], "r--", linewidth=2)
    plt.colorbar(sc, ax=ax2, label="|Error|")
    ax2.set_title(f"Predicted vs Actual (R²={zoo_df.loc[best_name,'R2']:.3f})",
                  fontweight="bold")
    ax2.set_xlabel("Actual"); ax2.set_ylabel("Predicted")

    # 3. Feature importance
    ax3 = fig.add_subplot(gs[0, 2])
    if hasattr(gbr_cv.best_estimator_, "feature_importances_"):
        import pandas as pd
        fi = pd.Series(gbr_cv.best_estimator_.feature_importances_,
                       index=feature_names).sort_values()
        fi.plot.barh(ax=ax3, color="steelblue", edgecolor="white")
        ax3.set_title("GBR Feature Importances", fontweight="bold")

    # 4. Residual histogram
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(y_test.values - gbr_preds, bins=25, color="coral",
             edgecolor="white", alpha=0.85)
    ax4.axvline(0, color="black", linestyle="--")
    ax4.set_title("Residual Distribution", fontweight="bold")
    ax4.set_xlabel("Residual")

    # 5. Learning curve
    ax5 = fig.add_subplot(gs[1, 0:2])
    sizes, tr_sc, val_sc = learning_curve(
        gbr_cv.best_estimator_, X, y, cv=5,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=seed)
    ax5.plot(sizes, -tr_sc.mean(1), "o-", label="Train RMSE", color="steelblue")
    ax5.plot(sizes, -val_sc.mean(1), "o-", label="CV RMSE", color="coral")
    ax5.fill_between(sizes,
                     -tr_sc.mean(1) - tr_sc.std(1),
                     -tr_sc.mean(1) + tr_sc.std(1), alpha=0.1, color="steelblue")
    ax5.set_title("Best Model Learning Curve", fontweight="bold")
    ax5.set_xlabel("Training Samples"); ax5.set_ylabel("RMSE"); ax5.legend()

    # 6. Conformal PI
    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.scatter(range(len(y_test)), y_test.values[idx_sorted],
                color="black", s=12, zorder=5, label="Actual")
    ax6.plot(range(len(test_preds)), test_preds[idx_sorted],
             color="steelblue", linewidth=1.5, label="Predicted")
    ax6.fill_between(range(len(y_test)), cp_lower[idx_sorted], cp_upper[idx_sorted],
                     alpha=0.2, color="steelblue",
                     label=f"90% CI (cov={coverage:.2f})")
    ax6.set_title("Conformal Prediction Intervals", fontweight="bold")
    ax6.set_xlabel("Sample"); ax6.set_ylabel("Target"); ax6.legend(fontsize=9)

    # 7. Quantile bands
    ax7 = fig.add_subplot(gs[2, 0:2])
    q10 = qr_models[0.1].predict(X_test)
    q90 = qr_models[0.9].predict(X_test)
    q50 = qr_models[0.5].predict(X_test)
    ax7.fill_between(range(len(y_test)), q10[idx_sorted], q90[idx_sorted],
                     alpha=0.2, color="steelblue", label="Q10-Q90 Band")
    ax7.plot(range(len(y_test)), q50[idx_sorted],
             color="steelblue", linewidth=2, label="Q50 (Median)")
    ax7.scatter(range(len(y_test)), y_test.values[idx_sorted],
                color="black", s=10, zorder=5, label="Actual")
    ax7.set_title("Quantile Regression Bands", fontweight="bold")
    ax7.set_xlabel("Sample"); ax7.set_ylabel("Target"); ax7.legend(fontsize=9)

    # 8. A/B test
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.hist(errors_A, bins=15, alpha=0.6, color="steelblue",
             label=f"Ridge  MAE={errors_A.mean():.1f}")
    ax8.hist(errors_B, bins=15, alpha=0.6, color="coral",
             label=f"GBR    MAE={errors_B.mean():.1f}")
    ax8.set_title(f"A/B Test (p={p_val:.3f})", fontweight="bold")
    ax8.set_xlabel("Abs Error"); ax8.legend(fontsize=9)

    # 9. Metrics summary table
    ax9 = fig.add_subplot(gs[2, 3])
    ax9.axis("off")
    table_data = [
        ["Metric",          "Value"],
        ["Best Model",      best_name[:15]],
        ["Test RMSE",       f"{zoo_df.loc[best_name,'RMSE']:.2f}"],
        ["Test MAE",        f"{zoo_df.loc[best_name,'MAE']:.2f}"],
        ["Test R²",         f"{zoo_df.loc[best_name,'R2']:.3f}"],
        ["CV RMSE",         f"{-cv_scores.mean():.2f}±{cv_scores.std():.2f}"],
        ["Conf. Coverage",  f"{coverage:.2f}"],
        ["A/B p-value",     f"{p_val:.4f}"],
        ["n (total)",       "442 (353 train)"],
    ]
    tbl = ax9.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc="center", loc="center", colWidths=[0.55, 0.45])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for i in range(len(table_data)):
        for j in range(2):
            cell = tbl[i, j]
            if i == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif i % 2 == 0:
                cell.set_facecolor("#ecf0f1")
    ax9.set_title("Summary Metrics", fontweight="bold")

    save_fig(fig, fp, "00_summary_dashboard.png", bbox_inches="tight")
    print("  📊 Summary dashboard saved.")
