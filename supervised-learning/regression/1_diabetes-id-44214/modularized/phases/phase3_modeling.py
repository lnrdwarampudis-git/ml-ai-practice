"""
phases/phase3_modeling.py
Phases 21–30: Pipelines → Model Families → Model Comparison → Hyperparameter Tuning
              → Feature Selection → Experiment Tracking → Interpretability
              → Error Analysis → Segment-Level Analysis → Reproducibility
"""

import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                   HuberRegressor, BayesianRidge)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, learning_curve)
from sklearn.feature_selection import (SelectKBest, f_regression,
                                        RFE, mutual_info_regression)
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

from utils.helpers import save_fig, regression_metrics, save_json, now, set_plot_style

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def run(cfg, ctx: dict) -> dict:
    set_plot_style()
    fp = cfg.figures_dir
    ap = cfg.artifacts_dir
    mp = cfg.models_dir
    seed = cfg.random_seed

    feature_names = ctx["feature_names"]
    X_train = ctx["X_train"]; X_test = ctx["X_test"]
    y_train = ctx["y_train"]; y_test  = ctx["y_test"]
    X = ctx["X"]; y = ctx["y"]

    print("\n" + "=" * 60)
    print("  PHASE 3 — MODELING & ENGINEERING")
    print("=" * 60)

    # ── 21. Pipelines ─────────────────────────────────────────────────────────
    print("\n[21] Pipelines")
    pipe_lr = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    pipe_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
        ("model",  Ridge(alpha=1.0)),
    ])
    for name, pipe in [("Linear", pipe_lr), ("Ridge+Poly", pipe_ridge)]:
        pipe.fit(X_train, y_train)
        r2 = r2_score(y_test, pipe.predict(X_test))
        print(f"  Pipeline [{name}]: R²={r2:.3f}")

    # ── 22 & 23. Model Families + Comparison ─────────────────────────────────
    print("\n[22] Model Families")
    model_zoo = {
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=1.0),
        "Lasso":            Lasso(alpha=0.1),
        "ElasticNet":       ElasticNet(alpha=0.1, l1_ratio=0.5),
        "SVR(rbf)":         Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf"))]),
        "KNN(k=5)":         KNeighborsRegressor(n_neighbors=5),
        "DecisionTree":     DecisionTreeRegressor(max_depth=5, random_state=seed),
        "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=seed),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=seed),
        "HuberRegressor":   HuberRegressor(max_iter=300),
        "BayesianRidge":    BayesianRidge(),
    }
    if HAS_XGB:
        model_zoo["XGBoost"] = xgb.XGBRegressor(n_estimators=100, random_state=seed, verbosity=0)
    if HAS_LGB:
        model_zoo["LightGBM"] = lgb.LGBMRegressor(n_estimators=100, random_state=seed, verbose=-1)

    zoo_results = {}
    for name, model in model_zoo.items():
        model.fit(X_train, y_train)
        zoo_results[name] = regression_metrics(y_test, model.predict(X_test), name)

    print("\n[23] Model Comparison")
    zoo_df = pd.DataFrame(zoo_results).T.sort_values("RMSE")
    print(zoo_df.round(3))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Phase 3 — Model Comparison", fontweight="bold")
    for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
        zoo_df[metric].plot.barh(
            ax=ax,
            color=["steelblue" if metric == "R2" else "coral"] * len(zoo_df),
            edgecolor="white")
        ax.set_title(metric); ax.set_xlabel(metric)
    plt.tight_layout()
    save_fig(fig, fp, "16_model_comparison.png")

    best_name = zoo_df["RMSE"].idxmin()
    best_model = model_zoo[best_name]
    print(f"\n  Best model (lowest RMSE): {best_name}")

    # ── 24. Hyperparameter Tuning ─────────────────────────────────────────────
    print("\n[24] Hyperparameter Tuning")
    ridge_cv = GridSearchCV(
        Ridge(), {"alpha": cfg.ridge_alphas},
        cv=cfg.cv_folds, scoring="neg_root_mean_squared_error", n_jobs=-1)
    ridge_cv.fit(X_train, y_train)
    print(f"  Ridge best α={ridge_cv.best_params_['alpha']}, "
          f"CV-RMSE={-ridge_cv.best_score_:.2f}")

    gbr_cv = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=seed),
        cfg.gbr_param_grid, n_iter=cfg.gbr_n_iter, cv=cfg.cv_folds,
        scoring="neg_root_mean_squared_error", random_state=seed, n_jobs=-1)
    gbr_cv.fit(X_train, y_train)
    print(f"  GBR best params: {gbr_cv.best_params_}, "
          f"CV-RMSE={-gbr_cv.best_score_:.2f}")

    alphas = cfg.ridge_alphas + [1000]
    alpha_rmse = [-cross_val_score(Ridge(alpha=a), X_train, y_train,
                                    cv=5, scoring="neg_root_mean_squared_error").mean()
                  for a in alphas]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(alphas, alpha_rmse, "o-", color="steelblue", linewidth=2)
    ax.axvline(ridge_cv.best_params_["alpha"], color="red", linestyle="--",
               label=f"Best α={ridge_cv.best_params_['alpha']}")
    ax.set_title("Phase 3 — Ridge α Sensitivity (CV-RMSE)", fontweight="bold")
    ax.set_xlabel("Alpha (log)"); ax.set_ylabel("CV-RMSE"); ax.legend()
    plt.tight_layout()
    save_fig(fig, fp, "17_hyperparameter_tuning.png")

    # ── 25. Feature Selection ─────────────────────────────────────────────────
    print("\n[25] Feature Selection")
    f_scores = pd.Series(
        SelectKBest(f_regression, k=6).fit(X_train, y_train).scores_,
        index=feature_names).sort_values(ascending=False)
    mi_scores = pd.Series(
        mutual_info_regression(X_train, y_train, random_state=seed),
        index=feature_names).sort_values(ascending=False)
    rfe = RFE(Ridge(alpha=1.0), n_features_to_select=6)
    rfe.fit(X_train, y_train)
    rfe_support = pd.Series(rfe.support_.astype(int), index=feature_names)
    perm_imp = permutation_importance(gbr_cv.best_estimator_, X_test, y_test,
                                       n_repeats=30, random_state=seed)
    perm_df = pd.Series(perm_imp.importances_mean, index=feature_names).sort_values(ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Phase 3 — Feature Selection Methods", fontweight="bold")
    f_scores.plot.barh(ax=axes[0][0], color="steelblue", edgecolor="white")
    axes[0][0].set_title("F-Regression Scores")
    mi_scores.plot.barh(ax=axes[0][1], color="coral", edgecolor="white")
    axes[0][1].set_title("Mutual Information")
    rfe_support.plot.barh(ax=axes[1][0], color="mediumseagreen", edgecolor="white")
    axes[1][0].set_title("RFE Selection (1=Selected)")
    perm_df.plot.barh(ax=axes[1][1], color="orchid", edgecolor="white")
    axes[1][1].set_title("Permutation Importance")
    plt.tight_layout()
    save_fig(fig, fp, "18_feature_selection.png")

    # ── 26. Experiment Tracking ───────────────────────────────────────────────
    print("\n[26] Experiment Tracking")
    exp_log = []
    for name, model in list(model_zoo.items())[:6]:
        preds = model.predict(X_test)
        m = regression_metrics(y_test, preds)
        exp_log.append({
            "run_id": hashlib.md5(f"{name}{seed}".encode()).hexdigest()[:8],
            "model": name,
            "params": str(model.get_params())[:80],
            "rmse": m["RMSE"], "r2": m["R2"],
            "timestamp": now(),
        })
    exp_df = pd.DataFrame(exp_log)
    exp_path = ap / "experiment_log.csv"
    exp_df.to_csv(exp_path, index=False)
    print(f"  Saved → {exp_path}")
    print(exp_df[["run_id", "model", "rmse", "r2"]].to_string(index=False))

    # ── 27. Interpretability ──────────────────────────────────────────────────
    print("\n[27] Interpretability — Ridge Coefficients + SHAP")
    best_ridge = ridge_cv.best_estimator_
    ridge_coefs = pd.Series(best_ridge.coef_, index=feature_names).sort_values()

    n_plots = 2 if HAS_SHAP else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(16 if HAS_SHAP else 8, 6))
    if n_plots == 1:
        axes = [axes]
    ridge_coefs.plot.barh(
        ax=axes[0],
        color=["coral" if v < 0 else "steelblue" for v in ridge_coefs],
        edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Ridge Coefficients", fontweight="bold")

    if HAS_SHAP:
        import warnings; warnings.filterwarnings("ignore")
        explainer = shap.LinearExplainer(best_ridge, X_train)
        shap_vals = explainer.shap_values(X_test)
        shap_mean = pd.DataFrame(np.abs(shap_vals), columns=feature_names).mean().sort_values()
        shap_mean.plot.barh(ax=axes[1], color="orchid", edgecolor="white")
        axes[1].set_title("Mean |SHAP| Values", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, fp, "19_interpretability.png")

    # ── 28. Error Analysis ────────────────────────────────────────────────────
    print("\n[28] Error Analysis")
    gbr_preds = gbr_cv.best_estimator_.predict(X_test)
    err_df = X_test.copy()
    err_df["y_true"]    = y_test.values
    err_df["y_pred"]    = gbr_preds
    err_df["error"]     = err_df["y_true"] - err_df["y_pred"]
    err_df["abs_error"] = err_df["error"].abs()
    err_df["pct_error"] = (err_df["abs_error"] / err_df["y_true"]) * 100

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Phase 3 — Error Analysis (Tuned GBR)", fontweight="bold")
    axes[0][0].scatter(err_df["y_true"], err_df["error"], alpha=0.5, color="steelblue", edgecolors="none")
    axes[0][0].axhline(0, color="red", linestyle="--"); axes[0][0].set_title("Error vs True")
    axes[0][1].hist(err_df["error"], bins=25, color="coral", edgecolor="white", alpha=0.85)
    axes[0][1].set_title("Error Distribution")
    axes[0][2].scatter(err_df["bmi"], err_df["abs_error"], alpha=0.5, color="mediumseagreen", edgecolors="none")
    axes[0][2].set_title("Abs Error vs BMI")
    sc = axes[1][0].scatter(err_df["y_pred"], err_df["y_true"], alpha=0.5,
                             c=err_df["abs_error"], cmap="RdYlGn_r", edgecolors="none")
    axes[1][0].plot([25, 350], [25, 350], "r--")
    axes[1][0].set_title("Predicted vs Actual")
    plt.colorbar(sc, ax=axes[1][0], label="|Error|")
    axes[1][1].scatter(err_df["age"], err_df["abs_error"], alpha=0.5, color="orchid", edgecolors="none")
    axes[1][1].set_title("Abs Error vs Age")
    axes[1][2].hist(err_df["pct_error"], bins=25, color="goldenrod", edgecolor="white", alpha=0.85)
    axes[1][2].set_title("% Error Distribution")
    plt.tight_layout()
    save_fig(fig, fp, "20_error_analysis.png")

    # ── 29. Segment-Level Analysis ────────────────────────────────────────────
    print("\n[29] Segment-Level Analysis")
    err_df["target_group"] = pd.cut(err_df["y_true"], bins=3,
                                     labels=["Low Risk", "Mid Risk", "High Risk"])
    err_df["bmi_group"]    = pd.cut(err_df["bmi"], bins=3,
                                     labels=["Low BMI", "Mid BMI", "High BMI"])
    seg_stats = err_df.groupby("target_group", observed=True).agg(
        count=("y_true", "count"),
        mean_abs_err=("abs_error", "mean"),
        median_abs_err=("abs_error", "median"),
    ).round(2)
    print(seg_stats)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 3 — Segment-Level Error Analysis", fontweight="bold")
    seg_stats["mean_abs_err"].plot.bar(ax=axes[0], color="coral", edgecolor="white", rot=0)
    axes[0].set_title("MAE by Risk Segment"); axes[0].set_ylabel("Mean Abs Error")
    err_df.groupby("bmi_group", observed=True)["abs_error"].mean().plot.bar(
        ax=axes[1], color="steelblue", edgecolor="white", rot=0)
    axes[1].set_title("MAE by BMI Segment"); axes[1].set_ylabel("Mean Abs Error")
    plt.tight_layout()
    save_fig(fig, fp, "21_segment_analysis.png")

    # ── 30. Reproducibility ───────────────────────────────────────────────────
    print("\n[30] Reproducibility Record")
    import sklearn
    repro = {
        "random_seed": seed,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "train_hash": hashlib.sha256(X_train.to_csv().encode()).hexdigest()[:12],
        "test_hash":  hashlib.sha256(X_test.to_csv().encode()).hexdigest()[:12],
        "best_model": best_name,
    }
    save_json(repro, ap, "reproducibility.json")

    print("\n  ✅ Phase 3 complete.")
    ctx.update(dict(
        model_zoo=model_zoo, zoo_df=zoo_df, best_name=best_name, best_model=best_model,
        ridge_cv=ridge_cv, gbr_cv=gbr_cv, gbr_preds=gbr_preds, err_df=err_df,
    ))
    return ctx
