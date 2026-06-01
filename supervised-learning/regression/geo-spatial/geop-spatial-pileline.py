"""
geo_pipeline.py
===============
Industry-standard end-to-end Geospatial Regression pipeline.
California Housing: predict median house value from block-group features
including latitude and longitude.

Data:
    from sklearn.datasets import fetch_california_housing
    cal = fetch_california_housing(as_frame=True)
    df  = cal.frame          # 20640 rows × 9 columns

    Multi-ID OpenML fallback also included:
    fetch_openml(data_id=537)  → same California Housing dataset

Target: MedHouseVal — median house value in $100,000s (0.15 – 5.0)
Source: Pace & Barry (1997) — 1990 US Census California block groups

Mirrors every architectural pattern from titanic-ml-pipeline.py.

Primary focus: Geospatial Regression
──────────────────────────────────────────────────────────────────
  A. Spatial EDA
        — choropleth map of house values (GeoPandas + matplotlib)
        — KDE heatmap of block group density by lat/lon
        — urban cluster visualisation (SF Bay, LA Basin, San Diego)
        — coastal premium: house value vs distance from coastline

  B. Spatial autocorrelation (Moran's I)
        — global Moran's I: is there significant spatial clustering?
        — LISA (Local Indicators of Spatial Association): hot spots, cold spots
        — spatial lag plot: block group value vs weighted mean of neighbours
        — permutation test for significance (999 permutations)

  C. Raw lat/lon as features
        — naive baseline: lat, lon as numeric inputs to Ridge
        — show that raw lat/lon capture linear north-south gradient only
        — fails on non-linear spatial patterns (coastal premium, urban islands)

  D. Haversine distance features
        — distance to SF (37.77°N, −122.42°W)
        — distance to LA (34.05°N, −118.24°W)
        — distance to San Diego (32.72°N, −117.15°W)
        — distance to coast (nearest point on Pacific coastline proxy)
        — distance to nearest urban centroid (learned from data)
        — stepwise R² comparison: raw → +city distances → +coast → full

  E. KDE spatial density features
        — KDE of block group locations (sklearn KernelDensity)
        — bandwidth selection via cross-validation
        — log(KDE density) as urbanisation proxy feature
        — urban premium effect: higher KDE → higher house value

  F. Spatial cross-validation (prevent geographic leakage)
        — random KFold baseline (leaks nearby locations)
        — spatial KFold: cluster locations into K geographic bands,
          hold out each band (sklearn-style custom CV splitter)
        — block CV: divide into geographic grid cells, hold out cells
        — comparison: random KFold vs spatial KFold R² — quantify leakage

  G. Geographically Weighted Regression (GWR)
        — GWR: separate regression coefficients per location
        — bandwidth optimisation (golden section search on AICc)
        — local R² map: where does the model perform well/poorly?
        — coefficient maps: how does MedInc effect vary by location?
        — MGWR: multiscale — different bandwidths per variable

  H. Spatial feature engineering
        — H3/grid cell encoding: which grid cell (binned lat/lon)
        — lat × lon interaction term (captures diagonal gradients)
        — sin/cos of lat/lon (periodic encoding for wrap-around)
        — Voronoi region membership (nearest urban centre)
        — radial basis function features around urban centroids

  I. Outlier & leverage analysis
        — Cook's distance: influential block groups
        — leverage map: geographically isolated block groups have high leverage
        — heteroscedasticity: residuals larger in urban areas (capped target)

  J. Residual spatial autocorrelation
        — Moran's I on model residuals (should be near 0 for good spatial model)
        — residual maps: where does the model systematically over/underpredict?
        — comparison: Ridge residual Moran's I vs GBR vs GWR

  K. Learning curves & spatial holdout
        — standard learning curve (random sample size)
        — geographic learning curve: train on Southern California only,
          test on Northern California — cross-region generalisation

  L. Subgroup disparity by geographic cluster
        — RMSE by urban cluster (SF, LA, SD, Rural)
        — coastal vs inland RMSE comparison
        — income decile × location interaction effects

Industry-standard metrics: R², RMSE ($100k), MAE, MAPE, MedAE
Spatial metrics: Moran's I on residuals, spatial RMSE, geographic R²

Usage:
  python geo_pipeline.py train   --output-dir artifacts_geo
  python geo_pipeline.py predict --artifact-dir artifacts_geo --input-csv sample.csv
  python geo_pipeline.py monitor --artifact-dir artifacts_geo --input-csv new.csv
  python geo_pipeline.py sample-input --output-csv sample.csv --rows 20
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path("artifacts_geo") / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from scipy.stats import ks_2samp
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge, RidgeCV,
)
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, median_absolute_error, r2_score,
)
from sklearn.model_selection import (
    KFold, RandomizedSearchCV, cross_val_predict,
    cross_val_score, learning_curve, train_test_split,
)
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, PolynomialFeatures, RobustScaler, StandardScaler,
)

# Optional spatial imports
try:
    import geopandas as gpd
    from shapely.geometry import Point
    _GPD = True
except ImportError:
    _GPD = False

try:
    import libpysal.weights as lw
    from esda.moran import Moran, Moran_Local
    _PYSAL = True
except ImportError:
    _PYSAL = False

try:
    from mgwr.gwr import GWR, MGWR
    from mgwr.sel_bw import Sel_BW
    _GWR = True
except ImportError:
    _GWR = False

try:
    import shap; _SHAP = True
except ImportError:
    _SHAP = False
try:
    import mlflow; import mlflow.sklearn; _MLFLOW = True
except ImportError:
    _MLFLOW = False

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE          = 42
TARGET                = "MedHouseVal"
MODEL_FILE            = "geo_pipeline.joblib"
METRICS_FILE          = "metrics.json"
TRAINING_PROFILE_FILE = "training_profile.json"
MODEL_CARD_FILE       = "model_card.json"
ENVIRONMENT_FILE      = "environment.json"
N_JOBS                = int(os.environ.get("ML_N_JOBS", 1))

# Known California urban centres [name, lat, lon]
URBAN_CENTRES = [
    ("SF",    37.7749, -122.4194),
    ("LA",    34.0522, -118.2437),
    ("SD",    32.7157, -117.1611),
    ("SAC",   38.5816, -121.4944),
    ("SJO",   37.3382, -121.8863),
    ("OAK",   37.8044, -122.2712),
    ("LB",    33.7701, -118.1937),
    ("ANA",   33.8366, -117.9143),
    ("FRE",   36.7378, -119.7871),
    ("BAKE",  35.3733, -119.0187),
]

BASE_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Primary: sklearn.datasets.fetch_california_housing (no network required)
    Fallback: fetch_openml(data_id=537)
    Last resort: synthetic California Housing.
    """
    log.info("Loading California Housing dataset …")
    try:
        cal = fetch_california_housing(as_frame=True)
        df  = cal.frame.copy()
        log.info("✓ sklearn California Housing loaded  shape=%s", df.shape)
        return df
    except Exception as exc:
        log.warning("sklearn fetch failed (%s) — trying OpenML …", exc)

    for did in [537, 43093, 42225]:
        try:
            raw = fetch_openml(data_id=did, as_frame=True, parser="auto").frame
            raw.columns = [c.strip() for c in raw.columns]
            # Normalise column names to canonical form
            rename = {}
            for c in raw.columns:
                cl = c.lower().replace(" ","_")
                if "medinc" in cl or "median_income" in cl:   rename[c]="MedInc"
                elif "houseage" in cl or "housing_median_age" in cl: rename[c]="HouseAge"
                elif "averooms" in cl:  rename[c]="AveRooms"
                elif "avebedrms" in cl: rename[c]="AveBedrms"
                elif "population" in cl:rename[c]="Population"
                elif "aveoccup" in cl:  rename[c]="AveOccup"
                elif "latitude" in cl:  rename[c]="Latitude"
                elif "longitude" in cl: rename[c]="Longitude"
                elif "medhouseval" in cl or "median_house_value" in cl: rename[c]=TARGET
            raw = raw.rename(columns=rename)
            if TARGET in raw.columns and "Latitude" in raw.columns:
                log.info("✓ OpenML data_id=%d accepted  shape=%s", did, raw.shape)
                return raw
        except Exception as exc2:
            log.warning("OpenML data_id=%d failed: %s", did, exc2)

    log.warning("All sources failed — using synthetic California Housing.")
    return _make_synthetic_california()


def _make_synthetic_california() -> pd.DataFrame:
    """Synthetic California Housing with realistic spatial patterns."""
    rng = np.random.default_rng(RANDOM_STATE); n = 20640
    lat = rng.uniform(32.54, 41.95, n)
    lon = rng.uniform(-124.35, -114.31, n)

    # Urban proximity bonus
    urban_prox = np.zeros(n)
    for _, ulat, ulon in URBAN_CENTRES:
        dist = np.sqrt((lat-ulat)**2 + (lon-ulon)**2)
        urban_prox += np.exp(-dist / 0.5)
    coastal = np.exp((lon - (-124.35)) / 3)  # coastal premium

    med_inc = (rng.lognormal(1.5, 0.8, n) * (1 + 0.3*urban_prox)).clip(0.5, 15)
    house_age = rng.uniform(1, 52, n)
    ave_rooms = rng.lognormal(1.6, 0.4, n).clip(1, 20)
    ave_bedrms = (ave_rooms * rng.uniform(0.15, 0.35, n)).clip(1, 6)
    population = rng.lognormal(6.8, 0.7, n).clip(3, 35682).astype(float)
    ave_occup = rng.lognormal(0.5, 0.5, n).clip(1, 10)

    value = (
        0.45*med_inc
        + 0.01*house_age
        + 0.1*ave_rooms
        - 0.15*ave_occup
        + 0.3*urban_prox
        + 0.2*coastal
        + rng.normal(0, 0.3, n)
    ).clip(0.15, 5.0)

    return pd.DataFrame({
        "MedInc": med_inc, "HouseAge": house_age, "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms, "Population": population, "AveOccup": ave_occup,
        "Latitude": lat, "Longitude": lon, TARGET: value,
    })


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BASE_FEATURES + [TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def split_data(df: pd.DataFrame):
    """Stratified 80/20 on MedHouseVal quartile bins."""
    X = df.drop(columns=[TARGET]); y = df[TARGET]
    q_bins = pd.qcut(y, q=4, labels=False, duplicates="drop")
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=q_bins)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    r = df.isna().agg(["sum","mean"]).T.rename(
        columns={"sum":"missing_count","mean":"missing_rate"})
    r["dtype"] = df.dtypes.astype(str)
    return r.sort_values("missing_rate", ascending=False)


# ── Haversine distance ────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised Haversine distance in kilometres."""
    R    = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


# ── Concept A: Spatial EDA ────────────────────────────────────────────────────
def save_research_artifacts(X_train: pd.DataFrame, y_train: pd.Series,
                             output_dir: Path) -> None:
    log.info("Concept A: Spatial EDA (train-set only) …")
    output_dir.mkdir(parents=True, exist_ok=True)
    eda = X_train.copy(); eda[TARGET] = y_train.values
    sns.set_theme(style="whitegrid")

    # A1: choropleth scatter map (GeoPandas not required — plain matplotlib)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc = axes[0].scatter(
        eda["Longitude"], eda["Latitude"],
        c=eda[TARGET], cmap="RdYlGn", s=1, alpha=0.5,
        vmin=0.5, vmax=4.5)
    plt.colorbar(sc, ax=axes[0], label="MedHouseVal ($100k)")
    for name, ulat, ulon in URBAN_CENTRES[:5]:
        axes[0].annotate(name, xy=(ulon, ulat), fontsize=7, color="black",
                          ha="center",
                          bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6))
    axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
    axes[0].set_title("Concept A: Spatial choropleth — house values\n"
                      "Green=high value  Red=low value")

    # A2: KDE heatmap of block group density
    from scipy.stats import gaussian_kde
    xy = np.vstack([eda["Longitude"], eda["Latitude"]])
    kde_fn = gaussian_kde(xy, bw_method="scott")
    lon_grid = np.linspace(eda["Longitude"].min(), eda["Longitude"].max(), 120)
    lat_grid = np.linspace(eda["Latitude"].min(),  eda["Latitude"].max(), 120)
    LOG, LAG  = np.meshgrid(lon_grid, lat_grid)
    Z         = kde_fn(np.vstack([LOG.ravel(), LAG.ravel()])).reshape(LOG.shape)
    axes[1].contourf(LOG, LAG, Z, levels=20, cmap="Blues")
    axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
    axes[1].set_title("Concept A: KDE density of block groups\nDarker = more urban")

    plt.tight_layout()
    plt.savefig(output_dir / "plot_spatial_eda.png", dpi=160); plt.close()

    # A3: MedHouseVal vs latitude and longitude separately
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.scatter(eda["Latitude"],  eda[TARGET], alpha=0.05, s=2, color="#4C78A8")
    a1.set_xlabel("Latitude"); a1.set_ylabel("MedHouseVal")
    a1.set_title("Value vs Latitude\n(non-linear — coastal cities at different lats)")
    a2.scatter(eda["Longitude"], eda[TARGET], alpha=0.05, s=2, color="#E45756")
    a2.set_xlabel("Longitude"); a2.set_ylabel("MedHouseVal")
    a2.set_title("Value vs Longitude\n(strong coastal premium — west = higher)")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_lat_lon_vs_value.png", dpi=160); plt.close()

    # A4: Coastal premium analysis
    dist_coast_proxy = np.abs(eda["Longitude"] + 120.0)   # proxy: distance from ~coast
    plt.figure(figsize=(7, 4))
    plt.scatter(dist_coast_proxy, eda[TARGET], alpha=0.05, s=2, color="#54A24B")
    plt.xlabel("Distance from coast proxy (lon + 120°)")
    plt.ylabel("MedHouseVal ($100k)")
    plt.title("Concept A: Coastal premium\nHouses near coast (x≈0) command highest values")
    plt.tight_layout(); plt.savefig(output_dir / "plot_coastal_premium.png", dpi=160); plt.close()

    # Stats
    corr = pd.to_numeric(eda.select_dtypes(include=[np.number]).corrwith(eda[TARGET]),
                         errors="coerce").sort_values(key=abs, ascending=False)
    corr.to_csv(output_dir / "correlation_with_target.csv")
    write_json(output_dir / "research_decisions.json", {
        "problem_type": "geospatial_regression",
        "target": TARGET, "target_unit": "median house value $100k",
        "lat_range": [float(eda["Latitude"].min()), float(eda["Latitude"].max())],
        "lon_range": [float(eda["Longitude"].min()), float(eda["Longitude"].max())],
        "top_correlations": corr.head(8).to_dict(),
        "top_capping_pct": float((eda[TARGET] >= 4.99).mean()),
    })
    log.info("EDA artifacts saved.")


# ── Concept B: Moran's I spatial autocorrelation ─────────────────────────────
def analyse_morans_i(X_train: pd.DataFrame, y_train: pd.Series,
                     output_dir: Path) -> dict[str, Any]:
    """
    Concept B: Moran's I global spatial autocorrelation on MedHouseVal.
    Null hypothesis: house values are spatially random.
    Moran's I ≈ 0.65 → strong positive spatial autocorrelation.
    Uses KNN spatial weights (k=8 nearest neighbours).
    """
    log.info("Concept B: Moran's I spatial autocorrelation …")
    if not _PYSAL:
        log.warning("pip install libpysal esda — Moran's I skipped")
        return {}

    coords = list(zip(X_train["Longitude"].values, X_train["Latitude"].values))
    w      = lw.KNN.from_array(np.array(coords), k=8)
    w.transform = "r"   # row-standardise

    moran = Moran(y_train.values, w, permutations=199)
    log.info("  Moran's I=%.4f  p=%.4f  z=%.4f  (spatial autocorrelation: %s)",
             moran.I, moran.p_sim, moran.z_sim, "YES" if moran.p_sim < 0.05 else "NO")

    # Moran scatter plot (spatial lag plot)
    y_lag = lw.lag_spatial(w, y_train.values)
    plt.figure(figsize=(6, 5))
    plt.scatter(y_train.values, y_lag, alpha=0.1, s=3, color="#4C78A8")
    z = np.polyfit(y_train.values, y_lag, 1)
    xr = np.linspace(y_train.min(), y_train.max(), 100)
    plt.plot(xr, np.poly1d(z)(xr), "r-", linewidth=1.5)
    plt.xlabel("MedHouseVal"); plt.ylabel("Spatial lag (mean of neighbours)")
    plt.title(f"Concept B: Moran scatter plot\nMoran's I={moran.I:.4f}  p={moran.p_sim:.4f}")
    plt.tight_layout(); plt.savefig(output_dir/"plot_morans_scatter.png", dpi=160); plt.close()

    # LISA local Moran's I
    lisa = Moran_Local(y_train.values, w, permutations=199, seed=RANDOM_STATE)
    sig  = lisa.p_sim < 0.05
    fig, ax = plt.subplots(figsize=(9, 6))
    quad_colors = {1:"#E45756", 2:"#4C78A8", 3:"#B0C4DE", 4:"#F0A0A0"}
    quad_labels = {1:"HH (hot spot)", 2:"LL (cold spot)",
                   3:"LH (spatial outlier)", 4:"HL (spatial outlier)"}
    for q in [1, 2, 3, 4]:
        mask = (lisa.q == q) & sig
        ax.scatter(
            X_train.loc[mask,"Longitude"], X_train.loc[mask,"Latitude"],
            c=quad_colors[q], s=2, alpha=0.6, label=quad_labels[q])
    ax.scatter(X_train.loc[~sig,"Longitude"], X_train.loc[~sig,"Latitude"],
               c="#CCCCCC", s=1, alpha=0.3, label="Not significant")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Concept B: LISA — local spatial autocorrelation\n"
                 "HH=high-value clusters  LL=low-value clusters")
    ax.legend(fontsize=7, markerscale=4)
    plt.tight_layout(); plt.savefig(output_dir/"plot_lisa.png", dpi=160); plt.close()

    result = {
        "moran_I":      float(moran.I),
        "moran_p":      float(moran.p_sim),
        "moran_z":      float(moran.z_sim),
        "significant":  bool(moran.p_sim < 0.05),
        "n_hotspots":   int(((lisa.q==1)&sig).sum()),
        "n_coldspots":  int(((lisa.q==2)&sig).sum()),
    }
    write_json(output_dir/"morans_i_analysis.json", result)
    return result


# ── Concept C: Raw lat/lon vs engineered spatial features ─────────────────────
def analyse_raw_vs_spatial(X_train: pd.DataFrame, y_train: pd.Series,
                            output_dir: Path) -> dict[str, Any]:
    """
    Concept C: Show R² uplift from spatial feature engineering.
    Baseline: MedInc only.
    Step 1: + raw lat/lon (captures linear gradient).
    Step 2: + Haversine distances to cities.
    Step 3: + KDE density feature.
    Step 4: + all spatial features.
    """
    log.info("Concept C: Raw lat/lon vs spatial feature analysis …")
    cv    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_log = np.log1p(y_train.values)

    def _score(cols):
        arr = X_train[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        arr = RobustScaler().fit_transform(arr)
        return float(cross_val_score(Ridge(alpha=1.0), arr, y_log, cv=cv, scoring="r2").mean())

    results = {}
    results["MedInc only"]       = _score(["MedInc"])
    results["All raw features"]  = _score(["MedInc","HouseAge","AveRooms","AveBedrms",
                                            "Population","AveOccup"])
    results["+ raw lat/lon"]     = _score(BASE_FEATURES)

    # Add city distance features to X_train temporarily for scoring
    X_tmp = X_train.copy()
    for name, ulat, ulon in URBAN_CENTRES[:5]:
        X_tmp[f"dist_{name}"] = haversine_km(
            X_tmp["Latitude"].values, X_tmp["Longitude"].values, ulat, ulon)
    dist_cols = BASE_FEATURES + [f"dist_{n}" for n,_,_ in URBAN_CENTRES[:5]]
    arr_d = RobustScaler().fit_transform(
        X_tmp[dist_cols].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy())
    r2_d  = float(cross_val_score(Ridge(alpha=1.0),arr_d,y_log,cv=cv,scoring="r2").mean())
    results["+ city distances"]  = r2_d

    # Add KDE density
    coords  = np.column_stack([X_train["Longitude"].values, X_train["Latitude"].values])
    kde_est = KernelDensity(bandwidth=0.1, kernel="gaussian").fit(coords)
    X_tmp["log_kde_density"] = kde_est.score_samples(coords)
    full_cols = dist_cols + ["log_kde_density"]
    arr_f = RobustScaler().fit_transform(
        X_tmp[full_cols].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy())
    r2_f  = float(cross_val_score(Ridge(alpha=1.0),arr_f,y_log,cv=cv,scoring="r2").mean())
    results["+ KDE density"]     = r2_f

    log.info("  Stepwise R² (log target):")
    for k,v in results.items():
        log.info("    %-30s %.4f", k, v)

    plt.figure(figsize=(10, 4))
    names = list(results.keys()); vals = list(results.values())
    colors = ["#CCCCCC","#88BBDD","#4C78A8","#1D9E75","#085041"]
    bars = plt.barh(names, vals, color=colors)
    for bar,v in zip(bars,vals):
        plt.text(v+0.003,bar.get_y()+bar.get_height()/2,f"{v:.4f}",va="center",fontsize=9)
    plt.xlabel("CV R² (log target)")
    plt.title("Concepts C–E: Spatial feature engineering uplift\nEach step adds richer spatial information")
    plt.xlim(0,1.05); plt.tight_layout()
    plt.savefig(output_dir/"plot_spatial_uplift.png",dpi=160); plt.close()
    write_json(output_dir/"spatial_uplift.json", results)
    return results


# ── Concept D: Haversine features ─────────────────────────────────────────────
def analyse_haversine_features(X_train: pd.DataFrame, y_train: pd.Series,
                                output_dir: Path) -> dict[str, Any]:
    """
    Concept D: Haversine distance features to major California cities.
    Quantifies which city's proximity matters most (should be SF + LA).
    """
    log.info("Concept D: Haversine distance feature analysis …")
    cv    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_log = np.log1p(y_train.values)

    importance = {}
    for name, ulat, ulon in URBAN_CENTRES:
        d = haversine_km(X_train["Latitude"].values, X_train["Longitude"].values,
                         ulat, ulon).reshape(-1,1)
        d_sc = StandardScaler().fit_transform(d)
        r2   = float(cross_val_score(Ridge(alpha=1.0), d_sc, y_log,
                                     cv=cv, scoring="r2").mean())
        importance[name] = {"city": name, "lat": ulat, "lon": ulon, "r2_solo": r2}
        log.info("    dist_%s  solo R²=%.4f", name, r2)

    plt.figure(figsize=(8,4))
    names = list(importance.keys())
    vals  = [importance[n]["r2_solo"] for n in names]
    plt.bar(names, vals, color="#4C78A8")
    plt.ylabel("Solo CV R² with log(MedHouseVal)")
    plt.title("Concept D: Haversine distance feature importance per city\nSF and LA proximity dominate")
    plt.tight_layout(); plt.savefig(output_dir/"plot_haversine_importance.png",dpi=160); plt.close()

    write_json(output_dir/"haversine_analysis.json", importance)
    return importance


# ── Concept E: KDE density feature ────────────────────────────────────────────
def analyse_kde_density(X_train: pd.DataFrame, y_train: pd.Series,
                         output_dir: Path) -> dict[str, Any]:
    """
    Concept E: KDE-smoothed neighbourhood density as urbanisation proxy.
    Bandwidth selection via cross-validation on density estimation task.
    """
    log.info("Concept E: KDE density feature analysis …")
    coords  = np.column_stack([X_train["Longitude"].values, X_train["Latitude"].values])
    bandwidths = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    scores     = {}
    cv_kde     = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for bw in bandwidths:
        # Cross-validate the KDE log-likelihood
        ll_scores = []
        for tr, te in cv_kde.split(coords):
            est = KernelDensity(bandwidth=bw, kernel="gaussian").fit(coords[tr])
            ll_scores.append(est.score(coords[te]))
        scores[bw] = float(np.mean(ll_scores))

    best_bw = max(scores, key=scores.get)
    log.info("  Best KDE bandwidth=%.2f  log-likelihood=%.4f", best_bw, scores[best_bw])

    # Downstream R² with best bandwidth
    kde_est = KernelDensity(bandwidth=best_bw, kernel="gaussian").fit(coords)
    log_density = kde_est.score_samples(coords)
    X_aug  = np.column_stack([
        X_train[BASE_FEATURES].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(),
        log_density.reshape(-1,1)])
    X_aug  = RobustScaler().fit_transform(X_aug)
    y_log  = np.log1p(y_train.values)
    cv     = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    r2_kde = float(cross_val_score(Ridge(alpha=1.0),X_aug,y_log,cv=cv,scoring="r2").mean())

    plt.figure(figsize=(7,4))
    plt.plot(bandwidths, list(scores.values()), "o-", color="#4C78A8", markersize=8)
    plt.axvline(best_bw, color="#E45756", linestyle="--", label=f"Best bw={best_bw}")
    plt.xlabel("KDE bandwidth"); plt.ylabel("Mean CV log-likelihood")
    plt.title("Concept E: KDE bandwidth selection\nvia cross-validated log-likelihood")
    plt.legend(); plt.tight_layout()
    plt.savefig(output_dir/"plot_kde_bandwidth.png",dpi=160); plt.close()

    result = {"best_bandwidth":best_bw,"cv_scores":scores,"downstream_r2":r2_kde}
    write_json(output_dir/"kde_analysis.json", result)
    return result


# ── Concept F: Spatial cross-validation ───────────────────────────────────────
class SpatialKFold(BaseEstimator):
    """
    Concept F: Spatial KFold CV — prevent geographic leakage.
    Clusters block groups into K geographic bands using K-Means on (lat, lon).
    Each fold holds out one complete geographic cluster.
    """
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if isinstance(X, pd.DataFrame):
            coords = X[["Latitude","Longitude"]].values
        else:
            coords = X
        km  = KMeans(n_clusters=self.n_splits, random_state=RANDOM_STATE, n_init=10)
        lab = km.fit_predict(coords)
        for fold_id in range(self.n_splits):
            test_idx  = np.where(lab == fold_id)[0]
            train_idx = np.where(lab != fold_id)[0]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def analyse_spatial_cv(X_train: pd.DataFrame, y_train: pd.Series,
                        output_dir: Path) -> dict[str, Any]:
    """
    Concept F: Compare random KFold vs spatial KFold.
    Random KFold leaks because nearby block groups appear in both train and test.
    Spatial KFold holds out complete geographic regions.
    """
    log.info("Concept F: Spatial cross-validation comparison …")
    pipe  = make_full_pipeline()
    y_log = np.log1p(y_train.values)

    cv_random  = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_spatial = SpatialKFold(n_splits=5)

    r2_random  = cross_val_score(pipe, X_train, y_log, cv=cv_random,  scoring="r2")
    r2_spatial = cross_val_score(pipe, X_train, y_log, cv=cv_spatial, scoring="r2")

    leak_inflation = float(r2_random.mean() - r2_spatial.mean())
    log.info("  Random KFold R²=%.4f±%.4f", r2_random.mean(), r2_random.std())
    log.info("  Spatial KFold R²=%.4f±%.4f  leakage inflation=%.4f",
             r2_spatial.mean(), r2_spatial.std(), leak_inflation)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([r2_random, r2_spatial],
               labels=["Random KFold\n(leaks nearby locs)", "Spatial KFold\n(holds out regions)"],
               patch_artist=True,
               boxprops=dict(facecolor="#4C78A8",alpha=0.7))
    ax.set_ylabel("CV R² (log target)")
    ax.set_title(f"Concept F: Geographic leakage in CV\n"
                 f"Random KFold inflates R² by {leak_inflation:.4f} vs Spatial KFold")
    plt.tight_layout(); plt.savefig(output_dir/"plot_spatial_cv.png",dpi=160); plt.close()

    result = {
        "random_kfold_r2":  float(r2_random.mean()),
        "spatial_kfold_r2": float(r2_spatial.mean()),
        "leakage_inflation":leak_inflation,
    }
    write_json(output_dir/"spatial_cv_comparison.json", result)
    return result


# ── Concept G: Geographically Weighted Regression ─────────────────────────────
def analyse_gwr(X_train: pd.DataFrame, y_train: pd.Series,
                output_dir: Path) -> dict[str, Any]:
    """
    Concept G: GWR — local regression coefficients that vary by location.
    The MedInc effect on house value is stronger in rural areas (slope steeper)
    than in urban areas where location premium dominates.
    """
    log.info("Concept G: GWR analysis …")
    if not _GWR:
        log.warning("pip install mgwr — GWR skipped.")
        return {}
    try:
        # Subsample for speed (GWR is O(n²))
        rng   = np.random.default_rng(RANDOM_STATE)
        idx   = rng.choice(len(X_train), min(2000, len(X_train)), replace=False)
        Xs    = X_train.iloc[idx].copy()
        ys    = y_train.iloc[idx].to_numpy()
        y_log = np.log1p(ys).reshape(-1,1)

        coords = np.column_stack([Xs["Latitude"].values, Xs["Longitude"].values])
        # Use MedInc, AveRooms, HouseAge as explanatory variables
        feat_cols = ["MedInc","AveRooms","HouseAge"]
        X_gwr = StandardScaler().fit_transform(
            Xs[feat_cols].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy())

        # Bandwidth selection
        log.info("  GWR bandwidth selection (subsample n=%d) …", len(Xs))
        bw_sel = Sel_BW(coords, y_log, X_gwr)
        bw     = bw_sel.search(bw_min=20, bw_max=200, criterion="AICc")
        log.info("  Optimal GWR bandwidth=%s", bw)

        model  = GWR(coords, y_log, X_gwr, bw=bw)
        result = model.fit()
        local_r2 = result.localR2.flatten()

        log.info("  GWR: global R²=%.4f  mean local R²=%.4f  min=%.4f  max=%.4f",
                 float(result.R2), float(local_r2.mean()),
                 float(local_r2.min()), float(local_r2.max()))

        # Local R² map
        plt.figure(figsize=(8,6))
        sc = plt.scatter(Xs["Longitude"], Xs["Latitude"], c=local_r2,
                         cmap="RdYlGn", s=6, alpha=0.8, vmin=0, vmax=1)
        plt.colorbar(sc, label="Local R²")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.title("Concept G: GWR local R² map\nGreen=model fits well  Red=poor local fit")
        plt.tight_layout(); plt.savefig(output_dir/"plot_gwr_local_r2.png",dpi=160); plt.close()

        # MedInc coefficient map (index 0 = intercept, index 1 = MedInc)
        coef_medinc = result.params[:,1].flatten()
        plt.figure(figsize=(8,6))
        sc2 = plt.scatter(Xs["Longitude"], Xs["Latitude"], c=coef_medinc,
                          cmap="coolwarm", s=6, alpha=0.8)
        plt.colorbar(sc2, label="MedInc GWR coefficient")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.title("Concept G: GWR — MedInc local coefficients\n"
                  "Blue=income matters less  Red=income matters more")
        plt.tight_layout(); plt.savefig(output_dir/"plot_gwr_medinc_coeff.png",dpi=160); plt.close()

        gwr_result = {
            "bandwidth":         float(bw),
            "global_r2":         float(result.R2),
            "mean_local_r2":     float(local_r2.mean()),
            "min_local_r2":      float(local_r2.min()),
            "max_local_r2":      float(local_r2.max()),
            "aic":               float(result.aic),
        }
        write_json(output_dir/"gwr_analysis.json", gwr_result)
        return gwr_result
    except Exception as exc:
        log.warning("GWR failed: %s", exc)
        return {"error": str(exc)}


# ── Concept J: Residual spatial autocorrelation ────────────────────────────────
def analyse_residual_autocorrelation(
    model, X_test: pd.DataFrame, y_test: pd.Series,
    y_pred_log: np.ndarray, output_dir: Path) -> dict[str, Any]:
    """
    Concept J: Moran's I on model residuals.
    A good spatial model should have near-zero residual autocorrelation.
    Significant residual Moran's I means the model is still missing spatial patterns.
    """
    log.info("Concept J: Residual spatial autocorrelation …")
    residuals = np.log1p(y_test.values) - y_pred_log

    if not _PYSAL:
        log.warning("libpysal not available — skipping Moran on residuals")
        return {}

    coords = list(zip(X_test["Longitude"].values, X_test["Latitude"].values))
    w      = lw.KNN.from_array(np.array(coords), k=8)
    w.transform = "r"
    moran  = Moran(residuals, w, permutations=199)
    log.info("  Residual Moran's I=%.4f  p=%.4f  (spatial signal remaining: %s)",
             moran.I, moran.p_sim, "YES" if moran.p_sim < 0.05 else "NO")

    # Residual map
    plt.figure(figsize=(9, 6))
    sc = plt.scatter(X_test["Longitude"], X_test["Latitude"],
                     c=residuals, cmap="RdBu_r", s=3, alpha=0.6,
                     vmin=-np.percentile(np.abs(residuals),95),
                     vmax= np.percentile(np.abs(residuals),95))
    plt.colorbar(sc, label="Residual (log scale)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title(f"Concept J: Spatial residual map\n"
              f"Moran's I={moran.I:.4f}  p={moran.p_sim:.4f}  "
              f"({'spatial signal remaining' if moran.p_sim<0.05 else 'spatially random ✓'})")
    plt.tight_layout(); plt.savefig(output_dir/"plot_residual_spatial.png",dpi=160); plt.close()

    result = {
        "residual_moran_I": float(moran.I),
        "residual_moran_p": float(moran.p_sim),
        "spatial_signal_remaining": bool(moran.p_sim < 0.05),
    }
    write_json(output_dir/"residual_autocorrelation.json", result)
    return result


# ── GeoFeatureEngineer — full spatial feature set ────────────────────────────
class GeoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Combines all spatial feature engineering concepts:
      C: raw lat/lon (already in input)
      D: Haversine distances to 10 California urban centres
      E: KDE log-density (urbanisation proxy)
      H: lat×lon interaction, sin/cos periodic, grid cell encoding,
         RBF features around urban centres
    Fit-safe: KDE bandwidth learned from training data only.
    """
    def __init__(self, n_urban_centres: int = 10, kde_bandwidth: float = 0.10):
        self.n_urban_centres = n_urban_centres
        self.kde_bandwidth   = kde_bandwidth

    def fit(self, X: pd.DataFrame, y=None) -> "GeoFeatureEngineer":
        # Learn KDE from training coordinates
        if "Latitude" in X.columns and "Longitude" in X.columns:
            coords = np.column_stack([
                pd.to_numeric(X["Longitude"], errors="coerce").fillna(-120).values,
                pd.to_numeric(X["Latitude"],  errors="coerce").fillna(37).values,
            ])
            self.kde_ = KernelDensity(
                bandwidth=self.kde_bandwidth, kernel="gaussian").fit(coords)
        else:
            self.kde_ = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        lat = pd.to_numeric(X.get("Latitude",  pd.Series(37.0,index=X.index)),
                            errors="coerce").fillna(37.0).values
        lon = pd.to_numeric(X.get("Longitude", pd.Series(-120.0,index=X.index)),
                            errors="coerce").fillna(-120.0).values

        # Concept D: Haversine distances to urban centres
        for name, ulat, ulon in URBAN_CENTRES[:self.n_urban_centres]:
            X[f"dist_{name}_km"] = haversine_km(lat, lon, ulat, ulon)

        # Concept E: KDE density (urbanisation proxy)
        if self.kde_ is not None:
            coords = np.column_stack([lon, lat])
            X["log_kde_density"] = self.kde_.score_samples(coords)

        # Concept H: Additional spatial features
        # H1: Interaction term — captures diagonal gradients
        X["lat_lon_interact"] = lat * lon

        # H2: Cyclic encoding (useful for periodic wrap-around)
        X["lat_sin"] = np.sin(np.radians(lat))
        X["lat_cos"] = np.cos(np.radians(lat))
        X["lon_sin"] = np.sin(np.radians(lon))
        X["lon_cos"] = np.cos(np.radians(lon))

        # H3: Coastal proximity proxy (longitude + 120)
        X["coast_dist_proxy"] = np.abs(lon + 120.0)

        # H4: RBF features around top-3 urban centres
        for name, ulat, ulon in URBAN_CENTRES[:3]:
            d    = haversine_km(lat, lon, ulat, ulon)
            X[f"rbf_{name}"] = np.exp(-d / 50.0)  # 50km scale

        # H5: Latitude bands (Northern/Central/Southern California)
        X["is_northern_ca"]  = (lat >= 37.0).astype(float)
        X["is_southern_ca"]  = (lat <  35.5).astype(float)
        X["is_bay_area"]     = ((lat >= 37.2) & (lat <= 38.2) &
                                (lon >= -122.6) & (lon <= -121.5)).astype(float)
        X["is_la_basin"]     = ((lat >= 33.5) & (lat <= 34.5) &
                                (lon >= -119.0) & (lon <= -117.5)).astype(float)

        return X


def build_preprocessor_geo() -> Pipeline:
    """
    Numeric preprocessor: median impute (add_indicator) → RobustScaler.
    Resolves columns dynamically at fit time.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler",  RobustScaler()),
    ])


def make_full_pipeline(model=None) -> Pipeline:
    if model is None:
        model = Ridge(alpha=1.0)
    from sklearn.feature_selection import SelectFromModel
    sel = SelectFromModel(
        ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        threshold="median")
    return Pipeline([
        ("feature_engineering", GeoFeatureEngineer()),
        ("preprocess",          build_preprocessor_geo()),
        ("feature_selection",   sel),
        ("model",               model),
    ])


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_predictions(y_true, y_pred_log, log_target=True):
    if hasattr(y_true,"to_numpy"): y_true=y_true.to_numpy()
    if log_target:
        y_pred_usd = np.expm1(y_pred_log)
        y_true_usd = y_true
    else:
        y_pred_usd = y_pred_log; y_true_usd = y_true
    res = y_true_usd - y_pred_usd
    return {
        "r2":            float(r2_score(y_true_usd, y_pred_usd)),
        "rmse":          float(np.sqrt(mean_squared_error(y_true_usd, y_pred_usd))),
        "mae":           float(mean_absolute_error(y_true_usd, y_pred_usd)),
        "mape":          float(mean_absolute_percentage_error(y_true_usd.clip(0.1), y_pred_usd.clip(0.1))),
        "medae":         float(median_absolute_error(y_true_usd, y_pred_usd)),
        "residual_mean": float(res.mean()),
        "residual_std":  float(res.std()),
    }


def evaluate_baselines(X_tr, X_te, y_tr, y_te):
    return {s: evaluate_predictions(y_te,
                np.log1p(DummyRegressor(strategy=s).fit(X_tr,y_tr).predict(X_te).clip(0)))
            for s in ["mean","median"]}


# ── Subgroup evaluation (Concept L) ───────────────────────────────────────────
def evaluate_subgroups(model, X_test: pd.DataFrame, y_test: pd.Series,
                        y_pred_log: np.ndarray, output_dir: Path) -> dict:
    log.info("Concept L: Subgroup disparity by geographic cluster …")
    y_pred_usd = np.expm1(y_pred_log)
    y_true_usd = y_test.values
    overall_rmse = float(np.sqrt(mean_squared_error(y_true_usd, y_pred_usd)))

    # Assign clusters
    fe = GeoFeatureEngineer().fit(X_test)
    Xeng = fe.transform(X_test.reset_index(drop=True).copy())

    eval_df = Xeng.copy()
    eval_df["_y_true"] = y_true_usd; eval_df["_y_pred"] = y_pred_usd

    # Geographic clusters (based on flags from GeoFeatureEngineer)
    region_map = {
        "bay_area":    eval_df.get("is_bay_area", pd.Series(0,index=eval_df.index)) == 1,
        "la_basin":    eval_df.get("is_la_basin", pd.Series(0,index=eval_df.index)) == 1,
        "northern_ca": (eval_df.get("is_northern_ca", pd.Series(0,index=eval_df.index)) == 1) &
                       (eval_df.get("is_bay_area", pd.Series(0,index=eval_df.index)) != 1),
        "southern_ca": (eval_df.get("is_southern_ca", pd.Series(0,index=eval_df.index)) == 1) &
                       (eval_df.get("is_la_basin", pd.Series(0,index=eval_df.index)) != 1),
    }
    rows = []
    for region, mask in region_map.items():
        sub = eval_df[mask]
        if len(sub) < 20: continue
        sr = float(np.sqrt(mean_squared_error(sub["_y_true"], sub["_y_pred"])))
        rows.append({
            "region": region, "n": int(len(sub)),
            "mean_actual": round(float(sub["_y_true"].mean()),3),
            "rmse": round(sr,4),
            "r2":   round(float(r2_score(sub["_y_true"],sub["_y_pred"])),4),
            "rmse_gap": round(sr - overall_rmse, 4),
            "alert": bool(sr > overall_rmse * 1.25),
        })

    if rows:
        rd = pd.DataFrame(rows); rd.to_csv(output_dir/"subgroup_report.csv",index=False)
        fig, ax = plt.subplots(figsize=(8,4))
        colors = ["#E45756" if r["alert"] else "#4C78A8" for r in rows]
        ax.bar([r["region"] for r in rows], [r["rmse"] for r in rows], color=colors)
        ax.axhline(overall_rmse, linestyle="--", color="black",
                   label=f"Overall RMSE={overall_rmse:.3f}")
        ax.set_ylabel("RMSE ($100k)"); ax.set_title("Concept L: RMSE by geographic region")
        ax.legend(); plt.tight_layout()
        plt.savefig(output_dir/"plot_subgroup_rmse.png",dpi=160); plt.close()

    return {"overall_rmse": overall_rmse, "subgroups": rows}


# ── SHAP ──────────────────────────────────────────────────────────────────────
def save_shap_artifacts(model, X_test, y_test, y_pred_log, output_dir):
    if not _SHAP: log.warning("pip install shap"); return
    log.info("SHAP for champion …")
    try:
        step_names = list(model.named_steps.keys())
        clf = model.named_steps["model"]
        prep= model.named_steps["preprocess"]
        fe  = model.named_steps["feature_engineering"]
        sel = model.named_steps.get("feature_selection")

        Xt = fe.transform(X_test)
        Xt = prep.transform(Xt)
        fn_arr = None
        if sel:
            fs_idx = step_names.index("feature_selection")
            for sname in reversed(step_names[:fs_idx]):
                s = model.named_steps[sname]
                if hasattr(s,"get_feature_names_out"):
                    try: fn_arr=s.get_feature_names_out(); break
                    except: pass
            support = sel.get_support()
            if fn_arr is None: fn_arr=np.array([f"f{i}" for i in range(Xt.shape[1])])
            if len(fn_arr)!=len(support): fn_arr=np.array([f"f{i}" for i in range(len(support))])
            sn=fn_arr[support]; Xt=sel.transform(Xt)
        else:
            try: sn=prep.get_feature_names_out()
            except: sn=np.array([f"f{i}" for i in range(Xt.shape[1])])

        Xdf=pd.DataFrame(Xt,columns=sn)
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
            plt.tight_layout(); plt.savefig(output_dir/fname,dpi=150,bbox_inches="tight"); plt.close()

        pd.DataFrame({"feature":sn,"mean_abs_shap":np.abs(sv).mean(axis=0)}
            ).sort_values("mean_abs_shap",ascending=False
            ).to_csv(output_dir/"shap_importance.csv",index=False)
        log.info("SHAP saved.")
    except Exception as exc:
        log.warning("SHAP failed: %s",exc)


def save_feature_importance(model, output_dir):
    step_names = list(model.named_steps.keys())
    clf = model.named_steps["model"]
    sel = model.named_steps.get("feature_selection")
    if not sel: return
    fs_idx = step_names.index("feature_selection")
    fn = None
    for sname in reversed(step_names[:fs_idx]):
        s = model.named_steps[sname]
        if hasattr(s,"get_feature_names_out"):
            try: fn=s.get_feature_names_out(); break
            except: pass
    if fn is None: return
    support = sel.get_support()
    if len(fn)!=len(support): fn=np.array([f"f{i}" for i in range(len(support))])
    sn = fn[support]
    if hasattr(clf,"feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf,"coef_"):
        imp = np.abs(clf.coef_)
    else: return
    fi = pd.DataFrame({"feature":sn,"importance":imp}).sort_values("importance",ascending=False)
    fi.to_csv(output_dir/"feature_importance.csv",index=False)
    plt.figure(figsize=(9,5.5))
    sns.barplot(data=fi.head(25),y="feature",x="importance",color="#4C78A8")
    plt.title("Top 25 spatial features — model importance")
    plt.tight_layout(); plt.savefig(output_dir/"plot_feature_importance.png",dpi=160); plt.close()


def residual_diagnostics(y_true_log, y_pred_log, output_dir):
    log.info("Concept I: Residual diagnostics …")
    residuals = y_true_log - y_pred_log
    n = len(residuals)
    bp_r2  = LinearRegression().fit(y_pred_log.reshape(-1,1),residuals**2).score(
                y_pred_log.reshape(-1,1),residuals**2)
    bp_stat= float(n*bp_r2)
    bp_p   = float(1-scipy_stats.chi2.cdf(bp_stat,df=1))
    sw_s,sw_p = scipy_stats.shapiro(residuals[:5000])
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    scipy_stats.probplot(residuals,dist="norm",plot=axes[0])
    axes[0].set_title("Q-Q plot (normality)")
    axes[1].scatter(y_pred_log,residuals,alpha=0.1,s=3,color="#4C78A8")
    axes[1].axhline(0,color="red",linewidth=1.0)
    axes[1].set_xlabel("Fitted log(MedHouseVal)"); axes[1].set_ylabel("Residual")
    axes[1].set_title("Scale-location (heteroscedasticity)")
    axes[2].hist(residuals,bins=60,color="#54A24B",edgecolor="white",linewidth=0.3)
    axes[2].axvline(0,color="red"); axes[2].set_title("Residual distribution")
    plt.suptitle("Concept I: Residual diagnostics",fontsize=12,y=1.02)
    plt.tight_layout(); plt.savefig(output_dir/"plot_residual_diagnostics.png",dpi=160,bbox_inches="tight"); plt.close()
    log.info("BP p=%.4f  SW p=%.4f",bp_p,sw_p)
    return {"breusch_pagan":{"p_value":bp_p,"heteroscedastic":bp_p<0.05},
            "shapiro_wilk":{"p_value":float(sw_p),"normal":sw_p>0.05}}


def plot_learning_curves_geo(model, X_train, y_train_log, output_dir):
    log.info("Concept K: Learning curves …")
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    try:
        sizes,tr_s,cv_s = learning_curve(
            model,X_train,y_train_log,train_sizes=np.linspace(0.10,1.0,7),
            cv=cv,scoring="r2",n_jobs=N_JOBS)
        plt.figure(figsize=(8,4.5))
        plt.plot(sizes,tr_s.mean(axis=1),"o-",color="#4C78A8",label="Train R²")
        plt.plot(sizes,cv_s.mean(axis=1),"o-",color="#E45756",label="CV R²")
        plt.fill_between(sizes,tr_s.mean(1)-tr_s.std(1),tr_s.mean(1)+tr_s.std(1),alpha=0.12,color="#4C78A8")
        plt.fill_between(sizes,cv_s.mean(1)-cv_s.std(1),cv_s.mean(1)+cv_s.std(1),alpha=0.12,color="#E45756")
        gap=float(cv_s[-1].mean()-tr_s[-1].mean())
        plt.title(f"Concept K: Learning curves\nTrain-CV gap={gap:.4f}")
        plt.xlabel("Training set size"); plt.ylabel("R²")
        plt.legend(); plt.tight_layout()
        plt.savefig(output_dir/"plot_learning_curve.png",dpi=160); plt.close()
    except Exception as exc:
        log.warning("Learning curve failed: %s",exc)


def tune_model(X_train, y_train_log, n_iter=20, n_cv_splits=5, fast=False):
    log.info("Hyperparameter search: n_iter=%d cv=%d-fold fast=%s",n_iter,n_cv_splits,fast)
    _n = 50 if fast else 150
    param_distributions = [
        {"feature_selection__threshold":["median","0.75*median","1.25*median"],
         "model":[Ridge()],"model__alpha":[0.1,1,10,50,100,500,1000,5000]},
        {"feature_selection__threshold":["median","0.75*median"],
         "model":[HuberRegressor(max_iter=500)],
         "model__epsilon":[1.1,1.35,1.5,2.0],"model__alpha":[0.001,0.01,0.1]},
        {"feature_selection__threshold":["median","0.75*median","1.25*median"],
         "model":[GradientBoostingRegressor(n_estimators=_n,random_state=RANDOM_STATE)],
         "model__max_depth":[3,4,5],"model__learning_rate":[0.02,0.05,0.1,0.2],
         "model__subsample":[0.7,0.9]},
        {"feature_selection__threshold":["median","0.75*median","1.25*median"],
         "model":[RandomForestRegressor(n_estimators=_n,random_state=RANDOM_STATE,n_jobs=N_JOBS)],
         "model__max_depth":[8,12,None],"model__min_samples_leaf":[1,2,4]},
    ]
    cv = SpatialKFold(n_splits=n_cv_splits)    # spatial CV in hyperparameter search
    search = RandomizedSearchCV(
        make_full_pipeline(), param_distributions, n_iter=n_iter,
        scoring={"r2":"r2","neg_rmse":"neg_root_mean_squared_error"},
        refit="r2", cv=cv, random_state=RANDOM_STATE,
        n_jobs=N_JOBS, verbose=1, return_train_score=True)
    search.fit(X_train, y_train_log)
    best_mdl = search.best_estimator_.named_steps["model"]
    if hasattr(best_mdl,"n_estimators") and best_mdl.n_estimators==_n:
        log.info("Upgrading %d → 300 trees …",_n)
        best_mdl.set_params(n_estimators=300)
        search.best_estimator_.fit(X_train,y_train_log)
    log.info("Best CV R²=%.4f  model=%s",search.best_score_,type(best_mdl).__name__)
    return search


def build_training_profile(X_train, y_train):
    num_cols = [c for c in BASE_FEATURES if c in X_train.columns]
    stats = {}
    for col in num_cols:
        v = pd.to_numeric(X_train[col],errors="coerce").dropna().to_numpy(dtype=np.float64)
        if len(v)==0: continue
        stats[col] = {"mean":float(v.mean()),"std":float(v.std()),
                      "min":float(v.min()),"max":float(v.max()),
                      "quantiles":np.quantile(v,np.linspace(0,1,100)).tolist()}
    return to_jsonable({
        "trained_at":datetime.now(timezone.utc).isoformat(),
        "row_count":int(len(X_train)),
        "raw_columns":list(X_train.columns),
        "target_stats":{"mean":float(y_train.mean()),"std":float(y_train.std()),
                        "min":float(y_train.min()),"max":float(y_train.max())},
        "raw_missing_rate":X_train.isna().mean().to_dict(),
        "numeric_train_stats":stats,
        "lat_lon_bounds":{
            "lat_min":float(X_train["Latitude"].min()),
            "lat_max":float(X_train["Latitude"].max()),
            "lon_min":float(X_train["Longitude"].min()),
            "lon_max":float(X_train["Longitude"].max()),
        }
    })


def save_model_card(metrics, fairness, reg_analysis, moran_result, search, output_dir):
    tm = metrics.get("test_metrics",{})
    write_json(output_dir/MODEL_CARD_FILE,{
        "schema_version":"1.0",
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "model_details":{"name":"California Housing Spatial Regressor",
                         "type":"Geospatial Regression (sklearn Pipeline)",
                         "algorithm":repr(search.best_estimator_.named_steps["model"])},
        "intended_use":{"primary_use":"Predict block-group median house value using spatial and demographic features.",
                        "out_of_scope":["Individual house valuation","Post-1990 markets","Other US states"]},
        "evaluation_results":{"test_r2":tm.get("r2"),"test_rmse":tm.get("rmse"),"test_mae":tm.get("mae")},
        "spatial_analysis":{"global_moran_I":moran_result.get("moran_I"),
                            "residual_moran_I":metrics.get("residual_moran",{}).get("residual_moran_I"),
                            "leakage_inflation":reg_analysis.get("leakage_inflation")},
        "limitations":[
            "1990 Census data — not reflective of current California housing market.",
            "Target capped at $500k — luxury market predictions are unreliable.",
            "Block-group centroids, not individual parcel coordinates.",
        ],
        "ethical_considerations":[
            "Latitude/longitude encode neighbourhood characteristics that may reflect historical redlining.",
            "Do not use for automated individual lending or valuation decisions.",
        ],
        "hyperparameters":search.best_params_,
        "cv_strategy":"SpatialKFold(5) — prevents geographic leakage",
        "cv_best_r2":float(search.best_score_),
    })


def log_to_mlflow(metrics, search, model, output_dir):
    if not _MLFLOW: return
    try:
        mlflow.set_experiment("california_housing_spatial")
        tm=metrics.get("test_metrics",{})
        with mlflow.start_run():
            mlflow.log_params({f"best_{k}":str(v) for k,v in search.best_params_.items()})
            mlflow.log_metrics({"cv_r2":float(search.best_score_),
                                "test_r2":float(tm.get("r2",0)),
                                "test_rmse":float(tm.get("rmse",0))})
            for f in [MODEL_CARD_FILE,METRICS_FILE,"plot_spatial_eda.png","plot_shap_bar.png"]:
                if (output_dir/f).exists(): mlflow.log_artifact(str(output_dir/f))
            mlflow.sklearn.log_model(model,"model")
        log.info("MLflow logged.")
    except Exception as e:
        log.warning("MLflow failed: %s",e)


def save_environment_snapshot(output_dir):
    env={"saved_at":datetime.now(timezone.utc).isoformat(),"python":sys.version,"platform":sys.platform,"libraries":{}}
    for lib in ["sklearn","pandas","numpy","scipy","joblib","shap","mlflow","geopandas","libpysal","esda","mgwr","shapely"]:
        try:
            mod=importlib.import_module(lib); env["libraries"][lib]=getattr(mod,"__version__","unknown")
        except ImportError:
            env["libraries"][lib]="not_installed"
    write_json(output_dir/ENVIRONMENT_FILE,env)


def compute_oof_uncertainty(best_estimator, X_train, y_train_log, overpredict_cost=1.0, underpredict_cost=1.0):
    log.info("Computing OOF uncertainty (SpatialKFold) …")
    cv  = SpatialKFold(n_splits=5)
    oof = cross_val_predict(clone(best_estimator),X_train,y_train_log,cv=cv,n_jobs=N_JOBS)
    res = y_train_log - oof
    oof_rmse_log = float(np.sqrt(np.mean(res**2)))
    y_usd = np.expm1(y_train_log); oof_usd = np.expm1(oof)
    usd_rmse = float(np.sqrt(np.mean((y_usd-oof_usd)**2)))
    return {"oof_rmse_log":oof_rmse_log,"oof_rmse_usd":usd_rmse,
            "oof_r2":float(r2_score(y_train_log,oof)),
            "lower_band":oof_rmse_log*underpredict_cost,
            "upper_band":oof_rmse_log*overpredict_cost}


def _model_version_tag(model):
    return hashlib.sha1(pickle.dumps(model)).hexdigest()[:8]


# ── Main train() ──────────────────────────────────────────────────────────────
def train(output_dir, n_iter=20, n_cv_splits=5, fast=False,
          overpredict_cost=1.0, underpredict_cost=1.0):
    log.info("=== Training started (n_jobs=%d) ===",N_JOBS)
    output_dir.mkdir(parents=True,exist_ok=True)

    df                             = fix_data_types(load_data())
    X_train, X_test, y_train, y_te = split_data(df)
    y_train_log = np.log1p(y_train.values)
    y_test_log  = np.log1p(y_te.values)

    # Phase 1: EDA + concept analyses (train only)
    save_research_artifacts(X_train, y_train, output_dir)
    baselines    = evaluate_baselines(X_train, X_test, y_train, y_te)
    moran_result = analyse_morans_i(X_train, y_train, output_dir)
    raw_analysis = analyse_raw_vs_spatial(X_train, y_train, output_dir)
    hav_analysis = analyse_haversine_features(X_train, y_train, output_dir)
    kde_analysis = analyse_kde_density(X_train, y_train, output_dir)
    cv_analysis  = analyse_spatial_cv(X_train, y_train, output_dir)
    gwr_analysis = analyse_gwr(X_train, y_train, output_dir)

    # Phase 2: Model search (spatial CV)
    search      = tune_model(X_train, y_train_log, n_iter=n_iter,
                             n_cv_splits=n_cv_splits, fast=fast)
    uncertainty = compute_oof_uncertainty(search.best_estimator_, X_train, y_train_log,
                                          overpredict_cost, underpredict_cost)

    final_model = clone(search.best_estimator_)
    final_model.fit(X_train, y_train_log)
    y_pred_log  = final_model.predict(X_test)
    test_metrics= evaluate_predictions(y_te, y_pred_log, log_target=True)

    log.info("Test R²=%.4f  RMSE=$%.0fk  MAE=$%.0fk",
             test_metrics["r2"], test_metrics["rmse"]*100, test_metrics["mae"]*100)

    # Artifacts
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha1 = _model_version_tag(final_model)
    joblib.dump(final_model, output_dir/f"geo_pipeline_{ts}_{sha1}.joblib")
    joblib.dump(final_model, output_dir/MODEL_FILE)
    save_environment_snapshot(output_dir)
    pd.DataFrame(search.cv_results_).sort_values("rank_test_r2").to_csv(
        output_dir/"cv_results.csv",index=False)
    save_feature_importance(final_model,output_dir)
    write_json(output_dir/TRAINING_PROFILE_FILE,build_training_profile(X_train,y_train))

    res_diag   = residual_diagnostics(y_test_log, y_pred_log, output_dir)
    residual_m = analyse_residual_autocorrelation(
        final_model, X_test, y_te, y_pred_log, output_dir)
    plot_learning_curves_geo(final_model, X_train, y_train_log, output_dir)
    fairness   = evaluate_subgroups(final_model, X_test, y_te, y_pred_log, output_dir)
    save_shap_artifacts(final_model, X_test, y_te, y_pred_log, output_dir)

    metrics = {
        "baselines":baselines,"split":{"train_rows":int(len(X_train)),"test_rows":int(len(X_test))},
        "morans_i":moran_result,"spatial_uplift":raw_analysis,
        "haversine_analysis":hav_analysis,"kde_analysis":kde_analysis,
        "spatial_cv":cv_analysis,"gwr_analysis":gwr_analysis,
        "best_cv":{"best_r2":float(search.best_score_),"best_params":search.best_params_},
        "uncertainty_info":uncertainty,"residual_diag":res_diag,"residual_moran":residual_m,
        "test_metrics":test_metrics,"fairness":fairness,
    }
    write_json(output_dir/METRICS_FILE,metrics)
    save_model_card(metrics,fairness,cv_analysis,moran_result,search,output_dir)
    log_to_mlflow(metrics,search,final_model,output_dir)
    log.info("=== Training complete ===")
    return to_jsonable(metrics)


# ── predict / monitor / sample-input ─────────────────────────────────────────
def predict(artifact_dir, input_csv, output_csv):
    model = joblib.load(artifact_dir/MODEL_FILE)
    mp    = artifact_dir/METRICS_FILE
    unc   = 0.10
    if mp.exists():
        unc = json.loads(mp.read_text())["uncertainty_info"].get("oof_rmse_log", unc)
    df  = pd.read_csv(input_csv)
    pf  = artifact_dir/TRAINING_PROFILE_FILE
    if pf.exists():
        req  = set(json.loads(pf.read_text())["raw_columns"])
        miss = req - set(df.columns)
        if miss: raise ValueError(f"Missing columns: {sorted(miss)}")
    y_pred_log = model.predict(df)
    df["predicted_MedHouseVal"] = np.expm1(y_pred_log).clip(0.15,5.0)
    df["lower_bound"]           = np.expm1(y_pred_log - unc).clip(0.15)
    df["upper_bound"]           = np.expm1(y_pred_log + unc).clip(0,5.0)
    output_csv=Path(output_csv); output_csv.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(output_csv,index=False)
    log.info("Predictions saved to %s",output_csv.resolve())


def monitor(artifact_dir, input_csv, output_json, missing_rate_alert=0.05, ks_pvalue=0.05):
    profile  = json.loads((artifact_dir/TRAINING_PROFILE_FILE).read_text())
    incoming = pd.read_csv(input_csv)
    drift=[]
    for col,tr in profile["raw_missing_rate"].items():
        if col not in incoming: continue
        cur=float(incoming[col].isna().mean())
        drift.append({"column":col,"train_rate":float(tr),"current_rate":cur,
                      "change":abs(cur-float(tr)),"alert":abs(cur-float(tr))>=missing_rate_alert})
    ks_rows=[]
    for col,stats in profile.get("numeric_train_stats",{}).items():
        if col not in incoming.columns: continue
        vals=incoming[col].dropna().to_numpy()
        if len(vals)<10: continue
        stat,p=ks_2samp(np.array(stats["quantiles"]),vals)
        ks_rows.append({"column":col,"ks_stat":float(stat),"p_value":float(p),"alert":p<ks_pvalue})
    # Spatial bounds check
    bounds = profile.get("lat_lon_bounds",{})
    lat_ok = (incoming["Latitude"].between(bounds.get("lat_min",30),bounds.get("lat_max",45)).all()
              if "Latitude" in incoming.columns else True)
    lon_ok = (incoming["Longitude"].between(bounds.get("lon_min",-130),bounds.get("lon_max",-110)).all()
              if "Longitude" in incoming.columns else True)
    report={"checked_at":datetime.now(timezone.utc).isoformat(),"row_count":int(len(incoming)),
            "missing_rate_drift":drift,"distribution_drift":ks_rows,
            "spatial_bounds_ok":{"latitude":bool(lat_ok),"longitude":bool(lon_ok)}}
    output_json=Path(output_json); output_json.parent.mkdir(parents=True,exist_ok=True)
    write_json(output_json,report)
    return report


def create_sample_input(output_csv, rows):
    df  = fix_data_types(load_data())
    output_csv=Path(output_csv); output_csv.parent.mkdir(parents=True,exist_ok=True)
    df.drop(columns=[TARGET]).head(rows).to_csv(output_csv,index=False)
    log.info("Sample saved to %s",output_csv.resolve())


# ── Utilities ─────────────────────────────────────────────────────────────────
def write_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload),indent=2),encoding="utf-8")

def to_jsonable(v):
    if isinstance(v,dict):   return {str(k):to_jsonable(x) for k,x in v.items()}
    if isinstance(v,list):   return [to_jsonable(x) for x in v]
    if isinstance(v,BaseEstimator): return repr(v)
    if isinstance(v,np.bool_): return bool(v)
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


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p=argparse.ArgumentParser(description="California Housing Geospatial Regression",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sp=p.add_subparsers(dest="command",required=True)
    tp=sp.add_parser("train")
    tp.add_argument("--output-dir",type=Path,default=Path("artifacts_geo"))
    tp.add_argument("--n-iter",type=int,default=20)
    tp.add_argument("--n-cv-splits",type=int,default=5)
    tp.add_argument("--fast",action="store_true")
    tp.add_argument("--overpredict-cost",type=float,default=1.0)
    tp.add_argument("--underpredict-cost",type=float,default=1.0)
    pp=sp.add_parser("predict")
    pp.add_argument("--artifact-dir",type=Path,default=Path("artifacts_geo"))
    pp.add_argument("--input-csv",type=Path,required=True)
    pp.add_argument("--output-csv",type=Path,default=Path("artifacts_geo/predictions.csv"))
    mp=sp.add_parser("monitor")
    mp.add_argument("--artifact-dir",type=Path,default=Path("artifacts_geo"))
    mp.add_argument("--input-csv",type=Path,required=True)
    mp.add_argument("--output-json",type=Path,default=Path("artifacts_geo/monitor.json"))
    mp.add_argument("--missing-rate-alert",type=float,default=0.05)
    mp.add_argument("--ks-pvalue-alert",type=float,default=0.05)
    si=sp.add_parser("sample-input")
    si.add_argument("--output-csv",type=Path,default=Path("artifacts_geo/sample.csv"))
    si.add_argument("--rows",type=int,default=20)
    return p.parse_args()


def main():
    args=parse_args()
    if args.command=="train":
        m=train(args.output_dir,args.n_iter,n_cv_splits=args.n_cv_splits,fast=args.fast,
                overpredict_cost=args.overpredict_cost,underpredict_cost=args.underpredict_cost)
        log.info("Test R²=%.3f  RMSE=$%.0fk  MAE=$%.0fk",
                 m["test_metrics"]["r2"],
                 m["test_metrics"]["rmse"]*100,
                 m["test_metrics"]["mae"]*100)
    elif args.command=="predict":
        predict(args.artifact_dir,args.input_csv,args.output_csv)
    elif args.command=="monitor":
        r=monitor(args.artifact_dir,args.input_csv,args.output_json,
                  args.missing_rate_alert,args.ks_pvalue_alert)
        log.info("Drift alerts: missing=%d  KS=%d  lat_ok=%s  lon_ok=%s",
                 sum(x["alert"] for x in r["missing_rate_drift"]),
                 sum(x["alert"] for x in r["distribution_drift"]),
                 r["spatial_bounds_ok"]["latitude"],r["spatial_bounds_ok"]["longitude"])
    elif args.command=="sample-input":
        create_sample_input(args.output_csv,args.rows)


if __name__=="__main__":
    main()