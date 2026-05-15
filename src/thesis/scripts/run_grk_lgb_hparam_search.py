"""GRK Stage 2 — LightGBM hyperparameter / loss-function search (cloud script).

Mechanical conversion of `notebooks/04_grk/grk_lgb_hparam_search.ipynb` for
headless cloud execution. Single stratified 80/20 station split (random,
seed=42) — NOT k-fold. The chosen hparams are re-validated by
`run_grk_kfold_cv.py` afterwards.

Outputs (to ``outputs/grk/``):
    - hparam_features_train80.parquet   (cached leakage-free geo-features)
    - hparam_search_results.csv         (one row per config)
    - hparam_search_summary.png         (RMSE / MAE_heavy / bias / ceiling)
    - hparam_search_top3_scatter.png    (y_true vs y_pred for top 3)
    - lgb_best_params.json              (best mean-targeting config)

Usage:
    python -m thesis.scripts.run_grk_lgb_hparam_search
    python -m thesis.scripts.run_grk_lgb_hparam_search --no-upload
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — must come before pyplot import
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from thesis.scripts._common import APP_ROOT, ensure_app_root, log

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

DATA = APP_ROOT / "data" / "rekis"
OUT  = APP_ROOT / "outputs" / "grk"

K_NEIGHBOURS    = 15
SVD_QUANTILES   = np.linspace(0.0, 1.0, 21)
SOILGRIDS_VARS  = ["bulk_density", "clay", "sand", "silt", "soc", "water_10kpa"]

ELEV_BINS   = [-np.inf, 250.0, 500.0, np.inf]
ELEV_LABELS = ["plains", "hills", "mountains"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRK Stage 2 LightGBM hparam search")
    p.add_argument("--no-upload", action="store_true",
                   help="Skip S3 upload at the end")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Heavy imports deferred so the module imports cleanly without lightgbm
    # installed locally (mirrors run_grk_kfold_cv.py).
    from joblib import Parallel, delayed
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from pyproj import Transformer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    from thesis.config import Config
    from thesis.data.soilgrids import SoilGridsSource
    from thesis.models.grk.features import compute_day_geo_features

    args = parse_args()
    ensure_app_root()
    OUT.mkdir(parents=True, exist_ok=True)

    log(f"repo: {APP_ROOT}")
    log(f"data: {DATA}")
    log(f"out : {OUT}")

    # ------------------------------------------------------------------
    # 1. Stations + precipitation
    # ------------------------------------------------------------------
    stations = pd.read_csv(DATA / "Stationsliste.txt")
    stations = stations.rename(columns={
        "Stat_ID": "station_id",
        "Laenge": "lon", "Breite": "lat", "Hoehe": "elevation_m",
    })[["station_id", "lon", "lat", "elevation_m"]]

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x, y = tr.transform(stations["lon"].values, stations["lat"].values)
    stations["x_proj"] = x
    stations["y_proj"] = y
    stations["elev_zone"] = pd.cut(
        stations["elevation_m"], bins=ELEV_BINS, labels=ELEV_LABELS,
    )

    rr = pd.read_csv(DATA / "RR.csv", sep=";", decimal=",", parse_dates=["zeit"])
    rr = rr.rename(columns={"zeit": "date"})
    mask = (rr["date"] >= "2000-01-01") & (rr["date"] <= "2023-12-31")
    rr = rr.loc[mask].copy()
    rr["date"] = rr["date"].dt.date

    long = rr.melt(id_vars="date", var_name="station_id", value_name="precip_mm")
    long = long.dropna(subset=["precip_mm"])
    long = long[long["precip_mm"] >= 0.5].copy()
    long = long.merge(stations[["station_id"]], on="station_id", how="inner")

    active = stations[stations["station_id"].isin(long["station_id"])].copy()
    active = active.dropna(subset=["elev_zone"]).reset_index(drop=True)
    active = active.drop_duplicates(subset="station_id", keep="first").reset_index(drop=True)
    long   = long.drop_duplicates(subset=["date", "station_id"], keep="first")
    long   = long[long["station_id"].isin(active["station_id"])].copy()

    log(f"active stations: {len(active):,}")
    log(f"wet station-days: {len(long):,}")
    log(f"unique days: {long['date'].nunique():,}")

    # ------------------------------------------------------------------
    # 1b. SoilGrids static features per station
    # ------------------------------------------------------------------
    cfg = Config()
    cfg.paths.root  = APP_ROOT / "data"
    cfg.paths.cache = APP_ROOT / "data" / "cache"

    for v in SOILGRIDS_VARS:
        src = SoilGridsSource(cfg, variable=v, depth=None)  # depth-averaged
        active[v] = src.sample_at_projected(
            active["x_proj"].to_numpy(), active["y_proj"].to_numpy(),
        )

    n_nan = active[SOILGRIDS_VARS].isna().any(axis=1).sum()
    log(f"SoilGrids attached for {len(active) - n_nan:,} stations  (NaN: {n_nan})")

    # ------------------------------------------------------------------
    # 2. Stratified 80/20 station split
    # ------------------------------------------------------------------
    train_sids, val_sids = train_test_split(
        active["station_id"].to_numpy(),
        test_size=0.2, random_state=42,
        stratify=active["elev_zone"].astype(str).to_numpy(),
    )
    train_set = set(train_sids)
    val_set   = set(val_sids)
    log(f"train stations: {len(train_set):,}, val stations: {len(val_set):,}")

    # ------------------------------------------------------------------
    # 3. Leakage-free geo-features (reuse cached parquet if present)
    # ------------------------------------------------------------------
    feats_path = OUT / "hparam_features_train80.parquet"
    if feats_path.exists():
        log(f"reusing cached features: {feats_path}")
        feats = pd.read_parquet(feats_path)
        log(f"  {len(feats):,} rows loaded")
    else:
        active_idx = active.set_index("station_id")[["x_proj", "y_proj", "elevation_m"]]

        def _per_day(date):
            day = long[long["date"] == date]
            if len(day) < 3:
                return []
            sids   = day["station_id"].values
            coords = active_idx.loc[sids, ["x_proj", "y_proj"]].values
            z      = day["precip_mm"].values
            train_mask = np.array([s in train_set for s in sids], dtype=bool)
            return compute_day_geo_features(
                date=str(date), xy_all=coords, z_all=z, sids_all=sids,
                train_mask=train_mask, k=K_NEIGHBOURS, svd_quantiles=SVD_QUANTILES,
            )

        days = sorted(long["date"].unique())
        t0 = time.time()
        out = Parallel(n_jobs=-1, verbose=0)(
            delayed(_per_day)(d) for d in tqdm(days, desc="geo-features")
        )
        feats_flat = [r for sub in out for r in sub]
        feats = pd.DataFrame(feats_flat)
        log(f"features: {len(feats):,} rows in {time.time()-t0:.1f}s")

        feats.to_parquet(feats_path)
        log(f"cached: {feats_path}")

    # ------------------------------------------------------------------
    # 4. Training matrix
    # ------------------------------------------------------------------
    FEATURE_COLS = (
        ["idw", "gos"]
        + [f"svd_{i:02d}" for i in range(21)]
        + ["x_proj", "y_proj", "elevation_m"]
        + SOILGRIDS_VARS
    )

    df = feats.merge(
        active[["station_id", "x_proj", "y_proj", "elevation_m", "elev_zone"] + SOILGRIDS_VARS],
        on="station_id", how="left",
    )
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.merge(long[["date", "station_id", "precip_mm"]], on=["date", "station_id"], how="inner")
    df = df.dropna(subset=FEATURE_COLS).copy()
    df["split"] = np.where(df["station_id"].isin(train_set), "train", "val")

    train = df[df["split"] == "train"].copy()
    val   = df[df["split"] == "val"].copy()
    X_tr, y_tr = train[FEATURE_COLS], train["precip_mm"]
    X_va, y_va = val[FEATURE_COLS],   val["precip_mm"]
    log(f"train: {len(X_tr):,}  val: {len(X_va):,}")

    # ------------------------------------------------------------------
    # 5. Hyperparameter grid
    # ------------------------------------------------------------------
    _METRIC_FOR = {
        "regression":    "l2",
        "regression_l1": "l1",
        "tweedie":       "tweedie",
        "quantile":      "quantile",
    }
    MONOTONE = [
        1 if c in ("idw", "gos") or c.startswith("svd_") else 0
        for c in FEATURE_COLS
    ]
    assert len(MONOTONE) == len(FEATURE_COLS)

    configs = [
        {"name": "mae",                 "objective": "regression_l1"},
        {"name": "mse",                 "objective": "regression"},
        {"name": "quantile_a050",       "objective": "quantile", "alpha": 0.5},
        {"name": "quantile_a070",       "objective": "quantile", "alpha": 0.7},
        {"name": "quantile_a090",       "objective": "quantile", "alpha": 0.9},
        {"name": "tweedie_p11",         "objective": "tweedie", "tweedie_variance_power": 1.1},
        {"name": "tweedie_p13",         "objective": "tweedie", "tweedie_variance_power": 1.3},
        {"name": "tweedie_p15",         "objective": "tweedie", "tweedie_variance_power": 1.5},
        {"name": "tweedie_p17",         "objective": "tweedie", "tweedie_variance_power": 1.7},
        {"name": "tweedie_p19",         "objective": "tweedie", "tweedie_variance_power": 1.9},
        {"name": "tweedie_p145",        "objective": "tweedie", "tweedie_variance_power": 1.45},
        {"name": "tweedie_p155",        "objective": "tweedie", "tweedie_variance_power": 1.55},
        {"name": "tweedie_p165",        "objective": "tweedie", "tweedie_variance_power": 1.65},
        {"name": "tweedie_p175",        "objective": "tweedie", "tweedie_variance_power": 1.75},
        {"name": "tweedie_p15_big",     "objective": "tweedie", "tweedie_variance_power": 1.5,
                                        "num_leaves": 127, "learning_rate": 0.03},
        {"name": "tweedie_p17_big",     "objective": "tweedie", "tweedie_variance_power": 1.7,
                                        "num_leaves": 127, "learning_rate": 0.03},
        {"name": "tweedie_p15_minleaf200", "objective": "tweedie", "tweedie_variance_power": 1.5,
                                           "min_child_samples": 200},
        {"name": "tweedie_p15_l2_10",      "objective": "tweedie", "tweedie_variance_power": 1.5,
                                           "reg_lambda": 10.0},
        {"name": "tweedie_p15_ff05",       "objective": "tweedie", "tweedie_variance_power": 1.5,
                                           "feature_fraction": 0.5},
        {"name": "tweedie_p15_monotone",   "objective": "tweedie", "tweedie_variance_power": 1.5,
                                           "monotone_constraints": MONOTONE},
        {"name": "mae_minleaf200",         "objective": "regression_l1",
                                           "min_child_samples": 200},
        {"name": "tweedie_p15_slow",       "objective": "tweedie", "tweedie_variance_power": 1.5,
                                           "learning_rate": 0.02, "n_estimators": 4000},
    ]
    log(f"{len(configs)} configs queued")

    # ------------------------------------------------------------------
    # 6. Run all configs
    # ------------------------------------------------------------------
    DEFAULTS = dict(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        verbosity=-1,
        random_state=42,
        feature_fraction_seed=42,
        bagging_seed=42,
    )

    def run_one(cfg):
        name = cfg["name"]
        params = {**DEFAULTS, **{k: v for k, v in cfg.items() if k != "name"}}
        obj = params.get("objective", "regression")
        native = _METRIC_FOR[obj]
        params["metric"] = [native] if native == "l1" else [native, "l1"]

        model = LGBMRegressor(**params)
        t1 = time.time()
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(50, first_metric_only=True), log_evaluation(0)],
        )
        dt = time.time() - t1

        y_pred = np.asarray(model.predict(X_va))
        if obj in ("regression", "regression_l1"):
            y_pred_eval = np.clip(y_pred, 0.0, None)
        else:
            y_pred_eval = y_pred

        heavy = y_va.values >= 20.0
        n_heavy = int(heavy.sum())

        return {
            "name":           name,
            "objective":      params.get("objective"),
            "tweedie_power":  params.get("tweedie_variance_power"),
            "alpha":          params.get("alpha"),
            "lr":             params["learning_rate"],
            "leaves":         params["num_leaves"],
            "min_child":      params["min_child_samples"],
            "reg_lambda":     params.get("reg_lambda", 0.0),
            "feat_frac":      params["feature_fraction"],
            "monotone":       params.get("monotone_constraints") is not None,
            "n_trees":        int(model.best_iteration_ or params["n_estimators"]),
            "fit_time_s":     round(dt, 1),
            "rmse_val":       float(np.sqrt(mean_squared_error(y_va, y_pred_eval))),
            "mae_val":        float(mean_absolute_error(y_va, y_pred_eval)),
            "r2_val":         float(r2_score(y_va, y_pred_eval)),
            "mae_val_heavy":  float(mean_absolute_error(y_va[heavy], y_pred_eval[heavy])) if n_heavy else np.nan,
            "bias_heavy":     float(np.mean(y_pred_eval[heavy] - y_va.values[heavy])) if n_heavy else np.nan,
            "pred_max":       float(y_pred_eval.max()),
            "pred_p99":       float(np.percentile(y_pred_eval, 99.0)),
            "n_heavy":        n_heavy,
        }, y_pred_eval

    results = []
    preds_per_cfg: dict[str, np.ndarray] = {}
    for cfg_i in configs:
        try:
            m, yp = run_one(cfg_i)
            results.append(m)
            preds_per_cfg[m["name"]] = yp
            log(f"  {m['name']:<26}  RMSE={m['rmse_val']:.3f}  MAE={m['mae_val']:.3f}  "
                f"MAE_heavy={m['mae_val_heavy']:.2f}  bias_heavy={m['bias_heavy']:+.2f}  "
                f"pred_max={m['pred_max']:.0f}  trees={m['n_trees']}  ({m['fit_time_s']}s)")
        except Exception as e:
            log(f"  {cfg_i['name']}: FAILED — {e}")
            results.append({"name": cfg_i["name"], "error": str(e)})

    results_df = pd.DataFrame(results).sort_values("rmse_val", na_position="last")
    results_path = OUT / "hparam_search_results.csv"
    results_df.to_csv(results_path, index=False)
    log(f"Saved: {results_path}")

    # ------------------------------------------------------------------
    # 7. Diagnostic plots
    # ------------------------------------------------------------------
    ok = results_df.dropna(subset=["rmse_val"]).copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.barh(ok["name"], ok["rmse_val"], color="steelblue")
    ax.invert_yaxis(); ax.set_xlabel("RMSE_val (mm)")
    ax.set_title("Overall RMSE — primary ranking criterion (lower = better)")
    for i, v in enumerate(ok["rmse_val"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)

    ax = axes[0, 1]
    ax.barh(ok["name"], ok["mae_val_heavy"], color="orange")
    ax.invert_yaxis(); ax.set_xlabel("MAE on y_true >= 20 mm")
    ax.set_title("Heavy-event MAE (lower = better)")
    for i, v in enumerate(ok["mae_val_heavy"]):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)

    ax = axes[1, 0]
    colors = ["red" if b < 0 else "green" for b in ok["bias_heavy"]]
    ax.barh(ok["name"], ok["bias_heavy"], color=colors)
    ax.invert_yaxis(); ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("bias on y_true >= 20 mm  (red = under-predicts)")
    ax.set_title("Heavy-event bias (closer to 0 = better)")
    for i, v in enumerate(ok["bias_heavy"]):
        ax.text(v, i, f" {v:+.1f}", va="center", fontsize=8)

    ax = axes[1, 1]
    ax.barh(ok["name"], ok["pred_max"], color="purple")
    ax.invert_yaxis(); ax.set_xlabel("max(y_pred)  (mm)")
    ax.set_title(f"Prediction ceiling — y_true.max() = {y_va.max():.1f} mm")
    for i, v in enumerate(ok["pred_max"]):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "hparam_search_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {OUT / 'hparam_search_summary.png'}")

    top3 = ok.head(3)["name"].tolist()
    log(f"Top 3 by RMSE_val: {top3}")
    fig, axes = plt.subplots(1, len(top3), figsize=(5 * len(top3), 5))
    if len(top3) == 1:
        axes = [axes]
    lim = max(y_va.max(), max(preds_per_cfg[n].max() for n in top3))
    for ax, name in zip(axes, top3):
        yp = preds_per_cfg[name]
        ax.scatter(y_va, yp, s=2, alpha=0.15)
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_xlabel("y_true (mm)"); ax.set_ylabel("y_pred (mm)")
        row = ok[ok.name == name].iloc[0]
        ax.set_title(f"{name}\nRMSE={row.rmse_val:.3f}  MAE={row.mae_val:.3f}  "
                     f"pred_max={row.pred_max:.0f}")
    plt.tight_layout()
    plt.savefig(OUT / "hparam_search_top3_scatter.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8. Save best config
    # ------------------------------------------------------------------
    _MEAN_OBJ = {"regression", "regression_l1", "tweedie"}
    _MEAN_NAMES = {
        c["name"] for c in configs
        if c.get("objective", "regression") in _MEAN_OBJ
        or c["name"] == "quantile_a050"
    }

    eligible = ok[ok["name"].isin(_MEAN_NAMES)].copy()
    eligible = eligible.sort_values(
        by=["rmse_val", "mae_val", "mae_val_heavy"],
        ascending=True, na_position="last",
    ).reset_index(drop=True)

    log("Top 5 mean-targeting configs (ranked by RMSE → MAE → MAE_heavy):")
    log(eligible[["name", "objective", "rmse_val", "mae_val",
                  "mae_val_heavy", "bias_heavy", "pred_max"]].head().to_string(index=False))

    best_name   = eligible.iloc[0]["name"]
    best_cfg    = next(c for c in configs if c["name"] == best_name)
    best_params = {**DEFAULTS, **{k: v for k, v in best_cfg.items() if k != "name"}}
    best_obj    = best_params.get("objective", "regression")
    native      = _METRIC_FOR[best_obj]
    best_params["metric"] = [native] if native == "l1" else [native, "l1"]

    best_path = OUT / "lgb_best_params.json"
    with open(best_path, "w") as f:
        json.dump(
            {
                "name":    best_name,
                "params":  best_params,
                "metrics": eligible.iloc[0].to_dict(),
                "ranking": "RMSE → MAE → MAE_heavy on mean-targeting objectives",
            },
            f,
            indent=2,
            default=str,
        )
    log(f"Saved: {best_path}")
    log(f"best: {best_name}")

    # ------------------------------------------------------------------
    # 9. S3 upload
    # ------------------------------------------------------------------
    if not args.no_upload:
        from thesis.scripts.s3_upload import sync_to_s3
        log("Uploading results to S3 …")
        sync_to_s3(OUT, "results/grk")
        log("  Uploaded → s3://thesis-data-ismaktam/results/grk/")
    else:
        log("S3 upload skipped (--no-upload)")


if __name__ == "__main__":
    main()
