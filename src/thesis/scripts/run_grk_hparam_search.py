"""GRK Stage 2 hyperparameter search via Optuna (LightGBM).

Computes LOO geo-features for all wet station-days (parallelised with joblib),
then runs Bayesian optimisation over LightGBM hyperparameters.

Usage:
    python -m thesis.scripts.run_grk_hparam_search
    python -m thesis.scripts.run_grk_hparam_search --n-trials 50 --no-upload
    python -m thesis.scripts.run_grk_hparam_search --force-recompute
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.scripts._common import APP_ROOT, ensure_app_root, log

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K_GEO         = 15
SVD_QUANTILES = np.arange(0.0, 1.05, 0.05)   # 21 quantiles
SOIL_VARS     = ["bulk_density", "clay", "sand", "silt", "soc", "water_10kpa"]
OVERFIT_PEN   = 0.3   # penalty weight when ratio > 1.3

DIR_GRK    = APP_ROOT / "outputs" / "grk"
LOO_CACHE  = DIR_GRK / "loo_features.parquet"
PARAMS_OUT = DIR_GRK / "stage2_best_params.json"
TRIALS_OUT = DIR_GRK / "stage2_optuna_trials.csv"


# ---------------------------------------------------------------------------
# Per-day LOO geo-feature worker (shared with k-fold CV).
#
# `_compute_day_loo` wraps `compute_day_geo_features` with the legacy
# all-stations-as-neighbours mask, reproducing the original per-station LOO
# behaviour. The k-fold script calls `compute_day_geo_features` directly with
# a fold-aware mask so test-fold stations are excluded from the neighbour
# pool, eliminating leakage.
# ---------------------------------------------------------------------------

from thesis.models.grk.features import compute_day_geo_features


def _compute_day_loo(
    date: str,
    xy: np.ndarray,
    z: np.ndarray,
    sids: np.ndarray,
    k: int,
    svd_quantiles: np.ndarray,
) -> list[dict]:
    train_mask = np.ones(len(z), dtype=bool)
    return compute_day_geo_features(
        date=date,
        xy_all=xy,
        z_all=z,
        sids_all=sids,
        train_mask=train_mask,
        k=k,
        svd_quantiles=svd_quantiles,
    )


# ---------------------------------------------------------------------------
# Optuna callback — logs each trial to stdout (→ entrypoint.sh → S3)
# ---------------------------------------------------------------------------

def _make_optuna_callback(n_trials: int):
    def callback(study, trial):
        attrs = trial.user_attrs
        log(
            f"Trial {trial.number:3d}/{n_trials}  "
            f"mae_val={attrs.get('mae_val', float('nan')):.4f}  "
            f"mae_train={attrs.get('mae_train', float('nan')):.4f}  "
            f"ratio={attrs.get('ratio', float('nan')):.2f}  "
            f"trees={attrs.get('n_trees', 0):4d}  "
            f"best={study.best_value:.4f}  "
            f"lr={trial.params.get('learning_rate', 0):.4f}  "
            f"leaves={trial.params.get('num_leaves', 0)}"
        )
    return callback


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    import os
    p = argparse.ArgumentParser(description="GRK Stage 2 Hyperparameter Search (Optuna + LightGBM)")
    p.add_argument("--n-trials",        type=int,   default=int(os.environ.get("N_TRIALS", 100)),
                   help="Number of Optuna trials (default: 100, or N_TRIALS env var)")
    p.add_argument("--val-frac",        type=float, default=float(os.environ.get("VAL_FRAC", 0.2)),
                   help="Fraction of stations for validation (default: 0.2, or VAL_FRAC env var)")
    p.add_argument("--force-recompute", action="store_true",
                   default=os.environ.get("FORCE_RECOMPUTE", "0") == "1",
                   help="Recompute LOO features (or set FORCE_RECOMPUTE=1)")
    p.add_argument("--no-upload",       action="store_true",
                   default=os.environ.get("NO_UPLOAD", "0") == "1",
                   help="Skip S3 upload (or set NO_UPLOAD=1)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    args = parse_args()
    ensure_app_root()
    DIR_GRK.mkdir(parents=True, exist_ok=True)

    log("GRK Stage 2 Hyperparameter Search (Optuna)")
    log(f"Period: 1961-01-01 → 2023-12-31  n_trials={args.n_trials}  val_frac={args.val_frac}")

    # ------------------------------------------------------------------
    # 1. Config + DataRegistry
    # ------------------------------------------------------------------
    from thesis.config import Config
    from thesis.data.registry import DataRegistry

    cfg = Config()
    registry = DataRegistry.from_config(cfg)

    # ------------------------------------------------------------------
    # 2. Load data — full 1961–2023, Projection + Indicator only
    # ------------------------------------------------------------------
    log("Loading station data 1961-01-01 → 2023-12-31 …")
    from thesis.transforms import ProjectionTransform, IndicatorTransform
    from thesis.transforms.pipeline import TransformPipeline

    all_raw = registry.stations.load("1961-01-01", "2023-12-31")
    log(f"  Loaded {len(all_raw):,} records, {all_raw['station_id'].nunique()} stations")

    proj = ProjectionTransform(target_crs=cfg.study_area.target_crs)
    ind  = IndicatorTransform(threshold_mm=cfg.wet_day_threshold_mm)
    all_proc = TransformPipeline([proj, ind]).fit_transform(all_raw)

    df_wet = all_proc[all_proc["rain_indicator"] == 1].copy()
    n_days = df_wet["date"].nunique()
    log(f"  Wet station-days: {len(df_wet):,}  ({n_days} unique wet days)")

    # ------------------------------------------------------------------
    # 3. SoilGrids static features — sample at station locations
    # ------------------------------------------------------------------
    log("Loading SoilGrids features …")
    station_coords = all_proc.groupby("station_id")[["x_proj", "y_proj"]].first()
    x_proj_arr = station_coords["x_proj"].values
    y_proj_arr = station_coords["y_proj"].values

    soil_rows: dict[str, np.ndarray] = {"station_id": station_coords.index.values}
    for var, src in registry.soilgrids.items():
        if var in SOIL_VARS:
            soil_rows[var] = src.sample_at_projected(x_proj_arr, y_proj_arr)

    soil_static = pd.DataFrame(soil_rows).set_index("station_id")
    available_soil = [v for v in SOIL_VARS if v in soil_static.columns]

    n_nan = int(soil_static[available_soil].isna().sum().sum())
    if n_nan > 0:
        for v in available_soil:
            soil_static[v] = soil_static[v].fillna(float(soil_static[v].median()))
        log(f"  SoilGrids: {len(available_soil)} vars at {len(station_coords)} stations, {n_nan} NaN → median")
    else:
        log(f"  SoilGrids: {len(available_soil)} vars at {len(station_coords)} stations, no NaN")

    # ------------------------------------------------------------------
    # 4. LOO geo-features — parallel or load from cache
    # ------------------------------------------------------------------
    if LOO_CACHE.exists() and not args.force_recompute:
        log(f"LOO features cache found — loading: {LOO_CACHE}")
        df_loo = pd.read_parquet(LOO_CACHE)
        df_loo["date"] = pd.to_datetime(df_loo["date"])
        log(f"  {len(df_loo):,} rows, {len(df_loo.columns)} columns")
    else:
        log("LOO features cache not found — computing (n_jobs=-1) …")

        # Pre-group by date into arrays (faster than filtering in worker)
        date_groups: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for date, grp in df_wet.groupby("date"):
            date_groups[str(date)] = (
                grp[["x_proj", "y_proj"]].values.astype(np.float64),
                grp["precip_mm"].values.astype(np.float64),
                grp["station_id"].values,
            )

        log(f"  {len(date_groups)} wet days to process …")

        from joblib import Parallel, delayed

        batches = Parallel(n_jobs=-1, backend="loky", verbose=10)(
            delayed(_compute_day_loo)(
                date, grp_xy, grp_z, grp_sids, K_GEO, SVD_QUANTILES,
            )
            for date, (grp_xy, grp_z, grp_sids) in date_groups.items()
        )

        records = [r for batch in batches for r in batch]
        df_loo = pd.DataFrame(records)
        df_loo["date"] = pd.to_datetime(df_loo["date"])

        df_loo.to_parquet(LOO_CACHE, index=False)
        log(f"  LOO features: {len(df_loo):,} station-days → {LOO_CACHE}")

    # ------------------------------------------------------------------
    # 5. Build full feature matrix
    # ------------------------------------------------------------------
    log("Building full feature matrix …")

    # Merge precip_mm + coords from df_wet if not already in LOO cache
    if "precip_mm" not in df_loo.columns:
        target_cols = ["station_id", "date", "precip_mm"]
        df_target = df_wet[target_cols].copy()
        df_target["date"] = pd.to_datetime(df_target["date"])
        df_all = df_loo.merge(df_target, on=["station_id", "date"], how="inner")
    else:
        df_all = df_loo.copy()

    # Merge soil static features
    soil_reset = soil_static.reset_index()
    df_all = df_all.merge(soil_reset, on="station_id", how="left")

    # Fill any remaining NaN in soil columns
    for sv in available_soil:
        if sv in df_all.columns:
            df_all[sv] = df_all[sv].fillna(float(df_all[sv].median()))

    # Feature columns: detect SVD cols by prefix (handles both svd_00 and svd_q000)
    svd_cols  = sorted([c for c in df_all.columns if c.startswith("svd_")])
    feat_cols = ["idw", "gos"] + svd_cols + available_soil
    feat_cols = [c for c in feat_cols if c in df_all.columns]

    log(f"  Feature matrix: {len(df_all):,} rows, {len(feat_cols)} features")

    # ------------------------------------------------------------------
    # 6. Random 80/20 station split
    # ------------------------------------------------------------------
    all_sids = df_all["station_id"].unique()
    rng = np.random.default_rng(42)
    n_val = int(len(all_sids) * args.val_frac)
    val_sids = set(rng.choice(all_sids, size=n_val, replace=False))

    train_mask = ~df_all["station_id"].isin(val_sids)
    val_mask   =  df_all["station_id"].isin(val_sids)

    X_tr = df_all.loc[train_mask, feat_cols].values.astype(np.float32)
    y_tr = df_all.loc[train_mask, "precip_mm"].values.astype(np.float32)
    X_va = df_all.loc[val_mask,   feat_cols].values.astype(np.float32)
    y_va = df_all.loc[val_mask,   "precip_mm"].values.astype(np.float32)

    log(
        f"Split: {len(all_sids) - n_val} train stations, {n_val} val  |  "
        f"X_train {X_tr.shape}  X_val {X_va.shape}"
    )

    # ------------------------------------------------------------------
    # 7. Optuna
    # ------------------------------------------------------------------
    import os
    # OMP_NUM_THREADS=1 was set in Dockerfile to prevent joblib worker oversubscription.
    # Must be unset BEFORE importing lightgbm — OpenMP reads it at C-library init time.
    for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.pop(_env, None)

    # cgroup-aware CPU detection: respects Docker / Vast.ai vCPU allocation,
    # not host-level os.cpu_count() which causes thread oversubscription.
    try:
        n_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cores = os.cpu_count() or 8
    log(f"  Using {n_cores} CPU cores for LightGBM (cgroup-aware)")

    import optuna
    import lightgbm as lgb

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log(f"Optuna: {args.n_trials} trials, overfit_penalty={OVERFIT_PEN} for ratio>1.3")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":       "regression_l1",
            "metric":          "mae",
            "verbosity":       -1,
            "n_jobs":          n_cores,
            "num_threads":     n_cores,
            "learning_rate":   trial.suggest_float("learning_rate",  0.005, 0.3,   log=True),
            "num_leaves":      trial.suggest_int  ("num_leaves",     15,    255),
            "max_depth":       trial.suggest_int  ("max_depth",      3,     12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":      1,
            "reg_alpha":         trial.suggest_float("reg_alpha",  1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        ds_tr = lgb.Dataset(X_tr, label=y_tr)
        ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        booster = lgb.train(
            params,
            ds_tr,
            num_boost_round=2000,
            valid_sets=[ds_tr, ds_va],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        n_trees  = booster.num_trees()
        mae_tr   = float(np.mean(np.abs(booster.predict(X_tr) - y_tr)))
        mae_va   = float(np.mean(np.abs(booster.predict(X_va) - y_va)))
        ratio    = mae_va / (mae_tr + 1e-8)
        penalty  = OVERFIT_PEN * max(0.0, ratio - 1.3) * mae_va
        score    = mae_va + penalty

        trial.set_user_attr("mae_val",   mae_va)
        trial.set_user_attr("mae_train", mae_tr)
        trial.set_user_attr("ratio",     ratio)
        trial.set_user_attr("n_trees",   n_trees)

        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[_make_optuna_callback(args.n_trials)],
        show_progress_bar=False,
    )

    # ------------------------------------------------------------------
    # 8. Report top-5 trials
    # ------------------------------------------------------------------
    log("=== TOP-5 TRIALS (by score = mae_val + overfit_penalty) ===")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))
    for t in trials_sorted[:5]:
        attrs = t.user_attrs
        log(
            f"  #{t.number:3d}  score={t.value:.4f}  "
            f"mae_val={attrs.get('mae_val', float('nan')):.4f}  "
            f"ratio={attrs.get('ratio', float('nan')):.2f}  "
            f"lr={t.params.get('learning_rate', 0):.4f}  "
            f"leaves={t.params.get('num_leaves', 0)}  "
            f"trees={attrs.get('n_trees', 0)}"
        )

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    best = study.best_trial

    best_params = dict(best.params)
    best_params["n_estimators"] = best.user_attrs.get("n_trees", 500)
    best_params["mae_val"]      = best.user_attrs.get("mae_val", None)
    best_params["mae_train"]    = best.user_attrs.get("mae_train", None)
    best_params["ratio"]        = best.user_attrs.get("ratio", None)

    PARAMS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_OUT, "w") as f:
        json.dump(best_params, f, indent=2)
    log(f"Saved: {PARAMS_OUT}")

    # Save all trials to CSV
    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        row = {"trial": t.number, "score": t.value}
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)
    pd.DataFrame(rows).to_csv(TRIALS_OUT, index=False)
    log(f"Saved: {TRIALS_OUT}")

    # ------------------------------------------------------------------
    # 10. S3 upload
    # ------------------------------------------------------------------
    if not args.no_upload:
        from thesis.scripts.s3_upload import sync_to_s3
        log("Uploading results to S3 …")
        sync_to_s3(DIR_GRK, "results/grk")
        log(f"  Uploaded → s3://thesis-data-ismaktam/results/grk/")
    else:
        log("S3 upload skipped (--no-upload)")

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    log(f"Done. Total: {mins}m {secs:02d}s")


if __name__ == "__main__":
    main()
