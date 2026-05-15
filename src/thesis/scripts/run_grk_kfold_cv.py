"""GRK Stage 2 random k-fold cross-validation (k=5 by default).

Replaces the per-station LOO evaluation. Stations are split into k folds
stratified by elevation zone (plains/hills/mountains) with a fixed seed, so
held-out stations stay spatially intermixed with training stations — the
correct CV regime for spatial *interpolation* (Roberts et al. 2017,
Hofstra/Haylock E-OBS protocol, gstat::krige.cv k-fold mode).

For every fold f, geo-features (IDW, GOS, SVD-quantiles) are *recomputed*
using only train-fold stations as the neighbour pool. This gives strict
leakage-free CV: held-out stations never appear in any other station's
neighbour set within the same fold.

Outputs are means ± std across folds (MAE, RMSE) plus a per-fold breakdown
by elevation zone.

Usage:
    python -m thesis.scripts.run_grk_kfold_cv
    python -m thesis.scripts.run_grk_kfold_cv --n-folds 5 --no-upload
    python -m thesis.scripts.run_grk_kfold_cv --force-recompute-features
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

# Elevation zones — same bins as the LOO script (Table tab:elevation_means
# in text/data_study_area.tex). Used here for *stratification* of the k-fold
# split rather than for sub-sampling.
ELEV_BINS    = [-np.inf, 250.0, 500.0, np.inf]
ELEV_LABELS  = ["plains", "hills", "mountains"]

DIR_GRK         = APP_ROOT / "outputs" / "grk"
PARAMS_BEST     = DIR_GRK / "stage2_best_params.json"
PRED_OUT        = DIR_GRK / "kfold_cv_predictions.parquet"
METRICS_OUT     = DIR_GRK / "kfold_cv_metrics.json"
ASSIGN_OUT      = DIR_GRK / "kfold_assignments.csv"
FEATURES_TMPL   = DIR_GRK / "kfold_features_fold{f}.parquet"

# Default LightGBM params used when stage2_best_params.json is unavailable.
DEFAULT_PARAMS = {
    "learning_rate":     0.05,
    "num_leaves":        63,
    "max_depth":         8,
    "min_child_samples": 50,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "reg_alpha":         1e-3,
    "reg_lambda":        1e-3,
    "n_estimators":      800,
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    import os
    p = argparse.ArgumentParser(description="GRK Stage 2 random k-fold CV")
    p.add_argument("--n-folds", type=int,
                   default=int(os.environ.get("N_FOLDS", 5)),
                   help="Number of folds (default: 5)")
    p.add_argument("--seed", type=int,
                   default=int(os.environ.get("SEED", 42)),
                   help="Random seed for stratified fold assignment")
    p.add_argument("--no-upload", action="store_true",
                   default=os.environ.get("NO_UPLOAD", "0") == "1",
                   help="Skip S3 upload (or set NO_UPLOAD=1)")
    p.add_argument("--force-recompute-features", action="store_true",
                   default=os.environ.get("FORCE_RECOMPUTE_FEATURES", "0") == "1",
                   help="Recompute per-fold geo-features even if cache exists")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_lgbm_params() -> dict:
    """Load tuned params from stage2_best_params.json or fall back to defaults."""
    if PARAMS_BEST.exists():
        with open(PARAMS_BEST, "r") as f:
            best = json.load(f)
        params = {k: v for k, v in best.items()
                  if k not in {"mae_val", "mae_train", "ratio"}}
        log(f"  Loaded tuned params from {PARAMS_BEST.name}")
        return params

    log("  stage2_best_params.json not found — falling back to DEFAULT_PARAMS")
    return dict(DEFAULT_PARAMS)


def assign_folds(
    station_meta: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Stratified k-fold assignment by elevation zone.

    Returns a copy of `station_meta` with added `elev_zone` and `fold`
    columns, where `fold ∈ [0, n_folds)`.
    """
    from sklearn.model_selection import StratifiedKFold

    df = station_meta.copy()
    df["elev_zone"] = pd.cut(
        df["elevation_m"], bins=ELEV_BINS, labels=ELEV_LABELS, right=False
    ).astype(str)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_arr = np.full(len(df), -1, dtype=np.int8)
    for f, (_, test_idx) in enumerate(skf.split(df, df["elev_zone"])):
        fold_arr[test_idx] = f
    df["fold"] = fold_arr

    log("  Fold assignment:")
    for f in range(n_folds):
        sub = df[df["fold"] == f]
        zones = sub["elev_zone"].value_counts().to_dict()
        zline = " | ".join(f"{z}={zones.get(z, 0)}" for z in ELEV_LABELS)
        log(f"    fold {f}: {len(sub):4d} stations  ({zline})")
    return df


def compute_fold_features(
    fold: int,
    df_wet: pd.DataFrame,
    train_station_ids: set[str],
    out_path: Path,
) -> pd.DataFrame:
    """Compute per-day geo-features for one fold (using train-fold neighbours
    only) and persist to parquet.
    """
    from joblib import Parallel, delayed
    from thesis.models.grk.features import compute_day_geo_features

    # Pre-group by date for joblib workers.
    date_groups: list[tuple] = []
    for date, grp in df_wet.groupby("date"):
        xy   = grp[["x_proj", "y_proj"]].values.astype(np.float64)
        z    = grp["precip_mm"].values.astype(np.float64)
        sids = grp["station_id"].values
        train_mask = np.array(
            [sid in train_station_ids for sid in sids], dtype=bool
        )
        # Skip days where the train side is too small to fit the kNN tree.
        if train_mask.sum() < 2:
            continue
        date_groups.append((str(date), xy, z, sids, train_mask))

    log(f"  fold {fold}: {len(date_groups)} wet days, "
        f"computing geo-features (n_jobs=-1) …")

    batches = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(compute_day_geo_features)(
            date, xy, z, sids, train_mask, K_GEO, SVD_QUANTILES,
        )
        for (date, xy, z, sids, train_mask) in date_groups
    )

    records = [r for batch in batches for r in batch]
    df_loo = pd.DataFrame(records)
    df_loo["date"] = pd.to_datetime(df_loo["date"])
    df_loo.to_parquet(out_path, index=False)
    log(f"  fold {fold}: saved {len(df_loo):,} rows → {out_path.name}")
    return df_loo


def fold_metrics(
    df: pd.DataFrame,
    label_col: str = "y_pred",
    truth_col: str = "y_true",
) -> dict:
    err = df[label_col] - df[truth_col]
    out = {
        "n_stations":   int(df["station_id"].nunique()),
        "n_predictions": int(len(df)),
        "mae":  float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "by_zone": {},
    }
    for zone in ELEV_LABELS:
        sub = df[df["elev_zone"] == zone]
        if len(sub) == 0:
            continue
        e = sub[label_col] - sub[truth_col]
        out["by_zone"][zone] = {
            "n_stations":   int(sub["station_id"].nunique()),
            "n_predictions": int(len(sub)),
            "mae":  float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    args = parse_args()
    ensure_app_root()
    DIR_GRK.mkdir(parents=True, exist_ok=True)

    log("GRK Stage 2 random k-fold CV (stratified by elevation zone)")
    log(f"  n_folds={args.n_folds}  seed={args.seed}")

    # ------------------------------------------------------------------
    # 0. Pull tuned params from S3 if missing locally.
    # ------------------------------------------------------------------
    import os
    if not PARAMS_BEST.exists():
        log("stage2_best_params.json missing locally — pulling from S3 …")
        os.system(
            f'aws s3 cp s3://thesis-data-ismaktam/results/grk/stage2_best_params.json "{PARAMS_BEST}"'
        )

    # ------------------------------------------------------------------
    # 1. Config + DataRegistry
    # ------------------------------------------------------------------
    from thesis.config import Config
    from thesis.data.registry import DataRegistry

    cfg = Config()
    registry = DataRegistry.from_config(cfg)

    # ------------------------------------------------------------------
    # 2. Load station data + project + indicator
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
    log(f"  Wet station-days: {len(df_wet):,}  "
        f"({df_wet['date'].nunique()} unique wet days)")

    # ------------------------------------------------------------------
    # 3. Station metadata for stratification
    # ------------------------------------------------------------------
    log("Building station metadata …")
    station_meta = (
        all_raw.groupby("station_id")
        .agg(lon=("lon", "first"),
             lat=("lat", "first"),
             elevation_m=("elevation_m", "first"))
        .reset_index()
    )

    n_missing = int(station_meta["elevation_m"].isna().sum())
    if n_missing > 0:
        log(f"  {n_missing} stations missing elevation_m — sampling from DEM")
        x_proj_full = all_proc.groupby("station_id")["x_proj"].first().loc[station_meta["station_id"]].values
        y_proj_full = all_proc.groupby("station_id")["y_proj"].first().loc[station_meta["station_id"]].values
        dem_z = registry.dem.sample_at_projected(x_proj_full, y_proj_full)
        station_meta["elevation_m"] = station_meta["elevation_m"].fillna(
            pd.Series(dem_z, index=station_meta.index)
        )

    # ------------------------------------------------------------------
    # 4. K-fold assignment
    # ------------------------------------------------------------------
    assignments = assign_folds(station_meta, args.n_folds, args.seed)
    ASSIGN_OUT.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(ASSIGN_OUT, index=False)
    log(f"  Saved fold assignments → {ASSIGN_OUT.name}")

    # Restrict to stations actually present in the wet-day universe.
    universe_ids = set(df_wet["station_id"].unique())
    assignments_in_universe = assignments[assignments["station_id"].isin(universe_ids)].reset_index(drop=True)
    fold_lookup = dict(zip(assignments_in_universe["station_id"], assignments_in_universe["fold"]))
    zone_lookup = dict(zip(assignments_in_universe["station_id"], assignments_in_universe["elev_zone"]))

    # ------------------------------------------------------------------
    # 5. SoilGrids static features (once for all folds)
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
    for v in available_soil:
        soil_static[v] = soil_static[v].fillna(float(soil_static[v].median()))
    log(f"  SoilGrids: {len(available_soil)} vars at {len(station_coords)} stations")

    # ------------------------------------------------------------------
    # 6. LightGBM threading config (cgroup-aware)
    # ------------------------------------------------------------------
    for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.pop(_env, None)
    try:
        n_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cores = os.cpu_count() or 8
    log(f"  Using {n_cores} CPU cores for LightGBM (cgroup-aware)")

    import lightgbm as lgb

    base_params = load_lgbm_params()
    n_estimators = int(base_params.pop("n_estimators", 800))
    base_params.update({
        "objective":   "regression_l1",
        "metric":      "mae",
        "verbosity":   -1,
        "device":      "cuda",
        "max_bin":     63,
        "n_jobs":      n_cores,
        "num_threads": n_cores,
        "bagging_freq": 1,
    })

    # ------------------------------------------------------------------
    # 7. Per-fold loop
    # ------------------------------------------------------------------
    df_target = df_wet[["station_id", "date", "precip_mm"]].copy()
    df_target["date"] = pd.to_datetime(df_target["date"])

    soil_reset = soil_static.reset_index()

    per_fold_metrics: list[dict] = []
    pred_records: list[pd.DataFrame] = []

    for f in range(args.n_folds):
        log(f"=== Fold {f+1}/{args.n_folds} ===")
        t_fold = time.time()

        train_station_ids = {
            sid for sid, fid in fold_lookup.items() if fid != f
        }
        test_station_ids = {
            sid for sid, fid in fold_lookup.items() if fid == f
        }
        log(f"  train stations: {len(train_station_ids)}  "
            f"test stations: {len(test_station_ids)}")

        # 7a. Geo-features for this fold (cached unless --force-recompute-features)
        feat_path = Path(str(FEATURES_TMPL).format(f=f))
        if feat_path.exists() and not args.force_recompute_features:
            log(f"  fold {f}: loading cached features {feat_path.name}")
            df_loo = pd.read_parquet(feat_path)
            df_loo["date"] = pd.to_datetime(df_loo["date"])
        else:
            df_loo = compute_fold_features(
                fold=f,
                df_wet=df_wet,
                train_station_ids=train_station_ids,
                out_path=feat_path,
            )

        # 7b. Build feature matrix
        df_all = df_loo.merge(df_target, on=["station_id", "date"], how="inner")
        df_all = df_all.merge(soil_reset, on="station_id", how="left")
        for sv in available_soil:
            if sv in df_all.columns:
                df_all[sv] = df_all[sv].fillna(float(df_all[sv].median()))

        svd_cols  = sorted([c for c in df_all.columns if c.startswith("svd_")])
        feat_cols = ["idw", "gos"] + svd_cols + available_soil
        feat_cols = [c for c in feat_cols if c in df_all.columns]

        # 7c. Split rows by station fold membership
        is_test  = df_all["station_id"].isin(test_station_ids).values
        is_train = ~is_test

        X_tr = df_all.loc[is_train, feat_cols].values.astype(np.float32)
        y_tr = df_all.loc[is_train, "precip_mm"].values.astype(np.float32)
        X_te = df_all.loc[is_test,  feat_cols].values.astype(np.float32)
        y_te = df_all.loc[is_test,  "precip_mm"].values.astype(np.float32)

        log(f"  fold {f}: X_train {X_tr.shape}  X_test {X_te.shape}  features={len(feat_cols)}")

        # 7d. Train LightGBM
        ds_tr = lgb.Dataset(X_tr, label=y_tr)
        booster = lgb.train(
            base_params,
            ds_tr,
            num_boost_round=n_estimators,
            callbacks=[lgb.log_evaluation(-1)],
        )

        # 7e. Predict on held-out fold
        y_pred = booster.predict(X_te)

        df_te = df_all.loc[is_test, ["station_id", "date"]].copy()
        df_te["fold"]      = f
        df_te["elev_zone"] = df_te["station_id"].map(zone_lookup)
        df_te["y_true"]    = y_te
        df_te["y_pred"]    = y_pred.astype(np.float32)

        m = fold_metrics(df_te)
        m["fold"] = f
        per_fold_metrics.append(m)
        pred_records.append(df_te)

        elapsed = time.time() - t_fold
        log(f"  fold {f}: MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
            f"({elapsed/60:.1f} min)")

    # ------------------------------------------------------------------
    # 8. Aggregate + save
    # ------------------------------------------------------------------
    df_pred = pd.concat(pred_records, ignore_index=True)
    df_pred.to_parquet(PRED_OUT, index=False)
    log(f"Saved predictions → {PRED_OUT.name}")

    mae_arr  = np.array([m["mae"]  for m in per_fold_metrics])
    rmse_arr = np.array([m["rmse"] for m in per_fold_metrics])

    err_pooled = df_pred["y_pred"] - df_pred["y_true"]
    summary = {
        "mae_mean":  float(mae_arr.mean()),
        "mae_std":   float(mae_arr.std(ddof=1)) if len(mae_arr) > 1 else 0.0,
        "rmse_mean": float(rmse_arr.mean()),
        "rmse_std":  float(rmse_arr.std(ddof=1)) if len(rmse_arr) > 1 else 0.0,
        "mae_pooled":  float(np.mean(np.abs(err_pooled))),
        "rmse_pooled": float(np.sqrt(np.mean(err_pooled ** 2))),
        "by_zone": {},
    }
    for zone in ELEV_LABELS:
        zone_mae  = [m["by_zone"].get(zone, {}).get("mae")  for m in per_fold_metrics]
        zone_rmse = [m["by_zone"].get(zone, {}).get("rmse") for m in per_fold_metrics]
        zone_mae  = [v for v in zone_mae  if v is not None]
        zone_rmse = [v for v in zone_rmse if v is not None]
        if not zone_mae:
            continue
        summary["by_zone"][zone] = {
            "mae_mean":  float(np.mean(zone_mae)),
            "mae_std":   float(np.std(zone_mae,  ddof=1)) if len(zone_mae)  > 1 else 0.0,
            "rmse_mean": float(np.mean(zone_rmse)),
            "rmse_std":  float(np.std(zone_rmse, ddof=1)) if len(zone_rmse) > 1 else 0.0,
        }

    metrics = {
        "n_folds":    args.n_folds,
        "seed":       args.seed,
        "n_stations": int(len(fold_lookup)),
        "per_fold":   per_fold_metrics,
        "summary":    summary,
    }

    with open(METRICS_OUT, "w") as fh:
        json.dump(metrics, fh, indent=2)
    log(f"Saved metrics → {METRICS_OUT.name}")
    log(f"  MAE  = {summary['mae_mean']:.3f} ± {summary['mae_std']:.3f}  "
        f"(pooled {summary['mae_pooled']:.3f})")
    log(f"  RMSE = {summary['rmse_mean']:.3f} ± {summary['rmse_std']:.3f}  "
        f"(pooled {summary['rmse_pooled']:.3f})")
    for zone, zm in summary["by_zone"].items():
        log(f"  {zone:<9} MAE={zm['mae_mean']:.3f}±{zm['mae_std']:.3f}  "
            f"RMSE={zm['rmse_mean']:.3f}±{zm['rmse_std']:.3f}")

    # ------------------------------------------------------------------
    # 9. S3 upload
    # ------------------------------------------------------------------
    if not args.no_upload:
        from thesis.scripts.s3_upload import sync_to_s3
        log("Uploading results to S3 …")
        sync_to_s3(DIR_GRK, "results/grk")
        log("  Uploaded → s3://thesis-data-ismaktam/results/grk/")
    else:
        log("S3 upload skipped (--no-upload)")

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    log(f"Done. Total: {mins}m {secs:02d}s")


if __name__ == "__main__":
    main()
