"""Single-day kriging predictions for all 9 (transform × model) combinations.

Prerequisites: run_variogram and build_monthly_grids must complete first.

Usage:
    python -m thesis.scripts.run_viz_day --date 2013-02-01
    python -m thesis.scripts.run_viz_day --auto-date
    python -m thesis.scripts.run_viz_day --date 2013-02-01 --loo-cv
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.scripts._common import (
    APP_ROOT, ensure_app_root, log, download_from_s3,
    load_and_fit_pipeline,
)
from thesis.transforms.kriging_transform import KrigingTransform, TRANSFORMS

warnings.filterwarnings("ignore")

# Prevent OpenBLAS/MKL thread-pool deadlocks after fork
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ensure_app_root()

DIR_OUT = APP_ROOT / "outputs" / "viz_day"
DIR_OUT.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-day predictions for all 9 combos")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", type=str,
                   help="Date to predict (YYYY-MM-DD)")
    g.add_argument("--auto-date", action="store_true",
                   help="Pick the day with the most wet stations automatically")
    p.add_argument("--loo-cv", action="store_true",
                   help="Also run LOO-CV for this day (slower)")
    p.add_argument("--no-upload", action="store_true",
                   help="Skip S3 upload")
    return p.parse_args()


# ── module-level workers (must be picklable for joblib/loky) ─────────────

_CHUNK_SIZE = 25_000
_K_MC = 100


def _predict_one_combo(
    tr: str,
    model_name: str,
    vgm_info: dict,
    x_wet: np.ndarray,
    y_wet: np.ndarray,
    z_wet: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    norm_grid: np.ndarray,
    kt: KrigingTransform,
    wet_grid: np.ndarray,
    max_wet: int | None = None,
    seed: int = 42,
) -> tuple[tuple[str, str], np.ndarray, np.ndarray, np.ndarray, str]:
    """Run one (transform × model) amount prediction + MC back-transformation."""
    from thesis.models.kriging.ordinary import _build_pykrige_ok, _predict_chunked

    key = (tr, model_name)
    n_cells = len(grid_x)

    t_start = time.time()
    krig = _build_pykrige_ok(
        x_wet, y_wet, z_wet,
        model_name, vgm_info["params_dict"],
    )
    t_build = time.time() - t_start

    # Stage 2: amount kriging (only at wet grid cells)
    t_pred_start = time.time()
    z_pred, z_var = _predict_chunked(
        krig, grid_x, grid_y, chunk_size=_CHUNK_SIZE,
        n_closest_points=max_wet,
    )
    t_pred = time.time() - t_pred_start

    # MC back-transformation with antithetic variates (matches ordinary.py)
    rng = np.random.default_rng(seed)
    z_sigma = np.sqrt(np.maximum(z_var, 1e-8))[:, None]
    half = _K_MC // 2
    eps = rng.standard_normal((len(z_pred), half))
    eps = np.concatenate([eps, -eps], axis=1)
    z_samp = z_pred[:, None] + z_sigma * eps
    quota_samp = np.maximum(kt.inv(z_samp.ravel()), 0.0).reshape(len(z_pred), _K_MC)
    quota_pred = quota_samp.mean(axis=1)
    var_quota = quota_samp.var(axis=1)

    # Apply indicator mask: dry cells → 0
    precip_pred = np.where(wet_grid, np.maximum(quota_pred * norm_grid, 0.0), 0.0)
    var_mm2 = np.where(wet_grid, var_quota * norm_grid ** 2, 0.0)
    quota_pred = np.where(wet_grid, quota_pred, 0.0)

    n_wet_cells = int(wet_grid.sum())
    msg = (f"  {tr:15} + {model_name:12}  "
           f"→ {precip_pred.max():.2f} mm max, "
           f"{n_wet_cells}/{n_cells} wet cells  "
           f"(build {t_build:.1f}s, predict {t_pred:.1f}s)")
    return key, precip_pred, var_mm2, quota_pred, msg


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t0 = time.time()

    from thesis.config import Config
    from thesis.data.registry import DataRegistry
    from thesis.datasets.protocols import PredictionGrid
    from thesis.data.dem import DEMSource
    from thesis.models.kriging.variogram_fitter import GlobalVariogramFitter

    cfg = Config()
    registry = DataRegistry.from_config(cfg)

    # ── 1. Load & transform data ──────────────────────────────────────────
    (all_raw, all_proc,
     fwd, inv, _proc_by_date, get_mean_monthly_total) = load_and_fit_pipeline(
        cfg, registry, cfg.date_start, cfg.date_end,
    )
    log(f"  Pipeline done. {all_proc.shape}")

    # ── 2. Pick date ──────────────────────────────────────────────────────
    wet_counts = all_proc.groupby("date")["rain_indicator"].sum()
    if args.auto_date:
        test_date = str(wet_counts.idxmax())
        log(f"Auto-selected date: {test_date}  ({int(wet_counts[test_date])} wet stations)")
    else:
        test_date = args.date
        if test_date not in wet_counts.index:
            log(f"ERROR: date {test_date} not found in dataset.")
            sys.exit(1)
        log(f"Date: {test_date}  ({int(wet_counts[test_date])} wet stations)")

    proc_day = all_proc[all_proc["date"] == test_date].copy()

    # ── 3. Load variograms ────────────────────────────────────────────────
    vgm_path = APP_ROOT / "outputs" / "ordinary_kriging" / "global_variograms.pkl"
    if not vgm_path.exists():
        log("Variograms not found locally. Downloading from S3…")
        if not download_from_s3("results/ordinary_kriging/global_variograms.pkl", vgm_path):
            log("ERROR: variograms not found. Run run_cv.py first.")
            sys.exit(1)

    global_vgm = GlobalVariogramFitter.load(str(vgm_path))
    log(f"Loaded variograms: {list(global_vgm.keys())}")

    # ── 4. Build prediction grid ──────────────────────────────────────────
    log("Building prediction grid…")
    dem = DEMSource(cfg)
    grid = PredictionGrid.from_config(cfg, dem=dem)
    log(f"  Grid: {grid.n_cells():,} cells, shape {grid.shape}")

    # ── 5. Stage 1: indicator kriging → wet/dry grid mask ───────────────────
    from thesis.models.kriging.ordinary import _build_pykrige_ok, _predict_chunked
    from joblib import Parallel, delayed

    log("="*60)
    n_cores = os.cpu_count() or 1

    MODELS = ["spherical", "exponential", "gaussian"]

    wet_mask   = proc_day["rain_indicator"] == 1
    x_all      = proc_day["x_proj"].values
    y_all      = proc_day["y_proj"].values
    x_wet      = proc_day.loc[wet_mask, "x_proj"].values
    y_wet      = proc_day.loc[wet_mask, "y_proj"].values
    grid_x     = grid.coords_proj[:, 0]
    grid_y     = grid.coords_proj[:, 1]

    # Stage 1: indicator kriging (global spherical variogram — Haylock 2008)
    ind_key = ("indicator", "spherical")
    ind_info = global_vgm.get(ind_key)
    if ind_info is None:
        log("ERROR: Global indicator variogram not found in pkl.")
        log("  Run `python -m thesis.scripts.run_variogram --force` to fit it.")
        sys.exit(1)

    log("Stage 1: indicator kriging (spherical, global variogram)…")
    t_ind = time.time()
    z_ind = proc_day["rain_indicator"].values.astype(float)
    n_wet_stations = int(z_ind.sum())

    if n_wet_stations == len(z_ind):
        p_rain = np.ones(len(grid_x))
        wet_grid = np.ones(len(grid_x), dtype=bool)
        log(f"  All {len(z_ind)} stations wet → skipping indicator kriging, "
            f"all {len(grid_x)} cells marked wet  ({time.time()-t_ind:.1f}s)")
    elif n_wet_stations == 0:
        p_rain = np.zeros(len(grid_x))
        wet_grid = np.zeros(len(grid_x), dtype=bool)
        log(f"  All {len(z_ind)} stations dry → skipping indicator kriging, "
            f"all cells marked dry  ({time.time()-t_ind:.1f}s)")
    else:
        indicator_krig = _build_pykrige_ok(
            x_all, y_all, z_ind,
            ind_info["model"], ind_info["params_dict"],
        )
        p_rain, _ = _predict_chunked(indicator_krig, grid_x, grid_y, chunk_size=_CHUNK_SIZE)
        p_rain = np.clip(p_rain, 0.0, 1.0)
        wet_grid = p_rain > cfg.kriging.indicator_probability_threshold
        n_wet_cells = int(wet_grid.sum())
        log(f"  P(rain) threshold={cfg.kriging.indicator_probability_threshold}  "
            f"→ {n_wet_cells}/{len(grid_x)} wet cells ({100*n_wet_cells/len(grid_x):.1f}%)  "
            f"({time.time()-t_ind:.1f}s)")

    # ── 6. Load monthly norm grid ────────────────────────────────────────
    # Load spatially-varying monthly norm grid (TPS 3D with elevation)
    norm_grids_path = APP_ROOT / "outputs" / "ordinary_kriging" / "monthly_norm_grids.pkl"
    if not norm_grids_path.exists():
        log("Monthly norm grids not found locally. Downloading from S3…")
        if not download_from_s3("results/ordinary_kriging/monthly_norm_grids.pkl", norm_grids_path):
            log("ERROR: monthly norm grids not found. Run run_cv.py first.")
            sys.exit(1)

    with open(norm_grids_path, "rb") as f:
        norm_grids_dict = pickle.load(f)
    month_idx = int(test_date[5:7]) - 1  # 0-based
    if "grids_3d" in norm_grids_dict and norm_grids_dict["grids_3d"] is not None:
        norm_grid = norm_grids_dict["grids_3d"][month_idx]
        log(f"Using TPS 3D monthly norm grid (month {month_idx + 1})")
    else:
        norm_grid = norm_grids_dict["grids_2d"][month_idx]
        log(f"Using TPS 2D monthly norm grid (month {month_idx + 1})")

    # ── 7. Stage 2: amount predictions for all 9 combos (parallel) ────────
    # Pre-compute z_wet per transform
    z_wet_per_tr = {tr: fwd(proc_day.loc[wet_mask, "precip_quota"].values, tr) for tr in TRANSFORMS}

    # Build list of jobs — skip missing variograms
    jobs = []
    for tr in TRANSFORMS:
        for model_name in MODELS:
            key = (tr, model_name)
            if key not in global_vgm or global_vgm[key] is None:
                log(f"  {tr:15} + {model_name:12}  → SKIP (no variogram)")
                continue
            jobs.append((tr, model_name, global_vgm[key], z_wet_per_tr[tr]))

    n_jobs = min(n_cores, len(jobs))
    log(f"Stage 2: amount kriging — {len(jobs)} combinations, "
        f"n_jobs={n_jobs}")

    t_pred_start = time.time()
    max_wet = cfg.kriging.max_wet
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_predict_one_combo)(
            tr, model_name, vgm_info,
            x_wet, y_wet, z_wet,
            grid_x, grid_y, norm_grid,
            fwd.kts[tr],
            wet_grid,
            max_wet,
            cfg.random_seed,
        )
        for tr, model_name, vgm_info, z_wet in jobs
    )

    predictions: dict = {}
    variances: dict = {}
    quotas: dict = {}
    for key, pred, var, quota, msg in results:
        log(msg)
        predictions[key] = pred
        variances[key] = var
        quotas[key] = quota
    log(f"All {len(jobs)} predictions done in {time.time() - t_pred_start:.1f}s")

    # ── 8. LOO-CV (optional, parallel) ────────────────────────────────────
    loo_results: dict = {}

    if args.loo_cv:
        from thesis.models.kriging.loo_cv import _process_one_day_multi_vgm

        log("="*60)

        # Group by transform — _process_one_day_multi_vgm handles all models at once
        loo_by_transform: dict[str, list[tuple[str, dict]]] = {}
        for tr in TRANSFORMS:
            for model_name in MODELS:
                key = (tr, model_name)
                if key not in global_vgm or global_vgm[key] is None:
                    continue
                loo_by_transform.setdefault(tr, []).append((model_name, global_vgm[key]))

        n_combos = sum(len(v) for v in loo_by_transform.values())
        log(f"LOO-CV — {n_combos} combinations across {len(loo_by_transform)} transforms")

        loo_raw: dict[tuple[str, str], dict] = {}
        for tr, model_list in loo_by_transform.items():
            vgm_names = [m for m, _ in model_list]
            vgm_infos = [info for _, info in model_list]
            result = _process_one_day_multi_vgm(
                date=test_date,
                proc=proc_day,
                vgm_infos=vgm_infos,
                vgm_names=vgm_names,
                sub_tr=tr,
                fwd_fn=fwd,
                inv_fn=inv,
                get_monthly_total_fn=get_mean_monthly_total,
                n_stations_min=cfg.kriging.n_stations_min,
                k_mc=_K_MC,
                seed=42,
                norm_mode="station",
                max_wet=cfg.kriging.max_wet,
            )
            for model_name in vgm_names:
                loo_raw[(tr, model_name)] = result[model_name]

        for (tr, model_name), res in loo_raw.items():
            key = (tr, model_name)
            mae_vals     = res.get("mae", [])
            crps_mm_vals = res.get("crps_mm", [])
            if mae_vals:
                loo_results[key] = {
                    "n":           len(mae_vals),
                    "mae_mm":      float(np.mean(mae_vals)),
                    "crps_mm":     float(np.mean(crps_mm_vals)),
                    "mae_list":    mae_vals,
                    "crps_mm_list": crps_mm_vals,
                }
                log(f"  {tr:15} + {model_name:12}  "
                    f"n={loo_results[key]['n']:4d}  "
                    f"MAE={loo_results[key]['mae_mm']:.3f}  "
                    f"CRPS={loo_results[key]['crps_mm']:.3f}")
            else:
                log(f"  {tr:15} + {model_name:12}  → no valid stations")

    # ── 8. Save artefacts ─────────────────────────────────────────────────
    log("="*60)
    log("Saving artefacts…")

    pred_path = DIR_OUT / f"predictions_{test_date}.pkl"
    with open(pred_path, "wb") as f:
        pickle.dump(predictions, f)
    log(f"  {pred_path.name}")

    var_path = DIR_OUT / f"variances_{test_date}.pkl"
    with open(var_path, "wb") as f:
        pickle.dump(variances, f)
    log(f"  {var_path.name}")

    quota_path = DIR_OUT / f"quotas_{test_date}.pkl"
    with open(quota_path, "wb") as f:
        pickle.dump(quotas, f)
    log(f"  {quota_path.name}")

    # Extract 2D norm grid from already-loaded dict (avoid second pickle.load)
    norm_grid_2d = norm_grids_dict["grids_2d"][month_idx]

    grid_meta = {
        "date":        test_date,
        "shape":       grid.shape,
        "coords_proj": grid.coords_proj,
        "elevation_m": grid.elevation_m,
        "n_cells":     grid.n_cells(),
        "norm_grid_3d": norm_grid,
        "norm_grid_2d": norm_grid_2d,
        "p_rain":      p_rain,
        "wet_grid":    wet_grid,
        "indicator_threshold": cfg.kriging.indicator_probability_threshold,
    }
    grid_path = DIR_OUT / f"grid_meta_{test_date}.pkl"
    with open(grid_path, "wb") as f:
        pickle.dump(grid_meta, f)
    log(f"  {grid_path.name}")

    station_data = {
        "x_proj":         proc_day["x_proj"].values,
        "y_proj":         proc_day["y_proj"].values,
        "precip_mm":      proc_day["precip_mm"].values,
        "rain_indicator": proc_day["rain_indicator"].values,
    }
    stn_path = DIR_OUT / f"stations_{test_date}.pkl"
    with open(stn_path, "wb") as f:
        pickle.dump(station_data, f)
    log(f"  {stn_path.name}")

    if loo_results:
        loo_path = DIR_OUT / f"loo_cv_{test_date}.pkl"
        with open(loo_path, "wb") as f:
            pickle.dump(loo_results, f)
        log(f"  {loo_path.name}")

        rows = []
        for (tr, model_name), r in sorted(loo_results.items()):
            rows.append({"transform": tr, "model": model_name,
                         "n": r["n"], "mae_mm": r["mae_mm"], "crps_mm": r["crps_mm"]})
        csv_path = DIR_OUT / f"summary_{test_date}.csv"
        pd.DataFrame(rows).sort_values("crps_mm").to_csv(csv_path, index=False)
        log(f"  {csv_path.name}")

    # ── 9. Upload to S3 ───────────────────────────────────────────────────
    if not args.no_upload:
        from thesis.scripts.s3_upload import sync_to_s3
        log("Uploading to S3…")
        sync_to_s3(DIR_OUT, "results/viz_day")
        log("S3 upload complete.")

    elapsed = time.time() - t0
    log(f"Done in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
