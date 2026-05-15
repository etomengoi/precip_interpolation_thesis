"""DEM experiment: compare norm_mode = station / tps_2d / tps_3d via LOO-CV.

Prerequisites: run_variogram, build_monthly_grids, and run_cv must complete first.

Usage:
    python -m thesis.scripts.run_dem
    python -m thesis.scripts.run_dem --n-test-days 30 --no-upload
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.scripts._common import (
    APP_ROOT, ensure_app_root, log, download_from_s3,
    ProcByDateLoader, load_and_fit_pipeline,
)

warnings.filterwarnings("ignore")

ensure_app_root()

DIR_OK  = APP_ROOT / "outputs" / "ordinary_kriging"
DIR_CV  = APP_ROOT / "outputs" / "cross_validation"
DIR_OUT = APP_ROOT / "outputs" / "results"

for d in (DIR_OK, DIR_CV, DIR_OUT):
    d.mkdir(parents=True, exist_ok=True)

K_MC = 100
N_JOBS = -1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEM experiment: station / tps_2d / tps_3d LOO-CV")
    p.add_argument("--n-test-days", type=int, default=None,
                   help="Number of random test days (default: all)")
    p.add_argument("--no-upload", action="store_true",
                   help="Skip S3 upload")
    return p.parse_args()


def _build_station_norms_from_grid(
    grids: np.ndarray,
    grid_coords: np.ndarray,
    station_coords: np.ndarray,
) -> dict[int, np.ndarray]:
    """Map grid norms to station locations via nearest neighbour."""
    from scipy.spatial import cKDTree

    tree = cKDTree(grid_coords)
    _, indices = tree.query(station_coords, k=1)

    return {m + 1: grids[m][indices] for m in range(12)}


class _PrecomputedNormsLookup:
    """Picklable callable: date -> precomputed norms array for wet stations."""

    def __init__(
        self,
        norms_by_month: dict[int, np.ndarray],
        proc_by_date: dict[str, pd.DataFrame],
        station_id_to_idx: dict,
    ):
        self._norms_by_month = norms_by_month
        self._proc_by_date = proc_by_date
        self._sid_to_idx = station_id_to_idx

    def __call__(self, date: str) -> np.ndarray | None:
        proc = self._proc_by_date.get(date)
        if proc is None or proc.empty:
            return None
        month = int(date[5:7])
        norms_month = self._norms_by_month.get(month)
        if norms_month is None:
            return None
        wet_mask = (proc["rain_indicator"] == 1).values
        station_ids = proc["station_id"].values[wet_mask]
        indices = np.array([self._sid_to_idx[sid] for sid in station_ids])
        return norms_month[indices]


def main() -> None:
    args = parse_args()
    t0 = time.time()

    log("DEM Experiment")
    log(f"  n_test_days={args.n_test_days or 'ALL'}")

    # ── Check prerequisites ──────────────────────────────────────────────
    grids_path = DIR_OK / "monthly_norm_grids.pkl"
    if not grids_path.exists():
        log("  Downloading monthly norm grids from S3…")
        if not download_from_s3("results/ordinary_kriging/monthly_norm_grids.pkl", grids_path):
            log("ERROR: Monthly norm grids not found.")
            log("  Run `python -m thesis.scripts.build_monthly_grids` first.")
            sys.exit(1)

    from thesis.config import Config
    from thesis.data.registry import DataRegistry

    cfg = Config()
    registry = DataRegistry.from_config(cfg)

    # ── Load data + fit transforms ───────────────────────────────────────
    (all_raw, all_proc,
     fwd, inv, proc_by_date, get_mean_monthly_total) = load_and_fit_pipeline(
        cfg, registry, cfg.date_start, cfg.date_end,
    )

    # ── Build station lookup table ───────────────────────────────────────
    stations_df = (
        all_proc[["station_id", "x_proj", "y_proj"]]
        .drop_duplicates("station_id")
        .sort_values("station_id")
        .reset_index(drop=True)
    )
    station_coords = stations_df[["x_proj", "y_proj"]].values
    station_id_to_idx = {sid: i for i, sid in enumerate(stations_df["station_id"].values)}
    log(f"  {len(stations_df)} unique stations")

    del all_raw, all_proc
    gc.collect()

    # ── Load variograms ─────────────────────────────────────────────────
    from thesis.models.kriging.variogram_fitter import GlobalVariogramFitter

    vgm_path = DIR_OK / "global_variograms.pkl"
    if not vgm_path.exists():
        log("  Downloading variograms from S3…")
        if not download_from_s3("results/ordinary_kriging/global_variograms.pkl", vgm_path):
            log("ERROR: Global variograms not found.")
            log("  Run `python -m thesis.scripts.run_variogram` first.")
            sys.exit(1)
    global_vgm = GlobalVariogramFitter.load(str(vgm_path))
    log(f"Loaded variograms: {vgm_path}")

    # ── Determine best combo from CV results ─────────────────────────────
    cv_path = DIR_CV / "cv_results.pkl"
    if not cv_path.exists():
        log("ERROR: cv_results.pkl not found.")
        log("  Run `python -m thesis.scripts.run_cv` first.")
        sys.exit(1)
    with open(cv_path, "rb") as f:
        cv_results = pickle.load(f)
    valid = {k: r for k, r in cv_results.items()
             if r and np.isfinite(r.get("crps_mm", np.nan))}
    best_key = min(valid, key=lambda k: valid[k]["crps_mm"])
    log(f"Best combo from cv_results.pkl: {best_key[0]} x {best_key[1]}")

    if best_key not in global_vgm or global_vgm[best_key] is None:
        log(f"ERROR: {best_key} not in global_vgm.")
        sys.exit(1)

    vgm_best = {best_key: global_vgm[best_key]}

    # ── Test dates ───────────────────────────────────────────────────────
    rng = np.random.default_rng(cfg.random_seed)
    all_dates = pd.date_range(cfg.date_start, cfg.date_end, freq="1D").strftime("%Y-%m-%d").tolist()
    if args.n_test_days is not None:
        test_dates = sorted(rng.choice(all_dates, size=min(args.n_test_days, len(all_dates)), replace=False).tolist())
    else:
        test_dates = sorted(all_dates)
    log(f"  n_test_days={len(test_dates)}")

    # ── Load pre-computed TPS grids + map to station locations ────────────
    log(f"Loading pre-computed monthly norm grids: {grids_path}")
    with open(grids_path, "rb") as f:
        grid_data = pickle.load(f)
    grids_2d = grid_data["grids_2d"]
    grids_3d = grid_data["grids_3d"]

    from thesis.datasets.protocols import PredictionGrid
    grid = PredictionGrid.from_config(cfg, dem=registry.dem)
    grid_coords = grid.coords_proj

    log("  Mapping grid norms to station locations (nearest neighbour)...")
    norms_2d = _build_station_norms_from_grid(grids_2d, grid_coords, station_coords)
    norms_3d = _build_station_norms_from_grid(grids_3d, grid_coords, station_coords) if grids_3d is not None else norms_2d
    log(f"  Done: {len(stations_df)} stations x 12 months")

    del grid_data, grids_2d, grids_3d, grid
    gc.collect()

    # ── Run LOO-CV for each norm_mode ────────────────────────────────────
    from thesis.models.kriging.loo_cv import SpatialLooCV

    dem_results = {}
    norm_configs = {
        "station": None,
        "tps_2d":  _PrecomputedNormsLookup(norms_2d, proc_by_date, station_id_to_idx),
        "tps_3d":  _PrecomputedNormsLookup(norms_3d, proc_by_date, station_id_to_idx),
    }

    for nm, precomputed_fn in norm_configs.items():
        log(f"  Running LOO-CV: norm_mode={nm}")
        ckpt = str(DIR_CV / f"cv_dem_{nm}.pkl")

        cv_dem = SpatialLooCV(
            global_vgm=vgm_best,
            fwd_fn=fwd,
            inv_fn=inv,
            get_monthly_total_fn=get_mean_monthly_total,
            cfg=cfg,
            rng=np.random.default_rng(cfg.random_seed),
            n_test_days=len(test_dates),
            k_mc=K_MC,
            checkpoint_path=ckpt,
            n_jobs=N_JOBS,
            norm_mode=nm,
            precomputed_norms_fn=precomputed_fn,
        )
        result = cv_dem.run(test_dates, load_proc_fn=ProcByDateLoader(proc_by_date))
        cv_dem.save(ckpt)
        dem_results[nm] = result[best_key]
        r = dem_results[nm]
        log(f"  norm_mode={nm:<10}  CRPS_mm={r['crps_mm']:.4f}  MAE={r['mae_mm']:.3f} mm  n={r['n']}")

    # ── Summary ──────────────────────────────────────────────────────────
    key_str = f"{best_key[0]} x {best_key[1]}"
    lines = [
        "=" * 70,
        f"DEM Experiment ({key_str})",
        "=" * 70,
        f"n_test_days={len(test_dates)}",
        "",
    ]
    for nm in ("station", "tps_2d", "tps_3d"):
        if nm in dem_results:
            r = dem_results[nm]
            lines.append(f"  {nm:<10}  CRPS_mm={r['crps_mm']:.4f}  MAE={r['mae_mm']:.3f} mm  n={r['n']}")
    if "station" in dem_results and "tps_3d" in dem_results:
        base_crps = dem_results["station"]["crps_mm"]
        best_crps = dem_results["tps_3d"]["crps_mm"]
        if base_crps > 0:
            delta = (base_crps - best_crps) / base_crps * 100
            lines.append(f"  DEM improvement (tps_3d vs station): {delta:+.1f}% CRPS_mm")

    summary = "\n".join(lines)
    path = DIR_OUT / "dem_summary.txt"
    path.write_text(summary, encoding="utf-8")
    log(f"Summary saved: {path}")
    print("\n" + summary)

    # ── Upload to S3 ─────────────────────────────────────────────────────
    if not args.no_upload:
        from thesis.scripts.s3_upload import sync_to_s3
        log("Uploading DEM results to S3...")
        sync_to_s3(DIR_CV, "results/cross_validation")
        sync_to_s3(DIR_OUT, "results/outputs")
        log("S3 upload complete.")

    elapsed = time.time() - t0
    log(f"DEM experiment complete in {elapsed / 60:.1f} min ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
