"""DEPRECATED — use notebooks/03_kriging/kriging_train.ipynb instead.

Kept as reference for the script-based pipeline. The k-fold notebook
fits the variogram on the same 5-fold split shared with LGBM and BayesNF,
which this script does not (it ran a station-level LOO that was not
aligned with the other models' fold definitions).

Original docstring follows.
--------------------------------------------------------------------
Fit global climatological variograms with parallel CPU workers.

Run this BEFORE run_cv to pre-compute variograms. The result is cached
to disk and optionally uploaded to S3.

Usage:
    python -m thesis.scripts.run_variogram
    python -m thesis.scripts.run_variogram --force       # re-fit even if cached
    python -m thesis.scripts.run_variogram --no-upload
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np

from thesis.scripts._common import APP_ROOT, ensure_app_root, log

warnings.filterwarnings("ignore")

ensure_app_root()

DIR_OK = APP_ROOT / "outputs" / "ordinary_kriging"
DIR_OK.mkdir(parents=True, exist_ok=True)

N_JOBS = -1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit global variograms for Ordinary Kriging")
    p.add_argument("--force", action="store_true",
                   help="Re-fit even if cached variograms already exist")
    p.add_argument("--no-upload", action="store_true",
                   help="Skip S3 upload")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    log("Global Variogram Fitting")

    from thesis.config import Config
    from thesis.data.registry import DataRegistry

    cfg      = Config()
    registry = DataRegistry.from_config(cfg)

    vgm_path = str(DIR_OK / "global_variograms.pkl")
    if Path(vgm_path).exists() and not args.force:
        log(f"Cached variograms already exist: {vgm_path}")
        log("  Use --force to re-fit. Exiting.")
        return

    # ── Step 1: Load data + fit transforms ──────────────────────────────
    from thesis.scripts._common import load_and_fit_pipeline

    (all_raw, all_proc,
     fwd, inv, _, _) = load_and_fit_pipeline(
        cfg, registry, cfg.date_start, cfg.date_end,
    )

    # ── Step 2: Collect pool days ────────────────────────────────────────
    log("Grouping all_proc by date…")
    grouped = {date: grp for date, grp in all_proc.groupby("date")}
    pool_dates = sorted(grouped.keys())
    log(f"  {len(pool_dates)} unique dates")

    log(f"  Filtering pool days (>=5 wet stations)…")
    pool_procs = []
    for d in pool_dates:
        proc_day = grouped[d]
        if (proc_day["rain_indicator"] == 1).sum() >= 5:
            pool_procs.append(proc_day)
    log(f"  Pool ready: {len(pool_procs)} days")

    # ── Step 3: Fit variograms ───────────────────────────────────────────
    from thesis.models.kriging.variogram_fitter import GlobalVariogramFitter

    fitter = GlobalVariogramFitter(
        transforms=["none", "log", "normal_score"],
        variogram_models=["spherical", "exponential", "gaussian"],
        n_lags=cfg.kriging.variogram_nlags,
        max_lag_km=cfg.kriging.search_radius_km,
        min_pairs=30,
        checkpoint_path=str(DIR_OK / "global_variograms_checkpoint.pkl"),
        n_jobs=N_JOBS,
    )
    global_vgm = fitter.fit(pool_procs, fwd_fn=fwd)

    # ── Step 3b: Fit global indicator variogram (Haylock 2008, §31) ──────
    log("Fitting global indicator variogram (spherical)…")
    # Use ALL pool days (including days with <5 wet stations, as long as
    # they are not constant). fit_indicator() skips constant days internally.
    all_pool_procs = [grouped[d] for d in pool_dates]
    fitter.fit_indicator(all_pool_procs)

    fitter.save(vgm_path)
    log(f"Variograms saved: {vgm_path}")

    # ── Step 4: Upload to S3 ─────────────────────────────────────────────
    if not args.no_upload:
        from thesis.scripts.s3_upload import sync_to_s3
        log("Uploading results to S3…")
        sync_to_s3(DIR_OK, "results/ordinary_kriging")
        log("S3 upload complete.")

    elapsed = time.time() - t0
    log(f"Variogram fitting complete in {elapsed / 60:.1f} min ({elapsed:.0f} s)")


if __name__ == "__main__":
    main()
