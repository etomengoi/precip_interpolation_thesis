"""DEPRECATED — use notebooks/03_kriging/{kriging_train,kriging_eval,kriging_test}.ipynb.

The notebook pipeline uses 5-fold cross-validation aligned with LGBM and
BayesNF (shared fold_assignment.parquet) instead of station-level LOO,
making model results directly comparable.

Original docstring follows.
--------------------------------------------------------------------
Main LOO-CV pipeline: cross-validation for all 9 (transform × model) combos.

Prerequisites: run_variogram and build_monthly_grids must complete first.

Usage:
    python -m thesis.scripts.run_cv                    # full run (all days)
    python -m thesis.scripts.run_cv --n-test-days 30   # quick test
"""
from __future__ import annotations

import argparse
import gc
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ordinary Kriging LOO-CV pipeline")
    p.add_argument("--n-test-days", type=int, default=None,
                   help="Number of random test days (default: all)")
    p.add_argument("--no-upload", action="store_true",
                   help="Skip S3 upload")
    return p.parse_args()


def _load_variograms() -> dict:
    """Load pre-fitted variograms from local cache or S3."""
    from thesis.models.kriging.variogram_fitter import GlobalVariogramFitter

    vgm_path = DIR_OK / "global_variograms.pkl"
    if not vgm_path.exists():
        log("Variograms not found locally. Trying S3…")
        if not download_from_s3("results/ordinary_kriging/global_variograms.pkl", vgm_path):
            log("ERROR: Global variograms not found.")
            log("  Run `python -m thesis.scripts.run_variogram` first.")
            sys.exit(1)
        log("  Downloaded from S3.")

    log(f"Loading variograms: {vgm_path}")
    return GlobalVariogramFitter.load(str(vgm_path))


def step_run_full_cv(
    cfg, global_vgm, fwd, inv, proc_by_date, get_mean_monthly_total,
    n_test_days: int | None,
):
    """Run LOO-CV for all 9 (transform × variogram_model) combinations."""
    from thesis.models.kriging.loo_cv import SpatialLooCV

    K_MC = 100
    N_JOBS = -1

    log("=" * 60)
    log("Running FULL LOO-CV: all transform × variogram_model combos")

    rng = np.random.default_rng(cfg.random_seed)
    all_dates = pd.date_range(
        cfg.date_start, cfg.date_end, freq="1D",
    ).strftime("%Y-%m-%d").tolist()

    if n_test_days is not None:
        n_days = min(n_test_days, len(all_dates))
        test_dates = sorted(rng.choice(all_dates, size=n_days, replace=False).tolist())
    else:
        test_dates = sorted(all_dates)

    log(f"  Test dates: {len(test_dates)} ({test_dates[0]} … {test_dates[-1]})")

    cv = SpatialLooCV(
        global_vgm=global_vgm,
        fwd_fn=fwd,
        inv_fn=inv,
        get_monthly_total_fn=get_mean_monthly_total,
        cfg=cfg,
        rng=rng,
        n_test_days=len(test_dates),
        k_mc=K_MC,
        checkpoint_path=str(DIR_CV / "cv_results_checkpoint.pkl"),
        n_jobs=N_JOBS,
    )
    cv_results = cv.run(test_dates, load_proc_fn=ProcByDateLoader(proc_by_date))
    cv.save(str(DIR_CV / "cv_results.pkl"))
    log("Full CV done.")
    return cv_results, test_dates


def step_save_summary(cv_results, cfg, n_test_days: int | None):
    """Write a human-readable summary to outputs/results/summary.txt."""
    n_days = n_test_days if n_test_days is not None else "ALL"

    valid = {k: r for k, r in cv_results.items()
             if r and np.isfinite(r.get("crps_mm", np.nan))}
    best_key = min(valid, key=lambda k: valid[k]["crps_mm"]) if valid else None

    lines = [
        "=" * 70,
        "Ordinary Kriging LOO-CV — Summary",
        "=" * 70,
        f"Date range: {cfg.date_start} … {cfg.date_end}",
        f"n_test_days={n_days}",
        "",
        "─" * 70,
        f"{'Transform':<18} {'Model':<12} {'CRPS_mm':>10} {'MAE_mm':>10} {'CRPS_z':>10} {'n':>8}",
        "─" * 70,
    ]
    for (t, vm), r in sorted(cv_results.items()):
        marker = " <-- BEST" if best_key and (t, vm) == best_key else ""
        lines.append(
            f"{t:<18} {vm:<12} {r['crps_mm']:>10.4f} {r['mae_mm']:>10.3f} "
            f"{r['crps_z']:>10.4f} {r['n']:>8,}{marker}"
        )

    summary = "\n".join(lines)
    path = DIR_OUT / "summary.txt"
    path.write_text(summary, encoding="utf-8")
    log(f"Summary saved: {path}")
    print("\n" + summary)


def step_upload_s3():
    """Upload all output folders to S3."""
    from thesis.scripts.s3_upload import sync_to_s3

    log("Uploading results to S3…")
    sync_to_s3(DIR_OK, "results/ordinary_kriging")
    sync_to_s3(DIR_CV, "results/cross_validation")
    sync_to_s3(DIR_OUT, "results/outputs")
    log("S3 upload complete.")


def main() -> None:
    args = parse_args()
    t0 = time.time()

    from thesis.config import Config
    from thesis.data.registry import DataRegistry

    cfg = Config()
    registry = DataRegistry.from_config(cfg)

    log("Ordinary Kriging LOO-CV Pipeline")
    n_days_str = str(args.n_test_days) if args.n_test_days is not None else "ALL"
    log(f"  n_test_days={n_days_str}")

    # ── Step 1: Load data + fit transforms ───────────────────────────────
    (all_raw, all_proc,
     fwd, inv, proc_by_date, get_mean_monthly_total) = load_and_fit_pipeline(
        cfg, registry, cfg.date_start, cfg.date_end,
    )

    # ── Step 2: Load pre-fitted variograms ───────────────────────────────
    global_vgm = _load_variograms()

    del all_raw, all_proc
    gc.collect()
    log("  Released all_raw/all_proc from memory")

    # ── Step 3: Full LOO-CV ──────────────────────────────────────────────
    cv_results, test_dates = step_run_full_cv(
        cfg, global_vgm, fwd, inv, proc_by_date, get_mean_monthly_total,
        args.n_test_days,
    )

    # ── Step 4: Summary ──────────────────────────────────────────────────
    step_save_summary(cv_results, cfg, args.n_test_days)

    # ── Step 5: Upload to S3 ─────────────────────────────────────────────
    if not args.no_upload:
        step_upload_s3()

    elapsed = time.time() - t0
    log(f"Pipeline complete in {elapsed / 3600:.1f} hours ({elapsed:.0f} seconds)")


if __name__ == "__main__":
    main()
