"""Sensitivity analysis for max_wet (local kriging neighborhood size).

Prerequisites: run_variogram must complete first.

Usage:
    python -m thesis.scripts.run_sensitivity_max_wet --n-test-days 30
    python -m thesis.scripts.run_sensitivity_max_wet --k-values 20,50,100 --include-global
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.scripts._common import (
    APP_ROOT, ensure_app_root, log, download_from_s3,
    load_and_fit_pipeline, ProcByDateLoader,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ensure_app_root()

DIR_OUT = APP_ROOT / "outputs" / "sensitivity_max_wet"
DIR_OUT.mkdir(parents=True, exist_ok=True)

K_MC = 100
N_JOBS = 1
TRANSFORM = "normal_score"
MODEL = "exponential"
MIN_WET_STATIONS = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensitivity analysis for max_wet")
    p.add_argument("--k-values", type=str, default="10,20,30,50,75,100,150,200,300,500",
                   help="Comma-separated K values to test")
    p.add_argument("--include-global", action="store_true",
                   help="Also test K=None (global kriging) as baseline")
    p.add_argument("--n-test-days", type=int, default=30,
                   help="Number of random test days (default: 30)")
    return p.parse_args()


def _run_loo_for_k(
    k_val: int | None,
    test_dates: list[str],
    load_proc_fn,
    vgm_info: dict,
    fwd, inv, get_monthly_total,
    n_stations_min: int,
) -> dict:
    """Run LOO-CV for a single K value and return aggregated metrics + timing."""
    from thesis.models.kriging.loo_cv import _process_one_day_multi_vgm

    ss = np.random.SeedSequence(42)
    child_seeds = ss.spawn(len(test_dates))
    seed_ints = [int(s.generate_state(1)[0]) for s in child_seeds]

    t_start = time.time()

    day_results = []
    for date, seed in zip(test_dates, seed_ints):
        proc = load_proc_fn(date)
        day_results.append(
            _process_one_day_multi_vgm(
                date, proc,
                [vgm_info], [MODEL],
                TRANSFORM,
                fwd, inv, get_monthly_total,
                n_stations_min, K_MC, seed,
                "station", None,
                max_wet=k_val,
            )
        )

    elapsed = time.time() - t_start

    # Aggregate
    day_counts = [len(r[MODEL]["mae"]) for r in day_results]
    chunks_mae = [r[MODEL]["mae"] for r in day_results if r[MODEL]["mae"]]
    chunks_crps_z = [r[MODEL]["crps_z"] for r in day_results if r[MODEL]["crps_z"]]
    chunks_crps_mm = [r[MODEL]["crps_mm"] for r in day_results if r[MODEL]["crps_mm"]]
    all_mae = np.concatenate(chunks_mae) if chunks_mae else np.array([])
    all_crps_z = np.concatenate(chunks_crps_z) if chunks_crps_z else np.array([])
    all_crps_mm = np.concatenate(chunks_crps_mm) if chunks_crps_mm else np.array([])

    n_days_valid = sum(1 for c in day_counts if c > 0)

    return {
        "max_wet": k_val if k_val is not None else "global",
        "crps_mm": float(all_crps_mm.mean()) if len(all_crps_mm) else np.nan,
        "mae_mm": float(all_mae.mean()) if len(all_mae) else np.nan,
        "crps_z": float(all_crps_z.mean()) if len(all_crps_z) else np.nan,
        "n_predictions": len(all_mae),
        "n_days_valid": n_days_valid,
        "avg_stations_per_day": round(len(all_mae) / n_days_valid, 0) if n_days_valid else 0,
        "time_sec": round(elapsed, 1),
    }


def _make_plot(df: pd.DataFrame, out_path: Path) -> None:
    """Generate sensitivity elbow plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    is_global = df["max_wet"] == "global"
    df_k = df[~is_global].copy()
    df_k["max_wet"] = df_k["max_wet"].astype(int)
    df_k = df_k.sort_values("max_wet")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df_k["max_wet"], df_k["crps_mm"], "o-", color="tab:blue", label="CRPS_mm")
    ax1.set_xlabel("max_wet (K)")
    ax1.set_ylabel("CRPS_mm", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax1b = ax1.twinx()
    ax1b.plot(df_k["max_wet"], df_k["mae_mm"], "s--", color="tab:red", label="MAE_mm")
    ax1b.set_ylabel("MAE_mm", color="tab:red")
    ax1b.tick_params(axis="y", labelcolor="tab:red")

    if is_global.any():
        g = df[is_global].iloc[0]
        ax1.axhline(g["crps_mm"], color="tab:blue", linestyle=":", alpha=0.5, label="global CRPS")
        ax1b.axhline(g["mae_mm"], color="tab:red", linestyle=":", alpha=0.5, label="global MAE")

    ax1.set_title("Prediction Quality vs Neighborhood Size")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax2.plot(df_k["max_wet"], df_k["time_sec"], "o-", color="tab:green")
    if is_global.any():
        ax2.axhline(df[is_global].iloc[0]["time_sec"], color="tab:green",
                     linestyle=":", alpha=0.5, label="global time")
        ax2.legend()
    ax2.set_xlabel("max_wet (K)")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Computation Time vs Neighborhood Size")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Plot saved: {out_path}")


def main() -> None:
    args = parse_args()
    t0 = time.time()

    k_values: list[int | None] = [int(k) for k in args.k_values.split(",")]
    if args.include_global:
        k_values.append(None)

    log("=" * 60)
    log("Sensitivity Analysis: max_wet (local kriging neighborhood)")
    log(f"  K values: {[k if k is not None else 'global' for k in k_values]}")
    log(f"  n_test_days={args.n_test_days}")

    from thesis.config import Config
    from thesis.data.registry import DataRegistry
    from thesis.models.kriging.variogram_fitter import GlobalVariogramFitter

    cfg = Config()
    registry = DataRegistry.from_config(cfg)

    (all_raw, all_proc, fwd, inv,
     proc_by_date, get_monthly_total) = load_and_fit_pipeline(
        cfg, registry, cfg.date_start, cfg.date_end,
    )

    # Load variograms
    vgm_path = APP_ROOT / "outputs" / "ordinary_kriging" / "global_variograms.pkl"
    if not vgm_path.exists():
        log("Variograms not found locally. Downloading from S3...")
        if not download_from_s3("results/ordinary_kriging/global_variograms.pkl", vgm_path):
            log("ERROR: Global variograms not found.")
            log("  Run `python -m thesis.scripts.run_variogram` first.")
            sys.exit(1)
    global_vgm = GlobalVariogramFitter.load(str(vgm_path))

    key = (TRANSFORM, MODEL)
    if key not in global_vgm or global_vgm[key] is None:
        log(f"ERROR: variogram not found for {key}")
        sys.exit(1)
    vgm_info = global_vgm[key]
    log(f"  Using variogram: {key}")

    # Sample test dates — only days with many wet stations
    wet_counts = all_proc.groupby("date")["rain_indicator"].sum()
    eligible = wet_counts[wet_counts >= MIN_WET_STATIONS].index.tolist()
    if not eligible:
        log(f"ERROR: no days with >= {MIN_WET_STATIONS} wet stations. "
            f"Max wet count: {int(wet_counts.max())}.")
        sys.exit(1)

    rng = np.random.default_rng(cfg.random_seed)
    n_days = min(args.n_test_days, len(eligible))
    test_dates = sorted(rng.choice(eligible, size=n_days, replace=False).tolist())
    log(f"  Selected {n_days} test days from {len(eligible)} eligible "
        f"(>= {MIN_WET_STATIONS} wet stations)")
    selected_wet = wet_counts[test_dates]
    log(f"  Wet stations per day: min={int(selected_wet.min())}, "
        f"median={int(selected_wet.median())}, max={int(selected_wet.max())}")

    del all_raw, all_proc
    gc.collect()

    load_proc_fn = ProcByDateLoader(proc_by_date)

    # Run sensitivity analysis
    log("=" * 60)
    rows = []
    for k_val in k_values:
        k_label = str(k_val) if k_val is not None else "global"
        log(f"Running LOO-CV with max_wet={k_label}...")

        result = _run_loo_for_k(
            k_val, test_dates, load_proc_fn,
            vgm_info, fwd, inv, get_monthly_total,
            cfg.kriging.n_stations_min,
        )
        rows.append(result)
        log(f"  max_wet={k_label:>6}  CRPS_mm={result['crps_mm']:.4f}  "
            f"MAE_mm={result['mae_mm']:.3f}  time={result['time_sec']:.1f}s  "
            f"n={result['n_predictions']} ({result['n_days_valid']} days, "
            f"~{result['avg_stations_per_day']:.0f} stations/day)")

    # Save results
    df = pd.DataFrame(rows)
    csv_path = DIR_OUT / "sensitivity_results.csv"
    df.to_csv(csv_path, index=False)
    log(f"\nResults saved: {csv_path}")
    log(f"\n{df.to_string(index=False)}")

    _make_plot(df, DIR_OUT / "sensitivity_max_wet.png")

    elapsed = time.time() - t0
    log(f"\nDone in {elapsed / 60:.1f} min.")


if __name__ == "__main__":
    main()
