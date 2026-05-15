"""Shared utilities for pipeline scripts.

Centralises picklable callables, logging, S3 helpers, and the
transform-pipeline setup that every script repeats.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ── Paths (Docker: /app/, local: project root) ────────────────────────────

APP_ROOT = Path("/app") if Path("/app/src").exists() else Path(".")


def ensure_app_root() -> None:
    """chdir to APP_ROOT so relative paths resolve correctly."""
    os.chdir(APP_ROOT)


# ── Logging ───────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    """Print with timestamp."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── S3 helpers ────────────────────────────────────────────────────────────

S3_BUCKET = "thesis-data-ismaktam"


def download_from_s3(s3_key: str, local_path: str | Path) -> bool:
    """Download a single file from S3.  Returns True on success."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = f'aws s3 cp "s3://{S3_BUCKET}/{s3_key}" "{local_path}" --quiet'
    rc = os.system(cmd)
    return rc == 0 and local_path.exists()


# ── Picklable callables for multiprocessing workers ───────────────────────
# Local closures cannot be pickled by the standard multiprocessing backend.
# Module-level classes are picklable.

class FwdFn:
    """Forward transform: quota → z-space."""

    def __init__(self, kts: dict) -> None:
        self.kts = kts

    def __call__(self, values: np.ndarray, transform: str) -> np.ndarray:
        return self.kts[transform].fwd(values)


class InvFn:
    """Inverse transform: z-space → quota."""

    def __init__(self, kts: dict) -> None:
        self.kts = kts

    def __call__(self, z: np.ndarray, transform: str) -> np.ndarray:
        return self.kts[transform].inv(z)


class GetMeanMonthlyTotal:
    """Mean monthly precipitation total across all stations."""

    def __init__(self, detrend) -> None:
        self.detrend = detrend

    def __call__(self, date: str) -> float:
        month = int(date[5:7])
        try:
            vals = self.detrend._monthly_totals.xs(month, level="_month")
        except KeyError:
            vals = self.detrend._monthly_totals
        return float(vals.mean())


class ProcByDateLoader:
    """Picklable callable: date → preprocessed DataFrame for that day."""

    def __init__(self, proc_by_date: dict):
        self._d = proc_by_date

    def __call__(self, date: str):
        return self._d[date]


# ── Transform pipeline setup ─────────────────────────────────────────────

def load_and_fit_pipeline(cfg, registry, date_start: str, date_end: str):
    """Load raw data, fit transforms, return (all_raw, all_proc, fwd, inv, proc_by_date, get_monthly_total)."""
    from thesis.transforms import (
        ProjectionTransform, IndicatorTransform,
        DetrendTransform, NormalScoreTransform, LogTransform, KrigingTransform,
    )
    from thesis.transforms.kriging_transform import TRANSFORMS
    from thesis.transforms.pipeline import TransformPipeline

    log(f"Loading raw data: {date_start} … {date_end}")
    all_raw = registry.stations.load(date_start, date_end)
    log(f"  {len(all_raw):,} rows, {all_raw['station_id'].nunique()} stations")

    log("Fitting base pipeline: Projection → Indicator → Detrend")
    proj = ProjectionTransform(target_crs=cfg.study_area.target_crs)
    ind = IndicatorTransform(threshold_mm=cfg.wet_day_threshold_mm)
    det = DetrendTransform()
    all_proc = TransformPipeline([proj, ind, det]).fit_transform(all_raw)

    log("  Fitting NormalScoreTransform…")
    ns = NormalScoreTransform()
    ns.fit(all_proc)
    log(f"  NormalScoreTransform CDF: {len(ns._sorted_vals):,} wet quotas")

    log_t = LogTransform(offset=cfg.log_transform_offset)
    kts = {
        kind: KrigingTransform(kind=kind, ns=ns, log=log_t)
        for kind in TRANSFORMS
    }

    # Write sorted_vals to a memmap file so workers share it via
    # OS page cache instead of each receiving a 174 MB pickle copy.
    memmap_dir = APP_ROOT / "outputs" / "_memmap"
    memmap_dir.mkdir(parents=True, exist_ok=True)
    ns.enable_memmap(str(memmap_dir / "sorted_vals.npy"))

    fwd = FwdFn(kts)
    inv = InvFn(kts)

    proc_by_date: dict[str, pd.DataFrame] = {
        str(d): grp for d, grp in all_proc.groupby("date")
    }

    get_mean_monthly_total = GetMeanMonthlyTotal(det)

    return all_raw, all_proc, fwd, inv, proc_by_date, get_mean_monthly_total
