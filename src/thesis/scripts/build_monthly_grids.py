"""DEPRECATED — use notebooks/03_kriging/kriging_train.ipynb instead.

The notebook computes leakage-free monthly TPS normals per fold (one set
per fold, fit on train-fold stations only). This script fitted a single
set on all training stations, which is acceptable for whole-grid
prediction but not for k-fold CV evaluation.

Original docstring follows.
--------------------------------------------------------------------
Pre-compute monthly normal grids (2-D and 3-D TPS) and save to disk / S3.

Output: outputs/ordinary_kriging/monthly_norm_grids.pkl

Usage:
    python -m thesis.scripts.build_monthly_grids
    python -m thesis.scripts.build_monthly_grids --no-upload
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

from thesis.scripts._common import APP_ROOT, ensure_app_root, log

warnings.filterwarnings("ignore")

ensure_app_root()

DIR_OK = APP_ROOT / "outputs" / "ordinary_kriging"
DIR_OK.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build monthly normal grids (TPS)")
    p.add_argument("--no-upload", action="store_true", help="Skip S3 upload")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_path = DIR_OK / "monthly_norm_grids.pkl"
    if out_path.exists():
        log(f"Already exists: {out_path} — nothing to do.")
        return

    from thesis.config import Config
    from thesis.data.registry import DataRegistry
    from thesis.transforms import (
        ProjectionTransform, IndicatorTransform, DetrendTransform,
    )
    from thesis.models.kriging.monthly_norms import build_monthly_norm_grids
    from thesis.datasets.protocols import PredictionGrid

    cfg      = Config()
    registry = DataRegistry.from_config(cfg)

    log(f"Loading raw data: {cfg.date_start} … {cfg.date_end}")
    all_raw = registry.stations.load(cfg.date_start, cfg.date_end)
    log(f"  {len(all_raw):,} rows, {all_raw['station_id'].nunique()} stations")

    log("Fitting base pipeline: Projection → Indicator → Detrend…")
    proj = ProjectionTransform(target_crs=cfg.study_area.target_crs)
    ind  = IndicatorTransform(threshold_mm=cfg.wet_day_threshold_mm)
    det  = DetrendTransform()

    proj.fit(all_raw);  current = proj.apply(all_raw)
    ind.fit(current);   current = ind.apply(current)
    det.fit(current);   all_proc = det.apply(current)

    log("Building monthly normal grids (12 months × 2D/3D TPS)…")
    grid = PredictionGrid.from_config(cfg, dem=registry.dem)
    log(f"  Grid: {grid.shape[0]}×{grid.shape[1]} = {grid.n_cells():,} cells")

    grids_2d, grids_3d = build_monthly_norm_grids(det, all_proc, grid)

    with open(out_path, "wb") as f:
        pickle.dump({"grids_2d": grids_2d, "grids_3d": grids_3d}, f)
    log(f"  Saved: {out_path}  (2D: {grids_2d.shape}, 3D: {grids_3d.shape if grids_3d is not None else None})")

    if not args.no_upload:
        from thesis.scripts.s3_upload import sync_to_s3
        log("Uploading to S3…")
        sync_to_s3(DIR_OK, "results/ordinary_kriging")
        log("Upload complete.")


if __name__ == "__main__":
    main()
