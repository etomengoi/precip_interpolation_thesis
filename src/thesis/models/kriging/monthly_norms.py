"""Pre-compute monthly normal grids via thin-plate spline interpolation.

Extracted from ordinary.py — used by build_monthly_grids.py and run_dem.py.
Follows Haylock et al. (2008) methodology: 2-D TPS in (X, Y) or 3-D TPS
in (X, Y, Elevation) to capture orographic effects on monthly totals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from thesis.datasets.protocols import PredictionGrid
from thesis.transforms.detrend import DetrendTransform


def build_monthly_norm_grids(
    detrend: DetrendTransform,
    all_proc: pd.DataFrame,
    grid: PredictionGrid,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Pre-compute 12 monthly normal maps via TPS interpolation.

    Returns (grids_2d, grids_3d) where each is (12, M) or None.
    """
    from scipy.interpolate import RBFInterpolator

    if detrend._monthly_totals is None:
        raise RuntimeError("DetrendTransform must be fitted before building grids.")

    mt         = detrend._monthly_totals          # Series index (station_id, _month)
    grand_mean = float(mt.mean())

    # One representative row per station (coordinates don't vary by day)
    elev_col = "elevation_m" if "elevation_m" in all_proc.columns else None
    keep_cols = ["station_id", "x_proj", "y_proj"] + ([elev_col] if elev_col else [])
    stations = (
        all_proc[keep_cols]
        .drop_duplicates("station_id")
        .reset_index(drop=True)
    )

    gx = grid.coords_proj[:, 0]
    gy = grid.coords_proj[:, 1]
    M  = len(gx)

    has_station_elev = (
        elev_col is not None
        and stations[elev_col].notna().any()
    )
    has_grid_elev = grid.elevation_m is not None and np.any(np.isfinite(grid.elevation_m))
    do_3d = has_station_elev and has_grid_elev

    grids_2d: np.ndarray       = np.full((12, M), grand_mean)
    grids_3d: np.ndarray | None = np.full((12, M), grand_mean) if do_3d else None

    raw_grid_2d = np.column_stack([gx, gy])
    raw_grid_3d = np.column_stack([gx, gy, grid.elevation_m]) if do_3d else None

    for m in range(1, 13):
        try:
            lookup = mt.xs(m, level="_month")
        except KeyError:
            continue

        norms = stations["station_id"].map(lookup).fillna(grand_mean).values.astype(float)
        valid = np.isfinite(norms) & (norms > 0)
        if valid.sum() < 3:
            continue

        xs = stations["x_proj"].values[valid]
        ys = stations["y_proj"].values[valid]
        nv = norms[valid]

        # 2-D TPS
        raw_st = np.column_stack([xs, ys])
        fm2    = raw_st.mean(axis=0)
        fs2    = np.maximum(raw_st.std(axis=0), 1.0)
        rbf2 = RBFInterpolator(
            (raw_st - fm2) / fs2, nv,
            kernel="thin_plate_spline", degree=1, smoothing=1.0,
        )
        grids_2d[m - 1] = np.clip(rbf2((raw_grid_2d - fm2) / fs2), 1.0, None)

        # 3-D TPS
        if grids_3d is not None:
            elev_st = stations[elev_col].values[valid]
            valid3  = np.isfinite(elev_st)
            if valid3.sum() < 3:
                raise ValueError(
                    f"Month {m}: too few stations with valid elevation for 3D TPS: {valid3.sum()} < 3"
                )
            raw_st3 = np.column_stack([xs[valid3], ys[valid3], elev_st[valid3]])
            fm3     = raw_st3.mean(axis=0)
            fs3     = np.maximum(raw_st3.std(axis=0), 1.0)
            rbf3 = RBFInterpolator(
                (raw_st3 - fm3) / fs3, nv[valid3],
                kernel="thin_plate_spline", degree=1, smoothing=1.0,
            )
            grids_3d[m - 1] = np.clip(rbf3((raw_grid_3d - fm3) / fs3), 1.0, None)

    return grids_2d, grids_3d


def build_monthly_norms_at_points(
    detrend: DetrendTransform,
    all_proc_train: pd.DataFrame,
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    elev_pts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate the same TPS monthly normals as build_monthly_norm_grids,
    but at arbitrary points (test-fold station coordinates) instead of the
    1 km prediction grid. Used by k-fold CV to avoid grid-to-point resampling.

    Returns (norms_2d, norms_3d) — each (12, n_pts) or norms_3d=None.
    """
    from scipy.interpolate import RBFInterpolator

    if detrend._monthly_totals is None:
        raise RuntimeError("DetrendTransform must be fitted before building norms.")

    mt         = detrend._monthly_totals
    grand_mean = float(mt.mean())

    elev_col = "elevation_m" if "elevation_m" in all_proc_train.columns else None
    keep_cols = ["station_id", "x_proj", "y_proj"] + ([elev_col] if elev_col else [])
    stations = (
        all_proc_train[keep_cols]
        .drop_duplicates("station_id")
        .reset_index(drop=True)
    )

    n_pts = len(x_pts)
    has_station_elev = elev_col is not None and stations[elev_col].notna().any()
    has_pt_elev      = elev_pts is not None and np.any(np.isfinite(elev_pts))
    do_3d = has_station_elev and has_pt_elev

    norms_2d: np.ndarray = np.full((12, n_pts), grand_mean)
    norms_3d: np.ndarray | None = np.full((12, n_pts), grand_mean) if do_3d else None

    raw_pts_2d = np.column_stack([x_pts, y_pts])
    raw_pts_3d = np.column_stack([x_pts, y_pts, elev_pts]) if do_3d else None

    for m in range(1, 13):
        try:
            lookup = mt.xs(m, level="_month")
        except KeyError:
            continue

        nv_all = stations["station_id"].map(lookup).fillna(grand_mean).values.astype(float)
        valid  = np.isfinite(nv_all) & (nv_all > 0)
        if valid.sum() < 3:
            continue

        xs = stations["x_proj"].values[valid]
        ys = stations["y_proj"].values[valid]
        nv = nv_all[valid]

        raw_st = np.column_stack([xs, ys])
        fm2    = raw_st.mean(axis=0)
        fs2    = np.maximum(raw_st.std(axis=0), 1.0)
        rbf2 = RBFInterpolator(
            (raw_st - fm2) / fs2, nv,
            kernel="thin_plate_spline", degree=1, smoothing=1.0,
        )
        norms_2d[m - 1] = np.clip(rbf2((raw_pts_2d - fm2) / fs2), 1.0, None)

        if norms_3d is not None:
            elev_st = stations[elev_col].values[valid]
            valid3  = np.isfinite(elev_st)
            if valid3.sum() < 3:
                continue
            raw_st3 = np.column_stack([xs[valid3], ys[valid3], elev_st[valid3]])
            fm3     = raw_st3.mean(axis=0)
            fs3     = np.maximum(raw_st3.std(axis=0), 1.0)
            rbf3 = RBFInterpolator(
                (raw_st3 - fm3) / fs3, nv[valid3],
                kernel="thin_plate_spline", degree=1, smoothing=1.0,
            )
            norms_3d[m - 1] = np.clip(rbf3((raw_pts_3d - fm3) / fs3), 1.0, None)

    return norms_2d, norms_3d
