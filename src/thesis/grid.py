"""PredictionGrid builder — creates the target ≤1 km output grid."""
from __future__ import annotations

import numpy as np
from pyproj import Transformer

from thesis.config import Config
from thesis.datasets.protocols import PredictionGrid


def build_prediction_grid(
    cfg: Config,
    dem: "DEMSource | None" = None,
) -> PredictionGrid:
    """Build a regular grid over the study area in the projected CRS.

    Grid cell centres are spaced cfg.study_area.grid_resolution_m metres apart.
    The bounding box is defined by cfg.study_area lon/lat limits, projected
    into cfg.study_area.target_crs.

    If a DEMSource is provided, elevation is sampled at every grid cell and
    stored in PredictionGrid.elevation_m.

    Returns a PredictionGrid with coords_proj of shape (H*W, 2).
    """
    sa = cfg.study_area
    t = Transformer.from_crs("EPSG:4326", sa.target_crs, always_xy=True)

    # Project bounding box corners.
    x_min, y_min = t.transform(sa.lon_min, sa.lat_min)
    x_max, y_max = t.transform(sa.lon_max, sa.lat_max)

    res = sa.grid_resolution_m
    xs = np.arange(x_min, x_max, res)
    ys = np.arange(y_min, y_max, res)
    xx, yy = np.meshgrid(xs, ys)

    coords = np.column_stack([xx.ravel(), yy.ravel()])

    elevation_m = None
    if dem is not None:
        elevation_m = dem.sample_at_projected(coords[:, 0], coords[:, 1])

    return PredictionGrid(
        coords_proj=coords,
        shape=(len(ys), len(xs)),
        crs=sa.target_crs,
        elevation_m=elevation_m,
    )
