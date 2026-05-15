"""Core data structures: StationDataset, PredictionGrid, InterpolationResult."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class StationDataset:
    """Preprocessed station data for one day (projected coords, quota, indicator)."""

    coords_proj: np.ndarray
    precip_quota: np.ndarray
    rain_indicator: np.ndarray
    monthly_normals: np.ndarray | None
    elevation_m: np.ndarray | None
    date: str
    crs: str

    def n_stations(self) -> int:
        return len(self.coords_proj)

    def wet_mask(self) -> np.ndarray:
        return self.rain_indicator == 1


@dataclass(frozen=True)
class PredictionGrid:
    """Target interpolation grid with (M, 2) coords and optional elevation."""

    coords_proj: np.ndarray
    shape: tuple[int, int]
    crs: str
    elevation_m: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: object, dem: object = None) -> "PredictionGrid":
        """Build the study-area grid from a Config object."""
        from thesis.grid import build_prediction_grid
        return build_prediction_grid(cfg, dem=dem)  # type: ignore[arg-type]

    def n_cells(self) -> int:
        return self.coords_proj.shape[0]


@dataclass(frozen=True)
class InterpolationResult:
    """Output of model.predict(): mean (mm), variance (mm²), date, model name."""

    mean: xr.DataArray
    variance: xr.DataArray | None
    date: str
    model: str



