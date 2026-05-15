"""Structural contracts for all data sources.

Using typing.Protocol (structural subtyping) so concrete implementations
don't need to inherit from these — any class with matching methods qualifies.
"""
from typing import Protocol, runtime_checkable
import numpy as np
import pandas as pd


@runtime_checkable
class StationSource(Protocol):
    """Loads daily precipitation measurements from weather stations.

    Returns a DataFrame with columns:
        station_id  (str)
        date        (str, ISO 8601)
        lon         (float, WGS84)
        lat         (float, WGS84)
        precip_mm   (float, millimetres; NaN for missing)
    """

    def load(self, date_start: str, date_end: str) -> pd.DataFrame: ...


@runtime_checkable
class GridSource(Protocol):
    """Loads a spatial raster (DEM, land use, …) for the study region.

    The raster is expected to be pre-aligned to a common CRS during loading;
    consumers should not manage reprojection themselves.
    """

    def load_raster(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
        resolution_m: int,
    ) -> tuple[np.ndarray, object]:
        """Return (array (H, W), affine_transform)."""
        ...

    def sample_at_points(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> np.ndarray:
        """Return 1-D array of raster values at the given WGS84 coordinates."""
        ...
