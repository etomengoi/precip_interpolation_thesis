"""Reprojects station coordinates from WGS84 to an equal-area CRS.

Kriging requires distances in metres (not degrees) so that the variogram
is isotropic. EPSG:3035 (ETRS89-LAEA) is the standard choice for Europe.
"""
from __future__ import annotations

import pandas as pd
from pyproj import Transformer


class ProjectionTransform:
    """Adds x_proj / y_proj columns (metres, equal-area CRS) to the DataFrame.

    The source CRS is always assumed to be WGS84 (EPSG:4326).
    """

    def __init__(self, target_crs: str = "EPSG:3035") -> None:
        self.target_crs = target_crs
        self._transformer: Transformer | None = None

    def fit(self, df: pd.DataFrame) -> "ProjectionTransform":
        self._transformer = Transformer.from_crs(
            "EPSG:4326", self.target_crs, always_xy=True
        )
        return self

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._transformer is None:
            raise RuntimeError("Call fit() before apply().")
        df = df.copy()
        x, y = self._transformer.transform(df["lon"].values, df["lat"].values)
        df["x_proj"] = x
        df["y_proj"] = y
        return df

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop projected columns — they carry no information in output DataFrames."""
        return df.drop(columns=["x_proj", "y_proj"], errors="ignore")
