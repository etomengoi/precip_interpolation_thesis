"""SoilGrids data loader.

Source: SoilGrids 250m global soil property maps.
Local rasters live in data/soilgrids/ as GeoTIFF files.

Available variables (6) × depth layers (3):
    bulk_density    — bulk density (cg/cm³ × 10)
    clay            — clay content (g/kg × 10)
    sand            — sand content (g/kg × 10)
    silt            — silt content (g/kg × 10)
    soc             — soil organic carbon (dg/kg × 10)
    water_10kpa     — volumetric water content at -10 kPa (cm³/100 cm³ × 10)

Depth layers: "0-5m", "5-15m", "15-30m"
(The "m" in filenames is a mislabel; actual depths are centimetres.)

Rasters are reprojected to the study CRS on first use; subsequent calls
are served from the joblib disk cache.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal

import numpy as np
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

from thesis.cache import make_memory
from thesis.config import Config

# ---------------------------------------------------------------------------
# File-name → variable mapping
# ---------------------------------------------------------------------------

#: Maps canonical Python names to the substring that appears in the file names.
VARIABLE_MAP: dict[str, str] = {
    "bulk_density": "bulk density",
    "clay": "clay content",
    "sand": "sand",
    "silt": "silt",
    "soc": "soilorganiccarbon",
    "water_10kpa": "water -10kPa",
}

DEPTH_LAYERS: list[str] = ["0-5m", "5-15m", "15-30m"]

# Thickness of each depth layer in cm (used for depth-weighted averaging).
_DEPTH_WEIGHTS: list[float] = [5.0, 10.0, 15.0]  # 0-5, 5-15, 15-30 cm


def _file_for(data_root: Path, variable: str, depth: str) -> Path:
    """Return the path for a single (variable, depth) GeoTIFF."""
    label = VARIABLE_MAP[variable]
    return data_root / "soilgrids" / f"soilgrids({label} {depth}).tif"


# ---------------------------------------------------------------------------
# Public source class
# ---------------------------------------------------------------------------


class SoilGridsSource:
    """Loads SoilGrids rasters and samples them at point locations.

    Parameters
    ----------
    cfg:
        Project configuration.
    variable:
        One of the keys in ``VARIABLE_MAP``.
    depth:
        One of the ``DEPTH_LAYERS`` strings, or ``None`` to return the
        depth-weighted mean across all three layers (weights proportional
        to layer thickness: 5 cm, 10 cm, 15 cm).
    """

    @classmethod
    def from_config(cls, cfg: Config) -> "SoilGridsSource":
        """Construct a default instance (clay, depth-averaged) from config."""
        return cls(cfg)

    def __init__(
        self,
        cfg: Config,
        variable: str = "clay",
        depth: str | None = None,
    ) -> None:
        if variable not in VARIABLE_MAP:
            raise ValueError(
                f"Unknown variable {variable!r}. "
                f"Choose one of: {sorted(VARIABLE_MAP)}"
            )
        if depth is not None and depth not in DEPTH_LAYERS:
            raise ValueError(
                f"Unknown depth {depth!r}. "
                f"Choose one of: {DEPTH_LAYERS} or None for depth-average."
            )
        self._cfg = cfg
        self.variable = variable
        self.depth = depth

        memory = make_memory(cfg)
        self._load_cached = memory.cache(self._load_uncached)

    # ------------------------------------------------------------------
    # Public interface (matches GridSource protocol)
    # ------------------------------------------------------------------

    def load_raster(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
        resolution_m: int,
    ) -> tuple[np.ndarray, object]:
        """Return ``(array (H, W), affine_transform)`` in study CRS.

        The array is a float32 raster. If *depth* was set to ``None`` at
        construction time, the value is the depth-weighted mean across all
        three SoilGrids layers.
        """
        return self._load_cached(lon_min, lat_min, lon_max, lat_max, resolution_m)

    def sample_at_points(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> np.ndarray:
        """Bilinearly interpolate soil values at WGS84 lon/lat points."""
        sa = self._cfg.study_area
        array, transform = self.load_raster(
            sa.lon_min, sa.lat_min, sa.lon_max, sa.lat_max,
            sa.grid_resolution_m,
        )
        return _sample_raster(array, transform, lons, lats, sa.target_crs)

    def sample_at_projected(
        self, x_proj: np.ndarray, y_proj: np.ndarray
    ) -> np.ndarray:
        """Sample soil values at projected CRS coordinates (e.g. EPSG:3035).

        Converts projected → WGS84 then delegates to ``sample_at_points()``.
        """
        t = Transformer.from_crs(
            self._cfg.study_area.target_crs, "EPSG:4326", always_xy=True
        )
        lons, lats = t.transform(x_proj, y_proj)
        return self.sample_at_points(lons, lats)

    def get_data(self) -> "pd.DataFrame":
        """Load all 6 SoilGrids variables × 3 depth layers into a flat DataFrame.

        Returns a DataFrame with one row per grid cell and columns:
            x, y     — projected coordinates (EPSG:3035, metres)
            lon, lat — WGS84 coordinates
            {var}_{depth} for every variable in VARIABLE_MAP and every depth
            in DEPTH_LAYERS, e.g.:
                bulk_density_0-5m, bulk_density_5-15m, bulk_density_15-30m,
                clay_0-5m, clay_5-15m, clay_15-30m, ...
        """
        import pandas as pd

        sa = self._cfg.study_area

        # Load the first raster to get the grid shape / transform
        first_src = SoilGridsSource(self._cfg, variable="bulk_density", depth=DEPTH_LAYERS[0])
        arr0, transform = first_src.load_raster(
            sa.lon_min, sa.lat_min, sa.lon_max, sa.lat_max, sa.grid_resolution_m
        )
        H, W = arr0.shape

        # Build projected coordinate arrays
        x_origin = transform.c
        y_origin = transform.f
        dx = transform.a
        dy = transform.e  # negative

        xs = x_origin + (np.arange(W) + 0.5) * dx
        ys = y_origin + (np.arange(H) + 0.5) * dy
        xx, yy = np.meshgrid(xs, ys)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        # Convert to WGS84
        t_inv = Transformer.from_crs(sa.target_crs, "EPSG:4326", always_xy=True)
        lons, lats = t_inv.transform(x_flat, y_flat)

        records: dict[str, np.ndarray] = {
            "x": x_flat,
            "y": y_flat,
            "lon": lons.astype(np.float32),
            "lat": lats.astype(np.float32),
        }

        for var in VARIABLE_MAP:
            for depth in DEPTH_LAYERS:
                col = f"{var}_{depth}"
                src = SoilGridsSource(self._cfg, variable=var, depth=depth)
                arr, _ = src.load_raster(
                    sa.lon_min, sa.lat_min, sa.lon_max, sa.lat_max, sa.grid_resolution_m
                )
                records[col] = arr.ravel()

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_uncached(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
        resolution_m: int,
    ) -> tuple[np.ndarray, object]:
        """Load, reproject and (optionally) depth-average the SoilGrids rasters."""
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_origin

        data_root = Path(self._cfg.paths.root)
        t_proj = Transformer.from_crs("EPSG:4326", self._cfg.study_area.target_crs, always_xy=True)
        _BUF = 0.1  # degrees — ensures boundary stations aren't clipped from raster
        x_min, y_min = t_proj.transform(lon_min - _BUF, lat_min - _BUF)
        x_max, y_max = t_proj.transform(lon_max + _BUF, lat_max + _BUF)

        dst_w = math.ceil((x_max - x_min) / resolution_m)
        dst_h = math.ceil((y_max - y_min) / resolution_m)
        dst_transform = from_origin(x_min, y_max, resolution_m, resolution_m)

        depths = [self.depth] if self.depth is not None else DEPTH_LAYERS
        weights = (
            [1.0]
            if self.depth is not None
            else [w / sum(_DEPTH_WEIGHTS) for w in _DEPTH_WEIGHTS]
        )

        accumulator = np.zeros((dst_h, dst_w), dtype=np.float64)

        for d, w in zip(depths, weights):
            tif_path = _file_for(data_root, self.variable, d)
            if not tif_path.exists():
                raise FileNotFoundError(
                    f"SoilGrids file not found: {tif_path}. "
                    "Expected files in data/soilgrids/."
                )
            with rasterio.open(tif_path) as src:
                raw = src.read(1).astype(np.float32)
                # Replace rasterio nodata and explicit sentinel (32767 for int16)
                nodata = src.nodata
                if nodata is not None:
                    raw[raw == nodata] = np.nan
                raw[raw == 32767] = np.nan

                layer = np.empty((dst_h, dst_w), dtype=np.float32)
                reproject(
                    source=raw,
                    destination=layer,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=self._cfg.study_area.target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                )
            accumulator += w * np.where(np.isnan(layer), 0.0, layer)

        return accumulator.astype(np.float32), dst_transform


# ---------------------------------------------------------------------------
# Shared raster-sampling helper (mirrors dem.py)
# ---------------------------------------------------------------------------


def _sample_raster(
    array: np.ndarray,
    transform: object,
    lons: np.ndarray,
    lats: np.ndarray,
    target_crs: str,
) -> np.ndarray:
    """Bilinear sampling of a projected raster at WGS84 coordinates."""
    t = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    x_q, y_q = t.transform(lons, lats)

    H, W = array.shape
    x_origin = transform.c
    y_origin = transform.f
    dx = transform.a
    dy = transform.e  # negative

    xs = x_origin + (np.arange(W) + 0.5) * dx
    ys = y_origin + (np.arange(H) + 0.5) * dy  # descending

    if ys[0] > ys[-1]:
        interp = RegularGridInterpolator(
            (ys[::-1], xs), array[::-1], method="linear", bounds_error=False, fill_value=np.nan
        )
    else:
        interp = RegularGridInterpolator(
            (ys, xs), array, method="linear", bounds_error=False, fill_value=np.nan
        )

    return interp(np.column_stack([y_q, x_q])).astype(np.float32)
