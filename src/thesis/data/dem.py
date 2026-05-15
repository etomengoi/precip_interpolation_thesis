"""Copernicus DEM (Digital Elevation Model) loader.

Source: Copernicus DEM GLO-30 (30 m resolution, global coverage).
Local tiles live in data/dem/ as GeoTIFF files downloaded from
the Copernicus S3 bucket.

The tiles are merged and reprojected to the study CRS on first use;
subsequent calls are served from the joblib cache.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

from thesis.config import Config
from thesis.cache import make_memory


class DEMSource:
    """Loads elevation raster from Copernicus DEM for the study area."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        memory = make_memory(cfg)
        self._load_tile = memory.cache(self._load_tile_uncached)
        self._raster_cache: tuple[np.ndarray, object] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_raster(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
        resolution_m: int,
    ) -> tuple[np.ndarray, object]:
        """Return (elevation_array (H, W), affine_transform) in study CRS."""
        return self._load_tile(lon_min, lat_min, lon_max, lat_max, resolution_m)

    def sample_at_points(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> np.ndarray:
        """Bilinearly interpolate elevation at the given WGS84 lon/lat points."""
        sa = self._cfg.study_area
        array, transform = self.load_raster(
            sa.lon_min, sa.lat_min, sa.lon_max, sa.lat_max,
            sa.grid_resolution_m,
        )
        return _sample_raster(array, transform, lons, lats, sa.target_crs)

    def sample_at_projected(
        self, x_proj: np.ndarray, y_proj: np.ndarray
    ) -> np.ndarray:
        """Sample elevation at projected CRS coordinates (e.g. EPSG:3035).

        Converts projected → WGS84 then delegates to sample_at_points().
        """
        t = Transformer.from_crs(
            self._cfg.study_area.target_crs, "EPSG:4326", always_xy=True
        )
        lons, lats = t.transform(x_proj, y_proj)
        return self.sample_at_points(lons, lats)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_tile_uncached(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
        resolution_m: int,
    ) -> tuple[np.ndarray, object]:
        """Merge DEM GeoTIFF tiles, crop to bbox, reproject to study CRS."""
        import rasterio
        from rasterio.merge import merge
        from rasterio.warp import reproject, Resampling
        import math
        from rasterio.transform import from_origin

        dem_dir = Path(self._cfg.paths.root) / "dem"
        tif_files = sorted(glob.glob(str(dem_dir / "Copernicus_DSM_COG_10_*.tif")))

        if not tif_files:
            raise FileNotFoundError(f"No DEM tiles found in {dem_dir}.")

        src_files = [rasterio.open(f) for f in tif_files]
        try:
            mosaic, mosaic_transform = merge(src_files)
        finally:
            for s in src_files:
                s.close()

        mosaic = mosaic[0]  # single band

        # Reproject to study CRS at target resolution
        t_proj = Transformer.from_crs("EPSG:4326", self._cfg.study_area.target_crs, always_xy=True)
        x_min, y_min = t_proj.transform(lon_min, lat_min)
        x_max, y_max = t_proj.transform(lon_max, lat_max)
        

        dst_w = math.ceil((x_max - x_min) / resolution_m)
        dst_h = math.ceil((y_max - y_min) / resolution_m)
        # from_origin: top-left corner, pixel_width, pixel_height (positive)
        dst_transform = from_origin(x_min, y_max, resolution_m, resolution_m)

        dst_array = np.empty((dst_h, dst_w), dtype=np.float32)
        reproject(
            source=mosaic,
            destination=dst_array,
            src_transform=mosaic_transform,
            src_crs="EPSG:4326",
            dst_transform=dst_transform,
            dst_crs=self._cfg.study_area.target_crs,
            resampling=Resampling.bilinear,
        )

        return dst_array, dst_transform


def _sample_raster(
    array: np.ndarray,
    transform: object,
    lons: np.ndarray,
    lats: np.ndarray,
    target_crs: str,
) -> np.ndarray:
    """Bilinear sampling of a reprojected raster at WGS84 coordinates.

    The raster is in the projected CRS (target_crs), so we first convert
    the lon/lat query points, then interpolate on the regular grid.
    """
    t = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    x_q, y_q = t.transform(lons, lats)

    # Build coordinate axes from the affine transform
    H, W = array.shape
    # transform: pixel (col, row) → projected (x, y)
    x_origin = transform.c  # top-left X
    y_origin = transform.f  # top-left Y
    dx = transform.a        # pixel width
    dy = transform.e        # pixel height (negative)

    xs = x_origin + (np.arange(W) + 0.5) * dx
    ys = y_origin + (np.arange(H) + 0.5) * dy  # descending

    # RegularGridInterpolator expects ascending axes
    if ys[0] > ys[-1]:
        ys = ys[::-1]
        array = array[::-1, :]

    interp = RegularGridInterpolator(
        (ys, xs), array, method="linear", bounds_error=False, fill_value=0.0
    )
    return interp(np.column_stack([y_q, x_q])).astype(np.float32)
