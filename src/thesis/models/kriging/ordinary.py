"""Ordinary Kriging — production interpolation using PyKrige.

Two-stage approach (Haylock et al. 2008, Hofstra et al. 2008):
  1. Indicator kriging on rain_indicator → P(rain) at each grid cell.
     Variogram: spherical (Hofstra 2008).
     Wet-day threshold: P > 0.4 (Hofstra 2008).
  2. Ordinary kriging on fwd_fn(precip_quota) at wet stations.

References:
    Haylock et al. (2008) JGR:Atm doi:10.1029/2008JD010201
    Hofstra et al. (2008) JGR:Atm doi:10.1029/2008JD010100
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import RBFInterpolator

from thesis.config import Config
from thesis.datasets.protocols import InterpolationResult, PredictionGrid, StationDataset
from thesis.models.kriging.pykrige_adapter import to_pykrige_params
from thesis.transforms.pipeline import TransformPipeline

_CHUNK_SIZE = 25_000


def _build_pykrige_ok(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    model: str,
    params_dict: dict,
) -> OrdinaryKriging:
    """Build a PyKrige OrdinaryKriging object with pre-fitted variogram params."""
    pk_params = to_pykrige_params(params_dict, model)
    return OrdinaryKriging(
        x, y, z,
        variogram_model=model,
        variogram_parameters=pk_params,
        coordinates_type="euclidean",
        exact_values=True,
    )


def _predict_chunked(
    ok: OrdinaryKriging,
    gx: np.ndarray,
    gy: np.ndarray,
    chunk_size: int = _CHUNK_SIZE,
    n_closest_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute pykrige prediction in chunks to limit memory usage."""
    import math

    # PyKrige loop backend is required for moving-window selection.
    # Cap at n_train so we never request more neighbours than exist.
    if n_closest_points is not None:
        n_closest_points = min(n_closest_points, len(ok.X_ADJUSTED))
    backend = "loop" if n_closest_points is not None else "vectorized"

    n_total = len(gx)
    n_chunks = math.ceil(n_total / chunk_size)
    z_parts, var_parts = [], []

    for i in range(n_chunks):
        lo = i * chunk_size
        hi = min(lo + chunk_size, n_total)
        z_chunk, var_chunk = ok.execute(
            "points", gx[lo:hi], gy[lo:hi],
            n_closest_points=n_closest_points,
            backend=backend,
        )
        z_parts.append(np.asarray(z_chunk).ravel())
        var_parts.append(np.asarray(var_chunk).ravel())

    return np.concatenate(z_parts), np.concatenate(var_parts)


class OrdinaryKrigingModel:
    """Two-stage Ordinary Kriging: indicator (wet/dry) + amount (quota → mm)."""

    def __init__(
        self,
        cfg: Config,
        pipeline: TransformPipeline,
        fwd_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        inv_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self._cfg = cfg
        self._pipeline = pipeline
        self._fwd_fn = fwd_fn
        self._inv_fn = inv_fn
        self._indicator_krig: OrdinaryKriging | None = None
        self._precip_krig: OrdinaryKriging | None = None
        self._dataset: StationDataset | None = None
        self._clim_amount_params: dict | None = None
        self._clim_indicator_params: dict | None = None
        self._monthly_norm_grids_2d: np.ndarray | None = None
        self._monthly_norm_grids_3d: np.ndarray | None = None
        self._all_wet: bool = False

    def set_global_variogram(
        self,
        global_vgm: dict,
        transform: str,
        amount_model: str = "exponential",
    ) -> "OrdinaryKrigingModel":
        """Set variogram parameters from GlobalVariogramFitter output."""
        amt_info = global_vgm[(transform, amount_model)]
        self._clim_amount_params = {
            "variogram_model": amt_info["model"],
            "variogram_parameters": amt_info["params_dict"],
        }
        # Indicator variogram (optional — present if fit_indicator() was run)
        ind_key = ("indicator", "spherical")
        if ind_key in global_vgm and global_vgm[ind_key] is not None:
            ind_info = global_vgm[ind_key]
            self._clim_indicator_params = {
                "variogram_model": ind_info["model"],
                "variogram_parameters": ind_info["params_dict"],
            }
        return self

    def set_monthly_norm_grids(
        self,
        grids_2d: np.ndarray,
        grids_3d: np.ndarray | None,
    ) -> "OrdinaryKrigingModel":
        """Attach pre-computed monthly normal maps."""
        self._monthly_norm_grids_2d = grids_2d
        self._monthly_norm_grids_3d = grids_3d
        return self

    def fit(self, dataset: StationDataset) -> None:
        """Bind station data for this day and build kriging objects."""
        params = self._cfg.kriging
        x = dataset.coords_proj[:, 0]
        y = dataset.coords_proj[:, 1]

        # --- Stage 1: indicator kriging ---
        z_ind = dataset.rain_indicator.astype(float)
        n_wet = int(z_ind.sum())
        if n_wet == 0 or n_wet == len(z_ind):
            # Constant indicator (all dry or all wet) → variogram undefined.
            # Store None; predict() handles the shortcut.
            self._indicator_krig = None
            self._all_wet = (n_wet == len(z_ind))
        else:
            self._all_wet = False
            if self._clim_indicator_params is None:
                raise RuntimeError(
                    "Global indicator variogram not set. "
                    "Call set_global_variogram() with a variogram dict "
                    "containing the ('indicator', 'spherical') key."
                )
            self._indicator_krig = _build_pykrige_ok(
                x, y, z_ind,
                self._clim_indicator_params["variogram_model"],
                self._clim_indicator_params["variogram_parameters"],
            )

        # --- Stage 2: OK on transformed quota (wet stations only) ---
        wet = dataset.wet_mask()
        if wet.sum() >= params.n_stations_min:
            z_wet = self._apply_fwd(dataset.precip_quota[wet])
            valid = np.isfinite(z_wet)
            if valid.sum() >= params.n_stations_min:
                xw, yw, zw = x[wet][valid], y[wet][valid], z_wet[valid]
                if self._clim_amount_params is not None:
                    self._precip_krig = _build_pykrige_ok(
                        xw, yw, zw,
                        self._clim_amount_params["variogram_model"],
                        self._clim_amount_params["variogram_parameters"],
                    )
                else:
                    # Auto-fit variogram per day
                    self._precip_krig = OrdinaryKriging(
                        xw, yw, zw,
                        variogram_model=params.variogram_model_amount,
                        coordinates_type="euclidean",
                        exact_values=True,
                    )
            else:
                self._precip_krig = None
        else:
            self._precip_krig = None

        self._dataset = dataset

    def predict(self, grid: PredictionGrid) -> InterpolationResult:
        """Interpolate to grid, returning precipitation in mm."""
        import xarray as xr

        if self._dataset is None:
            raise RuntimeError("Call fit() before predict().")

        params = self._cfg.kriging
        gx = grid.coords_proj[:, 0]
        gy = grid.coords_proj[:, 1]
        H, W = grid.shape
        date = self._dataset.date

        # --- Stage 1: indicator kriging → wet/dry mask ---
        if self._indicator_krig is None:
            # Constant indicator (all wet or all dry) — no variogram needed
            wet_grid = np.full(H * W, self._all_wet)
        else:
            p_rain, _ = _predict_chunked(self._indicator_krig, gx, gy)
            p_rain = np.clip(p_rain, 0.0, 1.0)
            wet_grid = p_rain > params.indicator_probability_threshold

        # --- Stage 2: amount kriging ---
        if self._precip_krig is None or not wet_grid.any():
            mean_mm = np.zeros(H * W)
            var_mm2 = np.zeros(H * W)
        else:
            z_pred, z_var = _predict_chunked(
                self._precip_krig, gx, gy,
                n_closest_points=params.max_wet,
            )
            z_var = np.clip(z_var, 0.0, None)

            if self._monthly_norm_grids_2d is not None:
                month_idx = int(date[5:7]) - 1
                g3 = self._monthly_norm_grids_3d
                grid_normals = g3[month_idx] if g3 is not None else self._monthly_norm_grids_2d[month_idx]
            else:
                grid_normals = self._interpolate_monthly_normals(gx, gy, date, grid.elevation_m)

            # MC back-transformation with antithetic variates for variance reduction
            K_MC = 100
            rng = np.random.default_rng(self._cfg.random_seed)
            z_sigma = np.sqrt(np.maximum(z_var, 1e-8))[:, None]
            half = K_MC // 2
            eps = rng.standard_normal((len(z_pred), half))
            eps = np.concatenate([eps, -eps], axis=1)   # antithetic pairs
            z_samp = z_pred[:, None] + z_sigma * eps
            quota_samp = np.maximum(self._apply_inv(z_samp.ravel()), 0.0).reshape(len(z_pred), K_MC)

            mean_quota = quota_samp.mean(axis=1)
            var_quota = quota_samp.var(axis=1)

            mean_mm = np.where(wet_grid, np.clip(mean_quota * grid_normals, 0.0, None), 0.0)
            var_mm2 = np.where(wet_grid, var_quota * grid_normals ** 2, 0.0)

        mean_da = xr.DataArray(mean_mm.reshape(H, W), dims=["y", "x"], name="precip_mm")
        var_da = xr.DataArray(var_mm2.reshape(H, W), dims=["y", "x"], name="precip_var_mm2")

        return InterpolationResult(mean=mean_da, variance=var_da, date=date, model="OrdinaryKriging")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _interpolate_monthly_normals(
        self, gx: np.ndarray, gy: np.ndarray, date: str, g_elev: np.ndarray | None = None,
    ) -> np.ndarray:
        norms = self._dataset.monthly_normals
        if norms is None or int(np.sum(np.isfinite(norms))) < 3:
            raise ValueError(
                f"Too few valid monthly normals for TPS: "
                f"{int(np.sum(np.isfinite(norms))) if norms is not None else 0} < 3"
            )

        coords = self._dataset.coords_proj
        valid = np.isfinite(norms)

        station_elev = getattr(self._dataset, "elevation_m", None)
        use_3d = (
            station_elev is not None and g_elev is not None
            and np.any(np.isfinite(station_elev))
        )

        if use_3d:
            raw_station = np.column_stack([coords[valid], station_elev[valid]])
            raw_grid = np.column_stack([gx, gy, g_elev])
        else:
            raw_station = coords[valid]
            raw_grid = np.column_stack([gx, gy])

        feat_mean = raw_station.mean(axis=0)
        feat_std = np.maximum(raw_station.std(axis=0), 1.0)

        rbf = RBFInterpolator(
            (raw_station - feat_mean) / feat_std, norms[valid],
            kernel="thin_plate_spline", degree=1, smoothing=1.0,
        )
        return np.clip(rbf((raw_grid - feat_mean) / feat_std), 1.0, None)

    def _apply_fwd(self, quota: np.ndarray) -> np.ndarray:
        if self._fwd_fn is None:
            raise RuntimeError("fwd_fn must be provided to OrdinaryKrigingModel.")
        return self._fwd_fn(quota)

    def _apply_inv(self, z: np.ndarray) -> np.ndarray:
        if self._inv_fn is None:
            raise RuntimeError("inv_fn must be provided to OrdinaryKrigingModel.")
        return self._inv_fn(z)

