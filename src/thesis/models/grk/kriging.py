"""GRK Stage 2 — Ordinary Kriging on LightGBM residuals.

Independent from `models/kriging/ordinary.py`: that file targets the baseline
two-stage indicator + amount kriging on quotas with NST/log back-transform and
TPS monthly-normal grids. None of that machinery applies to residuals
(already in mm, no nonlinear inverse, no occurrence stage).

Pipeline:
    1. fit a global exponential variogram on residuals pooled across train-days
       (`fit_global_residual_variogram`)
    2. per day, build a `pykrige.OrdinaryKriging` with the pre-fitted variogram
       and predict at test stations with `n_closest_points = 125` (baseline
       value from `config.py:KrigingParams.max_wet`)

Reuses Marquardt LS / σ=1/√N pair-count weighting from
`models/kriging/variogram_fitter.py` to keep the variogram fit identical in
spirit to the baseline OK.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pykrige.ok import OrdinaryKriging
from scipy.optimize import curve_fit

from thesis.models.kriging.pykrige_adapter import to_pykrige_params
from thesis.models.kriging.variogram_fitter import (
    _accumulate_day_arrays,
    _vgm_exponential,
)


@dataclass
class VariogramFit:
    """Fitted variogram together with the empirical cloud used to fit it."""
    params: dict           # {"nugget": float, "psill": float, "range": float}
    pk_params: dict        # to_pykrige_params(params, model)
    model: str             # "exponential"
    lag_centers_m: np.ndarray
    emp_gamma: np.ndarray
    bins_count: np.ndarray


def fit_global_residual_variogram(
    day_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_lags: int = 38,
    max_lag_m: float = 416_000.0,
    min_pairs: int = 30,
    model: str = "exponential",
) -> VariogramFit:
    """Pool semi-variances across days, fit a single global variogram.

    Parameters
    ----------
    day_arrays
        One (x, y, residual) numpy triple per day. x, y in metres (EPSG:3035).
    n_lags, max_lag_m, min_pairs
        Defaults mirror baseline OK config (`KrigingParams` + `kriging.tex`).
    model
        Currently only "exponential" supported (matches Hofstra 2008 / baseline
        choice for precipitation amounts).
    """
    if model != "exponential":
        raise NotImplementedError(f"Variogram model {model!r} not supported")

    lag_edges   = np.linspace(0.0, max_lag_m, n_lags + 1)
    lag_centers = 0.5 * (lag_edges[:-1] + lag_edges[1:])

    bins_sum   = np.zeros(n_lags)
    bins_count = np.zeros(n_lags, dtype=np.int64)
    for x, y, r in day_arrays:
        bs, bc = _accumulate_day_arrays(x, y, r, lag_edges, n_lags, max_lag_m)
        bins_sum   += bs
        bins_count += bc

    emp_gamma = np.where(bins_count >= min_pairs, bins_sum / bins_count, np.nan)
    sill_est  = float(np.nanmax(emp_gamma)) if np.any(np.isfinite(emp_gamma)) else 1.0

    valid = np.isfinite(emp_gamma) & (lag_centers > 0) & (bins_count >= min_pairs)
    if valid.sum() < 4:
        raise RuntimeError(
            f"Too few valid lag bins to fit variogram: {valid.sum()} < 4"
        )
    lc, eg, cnt = lag_centers[valid], emp_gamma[valid], bins_count[valid]

    sigma_w = 1.0 / np.sqrt(cnt.astype(float))   # Haylock 2008 weighting

    nugget0 = float(eg[0]) * 0.1
    psill0  = max(sill_est - nugget0, 1e-10)
    range0  = max_lag_m * 0.3

    popt, _ = curve_fit(
        _vgm_exponential, lc, eg,
        p0=[nugget0, psill0, range0],
        sigma=sigma_w, absolute_sigma=False,
        bounds=([1e-6, 0.0, 1000.0], [np.inf, np.inf, max_lag_m]),
        maxfev=10000,
    )
    nugget, psill, rng_m = popt
    params = {"nugget": float(nugget), "psill": float(psill), "range": float(rng_m)}
    return VariogramFit(
        params=params,
        pk_params=to_pykrige_params(params, model),
        model=model,
        lag_centers_m=lag_centers,
        emp_gamma=emp_gamma,
        bins_count=bins_count,
    )


class GRKResidualKriging:
    """Local Ordinary Kriging on residuals with a pre-fitted variogram.

    Single-day predictor: bind one day's training residuals via `fit`, then
    `predict` at any number of test points. Local kriging uses the
    `n_closest_points` nearest train stations (default 125 — baseline value).

    Variance returned by `predict` is the kriging variance σ²_K of the
    residual, in the same units (mm²). For CRPS, draw K MC samples from
    N(μ, σ²_K), shift by deterministic LGB prediction, clip at zero — the
    notebook handles that step.
    """

    def __init__(self, n_closest_points: int = 125) -> None:
        self._n_closest = n_closest_points
        self._variogram: VariogramFit | None = None
        self._ok: OrdinaryKriging | None = None

    def set_variogram(self, vgm: VariogramFit) -> "GRKResidualKriging":
        self._variogram = vgm
        return self

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        residuals: np.ndarray,
    ) -> "GRKResidualKriging":
        if self._variogram is None:
            raise RuntimeError("Call set_variogram(...) before fit().")
        if len(x) < 3:
            self._ok = None
            return self
        self._ok = OrdinaryKriging(
            x, y, residuals,
            variogram_model=self._variogram.model,
            variogram_parameters=self._variogram.pk_params,
            coordinates_type="euclidean",
            exact_values=True,
        )
        return self

    def predict(
        self,
        x_query: np.ndarray,
        y_query: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean_residual, variance_residual). Both shape = (n_query,)."""
        if self._ok is None:
            return np.zeros(len(x_query)), np.zeros(len(x_query))
        n_use = min(self._n_closest, len(self._ok.X_ADJUSTED))
        r_pred, var_pred = self._ok.execute(
            "points", x_query, y_query,
            n_closest_points=n_use, backend="loop",
        )
        return (
            np.asarray(r_pred).ravel(),
            np.clip(np.asarray(var_pred).ravel(), 0.0, None),
        )
