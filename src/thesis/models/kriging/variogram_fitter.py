"""Climatological variogram fitting (Haylock et al. 2008, Hofstra et al. 2008).

Pools empirical semi-variance from many days, fits theoretical models
(spherical/exponential/gaussian) via sigma-weighted curve_fit.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist

from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Module-level parallel worker (must be at module scope for joblib/loky)
# ---------------------------------------------------------------------------

def _accumulate_day_arrays(
    xw: np.ndarray,
    yw: np.ndarray,
    zw: np.ndarray,
    lag_edges: np.ndarray,
    n_lags: int,
    max_lag: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute semi-variance bins for one day. Returns (bins_sum, bins_count)."""
    bins_sum   = np.zeros(n_lags)
    bins_count = np.zeros(n_lags, dtype=np.int64)

    if len(xw) < 3:
        return bins_sum, bins_count

    coords = np.column_stack([xw, yw])
    dists  = pdist(coords)
    zdiffs = pdist(zw[:, None], metric="euclidean") ** 2 * 0.5  # γ_ij

    in_range = dists <= max_lag
    if not in_range.any():
        return bins_sum, bins_count

    bin_idx = np.searchsorted(lag_edges[1:], dists[in_range])
    bin_idx = np.clip(bin_idx, 0, n_lags - 1)
    bins_sum   += np.bincount(bin_idx, weights=zdiffs[in_range], minlength=n_lags)
    bins_count += np.bincount(bin_idx, minlength=n_lags).astype(np.int64)

    return bins_sum, bins_count


# ---------------------------------------------------------------------------
# Theoretical variogram models — thin wrappers around pykrige
# ---------------------------------------------------------------------------
from pykrige.variogram_models import (
    exponential_variogram_model as _pk_exponential,
    gaussian_variogram_model as _pk_gaussian,
    spherical_variogram_model as _pk_spherical,
)
from thesis.models.kriging.pykrige_adapter import to_pykrige_params_list as _to_pk


def _vgm_spherical(h: np.ndarray, nugget: float, psill: float, a: float) -> np.ndarray:
    pk = _to_pk({"nugget": nugget, "psill": psill, "range": a}, "spherical")
    return _pk_spherical(pk, h)


def _vgm_exponential(h: np.ndarray, nugget: float, psill: float, a: float) -> np.ndarray:
    pk = _to_pk({"nugget": nugget, "psill": psill, "range": a}, "exponential")
    return _pk_exponential(pk, h)


def _vgm_gaussian(h: np.ndarray, nugget: float, psill: float, a: float) -> np.ndarray:
    pk = _to_pk({"nugget": nugget, "psill": psill, "range": a}, "gaussian")
    return _pk_gaussian(pk, h)


_VGM_FN: dict[str, Callable] = {
    "spherical":   _vgm_spherical,
    "exponential": _vgm_exponential,
    "gaussian":    _vgm_gaussian,
}

# Type alias for the fitted variogram dict
VariogramParams = dict  # {"nugget": float, "psill": float, "range": float}
VariogramInfo   = dict  # {"model": str, "params_dict": VariogramParams}
GlobalVgm       = dict[tuple[str, str], VariogramInfo | None]


class GlobalVariogramFitter:
    """Fits one climatological variogram per (transform, model) pair."""

    def __init__(
        self,
        transforms: list[str],
        variogram_models: list[str],
        n_lags: int = 38,
        max_lag_km: float = 450.0,
        min_pairs: int = 30,
        checkpoint_path: str | None = "outputs/global_variograms_checkpoint.pkl",
        n_jobs: int = 1,
    ) -> None:
        self.transforms       = transforms
        self.variogram_models = variogram_models
        self.n_lags           = n_lags
        self.max_lag_km       = max_lag_km
        self.min_pairs        = min_pairs
        self.checkpoint_path  = checkpoint_path
        self.n_jobs           = n_jobs

        self._max_lag   = max_lag_km * 1000.0          # metres
        self._lag_edges = np.linspace(0.0, self._max_lag, n_lags + 1)
        self._lag_centers = 0.5 * (self._lag_edges[:-1] + self._lag_edges[1:])

        self._result: GlobalVgm | None = None
        # Empirical variograms per transform — populated during fit(), saved alongside result
        self._emp_data: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        pool_procs: list[pd.DataFrame],
        fwd_fn: Callable[[np.ndarray, str], np.ndarray],
    ) -> GlobalVgm:
        """Fit climatological variograms from pooled daily data."""
        from joblib import Parallel, delayed

        global_vgm: GlobalVgm = {}

        for t_idx, t_name in enumerate(self.transforms):
            sub_tr = t_name

            # Phase 1 — sequential: apply fwd_fn (closure, not picklable) and
            # extract plain numpy arrays for each day.
            desc = f"[{t_idx + 1}/{len(self.transforms)}] '{t_name}' — fwd transform"
            day_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for proc in tqdm(pool_procs, desc=desc, unit="day"):
                wet = (proc["rain_indicator"] == 1).values
                if wet.sum() < 3:
                    continue
                values = proc["precip_quota"].values[wet]
                z = fwd_fn(values, sub_tr)
                valid = np.isfinite(z)
                if valid.sum() < 3:
                    continue
                day_arrays.append((
                    proc["x_proj"].values[wet][valid],
                    proc["y_proj"].values[wet][valid],
                    z[valid],
                ))

            # Phase 2 — parallel CPU: pdist + binning over all days.
            # GPU avoided: 64 CUDA contexts × ~500 MB = OOM. CPU is sufficient.
            print(
                f"  Accumulating semi-variances: {len(day_arrays)} days, "
                f"n_jobs={self.n_jobs}, device=CPU"
            )
            partial_results: list[tuple[np.ndarray, np.ndarray]] = Parallel(
                n_jobs=self.n_jobs,
                backend="loky",
                verbose=0,
            )(
                delayed(_accumulate_day_arrays)(
                    xw, yw, zw, self._lag_edges, self.n_lags, self._max_lag,
                )
                for xw, yw, zw in day_arrays
            )

            bins_sum   = np.zeros(self.n_lags)
            bins_count = np.zeros(self.n_lags, dtype=np.int64)
            for bs, bc in partial_results:
                bins_sum   += bs
                bins_count += bc

            emp_gamma = np.where(bins_count >= self.min_pairs, bins_sum / bins_count, np.nan)
            sill_est  = float(np.nanmax(emp_gamma)) if np.any(np.isfinite(emp_gamma)) else 1.0

            total_pairs = int(bins_count.sum())
            filled_bins = int((bins_count >= self.min_pairs).sum())
            print(f"Transform '{t_name}': {total_pairs:,} pairs, sill_est={sill_est:.4f}")
            print(f"  Bins with pairs: {filled_bins}/{self.n_lags}")

            # Store empirical variogram for this transform (used in analytics notebook)
            self._emp_data[t_name] = {
                "lag_km":    self._lag_centers / 1000.0,
                "gamma":     emp_gamma.copy(),
                "count":     bins_count.copy(),
            }

            for vm in self.variogram_models:
                key = (t_name, vm)
                result = self._fit_one(vm, self._lag_centers, emp_gamma, bins_count, sill_est)
                if result is None:
                    print(f"  {vm:<12} -> FAILED")
                    global_vgm[key] = None
                else:
                    params, chi2r = result
                    global_vgm[key] = {"model": vm, "params_dict": params, "chi2_reduced": chi2r}
                    print(
                        f"  {vm:<12} -> "
                        f"nugget={params['nugget']:.4f}  "
                        f"psill={params['psill']:.4f}  "
                        f"range={params['range'] / 1000:.0f} km  "
                        f"chi2r={chi2r:.3f}"
                    )

            # Checkpoint after each transform — survive crashes
            if self.checkpoint_path is not None:
                Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.checkpoint_path, "wb") as f:
                    pickle.dump(global_vgm, f)
                print(
                    f"  [checkpoint] {t_idx + 1}/{len(self.transforms)} transforms"
                    f" -> {self.checkpoint_path}"
                )

        print("\nGlobal variograms fitted.")
        self._result = global_vgm
        return global_vgm

    def fit_indicator(
        self,
        pool_procs: list[pd.DataFrame],
    ) -> GlobalVgm:
        """Fit a single global spherical variogram for the binary indicator.

        Uses ALL stations (wet + dry) unlike fit() which uses only wet.
        Skips all-wet or all-dry days (indicator variance = 0).
        Only fits spherical model (Haylock 2008, §37).

        The result is stored under key ("indicator", "spherical") and merged
        into self._result so save() writes everything together.
        """
        from joblib import Parallel, delayed

        # Phase 1 — extract indicator arrays, skipping constant days
        day_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        n_skipped = 0
        for proc in tqdm(pool_procs, desc="Indicator — extracting days", unit="day"):
            z_ind = proc["rain_indicator"].values.astype(float)
            n_wet = int(z_ind.sum())
            if n_wet == 0 or n_wet == len(z_ind):
                n_skipped += 1
                continue
            if len(z_ind) < 3:
                continue
            day_arrays.append((
                proc["x_proj"].values,
                proc["y_proj"].values,
                z_ind,
            ))

        print(
            f"  Indicator: {len(day_arrays)} days pooled, "
            f"{n_skipped} constant days skipped"
        )

        # Phase 2 — parallel semi-variance binning
        print(
            f"  Accumulating indicator semi-variances: "
            f"{len(day_arrays)} days, n_jobs={self.n_jobs}"
        )
        partial_results: list[tuple[np.ndarray, np.ndarray]] = Parallel(
            n_jobs=self.n_jobs,
            backend="loky",
            verbose=0,
        )(
            delayed(_accumulate_day_arrays)(
                xw, yw, zw, self._lag_edges, self.n_lags, self._max_lag,
            )
            for xw, yw, zw in day_arrays
        )

        bins_sum   = np.zeros(self.n_lags)
        bins_count = np.zeros(self.n_lags, dtype=np.int64)
        for bs, bc in partial_results:
            bins_sum   += bs
            bins_count += bc

        emp_gamma = np.where(
            bins_count >= self.min_pairs, bins_sum / bins_count, np.nan,
        )
        sill_est = float(np.nanmax(emp_gamma)) if np.any(np.isfinite(emp_gamma)) else 0.25

        total_pairs = int(bins_count.sum())
        filled_bins = int((bins_count >= self.min_pairs).sum())
        print(f"  Indicator: {total_pairs:,} pairs, sill_est={sill_est:.4f}")
        print(f"  Bins with pairs: {filled_bins}/{self.n_lags}")

        # Store empirical variogram
        self._emp_data["indicator"] = {
            "lag_km": self._lag_centers / 1000.0,
            "gamma":  emp_gamma.copy(),
            "count":  bins_count.copy(),
        }

        # Phase 3 — fit spherical model only
        key = ("indicator", "spherical")
        result = self._fit_one("spherical", self._lag_centers, emp_gamma, bins_count, sill_est)

        if self._result is None:
            self._result = {}

        if result is None:
            print("  spherical  -> FAILED")
            self._result[key] = None
        else:
            params, chi2r = result
            self._result[key] = {
                "model": "spherical",
                "params_dict": params,
                "chi2_reduced": chi2r,
            }
            print(
                f"  spherical  -> "
                f"nugget={params['nugget']:.4f}  "
                f"psill={params['psill']:.4f}  "
                f"range={params['range'] / 1000:.0f} km  "
                f"chi2r={chi2r:.3f}"
            )

        return self._result

    def save(self, path: str) -> None:
        """Save fitted variograms + empirical data to pickle."""
        if self._result is None:
            raise RuntimeError("Call fit() before save().")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "variograms":    self._result,
            "empirical":     self._emp_data,
            "lag_centers_km": self._lag_centers / 1000.0,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved: {path}")

    @staticmethod
    def load(path: str) -> GlobalVgm:
        """Load fitted variograms dict from pickle (backward-compatible)."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        vgm = data["variograms"] if isinstance(data, dict) and "variograms" in data else data
        print(f"Loaded: {path}  ({len(vgm)} combinations)")
        return vgm

    @staticmethod
    def load_full(path: str) -> dict:
        """Load full payload: variograms + empirical data + lag centers."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "variograms" in data:
            return data
        # Old format (plain GlobalVgm dict) — no empirical data available
        return {"variograms": data, "empirical": {}, "lag_centers_km": None}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_one(
        self,
        vm: str,
        lag_centers: np.ndarray,
        emp_gamma: np.ndarray,
        bins_count: np.ndarray,
        sill_est: float,
    ) -> tuple[VariogramParams, float] | None:
        """Fit one variogram model. Returns (params_dict, chi2_reduced) or None."""
        fn = _VGM_FN[vm]
        valid = np.isfinite(emp_gamma) & (lag_centers > 0) & (bins_count >= self.min_pairs)
        if valid.sum() < 4:
            return None

        lc  = lag_centers[valid]
        eg  = emp_gamma[valid]
        cnt = bins_count[valid]

        # Sigma-weighting by bin count (Haylock 2008):
        # σ = 1/sqrt(N) → larger weight for well-populated bins
        sigma_w = 1.0 / np.sqrt(cnt.astype(float))

        nugget0 = float(eg[0]) * 0.1
        psill0  = max(float(sill_est) - nugget0, 1e-10)
        range0  = self._max_lag * 0.3

        try:
            popt, _ = curve_fit(
                fn, lc, eg,
                p0=[nugget0, psill0, range0],
                sigma=sigma_w,
                absolute_sigma=False,
                # nugget >= 1e-6: prevents kriging matrix singularity for dense networks
                bounds=([1e-6, 0.0, 1000.0], [np.inf, np.inf, self._max_lag]),
                maxfev=10000,
            )
            nugget, psill, range_m = popt

            # Weighted chi-square (Haylock 2008 model selection criterion)
            gamma_fitted = fn(lc, *popt)
            chi2 = float(np.sum((eg - gamma_fitted) ** 2 * cnt))
            chi2_reduced = chi2 / max(len(lc) - 3, 1)

            params = {"nugget": float(nugget), "psill": float(psill), "range": float(range_m)}
            return params, chi2_reduced
        except (RuntimeError, ValueError):
            return None
