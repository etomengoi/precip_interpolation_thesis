"""Spatial LOO-CV for kriging model screening (Haylock 2008, Hofstra 2008).

Evaluates (transform × variogram_model) combinations via station-level
LOO-CV. Uses Dubrule (1983) Cholesky LOO for small networks and local
Schur complement for large ones (n > max_wet). Parallelised over dates.
"""
from __future__ import annotations

import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from properscoring import crps_gaussian

from tqdm.auto import tqdm

from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve

from thesis.config import Config
from thesis.models.kriging.neighbors import effective_range as _eff_range
from thesis.models.kriging.covariance import apply_cov_nugget as _apply_cov_nugget_numpy
from thesis.models.kriging.variogram_fitter import GlobalVgm

# Type alias for returned results
CvResults = dict[tuple[str, str], dict]  # {"mae_mm", "crps_z", "crps_mm", "n"}

# Valid modes for monthly-normal estimation in LOO-CV
NORM_MODES = ("station", "tps_2d", "tps_3d")


@contextmanager
def _tqdm_joblib(tqdm_object):
    """Patch joblib to update tqdm on completion with line-buffered progress."""
    import time as _time
    from joblib.parallel import BatchCompletionCallBack

    old_call = BatchCompletionCallBack.__call__
    _state = {"done": 0, "t0": _time.monotonic(), "last_print": 0}
    _total = tqdm_object.total or 1
    _interval = max(1, _total // 20)  # ~5 % steps

    def new_call(self, *args, **kwargs):
        tqdm_object.update(n=self.batch_size)
        _state["done"] += self.batch_size
        if _state["done"] - _state["last_print"] >= _interval or _state["done"] >= _total:
            elapsed = _time.monotonic() - _state["t0"]
            pct = 100.0 * _state["done"] / _total
            rate = _state["done"] / max(elapsed, 0.01)
            eta = (_total - _state["done"]) / max(rate, 0.01)
            print(
                f"[progress] {_state['done']:>6d}/{_total}  "
                f"{pct:5.1f}%  {rate:.1f} day/s  "
                f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
                flush=True,
            )
            _state["last_print"] = _state["done"]
        return old_call(self, *args, **kwargs)

    BatchCompletionCallBack.__call__ = new_call
    try:
        yield tqdm_object
    finally:
        BatchCompletionCallBack.__call__ = old_call


def _write_checkpoint(path: str | None, data: CvResults) -> None:
    """Atomically write CV results dict to a pickle checkpoint file."""
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)



def _tps_predict_norm_at_point(
    x_train: np.ndarray,
    y_train: np.ndarray,
    elev_train: np.ndarray | None,
    norms_train: np.ndarray,
    x_test: float,
    y_test: float,
    elev_test: float | None,
    use_3d: bool,
) -> float:
    """TPS-interpolate monthly normal at one held-out station (Haylock 2008)."""
    from scipy.interpolate import RBFInterpolator

    valid = np.isfinite(norms_train)
    if valid.sum() < 3:
        raise ValueError(f"Too few valid norms for TPS: {valid.sum()} < 3")

    if use_3d and elev_train is not None and elev_test is not None:
        raw_s = np.column_stack([x_train[valid], y_train[valid], elev_train[valid]])
        raw_t = np.array([[x_test, y_test, elev_test]])
    else:
        raw_s = np.column_stack([x_train[valid], y_train[valid]])
        raw_t = np.array([[x_test, y_test]])

    feat_mean = raw_s.mean(axis=0)
    feat_std = raw_s.std(axis=0)
    feat_std[feat_std < 1.0] = 1.0

    rbf = RBFInterpolator(
        (raw_s - feat_mean) / feat_std,
        norms_train[valid],
        kernel="thin_plate_spline",
        degree=1,
        smoothing=1.0,
    )
    pred = float(rbf((raw_t - feat_mean) / feat_std)[0])
    return max(pred, 1.0)


def _compute_crps_vectorized(
    z_wet: np.ndarray,
    z_pred_all: np.ndarray,
    z_var_all: np.ndarray,
    mm_wet: np.ndarray,
    inv_fn: Callable,
    sub_tr: str,
    k_mc: int,
    rng: np.random.Generator,
    *,
    station_norms: np.ndarray | None = None,
    use_tps: bool = False,
    tps_3d: bool = False,
    x_wet: np.ndarray | None = None,
    y_wet: np.ndarray | None = None,
    elev_wet: np.ndarray | None = None,
    n_wet: int = 0,
    precomputed_norms: np.ndarray | None = None,
) -> dict[str, list]:
    """Compute MAE, CRPS_z, CRPS_mm for all stations — vectorized numpy."""
    valid = np.isfinite(z_pred_all) & np.isfinite(z_var_all)
    if not valid.any():
        return {"mae": [], "crps_z": [], "crps_mm": []}

    idx_v = np.where(valid)[0]
    n_v   = len(idx_v)
    zp    = z_pred_all[idx_v]
    zv    = z_var_all[idx_v]
    zo    = z_wet[idx_v]

    s_all  = np.sqrt(np.maximum(zv, 1e-16))
    crps_z = crps_gaussian(zo, mu=zp, sig=s_all)

    # MC sampling with antithetic variates
    half = k_mc // 2
    eps = rng.standard_normal((n_v, half))
    eps = np.concatenate([eps, -eps], axis=1)
    if k_mc % 2:
        eps = np.concatenate([eps, rng.standard_normal((n_v, 1))], axis=1)
    z_samp = zp[:, None] + s_all[:, None] * eps
    z_flat = z_samp.ravel()

    quota_samp = np.maximum(inv_fn(z_flat, sub_tr), 0.0).reshape(n_v, k_mc)
    quota_pred = np.maximum(inv_fn(zp, sub_tr), 0.0)

    # Get norms: precomputed > TPS LOO > station
    if precomputed_norms is not None:
        norms_v = precomputed_norms[idx_v]
    elif use_tps and x_wet is not None and y_wet is not None:
        norms_v = np.empty(n_v)
        for j, i in enumerate(idx_v):
            loo_mask = np.ones(n_wet, dtype=bool)
            loo_mask[i] = False
            norms_v[j] = _tps_predict_norm_at_point(
                x_wet[loo_mask], y_wet[loo_mask],
                elev_wet[loo_mask] if elev_wet is not None else None,
                station_norms[loo_mask],
                x_wet[i], y_wet[i],
                float(elev_wet[i]) if elev_wet is not None else None,
                use_3d=tps_3d,
            )
    else:
        norms_v = station_norms[idx_v]

    mm_samp = quota_samp * norms_v[:, None]
    mm_pred = quota_pred * norms_v

    mae = np.abs(mm_pred - mm_wet[idx_v])

    # Fair CRPS_mm (Ferro 2014) via Gini spread — O(nK log K), no properscoring.
    # CRPS_fair = E|X-y| - Gini/(K(K-1)) where Gini = Σ_j x_j (2j - K + 1)
    K = mm_samp.shape[1]
    mm_sorted = np.sort(mm_samp, axis=1)
    term1 = np.mean(np.abs(mm_sorted - mm_wet[idx_v, None]), axis=1)
    gini_coeff = 2.0 * np.arange(K) - K + 1.0
    crps_mm = term1 - (mm_sorted @ gini_coeff) / (K * (K - 1))

    return {
        "mae":     mae.tolist(),
        "crps_z":  crps_z.tolist(),
        "crps_mm": crps_mm.tolist(),
    }


# ---------------------------------------------------------------------------
# CPU worker — must be at module scope for joblib/loky pickling
# ---------------------------------------------------------------------------

def _process_one_day_multi_vgm(
    date: str,
    proc: pd.DataFrame | None,
    vgm_infos: list[dict | None],
    vgm_names: list[str],
    sub_tr: str,
    fwd_fn: Callable,
    inv_fn: Callable,
    get_monthly_total_fn: Callable,
    n_stations_min: int,
    k_mc: int,
    seed: int,
    norm_mode: str = "station",
    precomputed_norms: np.ndarray | None = None,
    max_wet: int | None = None,
) -> dict[str, dict[str, list]]:
    """Process one date for one transform × multiple variogram models.

    Returns {model_name: {"mae": [...], "crps_z": [...], "crps_mm": [...]}}.
    """
    import time as _time

    empty = {m: {"mae": [], "crps_z": [], "crps_mm": []} for m in vgm_names}

    if proc is None or proc.empty:
        return empty

    wet_mask = (proc["rain_indicator"] == 1).values
    n_wet_orig = int(wet_mask.sum())
    if n_wet_orig < n_stations_min + 1:
        return empty

    # -- Shared across all variogram models --
    x_wet_orig      = proc["x_proj"].values[wet_mask]
    y_wet_orig      = proc["y_proj"].values[wet_mask]
    mm_wet_orig     = proc["precip_mm"].values[wet_mask]
    values_wet_orig = proc["precip_quota"].values[wet_mask]
    z_wet_orig = fwd_fn(values_wet_orig, sub_tr)
    mean_t     = get_monthly_total_fn(date)

    use_tps = norm_mode in ("tps_2d", "tps_3d")
    elev_wet_orig: np.ndarray | None = None
    if use_tps and "elevation_m" in proc.columns:
        elev_wet_orig = proc["elevation_m"].values[wet_mask]

    centroid_x = float(x_wet_orig.mean())
    centroid_y = float(y_wet_orig.mean())
    dist_to_centroid = np.hypot(x_wet_orig - centroid_x, y_wet_orig - centroid_y)

    # -- Per-variogram model: filter, build kriging system, solve --
    results: dict[str, dict[str, list]] = {}
    for vgm_info, vm_name in zip(vgm_infos, vgm_names):
        if vgm_info is None:
            results[vm_name] = {"mae": [], "crps_z": [], "crps_mm": []}
            continue

        rng = np.random.default_rng(seed)

        # Effective-range filtering (variogram-dependent)
        eff_r = _eff_range(vgm_info["params_dict"], vgm_info["model"])
        in_range = dist_to_centroid <= eff_r
        if in_range.sum() >= n_stations_min + 1:
            mask = in_range
        else:
            mask = np.ones(n_wet_orig, dtype=bool)

        x_wet      = x_wet_orig[mask]
        y_wet      = y_wet_orig[mask]
        z_wet      = z_wet_orig[mask]
        mm_wet     = mm_wet_orig[mask]
        values_wet = values_wet_orig[mask]
        elev_wet   = elev_wet_orig[mask] if elev_wet_orig is not None else None
        n_wet      = int(mask.sum())

        valid = np.isfinite(z_wet)
        if valid.sum() < n_stations_min + 1:
            results[vm_name] = {"mae": [], "crps_z": [], "crps_mm": []}
            continue
        x_wet      = x_wet[valid]
        y_wet      = y_wet[valid]
        z_wet      = z_wet[valid]
        mm_wet     = mm_wet[valid]
        values_wet = values_wet[valid]
        if elev_wet is not None:
            elev_wet = elev_wet[valid]
        n_wet = len(x_wet)

        station_norms = np.where(values_wet > 0, mm_wet / values_wet, mean_t)

        tps_3d = norm_mode == "tps_3d" and elev_wet is not None

        # Slice precomputed norms to match effective-range + valid masks
        pc_norms_masked = None
        if precomputed_norms is not None:
            pc_norms_masked = precomputed_norms[mask][valid]

        # Kriging system — LOO cross-validation
        t0 = _time.perf_counter()
        coords = np.column_stack([x_wet, y_wet])
        dist_mat = cdist(coords, coords)

        if max_wet is not None and n_wet > max_wet:
            # --- Local kriging LOO (mirrors production max_wet cap) ---
            # Each station is predicted from its K nearest neighbors.
            # Vectorized: batch cKDTree query + Schur complement solve
            # on the max_wet×max_wet SPD covariance submatrix.
            from scipy.spatial import cKDTree

            tree = cKDTree(coords)
            cov_full = _apply_cov_nugget_numpy(dist_mat, vgm_info)

            k_query = min(max_wet + 1, n_wet)
            _, all_idxs = tree.query(coords, k=k_query)
            nbr = all_idxs[:, 1:max_wet + 1]  # (n_wet, max_wet)
            n_loc = nbr.shape[1]

            sill = float(vgm_info["params_dict"]["nugget"]) + float(vgm_info["params_dict"]["psill"])
            z_pred_all = np.full(n_wet, np.nan)
            z_var_all  = np.full(n_wet, np.nan)
            row_idx = np.arange(n_wet)

            _BATCH = 500
            for lo in range(0, n_wet, _BATCH):
                hi = min(lo + _BATCH, n_wet)
                bs = hi - lo
                b_nbr = nbr[lo:hi]

                b_target = cov_full[row_idx[lo:hi, None], b_nbr]              # (bs, n_loc)
                b_local  = cov_full[b_nbr[:, :, None], b_nbr[:, None, :]]    # (bs, n_loc, n_loc)

                # Schur complement: solve C·[v|u] = [c|1], then μ and w
                rhs2 = np.zeros((bs, n_loc, 2))
                rhs2[:, :, 0] = b_target
                rhs2[:, :, 1] = 1.0

                sol = np.linalg.solve(b_local, rhs2)                      # (bs, n_loc, 2)

                v = sol[:, :, 0]                                               # C⁻¹c
                u = sol[:, :, 1]                                               # C⁻¹·1
                mu = (np.sum(v, axis=1) - 1.0) / np.sum(u, axis=1)            # Lagrange
                wk = v - mu[:, None] * u                                       # kriging weights

                z_pred_all[lo:hi] = np.einsum("ij,ij->i", wk, z_wet[b_nbr])
                z_var_all[lo:hi]  = np.maximum(
                    sill - np.einsum("ij,ij->i", wk, b_target) - mu, 1e-8,
                )

            del cov_full

        else:
            # --- Global Dubrule/Cholesky LOO (Dubrule 1983) ---
            # n_wet ≤ max_wet: all stations fit in one neighbourhood,
            # so one Cholesky gives all LOO predictions at once.
            cov_mat = _apply_cov_nugget_numpy(dist_mat, vgm_info)

            L = cho_factor(cov_mat)
            ones = np.ones(n_wet)
            Cinv_z = cho_solve(L, z_wet)
            Cinv_ones = cho_solve(L, ones)
            denom = ones @ Cinv_ones

            Cinv = cho_solve(L, np.eye(n_wet))
            mu_num = ones @ Cinv_z
            diag_Cinv = np.diag(Cinv)

            diag_Q = diag_Cinv - (Cinv_ones ** 2) / denom
            Qz_top = Cinv_z - Cinv_ones * (mu_num / denom)
            del Cinv

            del cov_mat

            valid_diag = diag_Q > 1e-12
            z_pred_all = np.where(valid_diag, z_wet - Qz_top / diag_Q, np.nan)
            z_var_all = np.where(valid_diag, np.maximum(1.0 / diag_Q, 1e-8), np.nan)

        results[vm_name] = _compute_crps_vectorized(
            z_wet, z_pred_all, z_var_all, mm_wet,
            inv_fn, sub_tr, k_mc, rng,
            station_norms=station_norms,
            use_tps=use_tps, tps_3d=tps_3d,
            x_wet=x_wet, y_wet=y_wet, elev_wet=elev_wet,
            n_wet=n_wet,
            precomputed_norms=pc_norms_masked,
        )

    return results


def _process_one_day_all_combos(
    date: str,
    proc: pd.DataFrame | None,
    vgm_infos_by_transform: dict[str, list],
    models: list[str],
    fwd_fn: Callable,
    inv_fn: Callable,
    get_monthly_total_fn: Callable,
    n_stations_min: int,
    k_mc: int,
    seed: int,
    norm_mode: str,
    precomputed_norms: np.ndarray | None,
    max_wet: int | None,
) -> dict[tuple[str, str], dict[str, list]]:
    """Process one date for ALL (transform × model) combos in a single worker call.

    3× less IPC than running 3 separate Parallel() calls:
    - proc frame serialized once instead of once-per-transform
    - fwd_fn/inv_fn closures serialized once instead of once-per-transform

    Returns {(t_name, vm): {"mae": [...], "crps_z": [...], "crps_mm": [...]}}.
    """
    results: dict[tuple[str, str], dict[str, list]] = {}
    for t_name, vgm_infos in vgm_infos_by_transform.items():
        day_res = _process_one_day_multi_vgm(
            date, proc, vgm_infos, models,
            t_name,
            fwd_fn, inv_fn, get_monthly_total_fn,
            n_stations_min, k_mc, seed, norm_mode, precomputed_norms, max_wet,
        )
        for vm in models:
            results[(t_name, vm)] = day_res[vm]
    return results


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SpatialLooCV:
    """Station-level spatial LOO cross-validation for variogram model screening.

    Usage
    -----
    >>> cv = SpatialLooCV(
    ...     global_vgm=global_vgm,
    ...     fwd_fn=fwd,
    ...     inv_fn=inv,
    ...     get_monthly_total_fn=get_mean_monthly_total,
    ...     cfg=cfg,
    ...     rng=np.random.default_rng(42),
    ...     n_test_days=60,
    ...     k_mc=100,
    ...     checkpoint_path="outputs/cv_results_checkpoint.pkl",
    ...     n_jobs=4,   # parallel workers over dates; -1 = all cores
    ... )
    >>> cv_results = cv.run(test_dates, load_proc_fn=load_base_proc)

    To load a previously saved checkpoint (skip re-running):
    >>> cv_results = SpatialLooCV.load("outputs/cv_results_checkpoint.pkl")
    """

    def __init__(
        self,
        global_vgm: GlobalVgm,
        fwd_fn: Callable[[np.ndarray, str], np.ndarray],
        inv_fn: Callable[[np.ndarray, str], np.ndarray],
        get_monthly_total_fn: Callable[[str], float],
        cfg: Config,
        rng: np.random.Generator,
        n_test_days: int = 60,
        k_mc: int = 100,
        checkpoint_path: str | None = "outputs/cv_results_checkpoint.pkl",
        n_jobs: int = 1,
        norm_mode: str = "station",
        precomputed_norms_fn: Callable[[str], np.ndarray | None] | None = None,
    ) -> None:
        if norm_mode not in NORM_MODES:
            raise ValueError(f"norm_mode must be one of {NORM_MODES}, got {norm_mode!r}")
        self.global_vgm            = global_vgm
        self._fwd                  = fwd_fn
        self._inv                  = inv_fn
        self._get_monthly_total    = get_monthly_total_fn
        self._cfg                  = cfg
        self._rng                  = rng
        self.n_test_days           = n_test_days
        self.k_mc                  = k_mc
        self.checkpoint_path       = checkpoint_path
        self.n_jobs                = n_jobs
        self.norm_mode             = norm_mode
        self._precomputed_norms_fn = precomputed_norms_fn

        self._result: CvResults | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        test_dates: list[str],
        load_proc_fn: Callable[[str], pd.DataFrame],
    ) -> CvResults:
        """Run spatial LOO-CV over all (transform, variogram_model) combinations.

        Parameters
        ----------
        test_dates:
            List of date strings (YYYY-MM-DD) to use as test days. Typically
            a random subsample of the full period (n_test_days days).
        load_proc_fn:
            Callable that loads and pre-processes data for one date string,
            returning a DataFrame with columns: ``rain_indicator``,
            ``precip_mm``, ``precip_quota``, ``x_proj``, ``y_proj``.

        Returns
        -------
        cv_results : dict[(transform, model)] -> {mae_mm, crps_z, crps_mm, n}
        """
        print("LOO-CV workers: CPU (loky)")

        # Load existing checkpoint — skip already completed combos on resume
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            with open(self.checkpoint_path, "rb") as f:
                cv_results: CvResults = pickle.load(f)
            print(f"Loaded checkpoint: {self.checkpoint_path}  ({len(cv_results)} entries)")
        else:
            cv_results = {}

        transforms = list(dict.fromkeys(k[0] for k in self.global_vgm))
        models     = list(dict.fromkeys(k[1] for k in self.global_vgm))
        total      = len(transforms) * len(models)

        # Mark variogram-failed combos immediately
        for t_name in transforms:
            for vm in models:
                key = (t_name, vm)
                if key not in cv_results and self.global_vgm.get(key) is None:
                    cv_results[key] = {"mae_mm": np.nan, "crps_z": np.nan, "crps_mm": np.nan, "n": 0}
                    print(f"{t_name} x {vm} -> SKIP (variogram failed)")
                    _write_checkpoint(self.checkpoint_path, cv_results)

        pending_combos = [(t, m) for t in transforms for m in models
                          if (t, m) not in cv_results]

        if not pending_combos:
            print("All combos already in checkpoint — nothing to run.")
            self._result = cv_results
            return cv_results

        # Spawn one child seed per date — reproducible and independent across workers
        ss          = np.random.SeedSequence(int(self._rng.integers(2**31)))
        child_seeds = ss.spawn(len(test_dates))
        seed_ints   = [int(s.generate_state(1)[0]) for s in child_seeds]

        # Build vgm_infos_by_transform for ALL pending transforms
        pending_transforms = list(dict.fromkeys(t for t, _ in pending_combos))
        vgm_infos_by_transform: dict[str, list[dict | None]] = {
            t_name: [self.global_vgm.get((t_name, vm)) for vm in models]
            for t_name in pending_transforms
        }

        n_pending = len(pending_combos)
        print(
            f"Running {n_pending}/{total} combos "
            f"({len(pending_transforms)} transforms × {len(models)} models) "
            f"over {len(test_dates)} days — single Parallel dispatch"
        )

        # Single Parallel dispatch: each worker handles ONE day × ALL transforms × ALL models.
        # 3× less IPC than 3 sequential Parallel calls (proc frame serialized once per day).
        all_day_results = self._run_cpu_all(
            test_dates, seed_ints, load_proc_fn,
            vgm_infos_by_transform, pending_transforms, models, total,
        )

        # Aggregate per (transform, model)
        for t_name in pending_transforms:
            for vm in models:
                key = (t_name, vm)
                if key in cv_results:
                    continue  # already done from checkpoint

                chunks_mae     = [r[(t_name, vm)]["mae"]     for r in all_day_results if r[(t_name, vm)]["mae"]]
                chunks_crps_z  = [r[(t_name, vm)]["crps_z"]  for r in all_day_results if r[(t_name, vm)]["crps_z"]]
                chunks_crps_mm = [r[(t_name, vm)]["crps_mm"] for r in all_day_results if r[(t_name, vm)]["crps_mm"]]

                all_mae     = np.concatenate(chunks_mae)     if chunks_mae     else np.array([])
                all_crps_z  = np.concatenate(chunks_crps_z)  if chunks_crps_z  else np.array([])
                all_crps_mm = np.concatenate(chunks_crps_mm) if chunks_crps_mm else np.array([])

                cv_results[key] = {
                    "mae_mm":  float(all_mae.mean())     if len(all_mae)     else np.nan,
                    "crps_z":  float(all_crps_z.mean())  if len(all_crps_z)  else np.nan,
                    "crps_mm": float(all_crps_mm.mean()) if len(all_crps_mm) else np.nan,
                    "n":       len(all_mae),
                }
                r = cv_results[key]
                print(
                    f"{t_name:<18} x {vm:<12} -> "
                    f"MAE={r['mae_mm']:.3f} mm  "
                    f"CRPS_z={r['crps_z']:.4f}  "
                    f"CRPS_mm={r['crps_mm']:.4f}  "
                    f"(n={r['n']})"
                )
                _write_checkpoint(self.checkpoint_path, cv_results)

        print("\nSpatial LOO-CV complete.")
        self._result = cv_results
        return cv_results

    def save(self, path: str) -> None:
        """Save CV results to a pickle file."""
        if self._result is None:
            raise RuntimeError("Call run() before save().")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._result, f)
        print(f"Saved: {path}")

    @staticmethod
    def load(path: str) -> CvResults:
        """Load previously computed CV results from pickle (skips re-running)."""
        with open(path, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded: {path}  ({len(results)} combinations)")
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_cpu_all(
        self,
        test_dates: list[str],
        seed_ints: list[int],
        load_proc_fn: Callable,
        vgm_infos_by_transform: dict[str, list],
        transforms: list[str],
        models: list[str],
        total: int,
    ) -> list[dict]:
        """Single Parallel dispatch: one worker per day × ALL transforms × ALL models.

        Compared to 3 sequential Parallel() calls (one per transform):
        - proc frames loaded once instead of 3×
        - fwd_fn/inv_fn closures serialized once per day instead of 3×
        - 3× less IPC overhead overall
        """
        from joblib import Parallel, delayed

        # Pre-resolve DataFrames in parent process — each worker gets only its
        # day's small DataFrame (~300 KB), not the full 7 GB proc_by_date dict.
        proc_frames: list[pd.DataFrame | None] = [
            load_proc_fn(d) for d in test_dates
        ]

        norms_per_date: list[np.ndarray | None] = (
            [self._precomputed_norms_fn(d) for d in test_dates]
            if self._precomputed_norms_fn is not None
            else [None] * len(test_dates)
        )

        n_combos = len(transforms) * len(models)
        desc = f"LOO-CV {n_combos} combos"

        pbar = tqdm(
            total=len(test_dates),
            desc=desc,
            unit="day",
            miniters=max(1, len(test_dates) // 20),
        )
        with _tqdm_joblib(pbar):
            all_day_results = Parallel(
                n_jobs=self.n_jobs,
                backend="loky",
                verbose=0,
            )(
                delayed(_process_one_day_all_combos)(
                    date, proc,
                    vgm_infos_by_transform, models,
                    self._fwd, self._inv, self._get_monthly_total,
                    self._cfg.kriging.n_stations_min,
                    self.k_mc, seed, self.norm_mode, pc_norms,
                    self._cfg.kriging.max_wet,
                )
                for date, proc, seed, pc_norms in zip(
                    test_dates, proc_frames, seed_ints, norms_per_date,
                )
            )

        return all_day_results
