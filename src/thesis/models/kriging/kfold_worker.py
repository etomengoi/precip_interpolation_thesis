"""Per-day k-fold kriging prediction worker.

Trains pykrige Ordinary Kriging on fold!=k stations and predicts at
fold==k test station coordinates. Used by notebooks/03_kriging/.

Module-level so it pickles cleanly under joblib loky.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from thesis.models.kriging.ordinary import _build_pykrige_ok, _predict_chunked


def predict_one_day_one_combo(
    date: str,
    proc_train: pd.DataFrame,
    proc_test: pd.DataFrame,
    vgm_info: dict,
    indicator_vgm_info: dict | None,    # ('indicator', 'spherical') from global_vgm
    transform: str,
    fwd_fn: Callable,
    inv_fn: Callable,
    norms_2d_train: np.ndarray | None,   # (12, n_test) test-point monthly normals (TPS fit on train fold)
    norms_3d_train: np.ndarray | None,   # (12, n_test) or None
    n_stations_min: int,
    max_wet: int | None,
    k_mc: int,
    seed: int,
    use_3d_norms: bool,
    indicator_threshold: float = 0.4,
) -> pd.DataFrame:
    """Predict one day, one (transform × variogram_model) combo.

    Returns a per-record DataFrame:
      date, station_id, observed_mm, predicted_mm, predicted_var_mm2,
      observed_quota, predicted_z, predicted_z_var, transform, variogram_model
    Empty rows when prediction not possible (constant indicator, too few wet, etc.).
    """
    if proc_train is None or proc_test is None or proc_test.empty:
        return _empty_df()

    month_idx = int(date[5:7]) - 1

    # ── Test station info ───────────────────────────────────────────────
    test_x        = proc_test["x_proj"].values
    test_y        = proc_test["y_proj"].values
    test_sids     = proc_test["station_id"].values
    test_obs_mm   = proc_test["precip_mm"].values
    test_obs_ind  = proc_test["rain_indicator"].values
    test_obs_quota = proc_test["precip_quota"].values
    n_test = len(test_x)

    # Choose monthly normal source for back-transform at each test station
    if use_3d_norms and norms_3d_train is not None:
        test_norms = norms_3d_train[month_idx]
    elif norms_2d_train is not None:
        test_norms = norms_2d_train[month_idx]
    else:
        # Fallback: grand mean (should not happen if monthly norms were precomputed)
        test_norms = np.full(n_test, 1.0)

    # ── Stage 1: indicator kriging on train, predict at test ────────────
    train_x_all = proc_train["x_proj"].values
    train_y_all = proc_train["y_proj"].values
    train_ind   = proc_train["rain_indicator"].values.astype(float)

    n_wet_train = int(train_ind.sum())
    if n_wet_train == 0:
        # All-dry day on train fold → predict zero rain everywhere
        return _make_df(
            date, test_sids, test_obs_mm, test_obs_quota,
            np.zeros(n_test), np.zeros(n_test),
            np.full(n_test, np.nan), np.full(n_test, np.nan),
            transform, vgm_info["model"],
        )

    if n_wet_train == len(train_ind):
        # All-wet day on train fold → assume wet everywhere (skip indicator step)
        wet_test = np.ones(n_test, dtype=bool)
    elif indicator_vgm_info is None:
        # No indicator variogram supplied → fallback per-day auto-fit (slow but safe)
        from pykrige.ok import OrdinaryKriging
        ind_ok = OrdinaryKriging(
            train_x_all, train_y_all, train_ind,
            variogram_model="spherical",
            coordinates_type="euclidean",
            exact_values=True,
        )
        p_rain, _ = _predict_chunked(ind_ok, test_x, test_y)
        p_rain = np.clip(p_rain, 0.0, 1.0)
        wet_test = p_rain > indicator_threshold
    else:
        ind_ok = _build_pykrige_ok(
            train_x_all, train_y_all, train_ind,
            indicator_vgm_info["model"], indicator_vgm_info["params_dict"],
        )
        p_rain, _ = _predict_chunked(ind_ok, test_x, test_y)
        p_rain = np.clip(p_rain, 0.0, 1.0)
        wet_test = p_rain > indicator_threshold

    # ── Stage 2: amount kriging at wet test points ──────────────────────
    train_wet_mask = train_ind > 0.5
    if train_wet_mask.sum() < n_stations_min:
        # Too few wet train stations → predict zero rain
        return _make_df(
            date, test_sids, test_obs_mm, test_obs_quota,
            np.zeros(n_test), np.zeros(n_test),
            np.full(n_test, np.nan), np.full(n_test, np.nan),
            transform, vgm_info["model"],
        )

    train_x_wet = train_x_all[train_wet_mask]
    train_y_wet = train_y_all[train_wet_mask]
    train_quota = proc_train["precip_quota"].values[train_wet_mask]
    z_train = fwd_fn(train_quota, transform)
    valid = np.isfinite(z_train)
    if valid.sum() < n_stations_min:
        return _make_df(
            date, test_sids, test_obs_mm, test_obs_quota,
            np.zeros(n_test), np.zeros(n_test),
            np.full(n_test, np.nan), np.full(n_test, np.nan),
            transform, vgm_info["model"],
        )

    train_x_wet = train_x_wet[valid]
    train_y_wet = train_y_wet[valid]
    z_train     = z_train[valid]

    ok = _build_pykrige_ok(
        train_x_wet, train_y_wet, z_train,
        vgm_info["model"], vgm_info["params_dict"],
    )

    z_pred, z_var = _predict_chunked(
        ok, test_x, test_y, n_closest_points=max_wet,
    )
    z_var = np.clip(z_var, 0.0, None)

    # ── MC back-transform: z → quota → mm ───────────────────────────────
    rng = np.random.default_rng(seed + month_idx)
    z_sigma = np.sqrt(np.maximum(z_var, 1e-8))[:, None]
    half = k_mc // 2
    eps = rng.standard_normal((n_test, half))
    eps = np.concatenate([eps, -eps], axis=1)             # antithetic pairs
    z_samp = z_pred[:, None] + z_sigma * eps               # (n_test, k_mc)
    quota_samp = np.maximum(
        inv_fn(z_samp.ravel(), transform), 0.0,
    ).reshape(n_test, k_mc)

    mean_quota = quota_samp.mean(axis=1)
    var_quota  = quota_samp.var(axis=1)

    pred_mm  = np.where(wet_test, np.clip(mean_quota * test_norms, 0.0, None), 0.0)
    pred_var = np.where(wet_test, var_quota * test_norms ** 2, 0.0)

    return _make_df(
        date, test_sids, test_obs_mm, test_obs_quota,
        pred_mm, pred_var,
        z_pred, z_var,
        transform, vgm_info["model"],
    )


def predict_one_day_all_combos(
    date: str,
    proc_train: pd.DataFrame,
    proc_test: pd.DataFrame,
    global_vgm: dict,           # {(transform, vgm_model): vgm_info}
    transforms: list[str],
    vgm_models: list[str],
    fwd_fn: Callable,
    inv_fn: Callable,
    norms_2d_train: np.ndarray | None,
    norms_3d_train: np.ndarray | None,
    n_stations_min: int,
    max_wet: int | None,
    k_mc: int,
    seed: int,
    use_3d_norms: bool,
) -> pd.DataFrame:
    """Run all 9 (transform × variogram_model) combos for one day.
    Returns concatenated per-record DataFrame for all combos.
    """
    indicator_vgm_info = global_vgm.get(("indicator", "spherical"))
    parts = []
    for t in transforms:
        for vm in vgm_models:
            vgm_info = global_vgm.get((t, vm))
            if vgm_info is None:
                continue
            df = predict_one_day_one_combo(
                date, proc_train, proc_test, vgm_info, indicator_vgm_info, t,
                fwd_fn, inv_fn, norms_2d_train, norms_3d_train,
                n_stations_min, max_wet, k_mc, seed, use_3d_norms,
            )
            if not df.empty:
                parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else _empty_df()


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": [], "station_id": [],
        "observed_mm": [], "observed_quota": [],
        "predicted_mm": [], "predicted_var_mm2": [],
        "predicted_z": [], "predicted_z_var": [],
        "transform": [], "variogram_model": [],
    })


def _make_df(
    date: str,
    sids: np.ndarray,
    obs_mm: np.ndarray,
    obs_quota: np.ndarray,
    pred_mm: np.ndarray,
    pred_var: np.ndarray,
    z_pred: np.ndarray,
    z_var: np.ndarray,
    transform: str,
    vgm_model: str,
) -> pd.DataFrame:
    n = len(sids)
    return pd.DataFrame({
        "date":              np.repeat(date, n),
        "station_id":        sids,
        "observed_mm":       obs_mm,
        "observed_quota":    obs_quota,
        "predicted_mm":      pred_mm,
        "predicted_var_mm2": pred_var,
        "predicted_z":       z_pred,
        "predicted_z_var":   z_var,
        "transform":         np.repeat(transform, n),
        "variogram_model":   np.repeat(vgm_model, n),
    })
