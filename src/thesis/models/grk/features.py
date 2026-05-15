"""Per-day geo-features for GRK Stage 2 (IDW, GOS, SVD-quantiles).

The function generalises the per-station leave-one-out feature pass used in
the Stage 2 hparam search. It accepts a `train_mask` selecting which stations
in the day's wet set may serve as neighbours; this allows leakage-free
k-fold CV (where all fold-mates of a held-out station are excluded from the
neighbour pool, not just the station itself).

Setting `train_mask = np.ones(n, dtype=bool)` reproduces the original per-
station LOO behaviour (each station sees every other station, with self
excluded by the dist > 1 m filter).
"""
from __future__ import annotations

import numpy as np


def compute_day_geo_features(
    date: str,
    xy_all: np.ndarray,        # (n, 2) projected coords (metres)
    z_all: np.ndarray,         # (n,) precip_mm
    sids_all: np.ndarray,      # (n,) station_id
    train_mask: np.ndarray,    # (n,) bool — True = station is in neighbour pool
    k: int,
    svd_quantiles: np.ndarray,
) -> list[dict]:
    """Compute IDW / GOS / SVD-quantile features for every wet station of a day.

    Neighbours are drawn only from `xy_all[train_mask]`. For query stations
    in the train set, self-exclusion is handled by the `dists > 1 m` filter;
    for query stations in the test fold (train_mask=False), no self-exclusion
    is needed since they are not in the neighbour pool.
    """
    from sklearn.neighbors import BallTree

    n = len(z_all)
    if n < 3 or train_mask.sum() < 2:
        return []

    xy_train = xy_all[train_mask]
    z_train  = z_all[train_mask]
    n_train  = len(z_train)
    if n_train < 2:
        return []

    tree = BallTree(xy_train, metric="euclidean")
    k_query = min(k + 1, n_train)
    dists_all, idxs_all = tree.query(xy_all, k=k_query)

    sigma = np.ones(len(svd_quantiles), dtype=np.float64) + 1e-8

    records: list[dict] = []
    for j in range(n):
        # Self-exclusion via dist > 1 m only matters when j is itself in the
        # neighbour pool; for test-fold queries the filter is a no-op.
        mask = dists_all[j] > 1.0
        nbr_raw  = idxs_all[j][mask][:k]
        dist_raw = dists_all[j][mask][:k]

        if len(nbr_raw) == 0:
            continue

        z_nbr = z_train[nbr_raw]
        d_nbr = dist_raw

        # IDW
        w = 1.0 / (d_nbr ** 2 + 1e-8)
        w /= w.sum()
        idw = float(np.dot(w, z_nbr))

        # SVD quantiles of neighbours
        svd_j = np.quantile(z_nbr, svd_quantiles)

        # GOS — similarity between each neighbour's value and the local profile
        station_profile = np.quantile(z_nbr, svd_quantiles)
        nbr_profiles = np.stack(
            [np.quantile(z_train[[idx]], svd_quantiles) for idx in nbr_raw]
        )
        diff = (nbr_profiles - station_profile) / sigma
        S = np.exp(-(diff ** 2).sum(axis=1))
        S_sum = S.sum() + 1e-8
        gos = float(np.dot(S / S_sum, z_nbr))

        row = {
            "date":       date,
            "station_id": sids_all[j],
            "idw":        idw,
            "gos":        gos,
        }
        for qi, q in enumerate(svd_quantiles):
            row[f"svd_{qi:02d}"] = float(svd_j[qi])

        records.append(row)

    return records
