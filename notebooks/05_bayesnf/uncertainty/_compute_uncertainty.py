"""Compute uncertainty/calibration diagnostics for a BayesNF run.

Produces ``uncertainty.json`` and uploads it to the run folder on S3.

Diagnostics
-----------
1. **PIT** (Probability Integral Transform) — global + wet-only.
   Piecewise-linear interpolation of the predictive CDF at the observed
   value; histogram counts + Kolmogorov–Smirnov statistic against Uniform.
2. **Reliability per quantile level** — empirical coverage of each
   nominal quantile (global + wet).
3. **Sharpness** — distribution of interval widths
   ``q95−q05``, ``q90−q10``, ``q80−q20``, ``q60−q40``.
4. **Spread–skill** — Gaussian-equivalent ensemble σ vs RMSE per
   binned predicted mean.
5. **Hersbach (2000) CRPS decomposition** —
   ``CRPS = Reliability + CRPS_pot`` from per-bin (α, β) accumulators,
   computed exactly from the 11 quantile levels.
6. **CRPSS vs climatology** — skill score against (a) global pooled
   training rainfall and (b) 5-NN train-station pooled climatology
   (test stations are spatially held out, so per-test-station CDF
   has to be spatially imputed).
7. **Conditional CRPS** — per-row CRPS averaged within observed
   intensity bins ``[0, 0.5, 5, 20, 50, ∞]``.
8. **Wet-detector Brier decomposition** — Reliability/Resolution/
   Uncertainty for the binary event ``y ≥ 0.5 mm`` using the
   forecast probability ``P̂(Y ≥ 0.5 mm)`` interpolated from the
   quantile-CDF.

References
----------
- Hersbach (2000) *Decomposition of the CRPS for Ensemble Prediction Systems*.
  Wea. Forecasting 15:559–570.
- Gneiting & Raftery (2007) *Strictly Proper Scoring Rules*. JASA 102:359–378.
- Bröcker (2009) *Reliability, sufficiency, and the decomposition of
  proper scores*. Q.J.R. Meteorol. Soc. 135:1512–1519.

Usage
-----
::

    python notebooks/05_bayesnf/uncertainty/_compute_uncertainty.py \\
        vi__WY_h1_10__ffrk_full_32_5e-3_kl0.1_s5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_RUNS   = PROJECT_ROOT / "results" / "bayesnf" / "runs"
S3_BUCKET    = "thesis-data-ismaktam"
S3_RUNS_ROOT = "bayesnf/runs"
DATA_CACHE   = Path("/tmp/bnf_uncertainty")

QUANTILE_COLS    = ["q05","q10","q20","q30","q40","q50","q60","q70","q80","q90","q95"]
QUANTILE_LEVELS  = np.array([0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95])

WET_THRESHOLD_MM = 0.5
PIT_BINS         = 20

CLIM_KNN         = 5    # for nearest-station climatology baseline
INTENSITY_BINS   = [0.0, 0.5, 5.0, 20.0, 50.0, np.inf]
INTENSITY_NAMES  = ["dry (=0)", "0.5–5 mm", "5–20 mm", "20–50 mm", ">50 mm"]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Inject AWS creds from .env if present."""
    env = PROJECT_ROOT / ".env"
    if env.exists():
        import os
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _ensure_run_artifacts(run_name: str) -> dict:
    """Download config / metrics / preds locally; return path dict."""
    _load_env()
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name="eu-north-1")
    paths: dict[str, Path] = {}
    for fname in ("config.json", "metrics.json", "preds.parquet"):
        local = DATA_CACHE / fname
        if not local.exists():
            s3.download_file(S3_BUCKET, f"{S3_RUNS_ROOT}/{run_name}/{fname}", str(local))
        paths[fname] = local
    return paths


def _ensure_clim_data() -> tuple[Path, Path]:
    """Download fold0 train + test parquets for climatology baseline."""
    _load_env()
    s3 = boto3.client("s3", region_name="eu-north-1")
    train = DATA_CACHE / "fold0_train.parquet"
    test  = DATA_CACHE / "fold0_test.parquet"
    for local, key in [
        (train, "bayesnf/data/bayesnf_fold0_train.parquet"),
        (test,  "bayesnf/data/bayesnf_fold0_test.parquet"),
    ]:
        if not local.exists():
            s3.download_file(S3_BUCKET, key, str(local))
    return train, test


# ---------------------------------------------------------------------------
# Block 1 — PIT (Probability Integral Transform)
# ---------------------------------------------------------------------------

def _pit_from_quantiles(y: np.ndarray, q: np.ndarray,
                        levels: np.ndarray) -> np.ndarray:
    """Piecewise-linear PIT.

    For each row, interpolate the predictive CDF F(·) from the (q_p, p)
    points and evaluate at ``y``. Outside the quantile range, clamp to
    0 / 1. For ``y == q_p`` with ties (e.g. several q_p = 0), the PIT
    value is the largest p with q_p ≤ y — i.e. ``np.interp`` with the
    'right' insertion.
    """
    n = y.shape[0]
    pit = np.empty(n, dtype=np.float64)
    for i in range(n):
        pit[i] = np.interp(y[i], q[i], levels, left=0.0, right=1.0)
    return pit


def block_pit(y: np.ndarray, q: np.ndarray) -> dict:
    """PIT histogram + KS test against U[0,1]; global + wet subsets."""
    pit_all = _pit_from_quantiles(y, q, QUANTILE_LEVELS)
    out: dict = {}
    for name, mask in [("global", np.ones(len(y), dtype=bool)),
                       ("wet",    y >= WET_THRESHOLD_MM)]:
        p = pit_all[mask]
        counts, edges = np.histogram(p, bins=PIT_BINS, range=(0, 1))
        ks_stat, ks_p = stats.kstest(p, "uniform")
        out[name] = {
            "n": int(mask.sum()),
            "histogram_counts": counts.tolist(),
            "histogram_edges":  edges.tolist(),
            "ks_statistic":     float(ks_stat),
            "ks_p_value":       float(ks_p),
            "mean":             float(p.mean()),
            "std":              float(p.std()),
        }
    return out


# ---------------------------------------------------------------------------
# Block 2 — Reliability per quantile
# ---------------------------------------------------------------------------

def block_reliability(y: np.ndarray, q: np.ndarray) -> dict:
    """For each nominal quantile level p, empirical P(y ≤ q_p)."""
    out: dict = {}
    for name, mask in [("global", np.ones(len(y), dtype=bool)),
                       ("wet",    y >= WET_THRESHOLD_MM)]:
        yy = y[mask]; qq = q[mask]
        emp = (yy[:, None] <= qq).mean(axis=0)   # (N_levels,)
        # NB: for ties (q_p=0 with y=0) the inequality `y<=q` is True so
        # empirical coverage may be > nominal in the lower tail. That is
        # the *correct* interpretation for a discrete predictive distribution.
        dev = emp - QUANTILE_LEVELS
        out[name] = {
            "n": int(mask.sum()),
            "nominal":   QUANTILE_LEVELS.tolist(),
            "empirical": emp.tolist(),
            "deviation": dev.tolist(),
            "max_abs_deviation": float(np.abs(dev).max()),
            "rms_deviation":     float(np.sqrt((dev ** 2).mean())),
        }
    return out


# ---------------------------------------------------------------------------
# Block 3 — Sharpness
# ---------------------------------------------------------------------------

def block_sharpness(q: np.ndarray, y: np.ndarray) -> dict:
    """Distribution of interval widths."""
    LO_HI = [("q95_minus_q05", 0, 10),
             ("q90_minus_q10", 1, 9),
             ("q80_minus_q20", 2, 8),
             ("q60_minus_q40", 4, 6)]
    out: dict = {}
    for name, mask in [("global", np.ones(len(y), dtype=bool)),
                       ("wet",    y >= WET_THRESHOLD_MM)]:
        qm = q[mask]
        out[name] = {"n": int(mask.sum())}
        for label, i_lo, i_hi in LO_HI:
            w = qm[:, i_hi] - qm[:, i_lo]
            out[name][label] = {
                "mean":   float(w.mean()),
                "median": float(np.median(w)),
                "p25":    float(np.percentile(w, 25)),
                "p75":    float(np.percentile(w, 75)),
                "p95":    float(np.percentile(w, 95)),
                "max":    float(w.max()),
            }
    return out


# ---------------------------------------------------------------------------
# Block 4 — Spread–skill
# ---------------------------------------------------------------------------

def block_spread_skill(y: np.ndarray, mean: np.ndarray, q: np.ndarray,
                       n_bins: int = 20) -> dict:
    """Bin by predicted mean; per bin: RMSE(mean, y) vs σ̂.

    σ̂ from q90–q10 ≈ 2.5631·σ for Gaussian (closest defensible proxy).
    For a perfectly calibrated forecast, σ̂ ≈ RMSE per bin.
    """
    spread = (q[:, 9] - q[:, 1]) / (2.0 * 1.2816)   # q90-q10 to gaussian σ
    # Bin by quantile of predicted mean (so each bin has ~n/20 rows)
    edges = np.unique(np.quantile(mean, np.linspace(0, 1, n_bins + 1)))
    idx   = np.digitize(mean, edges[1:-1])  # 0..n_bins-1
    out = {
        "bin_edges_mean_mm":   edges.tolist(),
        "n_per_bin":           [],
        "mean_mm_per_bin":     [],
        "rmse_per_bin":        [],
        "spread_per_bin":      [],
        "ratio_spread_rmse":   [],
    }
    for b in range(int(idx.max()) + 1):
        m = idx == b
        if m.sum() < 10:
            continue
        rmse = float(np.sqrt(((y[m] - mean[m]) ** 2).mean()))
        sp   = float(spread[m].mean())
        out["n_per_bin"].append(int(m.sum()))
        out["mean_mm_per_bin"].append(float(mean[m].mean()))
        out["rmse_per_bin"].append(rmse)
        out["spread_per_bin"].append(sp)
        out["ratio_spread_rmse"].append(sp / rmse if rmse > 0 else None)
    # Aggregate ratio (RMSE-weighted)
    if out["ratio_spread_rmse"]:
        valid = [r for r in out["ratio_spread_rmse"] if r is not None]
        out["ratio_mean"] = float(np.mean(valid))
    else:
        out["ratio_mean"] = None
    return out


# ---------------------------------------------------------------------------
# Block 5 — Hersbach CRPS decomposition
# ---------------------------------------------------------------------------

def block_hersbach(y: np.ndarray, q: np.ndarray) -> dict:
    """Hersbach (2000) decomposition adapted to fixed quantile levels.

    For each row with 11 quantiles (q_1 ≤ … ≤ q_11) at levels p_1..p_11
    and observation y:

    * tail bin 0 (-∞, q_1)  : β_0 = max(0, q_1 - y), α_0 = 0
    * tail bin N (q_N, ∞)   : α_N = max(0, y - q_N), β_N = 0
    * interior bins k = 1..N-1, [q_k, q_{k+1}):
        - y < q_k         : α=0,   β=q_{k+1}-q_k
        - y > q_{k+1}     : α=q_{k+1}-q_k, β=0
        - else            : α=y-q_k, β=q_{k+1}-y

    Per-row CRPS = β_0 + α_N + Σ_k (p_k² α_k + (1-p_k)² β_k).

    Decomposition:
        Reliability = Σ_k (ᾱ_k + β̄_k) (p_k - o_k)²,  o_k = β̄_k/(ᾱ_k+β̄_k)
        CRPS_pot    = Σ_k (ᾱ_k + β̄_k) o_k (1 - o_k)
        Total      = β̄_0 + ᾱ_N + Reliability + CRPS_pot
    """
    q = np.sort(q, axis=1)                  # safety
    n_rows, N = q.shape
    levels = QUANTILE_LEVELS

    # Tail bins
    beta_0  = np.maximum(0.0, q[:, 0]  - y)            # (n,)
    alpha_N = np.maximum(0.0, y - q[:, -1])            # (n,)

    # Interior bins — vectorise over rows + bins
    qk   = q[:, :-1]                                   # (n, N-1)
    qk1  = q[:, 1:]                                    # (n, N-1)
    w    = qk1 - qk
    y_b  = y[:, None]
    above = y_b > qk1
    below = y_b < qk
    inside = ~(above | below)
    alpha = np.where(above, w, np.where(inside, y_b - qk, 0.0))
    beta  = np.where(below, w, np.where(inside, qk1 - y_b, 0.0))

    # Per-row CRPS
    p2  = (levels[:-1] ** 2)[None, :]                  # uses left level p_k
    om2 = ((1 - levels[:-1]) ** 2)[None, :]
    crps_per_row = beta_0 + alpha_N + (p2 * alpha + om2 * beta).sum(axis=1)

    # Averages over rows
    alpha_bar = alpha.mean(axis=0)                     # (N-1,)
    beta_bar  = beta.mean(axis=0)
    w_bar     = alpha_bar + beta_bar                   # bin mass
    safe      = np.where(w_bar > 1e-12, w_bar, 1.0)
    o_k       = np.where(w_bar > 1e-12, beta_bar / safe, 0.0)

    reli      = np.sum(w_bar * (levels[:-1] - o_k) ** 2)
    crps_pot  = np.sum(w_bar * o_k * (1.0 - o_k))
    tail_mean = float(beta_0.mean() + alpha_N.mean())
    total     = float(reli + crps_pot + tail_mean)

    return {
        "per_bin": {
            "left_level":           levels[:-1].tolist(),
            "alpha_mean":           alpha_bar.tolist(),
            "beta_mean":            beta_bar.tolist(),
            "bin_mass":             w_bar.tolist(),
            "empirical_o":          o_k.tolist(),
        },
        "reliability":              float(reli),
        "crps_pot":                 float(crps_pot),
        "tail_mean":                tail_mean,
        "crps_total_recomputed":    total,
        "crps_per_row_mean":        float(crps_per_row.mean()),
    }, crps_per_row


# ---------------------------------------------------------------------------
# Block 6 — CRPSS vs climatology
# ---------------------------------------------------------------------------

def _crps_empirical_vec(y: np.ndarray, z_sorted: np.ndarray) -> np.ndarray:
    """Vectorised CRPS of an empirical CDF for many y at once.

    Given sorted training samples z (1D), return per-row CRPS for each
    observation in ``y``.
    Uses the standard ``CRPS = E[|X−y|] − ½ E[|X−X'|]`` decomposition,
    where X, X' are iid draws from z's empirical distribution.
    """
    N = z_sorted.size
    S = np.concatenate([[0.0], np.cumsum(z_sorted.astype(np.float64))])  # (N+1,)
    Stot = S[-1]
    # second term: ½ E[|X−X'|] = (1/N²) Σ_k z_(k) (2k − N − 1)
    k = np.arange(1, N + 1, dtype=np.float64)
    A = float((z_sorted.astype(np.float64) * (2 * k - N - 1)).sum() / (N * N))

    m  = np.searchsorted(z_sorted, y, side="right")           # (n_y,)
    Sm = S[m]
    e1 = ((2 * m - N) * y + Stot - 2 * Sm) / N
    return e1 - A


def block_crpss(y: np.ndarray, stations: np.ndarray,
                test_coords: dict[str, tuple[float, float]],
                train_path: Path,
                crps_model_mean: float) -> dict:
    """CRPS skill score against two climatology baselines.

    (a) Global pooled climatology — empirical CDF of ALL train rainfall.
    (b) 5-NN station pooled climatology — per test station, pool train
        observations from its 5 nearest train stations into a single
        empirical CDF.
    """
    print("[crpss] loading train rainfall + station coords")
    train = pd.read_parquet(train_path,
                            columns=["station_id", "latitude", "longitude",
                                     "rainfall"])
    print(f"[crpss] train rows: {len(train):,}  "
          f"stations: {train['station_id'].nunique()}")

    # ---- (a) global ----
    z_glob = np.sort(train["rainfall"].to_numpy(dtype=np.float64))
    crps_glob = _crps_empirical_vec(y, z_glob)
    out = {
        "crps_model_mean":        float(crps_model_mean),
        "global_pooled": {
            "n_train_samples":    int(z_glob.size),
            "crps_climatology":   float(crps_glob.mean()),
            "skill_score":        float(1.0 - crps_model_mean / crps_glob.mean()),
        },
    }

    # ---- (b) 5-NN per test station ----
    print(f"[crpss] building 5-NN climatology over {len(test_coords)} test stations")
    train_stn = (train.groupby("station_id")
                 .agg(lat=("latitude", "first"),
                      lon=("longitude", "first")).reset_index())
    tree = cKDTree(train_stn[["lon", "lat"]].to_numpy())

    # Pre-index train rainfall by station for fast pooling
    rain_by_stn: dict[str, np.ndarray] = {
        sid: g["rainfall"].to_numpy(dtype=np.float64)
        for sid, g in train.groupby("station_id")
    }
    train_stations = train_stn["station_id"].to_numpy()

    crps_5nn_per_row = np.empty_like(y, dtype=np.float64)
    # Loop over test stations (394) and apply to their rows
    test_station_to_rows: dict[str, np.ndarray] = {}
    for i, sid in enumerate(stations):
        test_station_to_rows.setdefault(sid, []).append(i)
    test_station_to_rows = {k: np.asarray(v) for k, v in test_station_to_rows.items()}

    for sid, row_idx in test_station_to_rows.items():
        lon, lat = test_coords[sid]
        _, nn_ix = tree.query([lon, lat], k=CLIM_KNN)
        if np.isscalar(nn_ix):
            nn_ix = np.array([nn_ix])
        pooled = np.concatenate([rain_by_stn[train_stations[k]] for k in nn_ix])
        pooled.sort()
        crps_5nn_per_row[row_idx] = _crps_empirical_vec(y[row_idx], pooled)

    out["nn5_station_pooled"] = {
        "k":                      CLIM_KNN,
        "crps_climatology":       float(crps_5nn_per_row.mean()),
        "skill_score":            float(1.0 - crps_model_mean
                                        / crps_5nn_per_row.mean()),
    }
    return out


# ---------------------------------------------------------------------------
# Block 7 — Conditional CRPS by intensity
# ---------------------------------------------------------------------------

def block_conditional_crps(y: np.ndarray, crps_row: np.ndarray) -> dict:
    bins = INTENSITY_BINS
    idx  = np.digitize(y, bins[1:-1])
    out  = {"bin_edges_mm": bins[:-1] + [float("inf")],
            "bin_names":    INTENSITY_NAMES,
            "per_bin":      []}
    for b, name in enumerate(INTENSITY_NAMES):
        m = idx == b
        out["per_bin"].append({
            "name":          name,
            "n":             int(m.sum()),
            "crps_mean":     float(crps_row[m].mean()) if m.any() else None,
            "y_mean":        float(y[m].mean())        if m.any() else None,
        })
    return out


# ---------------------------------------------------------------------------
# Block 8 — Wet-detector Brier decomposition
# ---------------------------------------------------------------------------

def _pwet_from_quantiles(q: np.ndarray, thr: float = WET_THRESHOLD_MM) -> np.ndarray:
    """P(Y ≥ thr) interpolated from the quantile CDF."""
    n = q.shape[0]
    p = np.empty(n, dtype=np.float64)
    for i in range(n):
        # F(thr) by linear interp of (q_p, p)
        ft = np.interp(thr, q[i], QUANTILE_LEVELS, left=0.0, right=1.0)
        p[i] = 1.0 - ft
    return p


def block_brier_decomposition(y: np.ndarray, q: np.ndarray,
                              n_bins: int = 10) -> dict:
    """Brier-score reliability/resolution/uncertainty for P(Y ≥ 0.5).

    Murphy (1973) decomposition::

        BS = REL − RES + UNC
        REL = Σ_k (n_k/N) (p̄_k − ō_k)²        (≥ 0, want small)
        RES = Σ_k (n_k/N) (ō_k − ō)²            (≥ 0, want large)
        UNC = ō (1 − ō)                          (climatological baseline)
    """
    pwet = _pwet_from_quantiles(q)
    wet  = (y >= WET_THRESHOLD_MM).astype(np.float64)
    bs   = float(((pwet - wet) ** 2).mean())

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx   = np.clip(np.digitize(pwet, edges[1:-1]), 0, n_bins - 1)
    o_bar = wet.mean()
    rel = 0.0
    res = 0.0
    per_bin = []
    for b in range(n_bins):
        m = idx == b
        n_k = int(m.sum())
        if n_k == 0:
            per_bin.append({"bin": [float(edges[b]), float(edges[b+1])],
                            "n": 0})
            continue
        p_bar_k = float(pwet[m].mean())
        o_bar_k = float(wet[m].mean())
        rel += n_k / len(y) * (p_bar_k - o_bar_k) ** 2
        res += n_k / len(y) * (o_bar_k - o_bar) ** 2
        per_bin.append({
            "bin":   [float(edges[b]), float(edges[b+1])],
            "n":     n_k,
            "p_mean": p_bar_k,
            "o_mean": o_bar_k,
        })
    unc = float(o_bar * (1 - o_bar))
    return {
        "brier_score":   bs,
        "reliability":   float(rel),
        "resolution":    float(res),
        "uncertainty":   unc,
        "decomposition_residual": bs - (float(rel) - float(res) + unc),
        "obs_wet_rate":  float(o_bar),
        "per_bin":       per_bin,
        "n_bins":        n_bins,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def compute(run_name: str) -> Path:
    t_total = time.time()
    paths = _ensure_run_artifacts(run_name)
    train_path, test_path = _ensure_clim_data()

    cfg     = json.loads(paths["config.json"].read_text())
    metrics = json.loads(paths["metrics.json"].read_text())
    print(f"[load] preds.parquet")
    preds = pd.read_parquet(paths["preds.parquet"])
    print(f"       rows={len(preds):,}  stations={preds['station_id'].nunique()}")

    y   = preds["observed_mm"].to_numpy(dtype=np.float64)
    mu  = preds["mean_mm"].to_numpy(dtype=np.float64)
    q   = preds[QUANTILE_COLS].to_numpy(dtype=np.float64)
    sid = preds["station_id"].to_numpy()

    # Test station coords (need for 5-NN climatology)
    print(f"[load] test station coords from fold0_test.parquet")
    test_meta = (pd.read_parquet(test_path,
                                 columns=["station_id", "latitude", "longitude"])
                 .drop_duplicates("station_id"))
    test_coords = {row.station_id: (row.longitude, row.latitude)
                   for row in test_meta.itertuples()}

    # ------- Diagnostics -------
    out = {
        "run_name":          run_name,
        "config":            cfg,
        "n_total":           int(len(y)),
        "n_wet":             int((y >= WET_THRESHOLD_MM).sum()),
        "n_dry":             int((y <  WET_THRESHOLD_MM).sum()),
        "wet_threshold_mm":  WET_THRESHOLD_MM,
        "quantile_levels":   QUANTILE_LEVELS.tolist(),
    }

    print("[1/8] PIT histogram + KS"); t = time.time()
    out["pit"]            = block_pit(y, q);                  print(f"       {time.time()-t:.1f}s")
    print("[2/8] reliability per quantile"); t = time.time()
    out["reliability"]    = block_reliability(y, q);          print(f"       {time.time()-t:.1f}s")
    print("[3/8] sharpness"); t = time.time()
    out["sharpness"]      = block_sharpness(q, y);            print(f"       {time.time()-t:.1f}s")
    print("[4/8] spread-skill"); t = time.time()
    out["spread_skill"]   = block_spread_skill(y, mu, q);     print(f"       {time.time()-t:.1f}s")
    print("[5/8] Hersbach CRPS decomposition"); t = time.time()
    hersbach, crps_row    = block_hersbach(y, q);             print(f"       {time.time()-t:.1f}s")
    out["hersbach"]       = hersbach
    print("[6/8] CRPSS vs climatology"); t = time.time()
    out["crpss"]          = block_crpss(y, sid, test_coords,
                                        train_path,
                                        crps_model_mean=float(crps_row.mean()))
    print(f"       {time.time()-t:.1f}s")
    print("[7/8] conditional CRPS by intensity"); t = time.time()
    out["conditional_crps"] = block_conditional_crps(y, crps_row); print(f"       {time.time()-t:.1f}s")
    print("[8/8] Brier decomposition for wet detector"); t = time.time()
    out["brier_wet"]      = block_brier_decomposition(y, q);   print(f"       {time.time()-t:.1f}s")

    # ------- Save + upload -------
    local_dir = LOCAL_RUNS / run_name
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / "uncertainty.json"
    local_path.write_text(json.dumps(out, indent=2))
    print(f"[save] {local_path}  ({local_path.stat().st_size/1024:.1f} KB)")

    s3 = boto3.client("s3", region_name="eu-north-1")
    s3_key = f"{S3_RUNS_ROOT}/{run_name}/uncertainty.json"
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    print(f"[s3]   s3://{S3_BUCKET}/{s3_key}")

    print(f"[done] total {time.time() - t_total:.1f}s")
    return local_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name", help="bayesnf run name "
                    "(e.g. vi__WY_h1_10__ffrk_full_32_5e-3_kl0.1_s5)")
    args = ap.parse_args()
    compute(args.run_name)


if __name__ == "__main__":
    main()
