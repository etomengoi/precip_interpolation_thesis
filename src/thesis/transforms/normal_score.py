"""Normal-score (Gaussian anamorphosis) transform for precipitation quotas.

Maps wet-day quotas to N(0,1) via the empirical CDF + probit, guaranteeing
Gaussian marginals for kriging (Cecinati et al. 2017 WRR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri


class NormalScoreTransform:
    """Maps precip_quota → N(0,1) via empirical CDF (Blom positions) + probit."""

    def __init__(self) -> None:
        # Sorted wet-day quota values from training; used as empirical CDF
        self._sorted_vals: np.ndarray | None = None
        self._memmap_path: str | None = None

    # --- memmap for zero-copy multiprocessing --------------------------

    def enable_memmap(self, path: str) -> None:
        """Save _sorted_vals to .npy for zero-copy shared access across workers."""
        if self._sorted_vals is None:
            return
        np.save(path, self._sorted_vals)
        self._memmap_path = path

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._memmap_path is not None:
            state["_sorted_vals"] = None  # drop the 174 MB array
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._memmap_path is not None:
            self._sorted_vals = np.load(self._memmap_path, mmap_mode="r")

    def fit(self, df: pd.DataFrame) -> "NormalScoreTransform":
        """Learn empirical CDF from wet-day quota values."""
        if "precip_quota" not in df.columns:
            raise ValueError(
                "NormalScoreTransform requires precip_quota column; "
                "apply DetrendTransform before NormalScoreTransform."
            )
        if "rain_indicator" not in df.columns:
            raise RuntimeError(
                "NormalScoreTransform requires rain_indicator column; "
                "apply IndicatorTransform before NormalScoreTransform."
            )

        wet_df = df[df["rain_indicator"] == 1]
        vals = wet_df["precip_quota"].dropna().values
        if len(vals) == 0:
            raise ValueError("No wet-day quota values found in training data.")
        self._sorted_vals = np.sort(vals)
        return self

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add precip_normal_score column (dry days → sentinel ≈ -4.75)."""
        assert self._sorted_vals is not None, "Call fit() first."
        df = df.copy()

        n = len(self._sorted_vals)
        quota = df["precip_quota"].values

        # Interpolate empirical CDF: fraction of training values ≤ quota
        ranks = np.searchsorted(self._sorted_vals, quota, side="right")
        # Blom plotting position
        probs = (ranks - 0.375) / (n + 0.25)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)

        scores = ndtri(probs)

        # Dry days → sentinel (will be masked during kriging)
        dry_sentinel = float(ndtri(1e-6))
        if "rain_indicator" in df.columns:
            scores = np.where(df["rain_indicator"].values == 0, dry_sentinel, scores)

        df["precip_normal_score"] = scores
        return df

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recover precip_quota via inverse CDF with GSLIB tail extrapolation."""
        assert self._sorted_vals is not None, "Pipeline not fitted."
        df = df.copy()

        if "precip_normal_score" not in df.columns:
            return df

        from thesis.transforms.kriging_transform import gslib_inverse_nst

        df["precip_quota"] = gslib_inverse_nst(
            df["precip_normal_score"].values, self._sorted_vals,
        )
        return df
