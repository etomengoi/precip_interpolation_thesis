"""Transform adapter: quota ↔ z-space via NormalScoreTransform or LogTransform.

NST inverse uses GSLIB tail extrapolation (Deutsch & Journel 1997, §V.1.6):
lower tail power model (ω=1), upper tail hyperbolic (ω=1.5), interior via
analytic Blom-inverse for O(1) index lookup.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

from thesis.transforms.normal_score import NormalScoreTransform
from thesis.transforms.log_transform import LogTransform

_KINDS = ("none", "log", "normal_score")
TRANSFORMS = _KINDS  # public: canonical list of supported transform names

# GSLIB tail extrapolation parameters (Deutsch & Journel 1997, pp. 135-138)
_LOWER_TAIL_OMEGA = 1.0    # power model exponent; 1 = linear (conservative)
_LOWER_TAIL_ZMIN  = 0.0    # precipitation quota is non-negative
_UPPER_TAIL_OMEGA = 1.5    # hyperbolic model exponent; GSLIB general-purpose


def gslib_inverse_nst(z: np.ndarray, sorted_vals: np.ndarray) -> np.ndarray:
    """Inverse NST with GSLIB tail extrapolation (Deutsch & Journel 1997, §V.1.6)."""
    n = len(sorted_vals)
    n_plus = n + 0.25

    # Blom plotting-position bounds
    p_min = 0.625 / n_plus          # (1 - 0.375) / (n + 0.25)
    p_max = (n - 0.375) / n_plus    # (n + 1 - 1 - 0.375) / (n + 0.25)

    cdf_vals = np.clip(ndtr(z), 1e-10, 1.0 - 1e-10)
    result = np.empty_like(z, dtype=np.float64)

    # Interior: analytic Blom-inverse → O(1) index per element
    interior = (cdf_vals >= p_min) & (cdf_vals <= p_max)
    if interior.any():
        # Inverse Blom formula: p = (i + 0.625) / (n + 0.25)  ⟹  i = p*(n+0.25) - 0.625
        idx_float = cdf_vals[interior] * n_plus - 0.625
        lo = np.clip(idx_float.astype(np.intp), 0, n - 2)
        hi = lo + 1
        frac = idx_float - lo
        result[interior] = sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])

    # Lower tail: power model (GSLIB Eq. V.10)
    lower = cdf_vals < p_min
    if lower.any():
        p_ratio = cdf_vals[lower] / p_min
        result[lower] = (
            _LOWER_TAIL_ZMIN
            + (sorted_vals[0] - _LOWER_TAIL_ZMIN)
            * np.power(p_ratio, 1.0 / _LOWER_TAIL_OMEGA)
        )

    # Upper tail: hyperbolic model (GSLIB Eq. V.11)
    upper = cdf_vals > p_max
    if upper.any():
        lam = sorted_vals[-1] ** _UPPER_TAIL_OMEGA * (1.0 - p_max)
        result[upper] = np.power(
            lam / (1.0 - cdf_vals[upper]),
            1.0 / _UPPER_TAIL_OMEGA,
        )

    return np.maximum(result, 0.0)


class KrigingTransform:
    """Wraps a single transform choice (none / log / normal_score) for kriging."""

    def __init__(
        self,
        kind: str,
        ns: NormalScoreTransform,
        log: LogTransform,
    ) -> None:
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
        self.kind = kind
        self._ns  = ns
        self._log = log

    # ------------------------------------------------------------------
    def fwd(self, quota: np.ndarray) -> np.ndarray:
        """Quota → z (kriging space)."""
        if self.kind == "none":
            return quota.copy()
        if self.kind == "log":
            return np.log(np.maximum(quota, 0.0) + self._log.offset)
        # normal_score
        n = len(self._ns._sorted_vals)
        ranks = np.searchsorted(self._ns._sorted_vals, quota, side="right")
        probs = np.clip((ranks - 0.375) / (n + 0.25), 1e-6, 1 - 1e-6)
        return ndtri(probs)

    def inv(self, z: np.ndarray) -> np.ndarray:
        """z (kriging space) → quota, with GSLIB tail extrapolation."""
        if self.kind == "none":
            return np.maximum(z, 0.0)
        if self.kind == "log":
            return np.maximum(np.exp(z) - self._log.offset, 0.0)
        return gslib_inverse_nst(z, self._ns._sorted_vals)

    def __repr__(self) -> str:
        return f"KrigingTransform(kind={self.kind!r})"
