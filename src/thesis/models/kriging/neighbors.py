"""Effective range computation for local kriging neighborhood selection."""
from __future__ import annotations

import numpy as np


# Coefficients to convert variogram "range" parameter to practical effective
# range (the distance at which γ(h) ≈ 95 % of the sill).
_EFFECTIVE_RANGE_COEFFICIENTS: dict[str, float] = {
    "spherical":   1.0,     # exact sill at h = range
    "exponential": 3.0,     # solve 0.95 = 1 - exp(-h/a) → h = -ln(0.05)·a ≈ 3a
    "gaussian":    1.732,   # solve 0.95 = 1 - exp(-(h/a)²) → h = √(-ln(0.05))·a ≈ √3·a
}


def effective_range(params_dict: dict, model: str) -> float:
    """Return practical range in metres where γ(h) ≈ 95% of the sill."""
    r = float(params_dict["range"])
    coeff = _EFFECTIVE_RANGE_COEFFICIENTS[model]
    return r * coeff


