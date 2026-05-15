"""Covariance function for kriging — delegates to pykrige variogram models.

Single source of truth for both LOO cross-validation (loo_cv.py) and
production inference.

    C(0)   = sill = psill + nugget
    C(h>0) = sill - γ(h)

where γ(h) is computed by pykrige's variogram model functions.
"""
from __future__ import annotations

import numpy as np
from pykrige.variogram_models import (
    exponential_variogram_model,
    gaussian_variogram_model,
    spherical_variogram_model,
)

from thesis.models.kriging.pykrige_adapter import to_pykrige_params_list as to_pykrige_params

_PK_VGM_FN = {
    "spherical": spherical_variogram_model,
    "exponential": exponential_variogram_model,
    "gaussian": gaussian_variogram_model,
}


def apply_cov_nugget(dist: np.ndarray, vgm_info: dict) -> np.ndarray:
    """Covariance C(h) = sill - γ(h) with nugget, via pykrige variogram models."""
    model = vgm_info["model"]
    p = vgm_info["params_dict"]
    sill = float(p["nugget"]) + float(p["psill"])

    pk_params = to_pykrige_params(p, model)
    pk_fn = _PK_VGM_FN[model]

    # γ(h) from pykrige; C(h) = sill - γ(h) for h > 0, C(0) = sill
    gamma = pk_fn(pk_params, dist)
    cov = sill - gamma
    return np.where(dist == 0.0, sill, cov)
