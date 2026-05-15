"""Convert internal variogram params {nugget, psill, range} to PyKrige format."""
from __future__ import annotations

_RANGE_MULTIPLIER: dict[str, float] = {
    "exponential": 3.0,
    "spherical": 1.0,
    "gaussian": 7.0 / 4.0,
}


def to_pykrige_params(params_dict: dict, model: str) -> dict[str, float]:
    """Convert to PyKrige dict format (sill/range/nugget) — fixes params, no re-fitting."""
    mult = _RANGE_MULTIPLIER.get(model)
    if mult is None:
        raise ValueError(f"Unknown variogram model: {model!r}")

    psill = float(params_dict["psill"])
    nugget = float(params_dict["nugget"])
    a = float(params_dict["range"])

    return {"sill": psill + nugget, "range": a * mult, "nugget": nugget}


def to_pykrige_params_list(params_dict: dict, model: str) -> list[float]:
    """Convert to PyKrige [psill, range, nugget] list — for variogram functions only."""
    mult = _RANGE_MULTIPLIER.get(model)
    if mult is None:
        raise ValueError(f"Unknown variogram model: {model!r}")

    psill = float(params_dict["psill"])
    nugget = float(params_dict["nugget"])
    a = float(params_dict["range"])

    return [psill, a * mult, nugget]
