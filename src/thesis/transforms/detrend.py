"""Monthly-total detrending: precip_quota = precip_mm / monthly_total.

Removes seasonal amplitude (Haylock et al. 2008, Hofstra et al. 2008).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class DetrendTransform:
    """Expresses daily precip as a fraction of the station's monthly total."""

    def __init__(self) -> None:
        # (station_id, month) → mean monthly total [mm]; populated by fit()
        self._monthly_totals: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "DetrendTransform":
        """Learn per-station monthly totals from the training DataFrame."""
        dt = pd.to_datetime(df["date"])
        month = dt.dt.month
        year = dt.dt.year

        if "rain_indicator" not in df.columns:
            raise RuntimeError(
                "rain_indicator column missing — apply IndicatorTransform before DetrendTransform."
            )
        mask = df["rain_indicator"] == 1

        wet = df.loc[mask]

        monthly = (
            wet.assign(_year=year[mask], _month=month[mask])
            .groupby(["station_id", "_year", "_month"])["precip_mm"]
            .sum()
            .reset_index()
        )
        self._monthly_totals = (
            monthly.groupby(["station_id", "_month"])["precip_mm"].mean()
        )
        return self

    def _lookup_totals(self, df: pd.DataFrame) -> np.ndarray:
        """Look up per-row monthly totals, falling back to the grand mean."""
        assert self._monthly_totals is not None
        grand_mean = float(self._monthly_totals.mean())
        month = pd.to_datetime(df["date"]).dt.month
        lookup = self._monthly_totals.rename("_monthly_total").reset_index()
        keys = pd.DataFrame({"station_id": df["station_id"].values, "_month": month.values})
        return (
            keys.merge(lookup, on=["station_id", "_month"], how="left")["_monthly_total"]
            .fillna(grand_mean)
            .values
        )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add precip_quota = precip_mm / monthly_total."""
        if self._monthly_totals is None:
            raise RuntimeError("Call fit() before apply().")
        df = df.copy()
        totals_arr = self._lookup_totals(df)
        # Stations with no wet-day history get NaN total → quota fills to 0
        safe_totals = np.where(totals_arr > 0, totals_arr, np.nan)
        df["precip_quota"] = df["precip_mm"].values / safe_totals
        df["precip_quota"] = df["precip_quota"].fillna(0.0)
        if "rain_indicator" in df.columns:
            df.loc[df["rain_indicator"] == 0, "precip_quota"] = 0.0
        return df

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recover precip_mm from precip_quota × monthly_total."""
        assert self._monthly_totals is not None, "Pipeline not fitted."
        df = df.copy()
        if "station_id" in df.columns and "date" in df.columns:
            df["precip_mm"] = df["precip_quota"] * self._lookup_totals(df)
        return df
