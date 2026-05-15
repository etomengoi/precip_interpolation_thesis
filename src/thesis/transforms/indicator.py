"""Binary rain indicator (wet/dry) for two-stage kriging (Haylock et al. 2008)."""
from __future__ import annotations

import pandas as pd


class IndicatorTransform:
    """Adds `rain_indicator` (1 = wet, 0 = dry) based on a threshold."""

    def __init__(self, threshold_mm: float = 0.5) -> None:
        self.threshold_mm = threshold_mm

    def fit(self, df: pd.DataFrame) -> "IndicatorTransform":
        return self  # stateless

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rain_indicator"] = (df["precip_mm"] >= self.threshold_mm).astype(int)
        return df

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=["rain_indicator"], errors="ignore")
