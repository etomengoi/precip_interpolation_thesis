"""Log-transformation for precipitation quotas: log(quota + offset)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class LogTransform:
    """Applies log(quota + offset) to wet-day precip_quota values."""

    def __init__(self, offset: float = 1e-4) -> None:
        self.offset = offset

    def fit(self, df: pd.DataFrame) -> "LogTransform":
        return self  # stateless

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        wet_df = df[df["rain_indicator"] == 1]
        df.loc[wet_df.index, "precip_log"] = np.log(wet_df["precip_quota"] + self.offset)
        return df

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "precip_log" in df.columns:
            df["precip_quota_pred"] = (np.exp(df["precip_log"]) - self.offset).clip(lower=0.0)
        return df
