"""Transform protocol — the contract every preprocessing step must satisfy."""
from typing import Protocol
import pandas as pd


class Transform(Protocol):
    """A stateful, invertible data transformation.

    The fit/apply split mirrors sklearn's fit/transform, but with an explicit
    inverse() so predictions can be back-transformed to physical units (mm).

    Usage:
        t = SomeTransform()
        train_df = t.fit(train_df).apply(train_df)
        test_df  = t.apply(test_df)
        pred_mm  = t.inverse(pred_log_residuals)
    """

    def fit(self, df: pd.DataFrame) -> "Transform":
        """Learn parameters from df (e.g. monthly means for detrending).
        Returns self so calls can be chained: t.fit(df).apply(df).
        """
        ...

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the (fitted) transformation; must not modify df in place."""
        ...

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Invert the transformation to recover physical units."""
        ...
