"""TransformPipeline — composes multiple Transform steps in sequence."""
from __future__ import annotations

import pandas as pd

from thesis.transforms.protocols import Transform


class TransformPipeline:
    """Chains transforms — fitted and applied in order, inversed in reverse."""

    def __init__(self, transforms: list[Transform]) -> None:
        self._transforms = transforms
        self._fitted = False

    def add(self, transform: Transform) -> None:
        """Add a new transform to the end of the pipeline."""
        if self._fitted:
            raise RuntimeError("Cannot add transforms after fitting.")
        self._transforms.append(transform)

    def fit(self, df: pd.DataFrame) -> "TransformPipeline":
        """Fit each step sequentially, passing the transformed output forward."""
        current = df
        for t in self._transforms:
            t.fit(current)
            current = t.apply(current)
        self._fitted = True
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and return the transformed result without a second pass."""
        current = df
        for t in self._transforms:
            t.fit(current)
            current = t.apply(current)
        self._fitted = True
        return current

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all fitted transforms in order."""
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before apply().")
        current = df
        for t in self._transforms:
            current = t.apply(current)
        return current

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Invert transforms in reverse order to recover physical units."""
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before inverse().")
        current = df
        for t in reversed(self._transforms):
            current = t.inverse(current)
        return current

    def __repr__(self) -> str:
        steps = " → ".join(type(t).__name__ for t in self._transforms)
        status = "fitted" if self._fitted else "unfitted"
        return f"TransformPipeline({steps}) [{status}]"
