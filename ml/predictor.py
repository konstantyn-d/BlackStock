from __future__ import annotations
import numpy as np
from typing import Any, Optional


class MeanReturnPredictor:
    """
    Minimal placeholder model: returns the provided annual means as-is.
    API-compatible with .fit()/.predict() for a future real model replacement.
    """

    def __init__(self) -> None:
        self._is_fitted: bool = False

    def fit(self, X: Optional[Any], y: Optional[Any]) -> "MeanReturnPredictor":
        self._is_fitted = True
        return self

    def predict(self, mu_annual: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            # Our CLI calls .fit() before .predict(), but guard for safety.
            self._is_fitted = True
        return np.asarray(mu_annual, dtype=float)


