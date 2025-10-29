from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize

from .metrics import portfolio_stats

def _project_simplex(w: np.ndarray) -> np.ndarray:
    """
    Project weights onto the probability simplex: sum(w)=1, w>=0.
    This guards against minor numerical violations from the optimizer.
    """
    w = np.maximum(w, 0)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)

def min_variance(mu: np.ndarray, cov: np.ndarray, bounds: Optional[Tuple[float, float]] = (0.0, 1.0)) -> np.ndarray:
    n = len(mu)
    x0 = np.ones(n) / n

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]
    bnds = [bounds] * n if bounds else None

    def obj(w):
        return w @ cov @ w

    res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons)
    w = res.x if res.success else x0
    return _project_simplex(w)

def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, bounds: Optional[Tuple[float, float]] = (0.0, 1.0)) -> np.ndarray:
    n = len(mu)
    x0 = np.ones(n) / n

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [bounds] * n if bounds else None

    def neg_sharpe(w):
        ret, vol, sharpe = portfolio_stats(w, mu, cov, rf)
        return -sharpe

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bnds, constraints=cons)
    w = res.x if res.success else x0
    return _project_simplex(w)

def target_volatility(mu: np.ndarray, cov: np.ndarray, target_vol: float, rf: float = 0.0,
                      bounds: Optional[Tuple[float, float]] = (0.0, 1.0)) -> np.ndarray:
    """
    Locate a portfolio on the efficient frontier with the requested annual volatility.
    """
    n = len(mu)
    x0 = np.ones(n) / n

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [bounds] * n if bounds else None

    def obj(w):
        ret, vol, _ = portfolio_stats(w, mu, cov, rf)
        # Penalize deviation from target volatility with a light incentive for higher return.
        return (vol - target_vol) ** 2 - 1e-4 * ret

    res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons)
    w = res.x if res.success else x0
    return _project_simplex(w)
