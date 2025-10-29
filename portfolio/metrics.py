from __future__ import annotations
import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price levels into daily log-returns.
    Expects a DataFrame with tickers as columns and a DatetimeIndex.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas.DataFrame")
    returns = np.log(prices / prices.shift(1)).dropna(how="any")
    return returns.astype("float64")


def annualize_returns_daily(mu_daily: np.ndarray | pd.Series) -> np.ndarray:
    """
    Annualize mean daily returns by multiplying by 252 trading days.
    Returns a numpy.ndarray for downstream numeric routines.
    """
    arr = np.asarray(mu_daily, dtype=float)
    return arr * TRADING_DAYS_PER_YEAR


def annualize_cov_daily(cov_daily: np.ndarray | pd.DataFrame) -> np.ndarray:
    """
    Annualize daily covariance by multiplying by 252.
    Returns a numpy.ndarray.
    """
    arr = np.asarray(cov_daily, dtype=float)
    return arr * TRADING_DAYS_PER_YEAR


def portfolio_stats(w: np.ndarray,
                    mu_annual: np.ndarray,
                    cov_annual: np.ndarray,
                    rf: float = 0.0) -> tuple[float, float, float]:
    """
    Compute portfolio (annual_return, annual_volatility, sharpe).
    w: portfolio weights (sum ~ 1)
    mu_annual: annual expected returns per asset
    cov_annual: annual covariance of returns
    rf: annual risk-free rate
    """
    w = np.asarray(w, dtype=float)
    mu_annual = np.asarray(mu_annual, dtype=float)
    cov_annual = np.asarray(cov_annual, dtype=float)

    ann_return = float(w @ mu_annual)
    ann_vol = float(np.sqrt(w @ cov_annual @ w))
    excess = ann_return - rf
    sharpe = float(excess / ann_vol) if ann_vol > 0 else 0.0
    return ann_return, ann_vol, sharpe


