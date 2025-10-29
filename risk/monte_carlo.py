from __future__ import annotations
import numpy as np
import pandas as pd


def max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    dd = (series / peak - 1.0).min()
    return float(dd)


def simulate_gbm(port_ret: pd.Series, years: int, n_paths: int, dt_per_year: int = 252, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mu = port_ret.mean() * dt_per_year
    sigma = port_ret.std(ddof=1) * np.sqrt(dt_per_year)
    steps = years * dt_per_year
    drift = (mu - 0.5 * sigma ** 2) / dt_per_year
    vol = sigma / np.sqrt(dt_per_year)
    z = rng.standard_normal((steps, n_paths))
    log_paths = drift + vol * z
    cum_log = log_paths.cumsum(axis=0)
    factors = np.exp(cum_log)
    df = pd.DataFrame(factors, index=range(steps))
    df = pd.concat([pd.Series(1.0, index=[-1]), (1.0 * df).iloc[:, :]], axis=0)
    return df


def simulate_bootstrap(weights: dict, returns: pd.DataFrame, years: int, n_paths: int, dt_per_year: int = 252, block: int = 21, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = years * dt_per_year
    blocks = []
    R = returns.values
    n = len(returns)
    for _ in range(int(np.ceil(steps / block))):
        i = rng.integers(0, max(1, n - block))
        blocks.append(R[i:i+block])
    sim = np.vstack(blocks)[:steps]
    w = np.array([weights[t] for t in returns.columns], dtype=float)
    port_daily = sim @ w
    factors = (1.0 + port_daily).cumprod()
    df = pd.DataFrame(factors, index=range(steps))
    df = pd.concat([pd.Series(1.0, index=[-1]), df], axis=0)
    return df


def mc_summary(factors_df: pd.DataFrame) -> dict:
    final = factors_df.iloc[-1].values
    p5, p50, p95 = np.percentile(final, [5, 50, 95])
    mdds = []
    for col in factors_df.columns:
        ser = factors_df[col]
        mdds.append(max_drawdown(ser))
    mdd_med = float(np.median(mdds)) if mdds else float("nan")
    prob_below_1 = float((final < 1.0).mean())
    return {"p5": p5, "p50": p50, "p95": p95, "mdd_med": mdd_med, "prob_below_1": prob_below_1}


