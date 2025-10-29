from __future__ import annotations
import os
import pathlib
import datetime as dt
from typing import List

import pandas as pd
import yfinance as yf

from config import CONFIG
from utils.logging import get_logger

log = get_logger(__name__)

CACHE_DIR = pathlib.Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

def _cache_path(ticker: str, start: str, end: str) -> pathlib.Path:
    safe = f"{ticker}_{start}_{end}".replace(":", "-")
    return CACHE_DIR / f"{safe}.parquet"

def load_one(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.Series:
    path = _cache_path(ticker, start, end)
    if use_cache and path.exists():
        s = pd.read_parquet(path)
        s.name = ticker
        log.info(f"[CACHE] {ticker} → {len(s)} rows")
        return s

    log.info(f"[YF] download {ticker} {start}..{end}")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    s = df["Close"].astype("float64")
    s.name = ticker
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_parquet(path)
    return s

def load_prices(tickers: List[str], start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    series = [load_one(t.strip().upper(), start, end, use_cache) for t in tickers]
    prices = pd.concat(series, axis=1).dropna(how="any")
    prices.index.name = "date"
    return prices
