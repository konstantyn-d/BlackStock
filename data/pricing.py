from __future__ import annotations
import pandas as pd
import yfinance as yf


def last_close_from_window(prices: pd.DataFrame) -> pd.Series:
    return prices.iloc[-1].astype(float)


def live_last_prices(tickers: list[str]) -> pd.Series:
    tickers = [t.strip().upper() for t in tickers]
    try:
        df = yf.download(" ".join(tickers), period="1d", interval="1m", progress=False, auto_adjust=True, threads=True)
        if isinstance(df.columns, pd.MultiIndex):
            last = df["Close"].iloc[-1]
            last.name = None
            return last.astype(float)
        else:
            return pd.Series({tickers[0]: float(df["Close"].iloc[-1])})
    except Exception:
        pass

    data = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            p = tk.fast_info.last_price
            if p is None:
                p = tk.history(period="1d")["Close"].iloc[-1]
            data[t] = float(p)
        except Exception:
            data[t] = float("nan")
    return pd.Series(data)


