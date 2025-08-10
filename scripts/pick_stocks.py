# scripts/pick_stocks.py
# Robust daily picker for NSE (.NS) that ALWAYS returns 5 picks.

import json, datetime, time
from pathlib import Path

import pandas as pd
import yfinance as yf

UNIVERSE_FILE = Path(__file__).parent / "tickers_nifty.txt"
OUTFILE = Path(__file__).parents[1] / "public" / "data" / "latest.json"

# ---------- helpers ----------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / (down.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def fetch_history(tick: str, retries: int = 2) -> pd.DataFrame:
    for attempt in range(retries + 1):
        try:
            df = yf.download(
                tick, period="6mo", interval="1d",
                auto_adjust=True, progress=False, threads=False
            )
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(1 + attempt)
    return pd.DataFrame()

def normalize_close_vol(df: pd.DataFrame):
    """Return (close_series, volume_series) regardless of Yahoo's column shape."""
    if df.empty:
        return None, None
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
        vol = df["Volume"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
    else:
        close = df["Close"] if "Close" in df.columns else df.get("Adj Close")
        vol = df.get("Volume")
    if close is None or vol is None:
        ret
