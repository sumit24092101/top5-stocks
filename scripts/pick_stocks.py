# scripts/pick_stocks.py
# Robust daily picker for NSE (.NS) that ALWAYS returns 5 picks.
# Handles Yahoo multi-index columns safely.

import json, datetime, time
from pathlib import Path
import pandas as pd
import yfinance as yf

UNIVERSE_FILE = Path(__file__).parent / "tickers_nifty.txt"
OUTFILE = Path(__file__).parents[1] / "public" / "data" / "latest.json"

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / (down.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def fetch_history(tick, retries=2):
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

def normalize_ohlcv(df: pd.DataFrame):
    """Return (close_series, volume_series) regardless of column structure."""
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
        vol = df["Volume"]
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        if isinstance(vol, pd.DataFrame): vol = vol.iloc[:, 0]
    else:
        # Some downloads name 'Adj Close' only; prefer Close if present
        if "Close" in df.columns:
            close = df["Close"]
        elif "Adj Close" in df.columns:
            close = df["Adj Close"]
        else:
            return None, None
        vol = df["Volume"] if "Volume" in df.columns else None
    if vol is None:
        return None, None
    return close.astype(float), vol.astype(float)

def score_row(df):
    close, vol = normalize_ohlcv(df)
    if close is None or vol is None or len(close) < 25:
        return None

    last_close = float(close.iloc[-1])
    # Momentum: 5-day return
    if len(close) >= 6 and float(close.iloc[-6]) != 0.0:
        ret5 = (last_close / float(close.iloc[-6])) - 1.0
    else:
        ret5 = 0.0

    vol20 = vol.rolling(20).mean().iloc[-1]
    vol_surge = float(vol.iloc[-1]) / float(vol20) if pd.notna(vol20) and vol20 > 0 else 1.0

    rsi14_series = rsi(close, 14)
    rsi14 = float(rsi14_series.iloc[-1]) if pd.notna(rsi14_series.iloc[-1]) else 50.0

    rsi_penalty = 0.0
    if rsi14 >
