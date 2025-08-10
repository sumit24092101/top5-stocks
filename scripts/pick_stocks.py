# scripts/pick_stocks.py
# Robust daily picker for NSE (.NS) that ALWAYS returns 5 picks.
# Method: 5D momentum + volume surge + RSI sanity, with graceful fallbacks.

import json, datetime, math, time
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
    # 6 months ensures enough bars for 20d avg even with holidays.
    for attempt in range(retries + 1):
        try:
            df = yf.download(tick, period="6mo", interval="1d",
                             auto_adjust=True, progress=False, threads=False)
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(1 + attempt)  # tiny backoff
    return pd.DataFrame()

def score_row(df):
    if df.empty or len(df) < 25:
        return None
    df = df.copy()
    df["vol20"] = df["Volume"].rolling(20).mean()
    df["rsi14"] = rsi(df["Close"], 14)
    last = df.iloc[-1]

    # Momentum: 5-day close return (needs >=6 rows)
    if len(df) >= 6 and df["Close"].iloc[-6] != 0:
        ret5 = (last["Close"] / df["Close"].iloc[-6]) - 1
    else:
        ret5 = 0.0

    vol20 = last.get("vol20", float("nan"))
    if pd.notna(vol20) and vol20 > 0:
        vol_surge = last["Volume"] / vol20
    else:
        vol_surge = 1.0

    rsi14 = last.get("rsi14", float("nan"))
    if not pd.notna(rsi14):
        rsi14 = 50.0

    # Simple penalty to avoid very overbought/oversold
    rsi_penalty = 0.0
    if rsi14 > 72: rsi_penalty = -0.15
    if rsi14 < 35: rsi_penalty = -0.10

    score = 0.60 * ret5 + 0.30 * min(vol_surge / 3, 1.0) + 0.10 * (1 - abs(55 - rsi14) / 55) + rsi_penalty

    return {
        "close": round(float(last["Close"]), 2),
        "ret5": round(float(ret5), 4),
        "vol_surge": round(float(vol_surge), 2),
        "rsi14": round(float(rsi14), 1),
        "score": round(float(score), 4),
    }

def main():
    tickers = [t.strip() for t in UNIVERSE_FILE.read_text().splitlines()
               if t.strip() and not t.startswith("#")]

    results = []
    fallbacks = []  # keep simpler stats for fallback selection

    for t in tickers:
        df = fetch_history(t)
        if df.empty:
            continue
        row = score_row(df)
        if row:
            row["ticker"] = t
            results.append(row)

        # Keep a simpler version for fallback (even if <25 rows etc.)
        try:
            last = df.iloc[-1]
            if len(df) >= 6 and df["Close"].iloc[-6] != 0:
                ret5 = (last["Close"] / df["Close"].iloc[-6]) - 1
            else:
                ret5 = 0.0
            fallbacks.append({
                "ticker": t,
                "close": round(float(last["Close"]), 2),
                "ret5": round(float(ret5), 4),
            })
        except Exception:
            pass

    # Primary selection by score
    results.sort(key=lambda x: x["score"], reverse=True)
    picks = results[:5]

    # Fallback 1: if fewer than 5, fill with highest 5D return from fallback set not already picked
    if len(picks) < 5 and fallbacks:
        picked = {p["ticker"] for p in picks}
        rem = [r for r in sorted(fallbacks, key=lambda x: x["ret5"], reverse=True) if r["ticker"] not in picked]
        for r in rem:
            picks.append({
                "ticker": r["ticker"],
                "close": r["close"],
                "ret5": r["ret5"],
                "vol_surge": 1.0,
                "rsi14": 50.0,
                "score": round(0.60 * r["ret5"], 4)
            })
            if len(picks) == 5:
                break

    # Fallback 2: if still short (rare), pad with first tickers as neutral score
    i = 0
    while len(picks) < 5 and i < len(tickers):
        t = tickers[i]
        if all(p["ticker"] != t for p in picks):
            picks.append({
                "ticker": t,
                "close": None, "ret5": 0.0, "vol_surge": 1.0, "rsi14": 50.0, "score": 0.0
            })
        i += 1

    payload = {
        "as_of_ist": datetime.datetime.utcnow().isoformat() + "Z",
        "universe": "NIFTY50",
        "method": "momentum+volume+rsi (with safe fallbacks)",
        "picks": picks
    }
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    OUTFILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTFILE} with {len(picks)} picks.")

if __name__ == "__main__":
    main()
