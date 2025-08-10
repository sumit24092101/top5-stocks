# scripts/pick_stocks.py
# Ultra-stable NSE picker: always outputs 5 items, with graceful fallbacks.

import json, datetime, time
from pathlib import Path
import pandas as pd
import yfinance as yf

UNIVERSE_FILE = Path(__file__).parent / "tickers_nifty.txt"
OUTFILE = Path(__file__).parents[1] / "public" / "data" / "latest.json"

def fetch(ticker, retries=2):
    for i in range(retries + 1):
        try:
            df = yf.download(ticker, period="6mo", interval="1d",
                             auto_adjust=True, progress=False, threads=False)
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(1 + i)
    return pd.DataFrame()

def get_close_vol(df):
    if df.empty:
        return None, None
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
        vol = df["Volume"]
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        if isinstance(vol, pd.DataFrame): vol = vol.iloc[:, 0]
    else:
        close = df["Close"] if "Close" in df.columns else df.get("Adj Close")
        vol = df.get("Volume")
    if close is None or vol is None:
        return None, None
    return close.astype(float), vol.astype(float)

def main():
    tickers = [t.strip() for t in UNIVERSE_FILE.read_text().splitlines()
               if t.strip() and not t.startswith("#")]

    scored = []
    simple = []

    for t in tickers:
        df = fetch(t)
        if df.empty:
            continue
        close, vol = get_close_vol(df)
        if close is None or vol is None or len(close) == 0:
            continue

        last = float(close.iloc[-1])
        # 5-day return if possible
        ret5 = 0.0
        if len(close) >= 6 and float(close.iloc[-6]) != 0.0:
            ret5 = (last / float(close.iloc[-6])) - 1.0

        # volume surge vs 20d avg (safe)
        vol20 = vol.rolling(20).mean().iloc[-1] if len(vol) >= 20 else float("nan")
        if pd.notna(vol20) and vol20 > 0:
            vol_surge = float(vol.iloc[-1]) / float(vol20)
        else:
            vol_surge = 1.0

        # simple score: mostly momentum, tiny volume signal; no RSI to avoid edge cases
        score = 0.8 * ret5 + 0.2 * min(vol_surge / 3, 1.0)

        scored.append({
            "ticker": t,
            "close": round(last, 2),
            "ret5": round(ret5, 4),
            "vol_surge": round(vol_surge, 2),
            "rsi14": 50.0,          # placeholder, not used in score here
            "score": round(score, 4)
        })

        simple.append({"ticker": t, "close": round(last, 2), "ret5": round(ret5, 4)})

    # Primary: top 5 by score
    scored.sort(key=lambda x: x["score"], reverse=True)
    picks = scored[:5]

    # Fallback 1: fill with best 5D return from simple list
    if len(picks) < 5 and simple:
        chosen = {p["ticker"] for p in picks}
        for r in sorted(simple, key=lambda x: x["ret5"], reverse=True):
            if r["ticker"] in chosen:
                continue
            picks.append({
                "ticker": r["ticker"],
                "close": r["close"],
                "ret5": r["ret5"],
                "vol_surge": 1.0,
                "rsi14": 50.0,
                "score": round(0.8 * r["ret5"], 4)
            })
            if len(picks) == 5:
                break

    # Fallback 2: still short? pad with first tickers as neutral
    i = 0
    while len(picks) < 5 and i < len(tickers):
        t = tickers[i]
        if all(p["ticker"] != t for p in picks):
            picks.append({"ticker": t, "close": None, "ret5": 0.0, "vol_surge": 1.0, "rsi14": 50.0, "score": 0.0})
        i += 1

    payload = {
        "as_of_ist": datetime.datetime.utcnow().isoformat() + "Z",
        "universe": "NIFTY50",
        "method": "momentum + volume (stable fallback)",
        "picks": picks
    }
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    OUTFILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTFILE} with {len(picks)} picks.")

if __name__ == "__main__":
    main()
