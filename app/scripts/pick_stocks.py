# scripts/pick_stocks.py
import json, datetime
import pandas as pd
import yfinance as yf
from pathlib import Path

UNIVERSE_FILE = Path(__file__).parent / "tickers_nifty.txt"
OUTFILE = Path(__file__).parents[1] / "public" / "data" / "latest.json"

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / (down.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def score_ticker(tick):
    try:
        df = yf.download(tick, period="3mo", auto_adjust=True, progress=False, interval="1d")
        if df.empty or len(df) < 25:
            return None
        df["vol20"] = df["Volume"].rolling(20).mean()
        df["rsi14"] = rsi(df["Close"], 14)
        last = df.iloc[-1]

        # Momentum: 5-day return
        ret5 = (last["Close"] / df["Close"].iloc[-6]) - 1 if len(df) >= 6 else 0
        # Volume surge vs 20d avg
        vol20 = last["vol20"] if pd.notna(last["vol20"]) else 0
        vol_surge = (last["Volume"] / vol20) if vol20 else 1
        # RSI sanity; prefer mid-range
        rsi14 = last["rsi14"] if pd.notna(last["rsi14"]) else 50

        rsi_penalty = 0.0
        if rsi14 > 72: rsi_penalty = -0.15
        if rsi14 < 35: rsi_penalty = -0.10

        score = 0.60*ret5 + 0.30*min(vol_surge/3, 1.0) + 0.10*(1 - abs(55 - rsi14)/55) + rsi_penalty
        return {
            "ticker": tick,
            "close": round(float(last["Close"]), 2),
            "ret5": round(float(ret5), 4),
            "vol_surge": round(float(vol_surge), 2),
            "rsi14": round(float(rsi14), 1),
            "score": round(float(score), 4)
        }
    except Exception:
        return None

def main():
    universe = [t.strip() for t in UNIVERSE_FILE.read_text().splitlines()
                if t.strip() and not t.startswith("#")]
    rows = []
    for t in universe:
        r = score_ticker(t)
        if r: rows.append(r)
    rows.sort(key=lambda x: x["score"], reverse=True)
    picks = rows[:5]
    payload = {
        "as_of_ist": datetime.datetime.utcnow().isoformat() + "Z",
        "universe": "NIFTY50",
        "method": "momentum+volume+rsi",
        "picks": picks
    }
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    OUTFILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTFILE} with {len(picks)} picks.")

if __name__ == "__main__":
    main()
