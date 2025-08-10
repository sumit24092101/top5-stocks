# scripts/backtest.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

TOP_N = 5
COST = 0.001  # 0.10% per trade

TICKERS = [
    "ADANIPORTS.NS","ASIANPAINT.NS","AXISBANK.NS","BAJAJ-AUTO.NS","BAJFINANCE.NS",
    "BAJAJFINSV.NS","BHARTIARTL.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS","COALINDIA.NS",
    "DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS",
    "HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","ITC.NS",
    "INDUSINDBK.NS","INFY.NS","JSWSTEEL.NS","KOTAKBANK.NS","LTIM.NS","LT.NS","M&M.NS",
    "MARUTI.NS","NESTLEIND.NS","NTPC.NS","ONGC.NS","POWERGRID.NS","RELIANCE.NS","SBILIFE.NS",
    "SBIN.NS","SUNPHARMA.NS","TCS.NS","TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS",
    "TECHM.NS","TITAN.NS","ULTRACEMCO.NS","UPL.NS","WIPRO.NS"
]

def score_func(ret5, vol_surge):
    return 0.8 * ret5 + 0.2 * np.minimum(vol_surge / 3.0, 1.0)

def get_panel(data, field):
    cols = []
    for t in data.columns.levels[0]:
        if (t, field) in data.columns:
            cols.append(data[(t, field)].rename(t))
    return pd.concat(cols, axis=1).sort_index()

def metrics(eq):
    if isinstance(eq, pd.DataFrame): eq = eq.squeeze()
    if eq.empty or len(eq) < 2:
        return dict(CAGR=np.nan, MaxDD=np.nan, Sharpe=np.nan, Vol=np.nan)
    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float((eq.iloc[-1]/eq.iloc[0])**(1/years) - 1) if years>0 else np.nan
    maxdd = float((eq/eq.cummax() - 1).min())
    std = float(rets.std())
    sharpe = float(rets.mean()/std*np.sqrt(252)) if std>0 else np.nan
    vol = float(std*np.sqrt(252))
    return dict(CAGR=cagr, MaxDD=maxdd, Sharpe=sharpe, Vol=vol)

def slice_metrics(eq, years):
    cutoff = eq.index.max() - pd.Timedelta(days=int(365.25*years))
    return metrics(eq[eq.index >= cutoff])

def plot_curve(eq, bmk, years, path_png, title):
    cutoff = eq.index.max() - pd.Timedelta(days=int(365.25*years))
    plt.figure(figsize=(8,4))
    eq[eq.index >= cutoff].plot(label="Strategy")
    if not bmk.empty:
        bmk.loc[eq.index.min():eq.index.max()][bmk.index >= cutoff].plot(label="Benchmark")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(path_png); plt.close()

# ---------- Download data ----------
print("Downloading price data…")
px = yf.download(TICKERS, period="6y", interval="1d", auto_adjust=True, progress=False,
                 group_by="ticker", threads=True)
close = get_panel(px, "Close")
openp = get_panel(px, "Open")
vol = get_panel(px, "Volume")

# Signals at day t (close/volume)
ret5 = close / close.shift(5) - 1.0
vol20 = vol.rolling(20).mean()
vol_surge = (vol / vol20).replace([np.inf, -np.inf], np.nan)
score = score_func(ret5, vol_surge)

# Top N tickers each day (decided at t)
topN = score.apply(lambda r: list(r.dropna().sort_values(ascending=False).index[:TOP_N]), axis=1)

dates = close.index

# ---------- A) Intraday open->close, daily rebalance ----------
equity_A = [1.0]; prev = set()
for i in range(len(dates)-1):
    d, d1 = dates[i], dates[i+1]
    picks = set(topN.loc[d] or []); picks = set(list(picks)[:TOP_N])
    buys, sells = picks - prev, prev - picks
    n_trades = len(buys)+len(sells)
    if len(picks)==0:
        r = 0.0
    else:
        oc = (close.loc[d1, list(picks)].astype(float) / openp.loc[d1, list(picks)].astype(float) - 1.0).fillna(0)
        r = float(oc.mean())
    r -= COST * (n_trades * (1.0/TOP_N))
    equity_A.append(equity_A[-1]*(1+r))
    prev = picks
eqA = pd.Series(equity_A[1:], index=dates[1:])

# OC benchmark (index open->close)
nsei = yf.download("^NSEI", period="6y", interval="1d", auto_adjust=True, progress=False)
bmkOC = (nsei["Close"]/nsei["Open"] - 1.0).fillna(0)
bmkOC_eq = (1 + bmkOC).cumprod()
bmkOC_eq = bmkOC_eq.loc[eqA.index.min():eqA.index.max()]

# ---------- B) 1-day hold open->open ----------
equity_B = [1.0]; prev = set()
for i in range(len(dates)-2):
    d, d1, d2 = dates[i], dates[i+1], dates[i+2]
    picks = set(topN.loc[d] or []); picks = set(list(picks)[:TOP_N])
    buys, sells = picks - prev, prev - picks
    n_trades = len(buys)+len(sells)
    if len(picks)==0:
        r = 0.0
    else:
        oo = (openp.loc[d2, list(picks)].astype(float) / openp.loc[d1, list(picks)].astype(float) - 1.0).fillna(0)
        r = float(oo.mean())
    r -= COST * (n_trades * (1.0/TOP_N))
    equity_B.append(equity_B[-1]*(1+r))
    prev = picks
# align index to d2 dates
eqB = pd.Series(equity_B[1:], index=dates[2:])

# CC benchmark (index close->close)
bmkCC = (1 + nsei["Close"].pct_change().fillna(0)).cumprod()
bmkCC = bmkCC.loc[eqB.index.min():eqB.index.max()]

# ---------- C) Weekly hold (5 trading days), weekly rebalance ----------
equity_C = [1.0]; idxC = []
i = 0
while i <= len(dates)-1-6:  # need d, d1 buy, exit at close of d1+4 (i+5)
    d  = dates[i]
    d1 = dates[i+1]
    exit_close_day = dates[i+5]  # d1 + 4 trading days
    picks = set(topN.loc[d] or [])
    picks = set(list(picks)[:TOP_N])
    if len(picks)==0:
        r = 0.0
    else:
        ret = (close.loc[exit_close_day, list(picks)].astype(float) / openp.loc[d1, list(picks)].astype(float) - 1.0).fillna(0)
        r = float(ret.mean())
    # cost: buy at d1 open (TOP_N trades) + full sell at exit (TOP_N trades) on equal weights
    n_trades = TOP_N * 2
    r -= COST * (n_trades * (1.0/TOP_N))
    equity_C.append(equity_C[-1]*(1+r))
    idxC.append(exit_close_day)
    i += 5  # next rebalance week

eqC = pd.Series(equity_C[1:], index=pd.Index(idxC))

# CC benchmark sampled on the same exit dates
bmkCC_C = (1 + nsei["Close"].pct_change().fillna(0)).cumprod()
bmkCC_C = bmkCC_C.loc[eqC.index.min():eqC.index.max()]

# ---------- Save outputs ----------
out = Path("public/backtest"); out.mkdir(parents=True, exist_ok=True)

def pack_metrics(eq, bmk_eq):
    return {
        "1Y": slice_metrics(eq, 1),
        "3Y": slice_metrics(eq, 3),
        "5Y": slice_metrics(eq, 5),
        "Benchmark_OC_1Y": slice_metrics(bmk_eq, 1) if bmk_eq is not None else None
    }

summary = {
    "Strategy_A_intraday": {
        "1Y": slice_metrics(eqA, 1), "3Y": slice_metrics(eqA, 3), "5Y": slice_metrics(eqA, 5),
        "Benchmark_OC_1Y": slice_metrics(bmkOC_eq, 1), "Benchmark_OC_3Y": slice_metrics(bmkOC_eq, 3), "Benchmark_OC_5Y": slice_metrics(bmkOC_eq, 5)
    },
    "Strategy_B_1d_overnight": {
        "1Y": slice_metrics(eqB, 1), "3Y": slice_metrics(eqB, 3), "5Y": slice_metrics(eqB, 5),
        "Benchmark_CC_1Y": slice_metrics(bmkCC, 1), "Benchmark_CC_3Y": slice_metrics(bmkCC, 3), "Benchmark_CC_5Y": slice_metrics(bmkCC, 5)
    },
    "Strategy_C_weekly": {
        "1Y": slice_metrics(eqC, 1), "3Y": slice_metrics(eqC, 3), "5Y": slice_metrics(eqC, 5),
        "Benchmark_CC_1Y": slice_metrics(bmkCC_C, 1), "Benchmark_CC_3Y": slice_metrics(bmkCC_C, 3), "Benchmark_CC_5Y": slice_metrics(bmkCC_C, 5)
    }
}

with open(out/"summary.json","w") as f:
    json.dump(summary, f, indent=2)

# Save equity CSVs for each variant
eqA.to_csv(out/"equity_A.csv", header=["equity"])
eqB.to_csv(out/"equity_B.csv", header=["equity"])
eqC.to_csv(out/"equity_C.csv", header=["equity"])

# Plots
def plot_all():
    for (eq, bmk, tag, title) in [
        (eqA, bmkOC_eq, "A", "A) Intraday (open→close)"),
        (eqB, bmkCC,    "B", "B) 1-day hold (open→open)"),
        (eqC, bmkCC_C,  "C", "C) Weekly hold (open→close day+5)")
    ]:
        for yrs in [1,3,5]:
            plot_curve(eq, bmk, yrs, out/f"equity_{tag}_{yrs}y.png", f"{title} — {yrs}Y")

plot_all()

print("✅ Backtest complete. Files in public/backtest/")
