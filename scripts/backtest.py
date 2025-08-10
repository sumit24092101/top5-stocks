import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

TOP_N = 5
COST_PER_TRADE = 0.001  # 0.10%

# Universe: NIFTY 50 tickers
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
    frames = []
    for t in data.columns.levels[0]:
        if (t, field) in data.columns:
            frames.append(data[(t, field)].rename(t))
    return pd.concat(frames, axis=1).sort_index()

print("Downloading price data…")
data = yf.download(TICKERS, period="6y", interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True)

close_df = get_panel(data, "Close")
open_df = get_panel(data, "Open")
vol_df = get_panel(data, "Volume")

ret5 = close_df / close_df.shift(5) - 1.0
vol20 = vol_df.rolling(20).mean()
vol_surge = vol_df / vol20
score_df = score_func(ret5, vol_surge)

picks_series = score_df.apply(lambda row: list(row.dropna().sort_values(ascending=False).index[:TOP_N]), axis=1)

dates = close_df.index
equity = [1.0]
prev = set()
daily_returns = []

for i in range(len(dates)-1):
    d = dates[i]
    d1 = dates[i+1]
    todays_picks = set(picks_series.loc[d] or [])
    todays_picks = set(list(todays_picks)[:TOP_N])
    buys = todays_picks - prev
    sells = prev - todays_picks
    n_trades = len(buys) + len(sells)
    if len(todays_picks) == 0:
        r = 0.0
    else:
        op = open_df.loc[d1, list(todays_picks)].astype(float)
        cl = close_df.loc[d1, list(todays_picks)].astype(float)
        rets = (cl / op - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        r = rets.mean()
    cost = COST_PER_TRADE * (n_trades * (1.0/TOP_N))
    r_net = r - cost
    daily_returns.append(r_net)
    equity.append(equity[-1] * (1.0 + r_net))
    prev = todays_picks

eq_series = pd.Series(equity[1:], index=dates[1:])

# Benchmark
bmk_data = yf.download("^NSEI", period="6y", interval="1d", auto_adjust=True, progress=False)
bmk_eq = (1 + bmk_data["Close"].pct_change().fillna(0)).cumprod()
bmk_eq = bmk_eq.loc[eq_series.index.min():eq_series.index.max()]

def metrics(eq):
    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    maxdd = (eq/eq.cummax() - 1).min()
    sharpe = rets.mean()/rets.std()*np.sqrt(252) if rets.std()>0 else np.nan
    vol = rets.std()*np.sqrt(252)
    return dict(CAGR=cagr, MaxDD=maxdd, Sharpe=sharpe, Vol=vol)

def slice_metrics(eq, years):
    cutoff = eq.index.max() - pd.Timedelta(days=int(365.25*years))
    return metrics(eq[eq.index >= cutoff])

summary = {
    "1Y": slice_metrics(eq_series, 1),
    "3Y": slice_metrics(eq_series, 3),
    "5Y": slice_metrics(eq_series, 5),
    "Benchmark_1Y": slice_metrics(bmk_eq, 1),
    "Benchmark_3Y": slice_metrics(bmk_eq, 3),
    "Benchmark_5Y": slice_metrics(bmk_eq, 5),
}

# Save outputs
out_dir = Path("public/backtest")
out_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame([{"Date": d, "Return": r} for d, r in zip(eq_series.index, daily_returns)]).to_csv(out_dir / "daily_returns.csv", index=False)
pd.DataFrame(summary).to_csv(out_dir / "summary.csv")
pd.Series(summary).to_json(out_dir / "summary.json")

# Plots
for label, years in [("1y",1), ("3y",3), ("5y",5)]:
    cutoff = eq_series.index.max() - pd.Timedelta(days=int(365.25*years))
    plt.figure(figsize=(8,4))
    eq_series[eq_series.index >= cutoff].plot(label="Strategy")
    bmk_eq[bmk_eq.index >= cutoff].plot(label="Benchmark")
    plt.title(f"Equity Curve - Last {label.upper()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"equity_{label}.png")
    plt.close()

print("✅ Backtest complete.")
