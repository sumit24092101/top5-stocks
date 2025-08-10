import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

TOP_N = 5
COST_PER_TRADE = 0.001  # 0.10%

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

print("Downloading price data…")
data = yf.download(TICKERS, period="6y", interval="1d",
                   auto_adjust=True, progress=False, group_by="ticker", threads=True)

close_df = get_panel(data, "Close")
open_df  = get_panel(data, "Open")
vol_df   = get_panel(data, "Volume")

# Signals
ret5 = close_df / close_df.shift(5) - 1.0
vol20 = vol_df.rolling(20).mean()
vol_surge = (vol_df / vol20).replace([np.inf, -np.inf], np.nan)
score_df = score_func(ret5, vol_surge)

# Daily top N
picks_series = score_df.apply(lambda r: list(r.dropna().sort_values(ascending=False).index[:TOP_N]), axis=1)

# Sim: decide at t, buy at next open t+1, sell at t+1 close
dates = close_df.index
equity = [1.0]
prev = set()
daily_returns = []

for i in range(len(dates)-1):
    d, d1 = dates[i], dates[i+1]
    todays = set(picks_series.loc[d] or [])  # type: ignore
    todays = set(list(todays)[:TOP_N])

    buys  = todays - prev
    sells = prev - todays
    n_trades = len(buys) + len(sells)

    if len(todays) == 0:
        r = 0.0
    else:
        op = open_df.loc[d1, list(todays)].astype(float)
        cl = close_df.loc[d1, list(todays)].astype(float)
        rets = (cl / op - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        r = float(rets.mean())

    cost = COST_PER_TRADE * (n_trades * (1.0 / TOP_N))
    r_net = r - cost
    daily_returns.append(r_net)
    equity.append(equity[-1] * (1.0 + r_net))
    prev = todays

eq_series = pd.Series(equity[1:], index=dates[1:])

# Benchmark (NIFTY 50)
bmk = yf.download("^NSEI", period="6y", interval="1d", auto_adjust=True, progress=False)
bmk_eq = (1 + bmk["Close"].pct_change().fillna(0)).cumprod()
bmk_eq = bmk_eq.loc[eq_series.index.min():eq_series.index.max()]

def _squeeze_series(x):
    if isinstance(x, pd.DataFrame):
        x = x.squeeze()
    return x

def metrics(eq):
    eq = _squeeze_series(eq)
    if eq.empty or len(eq) < 2:
        return dict(CAGR=np.nan, MaxDD=np.nan, Sharpe=np.nan, Vol=np.nan)

    rets = _squeeze_series(eq.pct_change().dropna())
    if rets.empty:
        return dict(CAGR=np.nan, MaxDD=np.nan, Sharpe=np.nan, Vol=np.nan)

    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1) if years > 0 else np.nan
    rollmax = eq.cummax()
    maxdd = float((eq/rollmax - 1.0).min())
    std = float(rets.std())
    sharpe = float(rets.mean() / std * np.sqrt(252)) if std > 0 else np.nan
    vol = float(std * np.sqrt(252))
    return dict(CAGR=cagr, MaxDD=maxdd, Sharpe=sharpe, Vol=vol)

def slice_metrics(eq, years):
    eq = _squeeze_series(eq)
    cutoff = eq.index.max() - pd.Timedelta(days=int(365.25*years))
    return metrics(eq[eq.index >= cutoff])

summary = {
    "1Y":  slice_metrics(eq_series, 1),
    "3Y":  slice_metrics(eq_series, 3),
    "5Y":  slice_metrics(eq_series, 5),
    "Benchmark_1Y": slice_metrics(bmk_eq, 1),
    "Benchmark_3Y": slice_metrics(bmk_eq, 3),
    "Benchmark_5Y": slice_metrics(bmk_eq, 5),
}

# Save outputs
out_dir = Path("public/backtest")
out_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame({"Date": eq_series.index, "Equity": eq_series.values,
              "DailyReturn": [np.nan] + list(pd.Series(eq_series).pct_change().iloc[1:])}).to_csv(out_dir / "daily_returns.csv", index=False)

pd.DataFrame(summary).to_csv(out_dir / "summary.csv")

# Write clean JSON (plain floats)
with open(out_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Plots
def plot_curve(eq, bmk_eq, years, fname):
    cutoff = eq.index.max() - pd.Timedelta(days=int(365.25*years))
    eq_s = eq[eq.index >= cutoff]
    bmk_s = bmk_eq[bmk_eq.index >= cutoff]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    eq_s.plot(label="Strategy")
    bmk_s.plot(label="Benchmark")
    plt.title(f"Equity Curve — Last {years}Y")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / fname); plt.close()

plot_curve(eq_series, bmk_eq, 1, "equity_1y.png")
plot_curve(eq_series, bmk_eq, 3, "equity_3y.png")
plot_curve(eq_series, bmk_eq, 5, "equity_5y.png")

print("✅ Backtest complete; files in public/backtest")
