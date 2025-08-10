export default async function Page() {
  async function getData() {
    try {
      const res = await fetch("/data/latest.json", { cache: "no-store" });
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    }
  }
  const data = await getData();
  const picks = data?.picks || [];

  return (
    <main className="mx-auto max-w-3xl p-6">
      <h1 className="text-3xl font-semibold">Top 5 NSE Stock Picks for Today</h1>
      <p className="mt-2 text-sm opacity-70">
        Universe: {data?.universe || "—"} · Method: {data?.method || "—"} · Updated: {data?.as_of_ist || "—"}
      </p>

      <div className="mt-6 grid gap-4">
        {picks.length === 0 && (
          <div className="rounded-xl border p-4">
            <p>No picks yet. The daily job will populate data after the first run.</p>
          </div>
        )}
        {picks.map((p, i) => (
          <div key={p.ticker} className="rounded-2xl border p-4 shadow-sm">
            <div className="flex items-baseline justify-between">
              <h2 className="text-xl font-medium">{i + 1}. {p.ticker}</h2>
              <span className="text-sm">Score: {p.score}</span>
            </div>
            <div className="mt-2 text-sm">
              <div>Close: ₹{p.close}</div>
              <div>5-day return: {(p.ret5 * 100).toFixed(2)}%</div>
              <div>Volume surge vs 20d avg: {p.vol_surge}×</div>
              <div>RSI(14): {p.rsi14}</div>
            </div>
          </div>
        ))}
      </div>

      <footer className="mt-8 text-xs opacity-60">
        Not investment advice. Educational purposes only.
      </footer>
    </main>
  );
}
