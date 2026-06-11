"""
==============================================================================
 PAIRS TRADING — PART 5: THE PORTFOLIO BACKTESTER
==============================================================================
Every part so far judged ONE pair. Real stat-arb never does that — it runs a
basket, because blending many weakly-edged pairs cancels their idiosyncratic
noise and lifts the combined Sharpe (the ~sqrt(N) effect). This part:

    screen universe (on TRAIN)
        -> keep every pair that passes the artifact guards
        -> backtest EACH survivor independently (OOS, with costs)
        -> blend their daily returns equal-weight into ONE portfolio
        -> report the PORTFOLIO's out-of-sample edge vs the individual pairs
        -> measure how diversified the book actually is

What to look for in the output:
  * Does the PORTFOLIO test-Sharpe come in ABOVE the average individual pair?
    If yes, the per-pair edges are real and the noise is cancelling (good).
    If it just hovers near zero, the edges were noise — a smoother zero.
  * The AVG PAIRWISE CORRELATION tells you the ceiling on the benefit: near 0
    means strong diversification (closer to the sqrt(N) ideal); high means the
    pairs move together (e.g. several bank pairs) so the blend helps little.
  * Diversification MULTIPLIES a real edge; it cannot CREATE one. A near-zero
    portfolio is the honest signal that the universe has no broad edge.

Run with:   python3 pairs_trading_05_portfolio.py
Requires:   pip install yfinance pandas numpy statsmodels matplotlib
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from itertools import combinations
import yfinance as yf


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
UNIVERSE = [
    "KO", "PEP", "PG", "CL", "MDLZ", "KMB",
    "V", "MA",
    "XOM", "CVX", "COP",
    "JPM", "BAC", "WFC", "C",
    "GLD", "IAU",
]
START      = "2012-01-01"
END        = "2019-01-01"
TRAIN_FRAC = 0.60

P_MAX, HL_MIN, HL_MAX = 0.05, 2, 126
CORR_MIN, BETA_MIN    = 0.50, 0.05

Z_WINDOW, ENTRY_Z, EXIT_Z, STOP_Z = 30, 2.0, 0.0, 3.5
COST_BPS   = 20.0

# Optionally hand-pick the basket, e.g. [("V","MA"), ("KMB","MDLZ"), ("CL","PG")].
# None = use everything that survives screening.
PAIRS_OVERRIDE = [("V","MA"), ("CL","PG"), ("MDLZ","PG"), ("KO","PEP"),("KMB","MDLZ")]


# =============================================================================
# DATA / SCREENING  (carried over, verified earlier)
# =============================================================================
def get_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
    return raw["Close"].dropna(axis=1, how="all").dropna()


def half_life(spread):
    lag = spread.shift(1); delta = spread - lag
    df = pd.concat([delta, lag], axis=1).dropna(); df.columns = ["d", "l"]
    lam = sm.OLS(df["d"], sm.add_constant(df["l"])).fit().params["l"]
    return -np.log(2) / lam if lam < 0 else np.inf


def screen_pair(train, a, b):
    p_ab, p_ba = coint(train[a], train[b])[1], coint(train[b], train[a])[1]
    (y, x, p) = (a, b, p_ab) if p_ab <= p_ba else (b, a, p_ba)
    beta = sm.OLS(train[y], sm.add_constant(train[x])).fit().params.iloc[1]
    spread = train[y] - beta * train[x]
    return {"pair": f"{y}/{x}", "y": y, "x": x, "p_value": p,
            "half_life": half_life(spread), "beta": beta,
            "corr": train[a].pct_change().corr(train[b].pct_change())}


def survivors(train):
    rows = [screen_pair(train, a, b) for a, b in combinations(train.columns, 2)]
    t = pd.DataFrame(rows)
    keep = t[(t.p_value <= P_MAX) & (t.half_life.between(HL_MIN, HL_MAX))
             & (t["corr"].abs() >= CORR_MIN) & (t.beta.abs() >= BETA_MIN)]
    return keep.sort_values("half_life").reset_index(drop=True)


# =============================================================================
# BACKTEST ENGINE  (per pair)
# =============================================================================
def rolling_zscore(spread, window):
    return (spread - spread.rolling(window).mean()) / spread.rolling(window).std()


def generate_positions(z, entry, exit_, stop):
    pos = pd.Series(0.0, index=z.index); state = 0
    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi): pos.iloc[i] = 0; continue
        if state == 0:
            if zi > entry: state = -1
            elif zi < -entry: state = 1
        elif state == 1:
            if zi >= exit_ or zi < -stop: state = 0
        elif state == -1:
            if zi <= exit_ or zi > stop: state = 0
        pos.iloc[i] = state
    return pos


def backtest_pair(prices, rec):
    """Return (daily net-return series, positions) for one pair, train-beta fixed."""
    y, x, beta = rec["y"], rec["x"], rec["beta"]
    spread = prices[y] - beta * prices[x]
    z = rolling_zscore(spread, Z_WINDOW)
    pos = generate_positions(z, ENTRY_Z, EXIT_Z, STOP_Z)
    ret = pos.shift(1).fillna(0) * (prices[y].pct_change() - prices[x].pct_change())
    turnover = pos.diff().abs().fillna(0)
    net = (ret - turnover * (COST_BPS / 1e4)).fillna(0.0)
    return net, pos


def sharpe(r):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0


def max_dd(r):
    eq = (1 + r.fillna(0)).cumprod()
    return (eq / eq.cummax() - 1).min()


def total_ret(r):
    return (1 + r.fillna(0)).prod() - 1


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"Downloading {len(UNIVERSE)} tickers ...")
    prices = get_prices(UNIVERSE, START, END)
    split = int(len(prices) * TRAIN_FRAC)
    split_date = prices.index[split]
    train = prices.iloc[:split]

    # ---- choose the basket ----
    if PAIRS_OVERRIDE is not None:
        recs = [screen_pair(train, a, b) for a, b in PAIRS_OVERRIDE]
    else:
        recs = survivors(train).to_dict("records")
    if not recs:
        print("No pairs in the basket. Loosen filters or widen the universe.")
        return
    print(f"Basket of {len(recs)} pairs: {', '.join(r['pair'] for r in recs)}\n")

    # ---- backtest each pair, collect daily net returns ----
    nets = {}
    print(f"Per-pair OUT-OF-SAMPLE results (costs = {COST_BPS:.0f} bps):")
    print(f"  {'pair':<10}{'OOS return':>12}{'OOS Sharpe':>12}{'OOS maxDD':>11}")
    for r in recs:
        net, _ = backtest_pair(prices, r)
        nets[r["pair"]] = net
        te = net.iloc[split:]
        print(f"  {r['pair']:<10}{total_ret(te)*100:>11.1f}%{sharpe(te):>12.2f}{max_dd(te)*100:>10.1f}%")

    net_df = pd.DataFrame(nets)                 # one column per pair, aligned by date

    # ---- blend: equal capital per pair => portfolio daily return = row mean ----
    port = net_df.mean(axis=1)
    port_tr, port_te = port.iloc[:split], port.iloc[split:]

    # ---- diversification diagnostic: how correlated are the pair returns? ----
    # Use only days where pairs are actually trading, so idle 0s don't fake low corr.
    active = net_df.iloc[split:].replace(0.0, np.nan)
    cmat = active.corr()
    avg_corr = cmat.where(~np.eye(len(cmat), dtype=bool)).stack().mean()

    avg_indiv_sharpe = np.mean([sharpe(nets[r["pair"]].iloc[split:]) for r in recs])

    print("\n================  PORTFOLIO (equal-weight blend)  ================")
    print(f"  Avg individual OOS Sharpe : {avg_indiv_sharpe:7.2f}")
    print(f"  PORTFOLIO  OOS Sharpe      : {sharpe(port_te):7.2f}   <-- the headline")
    print(f"  PORTFOLIO  OOS return      : {total_ret(port_te)*100:7.1f}%")
    print(f"  PORTFOLIO  OOS max drawdown: {max_dd(port_te)*100:7.1f}%")
    print(f"  (in-sample portfolio Sharpe: {sharpe(port_tr):.2f})")
    print(f"  Avg pairwise return corr   : {avg_corr:7.2f}   "
          f"(near 0 = well diversified; high = pairs move together)")
    if sharpe(port_te) > avg_indiv_sharpe + 0.05:
        print("  -> Portfolio beats the average pair: noise is cancelling. Real-ish edge.")
    else:
        print("  -> No lift over the average pair: little real edge to diversify.")

    # ---- plot: portfolio equity curve, individual pairs faded behind ----
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in net_df.columns:
        eq = (1 + net_df[col].fillna(0)).cumprod()
        ax.plot(eq.index, eq, alpha=0.35, linewidth=0.9, label=col)
    port_eq = (1 + port.fillna(0)).cumprod()
    ax.plot(port_eq.index, port_eq, color="black", linewidth=2.4, label="PORTFOLIO")
    ax.axvline(split_date, color="orange", ls=":", lw=2, label="train | test")
    ax.axhline(1.0, color="gray", lw=0.6)
    ax.set_title(f"Portfolio of {len(recs)} pairs vs individual pairs "
                 f"(bold = blend; only post-orange is out-of-sample)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    print("\nClose the plot window to finish.")
    plt.show()


if __name__ == "__main__":
    main()