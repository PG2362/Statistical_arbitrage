"""
==============================================================================
 PAIRS TRADING — PART 2: BACKTESTING
==============================================================================
Part 1 ended with a z-score signal on a chart. This part answers the next
question: "if I had actually traded that signal, would I have made money?"

A backtest replays history one day at a time and simulates the trades your
rules would have produced. Done honestly it's the most useful tool you have;
done carelessly it's a machine for fooling yourself. This script is built to
be honest, and the three things that make it honest are called out in CAPS
where they happen:

    1. TRAIN/TEST SPLIT  - estimate the hedge ratio on early data only,
                           then judge the strategy on later data it never saw.
    2. EXECUTION LAG      - you trade on tomorrow's price, not the closing
                           price your signal was computed from (.shift(1)).
    3. TRANSACTION COSTS  - thin stat-arb edges die under real-world frictions.

Run with:   python3 pairs_trading_02_backtest.py
Requires:   pip install yfinance pandas numpy statsmodels matplotlib
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
TICKER_A   = "KMB"
TICKER_B   = "MDLZ"
START      = "2000-01-01"
END        = "2019-01-01"

TRAIN_FRAC = 0.60     # first 60% of days = training (fit beta), last 40% = out-of-sample test
Z_WINDOW   = 30       # rolling window for the z-score
ENTRY_Z    = 2.0      # open a trade when |z| crosses this
EXIT_Z     = 0.0      # close when the spread reverts to its mean
STOP_Z     = 3.5      # bail out if z keeps running — the relationship may be breaking
COST_BPS   = 20.0      # round-trip-ish cost per leg change, in basis points (1 bp = 0.01%)


# -----------------------------------------------------------------------------
# DATA  (same as Part 1)
# -----------------------------------------------------------------------------
def get_prices(a, b, start, end):
    raw = yf.download([a, b], start=start, end=end, auto_adjust=True)
    prices = raw["Close"][[a, b]].dropna()
    if prices.empty:
        raise ValueError("No data returned — check tickers / dates / connection.")
    return prices


# -----------------------------------------------------------------------------
# SPREAD  — note the (1) TRAIN/TEST SPLIT
# -----------------------------------------------------------------------------
# In Part 1 we fit beta on the whole history, which secretly used future data.
# Here we fit it on the TRAINING slice only. The test slice is then traded with
# a beta that was locked in *before* any test-period price existed — exactly the
# information you'd have had in real time.
def build_spread(prices, a, b, train_end_idx):
    train = prices.iloc[:train_end_idx]
    model = sm.OLS(train[a], sm.add_constant(train[b])).fit()
    beta = model.params[b]
    spread = prices[a] - beta * prices[b]          # applied to the FULL series
    return spread, beta


def rolling_zscore(spread, window):
    # Rolling (trailing-only) mean/std — no peeking ahead, just like Part 1.
    m = spread.rolling(window).mean()
    s = spread.rolling(window).std()
    return (spread - m) / s


# -----------------------------------------------------------------------------
# POSITIONS  — the state machine
# -----------------------------------------------------------------------------
# A z-score is continuous; a position is one of three states:
#     +1  long the spread   (z dropped below -ENTRY: spread cheap -> buy A, sell B)
#     -1  short the spread   (z rose above +ENTRY: spread rich -> sell A, buy B)
#      0  flat
# The key behaviour is HOLD-UNTIL-REVERSION: once in, we stay in until z crosses
# the exit band (0) or blows through the stop. That memory is why we loop with an
# explicit `state` variable instead of writing a clever one-liner — clarity beats
# cleverness, and at ~1000 daily rows speed is irrelevant.
def generate_positions(z, entry, exit_, stop):
    pos = pd.Series(0.0, index=z.index)
    state = 0
    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):                 # rolling-window warm-up: stay flat
            pos.iloc[i] = 0
            continue
        if state == 0:
            if zi > entry:      state = -1
            elif zi < -entry:   state = 1
        elif state == 1:                 # currently long the spread
            if zi >= exit_ or zi < -stop:
                state = 0
        elif state == -1:                # currently short the spread
            if zi <= exit_ or zi > stop:
                state = 0
        pos.iloc[i] = state
    return pos


# -----------------------------------------------------------------------------
# THE BACKTEST  — note (2) EXECUTION LAG and (3) TRANSACTION COSTS
# -----------------------------------------------------------------------------
def run_backtest(prices, pos, a, b, cost_bps):
    retA = prices[a].pct_change()
    retB = prices[b].pct_change()

    # (2) EXECUTION LAG: the position decided at the close of day t can only be
    # *held* starting day t+1, so it earns day t+1's return. Shifting by one day
    # is the difference between a real backtest and accidental time travel.
    pos_held = pos.shift(1).fillna(0)

    # Dollar-neutral spread return: long A / short B (or vice versa). Equal dollar
    # legs, so the daily P&L is direction * (retA - retB). (A more precise version
    # weights the B leg by beta; we keep it 1:1 here for a clean first pass.)
    gross = pos_held * (retA - retB)

    # (3) TRANSACTION COSTS: charged whenever the position changes. pos.diff()
    # is nonzero on the days we trade; each unit of change costs cost_bps.
    turnover = pos.diff().abs().fillna(0)
    costs = turnover * (cost_bps / 1e4)
    net = gross - costs
    return net


# -----------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------
def annualised_sharpe(r):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0

def max_drawdown(r):
    eq = (1 + r.fillna(0)).cumprod()
    return (eq / eq.cummax() - 1).min()

def trade_stats(pos, net):
    # A "trade" is a contiguous block of nonzero position. Group them and
    # compound each block's daily returns to get per-trade P&L -> win rate.
    in_mkt = pos != 0
    block = (in_mkt != in_mkt.shift()).cumsum()          # id that increments each switch
    trade_rets = []
    for _, grp in net.groupby(block):
        if (pos.loc[grp.index] != 0).any():              # only count blocks we were IN
            trade_rets.append((1 + grp.fillna(0)).prod() - 1)
    trade_rets = np.array(trade_rets)
    n = len(trade_rets)
    win = (trade_rets > 0).mean() if n else 0.0
    return n, win

def report(name, net, pos):
    total = (1 + net.fillna(0)).prod() - 1
    n, win = trade_stats(pos, net)
    print(f"\n--- {name} ---")
    print(f"  Total return     : {total*100:7.1f}%")
    print(f"  Annualised Sharpe: {annualised_sharpe(net):7.2f}")
    print(f"  Max drawdown     : {max_drawdown(net)*100:7.1f}%")
    print(f"  Number of trades : {n:7d}")
    print(f"  Win rate         : {win*100:7.1f}%")


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------
def make_plots(z, pos, net, split_date):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # z-score with shaded in-market periods + the train/test boundary
    axes[0].plot(z.index, z, color="teal", linewidth=0.8)
    axes[0].axhline(ENTRY_Z, color="red", ls="--"); axes[0].axhline(-ENTRY_Z, color="green", ls="--")
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].fill_between(z.index, z.min(), z.max(), where=(pos != 0), color="gray", alpha=0.12)
    axes[0].axvline(split_date, color="orange", ls=":", lw=2, label="train | test")
    axes[0].set_title("Z-score, shaded = in a position, orange = out-of-sample begins")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # equity curve (compounded net returns), train/test boundary marked
    eq = (1 + net.fillna(0)).cumprod()
    axes[1].plot(eq.index, eq, color="purple")
    axes[1].axvline(split_date, color="orange", ls=":", lw=2)
    axes[1].axhline(1.0, color="black", lw=0.6)
    axes[1].set_title("Equity curve (1.0 = starting capital). Only the post-orange part is the honest result.")
    axes[1].grid(alpha=0.3)

    plt.tight_layout(); plt.show()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print(f"Downloading {TICKER_A} / {TICKER_B} ...")
    prices = get_prices(TICKER_A, TICKER_B, START, END)
    split_idx = int(len(prices) * TRAIN_FRAC)
    split_date = prices.index[split_idx]
    print(f"{len(prices)} days | train = first {split_idx}, test = last {len(prices)-split_idx}")

    spread, beta = build_spread(prices, TICKER_A, TICKER_B, split_idx)
    print(f"Hedge ratio beta (fit on TRAIN only) = {beta:.3f}")

    z   = rolling_zscore(spread, Z_WINDOW)
    pos = generate_positions(z, ENTRY_Z, EXIT_Z, STOP_Z)
    net = run_backtest(prices, pos, TICKER_A, TICKER_B, COST_BPS)

    # Report train and test SEPARATELY. The test row is the one that matters;
    # if train looks great and test looks awful, you've overfit.
    report("TRAIN (in-sample — do NOT trust this)", net.iloc[:split_idx], pos.iloc[:split_idx])
    report("TEST  (out-of-sample — the real number)", net.iloc[split_idx:], pos.iloc[split_idx:])

    print("\nClose the plot window to finish.")
    make_plots(z, pos, net, split_date)


if __name__ == "__main__":
    main()