"""
==============================================================================
 PAIRS TRADING — PART 4: THE END-TO-END PIPELINE
==============================================================================
This stitches the screener (Part 3) onto the backtester (Part 2) so the whole
thing runs in one shot:

    download universe
        -> screen every pair ON THE TRAINING WINDOW (cointegration, half-life)
        -> throw out the statistical mirages (correlation + hedge-ratio guards)
        -> pick the best surviving pair
        -> run the full backtest on it OUT-OF-SAMPLE, with realistic costs
        -> report train vs test and plot

Two things to keep in mind while reading the result:

  * SELECTION BIAS: we pick the pair that looked best *in training*, so its
    TRAIN numbers are flattered by definition (we chose it for them). The TEST
    block — data the selection never touched — is the only honest verdict.

  * THE ARTIFACT GUARDS: Part 3's raw output was dominated by one range-bound
    stock (Citi) "cointegrating" with gold, oil, everything — because when the
    hedge ratio is ~0 the spread collapses to that single stock and the test is
    really just asking "is this one name stationary?". We now require a real
    return-correlation AND a non-trivial hedge ratio so those mirages get cut.
    (The correlation floor is a heuristic, not a theorem — cointegration is
    about price levels, correlation about returns — but it's an effective
    spurious-pair filter in practice.)

Run with:   python3 pairs_trading_04_pipeline.py
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
    "KO", "PEP", "PG", "CL", "KMB", "MDLZ",       # staples
    "V", "MA",                                     # payments
    "XOM", "CVX", "COP",                           # oil
    "JPM", "BAC", "WFC", "C",                      # banks
    "GLD", "IAU",                                  # gold ETFs (sanity-check pair)
]
START      = "2018-01-01"
END        = "2024-01-01"
TRAIN_FRAC = 0.60

# --- screening filters (selection happens on TRAIN only) ---
P_MAX      = 0.05     # cointegration gate
HL_MIN     = 2        # reject sub-2-day "reversion" (microstructure noise)
HL_MAX     = 126      # reject slower-than-~6-month reversion (untradeable)
CORR_MIN   = 0.50     # return-correlation floor — kills near-zero-corr mirages
BETA_MIN   = 0.05     # hedge-ratio floor — kills "spread is really just one stock"

# --- backtest params ---
Z_WINDOW   = 30
ENTRY_Z    = 2.0
EXIT_Z     = 0.0
STOP_Z     = 3.5
COST_BPS   = 20.0     # realistic round-trip-ish friction (the KO/PEP killer)

# Set to e.g. ("BAC", "JPM") to skip selection and backtest a specific pair.
FORCE_PAIR = None


# =============================================================================
# DATA
# =============================================================================
def get_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
    prices = raw["Close"].dropna(axis=1, how="all").dropna()
    return prices


# =============================================================================
# SCREENING  (Part 3)
# =============================================================================
def half_life(spread):
    lag = spread.shift(1)
    delta = spread - lag
    df = pd.concat([delta, lag], axis=1).dropna()
    df.columns = ["delta", "lag"]
    lam = sm.OLS(df["delta"], sm.add_constant(df["lag"])).fit().params["lag"]
    return -np.log(2) / lam if lam < 0 else np.inf


def screen_pair(train, a, b):
    # Engle-Granger is order-dependent, so test both ways and keep the better.
    p_ab = coint(train[a], train[b])[1]
    p_ba = coint(train[b], train[a])[1]
    if p_ab <= p_ba:
        y_name, x_name, p_value = a, b, p_ab
    else:
        y_name, x_name, p_value = b, a, p_ba

    beta = sm.OLS(train[y_name], sm.add_constant(train[x_name])).fit().params.iloc[1]
    spread = train[y_name] - beta * train[x_name]
    return {
        "pair": f"{y_name}/{x_name}",
        "y": y_name, "x": x_name,
        "p_value": p_value,
        "half_life": half_life(spread),
        "beta": beta,
        "corr": train[a].pct_change().corr(train[b].pct_change()),
    }


def screen_universe(train):
    rows = [screen_pair(train, a, b) for a, b in combinations(train.columns, 2)]
    return pd.DataFrame(rows)


def select_best(table):
    keep = table[
        (table["p_value"] <= P_MAX)
        & (table["half_life"].between(HL_MIN, HL_MAX))
        & (table["corr"].abs() >= CORR_MIN)
        & (table["beta"].abs() >= BETA_MIN)
    ].copy()
    if keep.empty:
        return None, keep
    # Among survivors, fastest reversion wins — speed is what beats costs.
    keep = keep.sort_values("half_life")
    return keep.iloc[0], keep


# =============================================================================
# BACKTEST  (Part 2)
# =============================================================================
def rolling_zscore(spread, window):
    m = spread.rolling(window).mean()
    s = spread.rolling(window).std()
    return (spread - m) / s


def generate_positions(z, entry, exit_, stop):
    pos = pd.Series(0.0, index=z.index)
    state = 0
    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            pos.iloc[i] = 0
            continue
        if state == 0:
            if zi > entry:    state = -1
            elif zi < -entry: state = 1
        elif state == 1:
            if zi >= exit_ or zi < -stop: state = 0
        elif state == -1:
            if zi <= exit_ or zi > stop: state = 0
        pos.iloc[i] = state
    return pos


def run_backtest(y, x, pos, cost_bps):
    ret_y, ret_x = y.pct_change(), x.pct_change()
    pos_held = pos.shift(1).fillna(0)                 # no look-ahead: trade next day
    gross = pos_held * (ret_y - ret_x)                # dollar-neutral 1:1 legs
    turnover = pos.diff().abs().fillna(0)
    return gross - turnover * (cost_bps / 1e4)


def annualised_sharpe(r):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0


def max_drawdown(r):
    eq = (1 + r.fillna(0)).cumprod()
    return (eq / eq.cummax() - 1).min()


def trade_stats(pos, net):
    in_mkt = pos != 0
    block = (in_mkt != in_mkt.shift()).cumsum()
    rets = [(1 + g.fillna(0)).prod() - 1
            for _, g in net.groupby(block) if (pos.loc[g.index] != 0).any()]
    rets = np.array(rets)
    return len(rets), (rets > 0).mean() if len(rets) else 0.0


def report(name, net, pos):
    n, win = trade_stats(pos, net)
    print(f"\n--- {name} ---")
    print(f"  Total return     : {((1 + net.fillna(0)).prod() - 1) * 100:7.1f}%")
    print(f"  Annualised Sharpe: {annualised_sharpe(net):7.2f}")
    print(f"  Max drawdown     : {max_drawdown(net) * 100:7.1f}%")
    print(f"  Number of trades : {n:7d}")
    print(f"  Win rate         : {win * 100:7.1f}%")


def make_plots(z, pos, net, split_date, pair):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(z.index, z, color="teal", linewidth=0.8)
    axes[0].axhline(ENTRY_Z, color="red", ls="--"); axes[0].axhline(-ENTRY_Z, color="green", ls="--")
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].fill_between(z.index, z.min(), z.max(), where=(pos != 0), color="gray", alpha=0.12)
    axes[0].axvline(split_date, color="orange", ls=":", lw=2, label="train | test")
    axes[0].set_title(f"{pair}  —  z-score (shaded = in position), orange = out-of-sample")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    eq = (1 + net.fillna(0)).cumprod()
    axes[1].plot(eq.index, eq, color="purple")
    axes[1].axvline(split_date, color="orange", ls=":", lw=2)
    axes[1].axhline(1.0, color="black", lw=0.6)
    axes[1].set_title("Equity curve — only the post-orange (out-of-sample) part is the honest result")
    axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"Downloading {len(UNIVERSE)} tickers ...")
    prices = get_prices(UNIVERSE, START, END)
    split = int(len(prices) * TRAIN_FRAC)
    split_date = prices.index[split]
    train = prices.iloc[:split]
    print(f"{len(prices.columns)} usable tickers, {len(prices)} shared days "
          f"(train = first {split}).")

    # ---- choose the pair ----
    if FORCE_PAIR is not None:
        a, b = FORCE_PAIR
        chosen = screen_pair(train, a, b)
        print(f"\nForced pair: {chosen['pair']}")
    else:
        table = screen_universe(train)
        chosen, survivors = select_best(table)
        if chosen is None:
            print("\nNo pair passed the filters. Loosen them or widen the universe.")
            print(table.sort_values("p_value").head(10).to_string(index=False))
            return
        print(f"\nScreened {len(table)} pairs; {len(survivors)} passed the artifact guards.")
        print("Top survivors (ranked by reversion speed):")
        pd.set_option("display.float_format", lambda v: f"{v:.4f}")
        print(survivors[["pair", "p_value", "half_life", "beta", "corr"]].head(8).to_string(index=False))
        print(f"\n>>> SELECTED (fastest reverting): {chosen['pair']}  "
              f"(p={chosen['p_value']:.4f}, half-life={chosen['half_life']:.1f}d, "
              f"corr={chosen['corr']:.2f})")

    # ---- backtest the chosen pair, beta fixed from TRAIN ----
    y_name, x_name, beta = chosen["y"], chosen["x"], chosen["beta"]
    spread = prices[y_name] - beta * prices[x_name]      # train-beta, full series
    z   = rolling_zscore(spread, Z_WINDOW)
    pos = generate_positions(z, ENTRY_Z, EXIT_Z, STOP_Z)
    net = run_backtest(prices[y_name], prices[x_name], pos, COST_BPS)

    print(f"\n================  BACKTEST: {chosen['pair']}  (costs = {COST_BPS:.0f} bps)  ================")
    report("TRAIN (in-sample — flattered by selection, do NOT trust)", net.iloc[:split], pos.iloc[:split])
    report("TEST  (out-of-sample — the real verdict)", net.iloc[split:], pos.iloc[split:])

    print("\nClose the plot window to finish.")
    make_plots(z, pos, net, split_date, chosen["pair"])


if __name__ == "__main__":
    main()