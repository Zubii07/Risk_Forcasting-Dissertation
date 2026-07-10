"""
Inputs used by the optimiser
-----------------------------
  - 252 days of trailing HISTORICAL asset returns (raw scenarios)
  - No model forecasts are used here at all

This isolates the question: "How much value do LSTM/Transformer
forecasts actually add, if the optimiser already has full access
to historical return scenarios?"
"""

import os
import sys
import time
import warnings
import importlib
import numpy as np
import pandas as pd
import cvxpy as cp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

config = importlib.import_module("config.config")
PROCESSED_DATA_PATH = config.PROCESSED_DATA_PATH
METRICS_PATH = config.METRICS_PATH
PORTFOLIO_TICKERS = config.PORTFOLIO_TICKERS
CONFIDENCE_LEVEL = config.CONFIDENCE_LEVEL
MAX_WEIGHT = config.MAX_WEIGHT
NO_TRADE_THRESHOLD = config.NO_TRADE_THRESHOLD
TRANSACTION_COST = config.TRANSACTION_COST
REBALANCE_FREQ = config.REBALANCE_FREQ
warnings.filterwarnings("ignore")



TRADING_DAYS  = 252
LOOKBACK      = 252
ALPHA         = 1 - CONFIDENCE_LEVEL

print("=" * 60)
print("  PHASE 4 — EXPERIMENT 1: Historical Optimisation")
print("  (Baseline — Pure CVaR, full historical scenarios)")
print("=" * 60)

print("""
  Inputs consciously chosen for this experiment:
      252-day trailing historical returns (scenario set)
      No model forecasts used
      No expected-return constraint
  This is the traditional / naive approach.
""")

# STEP 1: Load data

print("[1/6] Loading data...")
log_returns = pd.read_csv(f"{PROCESSED_DATA_PATH}log_returns.csv", index_col=0, parse_dates=True)
test_df     = pd.read_csv(f"{PROCESSED_DATA_PATH}test.csv",        index_col=0, parse_dates=True)
asset_cols  = [c for c in log_returns.columns if c in PORTFOLIO_TICKERS]
returns     = log_returns[asset_cols].copy()
n_assets    = len(asset_cols)

test_start, test_end = test_df.index[0], test_df.index[-1]
test_slice  = returns.loc[test_start:test_end]
idx_series  = test_slice.index.to_series()
rebal_dates = idx_series.resample(REBALANCE_FREQ).apply(lambda x: x.max() if len(x) > 0 else pd.NaT)
rebal_dates = pd.DatetimeIndex(rebal_dates.dropna().values)
print(f"     {n_assets} assets | {len(rebal_dates)} rebalance dates")


# STEP 2: Rockafellar-Uryasev CVaR optimiser (scenario-based)
def solve_historical_cvar(scenario_returns: np.ndarray, max_weight=MAX_WEIGHT):
    """
    Minimises portfolio CVaR at ALPHA using raw historical scenarios.
    scenario_returns: (T, N) matrix of historical daily returns.
    """
    T, N = scenario_returns.shape
    w    = cp.Variable(N)
    zeta = cp.Variable()
    u    = cp.Variable(T)

    port_ret = scenario_returns @ w
    constraints = [
        u >= 0,
        u >= -port_ret - zeta,
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
    ]
    cvar_expr = zeta + cp.sum(u) / (T * ALPHA)
    prob = cp.Problem(cp.Minimize(cvar_expr), constraints)

    try:
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            prob.solve(solver=cp.SCS)
        weights = np.clip(w.value, 0, None)
        weights = weights / weights.sum()
        return weights, prob.value
    except Exception:
        return np.ones(N) / N, np.nan

# STEP 3: Rolling rebalance loop
print("\n[2/6] Running rolling historical CVaR optimisation...")

weights_history = pd.DataFrame(index=rebal_dates, columns=asset_cols, dtype=float)
solve_times      = []
skipped_rebalances = 0
traded_rebalances = 0
bootstrap_rebalances = 0
prev_weights = np.ones(n_assets) / n_assets


def maybe_apply_no_trade_rule(candidate_weights: np.ndarray,
                              previous_weights: np.ndarray,
                              threshold: float):
    distance = np.abs(candidate_weights - previous_weights).sum()
    if distance < threshold:
        return previous_weights.copy(), True, float(distance)
    return candidate_weights, False, float(distance)

t0_total = time.time()
for i, dt in enumerate(rebal_dates):
    window = returns.loc[:dt].iloc[-LOOKBACK:]
    if len(window) < LOOKBACK // 2:
        weights_history.loc[dt] = prev_weights
        bootstrap_rebalances += 1
        continue

    t0 = time.time()
    w, _ = solve_historical_cvar(window.values, MAX_WEIGHT)
    solve_times.append(time.time() - t0)

    w_final, skipped, l1_distance = maybe_apply_no_trade_rule(
        np.asarray(w, dtype=float), prev_weights, NO_TRADE_THRESHOLD
    )
    if skipped:
        skipped_rebalances += 1
    else:
        traded_rebalances += 1
    prev_weights = w_final.copy()
    weights_history.loc[dt] = w_final

    if (i + 1) % 20 == 0 or i == len(rebal_dates) - 1:
        print(f"   [{i+1}/{len(rebal_dates)}] rebalanced {dt.date()}"
              f" | L1 change={l1_distance:.4f}")

total_time = time.time() - t0_total
print(f"\n   Optimisation complete — {len(rebal_dates)} rebalances")
print(f"     Total optimisation time : {total_time:.2f}s")
print(f"     Avg time per rebalance  : {np.mean(solve_times)*1000:.1f}ms")
print(f"     No-trade threshold      : {NO_TRADE_THRESHOLD:.3f} (L1 distance)")
print(f"     Bootstrap rebalance days: {bootstrap_rebalances}")
print(f"     Skipped rebalances       : {skipped_rebalances}")
print(f"     Traded rebalances        : {traded_rebalances}")

# STEP 4: Build daily portfolio returns with transaction costs
print("\n[3/6] Building daily portfolio returns (with transaction costs)...")

daily_weights = weights_history.reindex(returns.loc[test_start:test_end].index).ffill()
daily_weights = daily_weights.fillna(1.0 / n_assets)

port_returns = (daily_weights.shift(1) * returns.loc[test_start:test_end]).sum(axis=1)

turnover = daily_weights.diff().abs().sum(axis=1).fillna(0)
cost_drag = turnover * TRANSACTION_COST
port_returns_net = port_returns - cost_drag
port_returns_net = port_returns_net.dropna()
annual_cost_drag = turnover.mean() * TRANSACTION_COST * TRADING_DAYS

print(f"     {len(port_returns_net)} daily returns computed")
print(f"     Avg daily turnover: {turnover.mean():.4f}")
print(f"     Estimated annual cost drag: {annual_cost_drag*100:.2f}%")

# STEP 5: Performance summary
print("\n[4/6] Performance summary...")

rf_daily = 0.02 / TRADING_DAYS
r    = port_returns_net
cum  = (1 + r).cumprod()
ann_ret  = (cum.iloc[-1]) ** (TRADING_DAYS / len(r)) - 1
ann_vol  = r.std() * np.sqrt(TRADING_DAYS)
sharpe   = (r.mean() - rf_daily) / r.std() * np.sqrt(TRADING_DAYS)
mdd      = ((cum - cum.cummax()) / cum.cummax()).min()
v95      = r.quantile(0.05)
cv95     = r[r <= v95].mean()

print(f"""
   ── Experiment 1 (Historical) — Performance ──
   Annualised Return : {ann_ret*100:.2f}%
   Annualised Vol     : {ann_vol*100:.2f}%
   Sharpe Ratio       : {sharpe:.4f}
   Max Drawdown       : {mdd*100:.2f}%
   CVaR 95%           : {cv95*100:.4f}%
""")

# STEP 6: Save outputs
print("[5/6] Saving outputs...")
os.makedirs(METRICS_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

port_returns_net.to_csv(f"{PROCESSED_DATA_PATH}exp1_historical_returns.csv", header=["Historical"])
weights_history.to_csv(f"{PROCESSED_DATA_PATH}exp1_historical_weights.csv")

summary = pd.DataFrame([{
    "Experiment": "Exp1_Historical",
    "Ann. Return (%)": round(ann_ret*100, 2),
    "Ann. Vol (%)":    round(ann_vol*100, 2),
    "Sharpe":          round(sharpe, 4),
    "Max DD (%)":      round(mdd*100, 2),
    "CVaR 95% (%)":    round(cv95*100, 4),
    "Avg Turnover":    round(turnover.mean(), 4),
    "Est. Annual Cost Drag (%)": round(annual_cost_drag*100, 4),
    "Total Opt Time (s)": round(total_time, 2),
    "Avg Opt Time/Rebalance (ms)": round(np.mean(solve_times)*1000, 1),
    "No-Trade Threshold": NO_TRADE_THRESHOLD,
    "Skipped Rebalances": skipped_rebalances,
    "Traded Rebalances": traded_rebalances,
    "Inputs Used": "252-day historical scenarios only",
}])
summary.to_csv(f"{METRICS_PATH}exp1_historical_summary.csv", index=False)

print("    exp1_historical_returns.csv")
print("    exp1_historical_weights.csv")
print("    exp1_historical_summary.csv")

print("\n[6/6] Experiment 1 complete.")
print("=" * 60)
print("   Run phase4_experiment2_forecast_driven.py next")
print("=" * 60)