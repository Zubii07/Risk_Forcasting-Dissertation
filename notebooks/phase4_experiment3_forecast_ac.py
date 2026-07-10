"""
Purpose
-------
Builds on Experiment 2 (forecast-driven, CVaR-only optimisation)
by adding the two enhancements the client/supervisor asked for:

  A — Minimum Return Constraint
      Optimise CVaR while requiring the portfolio to achieve at
      least a target annualised return. Uses a COMPACT expected
      return proxy (60-day EWMA of returns) — NOT the full
      historical return series — to stay consistent with the
      "forecast-driven, minimal historical dependence" philosophy.

  C — Regime-Aware Allocation
      Blends between a purely defensive (Experiment 2 style) and
      a return-seeking weight vector, based on the current VIX
      regime:
        - Stress  (VIX > 25): 100% defensive weights
        - Normal  (VIX ≤ 25): blend toward the return-constrained
          weights (70% return-seeking / 30% defensive)

Objective
---------
Improve annualised return versus Experiment 2, while preserving
as much of the downside protection as possible. This experiment
is expected to show a genuine risk/return TRADE-OFF, not a free
lunch — that trade-off is the point.

============================================================
"""

import os
import sys
import time
import warnings
import importlib
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.stats import norm

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
VIX_STRESS_THRESHOLD = config.VIX_STRESS_THRESHOLD
MIN_RETURN_TARGET = config.MIN_RETURN_TARGET
EWMA_RETURN_SPAN = config.EWMA_RETURN_SPAN

warnings.filterwarnings("ignore")


TRADING_DAYS = 252
ALPHA        = 1 - CONFIDENCE_LEVEL
Z_ALPHA      = norm.ppf(ALPHA)
Z_CVAR       = norm.pdf(Z_ALPHA) / ALPHA
DAILY_RETURN_TARGET = (1 + MIN_RETURN_TARGET) ** (1/TRADING_DAYS) - 1

MODEL_FILES = {
    "HistVol":     "histvol_cvar.csv",
    "GARCH":       "garch_cvar.csv",
    "GJR-GARCH":   "gjr_garch_cvar.csv",
    "LSTM":        "lstm_cvar.csv",
    "Transformer": "transformer_cvar.csv",
}

# Which single model to run the full A+C experiment for (best from Exp2 /
# or the dissertation's primary model). Can loop all models if desired.
PRIMARY_MODEL = "Transformer"

print("=" * 60)
print("  PHASE 4 — EXPERIMENT 3: Forecast-Driven + A+C")
print("  (Return constraint + Regime-aware allocation)")
print("=" * 60)

print(f"""
  Inputs consciously chosen for this experiment:
     Per-asset CVaR forecasts ({PRIMARY_MODEL})
     Fixed correlation matrix (training period only)
     Compact EWMA expected return ({EWMA_RETURN_SPAN}-day span)
     VIX regime label (Normal / Stress) for blending
     No raw historical return scenarios

  Enhancement A — Minimum return target : {MIN_RETURN_TARGET*100:.1f}% annualised
  Enhancement C — Regime blend          : Stress=100% defensive,
                                           Normal=70% return-seeking
""")

# STEP 1: Load data
print("[1/8] Loading data...")
log_returns = pd.read_csv(f"{PROCESSED_DATA_PATH}log_returns.csv", index_col=0, parse_dates=True)
train_df    = pd.read_csv(f"{PROCESSED_DATA_PATH}train.csv",       index_col=0, parse_dates=True)
test_df     = pd.read_csv(f"{PROCESSED_DATA_PATH}test.csv",        index_col=0, parse_dates=True)
vix_df      = pd.read_csv(f"{PROCESSED_DATA_PATH}vix.csv",         index_col=0, parse_dates=True)

asset_cols  = [c for c in log_returns.columns if c in PORTFOLIO_TICKERS]
returns     = log_returns[asset_cols].copy()
n_assets    = len(asset_cols)
test_start, test_end = test_df.index[0], test_df.index[-1]

cvar_path = f"{PROCESSED_DATA_PATH}{MODEL_FILES[PRIMARY_MODEL]}"
if not os.path.exists(cvar_path):
    raise SystemExit(f" Missing {cvar_path}. Run Phase 2/3 first.")
cvar_df = pd.read_csv(cvar_path, index_col=0, parse_dates=True)[asset_cols]

corr_path = f"{PROCESSED_DATA_PATH}fixed_correlation_matrix.csv"
if os.path.exists(corr_path):
    corr_train = pd.read_csv(corr_path, index_col=0).values
    print("    Reusing fixed correlation matrix from Experiment 2")
else:
    corr_train = train_df[asset_cols].corr().values
    print("     Correlation matrix estimated fresh (training period only)")

vix_test = vix_df.reindex(returns.loc[test_start:test_end].index).ffill().bfill()
vix_series = vix_test.iloc[:, 0]

print(f"    {n_assets} assets | CVaR forecasts: {cvar_df.shape}")

# STEP 2: Compact expected-return proxy (EWMA, NOT full history)
print(f"\n[2/8] Building compact expected-return proxy "
      f"({EWMA_RETURN_SPAN}-day EWMA)...")
print("   This is intentionally minimal — a single decaying average,")
print("   not the full historical return series used in Experiment 1.")

ewma_returns = returns[asset_cols].ewm(span=EWMA_RETURN_SPAN, adjust=False).mean()
print(f"    EWMA expected-return series built: {ewma_returns.shape}")


# STEP 3: Two optimisers — Defensive (Exp2-style) & Return-Seeking
def solve_defensive(cvar_vec, corr, max_weight=MAX_WEIGHT):
    """Pure CVaR minimisation — identical to Experiment 2."""
    sigma = np.clip(cvar_vec, 1e-6, None) / Z_CVAR
    D = np.diag(sigma)
    Sigma = D @ corr @ D
    Sigma = (Sigma + Sigma.T) / 2
    eigvals = np.linalg.eigvalsh(Sigma)
    if eigvals.min() < 0:
        Sigma += np.eye(len(sigma)) * (abs(eigvals.min()) + 1e-8)

    N = len(sigma)
    w = cp.Variable(N)
    port_var = cp.quad_form(w, cp.psd_wrap(Sigma))
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    prob = cp.Problem(cp.Minimize(port_var), constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            prob.solve(solver=cp.SCS)
        weights = np.clip(w.value, 0, None)
        return weights / weights.sum()
    except Exception:
        return np.ones(N) / N


def solve_return_constrained(cvar_vec, corr, mu_vec, target_return,
                              max_weight=MAX_WEIGHT):
    """
    Enhancement A: CVaR minimisation subject to a minimum expected
    return constraint, using the compact EWMA return proxy.
    """
    sigma = np.clip(cvar_vec, 1e-6, None) / Z_CVAR
    D = np.diag(sigma)
    Sigma = D @ corr @ D
    Sigma = (Sigma + Sigma.T) / 2
    eigvals = np.linalg.eigvalsh(Sigma)
    if eigvals.min() < 0:
        Sigma += np.eye(len(sigma)) * (abs(eigvals.min()) + 1e-8)

    N = len(sigma)
    w = cp.Variable(N)
    port_var = cp.quad_form(w, cp.psd_wrap(Sigma))
    constraints = [
        cp.sum(w) == 1, w >= 0, w <= max_weight,
        mu_vec @ w >= target_return,
    ]
    prob = cp.Problem(cp.Minimize(port_var), constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            prob.solve(solver=cp.SCS)
        if w.value is None:
            raise ValueError("infeasible")
        weights = np.clip(w.value, 0, None)
        return weights / weights.sum(), True
    except Exception:
        # Fallback: relax the return constraint (drop it) if infeasible
        w_fallback = solve_defensive(cvar_vec, corr, max_weight)
        return w_fallback, False


def apply_no_trade_rule(candidate_weights: np.ndarray,
                         previous_weights: np.ndarray,
                         threshold: float):
    distance = np.abs(candidate_weights - previous_weights).sum()
    if distance < threshold:
        return previous_weights.copy(), True, float(distance)
    return candidate_weights, False, float(distance)

# STEP 4: Rolling rebalance loop — regime-aware blend
print("\n[3/8] Running Experiment 3 (regime-aware, return-constrained)...")

idx_series  = returns.loc[test_start:test_end].index.to_series()
rebal_dates = idx_series.resample(REBALANCE_FREQ).apply(lambda x: x.max() if len(x)>0 else pd.NaT)
rebal_dates = pd.DatetimeIndex(rebal_dates.dropna().values)
rebal_dates = rebal_dates.intersection(cvar_df.index)

weights_history   = pd.DataFrame(index=rebal_dates, columns=asset_cols, dtype=float)
regime_log        = pd.Series(index=rebal_dates, dtype=object)
infeasible_count  = 0
normal_infeasible_count = 0
stress_infeasible_count = 0
solve_times        = []
skipped_rebalances  = 0
traded_rebalances   = 0
prev_weights        = np.ones(n_assets) / n_assets

t0_total = time.time()
for i, dt in enumerate(rebal_dates):
    cvar_vec = cvar_df.loc[dt, asset_cols].values.astype(float)
    cvar_vec = np.nan_to_num(cvar_vec, nan=np.nanmean(cvar_vec))
    mu_vec   = ewma_returns.loc[dt, asset_cols].values.astype(float)

    current_vix = vix_series.loc[dt] if dt in vix_series.index else vix_series.asof(dt)
    is_stress   = current_vix > VIX_STRESS_THRESHOLD

    t0 = time.time()
    w_defensive = solve_defensive(cvar_vec, corr_train, MAX_WEIGHT)

    if is_stress:
        w_final = w_defensive
        regime_log.loc[dt] = "Stress (100% defensive)"
    else:
        w_return, feasible = solve_return_constrained(
            cvar_vec, corr_train, mu_vec, DAILY_RETURN_TARGET, MAX_WEIGHT
        )
        if not feasible:
            infeasible_count += 1
            normal_infeasible_count += 1
        # Blend: 70% return-seeking, 30% defensive during Normal regime
        w_final = 0.70 * w_return + 0.30 * w_defensive
        w_final = w_final / w_final.sum()
        regime_log.loc[dt] = "Normal (70% return-seeking blend)"

    w_final, skipped, l1_distance = apply_no_trade_rule(
        np.asarray(w_final, dtype=float), prev_weights, NO_TRADE_THRESHOLD
    )
    if skipped:
        skipped_rebalances += 1
    else:
        traded_rebalances += 1
    prev_weights = w_final.copy()

    solve_times.append(time.time() - t0)
    weights_history.loc[dt] = w_final

    if (i + 1) % 30 == 0 or i == len(rebal_dates) - 1:
        print(f"   [{i+1}/{len(rebal_dates)}] {dt.date()} — {regime_log.loc[dt]}"
              f" | L1 change={l1_distance:.4f}")

total_time = time.time() - t0_total
print(f"\n {len(rebal_dates)} rebalances complete")
print(f"   Total time: {total_time:.2f}s | Avg: {np.mean(solve_times)*1000:.1f}ms")
print(f"   Infeasible return constraint (fell back to defensive): "
      f"{infeasible_count}/{len(rebal_dates)} rebalances")
print(f"   No-trade threshold      : {NO_TRADE_THRESHOLD:.3f} (L1 distance)")
print(f"   Skipped rebalances       : {skipped_rebalances}")
print(f"   Traded rebalances        : {traded_rebalances}")

# STEP 5: Build daily portfolio returns
print("\n[4/8] Building daily portfolio returns (with transaction costs)...")

daily_weights = weights_history.reindex(returns.loc[test_start:test_end].index).ffill()
daily_weights = daily_weights.fillna(1.0 / n_assets)
port_returns  = (daily_weights.shift(1) * returns.loc[test_start:test_end]).sum(axis=1)
turnover      = daily_weights.diff().abs().sum(axis=1).fillna(0)
port_returns_net = (port_returns - turnover * TRANSACTION_COST).dropna()
annual_cost_drag = turnover.mean() * TRANSACTION_COST * TRADING_DAYS

print(f"  {len(port_returns_net)} daily returns | "
      f"Avg turnover: {turnover.mean():.4f}")
print(f"  Estimated annual cost drag: {annual_cost_drag*100:.2f}%")

# STEP 6: Performance summary
print("\n[5/8] Performance summary...")

rf_daily = 0.02 / TRADING_DAYS
r    = port_returns_net
cum  = (1 + r).cumprod()
ann_ret  = cum.iloc[-1] ** (TRADING_DAYS / len(r)) - 1
ann_vol  = r.std() * np.sqrt(TRADING_DAYS)
sharpe   = (r.mean() - rf_daily) / r.std() * np.sqrt(TRADING_DAYS)
mdd      = ((cum - cum.cummax()) / cum.cummax()).min()
v95      = r.quantile(0.05)
cv95     = r[r <= v95].mean()

print(f"""
   ── Experiment 3 (Forecast-Driven + A+C, {PRIMARY_MODEL}) — Performance ──
   Annualised Return : {ann_ret*100:.2f}%
   Annualised Vol     : {ann_vol*100:.2f}%
   Sharpe Ratio       : {sharpe:.4f}
   Max Drawdown       : {mdd*100:.2f}%
   CVaR 95%           : {cv95*100:.4f}%
""")

# STEP 7: Regime breakdown
print("[6/8] Regime allocation breakdown...")
regime_counts = regime_log.value_counts()
print(regime_counts.to_string())
if len(rebal_dates) > 0:
    infeasible_pct = (infeasible_count / len(rebal_dates)) * 100
    print(f"   Infeasible return constraint rate: {infeasible_pct:.2f}%")
    print(f"   Infeasible during Normal regime  : {normal_infeasible_count}")
    print(f"   Infeasible during Stress regime  : {stress_infeasible_count}")

# STEP 8: Save outputs
print("\n[7/8] Saving outputs...")
os.makedirs(METRICS_PATH, exist_ok=True)

port_returns_net.to_csv(f"{PROCESSED_DATA_PATH}exp3_forecast_ac_returns.csv",
                        header=[f"Exp3_{PRIMARY_MODEL}_AC"])
weights_history.to_csv(f"{PROCESSED_DATA_PATH}exp3_forecast_ac_weights.csv")
regime_log.to_csv(f"{METRICS_PATH}exp3_regime_log.csv", header=["Regime"])

summary = pd.DataFrame([{
    "Experiment": f"Exp3_ForecastDriven_AC_{PRIMARY_MODEL}",
    "Model": PRIMARY_MODEL,
    "Ann. Return (%)": round(ann_ret*100, 2),
    "Ann. Vol (%)":    round(ann_vol*100, 2),
    "Sharpe":          round(sharpe, 4),
    "Max DD (%)":      round(mdd*100, 2),
    "CVaR 95% (%)":    round(cv95*100, 4),
    "Avg Turnover":    round(turnover.mean(), 4),
    "Est. Annual Cost Drag (%)": round(annual_cost_drag*100, 4),
    "Total Opt Time (s)": round(total_time, 2),
    "Avg Opt Time/Rebalance (ms)": round(np.mean(solve_times)*1000, 1),
    "Infeasible Rebalances": infeasible_count,
    "Infeasible Rebalances (%)": round((infeasible_count / len(rebal_dates)) * 100, 2) if len(rebal_dates) else np.nan,
    "Normal Regime Infeasible": normal_infeasible_count,
    "Stress Regime Infeasible": stress_infeasible_count,
    "Skipped Rebalances": skipped_rebalances,
    "Traded Rebalances": traded_rebalances,
    "Min Return Target (Ann. %)": MIN_RETURN_TARGET*100,
    "Inputs Used": "CVaR forecast + fixed correlation + EWMA return + VIX regime",
}])
summary.to_csv(f"{METRICS_PATH}exp3_forecast_ac_summary.csv", index=False)

print(" exp3_forecast_ac_returns.csv")
print(" exp3_forecast_ac_weights.csv")
print(" exp3_regime_log.csv")
print(" exp3_forecast_ac_summary.csv")

print("\n[8/8] Experiment 3 complete.")
print("=" * 60)
print("  Run phase4_comparison.py next")
print("=" * 60)