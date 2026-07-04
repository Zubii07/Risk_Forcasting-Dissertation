import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.stats import norm

from config.config import (
    PROCESSED_DATA_PATH, METRICS_PATH,
    PORTFOLIO_TICKERS, CONFIDENCE_LEVEL, MAX_WEIGHT,
    TRANSACTION_COST, REBALANCE_FREQ,
)
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRADING_DAYS = 252
ALPHA        = 1 - CONFIDENCE_LEVEL
Z_ALPHA      = norm.ppf(ALPHA)
Z_CVAR       = norm.pdf(Z_ALPHA) / ALPHA     # converts sigma -> parametric CVaR

MODEL_FILES = {
    "HistVol":     "histvol_cvar.csv",
    "GARCH":       "garch_cvar.csv",
    "GJR-GARCH":   "gjr_garch_cvar.csv",
    "LSTM":        "lstm_cvar.csv",
    "Transformer": "transformer_cvar.csv",
}

print("=" * 60)
print("  PHASE 4 — EXPERIMENT 2: Forecast-Driven Optimisation")
print("  (No historical return scenarios given to optimiser)")
print("=" * 60)

print("""
  Inputs consciously chosen for this experiment:
     Per-asset one-day-ahead CVaR forecasts (model-specific)
     Fixed correlation matrix (estimated once, training period only)
     No raw historical return scenarios
     No expected-return constraint
  This isolates: "Are the CVaR forecasts alone sufficient
  to drive sensible portfolio construction?"
""")

# STEP 1: Load data

print("[1/7] Loading data...")
log_returns = pd.read_csv(f"{PROCESSED_DATA_PATH}log_returns.csv", index_col=0, parse_dates=True)
train_df    = pd.read_csv(f"{PROCESSED_DATA_PATH}train.csv",       index_col=0, parse_dates=True)
test_df     = pd.read_csv(f"{PROCESSED_DATA_PATH}test.csv",        index_col=0, parse_dates=True)

asset_cols  = [c for c in log_returns.columns if c in PORTFOLIO_TICKERS]
returns     = log_returns[asset_cols].copy()
n_assets    = len(asset_cols)
test_start, test_end = test_df.index[0], test_df.index[-1]

cvar_forecasts = {}
for model_name, fname in MODEL_FILES.items():
    path = f"{PROCESSED_DATA_PATH}{fname}"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        cvar_forecasts[model_name] = df[[c for c in asset_cols if c in df.columns]]
        print(f"    {model_name:<12} CVaR forecasts loaded: {cvar_forecasts[model_name].shape}")
    else:
        print(f"     {model_name:<12} file not found ({fname}) — skipped")

if not cvar_forecasts:
    raise SystemExit(" No CVaR forecast files found. Run Phase 2/3 first.")

# STEP 2: Fixed correlation matrix (training period ONLY)

print("\n[2/7] Estimating FIXED correlation matrix from training period...")
print("   (This structural input is estimated ONCE and never updated —")
print("    it does not use test-period data and is not a forecast.)")

corr_train = train_df[asset_cols].corr().values
print(f"     Correlation matrix shape: {corr_train.shape}")
print(f"     Mean off-diagonal correlation: "
      f"{(corr_train.sum() - n_assets) / (n_assets**2 - n_assets):.3f}")

pd.DataFrame(corr_train, index=asset_cols, columns=asset_cols).to_csv(
    f"{PROCESSED_DATA_PATH}fixed_correlation_matrix.csv"
)

# STEP 3: Parametric CVaR optimiser (forecast-driven)
def solve_forecast_driven_cvar(cvar_vec: np.ndarray, corr: np.ndarray,
                                max_weight=MAX_WEIGHT):
    """
    Minimises parametric portfolio CVaR using ONLY per-asset CVaR
    forecasts (converted to implied sigma) + fixed correlation.

    No raw historical returns are used in this function.
    """
    sigma = np.clip(cvar_vec, 1e-6, None) / Z_CVAR
    D     = np.diag(sigma)
    Sigma = D @ corr @ D
    Sigma = (Sigma + Sigma.T) / 2   # enforce symmetry (numerical safety)

    # Ensure positive semi-definite (numerical safety)
    eigvals = np.linalg.eigvalsh(Sigma)
    if eigvals.min() < 0:
        Sigma += np.eye(len(sigma)) * (abs(eigvals.min()) + 1e-8)

    N = len(sigma)
    w = cp.Variable(N)
    port_var  = cp.quad_form(w, cp.psd_wrap(Sigma))
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
    prob = cp.Problem(cp.Minimize(port_var), constraints)

    try:
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            prob.solve(solver=cp.SCS)
        weights = np.clip(w.value, 0, None)
        weights = weights / weights.sum()
        parametric_cvar = Z_CVAR * np.sqrt(weights @ Sigma @ weights)
        return weights, parametric_cvar
    except Exception:
        return np.ones(N) / N, np.nan


# STEP 4: Rolling rebalance loop — per model
print("\n[3/7] Running forecast-driven optimisation for each model...")

idx_series  = returns.loc[test_start:test_end].index.to_series()
rebal_dates_all = idx_series.resample(REBALANCE_FREQ).apply(lambda x: x.max() if len(x)>0 else pd.NaT)
rebal_dates_all = pd.DatetimeIndex(rebal_dates_all.dropna().values)

results = {}
timing_rows = []

for model_name, cvar_df in cvar_forecasts.items():
    print(f"\n   ── {model_name} ──")
    rebal_dates = rebal_dates_all.intersection(cvar_df.index)
    if len(rebal_dates) == 0:
        print(f"     No overlapping dates for {model_name} — skipped")
        continue

    weights_history = pd.DataFrame(index=rebal_dates, columns=asset_cols, dtype=float)
    solve_times = []

    t0_total = time.time()
    for i, dt in enumerate(rebal_dates):
        cvar_vec = cvar_df.loc[dt, asset_cols].values.astype(float)
        cvar_vec = np.nan_to_num(cvar_vec, nan=np.nanmean(cvar_vec))

        t0 = time.time()
        w, _ = solve_forecast_driven_cvar(cvar_vec, corr_train, MAX_WEIGHT)
        solve_times.append(time.time() - t0)
        weights_history.loc[dt] = w

    total_time = time.time() - t0_total
    print(f"     {len(rebal_dates)} rebalances | "
          f"total={total_time:.2f}s | avg={np.mean(solve_times)*1000:.1f}ms")

    daily_weights = weights_history.reindex(returns.loc[test_start:test_end].index).ffill()
    daily_weights = daily_weights.fillna(1.0 / n_assets)
    port_returns = (daily_weights.shift(1) * returns.loc[test_start:test_end]).sum(axis=1)
    turnover = daily_weights.diff().abs().sum(axis=1).fillna(0)
    port_returns_net = (port_returns - turnover * TRANSACTION_COST).dropna()

    results[model_name] = {
        "returns": port_returns_net,
        "weights": weights_history,
        "turnover": turnover.mean(),
        "total_time": total_time,
        "avg_time_ms": np.mean(solve_times) * 1000,
        "n_rebalances": len(rebal_dates),
    }
    timing_rows.append({
        "Model": model_name,
        "N Rebalances": len(rebal_dates),
        "Total Opt Time (s)": round(total_time, 2),
        "Avg Opt Time/Rebalance (ms)": round(np.mean(solve_times)*1000, 1),
    })

# STEP 5: Performance summary — all models
print("\n[4/7] Performance summary (all forecast-driven models)...")

rf_daily = 0.02 / TRADING_DAYS
summary_rows = []
for model_name, res in results.items():
    r   = res["returns"]
    cum = (1 + r).cumprod()
    ann_ret = cum.iloc[-1] ** (TRADING_DAYS / len(r)) - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe  = (r.mean() - rf_daily) / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    v95     = r.quantile(0.05)
    cv95    = r[r <= v95].mean()

    summary_rows.append({
        "Experiment": f"Exp2_ForecastDriven_{model_name}",
        "Model": model_name,
        "Ann. Return (%)": round(ann_ret*100, 2),
        "Ann. Vol (%)":    round(ann_vol*100, 2),
        "Sharpe":          round(sharpe, 4),
        "Max DD (%)":      round(mdd*100, 2),
        "CVaR 95% (%)":    round(cv95*100, 4),
        "Avg Turnover":    round(res["turnover"], 4),
        "Total Opt Time (s)": round(res["total_time"], 2),
        "Avg Opt Time/Rebalance (ms)": round(res["avg_time_ms"], 1),
        "Inputs Used": "CVaR forecast + fixed correlation only",
    })

summary_df = pd.DataFrame(summary_rows)
print("\n" + summary_df[["Model","Ann. Return (%)","Sharpe","Max DD (%)","CVaR 95% (%)"]].to_string(index=False))


# STEP 6: Weight differentiation check (supervisor point #10)
print("\n[5/7] Checking weight differentiation across models...")
print("   (Different forecasts SHOULD produce different allocations —")
print("    otherwise the forecasting model choice doesn't matter.)")

if len(results) >= 2:
    model_names = list(results.keys())
    avg_weights = pd.DataFrame({
        m: results[m]["weights"].mean() for m in model_names
    })
    print("\n   Average allocation by model:")
    print(avg_weights.round(3).to_string())

    # Pairwise weight distance
    dist_rows = []
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            common_dates = results[m1]["weights"].index.intersection(results[m2]["weights"].index)
            if len(common_dates) == 0:
                continue
            diff = (results[m1]["weights"].loc[common_dates] -
                    results[m2]["weights"].loc[common_dates]).abs().mean().mean()
            dist_rows.append({"Model A": m1, "Model B": m2, "Avg Weight Distance": round(diff, 4)})
    dist_df = pd.DataFrame(dist_rows)
    print("\n   Pairwise average weight distance (higher = more differentiated):")
    print(dist_df.to_string(index=False))
    dist_df.to_csv(f"{METRICS_PATH}exp2_weight_differentiation.csv", index=False)

# STEP 7: Save outputs
print("\n[6/7] Saving outputs...")
os.makedirs(METRICS_PATH, exist_ok=True)

for model_name, res in results.items():
    safe_name = model_name.replace("-", "_")
    res["returns"].to_csv(f"{PROCESSED_DATA_PATH}exp2_forecast_{safe_name}_returns.csv",
                          header=[model_name])
    res["weights"].to_csv(f"{PROCESSED_DATA_PATH}exp2_forecast_{safe_name}_weights.csv")

summary_df.to_csv(f"{METRICS_PATH}exp2_forecast_driven_summary.csv", index=False)
pd.DataFrame(timing_rows).to_csv(f"{METRICS_PATH}exp2_timing.csv", index=False)

print("     exp2_forecast_<model>_returns.csv (one per model)")
print("     exp2_forecast_<model>_weights.csv (one per model)")
print("     exp2_forecast_driven_summary.csv")
print("     exp2_weight_differentiation.csv")
print("     exp2_timing.csv")

print("\n[7/7] Experiment 2 complete.")
print("=" * 60)
print("    Run phase4_experiment3_forecast_ac.py next")
print("=" * 60)