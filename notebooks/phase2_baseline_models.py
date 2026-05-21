import os
import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.historical_volatility import HistoricalVolatilityModel
from src.models.garch_model import GARCHModel
from src.evaluation.risk_metrics import (
    parametric_var,
    parametric_cvar,
    evaluate_volatility_forecast,
    kupiec_pof_test,
    realised_volatility_proxy,
)
from config.config import (
    PROCESSED_DATA_PATH,
    METRICS_PATH,
    FIGURES_PATH,
    PORTFOLIO_TICKERS,
    CONFIDENCE_LEVEL,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

warnings.filterwarnings("ignore")


plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Load Phase 1 Processed Data

print("\n[1/7] Loading processed data from Phase 1...")

try:
    log_returns = pd.read_csv(f"{PROCESSED_DATA_PATH}log_returns.csv", index_col=0, parse_dates=True)
    train_df    = pd.read_csv(f"{PROCESSED_DATA_PATH}train.csv",       index_col=0, parse_dates=True)
    val_df      = pd.read_csv(f"{PROCESSED_DATA_PATH}val.csv",         index_col=0, parse_dates=True)
    test_df     = pd.read_csv(f"{PROCESSED_DATA_PATH}test.csv",        index_col=0, parse_dates=True)
except FileNotFoundError as e:
    raise SystemExit(f" Phase 1 outputs not found: {e}\n   Run notebooks/phase1_data_pipeline.py first.")

asset_cols = [c for c in log_returns.columns if c in PORTFOLIO_TICKERS]
if not asset_cols:
    raise SystemExit(f" No asset columns found. Expected: {PORTFOLIO_TICKERS}\n   Got: {list(log_returns.columns)}")

returns       = log_returns[asset_cols].copy()
train_end_idx = len(train_df)

print(f"    Loaded returns for {len(asset_cols)} assets")
print(f"    Total rows : {len(returns):,}")
print(f"    Date range : {returns.index[0].date()} → {returns.index[-1].date()}")
print(f"    Train/Val/Test sizes: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")

#  Historical Volatility Model

print("\n[2/7] Fitting Historical Volatility model...")

histvol      = HistoricalVolatilityModel(window=30, annualise=False)
hv_forecasts = pd.DataFrame(index=returns.index, columns=asset_cols, dtype=float)

for asset in asset_cols:
    hv_forecasts[asset] = histvol.forecast(returns[asset])

hv_forecasts = hv_forecasts.dropna()
print(f"    HistVol (30-day) forecasts: {hv_forecasts.shape}")


# GARCH(1,1) Model

print("\n[3/7] Fitting GARCH(1,1) model (refits every 252 days)...")

garch           = GARCHModel(p=1, q=1, dist="normal")
garch_forecasts = pd.DataFrame(index=returns.index, columns=asset_cols, dtype=float)

for i, asset in enumerate(asset_cols, 1):
    print(f"   [{i}/{len(asset_cols)}] GARCH on {asset} ...", end=" ", flush=True)
    try:
        fc = garch.fit_predict(returns[asset], train_size=train_end_idx, refit_every=252)
        garch_forecasts[asset] = fc
        print(f"  ({fc.dropna().shape[0]:,} forecasts)")
    except Exception as e:
        print(f"  ({e})")

garch_forecasts = garch_forecasts.dropna(how="all")
print(f"\n   GARCH forecasts: {garch_forecasts.shape}")


# STEP 4: VaR & CVaR Estimation

print("\n[4/7] Computing VaR and CVaR at 95% confidence...")

hv_var     = pd.DataFrame(index=hv_forecasts.index,    columns=asset_cols, dtype=float)
hv_cvar    = pd.DataFrame(index=hv_forecasts.index,    columns=asset_cols, dtype=float)
garch_var  = pd.DataFrame(index=garch_forecasts.index, columns=asset_cols, dtype=float)
garch_cvar = pd.DataFrame(index=garch_forecasts.index, columns=asset_cols, dtype=float)

for asset in asset_cols:
    hv_var [asset] = parametric_var (hv_forecasts[asset], CONFIDENCE_LEVEL)
    hv_cvar[asset] = parametric_cvar(hv_forecasts[asset], CONFIDENCE_LEVEL)
    if asset in garch_forecasts.columns:
        garch_var [asset] = parametric_var (garch_forecasts[asset], CONFIDENCE_LEVEL)
        garch_cvar[asset] = parametric_cvar(garch_forecasts[asset], CONFIDENCE_LEVEL)

print("   VaR & CVaR computed for both models")


# Evaluate on Test Set

print("\n[5/7] Evaluating models on test set...")

test_start = test_df.index[0]
test_end   = test_df.index[-1]
eval_rows  = []

for asset in asset_cols:
    rets_test = returns[asset].loc[test_start:test_end]
    hv_test   = hv_forecasts[asset].loc[test_start:test_end]
    g_test    = garch_forecasts[asset].loc[test_start:test_end] if asset in garch_forecasts.columns else pd.Series(dtype=float)

    if hv_test.dropna().empty or rets_test.dropna().empty:
        continue

    eval_rows.append({"Asset": asset, "Model": "HistVol_30d", **evaluate_volatility_forecast(hv_test, rets_test)})
    if not g_test.dropna().empty:
        eval_rows.append({"Asset": asset, "Model": "GARCH(1,1)", **evaluate_volatility_forecast(g_test, rets_test)})

eval_df = pd.DataFrame(eval_rows)
agg     = eval_df.groupby("Model")[["MAE", "RMSE", "QLIKE"]].mean().round(6)

print("\n   Forecast Accuracy on Test Set:")
print(eval_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("\n   Average across all assets:")
print(agg.to_string())


# STEP 6: Kupiec POF VaR Backtest

print("\n[6/7] Running Kupiec POF VaR backtests...")

backtest_rows = []
for asset in asset_cols:
    rets_test = returns[asset].loc[test_start:test_end]

    hv_var_test = hv_var[asset].loc[test_start:test_end]
    if not hv_var_test.dropna().empty:
        res = kupiec_pof_test(rets_test, hv_var_test, CONFIDENCE_LEVEL)
        backtest_rows.append({"Asset": asset, "Model": "HistVol_30d", **res})

    if asset in garch_var.columns:
        g_var_test = garch_var[asset].loc[test_start:test_end]
        if not g_var_test.dropna().empty:
            res = kupiec_pof_test(rets_test, g_var_test, CONFIDENCE_LEVEL)
            backtest_rows.append({"Asset": asset, "Model": "GARCH(1,1)", **res})

backtest_df = pd.DataFrame(backtest_rows)
summary = backtest_df[["Asset","Model","N","Exceptions","Expected_Exceptions","Observed_Rate","p_value","Model_Adequate"]].copy()
summary["Expected_Exceptions"] = summary["Expected_Exceptions"].round(1)
summary["Observed_Rate"]       = summary["Observed_Rate"].round(4)
summary["p_value"]             = summary["p_value"].round(4)

print("\n   Kupiec POF Test Results (p > 0.05 = model adequate):")
print(summary.to_string(index=False))

pct_adequate = backtest_df.groupby("Model")["Model_Adequate"].mean() * 100
print("\n   % of assets where VaR model is adequate:")
for model, pct in pct_adequate.items():
    print(f"     {model:<15} {pct:>5.1f}%")


# STEP 7: Save All Outputs

print("\n[7/7] Saving forecasts, metrics, and risk measures...")

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(METRICS_PATH,        exist_ok=True)
os.makedirs(FIGURES_PATH,        exist_ok=True)

hv_forecasts.to_csv   (f"{PROCESSED_DATA_PATH}histvol_forecasts.csv")
garch_forecasts.to_csv(f"{PROCESSED_DATA_PATH}garch_forecasts.csv")
hv_var.to_csv         (f"{PROCESSED_DATA_PATH}histvol_var.csv")
hv_cvar.to_csv        (f"{PROCESSED_DATA_PATH}histvol_cvar.csv")
garch_var.to_csv      (f"{PROCESSED_DATA_PATH}garch_var.csv")
garch_cvar.to_csv     (f"{PROCESSED_DATA_PATH}garch_cvar.csv")
eval_df.to_csv        (f"{METRICS_PATH}phase2_forecast_metrics.csv", index=False)
backtest_df.to_csv    (f"{METRICS_PATH}phase2_var_backtests.csv",    index=False)

print("   All Phase 2 outputs saved")


# VISUALISATIONS


# Plot 6: Volatility Forecasts vs Realised
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
axes = axes.flatten()
demo_assets = [a for a in ["SPY", "AAPL", "TLT", "NVDA"] if a in asset_cols]
for i, asset in enumerate(demo_assets):
    ax  = axes[i]
    rv  = realised_volatility_proxy(returns[asset]).loc[test_start:test_end]
    hv_ = hv_forecasts[asset].loc[test_start:test_end]
    g_  = garch_forecasts[asset].loc[test_start:test_end] if asset in garch_forecasts.columns else pd.Series(dtype=float)
    ax.plot(rv.index,  rv,  color="lightgray", linewidth=0.6, label="Realised (|r|)", alpha=0.7)
    ax.plot(hv_.index, hv_, color="#2196F3",   linewidth=1.1, label="HistVol_30d")
    if not g_.dropna().empty:
        ax.plot(g_.index, g_, color="#E91E63", linewidth=1.1, label="GARCH(1,1)")
    ax.set_title(f"{asset} — Volatility Forecasts (Test Set)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
plt.suptitle("Volatility Forecasts — Test Set", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}06_volatility_forecasts.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n Plot 6 saved: Volatility Forecasts")

# Plot 7: VaR Breaches (SPY)
if "SPY" in asset_cols:
    fig, ax = plt.subplots(figsize=(16, 6))
    spy_ret = returns["SPY"].loc[test_start:test_end]
    spy_hv  = hv_var["SPY"].loc[test_start:test_end]
    spy_g   = garch_var["SPY"].loc[test_start:test_end]
    ax.plot(spy_ret.index, spy_ret, color="lightgray", linewidth=0.6, label="SPY daily returns")
    ax.plot(spy_hv.index, -spy_hv, color="#2196F3",   linewidth=1.0, label="HistVol 95% VaR")
    ax.plot(spy_g.index,  -spy_g,  color="#E91E63",   linewidth=1.0, label="GARCH 95% VaR")
    breaches_hv = spy_ret[spy_ret < -spy_hv]
    breaches_g  = spy_ret[spy_ret < -spy_g]
    ax.scatter(breaches_hv.index, breaches_hv, color="#1565C0", s=20, zorder=5, label=f"HistVol breaches ({len(breaches_hv)})")
    ax.scatter(breaches_g.index,  breaches_g,  color="#AD1457", s=20, zorder=5, label=f"GARCH breaches ({len(breaches_g)})", marker="x")
    ax.set_title("SPY — VaR (95%) Breach Analysis on Test Set", fontsize=13, fontweight="bold")
    ax.set_ylabel("Return")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}07_var_breaches_spy.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(" Plot 7 saved: SPY VaR Breach Analysis")

# Plot 8: Forecast Accuracy Bars
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metric_pivot = eval_df.pivot(index="Asset", columns="Model", values=["MAE","RMSE","QLIKE"])
for i, metric in enumerate(["MAE","RMSE","QLIKE"]):
    metric_pivot[metric].plot(kind="bar", ax=axes[i], color=["#2196F3","#E91E63"], edgecolor="black", linewidth=0.5)
    axes[i].set_title(f"{metric} (lower = better)", fontsize=12, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=45, labelsize=9)
    axes[i].legend(fontsize=8)
    axes[i].grid(axis="y", alpha=0.3)
plt.suptitle("Phase 2 — Forecast Accuracy by Asset (Test Set)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}08_forecast_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()
print(" Plot 8 saved: Forecast Accuracy Comparison")

# Plot 9: VaR Exception Heatmap
fig, ax = plt.subplots(figsize=(11, 6))
exception_pivot = backtest_df.pivot(index="Asset", columns="Model", values="Observed_Rate")
expected_rate   = 1 - CONFIDENCE_LEVEL
sns.heatmap(exception_pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
            center=expected_rate, vmin=0, vmax=expected_rate * 2.5,
            ax=ax, linewidths=0.5,
            cbar_kws={"label": f"Observed exception rate (expected = {expected_rate})"})
ax.set_title(f"VaR Exception Rates — Closer to {expected_rate} = better", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}09_var_exception_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print(" Plot 9 saved: VaR Exception Heatmap")


# SUMMARY

print("\n" + "=" * 60)
print("  PHASE 2 COMPLETE ✅")
print("=" * 60)
print(f"""
  Models built:
     Historical Volatility (30-day rolling)
     GARCH(1,1) with periodic refitting

  Risk measures:
     VaR  @ {int(CONFIDENCE_LEVEL*100)}% confidence
     CVaR @ {int(CONFIDENCE_LEVEL*100)}% confidence

  Evaluation:
     MAE, RMSE, QLIKE on test set
     Kupiec POF backtests on every asset

  Average test-set metrics:
{agg.to_string()}

  Files saved to data/processed/ and results/

""")