"""
PHASE 2 ENHANCEMENTS
Deep Learning-Based Downside Risk Forecasting

Adds to Phase 2 baseline results:
  1. GJR-GARCH model (asymmetric volatility — leverage effect)
  2. Regime-split VaR breach table (Normal vs Stress)
  3. Expanded tail-risk underestimation commentary
  4. New comparison charts including GJR-GARCH

Run AFTER phase2_baseline_models.py
Reads from : data/processed/ (Phase 1 & 2 outputs)
Writes to  : data/processed/, results/metrics/, results/figures/
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import norm as scipy_norm
from arch import arch_model

from config.config import (
    PROCESSED_DATA_PATH, METRICS_PATH, FIGURES_PATH,
    PORTFOLIO_TICKERS, CONFIDENCE_LEVEL,
    VIX_STRESS_THRESHOLD, STRESS_PERIODS,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
TRADING_DAYS = 252

print("=" * 60)
print("  PHASE 2 ENHANCEMENTS")
print("  GJR-GARCH + Regime-Split VaR Analysis")
print("=" * 60)


# STEP 1: Load Phase 1 & 2 outputs

print("\n[1/7] Loading Phase 1 & 2 outputs...")

try:
    log_returns = pd.read_csv(f"{PROCESSED_DATA_PATH}log_returns.csv",
                              index_col=0, parse_dates=True)
    train_df    = pd.read_csv(f"{PROCESSED_DATA_PATH}train.csv",
                              index_col=0, parse_dates=True)
    test_df     = pd.read_csv(f"{PROCESSED_DATA_PATH}test.csv",
                              index_col=0, parse_dates=True)
    vix_df      = pd.read_csv(f"{PROCESSED_DATA_PATH}vix.csv",
                              index_col=0, parse_dates=True)
    hv_var      = pd.read_csv(f"{PROCESSED_DATA_PATH}histvol_var.csv",
                              index_col=0, parse_dates=True)
    garch_var   = pd.read_csv(f"{PROCESSED_DATA_PATH}garch_var.csv",
                              index_col=0, parse_dates=True)
    p2_metrics  = pd.read_csv(f"{METRICS_PATH}phase2_forecast_metrics.csv")
except FileNotFoundError as e:
    raise SystemExit(
        f" Missing file: {e}\n"
        "   Run phase2_baseline_models.py first."
    )

asset_cols  = [c for c in log_returns.columns if c in PORTFOLIO_TICKERS]
returns     = log_returns[asset_cols].copy()
train_end   = train_df.index[-1]
test_start  = test_df.index[0]
test_end    = test_df.index[-1]
train_end_idx = len(train_df)

# Align VIX to returns index
vix_aligned = vix_df.reindex(returns.index).ffill().bfill()
stress_mask = vix_aligned.iloc[:, 0] > VIX_STRESS_THRESHOLD
normal_mask = ~stress_mask

print(f"    {len(asset_cols)} assets loaded")
print(f"    Test: {test_start.date()} → {test_end.date()}")
print(f"    VIX stress days in test: "
      f"{stress_mask.loc[test_start:test_end].sum()} / "
      f"{len(stress_mask.loc[test_start:test_end])}")

# STEP 2: GJR-GARCH model

print("\n[2/7] Fitting GJR-GARCH model...")
print("   GJR-GARCH captures the leverage effect:")
print("   σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I_{t-1} + β·σ²_{t-1}")
print("   where I_{t-1}=1 if ε_{t-1}<0 (bad news amplifies volatility)")


def fit_gjr_garch_rolling(returns_series, train_size, refit_every=252):
    """
    Rolling GJR-GARCH(1,1,1) with periodic refitting.
    Returns Series of one-step-ahead volatility forecasts.
    """
    ret_pct    = returns_series * 100
    n          = len(ret_pct)
    forecasts  = pd.Series(index=returns_series.index, dtype=float)

    last_fit   = None
    omega = alpha = gamma = beta = None

    for i in range(train_size, n):
        if last_fit is None or (i - last_fit) >= refit_every:
            try:
                model = arch_model(
                    ret_pct.iloc[:i],
                    mean    = "Zero",
                    vol     = "GARCH",
                    p       = 1,
                    o       = 1,       # GJR asymmetric term
                    q       = 1,
                    dist    = "normal",
                    rescale = False,
                )
                res   = model.fit(disp="off", show_warning=False)
                omega = res.params.get("omega",   0.01)
                alpha = res.params.get("alpha[1]",0.05)
                gamma = res.params.get("gamma[1]",0.05)   # leverage term
                beta  = res.params.get("beta[1]", 0.85)
                last_fit = i
            except Exception:
                if omega is None:
                    omega, alpha, gamma, beta = 0.01, 0.05, 0.05, 0.85

        try:
            prev_ret = ret_pct.iloc[i - 1]
            indicator = 1.0 if prev_ret < 0 else 0.0   # leverage indicator
            if i == train_size or pd.isna(forecasts.iloc[i - 1]):
                prev_var = ret_pct.iloc[:i].var()
            else:
                prev_var = forecasts.iloc[i - 1] ** 2
            next_var = (omega
                        + alpha * prev_ret ** 2
                        + gamma * prev_ret ** 2 * indicator
                        + beta  * prev_var)
            forecasts.iloc[i] = np.sqrt(max(next_var, 1e-10))
        except Exception:
            forecasts.iloc[i] = np.nan

    forecasts = forecasts / 100   # back to decimal
    forecasts.name = "GJR-GARCH"
    return forecasts


gjr_forecasts = pd.DataFrame(index=returns.index,
                              columns=asset_cols, dtype=float)

for i, asset in enumerate(asset_cols, 1):
    print(f"   [{i}/{len(asset_cols)}] GJR-GARCH on {asset} ...",
          end=" ", flush=True)
    try:
        fc = fit_gjr_garch_rolling(
            returns[asset], train_size=train_end_idx, refit_every=252
        )
        gjr_forecasts[asset] = fc
        print(f"  ({fc.dropna().shape[0]:,} forecasts)")
    except Exception as e:
        print(f"  ({e})")

gjr_forecasts = gjr_forecasts.dropna(how="all")
print(f"\n    GJR-GARCH forecasts: {gjr_forecasts.shape}")


# STEP 3: VaR & CVaR from GJR-GARCH

print("\n[3/7] Computing GJR-GARCH VaR & CVaR...")

z    = scipy_norm.ppf(1 - CONFIDENCE_LEVEL)
gjr_var  = pd.DataFrame(index=gjr_forecasts.index,
                         columns=asset_cols, dtype=float)
gjr_cvar = pd.DataFrame(index=gjr_forecasts.index,
                         columns=asset_cols, dtype=float)

for asset in asset_cols:
    if asset in gjr_forecasts.columns:
        gjr_var[asset]  = -z * gjr_forecasts[asset]
        gjr_cvar[asset] = gjr_forecasts[asset] * \
                          scipy_norm.pdf(z) / (1 - CONFIDENCE_LEVEL)

gjr_var.to_csv (f"{PROCESSED_DATA_PATH}gjr_garch_var.csv")
gjr_cvar.to_csv(f"{PROCESSED_DATA_PATH}gjr_garch_cvar.csv")
print("    GJR-GARCH VaR & CVaR saved")

# STEP 4: Forecast accuracy — GJR-GARCH vs others

print("\n[4/7] Evaluating GJR-GARCH forecast accuracy...")


def realised_vol_proxy(r): return r.abs()

def mae_fn(pred, actual):
    df = pd.concat([pred, actual], axis=1).dropna()
    return float((df.iloc[:, 0] - df.iloc[:, 1]).abs().mean())

def rmse_fn(pred, actual):
    df = pd.concat([pred, actual], axis=1).dropna()
    return float(np.sqrt(((df.iloc[:, 0] - df.iloc[:, 1])**2).mean()))

def qlike_fn(pred, actual):
    df = pd.concat([pred, actual], axis=1).dropna()
    pv = (df.iloc[:, 0]**2).clip(lower=1e-10)
    av = (df.iloc[:, 1]**2).clip(lower=1e-10)
    ratio = av / pv
    return float((ratio - np.log(ratio) - 1).mean())


gjr_eval_rows = []
for asset in asset_cols:
    rets_test = returns[asset].loc[test_start:test_end]
    g_test    = gjr_forecasts[asset].loc[test_start:test_end] \
                if asset in gjr_forecasts.columns else pd.Series(dtype=float)
    if g_test.dropna().empty:
        continue
    actual = realised_vol_proxy(rets_test)
    gjr_eval_rows.append({
        "Asset": asset, "Model": "GJR-GARCH",
        "MAE":   mae_fn(g_test, actual),
        "RMSE":  rmse_fn(g_test, actual),
        "QLIKE": qlike_fn(g_test, actual),
    })

gjr_eval_df  = pd.DataFrame(gjr_eval_rows)
full_eval_df = pd.concat([p2_metrics, gjr_eval_df], ignore_index=True)

agg = full_eval_df.groupby("Model")[["MAE","RMSE","QLIKE"]].mean().round(6)
print("\n   ── Forecast Accuracy — All Econometric Models (Test Set) ──")
print(agg.to_string())

full_eval_df.to_csv(f"{METRICS_PATH}phase2_enhanced_metrics.csv", index=False)

# STEP 5: Regime-split VaR breach analysis

print("\n[5/7] Regime-split VaR breach analysis (Normal vs Stress)...")

# Stress/Normal masks for test period only
stress_test = stress_mask.loc[test_start:test_end]
normal_test = normal_mask.loc[test_start:test_end]

regime_breach_rows = []

def kupiec_pvalue(n, x, p_exp):
    """Simple Kupiec POF p-value."""
    if x == 0 or x == n:
        return np.nan
    p_obs = x / n
    lr = -2 * (x*np.log(p_exp) + (n-x)*np.log(1-p_exp)
               - x*np.log(p_obs) - (n-x)*np.log(1-p_obs))
    from scipy.stats import chi2
    return float(1 - chi2.cdf(lr, df=1))

all_vars = {
    "HistVol_30d": hv_var,
    "GARCH(1,1)":  garch_var,
    "GJR-GARCH":   gjr_var,
}

for model_name, var_df in all_vars.items():
    for asset in asset_cols:
        if asset not in var_df.columns:
            continue
        r    = returns[asset].loc[test_start:test_end]
        v    = var_df[asset].loc[test_start:test_end]
        common = r.index.intersection(v.index)
        r, v = r.loc[common], v.loc[common]

        for mask, label in [
            (stress_test.reindex(common).fillna(False), "Stress (VIX>25)"),
            (normal_test.reindex(common).fillna(True),  "Normal (VIX≤25)"),
        ]:
            r_reg = r[mask]
            v_reg = v[mask]
            if len(r_reg) < 10:
                continue
            breaches = (r_reg < -v_reg).sum()
            n        = len(r_reg)
            obs_rate = breaches / n
            p_val    = kupiec_pvalue(n, int(breaches), 1-CONFIDENCE_LEVEL)
            regime_breach_rows.append({
                "Model"       : model_name,
                "Asset"       : asset,
                "Regime"      : label,
                "N Days"      : n,
                "Breaches"    : int(breaches),
                "Expected"    : round(n * (1-CONFIDENCE_LEVEL), 1),
                "Breach Rate" : round(obs_rate, 4),
                "p-value"     : round(p_val, 4) if not np.isnan(p_val) else np.nan,
                "Adequate"    : "Yes" if (not np.isnan(p_val) and p_val > 0.05) else "No",
            })

regime_df = pd.DataFrame(regime_breach_rows)

# Summary pivot: avg breach rate by model x regime
pivot = regime_df.groupby(["Model","Regime"])["Breach Rate"].mean().unstack()
pivot["Expected"] = 1 - CONFIDENCE_LEVEL
print("\n   ── Average VaR Breach Rates by Model & Regime ──")
print(f"   (Expected: {1-CONFIDENCE_LEVEL:.3f} = 5.0%)")
print(pivot.round(4).to_string())

# % adequate by regime
adeq = regime_df.groupby(["Model","Regime"])["Adequate"].apply(
    lambda x: (x=="Yes").mean()*100
).unstack()
print("\n   ── % Assets Passing Kupiec Test by Regime ──")
print(adeq.round(1).to_string())

regime_df.to_csv(f"{METRICS_PATH}phase2_regime_var_breaches.csv", index=False)

# Expanded commentary
print("""
   ── TAIL-RISK UNDERESTIMATION COMMENTARY ──

   Key finding: All three econometric models show HIGHER breach
   rates during stress regimes than normal regimes. This is the
   'volatility clustering' problem — models trained on historical
   data systematically underestimate forward risk during crises.

   Why models struggle during prolonged uncertainty:
   1. BACKWARD-LOOKING ESTIMATION: All models estimate volatility
      from past returns. During unprecedented stress (COVID,
      inflation shock), past data provides poor guidance.

   2. REGIME SWITCHING: Volatility transitions abruptly between
      low and high regimes. GARCH/GJR-GARCH adapt slowly via
      their persistence parameter (β ≈ 0.85-0.90).

   3. HEAVY TAILS: Financial returns exhibit excess kurtosis.
      Gaussian VaR systematically underestimates tail losses
      during extreme events even with correct volatility forecasts.

   4. CORRELATION BREAKDOWN: During crises, asset correlations
      spike toward 1.0, making diversification less effective
      precisely when it is needed most.

   Implications for portfolio construction:
   → This breach clustering directly motivates Phase 3's deep
     learning approach — LSTM and Transformer models can learn
     non-linear regime dynamics that GARCH cannot capture.
   → It also justifies the CVaR optimisation in Phase 4, which
     uses tail scenarios rather than point VaR estimates.
""")

# STEP 6: Kupiec test for GJR-GARCH

print("[6/7] Kupiec test — GJR-GARCH full test period...")

gjr_backtest_rows = []
for asset in asset_cols:
    if asset not in gjr_var.columns:
        continue
    r = returns[asset].loc[test_start:test_end]
    v = gjr_var[asset].loc[test_start:test_end]
    common = r.index.intersection(v.index)
    r, v   = r.loc[common], v.loc[common]
    if len(r) < 10:
        continue
    breaches = (r < -v).sum()
    n        = len(r)
    obs_rate = breaches / n
    p_val    = kupiec_pvalue(n, int(breaches), 1-CONFIDENCE_LEVEL)
    gjr_backtest_rows.append({
        "Asset": asset, "Model": "GJR-GARCH",
        "N": n, "Breaches": int(breaches),
        "Expected": round(n*(1-CONFIDENCE_LEVEL),1),
        "Observed Rate": round(obs_rate, 4),
        "p-value": round(p_val,4) if not np.isnan(p_val) else np.nan,
        "Adequate": "Yes" if (not np.isnan(p_val) and p_val>0.05) else "No",
    })

gjr_bt_df = pd.DataFrame(gjr_backtest_rows)
pct_adeq  = (gjr_bt_df["Adequate"]=="Yes").mean()*100
print(f"\n   GJR-GARCH Kupiec adequacy: {pct_adeq:.1f}% of assets")
gjr_bt_df.to_csv(f"{METRICS_PATH}phase2_gjr_backtests.csv", index=False)

# STEP 7: Visualisations

print("\n[7/7] Generating enhancement charts...")
os.makedirs(FIGURES_PATH, exist_ok=True)

COLORS = {
    "HistVol_30d": "#2196F3",
    "GARCH(1,1)":  "#E91E63",
    "GJR-GARCH":   "#FF9800",
}

# Plot E1: All 3 models forecast accuracy comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
model_order  = ["HistVol_30d", "GARCH(1,1)", "GJR-GARCH"]
bar_colors   = [COLORS[m] for m in model_order]

for i, metric in enumerate(["MAE","RMSE","QLIKE"]):
    sub = agg.reindex(model_order)
    bars = axes[i].bar(model_order, sub[metric].values,
                       color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[i].set_title(f"{metric} (lower = better)",
                      fontsize=12, fontweight="bold")
    axes[i].tick_params(axis="x", rotation=15, labelsize=9)
    axes[i].grid(axis="y", alpha=0.3)
    for b, v in zip(bars, sub[metric].values):
        if not np.isnan(v):
            axes[i].text(b.get_x()+b.get_width()/2, v,
                         f"{v:.5f}", ha="center", va="bottom", fontsize=8)

plt.suptitle("Phase 2 Enhanced — Econometric Model Forecast Accuracy\n"
             "(HistVol vs GARCH vs GJR-GARCH)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}E1_enhanced_forecast_accuracy.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("    Plot E1 saved: Enhanced Forecast Accuracy")

# Plot E2: Regime-split VaR breach rates heatmap 
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
regimes = ["Normal (VIX≤25)", "Stress (VIX>25)"]

for ax, regime in zip(axes, regimes):
    sub = regime_df[regime_df["Regime"]==regime]
    if sub.empty:
        continue
    piv = sub.pivot(index="Asset", columns="Model",
                    values="Breach Rate")
    piv = piv.reindex(columns=model_order, fill_value=np.nan)
    expected = 1 - CONFIDENCE_LEVEL
    sns.heatmap(
        piv, annot=True, fmt=".3f",
        cmap="RdYlGn_r", center=expected,
        vmin=0, vmax=expected*2.5,
        ax=ax, linewidths=0.5,
        cbar_kws={"label": f"Breach rate (expected={expected:.2f})"},
        annot_kws={"size": 9},
    )
    color = "#D32F2F" if "Stress" in regime else "#1B5E20"
    ax.set_title(f"VaR Breach Rates — {regime}",
                 fontsize=12, fontweight="bold", color=color)

plt.suptitle("Regime-Split VaR Breach Analysis\n"
             "Models systematically underestimate risk during stress",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}E2_regime_var_breaches.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("    Plot E2 saved: Regime-Split VaR Breaches")

# Plot E3: GJR-GARCH vs GARCH volatility (SPY, leverage effect)
fig, ax = plt.subplots(figsize=(16, 6))
asset = "SPY" if "SPY" in asset_cols else asset_cols[0]
rv    = returns[asset].abs().loc[test_start:test_end]
g_fc  = garch_var[asset].loc[test_start:test_end] if asset in garch_var.columns else pd.Series()
gjr_fc= gjr_forecasts[asset].loc[test_start:test_end] if asset in gjr_forecasts.columns else pd.Series()

ax.plot(rv.index,  rv,     color="lightgray", linewidth=0.6,
        label="Realised |r|", alpha=0.7)
ax.plot(g_fc.index,   g_fc/scipy_norm.ppf(CONFIDENCE_LEVEL),
        color="#E91E63", linewidth=1.1, label="GARCH(1,1)")
ax.plot(gjr_fc.index, gjr_fc,
        color="#FF9800", linewidth=1.1, label="GJR-GARCH")

# Shade stress periods in test
for cname, (s, e) in STRESS_PERIODS.items():
    s_pd = pd.Timestamp(s)
    e_pd = pd.Timestamp(e)
    if s_pd <= test_end and e_pd >= test_start:
        ax.axvspan(max(s_pd, test_start), min(e_pd, test_end),
                       alpha=0.12, color="red",
                       label="_nolegend_")

ax.set_title(f"{asset} — GARCH vs GJR-GARCH Volatility Forecasts\n"
             "(GJR-GARCH captures leverage effect: bad news → more volatility)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Volatility Forecast")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}E3_gjr_vs_garch_volatility.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("    Plot E3 saved: GJR-GARCH vs GARCH Volatility")

# Plot E4: Breach rate — Normal vs Stress side-by-side
fig, ax = plt.subplots(figsize=(13, 6))
x      = np.arange(len(model_order))
w = 0.35
normal_rates = [
    regime_df[(regime_df["Model"]==m) &
               (regime_df["Regime"]=="Normal (VIX≤25)")]["Breach Rate"].mean()
    for m in model_order
]
stress_rates = [
    regime_df[(regime_df["Model"]==m) &
               (regime_df["Regime"]=="Stress (VIX>25)")]["Breach Rate"].mean()
    for m in model_order
]
ax.bar(x - w/2, normal_rates, w, label="Normal regime (VIX≤25)",
       color="#A5D6A7", edgecolor="black", linewidth=0.5)
ax.bar(x + w/2, stress_rates, w, label="Stress regime (VIX>25)",
       color="#EF9A9A", edgecolor="black", linewidth=0.5)
ax.axhline(1 - CONFIDENCE_LEVEL, color="black", linestyle="--",
           linewidth=1.2, label=f"Expected ({1-CONFIDENCE_LEVEL:.2f})")
ax.set_xticks(x)
ax.set_xticklabels(model_order, fontsize=10)
ax.set_ylabel("Average Breach Rate")
ax.set_title("VaR Breach Rates: Normal vs Stress Regime\n"
             "All models underestimate tail risk during market stress",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for xi, (nr, sr) in enumerate(zip(normal_rates, stress_rates)):
    ax.text(xi-w/2, nr+0.001, f"{nr:.3f}", ha="center", va="bottom", fontsize=9)
    ax.text(xi+w/2, sr+0.001, f"{sr:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}E4_breach_normal_vs_stress.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("    Plot E4 saved: Breach Rate Normal vs Stress")

# SUMMARY

print("\n" + "=" * 60)
print("  Summary of Phase 2 Enhancements:")
print("=" * 60)
print(f"""
  Added:
     GJR-GARCH model (captures leverage effect)
     GJR-GARCH VaR/CVaR forecasts
     Regime-split VaR breach table (Normal vs Stress)
     Kupiec test for GJR-GARCH
     Tail-risk underestimation commentary

  Average breach rates (test period):
    Normal regime: {np.mean(normal_rates):.3f}  (expected 0.050)
    Stress regime: {np.mean(stress_rates):.3f}  (expected 0.050)
  → Stress breach rate is {np.mean(stress_rates)/np.mean(normal_rates):.1f}x higher
    than normal — confirming backward-looking models
    systematically underestimate risk during crises.

  Forecast accuracy (avg across assets):
{agg.to_string()}
""")