import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

from config.config import (
    PROCESSED_DATA_PATH, METRICS_PATH, FIGURES_PATH,
    RISK_FREE_RATE, STRESS_PERIODS,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# CONSTANTS

TRADING_DAYS = 252
rf_daily     = RISK_FREE_RATE / TRADING_DAYS

# Bootstrap configuration
BOOTSTRAP_N_ITER  = 1000
BOOTSTRAP_BLOCK   = 5     # days — preserves serial dependence
BOOTSTRAP_SEED    = 42

COLORS = {
    "EqualWeight"     : "#E91E63",
    "CVaR_Historical" : "#FF9800",
    "CVaR_HistVol"    : "#4CAF50",
    "CVaR_GARCH"      : "#00BCD4",
    "CVaR_LSTM"       : "#2196F3",
    "CVaR_Transformer": "#9C27B0",
}

# STATISTICAL HELPERS

def block_bootstrap_means(
    data: np.ndarray,
    n_iter: int = BOOTSTRAP_N_ITER,
    block_size: int = BOOTSTRAP_BLOCK,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(data)
    if n == 0 or block_size <= 0:
        return np.array([])
    n_blocks = int(np.ceil(n / block_size))
    means = np.empty(n_iter)
    for i in range(n_iter):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        boot = np.concatenate([data[s : s + block_size] for s in starts])[:n]
        means[i] = boot.mean()
    return means


def jobson_korkie_test(
    returns_a: pd.Series, returns_b: pd.Series, rf: float = RISK_FREE_RATE
) -> dict:
    """
    Jobson-Korkie (1981) test with Memmel (2003) correction.
    Tests H0: Sharpe(A) = Sharpe(B).
    Returns z-statistic, p-value, and the annualised Sharpe difference.
    """
    a = returns_a.dropna()
    b = returns_b.dropna()
    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]
    n = len(common)
    if n < 30:
        return {"sharpe_a": np.nan, "sharpe_b": np.nan, "difference": np.nan,
                "z_stat": np.nan, "p_value": np.nan, "significant_5pct": False, "n": n}

    daily_rf = rf / TRADING_DAYS
    mu_a = (a - daily_rf).mean()
    mu_b = (b - daily_rf).mean()
    var_a = a.var()
    var_b = b.var()
    cov_ab = np.cov(a, b, ddof=1)[0, 1]
    sig_a = np.sqrt(var_a)
    sig_b = np.sqrt(var_b)

    sh_a = mu_a / sig_a
    sh_b = mu_b / sig_b

    # Memmel-corrected variance of the daily Sharpe difference
    theta = (
        2 * var_a * var_b
        - 2 * sig_a * sig_b * cov_ab
        + 0.5 * sh_a ** 2 * var_b ** 2
        + 0.5 * sh_b ** 2 * var_a ** 2
        - sh_a * sh_b * cov_ab ** 2
    ) / (var_a * var_b)

    # Annualised Sharpes
    sh_a_ann = sh_a * np.sqrt(TRADING_DAYS)
    sh_b_ann = sh_b * np.sqrt(TRADING_DAYS)

    variance = theta / n
    if variance <= 0:
        return {"sharpe_a": sh_a_ann, "sharpe_b": sh_b_ann,
                "difference": sh_a_ann - sh_b_ann,
                "z_stat": np.nan, "p_value": np.nan,
                "significant_5pct": False, "n": n}

    z_stat = (sh_a_ann - sh_b_ann) / np.sqrt(TRADING_DAYS * variance)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        "sharpe_a": float(sh_a_ann),
        "sharpe_b": float(sh_b_ann),
        "difference": float(sh_a_ann - sh_b_ann),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "significant_5pct": bool(p_value < 0.05),
        "n": int(n),
    }


def pairwise_sharpe_tests(
    port_dict: dict[str, pd.Series], rf: float = RISK_FREE_RATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pairwise Sharpe difference and p-value matrices."""
    names = list(port_dict.keys())
    diff_mat = pd.DataFrame(index=names, columns=names, dtype=object)
    pval_mat = pd.DataFrame(index=names, columns=names, dtype=float)

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                diff_mat.iloc[i, j] = "—"
                pval_mat.iloc[i, j] = np.nan
                continue
            res = jobson_korkie_test(port_dict[a], port_dict[b], rf=rf)
            diff = res["difference"]
            pv = res["p_value"]
            sig = "*" if (pd.notna(pv) and pv < 0.05) else " "
            diff_mat.iloc[i, j] = f"{diff:+.3f}{sig}"
            pval_mat.iloc[i, j] = pv
    return diff_mat, pval_mat

# MAIN

print("=" * 60)
print("  PHASE 5: Backtesting & Performance Evaluation")
print("=" * 60)

# STEP 1: Load Phase 4 outputs

print("\n[1/8] Loading Phase 4 portfolio returns...")

try:
    port_ret = pd.read_csv(
        f"{PROCESSED_DATA_PATH}phase4_portfolio_returns.csv",
        index_col=0, parse_dates=True,
    )
except FileNotFoundError:
    raise SystemExit(
        "   phase4_portfolio_returns.csv not found.\n"
        "   Run phase4_portfolio_optimiser.py first."
    )

# Drop any leading zero rows (warm-up period before backtest starts)
port_ret = port_ret[port_ret.abs().sum(axis=1) > 0]

# Load VIX for regime identification
try:
    vix = pd.read_csv(
        f"{PROCESSED_DATA_PATH}vix.csv", index_col=0, parse_dates=True
    )
    vix = vix.reindex(port_ret.index).ffill().bfill()
    has_vix = True
except FileNotFoundError:
    has_vix = False
    print("VIX data not found — stress regime analysis will use date ranges only")

strategies = list(port_ret.columns)
test_start = port_ret.index[0]
test_end   = port_ret.index[-1]

print(f"Loaded {len(strategies)} strategies")
print(f"Strategies: {strategies}")
print(f"Test period: {test_start.date()} → {test_end.date()} ({len(port_ret):,} days)")

# STEP 2: Comprehensive performance metrics

print("\n[2/8] Computing comprehensive performance metrics...")


def compute_full_metrics(returns: pd.Series, name: str) -> dict:
    r   = returns.dropna()
    cum = (1 + r).cumprod()

    # Return metrics 
    total_ret = cum.iloc[-1] - 1
    ann_ret   = (1 + total_ret) ** (TRADING_DAYS / len(r)) - 1
    ann_vol   = r.std() * np.sqrt(TRADING_DAYS)

    # Sharpe
    sharpe = (r.mean() - rf_daily) / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0

    # Sortino (CONVENTIONAL: uses returns < 0) 
    downside = r[r < 0]
    down_std = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 1 else np.nan
    sortino = (ann_ret - RISK_FREE_RATE) / down_std if (down_std and down_std > 0) else np.nan

    # Drawdown 
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd   = float(drawdown.min())
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Peak-to-trough days (current MDD only)
    dd_end_trough = drawdown.idxmin()
    dd_start = cum[:dd_end_trough].idxmax()
    peak_to_trough = (dd_end_trough - dd_start).days

    # Peak-to-recovery days (None if never recovered in window)
    after = cum.loc[dd_end_trough:]
    peak_value = cum.loc[dd_start]
    recovery_mask = after >= peak_value
    if recovery_mask.any():
        recovery_date = after[recovery_mask].index[0]
        peak_to_recovery = (recovery_date - dd_start).days
    else:
        peak_to_recovery = -1   # not recovered by end of test period

    # Tail risk 
    var_95  = float(r.quantile(0.05))
    cvar_95 = float(r[r <= var_95].mean()) if (r <= var_95).any() else var_95
    var_99  = float(r.quantile(0.01))
    cvar_99 = float(r[r <= var_99].mean()) if (r <= var_99).any() else var_99

    # Omega ratio (threshold = rf_daily)
    gains  = (r[r > rf_daily] - rf_daily).sum()
    losses = (rf_daily - r[r <= rf_daily]).sum()
    omega  = gains / losses if losses > 0 else np.nan

    # Hit rate
    hit_rate = (r > 0).mean()

    # Skewness & Excess Kurtosis
    skew = float(r.skew())
    kurt = float(r.kurtosis())

    # Jarque-Bera normality test
    if len(r) > 30:
        jb_p = float(stats.jarque_bera(r)[1])
    else:
        jb_p = np.nan

    return {
        "Strategy"              : name,
        "Total Return (%)"      : round(total_ret * 100, 2),
        "Ann. Return (%)"       : round(ann_ret   * 100, 2),
        "Ann. Volatility (%)"   : round(ann_vol  * 100, 2),
        "Sharpe Ratio"          : round(sharpe,           4),
        "Sortino Ratio"         : round(sortino,           4) if pd.notna(sortino) else np.nan,
        "Calmar Ratio"          : round(calmar,            4),
        "Omega Ratio"           : round(omega,             4) if pd.notna(omega) else np.nan,
        "Max Drawdown (%)"      : round(max_dd   * 100,   2),
        "MDD Peak-to-Trough (d)": int(peak_to_trough),
        "MDD Peak-to-Recovery (d)": int(peak_to_recovery),
        "VaR 95% (%)"           : round(var_95   * 100,   4),
        "CVaR 95% (%)"          : round(cvar_95  * 100,   4),
        "VaR 99% (%)"           : round(var_99   * 100,   4),
        "CVaR 99% (%)"          : round(cvar_99  * 100,   4),
        "Hit Rate (%)"          : round(hit_rate * 100,    2),
        "Skewness"              : round(skew,              4),
        "Excess Kurtosis"       : round(kurt,              4),
        "Jarque-Bera p-value"   : round(jb_p,              4) if pd.notna(jb_p) else np.nan,
    }


metrics_list = [compute_full_metrics(port_ret[s], s) for s in strategies]
metrics_df   = pd.DataFrame(metrics_list).set_index("Strategy")

print("\n   Headline Performance Metrics:")
display_cols = ["Ann. Return (%)", "Ann. Volatility (%)", "Sharpe Ratio",
                "Sortino Ratio", "Max Drawdown (%)", "CVaR 95% (%)", "Omega Ratio"]
print(metrics_df[display_cols].to_string())

# STEP 3: Statistical significance tests (vs benchmark)

print("\n[3/8] Running significance tests vs EqualWeight (BLOCK bootstrap)...")
print(f"      Bootstrap iterations: {BOOTSTRAP_N_ITER}, block size: {BOOTSTRAP_BLOCK} days")

benchmark = "EqualWeight" if "EqualWeight" in strategies else strategies[0]
bench_ret = port_ret[benchmark].dropna()
sig_rows  = []

for strat in strategies:
    if strat == benchmark:
        continue
    s_ret  = port_ret[strat].dropna()
    common = s_ret.index.intersection(bench_ret.index)
    diff   = (s_ret.loc[common] - bench_ret.loc[common]).dropna()

    if len(diff) < 30:
        continue

    # Paired t-test on daily excess returns
    t_stat, p_val = stats.ttest_1samp(diff, 0)

    # BLOCK bootstrap (preserves serial dependence)
    boot_means = block_bootstrap_means(diff.values)
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)

    sig_rows.append({
        "Strategy"              : strat,
        "Mean Excess Return"    : round(diff.mean() * TRADING_DAYS, 4),
        "t-statistic"           : round(t_stat, 4),
        "p-value"               : round(p_val,  4),
        "Significant (5%)"      : "Yes" if p_val < 0.05 else "No",
        "Block-Bootstrap CI Low": round(ci_lo * TRADING_DAYS, 4),
        "Block-Bootstrap CI High": round(ci_hi * TRADING_DAYS, 4),
    })

sig_df = pd.DataFrame(sig_rows).set_index("Strategy")
print(f"\n   Significance Tests vs {benchmark} (annualised excess returns):")
print(sig_df.to_string())


# STEP 3b: Pairwise Sharpe tests (Jobson-Korkie)
print("\n[3b/8] Pairwise Jobson-Korkie Sharpe difference tests...")

port_dict = {s: port_ret[s].dropna() for s in strategies}
diff_mat, pval_mat = pairwise_sharpe_tests(port_dict, rf=RISK_FREE_RATE)

print("\n   Pairwise Sharpe Differences (row - column, * = p<0.05):")
print(diff_mat.to_string())

# Key comparisons of interest for the dissertation
key_pairs = []
if "CVaR_Transformer" in strategies and "CVaR_GARCH" in strategies:
    key_pairs.append(("CVaR_Transformer", "CVaR_GARCH"))
if "CVaR_LSTM" in strategies and "CVaR_GARCH" in strategies:
    key_pairs.append(("CVaR_LSTM", "CVaR_GARCH"))
if "CVaR_Transformer" in strategies and "CVaR_Historical" in strategies:
    key_pairs.append(("CVaR_Transformer", "CVaR_Historical"))
if "CVaR_LSTM" in strategies and "CVaR_Historical" in strategies:
    key_pairs.append(("CVaR_LSTM", "CVaR_Historical"))
if "EqualWeight" in strategies and "CVaR_Transformer" in strategies:
    key_pairs.append(("EqualWeight", "CVaR_Transformer"))

print("\n   Key Pairwise Comparisons:")
for a, b in key_pairs:
    res = jobson_korkie_test(port_ret[a], port_ret[b], rf=RISK_FREE_RATE)
    sig = " SIGNIFICANT (p<0.05)" if res["significant_5pct"] else " not significant"
    print(f"     {a:<20} vs {b:<20}  "
          f"ΔSharpe = {res['difference']:+.3f}  "
          f"p = {res['p_value']:.3f}  {sig}")

# STEP 4: Year-by-year performance

print("\n[4/8] Computing year-by-year performance...")

years    = sorted(port_ret.index.year.unique())
yoy_rows = []

for yr in years:
    yr_ret = port_ret[port_ret.index.year == yr]
    for strat in strategies:
        r = yr_ret[strat].dropna()
        if len(r) < 30:
            continue
        ann_ret = (1 + r).prod() ** (TRADING_DAYS / max(len(r), 1)) - 1
        sharpe  = (r.mean() - rf_daily) / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else 0
        cum     = (1 + r).cumprod()
        max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
        yoy_rows.append({
            "Year": yr, "Strategy": strat,
            "Ann. Return (%)": round(ann_ret * 100, 2),
            "Sharpe": round(sharpe, 3),
            "Max DD (%)": round(max_dd * 100, 2),
        })

yoy_df = pd.DataFrame(yoy_rows)
yoy_pivot_ret = yoy_df.pivot(index="Strategy", columns="Year", values="Ann. Return (%)")
yoy_pivot_sh  = yoy_df.pivot(index="Strategy", columns="Year", values="Sharpe")

print("\n   Year-by-Year Annualised Returns (%):")
print(yoy_pivot_ret.to_string())
print("\n   Year-by-Year Sharpe Ratios:")
print(yoy_pivot_sh.to_string())

# STEP 5: Regime-conditional analysis

print("\n[5/8] Regime-conditional analysis (Normal vs Stress)...")

regime_rows = []
for strat in strategies:
    r = port_ret[strat].dropna()

    for regime_name, (s_date, e_date) in STRESS_PERIODS.items():
        s_mask = (r.index >= pd.Timestamp(s_date)) & (r.index <= pd.Timestamp(e_date))
        stress_r = r[s_mask]
        if len(stress_r) < 5:
            continue
        cum    = (1 + stress_r).prod() - 1
        vol    = stress_r.std() * np.sqrt(TRADING_DAYS)
        max_dd = ((1 + stress_r).cumprod() /
                  (1 + stress_r).cumprod().cummax() - 1).min()
        regime_rows.append({
            "Strategy": strat, "Regime": regime_name,
            "Cumulative Return (%)": round(cum * 100, 2),
            "Ann. Volatility (%)":   round(vol * 100, 2),
            "Max Drawdown (%)":       round(max_dd * 100, 2),
        })

    # "Normal" = everything outside ALL declared stress periods
    stress_mask = pd.Series(False, index=r.index)
    for s_date, e_date in STRESS_PERIODS.values():
        stress_mask |= (r.index >= pd.Timestamp(s_date)) & (r.index <= pd.Timestamp(e_date))
    normal_r = r[~stress_mask]
    if len(normal_r) > 5:
        cum    = (1 + normal_r).prod() - 1
        vol    = normal_r.std() * np.sqrt(TRADING_DAYS)
        max_dd = ((1 + normal_r).cumprod() /
                  (1 + normal_r).cumprod().cummax() - 1).min()
        regime_rows.append({
            "Strategy": strat, "Regime": "Normal",
            "Cumulative Return (%)": round(cum * 100, 2),
            "Ann. Volatility (%)":   round(vol * 100, 2),
            "Max Drawdown (%)":       round(max_dd * 100, 2),
        })

regime_df = pd.DataFrame(regime_rows)
regime_pivot = regime_df.pivot(
    index="Strategy", columns="Regime", values="Cumulative Return (%)"
)
print("\n   Cumulative Returns by Market Regime (%):")
print(regime_pivot.to_string())

# STEP 6: Save all outputs

print("\n[6/8] Saving outputs...")

os.makedirs(METRICS_PATH, exist_ok=True)
metrics_df.to_csv (f"{METRICS_PATH}phase5_full_metrics.csv")
sig_df.to_csv     (f"{METRICS_PATH}phase5_significance_tests.csv")
diff_mat.to_csv   (f"{METRICS_PATH}phase5_pairwise_sharpe_diff.csv")
pval_mat.to_csv   (f"{METRICS_PATH}phase5_pairwise_pvalues.csv")
yoy_df.to_csv     (f"{METRICS_PATH}phase5_year_by_year.csv",   index=False)
regime_df.to_csv  (f"{METRICS_PATH}phase5_regime_analysis.csv", index=False)
print("    All metrics saved")

# STEP 7: Visualisations

print("\n[7/8] Generating charts...")
os.makedirs(FIGURES_PATH, exist_ok=True)

equity_curves = pd.DataFrame({s: (1 + port_ret[s]).cumprod() for s in strategies})

# ── Plot 19: Full Metrics Heatmap ──
fig, ax = plt.subplots(figsize=(14, 6))
heatmap_cols = ["Ann. Return (%)", "Ann. Volatility (%)", "Sharpe Ratio",
                "Sortino Ratio", "Max Drawdown (%)", "CVaR 95% (%)",
                "Omega Ratio", "Hit Rate (%)"]
hm_data = metrics_df[heatmap_cols].copy().astype(float)
hm_norm = hm_data.copy()
for col in hm_norm.columns:
    mn, mx = hm_norm[col].min(), hm_norm[col].max()
    if mx != mn:
        hm_norm[col] = (hm_norm[col] - mn) / (mx - mn)
sns.heatmap(
    hm_norm, annot=hm_data.round(2), fmt="g",
    cmap="RdYlGn", ax=ax, linewidths=0.5,
    cbar_kws={"label": "Normalised Score (green = better)"},
    annot_kws={"size": 9},
)
ax.set_title("Phase 5 — Full Performance Metrics Heatmap",
             fontsize=13, fontweight="bold")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=30, labelsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}19_metrics_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("    All metrics saved")


# Plot 20: Year-by-Year Returns Heatmap
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    yoy_pivot_ret.astype(float), annot=True, fmt=".1f",
    cmap="RdYlGn", center=0, ax=ax,
    linewidths=0.5, annot_kws={"size": 10},
    cbar_kws={"label": "Annualised Return (%)"},
)
ax.set_title("Year-by-Year Annualised Returns (%) — All Strategies",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}20_year_by_year_returns.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Plot 20 saved: Year-by-Year Returns")


# Plot 21: Regime Analysis Bar Chart 
regimes_to_plot = [c for c in regime_pivot.columns if c in list(STRESS_PERIODS.keys()) + ["Normal"]]
n_plots = min(len(regimes_to_plot), 2)
fig, axes = plt.subplots(1, n_plots, figsize=(9 * n_plots, 6))
if n_plots == 1:
    axes = [axes]
for ax, regime in zip(axes, regimes_to_plot[:n_plots]):
    vals   = regime_pivot[regime].sort_values()
    colors = [COLORS.get(s, "#888") for s in vals.index]
    bars   = ax.bar(vals.index, vals.values, color=colors,
                    edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Cumulative Return (%) — {regime}",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals.values):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
plt.suptitle("Strategy Returns by Market Regime",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}21_regime_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Plot 21 saved: Regime Analysis")


# ── Plot 22: Rolling 63-day Sharpe ──
fig, ax = plt.subplots(figsize=(16, 6))
for strat in strategies:
    r      = port_ret[strat].dropna()
    roll_s = r.rolling(63).apply(
        lambda x: (x.mean() - rf_daily) / x.std() * np.sqrt(TRADING_DAYS)
        if x.std() > 0 else 0
    )
    ax.plot(roll_s.index, roll_s,
            label=strat, color=COLORS.get(strat, "#888"),
            linewidth=1.8 if "Transformer" in strat else 1.0,
            alpha=0.9)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
for regime_name, (s, e) in STRESS_PERIODS.items():
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.08, color="red")
ax.set_title("Rolling 63-Day Sharpe Ratio — All Strategies",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Annualised Sharpe Ratio")
ax.legend(fontsize=9, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}22_rolling_sharpe.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Plot 22 saved: Rolling Sharpe")


# Plot 23: Drawdowns with stress shading 
fig, ax = plt.subplots(figsize=(16, 7))
for strat in strategies:
    r      = port_ret[strat].dropna()
    cum    = (1 + r).cumprod()
    dd     = (cum - cum.cummax()) / cum.cummax() * 100
    ax.fill_between(dd.index, dd, 0, alpha=0.12, color=COLORS.get(strat, "#888"))
    ax.plot(dd.index, dd, label=strat,
            color=COLORS.get(strat, "#888"),
            linewidth=2.0 if "Transformer" in strat else 1.0)
for regime_name, (s, e) in STRESS_PERIODS.items():
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
               alpha=0.08, color="red", label="_nolegend_")
    ax.annotate(regime_name.replace("_", "\n"),
                xy=(pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2,
                    ax.get_ylim()[0] * 0.85),
                fontsize=7, ha="center", color="darkred", fontweight="bold")
ax.set_title("Portfolio Drawdowns — Full Test Period with Stress Regimes",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}23_drawdowns_with_regimes.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Plot 23 saved: Drawdowns with Stress Regimes")


#  Plot 24: Statistical significance bar chart (vs EqualWeight)
fig, ax = plt.subplots(figsize=(12, 5))
sig_strats = sig_df.index.tolist()
bar_colors = [COLORS.get(s, "#888") for s in sig_strats]
bars = ax.bar(sig_strats, sig_df["Mean Excess Return"],
              color=bar_colors, edgecolor="black", linewidth=0.5)
for b, (_, row) in zip(bars, sig_df.iterrows()):
    stars = ("***" if row["p-value"] < 0.01
             else "**" if row["p-value"] < 0.05
             else "*" if row["p-value"] < 0.10
             else "ns")
    ax.text(b.get_x() + b.get_width() / 2,
            row["Mean Excess Return"],
            f"{stars}\np={row['p-value']:.3f}",
            ha="center",
            va="bottom" if row["Mean Excess Return"] >= 0 else "top",
            fontsize=9, fontweight="bold")
    # CI as error bar
    lo = row["Block-Bootstrap CI Low"]
    hi = row["Block-Bootstrap CI High"]
    ax.plot([b.get_x() + b.get_width() / 2] * 2,
            [lo, hi], color="black", linewidth=1.5, alpha=0.6)
    ax.plot([b.get_x() + b.get_width() / 2 - 0.05,
             b.get_x() + b.get_width() / 2 + 0.05], [lo, lo],
            color="black", linewidth=1.5, alpha=0.6)
    ax.plot([b.get_x() + b.get_width() / 2 - 0.05,
             b.get_x() + b.get_width() / 2 + 0.05], [hi, hi],
            color="black", linewidth=1.5, alpha=0.6)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title(f"Mean Annualised Excess Return vs {benchmark}\n"
             f"(Error bars = 95% block-bootstrap CI; "
             f"*** p<0.01, ** p<0.05, * p<0.10, ns = not significant)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Excess Return (annualised)")
ax.tick_params(axis="x", rotation=20, labelsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}24_significance_tests.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Plot 24 saved: Statistical Significance Tests")


# ── Plot 25: Equity curves with stress shading ──
fig, ax = plt.subplots(figsize=(16, 7))
for strat in strategies:
    lw = 2.2 if "Transformer" in strat else 1.2
    ls = "--" if strat == "EqualWeight" else "-"
    ax.plot(equity_curves.index, equity_curves[strat],
            label=strat, color=COLORS.get(strat, "#888"),
            linewidth=lw, linestyle=ls)
for _, (s, e) in STRESS_PERIODS.items():
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.08, color="red")
ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
ax.set_title("Equity Curves — All Strategies with Stress Period Overlay",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Portfolio Value (start = $1.00)")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}25_equity_curves_stress.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Plot 25 saved: Equity Curves with Stress Overlay")


# ── Plot 26 (NEW): Pairwise Sharpe Significance Heatmap ──
fig, ax = plt.subplots(figsize=(10, 8))
pval_display = pval_mat.astype(float)
mask = pval_display.isna()
sns.heatmap(
    pval_display,
    annot=True, fmt=".3f",
    cmap="RdYlGn_r",
    center=0.05, vmin=0, vmax=0.2,
    mask=mask, linewidths=0.5,
    cbar_kws={"label": "p-value (< 0.05 = significant)"},
    ax=ax,
)
ax.set_title("Pairwise Sharpe Difference p-values (Jobson-Korkie)\n"
             "Green = significant difference, Red = not significant",
             fontsize=12, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="x", rotation=30, labelsize=9)
ax.tick_params(axis="y", rotation=0,  labelsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}26_pairwise_sharpe_significance.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Plot 26 saved: Pairwise Sharpe Significance Heatmap")


# STEP 8: Final Summary

print("\n[8/8] Building final summary...")

best_sharpe  = metrics_df["Sharpe Ratio"].idxmax()
best_sortino = metrics_df["Sortino Ratio"].idxmax()
best_mdd     = metrics_df["Max Drawdown (%)"].idxmax()      # closest to 0
best_cvar    = metrics_df["CVaR 95% (%)"].idxmax()          # closest to 0
best_omega   = metrics_df["Omega Ratio"].idxmax()

print("\n" + "=" * 60)
print("  PHASE 5 COMPLETE ")
print("=" * 60)
print(f"""
  ── Winners by metric ──
    Best Sharpe Ratio   : {best_sharpe:<20}  ({metrics_df.loc[best_sharpe,'Sharpe Ratio']:.4f})
    Best Sortino Ratio  : {best_sortino:<20}  ({metrics_df.loc[best_sortino,'Sortino Ratio']:.4f})
    Best Max Drawdown   : {best_mdd:<20}  ({metrics_df.loc[best_mdd,'Max Drawdown (%)']:.2f}%)
    Best CVaR 95%       : {best_cvar:<20}  ({metrics_df.loc[best_cvar,'CVaR 95% (%)']:.4f}%)
    Best Omega Ratio    : {best_omega:<20}  ({metrics_df.loc[best_omega,'Omega Ratio']:.4f})

  ── Full metrics summary ──
{metrics_df[["Ann. Return (%)","Sharpe Ratio","Sortino Ratio","Max Drawdown (%)","CVaR 95% (%)","Omega Ratio"]].to_string()}

  ── Significance vs EqualWeight (block bootstrap + t-test) ──
{sig_df[["Mean Excess Return","t-statistic","p-value","Significant (5%)"]].to_string()}
""")