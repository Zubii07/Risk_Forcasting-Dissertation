"""
Compares all three experiments side by side:
  Experiment 1 — Historical (baseline)
  Experiment 2 — Forecast-Driven (per model)
  Experiment 3 — Forecast-Driven + A+C (primary model)

Also produces:
  - Computational performance comparison
  - Weight allocation heatmaps
  - Cumulative return comparison chart
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from config.config import PROCESSED_DATA_PATH, METRICS_PATH, FIGURES_PATH
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


plt.style.use("seaborn-v0_8-darkgrid")
TRADING_DAYS = 252
rf_daily = 0.02 / TRADING_DAYS

print("=" * 60)
print("  PHASE 4 — COMPARISON: All Experiments")
print("=" * 60)

# STEP 1: Load all experiment outputs
print("\n[1/6] Loading all experiment outputs...")

exp1_ret = pd.read_csv(f"{PROCESSED_DATA_PATH}exp1_historical_returns.csv",
                       index_col=0, parse_dates=True).iloc[:, 0]

exp2_files = [f for f in os.listdir(PROCESSED_DATA_PATH)
              if f.startswith("exp2_forecast_") and f.endswith("_returns.csv")]
exp2_returns = {}
for f in exp2_files:
    model = f.replace("exp2_forecast_", "").replace("_returns.csv", "")
    exp2_returns[model] = pd.read_csv(f"{PROCESSED_DATA_PATH}{f}",
                                      index_col=0, parse_dates=True).iloc[:, 0]

exp3_path = f"{PROCESSED_DATA_PATH}exp3_forecast_ac_returns.csv"
exp3_ret = None
if os.path.exists(exp3_path):
    exp3_ret = pd.read_csv(exp3_path, index_col=0, parse_dates=True).iloc[:, 0]

print(f" Exp1 (Historical): {len(exp1_ret)} days")
print(f" Exp2 (Forecast-Driven): {len(exp2_returns)} models")
print(f" Exp3 (Forecast + A+C): {'loaded' if exp3_ret is not None else 'not found'}")

# STEP 2: Unified performance table
print("\n[2/6] Building unified performance comparison...")

def perf_metrics(r):
    r = r.dropna()
    cum = (1+r).cumprod()
    ann_ret = cum.iloc[-1] ** (TRADING_DAYS/len(r)) - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe  = (r.mean()-rf_daily)/r.std()*np.sqrt(TRADING_DAYS) if r.std()>0 else 0
    mdd     = ((cum-cum.cummax())/cum.cummax()).min()
    v95     = r.quantile(0.05)
    cv95    = r[r<=v95].mean()
    return {
        "Ann. Return (%)": round(ann_ret*100, 2),
        "Ann. Vol (%)":    round(ann_vol*100, 2),
        "Sharpe":          round(sharpe, 4),
        "Max DD (%)":      round(mdd*100, 2),
        "CVaR 95% (%)":    round(cv95*100, 4),
    }

comparison_rows = []
comparison_rows.append({"Experiment": "Exp1 — Historical (Baseline)", **perf_metrics(exp1_ret)})
for model, r in exp2_returns.items():
    comparison_rows.append({"Experiment": f"Exp2 — Forecast-Driven ({model})", **perf_metrics(r)})
if exp3_ret is not None:
    comparison_rows.append({"Experiment": "Exp3 — Forecast-Driven + A+C", **perf_metrics(exp3_ret)})

comparison_df = pd.DataFrame(comparison_rows).set_index("Experiment")
print("\n" + comparison_df.to_string())
comparison_df.to_csv(f"{METRICS_PATH}phase4_all_experiments_comparison.csv")

# STEP 3: Computational performance comparison
print("\n[3/6] Computational performance comparison...")

timing_rows = []
exp1_summary = pd.read_csv(f"{METRICS_PATH}exp1_historical_summary.csv")
timing_rows.append({
    "Experiment": "Exp1 — Historical",
    "Avg Time/Rebalance (ms)": exp1_summary["Avg Opt Time/Rebalance (ms)"].iloc[0],
    "Total Time (s)": exp1_summary["Total Opt Time (s)"].iloc[0],
    "Inputs": "252-day historical scenarios",
})

exp2_timing_path = f"{METRICS_PATH}exp2_timing.csv"
if os.path.exists(exp2_timing_path):
    exp2_timing = pd.read_csv(exp2_timing_path)
    for _, row in exp2_timing.iterrows():
        timing_rows.append({
            "Experiment": f"Exp2 — {row['Model']}",
            "Avg Time/Rebalance (ms)": row["Avg Opt Time/Rebalance (ms)"],
            "Total Time (s)": row["Total Opt Time (s)"],
            "Inputs": "CVaR forecast + fixed correlation",
        })

exp3_summary_path = f"{METRICS_PATH}exp3_forecast_ac_summary.csv"
if os.path.exists(exp3_summary_path):
    exp3_summary = pd.read_csv(exp3_summary_path)
    timing_rows.append({
        "Experiment": "Exp3 — Forecast + A+C",
        "Avg Time/Rebalance (ms)": exp3_summary["Avg Opt Time/Rebalance (ms)"].iloc[0],
        "Total Time (s)": exp3_summary["Total Opt Time (s)"].iloc[0],
        "Inputs": "CVaR forecast + correlation + EWMA return + regime",
    })

timing_df = pd.DataFrame(timing_rows)
print("\n" + timing_df.to_string(index=False))
timing_df.to_csv(f"{METRICS_PATH}phase4_computational_comparison.csv", index=False)

speedup = timing_df.loc[timing_df["Experiment"]=="Exp1 — Historical", "Avg Time/Rebalance (ms)"].iloc[0]
if len(exp2_timing) > 0:
    exp2_avg = exp2_timing["Avg Opt Time/Rebalance (ms)"].mean()
    print(f"\n   Forecast-driven optimisation is "
          f"{speedup/exp2_avg:.1f}x faster per rebalance than historical "
          f"(smaller optimisation problem: no scenario matrix needed).")

# STEP 4: Visualisations
print("\n[4/6] Generating comparison charts...")
os.makedirs(FIGURES_PATH, exist_ok=True)

COLORS = {
    "Exp1_Historical": "#616161",
    "HistVol": "#90A4AE", "GARCH": "#2196F3", "GJR-GARCH": "#FF9800",
    "LSTM": "#4CAF50", "Transformer": "#9C27B0", "Exp3_AC": "#E91E63",
}

# ── Chart A1: Cumulative returns — all experiments ──
fig, ax = plt.subplots(figsize=(16, 7))
cum1 = (1+exp1_ret).cumprod()
ax.plot(cum1.index, cum1, label="Exp1 — Historical (Baseline)",
        color=COLORS["Exp1_Historical"], linewidth=2.0, linestyle="--")

for model, r in exp2_returns.items():
    cum = (1+r).cumprod()
    ax.plot(cum.index, cum, label=f"Exp2 — {model}",
            color=COLORS.get(model, "#888"), linewidth=1.2, alpha=0.85)

if exp3_ret is not None:
    cum3 = (1+exp3_ret).cumprod()
    ax.plot(cum3.index, cum3, label="Exp3 — Forecast + A+C",
            color=COLORS["Exp3_AC"], linewidth=2.5)

ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)
ax.set_title("Cumulative Returns — Historical vs Forecast-Driven vs A+C Enhanced",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Portfolio Value (start = $1.00)")
ax.legend(fontsize=8, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}A1_all_experiments_cumulative_returns.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(" Chart A1 saved: All Experiments Cumulative Returns")

# ── Chart A2: Performance metrics grouped bar ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics_to_plot = ["Ann. Return (%)", "Sharpe", "Max DD (%)"]
bar_colors = []
for exp_name in comparison_df.index:
    if "Historical" in exp_name: 
        bar_colors.append(COLORS["Exp1_Historical"])
    elif "A+C" in exp_name: 
        bar_colors.append(COLORS["Exp3_AC"])
    else:
        for m in COLORS:
            if m in exp_name: 
                bar_colors.append(COLORS[m]) 
                break
        else: 
            bar_colors.append("#888")

for ax, metric in zip(axes, metrics_to_plot):
    vals = comparison_df[metric].values
    bars = ax.barh(comparison_df.index, vals, color=bar_colors,
                   edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)
    for b, v in zip(bars, vals):
        ax.text(v + (0.05 if v>=0 else -0.05), b.get_y()+b.get_height()/2,
                f"{v:.2f}", va="center", ha="left" if v>=0 else "right", fontsize=7)

plt.suptitle("Performance Metrics — All Experiments Compared",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}A2_performance_metrics_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(" Chart A2 saved: Performance Metrics Comparison")

# ── Chart A3: Computational time comparison ──
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(timing_df["Experiment"], timing_df["Avg Time/Rebalance (ms)"],
              color=["#616161" if "Historical" in e else
                     "#E91E63" if "A+C" in e else "#4CAF50"
                     for e in timing_df["Experiment"]],
              edgecolor="black", linewidth=0.5)
ax.set_title("Optimisation Time per Rebalance — All Experiments",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Avg Time (milliseconds)")
ax.tick_params(axis="x", rotation=30, labelsize=8)
for b, v in zip(bars, timing_df["Avg Time/Rebalance (ms)"]):
    ax.text(b.get_x()+b.get_width()/2, v, f"{v:.1f}ms",
            ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}A3_computational_time_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(" Chart A3 saved: Computational Time Comparison")

# ── Chart A4: Weight allocation heatmaps ──
weight_files = {
    "Exp1 — Historical": f"{PROCESSED_DATA_PATH}exp1_historical_weights.csv",
    "Exp2 — Transformer": f"{PROCESSED_DATA_PATH}exp2_forecast_Transformer_weights.csv",
    "Exp3 — Forecast+A+C": f"{PROCESSED_DATA_PATH}exp3_forecast_ac_weights.csv",
}
available_weights = {k: v for k, v in weight_files.items() if os.path.exists(v)}

if available_weights:
    fig, axes = plt.subplots(1, len(available_weights), figsize=(7*len(available_weights), 6))
    if len(available_weights) == 1:
        axes = [axes]
    for ax, (label, path) in zip(axes, available_weights.items()):
        w_df = pd.read_csv(path, index_col=0, parse_dates=True)
        sns.heatmap(w_df.T.astype(float), cmap="YlOrRd", ax=ax,
                   cbar_kws={"label": "Weight"}, vmin=0, vmax=0.3)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Rebalance Date")
        ax.set_ylabel("Asset")
        ax.set_xticks([])
    plt.suptitle("Portfolio Weight Allocation Over Time — Heatmap Comparison",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}A4_weight_allocation_heatmaps.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart A4 saved: Weight Allocation Heatmaps")

# ── Chart A5: Weight differentiation across forecast models ──
diff_path = f"{METRICS_PATH}exp2_weight_differentiation.csv"
if os.path.exists(diff_path):
    diff_df = pd.read_csv(diff_path)
    diff_df["Pair"] = diff_df["Model A"] + " vs " + diff_df["Model B"]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(diff_df["Pair"], diff_df["Avg Weight Distance"],
                   color="#5C6BC0", edgecolor="black", linewidth=0.5)
    ax.set_title("Weight Allocation Differentiation Across Forecast Models\n"
                 "(Higher = models produce more different portfolio decisions)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Avg Absolute Weight Distance")
    for b, v in zip(bars, diff_df["Avg Weight Distance"]):
        ax.text(v, b.get_y()+b.get_height()/2, f"{v:.4f}",
                va="center", ha="left", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}A5_weight_differentiation.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart A5 saved: Weight Differentiation")

# STEP 5: Regime allocation chart (Exp3)
regime_log_path = f"{METRICS_PATH}exp3_regime_log.csv"
if os.path.exists(regime_log_path) and exp3_ret is not None:
    regime_log = pd.read_csv(regime_log_path, index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(16, 5))
    cum3 = (1+exp3_ret).cumprod()
    ax.plot(cum3.index, cum3, color=COLORS["Exp3_AC"], linewidth=1.5)
    for dt, row in regime_log.iterrows():
        color = "#B71C1C" if "Stress" in str(row["Regime"]) else "#1B5E20"
        ax.axvline(dt, color=color, alpha=0.08, linewidth=8)
    ax.set_title("Experiment 3 — Equity Curve with Regime Allocation\n"
                 "(Red = Stress/Defensive rebalance, Green = Normal/Return-seeking blend)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Portfolio Value")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}A6_exp3_regime_overlay.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(" Chart A6 saved: Experiment 3 Regime Overlay")


# STEP 6: Summary
print("\n[5/6] Building final answer to supervisor's core question...")

best_exp2 = min(exp2_returns.items(),
                 key=lambda kv: perf_metrics(kv[1])["Max DD (%)"])
print(f"""
   ── DOES THE FORECAST-DRIVEN APPROACH WORK? ──

   Exp1 (Historical, full 252-day scenarios):
     Sharpe = {perf_metrics(exp1_ret)['Sharpe']:.4f} | Max DD = {perf_metrics(exp1_ret)['Max DD (%)']:.2f}%

   Exp2 (Forecast-driven, CVaR only — {best_exp2[0]}):
     Sharpe = {perf_metrics(best_exp2[1])['Sharpe']:.4f} | Max DD = {perf_metrics(best_exp2[1])['Max DD (%)']:.2f}%

   {"Exp3 (Forecast + A+C):" if exp3_ret is not None else ""}
   {f"    Sharpe = {perf_metrics(exp3_ret)['Sharpe']:.4f} | Max DD = {perf_metrics(exp3_ret)['Max DD (%)']:.2f}%" if exp3_ret is not None else ""}

   INTERPRETATION:
   If Exp2 achieves comparable risk metrics to Exp1 using only a
   single CVaR number per asset (instead of 252 raw data points),
   this demonstrates the forecasts capture the essential risk
   information — directly answering the supervisor's question.
""")

print("[6/6] Comparison complete.")
print("=" * 60)
print("  Files saved:")
print(" phase4_all_experiments_comparison.csv")
print(" phase4_computational_comparison.csv")
print(" Charts saved:")
print(" A1_all_experiments_cumulative_returns.png")
print(" A2_performance_metrics_comparison.png")
print(" A3_computational_time_comparison.png")
print(" A4_weight_allocation_heatmaps.png")
print(" A5_weight_differentiation.png")
print(" A6_exp3_regime_overlay.png")
print("=" * 60)