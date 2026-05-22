import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.portfolio.cvar_optimiser import cvar_optimise, equal_weight
from src.portfolio.backtest_engine import run_backtest
from config.config import (
    PROCESSED_DATA_PATH,
    METRICS_PATH,
    FIGURES_PATH,
    PORTFOLIO_TICKERS,
    CONFIDENCE_LEVEL,
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# CONFIGURATION

REBALANCE_FREQUENCY = "weekly"     # primary: weekly | also test: daily, monthly
LOOKBACK_DAYS = 252                # ~1 year of scenarios
COST_BPS = 5.0                     # 5 basis points per turnover unit
MAX_WEIGHT = 0.30                  # no single asset > 30%
MIN_WEIGHT = 0.0                   # long-only
ALPHA = 1 - CONFIDENCE_LEVEL       # 0.05 for 95% CVaR

# Whether to also run daily/monthly as robustness checks
RUN_ROBUSTNESS = True

# UTILITIES

def p(folder: str, fname: str = "") -> Path:
    return Path(folder) / fname if fname else Path(folder)


def build_forecast_provider(vol_forecasts: pd.DataFrame):
    """
    Build a forecast_provider callable from a volatility-forecast DataFrame.

    Returns risk_scaling = predicted vol on the rebalance date,
    used by cvar_optimise to scale historical scenarios.
    """
    def provider(date: pd.Timestamp, asset_list: list[str]) -> dict | None:
        if vol_forecasts is None:
            return None
        # Use most recent forecast at or before `date`
        try:
            row = vol_forecasts.loc[:date].iloc[-1]
        except (IndexError, KeyError):
            return None
        scaling = np.array(
            [row[a] if a in row.index and pd.notna(row[a]) else np.nan for a in asset_list]
        )
        if np.isnan(scaling).any():
            return None
        return {"risk_scaling": scaling}
    return provider


def make_optimiser():
    """Return a callable wrapping the CVaR optimiser with project defaults."""
    def opt(scenarios, **kwargs):
        weights, _, _ = cvar_optimise(
            scenarios,
            alpha=ALPHA,
            max_weight=MAX_WEIGHT,
            min_weight=MIN_WEIGHT,
            **kwargs,
        )
        return weights
    return opt


def make_eq_weight_optimiser(n_assets: int):
    """Equal-weight rebalances back to 1/N at each rebalance date."""
    def opt(scenarios, **kwargs):
        return equal_weight(n_assets)
    return opt

# MAIN

def main() -> None:
    print("=" * 64)
    print("  PHASE 4: CVaR Portfolio Optimisation")
    print("=" * 64)

    # STEP 1: Load returns & forecasts

    print("\n[1/6] Loading data and forecasts...")

    log_returns = pd.read_csv(
        p(PROCESSED_DATA_PATH, "log_returns.csv"),
        index_col=0, parse_dates=True,
    )
    test_df = pd.read_csv(p(PROCESSED_DATA_PATH, "test.csv"), index_col=0, parse_dates=True)

    asset_cols = [c for c in log_returns.columns if c in PORTFOLIO_TICKERS]
    returns = log_returns[asset_cols].copy()
    n_assets = len(asset_cols)
    print(f"   {n_assets} assets, {len(returns):,} days")

    # Load forecasts from Phase 2 + 3
    def safe_load(fname):
        path = p(PROCESSED_DATA_PATH, fname)
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df[[c for c in asset_cols if c in df.columns]]
        return None

    hv_forecasts  = safe_load("histvol_forecasts.csv")
    grch_forecasts = safe_load("garch_forecasts.csv")
    lstm_forecasts = safe_load("lstm_forecasts.csv")
    tfm_forecasts  = safe_load("transformer_forecasts.csv")

    for name, df in [("HistVol", hv_forecasts), ("GARCH", grch_forecasts),
                     ("LSTM", lstm_forecasts), ("Transformer", tfm_forecasts)]:
        status = f"shape {df.shape}" if df is not None else "NOT FOUND"
        print(f"   {'' if df is not None else '⚠'} {name}: {status}")

    # Test period bounds
    test_start = test_df.index[0]
    test_end   = test_df.index[-1]
    print(f"   Test period: {test_start.date()} → {test_end.date()}")

    # STEP 2: Define strategies
    
    print("\n[2/6] Defining portfolio strategies...")

    strategies = {
        "EqualWeight":    {"optimise_fn": make_eq_weight_optimiser(n_assets),
                           "forecast_provider": None},
        "CVaR_Historical": {"optimise_fn": make_optimiser(),
                            "forecast_provider": None},
        "CVaR_HistVol":   {"optimise_fn": make_optimiser(),
                           "forecast_provider": build_forecast_provider(hv_forecasts)},
        "CVaR_GARCH":     {"optimise_fn": make_optimiser(),
                           "forecast_provider": build_forecast_provider(grch_forecasts)},
        "CVaR_LSTM":      {"optimise_fn": make_optimiser(),
                           "forecast_provider": build_forecast_provider(lstm_forecasts)},
        "CVaR_Transformer": {"optimise_fn": make_optimiser(),
                             "forecast_provider": build_forecast_provider(tfm_forecasts)},
    }

    # Skip strategies whose forecasts couldn't be loaded
    available = {}
    for name, cfg in strategies.items():
        needs_forecast = cfg["forecast_provider"] is not None
        if needs_forecast:
            # Probe the provider: does it return a valid scaling at any test date?
            sample_date = test_df.index[len(test_df) // 2]
            sample_fc = cfg["forecast_provider"](sample_date, asset_cols)
            if sample_fc is None:
                print(f"   ⚠  Skipping {name}: forecasts unavailable")
                continue
        available[name] = cfg

    print(f"   {len(available)} strategies ready: {list(available.keys())}")

    # STEP 3: Run backtests at PRIMARY frequency (weekly)

    print(f"\n[3/6] Running backtests at PRIMARY frequency: {REBALANCE_FREQUENCY}...")

    # Limit returns to: from (test_start - lookback_days) to test_end
    bt_start_idx = max(0, returns.index.get_loc(test_start) - LOOKBACK_DAYS - 5)
    bt_returns = returns.iloc[bt_start_idx:returns.index.get_loc(test_end) + 1]

    primary_results = {}
    for name, cfg in available.items():
        print()
        result = run_backtest(
            returns=bt_returns,
            optimise_fn=cfg["optimise_fn"],
            forecast_provider=cfg["forecast_provider"],
            frequency=REBALANCE_FREQUENCY,
            lookback_days=LOOKBACK_DAYS,
            cost_bps=COST_BPS,
            strategy_name=name,
            verbose=True,
        )
        primary_results[name] = result

    # STEP 4: Compute performance metrics on TEST period

    print("\n[4/6] Computing performance metrics on test period...")

    perf_rows = []
    for name, res in primary_results.items():
        test_rets = res.portfolio_returns.loc[test_start:test_end]
        m = portfolio_metrics(test_rets)
        m["Strategy"] = name
        m["Turnover_avg"] = res.turnover_history.mean()
        m["Cost_total"]   = res.costs_history.sum()
        m["N_rebalances"] = len(res.weights_history)
        perf_rows.append(m)

    perf_df = pd.DataFrame(perf_rows).set_index("Strategy")
    # Order columns nicely
    cols = ["Ann_Return", "Ann_Vol", "Sharpe", "Sortino", "Max_DD",
            "Calmar", "VaR_95", "CVaR_95", "Turnover_avg", "Cost_total", "N_rebalances"]
    perf_df = perf_df[cols].round(4)

    print(f"\n  ── Performance summary ({REBALANCE_FREQUENCY} rebalancing) ──")
    print(perf_df.to_string())


    # STEP 5: Robustness checks (daily + monthly)

    robustness_results = {}
    if RUN_ROBUSTNESS:
        print("\n[5/6] Running robustness checks (daily + monthly)...")
        for freq in ["daily", "monthly"]:
            print(f"\n   ── Frequency: {freq} ──")
            freq_results = {}
            for name, cfg in available.items():
                res = run_backtest(
                    returns=bt_returns,
                    optimise_fn=cfg["optimise_fn"],
                    forecast_provider=cfg["forecast_provider"],
                    frequency=freq,
                    lookback_days=LOOKBACK_DAYS,
                    cost_bps=COST_BPS,
                    strategy_name=f"{name}_{freq}",
                    verbose=False,
                )
                freq_results[name] = res
            robustness_results[freq] = freq_results

        # Summary table across frequencies
        rob_rows = []
        for freq, fres in robustness_results.items():
            for name, res in fres.items():
                test_rets = res.portfolio_returns.loc[test_start:test_end]
                m = portfolio_metrics(test_rets)
                m["Strategy"]  = name
                m["Frequency"] = freq
                rob_rows.append(m)
        rob_df = pd.DataFrame(rob_rows)
        print("\n  ── Sharpe ratios by frequency ──")
        print(rob_df.pivot(index="Strategy", columns="Frequency", values="Sharpe").round(3).to_string())
    else:
        print("\n[5/6] Robustness checks SKIPPED (set RUN_ROBUSTNESS=True to enable)")


    # STEP 6: Save outputs + figures

    print("\n[6/6] Saving outputs and producing figures...")

    Path(METRICS_PATH).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_PATH).mkdir(parents=True, exist_ok=True)

    # Save daily portfolio returns
    rets_df = pd.DataFrame({
        name: res.portfolio_returns for name, res in primary_results.items()
    })
    rets_df.to_csv(p(PROCESSED_DATA_PATH, "phase4_portfolio_returns.csv"))

    # Save weights history (concatenated)
    wt_list = []
    for name, res in primary_results.items():
        wts = res.weights_history.copy()
        wts["Strategy"] = name
        wt_list.append(wts)
    pd.concat(wt_list).to_csv(p(PROCESSED_DATA_PATH, "phase4_weights_history.csv"))

    # Save metrics
    perf_df.to_csv(p(METRICS_PATH, "phase4_performance_metrics.csv"))
    if RUN_ROBUSTNESS:
        rob_df.to_csv(p(METRICS_PATH, "phase4_robustness_metrics.csv"), index=False)

    # Figures
    plot_phase4(primary_results, perf_df, test_start, test_end)

    # SUMMARY
 
    print("\n" + "=" * 64)
    print("  PHASE 4 COMPLETE ✅")
    print("=" * 64)
    print(f"""
  Configuration:
    Rebalancing : {REBALANCE_FREQUENCY}
    Lookback    : {LOOKBACK_DAYS} days
    Costs       : {COST_BPS} bps per turnover unit
    CVaR α      : {ALPHA} (95% CVaR)
    Max weight  : {MAX_WEIGHT * 100:.0f}%

  Strategies backtested: {len(primary_results)}
{perf_df[['Sharpe', 'Sortino', 'Max_DD', 'CVaR_95']].to_string()}

""")

# METRICS

TRADING_DAYS = 252

def portfolio_metrics(returns: pd.Series, rf: float = 0.02) -> dict:
    """Compute headline performance metrics."""
    r = returns.dropna()
    if len(r) == 0:
        return {k: np.nan for k in
                ["Ann_Return", "Ann_Vol", "Sharpe", "Sortino", "Max_DD", "Calmar", "VaR_95", "CVaR_95"]}

    ann_ret = (1 + r.mean()) ** TRADING_DAYS - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    excess  = r - rf / TRADING_DAYS
    sharpe  = excess.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan
    down    = r[r < 0].std()
    sortino = excess.mean() / down * np.sqrt(TRADING_DAYS) if down > 0 else np.nan

    cum  = (1 + r).cumprod()
    peak = cum.cummax()
    mdd  = float((cum / peak - 1).min())
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan

    var_q  = float(np.quantile(r, 0.05))
    var95  = -var_q
    tail   = r[r <= var_q]
    cvar95 = -tail.mean() if len(tail) else np.nan

    return dict(
        Ann_Return=ann_ret, Ann_Vol=ann_vol,
        Sharpe=sharpe, Sortino=sortino,
        Max_DD=mdd, Calmar=calmar,
        VaR_95=var95, CVaR_95=cvar95,
    )

# FIGURES

def plot_phase4(results: dict, perf_df: pd.DataFrame, test_start, test_end):
    figdir = Path(FIGURES_PATH)
    figdir.mkdir(parents=True, exist_ok=True)

    palette = sns.color_palette("husl", len(results))
    color_map = {name: palette[i] for i, name in enumerate(results.keys())}

    # Plot 14: Cumulative returns 
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, res in results.items():
        r = res.portfolio_returns.loc[test_start:test_end]
        cum = (1 + r).cumprod()
        ax.plot(cum.index, cum.values, label=name, linewidth=1.5, color=color_map[name])
    ax.set_title("Cumulative Portfolio Returns — Test Period", fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    plt.savefig(figdir / "14_cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Plot 14 saved: Cumulative returns")

    # Plot 15: Drawdown curves 
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, res in results.items():
        r = res.portfolio_returns.loc[test_start:test_end]
        cum = (1 + r).cumprod()
        dd = cum / cum.cummax() - 1
        ax.fill_between(dd.index, dd.values, 0, alpha=0.15, color=color_map[name])
        ax.plot(dd.index, dd.values, label=name, linewidth=1.0, color=color_map[name])
    ax.set_title("Portfolio Drawdowns — Test Period", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir / "15_drawdown_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Plot 15 saved: Drawdown curves")

    # ---- Plot 16: Performance metrics bar chart ----
    metrics_to_plot = ["Sharpe", "Sortino", "Max_DD", "CVaR_95"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for i, m in enumerate(metrics_to_plot):
        vals = perf_df[m]
        # For Max_DD and CVaR_95, larger negative = worse → invert for chart
        if m in ["Max_DD"]:
            display = vals.abs()
            title = "Max Drawdown (absolute, lower = better)"
        elif m in ["CVaR_95"]:
            display = vals
            title = "CVaR 95% (lower = better)"
        else:
            display = vals
            title = f"{m} (higher = better)"
        colors = [color_map[k] for k in vals.index]
        axes[i].bar(range(len(vals)), display.values, color=colors,
                    edgecolor="black", linewidth=0.5)
        axes[i].set_xticks(range(len(vals)))
        axes[i].set_xticklabels(vals.index, rotation=30, ha="right", fontsize=9)
        axes[i].set_title(title, fontsize=11, fontweight="bold")
        axes[i].grid(axis="y", alpha=0.3)
        for j, v in enumerate(display.values):
            axes[i].text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle("Phase 4 — Strategy Performance Metrics", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(figdir / "16_performance_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Plot 16 saved: Performance metrics")

    # ---- Plot 17: Average weights bar chart ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, (name, res) in enumerate(results.items()):
        if i >= len(axes):
            break
        wts = res.weights_history
        wts_in_test = wts.loc[test_start:test_end]
        avg = wts_in_test.mean()
        avg.plot(kind="bar", ax=axes[i], color=color_map[name],
                 edgecolor="black", linewidth=0.5)
        axes[i].set_title(f"{name} — Avg Weights (Test)", fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Weight")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=45, labelsize=8)
        axes[i].grid(axis="y", alpha=0.3)
        axes[i].axhline(0.3, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    plt.suptitle("Phase 4 — Average Portfolio Weights by Strategy",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(figdir / "17_weights_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Plot 17 saved: Weights evolution")

    # ---- Plot 18: Rolling 63-day Sharpe ratio ----
    fig, ax = plt.subplots(figsize=(14, 6))
    window = 63
    for name, res in results.items():
        r = res.portfolio_returns.loc[test_start:test_end]
        rolling_sharpe = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(TRADING_DAYS)
        ax.plot(rolling_sharpe.index, rolling_sharpe.values,
                label=name, linewidth=1.2, color=color_map[name])
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio — Test Period",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Annualised Sharpe")
    ax.set_xlabel("")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    plt.savefig(figdir / "18_rolling_sharpe.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   Plot 18 saved: Rolling Sharpe")


if __name__ == "__main__":
    main()