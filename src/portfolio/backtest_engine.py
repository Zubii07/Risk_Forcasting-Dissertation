"""
Walk-Forward Backtest Engine

Simulates a portfolio rebalanced on a fixed schedule (weekly / daily / monthly),
applying transaction costs and tracking weights, returns, and turnover over time.

Workflow per rebalance date:
    1. Use forecasts available AS OF the rebalance date (no look-ahead).
    2. Pull last `lookback_days` of historical returns as CVaR scenarios.
    3. Solve CVaR optimisation → new target weights.
    4. Apply transaction cost = turnover × cost_bps / 10000.
    5. Hold the new weights until the next rebalance date.

Returns a DataFrame with daily portfolio returns net of costs.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    portfolio_returns: pd.Series                 # Daily net returns
    weights_history:  pd.DataFrame               # Weights on each rebalance date
    turnover_history: pd.Series                  # Turnover per rebalance
    costs_history:    pd.Series                  # Transaction costs per rebalance
    cvar_history:     pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    strategy_name:    str = "unnamed"


def get_rebalance_dates(
    dates: pd.DatetimeIndex,
    frequency: str = "weekly",
) -> pd.DatetimeIndex:
    """
    Get rebalance dates from a daily date index.

    'weekly'  → first trading day of each ISO week (typically Monday)
    'daily'   → every trading day
    'monthly' → first trading day of each month
    """
    if frequency == "daily":
        return dates

    df = pd.DataFrame(index=dates)
    if frequency == "weekly":
        df["period"] = df.index.to_period("W")
    elif frequency == "monthly":
        df["period"] = df.index.to_period("M")
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    rebal_dates = df.groupby("period").apply(lambda x: x.index[0]).values
    return pd.DatetimeIndex(rebal_dates)


def run_backtest(
    returns: pd.DataFrame,
    optimise_fn,
    forecast_provider=None,
    rebalance_dates: pd.DatetimeIndex | None = None,
    frequency: str = "weekly",
    lookback_days: int = 252,
    cost_bps: float = 5.0,
    initial_weights: np.ndarray | None = None,
    strategy_name: str = "strategy",
    verbose: bool = True,
) -> BacktestResult:

    asset_list = returns.columns.tolist()
    n_assets = len(asset_list)
    dates = returns.index

    # Build rebalance schedule
    if rebalance_dates is None:
        rebalance_dates = get_rebalance_dates(dates, frequency=frequency)
    rebalance_dates = pd.DatetimeIndex(
        [d for d in rebalance_dates if d in dates]
    )

    # Need lookback history before the FIRST rebalance
    valid_starts = dates[dates >= dates[lookback_days]]
    rebalance_dates = rebalance_dates[rebalance_dates.isin(valid_starts)]
    if len(rebalance_dates) == 0:
        raise ValueError("No valid rebalance dates given lookback requirements.")

    # Initialise
    if initial_weights is None:
        current_weights = np.full(n_assets, 1.0 / n_assets)
    else:
        current_weights = np.asarray(initial_weights, dtype=float)

    cost_rate = cost_bps / 10_000.0

    # Storage
    weights_log  = []
    weights_dates = []
    turnover_log = []
    costs_log    = []
    cvar_log     = []
    daily_returns = pd.Series(index=dates, dtype=float)

    start_idx = dates.get_loc(rebalance_dates[0])
    daily_returns.iloc[:start_idx] = 0.0
    weights_arr = np.tile(current_weights, (len(dates), 1))

    next_rebal_iter = iter(rebalance_dates)
    next_rebal = next(next_rebal_iter)

    if verbose:
        print(f"   ▶ Backtest [{strategy_name}] "
              f"| {len(rebalance_dates)} rebalances "
              f"| {frequency} | costs={cost_bps}bps | lookback={lookback_days}d")

    for t, date in enumerate(dates):
        if t < start_idx:
            continue

        # ---- Rebalance? 
        if date == next_rebal:
            # Build scenario matrix from prior `lookback_days` of returns
            hist = returns.iloc[t - lookback_days : t].values

            # Get forecast inputs (if provider given)
            kwargs = {}
            if forecast_provider is not None:
                fc = forecast_provider(date, asset_list)
                if fc is not None:
                    kwargs.update(fc)

            try:
                new_weights = optimise_fn(hist, **kwargs)
            except Exception as e:
                logger.warning(f"Optimisation failed at {date.date()}: {e}")
                new_weights = current_weights

            # Transaction cost = turnover × cost_rate
            turnover = float(np.abs(new_weights - current_weights).sum())
            cost = turnover * cost_rate

            weights_log.append(new_weights.copy())
            weights_dates.append(date)
            turnover_log.append(turnover)
            costs_log.append(cost)

            # Apply cost on rebalance day
            daily_pnl = float(returns.iloc[t].values @ new_weights) - cost
            current_weights = new_weights

            # Advance to next rebalance date
            try:
                next_rebal = next(next_rebal_iter)
            except StopIteration:
                next_rebal = pd.Timestamp("2100-01-01")  # never trigger again
        else:
            # Hold weights, drift with returns
            daily_pnl = float(returns.iloc[t].values @ current_weights)
            # Weights drift (passive)
            current_weights = current_weights * (1 + returns.iloc[t].values)
            total = current_weights.sum()
            if total > 0:
                current_weights = current_weights / total

        daily_returns.iloc[t] = daily_pnl
        weights_arr[t] = current_weights

    weights_history_df = pd.DataFrame(
        weights_log, index=weights_dates, columns=asset_list
    )
    return BacktestResult(
        portfolio_returns=daily_returns.dropna(),
        weights_history=weights_history_df,
        turnover_history=pd.Series(turnover_log, index=weights_dates),
        costs_history=pd.Series(costs_log, index=weights_dates),
        cvar_history=pd.Series(cvar_log, index=weights_dates) if cvar_log else pd.Series(dtype=float),
        strategy_name=strategy_name,
    )