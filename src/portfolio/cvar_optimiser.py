"""
CVaR Portfolio Optimisation

Implements the Rockafellar-Uryasev (2000) linear-programming formulation
for minimising portfolio Conditional Value-at-Risk.
Outputs the weight vector w* that minimises expected CVaR loss.
"""
from __future__ import annotations
import numpy as np
import cvxpy as cp


def cvar_optimise(
    scenarios: np.ndarray,
    alpha: float = 0.05,
    max_weight: float = 0.30,
    min_weight: float = 0.0,
    target_return: float | None = None,
    risk_scaling: np.ndarray | None = None,
    solver: str = "ECOS",
) -> tuple[np.ndarray, float, str]:

    T, N = scenarios.shape

    # Apply risk scaling if provided (forecast-aware scenarios)
    if risk_scaling is not None:
        if len(risk_scaling) != N:
            raise ValueError(f"risk_scaling length {len(risk_scaling)} != N {N}")
        hist_std = scenarios.std(axis=0, ddof=1)
        # Avoid divide-by-zero
        ratio = np.where(hist_std > 1e-8, risk_scaling / hist_std, 1.0)
        scenarios = scenarios * ratio[np.newaxis, :]

    # CVXPY variables
    w = cp.Variable(N)
    zeta = cp.Variable()                  # Value-at-Risk threshold
    z = cp.Variable(T, nonneg=True)       # Excess-loss slack variables

    # Loss in scenario t = -returns_t @ w
    losses = -scenarios @ w
    cvar = zeta + (1.0 / (alpha * T)) * cp.sum(z)

    constraints = [
        z >= losses - zeta,
        cp.sum(w) == 1.0,
        w >= min_weight,
        w <= max_weight,
    ]
    if target_return is not None:
        constraints.append(scenarios.mean(axis=0) @ w >= target_return)

    prob = cp.Problem(cp.Minimize(cvar), constraints)
    try:
        prob.solve(solver=solver, verbose=False)
    except Exception:
        # Fallback solver
        prob.solve(solver="SCS", verbose=False)

    if w.value is None:
        # Optimisation failed — return equal weights as a safety fallback
        return np.full(N, 1.0 / N), float("nan"), prob.status or "failed"

    weights = np.clip(np.asarray(w.value), min_weight, max_weight)
    # Renormalise to sum to 1 (handles small numerical drift)
    weights = weights / weights.sum()
    return weights, float(prob.value), prob.status


def equal_weight(n_assets: int) -> np.ndarray:
    """1/N benchmark portfolio."""
    return np.full(n_assets, 1.0 / n_assets)


def cvar_of_portfolio(scenarios: np.ndarray, weights: np.ndarray, alpha: float = 0.05) -> float:
    """Compute realised CVaR of a portfolio over given scenarios."""
    portfolio_returns = scenarios @ weights
    var = np.quantile(portfolio_returns, alpha)
    tail = portfolio_returns[portfolio_returns <= var]
    return -tail.mean() if len(tail) else float("nan")