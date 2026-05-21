"""
Risk Metrics: VaR, CVaR, and Evaluation

- Parametric VaR/CVaR assuming Normal distribution
- Volatility forecast evaluation: MAE, RMSE, QLIKE
- VaR coverage backtests: Kupiec POF test
"""

import numpy as np
import pandas as pd
from scipy import stats

def parametric_var(volatility: pd.Series, confidence: float = 0.95) -> pd.Series:

    z = stats.norm.ppf(1 - confidence)   # negative quantile
    var = -z * volatility
    var.name = "VaR"
    return var


def parametric_cvar(volatility: pd.Series, confidence: float = 0.95) -> pd.Series:
    z      = stats.norm.ppf(1 - confidence)
    cvar   = volatility * stats.norm.pdf(z) / (1 - confidence)
    cvar.name = "CVaR"
    return cvar

# Volatility Forecast Evaluation Metrics

def realised_volatility_proxy(returns: pd.Series) -> pd.Series:
    return returns.abs()


def mae(predicted: pd.Series, actual: pd.Series) -> float:
    """Mean Absolute Error."""
    df = pd.concat([predicted, actual], axis=1).dropna()
    return float((df.iloc[:, 0] - df.iloc[:, 1]).abs().mean())


def rmse(predicted: pd.Series, actual: pd.Series) -> float:
    """Root Mean Squared Error."""
    df = pd.concat([predicted, actual], axis=1).dropna()
    return float(np.sqrt(((df.iloc[:, 0] - df.iloc[:, 1]) ** 2).mean()))


def qlike(predicted: pd.Series, actual: pd.Series) -> float:
    df = pd.concat([predicted, actual], axis=1).dropna()
    pred_var   = (df.iloc[:, 0] ** 2).clip(lower=1e-10)
    actual_var = (df.iloc[:, 1] ** 2).clip(lower=1e-10)
    ratio = actual_var / pred_var
    return float((ratio - np.log(ratio) - 1).mean())


def evaluate_volatility_forecast(
    predicted: pd.Series,
    returns:   pd.Series,
) -> dict:

    actual = realised_volatility_proxy(returns)
    return {
        "MAE":   mae(predicted, actual),
        "RMSE":  rmse(predicted, actual),
        "QLIKE": qlike(predicted, actual),
    }

# VaR Backtesting — Kupiec Proportion of Failures (POF) test

def kupiec_pof_test(
    returns:    pd.Series,
    var:        pd.Series,
    confidence: float = 0.95,
) -> dict:

    df = pd.concat([returns, var], axis=1).dropna()
    df.columns = ["ret", "var"]

    exceptions = (df["ret"] < -df["var"]).astype(int)
    N          = len(df)
    x          = int(exceptions.sum())
    p_exp      = 1 - confidence
    p_obs      = x / N if N > 0 else 0.0

    if 0 < x < N:
        log_lik_h0 = x * np.log(p_exp)     + (N - x) * np.log(1 - p_exp)
        log_lik_h1 = x * np.log(p_obs)     + (N - x) * np.log(1 - p_obs)
        lr_stat = -2 * (log_lik_h0 - log_lik_h1)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    else:
        lr_stat, p_value = np.nan, np.nan

    return {
        "N":                  N,
        "Exceptions":         x,
        "Expected_Exceptions":N * p_exp,
        "Observed_Rate":      p_obs,
        "Expected_Rate":      p_exp,
        "LR_Statistic":       lr_stat,
        "p_value":            p_value,
        "Model_Adequate":     bool(p_value > 0.05) if not np.isnan(p_value) else False,
    }