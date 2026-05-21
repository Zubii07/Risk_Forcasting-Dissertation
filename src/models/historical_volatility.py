"""
Historical Volatility Model
============================
Simple baseline: uses past N-day standard deviation as the volatility forecast.
Assumes future volatility = recent realised volatility.
"""

import numpy as np
import pandas as pd


class HistoricalVolatilityModel:
    """
    Historical (rolling-window) volatility forecaster.

    Parameters
    ----------
    window : int
        Lookback window size in days (default 30).
    annualise : bool
        If True, return annualised volatility (multiplied by sqrt(252)).
    """

    def __init__(self, window: int = 30, annualise: bool = False):
        self.window     = window
        self.annualise  = annualise
        self.name       = f"HistVol_{window}d"

    def forecast(self, returns: pd.Series) -> pd.Series:
        """
        Generate one-step-ahead volatility forecasts.
        Returns a Series indexed identically to `returns`,
        where each value is the predicted volatility for that day
        based on the previous `window` days.
        """
        vol = returns.rolling(self.window).std().shift(1)   # shift to avoid look-ahead
        if self.annualise:
            vol = vol * np.sqrt(252)
        vol.name = self.name
        return vol

    def __repr__(self):
        return f"HistoricalVolatilityModel(window={self.window})"