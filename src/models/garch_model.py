import numpy as np
import pandas as pd
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")


class GARCHModel:
    """
    GARCH(1,1) one-step-ahead volatility forecaster.

    Parameters
    ----------
    p : int  — lag order for ARCH terms (default 1)
    q : int  — lag order for GARCH terms (default 1)
    dist : str  — innovation distribution ('normal', 't', 'skewt')
    """

    def __init__(self, p: int = 1, q: int = 1, dist: str = "normal"):
        self.p     = p
        self.q     = q
        self.dist  = dist
        self.name  = f"GARCH({p},{q})"
        self.model = None
        self.fit_result = None

    def fit_predict(
        self,
        returns: pd.Series,
        train_size: int,
        refit_every: int = 252,   # Refit once per year (~252 trading days)
    ) -> pd.Series:
        """
        Rolling/expanding-window GARCH forecasting.

        Trains on first `train_size` points, forecasts next day,
        then expands window and refits every `refit_every` days.
        """
        returns = returns.dropna()
        n       = len(returns)
        # Scale returns to percentage so the optimiser doesn't choke
        ret_pct = returns * 100
        forecasts = pd.Series(index=returns.index, dtype=float)

        last_fit = None
        omega = alpha = beta = None

        for i in range(train_size, n):
            # Refit periodically (or on first iteration)
            if last_fit is None or (i - last_fit) >= refit_every:
                try:
                    train_data = ret_pct.iloc[:i]
                    model = arch_model(
                        train_data,
                        mean = "Zero",
                        vol  = "GARCH",
                        p    = self.p,
                        q    = self.q,
                        dist = self.dist,
                        rescale = False,
                    )
                    res = model.fit(disp="off", show_warning=False)
                    omega = res.params["omega"]
                    alpha = res.params.get("alpha[1]", 0)
                    beta  = res.params.get("beta[1]", 0)
                    self.fit_result = res
                    last_fit = i
                except Exception:
                    # Fall back to previous estimates if fit fails
                    if omega is None:
                        omega, alpha, beta = 0.01, 0.05, 0.90

            try:
                prev_ret = ret_pct.iloc[i - 1]
                if i == train_size or pd.isna(forecasts.iloc[i - 1]):
                    prev_var = ret_pct.iloc[:i].var()
                else:
                    prev_var = forecasts.iloc[i - 1] ** 2
                next_var = omega + alpha * prev_ret ** 2 + beta * prev_var
                forecasts.iloc[i] = np.sqrt(max(next_var, 1e-10))
            except Exception:
                forecasts.iloc[i] = np.nan

        # Convert back from % to decimal scale
        forecasts = forecasts / 100
        forecasts.name = self.name
        return forecasts

    def __repr__(self):
        return f"GARCHModel(p={self.p}, q={self.q}, dist='{self.dist}')"