

# --- Asset Universe ---
ASSETS = {
    "Technology":  ["AAPL", "MSFT", "NVDA"],
    "Finance":     ["JPM", "GS"],
    "Healthcare":  ["JNJ", "UNH"],
    "Energy":      ["XOM"],
    "ETFs":        ["SPY", "QQQ"],
    "Bonds":       ["TLT"],        # Long duration bonds (negatively correlated)
    "Gold":        ["GLD"],        # Safe haven asset
    "Volatility":  ["^VIX"],       # VIX for stress regime labeling
}

# Flat list of tickers (excluding VIX from portfolio)
PORTFOLIO_TICKERS = [
    "AAPL", "MSFT", "NVDA",
    "JPM", "GS",
    "JNJ", "UNH",
    "XOM",
    "SPY", "QQQ",
    "TLT", "GLD"
]

VIX_TICKER = "^VIX"

# --- Date Range ---
START_DATE = "2004-01-01"
END_DATE   = "2025-01-01"

# --- Stress Periods ---
STRESS_PERIODS = {
    "GFC_2008":        ("2008-09-01", "2009-03-31"),
    "COVID_2020":      ("2020-02-01", "2020-04-30"),
    "Inflation_2022":  ("2022-01-01", "2022-12-31"),
}

# --- VIX Stress Threshold ---
VIX_STRESS_THRESHOLD = 25   # VIX > 25 = high volatility regime

# --- Walk-Forward Split ---
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15          # Test set starts ~2020 onwards

# --- Feature Engineering ---
ROLLING_WINDOWS = [10, 30]  # Days for rolling volatility features
SEQUENCE_LENGTH = 30        # Lookback window for LSTM/Transformer (timesteps)

# --- Portfolio ---
REBALANCE_FREQ      = "D"     # Daily rebalancing
TRANSACTION_COST    = 0.001   # 0.1% per trade
RISK_FREE_RATE      = 0.02    # 2% annual (for Sharpe/Sortino)
CONFIDENCE_LEVEL    = 0.95    # For VaR/CVaR estimation
MAX_WEIGHT           = 0.30
MIN_RETURN_TARGET    = 0.06     # annualised, used in Experiment 3 (constraint A)
EWMA_RETURN_SPAN     = 60       # days, compact expected-return proxy (Experiment 3)

# --- Paths ---
RAW_DATA_PATH       = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
RESULTS_PATH        = "results/"
FIGURES_PATH        = "results/figures/"
METRICS_PATH        = "results/metrics/"