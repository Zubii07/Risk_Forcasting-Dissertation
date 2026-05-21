from pathlib import Path
import sys
import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (  # noqa: E402
    END_DATE,
    FIGURES_PATH,
    PORTFOLIO_TICKERS,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    ROLLING_WINDOWS,
    START_DATE,
    STRESS_PERIODS,
    TRAIN_RATIO,
    VAL_RATIO,
    VIX_STRESS_THRESHOLD,
    VIX_TICKER,
)

warnings.filterwarnings("ignore")
sys.path.append("..")

# ── Plot Style 
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

print("=" * 60)
print("  Data Collection & Preprocessing")
print("=" * 60)


# Download Raw Data from Yahoo Finance

print("\n[1/6] Downloading data from Yahoo Finance...")

all_tickers = PORTFOLIO_TICKERS + [VIX_TICKER]

raw_data = yf.download(
    tickers    = all_tickers,
    start      = START_DATE,
    end        = END_DATE,
    auto_adjust= True,
    progress   = True
)["Close"]

# Rename VIX column
raw_data.rename(columns={"^VIX": "VIX"}, inplace=True)

print(f"\n Downloaded {len(raw_data.columns)} assets")
print(f"   Date range : {raw_data.index[0].date()} → {raw_data.index[-1].date()}")
print(f"   Total rows : {len(raw_data):,}")
print(f"   Assets     : {list(raw_data.columns)}")

# Save raw data
os.makedirs(RAW_DATA_PATH, exist_ok=True)
raw_data.to_csv(f"{RAW_DATA_PATH}raw_prices.csv")
print(f"\n Raw data saved → {RAW_DATA_PATH}raw_prices.csv")

# Data Quality Check

print("\n[2/6] Running data quality checks...")

missing = raw_data.isnull().sum()
missing_pct = (missing / len(raw_data) * 100).round(2)

quality_report = pd.DataFrame({
    "Missing Values": missing,
    "Missing %": missing_pct,
    "Start Date": raw_data.apply(lambda col: col.first_valid_index().date()),
    "End Date"  : raw_data.apply(lambda col: col.last_valid_index().date()),
})

print("\nData Quality Report:")
print(quality_report.to_string())

# Flag assets with >5% missing
problematic = missing_pct[missing_pct > 5]
if not problematic.empty:
    print(f"\n⚠️  Assets with >5% missing data: {list(problematic.index)}")
else:
    print("\n All assets pass quality check (< 5% missing)")


# STEP 3: Preprocessing — Clean & Forward Fill

print("\n[3/6] Preprocessing data...")

# Separate VIX from portfolio prices
vix_prices   = raw_data[["VIX"]].copy()
port_prices  = raw_data[PORTFOLIO_TICKERS].copy()

# Forward fill missing values (market holidays etc.)
port_prices.ffill(inplace=True)
port_prices.bfill(inplace=True)
vix_prices.ffill(inplace=True)

# Drop rows where ALL assets are missing (non-trading days)
port_prices.dropna(how="all", inplace=True)

print(f" Clean price data: {port_prices.shape[0]:,} rows × {port_prices.shape[1]} assets")


# STEP 4: Feature Engineering

print("\n[4/6] Engineering features...")

# --- Log Returns ---
log_returns = np.log(port_prices / port_prices.shift(1)).dropna()
print(f"    Log returns computed: {log_returns.shape}")

# --- Rolling Volatility ---
roll_vol = pd.DataFrame(index=log_returns.index)
for d in ROLLING_WINDOWS:
    for col in log_returns.columns:
        roll_vol[f"{col}_vol_{d}d"] = log_returns[col].rolling(d).std() * np.sqrt(252)

roll_vol.dropna(inplace=True)
print(f"    Rolling volatility features: {roll_vol.shape[1]} columns")

# --- VIX Stress Regime Labels ---
vix_aligned = vix_prices.reindex(log_returns.index).ffill()
stress_regime = (vix_aligned["VIX"] > VIX_STRESS_THRESHOLD).astype(int)
stress_regime.name = "stress_regime"

n_stress = stress_regime.sum()
n_normal = (stress_regime == 0).sum()
print("    VIX stress regime labels:")
print(f"      Normal periods : {n_normal:,} days ({n_normal/len(stress_regime)*100:.1f}%)")
print(f"      Stress periods : {n_stress:,} days ({n_stress/len(stress_regime)*100:.1f}%)")

# --- Combined Feature Dataset ---
feature_df = pd.concat([log_returns, roll_vol, vix_aligned, stress_regime], axis=1).dropna()
print(f"\n    Final feature dataset: {feature_df.shape[0]:,} rows × {feature_df.shape[1]} columns")


# STEP 5: Walk-Forward Train / Validation / Test Split

print("\n[5/6] Creating walk-forward splits...")

n = len(feature_df)
train_end = int(n * TRAIN_RATIO)
val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

train_df = feature_df.iloc[:train_end]
val_df   = feature_df.iloc[train_end:val_end]
test_df  = feature_df.iloc[val_end:]

print("\n    Split Summary:")
print(f"   {'Set':<12} {'Start':<14} {'End':<14} {'Rows':>6} {'%':>6}")
print(f"   {'-'*54}")
print(f"   {'Train':<12} {str(train_df.index[0].date()):<14} {str(train_df.index[-1].date()):<14} {len(train_df):>6,} {len(train_df)/n*100:>5.1f}%")
print(f"   {'Validation':<12} {str(val_df.index[0].date()):<14} {str(val_df.index[-1].date()):<14} {len(val_df):>6,} {len(val_df)/n*100:>5.1f}%")
print(f"   {'Test':<12} {str(test_df.index[0].date()):<14} {str(test_df.index[-1].date()):<14} {len(test_df):>6,} {len(test_df)/n*100:>5.1f}%")

# Verify test set covers stress periods
print(f"\n    Test set starts: {test_df.index[0].date()} (covers COVID 2020 + 2022 shock)")


# STEP 6: Save Processed Data

print("\n[6/6] Saving processed data...")

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

port_prices.to_csv(f"{PROCESSED_DATA_PATH}clean_prices.csv")
log_returns.to_csv(f"{PROCESSED_DATA_PATH}log_returns.csv")
feature_df.to_csv(f"{PROCESSED_DATA_PATH}features.csv")
train_df.to_csv(f"{PROCESSED_DATA_PATH}train.csv")
val_df.to_csv(f"{PROCESSED_DATA_PATH}val.csv")
test_df.to_csv(f"{PROCESSED_DATA_PATH}test.csv")
vix_aligned.to_csv(f"{PROCESSED_DATA_PATH}vix.csv")
stress_regime.to_csv(f"{PROCESSED_DATA_PATH}stress_regimes.csv")

print(f"    All files saved to {PROCESSED_DATA_PATH}")


# VISUALISATIONS

os.makedirs(FIGURES_PATH, exist_ok=True)

# --- Plot 1: Asset Price History ---
fig, axes = plt.subplots(4, 3, figsize=(18, 14))
axes = axes.flatten()

for i, ticker in enumerate(PORTFOLIO_TICKERS):
    axes[i].plot(port_prices.index, port_prices[ticker], linewidth=0.8, color="#2196F3")
    axes[i].set_title(ticker, fontsize=11, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=30, labelsize=7)
    axes[i].tick_params(axis="y", labelsize=7)

    # Shade stress periods
    for name, (s, e) in STRESS_PERIODS.items():
        axes[i].axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.15, color="red")

plt.suptitle("Asset Price History (2004–2025)\n(Red shading = stress periods)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}01_asset_prices.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n Plot 1 saved: Asset Price History")

# --- Plot 2: Log Returns Distribution ---
fig, axes = plt.subplots(4, 3, figsize=(18, 14))
axes = axes.flatten()

for i, ticker in enumerate(PORTFOLIO_TICKERS):
    axes[i].hist(log_returns[ticker], bins=80, color="#4CAF50", alpha=0.75, edgecolor="none")
    axes[i].axvline(log_returns[ticker].mean(), color="red", linestyle="--", linewidth=1, label="Mean")
    axes[i].set_title(ticker, fontsize=11, fontweight="bold")
    axes[i].tick_params(labelsize=7)

plt.suptitle("Log Return Distributions (2004–2025)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}02_return_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print(" Plot 2 saved: Return Distributions")

# --- Plot 3: VIX History with Stress Regimes ---
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(vix_aligned.index, vix_aligned["VIX"], color="#9C27B0", linewidth=0.8, label="VIX")
ax.axhline(VIX_STRESS_THRESHOLD, color="red", linestyle="--", linewidth=1.2, label=f"Stress Threshold (VIX={VIX_STRESS_THRESHOLD})")
ax.fill_between(vix_aligned.index, vix_aligned["VIX"], VIX_STRESS_THRESHOLD,
                where=(vix_aligned["VIX"] > VIX_STRESS_THRESHOLD),
                alpha=0.3, color="red", label="Stress Regime")

for name, (s, e) in STRESS_PERIODS.items():
    mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
    ax.annotate(name.replace("_", "\n"), xy=(mid, vix_aligned["VIX"].max() * 0.85),
                fontsize=8, ha="center", color="darkred", fontweight="bold")

ax.set_title("VIX Index — Stress Regime Identification (2004–2025)", fontsize=13, fontweight="bold")
ax.set_ylabel("VIX Level")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}03_vix_stress_regimes.png", dpi=150, bbox_inches="tight")
plt.show()
print(" Plot 3 saved: VIX Stress Regimes")

# --- Plot 4: Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(13, 10))
corr = log_returns.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 8}, linewidths=0.5)
ax.set_title("Asset Return Correlation Matrix (2004–2025)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print(" Plot 4 saved: Correlation Heatmap")

# --- Plot 5: Train/Val/Test Split Timeline ---
fig, ax = plt.subplots(figsize=(16, 3))
ax.barh("Dataset", (val_df.index[0] - train_df.index[0]).days, left=0,
        color="#2196F3", alpha=0.8, label=f"Train ({len(train_df):,} days)")
ax.barh("Dataset", (test_df.index[0] - val_df.index[0]).days,
        left=(val_df.index[0] - train_df.index[0]).days,
        color="#FF9800", alpha=0.8, label=f"Validation ({len(val_df):,} days)")
ax.barh("Dataset", (test_df.index[-1] - test_df.index[0]).days,
        left=(test_df.index[0] - train_df.index[0]).days,
        color="#F44336", alpha=0.8, label=f"Test ({len(test_df):,} days)")

ax.set_title("Walk-Forward Train / Validation / Test Split", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlabel("Days from start")
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}05_train_val_test_split.png", dpi=150, bbox_inches="tight")
plt.show()
print(" Plot 5 saved: Train/Val/Test Split")


# SUMMARY

print("\n" + "=" * 60)
print("  PHASE 1 COMPLETE ")
print("=" * 60)
print(f"""
  Assets downloaded    : {len(PORTFOLIO_TICKERS)}
  Date range           : {START_DATE} → {END_DATE}
  Total trading days   : {len(feature_df):,}
  Features engineered  : {feature_df.shape[1]}
  
  Split:
    Train  : {train_df.index[0].date()} → {train_df.index[-1].date()} ({len(train_df):,} rows)
    Val    : {val_df.index[0].date()} → {val_df.index[-1].date()} ({len(val_df):,} rows)
    Test   : {test_df.index[0].date()} → {test_df.index[-1].date()} ({len(test_df):,} rows)

  Stress regime days   : {n_stress:,} ({n_stress/len(stress_regime)*100:.1f}% of data)
  
  Saved files:
     clean_prices.csv
     log_returns.csv
     features.csv
     train.csv / val.csv / test.csv
     vix.csv
     stress_regimes.csv
    
  Charts saved to: {FIGURES_PATH}
     01_asset_prices.png
     02_return_distributions.png
     03_vix_stress_regimes.png
     04_correlation_heatmap.png
     05_train_val_test_split.png
""")
