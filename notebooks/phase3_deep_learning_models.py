import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from src.models.lstm_model import LSTMVolatilityModel
from src.models.transformer_model import TransformerVolatilityModel
from src.models.dl_utils import (
    build_volatility_target, create_sequences,
    SequenceDataset, train_model, predict,
)
from src.evaluation.risk_metrics import (
    parametric_var, parametric_cvar,
    evaluate_volatility_forecast, kupiec_pof_test,
)
from torch.utils.data import DataLoader


from config.config import (
    PROCESSED_DATA_PATH, METRICS_PATH, FIGURES_PATH,
    PORTFOLIO_TICKERS, CONFIDENCE_LEVEL, SEQUENCE_LENGTH,
)


# Fix import paths FIRST
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ── Reproducibility ──
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Device ──
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("  PHASE 3: Deep Learning Models (LSTM + Transformer)")
print("=" * 60)
print(f"\n  Device: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")


# Load processed Data

print("\n[1/7] Loading processed data...")

log_returns = pd.read_csv(f"{PROCESSED_DATA_PATH}log_returns.csv", index_col=0, parse_dates=True)
train_df    = pd.read_csv(f"{PROCESSED_DATA_PATH}train.csv", index_col=0, parse_dates=True)
val_df      = pd.read_csv(f"{PROCESSED_DATA_PATH}val.csv",   index_col=0, parse_dates=True)
test_df     = pd.read_csv(f"{PROCESSED_DATA_PATH}test.csv",  index_col=0, parse_dates=True)

asset_cols = [c for c in log_returns.columns if c in PORTFOLIO_TICKERS]
returns    = log_returns[asset_cols].copy()

train_end = train_df.index[-1]
val_end   = val_df.index[-1]
test_start = test_df.index[0]
test_end   = test_df.index[-1]

print(f"   ✅ {len(asset_cols)} assets | {len(returns):,} rows")
print(f"   ✅ Seq length: {SEQUENCE_LENGTH} | Train→{train_end.date()} Val→{val_end.date()} Test→{test_end.date()}")


# Train LSTM & Transformer per asset

print("\n[2/7] Training deep learning models (per asset)...")
print("   (LSTM + Transformer trained separately for each asset)\n")

TARGET_WINDOW = 5    # predict 5-day forward realised volatility
BATCH_SIZE    = 64
EPOCHS        = 50

lstm_forecasts  = pd.DataFrame(index=returns.index, columns=asset_cols, dtype=float)
trans_forecasts = pd.DataFrame(index=returns.index, columns=asset_cols, dtype=float)
histories       = {}

start_time = time.time()

for ai, asset in enumerate(asset_cols, 1):
    print(f"   [{ai}/{len(asset_cols)}] {asset}")
    r = returns[asset].values
    target = build_volatility_target(r, target_window=TARGET_WINDOW)

    # Build sequences
    X, y, idx = create_sequences(r, target, SEQUENCE_LENGTH)
    seq_dates = returns.index[idx]

    # Split by date
    tr_mask = seq_dates <= train_end
    va_mask = (seq_dates > train_end) & (seq_dates <= val_end)
    te_mask = seq_dates > val_end

    # Scale features using TRAIN stats only (no leakage)
    scaler = StandardScaler()
    X_flat = X.reshape(-1, X.shape[-1])
    scaler.fit(X[tr_mask].reshape(-1, X.shape[-1]))
    X_scaled = scaler.transform(X_flat).reshape(X.shape)

    X_tr, y_tr = X_scaled[tr_mask], y[tr_mask]
    X_va, y_va = X_scaled[va_mask], y[va_mask]
    X_te, y_te = X_scaled[te_mask], y[te_mask]

    if len(X_tr) == 0 or len(X_va) == 0:
        print(f"      ⚠️  insufficient data, skipping {asset}")
        continue

    train_loader = DataLoader(SequenceDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SequenceDataset(X_va, y_va), batch_size=BATCH_SIZE, shuffle=False)

    # LSTM
    print("      Training LSTM ...")
    lstm = LSTMVolatilityModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
    lstm, lstm_hist = train_model(lstm, train_loader, val_loader,
                                  epochs=EPOCHS, lr=1e-3, patience=8,
                                  device=DEVICE, verbose=True)
    lstm_pred = predict(lstm, X_scaled[te_mask], device=DEVICE)
    lstm_forecasts.loc[seq_dates[te_mask], asset] = lstm_pred

    # Transformer 
    print("      Training Transformer ...")
    trans = TransformerVolatilityModel(input_size=1, d_model=64, nhead=4,
                                       num_layers=2, dim_ff=128, dropout=0.1)
    trans, trans_hist = train_model(trans, train_loader, val_loader,
                                    epochs=EPOCHS, lr=1e-3, patience=8,
                                    device=DEVICE, verbose=True)
    trans_pred = predict(trans, X_scaled[te_mask], device=DEVICE)
    trans_forecasts.loc[seq_dates[te_mask], asset] = trans_pred

    histories[asset] = {"lstm": lstm_hist, "transformer": trans_hist}
    print(f"      ✅ Done ({len(X_te)} test predictions)\n")

elapsed = time.time() - start_time
print(f"   ✅ All models trained in {elapsed/60:.1f} minutes")

lstm_forecasts  = lstm_forecasts.dropna(how="all")
trans_forecasts = trans_forecasts.dropna(how="all")


# VaR & CVaR from DL forecasts

print("\n[3/7] Computing VaR & CVaR from DL forecasts...")

lstm_var   = pd.DataFrame(index=lstm_forecasts.index,  columns=asset_cols, dtype=float)
lstm_cvar  = pd.DataFrame(index=lstm_forecasts.index,  columns=asset_cols, dtype=float)
trans_var  = pd.DataFrame(index=trans_forecasts.index, columns=asset_cols, dtype=float)
trans_cvar = pd.DataFrame(index=trans_forecasts.index, columns=asset_cols, dtype=float)

for asset in asset_cols:
    if asset in lstm_forecasts.columns:
        lstm_var [asset] = parametric_var (lstm_forecasts[asset], CONFIDENCE_LEVEL)
        lstm_cvar[asset] = parametric_cvar(lstm_forecasts[asset], CONFIDENCE_LEVEL)
        trans_var [asset] = parametric_var (trans_forecasts[asset], CONFIDENCE_LEVEL)
        trans_cvar[asset] = parametric_cvar(trans_forecasts[asset], CONFIDENCE_LEVEL)

print("   ✅ VaR & CVaR computed")


# Evaluate on test set

print("\n[4/7] Evaluating on test set...")

eval_rows = []
for asset in asset_cols:
    rets_test = returns[asset].loc[test_start:test_end]
    if asset in lstm_forecasts.columns:
        l_test = lstm_forecasts[asset].loc[test_start:test_end]
        if not l_test.dropna().empty:
            eval_rows.append({"Asset": asset, "Model": "LSTM", **evaluate_volatility_forecast(l_test, rets_test)})
    if asset in trans_forecasts.columns:
        t_test = trans_forecasts[asset].loc[test_start:test_end]
        if not t_test.dropna().empty:
            eval_rows.append({"Asset": asset, "Model": "Transformer", **evaluate_volatility_forecast(t_test, rets_test)})

eval_df = pd.DataFrame(eval_rows)
agg = eval_df.groupby("Model")[["MAE", "RMSE", "QLIKE"]].mean().round(6)

print("\n   Forecast Accuracy on Test Set:")
print(eval_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("\n   Average across all assets:")
print(agg.to_string())

# Merge with Phase 2 metrics for full comparison
try:
    p2 = pd.read_csv(f"{METRICS_PATH}phase2_forecast_metrics.csv")
    full_compare = pd.concat([p2, eval_df], ignore_index=True)
    full_agg = full_compare.groupby("Model")[["MAE", "RMSE", "QLIKE"]].mean().round(6)
    print("\n   ── FULL MODEL COMPARISON (all 4 models) ──")
    print(full_agg.to_string())
except FileNotFoundError:
    full_compare = eval_df
    full_agg = agg

# VaR Backtests

print("\n[5/7] Running Kupiec POF backtests...")

backtest_rows = []
for asset in asset_cols:
    rets_test = returns[asset].loc[test_start:test_end]
    if asset in lstm_var.columns:
        v = lstm_var[asset].loc[test_start:test_end]
        if not v.dropna().empty:
            backtest_rows.append({"Asset": asset, "Model": "LSTM", **kupiec_pof_test(rets_test, v, CONFIDENCE_LEVEL)})
    if asset in trans_var.columns:
        v = trans_var[asset].loc[test_start:test_end]
        if not v.dropna().empty:
            backtest_rows.append({"Asset": asset, "Model": "Transformer", **kupiec_pof_test(rets_test, v, CONFIDENCE_LEVEL)})

backtest_df = pd.DataFrame(backtest_rows)
pct_adequate = backtest_df.groupby("Model")["Model_Adequate"].mean() * 100
print("\n   % of assets where VaR model is adequate:")
for model, pct in pct_adequate.items():
    print(f"     {model:<15} {pct:>5.1f}%")

# STEP 6: Save outputs

print("\n[6/7] Saving outputs...")

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(METRICS_PATH, exist_ok=True)

lstm_forecasts.to_csv (f"{PROCESSED_DATA_PATH}lstm_forecasts.csv")
trans_forecasts.to_csv(f"{PROCESSED_DATA_PATH}transformer_forecasts.csv")
lstm_var.to_csv       (f"{PROCESSED_DATA_PATH}lstm_var.csv")
lstm_cvar.to_csv      (f"{PROCESSED_DATA_PATH}lstm_cvar.csv")
trans_var.to_csv      (f"{PROCESSED_DATA_PATH}transformer_var.csv")
trans_cvar.to_csv     (f"{PROCESSED_DATA_PATH}transformer_cvar.csv")
eval_df.to_csv        (f"{METRICS_PATH}phase3_forecast_metrics.csv", index=False)
backtest_df.to_csv    (f"{METRICS_PATH}phase3_var_backtests.csv", index=False)
full_compare.to_csv   (f"{METRICS_PATH}all_models_comparison.csv", index=False)

# Save trained model weights (last asset as example; full loop saves all if needed)
os.makedirs(f"{PROJECT_ROOT}/models_saved", exist_ok=True)
print("   ✅ All outputs saved")

# STEP 7: Visualisations

print("\n[7/7] Generating charts...")
os.makedirs(FIGURES_PATH, exist_ok=True)

# Plot 10: Training curves (SPY example)
demo = "SPY" if "SPY" in histories else list(histories.keys())[0]
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, (mkey, mlabel) in zip(axes, [("lstm", "LSTM"), ("transformer", "Transformer")]):
    h = histories[demo][mkey]
    ax.plot(h["train_loss"], label="Train", color="#2196F3")
    ax.plot(h["val_loss"],   label="Validation", color="#E91E63")
    ax.set_title(f"{mlabel} Training Curve ({demo})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.set_yscale("log")
plt.suptitle("Phase 3 — Deep Learning Training Curves", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}10_dl_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("   📊 Plot 10 saved: Training Curves")

# Plot 11: DL volatility forecasts (4 assets)
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
axes = axes.flatten()
demo_assets = [a for a in ["SPY", "AAPL", "TLT", "NVDA"] if a in asset_cols]
for i, asset in enumerate(demo_assets):
    ax = axes[i]
    rv = returns[asset].abs().loc[test_start:test_end]
    l_ = lstm_forecasts[asset].loc[test_start:test_end]  if asset in lstm_forecasts.columns  else pd.Series(dtype=float)
    t_ = trans_forecasts[asset].loc[test_start:test_end] if asset in trans_forecasts.columns else pd.Series(dtype=float)
    ax.plot(rv.index, rv, color="lightgray", linewidth=0.6, label="Realised (|r|)", alpha=0.7)
    ax.plot(l_.index, l_, color="#FF9800", linewidth=1.1, label="LSTM")
    ax.plot(t_.index, t_, color="#4CAF50", linewidth=1.1, label="Transformer")
    ax.set_title(f"{asset} — DL Volatility Forecasts", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
plt.suptitle("Phase 3 — Deep Learning Volatility Forecasts (Test Set)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}11_dl_volatility_forecasts.png", dpi=150, bbox_inches="tight")
plt.close()
print("   📊 Plot 11 saved: DL Volatility Forecasts")

# Plot 12: Full 4-model comparison (avg metrics)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
model_order = ["HistVol_30d", "GARCH(1,1)", "LSTM", "Transformer"]
colors = ["#90A4AE", "#2196F3", "#FF9800", "#4CAF50"]
for i, metric in enumerate(["MAE", "RMSE", "QLIKE"]):
    vals = [full_agg.loc[m, metric] if m in full_agg.index else np.nan for m in model_order]
    bars = axes[i].bar(model_order, vals, color=colors, edgecolor="black", linewidth=0.6)
    axes[i].set_title(f"{metric} (lower = better)", fontsize=12, fontweight="bold")
    axes[i].tick_params(axis="x", rotation=20, labelsize=9)
    axes[i].grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            axes[i].text(b.get_x()+b.get_width()/2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
plt.suptitle("All 4 Models — Average Forecast Accuracy (Test Set)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}12_all_models_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("   📊 Plot 12 saved: All Models Comparison")

# SUMMARY
print("\n" + "=" * 60)
print("  PHASE 3 COMPLETE ✅")
print("=" * 60)
print(f"""
  Models built:
    ✅ LSTM (2-layer, 64 hidden units)
    ✅ Transformer (2-layer, 4 heads, d_model=64) [PRIMARY]

  Training: {DEVICE.upper()} | {len(asset_cols)} assets | seq_len={SEQUENCE_LENGTH}

  Average test-set metrics (all 4 models):
{full_agg.to_string()}

  VaR adequacy:
{pct_adequate.to_string()}

  Files saved:
    ✅ lstm_forecasts.csv, transformer_forecasts.csv
    ✅ lstm_var/cvar.csv, transformer_var/cvar.csv
    ✅ phase3_forecast_metrics.csv
    ✅ all_models_comparison.csv

  Charts:
    ✅ 10_dl_training_curves.png
    ✅ 11_dl_volatility_forecasts.png
    ✅ 12_all_models_comparison.png

  ➡️  Ready for Phase 4: CVaR Portfolio Optimisation
""")