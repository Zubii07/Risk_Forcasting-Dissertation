# Deep Learning-Based Downside Risk Forecasting & CVaR Portfolio Optimisation Under Market Stress Conditions

A dissertation project comparing classical econometric models (HistVol, GARCH, GJR-GARCH) against deep learning models (LSTM, Transformer) for volatility/CVaR forecasting, and evaluating how forecast quality translates into portfolio level risk management under CVaR optimisation with a dedicated stress testing framework across three historical market crises.

---

## 1. Research Questions

1. Do deep learning models (LSTM, Transformer) forecast downside risk (CVaR) more accurately than classical econometric models (HistVol, GARCH, GJR-GARCH)?
2. Do more accurate risk forecasts translate into better portfolio construction (lower drawdown, lower tail risk)?
3. Are forecasted CVaR values alone — without raw historical return scenarios — sufficient information to drive sound portfolio optimisation?
4. How do all strategies perform specifically during market stress (GFC 2008, COVID-19 2020, Inflation 2022)?

---

## 2. Project Structure

```
risk_forecasting/
├── config/
│   └── config.py                          # All project settings in one place
├── notebooks/
│   ├── phase1_data_pipeline.py            # Data collection, features, VIX regime labels
│   ├── phase2_baseline_models.py          # HistVol + GARCH baselines
│   ├── phase2_enhancements.py             # + GJR-GARCH, regime-split VaR
│   ├── phase3_deep_learning_models.py     # LSTM + Transformer (run on Colab GPU)
│   ├── phase3_enhancements.py             # + DM tests, regime split, attention analysis
│   ├── phase4_experiment1_historical.py   # Historical scenario-based CVaR optimisation
│   ├── phase4_experiment2_forecast_driven.py  # Forecast-driven CVaR optimisation (core)
│   ├── phase4_experiment2b_rolling_corr.py     # Diagnostic: rolling vs fixed correlation
│   ├── phase4_experiment3_forecast_ac.py  # + Min return constraint (A) + regime allocation (C)
│   ├── phase4_comparison.py               # All 3 experiments compared + timing
│   ├── phase4_diagrams.py                 # Pipeline / workflow / framework diagrams
│   ├── phase5_backtesting.py              # Full statistical evaluation
│   └── phase6_stress_analysis.py          # GFC / COVID / Inflation stress analysis
├── src/
│   ├── models/                            # GARCH, GJR-GARCH, LSTM, Transformer, HistVol
│   ├── portfolio/                         # CVaR optimiser (historical + forecast-driven)
│   └── evaluation/                        # Risk metrics, backtesting utilities
├── data/
│   ├── raw/                               # Downloaded price data (regenerated, not delivered)
│   └── processed/                         # Returns, forecasts, weights, VIX, splits
├── results/
│   ├── figures/                           # All charts (44 core + 15 Phase 4 rebuild)
│   └── metrics/                           # All CSV metric outputs
└── requirements.txt
```

---

## 3. Data

| Setting | Value |
|---|---|
| Assets | AAPL, MSFT, NVDA, JPM, GS, JNJ, UNH, XOM, SPY, QQQ, TLT, GLD |
| Volatility Index | ^VIX |
| Date Range | 2004-01-01 → 2025-01-01 |
| Split | 70% Train / 15% Validation / 15% Test (walk-forward, no shuffling) |
| Test Period | ~Nov 2021 → Dec 2024 |
| VIX Stress Threshold | 25 |

**Stress periods analysed:**

| Period | Dates | Type |
|---|---|---|
| GFC 2008 | 2008-09-01 → 2009-03-31 | Credit-driven crash |
| COVID-19 2020 | 2020-02-01 → 2020-04-30 | Pandemic-driven crash |
| Inflation 2022 | 2022-01-01 → 2022-12-31 | Macro/policy-driven stress |

---

## 4. Methodology Overview

### Phase 1 — Data Pipeline
Downloads and cleans price data, computes log returns, rolling volatility (10d/30d), VIX-based stress regime labels, and the walk-forward train/val/test split.

### Phase 2 — Baseline Risk Models
Historical Volatility (30d) and GARCH(1,1) as classical volatility forecasts, converted to VaR/CVaR at 95% confidence. Backtested with Kupiec POF tests.

**Enhancement:** GJR-GARCH added to capture the leverage effect (asymmetric volatility response to negative shocks), plus a regime-split VaR breach table (Normal vs Stress).

### Phase 3 — Deep Learning Models
LSTM (2-layer, 64 units) and Transformer (2-layer, 4-head, d_model=64) trained on a rolling 30-day sequence window, predicting one-step-ahead volatility, converted to VaR/CVaR the same way as Phase 2.

**Enhancement:** Diebold-Mariano statistical significance tests comparing all model pairs, Normal vs Stress regime performance split, and an attention-weight proxy analysis.

### Phase 4 — Portfolio Optimisation (rebuilt into 3 experiments)

This phase was rebuilt following supervisor feedback questioning whether the deep learning forecasts were meaningfully contributing to portfolio construction, given the optimiser also had access to full historical return scenarios.

| Experiment | Optimiser Inputs | Purpose |
|---|---|---|
| **Exp 1 — Historical** | 252-day trailing historical return scenarios only. No forecasts used. | Traditional baseline (Rockafellar-Uryasev CVaR minimisation) |
| **Exp 2 — Forecast-Driven** | Per-asset one-day-ahead CVaR forecast only, + a correlation matrix estimated once from the training period and held fixed. No raw return scenarios. | Tests whether forecasts alone carry sufficient information |
| **Exp 2b — Rolling Correlation** | Same as Exp 2, but the correlation matrix is recomputed every rebalance using a rolling 252-day window. | Diagnostic — isolates whether a *stale* correlation matrix (not the CVaR forecasts) was limiting Exp 2 |
| **Exp 3 — Forecast-Driven + A+C** | Exp 2 + a minimum-return constraint (A, via a compact 60-day EWMA return proxy) + regime-aware allocation blending (C, based on VIX regime) | Tests whether return can be improved without abandoning the forecast-driven philosophy |

**Key finding on Experiment 3 (A+C):** extensive grid testing (return-tilt strength × shrinkage level) showed that *any* tilt toward the EWMA expected-return signal degraded performance monotonically — the best-performing configuration in every test was the one with zero return-tilt. This is consistent with the well-documented sensitivity of mean-variance-style optimisation to expected-return estimation error (Michaud, 1989), and is reported as a finding rather than a limitation.

**Turnover/cost-drag tuning:** a subsequent pass reducing unnecessary rebalancing turnover materially improved net returns across all experiments by cutting transaction cost drag (see Section 6 below) — the DL-forecast-driven CVaR strategies now show a consistent improvement over their pre-tuning results.

### Phase 5 — Backtesting & Statistical Evaluation
17 performance metrics per strategy, block-bootstrap significance tests vs. the EqualWeight benchmark, Jobson-Korkie pairwise Sharpe tests, year-by-year performance breakdown, and regime-conditional analysis.

### Phase 6 — Market Stress Analysis
Full VIX regime labeling across the entire 2004–2024 history, deep-dive performance analysis within each of the three crisis periods, drawdown/recovery analysis, and a master dissertation summary chart.

---

## 5. How to Run

### 5.1 Requirements
- Python 3.10+
- Google account (Phase 3 GPU training only)

### 5.2 Installation
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create the required package files:
```bash
touch src/__init__.py src/models/__init__.py src/evaluation/__init__.py src/portfolio/__init__.py config/__init__.py
```

### 5.3 Run Order

| Step | Script | Where |
|---|---|---|
| 1 | `phase1_data_pipeline.py` | Local |
| 2 | `phase2_baseline_models.py` | Local |
| 3 | `phase2_enhancements.py` | Local |
| 4 | `phase3_deep_learning_models.py` | **Google Colab (GPU)** |
| 5 | `phase3_enhancements.py` | Local |
| 6 | `phase4_experiment1_historical.py` | Local |
| 7 | `phase4_experiment2_forecast_driven.py` | Local |
| 8 | `phase4_experiment2b_rolling_corr.py` | Local |
| 9 | `phase4_experiment3_forecast_ac.py` | Local |
| 10 | `phase4_comparison.py` | Local |
| 11 | `phase4_diagrams.py` | Local (independent) |
| 12 | `phase5_backtesting.py` | Local |
| 13 | `phase6_stress_analysis.py` | Local |

Always run from the project root:
```bash
python -m notebooks.phase1_data_pipeline
```

Total reproduction time from scratch: ~70–90 minutes (Phase 3 GPU training ≈ 20 minutes on a free Colab T4).

All settings (tickers, date ranges, confidence levels, transaction costs, weight limits, return targets) live in `config/config.py`.

---

## 6. Results Summary

### 6.1 Forecast Accuracy (Phase 2 & 3, Test Set)

| Model | MAE | RMSE | QLIKE |
|---|---|---|---|
| HistVol_30d | 0.008835 | 0.011162 | 1.5499 |
| GARCH(1,1) | 0.008918 | 0.011174 | 1.5140 |
| GJR-GARCH | 0.008833 | 0.011109 | 1.5090 |
| LSTM | 0.007358 (best MAE) | 0.009783 | 1.6546 |
| Transformer | **0.007211** | **0.009674** | 1.6497 |

Diebold-Mariano tests confirm the Transformer significantly outperforms both GARCH and LSTM on the full test period (p < 0.001).

### 6.2 Portfolio Optimisation — Final Results (post turnover-tuning)

| Experiment | Ann. Return | Sharpe | Max DD | Avg Turnover | Annual Cost Drag |
|---|---|---|---|---|---|
| Exp1 — Historical | 1.87% | 0.0341 | -15.75% | 0.0138 | 0.35% |
| Exp2 — HistVol | 0.49% | -0.1007 | -18.64% | 0.0343 | 0.86% |
| Exp2 — GARCH | -0.19% | -0.1719 | -20.00% | 0.0439 | 1.11% |
| Exp2 — GJR-GARCH | 0.70% | -0.0819 | -18.99% | 0.0454 | 1.14% |
| Exp2 — LSTM | 0.75% | -0.0758 | -19.48% | 0.0101 | 0.26% |
| **Exp2 — Transformer** | **1.00%** | **-0.0534** | **-18.63%** | 0.0145 | 0.36% |
| Exp3 — Transformer + A+C | 0.66% | -0.0860 | -18.34% | 0.0283 | 0.71% |

Reducing rebalancing turnover materially cut transaction-cost drag and improved net returns across every strategy — most notably for the GARCH-family models, where cost drag more than halved.

### 6.3 Stress Period Performance (Inflation 2022, In-Sample Test Period)
CVaR-optimised strategies substantially outperformed the EqualWeight benchmark during the 2022 inflation shock, with the Transformer-driven strategy providing the strongest downside protection.

---

## 7. Key Findings

1. **Deep learning models forecast volatility/CVaR more accurately than classical models**, with statistical significance confirmed via Diebold-Mariano tests (p < 0.001).
2. **CVaR forecasts alone can meaningfully differentiate portfolio decisions** — different models produce measurably different weight allocations (weight-differentiation analysis, Experiment 2).
3. **Forecast-driven optimisation does not fully replicate the performance of full historical-scenario optimisation**, primarily due to (a) a fixed/non-adaptive correlation structure and (b) the loss of tail-shape information contained in raw scenarios. This is reported as a finding, not hidden as a shortfall.
4. **Return-tilted extensions of the CVaR optimiser (A+C) did not improve returns.** Grid testing across return-tilt strength and shrinkage levels showed that any deviation from pure risk-minimisation degraded performance monotonically — a result consistent with well-established critiques of mean-variance-style optimisation under noisy expected-return estimates.
5. **Portfolio turnover and transaction-cost drag were a material, previously under-examined performance driver.** Explicitly reducing unnecessary rebalancing turnover recovered a meaningful share of net return across all strategies without changing the underlying risk methodology.
6. **All CVaR-optimised strategies reduce maximum drawdown substantially versus an EqualWeight benchmark**, particularly during the 2022 stress period.

---

## 8. Limitations

- The correlation matrix in the forecast-driven experiments is estimated once from the training period and (except in the Exp2b diagnostic) held fixed — a structural simplification, explicitly documented.
- The parametric CVaR formulation assumes a first-order Gaussian relationship between forecasted CVaR and implied volatility; it does not capture fat-tail or skew effects present in raw historical scenarios.
- Statistical significance tests on portfolio-level return differences (Phase 5) did not reach conventional significance thresholds, consistent with the relatively short (~791 trading day) out-of-sample test window common in portfolio backtesting literature. Results are reported as economically, not statistically, significant.
- Expected-return forecasting was tested and found unreliable at this horizon; this is a known, general limitation of short-horizon return forecasting, not specific to this implementation.

---

## 9. Reproducibility

- Random seed 42 fixed for PyTorch/NumPy in Phase 3.
- Walk-forward splits use fixed ratios — no random shuffling.
- Price data may vary marginally by download date due to retrospective adjustments (splits/dividends) — expected and immaterial to conclusions.
- Phase 3 GPU results may vary slightly between Colab sessions due to GPU non-determinism; this does not affect the qualitative conclusions.

---

## 10. Author

**M. Zohaib**
AI/ML Engineer & Data Scientist
