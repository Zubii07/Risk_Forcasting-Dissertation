"""
PHASE 4 — CONCEPTUAL DIAGRAMS
"""

import os
import sys
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from config.config import FIGURES_PATH
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


os.makedirs(FIGURES_PATH, exist_ok=True)

NAVY   = "#1F3864"
BLUE   = "#2E75B6"
LBLUE  = "#DEEAF1"
GREEN  = "#1B5E20"
LGREEN = "#E8F5E9"
ORANGE = "#C55A11"
LORANGE= "#FCE4D6"
RED    = "#B71C1C"
LRED   = "#FFEBEE"
GREY   = "#616161"
LGREY  = "#F2F2F2"
WHITE  = "#FFFFFF"

print("=" * 60)
print("  PHASE 4 — CONCEPTUAL DIAGRAMS")
print("=" * 60)


def box(ax, x, y, w, h, text, facecolor=LBLUE, edgecolor=BLUE,
        fontsize=10, fontweight="bold", textcolor="black", lw=1.8):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=textcolor,
            wrap=True)
    return rect


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.8, style="-|>", connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=18,
                        color=color, linewidth=lw,
                        connectionstyle=connectionstyle)
    ax.add_patch(a)

# DIAGRAM 1: Complete End-to-End Pipeline
print("\n[1/4] Building Diagram D1 — End-to-End Pipeline...")

fig, ax = plt.subplots(figsize=(20, 6))
ax.set_xlim(0, 20)
ax.set_ylim(0, 6)
ax.axis("off")

stages = [
    ("DATA\n\n12 assets\n2004-2024\nVIX regime labels", 0.5, LBLUE, BLUE),
    ("FORECASTING\n\nHistVol / GARCH\nGJR-GARCH / LSTM\nTransformer\n\n→ CVaR forecasts", 5.0, LGREEN, GREEN),
    ("OPTIMISATION\n\nExp1: Historical\nExp2: Forecast-driven\nExp3: + A+C\n\n→ Portfolio weights", 10.0, LORANGE, ORANGE),
    ("EVALUATION\n\nReturns, Sharpe,\nDrawdown, CVaR\nStress-period tests", 15.0, LRED, RED),
]
box_w, box_h = 3.8, 4.0
y0 = 1.0

for text, x, fc, ec in stages:
    box(ax, x, y0, box_w, box_h, text, facecolor=fc, edgecolor=ec, fontsize=11)

for i in range(len(stages)-1):
    x1 = stages[i][1] + box_w
    x2 = stages[i+1][1]
    arrow(ax, x1+0.05, y0+box_h/2, x2-0.05, y0+box_h/2, color=NAVY, lw=2.5)

ax.text(10, 5.6, "Complete End-to-End Modelling Pipeline",
        ha="center", fontsize=16, fontweight="bold", color=NAVY)
ax.text(10, 0.3, "Each stage's output is intentionally chosen and understood — no unexamined inputs",
        ha="center", fontsize=10, fontstyle="italic", color=GREY)

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}D1_end_to_end_pipeline.png", dpi=150, bbox_inches="tight")
plt.close()
print("Diagram D1 saved: End-to-End Pipeline")

# DIAGRAM 2: Historical vs Forecast-Driven Workflow
print("\n[2/4] Building Diagram D2 — Historical vs Forecast-Driven Workflow...")

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")

ax.text(9, 9.5, "Experiment 1 vs Experiment 2 — What the Optimiser Actually Sees",
        ha="center", fontsize=15, fontweight="bold", color=NAVY)

# ── Top row: Experiment 1 (Historical) ──
ax.text(4.5, 8.5, "EXPERIMENT 1 — Historical Optimisation (Baseline)",
        ha="center", fontsize=12, fontweight="bold", color=GREY)
box(ax, 0.5, 6.5, 3.5, 1.5, "252 days of\nraw historical\nreturns (scenarios)", LGREY, GREY, 9)
arrow(ax, 4.1, 7.25, 5.4, 7.25, color=GREY, lw=2.2)
box(ax, 5.5, 6.5, 3.5, 1.5, "Rockafellar-Uryasev\nCVaR minimisation\n(scenario-based LP)", LGREY, GREY, 9)
arrow(ax, 9.1, 7.25, 10.4, 7.25, color=GREY, lw=2.2)
box(ax, 10.5, 6.5, 3.5, 1.5, "Portfolio\nweights", LGREY, GREY, 10)
ax.text(15.2, 7.25, "❌ No model forecasts\n   used at all",
        ha="left", fontsize=9, color=RED, fontweight="bold")

# ── Bottom row: Experiment 2 (Forecast-Driven) ──
ax.text(4.5, 4.8, "EXPERIMENT 2 — Forecast-Driven Optimisation",
        ha="center", fontsize=12, fontweight="bold", color=GREEN)
box(ax, 0.5, 2.5, 3.5, 1.7, "One CVaR number\nper asset\n(from LSTM/Transformer/\nGARCH etc.)", LGREEN, GREEN, 9)
arrow(ax, 4.1, 3.35, 5.4, 3.35, color=GREEN, lw=2.2)
box(ax, 5.5, 2.5, 3.5, 1.7, "Fixed correlation\n(training period only,\nheld constant)", LGREEN, GREEN, 9)
arrow(ax, 9.1, 3.35, 10.4, 3.35, color=GREEN, lw=2.2)
box(ax, 10.5, 2.5, 3.5, 1.7, "Parametric CVaR\nminimisation\n(convex QP)", LGREEN, GREEN, 9)
arrow(ax, 14.1, 3.35, 15.4, 3.35, color=GREEN, lw=2.2)
box(ax, 15.5, 2.5, 2.3, 1.7, "Portfolio\nweights", LGREEN, GREEN, 9)

ax.text(9, 1.0,
        "Key difference: Experiment 2 NEVER sees the 252-day return matrix — only a single forecasted risk number per asset.",
        ha="center", fontsize=10, fontstyle="italic", color=NAVY,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=LBLUE, edgecolor=BLUE))

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}D2_historical_vs_forecast_driven_workflow.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(" Diagram D2 saved: Historical vs Forecast-Driven Workflow")

# DIAGRAM 3: A+C Framework Diagram
print("\n[3/4] Building Diagram D3 — A+C Framework...")

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")

ax.text(9, 10.5, "Experiment 3 — Forecast-Driven + A+C Framework",
        ha="center", fontsize=15, fontweight="bold", color=NAVY)

# Base: Experiment 2 core
box(ax, 6.5, 8.3, 5.0, 1.3, "EXPERIMENT 2 CORE\nCVaR forecast + fixed correlation\n→ Defensive weights (w_defensive)",
    LGREEN, GREEN, 10)

# Enhancement A
box(ax, 0.5, 6.0, 5.0, 1.7,
    "ENHANCEMENT A\nMinimum Return Constraint\n\nCompact EWMA expected return\n(60-day span)\n+ target: 6% annualised",
    LORANGE, ORANGE, 9)
arrow(ax, 5.5, 6.85, 6.5, 8.5, color=ORANGE, lw=2, connectionstyle="arc3,rad=-0.2")

box(ax, 12.5, 6.0, 5.0, 1.7,
    "OUTPUT\nReturn-constrained weights\n(w_return)\n\nSolved via convex QP\nwith return ≥ target constraint",
    LORANGE, ORANGE, 9)
arrow(ax, 12.4, 6.85, 11.5, 8.5, color=ORANGE, lw=2, connectionstyle="arc3,rad=0.2")

# Enhancement C
box(ax, 3.0, 3.2, 5.0, 1.7,
    "ENHANCEMENT C\nRegime-Aware Allocation\n\nVIX ≤ 25 → Normal\nVIX > 25  → Stress",
    LBLUE, BLUE, 9)
arrow(ax, 5.5, 4.9, 8.5, 6.0, color=BLUE, lw=2, connectionstyle="arc3,rad=-0.2")

box(ax, 10.0, 3.2, 5.0, 1.7,
    "BLENDING RULE\n\nStress  → 100% w_defensive\nNormal → 70% w_return\n              + 30% w_defensive",
    LBLUE, BLUE, 9)
arrow(ax, 9.9, 4.9, 8.5, 6.0, color=BLUE, lw=2, connectionstyle="arc3,rad=0.2")
arrow(ax, 15.0, 4.9, 13.0, 6.0, color=BLUE, lw=2, connectionstyle="arc3,rad=-0.2")

# Final output
box(ax, 6.5, 0.6, 5.0, 1.5,
    "FINAL PORTFOLIO WEIGHTS\n(w_final, per rebalance)\n\nBalances downside protection\nwith improved annual return",
    LGREEN, GREEN, 10, textcolor="black")
arrow(ax, 6.0, 3.3, 7.5, 2.1, color=GREY, lw=2, connectionstyle="arc3,rad=0.15")
arrow(ax, 12.5, 3.3, 11.5, 2.1, color=GREY, lw=2, connectionstyle="arc3,rad=-0.15")

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}D3_ac_framework_diagram.png", dpi=150, bbox_inches="tight")
plt.close()
print(" Diagram D3 saved: A+C Framework")

# DIAGRAM 4: Regime-Switching Workflow
print("\n[4/4] Building Diagram D4 — Regime-Switching Workflow...")

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

ax.text(8, 8.5, "Regime-Switching Allocation Workflow (Rebalanced Weekly)",
        ha="center", fontsize=15, fontweight="bold", color=NAVY)

# Decision diamond (approximate with box)
box(ax, 6, 6.3, 4, 1.4, "Check current VIX\nvs threshold (25)", LBLUE, BLUE, 11)

arrow(ax, 6.5, 6.3, 3.5, 4.4, color=RED, lw=2.4, connectionstyle="arc3,rad=0.2")
ax.text(4.3, 5.5, "VIX > 25\n(Stress)", ha="center", fontsize=10,
        fontweight="bold", color=RED)

arrow(ax, 9.5, 6.3, 12.5, 4.4, color=GREEN, lw=2.4, connectionstyle="arc3,rad=-0.2")
ax.text(11.7, 5.5, "VIX ≤ 25\n(Normal)", ha="center", fontsize=10,
        fontweight="bold", color=GREEN)

box(ax, 1.0, 2.6, 5.0, 1.7,
    "STRESS REGIME\n\n100% Defensive weights\n(pure CVaR minimisation)\n\nPrioritise capital protection",
    LRED, RED, 10)

box(ax, 10.0, 2.6, 5.0, 1.7,
    "NORMAL REGIME\n\n70% Return-seeking +\n30% Defensive blend\n\nPursue higher return\nwhile capping risk",
    LGREEN, GREEN, 10)

arrow(ax, 3.5, 2.6, 6.5, 1.3, color=GREY, lw=2, connectionstyle="arc3,rad=0.15")
arrow(ax, 12.5, 2.6, 9.5, 1.3, color=GREY, lw=2, connectionstyle="arc3,rad=-0.15")

box(ax, 6.0, 0.3, 4.0, 1.1, "Rebalanced portfolio\napplied for the week",
    LBLUE, BLUE, 10)

ax.text(8, -0.3,
        "This switch happens automatically at every weekly rebalance — no manual intervention.",
        ha="center", fontsize=9, fontstyle="italic", color=GREY)

plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}D4_regime_switching_workflow.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(" Diagram D4 saved: Regime-Switching Workflow")

print("\n" + "=" * 60)
print("  All 4 conceptual diagrams complete")
print("=" * 60)
print("""
  Files saved:
    D1_end_to_end_pipeline.png
    D2_historical_vs_forecast_driven_workflow.png
    D3_ac_framework_diagram.png
    D4_regime_switching_workflow.png
""")