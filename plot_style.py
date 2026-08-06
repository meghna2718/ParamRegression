"""
plot_style.py
==============
Shared, publication-leaning matplotlib theme + color palette for
analyze_results.py and compare_runs.py. Import and call apply_style() once;
use PALETTE / CATEGORICAL for consistent colors across every plot in the
pipeline. All plotting functions still read from the saved JSON/CSV/pickle
artifacts, so re-running with a different style here (or writing your own
plotting from those files) never requires retraining.
"""
import matplotlib.pyplot as plt

PALETTE = {
    "ml": "#2563eb",       # blue -- this model
    "acts": "#dc2626",     # red  -- ACTS classical benchmark
    "best": "#16a34a",     # green -- highlight winner in comparisons
    "neutral": "#64748b",  # slate gray -- baselines / negative importance
    "train": "#2563eb",
    "val": "#f97316",      # orange -- validation curve
}

# For sweeps with >2 variants (e.g. none/sinusoidal/learnable)
CATEGORICAL = ["#2563eb", "#f97316", "#16a34a", "#9333ea", "#dc2626", "#0891b2"]


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
    })


def style_bars(bars, edgecolor="white", linewidth=0.6):
    for b in bars:
        b.set_edgecolor(edgecolor)
        b.set_linewidth(linewidth)
