"""Plot win-rate bar chart from eval_results.json files.

Usage:
    # Single run
    python scripts/plotting/plot_win_rates.py --results results/pe/save-pursuit_evasion-.../eval_results.json

    # Compare multiple strategies
    python scripts/plotting/plot_win_rates.py \
        --results results/pe/run_vp/eval_results.json results/pe/run_fp/eval_results.json \
        --labels "Vanilla Play" "Fictitious Play"

    # Save to file instead of showing
    python scripts/plotting/plot_win_rates.py --results ... --output win_rates.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

DARKNESS = "#060606"


def load_results(path: str) -> dict[str, int]:
    """Load eval_results.json and tally outcomes."""
    with open(path) as f:
        episodes = json.load(f)
    tallies: dict[str, int] = {}
    for ep in episodes:
        outcome = ep["outcome"]
        tallies[outcome] = tallies.get(outcome, 0) + 1
    return tallies


def plot_win_rates(labels: list[str], tallies_list: list[dict[str, int]],
                   output: str | None = None):
    """Stacked bar chart of outcomes per strategy/run."""
    # Collect all unique outcome names across runs
    all_outcomes = sorted({k for t in tallies_list for k in t})
    colors = {
        "pursuers_win": "cyan",
        "evaders_win": "magenta",
        "draw": "#F5F5F5",
        "team_0_wins": "cyan",
        "team_1_wins": "magenta",
        "ties": "#F5F5F5",
    }
    default_colors = plt.cm.Set2.colors

    x = np.arange(len(labels))
    bar_width = 0.6

    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.grid": True,
        "grid.color": "#444444",
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#d0d0d0",
        "xtick.color": "#d0d0d0",
        "ytick.color": "#d0d0d0",
        "legend.edgecolor": "#444444",
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARKNESS)
    ax.set_facecolor(DARKNESS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    bottom = np.zeros(len(labels))
    for i, outcome in enumerate(all_outcomes):
        values = [t.get(outcome, 0) for t in tallies_list]
        color = colors.get(outcome, default_colors[i % len(default_colors)])
        ax.bar(x, values, bar_width, bottom=bottom, label=outcome, color=color)

        # Value labels on each segment
        for j in range(len(labels)):
            if values[j] > 0:
                ax.text(x[j], bottom[j] + values[j] / 2, str(values[j]),
                        ha="center", va="center", color="black", fontsize=9)
        bottom += np.array(values, dtype=float)

    total = int(max(bottom)) if len(bottom) > 0 else 100
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Number of Games")
    ax.set_title(f"Game Outcomes ({total} games per strategy)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, total)
    ax.legend()

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved to {output}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot win-rate comparison chart")
    parser.add_argument("--results", nargs="+", required=True,
                        help="Paths to eval_results.json files")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each result file (default: filenames)")
    parser.add_argument("--output", default=None,
                        help="Save plot to file instead of showing")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.results):
        parser.error("--labels must have same length as --results")

    labels = args.labels or [p.split("/")[-2] for p in args.results]
    tallies_list = [load_results(p) for p in args.results]

    plot_win_rates(labels, tallies_list, output=args.output)
