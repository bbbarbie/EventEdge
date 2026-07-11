"""Core research experiments: calibration bias x informed-fraction sweep.

Runs the C++ simulator over a grid of (calibration_bias, informed_fraction)
with many seeds per cell, aggregates terminal P&L, and produces the two
headline plots:

  1. results/pnl_vs_calibration_bias.png
     Mean terminal P&L vs calibration bias, one line per informed fraction.

  2. results/heatmap_bias_x_informed.png
     Mean terminal P&L over the full bias x informed-fraction grid.

All simulation logic lives in C++; this script only orchestrates runs and
aggregates the per-run summary CSVs.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY = REPO_ROOT / "eventedge"
DATA_DIR = REPO_ROOT / "data" / "bias_sweep"
RESULTS_DIR = REPO_ROOT / "results"

BIAS_GRID = np.round(np.arange(-0.10, 0.1001, 0.02), 3)   # 11 values
INFORMED_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]            # 6 values
NUM_SEEDS = 30
NUM_STEPS = 2000
MAX_WORKERS = 8


def run_one(args: tuple[float, float, int]) -> Path:
    bias, informed, seed = args
    prefix = DATA_DIR / f"b{bias:+.2f}_i{informed:.1f}_s{seed}"
    summary_path = Path(f"{prefix}_summary.csv")
    if not summary_path.exists():  # cheap resume support
        subprocess.run(
            [str(BINARY),
             "--seed", str(seed),
             "--steps", str(NUM_STEPS),
             "--bias", f"{bias:.4f}",
             "--informed-fraction", f"{informed:.4f}",
             "--log-detail", "0",
             "--out-prefix", str(prefix)],
            check=True, capture_output=True,
        )
    return summary_path


def run_sweep() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(BIAS_GRID, INFORMED_GRID, range(NUM_SEEDS)))
    print(f"Running {len(grid)} simulations "
          f"({len(BIAS_GRID)} biases x {len(INFORMED_GRID)} informed fractions "
          f"x {NUM_SEEDS} seeds, {NUM_STEPS} steps each)...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        paths = list(pool.map(run_one, grid, chunksize=16))
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def plot_pnl_vs_bias(runs: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")

    for idx, informed in enumerate(INFORMED_GRID):
        sub = runs[np.isclose(runs["informed_fraction"], informed)]
        stats = sub.groupby("calibration_bias")["terminal_pnl"].agg(["mean", "sem"])
        color = cmap(idx / max(len(INFORMED_GRID) - 1, 1))
        ax.plot(stats.index, stats["mean"], marker="o", color=color,
                label=f"informed = {informed:.0%}")
        ax.fill_between(stats.index,
                        stats["mean"] - 1.96 * stats["sem"],
                        stats["mean"] + 1.96 * stats["sem"],
                        color=color, alpha=0.15)

    ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax.axvline(0.0, color="grey", linewidth=1, linestyle=":")
    ax.set_xlabel("Calibration bias (added to MM probability estimate)")
    ax.set_ylabel("Mean terminal P&L (contracts x $1)")
    ax.set_title("Market-maker terminal P&L vs calibration bias\n"
                 f"{NUM_SEEDS} seeds per point, {NUM_STEPS} steps, 95% CI")
    ax.legend(title="Informed fraction")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_heatmap(runs: pd.DataFrame, out_path: Path) -> None:
    pivot = (runs.groupby(["informed_fraction", "calibration_bias"])
             ["terminal_pnl"].mean().unstack("calibration_bias"))

    fig, ax = plt.subplots(figsize=(9, 5))
    vmax = np.abs(pivot.to_numpy()).max()
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", origin="lower",
                   vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)),
                  [f"{b:+.2f}" for b in pivot.columns])
    ax.set_yticks(range(len(pivot.index)),
                  [f"{i:.0%}" for i in pivot.index])
    ax.set_xlabel("Calibration bias")
    ax.set_ylabel("Informed fraction")
    ax.set_title("Mean terminal P&L: calibration bias x informed fraction\n"
                 f"{NUM_SEEDS} seeds per cell, {NUM_STEPS} steps")

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            ax.text(col, row, f"{value:.0f}", ha="center", va="center",
                    fontsize=7, color="black")

    fig.colorbar(im, ax=ax, label="Mean terminal P&L")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main() -> int:
    if not BINARY.exists():
        print(f"Binary not found: {BINARY}. Build the C++ simulator first.")
        return 1

    runs = run_sweep()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RESULTS_DIR / "bias_sweep_runs.csv", index=False)

    table = (runs.groupby(["informed_fraction", "calibration_bias"])
             ["terminal_pnl"].mean().unstack("calibration_bias"))
    print("\nMean terminal P&L (rows: informed fraction, cols: bias):")
    print(table.round(1).to_string())

    plot_pnl_vs_bias(runs, RESULTS_DIR / "pnl_vs_calibration_bias.png")
    plot_heatmap(runs, RESULTS_DIR / "heatmap_bias_x_informed.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
