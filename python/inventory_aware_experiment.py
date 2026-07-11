"""Fixed-spread vs inventory-aware quoting across calibration bias.

The P&L decomposition showed that a biased MM bleeds through two channels:
adverse selection on mispriced quotes and a large signed inventory position
carried into a binary settlement. Inventory-aware quoting attacks the second
channel by skewing the quote center against accumulated inventory:

    quote_center = mm_estimate - inventory_aversion * inventory

This experiment reruns the calibration-bias sweep under both strategies
(several aversion levels) and compares terminal P&L and terminal inventory.

Outputs:
    results/inventory_aware_vs_fixed.png
    results/inventory_aware_runs.csv
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
DATA_DIR = REPO_ROOT / "data" / "inventory_sweep"
RESULTS_DIR = REPO_ROOT / "results"

BIAS_GRID = np.round(np.arange(-0.10, 0.1001, 0.02), 3)
INFORMED_FRACTION = 0.3
NUM_SEEDS = 30
NUM_STEPS = 2000
MAX_WORKERS = 8

# (strategy flag, inventory aversion, label, color)
STRATEGIES = [
    ("fixed", 0.0, "Fixed spread", "#c0392b"),
    ("inventory", 0.001, "Inventory-aware, k=0.001", "#f39c12"),
    ("inventory", 0.002, "Inventory-aware, k=0.002", "#27ae60"),
    ("inventory", 0.005, "Inventory-aware, k=0.005", "#2980b9"),
]


def run_one(args: tuple[str, float, float, int]) -> Path:
    strategy, aversion, bias, seed = args
    prefix = DATA_DIR / f"{strategy}_k{aversion:.3f}_b{bias:+.2f}_s{seed}"
    summary_path = Path(f"{prefix}_summary.csv")
    if not summary_path.exists():
        subprocess.run(
            [str(BINARY),
             "--seed", str(seed),
             "--steps", str(NUM_STEPS),
             "--bias", f"{bias:.4f}",
             "--informed-fraction", f"{INFORMED_FRACTION:.4f}",
             "--mm-strategy", strategy,
             "--inventory-aversion", f"{aversion:.5f}",
             "--log-detail", "0",
             "--out-prefix", str(prefix)],
            check=True, capture_output=True,
        )
    return summary_path


def run_sweep() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    grid = [(s, k, b, seed)
            for (s, k, _, _) in STRATEGIES
            for b in BIAS_GRID
            for seed in range(NUM_SEEDS)]
    print(f"Running {len(grid)} simulations...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        paths = list(pool.map(run_one, grid, chunksize=16))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def plot(runs: pd.DataFrame, out_path: Path) -> None:
    runs = runs.copy()
    runs["abs_terminal_inventory"] = runs["terminal_inventory"].abs()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for strategy, aversion, label, color in STRATEGIES:
        sub = runs[(runs["mm_strategy"] == strategy)
                   & np.isclose(runs["inventory_aversion"], aversion)]

        pnl = sub.groupby("calibration_bias")["terminal_pnl"].agg(["mean", "sem"])
        ax1.plot(pnl.index, pnl["mean"], marker="o", color=color, label=label)
        ax1.fill_between(pnl.index,
                         pnl["mean"] - 1.96 * pnl["sem"],
                         pnl["mean"] + 1.96 * pnl["sem"],
                         color=color, alpha=0.15)

        inv = sub.groupby("calibration_bias")["abs_terminal_inventory"].mean()
        ax2.plot(inv.index, inv, marker="o", color=color, label=label)

    ax1.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax1.set_xlabel("Calibration bias")
    ax1.set_ylabel("Mean terminal P&L")
    ax1.set_title("Terminal P&L vs calibration bias by quoting strategy\n"
                  f"informed fraction {INFORMED_FRACTION:.0%}, "
                  f"{NUM_SEEDS} seeds per point, 95% CI")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Calibration bias")
    ax2.set_ylabel("Mean |terminal inventory| (contracts)")
    ax2.set_yscale("log")
    ax2.set_title("Inventory carried into settlement (log scale)")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main() -> int:
    if not BINARY.exists():
        print(f"Binary not found: {BINARY}. Build the C++ simulator first.")
        return 1

    runs = run_sweep()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RESULTS_DIR / "inventory_aware_runs.csv", index=False)

    table = (runs.assign(
        strategy=lambda d: d["mm_strategy"] + "_k" + d["inventory_aversion"].astype(str))
        .groupby(["strategy", "calibration_bias"])["terminal_pnl"]
        .mean().unstack("calibration_bias"))
    print("\nMean terminal P&L by strategy and bias:")
    print(table.round(1).to_string())

    plot(runs, RESULTS_DIR / "inventory_aware_vs_fixed.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
