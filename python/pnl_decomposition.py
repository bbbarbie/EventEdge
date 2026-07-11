"""Exact P&L decomposition: spread capture vs adverse selection vs settlement risk.

Terminal P&L is an exact sum over fills (with zero fees):

    terminal_pnl = sum_i dI_i * (Y - p_i)

where dI_i is the MM's signed inventory change and p_i the fill price.
Each fill term splits exactly into three components:

    dI * (mid_t - p)       spread capture     (always +half_spread per contract)
    dI * (ptrue_t - mid_t) adverse selection  (quote mispricing vs true prob,
                                               from calibration bias and from
                                               informed traders' selective timing)
    dI * (Y - ptrue_t)     settlement noise   (martingale term: zero mean,
                                               pure risk from holding to
                                               settlement)

The identity is checked numerically for every run: components must sum to the
C++ simulator's independently computed terminal_pnl.

Outputs:
    results/pnl_decomposition_vs_bias.png   (components vs calibration bias +
                                             adverse selection by trader type)
    results/pnl_decomposition.csv
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
DATA_DIR = REPO_ROOT / "data" / "decomposition_sweep"
RESULTS_DIR = REPO_ROOT / "results"

BIAS_GRID = np.round(np.arange(-0.10, 0.1001, 0.02), 3)
INFORMED_FRACTION = 0.3
NUM_SEEDS = 30
NUM_STEPS = 2000
MAX_WORKERS = 8
IDENTITY_TOL = 1e-6

TYPE_COLORS = {"INFORMED": "#c0392b", "VALUE": "#e67e22", "NOISE": "#2980b9"}


def run_one(args: tuple[float, int]) -> Path:
    bias, seed = args
    prefix = DATA_DIR / f"b{bias:+.2f}_s{seed}"
    if not Path(f"{prefix}_summary.csv").exists():
        subprocess.run(
            [str(BINARY),
             "--seed", str(seed),
             "--steps", str(NUM_STEPS),
             "--bias", f"{bias:.4f}",
             "--informed-fraction", f"{INFORMED_FRACTION:.4f}",
             "--log-detail", "1",
             "--out-prefix", str(prefix)],
            check=True, capture_output=True,
        )
    return prefix


def decompose_run(prefix: Path) -> dict:
    fills = pd.read_csv(f"{prefix}_fills.csv")
    summary = pd.read_csv(f"{prefix}_summary.csv").iloc[0]
    outcome = float(summary["event_outcome"])

    result = {
        "calibration_bias": float(summary["calibration_bias"]),
        "seed": int(summary["seed"]),
        "terminal_pnl": float(summary["terminal_pnl"]),
        "fill_count": int(summary["fill_count"]),
    }

    if fills.empty:
        result.update(spread_capture=0.0, adverse_selection=0.0,
                      settlement_noise=0.0, identity_error=0.0,
                      **{f"as_{t}": 0.0 for t in TYPE_COLORS})
        return result

    d_inv = fills["mm_inventory_change"].astype(float)
    mid = (fills["bid"] + fills["ask"]) / 2.0
    p_true = fills["latent_probability"]
    price = fills["fill_price"]

    spread_term = d_inv * (mid - price)
    adverse_term = d_inv * (p_true - mid)
    settle_term = d_inv * (outcome - p_true)

    result["spread_capture"] = spread_term.sum()
    result["adverse_selection"] = adverse_term.sum()
    result["settlement_noise"] = settle_term.sum()
    result["identity_error"] = abs(
        result["spread_capture"] + result["adverse_selection"]
        + result["settlement_noise"] - result["terminal_pnl"])

    for trader_type in TYPE_COLORS:
        mask = fills["trader_type"] == trader_type
        result[f"as_{trader_type}"] = adverse_term[mask].sum()
    return result


def plot(decomp: pd.DataFrame, out_path: Path) -> None:
    stats = decomp.groupby("calibration_bias").mean(numeric_only=True)
    x = stats.index.to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: components vs bias, stacked by sign so mixed-sign components
    # (settlement noise flips sign with bias) never overlap visually.
    components = [
        ("spread_capture", "Spread capture", "#27ae60"),
        ("adverse_selection", "Adverse selection", "#c0392b"),
        ("settlement_noise", "Settlement noise", "#95a5a6"),
    ]
    pos_base = np.zeros(len(x))
    neg_base = np.zeros(len(x))
    for column, label, color in components:
        values = stats[column].to_numpy()
        bottom = np.where(values >= 0, pos_base, neg_base)
        ax1.bar(x, values, width=0.012, bottom=bottom, color=color, label=label)
        pos_base += np.maximum(values, 0.0)
        neg_base += np.minimum(values, 0.0)
    ax1.plot(x, stats["terminal_pnl"], marker="o", color="black",
             linewidth=2, label="Terminal P&L (sum)")
    ax1.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax1.set_xlabel("Calibration bias")
    ax1.set_ylabel("Mean P&L component")
    ax1.set_title("P&L decomposition vs calibration bias\n"
                  f"informed fraction {INFORMED_FRACTION:.0%}, "
                  f"{NUM_SEEDS} seeds per bias")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2: adverse selection split by trader type.
    for trader_type, color in TYPE_COLORS.items():
        ax2.plot(x, stats[f"as_{trader_type}"], marker="o", color=color,
                 linewidth=2, label=trader_type)
    ax2.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax2.set_xlabel("Calibration bias")
    ax2.set_ylabel("Mean adverse-selection P&L")
    ax2.set_title("Adverse-selection component by trader type")
    ax2.legend(title="Trader type")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main() -> int:
    if not BINARY.exists():
        print(f"Binary not found: {BINARY}. Build the C++ simulator first.")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(BIAS_GRID, range(NUM_SEEDS)))
    print(f"Running {len(grid)} simulations...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        prefixes = list(pool.map(run_one, grid, chunksize=8))

    decomp = pd.DataFrame([decompose_run(p) for p in prefixes])

    max_err = decomp["identity_error"].max()
    print(f"Identity check: max |components - terminal_pnl| = {max_err:.2e} "
          f"across {len(decomp)} runs")
    if max_err > IDENTITY_TOL:
        bad = decomp[decomp["identity_error"] > IDENTITY_TOL]
        print(f"IDENTITY VIOLATED in {len(bad)} runs -- decomposition is wrong.")
        return 1

    stats = decomp.groupby("calibration_bias").mean(numeric_only=True)
    cols = ["spread_capture", "adverse_selection", "settlement_noise",
            "terminal_pnl", "as_INFORMED", "as_VALUE", "as_NOISE"]
    print("\nMean components by calibration bias:")
    print(stats[cols].round(2).to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    decomp.to_csv(RESULTS_DIR / "pnl_decomposition.csv", index=False)
    plot(decomp, RESULTS_DIR / "pnl_decomposition_vs_bias.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
