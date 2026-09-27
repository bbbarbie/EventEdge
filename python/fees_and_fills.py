"""Did you include fees? And does a fill mean what you think it means?

Two frictions the frictionless model leaves out, both Kalshi-specific:

1. Exchange fees. Kalshi charges fee_rate * C * P * (1 - P), rounded up to
   the cent per fill: 7% for takers, 1.75% for makers on the series that
   charge makers at all (0 elsewhere). At p = 0.5 the taker fee is 1.75c
   per contract, the same order as the 2c half-spread the MM earns.
   (Rates are from Kalshi's published fee schedule as of 2025; the
   `--fee-rate` flag takes whatever the current schedule says.)

2. Queue position. A real MM is not alone at its price: liquidity ahead of
   it in the queue takes the first contracts of every order. Unit-size
   noise orders are mostly absorbed; a larger informed sweep gets through.
   So conditional on a fill, the counterparty is more likely informed
   than the arrival mix says: the fill itself is adverse.

Design: bias 0 (a perfectly calibrated MM), informed fraction 0-50%,
fee in {none, maker exact, maker rounded, taker rounded} x fill model in
{alone, queued}, where
"queued" is a mean of 1 contract ahead of the MM and informed orders of
3 contracts. 30 seeds per cell. Reports pre-fee and post-fee P&L per
contract traded and the informed share of fills.

Outputs:
  results/fees_and_fills.csv
  results/fees_and_fills.png
"""

from __future__ import annotations

import itertools
import os
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
BINARY = Path(os.environ.get("EVENTEDGE_BINARY", REPO_ROOT / "eventedge"))
DATA_DIR = REPO_ROOT / "data" / "fees_and_fills"
RESULTS_DIR = REPO_ROOT / "results"

INFORMED_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# (rate, round up to the cent per fill). Kalshi rounds each fill's fee up
# to the next cent, which at unit size turns a 0.4c maker fee into 1c.
FEE_RATES = [(0.0, 1), (0.0175, 0), (0.0175, 1), (0.07, 1)]
FILL_MODELS = {"alone": (0.0, 1), "queued": (1.0, 3)}  # (queue_ahead, informed_size)
NUM_SEEDS = 30
NUM_STEPS = 2000
BIAS = 0.0
MAX_WORKERS = max(2, os.cpu_count() or 2)

FEE_COLORS = {(0.0, 1): "#7f8c8d", (0.0175, 0): "#5dade2", (0.0175, 1): "#2980b9",
              (0.07, 1): "#c0392b"}
FEE_LABELS = {(0.0, 1): "no fee", (0.0175, 0): "maker 1.75%·p(1−p), exact",
              (0.0175, 1): "maker 1.75%·p(1−p), rounded up per fill",
              (0.07, 1): "taker 7%·p(1−p), rounded up per fill"}


def run_one(args: tuple[str, tuple[float, int], float, int]) -> Path:
    model, (fee, rounding), informed, seed = args
    queue, size = FILL_MODELS[model]
    prefix = DATA_DIR / f"{model}_f{fee:.4f}r{rounding}_i{informed:.1f}_s{seed}"
    summary = Path(f"{prefix}_summary.csv")
    if not summary.exists():
        subprocess.run(
            [str(BINARY), "--seed", str(seed), "--steps", str(NUM_STEPS),
             "--bias", f"{BIAS:.4f}", "--informed-fraction", f"{informed:.4f}",
             "--fee-rate", f"{fee:.4f}", "--fee-round-cents", str(rounding),
             "--queue-ahead", f"{queue:.3f}",
             "--informed-size", str(size), "--log-detail", "0",
             "--out-prefix", str(prefix)],
            check=True, capture_output=True)
    return summary


def main() -> int:
    if not BINARY.exists():
        print(f"Binary not found: {BINARY}")
        return 1
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(FILL_MODELS, FEE_RATES, INFORMED_GRID, range(NUM_SEEDS)))
    print(f"Running {len(grid)} simulations...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        paths = list(pool.map(run_one, grid, chunksize=8))
    runs = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    runs["fill_model"] = [g[0] for g in grid]
    runs["fee_round"] = [g[1][1] for g in grid]
    runs["pre_fee_pnl"] = runs["terminal_pnl"] + runs["fees_paid"]
    runs["informed_share"] = runs["informed_fill_count"] / runs["fill_count"].clip(lower=1)
    runs["contracts"] = runs["fill_count"]  # unit orders except informed sweeps

    keys = ["fill_model", "fee_rate", "fee_round", "informed_fraction"]
    table = runs.groupby(keys).agg(
        pnl=("terminal_pnl", "mean"), pnl_sem=("terminal_pnl", "sem"),
        pre_fee_pnl=("pre_fee_pnl", "mean"), fees=("fees_paid", "mean"),
        fills=("fill_count", "mean"), informed_share=("informed_share", "mean"),
        absorbed=("absorbed_count", "mean")).reset_index()
    table.to_csv(RESULTS_DIR / "fees_and_fills.csv", index=False)
    print(table.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model in zip(axes, FILL_MODELS):
        for fee in FEE_RATES:
            sub = table[(table["fill_model"] == model) & np.isclose(table["fee_rate"], fee[0])
                        & (table["fee_round"] == fee[1])]
            ax.errorbar(sub["informed_fraction"], sub["pnl"], yerr=1.96 * sub["pnl_sem"],
                        marker="o", markersize=5, capsize=3, linewidth=2,
                        color=FEE_COLORS[fee], label=FEE_LABELS[fee])
        ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
        queue, size = FILL_MODELS[model]
        ax.set_title(f"fill model: {model}"
                     + (f" (mean {queue:.0f} ahead in queue, informed size {size})"
                        if queue > 0 else " (MM alone at its price)"))
        ax.set_xlabel("Informed fraction of arrivals")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean terminal P&L after fees (95% CI)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Calibrated MM (bias 0), spread 0.04, {NUM_SEEDS} seeds per point", y=1.02)
    fig.tight_layout()
    out = RESULTS_DIR / "fees_and_fills.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    # Break-even informed fraction per (model, fee): first grid point where mean P&L < 0.
    print("\nBreak-even informed fraction (first grid point with mean P&L < 0):")
    for model in FILL_MODELS:
        for fee in FEE_RATES:
            sub = table[(table["fill_model"] == model) & np.isclose(table["fee_rate"], fee[0])
                        & (table["fee_round"] == fee[1])]
            neg = sub[sub["pnl"] < 0]["informed_fraction"]
            print(f"  {model:7s} {FEE_LABELS[fee]:42s}: {neg.min() if len(neg) else 'never'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
