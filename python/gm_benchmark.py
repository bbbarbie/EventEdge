"""Glosten-Milgrom Bayesian quoting vs the naive calibrated quoter.

Prediction-market making is the textbook GM (1985) setting: a binary
value, informed and noise traders, a risk-neutral competitive MM. The GM
quoter (`--mm-strategy gm`) keeps a posterior over the latent probability
(diffused by the process it assumes, updated by its public signal and by
every order it sees, including the absence of one), and posts the
regret-free prices ask = E[p | buy], bid = E[p | sell] under the trader
population the simulator actually implements. Its spread is endogenous:
it widens with the informed fraction and with posterior uncertainty.

Three comparisons:

  A. Well-specified: GM (knows the population and the process) vs the
     fixed 0.04-spread quoter vs inventory-aware skew, over calibration
     bias x informed fraction. Bias enters GM through its signal exactly
     as it enters the naive quoter, so this isolates the value of Bayesian
     quoting given the same information.
  B. Misspecified population: GM told the informed fraction is 10% when it
     is really 30% / 50%.
  C. Misspecified process: GM that assumes no jumps (Gaussian steps only)
     facing the jumpy additive process.

P&L is reported with inventory marked at the terminal latent probability
(E[settled P&L | path], see run_experiments.py), 100 seeds per point.

Outputs:
  results/gm_benchmark.csv
  results/gm_benchmark.png
"""

from __future__ import annotations

import argparse
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
DATA_DIR = REPO_ROOT / "data" / "gm_benchmark"
RESULTS_DIR = REPO_ROOT / "results"

BIAS_GRID = [-0.10, -0.05, 0.0, 0.05, 0.10]
INFORMED_GRID = [0.1, 0.3, 0.5]
NUM_SEEDS = 100
NUM_STEPS = 2000
MAX_WORKERS = max(2, os.cpu_count() or 2)

# name -> extra flags. Fixed categorical order and colors.
QUOTERS = {
    "fixed 0.04":       ["--mm-strategy", "fixed"],
    "inventory k=.002": ["--mm-strategy", "inventory", "--inventory-aversion", "0.002"],
    "GM":               ["--mm-strategy", "gm"],
    "GM, thinks 10% informed": ["--mm-strategy", "gm", "--gm-informed", "0.1"],
    "GM, no-jump model": ["--mm-strategy", "gm", "--gm-jump-prob", "0"],
}
COLORS = {
    "fixed 0.04": "#c0392b", "inventory k=.002": "#e67e22", "GM": "#27ae60",
    "GM, thinks 10% informed": "#2980b9", "GM, no-jump model": "#8e44ad",
}


def run_one(args: tuple[str, float, float, int, str | None]) -> Path:
    quoter, bias, informed, seed, prob_path = args
    tag = "synthetic" if prob_path is None else Path(prob_path).stem
    slug = quoter.replace(" ", "").replace(",", "").replace("%", "").replace("=", "").replace(".", "")
    prefix = DATA_DIR / f"{tag}_{slug}_b{bias:+.2f}_i{informed:.1f}_s{seed}"
    summary = Path(f"{prefix}_summary.csv")
    if not summary.exists():
        cmd = [str(BINARY), "--seed", str(seed), "--bias", f"{bias:.4f}",
               "--informed-fraction", f"{informed:.4f}", "--log-detail", "1",
               "--out-prefix", str(prefix)] + QUOTERS[quoter]
        cmd += ["--prob-path", prob_path] if prob_path else ["--steps", str(NUM_STEPS)]
        subprocess.run(cmd, check=True, capture_output=True)
    return summary


def realized_spread(prefix: Path) -> float:
    steps = pd.read_csv(f"{prefix}_steps.csv", usecols=["bid", "ask"])
    return float((steps["ask"] - steps["bid"]).mean())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prob-path", help="benchmark on a real Kalshi path instead")
    parser.add_argument("--seeds", type=int, default=NUM_SEEDS)
    args = parser.parse_args(argv)
    if not BINARY.exists():
        print(f"Binary not found: {BINARY}")
        return 1
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = [(q, b, i, s, args.prob_path) for q, b, i, s in
            itertools.product(QUOTERS, BIAS_GRID, INFORMED_GRID, range(args.seeds))]
    print(f"Running {len(grid)} simulations...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        paths = list(pool.map(run_one, grid, chunksize=4))
    frames = []
    for (quoter, *_), path in zip(grid, paths):
        frame = pd.read_csv(path)
        frame["quoter"] = quoter
        frame["realized_spread"] = realized_spread(Path(str(path)[: -len("_summary.csv")]))
        frames.append(frame)
    runs = pd.concat(frames, ignore_index=True)
    runs["calibration_bias"] = runs["calibration_bias"].round(3)

    # Inventory marked at p_T: E[settled P&L | path] exactly, without the
    # one Bernoulli draw per run (see run_experiments.py).
    table = runs.groupby(["quoter", "informed_fraction", "calibration_bias"]).agg(
        pnl=("terminal_pnl_marked", "mean"), pnl_sem=("terminal_pnl_marked", "sem"),
        settled_pnl=("terminal_pnl", "mean"),
        fills=("fill_count", "mean"), spread=("realized_spread", "mean"),
        inventory=("terminal_inventory", lambda s: s.abs().mean())).reset_index()
    tag = "" if args.prob_path is None else "_" + Path(args.prob_path).stem
    table.to_csv(RESULTS_DIR / f"gm_benchmark{tag}.csv", index=False)
    print(table.pivot_table(index=["quoter", "informed_fraction"], columns="calibration_bias",
                            values="pnl").round(1).to_string())
    print("\nRealized spread (mean ask-bid over the run):")
    print(table.pivot_table(index="quoter", columns="informed_fraction",
                            values="spread").round(3).to_string())

    fig, axes = plt.subplots(1, len(INFORMED_GRID), figsize=(5 * len(INFORMED_GRID), 4.8),
                             sharey=True)
    for ax, informed in zip(axes, INFORMED_GRID):
        for quoter in QUOTERS:
            sub = table[(table["quoter"] == quoter)
                        & np.isclose(table["informed_fraction"], informed)]
            ax.errorbar(sub["calibration_bias"], sub["pnl"], yerr=1.96 * sub["pnl_sem"],
                        marker="o", markersize=4, capsize=3, linewidth=2,
                        color=COLORS[quoter], label=quoter)
        ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
        ax.set_title(f"informed {informed:.0%}")
        ax.set_xlabel("Calibration bias")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean terminal P&L, inventory marked at p_T (95% CI)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Glosten–Milgrom vs naive quoting, {args.seeds} seeds per point"
                 + ("" if args.prob_path is None else f", path {Path(args.prob_path).stem}"),
                 y=1.02)
    fig.tight_layout()
    out = RESULTS_DIR / f"gm_benchmark{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
