"""Task A6: P&L consequence of the clipped process.

Default MM (epsilon = 0, pi = 0.10, spread 0.04, T = 2000) on both latent
processes with the same seeds (trader-arrival and settlement streams shared;
the latent path differs by construction). Reports mean P&L per contract with
SE for each process, the paired-by-seed difference, and which decomposition
component absorbs it. Run at the repo default p0 = 0.6 and at the Task B
grid p0 in {0.2, 0.5, 0.8}.

Output: results/taskA6_tables.md, results/taskA6_cells.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cache_or_compute, mean_se  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY = REPO_ROOT / "eventedge"
DATA_DIR = REPO_ROOT / "data" / "task_a6"
RESULTS_DIR = REPO_ROOT / "results"
P0_GRID = [0.2, 0.5, 0.6, 0.8]
STEPS, PI, SPREAD = 2000, 0.10, 0.04
COMPONENTS = ["spread_capture", "adverse_selection", "settlement_drift", "settlement_draw"]


def run_one(args):
    proc, p0, seed = args
    prefix = DATA_DIR / f"{proc}_p{p0:.1f}_s{seed}"
    if not Path(f"{prefix}_stats.json").exists():
        subprocess.run([str(BINARY), "--seed", str(seed), "--steps", str(STEPS),
                        "--prob-process", proc, "--p0", f"{p0:.4f}", "--bias", "0",
                        "--informed-fraction", f"{PI:.4f}", "--spread", f"{SPREAD:.4f}",
                        "--log-detail", "1", "--out-prefix", str(prefix)],
                       check=True, capture_output=True)
    row = cache_or_compute(prefix)
    row.update(process=proc, p0=p0)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, required=True)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(("additive", "martingale"), P0_GRID, range(args.seeds)))
    with ProcessPoolExecutor(max_workers=max(2, os.cpu_count() or 2)) as pool:
        runs = pd.DataFrame(list(pool.map(run_one, grid, chunksize=8)))
    assert runs["identity_error"].max() < 1e-6, runs["identity_error"].max()
    for col in ["terminal_pnl", "terminal_pnl_marked"] + COMPONENTS:
        runs[f"{col}_pc"] = runs[col] / runs["contracts"].clip(lower=1)
    runs.to_csv(RESULTS_DIR / "taskA6_runs.csv", index=False)

    md = [f"### A6. Default MM on both processes (ε = 0, π = {PI}, spread {SPREAD}, T = {STEPS}, "
          f"N = {args.seeds} seeds per cell, same seeds)\n",
          "P&L per contract traded. Difference = additive − martingale, paired by seed "
          "(shared arrival and settlement streams).\n",
          "| p₀ | quantity | clipped additive | p(1−p) martingale | difference (paired SE) |",
          "|---|---|---|---|---|"]
    cells = []
    for p0 in P0_GRID:
        a = runs[(runs.process == "additive") & (runs.p0 == p0)].set_index("seed").sort_index()
        m = runs[(runs.process == "martingale") & (runs.p0 == p0)].set_index("seed").sort_index()
        for label, col in [("settled P&L / contract", "terminal_pnl_pc"),
                           ("marked P&L / contract", "terminal_pnl_marked_pc"),
                           ("spread capture / contract", "spread_capture_pc"),
                           ("adverse selection / contract", "adverse_selection_pc"),
                           ("settlement drift / contract", "settlement_drift_pc"),
                           ("settlement draw / contract", "settlement_draw_pc"),
                           ("contracts per run", "contracts"),
                           ("terminal inventory", "terminal_inventory"),
                           ("E[p_T]", "terminal_probability")]:
            ma, sa = mean_se(a[col]); mm, sm = mean_se(m[col])
            diff = a[col] - m[col]
            md_, sd_ = mean_se(diff)
            md.append(f"| {p0} | {label} | {ma:+.4f} ± {sa:.4f} | {mm:+.4f} ± {sm:.4f} | "
                      f"{md_:+.4f} ± {sd_:.4f} |")
            cells.append({"p0": p0, "quantity": label, "additive": ma, "additive_se": sa,
                          "martingale": mm, "martingale_se": sm, "diff": md_, "diff_se": sd_})
    pd.DataFrame(cells).to_csv(RESULTS_DIR / "taskA6_cells.csv", index=False)
    (RESULTS_DIR / "taskA6_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
