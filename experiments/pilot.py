"""Pilot run to size N for Tasks A6 and B.

Runs the default MM (spread 0.04, T = 2000, pi = 0.10 for A6; the extreme and
smallest-epsilon cells of the B design) with 50 seeds per cell on both
processes, and reports the per-run standard deviation of P&L per contract
(settled and marked-at-p_T), the smallest effects of interest, and the N
that puts SE(mean P&L per contract) below 5% of each effect.

Output: results/pilot.md
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY = REPO_ROOT / "eventedge"
DATA_DIR = REPO_ROOT / "data" / "pilot"
RESULTS_DIR = REPO_ROOT / "results"
SEEDS = 50
STEPS = 2000

CELLS = [(proc, p0, eps, pi)
         for proc in ("additive", "martingale")
         for p0 in (0.2, 0.5, 0.8)
         for eps in (-0.10, -0.02, 0.0, 0.02, 0.10)
         for pi in (0.0, 0.1, 0.4)]


def run_one(args):
    proc, p0, eps, pi, seed = args
    prefix = DATA_DIR / f"{proc}_p{p0:.1f}_e{eps:+.2f}_i{pi:.1f}_s{seed}"
    summary = Path(f"{prefix}_summary.csv")
    if not summary.exists():
        subprocess.run([str(BINARY), "--seed", str(seed), "--steps", str(STEPS),
                        "--prob-process", proc, "--p0", f"{p0:.4f}", "--bias", f"{eps:.4f}",
                        "--informed-fraction", f"{pi:.4f}", "--log-detail", "0",
                        "--out-prefix", str(prefix)], check=True, capture_output=True)
    s = pd.read_csv(summary).iloc[0]
    contracts = max(int(s["fill_count"]), 1)
    return {"process": proc, "p0": p0, "eps": eps, "pi": pi, "seed": seed,
            "contracts": contracts,
            "pnl_pc": s["terminal_pnl"] / contracts,
            "marked_pc": s["terminal_pnl_marked"] / contracts}


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    grid = [c + (s,) for c in CELLS for s in range(SEEDS)]
    with ProcessPoolExecutor(max_workers=max(2, os.cpu_count() or 2)) as pool:
        runs = pd.DataFrame(list(pool.map(run_one, grid, chunksize=8)))
    g = runs.groupby(["process", "p0", "eps", "pi"]).agg(
        mean=("pnl_pc", "mean"), sd=("pnl_pc", "std"), marked_mean=("marked_pc", "mean"),
        marked_sd=("marked_pc", "std"), contracts=("contracts", "mean")).reset_index()
    md = [f"## Pilot ({SEEDS} seeds per cell, T = {STEPS}, spread 0.04)\n",
          "Per-run SD of P&L per contract, settled vs marked at p_T:\n",
          "| process | p₀ | ε | π | mean settled | SD settled | mean marked | SD marked | contracts |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        md.append(f"| {r.process} | {r.p0} | {r.eps:+.2f} | {r.pi} | {r['mean']:+.4f} | {r.sd:.4f} | "
                  f"{r.marked_mean:+.4f} | {r.marked_sd:.4f} | {r.contracts:.0f} |")

    md.append("\nEffects of interest (per contract, from cell means) and the N per cell that puts "
              "SE(mean) ≤ 5% of the effect, using the pooled per-run SD:\n")
    md.append("| process | effect | size (settled) | size (marked) | SD settled | N settled | SD marked | N marked |")
    md.append("|---|---|---|---|---|---|---|---|")
    rows = []
    for proc in ("additive", "martingale"):
        s = g[g.process == proc]
        def cell(p0, eps, pi, col="mean"):
            return s[(s.p0 == p0) & (np.isclose(s.eps, eps)) & (s.pi == pi)][col].iloc[0]
        effects = {
            "ε = ±0.02 vs 0, π = 0.1, p₀ = 0.5": (
                0.5 * (cell(0.5, 0.02, 0.1) + cell(0.5, -0.02, 0.1)) - cell(0.5, 0, 0.1),
                0.5 * (cell(0.5, 0.02, 0.1, "marked_mean") + cell(0.5, -0.02, 0.1, "marked_mean")) - cell(0.5, 0, 0.1, "marked_mean")),
            "ε = ±0.10 vs 0, π = 0.1, p₀ = 0.5": (
                0.5 * (cell(0.5, 0.10, 0.1) + cell(0.5, -0.10, 0.1)) - cell(0.5, 0, 0.1),
                0.5 * (cell(0.5, 0.10, 0.1, "marked_mean") + cell(0.5, -0.10, 0.1, "marked_mean")) - cell(0.5, 0, 0.1, "marked_mean")),
            "π = 0.4 vs 0, ε = 0, p₀ = 0.5": (
                cell(0.5, 0, 0.4) - cell(0.5, 0, 0.0),
                cell(0.5, 0, 0.4, "marked_mean") - cell(0.5, 0, 0.0, "marked_mean")),
            "ε·π interaction: [P(0.10,0.4)−P(0,0.4)]−[P(0.10,0.1)−P(0,0.1)]": (
                (cell(0.5, 0.10, 0.4) - cell(0.5, 0, 0.4)) - (cell(0.5, 0.10, 0.1) - cell(0.5, 0, 0.1)),
                (cell(0.5, 0.10, 0.4, "marked_mean") - cell(0.5, 0, 0.4, "marked_mean")) - (cell(0.5, 0.10, 0.1, "marked_mean") - cell(0.5, 0, 0.1, "marked_mean"))),
        }
        sd_s = np.sqrt((s["sd"] ** 2).mean())
        sd_m = np.sqrt((s["marked_sd"] ** 2).mean())
        for name, (es, em) in effects.items():
            n_s = int(np.ceil((sd_s / (0.05 * abs(es))) ** 2)) if es != 0 else float("inf")
            n_m = int(np.ceil((sd_m / (0.05 * abs(em))) ** 2)) if em != 0 else float("inf")
            md.append(f"| {proc} | {name} | {es:+.4f} | {em:+.4f} | {sd_s:.4f} | {n_s:,} | {sd_m:.4f} | {n_m:,} |")
            rows.append((proc, name, es, em, n_s, n_m))
    # A6 effect: process difference at eps 0, pi 0.1, per p0
    md.append("\nA6 candidate effect (process difference at ε = 0, π = 0.1), settled / marked:\n")
    for p0 in (0.2, 0.5, 0.8):
        a = g[(g.process == "additive") & (g.p0 == p0) & (g.eps == 0) & (g.pi == 0.1)].iloc[0]
        m = g[(g.process == "martingale") & (g.p0 == p0) & (g.eps == 0) & (g.pi == 0.1)].iloc[0]
        md.append(f"- p₀ = {p0}: settled {a['mean'] - m['mean']:+.4f} (SDs {a.sd:.3f}/{m.sd:.3f}), "
                  f"marked {a.marked_mean - m.marked_mean:+.4f} (SDs {a.marked_sd:.3f}/{m.marked_sd:.3f})")
    (RESULTS_DIR / "pilot.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
