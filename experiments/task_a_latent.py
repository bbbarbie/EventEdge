"""Task A (A1-A5): martingale validation of the two latent processes.

Runs build/latent_paths (which uses EventProbabilityProcess itself at scale 1
and a bit-for-bit verified mirror for sigma scaling) for both processes over
the p0 grid, at the default sigma / horizon and at sigma x {0.5, 2},
horizon x {0.5, 2}. N = 200,000 paths per configuration, seeds 1..N.

Outputs (results/):
  taskA_bias_table.csv        A1/A2/A4 per (process, p0, config)
  taskA_conditional.csv       A3 per (process, config, p_mid bin), pooled over p0
  taskA_bias_vs_p0.png        A1 + A5 figure
  taskA_tables.md             markdown tables to paste into results.md
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
HARNESS = REPO_ROOT / "build" / "latent_paths"
DATA_DIR = REPO_ROOT / "data" / "task_a"
RESULTS_DIR = REPO_ROOT / "results"

P0_GRID = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]
PROCESSES = ["additive", "martingale"]
DEFAULT_STEPS = 2000
CONFIGS = {  # name -> (sigma_scale, steps)
    "default": (1.0, DEFAULT_STEPS),
    "sigma x0.5": (0.5, DEFAULT_STEPS),
    "sigma x2": (2.0, DEFAULT_STEPS),
    "T x0.5": (1.0, DEFAULT_STEPS // 2),
    "T x2": (1.0, DEFAULT_STEPS * 2),
}
N_PATHS = 200_000
BINS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
COLORS = {"additive": "#c0392b", "martingale": "#2980b9"}


def run_one(args):
    process, p0, config = args
    scale, steps = CONFIGS[config]
    out = DATA_DIR / f"{process}_p{p0:.2f}_{config.replace(' ', '')}.csv"
    if not out.exists():
        subprocess.run([str(HARNESS), "--process", process, "--p0", f"{p0:.4f}",
                        "--paths", str(N_PATHS), "--steps", str(steps),
                        "--sigma-scale", f"{scale:.4f}", "--seed-base", "1",
                        "--out", str(out)], check=True, capture_output=True)
    d = pd.read_csv(out)
    n = len(d)
    row = {
        "process": process, "p0": p0, "config": config, "sigma_scale": scale,
        "steps": steps, "N": n,
        "bias_pT_pp": 100 * (d["p_T"].mean() - p0),
        "bias_pT_se_pp": 100 * d["p_T"].std(ddof=1) / np.sqrt(n),
        "bias_settle_pp": 100 * (d["settled"].mean() - p0),
        "bias_settle_se_pp": 100 * np.sqrt(p0 * (1 - p0) / n),
        "frac_paths_clipped": (d["clip_count"] > 0).mean(),
        "mean_clips_per_path": d["clip_count"].mean(),
        "steps_left_unit_interval": int(d["left_unit_interval"].sum()),
        "frac_paths_left_unit": (d["left_unit_interval"] > 0).mean(),
    }
    # A3 material: conditional means by p_mid bin (kept per file, pooled later)
    d["bin"] = pd.cut(d["p_mid"], BINS, right=False, include_lowest=True)
    cond = d.groupby("bin", observed=True).agg(
        n=("p_T", "size"), sum_pT=("p_T", "sum"), sum_pmid=("p_mid", "sum"),
        sumsq_dev=("p_T", lambda s: 0.0))  # placeholder, filled below
    # sum of squared (p_T - p_mid) for SE of the conditional difference
    d["dev"] = d["p_T"] - d["p_mid"]
    cond["sum_dev"] = d.groupby("bin", observed=True)["dev"].sum()
    cond["sumsq_dev"] = d.groupby("bin", observed=True)["dev"].apply(lambda s: (s ** 2).sum())
    cond = cond.reset_index()
    cond["process"], cond["config"], cond["p0"] = process, config, p0
    return row, cond


def main() -> int:
    global N_PATHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=N_PATHS)
    args = parser.parse_args()
    N_PATHS = args.paths
    if not HARNESS.exists():
        print(f"build the harness first: {HARNESS}")
        return 1
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(PROCESSES, P0_GRID, CONFIGS))
    print(f"{len(grid)} configurations x {N_PATHS} paths")
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 2))) as pool:
        results = list(pool.map(run_one, grid))
    table = pd.DataFrame([r for r, _ in results])
    table.to_csv(RESULTS_DIR / "taskA_bias_table.csv", index=False)

    # A3: pool bins across p0 within (process, config)
    cond = pd.concat([c for _, c in results], ignore_index=True)
    pooled = cond.groupby(["process", "config", "bin"], observed=True).agg(
        n=("n", "sum"), sum_dev=("sum_dev", "sum"), sumsq_dev=("sumsq_dev", "sum"),
        sum_pmid=("sum_pmid", "sum")).reset_index()
    pooled = pooled[pooled["n"] > 0]
    pooled["mean_pmid"] = pooled["sum_pmid"] / pooled["n"]
    pooled["cond_bias_pp"] = 100 * pooled["sum_dev"] / pooled["n"]
    var = pooled["sumsq_dev"] / pooled["n"] - (pooled["sum_dev"] / pooled["n"]) ** 2
    pooled["cond_bias_se_pp"] = 100 * np.sqrt(var.clip(lower=0) / pooled["n"])
    pooled.to_csv(RESULTS_DIR / "taskA_conditional.csv", index=False)

    # ---- markdown tables
    md = []
    base = table[table["config"] == "default"]
    md.append("### A1. Bias at the default σ and horizon (pp)\n")
    md.append(f"N = {N_PATHS:,} paths per cell, T = {DEFAULT_STEPS} steps, σ = repo default "
              "(additive: 0.01/step + 2%×0.05 jumps, clamp [0.01, 0.99]; "
              "martingale: 0.042·p(1−p)/step + 2%×0.21·p(1−p) jumps, clamp [1e-6, 1−1e-6]).\n")
    md.append("| p₀ | clipped: E[p_T]−p₀ | clipped: settle freq−p₀ | p(1−p): E[p_T]−p₀ | p(1−p): settle freq−p₀ |")
    md.append("|---|---|---|---|---|")
    for p0 in P0_GRID:
        a = base[(base.process == "additive") & (base.p0 == p0)].iloc[0]
        m = base[(base.process == "martingale") & (base.p0 == p0)].iloc[0]
        md.append(f"| {p0:.2f} | {a.bias_pT_pp:+.2f} ± {a.bias_pT_se_pp:.2f} | "
                  f"{a.bias_settle_pp:+.2f} ± {a.bias_settle_se_pp:.2f} | "
                  f"{m.bias_pT_pp:+.2f} ± {m.bias_pT_se_pp:.2f} | "
                  f"{m.bias_settle_pp:+.2f} ± {m.bias_settle_se_pp:.2f} |")
    md.append("\n### A2. Max |bias| over the p₀ grid (pp)\n")
    md.append("| process | max |E[p_T]−p₀| | at p₀ | max |settle−p₀| | at p₀ |")
    md.append("|---|---|---|---|---|")
    for proc in PROCESSES:
        b = base[base.process == proc]
        i = b["bias_pT_pp"].abs().idxmax()
        j = b["bias_settle_pp"].abs().idxmax()
        md.append(f"| {proc} | {abs(b.loc[i, 'bias_pT_pp']):.2f} ± {b.loc[i, 'bias_pT_se_pp']:.2f} | "
                  f"{b.loc[i, 'p0']:.2f} | {abs(b.loc[j, 'bias_settle_pp']):.2f} ± "
                  f"{b.loc[j, 'bias_settle_se_pp']:.2f} | {b.loc[j, 'p0']:.2f} |")
    md.append("\n### A3. Conditional martingale check, E[p_T | p_{T/2}] − p_{T/2} (pp), pooled over the p₀ grid\n")
    md.append("| p_{T/2} bin | n (clipped) | clipped | n (p(1−p)) | p(1−p) |")
    md.append("|---|---|---|---|---|")
    pa = pooled[(pooled.process == "additive") & (pooled.config == "default")].set_index("bin")
    pm = pooled[(pooled.process == "martingale") & (pooled.config == "default")].set_index("bin")
    for b in pa.index.union(pm.index):
        ra = pa.loc[b] if b in pa.index else None
        rm = pm.loc[b] if b in pm.index else None
        fa = f"{ra.cond_bias_pp:+.2f} ± {ra.cond_bias_se_pp:.2f}" if ra is not None else "—"
        fm = f"{rm.cond_bias_pp:+.2f} ± {rm.cond_bias_se_pp:.2f}" if rm is not None else "—"
        md.append(f"| {b} | {int(ra.n) if ra is not None else 0:,} | {fa} | "
                  f"{int(rm.n) if rm is not None else 0:,} | {fm} |")
    md.append("\n### A4. Boundary accounting (default config)\n")
    md.append("| p₀ | clipped: paths clipped ≥1 | clipped: mean clips/path | p(1−p): Euler steps outside [0,1] | p(1−p): paths with any |")
    md.append("|---|---|---|---|---|")
    for p0 in P0_GRID:
        a = base[(base.process == "additive") & (base.p0 == p0)].iloc[0]
        m = base[(base.process == "martingale") & (base.p0 == p0)].iloc[0]
        md.append(f"| {p0:.2f} | {100*a.frac_paths_clipped:.1f}% | {a.mean_clips_per_path:.1f} | "
                  f"{m.steps_left_unit_interval} | {100*m.frac_paths_left_unit:.3f}% |")
    md.append("\n### A5. Sensitivity: max |E[p_T]−p₀| over the p₀ grid (pp)\n")
    md.append("| config | σ·√T relative | clipped max |bias| (at p₀) | p(1−p) max |bias| (at p₀) |")
    md.append("|---|---|---|---|")
    for cfg, (scale, steps) in CONFIGS.items():
        rel = scale * np.sqrt(steps / DEFAULT_STEPS)
        cells = []
        for proc in PROCESSES:
            b = table[(table.process == proc) & (table.config == cfg)]
            i = b["bias_pT_pp"].abs().idxmax()
            cells.append(f"{abs(b.loc[i, 'bias_pT_pp']):.2f} ± {b.loc[i, 'bias_pT_se_pp']:.2f} ({b.loc[i, 'p0']:.2f})")
        md.append(f"| {cfg} | {rel:.2f} | {cells[0]} | {cells[1]} |")
    (RESULTS_DIR / "taskA_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- figure: bias vs p0, default solid, sensitivity dashed/dotted
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    styles = {"default": ("-", 2.2, "o"), "sigma x0.5": (":", 1.4, None), "sigma x2": ("--", 1.4, None),
              "T x0.5": ((0, (1, 1)), 1.0, None), "T x2": ((0, (5, 2, 1, 2)), 1.0, None)}
    for ax, proc in zip(axes, PROCESSES):
        for cfg, (ls, lw, mk) in styles.items():
            b = table[(table.process == proc) & (table.config == cfg)].sort_values("p0")
            ax.errorbar(b["p0"], b["bias_pT_pp"], yerr=1.96 * b["bias_pT_se_pp"],
                        linestyle=ls, linewidth=lw, marker=mk, markersize=5, capsize=2,
                        color=COLORS[proc], label=cfg)
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")
        ax.set_title(f"{'clipped additive' if proc == 'additive' else 'p(1−p) martingale'} process")
        ax.set_xlabel("p₀")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="config")
    axes[0].set_ylabel("E[p_T] − p₀  (percentage points, 95% CI)")
    fig.suptitle(f"Task A: settlement bias of the latent process, N = {N_PATHS:,} paths per point", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "taskA_bias_vs_p0.png", dpi=150, bbox_inches="tight")
    print("saved figure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
