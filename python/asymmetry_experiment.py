"""Why is P&L asymmetric in the sign of calibration bias, and is it real?

In the headline sweeps (p0 = 0.6, clamped additive walk) a -0.10 bias costs
about -37 while +0.10 costs about -88. The exact decomposition already says
where to look: spread capture and adverse selection are symmetric to within
a few percent; the whole gap sits in the settlement term, whose standard
error over 30 seeds is ~70. So the asymmetry is either sampling noise or the
settlement *drift* term, which is non-zero only when the latent process is
not a martingale (the clamp reflects paths back toward the interior, so a
walk started above 0.5 drifts down on average, and a +bias MM's long
inventory is systematically on the wrong side of that drift).

Three candidate explanations, and a design that separates them:

  H1  bounded payoff: p0 = 0.6 sits nearer the upper clamp, so +bias and
      -bias quotes hit the bounds differently. Prediction: asymmetry
      persists under the martingale process at p0 = 0.6 (no drift, same
      bounds) and vanishes at p0 = 0.5 under both processes.
  H2  settlement drift of the clamped walk: asymmetry is carried by the
      drift component sum dI*(p_T - p_t), flips sign at p0 = 0.4, and
      vanishes under the martingale process at any p0.
  H3  sampling noise: asymmetry shrinks like 1/sqrt(seeds) and is
      insignificant once the draw component inv_T*(Y - p_T) is removed.

Design: p0 in {0.4, 0.5, 0.6} x process in {additive, martingale} x
bias in +-{0.02..0.10} x 600 seeds at 30% informed flow. For every cell the
P&L is decomposed exactly (spread / adverse / settlement drift / settlement
draw) and the asymmetry A(b) = X(+b) - X(-b) is reported per component with
a 95% CI.

Outputs:
  results/asymmetry.csv            A(b) per (process, p0, |bias|, component)
  results/asymmetry_by_design.png  P&L asymmetry vs |bias| per design
  results/asymmetry_components.png which component carries it (p0 = 0.6)
"""

from __future__ import annotations

import itertools
import json
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
DATA_DIR = REPO_ROOT / "data" / "asymmetry"
RESULTS_DIR = REPO_ROOT / "results"

P0_GRID = [0.4, 0.5, 0.6]
PROCESSES = ["additive", "martingale"]
ABS_BIAS = [0.02, 0.04, 0.06, 0.08, 0.10]
BIAS_GRID = sorted({0.0} | {b for a in ABS_BIAS for b in (a, -a)})
INFORMED_FRACTION = 0.3
NUM_SEEDS = 600
NUM_STEPS = 2000
MAX_WORKERS = max(2, os.cpu_count() or 2)
IDENTITY_TOL = 1e-6

COMPONENTS = ["terminal_pnl", "spread_capture", "adverse_selection",
              "settlement_drift", "settlement_draw"]
# Fixed categorical order for p0 (identity), never cycled.
P0_COLORS = {0.4: "#2980b9", 0.5: "#7f8c8d", 0.6: "#c0392b"}
COMPONENT_COLORS = {"spread_capture": "#27ae60", "adverse_selection": "#c0392b",
                    "settlement_drift": "#8e44ad", "settlement_draw": "#95a5a6"}


def run_one(args: tuple[str, float, float, int]) -> dict:
    """Run, decompose, and keep only the decomposed row (the per-fill logs
    of 40k runs would be gigabytes)."""
    process, p0, bias, seed = args
    prefix = DATA_DIR / f"{process}_p{p0:.1f}_b{bias:+.2f}_s{seed}"
    cached = Path(f"{prefix}_decomp.json")
    if cached.exists():
        return json.loads(cached.read_text())
    subprocess.run(
        [str(BINARY), "--seed", str(seed), "--steps", str(NUM_STEPS),
         "--p0", f"{p0:.4f}", "--prob-process", process,
         "--bias", f"{bias:.4f}",
         "--informed-fraction", f"{INFORMED_FRACTION:.4f}",
         "--log-detail", "1", "--out-prefix", str(prefix)],
        check=True, capture_output=True)
    row = decompose(prefix, process, p0)
    cached.write_text(json.dumps(row))
    for suffix in ("_fills.csv", "_steps.csv", "_summary.csv"):
        Path(f"{prefix}{suffix}").unlink(missing_ok=True)
    return row


def decompose(prefix: Path, process: str, p0: float) -> dict:
    """Exact split of terminal P&L, with the settlement term itself split
    into process drift (zero iff martingale) and the settlement draw."""
    summary = pd.read_csv(f"{prefix}_summary.csv").iloc[0]
    fills = pd.read_csv(f"{prefix}_fills.csv")
    outcome = float(summary["event_outcome"])
    p_terminal = float(summary["terminal_probability"])
    row = {"process": process, "p0": p0,
           "calibration_bias": round(float(summary["calibration_bias"]), 3),
           "seed": int(summary["seed"]),
           "terminal_pnl": float(summary["terminal_pnl"]),
           "terminal_inventory": int(summary["terminal_inventory"]),
           "fill_count": int(summary["fill_count"])}
    if fills.empty:
        row.update(spread_capture=0.0, adverse_selection=0.0,
                   settlement_drift=0.0, settlement_draw=0.0, identity_error=0.0)
        return row
    d_inv = fills["mm_inventory_change"].astype(float)
    mid = (fills["bid"] + fills["ask"]) / 2.0
    p_true = fills["latent_probability"]
    row["spread_capture"] = (d_inv * (mid - fills["fill_price"])).sum()
    row["adverse_selection"] = (d_inv * (p_true - mid)).sum()
    row["settlement_drift"] = (d_inv * (p_terminal - p_true)).sum()
    row["settlement_draw"] = float(summary["terminal_inventory"]) * (outcome - p_terminal)
    row["identity_error"] = abs(row["spread_capture"] + row["adverse_selection"]
                                + row["settlement_drift"] + row["settlement_draw"]
                                - row["terminal_pnl"])
    # inventory-neutral settlement draw: (Y - p_T) alone, to show the draw
    # is mean-zero regardless of which side the inventory ended on
    row["draw_sign"] = outcome - p_terminal
    return row


def asymmetry_table(runs: pd.DataFrame) -> pd.DataFrame:
    """A(b) = mean X(+b) - mean X(-b) with the two-sample standard error."""
    rows = []
    grouped = runs.groupby(["process", "p0", "calibration_bias"])
    for (process, p0), _ in runs.groupby(["process", "p0"]):
        for b in ABS_BIAS:
            plus = grouped.get_group((process, p0, b))
            minus = grouped.get_group((process, p0, -b))
            for comp in COMPONENTS:
                diff = plus[comp].mean() - minus[comp].mean()
                se = np.sqrt(plus[comp].sem() ** 2 + minus[comp].sem() ** 2)
                rows.append({"process": process, "p0": p0, "abs_bias": b,
                             "component": comp, "asymmetry": diff, "se": se,
                             "z": diff / se if se > 0 else np.nan,
                             "mean_plus": plus[comp].mean(),
                             "mean_minus": minus[comp].mean()})
    return pd.DataFrame(rows)


def plot_by_design(table: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, process in zip(axes, PROCESSES):
        for p0 in P0_GRID:
            sub = table[(table["process"] == process) & (table["p0"] == p0)
                        & (table["component"] == "terminal_pnl")]
            ax.errorbar(sub["abs_bias"], sub["asymmetry"], yerr=1.96 * sub["se"],
                        marker="o", markersize=5, capsize=3, linewidth=2,
                        color=P0_COLORS[p0], label=f"p0 = {p0:.1f}")
        ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
        ax.set_title(f"{process} process")
        ax.set_xlabel("|calibration bias|")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("P&L asymmetry  A(b) = P&L(+b) − P&L(−b)  (95% CI)")
    axes[0].legend(title="initial latent p")
    fig.suptitle(f"Is the bias asymmetry real?  {NUM_SEEDS} seeds per point, "
                 f"informed {INFORMED_FRACTION:.0%}", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


def plot_components(table: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    comps = [c for c in COMPONENTS if c != "terminal_pnl"]
    width = 0.004
    for ax, process in zip(axes, PROCESSES):
        sub = table[(table["process"] == process) & (table["p0"] == 0.6)]
        for k, comp in enumerate(comps):
            s = sub[sub["component"] == comp].sort_values("abs_bias")
            x = s["abs_bias"].to_numpy() + (k - 1.5) * width
            ax.bar(x, s["asymmetry"], width=width, color=COMPONENT_COLORS[comp],
                   yerr=1.96 * s["se"], capsize=2, label=comp.replace("_", " "))
        total = sub[sub["component"] == "terminal_pnl"].sort_values("abs_bias")
        ax.plot(total["abs_bias"], total["asymmetry"], marker="o", color="black",
                linewidth=2, label="terminal P&L")
        ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
        ax.set_title(f"{process} process, p0 = 0.6")
        ax.set_xlabel("|calibration bias|")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Asymmetry of component  X(+b) − X(−b)  (95% CI)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Which channel carries the asymmetry?", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


def main() -> int:
    if not BINARY.exists():
        print(f"Binary not found: {BINARY}")
        return 1
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(PROCESSES, P0_GRID, BIAS_GRID, range(NUM_SEEDS)))
    print(f"Running {len(grid)} simulations on {MAX_WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        runs = pd.DataFrame(list(pool.map(run_one, grid, chunksize=16)))

    max_err = runs["identity_error"].max()
    print(f"Identity check: max error {max_err:.2e}")
    if max_err > IDENTITY_TOL:
        print("IDENTITY VIOLATED")
        return 1

    table = asymmetry_table(runs)
    table.to_csv(RESULTS_DIR / "asymmetry.csv", index=False)

    print("\nP&L asymmetry at |bias| = 0.10 (mean(+b) - mean(-b), z-score):")
    at10 = table[table["abs_bias"] == 0.10].pivot_table(
        index=["process", "p0"], columns="component", values="asymmetry")
    z10 = table[table["abs_bias"] == 0.10].pivot_table(
        index=["process", "p0"], columns="component", values="z")
    print(at10[COMPONENTS].round(1).to_string())
    print("\nz-scores:")
    print(z10[COMPONENTS].round(1).to_string())

    print("\nMean settlement drift by sign of bias (p0=0.6, |b|=0.10):")
    for process in PROCESSES:
        for b in (-0.10, 0.10):
            s = runs[(runs["process"] == process) & (runs["p0"] == 0.6)
                     & (runs["calibration_bias"] == b)]
            print(f"  {process:10s} bias {b:+.2f}: drift {s['settlement_drift'].mean():8.1f} "
                  f"± {1.96 * s['settlement_drift'].sem():5.1f}   "
                  f"terminal inventory {s['terminal_inventory'].mean():7.1f}   "
                  f"P&L {s['terminal_pnl'].mean():7.1f} ± {1.96 * s['terminal_pnl'].sem():5.1f}")

    plot_by_design(table, RESULTS_DIR / "asymmetry_by_design.png")
    plot_components(table, RESULTS_DIR / "asymmetry_components.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
