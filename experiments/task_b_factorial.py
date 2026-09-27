"""Task B: calibration error x informed flow -> adverse selection and P&L.

Factorial: process in {additive (old clipped), martingale (new p(1-p))} x
p0 in {0.2, 0.5, 0.8} x eps in {-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10} x
pi in {0, 0.1, 0.2, 0.4}; sigma, horizon (2000), spread (0.04) at defaults;
N seeds per cell from the pilot (--seeds). Same seeds in every cell (common
random numbers: latent path, arrivals and settlement identical across eps).

Per cell: mean P&L per contract (settled and marked at p_T) with SE, the
decomposition components, fills by side, share of fills against informed
traders, markouts by trader type at h = 0, 10, 50 and settlement.

B1: figure P&L vs eps, one line per pi, one panel per p0 (one figure per
process); regression P&L_pc ~ eps + pi + eps*pi + |eps| + |eps|*pi + p0 FE
on run-level data with seed-clustered SEs.
B2: (a) asymmetry on both processes, (b) reflection test p0=0.2 vs 0.8,
(c) p0 = 0.5 asymmetry and quote-clip counts, (d) P&L conditional on outcome.

Outputs: results/taskB_cells.csv, taskB_runs.csv (gitignored), taskB_tables.md,
taskB_pnl_vs_eps_<process>.png, taskB_asymmetry.png
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MARKOUT_HORIZONS, TYPES, cache_or_compute, mean_se, ols_cluster  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY = REPO_ROOT / "eventedge"
DATA_DIR = REPO_ROOT / "data" / "task_b"
RESULTS_DIR = REPO_ROOT / "results"

PROCESSES = ["additive", "martingale"]
P0_GRID = [0.2, 0.5, 0.8]
EPS_GRID = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
PI_GRID = [0.0, 0.1, 0.2, 0.4]
STEPS, SPREAD = 2000, 0.04
PI_COLORS = {0.0: "#7f8c8d", 0.1: "#2980b9", 0.2: "#e67e22", 0.4: "#c0392b"}
COMPONENTS = ["spread_capture", "adverse_selection", "settlement_drift", "settlement_draw"]


def run_one(args):
    proc, p0, eps, pi, seed = args
    prefix = DATA_DIR / f"{proc}_p{p0:.1f}_e{eps:+.2f}_i{pi:.1f}_s{seed}"
    if not Path(f"{prefix}_stats.json").exists():
        subprocess.run([str(BINARY), "--seed", str(seed), "--steps", str(STEPS),
                        "--prob-process", proc, "--p0", f"{p0:.4f}", "--bias", f"{eps:.4f}",
                        "--informed-fraction", f"{pi:.4f}", "--spread", f"{SPREAD:.4f}",
                        "--log-detail", "1", "--out-prefix", str(prefix)],
                       check=True, capture_output=True)
    row = cache_or_compute(prefix)
    row.update(process=proc, p0=p0, eps=eps, pi=pi)
    return row


def fmt(m, s, d=4):
    return f"{m:+.{d}f} ± {s:.{d}f}"


def cell_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (proc, p0, eps, pi), g in runs.groupby(["process", "p0", "eps", "pi"]):
        r = {"process": proc, "p0": p0, "eps": eps, "pi": pi, "N": len(g)}
        for col in ["terminal_pnl_pc", "terminal_pnl_marked_pc"] + [f"{c}_pc" for c in COMPONENTS] \
                + ["contracts", "fills_BUY", "fills_SELL", "informed_share", "bid_clipped_steps",
                   "ask_clipped_steps", "estimate_clipped_steps", "terminal_inventory"]:
            r[col], r[f"{col}_se"] = mean_se(g[col])
        for t in TYPES:
            for h in MARKOUT_HORIZONS + ["settle"]:
                r[f"markout_{h}_{t}"], r[f"markout_{h}_{t}_se"] = mean_se(g[f"markout_{h}_{t}"])
        rows.append(r)
    return pd.DataFrame(rows)


def regression(runs: pd.DataFrame, ycol: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    eps, pi = runs["eps"].to_numpy(), runs["pi"].to_numpy()
    cols = ["const", "eps", "pi", "eps*pi", "|eps|", "|eps|*pi", "p0=0.5", "p0=0.8"]
    X = np.column_stack([np.ones(len(runs)), eps, pi, eps * pi, np.abs(eps), np.abs(eps) * pi,
                         (runs["p0"] == 0.5).astype(float), (runs["p0"] == 0.8).astype(float)])
    beta, se = ols_cluster(X, runs[ycol].to_numpy(), runs["seed"].to_numpy())
    return cols, beta, se


def paired_asymmetry(runs: pd.DataFrame, ycol: str) -> pd.DataFrame:
    """A(|eps|) = mean over seeds of [y(+eps) - y(-eps)], paired by seed (CRN)."""
    rows = []
    for (proc, p0, pi), g in runs.groupby(["process", "p0", "pi"]):
        for a in (0.02, 0.05, 0.10):
            plus = g[np.isclose(g.eps, a)].set_index("seed")[ycol]
            minus = g[np.isclose(g.eps, -a)].set_index("seed")[ycol]
            diff = (plus - minus).dropna()
            m, s = mean_se(diff)
            rows.append({"process": proc, "p0": p0, "pi": pi, "abs_eps": a,
                         "asymmetry": m, "se": s, "z": m / s if s else np.nan, "N": len(diff)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, required=True)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(PROCESSES, P0_GRID, EPS_GRID, PI_GRID, range(args.seeds)))
    print(f"{len(grid)} runs ({len(grid) // args.seeds} cells x {args.seeds} seeds)")
    with ProcessPoolExecutor(max_workers=max(2, os.cpu_count() or 2)) as pool:
        runs = pd.DataFrame(list(pool.map(run_one, grid, chunksize=16)))
    max_err = runs["identity_error"].max()
    print(f"decomposition identity max error {max_err:.2e}")
    assert max_err < 1e-6
    c = runs["contracts"].clip(lower=1)
    runs["terminal_pnl_pc"] = runs["terminal_pnl"] / c
    runs["terminal_pnl_marked_pc"] = runs["terminal_pnl_marked"] / c
    for comp in COMPONENTS:
        runs[f"{comp}_pc"] = runs[comp] / c
    runs["informed_share"] = runs["fills_INFORMED"] / c
    runs.to_csv(RESULTS_DIR / "taskB_runs.csv", index=False)
    cells = cell_table(runs)
    cells.to_csv(RESULTS_DIR / "taskB_cells.csv", index=False)

    md = [f"Design: {len(cells)} cells x N = {args.seeds} seeds, T = {STEPS}, spread {SPREAD}, "
          "σ at repo defaults. P&L per contract traded. SE over seeds.\n"]

    # ---- per-cell summary (martingale; additive in the CSV)
    for proc in PROCESSES:
        md.append(f"\n#### Per-cell summary, {proc} process (P&L per contract, settled; "
                  "components per contract; fills by side; informed share of fills)\n")
        md.append("| p₀ | π | ε | P&L (SE) | marked P&L (SE) | spread | adverse | settle drift | settle draw | fills B/S | informed share | markout INF h0/h50/settle |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in cells[cells.process == proc].sort_values(["p0", "pi", "eps"]).iterrows():
            md.append(f"| {r.p0} | {r.pi} | {r.eps:+.2f} | {fmt(r.terminal_pnl_pc, r.terminal_pnl_pc_se)} | "
                      f"{fmt(r.terminal_pnl_marked_pc, r.terminal_pnl_marked_pc_se)} | "
                      f"{r.spread_capture_pc:+.4f} | {r.adverse_selection_pc:+.4f} | "
                      f"{r.settlement_drift_pc:+.4f} | {r.settlement_draw_pc:+.4f} | "
                      f"{r.fills_BUY:.0f}/{r.fills_SELL:.0f} | {r.informed_share:.3f} | "
                      f"{r.markout_0_INFORMED:+.3f}/{r.markout_50_INFORMED:+.3f}/{r.markout_settle_INFORMED:+.3f} |")

    # ---- B1 regression
    md.append("\n### B1. Regression P&L_pc ~ ε + π + ε·π + |ε| + |ε|·π + p₀ FE (run level, seed-clustered SE, 95% CI)\n")
    md.append("| process | outcome | " + " | ".join(["const", "ε", "π", "ε·π", "|ε|", "|ε|·π", "p₀=0.5", "p₀=0.8"]) + " |")
    md.append("|---|---|" + "---|" * 8)
    reg_rows = []
    for proc in PROCESSES:
        sub = runs[runs.process == proc]
        for ycol, label in [("terminal_pnl_pc", "settled"), ("terminal_pnl_marked_pc", "marked"),
                            ("adverse_selection_pc", "adverse selection")]:
            cols, beta, se = regression(sub, ycol)
            md.append(f"| {proc} | {label} | " + " | ".join(
                f"{b:+.4f} [{b - 1.96 * s:+.4f}, {b + 1.96 * s:+.4f}]" for b, s in zip(beta, se)) + " |")
            for cname, b, s in zip(cols, beta, se):
                reg_rows.append({"process": proc, "outcome": label, "term": cname, "coef": b, "se": s})
    pd.DataFrame(reg_rows).to_csv(RESULTS_DIR / "taskB_regression.csv", index=False)

    # ---- B2 (a)/(c): paired asymmetry
    asym = paired_asymmetry(runs, "terminal_pnl_pc")
    asym_m = paired_asymmetry(runs, "terminal_pnl_marked_pc")
    asym_a = paired_asymmetry(runs, "adverse_selection_pc")
    asym_d = paired_asymmetry(runs, "settlement_drift_pc")
    asym.to_csv(RESULTS_DIR / "taskB_asymmetry.csv", index=False)
    md.append("\n### B2(a)/(c). Asymmetry A = P&L_pc(+ε) − P&L_pc(−ε), paired by seed, 95% CI\n")
    md.append("| process | p₀ | π | |ε| | settled A | marked A | adverse-selection A | settlement-drift A |")
    md.append("|---|---|---|---|---|---|---|---|")
    for (proc, p0, pi, a), _ in asym.groupby(["process", "p0", "pi", "abs_eps"]):
        def pick(df):
            r = df[(df.process == proc) & (df.p0 == p0) & (df.pi == pi) & (df.abs_eps == a)].iloc[0]
            return f"{r.asymmetry:+.4f} ± {1.96 * r.se:.4f}"
        md.append(f"| {proc} | {p0} | {pi} | {a} | {pick(asym)} | {pick(asym_m)} | {pick(asym_a)} | {pick(asym_d)} |")

    # ---- B2 (b): reflection test: (0.2, eps, pi) vs (0.8, -eps, pi), buy<->sell
    md.append("\n### B2(b). Reflection test: cell (p₀=0.2, ε, π) vs mirror (p₀=0.8, −ε, π), z-scores (two-sample)\n")
    md.append("| process | π | ε | ΔP&L_pc (z) | Δmarked (z) | fills BUY(0.2) vs SELL(0.8) (z) | Δinformed share (z) | Δadverse_pc (z) |")
    md.append("|---|---|---|---|---|---|---|---|")
    refl_rows = []
    for proc in PROCESSES:
        for pi in PI_GRID:
            for eps in EPS_GRID:
                g1 = runs[(runs.process == proc) & (runs.p0 == 0.2) & np.isclose(runs.eps, eps) & (runs.pi == pi)]
                g2 = runs[(runs.process == proc) & (runs.p0 == 0.8) & np.isclose(runs.eps, -eps) & (runs.pi == pi)]
                def z(c1, c2):
                    m1, s1 = mean_se(g1[c1]); m2, s2 = mean_se(g2[c2])
                    d = m1 - m2; s = np.sqrt(s1 ** 2 + s2 ** 2)
                    return d, (d / s if s else np.nan)
                dp, zp = z("terminal_pnl_pc", "terminal_pnl_pc")
                dm, zm = z("terminal_pnl_marked_pc", "terminal_pnl_marked_pc")
                df_, zf = z("fills_BUY", "fills_SELL")
                di, zi = z("informed_share", "informed_share")
                da, za = z("adverse_selection_pc", "adverse_selection_pc")
                refl_rows.append({"process": proc, "pi": pi, "eps": eps, "z_pnl": zp, "z_marked": zm,
                                  "z_fills": zf, "z_informed": zi, "z_adverse": za})
                md.append(f"| {proc} | {pi} | {eps:+.2f} | {dp:+.4f} ({zp:+.1f}) | {dm:+.4f} ({zm:+.1f}) | "
                          f"{df_:+.1f} ({zf:+.1f}) | {di:+.4f} ({zi:+.1f}) | {da:+.4f} ({za:+.1f}) |")
    refl = pd.DataFrame(refl_rows)
    refl.to_csv(RESULTS_DIR / "taskB_reflection.csv", index=False)
    md.append(f"\nReflection: max |z| over {len(refl)} cell pairs: P&L {refl.z_pnl.abs().max():.2f}, "
              f"marked {refl.z_marked.abs().max():.2f}, fills {refl.z_fills.abs().max():.2f}, "
              f"informed share {refl.z_informed.abs().max():.2f}, adverse {refl.z_adverse.abs().max():.2f}. "
              f"Share of |z| > 1.96 (expect ~5%): "
              f"{(refl[['z_pnl','z_marked','z_fills','z_informed','z_adverse']].abs() > 1.96).mean().mean():.1%}.")

    # ---- B2 (c): quote clipping per cell
    md.append("\n### B2(c). Quote clipping: mean steps per run (of 2000) with bid at 0.01 / ask at 0.99, by cell (martingale; additive in CSV)\n")
    md.append("| p₀ | π | " + " | ".join(f"ε={e:+.2f}" for e in EPS_GRID) + " |")
    md.append("|---|---|" + "---|" * len(EPS_GRID))
    for p0 in P0_GRID:
        for pi in PI_GRID:
            vals = []
            for e in EPS_GRID:
                r = cells[(cells.process == "martingale") & (cells.p0 == p0) & (cells.pi == pi) & np.isclose(cells.eps, e)].iloc[0]
                vals.append(f"{r.bid_clipped_steps:.0f}/{r.ask_clipped_steps:.0f}")
            md.append(f"| {p0} | {pi} | " + " | ".join(vals) + " |")

    # ---- B2 (d): conditional on outcome
    md.append("\n### B2(d). P&L per contract conditional on the settlement outcome, by ε (pooled over π and p₀), settled P&L\n")
    md.append("| process | ε | P(YES) | P&L | YES | P&L | NO | difference YES−NO |")
    md.append("|---|---|---|---|---|---|")
    for proc in PROCESSES:
        for eps in EPS_GRID:
            g = runs[(runs.process == proc) & np.isclose(runs.eps, eps)]
            y = g[g.event_outcome == 1]["terminal_pnl_pc"]; n = g[g.event_outcome == 0]["terminal_pnl_pc"]
            my, sy = mean_se(y); mn, sn = mean_se(n)
            md.append(f"| {proc} | {eps:+.2f} | {g.event_outcome.mean():.3f} | {fmt(my, sy)} | {fmt(mn, sn)} | "
                      f"{fmt(my - mn, np.sqrt(sy**2 + sn**2))} |")
    (RESULTS_DIR / "taskB_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:1] + md[-40:]))

    # ---- figures
    for proc in PROCESSES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
        for ax, p0 in zip(axes, P0_GRID):
            for pi in PI_GRID:
                s = cells[(cells.process == proc) & (cells.p0 == p0) & (cells.pi == pi)].sort_values("eps")
                ax.errorbar(s["eps"], s["terminal_pnl_pc"], yerr=1.96 * s["terminal_pnl_pc_se"],
                            marker="o", markersize=4, capsize=3, linewidth=2, color=PI_COLORS[pi],
                            label=f"π = {pi}")
            ax.axhline(0, color="grey", linewidth=1, linestyle="--")
            ax.set_title(f"p₀ = {p0}")
            ax.set_xlabel("calibration error ε (added to the MM's probability estimate)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("settled P&L per contract (95% CI)")
        axes[0].legend(title="informed share")
        fig.suptitle(f"Task B1: {proc} process, N = {args.seeds} seeds per cell, T = {STEPS}, spread {SPREAD}", y=1.02)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"taskB_pnl_vs_eps_{proc}.png", dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, proc in zip(axes, PROCESSES):
        for p0, color in zip(P0_GRID, ["#2980b9", "#7f8c8d", "#c0392b"]):
            s = asym[(asym.process == proc) & (asym.p0 == p0) & (asym.pi == 0.1)].sort_values("abs_eps")
            ax.errorbar(s["abs_eps"], s["asymmetry"], yerr=1.96 * s["se"], marker="o", markersize=5,
                        capsize=3, linewidth=2, color=color, label=f"p₀ = {p0}")
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")
        ax.set_title(f"{proc} process, π = 0.1")
        ax.set_xlabel("|ε|")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("A = P&L_pc(+ε) − P&L_pc(−ε), paired by seed (95% CI)")
    axes[0].legend()
    fig.suptitle("Task B2: asymmetry of settled P&L in the sign of ε", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "taskB_asymmetry.png", dpi=150, bbox_inches="tight")
    print("figures saved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
