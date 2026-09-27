"""Stale-quote sniping: how much does each step of quote latency cost?

The MM quotes off the public signal it saw `--quote-latency L` steps ago.
When the latent probability jumps, its quote is wrong for L steps and
informed traders (who see the new level) hit it. This measures the loss
as a function of L and attributes it to the post-jump windows.

Jumps are located on the latent path itself (|dp| > JUMP_THRESHOLD between
consecutive steps). For each fill, the adverse-selection term
dI*(p_true - mid) is assigned to "sniped" if the fill happened within L
steps after a jump (L = 0 counts nothing: a zero-latency MM re-quotes on
the jump step), else to "background". Works on the synthetic process
(jumps: 2% per step, sigma 0.05) or on a real Kalshi path (`--prob-path`),
where the jump timestamps are the market's own.

Outputs:
  results/latency_sniping.csv
  results/latency_sniping.png
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
DATA_DIR = REPO_ROOT / "data" / "latency"
RESULTS_DIR = REPO_ROOT / "results"

LATENCIES = [0, 1, 2, 3, 5, 10, 20]
INFORMED_FRACTION = 0.3
NUM_SEEDS = 30
NUM_STEPS = 2000
JUMP_THRESHOLD = 0.03
MAX_WORKERS = max(2, os.cpu_count() or 2)


def run_one(args: tuple[int, int, str | None]) -> dict:
    latency, seed, prob_path = args
    tag = "synthetic" if prob_path is None else Path(prob_path).stem
    prefix = DATA_DIR / f"{tag}_L{latency}_s{seed}"
    if not Path(f"{prefix}_summary.csv").exists():
        cmd = [str(BINARY), "--seed", str(seed), "--quote-latency", str(latency),
               "--informed-fraction", f"{INFORMED_FRACTION:.4f}", "--bias", "0",
               "--log-detail", "1", "--out-prefix", str(prefix)]
        cmd += ["--prob-path", prob_path] if prob_path else ["--steps", str(NUM_STEPS)]
        subprocess.run(cmd, check=True, capture_output=True)
    return attribute(prefix, latency, seed)


def attribute(prefix: Path, latency: int, seed: int) -> dict:
    steps = pd.read_csv(f"{prefix}_steps.csv")
    fills = pd.read_csv(f"{prefix}_fills.csv")
    summary = pd.read_csv(f"{prefix}_summary.csv").iloc[0]
    p = steps["latent_probability"].to_numpy()
    jump_steps = steps["time_step"].to_numpy()[1:][np.abs(np.diff(p)) > JUMP_THRESHOLD]
    # a fill at step t is "in a stale window" if some jump happened at
    # t-latency+1 .. t (the quote at t still reflects a pre-jump signal)
    sniped = np.zeros(len(fills), dtype=bool)
    if latency > 0 and len(fills) and len(jump_steps):
        t = fills["time_step"].to_numpy()
        for j in jump_steps:
            sniped |= (t >= j) & (t < j + latency)
    d_inv = fills["mm_inventory_change"].astype(float)
    mid = (fills["bid"] + fills["ask"]) / 2.0
    adverse = d_inv * (fills["latent_probability"] - mid)
    return {
        "latency": latency, "seed": seed,
        "terminal_pnl": float(summary["terminal_pnl"]),
        "adverse_total": adverse.sum(),
        "adverse_sniped": adverse[sniped].sum(),
        "adverse_background": adverse[~sniped].sum(),
        "sniped_fills": int(sniped.sum()),
        "sniped_informed": int((sniped & (fills["trader_type"] == "INFORMED")).sum()),
        "jumps": int(len(jump_steps)),
        "fills": int(len(fills)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prob-path", help="replay a real path instead of the synthetic process")
    parser.add_argument("--seeds", type=int, default=NUM_SEEDS)
    args = parser.parse_args(argv)
    if not BINARY.exists():
        print(f"Binary not found: {BINARY}")
        return 1
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = [(L, s, args.prob_path) for L, s in itertools.product(LATENCIES, range(args.seeds))]
    print(f"Running {len(grid)} simulations...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        rows = pd.DataFrame(list(pool.map(run_one, grid, chunksize=4)))

    stats = rows.groupby("latency").agg(
        pnl=("terminal_pnl", "mean"), pnl_sem=("terminal_pnl", "sem"),
        adverse=("adverse_total", "mean"), adverse_sem=("adverse_total", "sem"),
        sniped=("adverse_sniped", "mean"), sniped_sem=("adverse_sniped", "sem"),
        background=("adverse_background", "mean"),
        sniped_fills=("sniped_fills", "mean"), sniped_informed=("sniped_informed", "mean"),
        jumps=("jumps", "mean"), fills=("fills", "mean")).reset_index()
    stats["loss_per_jump"] = stats["sniped"] / stats["jumps"].clip(lower=1e-9)
    stats["informed_share_in_window"] = stats["sniped_informed"] / stats["sniped_fills"].clip(lower=1e-9)
    tag = "synthetic" if args.prob_path is None else Path(args.prob_path).stem
    stats.to_csv(RESULTS_DIR / f"latency_sniping{'' if tag == 'synthetic' else '_' + tag}.csv",
                 index=False)
    print(stats.round(2).to_string())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.errorbar(stats["latency"], stats["adverse"], yerr=1.96 * stats["adverse_sem"],
                 marker="o", markersize=5, capsize=3, linewidth=2, color="#c0392b",
                 label="adverse selection, all fills")
    ax1.errorbar(stats["latency"], stats["sniped"], yerr=1.96 * stats["sniped_sem"],
                 marker="s", markersize=5, capsize=3, linewidth=2, color="#8e44ad",
                 label="…of which within L steps after a jump")
    ax1.plot(stats["latency"], stats["background"], marker="^", markersize=5,
             linewidth=2, color="#7f8c8d", label="…background (no recent jump)")
    ax1.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax1.set_xlabel("Quote latency L (steps)")
    ax1.set_ylabel("Mean adverse-selection P&L (95% CI)")
    ax1.set_title("Loss vs latency")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(stats["latency"], stats["loss_per_jump"], marker="o", markersize=5,
             linewidth=2, color="#8e44ad")
    ax2.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax2.set_xlabel("Quote latency L (steps)")
    ax2.set_ylabel("Sniping loss per jump event")
    ax2.set_title(f"Cost of a stale quote per jump  ({stats['jumps'].iloc[0]:.0f} jumps per run)")
    ax2.grid(alpha=0.3)
    fig.suptitle(f"{tag} path, informed {INFORMED_FRACTION:.0%}, bias 0, {args.seeds} seeds", y=1.02)
    fig.tight_layout()
    out = RESULTS_DIR / f"latency_sniping{'' if tag == 'synthetic' else '_' + tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
