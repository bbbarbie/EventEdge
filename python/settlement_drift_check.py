"""Validate that the martingale probability process removes settlement drift.

The P&L decomposition under the CLAMPED_ADDITIVE process showed a systematic
"settlement noise" term: clamping to [0.01, 0.99] induces drift toward the
interior, which interacts with the signed inventory that calibration bias
creates. Under LOGISTIC_MARTINGALE (dp = vol * p(1-p) * Z) the settlement
term should be statistically indistinguishable from zero at every bias.

Reuses the exact decomposition from pnl_decomposition.py and reads its
already-computed additive results for comparison.

Output: results/settlement_drift_fix.png
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pnl_decomposition import (  # noqa: E402
    BIAS_GRID, BINARY, IDENTITY_TOL, INFORMED_FRACTION, MAX_WORKERS,
    NUM_SEEDS, NUM_STEPS, RESULTS_DIR,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "decomposition_martingale"
ADDITIVE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "decomposition_sweep"


def settlement_split(prefix: Path) -> dict:
    """Split the settlement term to isolate process drift from draw noise:

        sum dI*(Y - p_t) = sum dI*(p_T - p_t)   drift component
                                                 (zero iff p is a martingale)
                         + inv_T * (Y - p_T)    draw noise
                                                 (mean-zero by construction)
    """
    fills = pd.read_csv(f"{prefix}_fills.csv")
    summary = pd.read_csv(f"{prefix}_summary.csv").iloc[0]
    outcome = float(summary["event_outcome"])
    p_terminal = float(summary["terminal_probability"])

    d_inv = fills["mm_inventory_change"].astype(float)
    drift = (d_inv * (p_terminal - fills["latent_probability"])).sum()
    noise = float(summary["terminal_inventory"]) * (outcome - p_terminal)
    total = (d_inv * (outcome - fills["latent_probability"])).sum()
    return {
        "calibration_bias": float(summary["calibration_bias"]),
        "seed": int(summary["seed"]),
        "settlement_drift": drift,
        "settlement_draw_noise": noise,
        "identity_error": abs(drift + noise - total),
    }


# The drift signal is subtle and all bias columns share latent paths within a
# seed, so this check uses more seeds than the main decomposition sweep.
NUM_SEEDS_CHECK = 100


def run_one(args: tuple[str, float, int]) -> Path:
    process, bias, seed = args
    data_dir = DATA_DIR if process == "martingale" else ADDITIVE_DATA_DIR
    prefix = data_dir / f"b{bias:+.2f}_s{seed}"
    if not Path(f"{prefix}_summary.csv").exists():
        subprocess.run(
            [str(BINARY),
             "--seed", str(seed),
             "--steps", str(NUM_STEPS),
             "--bias", f"{bias:.4f}",
             "--informed-fraction", f"{INFORMED_FRACTION:.4f}",
             "--prob-process", process,
             "--log-detail", "1",
             "--out-prefix", str(prefix)],
            check=True, capture_output=True,
        )
    return prefix


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ADDITIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    grid = [(proc, b, s)
            for proc in ("martingale", "additive")
            for b in BIAS_GRID
            for s in range(NUM_SEEDS_CHECK)]
    print(f"Running up to {len(grid)} simulations ({NUM_SEEDS_CHECK} seeds/point)...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        prefixes = list(pool.map(run_one, grid, chunksize=8))

    n = len(BIAS_GRID) * NUM_SEEDS_CHECK
    martingale = pd.DataFrame([settlement_split(p) for p in prefixes[:n]])
    additive = pd.DataFrame([settlement_split(p) for p in prefixes[n:]])
    max_err = max(martingale["identity_error"].max(), additive["identity_error"].max())
    print(f"Split identity check: max error = {max_err:.2e}")
    if max_err > IDENTITY_TOL:
        print("IDENTITY VIOLATED -- split is wrong.")
        return 1

    fig, ax = plt.subplots(figsize=(8, 5))
    for frame, label, color in [
        (additive, "Clamped additive walk (artifact)", "#c0392b"),
        (martingale, "p(1-p)-scaled martingale (fixed)", "#27ae60"),
    ]:
        stats = frame.groupby("calibration_bias")["settlement_drift"].agg(["mean", "sem"])
        ax.errorbar(stats.index, stats["mean"], yerr=1.96 * stats["sem"],
                    marker="o", capsize=4, linewidth=2, color=color, label=label)

    ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax.set_xlabel("Calibration bias")
    ax.set_ylabel("Mean inventory-drift P&L:  sum dI x (p_T - p_fill)")
    ax.set_title("Latent-process drift captured by MM inventory\n"
                 f"informed fraction {INFORMED_FRACTION:.0%}, "
                 f"{NUM_SEEDS_CHECK} seeds per bias, 95% CI")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "settlement_drift_fix.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    table = pd.DataFrame({
        "additive_drift": additive.groupby("calibration_bias")["settlement_drift"].mean(),
        "additive_sem": additive.groupby("calibration_bias")["settlement_drift"].sem(),
        "martingale_drift": martingale.groupby("calibration_bias")["settlement_drift"].mean(),
        "martingale_sem": martingale.groupby("calibration_bias")["settlement_drift"].sem(),
    })
    print("\nMean inventory-drift component by bias:")
    print(table.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
