"""Markout analysis by trader type.

For every fill, record the true (latent) probability at T+0, T+10, T+50,
and T = settlement, and compute the market maker's markout per contract:

    markout_h = mm_inventory_change_sign * (reference_h - fill_price)

where reference_h is the latent probability h steps after the fill
(the realized 0/1 outcome for the settlement horizon), and
mm_inventory_change_sign is +1 when the MM buys (trader SELL) and
-1 when the MM sells (trader BUY).

A negative markout means the MM lost money on the fill at that horizon.
If adverse selection is working, INFORMED fills are systematically
negative and NOISE fills sit near zero (slightly positive: spread capture).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY = REPO_ROOT / "eventedge"
DATA_DIR = REPO_ROOT / "data" / "markout_runs"
RESULTS_DIR = REPO_ROOT / "results"

HORIZONS = [0, 10, 50]  # steps after the fill; settlement handled separately
NUM_SEEDS = 50
NUM_STEPS = 2000

TYPE_COLORS = {"INFORMED": "#c0392b", "VALUE": "#e67e22", "NOISE": "#2980b9"}


def run_simulations(num_seeds: int, num_steps: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for seed in range(num_seeds):
        prefix = DATA_DIR / f"seed{seed}"
        subprocess.run(
            [str(BINARY), "--seed", str(seed), "--steps", str(num_steps),
             "--out-prefix", str(prefix)],
            check=True, capture_output=True,
        )


def load_markouts(num_seeds: int) -> pd.DataFrame:
    rows = []
    for seed in range(num_seeds):
        prefix = DATA_DIR / f"seed{seed}"
        fills = pd.read_csv(f"{prefix}_fills.csv")
        steps = pd.read_csv(f"{prefix}_steps.csv")
        summary = pd.read_csv(f"{prefix}_summary.csv")
        if fills.empty:
            continue

        p_true = steps.set_index("time_step")["latent_probability"]
        outcome = float(summary["event_outcome"].iloc[0])
        num_steps = int(summary["num_steps"].iloc[0])

        # +1 when MM buys at the bid (trader SELL), -1 when MM sells at the ask
        sign = np.sign(fills["mm_inventory_change"])

        for h in HORIZONS:
            ref_step = fills["time_step"] + h
            ref = p_true.reindex(ref_step).to_numpy()  # NaN past the horizon
            valid = ref_step <= num_steps
            fills[f"markout_{h}"] = np.where(
                valid, sign * (ref - fills["fill_price"]), np.nan)
            fills[f"true_prob_t{h}"] = np.where(valid, ref, np.nan)
        fills["markout_settle"] = sign * (outcome - fills["fill_price"])
        fills["true_prob_settle"] = outcome
        rows.append(fills)

    return pd.concat(rows, ignore_index=True)


def summarize(markouts: pd.DataFrame) -> pd.DataFrame:
    cols = [f"markout_{h}" for h in HORIZONS] + ["markout_settle"]
    records = []
    for trader_type, group in markouts.groupby("trader_type"):
        for col in cols:
            values = group[col].dropna()
            records.append({
                "trader_type": trader_type,
                "horizon": col.replace("markout_", ""),
                "mean": values.mean(),
                "sem": values.sem(),
                "n": len(values),
            })
    return pd.DataFrame(records)


def plot(summary: pd.DataFrame, out_path: Path) -> None:
    horizon_order = [str(h) for h in HORIZONS] + ["settle"]
    labels = ["T+0", "T+10", "T+50", "Settlement"]
    x = np.arange(len(horizon_order))

    fig, ax = plt.subplots(figsize=(8, 5))
    for trader_type in ["INFORMED", "VALUE", "NOISE"]:
        sub = (summary[summary["trader_type"] == trader_type]
               .set_index("horizon").reindex(horizon_order))
        if sub["mean"].isna().all():
            continue
        n = int(sub["n"].max())
        ax.errorbar(
            x, sub["mean"], yerr=1.96 * sub["sem"],
            marker="o", capsize=4, linewidth=2,
            color=TYPE_COLORS[trader_type],
            label=f"{trader_type} (n={n:,} fills)",
        )

    ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Horizon after fill")
    ax.set_ylabel("Avg MM markout per contract\n(sign(inv chg) x (true prob - fill price))")
    ax.set_title("Market-maker markout by trader type\n"
                 f"{NUM_SEEDS} seeds x {NUM_STEPS} steps, 95% CI")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main() -> int:
    if not BINARY.exists():
        print(f"Binary not found: {BINARY}. Build the C++ simulator first.")
        return 1
    run_simulations(NUM_SEEDS, NUM_STEPS)
    markouts = load_markouts(NUM_SEEDS)
    summary = summarize(markouts)

    pivot = summary.pivot(index="trader_type", columns="horizon", values="mean")
    pivot = pivot[[str(h) for h in HORIZONS] + ["settle"]]
    print("\nAverage MM markout per contract by trader type:")
    print(pivot.round(4).to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    markouts.to_csv(RESULTS_DIR / "markouts_per_fill.csv", index=False)
    summary.to_csv(RESULTS_DIR / "markout_summary.csv", index=False)
    plot(summary, RESULTS_DIR / "markout_by_trader_type.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
