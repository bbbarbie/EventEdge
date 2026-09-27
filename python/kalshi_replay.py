"""Re-run the calibration-bias sweep with real Kalshi markets as the truth.

For every path fetched by python/kalshi.py (data/kalshi/*_path.csv) the
simulator replays the market's traded price as the latent probability
(`--prob-path`), settles at the market's real result when it has one
(`--outcome yes|no`, else a draw from the terminal price), and sweeps the
market maker's calibration bias over many seeds. Seeds vary the signals and
trader arrivals only: the path is the same in every run.

Outputs:
  results/kalshi_replay_runs.csv          one row per run (gitignored)
  results/kalshi_replay_pnl_vs_bias.png   mean terminal P&L vs bias, one
                                          line per market, synthetic
                                          additive process as reference
  results/kalshi_path_stats.csv           per-market path statistics vs the
                                          synthetic defaults

All simulation logic lives in C++; this script only orchestrates runs.
"""

from __future__ import annotations

import argparse
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
import kalshi  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY = REPO_ROOT / "eventedge"
DATA_DIR = REPO_ROOT / "data" / "kalshi_replay"
RESULTS_DIR = REPO_ROOT / "results"

BIAS_GRID = np.round(np.arange(-0.10, 0.1001, 0.02), 3)
INFORMED_FRACTION = 0.3
NUM_SEEDS = 30
MAX_WORKERS = 8


def run_one(args: tuple[str, Path, str | None, float, int]) -> Path:
    ticker, path, outcome, bias, seed = args
    prefix = DATA_DIR / f"{ticker}_b{bias:+.2f}_s{seed}"
    summary = Path(f"{prefix}_summary.csv")
    if not summary.exists():
        cmd = [str(BINARY),
               "--seed", str(seed),
               "--bias", f"{bias:.4f}",
               "--informed-fraction", f"{INFORMED_FRACTION:.4f}",
               "--prob-path", str(path),
               "--log-detail", "0",
               "--out-prefix", str(prefix)]
        if outcome:
            cmd += ["--outcome", outcome]
        subprocess.run(cmd, check=True, capture_output=True)
    return summary


def synthetic_reference(num_steps: int) -> pd.DataFrame:
    """Same sweep on the default additive process, matched in length, so the
    real-path curves have a like-for-like comparison."""
    frames = []
    for bias, seed in itertools.product(BIAS_GRID, range(NUM_SEEDS)):
        prefix = DATA_DIR / f"synthetic{num_steps}_b{bias:+.2f}_s{seed}"
        summary = Path(f"{prefix}_summary.csv")
        if not summary.exists():
            subprocess.run([str(BINARY), "--seed", str(seed), "--steps", str(num_steps),
                            "--bias", f"{bias:.4f}",
                            "--informed-fraction", f"{INFORMED_FRACTION:.4f}",
                            "--log-detail", "0", "--out-prefix", str(prefix)],
                           check=True, capture_output=True)
        frames.append(pd.read_csv(summary))
    ref = pd.concat(frames, ignore_index=True)
    ref["ticker"] = f"synthetic additive ({num_steps} steps)"
    return ref


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=kalshi.DATA_DIR,
                        help="where kalshi.py fetch wrote *_path.csv")
    parser.add_argument("--seeds", type=int, default=NUM_SEEDS)
    args = parser.parse_args(argv)
    seeds = range(args.seeds)

    pairs = kalshi.discover_paths(args.data_dir)
    if not pairs:
        print(f"No Kalshi paths under {args.data_dir}. Fetch one first:\n"
              "  python3 python/kalshi.py fetch --ticker <TICKER> --interval 1")
        return 1
    if not BINARY.exists():
        print(f"Simulator binary missing at {BINARY}; build it first.")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stats_rows = []
    grid = []
    for csv_file, meta in pairs:
        ticker = meta.get("ticker") or csv_file.stem
        probs = kalshi.read_path(csv_file)
        stats = kalshi.path_stats(probs)
        stats.update({"ticker": ticker, "result": meta.get("result"),
                      "title": meta.get("title")})
        stats_rows.append(stats)
        print(f"{ticker}  {meta.get('title', '')}  result={meta.get('result')}")
        print(kalshi.format_stats(stats))
        grid += [(ticker, csv_file, meta.get("result"), b, s)
                 for b, s in itertools.product(BIAS_GRID, seeds)]

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(RESULTS_DIR / "kalshi_path_stats.csv", index=False)

    print(f"Running {len(grid)} replay simulations "
          f"({len(pairs)} markets x {len(BIAS_GRID)} biases x {len(seeds)} seeds)...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        summaries = list(pool.map(run_one, grid, chunksize=8))
    runs = pd.concat([pd.read_csv(p) for p in summaries], ignore_index=True)
    runs["ticker"] = [g[0] for g in grid]
    runs.to_csv(RESULTS_DIR / "kalshi_replay_runs.csv", index=False)

    median_steps = int(runs.groupby("ticker")["num_steps"].first().median())
    reference = synthetic_reference(median_steps)

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    tickers = sorted(runs["ticker"].unique())
    for idx, ticker in enumerate(tickers):
        sub = runs[runs["ticker"] == ticker]
        stats = sub.groupby("calibration_bias")["terminal_pnl"].agg(["mean", "sem"])
        result = sub["event_outcome"].iloc[0] if sub["outcome_source"].iloc[0] == "forced" else None
        label = ticker if result is None else f"{ticker} (settled {'YES' if result else 'NO'})"
        ax.errorbar(stats.index, stats["mean"], yerr=1.96 * stats["sem"],
                    marker="o", capsize=3, linewidth=1.8,
                    color=cmap(idx / max(len(tickers) - 1, 1)), label=label)
    ref_stats = reference.groupby("calibration_bias")["terminal_pnl"].agg(["mean", "sem"])
    ax.errorbar(ref_stats.index, ref_stats["mean"], yerr=1.96 * ref_stats["sem"],
                marker="s", capsize=3, linewidth=1.5, color="grey", linestyle="--",
                label=reference["ticker"].iloc[0])
    ax.axhline(0.0, color="grey", linewidth=1, linestyle=":")
    ax.set_xlabel("Calibration bias")
    ax.set_ylabel("Mean terminal P&L (95% CI)")
    ax.set_title(f"Kalshi replay: informed fraction {INFORMED_FRACTION:.0%}, "
                 f"{len(seeds)} seeds per point")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = RESULTS_DIR / "kalshi_replay_pnl_vs_bias.png"
    fig.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")

    table = runs.groupby(["ticker", "calibration_bias"])["terminal_pnl"].mean().unstack()
    print(table.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
