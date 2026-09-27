"""Empirical calibration curve of Kalshi prices from settled markets.

Turns the simulator's calibration bias from a free parameter into a
measured distribution. For every settled market in the chosen series, the
traded price at fixed times before expiry is paired with the market's
result; binning prices gives the realised YES frequency per price bucket,
and (frequency - price) is the market's calibration error there, with a
Wilson interval. Stratified by series (Kalshi's category proxy) and by
time-to-expiry band.

Survivorship: only settled markets are used, but *all* settled markets in
the listing are used, whatever their result; nothing is conditioned on the
outcome or on later price behaviour. Illiquid markets are dropped by a
volume floor, which is a selection on activity, not on outcome.

    python3 python/kalshi_calibration.py --series KXBTCD KXETHD --max-markets 200
    python3 python/kalshi_calibration.py --from-cache      # re-analyse cached observations

Outputs:
  data/kalshi/calibration_observations.csv   one row per (market, horizon)
  results/kalshi_calibration.csv             per bucket x stratum
  results/kalshi_calibration.png
  results/kalshi_bias_distribution.csv       mean / sd of (freq - price) per
                                             stratum, the input for --bias
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kalshi  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OBS_FILE = kalshi.DATA_DIR / "calibration_observations.csv"

# Hours before expiry at which the price is sampled, and the band each
# horizon belongs to.
HORIZONS_H = [720, 168, 24, 6, 1]
BANDS = [(">7d", lambda h: h > 168), ("1–7d", lambda h: 24 < h <= 168),
         ("<1d", lambda h: h <= 24)]
BUCKETS = [(lo / 100.0, (lo + 10) / 100.0) for lo in range(0, 100, 10)]


def band_of(hours: float) -> str:
    for name, test in BANDS:
        if test(hours):
            return name
    return BANDS[-1][0]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def price_at(candles: Sequence[dict], ts: int) -> float | None:
    """Last known probability at or before ts (forward-filled candles)."""
    path = kalshi.candles_to_path(candles)
    last = None
    for end_ts, p in path:
        if end_ts <= ts:
            last = p
        else:
            break
    return last


def observe_market(market: dict, candles: Sequence[dict]) -> list[dict]:
    """One observation per horizon that the market's life covers."""
    result = kalshi.market_outcome(market)
    if result is None:
        return []
    close_ts = kalshi.parse_time(market["close_time"])
    open_ts = kalshi.parse_time(market["open_time"])
    rows = []
    for hours in HORIZONS_H:
        ts = close_ts - hours * 3600
        if ts < open_ts:
            continue
        p = price_at(candles, ts)
        if p is None:
            continue
        rows.append({"ticker": market["ticker"],
                     "series": kalshi.series_from_ticker(market["ticker"]),
                     "hours_to_expiry": hours, "band": band_of(hours),
                     "price": p, "yes": 1 if result == "yes" else 0,
                     "volume": market.get("volume", 0)})
    return rows


def calibration_table(obs: Sequence[dict]) -> list[dict]:
    """Realised frequency per price bucket, per stratum (series x band),
    plus pooled rows with series='ALL' and/or band='ALL'."""
    rows = []
    strata = {(o["series"], o["band"]) for o in obs}
    keys = sorted(strata) + sorted({("ALL", b) for _, b in strata}) \
        + sorted({(s, "ALL") for s, _ in strata}) + [("ALL", "ALL")]
    for series, band in keys:
        sel = [o for o in obs if (series == "ALL" or o["series"] == series)
               and (band == "ALL" or o["band"] == band)]
        for lo, hi in BUCKETS:
            inb = [o for o in sel if lo <= o["price"] < hi or (hi == 1.0 and o["price"] == 1.0)]
            n = len(inb)
            k = sum(o["yes"] for o in inb)
            mean_price = sum(o["price"] for o in inb) / n if n else float("nan")
            freq = k / n if n else float("nan")
            low, high = wilson(k, n)
            rows.append({"series": series, "band": band, "bucket_lo": lo, "bucket_hi": hi,
                         "n": n, "yes": k, "mean_price": mean_price, "freq": freq,
                         "error": freq - mean_price if n else float("nan"),
                         "ci_low": low, "ci_high": high})
    return rows


def bias_distribution(table: Sequence[dict]) -> list[dict]:
    """Observation-weighted mean and sd of the calibration error per stratum:
    what to feed the simulator as the bias grid."""
    out = []
    strata = sorted({(r["series"], r["band"]) for r in table})
    for series, band in strata:
        rows = [r for r in table if r["series"] == series and r["band"] == band and r["n"] > 0]
        n = sum(r["n"] for r in rows)
        if n == 0:
            continue
        mean = sum(r["error"] * r["n"] for r in rows) / n
        var = sum(r["n"] * (r["error"] - mean) ** 2 for r in rows) / n
        worst = max(rows, key=lambda r: abs(r["error"]))
        out.append({"series": series, "band": band, "observations": n,
                    "mean_error": mean, "sd_error": math.sqrt(var),
                    "worst_bucket_lo": worst["bucket_lo"], "worst_error": worst["error"]})
    return out


def write_rows(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        rows = []
        for r in csv.DictReader(fh):
            for k in ("price", "volume"):
                r[k] = float(r[k])
            r["yes"] = int(r["yes"])
            r["hours_to_expiry"] = int(float(r["hours_to_expiry"]))
            rows.append(r)
        return rows


def collect(series: Sequence[str], max_markets: int, min_volume: int,
            interval_min: int) -> list[dict]:
    obs: list[dict] = []
    for s in series:
        markets = kalshi.list_markets(series_ticker=s, status="settled", limit=200,
                                      max_pages=max(1, max_markets // 200 + 1))
        markets = [m for m in markets if (m.get("volume") or 0) >= min_volume][:max_markets]
        print(f"{s}: {len(markets)} settled markets with volume >= {min_volume}")
        for m in markets:
            start = kalshi.parse_time(m["open_time"])
            end = kalshi.parse_time(m["close_time"])
            try:
                candles = kalshi.get_candlesticks(s, m["ticker"], start_ts=start, end_ts=end,
                                                  period_interval=interval_min)
            except kalshi.KalshiError as err:
                print(f"  skip {m['ticker']}: {err}")
                continue
            obs.extend(observe_market(m, candles))
    return obs


def plot(table: Sequence[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = [b for b, _ in BANDS]
    colors = {">7d": "#2980b9", "1–7d": "#7f8c8d", "<1d": "#c0392b"}
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="grey", linewidth=1, linestyle="--", label="perfect calibration")
    for band in bands:
        rows = [r for r in table if r["series"] == "ALL" and r["band"] == band and r["n"] > 0]
        if not rows:
            continue
        x = [r["mean_price"] for r in rows]
        y = [r["freq"] for r in rows]
        lo = [r["freq"] - r["ci_low"] for r in rows]
        hi = [r["ci_high"] - r["freq"] for r in rows]
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", markersize=5, capsize=3, linewidth=2,
                    color=colors[band], label=f"{band} to expiry (n={sum(r['n'] for r in rows)})")
    ax.set_xlabel("Kalshi price (implied probability)")
    ax.set_ylabel("Realised YES frequency (Wilson 95% CI)")
    ax.set_title("Empirical calibration of Kalshi prices, settled markets")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--series", nargs="*", default=[], help="series tickers to pull")
    parser.add_argument("--max-markets", type=int, default=200, help="per series")
    parser.add_argument("--min-volume", type=int, default=100)
    parser.add_argument("--interval", type=int, default=60, help="candle minutes (1/60/1440)")
    parser.add_argument("--from-cache", action="store_true", help="skip fetching")
    parser.add_argument("--obs-file", type=Path, default=OBS_FILE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    if args.from_cache:
        if not args.obs_file.exists():
            print(f"no cached observations at {args.obs_file}")
            return 1
        obs = read_rows(args.obs_file)
    else:
        if not args.series:
            print("give --series (e.g. KXBTCD) or --from-cache")
            return 1
        try:
            obs = collect(args.series, args.max_markets, args.min_volume, args.interval)
        except kalshi.KalshiError as err:
            print(f"kalshi: {err}", file=sys.stderr)
            return 2
        write_rows(obs, args.obs_file)
    if not obs:
        print("no observations")
        return 1

    table = calibration_table(obs)
    write_rows(table, args.results_dir / "kalshi_calibration.csv")
    dist = bias_distribution(table)
    write_rows(dist, args.results_dir / "kalshi_bias_distribution.csv")

    print(f"{len(obs)} observations from {len({o['ticker'] for o in obs})} markets")
    print(f"{'series':<10} {'band':<6} {'n':>6} {'mean err':>9} {'sd err':>7}  worst bucket")
    for d in dist:
        print(f"{d['series']:<10} {d['band']:<6} {d['observations']:>6} "
              f"{d['mean_error']:>+9.3f} {d['sd_error']:>7.3f}  "
              f"[{d['worst_bucket_lo']:.1f},+.1) {d['worst_error']:+.3f}")
    plot(table, args.results_dir / "kalshi_calibration.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
