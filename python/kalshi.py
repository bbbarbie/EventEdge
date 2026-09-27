"""Kalshi market data -> EventEdge replay paths.

The simulator's latent probability is synthetic by default. This module pulls
real binary-contract price histories from Kalshi's public trade API and turns
them into replay paths the C++ binary consumes via `--prob-path`, so every
experiment can be re-run with a real market as the hidden truth and, for
settled markets, the real settlement as the outcome (`--outcome yes|no`).

Only public, unauthenticated GET endpoints are used (markets, trades,
candlesticks); nothing is sent to Kalshi beyond query parameters, and no
order-placement code exists here. Standard library only, so the data layer
adds no dependency the simulator does not already have.

    python3 python/kalshi.py list --series KXBTCD --status settled
    python3 python/kalshi.py fetch --ticker KXBTCD-25SEP2717-T115999.99 --interval 1
    python3 python/kalshi.py fetch --ticker <ticker> --source trades --interval 60
    python3 python/kalshi.py stats

`fetch` writes data/kalshi/<ticker>_path.csv (time_step, timestamp,
latent_probability) and <ticker>_meta.json (title, status, result, times),
and caches the raw JSON responses under data/kalshi/raw/. Prices arrive in
cents (1-99) or as dollar strings; both become probabilities in (0, 1).

Replay convention: step t of the simulator reads row t-1 of the path, one
trader-arrival opportunity per row. A 1-minute candle path over a day is
1440 steps; hourly candles over a month, ~720.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "kalshi"
BASE_URL = os.environ.get(
    "KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2"
).rstrip("/")
USER_AGENT = "EventEdge-research/0.1 (+python urllib)"

# Kalshi caps one candlestick request at this many periods.
MAX_CANDLES_PER_REQUEST = 5000
VALID_INTERVALS_MIN = (1, 60, 1440)

# Defaults of the synthetic processes, for the calibration comparison
# printed by `path_stats`. Keep in sync with src/event_probability.cpp.
SYNTHETIC_ADDITIVE_VOL = 0.01
SYNTHETIC_MARTINGALE_VOL = 0.042
SYNTHETIC_JUMP_RATE = 0.02


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class KalshiError(RuntimeError):
    pass


def _get(path: str, params: dict | None = None, *, retries: int = 4,
         timeout: float = 30.0) -> dict:
    """GET a public endpoint with retries on 429 / 5xx (exponential backoff)."""
    query = {k: v for k, v in (params or {}).items() if v is not None}
    url = f"{BASE_URL}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    delay = 1.0
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers={
            "Accept": "application/json", "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            retryable = err.code == 429 or err.code >= 500
            if not retryable or attempt == retries:
                body = err.read().decode("utf-8", "replace")[:300]
                raise KalshiError(f"HTTP {err.code} for {url}: {body}") from err
        except urllib.error.URLError as err:
            if attempt == retries:
                raise KalshiError(f"Cannot reach {url}: {err.reason}") from err
        time.sleep(delay)
        delay *= 2
    raise KalshiError(f"unreachable: {url}")  # pragma: no cover


# --------------------------------------------------------------------------
# Endpoints (public market data only)
# --------------------------------------------------------------------------

def list_markets(*, series_ticker: str | None = None,
                 event_ticker: str | None = None, status: str | None = None,
                 tickers: Sequence[str] | None = None, limit: int = 200,
                 max_pages: int = 10) -> list[dict]:
    """GET /markets, following the cursor. status: open|closed|settled|unopened."""
    out: list[dict] = []
    cursor = None
    for _ in range(max_pages):
        page = _get("/markets", {
            "limit": limit, "cursor": cursor, "series_ticker": series_ticker,
            "event_ticker": event_ticker, "status": status,
            "tickers": ",".join(tickers) if tickers else None,
        })
        out.extend(page.get("markets", []))
        cursor = page.get("cursor")
        if not cursor:
            break
    return out


def get_market(ticker: str) -> dict:
    """GET /markets/{ticker} -> the market object (status, result, times, prices)."""
    return _get(f"/markets/{ticker}").get("market", {})


def get_trades(ticker: str, *, min_ts: int | None = None,
               max_ts: int | None = None, limit: int = 1000,
               max_pages: int = 100) -> list[dict]:
    """GET /markets/trades for one ticker; returns trades oldest-first."""
    out: list[dict] = []
    cursor = None
    for _ in range(max_pages):
        page = _get("/markets/trades", {
            "ticker": ticker, "limit": limit, "cursor": cursor,
            "min_ts": min_ts, "max_ts": max_ts,
        })
        out.extend(page.get("trades", []))
        cursor = page.get("cursor")
        if not cursor:
            break
    out.sort(key=lambda t: parse_time(t["created_time"]))
    return out


def get_candlesticks(series_ticker: str, ticker: str, *, start_ts: int,
                     end_ts: int, period_interval: int = 60) -> list[dict]:
    """GET /series/{series}/markets/{ticker}/candlesticks, chunked to the
    5000-candle request cap. period_interval is in minutes (1, 60, 1440)."""
    if period_interval not in VALID_INTERVALS_MIN:
        raise ValueError(f"period_interval must be one of {VALID_INTERVALS_MIN}")
    step = period_interval * 60
    chunk = MAX_CANDLES_PER_REQUEST * step
    out: list[dict] = []
    lo = int(start_ts)
    while lo < end_ts:
        hi = min(lo + chunk, int(end_ts))
        page = _get(f"/series/{series_ticker}/markets/{ticker}/candlesticks", {
            "start_ts": lo, "end_ts": hi, "period_interval": period_interval})
        out.extend(page.get("candlesticks", []))
        lo = hi
    out.sort(key=lambda c: int(c["end_period_ts"]))
    return out


# --------------------------------------------------------------------------
# Normalisation (pure functions; unit-tested on recorded fixtures)
# --------------------------------------------------------------------------

def series_from_ticker(ticker: str) -> str:
    """Kalshi tickers are SERIES-EVENT-MARKET; the series is the first field."""
    return ticker.split("-")[0]


def parse_time(value: str | int | float) -> int:
    """RFC3339 (Z or offset, optional fractional seconds) or epoch -> epoch s."""
    if isinstance(value, (int, float)):
        return int(value)
    text = value.strip()
    if text.isdigit():
        return int(text)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:  # trim fractional seconds beyond microseconds
        head, tail = text.split(".", 1)
        frac, offset = tail, ""
        for sign in ("+", "-"):
            if sign in tail:
                frac, rest = tail.split(sign, 1)
                offset = sign + rest
                break
        text = f"{head}.{frac[:6]}{offset}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def price_to_prob(value) -> float | None:
    """Cents (int 1-99), a dollar string ("0.4200"), or a float already in
    [0, 1] -> probability. None / empty stays None (no trade in the period)."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, str):
        p = float(value)
    elif isinstance(value, int) or (isinstance(value, float) and value > 1.0):
        p = value / 100.0
    else:
        p = float(value)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"price {value!r} is not a probability")
    return p


def _leaf(obj, key: str):
    """Return obj[key], preferring a *_dollars sibling when present."""
    if not isinstance(obj, dict):
        return None
    dollars = obj.get(f"{key}_dollars")
    if dollars not in (None, ""):
        return dollars
    return obj.get(key)


def candle_prob(candle: dict) -> float | None:
    """One candle -> probability: last trade close, else bid/ask midpoint,
    else the previous close Kalshi carries in. None if nothing is known."""
    price = candle.get("price") or {}
    close = price_to_prob(_leaf(price, "close"))
    if close is not None:
        return close
    bid = price_to_prob(_leaf(candle.get("yes_bid") or {}, "close"))
    ask = price_to_prob(_leaf(candle.get("yes_ask") or {}, "close"))
    if bid is not None and ask is not None and bid > 0.0 and ask > 0.0:
        return 0.5 * (bid + ask)
    return price_to_prob(_leaf(price, "previous"))


def candles_to_path(candles: Iterable[dict]) -> list[tuple[int, float]]:
    """Candles -> [(end_ts, prob)], forward-filling periods with no
    information. Leading unknown periods are dropped (no truth to replay)."""
    rows: list[tuple[int, float]] = []
    last: float | None = None
    for candle in sorted(candles, key=lambda c: int(c["end_period_ts"])):
        p = candle_prob(candle)
        if p is None:
            p = last
        if p is None:
            continue
        last = p
        rows.append((int(candle["end_period_ts"]), p))
    return rows


def trades_to_path(trades: Iterable[dict], interval_s: int) -> list[tuple[int, float]]:
    """Trades -> a regular grid of `interval_s` seconds, last trade carried
    forward. Every trade is a yes-price observation whatever the taker side.
    Row at grid time t holds the last trade strictly before t."""
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    obs = sorted((parse_time(t["created_time"]), price_to_prob(t["yes_price"]))
                 for t in trades)
    obs = [(ts, p) for ts, p in obs if p is not None]
    if not obs:
        return []
    # Grid times are the first multiple of interval_s after the first trade,
    # then every interval_s until one grid time lies past the last trade, so
    # the final row carries the last trade too.
    first = obs[0][0] - (obs[0][0] % interval_s) + interval_s
    last_ts = obs[-1][0]
    rows: list[tuple[int, float]] = []
    i = 0
    current = obs[0][1]
    t = first
    while t - interval_s <= last_ts:
        while i < len(obs) and obs[i][0] < t:
            current = obs[i][1]
            i += 1
        rows.append((t, current))
        t += interval_s
    return rows


def path_stats(probs: Sequence[float]) -> dict:
    """Per-step statistics for calibrating the synthetic processes against a
    real path: additive vol (sd of dp), martingale vol (sd of dp / p(1-p)),
    jump rate (share of |dp| beyond 3 sd), and time spent at the bounds."""
    n = len(probs)
    if n < 3:
        return {"steps": n}
    dp = [probs[i] - probs[i - 1] for i in range(1, n)]
    scaled = [d / max(probs[i - 1] * (1.0 - probs[i - 1]), 1e-9)
              for i, d in enumerate(dp, start=1)]
    sd = statistics.pstdev(dp)
    sd_scaled = statistics.pstdev(scaled)
    moves = sum(1 for d in dp if d != 0.0)
    jumps = sum(1 for d in dp if sd > 0 and abs(d) > 3.0 * sd)
    return {
        "steps": n,
        "p_first": probs[0], "p_last": probs[-1],
        "p_min": min(probs), "p_max": max(probs),
        "additive_vol": sd,
        "martingale_vol": sd_scaled,
        "jump_rate": jumps / len(dp),
        "move_share": moves / len(dp),
        "at_bounds_share": sum(1 for p in probs if p <= 0.01 or p >= 0.99) / n,
        "synthetic_additive_vol": SYNTHETIC_ADDITIVE_VOL,
        "synthetic_martingale_vol": SYNTHETIC_MARTINGALE_VOL,
        "synthetic_jump_rate": SYNTHETIC_JUMP_RATE,
    }


def market_outcome(market: dict) -> str | None:
    """'yes' / 'no' for a settled market, else None (draw at settlement)."""
    result = (market.get("result") or "").lower()
    return result if result in ("yes", "no") else None


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------

def path_file(ticker: str, out_dir: Path = DATA_DIR) -> Path:
    return Path(out_dir) / f"{ticker}_path.csv"


def meta_file(ticker: str, out_dir: Path = DATA_DIR) -> Path:
    return Path(out_dir) / f"{ticker}_meta.json"


def write_path(rows: Sequence[tuple[int, float]], ticker: str,
               out_dir: Path = DATA_DIR) -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = path_file(ticker, out_dir)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_step", "timestamp", "latent_probability"])
        for step, (ts, p) in enumerate(rows, start=1):
            writer.writerow([step, ts, f"{p:.6f}"])
    return out


def read_path(file: Path) -> list[float]:
    with Path(file).open() as fh:
        return [float(row["latent_probability"]) for row in csv.DictReader(fh)]


def write_meta(market: dict, ticker: str, source: str, interval_s: int,
               n_rows: int, out_dir: Path = DATA_DIR) -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    meta = {
        "ticker": ticker,
        "series_ticker": series_from_ticker(ticker),
        "event_ticker": market.get("event_ticker"),
        "title": market.get("title"),
        "subtitle": market.get("yes_sub_title") or market.get("subtitle"),
        "status": market.get("status"),
        "result": market_outcome(market),
        "open_time": market.get("open_time"),
        "close_time": market.get("close_time"),
        "expiration_time": market.get("expiration_time"),
        "volume": market.get("volume"),
        "source": source,
        "interval_s": interval_s,
        "rows": n_rows,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = meta_file(ticker, out_dir)
    out.write_text(json.dumps(meta, indent=2) + "\n")
    return out


def read_meta(file: Path) -> dict:
    return json.loads(Path(file).read_text())


def discover_paths(data_dir: Path = DATA_DIR) -> list[tuple[Path, dict]]:
    """All fetched (path csv, meta) pairs; meta falls back to the ticker."""
    out = []
    for csv_file in sorted(Path(data_dir).glob("*_path.csv")):
        ticker = csv_file.name[: -len("_path.csv")]
        meta = meta_file(ticker, Path(data_dir))
        out.append((csv_file, read_meta(meta) if meta.exists() else {"ticker": ticker}))
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def fetch(ticker: str, *, source: str = "candles", interval_min: int = 60,
          start: str | None = None, end: str | None = None,
          out_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    market = get_market(ticker)
    if not market:
        raise KalshiError(f"market not found: {ticker}")
    raw_dir = Path(out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{ticker}_market.json").write_text(json.dumps(market, indent=2))

    now = int(time.time())
    start_ts = parse_time(start) if start else parse_time(market["open_time"])
    end_ts = parse_time(end) if end else min(now, parse_time(market["close_time"]))
    if end_ts <= start_ts:
        raise KalshiError(f"empty window {start_ts}..{end_ts} for {ticker}")

    if source == "candles":
        candles = get_candlesticks(series_from_ticker(ticker), ticker,
                                   start_ts=start_ts, end_ts=end_ts,
                                   period_interval=interval_min)
        (raw_dir / f"{ticker}_candles_{interval_min}m.json").write_text(
            json.dumps(candles))
        rows = candles_to_path(candles)
    elif source == "trades":
        trades = get_trades(ticker, min_ts=start_ts, max_ts=end_ts)
        (raw_dir / f"{ticker}_trades.json").write_text(json.dumps(trades))
        rows = trades_to_path(trades, interval_min * 60)
    else:
        raise ValueError("source must be candles or trades")
    if not rows:
        raise KalshiError(f"no price observations for {ticker} in window")

    path = write_path(rows, ticker, out_dir)
    meta = write_meta(market, ticker, source, interval_min * 60, len(rows), out_dir)
    return path, meta


def format_stats(stats: dict) -> str:
    if stats.get("steps", 0) < 3:
        return f"  steps={stats.get('steps', 0)} (too short for statistics)"
    return (
        f"  steps={stats['steps']}  p: {stats['p_first']:.2f} -> {stats['p_last']:.2f}"
        f"  [{stats['p_min']:.2f}, {stats['p_max']:.2f}]\n"
        f"  additive vol   {stats['additive_vol']:.4f}  (synthetic {stats['synthetic_additive_vol']:.4f})\n"
        f"  martingale vol {stats['martingale_vol']:.4f}  (synthetic {stats['synthetic_martingale_vol']:.4f})\n"
        f"  jump rate      {stats['jump_rate']:.4f}  (synthetic {stats['synthetic_jump_rate']:.4f})\n"
        f"  steps with a move {stats['move_share']:.2%}, at bounds {stats['at_bounds_share']:.2%}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list markets (public)")
    p_list.add_argument("--series", help="series ticker, e.g. KXBTCD")
    p_list.add_argument("--event", help="event ticker")
    p_list.add_argument("--status", choices=["open", "closed", "settled", "unopened"])
    p_list.add_argument("--limit", type=int, default=50)
    p_list.add_argument("--pages", type=int, default=1)

    p_fetch = sub.add_parser("fetch", help="download one market's price path")
    p_fetch.add_argument("--ticker", required=True)
    p_fetch.add_argument("--source", choices=["candles", "trades"], default="candles")
    p_fetch.add_argument("--interval", type=int, default=60,
                         help="minutes per step (candles: 1, 60 or 1440)")
    p_fetch.add_argument("--start", help="RFC3339 or epoch; default market open")
    p_fetch.add_argument("--end", help="RFC3339 or epoch; default min(now, close)")
    p_fetch.add_argument("--out-dir", type=Path, default=DATA_DIR)

    p_stats = sub.add_parser("stats", help="statistics of fetched paths vs synthetic")
    p_stats.add_argument("--data-dir", type=Path, default=DATA_DIR)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "list":
            markets = list_markets(series_ticker=args.series, event_ticker=args.event,
                                   status=args.status, limit=args.limit,
                                   max_pages=args.pages)
            print(f"{'ticker':<40} {'status':<8} {'result':<6} {'volume':>8}  title")
            for m in markets:
                print(f"{m.get('ticker', ''):<40} {m.get('status', ''):<8} "
                      f"{(market_outcome(m) or '-'):<6} {m.get('volume', 0):>8}  "
                      f"{m.get('title', '')}")
            print(f"{len(markets)} markets")
        elif args.cmd == "fetch":
            path, meta = fetch(args.ticker, source=args.source,
                               interval_min=args.interval, start=args.start,
                               end=args.end, out_dir=args.out_dir)
            info = read_meta(meta)
            print(f"wrote {path} ({info['rows']} rows) and {meta}")
            print(f"  {info['title']}  status={info['status']} result={info['result']}")
            print(format_stats(path_stats(read_path(path))))
        elif args.cmd == "stats":
            pairs = discover_paths(args.data_dir)
            if not pairs:
                print(f"no *_path.csv under {args.data_dir}; run `fetch` first")
                return 1
            for csv_file, meta in pairs:
                print(f"{meta.get('ticker')}  {meta.get('title', '')}  "
                      f"result={meta.get('result')}")
                print(format_stats(path_stats(read_path(csv_file))))
    except KalshiError as err:
        print(f"kalshi: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
