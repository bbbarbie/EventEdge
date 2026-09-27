"""Tests for python/kalshi.py: normalisation of recorded Kalshi API shapes
into replay paths, and the round trip through the C++ binary's --prob-path.

Standard library only (unittest); no network. Fixtures mirror the public
trade-api/v2 response shapes: cents-integer prices, *_dollars string prices,
null closes in quiet candles, and newest-first trade pages.

Run:  python3 -m unittest tests.test_kalshi   (from the repo root)
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import kalshi  # noqa: E402
import kalshi_calibration  # noqa: E402

BINARY = REPO_ROOT / "eventedge"
if not BINARY.exists():
    BINARY = REPO_ROOT / "build" / "eventedge"


def candle(end_ts, close=None, bid=None, ask=None, previous=None, dollars=False):
    def leaf(v):
        if v is None:
            return {}
        return {"close_dollars": f"{v / 100:.4f}"} if dollars else {"close": v}
    price = leaf(close)
    if previous is not None:
        price["previous"] = previous
    return {"end_period_ts": end_ts, "price": price,
            "yes_bid": leaf(bid), "yes_ask": leaf(ask), "volume": 0}


class PriceConversion(unittest.TestCase):
    def test_cents_and_dollars_and_floats(self):
        self.assertEqual(kalshi.price_to_prob(42), 0.42)
        self.assertEqual(kalshi.price_to_prob("0.4200"), 0.42)
        self.assertEqual(kalshi.price_to_prob(0.42), 0.42)
        self.assertEqual(kalshi.price_to_prob(99), 0.99)
        self.assertEqual(kalshi.price_to_prob(0), 0.0)
        self.assertIsNone(kalshi.price_to_prob(None))
        self.assertIsNone(kalshi.price_to_prob(""))

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            kalshi.price_to_prob(150)
        with self.assertRaises(ValueError):
            kalshi.price_to_prob("1.5")

    def test_series_from_ticker(self):
        self.assertEqual(kalshi.series_from_ticker("KXBTCD-25SEP2717-T115999.99"), "KXBTCD")
        self.assertEqual(kalshi.series_from_ticker("PRES-2028"), "PRES")

    def test_parse_time_formats(self):
        self.assertEqual(kalshi.parse_time("1970-01-01T00:01:00Z"), 60)
        self.assertEqual(kalshi.parse_time("1970-01-01T00:01:00.123456789Z"), 60)
        self.assertEqual(kalshi.parse_time("1970-01-01T01:01:00+01:00"), 60)
        self.assertEqual(kalshi.parse_time(60), 60)
        self.assertEqual(kalshi.parse_time("60"), 60)

    def test_market_outcome(self):
        self.assertEqual(kalshi.market_outcome({"result": "yes"}), "yes")
        self.assertEqual(kalshi.market_outcome({"result": "NO"}), "no")
        self.assertIsNone(kalshi.market_outcome({"result": ""}))
        self.assertIsNone(kalshi.market_outcome({}))


class Candles(unittest.TestCase):
    def test_close_preferred_then_mid_then_previous(self):
        self.assertEqual(kalshi.candle_prob(candle(1, close=55, bid=50, ask=60)), 0.55)
        self.assertEqual(kalshi.candle_prob(candle(1, bid=50, ask=60)), 0.55)
        self.assertEqual(kalshi.candle_prob(candle(1, previous=48)), 0.48)
        self.assertIsNone(kalshi.candle_prob(candle(1)))
        # a zero bid/ask side means "no quote", not a price of 0
        self.assertIsNone(kalshi.candle_prob(candle(1, bid=0, ask=60)))

    def test_dollar_strings(self):
        self.assertAlmostEqual(kalshi.candle_prob(candle(1, close=37, dollars=True)), 0.37)

    def test_forward_fill_and_sort_and_leading_drop(self):
        candles = [candle(300), candle(200, close=61), candle(100), candle(400, bid=58, ask=62)]
        rows = kalshi.candles_to_path(candles)
        self.assertEqual(rows, [(200, 0.61), (300, 0.61), (400, 0.60)])

    def test_stats_recover_built_in_volatility(self):
        # A path with known step sd must be measured back; jump rate ~0 for a
        # pure Gaussian walk, and the p(1-p) scaling reproduces the ratio.
        import random
        rng = random.Random(7)
        p, probs = 0.5, [0.5]
        for _ in range(20000):
            p = min(0.99, max(0.01, p + rng.gauss(0.0, 0.004)))
            probs.append(p)
        stats = kalshi.path_stats(probs)
        self.assertAlmostEqual(stats["additive_vol"], 0.004, delta=0.0004)
        self.assertLess(stats["jump_rate"], 0.01)
        self.assertGreater(stats["martingale_vol"], stats["additive_vol"])
        self.assertEqual(kalshi.path_stats([0.5, 0.6]), {"steps": 2})


class Trades(unittest.TestCase):
    def test_grid_with_last_trade_carried_forward(self):
        trades = [  # newest-first, as the API pages them
            {"created_time": "1970-01-01T00:04:30Z", "yes_price": 70, "taker_side": "no"},
            {"created_time": "1970-01-01T00:01:10Z", "yes_price": 40, "taker_side": "yes"},
            {"created_time": "1970-01-01T00:00:30Z", "yes_price": 35, "taker_side": "yes"},
        ]
        rows = kalshi.trades_to_path(trades, 60)
        self.assertEqual([ts for ts, _ in rows], [60, 120, 180, 240, 300])
        self.assertEqual([p for _, p in rows], [0.35, 0.40, 0.40, 0.40, 0.70])

    def test_empty(self):
        self.assertEqual(kalshi.trades_to_path([], 60), [])
        with self.assertRaises(ValueError):
            kalshi.trades_to_path([], 0)


class Files(unittest.TestCase):
    def test_path_and_meta_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rows = [(60, 0.5), (120, 0.512345678)]
            path = kalshi.write_path(rows, "T-1", out)
            self.assertEqual(kalshi.read_path(path), [0.5, 0.512346])
            with path.open() as fh:
                header = next(csv.reader(fh))
            self.assertEqual(header, ["time_step", "timestamp", "latent_probability"])
            market = {"title": "x", "status": "settled", "result": "no",
                      "open_time": "1970-01-01T00:00:00Z"}
            kalshi.write_meta(market, "T-1", "candles", 60, 2, out)
            pairs = kalshi.discover_paths(out)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][1]["result"], "no")
            self.assertEqual(pairs[0][1]["series_ticker"], "T")


@unittest.skipUnless(BINARY.exists(), "build the simulator first (cmake --build build)")
class ReplayThroughBinary(unittest.TestCase):
    def run_binary(self, *args):
        return subprocess.run([str(BINARY), *args], capture_output=True, text=True)

    def test_replay_reproduces_path_and_forced_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rows = [(60 * i, p) for i, p in enumerate([0.30, 0.31, 0.29, 0.35, 0.40, 0.38], 1)]
            path = kalshi.write_path(rows, "T-1", out)
            for outcome, expect in (("yes", 1), ("no", 0)):
                result = self.run_binary("--prob-path", str(path), "--outcome", outcome,
                                         "--seed", "3", "--out-prefix", str(out / outcome))
                self.assertEqual(result.returncode, 0, result.stderr)
                with (out / f"{outcome}_steps.csv").open() as fh:
                    steps = list(csv.DictReader(fh))
                self.assertEqual([float(r["latent_probability"]) for r in steps],
                                 [p for _, p in rows])
                with (out / f"{outcome}_summary.csv").open() as fh:
                    summary = next(csv.DictReader(fh))
                self.assertEqual(int(summary["num_steps"]), len(rows))
                self.assertEqual(int(summary["event_outcome"]), expect)
                self.assertEqual(summary["outcome_source"], "forced")
                self.assertEqual(summary["prob_process"], "replay")
                self.assertAlmostEqual(float(summary["terminal_probability"]), 0.38)
                self.assertAlmostEqual(
                    float(summary["terminal_pnl_marked"]),
                    float(summary["terminal_cash"]) + int(summary["terminal_inventory"]) * 0.38,
                    places=12)
                # accounting identity still holds with a forced outcome
                self.assertAlmostEqual(
                    float(summary["terminal_pnl"]),
                    float(summary["terminal_cash"]) + int(summary["terminal_inventory"]) * expect,
                    places=12)

    def test_steps_beyond_path_is_an_error_and_draw_is_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            path = kalshi.write_path([(60, 0.5), (120, 0.5)], "T-2", out)
            bad = self.run_binary("--prob-path", str(path), "--steps", "5",
                                  "--out-prefix", str(out / "bad"))
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("exceeds", bad.stderr)
            ok = self.run_binary("--prob-path", str(path), "--out-prefix", str(out / "ok"))
            self.assertEqual(ok.returncode, 0, ok.stderr)
            with (out / "ok_summary.csv").open() as fh:
                self.assertEqual(next(csv.DictReader(fh))["outcome_source"], "draw")
            missing = self.run_binary("--prob-process", "replay", "--out-prefix", str(out / "m"))
            self.assertNotEqual(missing.returncode, 0)


class Calibration(unittest.TestCase):
    def test_observations_and_buckets(self):
        market = {"ticker": "KXT-1", "result": "yes", "volume": 500,
                  "open_time": "1970-01-10T00:00:00Z", "close_time": "1970-01-20T00:00:00Z"}
        close = kalshi.parse_time(market["close_time"])
        # hourly candles; price rises from 0.30 to 0.80 over the last 10 days
        candles = [candle(close - h * 3600, close=30 + (240 - h) * 50 // 240)
                   for h in range(240, -1, -1)]
        obs = kalshi_calibration.observe_market(market, candles)
        # 720h horizon predates open_time -> dropped; the other four remain
        self.assertEqual([o["hours_to_expiry"] for o in obs], [168, 24, 6, 1])
        self.assertEqual([o["band"] for o in obs], ["1–7d", "<1d", "<1d", "<1d"])
        self.assertTrue(all(o["yes"] == 1 for o in obs))
        self.assertLess(obs[0]["price"], obs[-1]["price"])
        self.assertEqual(kalshi_calibration.observe_market(dict(market, result=""), candles), [])

        # a bucket with 3 yes of 4 at mean price 0.5 has error +0.25
        rows = [{"series": "S", "band": "<1d", "price": 0.5, "yes": y} for y in (1, 1, 1, 0)]
        table = kalshi_calibration.calibration_table(rows)
        b = [r for r in table if r["series"] == "S" and r["band"] == "<1d" and r["n"] > 0]
        self.assertEqual(len(b), 1)
        self.assertAlmostEqual(b[0]["error"], 0.25)
        self.assertTrue(b[0]["ci_low"] < 0.75 < b[0]["ci_high"])
        pooled = [r for r in table if r["series"] == "ALL" and r["band"] == "ALL" and r["n"] > 0]
        self.assertEqual(pooled[0]["n"], 4)
        dist = kalshi_calibration.bias_distribution(table)
        self.assertAlmostEqual(dist[0]["mean_error"], 0.25)
        self.assertEqual(dist[0]["sd_error"], 0.0)

    def test_wilson_interval(self):
        lo, hi = kalshi_calibration.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.404, places=2)
        self.assertAlmostEqual(hi, 0.596, places=2)
        self.assertTrue(all(math.isnan(v) for v in kalshi_calibration.wilson(0, 0)))


@unittest.skipUnless(BINARY.exists(), "build the simulator first (cmake --build build)")
class FrictionFlagsThroughBinary(unittest.TestCase):
    def run_binary(self, out: Path, name: str, *args):
        result = subprocess.run([str(BINARY), "--seed", "5", "--steps", "1500",
                                 "--informed-fraction", "0.3", "--log-detail", "0",
                                 "--out-prefix", str(out / name), *args],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (out / f"{name}_summary.csv").open() as fh:
            return next(csv.DictReader(fh))

    def test_fees_queue_latency_and_gm_change_only_what_they_should(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            base = self.run_binary(out, "base")
            fee = self.run_binary(out, "fee", "--fee-rate", "0.07")
            # same fills, same cash; fees are the only difference
            self.assertEqual(fee["fill_count"], base["fill_count"])
            self.assertEqual(fee["terminal_cash"], base["terminal_cash"])
            self.assertGreater(float(fee["fees_paid"]), 0.0)
            self.assertAlmostEqual(float(base["terminal_pnl"]) - float(fee["terminal_pnl"]),
                                   float(fee["fees_paid"]), places=9)
            self.assertEqual(float(base["fees_paid"]), 0.0)

            queued = self.run_binary(out, "queued", "--queue-ahead", "1.0", "--informed-size", "3")
            self.assertGreater(int(queued["absorbed_count"]), 0)
            self.assertLess(int(queued["fill_count"]), int(base["fill_count"]))
            informed_share = lambda r: int(r["informed_fill_count"]) / int(r["fill_count"])
            self.assertGreater(informed_share(queued), informed_share(base))  # adverse fill

            slow = self.run_binary(out, "slow", "--quote-latency", "10")
            self.assertEqual(slow["quote_latency"], "10")
            self.assertNotEqual(slow["terminal_pnl"], base["terminal_pnl"])
            # (direction is a statistical statement; python/latency_sniping.py measures it)

            gm = self.run_binary(out, "gm", "--mm-strategy", "gm")
            self.assertEqual(gm["mm_strategy"], "gm")
            self.assertGreater(int(gm["fill_count"]), 0)

            bad = subprocess.run([str(BINARY), "--queue-ahead", "-1", "--out-prefix", str(out / "x")],
                                 capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)


class _FakeKalshi(BaseHTTPRequestHandler):
    """Serves the public endpoints with the real response shapes, including
    cursor pagination and one transient 500, so the client's paging, retry
    and end-to-end `fetch` are exercised without touching the network."""

    market = {
        "ticker": "KXTEST-26JAN01-T1", "event_ticker": "KXTEST-26JAN01",
        "title": "Test market", "status": "settled", "result": "yes",
        "open_time": "1970-01-01T00:00:00Z", "close_time": "1970-01-01T01:00:00Z",
        "expiration_time": "1970-01-01T02:00:00Z", "volume": 123,
        "yes_bid": 60, "yes_ask": 62, "last_price": 61,
    }
    failures_left = 1
    requests: list[str] = []

    def log_message(self, *_):  # silence
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        type(self).requests.append(self.path)
        if type(self).failures_left > 0:
            type(self).failures_left -= 1
            return self._send(500, {"error": "transient"})
        if url.path.endswith("/markets/KXTEST-26JAN01-T1"):
            return self._send(200, {"market": self.market})
        if url.path.endswith("/markets"):
            page = q.get("cursor")
            if page is None:
                return self._send(200, {"markets": [self.market], "cursor": "p2"})
            return self._send(200, {"markets": [dict(self.market, ticker="X-2")], "cursor": ""})
        if url.path.endswith("/candlesticks"):
            assert q["period_interval"] == "1"
            lo, hi = int(q["start_ts"]), int(q["end_ts"])
            candles = [{"end_period_ts": t, "price": {"close": 50 + (t // 60) % 5},
                        "yes_bid": {"close": 49}, "yes_ask": {"close": 52}, "volume": 1}
                       for t in range(lo + 60, hi + 1, 60)]
            return self._send(200, {"ticker": q.get("ticker"), "candlesticks": candles})
        if url.path.endswith("/markets/trades"):
            if q.get("cursor") is None:
                trades = [{"created_time": "1970-01-01T00:10:00Z", "yes_price": 70,
                           "taker_side": "yes", "count": 1}]
                return self._send(200, {"trades": trades, "cursor": "more"})
            trades = [{"created_time": "1970-01-01T00:02:00Z", "yes_price": 40,
                       "taker_side": "no", "count": 2}]
            return self._send(200, {"trades": trades, "cursor": ""})
        return self._send(404, {"error": "not found"})


class ClientAgainstFakeServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _FakeKalshi)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.old_base = kalshi.BASE_URL
        kalshi.BASE_URL = f"http://127.0.0.1:{cls.server.server_port}/trade-api/v2"
        cls.old_sleep = kalshi.time.sleep
        kalshi.time.sleep = lambda _s: None  # retries without waiting

    @classmethod
    def tearDownClass(cls):
        kalshi.BASE_URL = cls.old_base
        kalshi.time.sleep = cls.old_sleep
        cls.server.shutdown()
        cls.server.server_close()

    def test_pagination_and_retry(self):
        _FakeKalshi.failures_left = 1
        markets = kalshi.list_markets(status="settled", max_pages=5)
        self.assertEqual([m["ticker"] for m in markets], ["KXTEST-26JAN01-T1", "X-2"])
        trades = kalshi.get_trades("KXTEST-26JAN01-T1")
        self.assertEqual([t["yes_price"] for t in trades], [40, 70])  # oldest first

    def test_candlesticks_are_chunked_at_request_cap(self):
        _FakeKalshi.failures_left = 0
        _FakeKalshi.requests.clear()
        n = kalshi.MAX_CANDLES_PER_REQUEST + 10
        candles = kalshi.get_candlesticks("KXTEST", "KXTEST-26JAN01-T1",
                                          start_ts=0, end_ts=n * 60, period_interval=1)
        self.assertEqual(len(candles), n)
        self.assertEqual(len([r for r in _FakeKalshi.requests if "candlesticks" in r]), 2)
        with self.assertRaises(ValueError):
            kalshi.get_candlesticks("S", "T", start_ts=0, end_ts=60, period_interval=5)

    def test_fetch_end_to_end(self):
        _FakeKalshi.failures_left = 0
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            path, meta = kalshi.fetch("KXTEST-26JAN01-T1", source="candles",
                                      interval_min=1, end="1970-01-01T00:05:00Z",
                                      out_dir=out)
            probs = kalshi.read_path(path)
            self.assertEqual(len(probs), 5)
            self.assertTrue(all(0.5 <= p <= 0.54 for p in probs))
            info = kalshi.read_meta(meta)
            self.assertEqual(info["result"], "yes")
            self.assertEqual(info["series_ticker"], "KXTEST")
            self.assertEqual(info["rows"], 5)
            self.assertTrue((out / "raw" / "KXTEST-26JAN01-T1_market.json").exists())

            path, _ = kalshi.fetch("KXTEST-26JAN01-T1", source="trades",
                                   interval_min=1, out_dir=out)
            probs = kalshi.read_path(path)
            self.assertEqual(probs[0], 0.40)
            self.assertEqual(probs[-1], 0.70)

    def test_missing_market_is_a_kalshi_error(self):
        _FakeKalshi.failures_left = 0
        with self.assertRaises(kalshi.KalshiError):
            kalshi.get_market("NOPE-1")


if __name__ == "__main__":
    unittest.main()
