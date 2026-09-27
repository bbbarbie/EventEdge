"""Shared per-run statistics for the experiments: exact P&L decomposition,
fills by side and trader type, markouts at the repo's horizons, and quote
clipping counts. Reads the simulator's _summary/_fills/_steps CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MARKOUT_HORIZONS = [0, 10, 50]
TYPES = ["INFORMED", "VALUE", "NOISE"]
CLIP_LO, CLIP_HI = 0.01, 0.99


def per_run_stats(prefix: Path) -> dict:
    summary = pd.read_csv(f"{prefix}_summary.csv").iloc[0]
    fills = pd.read_csv(f"{prefix}_fills.csv")
    steps = pd.read_csv(f"{prefix}_steps.csv")
    outcome = float(summary["event_outcome"])
    p_terminal = float(summary["terminal_probability"])
    contracts = int(fills["quantity"].sum()) if len(fills) else 0
    row = {
        "seed": int(summary["seed"]),
        "terminal_pnl": float(summary["terminal_pnl"]),
        "terminal_pnl_marked": float(summary["terminal_pnl_marked"]),
        "contracts": contracts,
        "fills": int(len(fills)),
        "event_outcome": int(outcome),
        "terminal_probability": p_terminal,
        "terminal_inventory": int(summary["terminal_inventory"]),
        "steps": int(summary["num_steps"]),
        # quote clipping: either side pinned at the clamp
        "bid_clipped_steps": int((steps["bid"] <= CLIP_LO + 1e-12).sum()),
        "ask_clipped_steps": int((steps["ask"] >= CLIP_HI - 1e-12).sum()),
        "estimate_clipped_steps": int(((steps["mm_estimate"] <= CLIP_LO + 1e-12)
                                       | (steps["mm_estimate"] >= CLIP_HI - 1e-12)).sum()),
    }
    for side in ("BUY", "SELL"):
        row[f"fills_{side}"] = int((fills["side"] == side).sum()) if len(fills) else 0
    for t in TYPES:
        row[f"fills_{t}"] = int((fills["trader_type"] == t).sum()) if len(fills) else 0
    if fills.empty:
        row.update(spread_capture=0.0, adverse_selection=0.0, settlement_drift=0.0,
                   settlement_draw=0.0, identity_error=0.0)
        for t in TYPES:
            row[f"adverse_{t}"] = 0.0
            for h in MARKOUT_HORIZONS + ["settle"]:
                row[f"markout_{h}_{t}"] = np.nan
        return row
    d_inv = fills["mm_inventory_change"].astype(float)
    mid = (fills["bid"] + fills["ask"]) / 2.0
    p_t = fills["latent_probability"]
    price = fills["fill_price"]
    spread = d_inv * (mid - price)
    adverse = d_inv * (p_t - mid)
    row["spread_capture"] = float(spread.sum())
    row["adverse_selection"] = float(adverse.sum())
    row["settlement_drift"] = float((d_inv * (p_terminal - p_t)).sum())
    row["settlement_draw"] = float(summary["terminal_inventory"]) * (outcome - p_terminal)
    row["identity_error"] = abs(row["spread_capture"] + row["adverse_selection"]
                                + row["settlement_drift"] + row["settlement_draw"]
                                + float(summary["fees_paid"]) - row["terminal_pnl"])
    # markouts per contract: sign(dI) * (p_{t+h} - price)
    p_by_step = steps.set_index("time_step")["latent_probability"]
    sign = np.sign(d_inv)
    t = fills["time_step"].to_numpy()
    for h in MARKOUT_HORIZONS:
        ref = p_by_step.reindex(t + h).to_numpy()
        mo = sign * (ref - price)
        for typ in TYPES:
            m = (fills["trader_type"] == typ).to_numpy()
            row[f"markout_{h}_{typ}"] = float(np.nanmean(mo[m])) if m.any() else np.nan
    mo_settle = sign * (outcome - price)
    for typ in TYPES:
        m = (fills["trader_type"] == typ).to_numpy()
        row[f"markout_settle_{typ}"] = float(mo_settle[m].mean()) if m.any() else np.nan
        row[f"adverse_{typ}"] = float(adverse[m].sum())
    return row


def cache_or_compute(prefix: Path) -> dict:
    cached = Path(f"{prefix}_stats.json")
    if cached.exists():
        return json.loads(cached.read_text())
    row = per_run_stats(prefix)
    cached.write_text(json.dumps(row))
    for suffix in ("_fills.csv", "_steps.csv", "_summary.csv"):
        Path(f"{prefix}{suffix}").unlink(missing_ok=True)
    return row


def mean_se(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    return (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan)


def ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """OLS with cluster-robust (by group) standard errors."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    meat = np.zeros_like(XtX_inv)
    for g in np.unique(groups):
        m = groups == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    n, k = X.shape
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    V = adj * XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(V))
