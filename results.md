# EventEdge — results log

Working notes for Tasks A–C. Every number carries N, σ, horizon, spread and a
standard error. Scripts that reproduce each figure live under `experiments/`
and are run from the repo root after `cmake --build build`.

## 0. Definitions as implemented (read before running anything)

All line numbers refer to the branch head at the time of writing.

**(1) How ε enters.** Additively in probability, then clamped:
`estimated_prob_ = clamp_probability(public_signal + config_.calibration_bias)`
(`src/market_maker.cpp:68`), with `clamp_probability(x) = std::clamp(x, 0.01, 0.99)`
(`src/market_maker.cpp:9`). The public signal is itself `clamp(p_t + N(0, 0.05))`
(`src/event_probability.cpp:90-93`). Quotes are
`bid = clamp(center − spread/2)`, `ask = clamp(center + spread/2)`
(`src/market_maker.cpp:85-86`), so **yes, quotes are clipped — at [0.01, 0.99], not
[0, 1]**. With spread 0.04 the bid clips whenever `center < 0.03` and the ask
whenever `center > 0.97`. The clip is symmetric under p → 1−p.

**(2) What informed traders know.** Neither the terminal outcome nor the next
move: the *current* latent level plus small noise,
`private_signal = clamp(p_t + N(0, σ_priv = 0.02))` (`src/event_probability.cpp:95-98`),
drawn at the same step the order is placed (`src/main.cpp`, step 3 of the loop). An
informed trader buys iff `private_signal − ask > 0.02` and sells iff
`bid − private_signal > 0.02` (`src/trader_agents.cpp:81-87`, `kMinEdge = 0.02` in
`include/types.hpp`), otherwise declines. Value traders (20% of arrivals) do the same
with an independent public-signal draw (σ = 0.05); noise traders (the remainder) trade
with probability 0.30 on a fair coin. There is no "next-move" or "terminal-outcome"
information mode in the code, so B2(e) cannot be run without adding one.

**(3) Settlement.** Bernoulli at the horizon, not absorbing:
`std::bernoulli_distribution outcome_dist(terminal_prob); event_outcome = outcome_dist(settlement_rng)`
(`src/main.cpp:364-367`) with `terminal_prob = p_T` after the last step; the path is
never absorbed at 0/1 (it cannot reach them: see the clamps below). A replayed Kalshi
market can force the outcome with `--outcome yes|no`. The settlement RNG is its own
stream (`seed*3+3`, `src/main.cpp:265`).

**(4) P&L decomposition and markout horizon.** Exact per-fill identity
(`python/pnl_decomposition.py`): with ΔI the MM's signed inventory change, `p` the fill
price, `mid` the quote midpoint at the fill, `p_t` the latent probability at the fill,
`Y` the outcome,
`terminal_pnl = Σ ΔI(mid − p)` [spread capture] `+ Σ ΔI(p_t − mid)` [adverse selection]
`+ Σ ΔI(Y − p_t)` [settlement]. The settlement term is further split
(`python/settlement_drift_check.py`) into `Σ ΔI(p_T − p_t)` [process drift, zero in
expectation iff the latent process is a martingale] and `I_T(Y − p_T)` [the settlement
draw, mean-zero by construction]. Fees are subtracted from terminal P&L and are zero
at defaults. Markouts (`python/markout_analysis.py:36`): `sign(ΔI) × (p_{t+h} − fill price)`
at **h ∈ {0, 10, 50} steps and at settlement** (`Y − fill price`). The summary also
carries `terminal_pnl_marked = cash + I_T·p_T − fees` = `E[terminal_pnl | path, fills]`.

**(5) Old clipped process.** Still present and still the **default**:
`--prob-process additive` is `ProbProcess::CLAMPED_ADDITIVE` (`include/types.hpp`),
`Δp = N(0, 0.01) + 1{U < 0.02}·N(0, 0.05)`, then `clamp(·, 0.01, 0.99)`
(`src/event_probability.cpp:33-40, 73-79`). The new process is
`--prob-process martingale` = `ProbProcess::LOGISTIC_MARTINGALE`:
`Δp = 0.042·p(1−p)·Z + 1{U < 0.02}·0.21·p(1−p)·Z'`, then a numerical safety clamp to
`[1e-6, 1 − 1e-6]` (`src/event_probability.cpp:14-15, 22-24, 67-72`). Both keep a
`clip_count`. Neither is modified here. Note the new process is not the pure diffusion
`dp = σ p(1−p) dW` of the task description: it has the same 2%-per-step jump mixture as
the old one, scaled by p(1−p). Defaults used throughout unless stated: horizon
T = 2000 steps, spread 0.04, σ_pub = 0.05, σ_priv = 0.02, informed share π = 0.10,
p₀ = 0.6 (overridden by `--p0`).

**Common random numbers.** Per-component streams derive from the seed
(`src/main.cpp:250-265`): latent process and signals `seed·3+1`, trader arrivals
`seed·3+2`, settlement `seed·3+3`. Per step the process stream is consumed by exactly
one `step()` and three signal draws, and the trader stream by one type draw plus two
noise-trader draws when a noise trader arrives — none of which depends on ε or on the
quotes. Verified on seed 11, p₀ = 0.2, π = 0.2, martingale, ε ∈ {−0.05, 0, +0.05}:
the 2000-step latent path, the terminal probability, the settlement outcome and the
370 noise-trader fills (step and side) are identical across ε. Informed and value
decisions depend on the quotes, as they must. Across the two *processes* the trader
and settlement streams are shared but the latent path differs by construction.

## 1. Pilot and choice of N

`experiments/pilot.py`: 50 seeds per cell on both processes, p₀ ∈ {0.2, 0.5, 0.8},
ε ∈ {−0.10, −0.02, 0, +0.02, +0.10}, π ∈ {0, 0.1, 0.4}, T = 2000, spread 0.04. Full
table in `results/pilot.md`. Per-run SD of P&L per contract at p₀ = 0.5:

| process | cell | SD settled P&L/contract | SD marked P&L/contract |
|---|---|---|---|
| martingale | ε = 0, π = 0.1 | 0.013 | 0.009 |
| martingale | ε = ±0.10, π = 0.1 | 0.18–0.19 | 0.10 |
| martingale | ε = ±0.10, π = 0.4 | 0.28–0.29 | 0.15 |
| additive | ε = ±0.10, π = 0.4 | 0.39 | 0.15 |

The SD grows with |ε| because a biased MM carries hundreds of contracts to settlement
and the P&L is then `inventory × (Y − average fill price)`: the settlement draw and the
path's net move dominate. Marking inventory at p_T removes the draw (exactly, it is
`E[settled | path]`) and roughly halves the SD; the path-move term remains.

N per cell needed for SE(mean P&L/contract) ≤ 5% of the effect (pooled SD over the
pilot cells of each process, settled / marked):

| effect (p₀ = 0.5) | size, martingale | N needed | size, additive | N needed |
|---|---|---|---|---|
| ε = ±0.02 vs 0 at π = 0.1 | −0.0010 / −0.0022 | 5.3 M / 380 k | −0.0011 / −0.0016 | 10.6 M / 910 k |
| ε = ±0.10 vs 0 at π = 0.1 | −0.032 / −0.033 | 5,200 / 1,600 | −0.033 / −0.035 | 12,900 / 2,000 |
| π = 0.4 vs 0 at ε = 0 | −0.025 / −0.026 | 8,900 / 2,600 | −0.023 / −0.025 | 26,400 / 4,000 |
| ε·π interaction (0.10 × 0.4 vs 0.1) | −0.041 / −0.021 | 3,100 / 4,100 | −0.020 / −0.019 | 34,600 / 6,800 |
| A6: process difference at ε = 0, π = 0.1 | −0.002 / −0.0007 | ≈ 10,000 / ≈ 2 M | | |

**Decision.** The ±0.02 effect cannot be resolved to 5% at any feasible N (millions of
runs per cell); it is reported with its SE and treated as "not resolved" where the SE
says so. For everything else, **N = 2,000 seeds per cell for Task B** (168 cells,
336,000 runs) gives SE(marked) ≈ 0.0015 and SE(settled) ≈ 0.0026 per contract at
p₀ = 0.5: 5–8% of the ε·π interaction (marked, both processes), 6% (settled,
martingale) and 13% (settled, additive) — the additive settled case misses the 5%
target and is flagged where it matters. Paired (common-random-number) contrasts
between ε levels have much smaller SEs than these two-sample figures, since the
latent path, arrivals and settlement draw cancel in the pair; those are reported as
paired SEs. **N = 10,000 seeds per cell for A6**, which resolves the settled
process difference to ≈ 5% and leaves the marked difference (an order of magnitude
smaller) unresolved. The adverse-selection component has a per-run SD of ~0.002 per
contract, so every contrast on it is resolved to well under 5% at these N.

