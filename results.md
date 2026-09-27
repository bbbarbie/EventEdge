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

## 2. Task A — martingale validation and the cost of clipping

Script: `experiments/task_a_latent.py` (harness `build/latent_paths`, built from
`experiments/latent_paths.cpp` against `EventProbabilityProcess`; the harness refuses
to run unless its σ-scalable mirror reproduces the class bit-for-bit over 20 seeds ×
T steps). N = 200,000 paths per (process, p₀, config), seeds 1…N, stream `seed·3+1`
as in the simulator; settlement is one Bernoulli(p_T) draw per path from stream
`seed·3+3`. Horizon T = 2000 steps, σ at repo defaults, no trading. Figure:
`results/taskA_bias_vs_p0.png`; tables: `results/taskA_bias_table.csv`,
`results/taskA_conditional.csv`.

![Task A](results/taskA_bias_vs_p0.png)

#### A1. Bias at the default σ and horizon (pp)

N = 200,000 paths per cell, T = 2000 steps, σ = repo default (additive: 0.01/step + 2%×0.05 jumps, clamp [0.01, 0.99]; martingale: 0.042·p(1−p)/step + 2%×0.21·p(1−p) jumps, clamp [1e-6, 1−1e-6]).

| p₀ | clipped: E[p_T]−p₀ | clipped: settle freq−p₀ | p(1−p): E[p_T]−p₀ | p(1−p): settle freq−p₀ |
|---|---|---|---|---|
| 0.05 | +35.81 ± 0.06 | +35.72 ± 0.05 | -0.01 ± 0.03 | +0.03 ± 0.05 |
| 0.10 | +31.15 ± 0.06 | +31.07 ± 0.07 | -0.01 ± 0.05 | +0.01 ± 0.07 |
| 0.20 | +22.48 ± 0.06 | +22.35 ± 0.09 | -0.02 ± 0.07 | -0.03 ± 0.09 |
| 0.35 | +10.80 ± 0.06 | +10.69 ± 0.11 | -0.05 ± 0.08 | -0.02 ± 0.11 |
| 0.50 | +0.04 ± 0.06 | -0.05 ± 0.11 | -0.05 ± 0.09 | -0.03 ± 0.11 |
| 0.65 | -10.72 ± 0.06 | -10.79 ± 0.11 | -0.03 ± 0.08 | -0.09 ± 0.11 |
| 0.80 | -22.42 ± 0.06 | -22.38 ± 0.09 | -0.01 ± 0.07 | -0.05 ± 0.09 |
| 0.90 | -31.10 ± 0.06 | -31.06 ± 0.07 | -0.03 ± 0.05 | -0.02 ± 0.07 |
| 0.95 | -35.76 ± 0.06 | -35.75 ± 0.05 | -0.04 ± 0.03 | -0.02 ± 0.05 |

#### A2. Max |bias| over the p₀ grid (pp)

| process | max |E[p_T]−p₀| | at p₀ | max |settle−p₀| | at p₀ |
|---|---|---|---|---|
| additive | 35.81 ± 0.06 | 0.05 | 35.75 ± 0.05 | 0.95 |
| martingale | 0.05 ± 0.09 | 0.50 | 0.09 ± 0.11 | 0.65 |

#### A3. Conditional martingale check, E[p_T | p_{T/2}] − p_{T/2} (pp), pooled over the p₀ grid

| p_{T/2} bin | n (clipped) | clipped | n (p(1−p)) | p(1−p) |
|---|---|---|---|---|
| [0.0, 0.05) | 92,615 | +28.00 ± 0.07 | 404,232 | +0.01 ± 0.01 |
| [0.05, 0.1) | 92,118 | +23.28 ± 0.07 | 129,153 | +0.01 ± 0.04 |
| [0.1, 0.15) | 91,781 | +19.38 ± 0.08 | 80,392 | +0.03 ± 0.06 |
| [0.15, 0.2) | 90,908 | +15.88 ± 0.08 | 59,487 | -0.02 ± 0.09 |
| [0.2, 0.25) | 90,327 | +12.63 ± 0.08 | 48,132 | -0.03 ± 0.12 |
| [0.25, 0.3) | 89,486 | +9.96 ± 0.09 | 41,676 | +0.15 ± 0.14 |
| [0.3, 0.35) | 89,129 | +7.58 ± 0.09 | 37,627 | +0.32 ± 0.15 |
| [0.35, 0.4) | 88,748 | +4.95 ± 0.09 | 34,787 | -0.19 ± 0.17 |
| [0.4, 0.45) | 88,738 | +3.15 ± 0.09 | 33,133 | -0.15 ± 0.18 |
| [0.45, 0.5) | 87,733 | +1.09 ± 0.09 | 32,204 | +0.04 ± 0.18 |
| [0.5, 0.55) | 87,583 | -0.89 ± 0.09 | 32,233 | +0.36 ± 0.18 |
| [0.55, 0.6) | 88,102 | -2.83 ± 0.09 | 33,054 | +0.04 ± 0.18 |
| [0.6, 0.65) | 88,739 | -5.10 ± 0.09 | 34,898 | -0.02 ± 0.17 |
| [0.65, 0.7) | 89,289 | -7.26 ± 0.09 | 37,681 | +0.04 ± 0.15 |
| [0.7, 0.75) | 89,914 | -9.49 ± 0.09 | 41,496 | +0.07 ± 0.14 |
| [0.75, 0.8) | 89,672 | -12.52 ± 0.08 | 48,299 | +0.04 ± 0.12 |
| [0.8, 0.85) | 90,536 | -15.96 ± 0.08 | 59,382 | -0.02 ± 0.09 |
| [0.85, 0.9) | 90,941 | -19.38 ± 0.08 | 80,416 | +0.08 ± 0.06 |
| [0.9, 0.95) | 90,340 | -23.43 ± 0.08 | 129,212 | -0.03 ± 0.04 |
| [0.95, 1.0) | 93,301 | -27.91 ± 0.07 | 402,506 | -0.00 ± 0.01 |

#### A4. Boundary accounting (default config)

| p₀ | clipped: paths clipped ≥1 | clipped: mean clips/path | p(1−p): Euler steps outside [0,1] | p(1−p): paths with any |
|---|---|---|---|---|
| 0.05 | 95.4% | 48.3 | 8 | 0.004% |
| 0.10 | 90.9% | 43.4 | 5 | 0.003% |
| 0.20 | 82.8% | 35.3 | 1 | 0.001% |
| 0.35 | 73.9% | 27.6 | 2 | 0.001% |
| 0.50 | 70.6% | 25.0 | 1 | 0.001% |
| 0.65 | 73.9% | 27.6 | 1 | 0.001% |
| 0.80 | 82.8% | 35.3 | 1 | 0.001% |
| 0.90 | 91.0% | 43.4 | 4 | 0.002% |
| 0.95 | 95.5% | 48.3 | 6 | 0.003% |

#### A5. Sensitivity: max |E[p_T]−p₀| over the p₀ grid (pp)

| config | σ·√T relative | clipped max |bias| (at p₀) | p(1−p) max |bias| (at p₀) |
|---|---|---|---|
| default | 1.00 | 35.81 ± 0.06 (0.05) | 0.05 ± 0.09 (0.50) |
| sigma x0.5 | 0.50 | 17.60 ± 0.04 (0.95) | 0.02 ± 0.04 (0.80) |
| sigma x2 | 2.00 | 44.93 ± 0.07 (0.05) | 0.11 ± 0.11 (0.65) |
| T x0.5 | 0.71 | 25.87 ± 0.05 (0.95) | 0.08 ± 0.07 (0.35) |
| T x2 | 1.41 | 42.99 ± 0.06 (0.05) | 0.09 ± 0.10 (0.50) |

**Reading.**

- **A1/A2 (headline).** The clipped process is biased by up to **35.8 ± 0.06 pp**
  (p₀ = 0.05; −35.8 at 0.95), and the bias is nearly linear in (0.5 − p₀): the
  reflecting clamp at [0.01, 0.99] pushes every path toward the interior, so
  E[p_T] ≈ 0.5·(1 − e^{−κT}) + p₀·e^{−κT}-like relaxation toward 0.5. At the repo
  default p₀ = 0.6 the bias is −7 pp (interpolating 0.5 → 0.65: −10.7). The p(1−p)
  process has **max |bias| 0.05 ± 0.09 pp** on E[p_T] and 0.09 ± 0.11 pp on the
  settlement frequency: zero within SE at every p₀. Settlement frequency and E[p_T]
  agree to within their SEs for both processes, as they must under Bernoulli(p_T)
  settlement.
- **A3.** The clipped process fails the conditional check in *every* bin, not only
  the boundary ones: +28.0 pp at p_{T/2} < 0.05, still +1.1 pp at [0.45, 0.5) and
  −0.9 at [0.5, 0.55), sign-antisymmetric about 0.5. That is because over the
  remaining 1000 steps (σ√T ≈ 0.32) most paths reach a clamp wherever they start.
  The p(1−p) process is flat: 20 bins, all within 2.1 SE of zero (largest +0.36 ± 0.18
  and +0.32 ± 0.15; with 20 bins one or two ~2 SE deviations are expected), and
  exactly zero to 0.01 pp in the two boundary bins that hold 40% of the mass.
- **A4.** 71–95% of clipped-process paths hit a clamp at least once, 25–48 times per
  path on average. The p(1−p) process left [0, 1] in 1–8 Euler steps out of 4×10⁸
  per cell (0.001–0.004% of paths, always in a jump step); it is then clamped to
  [10⁻⁶, 1 − 10⁻⁶]. That clamp binds ~10⁻⁸ of steps and moves p by < 10⁻⁶ when it
  does, so it cannot reintroduce measurable bias, and the A1/A3 numbers confirm
  none is visible at 0.05 pp resolution.
- **A5.** Clipping bias grows with σ·√T as predicted but saturates: relative
  σ√T of 0.5 / 0.71 / 1 / 1.41 / 2 gives max |bias| 17.6 / 25.9 / 35.8 / 43.0 /
  44.9 pp. The ceiling is 45 pp (E[p_T] → 0.5 from p₀ = 0.05 once every path has
  forgotten its start), so the growth is concave, not linear. The p(1−p) process
  stays ≤ 0.11 ± 0.11 pp in every configuration. Confirmed.

