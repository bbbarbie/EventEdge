// Task A harness: simulate the repo's latent probability processes without
// trading and emit per-path statistics. Uses EventProbabilityProcess itself
// (the code under test) at scale 1; for the sigma-sensitivity runs it
// re-implements the two step equations with a volatility multiplier and
// verifies, at scale 1, that the re-implementation reproduces the class
// bit-for-bit on the same seed before any scaled run is accepted.
//
//   latent_paths --process additive|martingale --p0 X --paths N --steps T
//                [--sigma-scale S] [--seed-base K] --out FILE
//
// FILE: one row per path: p_mid, p_T, settled (Bernoulli(p_T) with the
// path's own settlement stream), clip_count, left_unit_interval.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "event_probability.hpp"
#include "types.hpp"

namespace {

// Mirror of src/event_probability.cpp with a volatility multiplier. Same
// distribution objects, same draw order, so scale == 1 is identical.
struct ScaledProcess {
    ProbProcess type;
    double scale;
    double p;
    long clips = 0;
    long left_unit = 0;
    std::mt19937 rng;
    std::normal_distribution<double> drift_dist{0.0, 0.01};
    std::normal_distribution<double> shock_dist{0.0, 0.05};
    std::bernoulli_distribution shock_event_dist{0.02};
    std::normal_distribution<double> unit_normal{0.0, 1.0};

    ScaledProcess(ProbProcess t, double s, double p0, std::uint32_t seed)
        : type(t), scale(s), p(std::clamp(p0, 0.01, 0.99)), rng(seed) {}

    void step() {
        double next = p;
        if (type == ProbProcess::LOGISTIC_MARTINGALE) {
            const double sc = p * (1.0 - p);
            next += scale * 0.042 * sc * unit_normal(rng);
            if (shock_event_dist(rng)) next += scale * 0.21 * sc * unit_normal(rng);
        } else {
            next += scale * drift_dist(rng);
            if (shock_event_dist(rng)) next += scale * shock_dist(rng);
        }
        if (next < 0.0 || next > 1.0) ++left_unit;
        const double clamped = type == ProbProcess::LOGISTIC_MARTINGALE
            ? std::clamp(next, 1e-6, 1.0 - 1e-6) : std::clamp(next, 0.01, 0.99);
        if (clamped != next) ++clips;
        p = clamped;
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::string process = "additive", out;
    double p0 = 0.5, sigma_scale = 1.0;
    long paths = 1000;
    int steps = 2000;
    std::uint32_t seed_base = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string f = argv[i], v = argv[i + 1];
        if (f == "--process") process = v;
        else if (f == "--p0") p0 = std::stod(v);
        else if (f == "--paths") paths = std::stol(v);
        else if (f == "--steps") steps = std::stoi(v);
        else if (f == "--sigma-scale") sigma_scale = std::stod(v);
        else if (f == "--seed-base") seed_base = static_cast<std::uint32_t>(std::stoul(v));
        else if (f == "--out") out = v;
        else { std::cerr << "unknown flag " << f << '\n'; return 1; }
    }
    const ProbProcess type = process == "martingale" ? ProbProcess::LOGISTIC_MARTINGALE
                                                     : ProbProcess::CLAMPED_ADDITIVE;

    // Self-check: the scaled mirror must equal the class at scale 1.
    for (std::uint32_t s = 0; s < 20; ++s) {
        SimConfig cfg;
        cfg.true_prob_init = p0;
        cfg.prob_process = type;
        cfg.random_seed = seed_base + s;
        EventProbabilityProcess ref(cfg);
        ScaledProcess mirror(type, 1.0, p0, seed_base + s);
        for (int t = 0; t < steps; ++t) {
            ref.step();
            mirror.step();
            if (ref.true_prob() != mirror.p) {
                std::cerr << "SELF-CHECK FAILED: mirror diverges from class at seed "
                          << seed_base + s << " step " << t << '\n';
                return 2;
            }
        }
        if (ref.clip_count() != mirror.clips) {
            std::cerr << "SELF-CHECK FAILED: clip counts differ\n";
            return 2;
        }
    }

    std::ofstream fh(out);
    if (!fh) { std::cerr << "cannot open " << out << '\n'; return 1; }
    fh << "p_mid,p_T,settled,clip_count,left_unit_interval\n";
    const int mid = steps / 2;
    for (long i = 0; i < paths; ++i) {
        const std::uint32_t seed = seed_base + static_cast<std::uint32_t>(i);
        ScaledProcess proc(type, sigma_scale, p0, seed * 3u + 1u);  // same stream as the simulator
        std::mt19937 settlement_rng(seed * 3u + 3u);
        double p_mid = 0.0;
        for (int t = 1; t <= steps; ++t) {
            proc.step();
            if (t == mid) p_mid = proc.p;
        }
        std::bernoulli_distribution outcome(proc.p);
        fh << p_mid << ',' << proc.p << ',' << (outcome(settlement_rng) ? 1 : 0) << ','
           << proc.clips << ',' << proc.left_unit << '\n';
    }
    return 0;
}
