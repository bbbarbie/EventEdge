#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "types.hpp"

class EventProbabilityProcess {
public:
    explicit EventProbabilityProcess(const SimConfig& config);

    void step();
    double public_signal();
    double private_signal();
    double true_prob() const;
    void inject_shock(double magnitude);
    long clip_count() const;  // times the clamp actually bound (monitor per CLAUDE.md)

    // Replay mode (ProbProcess::REPLAY): each step() consumes the next value
    // of `path` as the latent probability; after the path is exhausted the
    // last value holds. Values are clamped to [0.01, 0.99] like the additive
    // process (Kalshi prices are already 1-99 cents, so this never binds on
    // real data). Signals are still drawn around the replayed value.
    void set_replay_path(std::vector<double> path);
    std::size_t replay_length() const;
    std::size_t replay_position() const;

private:
    double p_true_ = 0.0;
    ProbProcess process_type_ = ProbProcess::CLAMPED_ADDITIVE;
    long clip_count_ = 0;
    std::vector<double> replay_path_;
    std::size_t replay_index_ = 0;
    double signal_noise_pub_ = 0.0;
    double signal_noise_priv_ = 0.0;
    std::mt19937 rng_;
    std::normal_distribution<double> drift_dist_;
    std::normal_distribution<double> shock_dist_;
    std::bernoulli_distribution shock_event_dist_;
    std::normal_distribution<double> unit_normal_;
};
