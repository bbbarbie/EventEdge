#pragma once

#include <random>

#include "types.hpp"

class EventProbabilityProcess {
public:
    explicit EventProbabilityProcess(const SimConfig& config);

    void step();
    double public_signal();
    double private_signal();
    double true_prob() const;
    void inject_shock(double magnitude);

private:
    double p_true_ = 0.0;
    double signal_noise_pub_ = 0.0;
    double signal_noise_priv_ = 0.0;
    std::mt19937 rng_;
    std::normal_distribution<double> drift_dist_;
    std::normal_distribution<double> shock_dist_;
    std::bernoulli_distribution shock_event_dist_;
};
