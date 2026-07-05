#include "event_probability.hpp"

#include <algorithm>
#include <random>

namespace {

double clamp_probability(double value) {
    return std::clamp(value, 0.01, 0.99);
}

}  // namespace

EventProbabilityProcess::EventProbabilityProcess(const SimConfig& config)
    : p_true_(clamp_probability(config.true_prob_init)),
      signal_noise_pub_(config.signal_noise_pub),
      signal_noise_priv_(config.signal_noise_priv),
      rng_(config.random_seed),
      drift_dist_(0.0, 0.01),
      shock_dist_(0.0, 0.05),
      shock_event_dist_(0.02) {}

void EventProbabilityProcess::step() {
    double next_prob = p_true_ + drift_dist_(rng_);

    if (shock_event_dist_(rng_)) {
        next_prob += shock_dist_(rng_);
    }

    p_true_ = clamp_probability(next_prob);
}

double EventProbabilityProcess::public_signal() {
    std::normal_distribution<double> signal_dist(0.0, signal_noise_pub_);
    return clamp_probability(p_true_ + signal_dist(rng_));
}

double EventProbabilityProcess::private_signal() {
    std::normal_distribution<double> signal_dist(0.0, signal_noise_priv_);
    return clamp_probability(p_true_ + signal_dist(rng_));
}

double EventProbabilityProcess::true_prob() const {
    return p_true_;
}

void EventProbabilityProcess::inject_shock(double magnitude) {
    p_true_ = clamp_probability(p_true_ + magnitude);
}
