#include "event_probability.hpp"

#include <algorithm>
#include <random>

namespace {

double clamp_probability(double value) {
    return std::clamp(value, 0.01, 0.99);
}

// Martingale volatilities are scaled by p(1-p); chosen so that per-step and
// shock vol match the additive process (0.01 and 0.05) at p = 0.6 where
// p(1-p) ~= 0.24.
constexpr double kMartingaleVol = 0.042;
constexpr double kMartingaleShockVol = 0.21;

// The martingale process lives naturally in (0, 1): step size scales with
// p(1-p), so overshooting a bound needs a > 20-sigma draw. Only a numerical
// safety clamp is applied; the additive process keeps the tighter [0.01,
// 0.99] clamp (which is what induces its drift artifact near bounds).
double clamp_probability_martingale(double value) {
    return std::clamp(value, 1e-6, 1.0 - 1e-6);
}

}  // namespace

EventProbabilityProcess::EventProbabilityProcess(const SimConfig& config)
    : p_true_(clamp_probability(config.true_prob_init)),
      signal_noise_pub_(config.signal_noise_pub),
      signal_noise_priv_(config.signal_noise_priv),
      rng_(config.random_seed),
      drift_dist_(0.0, 0.01),
      shock_dist_(0.0, 0.05),
      shock_event_dist_(0.02),
      unit_normal_(0.0, 1.0) {
    process_type_ = config.prob_process;
}

void EventProbabilityProcess::step() {
    double next_prob = p_true_;

    if (process_type_ == ProbProcess::LOGISTIC_MARTINGALE) {
        const double scale = p_true_ * (1.0 - p_true_);
        next_prob += kMartingaleVol * scale * unit_normal_(rng_);
        if (shock_event_dist_(rng_)) {
            next_prob += kMartingaleShockVol * scale * unit_normal_(rng_);
        }
    } else {
        next_prob += drift_dist_(rng_);
        if (shock_event_dist_(rng_)) {
            next_prob += shock_dist_(rng_);
        }
    }

    const double clamped =
        process_type_ == ProbProcess::LOGISTIC_MARTINGALE
            ? clamp_probability_martingale(next_prob)
            : clamp_probability(next_prob);
    if (clamped != next_prob) {
        ++clip_count_;
    }
    p_true_ = clamped;
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

long EventProbabilityProcess::clip_count() const {
    return clip_count_;
}

void EventProbabilityProcess::inject_shock(double magnitude) {
    p_true_ = clamp_probability(p_true_ + magnitude);
}
