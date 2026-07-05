#include <iostream>

#include "event_probability.hpp"
#include "types.hpp"

int main() {
    SimConfig config;
    config.true_prob_init = 0.60;
    config.signal_noise_pub = 0.05;
    config.signal_noise_priv = 0.02;
    config.random_seed = 42;

    EventProbabilityProcess event_probability(config);
    std::cout << "EventEdge simulator initialized.\n";
    std::cout << "Initial true probability: " << event_probability.true_prob() << '\n';

    for (int t = 1; t <= 10; ++t) {
        event_probability.step();
        std::cout << "t=" << t
                  << ", true_prob=" << event_probability.true_prob()
                  << ", public_signal=" << event_probability.public_signal()
                  << ", private_signal=" << event_probability.private_signal() << '\n';
    }

    event_probability.inject_shock(0.10);
    std::cout << "After shock true probability: " << event_probability.true_prob() << '\n';

    return 0;
}
