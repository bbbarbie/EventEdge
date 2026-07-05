#pragma once

#include <optional>
#include <random>

#include "types.hpp"

class TraderAgents {
public:
    explicit TraderAgents(const SimConfig& config);

    std::optional<Order> generate_order(
        int timestep,
        double public_signal,
        double private_signal,
        const Quote& quote
    );

private:
    std::optional<Order> noise_trader_order(int timestep);
    std::optional<Order> value_trader_order(int timestep, double public_signal, const Quote& quote);
    std::optional<Order> informed_trader_order(int timestep, double private_signal, const Quote& quote);

    SimConfig config_;
    std::mt19937 rng_;
    int order_id_counter_ = 0;
};
