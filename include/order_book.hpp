#pragma once

#include <cstdint>
#include <optional>
#include <random>

#include "types.hpp"

class OrderBook {
public:
    void update_quotes(Quote q);
    Quote best_quote() const;
    std::optional<Fill> match_order(const Order& order);

    // Queue-position model: mean contracts ahead of the MM at its level,
    // drawn Poisson per order. 0 (default) means the MM is alone at the top.
    void configure_queue(double mean_depth_ahead, std::uint32_t seed);
    long absorbed_count() const;  // orders fully eaten by the queue ahead

private:
    bool has_valid_quote() const;

    Quote current_quote_;
    double queue_ahead_ = 0.0;
    std::mt19937 queue_rng_;
    long absorbed_ = 0;
};
