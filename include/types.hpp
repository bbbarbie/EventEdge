#pragma once

#include <cstddef>
#include <cstdint>

enum class Side {
    BUY,
    SELL
};

enum class TraderType {
    NOISE,
    VALUE,
    INFORMED,
    NONE
};

enum class OrderType {
    MARKET,
    LIMIT
};

enum class MMStrategy {
    FIXED_SPREAD,
    INVENTORY_AWARE,
    ADAPTIVE_SPREAD
};

struct Order {
    std::size_t id = 0;
    TraderType trader_type = TraderType::NONE;
    OrderType order_type = OrderType::MARKET;
    Side side = Side::BUY;
    double price = 0.0;
    int quantity = 0;
    double timestamp = 0.0;
};

struct Fill {
    std::size_t order_id = 0;
    Side side = Side::BUY;
    double price = 0.0;
    int quantity = 0;
    int mm_inventory_change = 0;
    TraderType trader_type = TraderType::NONE;
    double timestamp = 0.0;
};

struct Quote {
    double bid = 0.0;
    double ask = 0.0;
    int bid_size = 0;
    int ask_size = 0;
    double timestamp = 0.0;
};

struct PnLComponents {
    double cash = 0.0;
    int inventory = 0;
    double marked_to_market = 0.0;
    double realized = 0.0;
    double fees = 0.0;
    double total = 0.0;
};

struct SimConfig {
    double true_prob_init = 0.6;
    double true_probability = 0.6;
    double base_spread = 0.04;
    double calibration_bias = 0.0;
    double transaction_cost = 0.0;
    double signal_noise_pub = 0.05;
    double signal_noise_priv = 0.02;
    double initial_cash = 0.0;
    int initial_inventory = 0;
    int quote_size = 1;
    int num_steps = 0;
    std::uint32_t random_seed = 42;
    MMStrategy mm_strategy = MMStrategy::FIXED_SPREAD;
};
