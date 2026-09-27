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

enum class ProbProcess {
    // Gaussian random walk clamped to [0.01, 0.99]. Simple, but clamping
    // induces drift toward the interior (not a martingale near bounds).
    CLAMPED_ADDITIVE,
    // State-dependent volatility: dp = vol * p(1-p) * Z. A true martingale
    // that stays in (0, 1) naturally, so settlement risk is mean-zero.
    LOGISTIC_MARTINGALE,
    // Replay an externally supplied probability path (e.g. a Kalshi market's
    // traded price series, see python/kalshi.py). No synthetic dynamics: the
    // path is the latent truth, step t reads row t-1.
    REPLAY
};

enum class MMStrategy {
    FIXED_SPREAD,
    INVENTORY_AWARE,
    ADAPTIVE_SPREAD,
    // Glosten-Milgrom (1985) Bayesian quoter: keeps a posterior over the
    // latent probability, sets bid/ask at the regret-free prices
    // E[p | sell] / E[p | buy] under the trader model, and updates on the
    // observed flow (a no-trade step is informative too).
    GLOSTEN_MILGROM
};

// Trader-population constants shared by the agents and by the GM quoter,
// which must assume the same population to be the theory-optimal benchmark.
constexpr double kValueTraderProbability = 0.20;
constexpr double kNoiseTradeProbability = 0.30;
constexpr double kMinEdge = 0.02;

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
    double informed_fraction = 0.10;
    double inventory_aversion = 0.0;  // quote-center shift per contract of inventory
    double initial_cash = 0.0;
    int initial_inventory = 0;
    int quote_size = 1;
    int num_steps = 0;
    std::uint32_t random_seed = 42;
    MMStrategy mm_strategy = MMStrategy::FIXED_SPREAD;
    ProbProcess prob_process = ProbProcess::CLAMPED_ADDITIVE;

    // Exchange fee per contract, Kalshi-shaped: fee_rate * p * (1 - p) at the
    // fill price, rounded up to the next cent per fill when fee_round_cents.
    // Kalshi's published taker rate is 0.07 and its maker rate 0.0175 (on
    // the series that charge makers at all); 0 keeps the frictionless model.
    double fee_rate = 0.0;
    bool fee_round_cents = true;

    // Fill model. queue_ahead is the mean number of contracts from other
    // liquidity providers ahead of the MM at its price level (Poisson each
    // step); an order fills the MM only with the size left after the queue.
    // Informed traders send informed_size contracts, so with a queue they
    // reach the MM more often than unit-size noise flow does.
    double queue_ahead = 0.0;
    int informed_size = 1;

    // Quote latency in steps: the MM quotes off the public signal it
    // observed quote_latency steps ago.
    int quote_latency = 0;

    // Glosten-Milgrom quoter's model of the world (defaults match the
    // additive process); gm_informed < 0 means "assume the true fraction".
    double gm_vol = 0.01;
    double gm_jump_prob = 0.02;
    double gm_jump_vol = 0.05;
    double gm_informed = -1.0;
    double gm_markup = 0.0;  // extra half-spread on top of the regret-free quotes
};
