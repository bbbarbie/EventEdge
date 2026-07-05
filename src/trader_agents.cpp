#include "trader_agents.hpp"

#include <optional>
#include <random>

namespace {

constexpr double kValueTraderProbability = 0.20;
constexpr double kNoiseTradeProbability = 0.30;
constexpr double kMinEdge = 0.02;

Order make_market_order(int id, int timestep, TraderType trader_type, Side side) {
    Order order;
    order.id = id;
    order.trader_type = trader_type;
    order.order_type = OrderType::MARKET;
    order.side = side;
    order.quantity = 1;
    order.timestamp = timestep;
    return order;
}

}  // namespace

TraderAgents::TraderAgents(const SimConfig& config)
    : config_(config),
      rng_(config.random_seed),
      order_id_counter_(0) {}

std::optional<Order> TraderAgents::generate_order(
    int timestep,
    double public_signal,
    double private_signal,
    const Quote& quote
) {
    std::uniform_real_distribution<double> trader_type_dist(0.0, 1.0);
    const double draw = trader_type_dist(rng_);

    if (draw < config_.informed_fraction) {
        return informed_trader_order(timestep, private_signal, quote);
    }

    if (draw < config_.informed_fraction + kValueTraderProbability) {
        return value_trader_order(timestep, public_signal, quote);
    }

    return noise_trader_order(timestep);
}

std::optional<Order> TraderAgents::noise_trader_order(int timestep) {
    std::bernoulli_distribution trade_dist(kNoiseTradeProbability);
    if (!trade_dist(rng_)) {
        return std::nullopt;
    }

    std::bernoulli_distribution side_dist(0.5);
    const Side side = side_dist(rng_) ? Side::BUY : Side::SELL;
    return make_market_order(++order_id_counter_, timestep, TraderType::NOISE, side);
}

std::optional<Order> TraderAgents::value_trader_order(
    int timestep,
    double public_signal,
    const Quote& quote
) {
    if (public_signal - quote.ask > kMinEdge) {
        return make_market_order(++order_id_counter_, timestep, TraderType::VALUE, Side::BUY);
    }

    if (quote.bid - public_signal > kMinEdge) {
        return make_market_order(++order_id_counter_, timestep, TraderType::VALUE, Side::SELL);
    }

    return std::nullopt;
}

std::optional<Order> TraderAgents::informed_trader_order(
    int timestep,
    double private_signal,
    const Quote& quote
) {
    if (private_signal - quote.ask > kMinEdge) {
        return make_market_order(++order_id_counter_, timestep, TraderType::INFORMED, Side::BUY);
    }

    if (quote.bid - private_signal > kMinEdge) {
        return make_market_order(++order_id_counter_, timestep, TraderType::INFORMED, Side::SELL);
    }

    return std::nullopt;
}
