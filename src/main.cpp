#include <iostream>

#include "trader_agents.hpp"
#include "types.hpp"

namespace {

const char* trader_type_to_string(TraderType trader_type) {
    switch (trader_type) {
        case TraderType::NOISE:
            return "NOISE";
        case TraderType::VALUE:
            return "VALUE";
        case TraderType::INFORMED:
            return "INFORMED";
        case TraderType::NONE:
            return "NONE";
    }

    return "UNKNOWN";
}

const char* side_to_string(Side side) {
    switch (side) {
        case Side::BUY:
            return "BUY";
        case Side::SELL:
            return "SELL";
    }

    return "UNKNOWN";
}

}  // namespace

int main() {
    SimConfig config;
    config.informed_fraction = 0.50;
    config.random_seed = 42;

    TraderAgents trader_agents(config);

    Quote quote;
    quote.bid = 0.58;
    quote.ask = 0.62;

    std::cout << "EventEdge simulator initialized.\n";
    std::cout << "Quote: bid=" << quote.bid << ", ask=" << quote.ask << '\n';

    for (int t = 1; t <= 20; ++t) {
        const auto order = trader_agents.generate_order(t, 0.60, 0.70, quote);
        if (order.has_value()) {
            std::cout << "t=" << t
                      << ", trader_type=" << trader_type_to_string(order->trader_type)
                      << ", side=" << side_to_string(order->side)
                      << ", quantity=" << order->quantity << '\n';
        }
    }

    return 0;
}
