#include <iostream>

#include "types.hpp"

int main() {
    SimConfig config;
    Quote quote;
    quote.bid = 0.56;
    quote.ask = 0.64;

    std::cout << "EventEdge simulator initialized.\n";
    std::cout << "Initial true probability: " << config.true_probability << '\n';
    std::cout << "Market maker quote: bid=" << quote.bid
              << ", ask=" << quote.ask << '\n';

    return 0;
}
