#include <iostream>

#include "market_maker.hpp"
#include "order_book.hpp"
#include "types.hpp"

namespace {

void print_market_maker_state(const MarketMaker& market_maker) {
    std::cout << "Inventory: " << market_maker.inventory()
              << ", cash: " << market_maker.cash()
              << ", realized_pnl: " << market_maker.realized_pnl()
              << ", unrealized_pnl: " << market_maker.unrealized_pnl()
              << ", total_pnl: " << market_maker.total_pnl()
              << ", fees_paid: " << market_maker.fees_paid() << '\n';
}

}  // namespace

int main() {
    SimConfig config;
    config.true_prob_init = 0.60;
    config.base_spread = 0.04;
    config.transaction_cost = 0.001;

    MarketMaker market_maker(config);
    OrderBook order_book;

    const Quote quote = market_maker.compute_quote(0.60, 0);
    order_book.update_quotes(quote);

    Order buy_order;
    buy_order.id = 1;
    buy_order.order_type = OrderType::MARKET;
    buy_order.side = Side::BUY;
    buy_order.quantity = 1;

    Order sell_order;
    sell_order.id = 2;
    sell_order.order_type = OrderType::MARKET;
    sell_order.side = Side::SELL;
    sell_order.quantity = 1;

    std::cout << "EventEdge simulator initialized.\n";
    std::cout << "Market maker quote: bid=" << quote.bid
              << ", ask=" << quote.ask << '\n';

    const auto buy_fill = order_book.match_order(buy_order);
    if (buy_fill.has_value()) {
        market_maker.process_fill(*buy_fill, config.true_prob_init);
        std::cout << "Trader BUY fill: price=" << buy_fill->price
                  << ", mm_inventory_change=" << buy_fill->mm_inventory_change << '\n';
        print_market_maker_state(market_maker);
    }

    const auto sell_fill = order_book.match_order(sell_order);
    if (sell_fill.has_value()) {
        market_maker.process_fill(*sell_fill, config.true_prob_init);
        std::cout << "Trader SELL fill: price=" << sell_fill->price
                  << ", mm_inventory_change=" << sell_fill->mm_inventory_change << '\n';
        print_market_maker_state(market_maker);
    }

    return 0;
}
