#include <iostream>

#include "order_book.hpp"
#include "types.hpp"

int main() {
    OrderBook order_book;

    Quote quote;
    quote.bid = 0.56;
    quote.ask = 0.64;
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
        std::cout << "Trader BUY fill: price=" << buy_fill->price
                  << ", mm_inventory_change=" << buy_fill->mm_inventory_change << '\n';
    }

    const auto sell_fill = order_book.match_order(sell_order);
    if (sell_fill.has_value()) {
        std::cout << "Trader SELL fill: price=" << sell_fill->price
                  << ", mm_inventory_change=" << sell_fill->mm_inventory_change << '\n';
    }

    return 0;
}
