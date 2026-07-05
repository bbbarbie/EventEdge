#include "order_book.hpp"

void OrderBook::update_quotes(Quote q) {
    current_quote_ = q;
}

Quote OrderBook::best_quote() const {
    return current_quote_;
}

std::optional<Fill> OrderBook::match_order(const Order& order) const {
    if (!has_valid_quote() || order.order_type != OrderType::MARKET || order.quantity <= 0) {
        return std::nullopt;
    }

    Fill fill;
    fill.order_id = order.id;
    fill.side = order.side;
    fill.quantity = order.quantity;
    fill.trader_type = order.trader_type;
    fill.timestamp = order.timestamp;

    if (order.side == Side::BUY) {
        fill.price = current_quote_.ask;
        fill.mm_inventory_change = -order.quantity;
    } else {
        fill.price = current_quote_.bid;
        fill.mm_inventory_change = order.quantity;
    }

    return fill;
}

bool OrderBook::has_valid_quote() const {
    return current_quote_.bid > 0.0 &&
           current_quote_.ask > 0.0 &&
           current_quote_.bid <= current_quote_.ask;
}
