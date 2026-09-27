#include "order_book.hpp"

void OrderBook::update_quotes(Quote q) {
    current_quote_ = q;
}

Quote OrderBook::best_quote() const {
    return current_quote_;
}

void OrderBook::configure_queue(double mean_depth_ahead, std::uint32_t seed) {
    queue_ahead_ = mean_depth_ahead;
    queue_rng_.seed(seed);
}

long OrderBook::absorbed_count() const {
    return absorbed_;
}

std::optional<Fill> OrderBook::match_order(const Order& order) {
    if (!has_valid_quote() || order.order_type != OrderType::MARKET || order.quantity <= 0) {
        return std::nullopt;
    }

    // Liquidity ahead of the MM in the queue takes the first contracts of
    // every order. A unit noise order rarely gets through; a larger
    // informed sweep does. This is why a fill is worse news than an order.
    int remaining = order.quantity;
    if (queue_ahead_ > 0.0) {
        std::poisson_distribution<int> depth_dist(queue_ahead_);
        remaining -= depth_dist(queue_rng_);
        if (remaining <= 0) {
            ++absorbed_;
            return std::nullopt;
        }
    }

    Fill fill;
    fill.order_id = order.id;
    fill.side = order.side;
    fill.trader_type = order.trader_type;
    fill.timestamp = order.timestamp;

    // The MM fills whatever is left of the order at its quote (quote size is
    // not enforced, as before), so a larger informed order that reaches the
    // MM also moves more inventory against it.
    fill.quantity = remaining;
    if (order.side == Side::BUY) {
        fill.price = current_quote_.ask;
        fill.mm_inventory_change = -fill.quantity;
    } else {
        fill.price = current_quote_.bid;
        fill.mm_inventory_change = fill.quantity;
    }

    return fill;
}

bool OrderBook::has_valid_quote() const {
    return current_quote_.bid > 0.0 &&
           current_quote_.ask > 0.0 &&
           current_quote_.bid <= current_quote_.ask;
}
