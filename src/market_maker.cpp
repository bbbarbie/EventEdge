#include "market_maker.hpp"

#include <algorithm>
#include <cmath>

namespace {

double clamp_probability(double value) {
    return std::clamp(value, 0.01, 0.99);
}

}  // namespace

MarketMaker::MarketMaker(SimConfig config)
    : config_(config),
      inventory_(config.initial_inventory),
      cash_(config.initial_cash),
      estimated_prob_(config.true_prob_init),
      realized_pnl_(config.initial_cash),
      unrealized_pnl_(config.initial_inventory * config.true_prob_init) {}

Quote MarketMaker::compute_quote(double public_signal, int timestamp) {
    estimated_prob_ = clamp_probability(public_signal + config_.calibration_bias);

    Quote quote;
    quote.bid_size = config_.quote_size;
    quote.ask_size = config_.quote_size;
    quote.timestamp = timestamp;

    if (config_.mm_strategy == MMStrategy::FIXED_SPREAD) {
        const double half_spread = config_.base_spread / 2.0;
        quote.bid = clamp_probability(estimated_prob_ - half_spread);
        quote.ask = clamp_probability(estimated_prob_ + half_spread);
    }

    return quote;
}

void MarketMaker::process_fill(const Fill& fill, double true_prob_for_mtm) {
    inventory_ += fill.mm_inventory_change;

    if (fill.mm_inventory_change > 0) {
        cash_ -= fill.price * fill.quantity;
    } else if (fill.mm_inventory_change < 0) {
        cash_ += fill.price * fill.quantity;
    }

    fees_paid_ += config_.transaction_cost * fill.quantity;
    realized_pnl_ = cash_ - fees_paid_;
    unrealized_pnl_ = inventory_ * true_prob_for_mtm;
}

int MarketMaker::inventory() const {
    return inventory_;
}

double MarketMaker::cash() const {
    return cash_;
}

double MarketMaker::realized_pnl() const {
    return realized_pnl_;
}

double MarketMaker::unrealized_pnl() const {
    return unrealized_pnl_;
}

double MarketMaker::total_pnl() const {
    return realized_pnl_ + unrealized_pnl_;
}

double MarketMaker::estimated_prob() const {
    return estimated_prob_;
}

double MarketMaker::fees_paid() const {
    return fees_paid_;
}
