#include "market_maker.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

double clamp_probability(double value) {
    return std::clamp(value, 0.01, 0.99);
}

// GM belief grid: 0.005 spacing over [0.005, 0.995]. Fine enough that the
// regret-free prices move smoothly; coarse enough that a step costs ~10k
// flops.
constexpr double kGmGridStep = 0.005;
constexpr int kGmGridSize = 199;

double normal_cdf(double z) {
    return 0.5 * std::erfc(-z / std::sqrt(2.0));
}

double normal_pdf(double x, double sigma) {
    const double z = x / sigma;
    return std::exp(-0.5 * z * z) / sigma;
}

}  // namespace

MarketMaker::MarketMaker(SimConfig config)
    : config_(config),
      inventory_(config.initial_inventory),
      cash_(config.initial_cash),
      estimated_prob_(config.true_prob_init),
      realized_pnl_(config.initial_cash),
      unrealized_pnl_(config.initial_inventory * config.true_prob_init) {
    if (config_.mm_strategy == MMStrategy::GLOSTEN_MILGROM) {
        gm_grid_.resize(kGmGridSize);
        for (int i = 0; i < kGmGridSize; ++i) {
            gm_grid_[i] = kGmGridStep * (i + 1);
        }
        // Prior: flat. The first public signal dominates it anyway.
        gm_belief_.assign(kGmGridSize, 1.0 / kGmGridSize);
        gm_scratch_.assign(kGmGridSize, 0.0);
        // The quoter's model of the latent process: Gaussian steps with a
        // jump mixture, as the additive process. Kernel is tabulated once.
        const int half = kGmGridSize - 1;
        gm_kernel_.resize(2 * half + 1);
        for (int k = -half; k <= half; ++k) {
            const double dx = k * kGmGridStep;
            double density = (1.0 - config_.gm_jump_prob) * normal_pdf(dx, config_.gm_vol);
            if (config_.gm_jump_prob > 0.0) {
                const double jump_sigma = std::hypot(config_.gm_vol, config_.gm_jump_vol);
                density += config_.gm_jump_prob * normal_pdf(dx, jump_sigma);
            }
            gm_kernel_[k + half] = density;
        }
        gm_informed_ = config_.gm_informed >= 0.0 ? config_.gm_informed
                                                  : config_.informed_fraction;
    }
}

Quote MarketMaker::compute_quote(double public_signal, int timestamp) {
    if (config_.mm_strategy == MMStrategy::GLOSTEN_MILGROM) {
        return glosten_milgrom_quote(public_signal, timestamp);
    }

    estimated_prob_ = clamp_probability(public_signal + config_.calibration_bias);

    Quote quote;
    quote.bid_size = config_.quote_size;
    quote.ask_size = config_.quote_size;
    quote.timestamp = timestamp;

    const double half_spread = config_.base_spread / 2.0;
    double quote_center = estimated_prob_;

    if (config_.mm_strategy == MMStrategy::INVENTORY_AWARE) {
        // Long inventory shifts quotes down so traders buy from the MM and
        // reduce its position; short inventory shifts quotes up.
        quote_center = clamp_probability(
            estimated_prob_ - config_.inventory_aversion * inventory_);
    }

    quote.bid = clamp_probability(quote_center - half_spread);
    quote.ask = clamp_probability(quote_center + half_spread);

    last_quote_ = quote;
    return quote;
}

// ---------------------------------------------------------------------------
// Glosten-Milgrom quoter
// ---------------------------------------------------------------------------

void MarketMaker::gm_normalize() {
    const double total = std::accumulate(gm_belief_.begin(), gm_belief_.end(), 0.0);
    if (total <= 0.0 || !std::isfinite(total)) {
        gm_belief_.assign(kGmGridSize, 1.0 / kGmGridSize);
        return;
    }
    for (double& w : gm_belief_) w /= total;
}

double MarketMaker::gm_mean() const {
    double m = 0.0;
    for (int i = 0; i < kGmGridSize; ++i) m += gm_grid_[i] * gm_belief_[i];
    return m;
}

// Belief diffuses by the assumed latent process (prediction step).
void MarketMaker::gm_predict() {
    const int half = kGmGridSize - 1;
    for (int i = 0; i < kGmGridSize; ++i) {
        double acc = 0.0;
        for (int j = 0; j < kGmGridSize; ++j) {
            acc += gm_belief_[j] * gm_kernel_[i - j + half];
        }
        gm_scratch_[i] = acc;
    }
    gm_belief_.swap(gm_scratch_);
    gm_normalize();
}

// Public signal is p + N(0, sigma_pub); the MM's calibration bias shifts
// what it thinks it saw, so bias enters the GM quoter exactly as it enters
// the naive one.
void MarketMaker::gm_update_signal(double public_signal) {
    const double seen = clamp_probability(public_signal + config_.calibration_bias);
    const double sigma = std::max(config_.signal_noise_pub, 1e-4);
    for (int i = 0; i < kGmGridSize; ++i) {
        gm_belief_[i] *= normal_pdf(seen - gm_grid_[i], sigma);
    }
    gm_normalize();
}

// P(an order of this side arrives | latent p, quoted price), under the
// trader population the agents actually implement: informed with a private
// signal, value with a public signal, both needing kMinEdge past the quote;
// noise trades a fair coin kNoiseTradeProbability of the time.
double MarketMaker::gm_arrival_probability(double p, Side trader_side, double price) const {
    const double edge = trader_side == Side::BUY ? (p - price - kMinEdge)
                                                 : (price - kMinEdge - p);
    const double informed = gm_informed_;
    const double value = kValueTraderProbability;
    const double noise = std::max(0.0, 1.0 - informed - value);
    return informed * normal_cdf(edge / std::max(config_.signal_noise_priv, 1e-4))
         + value * normal_cdf(edge / std::max(config_.signal_noise_pub, 1e-4))
         + noise * kNoiseTradeProbability * 0.5;
}

// Regret-free price: a = E[p | a trader buys at a] (ask) or
// b = E[p | a trader sells at b] (bid). A fixed point, found by iteration
// from the posterior mean; the map is a contraction in practice.
double MarketMaker::gm_regret_free_price(Side trader_side) const {
    double price = gm_mean();
    for (int iter = 0; iter < 25; ++iter) {
        double num = 0.0;
        double den = 0.0;
        for (int i = 0; i < kGmGridSize; ++i) {
            const double w = gm_belief_[i]
                * gm_arrival_probability(gm_grid_[i], trader_side, price);
            num += w * gm_grid_[i];
            den += w;
        }
        if (den <= 0.0) break;
        const double next = num / den;
        if (std::fabs(next - price) < 1e-7) {
            price = next;
            break;
        }
        price = next;
    }
    return price;
}

Quote MarketMaker::glosten_milgrom_quote(double public_signal, int timestamp) {
    gm_predict();
    gm_update_signal(public_signal);
    estimated_prob_ = clamp_probability(gm_mean());

    Quote quote;
    quote.bid_size = config_.quote_size;
    quote.ask_size = config_.quote_size;
    quote.timestamp = timestamp;
    double ask = gm_regret_free_price(Side::BUY) + config_.gm_markup;
    double bid = gm_regret_free_price(Side::SELL) - config_.gm_markup;
    if (config_.mm_strategy == MMStrategy::GLOSTEN_MILGROM && config_.inventory_aversion > 0.0) {
        const double skew = config_.inventory_aversion * inventory_;
        ask -= skew;
        bid -= skew;
    }
    quote.ask = clamp_probability(std::max(ask, bid));
    quote.bid = clamp_probability(std::min(bid, quote.ask));
    last_quote_ = quote;
    return quote;
}

void MarketMaker::observe_flow(std::optional<Side> order_side) {
    if (config_.mm_strategy != MMStrategy::GLOSTEN_MILGROM) return;
    for (int i = 0; i < kGmGridSize; ++i) {
        const double p = gm_grid_[i];
        const double p_buy = gm_arrival_probability(p, Side::BUY, last_quote_.ask);
        const double p_sell = gm_arrival_probability(p, Side::SELL, last_quote_.bid);
        double likelihood;
        if (!order_side) {
            likelihood = std::max(0.0, 1.0 - p_buy - p_sell);
        } else if (*order_side == Side::BUY) {
            likelihood = p_buy;
        } else {
            likelihood = p_sell;
        }
        gm_belief_[i] *= likelihood;
    }
    gm_normalize();
}

// ---------------------------------------------------------------------------

void MarketMaker::process_fill(const Fill& fill, double true_prob_for_mtm) {
    inventory_ += fill.mm_inventory_change;

    if (fill.mm_inventory_change > 0) {
        cash_ -= fill.price * fill.quantity;
    } else if (fill.mm_inventory_change < 0) {
        cash_ += fill.price * fill.quantity;
    }

    // Kalshi's fee is proportional to p(1-p) at the trade price (it is a
    // percentage of the contract's maximum loss on either side), so it is
    // largest exactly where the spread is widest in probability terms.
    double fee = config_.transaction_cost * fill.quantity;
    if (config_.fee_rate > 0.0) {
        double variable = config_.fee_rate * fill.price * (1.0 - fill.price) * fill.quantity;
        if (config_.fee_round_cents) {
            variable = std::ceil(variable * 100.0 - 1e-9) / 100.0;
        }
        fee += variable;
    }
    fees_paid_ += fee;
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
