// Unit tests for EventEdge core simulation logic.
//
// Covers the required tests from CLAUDE.md:
//   - trader buy fills at ask / sell fills at bid
//   - trader buy decreases MM inventory / sell increases it
//   - cash updates with the correct sign
//   - zero-inventory wealth equals cash
//   - terminal settlement for long and short inventory
//   - probability outputs stay in bounds
//   - quote outputs satisfy bid <= ask
//   - seeded runs are reproducible
//   - fill-at-execution-price wealth invariant
//   - informed trader may decline to trade
//
// Minimal dependency-free harness: each CHECK reports failures with
// file:line; the process exit code is the failure count (0 = all pass).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <vector>

#include "event_probability.hpp"
#include "market_maker.hpp"
#include "order_book.hpp"
#include "trader_agents.hpp"
#include "types.hpp"

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        ++g_checks;                                                     \
        if (!(cond)) {                                                  \
            ++g_failures;                                               \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        }                                                               \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                             \
    do {                                                                  \
        ++g_checks;                                                       \
        const double a_ = (a);                                            \
        const double b_ = (b);                                            \
        if (std::fabs(a_ - b_) > (eps)) {                                 \
            ++g_failures;                                                 \
            std::printf("FAIL %s:%d  %s == %s  (%.10f vs %.10f)\n",       \
                        __FILE__, __LINE__, #a, #b, a_, b_);              \
        }                                                                 \
    } while (0)

SimConfig test_config() {
    SimConfig config;
    config.true_prob_init = 0.60;
    config.base_spread = 0.04;
    config.calibration_bias = 0.0;
    config.transaction_cost = 0.0;
    config.initial_cash = 0.0;
    config.initial_inventory = 0;
    config.quote_size = 1;
    config.random_seed = 7;
    return config;
}

Order market_order(TraderType trader_type, Side side, int quantity = 1) {
    Order order;
    order.id = 1;
    order.trader_type = trader_type;
    order.order_type = OrderType::MARKET;
    order.side = side;
    order.quantity = quantity;
    order.timestamp = 1.0;
    return order;
}

Quote make_quote(double bid, double ask) {
    Quote quote;
    quote.bid = bid;
    quote.ask = ask;
    quote.bid_size = 1;
    quote.ask_size = 1;
    return quote;
}

// --- Matching -------------------------------------------------------------

void test_trader_buy_fills_at_ask() {
    OrderBook book;
    book.update_quotes(make_quote(0.56, 0.64));

    const auto fill = book.match_order(market_order(TraderType::NOISE, Side::BUY));
    CHECK(fill.has_value());
    CHECK_NEAR(fill->price, 0.64, 1e-12);
    CHECK(fill->side == Side::BUY);
}

void test_trader_sell_fills_at_bid() {
    OrderBook book;
    book.update_quotes(make_quote(0.56, 0.64));

    const auto fill = book.match_order(market_order(TraderType::NOISE, Side::SELL));
    CHECK(fill.has_value());
    CHECK_NEAR(fill->price, 0.56, 1e-12);
    CHECK(fill->side == Side::SELL);
}

void test_inventory_change_signs() {
    OrderBook book;
    book.update_quotes(make_quote(0.56, 0.64));

    const auto buy_fill = book.match_order(market_order(TraderType::NOISE, Side::BUY, 3));
    CHECK(buy_fill.has_value());
    CHECK(buy_fill->mm_inventory_change == -3);  // trader BUY -> MM inventory falls

    const auto sell_fill = book.match_order(market_order(TraderType::NOISE, Side::SELL, 2));
    CHECK(sell_fill.has_value());
    CHECK(sell_fill->mm_inventory_change == +2);  // trader SELL -> MM inventory rises
}

void test_order_book_rejects_invalid() {
    OrderBook book;  // no quote posted yet
    CHECK(!book.match_order(market_order(TraderType::NOISE, Side::BUY)).has_value());

    book.update_quotes(make_quote(0.56, 0.64));
    Order limit = market_order(TraderType::NOISE, Side::BUY);
    limit.order_type = OrderType::LIMIT;
    CHECK(!book.match_order(limit).has_value());

    Order zero_qty = market_order(TraderType::NOISE, Side::BUY, 0);
    CHECK(!book.match_order(zero_qty).has_value());
}

// --- Market-maker accounting ----------------------------------------------

void test_cash_sign_on_fills() {
    MarketMaker mm(test_config());
    OrderBook book;
    book.update_quotes(make_quote(0.56, 0.64));

    // Trader BUY: MM sells at ask, receives cash.
    const auto buy_fill = book.match_order(market_order(TraderType::NOISE, Side::BUY));
    mm.process_fill(*buy_fill, 0.60);
    CHECK(mm.inventory() == -1);
    CHECK_NEAR(mm.cash(), 0.64, 1e-12);

    // Trader SELL: MM buys at bid, pays cash.
    const auto sell_fill = book.match_order(market_order(TraderType::NOISE, Side::SELL));
    mm.process_fill(*sell_fill, 0.60);
    CHECK(mm.inventory() == 0);
    CHECK_NEAR(mm.cash(), 0.64 - 0.56, 1e-12);
}

void test_zero_inventory_wealth_equals_cash() {
    MarketMaker mm(test_config());
    OrderBook book;
    book.update_quotes(make_quote(0.56, 0.64));

    mm.process_fill(*book.match_order(market_order(TraderType::NOISE, Side::SELL)), 0.60);
    mm.process_fill(*book.match_order(market_order(TraderType::NOISE, Side::BUY)), 0.60);

    CHECK(mm.inventory() == 0);
    CHECK_NEAR(mm.total_pnl(), mm.cash(), 1e-12);  // round trip: wealth == cash == spread
    CHECK_NEAR(mm.cash(), 0.08, 1e-12);
}

void test_fill_wealth_invariant_at_execution_price() {
    // A fill marked at its own execution price must not create wealth.
    MarketMaker mm(test_config());
    OrderBook book;
    book.update_quotes(make_quote(0.56, 0.64));

    const double wealth_before = mm.cash() + mm.inventory() * 0.56;
    mm.process_fill(*book.match_order(market_order(TraderType::NOISE, Side::SELL)), 0.56);
    const double wealth_after = mm.cash() + mm.inventory() * 0.56;
    CHECK_NEAR(wealth_after, wealth_before, 1e-12);

    const double wealth_before2 = mm.cash() + mm.inventory() * 0.64;
    mm.process_fill(*book.match_order(market_order(TraderType::NOISE, Side::BUY, 2)), 0.64);
    const double wealth_after2 = mm.cash() + mm.inventory() * 0.64;
    CHECK_NEAR(wealth_after2, wealth_before2, 1e-12);
}

void test_terminal_settlement_long_inventory() {
    MarketMaker mm(test_config());
    OrderBook book;
    book.update_quotes(make_quote(0.56, 0.64));

    // MM buys 3 contracts at the bid.
    for (int i = 0; i < 3; ++i) {
        mm.process_fill(*book.match_order(market_order(TraderType::NOISE, Side::SELL)), 0.60);
    }
    CHECK(mm.inventory() == 3);
    CHECK_NEAR(mm.cash(), -3 * 0.56, 1e-12);

    // Settlement: terminal_pnl = cash + inventory * Y
    const double pnl_if_yes = mm.cash() + mm.inventory() * 1.0;
    const double pnl_if_no = mm.cash() + mm.inventory() * 0.0;
    CHECK_NEAR(pnl_if_yes, 3 * (1.0 - 0.56), 1e-12);   // long wins when event occurs
    CHECK_NEAR(pnl_if_no, -3 * 0.56, 1e-12);           // long loses stake otherwise
}

void test_terminal_settlement_short_inventory() {
    MarketMaker mm(test_config());
    OrderBook book;
    book.update_quotes(make_quote(0.56, 0.64));

    // MM sells 2 contracts at the ask.
    for (int i = 0; i < 2; ++i) {
        mm.process_fill(*book.match_order(market_order(TraderType::NOISE, Side::BUY)), 0.60);
    }
    CHECK(mm.inventory() == -2);
    CHECK_NEAR(mm.cash(), 2 * 0.64, 1e-12);

    const double pnl_if_yes = mm.cash() + mm.inventory() * 1.0;
    const double pnl_if_no = mm.cash() + mm.inventory() * 0.0;
    CHECK_NEAR(pnl_if_yes, -2 * (1.0 - 0.64), 1e-12);  // short loses when event occurs
    CHECK_NEAR(pnl_if_no, 2 * 0.64, 1e-12);            // short keeps premium otherwise
}

void test_transaction_costs_reduce_realized_pnl() {
    SimConfig config = test_config();
    config.transaction_cost = 0.01;
    MarketMaker mm(config);
    OrderBook book;
    book.update_quotes(make_quote(0.56, 0.64));

    mm.process_fill(*book.match_order(market_order(TraderType::NOISE, Side::SELL)), 0.60);
    mm.process_fill(*book.match_order(market_order(TraderType::NOISE, Side::BUY)), 0.60);

    CHECK_NEAR(mm.fees_paid(), 0.02, 1e-12);
    CHECK_NEAR(mm.realized_pnl(), 0.08 - 0.02, 1e-12);
}

// --- Probabilities and quotes ----------------------------------------------

void test_probability_stays_in_bounds() {
    SimConfig config = test_config();
    config.true_prob_init = 0.98;  // start near the boundary
    EventProbabilityProcess process(config);

    for (int t = 0; t < 5000; ++t) {
        process.step();
        const double p = process.true_prob();
        CHECK(p >= 0.0 && p <= 1.0);
        const double pub = process.public_signal();
        CHECK(pub >= 0.0 && pub <= 1.0);
        const double priv = process.private_signal();
        CHECK(priv >= 0.0 && priv <= 1.0);
    }

    process.inject_shock(10.0);   // extreme shocks must still clamp
    CHECK(process.true_prob() <= 1.0);
    process.inject_shock(-10.0);
    CHECK(process.true_prob() >= 0.0);
}

void test_quote_bid_never_exceeds_ask() {
    for (const double bias : {-0.5, -0.1, 0.0, 0.1, 0.5}) {
        SimConfig config = test_config();
        config.calibration_bias = bias;
        MarketMaker mm(config);

        for (const double signal : {0.0, 0.01, 0.3, 0.6, 0.9, 0.99, 1.0}) {
            const Quote quote = mm.compute_quote(signal, 1);
            CHECK(quote.bid >= 0.0);
            CHECK(quote.ask <= 1.0);
            CHECK(quote.bid <= quote.ask);  // clipping must not invert the spread
        }
    }
}

void test_inventory_aware_quote_skew() {
    SimConfig config = test_config();
    config.mm_strategy = MMStrategy::INVENTORY_AWARE;
    config.inventory_aversion = 0.01;
    MarketMaker mm(config);
    OrderBook book;

    // Zero inventory: identical to fixed spread.
    const Quote flat = mm.compute_quote(0.60, 1);
    CHECK_NEAR(flat.bid, 0.58, 1e-12);
    CHECK_NEAR(flat.ask, 0.62, 1e-12);

    // MM buys 5 contracts -> long -> quotes must shift DOWN.
    book.update_quotes(flat);
    for (int i = 0; i < 5; ++i) {
        mm.process_fill(*book.match_order(market_order(TraderType::NOISE, Side::SELL)), 0.60);
    }
    CHECK(mm.inventory() == 5);
    const Quote skewed_down = mm.compute_quote(0.60, 2);
    CHECK_NEAR(skewed_down.bid, 0.58 - 0.05, 1e-12);
    CHECK_NEAR(skewed_down.ask, 0.62 - 0.05, 1e-12);
    CHECK(skewed_down.bid <= skewed_down.ask);

    // MM sells 10 contracts -> short 5 -> quotes must shift UP.
    book.update_quotes(flat);
    for (int i = 0; i < 10; ++i) {
        mm.process_fill(*book.match_order(market_order(TraderType::NOISE, Side::BUY)), 0.60);
    }
    CHECK(mm.inventory() == -5);
    const Quote skewed_up = mm.compute_quote(0.60, 3);
    CHECK_NEAR(skewed_up.bid, 0.58 + 0.05, 1e-12);
    CHECK_NEAR(skewed_up.ask, 0.62 + 0.05, 1e-12);
}

void test_inventory_aware_quotes_stay_bounded() {
    SimConfig config = test_config();
    config.mm_strategy = MMStrategy::INVENTORY_AWARE;
    config.inventory_aversion = 0.05;
    MarketMaker mm(config);
    OrderBook book;
    book.update_quotes(make_quote(0.56, 0.64));

    // Push inventory to an extreme; quotes must clamp without inverting.
    for (int i = 0; i < 100; ++i) {
        mm.process_fill(*book.match_order(market_order(TraderType::NOISE, Side::SELL)), 0.60);
    }
    const Quote quote = mm.compute_quote(0.60, 1);
    CHECK(quote.bid >= 0.0);
    CHECK(quote.ask <= 1.0);
    CHECK(quote.bid <= quote.ask);
}

void test_martingale_process_bounds_no_clipping() {
    SimConfig config = test_config();
    config.prob_process = ProbProcess::LOGISTIC_MARTINGALE;
    EventProbabilityProcess process(config);

    for (int t = 0; t < 5000; ++t) {
        process.step();
        const double p = process.true_prob();
        CHECK(p >= 0.0 && p <= 1.0);
    }
    // A discrete-step martingale can wander asymptotically close to the
    // bounds, so the 1e-6 safety clamp may bind on rare steps. That is
    // immaterial (there |Y - p| ~ 1e-6, so no drift leaks into P&L), but the
    // rate must be tiny -- the additive process clips ~50-150 times here.
    CHECK(process.clip_count() < 25);
}

void test_martingale_process_has_no_drift() {
    // The additive process is clamped (not a martingale near bounds); the
    // martingale process must have mean drift ~ 0. Average the terminal
    // probability over many seeded paths and compare to the start.
    const double p0 = 0.60;
    const int num_paths = 500;
    const int num_steps = 200;

    double sum_terminal = 0.0;
    for (int path = 0; path < num_paths; ++path) {
        SimConfig config = test_config();
        config.true_prob_init = p0;
        config.prob_process = ProbProcess::LOGISTIC_MARTINGALE;
        config.random_seed = static_cast<std::uint32_t>(1000 + path);
        EventProbabilityProcess process(config);
        for (int t = 0; t < num_steps; ++t) {
            process.step();
        }
        sum_terminal += process.true_prob();
    }

    const double mean_terminal = sum_terminal / num_paths;
    // Per-path terminal sd ~ 0.042*0.24*sqrt(200) ~ 0.14; SE over 500 paths
    // ~ 0.006. Tolerance 0.02 gives > 3 sigma of headroom, and the seeds are
    // fixed so this is deterministic.
    CHECK(std::fabs(mean_terminal - p0) < 0.02);
}

// --- Trader behavior --------------------------------------------------------

void test_informed_trader_may_decline_to_trade() {
    SimConfig config = test_config();
    config.informed_fraction = 1.0;  // every arrival is informed
    TraderAgents agents(config);

    // Signal inside the quoted spread: no positive-EV trade exists.
    const Quote quote = make_quote(0.55, 0.65);
    for (int t = 1; t <= 100; ++t) {
        const auto order = agents.generate_order(t, 0.60, 0.60, quote);
        CHECK(!order.has_value());
    }
}

void test_informed_trader_direction() {
    SimConfig config = test_config();
    config.informed_fraction = 1.0;
    TraderAgents agents(config);
    const Quote quote = make_quote(0.55, 0.65);

    const auto buy = agents.generate_order(1, 0.60, 0.75, quote);  // signal >> ask
    CHECK(buy.has_value());
    CHECK(buy->side == Side::BUY);
    CHECK(buy->trader_type == TraderType::INFORMED);

    const auto sell = agents.generate_order(2, 0.60, 0.45, quote);  // signal << bid
    CHECK(sell.has_value());
    CHECK(sell->side == Side::SELL);
}

// --- Reproducibility ---------------------------------------------------------

void test_seeded_runs_are_reproducible() {
    SimConfig config = test_config();

    EventProbabilityProcess process_a(config);
    EventProbabilityProcess process_b(config);
    for (int t = 0; t < 200; ++t) {
        process_a.step();
        process_b.step();
        CHECK_NEAR(process_a.true_prob(), process_b.true_prob(), 0.0);
        CHECK_NEAR(process_a.public_signal(), process_b.public_signal(), 0.0);
        CHECK_NEAR(process_a.private_signal(), process_b.private_signal(), 0.0);
    }

    TraderAgents agents_a(config);
    TraderAgents agents_b(config);
    const Quote quote = make_quote(0.55, 0.65);
    for (int t = 1; t <= 200; ++t) {
        const auto order_a = agents_a.generate_order(t, 0.62, 0.70, quote);
        const auto order_b = agents_b.generate_order(t, 0.62, 0.70, quote);
        CHECK(order_a.has_value() == order_b.has_value());
        if (order_a.has_value() && order_b.has_value()) {
            CHECK(order_a->side == order_b->side);
            CHECK(order_a->trader_type == order_b->trader_type);
        }
    }
}

}  // namespace

int main() {
    test_trader_buy_fills_at_ask();
    test_trader_sell_fills_at_bid();
    test_inventory_change_signs();
    test_order_book_rejects_invalid();
    test_cash_sign_on_fills();
    test_zero_inventory_wealth_equals_cash();
    test_fill_wealth_invariant_at_execution_price();
    test_terminal_settlement_long_inventory();
    test_terminal_settlement_short_inventory();
    test_transaction_costs_reduce_realized_pnl();
    test_probability_stays_in_bounds();
    test_quote_bid_never_exceeds_ask();
    test_inventory_aware_quote_skew();
    test_inventory_aware_quotes_stay_bounded();
    test_martingale_process_bounds_no_clipping();
    test_martingale_process_has_no_drift();
    test_informed_trader_may_decline_to_trade();
    test_informed_trader_direction();
    test_seeded_runs_are_reproducible();

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures;
}
