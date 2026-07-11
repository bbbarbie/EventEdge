#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "event_probability.hpp"
#include "market_maker.hpp"
#include "order_book.hpp"
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
    return side == Side::BUY ? "BUY" : "SELL";
}

}  // namespace

// Usage: eventedge [--seed N] [--steps N] [--out-prefix PATH]
//                  [--bias X] [--informed-fraction X] [--spread X]
//                  [--noise-pub X] [--noise-priv X] [--log-detail 0|1]
//
// Writes <prefix>_summary.csv always; <prefix>_steps.csv and
// <prefix>_fills.csv only when --log-detail is 1 (the default).
int main(int argc, char** argv) {
    SimConfig config;
    config.num_steps = 2000;
    std::string out_prefix = "run";
    bool log_detail = true;

    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        const std::string value = argv[i + 1];
        if (flag == "--seed") {
            config.random_seed = static_cast<std::uint32_t>(std::stoul(value));
        } else if (flag == "--steps") {
            config.num_steps = std::stoi(value);
        } else if (flag == "--out-prefix") {
            out_prefix = value;
        } else if (flag == "--bias") {
            config.calibration_bias = std::stod(value);
        } else if (flag == "--informed-fraction") {
            config.informed_fraction = std::stod(value);
        } else if (flag == "--spread") {
            config.base_spread = std::stod(value);
        } else if (flag == "--noise-pub") {
            config.signal_noise_pub = std::stod(value);
        } else if (flag == "--noise-priv") {
            config.signal_noise_priv = std::stod(value);
        } else if (flag == "--inventory-aversion") {
            config.inventory_aversion = std::stod(value);
        } else if (flag == "--mm-strategy") {
            if (value == "fixed") {
                config.mm_strategy = MMStrategy::FIXED_SPREAD;
            } else if (value == "inventory") {
                config.mm_strategy = MMStrategy::INVENTORY_AWARE;
            } else {
                std::cerr << "Unknown --mm-strategy: " << value << '\n';
                return 1;
            }
        } else if (flag == "--prob-process") {
            if (value == "additive") {
                config.prob_process = ProbProcess::CLAMPED_ADDITIVE;
            } else if (value == "martingale") {
                config.prob_process = ProbProcess::LOGISTIC_MARTINGALE;
            } else {
                std::cerr << "Unknown --prob-process: " << value << '\n';
                return 1;
            }
        } else if (flag == "--log-detail") {
            log_detail = std::stoi(value) != 0;
        } else {
            std::cerr << "Unknown flag: " << flag << '\n';
            return 1;
        }
    }

    if (config.num_steps <= 0 || config.base_spread < 0.0 ||
        config.informed_fraction < 0.0 || config.informed_fraction > 1.0 ||
        config.signal_noise_pub < 0.0 || config.signal_noise_priv < 0.0 ||
        config.inventory_aversion < 0.0) {
        std::cerr << "Invalid configuration values.\n";
        return 1;
    }

    // Derive distinct seeds per component so the RNG streams are not
    // accidentally identical (all constructors seed from config.random_seed).
    SimConfig prob_config = config;
    prob_config.random_seed = config.random_seed * 3u + 1u;
    SimConfig trader_config = config;
    trader_config.random_seed = config.random_seed * 3u + 2u;

    EventProbabilityProcess prob_process(prob_config);
    TraderAgents trader_agents(trader_config);
    MarketMaker market_maker(config);
    OrderBook order_book;
    std::mt19937 settlement_rng(config.random_seed * 3u + 3u);

    std::ofstream summary_out(out_prefix + "_summary.csv");
    if (!summary_out) {
        std::cerr << "Failed to open output files with prefix: " << out_prefix << '\n';
        return 1;
    }
    // Full double precision: Python decomposes terminal P&L from these logs
    // and checks an exact accounting identity, so rounding is not acceptable.
    summary_out << std::setprecision(17);

    std::ofstream steps_out;
    std::ofstream fills_out;
    if (log_detail) {
        steps_out.open(out_prefix + "_steps.csv");
        fills_out.open(out_prefix + "_fills.csv");
        if (!steps_out || !fills_out) {
            std::cerr << "Failed to open output files with prefix: " << out_prefix << '\n';
            return 1;
        }
        steps_out << std::setprecision(17);
        fills_out << std::setprecision(17);
        steps_out << "time_step,latent_probability,mm_estimate,bid,ask\n";
        fills_out << "seed,time_step,trader_type,side,quantity,fill_price,"
                     "mm_inventory_change,bid,ask,latent_probability,mm_estimate,"
                     "mm_inventory_after,mm_cash_after\n";
    }

    int fill_count = 0;
    int informed_fill_count = 0;
    int value_fill_count = 0;
    int noise_fill_count = 0;

    for (int t = 1; t <= config.num_steps; ++t) {
        // 1. latent probability evolves
        prob_process.step();
        const double p_true = prob_process.true_prob();

        // 2. MM observes its own public signal and posts a quote
        const double mm_signal = prob_process.public_signal();
        const Quote quote = market_maker.compute_quote(mm_signal, t);
        order_book.update_quotes(quote);

        // 3. arriving trader observes its own (independent) signal
        const double trader_public_signal = prob_process.public_signal();
        const double trader_private_signal = prob_process.private_signal();
        const auto order = trader_agents.generate_order(
            t, trader_public_signal, trader_private_signal, quote);

        // 4. match and process fill
        if (order.has_value()) {
            const auto fill = order_book.match_order(*order);
            if (fill.has_value()) {
                market_maker.process_fill(*fill, p_true);
                ++fill_count;
                switch (fill->trader_type) {
                    case TraderType::INFORMED: ++informed_fill_count; break;
                    case TraderType::VALUE:    ++value_fill_count;    break;
                    case TraderType::NOISE:    ++noise_fill_count;    break;
                    case TraderType::NONE:                            break;
                }
                if (log_detail) {
                    fills_out << config.random_seed << ',' << t << ','
                              << trader_type_to_string(fill->trader_type) << ','
                              << side_to_string(fill->side) << ','
                              << fill->quantity << ','
                              << fill->price << ','
                              << fill->mm_inventory_change << ','
                              << quote.bid << ',' << quote.ask << ','
                              << p_true << ','
                              << market_maker.estimated_prob() << ','
                              << market_maker.inventory() << ','
                              << market_maker.cash() << '\n';
                }
            }
        }

        if (log_detail) {
            steps_out << t << ',' << p_true << ','
                      << market_maker.estimated_prob() << ','
                      << quote.bid << ',' << quote.ask << '\n';
        }
    }

    // 5. settlement: draw outcome from terminal latent probability
    const double terminal_prob = prob_process.true_prob();
    std::bernoulli_distribution outcome_dist(terminal_prob);
    const int event_outcome = outcome_dist(settlement_rng) ? 1 : 0;
    const double terminal_pnl = market_maker.cash()
        + market_maker.inventory() * static_cast<double>(event_outcome)
        - market_maker.fees_paid();

    const char* process_name =
        config.prob_process == ProbProcess::LOGISTIC_MARTINGALE
            ? "martingale" : "additive";
    const char* strategy_name =
        config.mm_strategy == MMStrategy::INVENTORY_AWARE ? "inventory" : "fixed";
    summary_out << "seed,num_steps,calibration_bias,informed_fraction,"
                   "base_spread,signal_noise_pub,signal_noise_priv,"
                   "mm_strategy,inventory_aversion,prob_process,"
                   "terminal_probability,event_outcome,fill_count,"
                   "informed_fill_count,value_fill_count,noise_fill_count,"
                   "terminal_cash,terminal_inventory,terminal_pnl\n";
    summary_out << config.random_seed << ',' << config.num_steps << ','
                << config.calibration_bias << ',' << config.informed_fraction << ','
                << config.base_spread << ',' << config.signal_noise_pub << ','
                << config.signal_noise_priv << ','
                << strategy_name << ',' << config.inventory_aversion << ','
                << process_name << ','
                << terminal_prob << ',' << event_outcome << ','
                << fill_count << ','
                << informed_fill_count << ',' << value_fill_count << ','
                << noise_fill_count << ','
                << market_maker.cash() << ','
                << market_maker.inventory() << ',' << terminal_pnl << '\n';

    std::cout << "seed=" << config.random_seed
              << " steps=" << config.num_steps
              << " bias=" << config.calibration_bias
              << " informed=" << config.informed_fraction
              << " fills=" << fill_count
              << " outcome=" << event_outcome
              << " terminal_pnl=" << terminal_pnl << '\n';
    return 0;
}
