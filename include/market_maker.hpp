#pragma once

#include <optional>
#include <vector>

#include "types.hpp"

class MarketMaker {
public:
    explicit MarketMaker(SimConfig config);

    Quote compute_quote(double public_signal, int timestamp);
    void process_fill(const Fill& fill, double true_prob_for_mtm);

    // What the MM saw on the tape this step: an order's side, or none. The
    // Glosten-Milgrom quoter updates its posterior from this (a quiet step
    // is evidence too); the other strategies ignore it.
    void observe_flow(std::optional<Side> order_side);

    int inventory() const;
    double cash() const;
    double realized_pnl() const;
    double unrealized_pnl() const;
    double total_pnl() const;
    double estimated_prob() const;
    double fees_paid() const;

private:
    Quote glosten_milgrom_quote(double public_signal, int timestamp);
    void gm_predict();
    void gm_update_signal(double public_signal);
    double gm_regret_free_price(Side trader_side) const;
    double gm_arrival_probability(double p, Side trader_side, double price) const;
    double gm_mean() const;
    void gm_normalize();

    SimConfig config_;
    int inventory_ = 0;
    double cash_ = 0.0;
    double estimated_prob_ = 0.0;
    double realized_pnl_ = 0.0;
    double unrealized_pnl_ = 0.0;
    double fees_paid_ = 0.0;

    // GM posterior over the latent probability on a fixed grid.
    std::vector<double> gm_grid_;
    std::vector<double> gm_belief_;
    std::vector<double> gm_scratch_;
    std::vector<double> gm_kernel_;  // one-step transition kernel, symmetric
    Quote last_quote_;
    double gm_informed_ = 0.0;
};
