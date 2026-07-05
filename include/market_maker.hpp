#pragma once

#include "types.hpp"

class MarketMaker {
public:
    explicit MarketMaker(SimConfig config);

    Quote compute_quote(double public_signal, int timestamp);
    void process_fill(const Fill& fill, double true_prob_for_mtm);

    int inventory() const;
    double cash() const;
    double realized_pnl() const;
    double unrealized_pnl() const;
    double total_pnl() const;
    double estimated_prob() const;
    double fees_paid() const;

private:
    SimConfig config_;
    int inventory_ = 0;
    double cash_ = 0.0;
    double estimated_prob_ = 0.0;
    double realized_pnl_ = 0.0;
    double unrealized_pnl_ = 0.0;
    double fees_paid_ = 0.0;
};
