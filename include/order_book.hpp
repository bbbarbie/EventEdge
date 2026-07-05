#pragma once

#include <optional>

#include "types.hpp"

class OrderBook {
public:
    void update_quotes(Quote q);
    Quote best_quote() const;
    std::optional<Fill> match_order(const Order& order) const;

private:
    bool has_valid_quote() const;

    Quote current_quote_;
};
