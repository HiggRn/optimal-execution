import math

import numpy as np
from base_strategy import BaseStrategy


class Strategy(BaseStrategy):
    def __init__(self, side, window=100, tau=5.0):
        super().__init__(side)
        self.window = window
        self.history = []
        self.daily_return = 0.0
        self.tick_size = 0.01

        self.avg_spread = None
        self.prev_time = None
        self.tau = tau

    def on_tick(self, row, current_time) -> bool:
        current_spread = row["AskPrice_1"] - row["BidPrice_1"]

        if self.avg_spread is None or self.prev_time is None:
            self.avg_spread = current_spread
        else:
            dt = (current_time - self.prev_time).total_seconds()
            decay = math.exp(-dt / self.tau)
            self.avg_spread = current_spread * (1 - decay) + self.avg_spread * decay
        self.prev_time = current_time

        self.history.append(row)
        if len(self.history) < 2:
            return False

        if len(self.history) > self.window:
            self.history.pop(0)

        self.daily_return = (
            row["MidPrice"] - self.history[0]["MidPrice"]
        ) / self.history[0]["MidPrice"]

        return should_execute(
            row,
            self.history,
            self.side,
            self.daily_return,
            self.tick_size,
            self.avg_spread,
        )


def should_execute(current, history, side, daily_trend, tick_size, avg_spread) -> bool:
    imbalance = (current["BidSize_1"] - current["AskSize_1"]) / (
        current["BidSize_1"] + current["AskSize_1"]
    )

    is_large_tick = avg_spread <= 1.5 * tick_size

    trend_adj = np.clip(daily_trend * 10, -0.2, 0.2)

    if side == "BUY":
        if is_large_tick:
            return imbalance > (0.6 - trend_adj)
        else:
            return imbalance > (0.2 - trend_adj)

    else:  # SIDE == "SELL"
        if is_large_tick:
            return imbalance < (-0.6 - trend_adj)
        else:
            return imbalance < (-0.2 - trend_adj)
