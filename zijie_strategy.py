import numpy as np
from base_strategy import BaseStrategy


class Strategy(BaseStrategy):
    def __init__(self, side, window=100):
        super().__init__(side)
        self.window = window
        self.history = []
        self.daily_return = 0.0
        self.tick_size = 0.01

    def on_tick(self, row, current_time) -> bool:
        self.history.append(row)
        if len(self.history) < 2:
            return False

        if len(self.history) > self.window:
            self.history.pop(0)

        self.daily_return = (
            row["MidPrice"] - self.history[0]["MidPrice"]
        ) / self.history[0]["MidPrice"]

        return should_execute(
            row, self.history, self.side, self.daily_return, self.tick_size
        )


def should_execute(current, history, side, daily_trend, tick_size) -> bool:
    # 1. Feature extraction
    imbalance = (current["BidSize_1"] - current["AskSize_1"]) / (
        current["BidSize_1"] + current["AskSize_1"]
    )
    spread = current["AskPrice_1"] - current["BidPrice_1"]
    is_large_tick = spread <= 1.01 * tick_size

    # 2. Price momentum
    mid_change = current["MidPrice"] - history[-2]["MidPrice"]

    # 3. Global trend adjustment
    trend_adj = np.clip(daily_trend * 10, -0.2, 0.2)

    if side == "BUY":
        # - Large Tick: imbalance is high
        # - Small Tick: mid price starts to fall
        if is_large_tick:
            return imbalance > (0.6 - trend_adj)
        else:
            return mid_change < 0 and imbalance > (0.4 - trend_adj)

    else:  # SIDE == "SELL"
        # - Large Tick: imbalance is low
        # - Small Tick: mid price starts to rise
        if is_large_tick:
            return imbalance < (-0.6 - trend_adj)
        else:
            return mid_change > 0 and imbalance < (-0.4 - trend_adj)
