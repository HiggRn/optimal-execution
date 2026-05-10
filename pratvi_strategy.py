from collections import deque
import math
import numpy as np
from base_strategy import BaseStrategy


class Strategy(BaseStrategy):
    def __init__(self, side, window=100, tau=5.0):
        super().__init__(side)
        self.window = window
        self.history = deque(maxlen=window)
        self.rolling_tfi = 0.0
        self.tick_size = 0.01

        self.prev_time = None
        self.tau = tau

        self.ema_fast = None
        self.ema_slow = None

        self.spread_history = deque(maxlen=100)
        self.median_spread = None
        self.tick_count = 0

    def on_tick(self, row, idx) -> bool:
        current_spread = row["AskPrice_1"] - row["BidPrice_1"]

        self.spread_history.append(current_spread)
        self.tick_count += 1

        if self.median_spread is None or self.tick_count % 10 == 0:
            sorted_spreads = sorted(self.spread_history)
            self.median_spread = sorted_spreads[len(sorted_spreads) // 2]

        return should_execute(
            current=row,
            side=self.side,
            tick_size=self.tick_size,
            median_spread=self.median_spread,
            current_spread=current_spread
        )


def should_execute(
    current,
    side,
    tick_size,
    median_spread,
    current_spread
) -> bool:
    is_large_tick = median_spread <= 1.5 * tick_size

    safe_spread = max(2.0 * tick_size, median_spread)
    if current_spread > safe_spread + 1e-5:
        return False

    if is_large_tick:
        # Order Book Imbalance
        if side == "BUY":
            if current['Microprice'] > current['MidPrice'] + (0.4 * current['Spread']):
                return True
        else:
            if current['Microprice'] < current['MidPrice'] - (0.4 * current['Spread']):
                return True
        return False

    else:  # small tick
        # Trade Flow Imbalance
        if side == "BUY":
            return current['TFI'] < (current['Rolling_TFI'] - current['Rolling_std_TFI'])
        else:  # SELL
            return current['TFI'] > (current['Rolling_TFI'] + current['Rolling_std_TFI'])

