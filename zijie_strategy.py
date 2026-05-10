from collections import deque
import numpy as np
from base_strategy import BaseStrategy


class Strategy(BaseStrategy):
    def __init__(self, side, window=100):
        super().__init__(side)
        self.window = window
        self.tfi_history = deque(maxlen=window)
        self.tick_size = 0.01

        self.spread_history = deque(maxlen=100)
        self.median_spread = None
        self.tick_count = 0

    def on_tick(self, row, current_time) -> bool:
        current_spread = row["AskPrice_1"] - row["BidPrice_1"]

        self.spread_history.append(current_spread)
        self.tick_count += 1

        if self.median_spread is None or self.tick_count % 10 == 0:
            sorted_spreads = sorted(self.spread_history)
            self.median_spread = sorted_spreads[len(sorted_spreads) // 2]

        total_size = row["BidSize_1"] + row["AskSize_1"]
        safe_total_size = total_size if total_size > 0 else 1.0

        # Order Book Imbalance
        obi = (row["BidSize_1"] - row["AskSize_1"]) / safe_total_size

        # TFI
        vis = row.get("VisibleExecution_1=Yes_0=No", 0)
        hid = row.get("HiddenExecution_1=Yes_0=No", 0)
        tfi = 0.0
        if vis == 1 or hid == 1:
            direction = row.get("Direction_1=Buy_-1=Sell", 0)
            size = row.get("Size", 0)
            tfi = (direction * size) / safe_total_size

        # Rolling TFI
        if len(self.tfi_history) > 0:
            rolling_tfi_mean = np.mean(self.tfi_history)
            rolling_tfi_std = (
                np.std(self.tfi_history, ddof=1) if len(self.tfi_history) > 1 else 0.0
            )
        else:
            rolling_tfi_mean = 0.0
            rolling_tfi_std = 0.0

        execute = should_execute(
            side=self.side,
            tick_size=self.tick_size,
            median_spread=self.median_spread,
            current_spread=current_spread,
            obi=obi,
            tfi=tfi,
            rolling_tfi_mean=rolling_tfi_mean,
            rolling_tfi_std=rolling_tfi_std,
        )

        self.tfi_history.append(tfi)

        return execute


def should_execute(
    side,
    tick_size,
    median_spread,
    current_spread,
    obi,
    tfi,
    rolling_tfi_mean,
    rolling_tfi_std,
) -> bool:
    is_large_tick = median_spread <= 1.5 * tick_size

    safe_spread = max(2.0 * tick_size, median_spread)
    if current_spread > safe_spread + 1e-5:
        return False

    if is_large_tick:
        # Order Book Imbalance
        if side == "BUY":
            return obi > 0.8
        else:  # SELL
            return obi < -0.8

    else:  # small tick
        # Trade Flow Imbalance
        if side == "BUY":
            return tfi < (rolling_tfi_mean - rolling_tfi_std)
        else:  # SELL
            return tfi > (rolling_tfi_mean + rolling_tfi_std)
