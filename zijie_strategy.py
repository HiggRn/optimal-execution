import math

import numpy as np
from base_strategy import BaseStrategy


class Strategy(BaseStrategy):
    def __init__(self, side, window=100, tau=5.0):
        super().__init__(side)
        self.window = window
        self.history = []
        self.tick_size = 0.01

        self.avg_spread = None
        self.prev_time = None
        self.tau = tau

        self.ema_fast = None
        self.ema_slow = None

    def on_tick(self, row, current_time) -> bool:
        current_spread = row["AskPrice_1"] - row["BidPrice_1"]

        if self.avg_spread is None or self.prev_time is None:
            self.avg_spread = current_spread
        else:
            dt = (current_time - self.prev_time).total_seconds()
            decay = math.exp(-dt / self.tau)
            self.avg_spread = current_spread * (1 - decay) + self.avg_spread * decay

        self.ema_fast, self.ema_slow, macd_trend = update_macd_trend(
            row, current_time, self.prev_time, self.ema_fast, self.ema_slow
        )

        self.prev_time = current_time

        self.history.append(row)
        if len(self.history) < 2:
            return False

        if len(self.history) > self.window:
            self.history.pop(0)

        return should_execute(
            current=row,
            history=self.history,
            side=self.side,
            daily_trend=macd_trend,
            tick_size=self.tick_size,
            avg_spread=self.avg_spread,
        )


def should_execute(current, history, side, daily_trend, tick_size, avg_spread) -> bool:
    imbalance = (current["BidSize_1"] - current["AskSize_1"]) / (
        current["BidSize_1"] + current["AskSize_1"]
    )

    is_large_tick = avg_spread <= 1.5 * tick_size

    trend_adj = np.clip(daily_trend * 10, -0.2, 0.2)

    if is_large_tick:
        if side == "BUY":
            return imbalance > (0.6 - trend_adj)
        else:  # SELL
            return imbalance < (-0.6 - trend_adj)
    else:  # small tick
        rolling_ofi = calculate_rolling_ofi(history)

        current_volume = current["BidSize_1"] + current["AskSize_1"]
        ofi_norm = rolling_ofi / current_volume if current_volume > 0 else 0.0

        if side == "BUY":
            return ofi_norm > (2.0 - trend_adj * 5)
        else:  # SELL
            return ofi_norm < (-2.0 - trend_adj * 5)


def calculate_rolling_ofi(history) -> float:
    if len(history) < 2:
        return 0.0

    total_ofi = 0.0
    for i in range(1, len(history)):
        curr = history[i]
        prev = history[i - 1]

        # Bid side OFI
        if curr["BidPrice_1"] > prev["BidPrice_1"]:
            bid_flow = curr["BidSize_1"]
        elif curr["BidPrice_1"] == prev["BidPrice_1"]:
            bid_flow = curr["BidSize_1"] - prev["BidSize_1"]
        else:
            bid_flow = -prev["BidSize_1"]

        # Ask side OFI
        if curr["AskPrice_1"] < prev["AskPrice_1"]:
            ask_flow = curr["AskSize_1"]
        elif curr["AskPrice_1"] == prev["AskPrice_1"]:
            ask_flow = curr["AskSize_1"] - prev["AskSize_1"]
        else:
            ask_flow = -prev["AskSize_1"]

        total_ofi += bid_flow - ask_flow

    return total_ofi


def update_macd_trend(
    row, current_time, prev_time, ema_fast, ema_slow, tau_fast=5.0, tau_slow=60.0
):
    current_spread = row["AskPrice_1"] - row["BidPrice_1"]
    current_vol = row["BidSize_1"] + row["AskSize_1"]

    imbalance_ratio = row["BidSize_1"] / current_vol if current_vol > 0 else 0.5
    micro_price = row["BidPrice_1"] + imbalance_ratio * current_spread

    if prev_time is None or ema_fast is None or ema_slow is None:
        return micro_price, micro_price, 0.0

    dt = (current_time - prev_time).total_seconds()
    decay_fast = math.exp(-dt / tau_fast)
    decay_slow = math.exp(-dt / tau_slow)

    new_ema_fast = micro_price * (1 - decay_fast) + ema_fast * decay_fast
    new_ema_slow = micro_price * (1 - decay_slow) + ema_slow * decay_slow

    macd_trend = (new_ema_fast - new_ema_slow) / new_ema_slow

    return new_ema_fast, new_ema_slow, macd_trend
