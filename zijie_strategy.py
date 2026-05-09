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

        current_tf = compute_trade_flow(row)
        self.rolling_tfi += current_tf
        if len(self.history) == self.window:
            old_tf = self.history[0]
            self.rolling_tfi -= old_tf
        self.history.append(current_tf)

        sec = current_time.second + current_time.microsecond / 1e6

        return should_execute(
            current=row,
            rolling_tfi=self.rolling_tfi,
            side=self.side,
            macro_trend=macd_trend,
            tick_size=self.tick_size,
            avg_spread=self.avg_spread,
            current_spread=current_spread,
            sec=sec,
        )


def should_execute(
    current, rolling_tfi, side, macro_trend, tick_size, avg_spread, current_spread, sec
) -> bool:
    is_large_tick = avg_spread <= 1.5 * tick_size

    trend_adj = np.clip(macro_trend * 10, -0.2, 0.2)

    urgency = 1.0
    if sec > 40.0:
        urgency = (60.0 - sec) / 20.0

    is_tight_spread = current_spread <= avg_spread * 1.2

    if sec >= 58.0 and current_spread <= avg_spread * 1.5:
        return True

    if is_large_tick:
        # Order Book Imbalance
        obi = (current["BidSize_1"] - current["AskSize_1"]) / (
            current["BidSize_1"] + current["AskSize_1"] + 1e-9
        )
        trigger_obi = 0.6 * urgency

        if side == "BUY":
            signal_fire = obi > (trigger_obi - trend_adj)
        else:  # SELL
            signal_fire = obi < (-trigger_obi - trend_adj)

        return signal_fire and is_tight_spread

    else:  # small tick
        # Trade Flow Imbalance
        current_volume = current["BidSize_1"] + current["AskSize_1"]
        tfi_norm = rolling_tfi / current_volume if current_volume > 0 else 0.0

        trigger_tfi = 2.0 * urgency

        if side == "BUY":
            signal_fire = tfi_norm < (-trigger_tfi + trend_adj * 5)
        else:  # SELL
            signal_fire = tfi_norm > (trigger_tfi + trend_adj * 5)

        return signal_fire and is_tight_spread


def compute_trade_flow(row):
    vis = row.get("VisibleExecution_1=Yes_0=No", 0)
    hid = row.get("HiddenExecution_1=Yes_0=No", 0)

    if vis == 1 or hid == 1:
        direction = row.get("Direction_1=Buy_-1=Sell", 0)
        size = row.get("Size", 0)
        return direction * size

    return 0.0


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
