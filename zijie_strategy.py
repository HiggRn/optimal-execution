import math
from collections import deque
from typing import Optional

import pandas as pd

from base_strategy import BaseStrategy


class TimeRollingSum:
    __slots__ = ("_window_sec", "_buf", "_total")

    def __init__(self, window_sec: float) -> None:
        self._window_sec = window_sec
        self._buf: deque[tuple[pd.Timestamp, float]] = deque()
        self._total: float = 0.0

    def _evict(self, current_time: pd.Timestamp) -> None:
        cutoff = current_time - pd.Timedelta(seconds=self._window_sec)
        while self._buf and self._buf[0][0] < cutoff:
            _, v = self._buf.popleft()
            self._total -= v

    def read(self, current_time: pd.Timestamp) -> float:
        self._evict(current_time)
        return self._total

    def push(self, current_time: pd.Timestamp, val: float) -> None:
        self._buf.append((current_time, val))
        self._total += val


class Strategy(BaseStrategy):
    def __init__(self, side: str) -> None:
        super().__init__(side)
        self._current_minute: Optional[pd.Timestamp] = None
        self._executed: bool = False
        self._arrival_price: Optional[float] = None

        self._obi_buf = deque(maxlen=100)
        self._spread_buf = deque(maxlen=100)
        self._mid_buf = deque(maxlen=50)

        self._buy_flow = TimeRollingSum(3.0)
        self._sell_flow = TimeRollingSum(3.0)

    def _reset_minute(self):
        self._executed = False
        self._arrival_price = None
        self._obi_buf.clear()
        self._spread_buf.clear()
        self._mid_buf.clear()

    def on_tick(self, current_row: pd.Series, current_time: pd.Timestamp) -> bool:
        minute = current_time.floor("min")
        if minute != self._current_minute:
            self._current_minute = minute
            self._reset_minute()

        sec = current_time.second + current_time.microsecond / 1e6

        bid1 = float(current_row["BidPrice_1"])
        ask1 = float(current_row["AskPrice_1"])
        bsz1 = float(current_row["BidSize_1"])
        asz1 = float(current_row["AskSize_1"])

        if self._arrival_price is None:
            self._arrival_price = ask1 if self.side == "BUY" else bid1

        spread = ask1 - bid1
        mid = (ask1 + bid1) * 0.5
        self._spread_buf.append(spread)
        self._mid_buf.append(mid)

        micro = (ask1 * bsz1 + bid1 * asz1) / (bsz1 + asz1 + 1e-9)
        micro_edge = micro - mid

        vis_exec = float(current_row.get("VisibleExecution_1=Yes_0=No", 0) or 0)
        hid_exec = float(current_row.get("HiddenExecution_1=Yes_0=No", 0) or 0)
        direction = float(current_row.get("Direction_1=Buy_-1=Sell", 0) or 0)
        size = float(current_row.get("Size", 0) or 0)

        is_exec = (vis_exec == 1.0) or (hid_exec == 1.0)
        exec_vol = size if is_exec else 0.0

        self._buy_flow.push(current_time, exec_vol if direction == 1.0 else 0.0)
        self._sell_flow.push(current_time, exec_vol if direction == -1.0 else 0.0)

        bid_v = sum(float(current_row.get(f"BidSize_{i}", 0) or 0) for i in range(1, 6))
        ask_v = sum(float(current_row.get(f"AskSize_{i}", 0) or 0) for i in range(1, 6))
        obi = (bid_v - ask_v) / (bid_v + ask_v + 1e-9)
        self._obi_buf.append(obi)

        fire = False
        if not self._executed:
            if sec >= 55.0:
                fire = True
            elif sec >= 2.0 and len(self._obi_buf) >= 30:
                obi_mean = sum(self._obi_buf) / len(self._obi_buf)
                variance = sum((x - obi_mean) ** 2 for x in self._obi_buf) / len(
                    self._obi_buf
                )
                obi_std = math.sqrt(variance) + 1e-9
                obi_z = (obi - obi_mean) / obi_std

                local_mid_mean = sum(self._mid_buf) / len(self._mid_buf)
                is_downward_trend = local_mid_mean < self._arrival_price

                is_fav_regime = (self.side == "BUY" and is_downward_trend) or (
                    self.side == "SELL" and not is_downward_trend
                )

                curr_p = ask1 if self.side == "BUY" else bid1
                profit_usd = (
                    (self._arrival_price - curr_p)
                    if self.side == "BUY"
                    else (curr_p - self._arrival_price)
                )

                time_fraction_left = max(0.0, (55.0 - sec) / 53.0)
                local_spread_mean = sum(self._spread_buf) / len(self._spread_buf)

                if self.side == "BUY":
                    fav_obi_z = obi_z
                    fav_micro_edge = micro_edge
                else:
                    fav_obi_z = -obi_z
                    fav_micro_edge = -micro_edge

                if is_fav_regime:
                    expected_profit_usd = 1.5 * local_spread_mean * time_fraction_left
                    reversion_signal = (fav_obi_z > 0.5) and (fav_micro_edge > 0)
                    panic_signal = fav_obi_z > 1.5
                else:
                    expected_profit_usd = 0.5 * local_spread_mean * time_fraction_left
                    reversion_signal = (fav_obi_z < 0.8) and (fav_micro_edge < 0.5)
                    panic_signal = fav_obi_z < 0.0

                if profit_usd >= expected_profit_usd and reversion_signal:
                    fire = True
                elif profit_usd < 0 and panic_signal:
                    fire = True

        if fire:
            self._executed = True

        return fire
