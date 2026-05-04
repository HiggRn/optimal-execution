import math
from collections import deque
from typing import Optional

import pandas as pd

from base_strategy import BaseStrategy


class Strategy(BaseStrategy):
    def __init__(self, side: str) -> None:
        super().__init__(side)
        self._current_minute: Optional[pd.Timestamp] = None
        self._executed: bool = False
        self._arrival_price: Optional[float] = None

        # 仅维护一个用于计算局部 Z-score 的滚动窗口
        self._obi_buf = deque(maxlen=100)

    def _reset_minute(self):
        self._executed = False
        self._arrival_price = None
        self._obi_buf.clear()

    def on_tick(self, current_row: pd.Series, current_time: pd.Timestamp) -> bool:
        minute = current_time.floor("min")
        if minute != self._current_minute:
            self._current_minute = minute
            self._reset_minute()

        sec = current_time.second + current_time.microsecond / 1e6

        bid1 = float(current_row["BidPrice_1"])
        ask1 = float(current_row["AskPrice_1"])

        if self._arrival_price is None:
            self._arrival_price = ask1 if self.side == "BUY" else bid1

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

                curr_p = ask1 if self.side == "BUY" else bid1
                profit = (
                    (self._arrival_price - curr_p)
                    if self.side == "BUY"
                    else (curr_p - self._arrival_price)
                )

                if self.side == "BUY":
                    if profit > 0 and obi_z > 0.5:
                        fire = True
                    elif profit <= 0 and obi_z > 1.5:
                        fire = True
                else:  # SELL
                    if profit > 0 and obi_z < -0.5:
                        fire = True
                    elif profit <= 0 and obi_z < -1.5:
                        fire = True

        if fire:
            self._executed = True

        return fire
