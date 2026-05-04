import math
from typing import Optional

import pandas as pd

from base_strategy import BaseStrategy

# 依据 Notebook 物理规律设定的先验权重
PRIOR_WEIGHTS = {
    "W_rev": 1.0,  # 微观均值回归权重
    "W_trend": 0.5,  # 宏观低频漂移权重
    "W_risk": 1.65,  # 方差风险惩罚系数 (90% Confidence)
}


class OnlineEstimator:
    """依据 Lecture 12 Notebook 实现的 O(1) 在线均值与方差估计器"""

    __slots__ = ("alpha", "mean", "var")

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.mean: Optional[float] = None
        self.var: float = 0.0

    def update(self, x: float) -> None:
        if math.isnan(x):
            return

        if self.mean is None:
            self.mean = x
            self.var = 0.0
        else:
            # 公式: m_t = alpha * m_{t-1} + (1 - alpha) * x_t
            old_mean = self.mean
            self.mean = self.alpha * self.mean + (1.0 - self.alpha) * x
            # 公式: v_t = alpha * v_{t-1} + (1 - alpha) * (x_t - m_t)^2
            self.var = self.alpha * self.var + (1.0 - self.alpha) * (
                (x - self.mean) ** 2
            )

    @property
    def vol(self) -> float:
        return math.sqrt(self.var)


class Strategy(BaseStrategy):
    def __init__(self, side: str) -> None:
        super().__init__(side)
        self._current_minute: Optional[pd.Timestamp] = None
        self._executed: bool = False
        self._arrival_price: Optional[float] = None

        # 在线估计器 (跨分钟保持状态，不清空)
        self._mid_fast = OnlineEstimator(alpha=0.98)  # 捕捉高频反转
        self._mid_slow = OnlineEstimator(alpha=0.999)  # 捕捉低频趋势
        self._spread_ema = OnlineEstimator(alpha=0.99)  # 捕捉动态流动性成本

    def _reset_minute(self):
        self._executed = False
        self._arrival_price = None
        # 注意：这里绝对不能清空 OnlineEstimator，保证行情的连续性

    def on_tick(self, current_row: pd.Series, current_time: pd.Timestamp) -> bool:
        minute = current_time.floor("min")
        if minute != self._current_minute:
            self._current_minute = minute
            self._reset_minute()

        sec = current_time.second + current_time.microsecond / 1e6

        bid1 = float(current_row["BidPrice_1"])
        ask1 = float(current_row["AskPrice_1"])

        mid = (ask1 + bid1) * 0.5
        spread = ask1 - bid1

        if self._arrival_price is None:
            self._arrival_price = ask1 if self.side == "BUY" else bid1

        # 1. O(1) 在线更新状态
        self._mid_fast.update(mid)
        self._mid_slow.update(mid)
        self._spread_ema.update(spread)

        fire = False
        if not self._executed:
            # 强制执行兜底
            if sec >= 55.0:
                fire = True
            # 等待开局 2 秒以确保 Online Estimator 收集到初始方差
            elif sec >= 2.0 and self._mid_fast.mean is not None:
                # 时间衰减因子 (1.0 -> 0.0)
                tau = max(0.0, (55.0 - sec) / 53.0)

                # 动态执行门槛 (Dynamic Threshold)
                current_spread_cost = (
                    self._spread_ema.mean if self._spread_ema.mean else spread
                )
                threshold = current_spread_cost * tau

                # 提取先验参数
                w_rev = PRIOR_WEIGHTS["W_rev"]
                w_trend = PRIOR_WEIGHTS["W_trend"]
                w_risk = PRIOR_WEIGHTS["W_risk"]

                sigma_fast = self._mid_fast.vol

                # 2. 计算连续预期优势方程 (Expected Edge)
                if self.side == "BUY":
                    profit = self._arrival_price - ask1
                    micro_reversion = self._mid_fast.mean - mid
                    macro_trend = self._mid_slow.mean - self._arrival_price

                    expected_edge = (
                        profit
                        + w_rev * micro_reversion
                        + w_trend * macro_trend
                        - w_risk * sigma_fast * tau
                    )

                else:  # SELL
                    profit = bid1 - self._arrival_price
                    micro_reversion = mid - self._mid_fast.mean
                    macro_trend = self._arrival_price - self._mid_slow.mean

                    expected_edge = (
                        profit
                        + w_rev * micro_reversion
                        + w_trend * macro_trend
                        - w_risk * sigma_fast * tau
                    )

                # 3. 最优停止判决
                if expected_edge > threshold:
                    fire = True

        if fire:
            self._executed = True

        return fire
