from collections import deque
import math
from typing import Optional

import numpy as np
import pandas as pd

from base_strategy import BaseStrategy


# 0. Global configuration

BASE_PARAMS: dict = {
    # Execution windows
    "min_trade_sec": 3.0,
    "search_end_sec": 50.0,
    "fallback_start_sec": 55.0,

    # Spread filter
    "max_spread_mult": 1.20,

    # Score trigger threshold
    "trigger_q_buy": 0.90,
    "trigger_q_sell": 0.95,

    # Rolling feature windows
    "short": "1s",
    "mid": "5s",

    # Count-based rolling buffers
    "roll_win": 500,
    "roll_min_periods": 30,

    # Buy score weights
    "buy_w": {
        "ask_z": -0.70,
        "spread_z": -0.80,
        "micro_edge_z": -1.00,
        "imbalance_z": 0.80,
        "ask_cancel_z": -1.00,
        "buy_exec_flow_z": -0.90,
        "quote_instability_z": -0.80,
        "ask_depth_z": 0.50,
    },

    # Sell score weights
    "sell_w": {
        "bid_z": 0.70,
        "spread_z": -0.80,
        "micro_edge_z": -1.00,
        "imbalance_z": -0.80,
        "bid_cancel_z": -1.00,
        "sell_exec_flow_z": -0.90,
        "quote_instability_z": -0.80,
        "bid_depth_z": 0.50,
    },
}


# Four separate stock-specific algorithms for the original four stocks.
# These are fixed before the final test data.
STOCK_PARAMS: dict = {
    "AMZN": {
        "min_trade_sec": 8.0,
        "trigger_q_buy": 0.90,
        "trigger_q_sell": 0.90,
        "max_spread_mult": 1.20,
    },
    "GOOG": {
        "min_trade_sec": 8.0,
        "trigger_q_buy": 0.90,
        "trigger_q_sell": 0.90,
        "max_spread_mult": 1.20,
    },
    "INTC": {
        "min_trade_sec": 15.0,
        "trigger_q_buy": 0.97,
        "trigger_q_sell": 0.99,
        "max_spread_mult": 1.05,
    },
    "MSFT": {
        "min_trade_sec": 15.0,
        "trigger_q_buy": 0.97,
        "trigger_q_sell": 0.99,
        "max_spread_mult": 1.05,
    },
}


# 1. Helper classes

class RollingBuffer:
    """
    Fixed-size rolling buffer.

    It stores only past values. Current tick values are pushed only after
    the execution decision is made. Therefore, mean/std/median/quantile are
    based only on historical information.
    """

    def __init__(self, maxlen: int) -> None:
        self._buf: deque[float] = deque(maxlen=maxlen)

    def push(self, val: float) -> None:
        try:
            val = float(val)
        except Exception:
            return

        if not math.isnan(val) and not math.isinf(val):
            self._buf.append(val)

    def __len__(self) -> int:
        return len(self._buf)

    def mean(self) -> float:
        if len(self._buf) == 0:
            return math.nan
        return float(sum(self._buf) / len(self._buf))

    def std(self) -> float:
        n = len(self._buf)
        if n < 2:
            return math.nan

        arr = list(self._buf)
        mu = sum(arr) / n
        var = sum((x - mu) ** 2 for x in arr) / (n - 1)

        if var <= 0:
            return math.nan

        return float(math.sqrt(var))

    def quantile(self, q: float) -> float:
        if len(self._buf) == 0:
            return math.nan
        return float(np.quantile(list(self._buf), q))

    def median(self) -> float:
        return self.quantile(0.5)

    def zscore(self, val: float) -> float:
        """
        Current value is standardized using historical values only.
        """
        try:
            val = float(val)
        except Exception:
            return math.nan

        mu = self.mean()
        sd = self.std()

        if math.isnan(mu) or math.isnan(sd) or sd == 0.0:
            return math.nan

        return float((val - mu) / sd)


class TimeRollingSum:
    """
    Time-based rolling sum over past ticks.

    The important pattern is:
        read current historical sum
        decide whether to execute
        push current tick after decision

    This prevents the current tick from leaking into its own features.
    """

    def __init__(self, window_sec: float) -> None:
        self._window_sec = float(window_sec)
        self._buf: deque[tuple[pd.Timestamp, float]] = deque()
        self._total: float = 0.0

    def _evict(self, current_time: pd.Timestamp) -> None:
        cutoff = current_time - pd.Timedelta(seconds=self._window_sec)

        while self._buf and self._buf[0][0] < cutoff:
            _, old_val = self._buf.popleft()
            self._total -= old_val

    def read(self, current_time: pd.Timestamp) -> float:
        self._evict(current_time)
        return float(self._total)

    def push(self, current_time: pd.Timestamp, val: float) -> None:
        try:
            val = float(val)
        except Exception:
            val = 0.0

        if math.isnan(val) or math.isinf(val):
            val = 0.0

        self._buf.append((current_time, val))
        self._total += val


# 2. Utility functions

def _parse_sec(window: str) -> float:
    window = str(window).strip().lower()

    if window.endswith("min"):
        return float(window[:-3]) * 60.0

    if window.endswith("s"):
        return float(window[:-1])

    return float(window)


def _safe_float(x, default: float = 0.0) -> float:
    try:
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return default
        return y
    except Exception:
        return default


def _merge_params(ticker: Optional[str], params_override: Optional[dict] = None) -> dict:
    """
    Merge global parameters with stock-specific parameters and optional override.

    For AMZN/GOOG/INTC/MSFT:
        use pre-fixed STOCK_PARAMS.

    For AAPL:
        params_override should be selected by the backtester using AAPL training data.
    """
    params = dict(BASE_PARAMS)

    # Deep copy weight dicts so they are not accidentally modified.
    params["buy_w"] = dict(BASE_PARAMS["buy_w"])
    params["sell_w"] = dict(BASE_PARAMS["sell_w"])

    if ticker is not None:
        ticker = ticker.upper()
        if ticker in STOCK_PARAMS:
            params.update(STOCK_PARAMS[ticker])

    if params_override is not None:
        params.update(params_override)

    return params


# 3. Adjusted Ye Strategy

class Strategy(BaseStrategy):
    """
    Adjusted Ye Strategy: Queue-Aware Toxicity Timing.

    Main design:
    1. Tick-by-tick decision.
    2. Past-only rolling z-score.
    3. Past-only spread median.
    4. Past-only score quantile.
    5. First acceptable fallback tick.
    6. Minimum waiting time to avoid over-early execution.
    7. Fixed stock-specific parameters for AMZN/GOOG/INTC/MSFT.
    8. Optional autonomous params for AAPL, selected by training data only.

    on_tick(row, current_time) returns:
        True  -> execute now
        False -> do not execute
    """

    def __init__(
        self,
        side: str,
        ticker: Optional[str] = None,
        params_override: Optional[dict] = None,
    ) -> None:
        super().__init__(side)

        self.side = side.upper()
        self.ticker = ticker.upper() if ticker is not None else None
        self.params = _merge_params(self.ticker, params_override=params_override)

        rw = self.params["roll_win"]
        mid_sec = _parse_sec(self.params["mid"])
        short_sec = _parse_sec(self.params["short"])

        # Time-based rolling sums
        self._ts_buy_exec = TimeRollingSum(mid_sec)
        self._ts_sell_exec = TimeRollingSum(mid_sec)
        self._ts_ask_cancel = TimeRollingSum(mid_sec)
        self._ts_bid_cancel = TimeRollingSum(mid_sec)
        self._ts_quote_inst = TimeRollingSum(short_sec)

        # Side-specific feature buffers and weights
        if self.side == "BUY":
            self._weights = self.params["buy_w"]
            self._feat_bufs = {
                "ask_z": RollingBuffer(rw),
                "spread_z": RollingBuffer(rw),
                "micro_edge_z": RollingBuffer(rw),
                "imbalance_z": RollingBuffer(rw),
                "ask_cancel_z": RollingBuffer(rw),
                "buy_exec_flow_z": RollingBuffer(rw),
                "quote_instability_z": RollingBuffer(rw),
                "ask_depth_z": RollingBuffer(rw),
            }

        elif self.side == "SELL":
            self._weights = self.params["sell_w"]
            self._feat_bufs = {
                "bid_z": RollingBuffer(rw),
                "spread_z": RollingBuffer(rw),
                "micro_edge_z": RollingBuffer(rw),
                "imbalance_z": RollingBuffer(rw),
                "bid_cancel_z": RollingBuffer(rw),
                "sell_exec_flow_z": RollingBuffer(rw),
                "quote_instability_z": RollingBuffer(rw),
                "bid_depth_z": RollingBuffer(rw),
            }

        else:
            raise ValueError("side must be either BUY or SELL")

        # Historical score and spread buffers
        self._rb_score = RollingBuffer(rw)
        self._rb_spread = RollingBuffer(rw)

        # Minute-level state
        self._current_minute: Optional[pd.Timestamp] = None
        self._executed: bool = False

        # Previous quote for quote instability
        self._prev_bid1: Optional[float] = None
        self._prev_ask1: Optional[float] = None

    def on_tick(self, current_row: pd.Series, current_time: pd.Timestamp) -> bool:
        # 0. Minute rollover
        minute = current_time.floor("min")

        if minute != self._current_minute:
            self._current_minute = minute
            self._executed = False

        if self._executed:
            return False

        sec = current_time.second + current_time.microsecond / 1e6

        # 1. Current LOB values
        bid1 = _safe_float(current_row["BidPrice_1"])
        ask1 = _safe_float(current_row["AskPrice_1"])
        bsz1 = _safe_float(current_row["BidSize_1"])
        asz1 = _safe_float(current_row["AskSize_1"])

        spread = ask1 - bid1
        mid = 0.5 * (ask1 + bid1)

        micro = (ask1 * bsz1 + bid1 * asz1) / (bsz1 + asz1 + 1e-9)
        micro_edge = micro - mid

        bid_depth_5 = sum(
            _safe_float(current_row.get(f"BidSize_{i}", 0.0))
            for i in range(1, 6)
        )
        ask_depth_5 = sum(
            _safe_float(current_row.get(f"AskSize_{i}", 0.0))
            for i in range(1, 6)
        )

        imbalance_5 = (bid_depth_5 - ask_depth_5) / (
            bid_depth_5 + ask_depth_5 + 1e-9
        )

        quote_change = int(
            self._prev_bid1 is not None
            and (
                bid1 != self._prev_bid1
                or ask1 != self._prev_ask1
            )
        )

        # 2. Event information
        visible_exec = _safe_float(
            current_row.get("VisibleExecution_1=Yes_0=No", 0.0)
        )
        hidden_exec = _safe_float(
            current_row.get("HiddenExecution_1=Yes_0=No", 0.0)
        )
        partial_cancel = _safe_float(
            current_row.get("PartialCancel_1=Yes_0=No", 0.0)
        )
        full_delete = _safe_float(
            current_row.get("FullDelete_1=Yes_0=No", 0.0)
        )
        direction = _safe_float(
            current_row.get("Direction_1=Buy_-1=Sell", 0.0)
        )
        size = _safe_float(current_row.get("Size", 0.0))

        cancel_flag = (partial_cancel == 1) or (full_delete == 1)

        total_exec_size = (visible_exec + hidden_exec) * size

        buy_exec_flow = total_exec_size if direction == 1 else 0.0
        sell_exec_flow = total_exec_size if direction == -1 else 0.0

        # Practical cancellation-side proxy.
        ask_cancel = size if (cancel_flag and direction == -1) else 0.0
        bid_cancel = size if (cancel_flag and direction == 1) else 0.0

        # 3. Read past rolling sums before updating current tick
        buy_exec_sum = self._ts_buy_exec.read(current_time)
        sell_exec_sum = self._ts_sell_exec.read(current_time)
        ask_cancel_sum = self._ts_ask_cancel.read(current_time)
        bid_cancel_sum = self._ts_bid_cancel.read(current_time)
        quote_inst_sum = self._ts_quote_inst.read(current_time)

        # 4. Assemble features
        if self.side == "BUY":
            feat_vals = {
                "ask_z": ask1,
                "spread_z": spread,
                "micro_edge_z": micro_edge,
                "imbalance_z": imbalance_5,
                "ask_cancel_z": ask_cancel_sum,
                "buy_exec_flow_z": buy_exec_sum,
                "quote_instability_z": quote_inst_sum,
                "ask_depth_z": asz1,
            }
        else:
            feat_vals = {
                "bid_z": bid1,
                "spread_z": spread,
                "micro_edge_z": micro_edge,
                "imbalance_z": imbalance_5,
                "bid_cancel_z": bid_cancel_sum,
                "sell_exec_flow_z": sell_exec_sum,
                "quote_instability_z": quote_inst_sum,
                "bid_depth_z": bsz1,
            }

        # 5. Score using past-only z-scores
        score = 0.0
        all_features_warm = True

        for key, val in feat_vals.items():
            z = self._feat_bufs[key].zscore(val)

            if math.isnan(z):
                all_features_warm = False
                z = 0.0

            score += self._weights[key] * z

        # 6. Read past-only thresholds
        min_periods = self.params["roll_min_periods"]

        warm_enough = (
            all_features_warm
            and len(self._rb_score) >= min_periods
            and len(self._rb_spread) >= min_periods
        )

        if self.side == "BUY":
            trigger_q = self.params["trigger_q_buy"]
        else:
            trigger_q = self.params["trigger_q_sell"]

        score_threshold = self._rb_score.quantile(trigger_q)
        spread_cap = self._rb_spread.median() * self.params["max_spread_mult"]

        # 7. Online execution decision
        fire = False

        min_trade_sec = self.params["min_trade_sec"]
        search_end = self.params["search_end_sec"]
        fallback_start = self.params["fallback_start_sec"]

        if sec < min_trade_sec:
            fire = False

        elif sec <= search_end:
            if (
                warm_enough
                and not math.isnan(score_threshold)
                and not math.isnan(spread_cap)
                and spread <= spread_cap
                and score >= score_threshold
            ):
                fire = True

        elif sec >= fallback_start:
            # No hindsight selection.
            # Execute at first acceptable fallback tick.
            if warm_enough and not math.isnan(spread_cap):
                if spread <= spread_cap:
                    fire = True
            else:
                fire = True

        # 8. Update buffers after decision
        self._ts_buy_exec.push(current_time, buy_exec_flow)
        self._ts_sell_exec.push(current_time, sell_exec_flow)
        self._ts_ask_cancel.push(current_time, ask_cancel)
        self._ts_bid_cancel.push(current_time, bid_cancel)
        self._ts_quote_inst.push(current_time, quote_change)

        for key, val in feat_vals.items():
            self._feat_bufs[key].push(val)

        self._rb_score.push(score)
        self._rb_spread.push(spread)

        self._prev_bid1 = bid1
        self._prev_ask1 = ask1

        if fire:
            self._executed = True

        return fire