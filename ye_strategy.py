from collections import deque
import math
from typing import Optional

import numpy as np
import pandas as pd

from base_strategy import BaseStrategy

PARAMS: dict = {
    "search_end_sec": 50,
    "fallback_start_sec": 55,
    "max_spread_mult": 1.20,
    "trigger_q_buy": 0.85,
    "trigger_q_sell": 0.85,
    "short": "1s",
    "mid": "5s",
    "long": "15s",
    "roll_win": 500,  # count-based window for normalisation buffers
    "roll_min_periods": 30,  # min history before threshold is trusted
    "buy_w": {
        "ask_z": -1.00,
        "spread_z": -0.60,
        "micro_edge_z": -0.80,
        "imbalance_z": 0.70,
        "ask_cancel_z": -0.80,
        "buy_exec_flow_z": -0.60,
        "quote_instability_z": -0.50,
        "ask_depth_z": 0.40,
    },
    "sell_w": {
        "bid_z": 1.00,
        "spread_z": -0.60,
        "micro_edge_z": -0.80,
        "imbalance_z": -0.70,
        "bid_cancel_z": -0.80,
        "sell_exec_flow_z": -0.60,
        "quote_instability_z": -0.50,
        "bid_depth_z": 0.40,
    },
}


class RollingBuffer:
    """
    Fixed-size circular buffer exposing mean, std, quantile, median, and
    z-score over the last `maxlen` values pushed.

    All reads are O(n); fine for n=500 at LOB tick rates.
    """

    __slots__ = ("_buf", "_maxlen")

    def __init__(self, maxlen: int) -> None:
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._maxlen = maxlen

    def push(self, val: float) -> None:
        if not math.isnan(val):
            self._buf.append(val)

    def __len__(self) -> int:
        return len(self._buf)

    def mean(self) -> float:
        if not self._buf:
            return math.nan
        return sum(self._buf) / len(self._buf)

    def std(self) -> float:
        n = len(self._buf)
        if n < 2:
            return math.nan
        arr = list(self._buf)
        mu = sum(arr) / n
        return math.sqrt(sum((x - mu) ** 2 for x in arr) / (n - 1))

    def quantile(self, q: float) -> float:
        if not self._buf:
            return math.nan
        return float(np.quantile(list(self._buf), q))

    def median(self) -> float:
        return self.quantile(0.5)

    def zscore(self, val: float) -> float:
        """z-score of val against the buffer's current distribution."""
        mu = self.mean()
        sd = self.std()
        if math.isnan(mu) or math.isnan(sd) or sd == 0.0:
            return math.nan
        return (val - mu) / sd


class TimeRollingSum:
    """
    Maintains a running sum of (timestamp, value) pairs within a trailing
    time window of `window_sec` seconds.

    Mirrors pandas rolling(time_str).sum() but operates one event at a time.

    The read-before-push pattern replicates the .shift(1) used in the original
    feature engineering:
        val = trs.read(t)    # sum of past ticks in window — used for decision
        trs.push(t, x)       # record current tick — used by future decisions
    """

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
        """Past-data sum: evicts stale entries, returns total (current tick NOT included)."""
        self._evict(current_time)
        return self._total

    def push(self, current_time: pd.Timestamp, val: float) -> None:
        """Record current tick (always call AFTER read/decision for this tick)."""
        self._buf.append((current_time, val))
        self._total += val


def _parse_sec(w: str) -> float:
    """Convert a pandas offset string like '5s' or '1min' to seconds."""
    w = w.strip()
    if w.endswith("min"):
        return float(w[:-3]) * 60.0
    if w.endswith("s"):
        return float(w[:-1])
    return float(w)


class Strategy(BaseStrategy):
    def __init__(self, side: str) -> None:
        super().__init__(side)

        rw = PARAMS["roll_win"]
        mid_sec = _parse_sec(PARAMS["mid"])
        sht_sec = _parse_sec(PARAMS["short"])

        # Time-based rolling sums
        self._ts_buy_exec = TimeRollingSum(mid_sec)
        self._ts_sell_exec = TimeRollingSum(mid_sec)
        self._ts_ask_cancel = TimeRollingSum(mid_sec)
        self._ts_bid_cancel = TimeRollingSum(mid_sec)
        self._ts_quote_inst = TimeRollingSum(sht_sec)

        # Per-feature normalisation buffers (keyed by weight-dict key for convenience)
        if side == "BUY":
            self._weights = PARAMS["buy_w"]
            self._feat_bufs: dict[str, RollingBuffer] = {
                "ask_z": RollingBuffer(rw),  # AskPrice_1
                "spread_z": RollingBuffer(rw),  # Spread
                "micro_edge_z": RollingBuffer(rw),  # MicroEdge
                "imbalance_z": RollingBuffer(rw),  # Imbalance_5
                "ask_cancel_z": RollingBuffer(rw),  # AskCancel time-sum
                "buy_exec_flow_z": RollingBuffer(rw),  # BuyExecFlow time-sum
                "quote_instability_z": RollingBuffer(rw),  # QuoteInstability time-sum
                "ask_depth_z": RollingBuffer(rw),  # AskSize_1
            }
        else:
            self._weights = PARAMS["sell_w"]
            self._feat_bufs = {
                "bid_z": RollingBuffer(rw),  # BidPrice_1
                "spread_z": RollingBuffer(rw),  # Spread
                "micro_edge_z": RollingBuffer(rw),  # MicroEdge
                "imbalance_z": RollingBuffer(rw),  # Imbalance_5
                "bid_cancel_z": RollingBuffer(rw),  # BidCancel time-sum
                "sell_exec_flow_z": RollingBuffer(rw),  # SellExecFlow time-sum
                "quote_instability_z": RollingBuffer(rw),  # QuoteInstability time-sum
                "bid_depth_z": RollingBuffer(rw),  # BidSize_1
            }

        self._rb_score = RollingBuffer(rw)
        self._rb_spread = RollingBuffer(rw)

        # Minute-level state
        self._current_minute: Optional[pd.Timestamp] = None
        self._executed: bool = False
        self._prev_bid1: Optional[float] = None
        self._prev_ask1: Optional[float] = None

    # ------------------------------------------------------------------

    def on_tick(self, current_row: pd.Series, current_time: pd.Timestamp) -> bool:

        # ── 0. Minute rollover ─────────────────────────────────────────────
        minute = current_time.floor("min")
        if minute != self._current_minute:
            self._current_minute = minute
            self._executed = False

        if self._executed:
            return False

        sec = current_time.second + current_time.microsecond / 1e6

        # ── 1. Raw LOB values from the current row ─────────────────────────
        bid1 = float(current_row["BidPrice_1"])
        ask1 = float(current_row["AskPrice_1"])
        bsz1 = float(current_row["BidSize_1"])
        asz1 = float(current_row["AskSize_1"])

        spread = ask1 - bid1
        mid = (ask1 + bid1) * 0.5
        micro = (ask1 * bsz1 + bid1 * asz1) / (bsz1 + asz1 + 1e-9)
        micro_edge = micro - mid

        bid_d5 = sum(
            float(current_row.get(f"BidSize_{i}", 0) or 0) for i in range(1, 6)
        )
        ask_d5 = sum(
            float(current_row.get(f"AskSize_{i}", 0) or 0) for i in range(1, 6)
        )
        imb5 = (bid_d5 - ask_d5) / (bid_d5 + ask_d5 + 1e-9)

        quote_chg = int(
            self._prev_bid1 is not None
            and (bid1 != self._prev_bid1 or ask1 != self._prev_ask1)
        )

        vis_exec = float(current_row.get("VisibleExecution_1=Yes_0=No", 0) or 0)
        hid_exec = float(current_row.get("HiddenExecution_1=Yes_0=No", 0) or 0)
        p_cancel = float(current_row.get("PartialCancel_1=Yes_0=No", 0) or 0)
        f_delete = float(current_row.get("FullDelete_1=Yes_0=No", 0) or 0)
        direction = float(current_row.get("Direction_1=Buy_-1=Sell", 0) or 0)
        size = float(current_row.get("Size", 0) or 0)

        cancel_flag = (p_cancel == 1) or (f_delete == 1)
        total_exec = (vis_exec + hid_exec) * size
        buy_exec_flow = total_exec if direction == 1 else 0.0
        sell_exec_flow = total_exec if direction == -1 else 0.0
        ask_cancel = size if (cancel_flag and direction == -1) else 0.0
        bid_cancel = size if (cancel_flag and direction == 1) else 0.0

        # ── 2. READ time-based rolling sums (before this tick) ─────────────
        #    Mirrors rolling(time_str).sum().shift(1) from the original.
        buy_exec_sum = self._ts_buy_exec.read(current_time)
        sell_exec_sum = self._ts_sell_exec.read(current_time)
        ask_cancel_sum = self._ts_ask_cancel.read(current_time)
        bid_cancel_sum = self._ts_bid_cancel.read(current_time)
        quote_inst_sum = self._ts_quote_inst.read(current_time)

        # ── 3. Assemble feature vector for this side ───────────────────────
        if self.side == "BUY":
            feat_vals: dict[str, float] = {
                "ask_z": ask1,
                "spread_z": spread,
                "micro_edge_z": micro_edge,
                "imbalance_z": imb5,
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
                "imbalance_z": imb5,
                "bid_cancel_z": bid_cancel_sum,
                "sell_exec_flow_z": sell_exec_sum,
                "quote_instability_z": quote_inst_sum,
                "bid_depth_z": bsz1,
            }

        # ── 4. Compute score using past-only feature buffers ───────────────
        score = 0.0
        all_warm = True
        for key, val in feat_vals.items():
            z = self._feat_bufs[key].zscore(val)
            if math.isnan(z):
                all_warm = False
                z = 0.0
            score += self._weights[key] * z

        # ── 5. READ threshold and spread cap from past-only buffers ────────
        mp = PARAMS["roll_min_periods"]
        warm_enough = (
            all_warm and len(self._rb_score) >= mp and len(self._rb_spread) >= mp
        )

        tq = PARAMS[f"trigger_q_{self.side.lower()}"]
        thresh = self._rb_score.quantile(tq)
        spread_cap = self._rb_spread.median() * PARAMS["max_spread_mult"]

        # ── 6. Execution decision ──────────────────────────────────────────
        search_end = PARAMS["search_end_sec"]
        fallback_start = PARAMS["fallback_start_sec"]
        fire = False

        if sec <= search_end:
            # Cold start guard: skip search window until buffers are reliable.
            if (
                warm_enough
                and not math.isnan(thresh)
                and not math.isnan(spread_cap)
                and spread <= spread_cap
                and score >= thresh
            ):
                fire = True

        elif sec >= fallback_start:
            # Execute on the first tick in the fallback window with a tight
            # spread.  No lookahead into the rest of the window.
            ref = spread_cap if (warm_enough and not math.isnan(spread_cap)) else spread
            if spread <= ref:
                fire = True

        # ── 7. UPDATE all buffers with current tick (always after decision) ─
        self._ts_buy_exec.push(current_time, buy_exec_flow)
        self._ts_sell_exec.push(current_time, sell_exec_flow)
        self._ts_ask_cancel.push(current_time, ask_cancel)
        self._ts_bid_cancel.push(current_time, bid_cancel)
        self._ts_quote_inst.push(current_time, quote_chg)

        for key, val in feat_vals.items():
            self._feat_bufs[key].push(val)
        self._rb_score.push(score)
        self._rb_spread.push(spread)

        self._prev_bid1 = bid1
        self._prev_ask1 = ask1

        if fire:
            self._executed = True
        return fire
