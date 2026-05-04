import math
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from base_strategy import BaseStrategy

# Global parameters modified for Classification / Ranking Scheme
PARAMS: dict = {
    "force_exec_sec": 55.0,
    "min_wait_sec": 3.0,
    "roll_win": 500,
    "roll_min_periods": 30,
    "short": "1s",
    "mid": "5s",
    # ML Parameters
    "top_k_pct": 15,
    "learning_rate": 0.01,
    "l2_reg": 0.001,
    "sgd_epochs": 2,
    "consecutive_bins": 2,
}

FEATURE_KEYS = [
    "price_z",
    "spread_z",
    "micro_edge_z",
    "imbalance_z",
    "cancel_flow_z",
    "exec_flow_z",
    "quote_instability_z",
    "depth_z",
]
N_FEAT = len(FEATURE_KEYS)


class O1RollingBuffer:
    __slots__ = ("_buf", "_maxlen", "_sum", "_sum_sq")

    def __init__(self, maxlen: int) -> None:
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self._sum = 0.0
        self._sum_sq = 0.0

    def push(self, val: float) -> None:
        if math.isnan(val):
            return

        self._buf.append(val)
        self._sum += val
        self._sum_sq += val**2

        if len(self._buf) > self._maxlen:
            old_val = self._buf.popleft()
            self._sum -= old_val
            self._sum_sq -= old_val**2

    def __len__(self) -> int:
        return len(self._buf)

    def mean(self) -> float:
        n = len(self._buf)
        if n == 0:
            return math.nan
        return self._sum / n

    def std(self) -> float:
        n = len(self._buf)
        if n < 2:
            return math.nan
        var = (self._sum_sq - (self._sum**2) / n) / (n - 1)
        return math.sqrt(max(0.0, var))

    def zscore(self, val: float) -> float:
        mu = self.mean()
        sd = self.std()
        if math.isnan(mu) or math.isnan(sd) or sd < 1e-8:
            return 0.0
        return (val - mu) / sd


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


def _parse_sec(w: str) -> float:
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

        self._ts_exec = TimeRollingSum(mid_sec)
        self._ts_cancel = TimeRollingSum(mid_sec)
        self._ts_quote_inst = TimeRollingSum(sht_sec)

        self._feat_bufs: dict[str, O1RollingBuffer] = {
            k: O1RollingBuffer(rw) for k in FEATURE_KEYS
        }

        self.w = np.zeros(N_FEAT)
        self.b = 0.0

        self._minute_data = []
        self._score_buf = deque(maxlen=rw)
        self._consecutive_hits = 0

        self._current_minute: Optional[pd.Timestamp] = None
        self._executed: bool = False
        self._prev_bid1: Optional[float] = None
        self._prev_ask1: Optional[float] = None

    def _train_model(self):
        if len(self._minute_data) < 20:
            return

        prices = np.array([data[1] for data in self._minute_data])

        k_pct = PARAMS["top_k_pct"]
        if self.side == "BUY":
            threshold = np.percentile(prices, k_pct)
            labels = (prices <= threshold).astype(int)
        else:
            threshold = np.percentile(prices, 100 - k_pct)
            labels = (prices >= threshold).astype(int)

        w_1 = 1.0 / (k_pct / 100.0)
        w_0 = 1.0 / (1.0 - k_pct / 100.0)

        lr = PARAMS["learning_rate"]
        l2 = PARAMS["l2_reg"]

        for _ in range(PARAMS["sgd_epochs"]):
            indices = np.random.permutation(len(self._minute_data))
            for i in indices:
                z, _ = self._minute_data[i]
                y = labels[i]

                logit = np.dot(self.w, z) + self.b
                logit = max(min(logit, 20), -20)
                y_hat = 1.0 / (1.0 + math.exp(-logit))

                error = y_hat - y
                weight = w_1 if y == 1 else w_0
                grad = weight * error

                self.w -= lr * (grad * z + l2 * self.w)
                self.b -= lr * grad

    def _reset_minute(self):
        if self._minute_data:
            self._train_model()

        self._minute_data.clear()
        self._consecutive_hits = 0
        self._executed = False

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

        if self.side == "BUY":
            exec_flow = total_exec if direction == 1 else 0.0
            cancel_flow = size if (cancel_flag and direction == -1) else 0.0
            price_lvl = ask1
            depth = asz1
            exec_price = ask1
        else:
            exec_flow = total_exec if direction == -1 else 0.0
            cancel_flow = size if (cancel_flag and direction == 1) else 0.0
            price_lvl = bid1
            depth = bsz1
            exec_price = bid1

        exec_sum = self._ts_exec.read(current_time)
        cancel_sum = self._ts_cancel.read(current_time)
        quote_inst_sum = self._ts_quote_inst.read(current_time)

        feat_raw_vals = {
            "price_z": price_lvl,
            "spread_z": spread,
            "micro_edge_z": micro_edge,
            "imbalance_z": imb5,
            "cancel_flow_z": cancel_sum,
            "exec_flow_z": exec_sum,
            "quote_instability_z": quote_inst_sum,
            "depth_z": depth,
        }

        z_list = []
        all_warm = True
        for k in FEATURE_KEYS:
            buf = self._feat_bufs[k]
            if len(buf) < PARAMS["roll_min_periods"]:
                all_warm = False
            z_list.append(buf.zscore(feat_raw_vals[k]))

        z_vec = np.array(z_list)

        self._minute_data.append((z_vec.copy(), exec_price))

        s_t = float(np.dot(self.w, z_vec) + self.b)
        self._score_buf.append(s_t)

        fire = False
        if not self._executed:
            if sec < PARAMS["min_wait_sec"]:
                self._consecutive_hits = 0
            elif sec >= PARAMS["force_exec_sec"]:
                fire = True
            elif all_warm and len(self._score_buf) >= PARAMS["roll_min_periods"]:
                q_85 = np.percentile(self._score_buf, 85)

                if s_t > q_85:
                    self._consecutive_hits += 1
                else:
                    self._consecutive_hits = 0

                if self._consecutive_hits >= PARAMS["consecutive_bins"]:
                    fire = True

        if fire:
            self._executed = True

        self._ts_exec.push(current_time, exec_flow)
        self._ts_cancel.push(current_time, cancel_flow)
        self._ts_quote_inst.push(current_time, quote_chg)

        for k, v in feat_raw_vals.items():
            self._feat_bufs[k].push(v)

        self._prev_bid1 = bid1
        self._prev_ask1 = ask1

        return fire
