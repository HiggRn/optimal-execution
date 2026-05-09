import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

from zijie_strategy import compute_trade_flow, update_macd_trend


TICKERS = ["AMZN", "GOOG", "INTC", "MSFT"]
TICK_SIZE = 0.01


def enrich_features(df):
    df = df.copy()

    ema_fast = None
    ema_slow = None
    prev_time = None

    spread_history = deque(maxlen=100)
    rolling_tfi = 0.0
    history = deque(maxlen=100)

    median_spread = None
    tick_count = 0

    records = []

    for idx, row in df.iterrows():
        current_time = idx
        current_spread = row["AskPrice_1"] - row["BidPrice_1"]

        # === median spread ===
        spread_history.append(current_spread)
        tick_count += 1
        if median_spread is None or tick_count % 10 == 0:
            sorted_spreads = sorted(spread_history)
            median_spread = sorted_spreads[len(sorted_spreads) // 2]

        # === MACD ===
        ema_fast, ema_slow, macd_trend = update_macd_trend(
            row, current_time, prev_time, ema_fast, ema_slow
        )
        prev_time = current_time

        # === TFI ===
        current_tf = compute_trade_flow(row)
        rolling_tfi += current_tf
        if len(history) == history.maxlen:
            rolling_tfi -= history[0]
        history.append(current_tf)

        # === derived ===
        current_volume = row["BidSize_1"] + row["AskSize_1"]
        tfi_norm = rolling_tfi / current_volume if current_volume > 0 else 0.0

        obi = (row["BidSize_1"] - row["AskSize_1"]) / (
            row["BidSize_1"] + row["AskSize_1"] + 1e-9
        )

        is_large_tick = median_spread <= 1.5 * TICK_SIZE

        sec = current_time.second + current_time.microsecond / 1e6

        safe_spread = max(2.0 * TICK_SIZE, median_spread)

        records.append(
            {
                "time": current_time,
                "spread": current_spread,
                "median_spread": median_spread,
                "is_large_tick": is_large_tick,
                "obi": obi,
                "tfi_norm": tfi_norm,
                "macd": macd_trend,
                "sec": sec,
                "safe_spread": safe_spread,
                "bid": row["BidPrice_1"],
                "ask": row["AskPrice_1"],
            }
        )

    feat_df = pd.DataFrame(records).set_index("time")
    return feat_df


# 1. large vs small tick
def analyze_tick_regime(df, feat_df, ticker):
    regime = feat_df["is_large_tick"]

    time_diff = feat_df.index.to_series().diff().shift(-1)
    duration = time_diff.dt.total_seconds().fillna(0)

    large_ratio = duration[regime].sum() / duration.sum()

    print(f"{ticker} large tick ratio: {large_ratio:.3f}")

    plt.figure()
    regime.astype(int).rolling(1000).mean().plot()
    plt.title(f"{ticker} Large Tick Regime (rolling)")
    plt.savefig(f"plots/{ticker}_regime.png")

    # spread vs threshold
    plt.figure()
    plt.hist(feat_df["spread"], bins=100, alpha=0.5, label="spread")
    plt.axvline(1.5 * TICK_SIZE, color="red", label="1.5 tick")
    plt.title(f"{ticker} Spread Distribution")
    plt.legend()
    plt.savefig(f"plots/{ticker}_spread_hist.png")


# event: L1 emptied
def detect_l1_clear_events(df):
    bid = df["BidPrice_1"]
    ask = df["AskPrice_1"]

    next_bid = bid.shift(-1)
    next_ask = ask.shift(-1)

    mid = (bid + ask) / 2
    next_mid = (next_bid + next_ask) / 2

    price_move = next_mid - mid

    return price_move


# 3. large tick: OBI
def analyze_obi(feat_df, price_move, ticker):
    mask = feat_df["is_large_tick"]

    df = feat_df[mask].copy()
    df["price_move"] = price_move.values[mask.values]

    thresholds = np.linspace(0.3, 0.9, 20)
    acc = []

    for th in thresholds:
        signal = df["obi"] > th
        correct = df["price_move"] > 0

        if signal.sum() > 0:
            acc.append((signal & correct).sum() / signal.sum())
        else:
            acc.append(np.nan)

    plt.figure()
    plt.plot(thresholds, acc)
    plt.title(f"{ticker} OBI Predictive Power")
    plt.savefig(f"plots/{ticker}_obi.png")


# 4. small tick: TFI
def analyze_tfi(feat_df, price_move, ticker):
    mask = ~feat_df["is_large_tick"]

    df = feat_df[mask].copy()
    df["price_move"] = price_move.values[mask.values]

    thresholds = np.linspace(0.5, 3.0, 20)
    acc = []

    for th in thresholds:
        signal = df["tfi_norm"] > th
        correct = df["price_move"] > 0

        if signal.sum() > 0:
            acc.append((signal & correct).sum() / signal.sum())
        else:
            acc.append(np.nan)

    plt.figure()
    plt.plot(thresholds, acc)
    plt.title(f"{ticker} TFI Predictive Power")
    plt.savefig(f"plots/{ticker}_tfi.png")


# 5. 58s guard
def analyze_58s_guard(feat_df, price_move, ticker):
    mask = (~feat_df["is_large_tick"]) & (feat_df["sec"] >= 58)

    df = feat_df[mask].copy()
    df["price_move"] = price_move.values[mask.values]

    cond = df["spread"] <= (df["safe_spread"] + TICK_SIZE + 1e-5)

    good = df[cond]["price_move"].mean()
    bad = df[~cond]["price_move"].mean()

    print(f"{ticker} 58s guard: good={good}, bad={bad}")


# 6. spread filter
def analyze_spread_filter(feat_df, price_move, ticker):
    mask = ~feat_df["is_large_tick"]

    df = feat_df[mask].copy()
    df["price_move"] = price_move.values[mask.values]

    bad_cond = df["spread"] > df["safe_spread"] + 1e-5

    good = df[~bad_cond]["price_move"].mean()
    bad = df[bad_cond]["price_move"].mean()

    print(f"{ticker} spread filter: good={good}, bad={bad}")


def run_research():
    os.makedirs("plots", exist_ok=True)

    for ticker in TICKERS:
        path = f"data/{ticker}_5levels_train.csv"
        if not os.path.exists(path):
            continue

        print(f"Processing {ticker}...")

        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]

        df["Time_dt"] = pd.to_datetime(df["Time"], format="%H:%M:%S.%f")
        df = df.sort_values("Time_dt")
        df.set_index("Time_dt", inplace=True)

        feat_df = enrich_features(df)
        price_move = detect_l1_clear_events(df)

        analyze_tick_regime(df, feat_df, ticker)
        analyze_obi(feat_df, price_move, ticker)
        analyze_tfi(feat_df, price_move, ticker)
        analyze_58s_guard(feat_df, price_move, ticker)
        analyze_spread_filter(feat_df, price_move, ticker)


if __name__ == "__main__":
    run_research()
