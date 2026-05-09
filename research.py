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

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Enriching Features"):
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


def detect_next_mid_change_events(df):
    mid = (df["BidPrice_1"] + df["AskPrice_1"]) / 2.0

    mid_reset = mid.reset_index(drop=True)

    changed_mid = mid_reset.loc[mid_reset.diff() != 0]

    next_actual_mid = changed_mid.shift(-1).reindex(mid_reset.index).bfill()

    price_move = next_actual_mid.values - mid.values
    return pd.Series(np.sign(price_move), index=df.index)


# 1. Large/Small Tick
def analyze_tick_regime(feat_df, ticker):
    regime = feat_df["is_large_tick"]
    time_diff = feat_df.index.to_series().diff().shift(-1).dt.total_seconds().fillna(0)

    large_duration = time_diff[regime].sum()
    total_duration = time_diff.sum()
    large_ratio = large_duration / total_duration if total_duration > 0 else 0

    print(f"[{ticker}] Large tick regime time ratio: {large_ratio:.2%}")
    if large_ratio < 0.05 or large_ratio > 0.95:
        print(f"  -> Warning: {ticker} 几乎单一 Regime，可能不适合双轨策略。")

    plt.figure(figsize=(10, 5))
    plt.hist(
        feat_df["median_spread"],
        bins=np.arange(0, 0.05, 0.001),
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )
    plt.axvline(
        1.5 * TICK_SIZE, color="red", linestyle="--", label=f"Threshold (1.5 tick)"
    )
    plt.title(f"{ticker} Median Spread Distribution & Regime Threshold")
    plt.xlabel("Median Spread")
    plt.ylabel("Tick Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{ticker}_regime_spread.png")
    plt.close()


# 2. MACD
def analyze_macd_trend(feat_df, df, ticker):
    mid = (df["BidPrice_1"] + df["AskPrice_1"]) / 2.0

    unique_mid = mid.groupby(level=0).last()

    forward_times = feat_df.index + pd.Timedelta(seconds=30)

    target_idx = unique_mid.index.get_indexer(forward_times, method="nearest")
    target_mids = pd.Series(unique_mid.iloc[target_idx].values, index=feat_df.index)

    forward_return = (target_mids - mid) / mid * 10000  # 转换为 bps

    valid_mask = forward_return.notna() & feat_df["macd"].notna()
    macd_vals = feat_df.loc[valid_mask, "macd"]
    fwd_ret_vals = forward_return[valid_mask]

    bins = pd.qcut(macd_vals, q=10, duplicates="drop")
    grouped_ret = fwd_ret_vals.groupby(bins).mean()

    plt.figure(figsize=(10, 5))
    grouped_ret.plot(kind="bar", color="darkorange")
    plt.title(f"{ticker} MACD Deciles vs 30s Forward Mid-Price Return (bps)")
    plt.xlabel("MACD Deciles")
    plt.ylabel("Mean 30s Forward Return (bps)")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/{ticker}_macd_trend.png")
    plt.close()


# 3. Large Tick: OBI (Accuracy vs Count Tradeoff)
def analyze_obi(feat_df, price_move_sign, ticker):
    mask = feat_df["is_large_tick"]
    if mask.sum() < 100:
        return

    df_sub = feat_df[mask].copy()
    y_true = price_move_sign[mask].values

    thresholds = np.linspace(0.1, 0.9, 30)
    acc_list = []
    count_list = []

    for th in thresholds:
        signal = df_sub["obi"] > th
        total_signals = signal.sum()

        if total_signals > 50:
            correct = (y_true == 1)[signal].sum()
            acc_list.append(correct / total_signals)
        else:
            acc_list.append(np.nan)
        count_list.append(total_signals)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(
        thresholds, acc_list, "b-", marker="o", label="Accuracy (L1 Clear Direction)"
    )
    ax1.set_xlabel("OBI Threshold (Buy Signal > th)")
    ax1.set_ylabel("Predictive Accuracy", color="b")
    ax1.tick_params("y", colors="b")
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, count_list, "r--", marker="x", label="Trade Count")
    ax2.set_ylabel("Number of Signals", color="r")
    ax2.tick_params("y", colors="r")

    plt.title(f"{ticker} Large Tick: OBI Predictive Power Tradeoff")
    fig.tight_layout()
    plt.savefig(f"plots/{ticker}_obi_tradeoff.png")
    plt.close()


# 4. Small Tick: TFI (Accuracy vs Count Tradeoff)
def analyze_tfi(feat_df, price_move_sign, ticker):
    mask = ~feat_df["is_large_tick"]
    if mask.sum() < 100:
        return

    df_sub = feat_df[mask].copy()
    y_true = price_move_sign[mask].values

    thresholds = np.linspace(0.5, 5.0, 30)
    acc_list = []
    count_list = []

    for th in thresholds:
        signal = df_sub["tfi_norm"] < -th
        total_signals = signal.sum()

        if total_signals > 50:
            correct = (y_true == -1)[signal].sum()
            acc_list.append(correct / total_signals)
        else:
            acc_list.append(np.nan)
        count_list.append(total_signals)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(thresholds, acc_list, "g-", marker="o", label="Accuracy (Mid Move Down)")
    ax1.set_xlabel("TFI Threshold (Signal < -th)")
    ax1.set_ylabel("Predictive Accuracy", color="g")
    ax1.tick_params("y", colors="g")
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, count_list, "r--", marker="x", label="Trade Count")
    ax2.set_ylabel("Number of Signals", color="r")
    ax2.tick_params("y", colors="r")

    plt.title(f"{ticker} Small Tick: TFI (Normalized) Predictive Power Tradeoff")
    fig.tight_layout()
    plt.savefig(f"plots/{ticker}_tfi_tradeoff.png")
    plt.close()


# 5. Spread Filter
def analyze_spread_filter(feat_df, df, ticker):
    mask = ~feat_df["is_large_tick"]
    if mask.sum() < 100:
        return

    df_sub = feat_df[mask].copy()
    bad_cond = df_sub["spread"] > df_sub["safe_spread"] + 1e-5

    mid = (df["BidPrice_1"] + df["AskPrice_1"]) / 2.0
    unique_mid = mid.groupby(level=0).last()

    future_time = df_sub.index + pd.Timedelta(seconds=1)
    target_idx = unique_mid.index.get_indexer(future_time, method="nearest")
    future_mid = pd.Series(unique_mid.iloc[target_idx].values, index=df_sub.index)

    current_mid = (df_sub["bid"] + df_sub["ask"]) / 2.0
    abs_price_move = (future_mid - current_mid).abs()

    mean_risk_good = abs_price_move[~bad_cond].mean()
    mean_risk_bad = abs_price_move[bad_cond].mean()

    print(f"[{ticker}] Spread Filter Risk Analysis (Mean Abs Price Move in 1s):")
    print(f"  -> Spread Normal (Trade Allowed): {mean_risk_good:.5f}")
    print(f"  -> Spread Expanded (Trade Blocked): {mean_risk_bad:.5f}")

    plt.figure(figsize=(10, 5))
    plot_good = abs_price_move[~bad_cond].clip(upper=0.1)
    plot_bad = abs_price_move[bad_cond].clip(upper=0.1)

    plt.hist(
        plot_good,
        bins=50,
        density=True,
        histtype="step",
        cumulative=True,
        label="Normal Spread",
        color="blue",
    )
    plt.hist(
        plot_bad,
        bins=50,
        density=True,
        histtype="step",
        cumulative=True,
        label="Expanded Spread (Blocked)",
        color="red",
    )

    plt.title(f"{ticker} CDF of 1s Absolute Price Move (Risk Profile)")
    plt.xlabel("Absolute Price Move in 1 second")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{ticker}_spread_filter_cdf.png")
    plt.close()


def run_research():
    os.makedirs("plots", exist_ok=True)

    for ticker in TICKERS:
        path = f"data/{ticker}_5levels_train.csv"
        if not os.path.exists(path):
            print(f"Data not found for {ticker}, skipping.")
            continue

        print(f"\n================ Processing {ticker} ================")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]

        df["Time_dt"] = pd.to_datetime(df["Time"], format="%H:%M:%S.%f")
        df = df.sort_values("Time_dt")
        df.set_index("Time_dt", inplace=True)

        print("1. Enriching Features (Running MACD, OBI, TFI state machines)...")
        feat_df = enrich_features(df)

        print("2. Detecting Forward Labels...")
        price_move_sign = detect_next_mid_change_events(df)

        print("3. Analyzing Regime Validity...")
        analyze_tick_regime(feat_df, ticker)

        print("4. Analyzing MACD Macro Trend...")
        analyze_macd_trend(feat_df, df, ticker)

        print("5. Analyzing OBI Tradeoffs (Large Tick)...")
        analyze_obi(feat_df, price_move_sign, ticker)

        print("6. Analyzing TFI Tradeoffs (Small Tick)...")
        analyze_tfi(feat_df, price_move_sign, ticker)

        print("7. Analyzing Spread Filter Execution Logic...")
        analyze_spread_filter(feat_df, df, ticker)


if __name__ == "__main__":
    run_research()
