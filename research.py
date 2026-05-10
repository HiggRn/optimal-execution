import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import deque
from tqdm import tqdm

TICKERS = ["AMZN", "GOOG", "INTC", "MSFT"]
TICK_SIZE = 0.01


def enrich_features(df):
    df = df.copy()
    spread_history = deque(maxlen=100)
    rolling_tfi = 0.0
    history = deque(maxlen=100)

    median_spread = None
    tick_count = 0
    records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Enriching Features"):
        current_time = idx
        current_spread = row["AskPrice_1"] - row["BidPrice_1"]

        spread_history.append(current_spread)
        tick_count += 1
        if median_spread is None or tick_count % 10 == 0:
            sorted_spreads = sorted(spread_history)
            median_spread = sorted_spreads[len(sorted_spreads) // 2]

        total_size = row["BidSize_1"] + row["AskSize_1"]
        if total_size == 0:
            obi = np.nan
            tfi_norm = np.nan
        else:
            obi = (row["BidSize_1"] - row["AskSize_1"]) / total_size

            vis = row.get("VisibleExecution_1=Yes_0=No", 0)
            hid = row.get("HiddenExecution_1=Yes_0=No", 0)
            tfi = 0.0
            if vis == 1 or hid == 1:
                direction = row.get("Direction_1=Buy_-1=Sell", 0)
                size = row.get("Size", 0)
                tfi = (direction * size) / total_size

            rolling_tfi += tfi
            if len(history) == history.maxlen:
                rolling_tfi -= history[0]
            history.append(tfi)
            tfi_norm = rolling_tfi

        is_large_tick = median_spread <= 1.5 * TICK_SIZE

        records.append(
            {
                "time": current_time,
                "mid_price": (row["AskPrice_1"] + row["BidPrice_1"]) / 2.0,
                "spread": current_spread,
                "median_spread": median_spread,
                "is_large_tick": is_large_tick,
                "obi": obi,
                "tfi_norm": tfi_norm,
            }
        )

    feat_df = pd.DataFrame(records).set_index("time")
    return feat_df.dropna(subset=["obi", "tfi_norm"])


def compute_forward_returns(feat_df, horizon_seconds=1.0):
    unique_mid = feat_df["mid_price"].groupby(level=0).last()

    forward_time = feat_df.index + pd.Timedelta(seconds=horizon_seconds)
    target_idx = unique_mid.index.get_indexer(forward_time, method="nearest")
    future_mid = pd.Series(unique_mid.iloc[target_idx].values, index=feat_df.index)

    fwd_ret_bps = (future_mid - feat_df["mid_price"]) / feat_df["mid_price"] * 10000
    return fwd_ret_bps


def analyze_signal_rigorous(df_sub, signal_col, ticker, regime_name, horizon=1.0):
    df_sub = df_sub.copy()
    df_sub["fwd_ret"] = compute_forward_returns(df_sub, horizon_seconds=horizon)
    df_sub = df_sub.dropna(subset=[signal_col, "fwd_ret"])

    if len(df_sub) < 100:
        return

    # 1. Information Coefficient (IC)
    ic, p_val = stats.spearmanr(df_sub[signal_col], df_sub["fwd_ret"])
    print(
        f"[{ticker} - {regime_name}] {signal_col.upper()} {horizon}s Forward IC: {ic:.4f} (p-value: {p_val:.4e})"
    )

    # 2. Decile Analysis
    try:
        df_sub["decile"] = pd.qcut(
            df_sub[signal_col], q=10, labels=False, duplicates="drop"
        )
        decile_returns = df_sub.groupby("decile")["fwd_ret"].mean()

        plt.figure(figsize=(8, 5))
        decile_returns.plot(kind="bar", color="steelblue", edgecolor="black")
        plt.title(
            f"{ticker} {regime_name}: {signal_col.upper()} Deciles vs {horizon}s Forward Return"
        )
        plt.xlabel(f"{signal_col.upper()} Deciles (Low -> High)")
        plt.ylabel(f"Mean {horizon}s Forward Return (bps)")
        plt.axhline(0, color="red", linewidth=1, linestyle="--")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plots/{ticker}_{regime_name}_{signal_col}_deciles.png")
        plt.close()
    except ValueError:
        print(
            f"  -> Skipping decile plot for {ticker} due to insufficient unique values."
        )

    # 3. T-Test for Extreme Thresholds
    threshold = df_sub[signal_col].quantile(0.90)
    neg_threshold = df_sub[signal_col].quantile(0.10)

    buy_signals = df_sub[df_sub[signal_col] > threshold]["fwd_ret"]
    sell_signals = df_sub[df_sub[signal_col] < neg_threshold]["fwd_ret"]

    if len(buy_signals) > 0 and len(sell_signals) > 0:
        t_stat, t_pval = stats.ttest_ind(buy_signals, sell_signals, equal_var=False)
        print(
            f"  -> Top 10% vs Bottom 10% T-Test: T-stat = {t_stat:.4f}, p-val = {t_pval:.4e}"
        )
        print(f"  -> Top 10% Mean Ret: {buy_signals.mean():.4f} bps")
        print(f"  -> Bottom 10% Mean Ret: {sell_signals.mean():.4f} bps")


def analyze_event_study(feat_df, signal_col, ticker, regime_name, is_buy_signal=True):
    if is_buy_signal:
        threshold = feat_df[signal_col].quantile(0.95)
        triggers = feat_df[feat_df[signal_col] > threshold].index
    else:
        threshold = feat_df[signal_col].quantile(0.05)
        triggers = feat_df[feat_df[signal_col] < threshold].index

    if len(triggers) == 0:
        return

    if len(triggers) > 1000:
        triggers = np.random.choice(triggers, 1000, replace=False)

    unique_mid = feat_df["mid_price"].groupby(level=0).last()

    offsets = np.arange(-1.0, 2.1, 0.1)
    paths = []

    for t in triggers:
        sample_times = t + pd.to_timedelta(offsets, unit="s")
        idx = unique_mid.index.get_indexer(sample_times, method="nearest")
        prices = unique_mid.iloc[idx].values

        t0_idx = np.where(np.isclose(offsets, 0.0))[0][0]
        base_price = prices[t0_idx]

        if base_price > 0:
            path_bps = (prices - base_price) / base_price * 10000
            paths.append(path_bps)

    if paths:
        mean_path = np.nanmean(paths, axis=0)
        std_err = np.nanstd(paths, axis=0) / np.sqrt(len(paths))

        plt.figure(figsize=(8, 5))
        plt.plot(
            offsets,
            mean_path,
            label="Mean Mid Price Path (bps)",
            color="darkred",
            linewidth=2,
        )
        plt.fill_between(
            offsets,
            mean_path - 1.96 * std_err,
            mean_path + 1.96 * std_err,
            color="red",
            alpha=0.2,
        )
        plt.axvline(0, color="black", linestyle="--", label="Signal Trigger (T=0)")
        plt.axhline(0, color="gray", linewidth=1)

        direction = "Buy" if is_buy_signal else "Sell"
        plt.title(
            f"{ticker} {regime_name}: Event Study around Extreme {direction} {signal_col.upper()}"
        )
        plt.xlabel("Time relative to trigger (seconds)")
        plt.ylabel("Mid Price Change (bps)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            f"plots/{ticker}_{regime_name}_{signal_col}_{direction}_event_study.png"
        )
        plt.close()


def run_research():
    os.makedirs("plots", exist_ok=True)

    for ticker in TICKERS:
        path = f"data/{ticker}_5levels_train.csv"
        if not os.path.exists(path):
            print(f"Data not found for {ticker}, skipping.")
            continue

        print(f"\n{'=' * 20} Processing {ticker} {'=' * 20}")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]

        df["Time_dt"] = pd.to_datetime(df["Time"], format="%H:%M:%S.%f")
        df = df.sort_values("Time_dt")
        df.set_index("Time_dt", inplace=True)

        print("1. Enriching Features...")
        feat_df = enrich_features(df)

        print("2. Analyzing Large Tick Regime (OBI)...")
        large_tick_df = feat_df[feat_df["is_large_tick"]]
        if not large_tick_df.empty:
            analyze_signal_rigorous(
                large_tick_df, "obi", ticker, "LargeTick", horizon=1.0
            )
            analyze_event_study(
                large_tick_df, "obi", ticker, "LargeTick", is_buy_signal=True
            )

        print("3. Analyzing Small Tick Regime (TFI)...")
        small_tick_df = feat_df[~feat_df["is_large_tick"]]
        if not small_tick_df.empty:
            analyze_signal_rigorous(
                small_tick_df, "tfi_norm", ticker, "SmallTick", horizon=1.0
            )
            analyze_event_study(
                small_tick_df, "tfi_norm", ticker, "SmallTick", is_buy_signal=True
            )


if __name__ == "__main__":
    run_research()
