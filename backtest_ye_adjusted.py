import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from ye_adjusted import Strategy, STOCK_PARAMS


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"


# 1. Data loading

def find_dataset_file(ticker: str, split: str) -> Optional[Path]:
    """
    Find dataset file flexibly.

    Expected final format:
        data/AMZN_5levels_train.csv
        data/AMZN_5levels_test.csv
        data/AAPL_5levels_train.csv
        data/AAPL_5levels_test.csv

    Also supports simple variants.
    """
    ticker = ticker.upper()
    split = split.lower()

    candidates = [
        DATA_DIR / f"{ticker}_5levels_{split}.csv",
        DATA_DIR / f"{ticker}_5levels_{split}(1).csv",
        DATA_DIR / f"{ticker}_{split}.csv",
        DATA_DIR / f"{ticker}_{split}(1).csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    # final fallback: glob
    patterns = [
        f"{ticker}*{split}*.csv",
        f"{ticker.lower()}*{split}*.csv",
    ]

    for pattern in patterns:
        matches = sorted(DATA_DIR.glob(pattern))
        if matches:
            return matches[0]

    return None


def load_lob_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Time" not in df.columns:
        raise ValueError(f"{path} does not contain a Time column.")

    time_raw = df["Time"].astype(str)

    # Most files use HH:MM:SS.microsecond.
    time_dt = pd.to_datetime(
        time_raw,
        format="%H:%M:%S.%f",
        errors="coerce",
    )

    # Fallback if the file has a full datetime string or no milliseconds.
    if time_dt.isna().mean() > 0.50:
        time_dt = pd.to_datetime(time_raw, errors="coerce")

    df["Time_dt"] = time_dt
    df = df.dropna(subset=["Time_dt"]).copy()
    df = df.sort_values("Time_dt")
    df.set_index("Time_dt", inplace=True)
    df["Minute"] = df.index.floor("min")

    return df


# 2. Backtest engine

def run_backtest(
    df: pd.DataFrame,
    strategy_class,
    ticker: str,
    params_override: Optional[dict] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Tick-by-tick backtest.

    For each stock:
        run BUY strategy
        run SELL strategy

    Each minute:
        execute exactly once per side.
    """
    results = []

    for side in ["BUY", "SELL"]:
        strat = strategy_class(
            side,
            ticker=ticker,
            params_override=params_override,
        )

        grouped = df.groupby("Minute")

        iterator = tqdm(
            grouped,
            desc=f"{ticker} {side}",
            disable=not show_progress,
        )

        for minute, grp in iterator:
            if len(grp) == 0:
                continue

            twap_price = (
                float(grp["AskPrice_1"].iloc[0])
                if side == "BUY"
                else float(grp["BidPrice_1"].iloc[0])
            )

            exec_price = None
            exec_time = None
            executed = False

            for idx, row in grp.iterrows():
                if strat.on_tick(row, idx):
                    if not executed:
                        exec_price = (
                            float(row["AskPrice_1"])
                            if side == "BUY"
                            else float(row["BidPrice_1"])
                        )
                        exec_time = idx
                        executed = True

            # If no signal fires within the minute, force execution at last tick.
            if exec_price is None:
                last_row = grp.iloc[-1]
                exec_price = (
                    float(last_row["AskPrice_1"])
                    if side == "BUY"
                    else float(last_row["BidPrice_1"])
                )
                exec_time = grp.index[-1]

            improvement = (
                twap_price - exec_price
                if side == "BUY"
                else exec_price - twap_price
            )

            optm_price = (
                float(grp["AskPrice_1"].min())
                if side == "BUY"
                else float(grp["BidPrice_1"].max())
            )

            results.append(
                {
                    "Ticker": ticker,
                    "Minute": minute,
                    "Side": side,
                    "TWAP_Price": twap_price,
                    "Exec_Price": exec_price,
                    "Optm_Price": optm_price,
                    "Improvement": improvement,
                    "Improvement_bps": improvement / twap_price * 10000,
                    "Optm_Improvement_bps": abs(optm_price - twap_price)
                    / twap_price
                    * 10000,
                    "Exec_Time": exec_time,
                    "Exec_Second": exec_time.second
                    + exec_time.microsecond / 1_000_000,
                    "Forced": exec_time == grp.index[-1],
                }
            )

    return pd.DataFrame(results)


# 3. Metrics

def final_percentage_improvement(result_df: pd.DataFrame) -> dict:
    """
    Required metric:

    100 - 100 * (TOTAL_YOURALGO_BUY - TOTAL_YOURALGO_SELL)
                / (TOTAL_TWAP_BUY - TOTAL_TWAP_SELL)

    Interpretation:
        If algorithm = TWAP, improvement = 0%.
        If algorithm cost is 30% of TWAP cost, improvement = 70%.
    """
    buy_df = result_df[result_df["Side"] == "BUY"]
    sell_df = result_df[result_df["Side"] == "SELL"]

    total_algo_buy = buy_df["Exec_Price"].sum()
    total_algo_sell = sell_df["Exec_Price"].sum()

    total_twap_buy = buy_df["TWAP_Price"].sum()
    total_twap_sell = sell_df["TWAP_Price"].sum()

    algo_cost = total_algo_buy - total_algo_sell
    twap_cost = total_twap_buy - total_twap_sell

    if pd.isna(twap_cost) or abs(twap_cost) < 1e-12:
        pct_improvement = np.nan
    else:
        pct_improvement = 100.0 - 100.0 * algo_cost / twap_cost

    return {
        "TOTAL_YOURALGO_BUY": total_algo_buy,
        "TOTAL_YOURALGO_SELL": total_algo_sell,
        "TOTAL_TWAP_BUY": total_twap_buy,
        "TOTAL_TWAP_SELL": total_twap_sell,
        "YOURALGO_COST": algo_cost,
        "TWAP_COST": twap_cost,
        "Pct_Improvement": pct_improvement,
    }


def side_summary(result_df: pd.DataFrame, side: str) -> dict:
    side_df = result_df[result_df["Side"] == side].copy()
    imp_series = side_df["Improvement_bps"].dropna()

    if len(imp_series) > 1:
        t_stat, p_val = stats.ttest_1samp(imp_series, 0)
    else:
        t_stat, p_val = np.nan, np.nan

    return {
        f"{side}_Mean_bps": imp_series.mean(),
        f"{side}_Std_bps": imp_series.std(),
        f"{side}_T_Statistic": t_stat,
        f"{side}_P_Value": p_val,
        f"{side}_WinRate": (imp_series > 0).mean(),
        f"{side}_ForcedRate": side_df["Forced"].mean(),
        f"{side}_AvgExecSecond": side_df["Exec_Second"].mean(),
    }


def summarize_result(result_df: pd.DataFrame, ticker: str, split: str) -> dict:
    final_metric = final_percentage_improvement(result_df)

    summary = {
        "Ticker": ticker,
        "Split": split,
        "NumBuyMinutes": int((result_df["Side"] == "BUY").sum()),
        "NumSellMinutes": int((result_df["Side"] == "SELL").sum()),
    }

    summary.update(final_metric)
    summary.update(side_summary(result_df, "BUY"))
    summary.update(side_summary(result_df, "SELL"))

    return summary


# 4. AAPL autonomous parameter selection

def generate_aapl_candidate_params() -> list[dict]:
    """
    Candidate parameter sets for AAPL.

    This is autonomous because the code evaluates these candidates on AAPL
    training data and selects the best one automatically. No parameter is
    hand-tuned specifically after looking at AAPL test data.
    """
    candidates = []

    # Start with the four existing stock algorithms.
    for source_ticker, params in STOCK_PARAMS.items():
        candidate = dict(params)
        candidate["candidate_name"] = f"copy_{source_ticker}"
        candidates.append(candidate)

    # Add moderate variants.
    for min_trade_sec in [5.0, 8.0, 12.0, 15.0]:
        for trigger_q in [0.90, 0.95, 0.97]:
            for spread_mult in [1.05, 1.10, 1.20]:
                candidates.append(
                    {
                        "candidate_name": (
                            f"grid_wait{min_trade_sec}_q{trigger_q}_spr{spread_mult}"
                        ),
                        "min_trade_sec": min_trade_sec,
                        "trigger_q_buy": trigger_q,
                        "trigger_q_sell": trigger_q,
                        "max_spread_mult": spread_mult,
                    }
                )

    # Add sell-more-conservative variants.
    for min_trade_sec in [8.0, 12.0, 15.0]:
        for q_buy, q_sell in [(0.90, 0.95), (0.95, 0.97), (0.97, 0.99)]:
            candidates.append(
                {
                    "candidate_name": (
                        f"grid_wait{min_trade_sec}_qb{q_buy}_qs{q_sell}"
                    ),
                    "min_trade_sec": min_trade_sec,
                    "trigger_q_buy": q_buy,
                    "trigger_q_sell": q_sell,
                    "max_spread_mult": 1.10,
                }
            )

    return candidates


def clean_params_for_strategy(params: dict) -> dict:
    """
    Remove metadata keys before passing params into Strategy.
    """
    clean = dict(params)
    clean.pop("candidate_name", None)
    return clean


def select_aapl_params_from_training(aapl_train_df: pd.DataFrame) -> dict:
    """
    Select AAPL parameters using AAPL training data only.

    Objective:
        maximize the required final percentage improvement metric on AAPL train.
    """
    candidates = generate_aapl_candidate_params()

    best_score = -np.inf
    best_params = None
    all_rows = []

    print("\nSelecting AAPL autonomous parameters from AAPL training data only...")

    for i, candidate in enumerate(candidates, start=1):
        params_override = clean_params_for_strategy(candidate)

        result_df = run_backtest(
            aapl_train_df,
            Strategy,
            ticker="AAPL",
            params_override=params_override,
            show_progress=False,
        )

        metric = final_percentage_improvement(result_df)
        score = metric["Pct_Improvement"]

        row = {
            "RankInput": i,
            "Candidate": candidate.get("candidate_name", f"candidate_{i}"),
            "Pct_Improvement": score,
            "YOURALGO_COST": metric["YOURALGO_COST"],
            "TWAP_COST": metric["TWAP_COST"],
            **params_override,
        }
        all_rows.append(row)

        if pd.notna(score) and score > best_score:
            best_score = score
            best_params = dict(candidate)

    if best_params is None:
        # Conservative fallback if all candidates fail.
        best_params = {
            "candidate_name": "fallback_base",
            "min_trade_sec": 8.0,
            "trigger_q_buy": 0.90,
            "trigger_q_sell": 0.95,
            "max_spread_mult": 1.20,
        }

    selection_df = pd.DataFrame(all_rows).sort_values(
        "Pct_Improvement",
        ascending=False,
    )
    selection_path = RESULTS_DIR / "aapl_param_selection.csv"
    selection_df.to_csv(selection_path, index=False)

    selected_strategy_params = clean_params_for_strategy(best_params)

    selected_json = {
        "selected_candidate": best_params.get("candidate_name", "unknown"),
        "selected_train_pct_improvement": best_score,
        "selected_params": selected_strategy_params,
        "note": "Selected automatically using AAPL training data only.",
    }

    with open(RESULTS_DIR / "aapl_selected_params.json", "w") as f:
        json.dump(selected_json, f, indent=2)

    print("AAPL selected candidate:", selected_json["selected_candidate"])
    print("AAPL selected params:", selected_strategy_params)
    print(f"AAPL parameter search saved to {selection_path}")

    return selected_strategy_params


# 5. Main routine

def run_one_dataset(
    ticker: str,
    split: str,
    params_override: Optional[dict] = None,
) -> Optional[dict]:
    path = find_dataset_file(ticker, split)

    if path is None:
        print(f"[Skip] No {split} file found for {ticker}.")
        return None

    print(f"\nBacktesting {ticker} {split} with ye_adjusted.py ...")
    print(f"Using file: {path}")

    df = load_lob_csv(path)

    result_df = run_backtest(
        df,
        Strategy,
        ticker=ticker,
        params_override=params_override,
        show_progress=True,
    )

    output_name = RESULTS_DIR / f"{ticker}_{split}_ye_adjusted_backtest_result.csv"
    result_df.to_csv(output_name, index=False)

    summary = summarize_result(result_df, ticker, split)

    print(f">>> {ticker} {split} results saved to {output_name}.")
    print(
        f"Required Metric | {ticker} {split}: "
        f"{summary['Pct_Improvement']:.4f}% improvement"
    )
    print(
        f"BUY  mean: {summary['BUY_Mean_bps']:.4f} bps | "
        f"SELL mean: {summary['SELL_Mean_bps']:.4f} bps"
    )

    return summary


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)

    fixed_tickers = ["AMZN", "GOOG", "INTC", "MSFT"]
    all_summary_rows = []

    # 1. Run four fixed stock-specific algorithms
    for ticker in fixed_tickers:
        for split in ["train", "test"]:
            summary = run_one_dataset(
                ticker=ticker,
                split=split,
                params_override=None,
            )
            if summary is not None:
                all_summary_rows.append(summary)

    # 2. AAPL autonomous algorithm
    aapl_train_path = find_dataset_file("AAPL", "train")

    aapl_params = None

    if aapl_train_path is not None:
        print("\nFound AAPL training data.")
        aapl_train_df = load_lob_csv(aapl_train_path)

        # Select AAPL params from training data only.
        aapl_params = select_aapl_params_from_training(aapl_train_df)

        # Report AAPL train performance using selected params.
        summary = run_one_dataset(
            ticker="AAPL",
            split="train",
            params_override=aapl_params,
        )
        if summary is not None:
            all_summary_rows.append(summary)

        # Report AAPL test performance using the same selected params.
        summary = run_one_dataset(
            ticker="AAPL",
            split="test",
            params_override=aapl_params,
        )
        if summary is not None:
            all_summary_rows.append(summary)

    else:
        print(
            "\n[Warning] No AAPL training data found. "
            "AAPL autonomous parameter selection was skipped."
        )

    # 3. Save final summary
    if all_summary_rows:
        summary_df = pd.DataFrame(all_summary_rows)

        # Sort summary in desired order.
        ticker_order = {"AMZN": 0, "GOOG": 1, "INTC": 2, "MSFT": 3, "AAPL": 4}
        split_order = {"train": 0, "test": 1}

        summary_df["_ticker_order"] = summary_df["Ticker"].map(ticker_order)
        summary_df["_split_order"] = summary_df["Split"].map(split_order)
        summary_df = summary_df.sort_values(
            ["_ticker_order", "_split_order"]
        ).drop(columns=["_ticker_order", "_split_order"])

        summary_path = RESULTS_DIR / "final_performance_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\nAll ye_adjusted backtests completed.")
        print(f"Final summary saved to {summary_path}")
        print(
            summary_df[
                [
                    "Ticker",
                    "Split",
                    "Pct_Improvement",
                    "TOTAL_YOURALGO_BUY",
                    "TOTAL_YOURALGO_SELL",
                    "TOTAL_TWAP_BUY",
                    "TOTAL_TWAP_SELL",
                    "YOURALGO_COST",
                    "TWAP_COST",
                    "BUY_Mean_bps",
                    "SELL_Mean_bps",
                ]
            ].round(4)
        )
    else:
        print("\nNo datasets were found. Please check the data folder.")