import os

import pandas as pd
from scipy import stats
from tqdm import tqdm

from ye_adjusted import Strategy


def run_backtest(df, strategy_class, ticker):
    results = []

    for side in ["BUY", "SELL"]:
        strat = strategy_class(side, ticker=ticker)

        for minute, grp in tqdm(df.groupby("Minute"), desc=f"{ticker} {side}"):
            twap_price = (
                grp["AskPrice_1"].iloc[0]
                if side == "BUY"
                else grp["BidPrice_1"].iloc[0]
            )

            exec_price = None
            exec_time = None
            executed = False

            for idx, row in grp.iterrows():
                if strat.on_tick(row, idx):
                    if not executed:
                        exec_price = (
                            row["AskPrice_1"]
                            if side == "BUY"
                            else row["BidPrice_1"]
                        )
                        exec_time = idx
                        executed = True

            # If no signal fires within the minute, force execution at last tick.
            if exec_price is None:
                last_row = grp.iloc[-1]
                exec_price = (
                    last_row["AskPrice_1"]
                    if side == "BUY"
                    else last_row["BidPrice_1"]
                )
                exec_time = grp.index[-1]

            improvement = (
                twap_price - exec_price
                if side == "BUY"
                else exec_price - twap_price
            )

            optm_price = (
                grp["AskPrice_1"].min()
                if side == "BUY"
                else grp["BidPrice_1"].max()
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


def summarize_one_side(result_df, ticker, side):
    side_df = result_df[result_df["Side"] == side]
    imp_series = side_df["Improvement_bps"].dropna()

    if len(imp_series) > 1:
        mean_imp = imp_series.mean()
        std_imp = imp_series.std()
        t_stat, p_val = stats.ttest_1samp(imp_series, 0)
        win_rate = (imp_series > 0).mean()
        forced_rate = side_df["Forced"].mean()
        avg_exec_second = side_df["Exec_Second"].mean()
    else:
        mean_imp = 0.0
        std_imp = 0.0
        t_stat = 0.0
        p_val = 1.0
        win_rate = 0.0
        forced_rate = 0.0
        avg_exec_second = 0.0

    return {
        "Ticker": ticker,
        "Side": side,
        "Mean_bps": mean_imp,
        "Std_bps": std_imp,
        "T_Statistic": t_stat,
        "P_Value": p_val,
        "WinRate": win_rate,
        "ForcedRate": forced_rate,
        "AvgExecSecond": avg_exec_second,
    }


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    tickers = ["AMZN", "GOOG", "INTC", "MSFT"]

    summary_data = []

    for ticker in tickers:
        file_path = f"data/{ticker}_5levels_train.csv"

        if not os.path.exists(file_path):
            print(f"File '{file_path}' doesn't exist.")
            continue

        print(f"Backtesting {ticker} with ye_adjusted.py ...")

        df = pd.read_csv(file_path)
        df.columns = [c.strip() for c in df.columns]

        df["Time_dt"] = pd.to_datetime(
            df["Time"],
            format="%H:%M:%S.%f",
            errors="coerce",
        )

        df = df.dropna(subset=["Time_dt"])
        df = df.sort_values("Time_dt")
        df.set_index("Time_dt", inplace=True)
        df["Minute"] = df.index.floor("min")

        result_df = run_backtest(df, Strategy, ticker=ticker)

        output_name = f"results/{ticker}_ye_adjusted_backtest_result.csv"
        result_df.to_csv(output_name, index=False)

        print(f">>> {ticker} results saved to {output_name}.")

        for side in ["BUY", "SELL"]:
            summary = summarize_one_side(result_df, ticker, side)
            summary_data.append(summary)

            print(
                f" [{side:4s}] "
                f"Mean: {summary['Mean_bps']:>7.4f} bps | "
                f"Std: {summary['Std_bps']:>6.4f} | "
                f"T-Stat: {summary['T_Statistic']:>7.4f} "
                f"(p-val: {summary['P_Value']:>6.4f}) | "
                f"WinRate: {summary['WinRate']:>6.4f} | "
                f"AvgSec: {summary['AvgExecSecond']:>6.2f}"
            )

        print("-" * 80)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Add overall by side
        overall_rows = []
        for side in ["BUY", "SELL"]:
            side_rows = []
            for ticker in tickers:
                path = f"results/{ticker}_ye_adjusted_backtest_result.csv"
                if os.path.exists(path):
                    tmp = pd.read_csv(path)
                    side_rows.append(tmp[tmp["Side"] == side])

            if side_rows:
                all_side_df = pd.concat(side_rows, ignore_index=True)
                imp_series = all_side_df["Improvement_bps"].dropna()

                if len(imp_series) > 1:
                    t_stat, p_val = stats.ttest_1samp(imp_series, 0)
                else:
                    t_stat, p_val = 0.0, 1.0

                overall_rows.append(
                    {
                        "Ticker": "OVERALL",
                        "Side": side,
                        "Mean_bps": imp_series.mean(),
                        "Std_bps": imp_series.std(),
                        "T_Statistic": t_stat,
                        "P_Value": p_val,
                        "WinRate": (imp_series > 0).mean(),
                        "ForcedRate": all_side_df["Forced"].mean(),
                        "AvgExecSecond": all_side_df["Exec_Second"].mean(),
                    }
                )

        if overall_rows:
            summary_df = pd.concat(
                [summary_df, pd.DataFrame(overall_rows)],
                ignore_index=True,
            )

        summary_path = "results/summary_ye_adjusted.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\nAll ye_adjusted backtests completed.")
        print(f"Summary saved to {summary_path}")
        print(summary_df.round(4))