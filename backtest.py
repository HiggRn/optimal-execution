import os

import pandas as pd
from scipy import stats
from tqdm import tqdm

from zijie_strategy import Strategy


def run_backtest(df, strategy_class, show_progess=True, *args, **kwargs):
    results = []

    for side in ["BUY", "SELL"]:
        strat = strategy_class(side, *args, **kwargs)

        if show_progess:
            it = tqdm(df.groupby("Minute"))
        else:
            it = df.groupby("Minute")

        for minute, grp in it:
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
                            row["AskPrice_1"] if side == "BUY" else row["BidPrice_1"]
                        )
                        exec_time = idx
                        executed = True

            if exec_price is None:
                last_row = grp.iloc[-1]
                exec_price = (
                    last_row["AskPrice_1"] if side == "BUY" else last_row["BidPrice_1"]
                )
                exec_time = grp.index[-1]

            improvement = (
                twap_price - exec_price if side == "BUY" else exec_price - twap_price
            )

            optm_price = (
                grp["AskPrice_1"].min() if side == "BUY" else grp["BidPrice_1"].max()
            )

            results.append(
                {
                    "Minute": minute,
                    "Side": side,
                    "TWAP_Price": twap_price,
                    "Exec_Price": exec_price,
                    "Optm_Price": optm_price,
                    "Improvement_bps": improvement / twap_price * 10000,
                    "Optm_Improvement_bps": abs(optm_price - twap_price)
                    / twap_price
                    * 10000,
                    "Exec_Time": exec_time,
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    tickers = ["AMZN", "GOOG", "INTC", "MSFT"]

    summary_data = []
    for ticker in tickers:
        file_path = f"data/{ticker}_5levels_train.csv"

        if not os.path.exists(file_path):
            print(f"File '{file_path}' doesn't exist.")
            continue

        print(f"Backtesting {ticker} ...")

        df = pd.read_csv(file_path)

        df.columns = [c.strip() for c in df.columns]
        df["Time_dt"] = pd.to_datetime(
            df["Time"], format="%H:%M:%S.%f", errors="coerce"
        )
        df = df.sort_values("Time_dt")
        df.set_index("Time_dt", inplace=True)
        df["Minute"] = df.index.floor("min")

        result_df = run_backtest(df, Strategy)

        output_name = f"results/{ticker}_backtest_result.csv"
        result_df.to_csv(output_name, index=False)

        print(f">>> {ticker} Results saved to {output_name}.")

        for side in ["BUY", "SELL"]:
            side_df = result_df[result_df["Side"] == side]
            imp_series = side_df["Improvement_bps"]

            if len(imp_series) > 1:
                mean_imp = imp_series.mean()
                std_imp = imp_series.std()
                t_stat, p_val = stats.ttest_1samp(imp_series, 0)
            else:
                mean_imp, std_imp, t_stat, p_val = 0.0, 0.0, 0.0, 1.0

            print(
                f"    [{side:4s}] Mean: {mean_imp:>7.4f} bps | Std: {std_imp:>6.4f} | T-Stat: {t_stat:>7.4f} (p-val: {p_val:>6.4f})"
            )

            summary_data.append(
                {
                    "Ticker": ticker,
                    "Side": side,
                    "Mean_Improvement": mean_imp,
                    "Std_Improvement": std_imp,
                    "T_Statistic": t_stat,
                    "P_Value": p_val,
                }
            )

        total_buy_exec = result_df.loc[result_df["Side"] == "BUY", "Exec_Price"].sum()
        total_sell_exec = result_df.loc[result_df["Side"] == "SELL", "Exec_Price"].sum()
        total_buy_twap = result_df.loc[result_df["Side"] == "BUY", "TWAP_Price"].sum()
        total_sell_twap = result_df.loc[result_df["Side"] == "SELL", "TWAP_Price"].sum()
        pct_improv = 100 - 100 * (total_buy_exec - total_sell_exec) / (
            total_buy_twap - total_sell_twap
        )
        print(f"Percentage Improvement: {pct_improv:>7.4f}%")

        print("-" * 70)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("results/summary_stats.csv", index=False)
        print("\nAll backtests completed. Summary saved to results/summary_stats.csv")
