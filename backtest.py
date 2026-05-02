import os

import pandas as pd
from scipy import stats

from zijie_strategy import Strategy


def run_backtest(df, strategy_class, *args, **kwargs):
    results = []

    for side in ["BUY", "SELL"]:
        strat = strategy_class(side, *args, **kwargs)
        for minute, grp in df.groupby("Minute"):
            twap_price = (
                grp["AskPrice_1"].iloc[0]
                if side == "BUY"
                else grp["BidPrice_1"].iloc[0]
            )

            exec_price = None

            for idx, row in grp.iterrows():
                if strat.on_tick(row, idx):
                    exec_price = (
                        row["AskPrice_1"] if side == "BUY" else row["BidPrice_1"]
                    )
                    break

            if exec_price is None:
                last_row = grp.iloc[-1]
                exec_price = (
                    last_row["AskPrice_1"] if side == "BUY" else last_row["BidPrice_1"]
                )

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
                    "Improvement_bps": (improvement / twap_price) * 10000,
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

        print(f"Backtested {ticker}. Results saved to {output_name}.")
        print("-" * 30)

        imp_series = result_df["Improvement_bps"]
        mean_imp = imp_series.mean()
        std_imp = imp_series.std()

        t_stat, p_val = stats.ttest_1samp(imp_series, 0)

        print(
            f">>> {ticker} Statistics | Mean: {mean_imp:.4f} bps | Std: {std_imp:.4f} | T-Stat: {t_stat:.4f} (p-val: {p_val:.4f})"
        )

        summary_data.append(
            {
                "Ticker": ticker,
                "Mean_Improvement": mean_imp,
                "Std_Improvement": std_imp,
                "T_Statistic": t_stat,
                "P_Value": p_val,
            }
        )

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("results/summary_stats.csv", index=False)
        print("\nAll backtests completed. Summary saved to results/summary_stats.csv")
