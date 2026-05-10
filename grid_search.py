import os
import argparse
import itertools
import pandas as pd
import json

from backtest import run_backtest
from zijie_strategy import Strategy

PARAM_GRID = {
    "obi_threshold": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "tfi_multiplier": [0.5, 1.0, 1.5, 2.0],
}


def load_and_prep_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    df["Time_dt"] = pd.to_datetime(df["Time"], format="%H:%M:%S.%f", errors="coerce")
    df = df.sort_values("Time_dt")
    df.set_index("Time_dt", inplace=True)
    df["Minute"] = df.index.floor("min")
    return df


def pct_improvement(result_df):
    if result_df.empty:
        return 0.0
    total_buy_exec = result_df.loc[result_df["Side"] == "BUY", "Exec_Price"].sum()
    total_sell_exec = result_df.loc[result_df["Side"] == "SELL", "Exec_Price"].sum()
    total_buy_twap = result_df.loc[result_df["Side"] == "BUY", "TWAP_Price"].sum()
    total_sell_twap = result_df.loc[result_df["Side"] == "SELL", "TWAP_Price"].sum()

    twap_spread = total_buy_twap - total_sell_twap
    if twap_spread == 0:
        return 0.0
    pct_improv = 100 - 100 * (total_buy_exec - total_sell_exec) / twap_spread
    return pct_improv


def grid_search(df_train, ticker):
    print(f"\n[Grid Search] Start searching on {ticker} (Rows: {len(df_train)})")

    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = -float("inf")
    best_params = None

    for params in combinations:
        res_df = run_backtest(df_train, Strategy, show_progess=False, **params)
        score = pct_improvement(res_df)

        if score > best_score:
            best_score = score
            best_params = params
            print(f"  -> Params: {params} | Improvement: {score:.4f}%")

    print(
        f"[Grid Search] Best Params for {ticker}: {best_params} | Score: {best_score:.4f}%"
    )
    return best_params


def split_mode(data_path, train_ratio):
    ticker = os.path.basename(data_path).split("_")[0]
    df = load_and_prep_data(data_path)

    unique_minutes = df["Minute"].unique()
    split_idx = int(len(unique_minutes) * train_ratio)
    train_minutes = unique_minutes[:split_idx]
    test_minutes = unique_minutes[split_idx:]

    df_train = df[df["Minute"].isin(train_minutes)]
    df_test = df[df["Minute"].isin(test_minutes)]

    best_params = grid_search(df_train, ticker)

    print(f"\n[Testing] Evaluating on test split (Rows: {len(df_test)})")
    test_res_df = run_backtest(df_test, Strategy, **best_params)
    test_score = pct_improvement(test_res_df)
    print(f"[Result] {ticker} Test Split Improvement: {test_score:.4f}%")

    return best_params, test_score


def dual_mode(train_path, test_path):
    ticker = os.path.basename(train_path).split("_")[0]
    df_train = load_and_prep_data(train_path)
    df_test = load_and_prep_data(test_path)

    best_params = grid_search(df_train, ticker)

    print(f"\n[Testing] Evaluating on isolated test file (Rows: {len(df_test)})")
    test_res_df = run_backtest(df_test, Strategy, **best_params)
    test_score = pct_improvement(test_res_df)
    print(f"[Result] {ticker} Test File Improvement: {test_score:.4f}%")

    return best_params, test_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy Grid Search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Mode 1: single file
    p_split = subparsers.add_parser("split")
    p_split.add_argument("--data", required=True, help="Path to train CSV")
    p_split.add_argument("--ratio", type=float, default=0.8, help="Train ratio")

    # Mode 2: double files
    p_dual = subparsers.add_parser("dual")
    p_dual.add_argument("--train", required=True, help="Path to train CSV")
    p_dual.add_argument("--test", required=True, help="Path to test CSV")

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.command == "split":
        best_params, test_score = split_mode(args.data, args.ratio)
        ticker = os.path.basename(args.data).split("_")[0]
    elif args.command == "dual":
        best_params, test_score = dual_mode(args.train, args.test)
        ticker = os.path.basename(args.train).split("_")[0]
    else:
        raise ValueError(f"Unknown command: {args.command}")

    output_file = f"results/{ticker}_best_params.json"
    with open(output_file, "w") as f:
        json.dump(
            {"best_params": best_params, "test_improvement": test_score}, f, indent=4
        )
    print(f"Parameters saved to {output_file}")
