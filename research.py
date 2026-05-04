import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import statsmodels.api as sm

TICKERS = ["AMZN", "GOOG", "INTC", "MSFT"]
DATA_DIR = Path("./data")

DIR_COL = "Direction_1=Buy_-1=Sell"
VEXEC_COL = "VisibleExecution_1=Yes_0=No"
HEXEC_COL = "HiddenExecution_1=Yes_0=No"

# ── I/O & Features ────────────────────────────────────────────────────────────


def load(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{ticker}_5levels_train.csv", parse_dates=["Time"])
    return df.sort_values("Time").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["exec_buy"] = df["AskPrice_1"]
    df["exec_sell"] = df["BidPrice_1"]

    # Weighted mid
    df["wmid"] = (
        df["BidPrice_1"] * df["AskSize_1"] + df["AskPrice_1"] * df["BidSize_1"]
    ) / (df["BidSize_1"] + df["AskSize_1"] + 1e-9)
    df["wmid_dev"] = (df["wmid"] - df["MidPrice"]) / (df["Spread"] + 1e-9)

    # Aggregated order imbalance
    bid_w = sum((6 - k) * df[f"BidSize_{k}"] for k in range(1, 6))
    ask_w = sum((6 - k) * df[f"AskSize_{k}"] for k in range(1, 6))
    df["oib_wgt"] = (bid_w - ask_w) / (bid_w + ask_w + 1e-9)

    # Spread metrics
    df["spread_bps"] = df["Spread"] / df["MidPrice"] * 1e4
    df["ask_pos"] = (df["AskPrice_1"] - df["MidPrice"]) / (df["Spread"] + 1e-9)
    df["bid_pos"] = (df["MidPrice"] - df["BidPrice_1"]) / (df["Spread"] + 1e-9)

    # Depth slope
    bid_vol = sum(df[f"BidSize_{k}"] for k in range(1, 6))
    ask_vol = sum(df[f"AskSize_{k}"] for k in range(1, 6))
    df["bid_slope"] = (df["BidPrice_1"] - df["BidPrice_5"]) / (bid_vol + 1e-9)
    df["ask_slope"] = (df["AskPrice_5"] - df["AskPrice_1"]) / (ask_vol + 1e-9)

    # Signed trade flow
    df["signed_vol"] = df[DIR_COL] * df["Size"]
    roll_sv = df["signed_vol"].rolling(20, min_periods=1)
    roll_v = df["Size"].rolling(20, min_periods=1)
    df["tflow_20"] = roll_sv.sum() / (roll_v.sum() + 1e-9)

    return df


FEATURE_COLS = [
    "oib_wgt",
    "wmid_dev",
    "spread_bps",
    "ask_pos",
    "bid_pos",
    "bid_slope",
    "ask_slope",
    "tflow_20",
]

# ── Target Processing ─────────────────────────────────────────────────────────


def smooth_time(
    prices: np.ndarray, times: pd.Series, halflife_sec: float
) -> np.ndarray:
    s = pd.Series(prices, index=times)
    hl = pd.Timedelta(f"{halflife_sec}s")
    return s.ewm(halflife=hl, times=times, adjust=False).mean().values


def local_dir(
    smooth_px: np.ndarray, times: pd.Series, horizon_sec: float
) -> np.ndarray:
    t = times.to_numpy(dtype="datetime64[ns]").astype("int64")
    h = int(horizon_sec * 1e9)
    n = len(smooth_px)
    direction = np.full(n, np.nan)
    js = np.searchsorted(t, t + h)
    valid = js < n
    direction[valid] = np.sign(smooth_px[js[valid]] - smooth_px[valid])
    return direction


def local_return(
    smooth_px: np.ndarray, times: pd.Series, horizon_sec: float
) -> np.ndarray:
    t = times.to_numpy(dtype="datetime64[ns]").astype("int64")
    h = int(horizon_sec * 1e9)
    n = len(smooth_px)
    ret = np.full(n, np.nan)
    js = np.searchsorted(t, t + h)
    valid = js < n
    ret[valid] = smooth_px[js[valid]] - smooth_px[valid]
    return ret


# ── Goal 1: Regime-Conditioned IC ─────────────────────────────────────────────


def regime_ic_analysis(df: pd.DataFrame, target_col: str, horizon_sec: float):
    direction = local_dir(df[target_col].values, df["Time"], horizon_sec)
    df["_target_dir"] = direction

    try:
        df["spread_regime"] = pd.qcut(
            df["spread_bps"], q=3, labels=["Tight", "Normal", "Wide"]
        )
    except ValueError:
        df["spread_regime"] = "Normal"

    print(f"\n--- Goal 1: Regime-Conditioned IC (Horizon: {horizon_sec}s) ---")

    features_to_check = ["oib_wgt", "bid_pos", "bid_slope"]
    for feat in features_to_check:
        if feat not in df.columns:
            continue

        print(f"\nFeature: {feat}")
        for regime in ["Tight", "Normal", "Wide"]:
            mask = (df["spread_regime"] == regime) & ~(
                df["_target_dir"].isna() | df[feat].isna()
            )
            if mask.sum() < 100 or np.var(df.loc[mask, "_target_dir"]) < 1e-6:
                print(f"  [{regime:<6} Spread]  IC: N/A  (Insufficient variance)")
                continue

            ic, _ = spearmanr(df.loc[mask, feat], df.loc[mask, "_target_dir"])
            print(f"  [{regime:<6} Spread]  IC: {ic:>7.4f}")


# ── Goal 2: Multi-variate OLS ─────────────────────────────────────────────────


def multivariate_ols_analysis(df: pd.DataFrame, target_col: str, horizon_sec: float):
    future_ret = local_return(df[target_col].values, df["Time"], horizon_sec)

    print(f"\n--- Goal 2: Multivariate OLS Regression (Horizon: {horizon_sec}s) ---")

    X_cols = ["oib_wgt", "bid_pos", "bid_slope", "tflow_20"]

    reg_df = df[X_cols].copy()
    reg_df["Target"] = future_ret
    reg_df = reg_df.dropna()

    if len(reg_df) < 500 or np.var(reg_df["Target"]) < 1e-12:
        print("Not enough variance for OLS regression.")
        return

    X = reg_df[X_cols]
    X_z = (X - X.mean()) / (X.std() + 1e-9)
    Y = reg_df["Target"]

    X_z = sm.add_constant(X_z)
    model = sm.OLS(Y, X_z).fit()

    summary = pd.DataFrame(
        {"Coefficient": model.params, "t-stat": model.tvalues, "P>|t|": model.pvalues}
    ).drop("const")

    print(summary.round(4).to_string())
    print(f"\nR-squared: {model.rsquared:.6f}")


# ── Goal 3: Oracle Sensitivity Analysis ───────────────────────────────────────


def _get_minute_opt(df: pd.DataFrame, col: str, side: str) -> pd.Series:
    fn = "min" if side == "buy" else "max"
    return df.groupby(df["Time"].dt.floor("min"))[col].transform(fn)


def oracle_sensitivity_analysis(df: pd.DataFrame, side: str):
    print(f"\n--- Goal 3: Oracle Sensitivity Analysis ({side}) ---")
    halflifes = [1.0, 5.0, 15.0, 30.0]

    raw_col = "exec_buy" if side == "buy" else "exec_sell"
    df["minute_opt_raw"] = _get_minute_opt(df, raw_col, side)
    sign = 1 if side == "buy" else -1

    results = []

    for hl in halflifes:
        smooth_col = f"smooth_{hl}s"
        df[smooth_col] = smooth_time(df[raw_col].values, df["Time"], halflife_sec=hl)

        regrets = []
        for minute, grp in df.groupby(df["Time"].dt.floor("min")):
            if len(grp) < 3:
                continue

            idx = (
                grp[smooth_col].argmin() if side == "buy" else grp[smooth_col].argmax()
            )
            exec_price = grp[raw_col].iloc[idx]
            opt_price = grp["minute_opt_raw"].iloc[0]

            regrets.append((exec_price - opt_price) * sign)

        results.append(
            {
                "Halflife(s)": hl,
                "Mean Regret": np.mean(regrets),
                "Median Regret": np.median(regrets),
            }
        )

    print(pd.DataFrame(results).to_string(index=False))


# ── Main Entry ────────────────────────────────────────────────────────────────


def run_research(ticker: str):
    print(f"\n{'=' * 70}")
    print(f"  RESEARCH REPORT: {ticker}")
    print(f"{'=' * 70}")

    df = load(ticker)
    df = build_features(df)

    horizon_sec = 10.0
    target_col = "smooth_wmid"

    df[target_col] = smooth_time(df["wmid"].values, df["Time"], halflife_sec=5.0)

    # Goal 1: Regime-Conditioned IC
    regime_ic_analysis(df, target_col, horizon_sec)

    # Goal 2: Multivariate OLS
    multivariate_ols_analysis(df, target_col, horizon_sec)

    # Goal 3: Oracle Sensitivity
    for side in ["buy", "sell"]:
        oracle_sensitivity_analysis(df, side)


if __name__ == "__main__":
    for ticker in TICKERS:
        run_research(ticker)
