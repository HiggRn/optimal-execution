import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

DATA_DIR = Path("./data")


def load_data(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{ticker}_5levels_train.csv")
    df["Time"] = pd.to_datetime(df["Time"], format="mixed")
    return df.sort_values("Time").reset_index(drop=True)


def plot_interactive_lob(
    df: pd.DataFrame, ticker: str, start_idx: int = 0, num_ticks: int = 10000
):
    sub_df = df.iloc[start_idx : start_idx + num_ticks].copy()
    if sub_df.empty:
        print("Empty slice.")
        return

    fig = go.Figure()
    times = sub_df["Time"]

    for i in range(1, 6):
        ask_sizes = np.log1p(sub_df[f"AskSize_{i}"])
        fig.add_trace(
            go.Scattergl(
                x=times,
                y=sub_df[f"AskPrice_{i}"],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=4,
                    color=ask_sizes,
                    colorscale="Reds",
                    cmin=0,
                    cmax=np.percentile(ask_sizes, 95),
                    opacity=0.6,
                ),
                name=f"Ask L{i}",
                hoverinfo="y+text",
                text=sub_df[f"AskSize_{i}"].astype(str) + " shares",
                showlegend=False,
            )
        )

        bid_sizes = np.log1p(sub_df[f"BidSize_{i}"])
        fig.add_trace(
            go.Scattergl(
                x=times,
                y=sub_df[f"BidPrice_{i}"],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=4,
                    color=bid_sizes,
                    colorscale="Greens",
                    cmin=0,
                    cmax=np.percentile(bid_sizes, 95),
                    opacity=0.6,
                ),
                name=f"Bid L{i}",
                hoverinfo="y+text",
                text=sub_df[f"BidSize_{i}"].astype(str) + " shares",
                showlegend=False,
            )
        )

    is_trade = (sub_df["VisibleExecution_1=Yes_0=No"] == 1) | (
        sub_df["HiddenExecution_1=Yes_0=No"] == 1
    )
    trades = sub_df[is_trade].copy()

    if not trades.empty:
        buy_trades = trades[trades["Direction_1=Buy_-1=Sell"] == 1]
        sell_trades = trades[trades["Direction_1=Buy_-1=Sell"] == -1]

        if not buy_trades.empty:
            fig.add_trace(
                go.Scattergl(
                    x=buy_trades["Time"],
                    y=buy_trades["Price"],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=np.log1p(buy_trades["Size"]) * 3,
                        color="crimson",
                        line=dict(color="black", width=1),
                    ),
                    name="Trade (Buy)",
                    hoverinfo="x+y+text",
                    text="Buy "
                    + buy_trades["Size"].astype(str)
                    + " @ "
                    + buy_trades["Price"].astype(str),
                )
            )

        if not sell_trades.empty:
            fig.add_trace(
                go.Scattergl(
                    x=sell_trades["Time"],
                    y=sell_trades["Price"],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=np.log1p(sell_trades["Size"]) * 3,
                        color="lime",
                        line=dict(color="black", width=1),
                    ),
                    name="Trade (Sell)",
                    hoverinfo="x+y+text",
                    text="Sell "
                    + sell_trades["Size"].astype(str)
                    + " @ "
                    + sell_trades["Price"].astype(str),
                )
            )

    fig.update_layout(
        title=f"{ticker} LOB Interactive Heatmap (Ticks {start_idx} to {start_idx + num_ticks})",
        yaxis_title="Price",
        xaxis_title="Time",
        template="plotly_white",
        hovermode="closest",
        dragmode="zoom",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    )

    fig.update_xaxes(rangeslider=dict(visible=True))

    fig.show()


if __name__ == "__main__":
    ticker = "GOOG"
    df = load_data(ticker)

    start = int(len(df) * 0.5)
    plot_interactive_lob(df, ticker, start_idx=start, num_ticks=20000)
