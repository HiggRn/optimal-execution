# Optimal Execution

This is the repository for the final project of ORIE 5259 Market Microstructure and Algorithmic Trading.

The repo is managed by `uv`. To run the backtest, run
```bash
$ uv run backtest.py
```

Note that the `data` folder is not included.

## Strategy Idea

1. Large / small tick detection: Stocks are classified into large-tick and small-tick categories to apply specialized logic. The current implementation uses a rolling median of the bid-ask spread for classification.
2. Global trend: To account for macro trends during high-frequency execution, the strategy fine-tunes execution urgency within each minute by adjusting thresholds. This is implemented using a MACD (Moving Average Convergence Divergence) of the micro-price.
3. Execution timing: This is the core of the strategy, determining the optimal moment to execute within a one-minute window. If no signal is triggered, the strategy falls back to execution at the end of the minute.
    - Large-tick stocks: Characterized by thick order books; timing is driven by Order Book Imbalance (OBI).
    - Small-tick stocks: Characterized by thin order books; timing is driven by Trade Flow Imbalance (TFI) over a rolling window.
