from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import cumulative_returns


def plot_regimes_on_price(prices: pd.DataFrame, regimes: pd.DataFrame, asset: str = "SPY") -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(prices.index, prices[asset], color="black", linewidth=1.2, label=asset)
    colors = {"bull": "#2ca02c", "neutral": "#ffbf00", "crisis": "#d62728"}

    for label, group in regimes.groupby("regime_label"):
        aligned = prices.loc[prices.index.intersection(group.index)]
        ax.scatter(aligned.index, aligned[asset], s=18, color=colors.get(label, "gray"), label=label, alpha=0.85)

    ax.set_title(f"{asset} with Walk-Forward HMM Regime Labels")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.show()


def plot_equity_curves(return_frame: pd.DataFrame) -> None:
    equity = cumulative_returns(return_frame)
    ax = equity.plot(figsize=(14, 6), linewidth=1.5)
    ax.set_title("Strategy vs Static Benchmarks")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    plt.show()


def plot_weights(weights: pd.DataFrame) -> None:
    ax = weights[["SPY", "TLT", "GLD"]].plot.area(figsize=(14, 5), alpha=0.85)
    ax.set_title("Dynamic Portfolio Weights")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    plt.show()


def plot_drawdowns(return_frame: pd.DataFrame) -> None:
    equity = cumulative_returns(return_frame)
    drawdowns = equity / equity.cummax() - 1
    ax = drawdowns.plot(figsize=(14, 5), linewidth=1.2)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    plt.show()
