from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MarketConfig:
    start: str = "2005-01-01"
    end: str | None = None
    assets: tuple[str, ...] = ("SPY", "TLT", "GLD")
    proxies: tuple[str, ...] = ("^VIX",)


def download_market_data(config: MarketConfig | None = None) -> pd.DataFrame:
   
    config = config or MarketConfig()
    tickers = list(config.assets + config.proxies)
    raw = yf.download(
        tickers,
        start=config.start,
        end=config.end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if raw.empty:
        raise ValueError("No data downloaded. Check ticker symbols, dates, or internet connection.")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.sort_index().ffill().dropna(how="all")
    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        raise ValueError(f"Missing downloaded tickers: {missing}")

    return prices[tickers]


def asset_returns(prices: pd.DataFrame, assets: tuple[str, ...] = ("SPY", "TLT", "GLD")) -> pd.DataFrame:
    #Compute daily asset returns used by the optimizer and backtester.
    return prices.loc[:, list(assets)].pct_change().dropna()
