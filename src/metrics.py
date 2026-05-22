from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return (1 + returns).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    equity = cumulative_returns(returns)
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min())


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0].std() * np.sqrt(periods_per_year)
    ann_return = returns.mean() * periods_per_year
    return float(ann_return / downside) if downside and downside > 0 else np.nan


def performance_table(returns: pd.DataFrame, turnover: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for column in returns.columns:
        series = returns[column].dropna()
        ann_return = series.mean() * 252
        ann_vol = series.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
        mdd = max_drawdown(series)
        calmar = ann_return / abs(mdd) if mdd < 0 else np.nan
        rows.append(
            {
                "strategy": column,
                "annual_return": ann_return,
                "annual_volatility": ann_vol,
                "sharpe": sharpe,
                "sortino": sortino_ratio(series),
                "max_drawdown": mdd,
                "calmar": calmar,
                "total_return": cumulative_returns(series).iloc[-1] - 1,
                "avg_monthly_turnover": turnover.mean() if turnover is not None and column == "strategy" else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("strategy")
