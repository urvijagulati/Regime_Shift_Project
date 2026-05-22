from __future__ import annotations

import numpy as np
import pandas as pd

from .features import scale_train_test
from .hmm_model import fit_hmm, label_regimes, predict_regime_probabilities, predict_regimes
from .optimizer import ASSETS, optimize_weights


def monthly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.Series(index=index, data=index).resample("ME").last().dropna().values


def run_walk_forward_backtest(
    features: pd.DataFrame,
    returns: pd.DataFrame,
    train_years: int = 5,
    rebalance_frequency: str = "ME",
    transaction_cost_bps: float = 10.0,
    n_components: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_index = features.index.intersection(returns.index)
    features = features.loc[common_index]
    returns = returns.loc[common_index, list(ASSETS)]

    rebalance_dates = pd.Series(index=common_index, data=common_index).resample(rebalance_frequency).last().dropna()
    first_date = common_index.min() + pd.DateOffset(years=train_years)
    rebalance_dates = rebalance_dates[rebalance_dates >= first_date]

    previous_weights = np.repeat(1 / len(ASSETS), len(ASSETS))
    daily_records = []
    weight_records = []
    regime_records = []

    for i, rebalance_date in enumerate(rebalance_dates):
        train_mask = features.index < rebalance_date
        train_features = features.loc[train_mask]
        train_returns = returns.loc[returns.index < rebalance_date]
        if len(train_features) < 252 or len(train_returns) < 252:
            continue

        if i + 1 < len(rebalance_dates):
            next_rebalance = rebalance_dates.iloc[i + 1]
            period_index = returns.loc[(returns.index > rebalance_date) & (returns.index <= next_rebalance)].index
        else:
            period_index = returns.loc[returns.index > rebalance_date].index

        if period_index.empty:
            continue

        test_features = features.loc[[rebalance_date]]
        train_scaled, test_scaled, _ = scale_train_test(train_features, test_features)
        model = fit_hmm(train_scaled, n_components=n_components)
        train_regimes = predict_regimes(model, train_scaled)
        current_sequence = pd.concat([train_scaled, test_scaled])
        current_regime = int(predict_regimes(model, current_sequence).iloc[-1])
        regime_map = label_regimes(train_regimes, train_returns, train_features)
        regime_label = regime_map.get(current_regime, "neutral")
        raw_probabilities = predict_regime_probabilities(model, current_sequence).iloc[-1]
        label_probabilities = {"bull": 0.0, "neutral": 0.0, "crisis": 0.0}
        for regime_column, probability in raw_probabilities.items():
            regime_number = int(regime_column.split("_")[-1])
            label = regime_map.get(regime_number, "neutral")
            label_probabilities[label] += float(probability)

        trailing_returns = train_returns.tail(252)
        weights = optimize_weights(
            trailing_returns,
            regime_label,
            previous_weights=previous_weights,
        )
        turnover = float(np.abs(weights.values - previous_weights).sum())
        cost = turnover * (transaction_cost_bps / 10000.0)

        period_returns = returns.loc[period_index]
        portfolio_returns = period_returns.dot(weights)
        if len(portfolio_returns) > 0:
            portfolio_returns.iloc[0] -= cost

        for date, value in portfolio_returns.items():
            daily_records.append(
                {
                    "date": date,
                    "strategy_return": value,
                    "regime": current_regime,
                    "regime_label": regime_label,
                    "turnover": turnover if date == portfolio_returns.index[0] else 0.0,
                    "transaction_cost": cost if date == portfolio_returns.index[0] else 0.0,
                    "bull_probability": label_probabilities["bull"],
                    "neutral_probability": label_probabilities["neutral"],
                    "crisis_probability": label_probabilities["crisis"],
                }
            )

        weight_records.append(
            {
                "date": rebalance_date,
                "SPY": weights["SPY"],
                "TLT": weights["TLT"],
                "GLD": weights["GLD"],
                "regime": current_regime,
                "regime_label": regime_label,
                "bull_probability": label_probabilities["bull"],
                "neutral_probability": label_probabilities["neutral"],
                "crisis_probability": label_probabilities["crisis"],
                "turnover": turnover,
            }
        )
        regime_records.append(
            {
                "date": rebalance_date,
                "regime": current_regime,
                "regime_label": regime_label,
                "bull_probability": label_probabilities["bull"],
                "neutral_probability": label_probabilities["neutral"],
                "crisis_probability": label_probabilities["crisis"],
            }
        )
        previous_weights = weights.values

    strategy = pd.DataFrame(daily_records).set_index("date")
    weights = pd.DataFrame(weight_records).set_index("date")
    regimes = pd.DataFrame(regime_records).set_index("date")
    return strategy, weights, regimes


def build_benchmarks(returns: pd.DataFrame, strategy_index: pd.Index) -> pd.DataFrame:
    aligned = returns.loc[strategy_index, list(ASSETS)]
    benchmarks = pd.DataFrame(index=aligned.index)
    benchmarks["60_40"] = aligned["SPY"] * 0.60 + aligned["TLT"] * 0.40
    benchmarks["equal_weight"] = aligned.mean(axis=1)
    return benchmarks
