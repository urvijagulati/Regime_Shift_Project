from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_hmm_features(prices: pd.DataFrame) -> pd.DataFrame:
    spy = prices["SPY"]
    returns = spy.pct_change()

    features = pd.DataFrame(index=prices.index)
    features["equity_return_1d"] = returns
    features["equity_vol_21d"] = returns.rolling(21).std()
    features["equity_momentum_63d"] = spy.pct_change(63)
    features["bond_return_1d"] = prices["TLT"].pct_change()
    features["gold_return_1d"] = prices["GLD"].pct_change()
    features["vix_level"] = prices["^VIX"]
    features["vix_change_5d"] = prices["^VIX"].pct_change(5)
    return features.replace([float("inf"), float("-inf")], pd.NA).dropna()


def align_returns_to_features(returns: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    return returns.loc[returns.index.intersection(features.index)].copy()


def scale_train_test(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train), index=train.index, columns=train.columns)
    test_scaled = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
    return train_scaled, test_scaled, scaler
