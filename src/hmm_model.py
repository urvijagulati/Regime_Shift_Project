from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def fit_hmm(features: pd.DataFrame, n_components: int = 3, random_state: int = 42) -> GaussianHMM:

    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=500,
        random_state=random_state,
        min_covar=1e-4,
    )
    model.fit(features)
    return model


def predict_regimes(model: GaussianHMM, features: pd.DataFrame) -> pd.Series:
    states = model.predict(features)
    return pd.Series(states, index=features.index, name="regime")


def predict_regime_probabilities(model: GaussianHMM, features: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(features)
    columns = [f"regime_{i}" for i in range(model.n_components)]
    return pd.DataFrame(probabilities, index=features.index, columns=columns)


def label_regimes(
    regimes: pd.Series,
    asset_returns: pd.DataFrame,
    features: pd.DataFrame | None = None,
) -> dict[int, str]:
    frame = pd.concat([regimes, asset_returns["SPY"].rename("spy_return")], axis=1)
    if features is not None:
        stress_columns = [column for column in ["equity_vol_21d", "vix_level", "vix_change_5d"] if column in features]
        frame = frame.join(features[stress_columns])
    frame = frame.dropna()

    summary = frame.groupby("regime").agg(
        spy_mean=("spy_return", "mean"),
        spy_vol=("spy_return", "std"),
    )

    if features is not None and {"equity_vol_21d", "vix_level"}.issubset(frame.columns):
        summary["realized_vol"] = frame.groupby("regime")["equity_vol_21d"].mean()
        summary["vix_level"] = frame.groupby("regime")["vix_level"].mean()
        if "vix_change_5d" in frame.columns:
            summary["vix_change"] = frame.groupby("regime")["vix_change_5d"].mean()
        else:
            summary["vix_change"] = 0.0

        summary["stress_score"] = (
            summary["spy_mean"].rank(ascending=False)
            + summary["spy_vol"].rank(ascending=True)
            + summary["realized_vol"].rank(ascending=True)
            + summary["vix_level"].rank(ascending=True)
            + summary["vix_change"].rank(ascending=True)
        )
        crisis_regime = int(summary["stress_score"].idxmax())
        remaining = summary.drop(index=crisis_regime)
        bull_regime = int((remaining["spy_mean"].rank() - remaining["spy_vol"].rank()).idxmax())

        labels = {crisis_regime: "crisis", bull_regime: "bull"}
        for regime in summary.index:
            labels.setdefault(int(regime), "neutral")
        return labels

    summary["score"] = summary["spy_mean"] - summary["spy_vol"]
    ordered = summary.sort_values("score", ascending=False).index.tolist()
    names = ["bull", "neutral", "crisis"]
    return {int(regime): names[min(index, len(names) - 1)] for index, regime in enumerate(ordered)}


def summarize_regimes(regimes: pd.Series, features: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
   
    joined = features.join(regimes).join(asset_returns.add_suffix("_return"))
    return joined.groupby("regime").mean(numeric_only=True)
