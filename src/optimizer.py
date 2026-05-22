from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ASSETS = ("SPY", "TLT", "GLD")

REGIME_CONSTRAINTS = {
    "bull": {"bounds": ((0.80, 0.90), (0.05, 0.15), (0.00, 0.10))},
    "neutral": {"bounds": ((0.65, 0.75), (0.15, 0.25), (0.05, 0.15))},
    "crisis": {"bounds": ((0.35, 0.50), (0.05, 0.20), (0.35, 0.55))},
}

REGIME_BASE_WEIGHTS = {
    "bull": np.array([0.85, 0.10, 0.05]),
    "neutral": np.array([0.70, 0.20, 0.10]),
    "crisis": np.array([0.45, 0.10, 0.45]),
}


def portfolio_stats(weights: np.ndarray, returns: pd.DataFrame) -> tuple[float, float, float]:
    """Return annualized return, volatility, and Sharpe ratio."""
    mean_returns = returns.mean().values * 252
    cov = returns.cov().values * 252
    ann_return = float(weights @ mean_returns)
    ann_vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    return ann_return, ann_vol, sharpe


def optimize_weights(
    trailing_returns: pd.DataFrame,
    regime_label: str,
    previous_weights: np.ndarray | None = None,
    anchor_weights: np.ndarray | None = None,
    turnover_aversion: float = 0.05,
    anchor_aversion: float = 1000.0,
) -> pd.Series:
    
    trailing_returns = trailing_returns.loc[:, list(ASSETS)].dropna()
    if len(trailing_returns) < 60:
        return pd.Series(np.repeat(1 / len(ASSETS), len(ASSETS)), index=ASSETS)

    bounds = REGIME_CONSTRAINTS.get(regime_label, REGIME_CONSTRAINTS["neutral"])["bounds"]
    previous = previous_weights if previous_weights is not None else np.repeat(1 / len(ASSETS), len(ASSETS))
    base = anchor_weights if anchor_weights is not None else REGIME_BASE_WEIGHTS.get(regime_label, REGIME_BASE_WEIGHTS["neutral"])
    base = np.clip(base, [bound[0] for bound in bounds], [bound[1] for bound in bounds])
    base = base / base.sum()

    def objective(weights: np.ndarray) -> float:
        ann_return, ann_vol, sharpe = portfolio_stats(weights, trailing_returns)
        turnover_penalty = turnover_aversion * np.abs(weights - previous).sum()
        anchor_penalty = anchor_aversion * np.square(weights - base).sum()
        if regime_label == "crisis":
            return ann_vol - 0.25 * ann_return + turnover_penalty + anchor_penalty
        return -sharpe + turnover_penalty + anchor_penalty

    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]
    result = minimize(
        objective,
        base,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9},
    )

    weights = result.x if result.success else base
    weights = np.clip(weights, 0, 1)
    weights = weights / weights.sum()
    return pd.Series(weights, index=ASSETS)
