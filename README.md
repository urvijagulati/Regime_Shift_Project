# HMM Regime Shift Portfolio Allocation

This project builds a dynamic allocation strategy that detects market regimes with a Hidden Markov Model and changes portfolio constraints across equities, fixed income, and safe-haven assets. It includes walk-forward validation, transaction cost modeling, and benchmark comparisons against static portfolios.

## Deliverables

- Trained HMM regime model with regime labels overlaid on historical market data.
- Dynamic allocation backtester with strict walk-forward validation.
- Explicit transaction cost model using a 10 bps turnover penalty per rebalance.
- Performance tear sheet with Sharpe, Sortino, max drawdown, Calmar ratio, annual return, volatility, and turnover.
- Jupyter notebook covering data ingestion, regime detection, optimization, backtesting, and reporting.
- Reproducible Python modules and README.

## Project Structure

```text
regime_shift/
  notebooks/
    regime_shift_pipeline.ipynb
    regime_shift_pipeline.html 
  src/
    data_loader.py
    features.py
    hmm_model.py
    optimizer.py
    backtester.py
    metrics.py
    plots.py
  requirements.txt
  README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook notebooks/regime_shift_pipeline.ipynb
```

## Methodology

The notebook downloads daily data for:

- `SPY`: equity proxy
- `TLT`: fixed income proxy
- `GLD`: safe-haven/gold proxy
- `^VIX`: volatility proxy

The HMM uses features derived only from information available at each point in time, including returns, rolling volatility, momentum, and VIX changes. During walk-forward validation, the HMM is re-fit using only historical data available before each rebalance date.

## Regime Mapping

The model discovers numerical regimes. These are converted into economic labels by ranking each regime on equity return and volatility:

- `bull`: higher return and lower volatility
- `neutral`: mixed behavior
- `crisis`: lower return and higher volatility

Each regime maps to different optimization constraints:

- Bull: allows higher equity exposure.
- Neutral: balances equity, bonds, and gold.
- Crisis: caps equity exposure and requires larger defensive allocations.

## Backtest Rules

- Monthly rebalancing.
- HMM trained only on past data.
- Portfolio weights are optimized from trailing historical returns.
- Transaction cost = `turnover * 0.0010`.
- Benchmarks:
  - 60/40: 60% SPY, 40% TLT
  - Equal weight: SPY, TLT, GLD equally weighted
