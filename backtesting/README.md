# `backtesting/` – Full-Featured Backtesting Framework

This folder contains the **complete production-grade backtesting framework**
used for strategy research, optimisation and reporting.

| File | Purpose |
|------|---------|
| `backtest_engine.py` | Core event-driven backtesting engine |
| `data_handler.py` | Historical data loading and normalisation |
| `data_sources.py` | Connectors for external data feeds |
| `engine.py` | High-level orchestration entry point |
| `events.py` | Event types (MarketEvent, SignalEvent, OrderEvent, FillEvent) |
| `execution.py` | Simulated and live execution handlers |
| `hyperopt.py` | Bayesian / grid hyperparameter optimisation |
| `metrics.py` | Performance metrics (Sharpe, Sortino, Calmar, …) |
| `optimizer.py` | Multi-objective strategy optimiser |
| `plots.py` | Equity-curve and drawdown visualisations |
| `portfolio.py` | Multi-asset portfolio state tracking |
| `reports.py` | HTML / PDF report generation |
| `walk_forward.py` | Walk-forward analysis and anchored cross-validation |

## When to use `backtesting/`

Use this folder for **production strategy research**: walk-forward validation,
hyperparameter search, multi-strategy portfolio analytics and detailed
reporting.

## Difference from `backtest/`

| | `backtesting/` | `backtest/` |
|-|----------------|-------------|
| Scope | Full-featured framework | Single engine file |
| Dependencies | pandas, numpy, scipy, optional GPU | Minimal |
| Features | Walk-forward, hyperopt, portfolio | Tick simulation |
| Use when | Production research | Quick prototype |

See also [`backtest/README.md`](../backtest/README.md).
