# `backtest/` – Lightweight Single-File Engine

This folder contains a **single, self-contained backtest engine** designed for
rapid experimentation.

| File | Purpose |
|------|---------|
| `engine.py` | Minimal tick-level simulation engine |

## When to use `backtest/`

Use this folder when you want a **simple, dependency-free** backtest engine that
you can drop into a script or notebook without importing the full framework.

## Difference from `backtesting/`

| | `backtest/` | `backtesting/` |
|-|-------------|----------------|
| Scope | Single engine file | Full-featured backtesting framework |
| Dependencies | Minimal | pandas, numpy, scipy, optional GPU libs |
| Features | Tick simulation | Walk-forward, hyperopt, portfolio analytics |
| Use when | Quick prototype / single strategy | Production-grade multi-strategy research |

See also [`backtesting/README.md`](../backtesting/README.md).
