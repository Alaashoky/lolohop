"""
Backtesting engine — runs signal-based backtests on OHLCV DataFrames.
Supports ATR-based Stop Loss and Take Profit for realistic results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trades: List[dict] = field(default_factory=list)


class BacktestEngine:
    """
    Vectorised backtester with ATR-based Stop Loss & Take Profit.

    Signals: +1 = long, -1 = short, 0 = flat.
    Positions are sized as a fixed fraction of equity (default 10%).

    ATR Stops (default):
        SL = entry_price ± atr_sl_mult × ATR(14)   [default 1.0×]
        TP = entry_price ± atr_tp_mult × ATR(14)   [default 3.0×]
    """

    def __init__(
        self,
        initial_balance: float = 100_000.0,
        position_size_pct: float = 0.10,
        commission_pct: float = 0.0002,
        atr_sl_mult: float = 1.0,
        atr_tp_mult: float = 3.0,
        use_atr_stops: bool = True,
    ):
        self.initial_balance   = initial_balance
        self.balance           = initial_balance
        self.position_size_pct = position_size_pct
        self.commission_pct    = commission_pct
        self.atr_sl_mult       = atr_sl_mult
        self.atr_tp_mult       = atr_tp_mult
        self.use_atr_stops     = use_atr_stops
        self.trades: List[dict]        = []
        self.equity_curve: List[float] = [initial_balance]

    # ------------------------------------------------------------------
    def run_backtest(
        self,
        data: pd.DataFrame,
        signals,          # pd.Series or callable(data) -> pd.Series
    ) -> BacktestResult:
        """
        Run a backtest with ATR-based TP/SL.

        Parameters
        ----------
        data    : OHLCV DataFrame with columns open/high/low/close/volume
                  and optionally atr_14 (calculated by calculate_features)
        signals : pd.Series of {-1, 0, 1} aligned to data.index,
                  OR a callable that receives data and returns such a Series
        """
        if callable(signals):
            signals = signals(data)

        signals = pd.Series(signals, index=data.index).fillna(0)

        # Reset state
        self.balance = self.initial_balance
        self.trades  = []
        equity       = [self.initial_balance]

        position     = 0        # current position size (units)
        entry_price  = 0.0
        entry_signal = 0
        entry_atr    = 0.0
        tp_price     = 0.0
        sl_price     = 0.0

        has_atr = "atr_14" in data.columns

        for i in range(1, len(data)):
            price    = data["close"].iloc[i]
            high     = data["high"].iloc[i]
            low      = data["low"].iloc[i]
            prev_sig = signals.iloc[i - 1]

            # ATR at current bar (fallback: 0.1% of price)
            atr = float(data["atr_14"].iloc[i]) if has_atr else price * 0.001

            # ── إغلاق صفقة مفتوحة ──────────────────────────────────
            if position != 0:
                hit_tp = False
                hit_sl = False
                exit_price = None

                if self.use_atr_stops and entry_atr > 0:
                    if position > 0:  # Long
                        hit_tp = high >= tp_price
                        hit_sl = low  <= sl_price
                    else:             # Short
                        hit_tp = low  <= tp_price
                        hit_sl = high >= sl_price

                if hit_tp:
                    exit_price  = tp_price
                    exit_reason = "TP"
                elif hit_sl:
                    exit_price  = sl_price
                    exit_reason = "SL"
                elif prev_sig == 0 or prev_sig != entry_signal:
                    exit_price  = price
                    exit_reason = "signal"

                if exit_price is not None:
                    pnl        = position * (exit_price - entry_price)
                    commission = abs(position) * exit_price * self.commission_pct
                    net_pnl    = pnl - commission
                    self.balance += net_pnl
                    self.trades.append({
                        "entry_price": entry_price,
                        "exit_price":  exit_price,
                        "size":        position,
                        "pnl":         net_pnl,
                        "win":         net_pnl > 0,
                        "exit_reason": exit_reason,
                    })
                    position  = 0
                    entry_atr = 0.0

            # ── فتح صفقة جديدة ──────────────────────────────────────
            if prev_sig != 0 and position == 0:
                size         = (self.balance * self.position_size_pct) / price
                position     = size if prev_sig == 1 else -size
                entry_price  = price
                entry_signal = prev_sig
                entry_atr    = atr

                if self.use_atr_stops and entry_atr > 0:
                    if position > 0:  # Long
                        tp_price = entry_price + self.atr_tp_mult * entry_atr
                        sl_price = entry_price - self.atr_sl_mult * entry_atr
                    else:             # Short
                        tp_price = entry_price - self.atr_tp_mult * entry_atr
                        sl_price = entry_price + self.atr_sl_mult * entry_atr

            equity.append(self.balance)

        # ── Force-close at end ───────────────────────────────────────
        if position != 0:
            price      = data["close"].iloc[-1]
            pnl        = position * (price - entry_price)
            commission = abs(position) * price * self.commission_pct
            net_pnl    = pnl - commission
            self.balance += net_pnl
            self.trades.append({
                "entry_price": entry_price,
                "exit_price":  price,
                "size":        position,
                "pnl":         net_pnl,
                "win":         net_pnl > 0,
                "exit_reason": "end",
            })
            equity[-1] = self.balance

        self.equity_curve = equity

        # ── Metrics ──────────────────────────────────────────────────
        eq           = np.array(equity, dtype=float)
        total_return = (eq[-1] - eq[0]) / eq[0]

        daily_ret = np.diff(eq) / eq[:-1]
        sharpe = (
            float(np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252))
            if np.std(daily_ret) > 0 else 0.0
        )

        roll_max     = np.maximum.accumulate(eq)
        drawdowns    = (eq - roll_max) / roll_max
        max_drawdown = float(drawdowns.min())

        n        = len(self.trades)
        win_rate = sum(1 for t in self.trades if t["win"]) / n if n > 0 else 0.0

        gross_profit  = sum(t["pnl"] for t in self.trades if t["pnl"] > 0)
        gross_loss    = abs(sum(t["pnl"] for t in self.trades if t["pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return BacktestResult(
            total_return=float(total_return),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_drawdown),
            win_rate=float(win_rate),
            total_trades=n,
            profit_factor=float(profit_factor),
            equity_curve=equity,
            trades=self.trades,
        )

    # ------------------------------------------------------------------
    def monte_carlo_analysis(self, n_simulations: int = 1000) -> dict:
        """
        Bootstrap-resample trade PnLs to estimate return distribution.
        Returns empty dict if no trades have been run.
        """
        if not self.trades:
            return {}

        pnls = np.array([t["pnl"] for t in self.trades])
        sim_returns = []

        rng = np.random.default_rng(42)
        for _ in range(n_simulations):
            sample = rng.choice(pnls, size=len(pnls), replace=True)
            total = float(np.sum(sample) / self.initial_balance)
            sim_returns.append(total)

        sim_returns = np.array(sim_returns)
        return {
            "mean_return":   float(np.mean(sim_returns)),
            "std_return":    float(np.std(sim_returns)),
            "percentile_5":  float(np.percentile(sim_returns, 5)),
            "percentile_95": float(np.percentile(sim_returns, 95)),
            "n_simulations": n_simulations,
        }


# Alias kept for backward compatibility
Backtest = BacktestEngine
