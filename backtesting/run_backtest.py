#!/usr/bin/env python3
"""
HOPEFX — Backtest Runner
يجيب بيانات XAUUSD.m من MT5 (2024-2026) ويشغّل backtest على الموديلات المحفوظة
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── إعدادات ──────────────────────────────────────────────────────────────────
SYMBOL         = "XAUUSD.m"
BACKTEST_START = datetime(2024, 1, 1)
BACKTEST_END   = datetime(2026, 1, 1)
TIMEFRAME_STR  = "1h"
MODEL_DIR        = "ml/models"
INITIAL_BALANCE  = 10_000.0
POSITION_SIZE_PCT = 0.10
COMMISSION_PCT    = 0.0002


# ── Step 1: جلب البيانات من MT5 ───────────────────────────────────────────────
def fetch_mt5_data(
    symbol: str, start: datetime, end: datetime, interval: str = "1h"
) -> pd.DataFrame:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise RuntimeError("MetaTrader5 not installed — run: pip install MetaTrader5")

    tf_map = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }
    timeframe = tf_map.get(interval, mt5.TIMEFRAME_H1)

    logger.info("Connecting to MT5...")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    logger.info("Fetching %s from %s to %s...", symbol, start.date(), end.date())
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)

    if rates is None or len(rates) == 0:
        fallback = symbol.replace(".m", "").replace(".M", "")
        logger.warning("No data for %s — trying %s ...", symbol, fallback)
        rates = mt5.copy_rates_range(fallback, timeframe, start, end)

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError(f"No data returned for {symbol}. تأكد أن MT5 شغال.")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"time": "timestamp", "tick_volume": "volume"})
    df = df.set_index("timestamp")
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    logger.info(
        "  %d rows fetched (%s → %s)",
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
    )
    return df


# ── Step 2: حساب الـ Features (نفس الـ 79 Feature اللي اتدرب عليها الموديل) ──
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    c = data["close"]
    h = data["high"]
    l = data["low"]
    o = data["open"]
    v = data["volume"]

    # Strategy 1: MA Crossover
    for p in [5, 10, 20, 50, 100, 200]:
        data[f"ma_{p}"] = c.rolling(p).mean()
    data["sig_ma_5_20"] = np.where(data["ma_5"] > data["ma_20"], 1, -1)
    data["sig_ma_10_50"] = np.where(data["ma_10"] > data["ma_50"], 1, -1)
    data["sig_ma_50_200"] = np.where(data["ma_50"] > data["ma_200"], 1, -1)
    data["ma_dist_5_20"] = (data["ma_5"] - data["ma_20"]) / (data["ma_20"] + 1e-9)
    data["ma_dist_10_50"] = (data["ma_10"] - data["ma_50"]) / (data["ma_50"] + 1e-9)

    # Strategy 2: EMA Crossover
    for p in [9, 21, 50, 100]:
        data[f"ema_{p}"] = c.ewm(span=p, adjust=False).mean()
    data["sig_ema_9_21"] = np.where(data["ema_9"] > data["ema_21"], 1, -1)
    data["sig_ema_21_50"] = np.where(data["ema_21"] > data["ema_50"], 1, -1)
    data["ema_dist_9_21"] = (data["ema_9"] - data["ema_21"]) / (data["ema_21"] + 1e-9)

    # Strategy 3: RSI
    for period in [7, 14, 21]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        data[f"rsi_{period}"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    data["sig_rsi"] = np.where(
        data["rsi_14"] < 30, 1, np.where(data["rsi_14"] > 70, -1, 0)
    )
    data["rsi_slope"] = data["rsi_14"].diff(3)

    # Strategy 4: MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]
    data["sig_macd"] = np.where(data["macd"] > data["macd_signal"], 1, -1)
    data["macd_hist_slope"] = data["macd_hist"].diff(2)

    # Strategy 5: Bollinger Bands
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    data["bb_upper"] = bb_mid + 2 * bb_std
    data["bb_lower"] = bb_mid - 2 * bb_std
    data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / (bb_mid + 1e-9)
    data["bb_pos"] = (c - data["bb_lower"]) / (
        data["bb_upper"] - data["bb_lower"] + 1e-9
    )
    data["sig_bb"] = np.where(
        c < data["bb_lower"], 1, np.where(c > data["bb_upper"], -1, 0)
    )
    data["bb_squeeze"] = (
        data["bb_width"] < data["bb_width"].rolling(50).mean()
    ).astype(int)

    # Strategy 6: Stochastic
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    data["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-9)
    data["stoch_d"] = data["stoch_k"].rolling(3).mean()
    data["sig_stoch"] = np.where(
        data["stoch_k"] < 20, 1, np.where(data["stoch_k"] > 80, -1, 0)
    )
    data["stoch_cross"] = np.where(data["stoch_k"] > data["stoch_d"], 1, -1)

    # Strategy 7: Breakout
    data["high_20"] = h.rolling(20).max().shift(1)
    data["low_20"] = l.rolling(20).min().shift(1)
    data["high_50"] = h.rolling(50).max().shift(1)
    data["low_50"] = l.rolling(50).min().shift(1)
    data["sig_breakout_20"] = np.where(
        c > data["high_20"], 1, np.where(c < data["low_20"], -1, 0)
    )
    data["sig_breakout_50"] = np.where(
        c > data["high_50"], 1, np.where(c < data["low_50"], -1, 0)
    )
    data["price_vs_high20"] = (c - data["high_20"]) / (data["high_20"] + 1e-9)

    # Strategy 8: Mean Reversion
    data["deviation_ma20"] = (c - data["ma_20"]) / (data["ma_20"] + 1e-9)
    data["deviation_ma50"] = (c - data["ma_50"]) / (data["ma_50"] + 1e-9)
    data["zscore_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-9)
    data["zscore_50"] = (c - c.rolling(50).mean()) / (c.rolling(50).std() + 1e-9)
    data["sig_mean_rev"] = np.where(
        data["zscore_20"] < -1.5, 1, np.where(data["zscore_20"] > 1.5, -1, 0)
    )

    # Strategy 9: ATR & Volatility
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    data["atr_14"] = tr.rolling(14).mean()
    data["atr_pct"] = data["atr_14"] / (c + 1e-9)
    data["volatility_20"] = c.pct_change().rolling(20).std()
    data["volatility_50"] = c.pct_change().rolling(50).std()
    data["vol_ratio"] = data["volatility_20"] / (data["volatility_50"] + 1e-9)

    # Strategy 10: SMC/ICT + Session + Price Action
    data["hh"] = (h > h.shift(1)).astype(int)
    data["ll"] = (l < l.shift(1)).astype(int)
    data["hl"] = (l > l.shift(1)).astype(int)
    data["lh"] = (h < h.shift(1)).astype(int)
    data["fvg_bull"] = np.where(l > h.shift(2), 1, 0)
    data["fvg_bear"] = np.where(h < l.shift(2), 1, 0)

    hour = data.index.hour
    data["is_london"] = ((hour >= 7) & (hour < 16)).astype(int)
    data["is_ny"] = ((hour >= 13) & (hour < 22)).astype(int)
    data["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    data["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    data["day_of_week"] = data.index.dayofweek

    data["returns_1"] = c.pct_change(1)
    data["returns_5"] = c.pct_change(5)
    data["returns_10"] = c.pct_change(10)
    data["returns_20"] = c.pct_change(20)
    data["body_size"] = abs(c - o) / (h - l + 1e-9)
    data["upper_wick"] = (
        h - pd.concat([c, o], axis=1).max(axis=1)
    ) / (h - l + 1e-9)
    data["lower_wick"] = (
        pd.concat([c, o], axis=1).min(axis=1) - l
    ) / (h - l + 1e-9)
    data["is_bullish"] = (c > o).astype(int)

    data["vol_ma_20"] = v.rolling(20).mean()
    data["vol_ratio_20"] = v / (data["vol_ma_20"] + 1e-9)
    data["obv"] = (np.sign(c.diff()) * v).cumsum()
    data["obv_ma"] = data["obv"].rolling(20).mean()
    data["sig_obv"] = np.where(data["obv"] > data["obv_ma"], 1, -1)

    return data.dropna()


# ── Step 3: تحميل الموديل والـ Scaler ────────────────────────────────────────
def load_model(model_dir: str, model_name: str):
    """Load model from ml/models/ directory"""
    model_files = {
        "LogisticRegression": "LogisticRegression.pkl",
        "XGBoost": "XGBoost.pkl",
        "RandomForest": "RandomForest.pkl",
        "GradientBoosting": "GradientBoosting.pkl",
    }
    filename = model_files.get(model_name)
    if filename is None:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(model_files.keys())}"
        )

    filepath = os.path.join(model_dir, filename)
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    return joblib.load(filepath)


def load_scaler(model_dir: str):
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not Path(scaler_path).exists():
        logger.warning("Scaler not found at %s", scaler_path)
        return None
    scaler = joblib.load(scaler_path)
    logger.info("Scaler loaded from %s", scaler_path)
    return scaler


def load_feature_cols(model_dir: str) -> list:
    """Load the exact feature columns used during training"""
    path = os.path.join(model_dir, "feature_cols.pkl")
    if Path(path).exists():
        cols = joblib.load(path)
        logger.info("Feature columns loaded: %d features", len(cols))
        return cols
    logger.warning("feature_cols.pkl not found — using all available features")
    return None


# ── Step 4: توليد الـ Signals ─────────────────────────────────────────────────
def generate_signals(
    data_with_features: pd.DataFrame,
    model,
    scaler,
    feature_cols: list,
) -> pd.Series:
    available = [f for f in feature_cols if f in data_with_features.columns]
    missing = [f for f in feature_cols if f not in data_with_features.columns]
    if missing:
        logger.warning("Missing %d features: %s", len(missing), missing[:5])

    X = data_with_features[available]  # keep as DataFrame to preserve feature names

    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=available, index=data_with_features.index)

    predictions = model.predict(X)
    # تحويل: 1 = buy signal, 0 = no signal / sell
    signals = pd.Series(
        np.where(predictions == 1, 1, -1),
        index=data_with_features.index,
    )
    return signals


# ── Step 5: تشغيل الـ Backtest ────────────────────────────────────────────────
def run_full_backtest():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()
    model_dir = args.model_dir

    # إضافة project root للـ path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from backtesting.backtest_engine import BacktestEngine

    print("\n" + "=" * 60)
    print("  🚀 HOPEFX Backtest Runner")
    print(f"  📊 Symbol:  {SYMBOL}")
    print(f"  📅 Period:  {BACKTEST_START.date()} → {BACKTEST_END.date()}")
    print(f"  💰 Balance: ${INITIAL_BALANCE:,.0f}")
    print("=" * 60)

    # 1. جلب البيانات
    df = fetch_mt5_data(SYMBOL, BACKTEST_START, BACKTEST_END, TIMEFRAME_STR)

    # 2. حساب الـ Features
    logger.info("Calculating features...")
    data = calculate_features(df)
    logger.info("  %d rows with features ready", len(data))

    # تحديد الـ feature columns
    exclude_cols = ["open", "high", "low", "close", "volume"]
    feature_cols = [c for c in data.columns if c not in exclude_cols]

    # 3. تحميل الـ Scaler
    scaler = load_scaler(model_dir)

    # تحميل feature_cols المحفوظة من التدريب (لضمان نفس الـ features)
    saved_feature_cols = load_feature_cols(model_dir)
    if saved_feature_cols:
        feature_cols = saved_feature_cols

    # 4. تشغيل الـ backtest لكل موديل
    models_to_test = ["LogisticRegression", "XGBoost", "RandomForest", "GradientBoosting"]
    all_results = {}

    for model_name in models_to_test:
        try:
            logger.info("Loading %s model...", model_name)
            model = load_model(model_dir, model_name)

            # توليد الـ signals
            signals = generate_signals(data, model, scaler, feature_cols)

            # تشغيل الـ backtest
            engine = BacktestEngine(
                initial_balance=INITIAL_BALANCE,
                position_size_pct=POSITION_SIZE_PCT,
                commission_pct=COMMISSION_PCT,
                atr_sl_mult=1.5,
                atr_tp_mult=2.5,
                use_atr_stops=True,
            )
            result = engine.run_backtest(data, signals)

            all_results[model_name] = (result, engine)

            # عرض النتائج
            print(f"\n{'='*50}")
            print(f"  📈 نتائج {model_name.upper()}")
            print(f"{'='*50}")
            print(f"  💰 رأس المال الأولي:   ${INITIAL_BALANCE:>12,.2f}")
            print(
                f"  💵 رأس المال النهائي:  "
                f"${INITIAL_BALANCE * (1 + result.total_return):>12,.2f}"
            )
            print(f"  📈 إجمالي العائد:      {result.total_return*100:>+11.2f}%")
            print(f"  ✅ نسبة الفوز:         {result.win_rate*100:>11.1f}%")
            print(f"  🔄 عدد الصفقات:        {result.total_trades:>12}")
            print(f"  📉 أقصى سحب:           {result.max_drawdown*100:>11.2f}%")
            print(f"  📊 Sharpe Ratio:       {result.sharpe_ratio:>12.3f}")
            print(f"  💎 Profit Factor:      {result.profit_factor:>12.3f}")
            if result.trades:
                tp_count  = sum(1 for t in result.trades if t.get("exit_reason") == "TP")
                sl_count  = sum(1 for t in result.trades if t.get("exit_reason") == "SL")
                sig_count = sum(1 for t in result.trades if t.get("exit_reason") == "signal")
                print(f"  🎯 TP hits:            {tp_count:>12}")
                print(f"  🛑 SL hits:            {sl_count:>12}")
                print(f"  🔄 Signal exits:       {sig_count:>12}")

            # Monte Carlo
            mc = engine.monte_carlo_analysis(n_simulations=1000)
            if mc:
                print(f"\n  🎲 Monte Carlo (1000 simulations):")
                print(f"     متوسط العائد:    {mc['mean_return']*100:>+8.2f}%")
                print(f"     أسوأ 5%:         {mc['percentile_5']*100:>+8.2f}%")
                print(f"     أفضل 95%:        {mc['percentile_95']*100:>+8.2f}%")

        except Exception as e:
            logger.error("Failed for %s: %s", model_name, e)

    # ملخص نهائي
    if all_results:
        print(f"\n{'='*50}")
        print("  🏆 ملخص المقارنة")
        print(f"{'='*50}")
        print(
            f"  {'الموديل':<20} {'العائد':>10}  {'نسبة الفوز':>12}  {'Sharpe':>8}"
        )
        print(f"  {'-'*52}")
        for name, (res, _) in all_results.items():
            print(
                f"  {name:<20} {res.total_return*100:>+9.2f}%"
                f"  {res.win_rate*100:>11.1f}%  {res.sharpe_ratio:>8.3f}"
            )
        print(f"{'='*50}\n")


if __name__ == "__main__":
    run_full_backtest()
