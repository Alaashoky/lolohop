#!/usr/bin/env python3
"""
HOPEFX — Improved Training Pipeline
يجمع: Confidence Threshold + Ensemble + More Data + Hyperparameter Tuning
"""

from __future__ import annotations

import os
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── إعدادات ───────────────────────────────────────────────────────────────
SYMBOL         = "XAUUSD.m"
TRAIN_START    = datetime(2020, 1, 1)
TRAIN_END      = datetime(2025, 1, 1)   # ← زيادة بيانات: أضفنا 2024 كاملة
TIMEFRAME      = "1h"
OUTPUT_DIR     = "ml/models_improved"
CONFIDENCE_THR = 0.60                   # ← Confidence Threshold
HORIZON        = 5                      # شمعات للأمام للهدف
N_SPLITS       = 5                      # TimeSeriesSplit


# ── Step 1: جلب البيانات ──────────────────────────────────────────────────
def fetch_data(symbol: str, start: datetime, end: datetime, interval: str = "1h") -> pd.DataFrame:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise RuntimeError("pip install MetaTrader5")

    tf_map = {
        "1m": mt5.TIMEFRAME_M1, "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15, "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }

    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    logger.info("Fetching %s %s → %s", symbol, start.date(), end.date())
    rates = mt5.copy_rates_range(symbol, tf_map.get(interval, mt5.TIMEFRAME_H1), start, end)
    if rates is None or len(rates) == 0:
        sym2 = symbol.replace(".m", "").replace(".M", "")
        rates = mt5.copy_rates_range(sym2, tf_map.get(interval, mt5.TIMEFRAME_H1), start, end)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError(f"No data for {symbol}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"time": "timestamp", "tick_volume": "volume"})
    df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].dropna()
    logger.info("  %d rows (%s → %s)", len(df), df.index[0].date(), df.index[-1].date())
    return df


# ── Step 2: حساب الـ Features (79 feature) ───────────────────────────────
def calculate_features(df: pd.DataFrame, horizon: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    data = df.copy()
    c = data["close"]; h = data["high"]; l = data["low"]
    o = data["open"];  v = data["volume"]

    # Strategy 1: MA
    for p in [5, 10, 20, 50, 100, 200]:
        data[f"ma_{p}"] = c.rolling(p).mean()
    data["sig_ma_5_20"]   = np.where(data["ma_5"]  > data["ma_20"],  1, -1)
    data["sig_ma_10_50"]  = np.where(data["ma_10"] > data["ma_50"],  1, -1)
    data["sig_ma_50_200"] = np.where(data["ma_50"] > data["ma_200"], 1, -1)
    data["ma_dist_5_20"]  = (data["ma_5"]  - data["ma_20"])  / (data["ma_20"]  + 1e-9)
    data["ma_dist_10_50"] = (data["ma_10"] - data["ma_50"])  / (data["ma_50"]  + 1e-9)

    # Strategy 2: EMA
    for p in [9, 21, 50, 100]:
        data[f"ema_{p}"] = c.ewm(span=p, adjust=False).mean()
    data["sig_ema_9_21"]  = np.where(data["ema_9"]  > data["ema_21"], 1, -1)
    data["sig_ema_21_50"] = np.where(data["ema_21"] > data["ema_50"], 1, -1)
    data["ema_dist_9_21"] = (data["ema_9"] - data["ema_21"]) / (data["ema_21"] + 1e-9)

    # Strategy 3: RSI
    for period in [7, 14, 21]:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        data[f"rsi_{period}"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    data["sig_rsi"]   = np.where(data["rsi_14"] < 30, 1, np.where(data["rsi_14"] > 70, -1, 0))
    data["rsi_slope"] = data["rsi_14"].diff(3)

    # Strategy 4: MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    data["macd"]            = ema12 - ema26
    data["macd_signal"]     = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"]       = data["macd"] - data["macd_signal"]
    data["sig_macd"]        = np.where(data["macd"] > data["macd_signal"], 1, -1)
    data["macd_hist_slope"] = data["macd_hist"].diff(2)

    # Strategy 5: Bollinger Bands
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    data["bb_upper"]   = bb_mid + 2 * bb_std
    data["bb_lower"]   = bb_mid - 2 * bb_std
    data["bb_width"]   = (data["bb_upper"] - data["bb_lower"]) / (bb_mid + 1e-9)
    data["bb_pos"]     = (c - data["bb_lower"]) / (data["bb_upper"] - data["bb_lower"] + 1e-9)
    data["sig_bb"]     = np.where(c < data["bb_lower"], 1, np.where(c > data["bb_upper"], -1, 0))
    data["bb_squeeze"] = (data["bb_width"] < data["bb_width"].rolling(50).mean()).astype(int)

    # Strategy 6: Stochastic
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    data["stoch_k"]     = 100 * (c - low14) / (high14 - low14 + 1e-9)
    data["stoch_d"]     = data["stoch_k"].rolling(3).mean()
    data["sig_stoch"]   = np.where(data["stoch_k"] < 20, 1, np.where(data["stoch_k"] > 80, -1, 0))
    data["stoch_cross"] = np.where(data["stoch_k"] > data["stoch_d"], 1, -1)

    # Strategy 7: Breakout
    data["high_20"] = h.rolling(20).max().shift(1)
    data["low_20"]  = l.rolling(20).min().shift(1)
    data["high_50"] = h.rolling(50).max().shift(1)
    data["low_50"]  = l.rolling(50).min().shift(1)
    data["sig_breakout_20"] = np.where(c > data["high_20"], 1, np.where(c < data["low_20"], -1, 0))
    data["sig_breakout_50"] = np.where(c > data["high_50"], 1, np.where(c < data["low_50"], -1, 0))
    data["price_vs_high20"] = (c - data["high_20"]) / (data["high_20"] + 1e-9)

    # Strategy 8: Mean Reversion
    data["deviation_ma20"] = (c - data["ma_20"]) / (data["ma_20"] + 1e-9)
    data["deviation_ma50"] = (c - data["ma_50"]) / (data["ma_50"] + 1e-9)
    data["zscore_20"]      = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-9)
    data["zscore_50"]      = (c - c.rolling(50).mean()) / (c.rolling(50).std() + 1e-9)
    data["sig_mean_rev"]   = np.where(data["zscore_20"] < -1.5, 1, np.where(data["zscore_20"] > 1.5, -1, 0))

    # Strategy 9: ATR & Volatility
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    data["atr_14"]        = tr.rolling(14).mean()
    data["atr_pct"]       = data["atr_14"] / (c + 1e-9)
    data["volatility_20"] = c.pct_change().rolling(20).std()
    data["volatility_50"] = c.pct_change().rolling(50).std()
    data["vol_ratio"]     = data["volatility_20"] / (data["volatility_50"] + 1e-9)

    # Strategy 10: SMC/ICT + Session + Price Action
    data["hh"] = (h > h.shift(1)).astype(int)
    data["ll"] = (l < l.shift(1)).astype(int)
    data["hl"] = (l > l.shift(1)).astype(int)
    data["lh"] = (h < h.shift(1)).astype(int)
    data["fvg_bull"] = np.where(l > h.shift(2), 1, 0)
    data["fvg_bear"] = np.where(h < l.shift(2), 1, 0)

    hour = data.index.hour
    data["is_london"]   = ((hour >= 7)  & (hour < 16)).astype(int)
    data["is_ny"]       = ((hour >= 13) & (hour < 22)).astype(int)
    data["sin_hour"]    = np.sin(2 * np.pi * hour / 24)
    data["cos_hour"]    = np.cos(2 * np.pi * hour / 24)
    data["day_of_week"] = data.index.dayofweek

    data["returns_1"]  = c.pct_change(1)
    data["returns_5"]  = c.pct_change(5)
    data["returns_10"] = c.pct_change(10)
    data["returns_20"] = c.pct_change(20)
    data["body_size"]  = abs(c - o) / (h - l + 1e-9)
    data["upper_wick"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (h - l + 1e-9)
    data["lower_wick"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / (h - l + 1e-9)
    data["is_bullish"] = (c > o).astype(int)

    data["vol_ma_20"]    = v.rolling(20).mean()
    data["vol_ratio_20"] = v / (data["vol_ma_20"] + 1e-9)
    data["obv"]          = (np.sign(c.diff()) * v).cumsum()
    data["obv_ma"]       = data["obv"].rolling(20).mean()
    data["sig_obv"]      = np.where(data["obv"] > data["obv_ma"], 1, -1)

    # Target: هل السعر هيرتفع خلال horizon شمعات؟
    future_ret = c.pct_change(horizon).shift(-horizon)
    data["target"] = (future_ret > 0).astype(int)

    data = data.dropna()

    exclude = {"open", "high", "low", "close", "volume", "target",
               "spread", "real_volume"}
    feature_cols = [col for col in data.columns if col not in exclude]

    return data[feature_cols], data["target"]


# ── Step 3: Hyperparameter Tuning ────────────────────────────────────────
def tune_logistic(X_train, y_train) -> LogisticRegression:
    logger.info("Tuning LogisticRegression...")
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [500, 1000],
        "solver": ["lbfgs", "saga"],
    }
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    gs = GridSearchCV(
        LogisticRegression(class_weight="balanced"),
        param_grid,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    logger.info("  Best LR params: %s  →  CV accuracy: %.4f", gs.best_params_, gs.best_score_)
    return gs.best_estimator_


def tune_xgboost(X_train, y_train) -> XGBClassifier:
    logger.info("Tuning XGBoost...")
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    gs = GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0),
        param_grid,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    logger.info("  Best XGB params: %s  →  CV accuracy: %.4f", gs.best_params_, gs.best_score_)
    return gs.best_estimator_


# ── Step 4: Ensemble (Voting) ────────────────────────────────────────────
def build_ensemble(lr_model, xgb_model) -> VotingClassifier:
    logger.info("Building Ensemble (LR + XGB)...")
    ensemble = VotingClassifier(
        estimators=[
            ("lr",  lr_model),
            ("xgb", xgb_model),
        ],
        voting="soft",   # يستخدم الاحتمالات — أدق من hard voting
        weights=[1, 2],  # XGBoost أثقل لأنه أقوى
    )
    return ensemble


# ── Step 5: Confidence Threshold Backtest ────────────────────────────────
def backtest_with_confidence(
    model,
    X_test: pd.DataFrame,
    data_test: pd.DataFrame,
    confidence_thr: float = 0.60,
    initial_balance: float = 10_000.0,
    position_size_pct: float = 0.10,
    commission_pct: float = 0.0002,
) -> dict:
    """
    Backtest مع Confidence Threshold:
    - يفتح صفقة buy  لو P(up) > confidence_thr
    - يفتح صفقة sell لو P(down) > confidence_thr
    - يبقى flat لو مفيش ثقة كافية
    """
    proba = model.predict_proba(X_test)
    prob_up   = proba[:, 1]
    prob_down = proba[:, 0]

    signals = np.zeros(len(X_test))
    signals[prob_up   > confidence_thr] =  1
    signals[prob_down > confidence_thr] = -1

    balance = initial_balance
    trades = []
    equity = [balance]
    position = 0
    entry_price = 0.0
    entry_sig = 0

    prices = data_test["close"].values
    sigs   = signals

    for i in range(1, len(prices)):
        prev_sig = sigs[i - 1]
        price    = prices[i]

        # Close
        if position != 0 and (prev_sig == 0 or prev_sig != entry_sig):
            pnl        = position * (price - entry_price)
            commission = abs(position) * price * commission_pct
            net_pnl    = pnl - commission
            balance   += net_pnl
            trades.append({"pnl": net_pnl, "win": net_pnl > 0})
            position = 0

        # Open
        if prev_sig != 0 and position == 0:
            size     = (balance * position_size_pct) / price
            position = size if prev_sig == 1 else -size
            entry_price = price
            entry_sig   = prev_sig

        equity.append(balance)

    # Force close
    if position != 0:
        price      = prices[-1]
        pnl        = position * (price - entry_price)
        commission = abs(position) * price * commission_pct
        net_pnl    = pnl - commission
        balance   += net_pnl
        trades.append({"pnl": net_pnl, "win": net_pnl > 0})
        equity[-1] = balance

    n          = len(trades)
    win_rate   = sum(1 for t in trades if t["win"]) / n if n > 0 else 0
    total_ret  = (balance - initial_balance) / initial_balance
    eq         = np.array(equity)
    roll_max   = np.maximum.accumulate(eq)
    max_dd     = float(((eq - roll_max) / roll_max).min())
    dr         = np.diff(eq) / eq[:-1]
    sharpe     = float(np.mean(dr) / np.std(dr) * np.sqrt(252)) if np.std(dr) > 0 else 0.0
    filtered_n = int(np.sum(sigs != 0))

    return {
        "total_return":     total_ret,
        "win_rate":         win_rate,
        "total_trades":     n,
        "filtered_signals": filtered_n,
        "max_drawdown":     max_dd,
        "sharpe_ratio":     sharpe,
        "final_balance":    balance,
    }


# ── Main Pipeline ─────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  🚀 HOPEFX Improved Training Pipeline")
    print(f"  📊 Symbol:     {SYMBOL}")
    print(f"  📅 Train:      {TRAIN_START.date()} → {TRAIN_END.date()}")
    print(f"  🎯 Confidence: {CONFIDENCE_THR*100:.0f}%")
    print(f"  🔮 Horizon:    {HORIZON} candles")
    print("=" * 65)

    # ── 1. جلب البيانات ──────────────────────────────────────────────
    df = fetch_data(SYMBOL, TRAIN_START, TRAIN_END, TIMEFRAME)

    # ── 2. حساب الـ Features ─────────────────────────────────────────
    logger.info("Calculating features (horizon=%d)...", HORIZON)
    X, y = calculate_features(df, horizon=HORIZON)
    logger.info("  Dataset: %d rows, %d features", len(X), X.shape[1])

    # ── 3. Train/Test split (آخر 20% للاختبار — زمنياً) ─────────────
    split = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    data_test       = df.loc[X_test.index]

    logger.info("  Train: %d | Test: %d", len(X_train), len(X_test))

    # ── 4. Scale ──────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # تحويل لـ DataFrame مع أسماء الـ features لتجنب الـ warnings
    X_train_s = pd.DataFrame(X_train_s, columns=X_train.columns, index=X_train.index)
    X_test_s  = pd.DataFrame(X_test_s,  columns=X_test.columns,  index=X_test.index)

    # ── 5. Hyperparameter Tuning ──────────────────────────────────────
    lr_model  = tune_logistic(X_train_s, y_train)
    xgb_model = tune_xgboost(X_train_s, y_train)

    # ── 6. Ensemble ───────────────────────────────────────────────────
    ensemble = build_ensemble(lr_model, xgb_model)
    ensemble.fit(X_train_s, y_train)

    # ── 7. تقييم الموديلات ───────────────────────────────────────────
    models = {
        "LogisticRegression (tuned)": lr_model,
        "XGBoost (tuned)":            xgb_model,
        "Ensemble (LR+XGB)":          ensemble,
    }

    print(f"\n{'='*65}")
    print(f"  {'الموديل':<28} {'Accuracy':>10}  {'F1':>8}  {'حالة':>12}")
    print(f"{'='*65}")

    best_model      = None
    best_model_name = ""
    best_accuracy   = 0.0

    for name, model in models.items():
        y_pred    = model.predict(X_test_s)
        acc       = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        tag       = ""
        if acc > best_accuracy:
            best_accuracy   = acc
            best_model      = model
            best_model_name = name
            tag = "🏆 الأفضل!"
        print(f"  {name:<28} {acc*100:>9.2f}%  {f1:>8.4f}  {tag}")

    print(f"{'='*65}\n")

    # ── 8. Backtest بـ Confidence Threshold على أفضل موديل ──────────
    logger.info("Running Backtest with Confidence=%.0f%%...", CONFIDENCE_THR * 100)
    bt = backtest_with_confidence(
        best_model, X_test_s, data_test,
        confidence_thr=CONFIDENCE_THR,
    )

    print(f"\n{'='*55}")
    print(f"  📊 نتائج Backtest — {best_model_name}")
    print(f"  🎯 Confidence Threshold: {CONFIDENCE_THR*100:.0f}%")
    print(f"{'='*55}")
    print(f"  💰 رأس المال الأولي:   $10,000.00")
    print(f"  💵 رأس المال النهائي:  ${bt['final_balance']:>12,.2f}")
    print(f"  📈 إجمالي العائد:      {bt['total_return']*100:>+11.2f}%")
    print(f"  ✅ نسبة الفوز:         {bt['win_rate']*100:>11.1f}%")
    print(f"  🔄 عدد الصفقات:        {bt['total_trades']:>12}")
    print(f"  🔽 إشارات مفلترة:      {bt['filtered_signals']:>12}")
    print(f"  📉 أقصى سحب:           {bt['max_drawdown']*100:>11.2f}%")
    print(f"  📊 Sharpe Ratio:       {bt['sharpe_ratio']:>12.3f}")
    print(f"{'='*55}\n")

    # ── 9. حفظ الموديل والـ Scaler ───────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, os.path.join(OUTPUT_DIR, f"{best_model_name.split()[0]}.pkl"))
    joblib.dump(scaler,     os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(list(X_train.columns), os.path.join(OUTPUT_DIR, "feature_cols.pkl"))

    # حفظ كل الموديلات
    for name, model in models.items():
        fname = name.split()[0].replace("(", "").replace(")", "") + ".pkl"
        joblib.dump(model, os.path.join(OUTPUT_DIR, fname))

    # حفظ الـ confidence threshold
    joblib.dump({"confidence_thr": CONFIDENCE_THR, "horizon": HORIZON},
                os.path.join(OUTPUT_DIR, "config.pkl"))

    logger.info("✅ All models saved to %s", OUTPUT_DIR)
    print(f"  💾 Saved to: {OUTPUT_DIR}/")
    print(f"  🏆 Best model: {best_model_name}")
    print(f"  📈 Best accuracy: {best_accuracy*100:.2f}%\n")


if __name__ == "__main__":
    main()
