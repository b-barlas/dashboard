from __future__ import annotations

from typing import Callable

import pandas as pd
import ta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def ml_predict_direction(
    df: pd.DataFrame, debug_fn: Callable[[str], None] | None = None
) -> tuple[float, str]:
    """Predict next-candle direction using GradientBoosting with LR fallback."""
    if df is None or len(df) < 60:
        return 0.5, "NEUTRAL"

    debug = debug_fn or (lambda _msg: None)
    df = df.copy().reset_index(drop=True)

    df["ema5"] = ta.trend.ema_indicator(df["close"], window=5)
    df["ema9"] = ta.trend.ema_indicator(df["close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    macd_ind = ta.trend.MACD(df["close"])
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["returns"] = df["close"].pct_change()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    feature_cols = [
        "ema5",
        "ema9",
        "ema21",
        "rsi",
        "macd",
        "macd_signal",
        "macd_diff",
        "obv",
        "atr",
        "returns",
        "volume_ratio",
        "bb_width",
    ]
    df_features = df[feature_cols].shift(1)
    df_model = pd.concat([df_features, df["target"]], axis=1).dropna()

    if len(df_model) < 50:
        return 0.5, "NEUTRAL"

    X = df_model[feature_cols].astype(float).values
    y = df_model["target"].astype(int).values

    try:
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_pred = X[-1:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_pred_scaled = scaler.transform(X_pred)

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)
        prob_up = float(model.predict_proba(X_pred_scaled)[0][1])
    except Exception as e:
        debug(f"GradientBoosting failed ({e}), falling back to LogisticRegression")
        try:
            split_idx = int(len(X) * 0.8)
            model = LogisticRegression(max_iter=1000)
            model.fit(X[:split_idx], y[:split_idx])
            prob_up = float(model.predict_proba(X[-1:].reshape(1, -1))[0][1])
        except Exception:
            return 0.5, "NEUTRAL"

    direction = "LONG" if prob_up >= 0.6 else ("SHORT" if prob_up <= 0.4 else "NEUTRAL")
    return prob_up, direction


def ml_ensemble_predict(df: pd.DataFrame) -> tuple[float, str, dict]:
    """Ensemble ML prediction combining GB, RF, and LR models."""
    if df is None or len(df) < 60:
        return 0.5, "NEUTRAL", {}

    df = df.copy().reset_index(drop=True)
    df["ema5"] = ta.trend.ema_indicator(df["close"], window=5)
    df["ema9"] = ta.trend.ema_indicator(df["close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    macd_ind = ta.trend.MACD(df["close"])
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["returns"] = df["close"].pct_change()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df["ema_spread"] = (df["ema5"] - df["ema21"]) / df["close"]
    df["rsi_slope"] = df["rsi"].diff(3)
    df["vol_trend"] = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    feature_cols = [
        "ema5",
        "ema9",
        "ema21",
        "rsi",
        "macd",
        "macd_signal",
        "macd_diff",
        "obv",
        "atr",
        "returns",
        "volume_ratio",
        "bb_width",
        "ema_spread",
        "rsi_slope",
        "vol_trend",
    ]
    df_features = df[feature_cols].shift(1)
    df_model = pd.concat([df_features, df["target"]], axis=1).dropna()

    if len(df_model) < 50:
        return 0.5, "NEUTRAL", {}

    X = df_model[feature_cols].astype(float).values
    y = df_model["target"].astype(int).values

    try:
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_pred = X[-1:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_pred_scaled = scaler.transform(X_pred)

        gb = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08, subsample=0.85, random_state=42
        )
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)

        gb.fit(X_train_scaled, y_train)
        rf.fit(X_train_scaled, y_train)
        lr.fit(X_train_scaled, y_train)

        gb_prob = float(gb.predict_proba(X_pred_scaled)[0][1])
        rf_prob = float(rf.predict_proba(X_pred_scaled)[0][1])
        lr_prob = float(lr.predict_proba(X_pred_scaled)[0][1])

        prob_up = gb_prob * 0.45 + rf_prob * 0.35 + lr_prob * 0.20
        model_dirs = [
            "LONG" if p >= 0.58 else ("SHORT" if p <= 0.42 else "NEUTRAL")
            for p in (gb_prob, rf_prob, lr_prob)
        ]
        # Agreement = fraction of models voting for the dominant direction label.
        agreement = max(model_dirs.count("LONG"), model_dirs.count("SHORT"), model_dirs.count("NEUTRAL")) / 3.0
        details = {
            "gradient_boosting": gb_prob,
            "random_forest": rf_prob,
            "logistic_regression": lr_prob,
            "ensemble": prob_up,
            "agreement": agreement,
        }
    except Exception:
        return 0.5, "NEUTRAL", {}

    direction = "LONG" if prob_up >= 0.58 else ("SHORT" if prob_up <= 0.42 else "NEUTRAL")
    return prob_up, direction, details
