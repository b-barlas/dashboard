from __future__ import annotations

from typing import Callable

import numpy as np

import pandas as pd
import ta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _has_trusted_volume(df: pd.DataFrame) -> bool:
    if df is None or "volume" not in df.columns:
        return False
    attrs = getattr(df, "attrs", {}) or {}
    if "volume_is_24h_aggregate" in attrs:
        return not bool(attrs.get("volume_is_24h_aggregate"))
    if "volume_is_24h_aggregate" not in df.columns:
        return True
    try:
        return not bool(df["volume_is_24h_aggregate"].fillna(False).astype(bool).any())
    except Exception:
        return False


def ml_predict_direction(
    df: pd.DataFrame, debug_fn: Callable[[str], None] | None = None
) -> tuple[float, str]:
    """Predict next-candle direction using GradientBoosting with LR fallback."""
    if df is None or len(df) < 60:
        return 0.5, "NEUTRAL"

    debug = debug_fn or (lambda _msg: None)
    original_attrs = dict(getattr(df, "attrs", {}) or {})
    df = df.copy().reset_index(drop=True)
    df.attrs.update(original_attrs)

    df["ema5"] = ta.trend.ema_indicator(df["close"], window=5)
    df["ema9"] = ta.trend.ema_indicator(df["close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    macd_ind = ta.trend.MACD(df["close"])
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()
    trusted_volume = _has_trusted_volume(df)
    if trusted_volume:
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    else:
        df["obv"] = 0.0
        df["volume_ratio"] = 1.0
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["returns"] = df["close"].pct_change()
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
    # Predict t+1 direction from features known at t close.
    df_features = df[feature_cols]
    df_model = pd.concat([df_features, df["target"]], axis=1)
    df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df_model) < 50:
        return 0.5, "NEUTRAL"

    X = df_model[feature_cols].astype(float).values
    y = df_model["target"].astype(int).values

    try:
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        if len(set(y_train.tolist())) < 2:
            return 0.5, "NEUTRAL"
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
        return 0.5, "NEUTRAL", {
            "ensemble": 0.5,
            "ensemble_raw": 0.5,
            "agreement": 0.0,
            "directional_agreement": 0.0,
            "consensus_agreement": 0.0,
            "consensus_label": "NEUTRAL",
            "model_votes": ["NEUTRAL", "NEUTRAL", "NEUTRAL"],
            "gradient_boosting": 0.5,
            "random_forest": 0.5,
            "logistic_regression": 0.5,
            "status": "insufficient_candles",
            "error": "Need at least 60 candles.",
        }

    original_attrs = dict(getattr(df, "attrs", {}) or {})
    df = df.copy().reset_index(drop=True)
    df.attrs.update(original_attrs)
    df["ema5"] = ta.trend.ema_indicator(df["close"], window=5)
    df["ema9"] = ta.trend.ema_indicator(df["close"], window=9)
    df["ema21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    macd_ind = ta.trend.MACD(df["close"])
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()
    trusted_volume = _has_trusted_volume(df)
    if trusted_volume:
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        df["vol_trend"] = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()
    else:
        df["obv"] = 0.0
        df["volume_ratio"] = 1.0
        df["vol_trend"] = 1.0
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["returns"] = df["close"].pct_change()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df["ema_spread"] = (df["ema5"] - df["ema21"]) / df["close"]
    df["rsi_slope"] = df["rsi"].diff(3)
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
    # Predict t+1 direction from features known at t close.
    df_features = df[feature_cols]
    df_model = pd.concat([df_features, df["target"]], axis=1)
    df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df_model) < 50:
        return 0.5, "NEUTRAL", {
            "ensemble": 0.5,
            "ensemble_raw": 0.5,
            "agreement": 0.0,
            "directional_agreement": 0.0,
            "consensus_agreement": 0.0,
            "consensus_label": "NEUTRAL",
            "model_votes": ["NEUTRAL", "NEUTRAL", "NEUTRAL"],
            "gradient_boosting": 0.5,
            "random_forest": 0.5,
            "logistic_regression": 0.5,
            "status": "insufficient_features",
            "error": "Not enough clean feature rows after indicator warm-up.",
        }

    X = df_model[feature_cols].astype(float).values
    y = df_model["target"].astype(int).values

    try:
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        if len(set(y_train.tolist())) < 2:
            return 0.5, "NEUTRAL", {
                "ensemble": 0.5,
                "ensemble_raw": 0.5,
                "agreement": 0.0,
                "directional_agreement": 0.0,
                "consensus_agreement": 1.0,
                "consensus_label": "NEUTRAL",
                "model_votes": ["NEUTRAL", "NEUTRAL", "NEUTRAL"],
                "gradient_boosting": 0.5,
                "random_forest": 0.5,
                "logistic_regression": 0.5,
                "status": "single_class_window",
            }
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

        raw_prob_up = gb_prob * 0.45 + rf_prob * 0.35 + lr_prob * 0.20
        model_dirs = [
            "LONG" if p >= 0.58 else ("SHORT" if p <= 0.42 else "NEUTRAL")
            for p in (gb_prob, rf_prob, lr_prob)
        ]
        raw_direction = "LONG" if raw_prob_up >= 0.58 else ("SHORT" if raw_prob_up <= 0.42 else "NEUTRAL")
        vote_counts = {label: model_dirs.count(label) for label in ("LONG", "SHORT", "NEUTRAL")}
        max_votes = max(vote_counts.values())
        top_labels = [label for label, cnt in vote_counts.items() if cnt == max_votes]
        if len(top_labels) == 1:
            consensus_label = top_labels[0]
        elif raw_direction in top_labels:
            consensus_label = raw_direction
        elif "NEUTRAL" in top_labels:
            consensus_label = "NEUTRAL"
        else:
            consensus_label = "LONG" if raw_prob_up >= 0.5 else "SHORT"
        consensus_agreement = max_votes / 3.0
        raw_directional_agreement = (
            model_dirs.count(raw_direction) / 3.0 if raw_direction in {"LONG", "SHORT"} else 0.0
        )
        # Reliability-aware shrinkage: damp overconfident probabilities in low-agreement / low-sample regimes.
        sample_factor = max(0.35, min(1.0, (len(df_model) - 50) / 250.0))
        agreement_factor = 0.60 + 0.40 * raw_directional_agreement
        shrink = sample_factor * agreement_factor
        prob_up = 0.5 + (raw_prob_up - 0.5) * shrink
        direction = "LONG" if prob_up >= 0.58 else ("SHORT" if prob_up <= 0.42 else "NEUTRAL")
        directional_agreement = model_dirs.count(direction) / 3.0 if direction in {"LONG", "SHORT"} else 0.0
        details = {
            "gradient_boosting": gb_prob,
            "random_forest": rf_prob,
            "logistic_regression": lr_prob,
            "ensemble_raw": raw_prob_up,
            "ensemble": prob_up,
            # Backward-compatible key; now directional (LONG/SHORT only).
            "agreement": directional_agreement,
            "directional_agreement": directional_agreement,
            "consensus_agreement": consensus_agreement,
            "consensus_label": consensus_label,
            "model_votes": model_dirs,
        }
    except Exception as e:
        return 0.5, "NEUTRAL", {
            "ensemble": 0.5,
            "ensemble_raw": 0.5,
            "agreement": 0.0,
            "directional_agreement": 0.0,
            "consensus_agreement": 0.0,
            "consensus_label": "NEUTRAL",
            "model_votes": ["NEUTRAL", "NEUTRAL", "NEUTRAL"],
            "gradient_boosting": 0.5,
            "random_forest": 0.5,
            "logistic_regression": 0.5,
            "status": "model_exception",
            "error": str(e),
        }

    return prob_up, direction, details
