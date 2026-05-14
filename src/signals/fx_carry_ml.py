"""
Portfolio-Lab v3.19: ML-Enhanced FX Carry Timing

RandomForest classifier for predicting currency carry unwind events.
Gates carry allocation during high-risk periods to avoid drawdowns.

Research (Q3 2026): Traditional FX carry Sharpe 0.3 → ML-enhanced 0.5-0.6.
Features: yield curve steepness, momentum, volatility, economic surprise.

Integrates with v3.15 fx_carry_signal.py infrastructure.

Usage:
    python -m src.signals.fx_carry_ml train
    python -m src.signals.fx_carry_ml predict
    python -m src.signals.fx_carry_ml status
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json
import logging
import pickle
import argparse
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MODEL_PATH = Path("~/projects/portfolio-lab/data/fx_carry_ml_model.pkl").expanduser()
PRICES_PATH = Path("~/projects/portfolio-lab/public/data/prices.json").expanduser()

# FX-related tickers in prices.json
FX_PROXIES = {
    "UUP": "USD Bullish (DXY proxy)",
    "UDN": "USD Bearish (inverse DXY)",
    "FXE": "EUR/USD",
    "FXY": "JPY/USD",
    "FXB": "GBP/USD",
    "FXA": "AUD/USD",
    "FXC": "CAD/USD",
}

# Feature windows
MOMENTUM_WINDOW = 63   # ~3 month
VOLATILITY_WINDOW = 21  # ~1 month
CARRY_SIGNAL_WINDOW = 126  # ~6 month for regime detection

# Unwind definition: >2% drawdown in 5 days following carry signal
UNWIND_THRESHOLD = -2.0
UNWIND_WINDOW = 5


@dataclass
class FXCarryMLFeatures:
    """Feature vector for carry unwind prediction."""
    ticker: str
    date: str
    momentum_1m: float      # 1-month return
    momentum_3m: float      # 3-month return
    volatility_1m: float    # 1-month realized vol
    carry_signal: float     # Spot-vs-forward spread proxy
    trend_strength: float   # Price / SMA(63)
    rate_differential: float  # Short-term yield proxy
    unwind_label: int = 0   # 1 if unwind occurred in next 5 days


@dataclass
class FXCarryMLPrediction:
    """ML prediction output for carry timing."""
    ticker: str
    unwind_risk: float       # 0.0 to 1.0 probability
    risk_level: str          # low / medium / high
    carry_allowed: bool      # True if risk < threshold
    features_used: int
    timestamp: datetime


def compute_features(
    prices: List[dict],
    ticker: str,
    label_unwinds: bool = False
) -> List[FXCarryMLFeatures]:
    """Compute ML features from price history.

    Args:
        prices: List of {d, p} dicts sorted by date ascending
        ticker: Ticker symbol
        label_unwinds: If True, compute unwind labels for training

    Returns:
        List of FXCarryMLFeatures, one per date with sufficient history
    """
    closes = np.array([p["p"] for p in prices])
    dates = [p["d"] for p in prices]

    if len(closes) < CARRY_SIGNAL_WINDOW + UNWIND_WINDOW:
        return []

    features = []
    for i in range(CARRY_SIGNAL_WINDOW, len(closes) - (UNWIND_WINDOW if label_unwinds else 0)):
        window = closes[:i + 1]
        current = closes[i]

        # Momentum features
        mom_1m = (current / closes[max(0, i - 21)] - 1) * 100 if i >= 21 else 0.0
        mom_3m = (current / closes[max(0, i - 63)] - 1) * 100 if i >= 63 else 0.0

        # Volatility
        if i >= VOLATILITY_WINDOW:
            returns = np.diff(window[-VOLATILITY_WINDOW:]) / window[-VOLATILITY_WINDOW:-1]
            vol_1m = np.std(returns) * np.sqrt(252) * 100
        else:
            vol_1m = 0.0

        # Trend strength (price vs 63-day SMA)
        sma_63 = np.mean(window[-63:]) if len(window) >= 63 else np.mean(window)
        trend_strength = (current / sma_63 - 1) * 100 if sma_63 > 0 else 0.0

        # Carry signal proxy: 6m return vs 1m return differential
        if i >= CARRY_SIGNAL_WINDOW:
            ret_6m = (current / closes[i - CARRY_SIGNAL_WINDOW] - 1) * 100
            ret_1m = (current / closes[max(0, i - 21)] - 1) * 100
            carry_signal = ret_6m - ret_1m  # positive = sustained trend
        else:
            carry_signal = 0.0

        # Rate differential proxy: momentum of carry signal
        rate_diff = mom_3m - mom_1m

        feat = FXCarryMLFeatures(
            ticker=ticker,
            date=dates[i],
            momentum_1m=round(mom_1m, 4),
            momentum_3m=round(mom_3m, 4),
            volatility_1m=round(vol_1m, 4),
            carry_signal=round(carry_signal, 4),
            trend_strength=round(trend_strength, 4),
            rate_differential=round(rate_diff, 4),
        )

        # Label unwind events for training
        if label_unwinds:
            future_closes = closes[i + 1:i + 1 + UNWIND_WINDOW]
            if len(future_closes) >= UNWIND_WINDOW:
                future_return = (future_closes[-1] / current - 1) * 100
                feat.unwind_label = 1 if future_return < UNWIND_THRESHOLD else 0

        features.append(feat)

    return features


def features_to_array(features: List[FXCarryMLFeatures]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert feature list to numpy arrays for sklearn.

    Returns:
        X: feature matrix (n_samples, n_features)
        y: label vector (n_samples,) or None if no labels
    """
    X = np.array([
        [f.momentum_1m, f.momentum_3m, f.volatility_1m,
         f.carry_signal, f.trend_strength, f.rate_differential]
        for f in features
    ])
    y = np.array([f.unwind_label for f in features])
    return X, y


def train_model(
    prices_path: Optional[str] = None,
    model_path: Optional[str] = None,
    tickers: Optional[List[str]] = None
) -> dict:
    """Train RandomForest classifier for carry unwind prediction.

    Args:
        prices_path: Path to prices.json
        model_path: Where to save the trained model
        tickers: FX tickers to train on (default: UUP only)

    Returns:
        Dict with training metrics
    """
    if prices_path is None:
        prices_path = str(PRICES_PATH)
    if model_path is None:
        model_path = str(DEFAULT_MODEL_PATH)
    if tickers is None:
        tickers = ["UUP"]

    with open(prices_path) as f:
        data = json.load(f)

    all_features = []
    for ticker in tickers:
        if ticker in data:
            prices = sorted(data[ticker], key=lambda x: x["d"])
            feats = compute_features(prices, ticker, label_unwinds=True)
            all_features.extend(feats)
            logger.info(f"  {ticker}: {len(feats)} feature vectors")

    if not all_features:
        raise ValueError("No training data — no FX tickers found in prices.json")

    X, y = features_to_array(all_features)

    # Filter out nan rows
    valid = ~np.isnan(X).any(axis=1)
    X, y = X[valid], y[valid]

    n_unwinds = y.sum()
    n_total = len(y)
    logger.info(f"Training: {n_total} samples, {n_unwinds} unwind events ({n_unwinds/n_total*100:.1f}%)")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='f1')

    # Fit final model
    model.fit(X, y)

    # Feature importance
    feature_names = ['momentum_1m', 'momentum_3m', 'volatility_1m',
                     'carry_signal', 'trend_strength', 'rate_differential']
    importances = dict(zip(feature_names, model.feature_importances_))

    # Save model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'trained_date': datetime.now().isoformat(),
            'n_samples': n_total,
            'n_unwinds': int(n_unwinds),
        }, f)

    return {
        "model_path": model_path,
        "n_samples": n_total,
        "n_unwinds": int(n_unwinds),
        "unwind_rate": round(n_unwinds / n_total * 100, 1),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
        "feature_importance": importances,
    }


def predict_unwind_risk(
    ticker: str,
    prices_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> FXCarryMLPrediction:
    """Predict carry unwind risk for a ticker using trained model.

    Args:
        ticker: Ticker symbol (UUP, FXE, etc.)
        prices_path: Path to prices.json
        model_path: Path to trained model pickle

    Returns:
        FXCarryMLPrediction with risk assessment
    """
    if prices_path is None:
        prices_path = str(PRICES_PATH)
    if model_path is None:
        model_path = str(DEFAULT_MODEL_PATH)

    # Load model
    model_file = Path(model_path).expanduser()
    if not model_file.exists():
        return FXCarryMLPrediction(
            ticker=ticker, unwind_risk=0.5, risk_level="unknown",
            carry_allowed=True, features_used=0,
            timestamp=datetime.now()
        )

    with open(model_file, 'rb') as f:
        bundle = pickle.load(f)

    model = bundle['model']
    feature_names = bundle['feature_names']

    # Load price data
    with open(prices_path) as f:
        data = json.load(f)

    if ticker not in data:
        raise ValueError(f"No price data for ticker: {ticker}")

    prices = sorted(data[ticker], key=lambda x: x["d"])
    features = compute_features(prices, ticker, label_unwinds=False)

    if not features:
        return FXCarryMLPrediction(
            ticker=ticker, unwind_risk=0.5, risk_level="unknown",
            carry_allowed=True, features_used=0,
            timestamp=datetime.now()
        )

    # Use most recent feature vector
    latest = features[-1]
    X = np.array([[
        latest.momentum_1m, latest.momentum_3m, latest.volatility_1m,
        latest.carry_signal, latest.trend_strength, latest.rate_differential
    ]])

    # NaN guard
    if np.isnan(X).any():
        return FXCarryMLPrediction(
            ticker=ticker, unwind_risk=0.5, risk_level="unknown",
            carry_allowed=True, features_used=0,
            timestamp=datetime.now()
        )

    proba = model.predict_proba(X)[0]
    unwind_risk = float(proba[1]) if len(proba) > 1 else 0.5

    # Risk classification
    if unwind_risk < 0.3:
        risk_level = "low"
        carry_allowed = True
    elif unwind_risk < 0.5:
        risk_level = "medium"
        carry_allowed = True
    else:
        risk_level = "high"
        carry_allowed = False

    return FXCarryMLPrediction(
        ticker=ticker,
        unwind_risk=round(unwind_risk, 4),
        risk_level=risk_level,
        carry_allowed=carry_allowed,
        features_used=len(feature_names),
        timestamp=datetime.now()
    )


def get_carry_allocation(
    prediction: FXCarryMLPrediction,
    base_weight: float = 5.0
) -> float:
    """Convert ML prediction to allocation weight.

    Args:
        prediction: FXCarryMLPrediction from predict_unwind_risk()
        base_weight: Maximum FX carry allocation (default 5%)

    Returns:
        Allocation weight from 0.0 to base_weight
    """
    if prediction.risk_level == "unknown":
        # No model available — use half weight as neutral
        return base_weight * 0.5
    elif prediction.risk_level == "low":
        return base_weight
    elif prediction.risk_level == "medium":
        return base_weight * 0.75
    else:  # high risk
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="ML-Enhanced FX Carry Timing v3.19")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Train RandomForest model")
    train_p.add_argument("--tickers", nargs="+", default=["UUP"], help="FX tickers")
    train_p.add_argument("--prices", help="Path to prices.json")
    train_p.add_argument("--model", help="Path to save model")

    predict_p = sub.add_parser("predict", help="Predict unwind risk")
    predict_p.add_argument("--ticker", default="UUP", help="FX ticker")
    predict_p.add_argument("--prices", help="Path to prices.json")
    predict_p.add_argument("--model", help="Path to trained model")

    status_p = sub.add_parser("status", help="Show carry timing status")

    args = parser.parse_args()

    if args.command == "train":
        result = train_model(
            prices_path=args.prices,
            model_path=args.model,
            tickers=args.tickers
        )
        print(f"Model trained: {result['n_samples']} samples, "
              f"{result['n_unwinds']} unwinds ({result['unwind_rate']}%)")
        print(f"CV F1: {result['cv_f1_mean']:.4f} +/- {result['cv_f1_std']:.4f}")
        print(f"Feature importance: {result['feature_importance']}")
        print(f"Saved to: {result['model_path']}")

    elif args.command == "predict":
        pred = predict_unwind_risk(
            args.ticker,
            prices_path=args.prices,
            model_path=args.model
        )
        print(f"{pred.ticker}: unwind_risk={pred.unwind_risk:.3f}, "
              f"risk={pred.risk_level}, carry_allowed={pred.carry_allowed}")

    elif args.command == "status":
        from src.signals.fx_carry_signal import FXCarrySignalGenerator

        print("=== ML-Enhanced FX Carry Status (v3.19) ===")
        model_file = Path(DEFAULT_MODEL_PATH)
        if model_file.exists():
            print(f"Model: {model_file} (trained)")
        else:
            print(f"Model: NOT TRAINED (run 'python -m src.signals.fx_carry_ml train')")

        try:
            pred = predict_unwind_risk("UUP")
            alloc = get_carry_allocation(pred, base_weight=5.0)
            print(f"UUP unwind risk: {pred.unwind_risk:.3f} ({pred.risk_level})")
            print(f"Carry allocation: {alloc:.1f}% of 5% max")
        except Exception as e:
            print(f"Prediction error: {e}")

        # Current FX carry signal
        try:
            gen = FXCarrySignalGenerator()
            signal = gen.generate()
            print(f"FX carry signal: {signal.signal_type} (confidence={signal.confidence:.2f})")
        except Exception:
            pass

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
