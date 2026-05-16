#!/usr/bin/env python3
"""
v5.43 — Pipeline Integration for v5.10-v5.30 Modules

Wires standalone modules into the active execution pipeline:
  1. Bayesian Vol (v5.20) → Vol Targeting (v2.42b)
  2. Vol-Volume-Gap (v5.30) → Execution Timing/Rebalance Scheduler
  3. Realized Vol Estimates → Signal Input Normalization

No ML deps — pure numpy/scipy integration code.

Usage:
    python -m src.pipeline.integration check           # Verify all modules are wired
    python -m src.pipeline.integration bayesian-vol    # Test Bayesian → vol targeting
    python -m src.pipeline.integration vol-volume-gap  # Test vol-volume-gap → execution
    python -m src.pipeline.integration status           # Show integration status
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# Integration 1: Bayesian Vol → Vol Targeting
# =============================================================================

def get_bayesian_vol_adjusted_target(
    symbol: str = "SPY",
    base_target: float = 0.10,
    bayesian_weight: float = 0.3
) -> Optional[dict]:
    """
    Replace simple rolling vol in vol targeting with Bayesian posterior.

    Combines:
    - Simple rolling vol estimate (current behavior)
    - Bayesian posterior vol (v5.20)

    Returns adjusted vol target or None if Bayesian module unavailable.
    """
    try:
        from src.monitor.bayesian_vol import estimate_bayesian_vol
    except ImportError:
        logger.warning("Bayesian vol module not available")
        return None

    try:
        result = estimate_bayesian_vol(symbol)
        if result is None:
            return None

        bayesian_vol = result.posterior_vol
        prior_vol = result.prior_vol

        # Blend: (1 - weight) * simple + weight * bayesian
        blended_vol = (1 - bayesian_weight) * prior_vol + bayesian_weight * bayesian_vol

        # Adjust target: if Bayesian vol is higher, reduce target
        vol_ratio = prior_vol / max(blended_vol, 0.001)
        adjusted_target = base_target * min(vol_ratio, 1.5)  # Cap at 1.5x

        return {
            "symbol": symbol,
            "base_target": base_target,
            "prior_vol": round(prior_vol, 4),
            "bayesian_vol": round(bayesian_vol, 4),
            "blended_vol": round(blended_vol, 4),
            "vol_ratio": round(vol_ratio, 4),
            "adjusted_target": round(adjusted_target, 4),
            "credible_interval": getattr(result, 'credible_interval', None),
            "posterior_df": getattr(result, 'posterior_df', None),
        }
    except Exception as e:
        logger.error(f"Bayesian vol adjustment failed: {e}")
        return None


# =============================================================================
# Integration 2: Vol-Volume-Gap → Execution Timing
# =============================================================================

def get_execution_adjustment(
    symbol: str = "SPY"
) -> Optional[dict]:
    """
    Get execution adjustment factor from Vol-Volume-Gap Day Classifier.

    Regime → adjustment mapping:
        CRISIS     → defer execution (0.0 factor)
        HIGH_VOL   → scale position sizes down (0.6 factor)
        TREND_UP   → normal execution (1.0 factor)
        TREND_DOWN → normal execution (1.0 factor)
        MEAN_REVERT → slight caution (0.8 factor)
        UNKNOWN    → normal execution (1.0 factor)
    """
    try:
        from src.regime.vol_volume_gap import (
            compute_features,
            classify_day,
            ClassifierConfig,
            DayRegime,
        )
    except ImportError:
        logger.warning("Vol-Volume-Gap module not available")
        return None

    try:
        # Load price data and compute features
        prices = _load_ohlcv(symbol)
        if prices is None or len(prices) < 30:
            logger.warning(f"Insufficient data for {symbol} vol-volume-gap classification")
            return None

        config = ClassifierConfig()
        features = compute_features(prices, config)
        if features is None:
            return None

        classified = classify_day(features, config)
        regime = classified.regime.value

        # Adjustment mapping
        adjustments = {
            "crisis": 0.0,      # Defer — freeze execution
            "high_vol": 0.6,    # Reduce position sizes
            "trend_up": 1.0,    # Normal
            "trend_down": 1.0,  # Normal
            "mean_revert": 0.8, # Slight caution
            "unknown": 1.0,     # Normal (fallback)
        }

        confidence = getattr(classified, 'confidence', 0.5)
        adjustment = adjustments.get(regime, 1.0)

        # Confidence-scale the adjustment: low confidence → closer to 1.0
        if confidence < 0.5:
            # Blend toward neutral (1.0)
            adjustment = 1.0 - (1.0 - adjustment) * confidence * 2

        return {
            "symbol": symbol,
            "regime": regime,
            "confidence": round(confidence, 4),
            "base_adjustment": adjustment,
            "adjusted_factor": round(adjustment, 4),
            "action": _get_execution_action(regime, adjustment),
        }
    except Exception as e:
        logger.error(f"Execution adjustment failed: {e}")
        return None


def _get_execution_action(regime: str, adjustment: float) -> str:
    """Determine execution action from regime and adjustment."""
    if adjustment == 0.0:
        return "DEFER_EXECUTION"
    elif adjustment < 0.7:
        return "REDUCE_SIZE"
    elif adjustment < 0.9:
        return "SLIGHT_CAUTION"
    else:
        return "NORMAL_EXECUTION"


# =============================================================================
# Integration 3: Realized Vol → Signal Normalization
# =============================================================================

def get_vol_normalized_signal(
    signal_value: float,
    symbol: str = "SPY",
    window: int = 63
) -> Optional[dict]:
    """
    Normalize signal value by current realized volatility.

    Divides signal by realized vol to produce vol-standardized signal.
    Higher vol → lower effective signal (risk-reduce).
    """
    try:
        prices = _load_prices(symbol)
        if prices is None or len(prices) < window + 5:
            return None

        returns = np.diff(prices) / prices[:-1]
        recent_returns = returns[-window:]

        # Realized volatility (annualized)
        realized_vol = float(np.std(recent_returns) * np.sqrt(252))
        if realized_vol < 0.01:  # Prevent division by near-zero
            realized_vol = 0.01

        # Long-term vol baseline
        long_vol = float(np.std(returns) * np.sqrt(252))
        if long_vol < 0.01:
            long_vol = 0.10

        # Vol ratio: current vs long-term
        vol_ratio = realized_vol / long_vol

        # Normalized signal: signal / vol_ratio
        # If vol is elevated, signal is reduced
        normalized_signal = signal_value / max(vol_ratio, 0.3)
        normalized_signal = max(-1.0, min(1.0, normalized_signal))  # Clamp

        return {
            "symbol": symbol,
            "raw_signal": signal_value,
            "realized_vol": round(realized_vol, 4),
            "long_term_vol": round(long_vol, 4),
            "vol_ratio": round(vol_ratio, 4),
            "normalized_signal": round(normalized_signal, 4),
            "window_days": window,
        }
    except Exception as e:
        logger.error(f"Signal normalization failed: {e}")
        return None


def _load_prices(symbol: str) -> Optional[np.ndarray]:
    """Load prices from the project price file."""
    prices_path = PROJECT_ROOT / "public/data/prices.json"
    if not prices_path.exists():
        return None
    try:
        with open(prices_path) as f:
            data = json.load(f)
        if symbol not in data:
            return None
        raw = data[symbol]
        if isinstance(raw, dict) and "p" in raw:
            return np.array(raw["p"], dtype=np.float64)
        elif isinstance(raw, list):
            return np.array(raw, dtype=np.float64)
        return None
    except Exception:
        return None


def _load_ohlcv(symbol: str) -> Optional[np.ndarray]:
    """Load OHLCV-style data (columnar) for vol_volume_gap module.

    Returns n×1 array of close prices (close-only mode).
    The vol_volume_gap module handles the 1-column case.
    """
    prices = _load_prices(symbol)
    if prices is None:
        return None
    # Reshape to n×1 for vol_volume_gap's OHLCV-compatible interface
    return prices.reshape(-1, 1)


# =============================================================================
# Integration Status
# =============================================================================

def check_integration_status() -> dict:
    """Check which integrations are operational."""
    # 1. Bayesian Vol module availability
    bayesian_ok = False
    try:
        from src.monitor import bayesian_vol
        bayesian_ok = hasattr(bayesian_vol, "estimate_bayesian_vol")
    except (ImportError, Exception):
        pass

    # 2. Bayesian → Vol Targeting test
    bayesian_target = None
    if bayesian_ok:
        bayesian_target = get_bayesian_vol_adjusted_target()

    # 3. Vol-Volume-Gap module availability
    vvg_ok = False
    try:
        from src.regime import vol_volume_gap
        vvg_ok = hasattr(vol_volume_gap, "classify_day")
    except (ImportError, Exception):
        pass

    # 4. Vol-Volume-Gap execution test
    execution_adjustment = get_execution_adjustment() if vvg_ok else None

    # 5. Check price data availability
    prices = _load_prices("SPY")
    data_ok = prices is not None and len(prices) > 100

    # 6. Realized vol normalization test
    norm_test = get_vol_normalized_signal(0.5) if data_ok else None

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "integrations": {
            "bayesian_vol_to_targeting": {
                "available": bayesian_ok,
                "operational": bayesian_target is not None,
                "test_result": bayesian_target,
            },
            "vol_volume_gap_to_execution": {
                "available": vvg_ok,
                "operational": execution_adjustment is not None,
                "test_result": execution_adjustment,
            },
            "realized_vol_to_signal": {
                "available": data_ok,
                "operational": norm_test is not None,
                "test_result": norm_test,
            },
        },
        "overall_status": "operational" if (
            bayesian_target is not None or execution_adjustment is not None or norm_test is not None
        ) else "partial",
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Portfolio-Lab v5.43: Pipeline Integration"
    )
    parser.add_argument(
        "command",
        choices=["check", "bayesian-vol", "vol-volume-gap", "normalize", "status"],
        help="Integration command"
    )
    parser.add_argument(
        "--symbol", "-s",
        default="SPY",
        help="Symbol (default: SPY)"
    )
    parser.add_argument(
        "--signal", "-v",
        type=float,
        default=0.5,
        help="Signal value for normalization test (default: 0.5)"
    )
    parser.add_argument(
        "--target", "-t",
        type=float,
        default=0.10,
        help="Vol target (default: 0.10 = 10%)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to file"
    )

    args = parser.parse_args()

    if args.command == "check":
        status = check_integration_status()
        print(f"\n=== Pipeline Integration Status ===")
        print(f"Time: {status['timestamp']}")
        print(f"Overall: {status['overall_status']}")
        for name, info in status["integrations"].items():
            print(f"\n  {name}:")
            print(f"    Available: {'YES' if info['available'] else 'NO'}")
            print(f"    Operational: {'YES' if info['operational'] else 'NO'}")
            if info.get("test_result"):
                tr = info["test_result"]
                if name == "bayesian_vol_to_targeting":
                    print(f"    Adjusted target: {tr.get('adjusted_target', 'N/A')}")
                elif name == "vol_volume_gap_to_execution":
                    print(f"    Regime: {tr.get('regime', 'N/A')}")
                    print(f"    Action: {tr.get('action', 'N/A')}")
                elif name == "realized_vol_to_signal":
                    print(f"    Normalized signal: {tr.get('normalized_signal', 'N/A')}")

        if args.save:
            status_path = DATA_DIR / "pipeline_integration_status.json"
            status_path.parent.mkdir(parents=True, exist_ok=True)
            with open(status_path, "w") as f:
                json.dump(status, f, indent=2, default=str)
            logger.info(f"Status saved to {status_path}")

    elif args.command == "bayesian-vol":
        result = get_bayesian_vol_adjusted_target(args.symbol, args.target)
        if result:
            print(f"\n=== Bayesian Vol → Vol Targeting: {args.symbol} ===")
            print(f"Base target:    {result['base_target']:.2%}")
            print(f"Prior vol:      {result['prior_vol']:.2%}")
            print(f"Bayesian vol:   {result['bayesian_vol']:.2%}")
            print(f"Blended vol:    {result['blended_vol']:.2%}")
            print(f"Vol ratio:      {result['vol_ratio']:.3f}")
            print(f"Adjusted target: {result['adjusted_target']:.2%}")
        else:
            print(f"Bayesian vol adjustment unavailable for {args.symbol}")

    elif args.command == "vol-volume-gap":
        result = get_execution_adjustment(args.symbol)
        if result:
            print(f"\n=== Vol-Volume-Gap → Execution Timing: {args.symbol} ===")
            print(f"Regime:         {result['regime']}")
            print(f"Confidence:     {result['confidence']:.1%}")
            print(f"Adjustment:     {result['adjusted_factor']:.2f}")
            print(f"Action:         {result['action']}")
        else:
            print(f"Execution adjustment unavailable for {args.symbol}")

    elif args.command == "normalize":
        result = get_vol_normalized_signal(args.signal, args.symbol)
        if result:
            print(f"\n=== Vol-Normalized Signal: {args.symbol} ===")
            print(f"Raw signal:      {result['raw_signal']:+.4f}")
            print(f"Realized vol:    {result['realized_vol']:.2%}")
            print(f"Long-term vol:   {result['long_term_vol']:.2%}")
            print(f"Vol ratio:       {result['vol_ratio']:.3f}")
            print(f"Normalized:      {result['normalized_signal']:+.4f}")
        else:
            print(f"Signal normalization unavailable for {args.symbol}")

    elif args.command == "status":
        run_integration_status_check()


def run_integration_status_check() -> None:
    """Comprehensive integration status check for periodic monitoring."""
    status = check_integration_status()
    print(f"\n{'='*60}")
    print(f"  Pipeline Integration Status (v5.43)")
    print(f"{'='*60}")
    print(f"  Timestamp: {status['timestamp']}")
    print(f"  Overall:   {status['overall_status'].upper()}")
    print(f"{'='*60}")

    for name, info in status["integrations"].items():
        label = name.replace("_", " ").title()
        icon = "✅" if info.get("operational") else "❌" if info.get("available") else "⚠️"
        print(f"  {icon} {label}: {'Operational' if info.get('operational') else 'Not Operational' if info.get('available') else 'Unavailable'}")

    # Save status
    status_path = DATA_DIR / "pipeline_integration_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2, default=str)
    logger.info(f"Integration status saved to {status_path}")


if __name__ == "__main__":
    main()
