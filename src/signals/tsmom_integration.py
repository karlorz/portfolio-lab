#!/usr/bin/env python3
"""
Portfolio-Lab v2.52: TSMOM Integration with Signal Integrator

Bridge module to connect TSMOM overlay signals with the v2.51 signal integrator.

This module provides:
- TSMOMSignalAdapter: Converts TSMOM overlay signals to SignalSourceResult format
- TSMOMIntegrator: High-level interface for integrator to use TSMOM signals
- Delta calculation: Converts TSMOM signals to portfolio allocation deltas

Usage:
    from src.signals.tsmom_integration import TSMOMSignalAdapter
    
    adapter = TSMOMSignalAdapter()
    signal = adapter.get_signal("SPY")
    
    # For full portfolio
    signals = adapter.get_portfolio_signals(["SPY", "GLD", "TLT"])
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.signals.tsmom_overlay import TSMOMOverlay, TSMOMSignal, DEFAULT_BASE_ALLOCATION
from src.signals.integrator import SignalSourceResult
from src.signals.sample_counts import get_live_sample_count
logger = logging.getLogger(__name__)




__all__ = ['TSMOMSignalAdapter', 'get_tsmom_integrator_result']

class TSMOMSignalAdapter:
    """
    Adapter to convert TSMOM overlay signals to SignalSourceResult format
    for integration with v2.51 signal integrator.
    """
    
    def __init__(
        self,
        base_allocation: Dict[str, float] = None,
        lookback_days: int = 252,
        max_deviation: float = 0.10
    ):
        self.overlay = TSMOMOverlay(
            lookback_days=lookback_days,
            max_deviation=max_deviation
        )
        self.base_allocation = base_allocation or DEFAULT_BASE_ALLOCATION.copy()
    
    def get_signal(self, ticker: str) -> Optional[SignalSourceResult]:
        """
        Get TSMOM signal for a single ticker as SignalSourceResult.
        """
        tsmom_signal = self.overlay.compute_signal(ticker)
        if tsmom_signal is None:
            return None
        
        return SignalSourceResult(
            source_type="tsmom",
            source_name="aqrs_tsmom",
            signal=tsmom_signal.signal,  # -1, 0, or +1
            confidence=self._compute_confidence(tsmom_signal),
            raw_score=tsmom_signal.lookback_return,
            raw_unit="formation_return_12m",
            historical_accuracy=0.68,  # Based on AQR research
            sample_count=get_live_sample_count(ticker),
            timestamp=tsmom_signal.timestamp,
            metadata={
                "lookback_return": tsmom_signal.lookback_return,
                "realized_vol": tsmom_signal.realized_vol,
                "vol_scaled_position": tsmom_signal.vol_scaled_position,
                "base_weight": tsmom_signal.base_weight,
                "adjustment": tsmom_signal.adjustment,
                "target_weight": tsmom_signal.target_weight,
            }
        )
    
    # Alias for SignalSource interface compatibility
    generate_signal = get_signal
    
    def get_portfolio_signals(
        self,
        tickers: List[str]
    ) -> Dict[str, SignalSourceResult]:
        """
        Get TSMOM signals for all tickers in portfolio.
        """
        signals = {}
        for ticker in tickers:
            signal = self.get_signal(ticker)
            if signal:
                signals[ticker] = signal
        return signals
    
    def get_allocation_deltas(
        self,
        tickers: List[str] = None
    ) -> Dict[str, float]:
        """
        Get allocation deltas for all tickers based on TSMOM signals.
        
        Returns mapping of ticker to delta (adjustment from base allocation).
        """
        tickers = tickers or ["SPY", "GLD", "TLT"]
        deltas = {}
        
        for ticker in tickers:
            tsmom_signal = self.overlay.compute_signal(ticker)
            if tsmom_signal:
                deltas[ticker] = tsmom_signal.adjustment
            else:
                deltas[ticker] = 0.0
        
        return deltas
    
    def _compute_confidence(self, signal: TSMOMSignal) -> float:
        """
        Compute confidence score for TSMOM signal.
        
        Factors:
        - Trend strength (magnitude of formation return)
        - Data quality (volatility stability)
        - Signal clarity (near-zero vs strong signal)
        """
        # Base confidence
        confidence = 0.50
        
        # Trend strength contribution (0 to 0.25)
        trend_strength = min(abs(signal.lookback_return) / 0.30, 1.0)
        confidence += trend_strength * 0.25
        
        # Volatility stability (inverse of vol, 0 to 0.15)
        # Lower vol = more reliable signal
        vol_stability = max(0, 1.0 - signal.realized_vol / 0.30)
        confidence += vol_stability * 0.15
        
        # Signal clarity (0 to 0.10)
        # Stronger signals (farther from 0) get higher confidence
        if abs(signal.signal) > 0:
            clarity = min(abs(signal.lookback_return) / 0.10, 1.0)
            confidence += clarity * 0.10
        
        return min(1.0, confidence)

    def get_signal_snapshot(self, tickers: List[str] = None):
        """Generate aggregated SignalSnapshot for ensemble voter consumption."""
        from src.signals.signal_snapshot import SignalSnapshot

        tickers = tickers or ["SPY", "GLD", "TLT"]
        deltas = self.get_allocation_deltas(tickers)
        signals = {}
        confidences = []
        for ticker in tickers:
            tsmom_signal = self.overlay.compute_signal(ticker)
            if tsmom_signal is not None:
                signals[ticker] = tsmom_signal.adjustment
                confidences.append(self._compute_confidence(tsmom_signal))

        if not signals:
            return SignalSnapshot(
                source="tsmom_integration",
                timestamp=str(datetime.now()),
                value=0.0,
                confidence=0.0,
                is_active=False,
                explanation="TSMOM: no signals available",
            )

        avg_value = sum(signals.values()) / len(signals)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.5

        return SignalSnapshot(
            source="tsmom_integration",
            timestamp=str(datetime.now()),
            value=float(np.clip(avg_value * 10, -1, 1)),  # Scale adjustments to signal range
            confidence=avg_conf,
            asset_signals=deltas,
            regime_fit="all",
            is_active=any(v != 0 for v in deltas.values()),
            explanation=f"TSMOM: avg_adj={avg_value:+.3f}, "
                        f"assets={list(signals)}, "
                        f"avg_conf={avg_conf:.3f}",
            metadata={"deltas": deltas},
        )


def get_tsmom_integrator_result(
    tickers: List[str] = None,
    base_allocation: Dict[str, float] = None
) -> Dict[str, SignalSourceResult]:
    """
    Convenience function to get all TSMOM signals for integrator.
    
    Usage in SignalIntegrator:
        tsmom_signals = get_tsmom_integrator_result(["SPY", "GLD", "TLT"])
        for ticker, signal in tsmom_signals.items():
            # signal is SignalSourceResult ready for integrator
    """
    adapter = TSMOMSignalAdapter(base_allocation=base_allocation)
    return adapter.get_portfolio_signals(tickers or ["SPY", "GLD", "TLT"])


# CLI for testing
if __name__ == "__main__":
    from src.utils.log_config import configure_logging

    configure_logging()
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="TSMOM Integration Adapter")
    parser.add_argument("--ticker", help="Single ticker to get signal for")
    parser.add_argument("--portfolio", action="store_true", help="Get full portfolio signals")
    parser.add_argument("--deltas", action="store_true", help="Get allocation deltas")
    
    args = parser.parse_args()
    
    adapter = TSMOMSignalAdapter()
    
    if args.ticker:
        signal = adapter.get_signal(args.ticker)
        if signal:
            logger.info(json.dumps({
                "ticker": args.ticker,
                "source_type": signal.source_type,
                "source_name": signal.source_name,
                "signal": signal.signal,
                "confidence": signal.confidence,
                "raw_score": signal.raw_score,
                "timestamp": signal.timestamp,
                "metadata": signal.metadata
            }, indent=2))
        else:
            logger.error("No signal for %s", args.ticker)

    elif args.portfolio:
        signals = adapter.get_portfolio_signals(["SPY", "GLD", "TLT"])
        logger.info(json.dumps({
            ticker: {
                "signal": s.signal,
                "confidence": s.confidence,
                "raw_score": s.raw_score,
                "adjustment": s.metadata.get("adjustment")
            }
            for ticker, s in signals.items()
        }, indent=2))

    elif args.deltas:
        deltas = adapter.get_allocation_deltas(["SPY", "GLD", "TLT"])
        logger.info(json.dumps(deltas, indent=2))

    else:
        parser.print_help()

        parser.print_help()
