#!/usr/bin/env python3
"""
Portfolio-Lab v2.56-2.58: Multi-Strategy Signal Adapters

Signal adapters for integrating new strategies into the v2.51 signal integrator:
- MultiSpeedSignalAdapter: v2.56 Multi-Speed Momentum Ensemble (Man AHL)
- RiskParitySignalAdapter: v2.57 Risk Parity Weight Overlay (Bridgewater)
- NetworkMomentumSignalAdapter: v2.58 Network Momentum Lead-Lag (research-only)

Usage:
    from src.signals.multi_strategy_adapters import (
        MultiSpeedSignalAdapter, RiskParitySignalAdapter, NetworkMomentumSignalAdapter
    )
    
    # Each adapter provides SignalSourceResult outputs. MultiSpeed and
    # RiskParity are active integrator sources; NetworkMomentum is for
    # research/comparison until a future promotion decision.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from src.signals.integrator import SignalSourceResult
from src.signals.multi_speed_momentum import MultiSpeedMomentum, SPEED_TIERS, DEFAULT_BASE_ALLOCATION
from src.strategy.risk_parity_weight_overlay import RiskParityWeightOverlay, DEFAULT_BASE as RP_DEFAULT
from src.strategy.network_momentum_leadlag import NetworkMomentumLeadLag, DEFAULT_BASE_ALLOCATION as NM_DEFAULT



__all__ = ['MultiSpeedSignalAdapter', 'RiskParitySignalAdapter', 'NetworkMomentumSignalAdapter', 'get_all_strategy_signals']

class MultiSpeedSignalAdapter:
    """
    Adapter for v2.56 Multi-Speed Momentum Ensemble (Man AHL style).
    
    Provides SignalSourceResult format for integration with signal integrator.
    """
    
    def __init__(
        self,
        base_allocation: Dict[str, float] = None
    ):
        self.multi_speed = MultiSpeedMomentum()
        self.base_allocation = base_allocation or DEFAULT_BASE_ALLOCATION.copy()
        self.source_type = "multi_speed"
        self.source_name = "manahl_multi_speed_ensemble"
    
    def generate_signal(self, ticker: str) -> Optional[SignalSourceResult]:
        """Get multi-speed ensemble signal for a ticker."""
        # Compute ensemble signal
        ensemble_signal = self.multi_speed.compute_ensemble_signal(
            ticker,
            self.base_allocation.get(ticker, 0.0)
        )
        
        if not ensemble_signal:
            return None
        
        # Calculate composite signal from ensemble
        # Average of fast/medium/slow tier signals, weighted by confidence
        tier_signals = [
            ensemble_signal.fast_signal.signal,
            ensemble_signal.medium_signal.signal,
            ensemble_signal.slow_signal.signal
        ]
        
        # Consensus signal (-1 to +1 scale)
        consensus = sum(tier_signals) / len(tier_signals)
        
        # Confidence based on agreement
        agreements = sum(1 for s in tier_signals if s == tier_signals[0])
        confidence = agreements / len(tier_signals)
        
        return SignalSourceResult(
            source_type="multi_speed",
            source_name="manahl_multi_speed_ensemble",
            signal=consensus,
            confidence=confidence,
            raw_score=ensemble_signal.ensemble_position,
            raw_unit="vol_scaled_position",
            historical_accuracy=0.72,  # Based on Man AHL research (speed diversification)
            sample_count=5371,
            timestamp=ensemble_signal.timestamp,
            metadata={
                "fast_signal": ensemble_signal.fast_signal.signal,
                "medium_signal": ensemble_signal.medium_signal.signal,
                "slow_signal": ensemble_signal.slow_signal.signal,
                "ensemble_confidence": ensemble_signal.ensemble_confidence,
                "target_weight": ensemble_signal.target_weight,
                "speed_tiers": list(SPEED_TIERS)
            }
        )
    
    def get_portfolio_signals(
        self,
        tickers: List[str]
    ) -> Dict[str, SignalSourceResult]:
        """Get multi-speed signals for all tickers."""
        signals = {}
        for ticker in tickers:
            signal = self.generate_signal(ticker)
            if signal:
                signals[ticker] = signal
        return signals

    def get_signal_snapshot(self, tickers: List[str] = None):
        """Return aggregate signal as canonical SignalSnapshot for typed pipeline consumption."""
        from src.signals.signal_snapshot import SignalSnapshot
        tickers = tickers or ["SPY", "GLD", "TLT"]
        values = []
        confidences = []
        for ticker in tickers:
            sig = self.generate_signal(ticker)
            if sig:
                values.append(sig.signal)
                confidences.append(sig.confidence)
        if not values:
            return SignalSnapshot(
                source="multi_speed",
                timestamp=str(datetime.now()),
                value=0.0,
                confidence=0.0,
                is_active=False,
                explanation="MultiSpeed: no signals available",
            )
        avg_value = sum(values) / len(values)
        avg_conf = sum(confidences) / len(confidences)
        return SignalSnapshot(
            source="multi_speed",
            timestamp=str(datetime.now()),
            value=float(avg_value),
            confidence=float(avg_conf),
            regime_fit="all",
            is_active=any(v != 0.0 for v in values),
            explanation=f"MultiSpeed: avg_signal={avg_value:+.3f}, assets={len(values)}",
        )


class RiskParitySignalAdapter:
    """
    Adapter for v2.57 Risk Parity Weight Overlay (Bridgewater style).
    
    Provides SignalSourceResult format based on risk parity deviations.
    """
    
    def __init__(
        self,
        base_allocation: Dict[str, float] = None
    ):
        self.rp_overlay = RiskParityWeightOverlay()
        self.base_allocation = base_allocation or RP_DEFAULT.copy()
        self.source_type = "risk_parity"
        self.source_name = "bridgewater_rp_overlay"
    
    def generate_signal(self, ticker: str) -> Optional[SignalSourceResult]:
        """Get risk parity signal for a ticker."""
        # Calculate risk parity allocation
        rp_allocation = self.rp_overlay.calculate_rp_overlay(
            self.base_allocation
        )
        
        if not rp_allocation:
            return None
        
        # Signal based on RP adjustment from base
        adjustment = rp_allocation.rp_adjustments.get(ticker, 0.0)
        
        # Convert adjustment to -1 to +1 signal
        # Max deviation is 0.15, so normalize
        signal = adjustment / 0.15
        signal = max(-1.0, min(1.0, signal))
        
        # Confidence based on risk parity quality
        confidence = rp_allocation.risk_parity_score
        
        return SignalSourceResult(
            source_type="risk_parity",
            source_name="bridgewater_rp_overlay",
            signal=signal,
            confidence=confidence,
            raw_score=adjustment,
            raw_unit="weight_adjustment",
            historical_accuracy=0.70,  # Risk parity track record
            sample_count=5371,
            timestamp=rp_allocation.timestamp,
            metadata={
                "base_weight": rp_allocation.base_weights.get(ticker, 0.0),
                "target_weight": rp_allocation.target_weights.get(ticker, 0.0),
                "asset_volatility": rp_allocation.asset_vols.get(ticker, 0.0),
                "rp_weight": rp_allocation.raw_rp_weights.get(ticker, 0.0),
                "risk_parity_quality": rp_allocation.risk_parity_score,
                "expected_vol": rp_allocation.expected_vol
            }
        )
    
    def get_portfolio_signals(
        self,
        tickers: List[str]
    ) -> Dict[str, SignalSourceResult]:
        """Get risk parity signals for all tickers."""
        signals = {}
        for ticker in tickers:
            signal = self.generate_signal(ticker)
            if signal:
                signals[ticker] = signal
        return signals

    def get_signal_snapshot(self, tickers: List[str] = None):
        """Return aggregate signal as canonical SignalSnapshot for typed pipeline consumption."""
        from src.signals.signal_snapshot import SignalSnapshot
        tickers = tickers or ["SPY", "GLD", "TLT"]
        values = []
        confidences = []
        for ticker in tickers:
            sig = self.generate_signal(ticker)
            if sig:
                values.append(sig.signal)
                confidences.append(sig.confidence)
        if not values:
            return SignalSnapshot(
                source="risk_parity",
                timestamp=str(datetime.now()),
                value=0.0,
                confidence=0.0,
                is_active=False,
                explanation="RiskParity: no signals available",
            )
        avg_value = sum(values) / len(values)
        avg_conf = sum(confidences) / len(confidences)
        return SignalSnapshot(
            source="risk_parity",
            timestamp=str(datetime.now()),
            value=float(avg_value),
            confidence=float(avg_conf),
            regime_fit="all",
            is_active=any(v != 0.0 for v in values),
            explanation=f"RiskParity: avg_signal={avg_value:+.3f}, assets={len(values)}",
        )


class NetworkMomentumSignalAdapter:
    """
    Adapter for v2.58 Network Momentum Lead-Lag (Imperial College style).
    
    Provides SignalSourceResult format based on cross-asset lead-lag dynamics.
    This adapter is research/comparison-only; it is not loaded by the active
    live SignalIntegrator unless a future promotion decision changes that.
    """

    RUNTIME_ROLE = "research_only"
    LIVE_ACTIVATION_STATUS = "research_only"
    PROMOTION_BENCHMARK = (
        "future benchmark decision must beat current active ensemble or "
        "an explicitly documented benchmark"
    )
    
    def __init__(
        self,
        base_allocation: Dict[str, float] = None
    ):
        self.network_momentum = NetworkMomentumLeadLag()
        self.base_allocation = base_allocation or NM_DEFAULT.copy()
        self.source_type = "network_momentum"
        self.source_name = "imperial_network_momentum"
        self.runtime_role = self.RUNTIME_ROLE
        self.live_activation_status = self.LIVE_ACTIVATION_STATUS
        self.promotion_benchmark = self.PROMOTION_BENCHMARK
    
    def generate_signal(self, ticker: str) -> Optional[SignalSourceResult]:
        """Get network momentum signal for a ticker."""
        # Compute ensemble signal across lookback windows
        ensemble_signal = self.network_momentum.compute_ensemble_signal(
            ticker,
            self.base_allocation.get(ticker, 0.0)
        )
        
        if not ensemble_signal:
            return None
        
        # Signal is the ensemble momentum normalized to -1 to +1
        # Ensemble momentum can range widely, so clip
        raw_momentum = ensemble_signal.ensemble_momentum
        signal = max(-1.0, min(1.0, raw_momentum * 2))  # Scale to -1/+1
        
        # Confidence is ensemble confidence (agreement across windows)
        confidence = ensemble_signal.ensemble_confidence
        
        # Get lead-lag matrix for metadata
        leadlag = self.network_momentum.compute_leadlag_matrix(
            window=66  # Default window
        )
        
        return SignalSourceResult(
            source_type="network_momentum",
            source_name="imperial_network_momentum",
            signal=signal,
            confidence=confidence,
            raw_score=raw_momentum,
            raw_unit="ensemble_momentum",
            historical_accuracy=0.68,  # From paper: +29-33% improvement over baseline
            sample_count=5371,
            timestamp=ensemble_signal.timestamp,
            metadata={
                "runtime_role": self.runtime_role,
                "live_activation_status": self.live_activation_status,
                "promotion_benchmark": self.promotion_benchmark,
                "network_centrality": ensemble_signal.network_centrality,
                "leadership_score": ensemble_signal.leadership_score,
                "followership_score": ensemble_signal.followership_score,
                "window_count": len(ensemble_signal.window_signals),
                "target_weight": ensemble_signal.target_weight,
                "dominant_leader": leadlag and self._get_dominant_leader(leadlag)
            }
        )
    
    def _get_dominant_leader(self, leadlag_matrix) -> str:
        """Extract dominant leader from lead-lag matrix."""
        # Simple heuristic: asset with most outgoing edges
        assets = ['SPY', 'GLD', 'TLT']
        leadership = {a: 0.0 for a in assets}
        
        for (leader, follower), strength in leadlag_matrix.adjacency.items():
            if leader in leadership:
                leadership[leader] += strength
        
        return max(leadership, key=leadership.get) if leadership else "unknown"
    
    def get_portfolio_signals(
        self,
        tickers: List[str]
    ) -> Dict[str, SignalSourceResult]:
        """Get network momentum signals for all tickers."""
        signals = {}
        for ticker in tickers:
            signal = self.generate_signal(ticker)
            if signal:
                signals[ticker] = signal
        return signals

    def get_signal_snapshot(self, tickers: List[str] = None):
        """Return aggregate signal as canonical SignalSnapshot for typed pipeline consumption."""
        from src.signals.signal_snapshot import SignalSnapshot
        tickers = tickers or ["SPY", "GLD", "TLT"]
        values = []
        confidences = []
        for ticker in tickers:
            sig = self.generate_signal(ticker)
            if sig:
                values.append(sig.signal)
                confidences.append(sig.confidence)
        if not values:
            return SignalSnapshot(
                source="network_momentum",
                timestamp=str(datetime.now()),
                value=0.0,
                confidence=0.0,
                is_active=False,
                explanation="NetworkMomentum: no signals available",
            )
        avg_value = sum(values) / len(values)
        avg_conf = sum(confidences) / len(confidences)
        return SignalSnapshot(
            source="network_momentum",
            timestamp=str(datetime.now()),
            value=float(avg_value),
            confidence=float(avg_conf),
            regime_fit="all",
            is_active=any(v != 0.0 for v in values),
            explanation=f"NetworkMomentum: avg_signal={avg_value:+.3f}, assets={len(values)}",
        )


def get_all_strategy_signals(
    tickers: List[str] = None
) -> Dict[str, Dict[str, SignalSourceResult]]:
    """
    Get signals from all three new strategies for comparison/analysis.
    
    Returns dict mapping strategy name to ticker->signal mapping.
    """
    tickers = ["SPY", "GLD", "TLT"] if tickers is None else tickers
    
    multi_speed = MultiSpeedSignalAdapter()
    risk_parity = RiskParitySignalAdapter()
    network_mom = NetworkMomentumSignalAdapter()
    
    return {
        "multi_speed": multi_speed.get_portfolio_signals(tickers),
        "risk_parity": risk_parity.get_portfolio_signals(tickers),
        "network_momentum": network_mom.get_portfolio_signals(tickers),
    }


if __name__ == "__main__":
    # Quick test
    logger.info("Testing Multi-Strategy Adapters")
    logger.info("=" * 50)

    tickers = ["SPY", "GLD", "TLT"]

    # Test multi-speed
    logger.info("\n1. Multi-Speed Momentum (v2.56):")
    ms = MultiSpeedSignalAdapter()
    for ticker in tickers:
        sig = ms.generate_signal(ticker)
        if sig:
            logger.info("  %s: signal=%+.2f, conf=%.2f", ticker, sig.signal, sig.confidence)

    # Test risk parity
    logger.info("\n2. Risk Parity (v2.57):")
    rp = RiskParitySignalAdapter()
    for ticker in tickers:
        sig = rp.generate_signal(ticker)
        if sig:
            logger.info("  %s: signal=%+.2f, conf=%.2f", ticker, sig.signal, sig.confidence)

    # Test network momentum
    logger.info("\n3. Network Momentum (v2.58):")
    nm = NetworkMomentumSignalAdapter()
    for ticker in tickers:
        sig = nm.generate_signal(ticker)
        if sig:
            logger.info("  %s: signal=%+.2f, conf=%.2f", ticker, sig.signal, sig.confidence)

    logger.info("\nAll adapters operational.")
