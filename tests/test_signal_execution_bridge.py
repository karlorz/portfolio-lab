#!/usr/bin/env python3
"""
Tests for signal execution bridge — urgency calculation, allocation deltas,
order generation, and bridge result serialization.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.execution.signal_execution_bridge import (
    SignalExecutionBridge, AllocationDelta, BridgeResult,
)
from src.execution.rebalance_scheduler import OrderUrgency, ScheduledOrder
from src.signals.integrator import CompositeSignal


class TestCalculateUrgency:
    """Test urgency classification from signal characteristics."""

    def _make_bridge(self):
        bridge = SignalExecutionBridge.__new__(SignalExecutionBridge)
        bridge.portfolio_value = 100000.0
        bridge._price_cache = {}
        return bridge

    def test_urgent_high_score_high_confidence(self):
        bridge = self._make_bridge()
        urgency = bridge._calculate_urgency(0.80, 0.85, "normal")
        assert urgency == OrderUrgency.URGENT

    def test_high_moderate_score(self):
        bridge = self._make_bridge()
        urgency = bridge._calculate_urgency(0.55, 0.65, "normal")
        assert urgency == OrderUrgency.HIGH

    def test_normal_mild_signal(self):
        bridge = self._make_bridge()
        urgency = bridge._calculate_urgency(0.30, 0.45, "normal")
        assert urgency == OrderUrgency.NORMAL

    def test_low_neutral_signal(self):
        bridge = self._make_bridge()
        urgency = bridge._calculate_urgency(0.05, 0.20, "normal")
        assert urgency == OrderUrgency.LOW

    def test_crisis_boosts_urgency(self):
        bridge = self._make_bridge()
        # Same signal in crisis should be higher urgency than normal
        normal = bridge._calculate_urgency(0.40, 0.50, "normal")
        crisis = bridge._calculate_urgency(0.40, 0.50, "crisis")
        assert crisis.value >= normal.value

    def test_high_vol_boosts_urgency(self):
        bridge = self._make_bridge()
        normal = bridge._calculate_urgency(0.40, 0.50, "normal")
        high_vol = bridge._calculate_urgency(0.40, 0.50, "high_vol")
        assert high_vol.value >= normal.value

    def test_bearish_in_bull_reduces_urgency(self):
        bridge = self._make_bridge()
        # Bearish signal (negative score) in bull market gets caution reduction
        normal = bridge._calculate_urgency(-0.40, 0.50, "neutral")
        bull_cautious = bridge._calculate_urgency(-0.40, 0.50, "bull")
        # The regime_boost is -0.10, so urgency should be same or lower
        urgency_order = ["low", "normal", "high", "urgent"]
        assert urgency_order.index(bull_cautious.value) <= urgency_order.index(normal.value)


class TestGenerateAllocationDeltas:
    """Test delta generation from composite signals."""

    def _make_bridge(self, mock_signal_score=0.5, mock_confidence=0.7, mock_regime="normal"):
        bridge = SignalExecutionBridge.__new__(SignalExecutionBridge)
        bridge.portfolio_value = 100000.0
        bridge._price_cache = {}
        bridge.MAX_SINGLE_DELTA = 0.10

        # Mock integrator
        mock_integrator = MagicMock()
        mock_signal = CompositeSignal(
            ticker="SPY",
            timestamp=datetime.now().isoformat(),
            component_signals=[],
            composite_score=mock_signal_score,
            composite_confidence=mock_confidence,
            primary_drivers=["test"],
            signal_agreement="aligned",
            detected_regime=mock_regime,
            weights_used={},
            expected_accuracy=None,
        )
        mock_integrator.get_composite_signal.return_value = mock_signal
        bridge.integrator = mock_integrator
        return bridge

    def test_bullish_signal_creates_buy_delta(self):
        bridge = self._make_bridge(mock_signal_score=0.6, mock_confidence=0.8)
        deltas, regime = bridge.generate_allocation_deltas({"SPY": 0.46})
        assert len(deltas) == 1
        assert deltas[0].delta > 0  # bullish → increase weight
        assert deltas[0].symbol == "SPY"

    def test_bearish_signal_creates_sell_delta(self):
        bridge = self._make_bridge(mock_signal_score=-0.6, mock_confidence=0.8)
        deltas, regime = bridge.generate_allocation_deltas({"SPY": 0.46})
        assert len(deltas) == 1
        assert deltas[0].delta < 0  # bearish → decrease weight

    def test_neutral_signal_skipped(self):
        bridge = self._make_bridge(mock_signal_score=0.0, mock_confidence=0.0)
        deltas, regime = bridge.generate_allocation_deltas({"SPY": 0.46})
        assert len(deltas) == 0

    def test_weak_signal_below_threshold_skipped(self):
        # Very low score * confidence → delta < 1% → skipped
        bridge = self._make_bridge(mock_signal_score=0.05, mock_confidence=0.10)
        deltas, regime = bridge.generate_allocation_deltas({"SPY": 0.46})
        assert len(deltas) == 0

    def test_clamped_to_max_delta(self):
        bridge = self._make_bridge(mock_signal_score=1.0, mock_confidence=1.0)
        deltas, regime = bridge.generate_allocation_deltas({"SPY": 0.46}, max_delta=0.05)
        if deltas:
            assert abs(deltas[0].delta) <= 0.05 + 0.001  # small float tolerance

    def test_target_weight_clamped_to_bounds(self):
        bridge = self._make_bridge(mock_signal_score=-1.0, mock_confidence=1.0)
        deltas, regime = bridge.generate_allocation_deltas({"SPY": 0.08})
        if deltas:
            assert deltas[0].target_weight >= 0.05  # floor

    def test_multiple_symbols(self):
        bridge = self._make_bridge(mock_signal_score=0.5, mock_confidence=0.7)
        portfolio = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        deltas, regime = bridge.generate_allocation_deltas(portfolio)
        # All three should get deltas (same mock signal)
        assert len(deltas) == 3

    def test_regime_detected(self):
        bridge = self._make_bridge(mock_regime="crisis")
        deltas, regime = bridge.generate_allocation_deltas({"SPY": 0.46})
        assert regime == "crisis"

    def test_integrator_exception_fallback(self):
        bridge = SignalExecutionBridge.__new__(SignalExecutionBridge)
        bridge.portfolio_value = 100000.0
        bridge._price_cache = {}
        bridge.MAX_SINGLE_DELTA = 0.10

        mock_integrator = MagicMock()
        mock_integrator.get_composite_signal.side_effect = RuntimeError("no data")
        bridge.integrator = mock_integrator

        deltas, regime = bridge.generate_allocation_deltas({"SPY": 0.46})
        # Fallback to neutral signal → score=0, confidence=0 → no delta
        assert len(deltas) == 0
        assert regime == "neutral"


class TestDeltasToOrders:
    """Test conversion of allocation deltas to scheduled orders."""

    def _make_bridge(self, price=500.0):
        bridge = SignalExecutionBridge.__new__(SignalExecutionBridge)
        bridge.portfolio_value = 100000.0
        bridge._price_cache = {"SPY": price}
        bridge.db_path = MagicMock()
        bridge.db_path.exists.return_value = True

        mock_scheduler = MagicMock()
        mock_order = ScheduledOrder(
            order_id="test_001",
            symbol="SPY",
            side="buy",
            target_shares=10.0,
            target_value=5000.0,
            urgency=OrderUrgency.NORMAL,
            created_at=datetime.now(),
            estimated_cost_bps=5.0,
        )
        mock_scheduler.schedule_order.return_value = mock_order
        bridge.scheduler = mock_scheduler
        return bridge

    def test_buy_order_from_positive_delta(self):
        bridge = self._make_bridge()
        delta = AllocationDelta(
            symbol="SPY", current_weight=0.46, target_weight=0.50,
            delta=0.04, confidence=0.8, urgency=OrderUrgency.NORMAL,
            signal_score=0.5, estimated_value=4000.0,
        )
        orders = bridge._deltas_to_orders([delta])
        assert len(orders) == 1
        bridge.scheduler.schedule_order.assert_called_once()
        call_kwargs = bridge.scheduler.schedule_order.call_args
        assert call_kwargs.kwargs["side"] == "buy"

    def test_sell_order_from_negative_delta(self):
        bridge = self._make_bridge()
        delta = AllocationDelta(
            symbol="SPY", current_weight=0.50, target_weight=0.46,
            delta=-0.04, confidence=0.8, urgency=OrderUrgency.NORMAL,
            signal_score=-0.5, estimated_value=4000.0,
        )
        orders = bridge._deltas_to_orders([delta])
        assert len(orders) == 1
        call_kwargs = bridge.scheduler.schedule_order.call_args
        assert call_kwargs.kwargs["side"] == "sell"

    def test_skip_below_min_trade_value(self):
        bridge = self._make_bridge()
        delta = AllocationDelta(
            symbol="SPY", current_weight=0.46, target_weight=0.461,
            delta=0.001, confidence=0.8, urgency=OrderUrgency.LOW,
            signal_score=0.05, estimated_value=100.0,  # < MIN_TRADE_VALUE
        )
        orders = bridge._deltas_to_orders([delta])
        assert len(orders) == 0

    def test_skip_no_price(self):
        bridge = self._make_bridge(price=None)
        bridge._price_cache = {}
        bridge.db_path.exists.return_value = False
        delta = AllocationDelta(
            symbol="SPY", current_weight=0.46, target_weight=0.50,
            delta=0.04, confidence=0.8, urgency=OrderUrgency.NORMAL,
            signal_score=0.5, estimated_value=4000.0,
        )
        orders = bridge._deltas_to_orders([delta])
        assert len(orders) == 0


class TestBridgeResult:
    """Test BridgeResult serialization."""

    def test_to_dict(self):
        delta = AllocationDelta(
            symbol="SPY", current_weight=0.46, target_weight=0.50,
            delta=0.04, confidence=0.8, urgency=OrderUrgency.NORMAL,
            signal_score=0.5, estimated_value=4000.0,
        )
        result = BridgeResult(
            timestamp=datetime.now().isoformat(),
            portfolio_value=100000.0,
            regime="normal",
            deltas=[delta],
            orders=[],
            total_estimated_cost_bps=5.0,
            dry_run=True,
        )
        d = result.to_dict()
        assert d["portfolio_value"] == 100000.0
        assert d["regime"] == "normal"
        assert d["dry_run"] is True
        assert d["num_deltas"] == 1
        assert d["num_orders"] == 0
        assert len(d["deltas"]) == 1
        assert d["deltas"][0]["symbol"] == "SPY"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
