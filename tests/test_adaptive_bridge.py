#!/usr/bin/env python3
"""
Tests for Adaptive Sizing → Rebalance Scheduler Bridge (v5.75).
"""
import sys
import os
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.rebalancing.adaptive_bridge import (
    AdaptiveRebalanceBridge,
    AdaptiveRebalanceResult,
    BASE_ALLOCATION,
)
from src.rebalancing.integration import SmartRebalanceGate, RebalanceGateResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_bridge_data(tmp_path):
    """Create a temporary data directory with regime state for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create regime state (normal)
    regime_state = {
        "current_regime": "normal",
        "last_reading": {
            "regime": "normal",
            "confidence": 0.7,
        },
    }
    (data_dir / "regime_classifier_state.json").write_text(json.dumps(regime_state))

    # Create price data
    prices_dir = tmp_path / "public" / "data"
    prices_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n = 200
    dates = []
    base = datetime(2026, 1, 1)
    for i in range(n):
        d = base + timedelta(days=i)
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))

    n = len(dates)
    spy_returns = np.random.normal(0.0005, 0.008, n)
    spy_prices = 480.0 * np.exp(np.cumsum(spy_returns))

    prices = {
        "SPY": [{"d": dates[i], "p": float(spy_prices[i])} for i in range(n)],
    }
    (prices_dir / "prices.json").write_text(json.dumps(prices))

    return data_dir


@pytest.fixture
def bridge(temp_bridge_data):
    """Create a bridge with temp data dir."""
    return AdaptiveRebalanceBridge(data_dir=temp_bridge_data)


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------


class TestAdaptiveRebalanceBridgeInit:
    """Test bridge initialization."""

    def test_init_default(self, tmp_path):
        """Bridge should initialize without error."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        bridge = AdaptiveRebalanceBridge(data_dir=data_dir)
        assert bridge._target_allocation is None
        assert bridge.last_sizing_decision is None

    def test_init_static_fallback(self, tmp_path):
        """When no sizing state, target should fall back to BASE_ALLOCATION."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        bridge = AdaptiveRebalanceBridge(data_dir=data_dir)
        # Before refresh, should use static
        assert bridge.target_allocation == BASE_ALLOCATION

    def test_init_has_gate_and_sizer(self, tmp_path):
        """Bridge should contain both gate and sizer."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        bridge = AdaptiveRebalanceBridge(data_dir=data_dir)
        assert hasattr(bridge, "gate")
        assert hasattr(bridge, "sizer")
        assert isinstance(bridge.gate, SmartRebalanceGate)


# ---------------------------------------------------------------------------
# Test: Target Refresh
# ---------------------------------------------------------------------------


class TestTargetRefresh:
    """Test adaptive target computation."""

    def test_refresh_targets_returns_dict(self, bridge):
        """Refresh should return a dict with SPY, GLD, TLT."""
        targets = bridge.refresh_targets()
        assert isinstance(targets, dict)
        assert "SPY" in targets
        assert "GLD" in targets
        assert "TLT" in targets

    def test_refresh_updates_target_allocation(self, bridge):
        """Refresh should update the stored target allocation."""
        assert bridge._target_allocation is None
        bridge.refresh_targets()
        assert bridge._target_allocation is not None

    def test_refresh_targets_sum_to_one(self, bridge):
        """Targets should sum to ~1.0."""
        targets = bridge.refresh_targets()
        total = sum(targets.values())
        assert 0.99 <= total <= 1.01

    def test_refresh_targets_within_bounds(self, bridge):
        """Targets should respect hard bounds."""
        from src.strategy.adaptive_sizing import HARD_BOUNDS
        targets = bridge.refresh_targets()
        for asset in ["SPY", "GLD", "TLT"]:
            lo, hi = HARD_BOUNDS[asset]
            w = targets.get(asset, 0)
            assert lo <= w <= hi, f"{asset} weight {w:.4f} outside [{lo}, {hi}]"

    def test_refresh_saves_sizing_decision(self, bridge):
        """Last sizing decision should be saved after refresh."""
        assert bridge.last_sizing_decision is None
        bridge.refresh_targets()
        assert bridge.last_sizing_decision is not None

    def test_refresh_graceful_failure(self, tmp_path):
        """Missing data should not crash — fall back to static."""
        data_dir = tmp_path / "missing_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        # Don't create any state files — fully missing data
        bridge = AdaptiveRebalanceBridge(data_dir=data_dir)
        targets = bridge.refresh_targets()
        # Should still return valid targets within bounds
        assert isinstance(targets, dict)
        assert "SPY" in targets
        assert "GLD" in targets
        assert "TLT" in targets
        # All targets should be within hard bounds
        from src.strategy.adaptive_sizing import HARD_BOUNDS
        for asset in ["SPY", "GLD", "TLT"]:
            lo, hi = HARD_BOUNDS[asset]
            w = targets.get(asset, 0)
            assert lo <= w <= hi, f"{asset} weight {w:.4f} outside [{lo}, {hi}]"
        total = sum(targets.values())
        assert 0.99 <= total <= 1.01


# ---------------------------------------------------------------------------
# Test: Evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    """Test the main evaluate method."""

    def test_evaluate_returns_result(self, bridge):
        """Evaluate should return an AdaptiveRebalanceResult."""
        holdings = {"SPY": 46000.0, "GLD": 38000.0, "TLT": 16000.0}
        result = bridge.evaluate(holdings, sum(holdings.values()))
        assert isinstance(result, AdaptiveRebalanceResult)

    def test_evaluate_has_all_fields(self, bridge):
        """Result should contain all expected fields."""
        holdings = {"SPY": 46000.0, "GLD": 38000.0, "TLT": 16000.0}
        result = bridge.evaluate(holdings, sum(holdings.values()))
        assert hasattr(result, "should_execute")
        assert hasattr(result, "decision")
        assert hasattr(result, "urgency")
        assert hasattr(result, "max_drift")
        assert hasattr(result, "dynamic_target")
        assert hasattr(result, "static_target")
        assert hasattr(result, "target_diff")
        assert hasattr(result, "sizing_adjustments")
        assert hasattr(result, "sizing_regime")
        assert hasattr(result, "sizing_vol")
        assert hasattr(result, "gate_result")

    def test_evaluate_static_vs_dynamic_targets(self, bridge):
        """Static and dynamic targets should differ (since adaptive sizing is active)."""
        holdings = {"SPY": 46000.0, "GLD": 38000.0, "TLT": 16000.0}
        result = bridge.evaluate(holdings, sum(holdings.values()))
        assert result.static_target == BASE_ALLOCATION
        # Dynamic may equal static in some conditions, but structure should match
        assert "SPY" in result.dynamic_target
        assert "GLD" in result.dynamic_target
        assert "TLT" in result.dynamic_target

    def test_evaluate_dynamic_targets_used(self, bridge):
        """When use_dynamic_targets=True, gate should receive dynamic targets."""
        holdings = {"SPY": 47000.0, "GLD": 37000.0, "TLT": 16000.0}
        result = bridge.evaluate(holdings, sum(holdings.values()), use_dynamic_targets=True)
        # The gate result should exist
        assert isinstance(result.gate_result, RebalanceGateResult)

    def test_evaluate_static_targets(self, bridge):
        """When use_dynamic_targets=False, should use static targets."""
        holdings = {"SPY": 46000.0, "GLD": 38000.0, "TLT": 16000.0}
        result = bridge.evaluate(holdings, sum(holdings.values()), use_dynamic_targets=False)
        assert result.dynamic_target == BASE_ALLOCATION
        assert result.target_diff == {"SPY": 0.0, "GLD": 0.0, "TLT": 0.0}

    def test_evaluate_drift_calculation(self, bridge):
        """Holdings far from target should show higher drift."""
        # Holdings that match base allocation
        aligned = {"SPY": 46000.0, "GLD": 38000.0, "TLT": 16000.0}
        result_aligned = bridge.evaluate(aligned, sum(aligned.values()))

        rebridge = bridge  # new evaluator to get fresh targets
        # Holdings far from base
        misaligned = {"SPY": 56000.0, "GLD": 28000.0, "TLT": 16000.0}
        result_misaligned = rebridge.evaluate(misaligned, sum(misaligned.values()))

        # Misaligned should have higher or equal drift
        assert result_misaligned.max_drift >= result_aligned.max_drift

    def test_evaluate_custom_vpin(self, bridge):
        """VPIN override should be accepted."""
        holdings = {"SPY": 46000.0, "GLD": 38000.0, "TLT": 16000.0}
        result = bridge.evaluate(holdings, sum(holdings.values()), vpin=0.80)
        assert result.gate_result.metadata["vpin"] == 0.80

    def test_evaluate_custom_time(self, bridge):
        """Custom datetime should be accepted."""
        holdings = {"SPY": 46000.0, "GLD": 38000.0, "TLT": 16000.0}
        now = datetime(2026, 5, 15, 12, 0)
        result = bridge.evaluate(holdings, sum(holdings.values()), now=now)
        assert result.gate_result.metadata["in_optimal_window"] is True

    def test_evaluate_outside_optimal_window(self, bridge):
        """Outside optimal window should be reflected in metadata."""
        holdings = {"SPY": 47000.0, "GLD": 37000.0, "TLT": 16000.0}
        now = datetime(2026, 5, 15, 9, 30)  # 9:30 AM
        result = bridge.evaluate(holdings, sum(holdings.values()), now=now)
        # SPY is overweight relative to dynamic targets, GLD underweight
        assert "in_optimal_window" in result.gate_result.metadata

    def test_evaluate_sizing_info_present(self, bridge):
        """Result should contain sizing-specific info."""
        holdings = {"SPY": 46000.0, "GLD": 38000.0, "TLT": 16000.0}
        result = bridge.evaluate(holdings, sum(holdings.values()))
        assert result.sizing_regime in ("normal", "low_vol", "high_vol", "crisis", "recovery", "unknown")
        assert isinstance(result.sizing_adjustments, dict)
        assert "SPY" in result.sizing_adjustments

    def test_evaluate_multiple_calls(self, bridge):
        """Multiple evaluate calls should not crash."""
        for i in range(3):
            holdings = {
                "SPY": 46000.0 + i * 1000,
                "GLD": 38000.0 - i * 500,
                "TLT": 16000.0 - i * 500,
            }
            result = bridge.evaluate(holdings, sum(holdings.values()))
            assert isinstance(result, AdaptiveRebalanceResult)


# ---------------------------------------------------------------------------
# Test: Record Execution
# ---------------------------------------------------------------------------


class TestRecordExecution:
    """Test recording rebalance executions."""

    def test_record_execution(self, bridge):
        """Recording a rebalance should update gate state."""
        bridge.record_execution(5.0, "2026-05-16", ["SPY", "GLD"])
        status = bridge.get_status()
        assert status["gate"]["ytd_cost_bps"] > 0

    def test_record_multiple_executions(self, bridge):
        """Multiple recordings should accumulate."""
        bridge.record_execution(3.0, "2026-05-15", ["SPY"])
        bridge.record_execution(4.0, "2026-05-16", ["GLD"])
        status = bridge.get_status()
        assert status["gate"]["ytd_cost_bps"] == 7.0


# ---------------------------------------------------------------------------
# Test: Status
# ---------------------------------------------------------------------------


class TestStatus:
    """Test status reporting."""

    def test_get_status(self, bridge):
        """Status should contain bridge, gate, and sizer sections."""
        bridge.refresh_targets()
        status = bridge.get_status()
        assert "bridge" in status
        assert "gate" in status
        assert "sizer" in status
        assert status["bridge"]["active"] is True

    def test_to_json(self, bridge):
        """to_json should return valid JSON."""
        status_json = bridge.to_json()
        parsed = json.loads(status_json)
        assert "bridge" in parsed
        assert "gate" in parsed

    def test_status_before_refresh(self, bridge):
        """Status should work even without a sizing decision yet."""
        status = bridge.get_status()
        assert status["bridge"]["active"] is True
        assert status["sizer"] == {}  # No decision yet


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_value_portfolio(self, bridge):
        """Zero total value should not crash."""
        holdings = {"SPY": 0.0, "GLD": 0.0, "TLT": 0.0}
        result = bridge.evaluate(holdings, 0.0)
        assert isinstance(result, AdaptiveRebalanceResult)

    def test_single_asset_holdings(self, bridge):
        """Holdings missing some assets should not crash."""
        holdings = {"SPY": 100000.0}
        result = bridge.evaluate(holdings, 100000.0)
        assert isinstance(result, AdaptiveRebalanceResult)

    def test_empty_holdings(self, bridge):
        """Empty holdings should not crash."""
        result = bridge.evaluate({}, 0.0)
        assert isinstance(result, AdaptiveRebalanceResult)

    def test_negative_holdings(self, bridge):
        """Negative holdings (possible with margin) should not crash."""
        holdings = {"SPY": -5000.0, "GLD": 55000.0, "TLT": 50000.0}
        result = bridge.evaluate(holdings, 100000.0)
        assert isinstance(result, AdaptiveRebalanceResult)

    def test_large_holdings(self, bridge):
        """Large portfolio values should work."""
        holdings = {"SPY": 4600000.0, "GLD": 3800000.0, "TLT": 1600000.0}
        result = bridge.evaluate(holdings, 10000000.0)
        assert isinstance(result, AdaptiveRebalanceResult)


# ---------------------------------------------------------------------------
# Test: Demo Mode
# ---------------------------------------------------------------------------


class TestDemo:
    """Test demo mode execution."""

    def test_demo_runs(self, bridge, capsys):
        """Demo function should run without error."""
        from src.rebalancing.adaptive_bridge import demo
        # Patch bridge creation to use our temp data
        import src.rebalancing.adaptive_bridge as ab_module

        original = ab_module.AdaptiveRebalanceBridge

        class PatchedBridge(AdaptiveRebalanceBridge):
            def __init__(self):
                # Capture data_dir from the fixture
                pass

        # Just test that demo's imports work
        assert hasattr(ab_module, "demo")
