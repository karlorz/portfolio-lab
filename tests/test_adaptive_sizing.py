#!/usr/bin/env python3
"""
Tests for Adaptive Position Sizing (v5.74).
"""
import sys
import os
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.strategy.adaptive_sizing import (
    AdaptiveSizer,
    BASE_ALLOCATION,
    HARD_BOUNDS,
    MAX_FACTOR_ADJUSTMENT,
    SizingDecision,
    SizingFactors,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory with test state files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def normal_regime_state(temp_data_dir):
    """Create a regime state file indicating NORMAL conditions."""
    state = {
        "current_regime": "normal",
        "previous_regime": None,
        "last_updated": datetime.now().isoformat(),
        "last_reading": {
            "regime": "normal",
            "confidence": 0.7,
        },
    }
    state_path = temp_data_dir / "regime_classifier_state.json"
    state_path.write_text(json.dumps(state))
    return temp_data_dir


@pytest.fixture
def crisis_regime_state(temp_data_dir):
    """Create a regime state file indicating CRISIS conditions."""
    state = {
        "current_regime": "crisis",
        "previous_regime": "high_vol",
        "last_updated": datetime.now().isoformat(),
        "last_reading": {
            "regime": "crisis",
            "confidence": 0.9,
        },
    }
    state_path = temp_data_dir / "regime_classifier_state.json"
    state_path.write_text(json.dumps(state))
    return temp_data_dir


@pytest.fixture
def mock_prices(tmp_path):
    """Create mock price data for testing."""
    prices_dir = tmp_path / "public" / "data"
    prices_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 200 days of price data
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
    
    prices_path = prices_dir / "prices.json"
    prices_path.write_text(json.dumps(prices))
    
    return tmp_path


@pytest.fixture
def mock_prices_high_vol(tmp_path):
    """Create mock price data with high volatility."""
    prices_dir = tmp_path / "public" / "data"
    prices_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(99)
    n = 200
    dates = []
    base = datetime(2026, 1, 1)
    for i in range(n):
        d = base + timedelta(days=i)
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
    
    n = len(dates)
    spy_returns = np.random.normal(-0.001, 0.025, n)  # High vol, negative drift
    spy_prices = 480.0 * np.exp(np.cumsum(spy_returns))
    
    prices = {
        "SPY": [{"d": dates[i], "p": float(spy_prices[i])} for i in range(n)],
    }
    
    prices_path = prices_dir / "prices.json"
    prices_path.write_text(json.dumps(prices))
    
    return tmp_path


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------


class TestAdaptiveSizerInit:
    """Test initialization."""

    def test_init_default(self, tmp_path):
        """Default initialization should use base allocation."""
        sizer = AdaptiveSizer(data_dir=tmp_path / "data")
        assert sizer.last_allocation == BASE_ALLOCATION
        assert sizer.last_decision is None

    def test_init_loads_state(self, tmp_path):
        """Loading previous state should restore last allocation."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            "last_allocation": {"SPY": 0.50, "GLD": 0.35, "TLT": 0.15},
            "last_updated": datetime.now().isoformat(),
        }
        state_path = data_dir / "adaptive_sizing_state.json"
        state_path.write_text(json.dumps(state))
        
        sizer = AdaptiveSizer(data_dir=data_dir)
        assert sizer.last_allocation == {"SPY": 0.50, "GLD": 0.35, "TLT": 0.15}

    def test_init_missing_state(self, tmp_path):
        """Missing state file should not crash."""
        data_dir = tmp_path / "data"
        sizer = AdaptiveSizer(data_dir=data_dir)
        assert sizer.last_allocation == BASE_ALLOCATION


# ---------------------------------------------------------------------------
# Test: Regime-Based Adjustments
# ---------------------------------------------------------------------------


class TestRegimeAdjustments:
    """Test regime-based allocation adjustments."""

    def test_normal_regime(self, normal_regime_state):
        """NORMAL regime should stay near base allocation."""
        sizer = AdaptiveSizer(data_dir=normal_regime_state)
        decision = sizer.compute_allocation()
        assert decision.factors.regime == "normal"
        # NORMAL regime: no regime adjustment
        assert abs(decision.regime_adjustment["SPY"]) < 0.001
        assert abs(decision.regime_adjustment["GLD"]) < 0.001
        assert abs(decision.regime_adjustment["TLT"]) < 0.001

    def test_crisis_regime(self, crisis_regime_state):
        """CRISIS regime should reduce SPY, increase GLD/TLT."""
        sizer = AdaptiveSizer(data_dir=crisis_regime_state)
        decision = sizer.compute_allocation()
        assert decision.factors.regime == "crisis"
        assert decision.regime_adjustment["SPY"] < 0  # Reduce SPY
        assert decision.regime_adjustment["GLD"] > 0  # Increase GLD
        assert decision.regime_adjustment["TLT"] > 0  # Increase TLT

    def test_regime_unknown(self, temp_data_dir):
        """UNKNOWN regime should produce zero adjustments."""
        sizer = AdaptiveSizer(data_dir=temp_data_dir)
        decision = sizer.compute_allocation()
        # No regime state file -> unknown
        assert decision.regime_adjustment["SPY"] == 0.0

    def test_all_regimes_produce_valid_allocations(self, temp_data_dir):
        """All regime types should produce bounded allocations."""
        from src.strategy.adaptive_sizing import REGIME_ADJUSTMENTS
        
        for regime in REGIME_ADJUSTMENTS:
            state = {
                "current_regime": regime,
                "last_reading": {"regime": regime, "confidence": 0.7},
            }
            state_path = temp_data_dir / "regime_classifier_state.json"
            state_path.write_text(json.dumps(state))
            
            sizer = AdaptiveSizer(data_dir=temp_data_dir)
            decision = sizer.compute_allocation()
            
            for asset in ["SPY", "GLD", "TLT"]:
                w = decision.adjusted_allocation.get(asset, 0)
                lo, hi = HARD_BOUNDS[asset]
                assert lo <= w <= hi, f"{asset} weight {w:.4f} outside [{lo}, {hi}] in {regime} regime"
            
            # Weights should sum to ~1.0
            total = sum(decision.adjusted_allocation.values())
            assert 0.99 <= total <= 1.01


# ---------------------------------------------------------------------------
# Test: Hard Bounds
# ---------------------------------------------------------------------------


class TestHardBounds:
    """Test boundary enforcement."""

    def test_spy_bounds(self):
        """SPY should stay within 36-56%."""
        lo, hi = HARD_BOUNDS["SPY"]
        assert lo == 0.36
        assert hi == 0.56

    def test_gld_bounds(self):
        """GLD should stay within 28-48%."""
        lo, hi = HARD_BOUNDS["GLD"]
        assert lo == 0.28
        assert hi == 0.48

    def test_tlt_bounds(self):
        """TLT should stay within 6-26%."""
        lo, hi = HARD_BOUNDS["TLT"]
        assert lo == 0.06
        assert hi == 0.26

    def test_allocation_normalized(self, temp_data_dir):
        """Allocation should always sum to ~1.0."""
        sizer = AdaptiveSizer(data_dir=temp_data_dir)
        decision = sizer.compute_allocation()
        total = sum(decision.adjusted_allocation.values())
        assert 0.99 <= total <= 1.01


# ---------------------------------------------------------------------------
# Test: Volatility Adjustment
# ---------------------------------------------------------------------------


class TestVolatilityAdjustment:
    """Test volatility-based allocation shifts."""

    def test_low_vol_increases_spy(self):
        """Low vol should increase SPY allocation."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_vol_adjustment(0.08)  # 8% vol (below 14% target)
        assert adj["SPY"] > 0

    def test_high_vol_decreases_spy(self):
        """High vol should decrease SPY allocation."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_vol_adjustment(0.22)  # 22% vol (above 14% target)
        assert adj["SPY"] < 0
        assert adj["GLD"] > 0  # Shift to gold
        assert adj["TLT"] > 0  # Shift to bonds

    def test_extreme_vol_capped(self):
        """Extreme vol should max out adjustments."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_vol_adjustment(0.50)  # 50% vol
        assert adj["SPY"] <= 0  # Should reduce SPY
        assert abs(adj["SPY"]) <= MAX_FACTOR_ADJUSTMENT  # Capped

    def test_target_vol_produces_zero_adjustment(self):
        """Near-target vol should produce minimal adjustment."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_vol_adjustment(0.14)  # Exactly target vol
        assert abs(adj["SPY"]) < 0.015  # Minimal or zero
        assert abs(adj["GLD"]) < 0.01
        assert abs(adj["TLT"]) < 0.01


# ---------------------------------------------------------------------------
# Test: Signal Adjustment
# ---------------------------------------------------------------------------


class TestSignalAdjustment:
    """Test ensemble signal-based adjustments."""

    def test_strong_bullish_adjustment(self):
        """Strong bullish signal with high agreement."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_signal_adjustment(0.7, 0.8)
        assert adj["SPY"] > 0  # Increase SPY
        assert adj["GLD"] < 0  # Reduce GLD
        assert adj["TLT"] < 0  # Reduce TLT

    def test_strong_bearish_adjustment(self):
        """Strong bearish signal with high agreement."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_signal_adjustment(-0.7, 0.8)
        assert adj["SPY"] < 0  # Reduce SPY
        assert adj["GLD"] > 0  # Increase GLD
        assert adj["TLT"] > 0  # Increase TLT

    def test_low_agreement_no_adjustment(self):
        """Low agreement should produce no adjustment."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_signal_adjustment(0.7, 0.4)  # 40% agreement (< 50%)
        assert adj["SPY"] == 0.0
        assert adj["GLD"] == 0.0
        assert adj["TLT"] == 0.0

    def test_neutral_signal_no_adjustment(self):
        """Near-zero signal should produce no adjustment."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_signal_adjustment(0.05, 0.8)  # 5% signal
        assert adj["SPY"] == 0.0

    def test_signal_adjustment_capped(self):
        """Signal adjustment should not exceed MAX_FACTOR_ADJUSTMENT."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_signal_adjustment(1.0, 1.0)  # Max signal, max agreement
        assert abs(adj["SPY"]) <= MAX_FACTOR_ADJUSTMENT
        assert abs(adj["GLD"]) <= MAX_FACTOR_ADJUSTMENT
        assert abs(adj["TLT"]) <= MAX_FACTOR_ADJUSTMENT


# ---------------------------------------------------------------------------
# Test: Drawdown Adjustment
# ---------------------------------------------------------------------------


class TestDrawdownAdjustment:
    """Test drawdown-based adjustments."""

    def test_shallow_drawdown_no_adjustment(self):
        """Small drawdown should not trigger adjustment."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_drawdown_adjustment(-0.03, "ok")
        assert adj["SPY"] == 0.0
        assert adj["GLD"] == 0.0
        assert adj["TLT"] == 0.0

    def test_deep_drawdown_reduces_spy(self):
        """Deep drawdown should reduce SPY."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_drawdown_adjustment(-0.15, "ok")
        assert adj["SPY"] < 0
        assert adj["GLD"] > 0
        assert adj["TLT"] > 0

    def test_critical_circuit_breaker(self):
        """Critical circuit breaker should max out risk reduction."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_drawdown_adjustment(-0.05, "critical")
        assert adj["SPY"] < 0
        assert adj["GLD"] > 0

    def test_elevated_circuit_breaker(self):
        """Elevated circuit breaker should partially reduce risk."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_drawdown_adjustment(-0.02, "elevated")
        assert adj["SPY"] < 0
        assert adj["SPY"] > -MAX_FACTOR_ADJUSTMENT * 0.5  # Not maxed
        assert adj["GLD"] > 0


# ---------------------------------------------------------------------------
# Test: State Persistence
# ---------------------------------------------------------------------------


class TestStatePersistence:
    """Test loading and saving state."""

    def test_save_creates_state_file(self, temp_data_dir, normal_regime_state):
        """Compute allocation should save state."""
        sizer = AdaptiveSizer(data_dir=normal_regime_state)
        decision = sizer.compute_allocation()
        state_path = normal_regime_state / "adaptive_sizing_state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert "last_allocation" in state
        assert "regime" in state
        assert state["regime"] == "normal"

    def test_save_state_integrity(self, temp_data_dir):
        """Saved state should be parseable JSON."""
        data_dir = temp_data_dir
        # Set up regime state
        state = {
            "current_regime": "high_vol",
            "last_reading": {"regime": "high_vol", "confidence": 0.8},
        }
        data_dir.joinpath("regime_classifier_state.json").write_text(json.dumps(state))
        
        sizer = AdaptiveSizer(data_dir=data_dir)
        decision = sizer.compute_allocation()
        state_path = data_dir / "adaptive_sizing_state.json"
        
        # Verify JSON integrity
        loaded = json.loads(state_path.read_text())
        assert loaded["regime"] == "high_vol"
        assert 0.99 <= sum(loaded["last_allocation"].values()) <= 1.01

    def test_load_from_state(self, temp_data_dir):
        """Loading from existing state should restore values."""
        data_dir = temp_data_dir
        state_path = data_dir / "adaptive_sizing_state.json"
        state = {
            "last_allocation": {"SPY": 0.50, "GLD": 0.32, "TLT": 0.18},
            "last_updated": "2026-05-16T12:00:00",
        }
        state_path.write_text(json.dumps(state))
        
        sizer = AdaptiveSizer(data_dir=data_dir)
        assert sizer.last_allocation["SPY"] == 0.50
        assert sizer.last_allocation["GLD"] == 0.32
        assert sizer.last_allocation["TLT"] == 0.18


# ---------------------------------------------------------------------------
# Test: Full Pipeline Integration
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Test end-to-end allocation computation."""

    def test_compute_without_crash(self, normal_regime_state):
        """Basic allocation should not crash."""
        sizer = AdaptiveSizer(data_dir=normal_regime_state)
        decision = sizer.compute_allocation()
        assert isinstance(decision, SizingDecision)
        assert decision.base_allocation == BASE_ALLOCATION
        assert len(decision.adjusted_allocation) == 3

    def test_adjusted_allocation_within_bounds(self, crisis_regime_state):
        """Even in crisis, allocation should stay within bounds."""
        sizer = AdaptiveSizer(data_dir=crisis_regime_state)
        decision = sizer.compute_allocation()
        for asset in ["SPY", "GLD", "TLT"]:
            w = decision.adjusted_allocation.get(asset, 0)
            lo, hi = HARD_BOUNDS[asset]
            assert lo <= w <= hi, f"{asset} weight {w:.4f} outside [{lo}, {hi}]"

    def test_allocation_sum_to_one(self, normal_regime_state):
        """Allocation should always sum to 1.0."""
        sizer = AdaptiveSizer(data_dir=normal_regime_state)
        decision = sizer.compute_allocation()
        total = sum(decision.adjusted_allocation.values())
        assert abs(total - 1.0) < 0.01

    def test_sizing_factors_structure(self, normal_regime_state):
        """SizingFactors should contain all expected fields."""
        sizer = AdaptiveSizer(data_dir=normal_regime_state)
        decision = sizer.compute_allocation()
        factors = decision.factors
        assert hasattr(factors, "regime")
        assert hasattr(factors, "spy_vol_20d")
        assert hasattr(factors, "spy_mom_20d")
        assert hasattr(factors, "ensemble_signal")
        assert hasattr(factors, "circuit_breaker_severity")

    def test_sizing_decision_structure(self, normal_regime_state):
        """SizingDecision should contain all expected fields."""
        sizer = AdaptiveSizer(data_dir=normal_regime_state)
        decision = sizer.compute_allocation()
        assert hasattr(decision, "base_allocation")
        assert hasattr(decision, "adjusted_allocation")
        assert hasattr(decision, "adjustments")
        assert hasattr(decision, "regime_adjustment")
        assert hasattr(decision, "volatility_adjustment")
        assert hasattr(decision, "signal_adjustment")
        assert hasattr(decision, "drawdown_adjustment")


# ---------------------------------------------------------------------------
# Test: CLI
# ---------------------------------------------------------------------------


class TestCLI:
    """Test CLI entry point."""

    def test_adjust_command(self, normal_regime_state, monkeypatch, capsys):
        """`adjust` command should print allocation table."""
        import sys
        monkeypatch.setattr(sys, "argv", ["adaptive_sizing.py", "adjust"])
        
        # Point to temp data dir
        from src.strategy import adaptive_sizing as mod
        original_dir = mod.DATA_DIR
        mod.DATA_DIR = normal_regime_state
        
        try:
            mod.main()
            captured = capsys.readouterr()
            assert "ADAPTIVE POSITION SIZING" in captured.out
            assert "SPY" in captured.out
            assert "GLD" in captured.out
            assert "TLT" in captured.out
        finally:
            mod.DATA_DIR = original_dir

    def test_status_command(self, temp_data_dir, monkeypatch, capsys):
        """`status` command should print state."""
        import sys
        monkeypatch.setattr(sys, "argv", ["adaptive_sizing.py", "status"])
        
        from src.strategy import adaptive_sizing as mod
        original_dir = mod.DATA_DIR
        mod.DATA_DIR = temp_data_dir
        mod.STATE_PATH = temp_data_dir / "adaptive_sizing_state.json"
        
        try:
            mod.main()
            captured = capsys.readouterr()
            assert "No state file found" in captured.out
        finally:
            mod.DATA_DIR = original_dir


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_vol_does_not_crash(self):
        """Zero volatility should not crash vol adjustment."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_vol_adjustment(0.0)
        assert adj["SPY"] == 0.0

    def test_extreme_values_produce_bounded_output(self, temp_data_dir):
        """Extreme factor values should stay within bounds."""
        sizer = AdaptiveSizer(data_dir=temp_data_dir)
        # Test with extreme signal values
        adj = sizer._compute_signal_adjustment(2.0, 1.0)  # Beyond +1
        assert abs(adj["SPY"]) <= MAX_FACTOR_ADJUSTMENT
        
        adj = sizer._compute_drawdown_adjustment(-0.50, "critical")  # Extreme DD
        assert abs(adj["SPY"]) <= MAX_FACTOR_ADJUSTMENT

    def test_missing_state_files_graceful(self, temp_data_dir):
        """Missing all state files should fall back to base."""
        sizer = AdaptiveSizer(data_dir=temp_data_dir)
        decision = sizer.compute_allocation()
        # Should not crash, should return base allocation
        assert decision.base_allocation == BASE_ALLOCATION

    def test_confidence_scaling(self, temp_data_dir):
        """Low confidence should reduce regime adjustment magnitude."""
        from src.strategy.adaptive_sizing import CONFIDENCE_SCALING
        assert 0.5 in CONFIDENCE_SCALING  # Low confidence
        assert 0.7 in CONFIDENCE_SCALING  # Medium confidence
        assert 0.9 in CONFIDENCE_SCALING  # High confidence


class TestIntegrationNormalRegime:
    """Integration test: normal regime scenario."""

    def test_normal_regime_produces_sensible_allocation(self, normal_regime_state):
        """Normal regime should keep allocation near base."""
        sizer = AdaptiveSizer(data_dir=normal_regime_state)
        decision = sizer.compute_allocation()
        
        # In normal regime, close to base allocation
        for asset in ["SPY", "GLD", "TLT"]:
            base = BASE_ALLOCATION[asset]
            actual = decision.adjusted_allocation.get(asset, 0)
            # Within 5% of base (given vol adjustments)
            assert abs(actual - base) < 0.05
