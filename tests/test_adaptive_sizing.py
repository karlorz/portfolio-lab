#!/usr/bin/env python3
"""
Tests for Adaptive Position Sizing (v5.74).
"""
import logging
import sys
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


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


def _isolate_live_regime_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin unit tests to the local-degradation path of _load_regime_state.

    Since the live-SSOT refactor (_load_regime_state order: regime_state.json
    -> signals.json -> classifier state -> VIX/market.db), the loader consults
    live signals.json / market.db before local classifier state. Unit tests of
    the local file handling must not depend on live regime state (it flips
    with live data), so both live sources are patched out here.
    """
    monkeypatch.setattr(
        "src.strategy.adaptive_sizing.AdaptiveSizer._load_regime_from_signals",
        lambda self: None,
    )
    monkeypatch.setattr("src.paths.MARKET_DB", Path("/nonexistent/market.db"))


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

    def test_load_ensemble_signal_uses_env_override(self, temp_data_dir, monkeypatch, tmp_path):
        """ENSEMBLE_WEIGHTS_FILE used when no signals.json vote is available."""
        # Isolate from live PUBLIC_DATA_DIR / WWW signals.json
        empty_public = tmp_path / "empty_public"
        empty_public.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PUBLIC_DATA_DIR", str(empty_public))
        monkeypatch.setenv("PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA", "1")

        weights_path = tmp_path / "custom_ensemble_weights.json"
        weights_path.write_text(json.dumps({
            "weighted_consensus": 0.42,
            "agreement_ratio": 0.73,
        }))
        monkeypatch.setenv("ENSEMBLE_WEIGHTS_FILE", str(weights_path))

        sizer = AdaptiveSizer(data_dir=temp_data_dir)
        signal, agreement = sizer._load_ensemble_signal()

        assert signal == 0.42
        assert agreement == 0.73


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

    def test_regime_unknown(self, temp_data_dir, monkeypatch):
        """UNKNOWN regime should produce zero adjustments."""
        _isolate_live_regime_sources(monkeypatch)
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
        _ = sizer.compute_allocation()
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
        _ = sizer.compute_allocation()
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

    def test_adjust_command(self, normal_regime_state, monkeypatch, caplog):
        """`adjust` command should print allocation table."""
        import sys
        monkeypatch.setattr(sys, "argv", ["adaptive_sizing.py", "adjust"])

        # Point to temp data dir
        from src.strategy import adaptive_sizing as mod
        original_dir = mod.DATA_DIR
        mod.DATA_DIR = normal_regime_state

        try:
            with caplog.at_level(logging.INFO, logger="src.strategy.adaptive_sizing"):
                mod.main()
            assert "ADAPTIVE POSITION SIZING" in caplog.text
            assert "SPY" in caplog.text
            assert "GLD" in caplog.text
            assert "TLT" in caplog.text
        finally:
            mod.DATA_DIR = original_dir

    def test_status_command(self, temp_data_dir, monkeypatch, caplog):
        """`status` command should log state."""
        monkeypatch.setattr(sys, "argv", ["adaptive_sizing.py", "status"])

        from src.strategy import adaptive_sizing as mod
        original_dir = mod.DATA_DIR
        mod.DATA_DIR = temp_data_dir
        mod.STATE_PATH = temp_data_dir / "adaptive_sizing_state.json"

        try:
            with caplog.at_level(logging.WARNING, logger="src.strategy.adaptive_sizing"):
                mod.main()
            assert "No state file found" in caplog.text
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


# ---------------------------------------------------------------------------
# Extended test coverage: to_dict, computation edge cases, constants, signals
# ---------------------------------------------------------------------------


class TestExtendedCoverage:
    """Extended tests: dataclass to_dict, computation edge cases, constants, signals."""

    # --- to_dict tests ---

    def test_sizing_factors_to_dict_all_fields(self, normal_regime_state):
        """SizingFactors.to_dict() should contain all dataclass fields."""
        sizer = AdaptiveSizer(data_dir=normal_regime_state)
        decision = sizer.compute_allocation()
        d = decision.factors.to_dict()
        expected = {
            "timestamp", "regime", "regime_confidence", "spy_vol_20d",
            "spy_mom_20d", "spy_drawdown_60d", "ensemble_signal",
            "ensemble_agreement", "circuit_breaker_severity",
        }
        assert set(d.keys()) == expected, f"Keys mismatch: diff={expected ^ set(d.keys())}"
        assert isinstance(d["timestamp"], str)
        assert isinstance(d["regime"], str)
        assert isinstance(d["regime_confidence"], float)

    def test_sizing_decision_to_dict_all_fields(self, normal_regime_state):
        """SizingDecision.to_dict() should contain all dataclass fields."""
        sizer = AdaptiveSizer(data_dir=normal_regime_state)
        decision = sizer.compute_allocation()
        d = decision.to_dict()
        expected = {
            "timestamp", "base_allocation", "adjusted_allocation",
            "adjustments", "regime_adjustment", "volatility_adjustment",
            "signal_adjustment", "drawdown_adjustment", "factors",
        }
        assert set(d.keys()) == expected, f"Keys mismatch: diff={expected ^ set(d.keys())}"
        # Nested factors should also be a dict (via asdict recursion)
        assert isinstance(d["factors"], dict)
        assert "regime" in d["factors"]

    def test_sizing_decision_to_dict_preserves_allocations(self, normal_regime_state):
        """to_dict should preserve base_allocation and adjusted_allocation."""
        sizer = AdaptiveSizer(data_dir=normal_regime_state)
        decision = sizer.compute_allocation()
        d = decision.to_dict()
        assert d["base_allocation"] == BASE_ALLOCATION
        assert isinstance(d["adjusted_allocation"], dict)
        # Should have all three core assets
        for asset in ("SPY", "GLD", "TLT"):
            assert asset in d["adjusted_allocation"]

    # --- Volatility computation edge cases ---

    def test_vol_adjustment_at_high_boundary_ratio(self):
        """Vol ratio exactly at 0.8 (high-vol threshold) should give zero adjustment."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_vol_adjustment(0.175)  # 0.14/0.175 = 0.8
        assert adj["SPY"] == 0.0
        assert adj["GLD"] == 0.0
        assert adj["TLT"] == 0.0

    def test_vol_adjustment_at_low_boundary_ratio(self):
        """Vol ratio exactly at 1.2 (low-vol threshold) should give zero adjustment."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_vol_adjustment(0.14 / 1.2)  # vol_ratio = 1.2
        assert adj["SPY"] == 0.0
        assert adj["GLD"] == 0.0
        assert adj["TLT"] == 0.0

    def test_vol_adjustment_very_low_vol_clamped(self):
        """Extremely low vol should be clamped to vol_ratio=1.5, producing max positive SPY adj."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        # Must be just above the <= 0.001 guard threshold
        adj = sizer._compute_vol_adjustment(0.0011)  # vol_ratio ~= 127, clamped to 1.5
        assert adj["SPY"] > 0
        assert adj["GLD"] < 0
        assert adj["TLT"] < 0
        # Check exact values at max factor
        assert adj["SPY"] == pytest.approx(MAX_FACTOR_ADJUSTMENT * 0.7)
        assert adj["GLD"] == pytest.approx(-MAX_FACTOR_ADJUSTMENT * 0.3)
        assert adj["TLT"] == pytest.approx(-MAX_FACTOR_ADJUSTMENT * 0.3)

    def test_vol_adjustment_mid_high_vol_scaling(self):
        """Vol ratio at midpoint between 0.5 and 0.8 should produce proportional adjustment."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        # vol_ratio = 0.65, factor = (0.8-0.65)/0.3 = 0.5
        vol = 0.14 / 0.65
        adj = sizer._compute_vol_adjustment(vol)
        expected_spy = -MAX_FACTOR_ADJUSTMENT * 0.5 * 0.7
        expected_gld = +MAX_FACTOR_ADJUSTMENT * 0.5 * 0.5
        expected_tlt = +MAX_FACTOR_ADJUSTMENT * 0.5 * 0.3
        assert adj["SPY"] == pytest.approx(expected_spy)
        assert adj["GLD"] == pytest.approx(expected_gld)
        assert adj["TLT"] == pytest.approx(expected_tlt)

    # --- Signal adjustment edge cases ---

    def test_signal_at_one_tenth_boundary(self):
        """Ensemble signal exactly at 0.1 should proceed (not blocked by < 0.1 check)."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_signal_adjustment(0.1, 0.8)
        # Should have a tiny positive adjustment
        assert adj["SPY"] > 0

    def test_signal_just_below_one_tenth(self):
        """Ensemble signal just below 0.1 should produce zero adjustment."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_signal_adjustment(0.099, 0.8)
        assert adj["SPY"] == 0.0

    def test_signal_agreement_at_exactly_half(self):
        """Agreement exactly at 0.5 should produce zero adjustment."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_signal_adjustment(0.7, 0.5)
        assert adj["SPY"] == 0.0

    def test_signal_agreement_just_above_half(self):
        """Agreement just above 0.5 should produce a very small adjustment."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_signal_adjustment(0.7, 0.51)
        # factor = 0.7 * (0.51 - 0.5) * 2 = 0.014
        assert adj["SPY"] > 0
        assert adj["SPY"] < MAX_FACTOR_ADJUSTMENT * 0.01  # Very small

    def test_signal_max_bullish(self):
        """Signal=1.0 and agreement=1.0 should produce maximum bullish adjustment."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_signal_adjustment(1.0, 1.0)
        # factor = 1.0 * (1.0-0.5) * 2 = 1.0
        assert adj["SPY"] == pytest.approx(+MAX_FACTOR_ADJUSTMENT * 0.5)
        assert adj["GLD"] == pytest.approx(-MAX_FACTOR_ADJUSTMENT * 0.3)
        assert adj["TLT"] == pytest.approx(-MAX_FACTOR_ADJUSTMENT * 0.3)

    def test_signal_max_bearish(self):
        """Signal=-1.0 and agreement=1.0 should produce maximum bearish adjustment."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_signal_adjustment(-1.0, 1.0)
        # factor = 1.0 * (1.0-0.5) * 2 = 1.0
        assert adj["SPY"] == pytest.approx(-MAX_FACTOR_ADJUSTMENT * 0.5)
        assert adj["GLD"] == pytest.approx(+MAX_FACTOR_ADJUSTMENT * 0.3)
        assert adj["TLT"] == pytest.approx(+MAX_FACTOR_ADJUSTMENT * 0.3)

    # --- Drawdown adjustment edge cases ---

    def test_drawdown_exactly_at_ten_percent(self):
        """Drawdown exactly at -0.10 should produce zero factor (threshold boundary)."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_drawdown_adjustment(-0.10, "ok")
        # factor = min(1.0, (0.10 - 0.10) / 0.10) = 0.0
        assert adj["SPY"] == 0.0
        assert adj["GLD"] == 0.0
        assert adj["TLT"] == 0.0

    def test_drawdown_at_twenty_percent_max_factor(self):
        """Drawdown at -0.20 should produce maximum factor of 1.0."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_drawdown_adjustment(-0.20, "ok")
        # factor = min(1.0, (0.20 - 0.10) / 0.10) = 1.0
        assert adj["SPY"] == pytest.approx(-MAX_FACTOR_ADJUSTMENT * 0.5)
        assert adj["GLD"] == pytest.approx(+MAX_FACTOR_ADJUSTMENT * 0.3)
        assert adj["TLT"] == pytest.approx(+MAX_FACTOR_ADJUSTMENT * 0.3)

    def test_drawdown_severe_circuit_breaker_bypasses_dd(self):
        """Severe circuit breaker should apply fixed reduction regardless of drawdown depth."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_drawdown_adjustment(-0.03, "severe")
        assert adj["SPY"] == -MAX_FACTOR_ADJUSTMENT * 0.8
        assert adj["GLD"] == +MAX_FACTOR_ADJUSTMENT * 0.5
        assert adj["TLT"] == +MAX_FACTOR_ADJUSTMENT * 0.4

    def test_drawdown_shallow_with_elevated_cb(self):
        """Elevated circuit breaker with shallow drawdown still applies partial reduction."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_drawdown_adjustment(-0.01, "elevated")
        assert adj["SPY"] == -MAX_FACTOR_ADJUSTMENT * 0.4
        assert adj["GLD"] == +MAX_FACTOR_ADJUSTMENT * 0.3
        assert adj["TLT"] == +MAX_FACTOR_ADJUSTMENT * 0.2

    def test_drawdown_unknown_severity_ignored(self):
        """Unknown circuit breaker severity (not ok/elevated/severe/critical) should be treated as ok."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_drawdown_adjustment(-0.03, "unknown_severity")
        # Not in critical/severe/elevated, and -0.03 > -0.10, so no adjustment
        assert adj["SPY"] == 0.0
        assert adj["GLD"] == 0.0
        assert adj["TLT"] == 0.0

    # --- _apply_bounds edge cases ---

    def test_apply_bounds_already_within(self):
        """_apply_bounds on already-bounded weights should return normalized weights near original."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        bounded = sizer._apply_bounds({"SPY": 0.46, "GLD": 0.38, "TLT": 0.16})
        # Each value within bounds, sum=1.0, so unchanged by clamp and normalization
        assert bounded["SPY"] == pytest.approx(0.46, abs=0.005)
        assert bounded["GLD"] == pytest.approx(0.38, abs=0.005)
        assert bounded["TLT"] == pytest.approx(0.16, abs=0.005)

    def test_apply_bounds_clamps_excessive_spy(self):
        """_apply_bounds should clamp SPY above its 56% upper bound."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        bounded = sizer._apply_bounds({"SPY": 0.70, "GLD": 0.20, "TLT": 0.10})
        # After clamp: SPY=0.56, GLD=0.28, TLT=0.10, sum=0.94
        # After normalize: SPY=0.5957, GLD=0.2979, TLT=0.1064
        assert bounded["SPY"] <= 0.56 / 0.94 + 0.01  # Slight tolerance after normalization
        assert bounded["GLD"] >= 0.28 / 0.94 - 0.01
        assert abs(sum(bounded.values()) - 1.0) < 0.01

    def test_apply_bounds_zero_total_no_normalize(self):
        """_apply_bounds with zero total after clamping should not divide by zero."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        # 'OTHER' has default bounds (0.0, 1.0), so 0.0 stays at 0.0 -> total=0
        bounded = sizer._apply_bounds({"OTHER": 0.0})
        assert bounded["OTHER"] == 0.0

    def test_apply_bounds_single_asset_clamped(self):
        """_apply_bounds with single asset should clamp to bounds."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        bounded = sizer._apply_bounds({"SPY": 0.90})
        # Clamp to 0.56, then normalize (single asset -> 1.0)
        assert bounded["SPY"] == pytest.approx(1.0)

    # --- Constants validation ---

    def test_regime_adjustments_covers_all_regimes(self):
        """REGIME_ADJUSTMENTS should include low_vol, normal, high_vol, crisis, recovery, unknown."""
        from src.strategy.adaptive_sizing import REGIME_ADJUSTMENTS
        expected = {"low_vol", "normal", "high_vol", "crisis", "recovery", "unknown"}
        assert set(REGIME_ADJUSTMENTS.keys()) == expected

    def test_regime_adjustments_all_have_three_assets(self):
        """Every regime adjustment dict should have SPY, GLD, TLT keys."""
        from src.strategy.adaptive_sizing import REGIME_ADJUSTMENTS
        for regime, adj in REGIME_ADJUSTMENTS.items():
            assert "SPY" in adj, f"{regime} missing SPY"
            assert "GLD" in adj, f"{regime} missing GLD"
            assert "TLT" in adj, f"{regime} missing TLT"

    def test_regime_adjustments_symmetric(self):
        """Collective sum of adjustments per regime should not exceed +-0.15."""
        from src.strategy.adaptive_sizing import REGIME_ADJUSTMENTS
        for regime, adj in REGIME_ADJUSTMENTS.items():
            total_adj = sum(adj.values())
            assert abs(total_adj) <= 0.15, f"{regime} total adjustment {total_adj:.3f} too large"

    def test_confidence_scaling_is_monotonic(self):
        """CONFIDENCE_SCALING values should increase with confidence."""
        from src.strategy.adaptive_sizing import CONFIDENCE_SCALING
        confidences = sorted(CONFIDENCE_SCALING.keys())
        for i in range(len(confidences) - 1):
            assert CONFIDENCE_SCALING[confidences[i]] <= CONFIDENCE_SCALING[confidences[i + 1]], (
                f"Non-monotonic at confidence {confidences[i]}"
            )

    def test_confidence_scaling_reaches_full(self):
        """Highest confidence level should map to 1.0 scaling."""
        from src.strategy.adaptive_sizing import CONFIDENCE_SCALING
        max_conf = max(CONFIDENCE_SCALING.keys())
        assert CONFIDENCE_SCALING[max_conf] == 1.0

    def test_base_allocation_sums_to_one(self):
        """BASE_ALLOCATION should sum to 1.0."""
        total = sum(BASE_ALLOCATION.values())
        assert abs(total - 1.0) < 0.01

    def test_hard_bounds_all_core_assets_present(self):
        """HARD_BOUNDS should include SPY, GLD, TLT with valid ranges."""
        for asset in ["SPY", "GLD", "TLT"]:
            assert asset in HARD_BOUNDS, f"{asset} missing from HARD_BOUNDS"
            lo, hi = HARD_BOUNDS[asset]
            assert 0.0 <= lo < hi <= 1.0, f"{asset} bounds [{lo}, {hi}] invalid"

    def test_hard_bounds_bond_overlap(self):
        """Bond bounds should not overlap with equity/gold ranges."""
        # TLT 6-26%, IEF 0-10%, SHY 0-10%
        assert HARD_BOUNDS["TLT"] == (0.06, 0.26)
        assert HARD_BOUNDS["IEF"] == (0.00, 0.10)
        assert HARD_BOUNDS["SHY"] == (0.00, 0.10)


class TestExports:
    """Verify __all__ exports and import completeness."""

    def test_all_exports_importable(self):
        from src.strategy.adaptive_sizing import __all__
        import src.strategy.adaptive_sizing as mod
        for name in __all__:
            assert hasattr(mod, name), f"{name} in __all__ but not in module"

    def test_all_contains_key_names(self):
        from src.strategy.adaptive_sizing import __all__
        expected = {'HARD_BOUNDS', 'MAX_FACTOR_ADJUSTMENT', 'REGIME_ADJUSTMENTS',
                    'CONFIDENCE_SCALING', 'SizingFactors', 'SizingDecision', 'AdaptiveSizer'}
        assert expected.issubset(set(__all__))


class TestSizingFactorsDataclass:
    """Comprehensive SizingFactors dataclass tests."""

    def test_all_fields_present(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(SizingFactors)}
        expected = {"timestamp", "regime", "regime_confidence", "spy_vol_20d",
                    "spy_mom_20d", "spy_drawdown_60d", "ensemble_signal",
                    "ensemble_agreement", "circuit_breaker_severity"}
        assert field_names == expected

    def test_create_with_all_fields(self):
        sf = SizingFactors(
            timestamp="2026-05-24T10:00:00", regime="normal", regime_confidence=0.7,
            spy_vol_20d=0.14, spy_mom_20d=0.02, spy_drawdown_60d=-0.03,
            ensemble_signal=0.5, ensemble_agreement=0.8, circuit_breaker_severity="ok",
        )
        assert sf.regime == "normal"
        assert sf.regime_confidence == 0.7

    def test_to_dict_keys_match_fields(self):
        from dataclasses import fields
        sf = SizingFactors(
            timestamp="2026-05-24T10:00:00", regime="normal", regime_confidence=0.7,
            spy_vol_20d=0.14, spy_mom_20d=0.02, spy_drawdown_60d=-0.03,
            ensemble_signal=0.5, ensemble_agreement=0.8, circuit_breaker_severity="ok",
        )
        d = sf.to_dict()
        expected = {f.name for f in fields(SizingFactors)}
        assert set(d.keys()) == expected


class TestSizingDecisionDataclass:
    """Comprehensive SizingDecision dataclass tests."""

    def test_all_fields_present(self):
        from dataclasses import fields
        field_names = {f.name for f in fields(SizingDecision)}
        expected = {"timestamp", "base_allocation", "adjusted_allocation", "adjustments",
                    "regime_adjustment", "volatility_adjustment", "signal_adjustment",
                    "drawdown_adjustment", "factors"}
        assert field_names == expected

    def test_to_dict_contains_all_fields(self, normal_regime_state):
        sizer = AdaptiveSizer(data_dir=normal_regime_state)
        decision = sizer.compute_allocation()
        d = decision.to_dict()
        assert "factors" in d
        assert isinstance(d["factors"], dict)


class TestRegimeAdjustmentsExtended:
    """Extended parametrized regime adjustment tests."""

    @pytest.mark.parametrize("regime,spy_sign,gld_sign,tlt_sign", [
        ("low_vol", 1, -1, -1),
        ("normal", 0, 0, 0),
        ("high_vol", -1, 1, 1),
        ("crisis", -1, 1, 1),
        ("recovery", 1, -1, -1),
    ])
    def test_regime_adjustment_directions(self, regime, spy_sign, gld_sign, tlt_sign, temp_data_dir):
        from src.strategy.adaptive_sizing import REGIME_ADJUSTMENTS
        adj = REGIME_ADJUSTMENTS[regime]
        if spy_sign != 0:
            assert (adj["SPY"] > 0) == (spy_sign > 0), f"{regime} SPY direction wrong"
        else:
            assert adj["SPY"] == 0.0, f"{regime} SPY should be zero"
        if gld_sign != 0:
            assert (adj["GLD"] > 0) == (gld_sign > 0), f"{regime} GLD direction wrong"
        else:
            assert adj["GLD"] == 0.0, f"{regime} GLD should be zero"
        if tlt_sign != 0:
            assert (adj["TLT"] > 0) == (tlt_sign > 0), f"{regime} TLT direction wrong"
        else:
            assert adj["TLT"] == 0.0, f"{regime} TLT should be zero"

    def test_crisis_has_largest_spy_reduction(self):
        from src.strategy.adaptive_sizing import REGIME_ADJUSTMENTS
        crisis_spy = REGIME_ADJUSTMENTS["crisis"]["SPY"]
        for regime, adj in REGIME_ADJUSTMENTS.items():
            if regime != "crisis":
                assert adj["SPY"] >= crisis_spy, f"{regime} SPY adj more negative than crisis"


class TestComputeVolAdjustmentExtended:
    """Extended vol adjustment edge cases."""

    def test_vol_adjustment_negative_vol_returns_zero(self):
        """Negative vol is unphysical; function should handle gracefully."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_vol_adjustment(-0.05)
        # Negative vol <= 0.001 triggers early return
        assert adj["SPY"] == 0.0

    def test_vol_adjustment_exact_target(self):
        """Vol exactly at target (14%) should produce zero adjustment."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_vol_adjustment(0.14)
        # vol_ratio = 0.14/0.14 = 1.0 -> between 0.8 and 1.2 -> zero
        assert adj["SPY"] == 0.0
        assert adj["GLD"] == 0.0
        assert adj["TLT"] == 0.0

    @pytest.mark.parametrize("vol,expect_spy_positive", [
        (0.08, True),   # Low vol -> increase SPY
        (0.10, True),   # Low vol -> increase SPY
        (0.14, None),   # At target -> zero
        (0.20, False),  # High vol -> decrease SPY
        (0.30, False),  # Very high vol -> decrease SPY
    ])
    def test_vol_adjustment_direction(self, vol, expect_spy_positive):
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_vol_adjustment(vol)
        if expect_spy_positive is True:
            assert adj["SPY"] > 0
        elif expect_spy_positive is False:
            assert adj["SPY"] < 0
        else:
            assert adj["SPY"] == 0.0


class TestComputeSignalAdjustmentExtended:
    """Extended signal adjustment tests."""

    @pytest.mark.parametrize("signal,agreement,expect_nonzero", [
        (0.5, 0.6, True),    # Above both thresholds
        (0.1, 0.6, True),    # At signal boundary
        (0.09, 0.6, False),  # Below signal threshold
        (0.5, 0.5, False),   # At agreement boundary (<=0.5)
        (-0.5, 0.6, True),   # Negative signal
    ])
    def test_signal_thresholds(self, signal, agreement, expect_nonzero):
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_signal_adjustment(signal, agreement)
        if expect_nonzero:
            assert adj["SPY"] != 0.0
        else:
            assert adj["SPY"] == 0.0


class TestComputeDrawdownExtended:
    """Extended drawdown adjustment tests."""

    @pytest.mark.parametrize("dd,severity,expect_spy_negative", [
        (-0.03, "ok", False),        # Shallow DD, no CB -> zero
        (-0.15, "ok", True),         # Deep DD -> reduce SPY
        (-0.05, "critical", True),   # Critical CB -> reduce SPY
        (-0.05, "severe", True),     # Severe CB -> reduce SPY
        (-0.05, "elevated", True),   # Elevated CB -> reduce SPY
        (-0.05, "ok", False),        # Moderate DD, no CB -> zero
    ])
    def test_drawdown_directions(self, dd, severity, expect_spy_negative):
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        adj = sizer._compute_drawdown_adjustment(dd, severity)
        if expect_spy_negative:
            assert adj["SPY"] < 0
        else:
            assert adj["SPY"] == 0.0


class TestApplyBoundsExtended:
    """Extended _apply_bounds tests."""

    def test_unknown_asset_default_bounds(self):
        """Unknown asset should get default bounds (0.0, 1.0)."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        bounded = sizer._apply_bounds({"UNKNOWN_ASSET": 0.5})
        assert bounded["UNKNOWN_ASSET"] == pytest.approx(1.0)  # Single asset normalizes to 1.0

    def test_below_lower_bound_clamped(self):
        """Weight below lower bound should be clamped up."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        bounded = sizer._apply_bounds({"SPY": 0.20, "GLD": 0.50, "TLT": 0.30})
        # SPY 0.20 < 0.36, clamped to 0.36
        assert bounded["SPY"] >= 0.36 / 1.16 - 0.01  # After normalization

    def test_all_assets_at_bounds(self):
        """All assets exactly at their bounds should remain."""
        sizer = AdaptiveSizer(data_dir=Path("/tmp/nonexistent"))
        bounded = sizer._apply_bounds({"SPY": 0.56, "GLD": 0.38, "TLT": 0.06})
        total = sum(bounded.values())
        assert abs(total - 1.0) < 0.01


class TestStatePersistenceExtended:
    """Extended state persistence tests."""

    def test_corrupted_state_file_falls_back(self, tmp_path):
        """Corrupted JSON state file should fall back gracefully."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "adaptive_sizing_state.json").write_text("NOT JSON!!!")
        sizer = AdaptiveSizer(data_dir=data_dir)
        assert sizer.last_allocation == BASE_ALLOCATION

    def test_state_file_missing_last_allocation(self, tmp_path):
        """State file without last_allocation should use BASE_ALLOCATION."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        state = {"last_updated": "2026-05-24"}
        (data_dir / "adaptive_sizing_state.json").write_text(json.dumps(state))
        sizer = AdaptiveSizer(data_dir=data_dir)
        assert sizer.last_allocation == BASE_ALLOCATION


class TestCLIExtended:
    """Extended CLI tests."""

    def test_simulate_command_insufficient_data(self, temp_data_dir, monkeypatch, caplog):
        """simulate command with no prices should report insufficient data."""
        import sys as _sys
        monkeypatch.setattr(_sys, "argv", ["adaptive_sizing.py", "simulate"])
        from src.strategy import adaptive_sizing as mod
        original_dir = mod.DATA_DIR
        mod.DATA_DIR = temp_data_dir
        mod.STATE_PATH = temp_data_dir / "adaptive_sizing_state.json"
        try:
            with caplog.at_level(logging.DEBUG, logger="src.strategy.adaptive_sizing"):
                mod.main()
            combined = caplog.text.lower()
            assert "insufficient data" in combined or "simulation" in combined or "error" in combined
        finally:
            mod.DATA_DIR = original_dir

    def test_unknown_command_prints_usage(self, temp_data_dir, monkeypatch, caplog):
        """Unknown command should log usage."""
        import sys as _sys
        monkeypatch.setattr(_sys, "argv", ["adaptive_sizing.py", "foobar"])
        from src.strategy import adaptive_sizing as mod
        original_dir = mod.DATA_DIR
        mod.DATA_DIR = temp_data_dir
        mod.STATE_PATH = temp_data_dir / "adaptive_sizing_state.json"
        try:
            with caplog.at_level(logging.WARNING, logger="src.strategy.adaptive_sizing"):
                mod.main()
            assert "Usage" in caplog.text
        finally:
            mod.DATA_DIR = original_dir

    def test_no_args_defaults_to_adjust(self, temp_data_dir, monkeypatch, caplog):
        """No args should default to adjust command."""
        import sys as _sys
        monkeypatch.setattr(_sys, "argv", ["adaptive_sizing.py"])
        from src.strategy import adaptive_sizing as mod
        original_dir = mod.DATA_DIR
        mod.DATA_DIR = temp_data_dir
        mod.STATE_PATH = temp_data_dir / "adaptive_sizing_state.json"
        try:
            with caplog.at_level(logging.INFO, logger="src.strategy.adaptive_sizing"):
                mod.main()
            assert "ADAPTIVE POSITION SIZING" in caplog.text
        finally:
            mod.DATA_DIR = original_dir


class TestConstantsValidation:
    """Validate all module constants."""

    def test_max_factor_adjustment_positive(self):
        assert MAX_FACTOR_ADJUSTMENT > 0
        assert MAX_FACTOR_ADJUSTMENT <= 0.10

    def test_hard_bounds_lo_less_than_hi(self):
        for asset, (lo, hi) in HARD_BOUNDS.items():
            assert lo < hi, f"{asset}: lo={lo} >= hi={hi}"

    def test_hard_bounds_within_zero_one(self):
        for asset, (lo, hi) in HARD_BOUNDS.items():
            assert 0.0 <= lo <= 1.0, f"{asset}: lo={lo} out of range"
            assert 0.0 <= hi <= 1.0, f"{asset}: hi={hi} out of range"

    def test_regime_adjustments_all_numeric(self):
        from src.strategy.adaptive_sizing import REGIME_ADJUSTMENTS
        for regime, adj in REGIME_ADJUSTMENTS.items():
            for asset, val in adj.items():
                assert isinstance(val, (int, float)), f"{regime}.{asset} is not numeric"

    def test_base_allocation_all_positive(self):
        for asset, weight in BASE_ALLOCATION.items():
            assert weight > 0, f"{asset} weight={weight} not positive"

    def test_confidence_scaling_all_positive(self):
        from src.strategy.adaptive_sizing import CONFIDENCE_SCALING
        for conf, scale in CONFIDENCE_SCALING.items():
            assert 0 < scale <= 1.0, f"confidence={conf} scale={scale} invalid"


class TestLoadRegimeStateExtended:
    """Extended _load_regime_state tests."""

    def test_invalid_regime_falls_to_unknown(self, tmp_path):
        """Regime not in REGIME_ADJUSTMENTS should fall to unknown."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        state = {"current_regime": "nuclear_war", "last_reading": {"confidence": 0.9}}
        (data_dir / "regime_classifier_state.json").write_text(json.dumps(state))
        sizer = AdaptiveSizer(data_dir=data_dir)
        regime, conf = sizer._load_regime_state()
        assert regime == "unknown"
        assert conf == 0.3

    def test_corrupted_regime_file_falls_to_unknown(self, tmp_path, monkeypatch):
        """Corrupted JSON regime file should fall to unknown or fallback."""
        _isolate_live_regime_sources(monkeypatch)
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "regime_classifier_state.json").write_text("BROKEN JSON")
        sizer = AdaptiveSizer(data_dir=data_dir)
        regime, conf = sizer._load_regime_state()
        # May fall to VIX-based fallback (normal) or unknown
        assert regime in ("unknown", "normal")

    def test_no_regime_file_falls_to_fallback(self, tmp_path, monkeypatch):
        """No regime file should return unknown or VIX-based fallback."""
        _isolate_live_regime_sources(monkeypatch)
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        sizer = AdaptiveSizer(data_dir=data_dir)
        regime, conf = sizer._load_regime_state()
        # VIX-based fallback may return "normal"; only "unknown" if both paths fail
        assert regime in ("unknown", "normal")


class TestLoadCircuitBreaker:
    """Circuit breaker loading tests."""

    def test_missing_cb_file_returns_ok(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        sizer = AdaptiveSizer(data_dir=data_dir)
        assert sizer._load_circuit_breaker() == "ok"

    def test_corrupted_cb_file_returns_ok(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / ".circuit_breaker_state.json").write_text("NOT JSON")
        sizer = AdaptiveSizer(data_dir=data_dir)
        assert sizer._load_circuit_breaker() == "ok"

    def test_valid_cb_file(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        state = {"severity": "elevated"}
        (data_dir / ".circuit_breaker_state.json").write_text(json.dumps(state))
        sizer = AdaptiveSizer(data_dir=data_dir)
        assert sizer._load_circuit_breaker() == "elevated"

    def test_status_green_maps_to_ok_severity(self, tmp_path):
        """Paper risk CB files use status=green, not severity key."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / ".circuit_breaker_state.json").write_text(json.dumps({
            "status": "green",
            "last_check": "2026-05-22T22:37:01",
            "max_drawdown": 0.001,
        }))
        sizer = AdaptiveSizer(data_dir=data_dir)
        assert sizer._load_circuit_breaker() in ("ok", "green")

    def test_status_red_maps_to_elevated_or_red(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / ".circuit_breaker_state.json").write_text(json.dumps({
            "status": "red",
            "trips": 2,
        }))
        sizer = AdaptiveSizer(data_dir=data_dir)
        sev = sizer._load_circuit_breaker()
        assert sev in ("red", "elevated", "critical", "halt")


class TestLoadEnsembleSignal:
    """Ensemble signal loading tests."""

    def test_missing_file_returns_defaults(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        empty_public = tmp_path / "empty_public"
        empty_public.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PUBLIC_DATA_DIR", str(empty_public))
        monkeypatch.setenv("PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA", "1")
        sizer = AdaptiveSizer(data_dir=data_dir)
        signal, agreement = sizer._load_ensemble_signal()
        assert signal == 0.0
        assert agreement == 0.5

    def test_corrupted_file_returns_defaults(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        empty_public = tmp_path / "empty_public"
        empty_public.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PUBLIC_DATA_DIR", str(empty_public))
        monkeypatch.setenv("PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA", "1")
        (data_dir / "ensemble_weights.json").write_text("BROKEN")
        sizer = AdaptiveSizer(data_dir=data_dir)
        signal, agreement = sizer._load_ensemble_signal()
        assert signal == 0.0
        assert agreement == 0.5

    def test_valid_file_with_composite_signal(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        empty_public = tmp_path / "empty_public"
        empty_public.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PUBLIC_DATA_DIR", str(empty_public))
        monkeypatch.setenv("PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA", "1")
        state = {"composite_signal": 0.3, "agreement_ratio": 0.7}
        (data_dir / "ensemble_weights.json").write_text(json.dumps(state))
        sizer = AdaptiveSizer(data_dir=data_dir)
        signal, agreement = sizer._load_ensemble_signal()
        assert signal == 0.3
        assert agreement == 0.7

    def test_valid_file_with_weighted_consensus(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        empty_public = tmp_path / "empty_public"
        empty_public.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PUBLIC_DATA_DIR", str(empty_public))
        monkeypatch.setenv("PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA", "1")
        state = {"weighted_consensus": -0.4, "agreement_ratio": 0.6}
        (data_dir / "ensemble_weights.json").write_text(json.dumps(state))
        sizer = AdaptiveSizer(data_dir=data_dir)
        signal, agreement = sizer._load_ensemble_signal()
        assert signal == -0.4
        assert agreement == 0.6

    def test_signals_json_ensemble_voting_is_primary_ssot(self, tmp_path):
        """Live ensemble_voting must win over regime weight tables."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        # Regime weight table (real shape) — must NOT be read as vote consensus
        (data_dir / "ensemble_weights.json").write_text(json.dumps({
            "normal": {"alternative_data": 0.21, "momentum": 0.3},
            "crisis": {"alternative_data": 0.1},
        }))
        (data_dir / "signals.json").write_text(json.dumps({
            "ensemble_voting": {
                "weighted_consensus": 0.0483,
                "agreement_ratio": 0.9364,
                "regime": "normal",
                "regime_confidence": 0.755,
            }
        }))
        sizer = AdaptiveSizer(data_dir=data_dir)
        signal, agreement = sizer._load_ensemble_signal()
        assert abs(signal - 0.0483) < 1e-9
        assert abs(agreement - 0.9364) < 1e-9

    def test_signals_json_prefers_public_path_over_stale_weights(self, tmp_path, monkeypatch):
        """PUBLIC SIGNALS_JSON path should be consulted when data_dir has only weights."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        public = tmp_path / "public" / "data"
        public.mkdir(parents=True, exist_ok=True)
        (data_dir / "ensemble_weights.json").write_text(json.dumps({
            "normal": {"momentum": 0.5},
        }))
        (public / "signals.json").write_text(json.dumps({
            "ensemble_voting": {
                "weighted_consensus": -0.12,
                "agreement_ratio": 0.81,
            }
        }))
        monkeypatch.setenv("PUBLIC_DATA_DIR", str(public))
        # Clear any module-level path cache by re-reading via env in loader
        sizer = AdaptiveSizer(data_dir=data_dir)
        signal, agreement = sizer._load_ensemble_signal()
        assert abs(signal - (-0.12)) < 1e-9
        assert abs(agreement - 0.81) < 1e-9

    def test_weight_table_without_vote_keys_returns_defaults(self, tmp_path, monkeypatch):
        """Per-regime weight tables must not invent composite_signal=0 from missing keys only."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        empty_public = tmp_path / "empty_public"
        empty_public.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PUBLIC_DATA_DIR", str(empty_public))
        monkeypatch.setenv("PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA", "1")
        (data_dir / "ensemble_weights.json").write_text(json.dumps({
            "normal": {"momentum": 0.4, "mean_reversion": 0.2},
            "crisis": {"momentum": 0.1},
        }))
        sizer = AdaptiveSizer(data_dir=data_dir)
        signal, agreement = sizer._load_ensemble_signal()
        # No vote keys → defaults (not a false "0.0 consensus from weights")
        assert signal == 0.0
        assert agreement == 0.5



class TestLoadRegimeStateSSOT:
    """Regime confidence should follow live SSOT, not stale classifier fixtures."""

    def test_stale_classifier_defers_to_signals_regime_confidence(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        empty_public = tmp_path / "empty_public"
        empty_public.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PUBLIC_DATA_DIR", str(empty_public))
        monkeypatch.setenv("PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA", "1")
        # May-stale classifier with failed low confidence
        (data_dir / "regime_classifier_state.json").write_text(json.dumps({
            "current_regime": "normal",
            "last_updated": "2026-05-21T00:00:00",
            "last_reading": {
                "regime": "normal",
                "confidence": 0.3,
                "regime_reason": "Insufficient data",
            },
        }))
        (data_dir / "signals.json").write_text(json.dumps({
            "ensemble_voting": {
                "regime": "normal",
                "regime_confidence": 0.755,
                "weighted_consensus": 0.05,
                "agreement_ratio": 0.9,
            },
            "regime": {"name": "normal", "confidence": 0.755},
        }))
        sizer = AdaptiveSizer(data_dir=data_dir)
        regime, conf = sizer._load_regime_state()
        assert regime == "normal"
        assert abs(conf - 0.755) < 1e-9

    def test_fresh_regime_state_json_preferred(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        empty_public = tmp_path / "empty_public"
        empty_public.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PUBLIC_DATA_DIR", str(empty_public))
        monkeypatch.setenv("PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA", "1")
        now = datetime.now().isoformat()
        (data_dir / "regime_state.json").write_text(json.dumps({
            "regime": "high_vol",
            "confidence": 0.88,
            "updated_at": now,
            "generated_at": now,
        }))
        (data_dir / "regime_classifier_state.json").write_text(json.dumps({
            "current_regime": "normal",
            "last_updated": "2026-05-21T00:00:00",
            "last_reading": {"regime": "normal", "confidence": 0.3},
        }))
        sizer = AdaptiveSizer(data_dir=data_dir)
        regime, conf = sizer._load_regime_state()
        assert regime == "high_vol"
        assert abs(conf - 0.88) < 1e-9

    def test_fresh_classifier_still_used_when_no_better_source(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        empty_public = tmp_path / "empty_public"
        empty_public.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PUBLIC_DATA_DIR", str(empty_public))
        monkeypatch.setenv("PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA", "1")
        (data_dir / "regime_classifier_state.json").write_text(json.dumps({
            "current_regime": "crisis",
            "last_updated": datetime.now().isoformat(),
            "last_reading": {"regime": "crisis", "confidence": 0.91},
        }))
        sizer = AdaptiveSizer(data_dir=data_dir)
        regime, conf = sizer._load_regime_state()
        assert regime == "crisis"
        assert abs(conf - 0.91) < 1e-9


class TestGetSeries:
    """_get_series edge cases."""

    def test_no_prices_returns_none(self, tmp_path):
        """When prices DataFrame is explicitly empty, returns None."""
        import pandas as pd
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        sizer = AdaptiveSizer(data_dir=data_dir)
        sizer._prices_df = pd.DataFrame()  # Empty DataFrame — triggers the symbol-not-found path
        assert sizer._get_series("SPY") is None

    def test_missing_symbol_returns_none(self, tmp_path):
        import pandas as pd
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        sizer = AdaptiveSizer(data_dir=data_dir)
        sizer._prices_df = pd.DataFrame({"QQQ": [400.0]})
        assert sizer._get_series("SPY") is None
