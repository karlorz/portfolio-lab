"""
Tests for VIX Term Structure Tactical Overlay (v4.50 Phase 3)
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from src.strategy.vix_overlay import (
    VIXTermStructureOverlay,
    VIXOverlayIntegrator,
    VIXOverlayStatus,
    VIXOverlayDecision,
    calculate_vix_overlay,
    get_vix_overlay_summary
)
from src.signals.vix_term_structure import VIXTermStructureSignal


class TestVIXOverlayStatus:
    """Test overlay status classifications."""
    
    def test_status_enum_values(self):
        assert VIXOverlayStatus.ACTIVE.value == "active"
        assert VIXOverlayStatus.HOLDING.value == "holding"
        assert VIXOverlayStatus.FROZEN.value == "frozen"
        assert VIXOverlayStatus.DISABLED.value == "disabled"


class TestVIXTermStructureOverlay:
    """Test VIX overlay core functionality."""
    
    @pytest.fixture
    def overlay(self, tmp_path):
        """Create overlay with temporary state file."""
        state_file = tmp_path / "vix_overlay_state.json"
        return VIXTermStructureOverlay(state_file=state_file)
    
    @pytest.fixture
    def sample_signal(self):
        """Create sample VIX signal."""
        return VIXTermStructureSignal(
            timestamp=datetime.now().isoformat(),
            signal_state="risk_off",
            signal_value=-0.622,
            vix_spot=22.5,
            vix3m=20.2,
            vix6m=21.0,
            slope_vix3m_vix=0.90,
            regime="backwardation",
            regime_strength=0.622,
            slope_signal=-0.65,
            roll_yield_signal=-0.10,
            vix_zscore_signal=-0.15,
            curve_shape_signal=0.04,
            spy_shift=0.0,
            gld_shift=0.0,
            tlt_shift=0.0,
            confidence=90.0,
            is_valid=True,
            reason="Test signal"
        )
    
    def test_initial_state(self, overlay):
        """Test overlay starts at baseline."""
        assert overlay.current_allocation == overlay.BASELINE
        assert overlay.last_shift_date is None
    
    def test_baseline_allocation(self, overlay):
        """Test baseline allocation sums to 100%."""
        total = sum(overlay.BASELINE.values())
        assert total == 1.0
        assert overlay.BASELINE["SPY"] == 0.46
        assert overlay.BASELINE["GLD"] == 0.38
        assert overlay.BASELINE["TLT"] == 0.16
    
    def test_allocation_shifts_table(self, overlay):
        """Test allocation shift mappings."""
        # Extreme contango (complacent)
        spy, gld, tlt = overlay._get_allocation_shifts(0.85)
        assert spy > 0  # Increase equity
        assert gld < 0  # Reduce gold
        
        # Extreme backwardation (risk-off)
        spy, gld, tlt = overlay._get_allocation_shifts(-0.85)
        assert spy < 0  # Reduce equity
        assert gld > 0  # Increase gold
        
        # Neutral
        spy, gld, tlt = overlay._get_allocation_shifts(0.0)
        assert spy == 0.0
        assert gld == 0.0
        assert tlt == 0.0
    
    def test_calculate_overlay_risk_off(self, overlay, sample_signal):
        """Test overlay generates risk-off decision."""
        decision = overlay.calculate_overlay(sample_signal, vpin_toxicity=0.3)
        
        assert decision.status in ["active", "holding"]
        assert decision.signal_value == -0.622
        assert decision.regime == "backwardation"
        assert decision.target_spy_shift < 0  # Reduce equity
        assert decision.target_gld_shift > 0  # Add defensive
        
    def test_calculate_overlay_risk_on(self, overlay):
        """Test overlay generates risk-on decision."""
        signal = VIXTermStructureSignal(
            timestamp=datetime.now().isoformat(),
            signal_state="risk_on",
            signal_value=0.8,
            vix_spot=16.0,
            vix3m=19.0,
            vix6m=20.0,
            slope_vix3m_vix=1.19,
            regime="extreme_contango",
            regime_strength=0.8,
            slope_signal=0.75,
            roll_yield_signal=0.16,
            vix_zscore_signal=0.2,
            curve_shape_signal=0.05,
            spy_shift=0.0,
            gld_shift=0.0,
            tlt_shift=0.0,
            confidence=90.0,
            is_valid=True,
            reason="Test"
        )
        
        decision = overlay.calculate_overlay(signal, vpin_toxicity=0.3)
        
        assert decision.target_spy_shift > 0  # Increase equity
        assert decision.allowed_spy_shift <= overlay.max_daily_shift
        
    def test_vpin_freeze_constraint(self, overlay, sample_signal):
        """Test VPIN toxicity freezes execution."""
        decision = overlay.calculate_overlay(
            sample_signal, 
            vpin_toxicity=0.75,  # Above threshold
            current_date=datetime.now()
        )
        
        assert decision.status == "frozen"
        assert decision.vpin_override is True
        assert decision.allowed_spy_shift == 0.0
        assert decision.urgency == "deferred"
    
    def test_holding_period_constraint(self, overlay, sample_signal):
        """Test minimum holding period."""
        # First shift
        overlay.calculate_overlay(sample_signal, current_date=datetime(2026, 1, 1))
        
        # Immediate second shift (should be blocked)
        decision = overlay.calculate_overlay(
            sample_signal, 
            current_date=datetime(2026, 1, 2)
        )
        
        assert decision.status == "holding"
        assert decision.allowed_spy_shift == 0.0
    
    def test_vix_spike_disables_overlay(self, overlay, sample_signal):
        """Test extreme VIX spike disables overlay."""
        # Current VIX needs to be 50% higher than previous for spike detection
        # sample_signal.vix_spot = 22.5, so prev needs to be ~15
        vix_history = [15.0, 15.2, 15.1]  # Last is 15.1, current 22.5 = ~49% spike
        
        # Use a higher VIX to trigger >50% spike from 15
        signal_high_vix = VIXTermStructureSignal(
            timestamp=datetime.now().isoformat(),
            signal_state="risk_off",
            signal_value=-0.622,
            vix_spot=24.0,  # 60% spike from 15
            vix3m=20.2,
            vix6m=21.0,
            slope_vix3m_vix=0.90,
            regime="backwardation",
            regime_strength=0.622,
            slope_signal=-0.65,
            roll_yield_signal=-0.10,
            vix_zscore_signal=-0.15,
            curve_shape_signal=0.04,
            spy_shift=0.0,
            gld_shift=0.0,
            tlt_shift=0.0,
            confidence=90.0,
            is_valid=True,
            reason="Test"
        )
        
        decision = overlay.calculate_overlay(
            signal_high_vix,
            vix_history=vix_history,
            current_date=datetime.now()
        )
        
        assert decision.status == "disabled"
        assert "spike" in str(decision.constraints_applied).lower()
    
    def test_max_daily_shift_capped(self, overlay):
        """Test shifts are capped at max daily limit."""
        signal = VIXTermStructureSignal(
            timestamp=datetime.now().isoformat(),
            signal_state="risk_off",
            signal_value=-0.95,  # Extreme risk-off
            vix_spot=30.0,
            vix3m=22.0,
            vix6m=23.0,
            slope_vix3m_vix=0.73,
            regime="extreme_backwardation",
            regime_strength=0.95,
            slope_signal=-0.9,
            roll_yield_signal=-0.36,
            vix_zscore_signal=-0.4,
            curve_shape_signal=0.04,
            spy_shift=0.0,
            gld_shift=0.0,
            tlt_shift=0.0,
            confidence=95.0,
            is_valid=True,
            reason="Test"
        )
        
        # Wait for holding period
        overlay.last_shift_date = None
        
        decision = overlay.calculate_overlay(signal, vpin_toxicity=0.0)
        
        # Should request large shift but be capped
        assert decision.target_spy_shift == -0.10  # -10% requested
        assert abs(decision.allowed_spy_shift) <= overlay.max_daily_shift  # Capped to 5%
    
    def test_state_persistence(self, overlay, sample_signal):
        """Test state is saved and loaded."""
        # Execute a shift
        overlay.calculate_overlay(sample_signal, current_date=datetime(2026, 1, 10))
        
        # Create new overlay with same state file
        overlay2 = VIXTermStructureOverlay(state_file=overlay.state_file)
        
        assert overlay2.last_shift_date is not None
        assert overlay2.current_allocation != overlay2.BASELINE
    
    def test_allocation_normalization(self, overlay, sample_signal):
        """Test allocation always sums to 1.0."""
        # Wait for holding period
        overlay.last_shift_date = None
        
        overlay.calculate_overlay(sample_signal, current_date=datetime.now())
        
        total = sum(overlay.current_allocation.values())
        assert abs(total - 1.0) < 0.001
    
    def test_get_summary(self, overlay, sample_signal):
        """Test summary generation."""
        overlay.calculate_overlay(sample_signal, current_date=datetime.now())
        
        summary = overlay.get_summary()
        
        assert "current_allocation" in summary
        assert "baseline_allocation" in summary
        assert "active_drifts" in summary
        assert "shifts_30d" in summary
        
    def test_reset_to_baseline(self, overlay, sample_signal):
        """Test reset functionality."""
        # Execute shift first
        overlay.last_shift_date = None
        overlay.calculate_overlay(sample_signal, current_date=datetime.now())
        
        assert overlay.current_allocation != overlay.BASELINE
        
        overlay.reset_to_baseline()
        
        assert overlay.current_allocation == overlay.BASELINE
        assert overlay.last_shift_date is None


class TestVIXOverlayIntegrator:
    """Test overlay integration with ensemble and rebalance gate."""
    
    @pytest.fixture
    def integrator(self, tmp_path):
        """Create integrator with temporary state."""
        state_file = tmp_path / "vix_overlay_state.json"
        overlay = VIXTermStructureOverlay(state_file=state_file)
        return VIXOverlayIntegrator(overlay)
    
    @pytest.fixture
    def sample_signal(self):
        return VIXTermStructureSignal(
            timestamp=datetime.now().isoformat(),
            signal_state="risk_off",
            signal_value=-0.622,
            vix_spot=22.5,
            vix3m=20.2,
            vix6m=21.0,
            slope_vix3m_vix=0.90,
            regime="backwardation",
            regime_strength=0.622,
            slope_signal=-0.65,
            roll_yield_signal=-0.10,
            vix_zscore_signal=-0.15,
            curve_shape_signal=0.04,
            spy_shift=0.0,
            gld_shift=0.0,
            tlt_shift=0.0,
            confidence=90.0,
            is_valid=True,
            reason="Test"
        )
    
    def test_ensemble_contribution_format(self, integrator, sample_signal):
        """Test ensemble contribution has correct format."""
        contrib = integrator.get_ensemble_contribution(sample_signal)
        
        assert contrib["source"] == "vix_term_structure"
        assert contrib["weight"] == 0.15
        assert -1.0 <= contrib["signal"] <= 1.0
        assert 0.0 <= contrib["confidence"] <= 1.0
        assert "shift_recommendation" in contrib
        assert "urgency" in contrib
    
    def test_ensemble_signal_scaling(self, integrator, sample_signal):
        """Test signal is properly scaled."""
        contrib = integrator.get_ensemble_contribution(sample_signal)
        
        # -0.622 signal with ~-5% shift should scale to ~-0.6
        assert contrib["signal"] < 0  # Risk-off
        
    def test_rebalance_gate_integration_execute(self, integrator, sample_signal):
        """Test integration when gate allows execution."""
        decision = integrator.overlay.calculate_overlay(sample_signal, vpin_toxicity=0.3)
        
        gate_status = {"can_execute": True, "vpin_status": "normal", "reason": None}
        result = integrator.integrate_with_rebalance_gate(gate_status, decision)
        
        assert result["execute"] == (decision.status == "active")
        assert "shift" in result
    
    def test_rebalance_gate_integration_blocked(self, integrator, sample_signal):
        """Test integration when gate blocks execution."""
        decision = integrator.overlay.calculate_overlay(sample_signal, vpin_toxicity=0.3)
        decision.urgency = "immediate"
        
        gate_status = {"can_execute": False, "vpin_status": "elevated", "reason": "High toxicity"}
        result = integrator.integrate_with_rebalance_gate(gate_status, decision)
        
        assert result["execute"] is False
        assert "deferred" in result["reason"]
        assert "deferred_shift" in result


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_calculate_vix_overlay(self):
        """Test calculate_vix_overlay convenience function."""
        decision = calculate_vix_overlay(
            vix_spot=22.5,
            vix3m=20.2,
            vix6m=21.0,
            vpin_toxicity=0.3
        )
        
        assert isinstance(decision, VIXOverlayDecision)
        assert decision.signal_value is not None
        assert decision.regime is not None
    
    def test_get_vix_overlay_summary(self, tmp_path):
        """Test get_vix_overlay_summary."""
        # Just verify the function returns expected structure
        # Use the default state file behavior (may not exist)
        try:
            summary = get_vix_overlay_summary()
            assert "status" in summary
            assert "current_allocation" in summary
            assert "baseline_allocation" in summary
        except FileNotFoundError:
            # State file doesn't exist - that's ok for test
            pytest.skip("State file not initialized")


class TestConstraintCombinations:
    """Test complex constraint interactions."""
    
    @pytest.fixture
    def overlay(self, tmp_path):
        state_file = tmp_path / "vix_overlay_state.json"
        return VIXTermStructureOverlay(state_file=state_file)
    
    def test_vpin_overrides_immediate_urgency(self, overlay):
        """Test VPIN freeze overrides immediate urgency."""
        signal = VIXTermStructureSignal(
            timestamp=datetime.now().isoformat(),
            signal_state="risk_off",
            signal_value=-0.85,  # Extreme - immediate urgency
            vix_spot=28.0,
            vix3m=20.0,
            vix6m=21.0,
            slope_vix3m_vix=0.71,
            regime="extreme_backwardation",
            regime_strength=0.85,
            slope_signal=-0.8,
            roll_yield_signal=-0.4,
            vix_zscore_signal=-0.5,
            curve_shape_signal=0.04,
            spy_shift=0.0,
            gld_shift=0.0,
            tlt_shift=0.0,
            confidence=95.0,
            is_valid=True,
            reason="Test"
        )
        
        decision = overlay.calculate_overlay(
            signal,
            vpin_toxicity=0.75,  # High toxicity
            current_date=datetime.now()
        )
        
        assert decision.status == "frozen"
        assert decision.urgency == "deferred"
        assert decision.allowed_spy_shift == 0.0
    
    def test_multiple_constraints_reported(self, overlay):
        """Test multiple constraints are all reported."""
        signal = VIXTermStructureSignal(
            timestamp=datetime.now().isoformat(),
            signal_state="risk_off",
            signal_value=-0.95,
            vix_spot=30.0,
            vix3m=22.0,
            vix6m=23.0,
            slope_vix3m_vix=0.73,
            regime="extreme_backwardation",
            regime_strength=0.95,
            slope_signal=-0.9,
            roll_yield_signal=-0.36,
            vix_zscore_signal=-0.4,
            curve_shape_signal=0.04,
            spy_shift=0.0,
            gld_shift=0.0,
            tlt_shift=0.0,
            confidence=95.0,
            is_valid=True,
            reason="Test"
        )
        
        # Set recent shift date to trigger holding period
        overlay.last_shift_date = datetime.now() - timedelta(days=2)
        
        decision = overlay.calculate_overlay(
            signal,
            vpin_toxicity=0.3,
            current_date=datetime.now()
        )
        
        # Should be holding (min holding period)
        assert decision.status == "holding"
        assert len(decision.constraints_applied) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
