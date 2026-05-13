#!/usr/bin/env python3
"""
Test Duration Overlay - v3.11 Phase 2

Tests for duration_overlay.py implementing yield curve regime-based allocation.
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals.yield_curve_regime import (
    YieldCurveRegimeClassifier,
    YieldCurveRegime,
    YieldCurveData,
    RegimeClassification
)
from strategy.duration_overlay import (
    DurationOverlay,
    DurationAllocation,
    RegimeShift,
    OverlayRecommendation,
    STATE_PATH,
    YIELDS_PATH,
    DB_PATH,
)


class TestDurationAllocation:
    """Test DurationAllocation dataclass."""
    
    def test_basic_allocation(self):
        """Test basic allocation creation."""
        alloc = DurationAllocation(tlt=0.16, ief=0.15, shy=0.05, bil=0.00)
        
        assert alloc.tlt == 0.16
        assert alloc.ief == 0.15
        assert alloc.shy == 0.05
        assert alloc.bil == 0.00
        assert alloc.total_allocation == 0.36
    
    def test_effective_duration_calculation(self):
        """Test effective duration calculation."""
        # 100% TLT should be ~18.5 years
        alloc_all_tlt = DurationAllocation(tlt=1.0, ief=0.0, shy=0.0, bil=0.0)
        assert alloc_all_tlt.effective_duration == pytest.approx(18.5, abs=0.1)
        
        # 100% IEF should be ~7.5 years
        alloc_all_ief = DurationAllocation(tlt=0.0, ief=1.0, shy=0.0, bil=0.0)
        assert alloc_all_ief.effective_duration == pytest.approx(7.5, abs=0.1)
        
        # 100% SHY should be ~1.9 years
        alloc_all_shy = DurationAllocation(tlt=0.0, ief=0.0, shy=1.0, bil=0.0)
        assert alloc_all_shy.effective_duration == pytest.approx(1.9, abs=0.1)
        
        # Flat regime: ~10.6 years effective
        alloc_flat = DurationAllocation(tlt=0.16, ief=0.15, shy=0.05, bil=0.0)
        assert alloc_flat.effective_duration == pytest.approx(11.6, abs=0.5)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        alloc = DurationAllocation(tlt=0.16, ief=0.15, shy=0.05, bil=0.00)
        d = alloc.to_dict()
        
        assert d == {"tlt": 0.16, "ief": 0.15, "shy": 0.05, "bil": 0.0}


class TestDurationOverlayInitialization:
    """Test DurationOverlay initialization."""
    
    @patch.object(Path, 'exists', return_value=False)
    def test_default_initialization(self, mock_exists):
        """Test overlay with default parameters."""
        classifier = Mock()
        overlay = DurationOverlay(
            base_spy=0.46,
            base_gld=0.38,
            base_bond_total=0.16,
            classifier=classifier
        )
        
        assert overlay.base_spy == 0.46
        assert overlay.base_gld == 0.38
        assert overlay.base_bond_total == 0.16
        assert overlay.classifier == classifier
    
    @patch.object(Path, 'exists', return_value=False)
    def test_custom_allocation(self, mock_exists):
        """Test overlay with custom base allocations."""
        overlay = DurationOverlay(
            base_spy=0.50,
            base_gld=0.30,
            base_bond_total=0.20,
            classifier=Mock()
        )
        
        assert overlay.base_spy == 0.50
        assert overlay.base_gld == 0.30
        assert overlay.base_bond_total == 0.20


class TestDurationOverlayRegimeAllocations:
    """Test regime-specific allocations."""
    
    @patch.object(Path, 'exists', return_value=False)
    def test_inverted_regime_allocation(self, mock_exists):
        """Test allocation during inverted curve regime."""
        overlay = DurationOverlay(classifier=Mock())
        alloc = overlay.REGIME_ALLOCATIONS["inverted"]
        
        assert alloc.tlt == 0.05  # Minimal long duration
        assert alloc.ief == 0.25  # Maximize intermediate
        assert alloc.shy == 0.06  # Maximize short
        assert alloc.effective_duration < 9.0  # Shorter effective duration (~8 years)
    
    @patch.object(Path, 'exists', return_value=False)
    def test_flat_regime_allocation(self, mock_exists):
        """Test allocation during flat curve regime."""
        overlay = DurationOverlay(classifier=Mock())
        alloc = overlay.REGIME_ALLOCATIONS["flat"]
        
        assert alloc.tlt == 0.16  # Neutral
        assert alloc.ief == 0.15
        assert alloc.shy == 0.05
        assert alloc.total_allocation == 0.36
    
    @patch.object(Path, 'exists', return_value=False)
    def test_steep_regime_allocation(self, mock_exists):
        """Test allocation during steep curve regime."""
        overlay = DurationOverlay(classifier=Mock())
        alloc = overlay.REGIME_ALLOCATIONS["steep"]
        
        assert alloc.tlt == 0.22  # Maximize long duration
        assert alloc.ief == 0.10
        assert alloc.shy == 0.04  # Minimize short
        assert alloc.effective_duration > 12.0  # Long effective duration
    
    @patch.object(Path, 'exists', return_value=False)
    def test_unknown_regime_fallback(self, mock_exists):
        """Test fallback allocation for unknown regime."""
        overlay = DurationOverlay(classifier=Mock())
        alloc = overlay.REGIME_ALLOCATIONS["unknown"]
        
        # Should fall back to flat allocation
        assert alloc.tlt == 0.16
        assert alloc.ief == 0.15
        assert alloc.shy == 0.05


class TestTransitionConstraints:
    """Test transition constraint logic."""
    
    @patch.object(Path, 'exists', return_value=False)
    def test_max_shift_per_month(self, mock_exists):
        """Test that shifts are constrained to 25% per month."""
        overlay = DurationOverlay(classifier=Mock())
        overlay.base_bond_total = 0.36  # Full 36% bond allocation
        
        # Try to shift from all flat to all inverted
        current = DurationAllocation(tlt=0.16, ief=0.15, shy=0.05, bil=0.0)
        target = DurationAllocation(tlt=0.05, ief=0.25, shy=0.06, bil=0.0)
        
        constrained = overlay._apply_transition_constraints(current, target)
        
        # Max shift is 25% of 0.36 = 0.09 per component
        max_shift = 0.25 * 0.36
        
        # Allow slightly more tolerance for floating point and normalization
        assert abs(constrained.tlt - current.tlt) <= max_shift + 0.005
        assert abs(constrained.ief - current.ief) <= max_shift + 0.005
        assert abs(constrained.shy - current.shy) <= max_shift + 0.005
    
    @patch.object(Path, 'exists', return_value=False)
    def test_allocation_normalization(self, mock_exists):
        """Test that constrained allocation still sums to base_bond_total."""
        overlay = DurationOverlay(classifier=Mock())
        overlay.base_bond_total = 0.16
        
        current = DurationAllocation(tlt=0.16, ief=0.0, shy=0.0, bil=0.0)
        target = DurationAllocation(tlt=0.0, ief=0.16, shy=0.0, bil=0.0)
        
        constrained = overlay._apply_transition_constraints(current, target)
        
        # Should still sum to base bond total
        total = constrained.tlt + constrained.ief + constrained.shy + constrained.bil
        assert abs(total - 0.16) < 0.001


class TestFallbackBehavior:
    """Test fallback behavior when data is unavailable."""
    
    @patch.object(Path, 'exists', return_value=False)
    def test_fallback_recommendation(self, mock_exists):
        """Test fallback when yield data unavailable."""
        overlay = DurationOverlay(classifier=Mock())
        
        rec = overlay._fallback_recommendation("Test reason")
        
        assert rec.current_regime == "unknown (fallback)"
        assert rec.base_allocation["SPY"] == 0.46
        assert rec.base_allocation["TLT"] == 0.16
        assert rec.base_allocation["IEF"] == 0.15
        assert rec.base_allocation["SHY"] == 0.05
        assert rec.confidence == "low"
        assert "Test reason" in rec.rationale
        assert rec.expected_improvement_bps == 0.0


class TestExpectedImprovements:
    """Test expected Sharpe improvement calculations."""
    
    @patch.object(Path, 'exists', return_value=False)
    def test_inverted_expected_improvement(self, mock_exists):
        """Test expected improvement in inverted regime."""
        overlay = DurationOverlay(classifier=Mock())
        improvement = overlay.EXPECTED_IMPROVEMENT["inverted"]
        
        assert improvement == 0.020  # 20 bps
    
    @patch.object(Path, 'exists', return_value=False)
    def test_flat_expected_improvement(self, mock_exists):
        """Test expected improvement in flat regime."""
        overlay = DurationOverlay(classifier=Mock())
        improvement = overlay.EXPECTED_IMPROVEMENT["flat"]
        
        assert improvement == 0.000  # No improvement (baseline)
    
    @patch.object(Path, 'exists', return_value=False)
    def test_steep_expected_improvement(self, mock_exists):
        """Test expected improvement in steep regime."""
        overlay = DurationOverlay(classifier=Mock())
        improvement = overlay.EXPECTED_IMPROVEMENT["steep"]
        
        assert improvement == 0.015  # 15 bps


class TestRationaleGeneration:
    """Test rationale generation for different regimes."""
    
    @patch.object(Path, 'exists', return_value=False)
    def test_inverted_rationale(self, mock_exists):
        """Test rationale for inverted regime."""
        overlay = DurationOverlay(classifier=Mock())
        
        mock_classification = Mock()
        mock_classification.spread_2s10s = -0.0050  # -50 bps
        
        alloc = overlay.REGIME_ALLOCATIONS["inverted"]
        rationale = overlay._generate_rationale(
            YieldCurveRegime.INVERTED,
            mock_classification,
            alloc
        )
        
        assert "inverted" in rationale.lower()
        assert "-50bps" in rationale or "short duration" in rationale.lower()
        assert "recession" in rationale.lower() or "risk" in rationale.lower()
    
    @patch.object(Path, 'exists', return_value=False)
    def test_steep_rationale(self, mock_exists):
        """Test rationale for steep regime."""
        overlay = DurationOverlay(classifier=Mock())
        
        mock_classification = Mock()
        mock_classification.spread_2s10s = 0.0120  # +120 bps
        
        alloc = overlay.REGIME_ALLOCATIONS["steep"]
        rationale = overlay._generate_rationale(
            YieldCurveRegime.STEEP,
            mock_classification,
            alloc
        )
        
        assert "steep" in rationale.lower()
        assert "growth" in rationale.lower() or "premium" in rationale.lower()


class TestAllocationDelta:
    """Test allocation delta calculations."""
    
    @patch.object(Path, 'exists', return_value=False)
    def test_delta_vs_static(self, mock_exists):
        """Test allocation delta vs static 46/38/16 base."""
        overlay = DurationOverlay(classifier=Mock())
        
        # Create a mock recommendation with inverted allocation
        rec = OverlayRecommendation(
            timestamp=datetime.now().isoformat(),
            current_regime="inverted",
            base_allocation={},
            duration_breakdown={"tlt": 0.05, "ief": 0.25, "shy": 0.06, "bil": 0.0},
            effective_duration=4.0,
            shift_pending=False,
            days_until_shift=0,
            confidence="high",
            rationale="Test",
            expected_improvement_bps=20.0
        )
        
        delta = overlay.get_allocation_delta(rec)
        
        # Static is 16/15/5, inverted is 5/25/6
        assert delta["TLT"] == 0.05 - 0.16  # -11%
        assert delta["IEF"] == 0.25 - 0.15  # +10%
        assert delta["SHY"] == 0.06 - 0.05  # +1%


class TestStatePersistence:
    """Test state loading and saving."""
    
    @patch("builtins.open")
    @patch.object(Path, 'exists', return_value=True)
    def test_load_state(self, mock_exists, mock_open):
        """Test loading state from disk."""
        mock_state = {
            "current_allocation": {"tlt": 0.16, "ief": 0.15, "shy": 0.05, "bil": 0.0},
            "shift_history": [],
        }
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_state)
        
        overlay = DurationOverlay(classifier=Mock())
        state = overlay._load_state()
        
        assert "current_allocation" in state
        assert state["current_allocation"]["tlt"] == 0.16


class TestIntegrationWithClassifier:
    """Test integration with YieldCurveRegimeClassifier."""
    
    @patch.object(Path, 'exists', return_value=False)
    @patch.object(DurationOverlay, '_get_current_yields')
    def test_recommendation_uses_classifier(self, mock_get_yields, mock_exists):
        """Test that recommendation uses classifier for regime detection."""
        # Create mock yield data
        mock_yields = YieldCurveData(
            timestamp=datetime.now().isoformat(),
            dgs10=0.045,
            dgs2=0.050,
            dgs30=None,
            dgs5=None,
            spread_2s10s=-0.005,  # Inverted
            spread_10s30s=None
        )
        mock_get_yields.return_value = mock_yields
        
        # Create mock classifier
        mock_classifier = Mock()
        mock_classification = RegimeClassification(
            timestamp=datetime.now().isoformat(),
            regime=YieldCurveRegime.INVERTED,
            spread_2s10s=-0.005,
            dgs10=0.045,
            dgs2=0.050,
            days_in_regime=45,
            regime_start_date="2026-01-01",
            is_transition_pending=False,
            days_until_eligible=0,
            confidence="high"
        )
        mock_classifier.classify.return_value = mock_classification
        mock_classifier.state = {"current_regime": YieldCurveRegime.INVERTED}
        
        overlay = DurationOverlay(classifier=mock_classifier)
        
        # Temporarily set fresh data
        with patch.object(overlay, '_is_data_fresh', return_value=True):
            rec = overlay.get_recommendation()
        
        # Verify classifier was called
        mock_classifier.classify.assert_called_once()


class TestCLIFunctionality:
    """Test CLI functionality."""
    
    @patch.object(Path, 'exists', return_value=False)
    @patch('strategy.duration_overlay.DurationOverlay._get_current_yields')
    def test_cli_status(self, mock_get_yields, mock_exists):
        """Test CLI status output."""
        mock_get_yields.return_value = None  # Trigger fallback
        
        overlay = DurationOverlay(classifier=Mock())
        status = overlay.cli_status()
        
        assert "Duration Overlay Status" in status
        assert "UNKNOWN" in status.upper() or "fallback" in status.lower()
        assert "Allocation:" in status
        assert "SPY:" in status
        assert "GLD:" in status
        assert "TLT:" in status
    
    @patch.object(Path, 'exists', return_value=False)
    @patch('strategy.duration_overlay.DurationOverlay._get_current_yields')
    def test_cli_recommendation(self, mock_get_yields, mock_exists):
        """Test CLI recommendation output."""
        mock_get_yields.return_value = None  # Trigger fallback
        
        overlay = DurationOverlay(classifier=Mock())
        rec = overlay.get_recommendation()
        
        assert rec.timestamp is not None
        assert rec.current_regime is not None
        assert rec.base_allocation is not None
        assert rec.duration_breakdown is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
