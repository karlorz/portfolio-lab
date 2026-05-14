"""Tests for convexity harvest strategy module."""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
import sys


class TestConvexityPosition:
    """ConvexityPosition dataclass."""

    def test_position_creation(self):
        from src.strategy.convexity_harvest import ConvexityPosition
        pos = ConvexityPosition(
            date="2026-05-14", allocation_pct=3.5,
            position_type="short_vix", vix_level=18.0,
            contango_pct=7.5, expected_roll_yield=0.5,
            entry_reason="Strong contango", exit_triggered=False,
            exit_reason=None, unrealized_pnl=0.0, realized_pnl=0.0
        )
        assert pos.allocation_pct == 3.5
        assert pos.position_type == "short_vix"
        assert pos.vix_level == 18.0

    def test_position_flat(self):
        from src.strategy.convexity_harvest import ConvexityPosition
        pos = ConvexityPosition(
            date="2026-05-14", allocation_pct=0.0,
            position_type="flat", vix_level=32.0,
            contango_pct=-2.0, expected_roll_yield=0.0,
            entry_reason="VIX stress", exit_triggered=True,
            exit_reason="VIX > 30", unrealized_pnl=0.0, realized_pnl=-150.0
        )
        assert pos.position_type == "flat"
        assert pos.allocation_pct == 0.0

    def test_to_dict(self):
        from src.strategy.convexity_harvest import ConvexityPosition
        pos = ConvexityPosition(
            date="2026-05-14", allocation_pct=2.0,
            position_type="short_vix", vix_level=15.0,
            contango_pct=6.0, expected_roll_yield=0.3,
            entry_reason="Moderate contango", exit_triggered=False,
            exit_reason=None, unrealized_pnl=100.0, realized_pnl=0.0
        )
        d = pos.to_dict()
        assert d["date"] == "2026-05-14"
        assert d["allocation_pct"] == 2.0
        assert d["position_type"] == "short_vix"


class TestStrategyConstants:
    """ConvexityHarvestStrategy constants."""

    @pytest.fixture
    def strategy(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        return ConvexityHarvestStrategy(vix_data_manager=mock_mgr)

    def test_contango_thresholds(self, strategy):
        assert strategy.CONTANGO_ENTRY_THRESHOLD > 0
        assert strategy.STRONG_CONTANGO_THRESHOLD > strategy.CONTANGO_ENTRY_THRESHOLD

    def test_vix_stress_threshold(self, strategy):
        assert strategy.VIX_STRESS_THRESHOLD > 0

    def test_max_allocation(self, strategy):
        assert 0 < strategy.MAX_ALLOCATION_PCT <= 10


class TestCalculatePositionSize:
    """Position sizing based on contango and VIX level."""

    @pytest.fixture
    def strategy(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        return ConvexityHarvestStrategy(vix_data_manager=mock_mgr)

    def test_stress_vix_returns_zero(self, strategy):
        alloc, reason = strategy.calculate_position_size(
            contango_pct=10.0, vix_level=strategy.VIX_STRESS_THRESHOLD + 1
        )
        assert alloc == 0.0
        assert "stress" in reason.lower()

    def test_backwardation_returns_zero(self, strategy):
        alloc, reason = strategy.calculate_position_size(
            contango_pct=-5.0, vix_level=15.0
        )
        assert alloc == 0.0
        assert "backwardation" in reason.lower()

    def test_flat_contango_returns_zero(self, strategy):
        alloc, reason = strategy.calculate_position_size(
            contango_pct=1.0, vix_level=15.0
        )
        assert alloc == 0.0

    def test_moderate_contango_positive_allocation(self, strategy):
        alloc, reason = strategy.calculate_position_size(
            contango_pct=7.0, vix_level=15.0
        )
        assert alloc > 0
        assert alloc <= strategy.MAX_ALLOCATION_PCT

    def test_strong_contango_higher_allocation(self, strategy):
        alloc_moderate, _ = strategy.calculate_position_size(contango_pct=7.0, vix_level=15.0)
        alloc_strong, _ = strategy.calculate_position_size(contango_pct=15.0, vix_level=15.0)
        assert alloc_strong > alloc_moderate

    def test_high_vix_reduces_allocation(self, strategy):
        alloc_low_vix, _ = strategy.calculate_position_size(contango_pct=10.0, vix_level=12.0)
        alloc_high_vix, _ = strategy.calculate_position_size(contango_pct=10.0, vix_level=25.0)
        assert alloc_high_vix < alloc_low_vix

    def test_allocation_capped_at_max(self, strategy):
        alloc, _ = strategy.calculate_position_size(
            contango_pct=50.0, vix_level=10.0
        )
        assert alloc <= strategy.MAX_ALLOCATION_PCT

    def test_contango_near_entry_threshold(self, strategy):
        """Right at the entry threshold should produce allocation."""
        alloc, _ = strategy.calculate_position_size(
            contango_pct=strategy.CONTANGO_ENTRY_THRESHOLD + 0.1, vix_level=15.0
        )
        assert alloc > 0


class TestExitTriggers:
    """Exit trigger logic."""

    @pytest.fixture
    def strategy(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        return ConvexityHarvestStrategy(vix_data_manager=mock_mgr)

    def test_vix_stress_triggers_exit(self, strategy):
        should_exit, reason = strategy.check_exit_triggers(
            vix_level=strategy.VIX_STRESS_THRESHOLD + 1,
            contango_pct=10.0, date="2026-05-14"
        )
        assert should_exit is True

    def test_normal_vix_no_exit(self, strategy):
        should_exit, reason = strategy.check_exit_triggers(
            vix_level=15.0, contango_pct=10.0, date="2026-05-14"
        )
        assert should_exit is False

    def test_backwardation_triggers_exit(self, strategy):
        should_exit, reason = strategy.check_exit_triggers(
            vix_level=15.0, contango_pct=-3.0, date="2026-05-14"
        )
        assert should_exit is True

    def test_flat_contango_triggers_exit(self, strategy):
        """Contango below entry threshold should trigger exit."""
        should_exit, reason = strategy.check_exit_triggers(
            vix_level=15.0, contango_pct=1.0, date="2026-05-14"
        )
        assert should_exit is True

    def test_exit_returns_reason_string(self, strategy):
        should_exit, reason = strategy.check_exit_triggers(
            vix_level=strategy.VIX_STRESS_THRESHOLD + 1,
            contango_pct=10.0, date="2026-05-14"
        )
        assert reason is not None
        assert isinstance(reason, str)
        assert len(reason) > 0


class TestGenerateSignal:
    """Signal generation flow."""

    @pytest.fixture
    def strategy(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_vix_term_structure.return_value = {
            "spot": 18.0,
            "m1": 20.0, "m2": 21.5,
            "contango_pct": 8.3
        }
        mock_mgr.get_vix_history.return_value = [18.0, 17.5, 17.0]
        return ConvexityHarvestStrategy(vix_data_manager=mock_mgr)

    def test_generate_signal_returns_position(self, strategy):
        pos = strategy.generate_signal("2026-05-14")
        from src.strategy.convexity_harvest import ConvexityPosition
        assert isinstance(pos, ConvexityPosition)
        assert pos.date == "2026-05-14"
        assert pos.position_type in ("short_vix", "long_vix", "flat")

    def test_generate_signal_stores_in_history(self, strategy):
        pos = strategy.generate_signal("2026-05-14")
        assert len(strategy.position_history) >= 1


class TestGetCurrentSignal:
    """get_current_signal method."""

    def test_get_current_signal_no_history(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        result = strategy.get_current_signal()
        assert isinstance(result, dict)

    def test_get_current_signal_with_history(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy, ConvexityPosition
        mock_mgr = MagicMock()
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        strategy.position_history.append(ConvexityPosition(
            date="2026-05-14", allocation_pct=3.0,
            position_type="short_vix", vix_level=18.0,
            contango_pct=7.0, expected_roll_yield=0.4,
            entry_reason="Test", exit_triggered=False,
            exit_reason=None, unrealized_pnl=0.0, realized_pnl=0.0
        ))
        result = strategy.get_current_signal()
        assert isinstance(result, dict)
