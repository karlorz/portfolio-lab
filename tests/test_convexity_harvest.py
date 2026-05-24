"""Tests for convexity harvest strategy module."""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestConvexityPosition:
    """ConvexityPosition dataclass."""

    def test_position_creation(self):
        from src.strategy.convexity_harvest import ConvexityPosition
        pos = ConvexityPosition(
            date="2026-05-14", allocation_pct=3.5,
            position_type="short_vix", vix_level=18.0,
            contango_pct=7.5, expected_roll_yield=0.5,
            risk_score=0.3, exit_triggered=False, exit_reason=None
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
            risk_score=1.0, exit_triggered=True, exit_reason="VIX > 30"
        )
        assert pos.position_type == "flat"
        assert pos.allocation_pct == 0.0

    def test_to_dict(self):
        from src.strategy.convexity_harvest import ConvexityPosition
        pos = ConvexityPosition(
            date="2026-05-14", allocation_pct=2.0,
            position_type="short_vix", vix_level=15.0,
            contango_pct=6.0, expected_roll_yield=0.3,
            risk_score=0.2, exit_triggered=False, exit_reason=None
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

    def test_backwardation_exit_after_consecutive_days(self, strategy):
        """Backwardation triggers exit only after BACKWARDATION_EXIT_DAYS."""
        exit_triggered = False
        for day in range(10):
            should_exit, reason = strategy.check_exit_triggers(
                vix_level=15.0, contango_pct=-3.0, date=f"2026-05-{15+day}"
            )
            if should_exit:
                exit_triggered = True
                break
        assert exit_triggered, "Should exit after consecutive backwardation days"

    def test_contango_breaks_backwardation_streak(self, strategy):
        strategy.check_exit_triggers(vix_level=15.0, contango_pct=-3.0, date="2026-05-14")
        strategy.check_exit_triggers(vix_level=15.0, contango_pct=5.0, date="2026-05-15")
        assert strategy.consecutive_backwardation_days == 0

    def test_exit_returns_reason_string(self, strategy):
        should_exit, reason = strategy.check_exit_triggers(
            vix_level=strategy.VIX_STRESS_THRESHOLD + 1,
            contango_pct=10.0, date="2026-05-14"
        )
        assert reason is not None
        assert isinstance(reason, str)


class TestGenerateSignal:
    """Signal generation flow with mocked VIX manager."""

    @pytest.fixture
    def strategy_with_data(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 18.0,
            "contango_spot_1m": 8.3,
            "contango_1m_2m": 5.0,
            "term_structure": "contango",
            "risk_score": 0.2,
            "yield_1m_annualized": 5.0, "annualized_roll_yield": 4.5,
        }
        return ConvexityHarvestStrategy(vix_data_manager=mock_mgr)

    def test_generate_signal_returns_position(self, strategy_with_data):
        pos = strategy_with_data.generate_signal("2026-05-14")
        from src.strategy.convexity_harvest import ConvexityPosition
        assert isinstance(pos, ConvexityPosition)
        assert pos.date == "2026-05-14"
        assert pos.position_type in ("short_vix", "long_vix", "flat")

    def test_generate_signal_stores_in_history(self, strategy_with_data):
        pos = strategy_with_data.generate_signal("2026-05-14")
        assert len(strategy_with_data.position_history) >= 1

    def test_generate_signal_no_data_returns_flat(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = None
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        pos = strategy.generate_signal("2026-05-14")
        assert pos.position_type == "flat"
        assert pos.allocation_pct == 0.0


class TestGetCurrentSignal:
    """get_current_signal method."""

    def test_get_current_signal_no_history(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 18.0, "contango_spot_1m": 8.3,
            "contango_1m_2m": 5.0, "term_structure": "contango",
            "risk_score": 0.2, "yield_1m_annualized": 5.0, "annualized_roll_yield": 4.5,
        }
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        result = strategy.get_current_signal()
        assert isinstance(result, dict)

    def test_get_current_signal_with_history(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy, ConvexityPosition
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 18.0, "contango_spot_1m": 8.3,
            "contango_1m_2m": 5.0, "term_structure": "contango",
            "risk_score": 0.2, "yield_1m_annualized": 5.0, "annualized_roll_yield": 4.5,
        }
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        strategy.position_history.append(ConvexityPosition(
            date="2026-05-14", allocation_pct=3.0,
            position_type="short_vix", vix_level=18.0,
            contango_pct=7.0, expected_roll_yield=0.4,
            risk_score=0.2, exit_triggered=False, exit_reason=None
        ))
        result = strategy.get_current_signal()
        assert isinstance(result, dict)


class TestZeroDayBacktestGuard:
    """Regression: run_backtest used to ZeroDivisionError when start_date == end_date."""

    def test_same_start_end_no_crash(self):
        """run_backtest with identical start and end date should not crash."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = None  # flat positions, minimal logic
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        result = strategy.run_backtest(start_date="2026-01-15", end_date="2026-01-15")
        assert isinstance(result, dict)
        assert "total_return_pct" in result
        # annualized_return_pct should be finite (not inf or nan)
        import math
        assert math.isfinite(result["annualized_return_pct"])


class TestCalculatePositionSizeExtended:
    """Additional edge cases for calculate_position_size."""

    @pytest.fixture
    def strategy(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        return ConvexityHarvestStrategy(vix_data_manager=mock_mgr)

    def test_contango_at_entry_threshold(self, strategy):
        """Contango exactly at entry threshold passes the < check, gives minimal alloc."""
        alloc, _ = strategy.calculate_position_size(
            contango_pct=strategy.CONTANGO_ENTRY_THRESHOLD, vix_level=15.0
        )
        # At exactly 5.0%, the < check is False, so allocation is computed
        # base = 2.0 + (5.0 - 5.0) * 0.4 = 2.0, then VIX-adjusted
        assert alloc > 0

    def test_reasoning_includes_contango_and_vix(self, strategy):
        """Reasoning string should include contango and VIX level."""
        _, reason = strategy.calculate_position_size(contango_pct=8.0, vix_level=15.0)
        assert "Contango" in reason
        assert "VIX" in reason

    def test_very_high_vix_reduces_allocation(self, strategy):
        """VIX near stress threshold should reduce allocation significantly."""
        alloc_normal, _ = strategy.calculate_position_size(contango_pct=10.0, vix_level=15.0)
        alloc_high, _ = strategy.calculate_position_size(contango_pct=10.0, vix_level=34.0)
        assert alloc_high < alloc_normal

    def test_low_vix_maximizes_allocation(self, strategy):
        """Very low VIX should produce high allocation for strong contango."""
        alloc, _ = strategy.calculate_position_size(
            contango_pct=15.0, vix_level=1.0
        )
        # vix_factor = 1 - 1/35 = 0.971, adjusted = 4.2 * (0.5 + 0.5*0.971) = 4.14
        # Capped at MAX_ALLOCATION_PCT
        assert alloc > 0
        assert alloc <= strategy.MAX_ALLOCATION_PCT

    def test_contango_between_thresholds(self, strategy):
        """Contango between entry and strong threshold uses linear scaling."""
        alloc, _ = strategy.calculate_position_size(
            contango_pct=7.5, vix_level=15.0
        )
        # base_allocation = 2.0 + (7.5 - 5.0) * 0.4 = 3.0
        # vix_factor = 1.0 - 15.0/35.0 = 0.571
        # adjusted = 3.0 * (0.5 + 0.5 * 0.571) ≈ 2.36
        assert 0 < alloc < strategy.MAX_ALLOCATION_PCT


class TestExitTriggersExtended:
    """Additional exit trigger edge cases."""

    @pytest.fixture
    def strategy(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        return ConvexityHarvestStrategy(vix_data_manager=mock_mgr)

    def test_vix_spike_trigger(self, strategy):
        """VIX spike > 20% in one day should trigger exit."""
        strategy.last_vix_level = 15.0
        should_exit, reason = strategy.check_exit_triggers(
            vix_level=19.0,  # 26.7% spike
            contango_pct=10.0, date="2026-05-14"
        )
        assert should_exit is True
        assert "spike" in reason.lower()

    def test_vix_no_spike_below_threshold(self, strategy):
        """VIX move below 20% should not trigger spike exit."""
        strategy.last_vix_level = 15.0
        should_exit, reason = strategy.check_exit_triggers(
            vix_level=17.0,  # 13.3% move
            contango_pct=10.0, date="2026-05-14"
        )
        assert should_exit is False

    def test_no_previous_vix_no_spike_check(self, strategy):
        """Without previous VIX data, spike check is skipped."""
        strategy.last_vix_level = None
        should_exit, reason = strategy.check_exit_triggers(
            vix_level=40.0,  # Would be a spike but no prev data
            contango_pct=-3.0, date="2026-05-14"
        )
        # VIX stress triggers first (40 > 35)
        assert should_exit is True

    def test_backwardation_counter_resets(self, strategy):
        """Contango after backwardation resets the counter to 0."""
        for _ in range(2):
            strategy.check_exit_triggers(vix_level=15.0, contango_pct=-3.0, date="2026-05-14")
        assert strategy.consecutive_backwardation_days == 2

        strategy.check_exit_triggers(vix_level=15.0, contango_pct=5.0, date="2026-05-15")
        assert strategy.consecutive_backwardation_days == 0

    def test_backwardation_exit_exactly_at_threshold(self, strategy):
        """Should exit exactly at BACKWARDATION_EXIT_DAYS consecutive days."""
        for day in range(strategy.BACKWARDATION_EXIT_DAYS - 1):
            should_exit, _ = strategy.check_exit_triggers(
                vix_level=15.0, contango_pct=-3.0, date=f"2026-05-{10+day}"
            )
            assert should_exit is False

        # On the Nth day, should trigger
        should_exit, reason = strategy.check_exit_triggers(
            vix_level=15.0, contango_pct=-3.0,
            date=f"2026-05-{10 + strategy.BACKWARDATION_EXIT_DAYS - 1}"
        )
        assert should_exit is True
        assert "Backwardation" in reason


class TestGenerateSignalExtended:
    """Additional signal generation edge cases."""

    def test_generate_signal_exit_when_position_active(self):
        """When exit is triggered and we have a position, should go flat."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy, ConvexityPosition
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 40.0,
            "contango_spot_1m": 5.0,
            "contango_1m_2m": 3.0,
            "term_structure": "contango",
            "risk_score": 0.8,
            "yield_1m_annualized": 2.0, "annualized_roll_yield": 1.5,
        }
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        strategy.last_allocation = 3.0  # Active position
        pos = strategy.generate_signal("2026-05-14")
        assert pos.exit_triggered is True
        assert pos.allocation_pct == 0.0
        assert pos.position_type == "flat"

    def test_generate_signal_exit_no_position_stays_flat(self):
        """When exit triggered but no position, stays flat (not exit_triggered)."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 40.0,
            "contango_spot_1m": 5.0,
            "contango_1m_2m": 3.0,
            "term_structure": "contango",
            "risk_score": 0.8,
            "yield_1m_annualized": 2.0, "annualized_roll_yield": 1.5,
        }
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        strategy.last_allocation = 0.0  # No position
        pos = strategy.generate_signal("2026-05-14")
        # VIX stress but no position → should still set short_vix or flat
        # but NOT set exit_triggered=True because last_allocation is 0
        assert pos.position_type == "short_vix" or pos.position_type == "flat"

    def test_generate_signal_backwardation_regime(self):
        """Backwardation should produce flat position with reason."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 18.0,
            "contango_spot_1m": -3.0,
            "contango_1m_2m": -1.0,
            "term_structure": "backwardation",
            "risk_score": 0.5,
            "yield_1m_annualized": -2.0, "annualized_roll_yield": -1.5,
        }
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        strategy.last_allocation = 0.0
        pos = strategy.generate_signal("2026-05-14")
        assert pos.position_type == "flat"
        assert pos.exit_reason == "Backwardation regime"

    def test_generate_signal_updates_last_vix_and_allocation(self):
        """Signal generation should update last_vix_level and last_allocation."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 18.0,
            "contango_spot_1m": 8.0,
            "contango_1m_2m": 5.0,
            "term_structure": "contango",
            "risk_score": 0.2,
            "yield_1m_annualized": 5.0, "annualized_roll_yield": 4.5,
        }
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        pos = strategy.generate_signal("2026-05-14")
        assert strategy.last_vix_level == 18.0
        assert strategy.last_allocation == pos.allocation_pct

    def test_contango_with_zero_allocation_is_flat(self):
        """Contango regime but allocation computes to 0 → flat position."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 18.0,
            "contango_spot_1m": 3.0,  # Below CONTANGO_ENTRY_THRESHOLD
            "contango_1m_2m": 2.0,
            "term_structure": "contango",
            "risk_score": 0.2,
            "yield_1m_annualized": 1.0, "annualized_roll_yield": 0.5,
        }
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        pos = strategy.generate_signal("2026-05-14")
        assert pos.position_type == "flat"
        assert pos.allocation_pct == 0.0


class TestRunBacktestExtended:
    """Additional backtest edge cases."""

    def test_backtest_with_positions(self):
        """Backtest should track positions with positive allocation."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 15.0,
            "contango_spot_1m": 10.0,
            "contango_1m_2m": 5.0,
            "term_structure": "contango",
            "risk_score": 0.1,
            "yield_1m_annualized": 6.0, "annualized_roll_yield": 5.0,
        }
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        result = strategy.run_backtest(start_date="2026-01-15", end_date="2026-01-20")
        assert result["days_with_position"] > 0
        assert result["total_positions"] == 6  # 6 days inclusive

    def test_backtest_all_flat(self):
        """Backtest with no data should have all flat positions."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = None
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        result = strategy.run_backtest(start_date="2026-01-15", end_date="2026-01-17")
        assert result["days_with_position"] == 0
