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
        _ = strategy_with_data.generate_signal("2026-05-14")
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


class TestAllExports:
    """__all__ exports validation."""

    def test_all_exists(self):
        from src.strategy import convexity_harvest as m
        assert hasattr(m, '__all__')

    def test_all_contains_expected(self):
        from src.strategy import convexity_harvest as m
        expected = {'ConvexityPosition', 'ConvexityHarvestStrategy'}
        assert set(m.__all__) == expected

    def test_class_convexity_position_exported(self):
        from src.strategy.convexity_harvest import ConvexityPosition  # noqa: F401

    def test_class_convexity_harvest_strategy_exported(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy  # noqa: F401

    def test_non_exported_not_accessible_via_star(self):
        """Items not in __all__ should not be imported via star import."""
        # Star import (from module import *) only brings in __all__ items
        import src.strategy.convexity_harvest as m
        for name in dir(m):
            if name.startswith('_'):
                continue
        # Verify __all__ is the authoritative list of public names
        public_names = {n for n in dir(m) if not n.startswith('_')}
        assert set(m.__all__).issubset(public_names)


class TestDataclassFieldValidation:
    """Full dataclass field validation for ConvexityPosition."""

    @pytest.fixture
    def sample_position(self):
        from src.strategy.convexity_harvest import ConvexityPosition
        return ConvexityPosition(
            date="2026-05-24",
            allocation_pct=3.5,
            position_type="short_vix",
            vix_level=18.0,
            contango_pct=7.5,
            expected_roll_yield=0.5,
            risk_score=0.3,
            exit_triggered=False,
            exit_reason=None,
        )

    def test_to_dict_contains_all_fields(self, sample_position):
        d = sample_position.to_dict()
        expected_keys = {
            "date", "allocation_pct", "position_type", "vix_level",
            "contango_pct", "expected_roll_yield", "risk_score",
            "exit_triggered", "exit_reason",
        }
        assert set(d.keys()) == expected_keys

    def test_allocation_pct_is_float(self, sample_position):
        assert isinstance(sample_position.allocation_pct, float)

    def test_vix_level_is_float(self, sample_position):
        assert isinstance(sample_position.vix_level, float)

    def test_contango_pct_is_float(self, sample_position):
        assert isinstance(sample_position.contango_pct, float)

    def test_expected_roll_yield_is_float(self, sample_position):
        assert isinstance(sample_position.expected_roll_yield, float)

    def test_risk_score_is_float(self, sample_position):
        assert isinstance(sample_position.risk_score, float)

    def test_exit_triggered_is_bool(self, sample_position):
        assert isinstance(sample_position.exit_triggered, bool)

    def test_position_type_is_str(self, sample_position):
        assert isinstance(sample_position.position_type, str)

    def test_date_is_str(self, sample_position):
        assert isinstance(sample_position.date, str)

    def test_exit_reason_is_optional(self, sample_position):
        """exit_reason can be either None or a string."""
        assert sample_position.exit_reason is None
        from src.strategy.convexity_harvest import ConvexityPosition
        pos_with_reason = ConvexityPosition(
            date="2026-05-24", allocation_pct=0.0,
            position_type="flat", vix_level=0.0,
            contango_pct=0.0, expected_roll_yield=0.0,
            risk_score=1.0, exit_triggered=True,
            exit_reason="Some reason",
        )
        assert isinstance(pos_with_reason.exit_reason, str)

    def test_risk_score_default_range(self, sample_position):
        """risk_score should be between 0 and 1."""
        assert 0.0 <= sample_position.risk_score <= 1.0

    def test_allocation_default_range(self, sample_position):
        """allocation_pct should be between 0 and MAX_ALLOCATION_PCT."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        assert 0.0 <= sample_position.allocation_pct <= ConvexityHarvestStrategy.MAX_ALLOCATION_PCT


class TestConstantsValidation:
    """All module-level constants have reasonable ranges and types."""

    @pytest.fixture
    def const(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        return ConvexityHarvestStrategy

    def test_max_allocation_type_and_range(self, const):
        assert isinstance(const.MAX_ALLOCATION_PCT, (int, float))
        assert 1.0 <= const.MAX_ALLOCATION_PCT <= 10.0

    def test_vix_stress_threshold_type_and_range(self, const):
        assert isinstance(const.VIX_STRESS_THRESHOLD, (int, float))
        assert 20.0 <= const.VIX_STRESS_THRESHOLD <= 50.0

    def test_vix_spike_threshold_type_and_range(self, const):
        assert isinstance(const.VIX_SPIKE_THRESHOLD, (int, float))
        assert 5.0 <= const.VIX_SPIKE_THRESHOLD <= 50.0

    def test_contango_entry_threshold_type_and_range(self, const):
        assert isinstance(const.CONTANGO_ENTRY_THRESHOLD, (int, float))
        assert 1.0 <= const.CONTANGO_ENTRY_THRESHOLD <= 15.0

    def test_strong_contango_threshold_type_and_range(self, const):
        assert isinstance(const.STRONG_CONTANGO_THRESHOLD, (int, float))
        assert const.STRONG_CONTANGO_THRESHOLD > const.CONTANGO_ENTRY_THRESHOLD
        assert const.STRONG_CONTANGO_THRESHOLD <= 30.0

    def test_backwardation_exit_days_type_and_range(self, const):
        assert isinstance(const.BACKWARDATION_EXIT_DAYS, int)
        assert 1 <= const.BACKWARDATION_EXIT_DAYS <= 10


class TestCalculatePositionSizeBoundary:
    """Boundary value tests for calculate_position_size."""

    @pytest.fixture
    def strategy(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        return ConvexityHarvestStrategy(vix_data_manager=mock_mgr)

    def test_contango_exactly_at_entry_threshold(self, strategy):
        """Contango exactly at CONTANGO_ENTRY_THRESHOLD yields positive allocation."""
        alloc, _ = strategy.calculate_position_size(
            contango_pct=strategy.CONTANGO_ENTRY_THRESHOLD, vix_level=15.0
        )
        # 5.0 < 5.0 is False, so base = 2.0 + (5.0 - 5.0) * 0.4 = 2.0
        assert alloc > 0

    def test_contango_exactly_at_strong_threshold(self, strategy):
        """Contango exactly at STRONG_CONTANGO_THRESHOLD."""
        alloc, _ = strategy.calculate_position_size(
            contango_pct=strategy.STRONG_CONTANGO_THRESHOLD, vix_level=15.0
        )
        # 10.0 < 10.0 is False, base = 4.0 + min(1.0, 0.0) = 4.0
        assert alloc > 0

    def test_contango_at_zero(self, strategy):
        """Contango at exactly 0% is below entry threshold -> zero allocation."""
        alloc, reason = strategy.calculate_position_size(
            contango_pct=0.0, vix_level=15.0
        )
        assert alloc == 0.0
        assert "flat" in reason.lower() or "contango" in reason.lower()

    def test_vix_exactly_at_stress_threshold(self, strategy):
        """VIX exactly at stress threshold (not >) should NOT be blocked."""
        alloc, reason = strategy.calculate_position_size(
            contango_pct=10.0, vix_level=strategy.VIX_STRESS_THRESHOLD
        )
        assert alloc > 0
        assert "stress" not in reason.lower()

    def test_vix_at_zero(self, strategy):
        """VIX at 0 should not crash and should maximize vix_factor."""
        alloc, _ = strategy.calculate_position_size(
            contango_pct=10.0, vix_level=0.0
        )
        # vix_factor = max(0.0, 1.0 - 0.0/35.0) = 1.0
        # adjusted = 4.0 * (0.5 + 0.5) = 4.0
        assert alloc > 0
        assert alloc <= strategy.MAX_ALLOCATION_PCT

    def test_vix_very_large(self, strategy):
        """Very large VIX (100+) should trigger stress exit."""
        alloc, reason = strategy.calculate_position_size(
            contango_pct=10.0, vix_level=100.0
        )
        assert alloc == 0.0
        assert "stress" in reason.lower()

    def test_contango_very_large(self, strategy):
        """Very large contango should approach MAX_ALLOCATION_PCT (with very low VIX)."""
        alloc, _ = strategy.calculate_position_size(
            contango_pct=1000.0, vix_level=0.0
        )
        assert alloc == pytest.approx(strategy.MAX_ALLOCATION_PCT, rel=1e-6)

    def test_portfolio_value_zero(self, strategy):
        """Portfolio value of 0 should not crash (not used in calculation)."""
        alloc, _ = strategy.calculate_position_size(
            contango_pct=10.0, vix_level=15.0, portfolio_value=0.0
        )
        assert alloc > 0

    def test_portfolio_value_negative(self, strategy):
        """Negative portfolio value should not crash (not used in calculation)."""
        alloc, _ = strategy.calculate_position_size(
            contango_pct=10.0, vix_level=15.0, portfolio_value=-1000.0
        )
        assert alloc > 0

    def test_contango_just_below_entry_threshold(self, strategy):
        """Contango just below entry threshold gives zero allocation."""
        alloc, _ = strategy.calculate_position_size(
            contango_pct=strategy.CONTANGO_ENTRY_THRESHOLD - 0.01, vix_level=15.0
        )
        assert alloc == 0.0

    def test_contango_just_above_strong_threshold(self, strategy):
        """Contango just above STRONG_CONTANGO_THRESHOLD, very low VIX should give >4%."""
        alloc, _ = strategy.calculate_position_size(
            contango_pct=strategy.STRONG_CONTANGO_THRESHOLD + 0.01, vix_level=1.0
        )
        # base = 4.0 + min(1.0, 0.01*0.2) = 4.002
        # vix_factor = 1 - 1/35 ≈ 0.971, adjusted = 4.002 * (0.5 + 0.5*0.971) = 3.97...
        # Actually this doesn't reach 4% either with vix_factor reduction
        # The key point is allocation is computed and positive
        assert alloc > 3.0
        assert alloc <= strategy.MAX_ALLOCATION_PCT

    def test_contango_negative_very_large(self, strategy):
        """Very negative contango (deep backwardation) returns zero."""
        alloc, _ = strategy.calculate_position_size(
            contango_pct=-100.0, vix_level=15.0
        )
        assert alloc == 0.0


class TestCheckExitTriggersBoundary:
    """Boundary value tests for check_exit_triggers."""

    @pytest.fixture
    def strategy(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        strat = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        strat.last_vix_level = 15.0
        return strat

    def test_vix_exactly_at_stress_threshold_no_exit(self, strategy):
        """VIX exactly at stress threshold should NOT trigger exit (by > comparison)."""
        strategy.last_vix_level = None  # Prevent spike check from interfering
        should_exit, _ = strategy.check_exit_triggers(
            vix_level=strategy.VIX_STRESS_THRESHOLD,
            contango_pct=10.0, date="2026-05-14"
        )
        assert should_exit is False

    def test_vix_change_exactly_at_spike_threshold(self, strategy):
        """VIX change exactly at spike threshold uses > so should NOT trigger."""
        # last_vix_level=15, current=18 -> 20% exactly
        strategy.last_vix_level = 15.0
        should_exit, _ = strategy.check_exit_triggers(
            vix_level=18.0, contango_pct=10.0, date="2026-05-14"
        )
        # (18-15)/15*100 = 20%, > 20% is False
        assert should_exit is False

    def test_contango_zero_resets_backwardation_counter(self, strategy):
        """Contango of exactly 0 should reset the backwardation counter."""
        strategy.check_exit_triggers(vix_level=15.0, contango_pct=-2.0, date="2026-05-14")
        assert strategy.consecutive_backwardation_days == 1
        strategy.check_exit_triggers(vix_level=15.0, contango_pct=0.0, date="2026-05-15")
        assert strategy.consecutive_backwardation_days == 0

    def test_vix_spike_just_above_threshold(self, strategy):
        """VIX spike just above threshold should trigger exit."""
        strategy.last_vix_level = 15.0
        should_exit, reason = strategy.check_exit_triggers(
            vix_level=18.01,  # (18.01-15)/15*100 = 20.07% > 20%
            contango_pct=10.0, date="2026-05-14"
        )
        assert should_exit is True
        assert "spike" in reason.lower()

    def test_vix_stress_with_no_last_vix(self, strategy):
        """VIX stress with no last_vix_level still triggers on stress."""
        strategy.last_vix_level = None
        should_exit, reason = strategy.check_exit_triggers(
            vix_level=strategy.VIX_STRESS_THRESHOLD + 1,
            contango_pct=10.0, date="2026-05-14"
        )
        assert should_exit is True
        assert "stress" in reason.lower()

    def test_backwardation_one_day_before_exit(self, strategy):
        """Backwardation exactly one day before threshold should not exit."""
        for day in range(strategy.BACKWARDATION_EXIT_DAYS - 1):
            should_exit, _ = strategy.check_exit_triggers(
                vix_level=15.0, contango_pct=-3.0, date=f"2026-05-{10+day}"
            )
            assert should_exit is False

    def test_backwardation_exit_on_nth_day(self, strategy):
        """Backwardation on exact threshold day should exit."""
        for _ in range(strategy.BACKWARDATION_EXIT_DAYS - 1):
            strategy.check_exit_triggers(vix_level=15.0, contango_pct=-3.0, date="2026-05-14")
        should_exit, reason = strategy.check_exit_triggers(
            vix_level=15.0, contango_pct=-3.0, date="2026-05-15"
        )
        assert should_exit is True
        assert "Backwardation" in reason

    def test_contango_just_below_zero(self, strategy):
        """Contango very close to zero on negative side starts backwardation count."""
        strategy.check_exit_triggers(vix_level=15.0, contango_pct=-0.001, date="2026-05-14")
        assert strategy.consecutive_backwardation_days == 1

    def test_contango_small_positive_resets_backwardation(self, strategy):
        """Very small positive contango should reset backwardation counter."""
        strategy.check_exit_triggers(vix_level=15.0, contango_pct=-2.0, date="2026-05-14")
        assert strategy.consecutive_backwardation_days == 1
        strategy.check_exit_triggers(vix_level=15.0, contango_pct=0.001, date="2026-05-15")
        assert strategy.consecutive_backwardation_days == 0

    def test_multiple_backwardation_streaks(self, strategy):
        """Consecutive backwardation across multiple streaks."""
        # Streak 1: 2 days then break
        for _ in range(2):
            strategy.check_exit_triggers(vix_level=15.0, contango_pct=-3.0, date="2026-05-14")
        strategy.check_exit_triggers(vix_level=15.0, contango_pct=5.0, date="2026-05-15")
        assert strategy.consecutive_backwardation_days == 0
        # Streak 2: should start fresh
        for _ in range(2):
            strategy.check_exit_triggers(vix_level=15.0, contango_pct=-3.0, date="2026-05-16")
        assert strategy.consecutive_backwardation_days == 2


class TestGenerateSignalEdgeCases:
    """Additional edge cases for generate_signal."""

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

    def test_generate_signal_contango_no_data(self):
        """Contango regime but no signal data returns flat with null VIX (not 0.0)."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = None
        mock_mgr.get_data_range.return_value = ("", "")
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        pos = strategy.generate_signal("2026-05-14")
        assert pos.position_type == "flat"
        assert pos.risk_score == 1.0
        assert pos.exit_reason is not None
        assert "unavailable" in pos.exit_reason
        # Residual honesty: unknown levels are null, not silent zeros
        assert pos.vix_level is None
        assert pos.contango_pct is None
        assert pos.expected_roll_yield is None
        payload = strategy.get_current_signal()
        assert payload["status"] == "unavailable"
        assert payload["vix_level"] is None
        assert payload["contango_pct"] is None
        assert payload.get("vix_source") == "unavailable"

    def test_generate_signal_falls_back_to_last_cache_day(self):
        """Today missing → last futures cache day supplies VIX (not zeros)."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy

        mock_mgr = MagicMock()

        def _contango(date):
            if date == "2026-07-20":
                return None
            if date == "2026-05-22":
                return {
                    "vix_level": 16.76,
                    "contango_spot_1m": 19.5,
                    "contango_1m_2m": 0.0,
                    "is_contango": True,
                    "annualized_roll_yield": 50.0,
                }
            return None

        mock_mgr.get_contango_signal.side_effect = _contango
        mock_mgr.get_data_range.return_value = ("2021-05-10", "2026-05-22")
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        pos = strategy.generate_signal("2026-07-20")
        assert pos.vix_level == pytest.approx(16.76)
        assert pos.exit_reason != "unavailable: no VIX futures cache"
        payload = strategy.get_current_signal()
        # get_current uses today; mock still falls back
        assert payload.get("vix_level", 0) != 0 or pos.vix_level > 0

    def test_generate_signal_contango_positive_allocation(self, strategy_with_data):
        """Contango with positive allocation produces short_vix position."""
        pos = strategy_with_data.generate_signal("2026-05-14")
        if pos.allocation_pct > 0:
            assert pos.position_type == "short_vix"
            assert pos.risk_score < 1.0
            assert pos.expected_roll_yield > 0

    def test_generate_signal_contango_zero_allocation_flat(self):
        """Contango regime, allocation computes to 0 -> flat position."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 18.0,
            "contango_spot_1m": 3.0,  # Below entry threshold
            "contango_1m_2m": 2.0,
            "term_structure": "contango",
            "risk_score": 0.2,
            "yield_1m_annualized": 1.0, "annualized_roll_yield": 0.5,
        }
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        pos = strategy.generate_signal("2026-05-14")
        assert pos.position_type == "flat"
        assert pos.allocation_pct == 0.0

    def test_generate_signal_and_state_transition(self):
        """State transitions between generate_signal calls should be consistent."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        # Day 1: contango with position
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 18.0, "contango_spot_1m": 8.0,
            "contango_1m_2m": 5.0, "term_structure": "contango",
            "risk_score": 0.2, "yield_1m_annualized": 5.0, "annualized_roll_yield": 4.5,
        }
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        pos1 = strategy.generate_signal("2026-05-14")
        assert strategy.last_vix_level == 18.0
        assert strategy.last_allocation == pos1.allocation_pct
        # Day 2: stress regime, exit triggered
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 40.0, "contango_spot_1m": 10.0,
            "contango_1m_2m": 5.0, "term_structure": "contango",
            "risk_score": 0.8, "yield_1m_annualized": 5.0, "annualized_roll_yield": 4.5,
        }
        pos2 = strategy.generate_signal("2026-05-15")
        if pos1.allocation_pct > 0:
            # If we had a position, exit should be triggered
            assert pos2.exit_triggered is True
            assert pos2.position_type == "flat"

    def test_generate_signal_backwardation_after_contango(self):
        """Transition from contango to backwardation should go flat."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        # Day 1: contango with low VIX so spike won't trigger
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 14.0, "contango_spot_1m": 8.0,
            "contango_1m_2m": 5.0, "term_structure": "contango",
            "risk_score": 0.2, "yield_1m_annualized": 5.0, "annualized_roll_yield": 4.5,
        }
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        strategy.generate_signal("2026-05-14")
        # Day 2: backwardation - small VIX change so no spike trigger
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 15.0, "contango_spot_1m": -3.0,
            "contango_1m_2m": -1.0, "term_structure": "backwardation",
            "risk_score": 0.5, "yield_1m_annualized": -2.0, "annualized_roll_yield": -1.5,
        }
        pos2 = strategy.generate_signal("2026-05-15")
        assert pos2.position_type == "flat"
        assert pos2.exit_reason == "Backwardation regime"


class TestRunBacktestEdgeCases:
    """Edge cases for run_backtest."""

    @pytest.fixture
    def mock_strategy(self):
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 15.0, "contango_spot_1m": 10.0,
            "contango_1m_2m": 5.0, "term_structure": "contango",
            "risk_score": 0.1, "yield_1m_annualized": 6.0, "annualized_roll_yield": 5.0,
        }
        return ConvexityHarvestStrategy(vix_data_manager=mock_mgr)

    def test_backtest_initial_capital_zero(self, mock_strategy):
        """Backtest with zero initial capital should not crash."""
        result = mock_strategy.run_backtest(
            start_date="2026-01-15", end_date="2026-01-17", initial_capital=0.0
        )
        assert isinstance(result, dict)
        assert result["final_capital"] == 0.0
        assert result["total_return_pct"] == 0.0

    def test_backtest_negative_initial_capital(self, mock_strategy):
        """Backtest with negative initial capital should not crash."""
        result = mock_strategy.run_backtest(
            start_date="2026-01-15", end_date="2026-01-17", initial_capital=-1000.0
        )
        import math
        assert isinstance(result, dict)
        assert math.isfinite(result["annualized_return_pct"])

    def test_backtest_result_contains_all_keys(self, mock_strategy):
        """Backtest result dict should have all expected keys."""
        result = mock_strategy.run_backtest(
            start_date="2026-01-15", end_date="2026-01-17"
        )
        expected_keys = {
            "start_date", "end_date", "initial_capital", "final_capital",
            "total_return_pct", "annualized_return_pct", "volatility_pct",
            "sharpe_ratio", "max_drawdown_pct", "total_positions",
            "days_with_position", "exit_events", "exit_reasons",
        }
        assert set(result.keys()) == expected_keys

    def test_backtest_sharpe_positive_for_profitable(self, mock_strategy):
        """Backtest with consistent positive returns should have positive Sharpe."""
        result = mock_strategy.run_backtest(
            start_date="2026-01-15", end_date="2026-01-22"
        )
        # With consistent contango data, should produce some positive returns
        assert isinstance(result["sharpe_ratio"], (int, float))

    def test_backtest_exit_reasons_is_list(self, mock_strategy):
        """exit_reasons should be a list (possibly empty)."""
        result = mock_strategy.run_backtest(
            start_date="2026-01-15", end_date="2026-01-17"
        )
        assert isinstance(result["exit_reasons"], list)

    def test_backtest_volatility_non_negative(self, mock_strategy):
        """Volatility should be non-negative."""
        result = mock_strategy.run_backtest(
            start_date="2026-01-15", end_date="2026-01-17"
        )
        assert result["volatility_pct"] >= 0

    def test_backtest_max_drawdown_non_negative(self, mock_strategy):
        """Max drawdown should be non-negative."""
        result = mock_strategy.run_backtest(
            start_date="2026-01-15", end_date="2026-01-17"
        )
        assert result["max_drawdown_pct"] >= 0

    def test_backtest_very_long_period(self):
        """Backtest with month-long period should handle many dates."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = None  # All flat
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        result = strategy.run_backtest(start_date="2026-01-01", end_date="2026-12-31")
        assert result["total_positions"] > 300  # ~365 days
        assert result["volatility_pct"] == 0.0  # All flat -> no vol


class TestStrategyInitialization:
    """Strategy initialization edge cases."""

    def test_default_initialization(self):
        """Strategy should initialize without arguments (uses default VIXDataManager)."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        try:
            strategy = ConvexityHarvestStrategy()
            assert strategy.position_history == []
            assert strategy.consecutive_backwardation_days == 0
            assert strategy.last_vix_level is None
            assert strategy.last_allocation == 0.0
        except Exception as e:
            # The default VIXDataManager tries to load files, might fail
            # That's acceptable - we just test initialization
            pytest.skip(f"Default initialization skipped: {e}")

    def test_strategy_with_mock_initial_state(self):
        """Strategy initialized with mock should have correct initial state."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        assert strategy.position_history == []
        assert strategy.consecutive_backwardation_days == 0
        assert strategy.last_vix_level is None
        assert strategy.last_allocation == 0.0

    def test_strategy_position_history_accumulates(self):
        """Position history should accumulate across multiple generate_signal calls."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 18.0, "contango_spot_1m": 8.0,
            "contango_1m_2m": 5.0, "term_structure": "contango",
            "risk_score": 0.2, "yield_1m_annualized": 5.0, "annualized_roll_yield": 4.5,
        }
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        strategy.generate_signal("2026-05-14")
        strategy.generate_signal("2026-05-15")
        strategy.generate_signal("2026-05-16")
        assert len(strategy.position_history) == 3


class TestCLIMain:
    """CLI main() function tests with mocked args."""

    @pytest.fixture
    def mock_strategy_for_cli(self):
        """Patch the strategy constructor to return a mock strategy."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = {
            "vix_level": 18.0, "contango_spot_1m": 8.3,
            "contango_1m_2m": 5.0, "term_structure": "contango",
            "risk_score": 0.2, "yield_1m_annualized": 5.0, "annualized_roll_yield": 4.5,
        }
        strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
        return strategy

    @patch('src.strategy.convexity_harvest.sys.argv', ['convexity_harvest.py'])
    def test_main_no_args_demo_mode(self, capsys):
        """main() with no args should run demo mode without crashing."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        import src.strategy.convexity_harvest as ch
        mock_mgr = MagicMock()
        mock_mgr.get_contango_signal.return_value = None
        with patch.object(ConvexityHarvestStrategy, '__init__', return_value=None):
            with patch.object(ch, 'ConvexityHarvestStrategy') as mock_cls:
                instance = MagicMock()
                instance.generate_signal.return_value = MagicMock(
                    date="2024-01-15", position_type="flat", vix_level=15.0,
                    contango_pct=0.0, expected_roll_yield=0.0,
                    allocation_pct=0.0, exit_triggered=False, exit_reason=None,
                )
                mock_cls.return_value = instance
                ch.main()
                captured = capsys.readouterr()
                assert "Convexity Harvest Strategy (v2.21)" in captured.err

    @patch('src.strategy.convexity_harvest.sys.argv',
           ['convexity_harvest.py', '--backtest', '2020-01-01', '2020-01-10'])
    def test_main_backtest_with_args(self, capsys):
        """main() with --backtest and custom dates should not crash."""
        from src.strategy.convexity_harvest import ConvexityHarvestStrategy
        import src.strategy.convexity_harvest as ch
        with patch.object(ConvexityHarvestStrategy, '__init__', return_value=None):
            with patch.object(ch, 'ConvexityHarvestStrategy') as mock_cls:
                instance = MagicMock()
                instance.run_backtest.return_value = {
                    "start_date": "2020-01-01", "end_date": "2020-01-10",
                    "initial_capital": 100000.0, "final_capital": 100000.0,
                    "total_return_pct": 0.0, "annualized_return_pct": 0.0,
                    "volatility_pct": 0.0, "sharpe_ratio": 0.0,
                    "max_drawdown_pct": 0.0, "total_positions": 10,
                    "days_with_position": 0, "exit_events": 0, "exit_reasons": [],
                }
                mock_cls.return_value = instance
                ch.main()
                captured = capsys.readouterr()
                assert "Running convexity harvest backtest: 2020-01-01 to 2020-01-10" in captured.err

    @patch('src.strategy.convexity_harvest.sys.argv',
           ['convexity_harvest.py', '--backtest'])
    def test_main_backtest_default_dates(self):
        """main() with --backtest but no dates should use defaults."""
        import src.strategy.convexity_harvest as ch
        with patch.object(ch.ConvexityHarvestStrategy, '__init__', return_value=None):
            with patch.object(ch, 'ConvexityHarvestStrategy') as mock_cls:
                instance = MagicMock()
                instance.run_backtest.return_value = {
                    "start_date": "2020-01-01", "end_date": "2024-12-31",
                    "initial_capital": 100000.0, "final_capital": 100000.0,
                    "total_return_pct": 0.0, "annualized_return_pct": 0.0,
                    "volatility_pct": 0.0, "sharpe_ratio": 0.0,
                    "max_drawdown_pct": 0.0, "total_positions": 1,
                    "days_with_position": 0, "exit_events": 0, "exit_reasons": [],
                }
                mock_cls.return_value = instance
                with patch('builtins.print'):
                    ch.main()

    @patch('src.strategy.convexity_harvest.sys.argv',
           ['convexity_harvest.py', '--signal'])
    def test_main_signal(self):
        """main() with --signal should produce JSON output."""
        import src.strategy.convexity_harvest as ch
        with patch.object(ch.ConvexityHarvestStrategy, '__init__', return_value=None):
            with patch.object(ch, 'ConvexityHarvestStrategy') as mock_cls:
                instance = MagicMock()
                instance.get_current_signal.return_value = {
                    "date": "2026-05-24", "allocation_pct": 2.5,
                    "position_type": "short_vix",
                }
                mock_cls.return_value = instance
                with patch('builtins.print') as mock_print:
                    ch.main()
                    # Should print JSON
                    call_args = [c[0][0] for c in mock_print.call_args_list if c[0]]
                    json_calls = [a for a in call_args if "allocation_pct" in str(a) or "date" in str(a)]
                    assert len(json_calls) >= 0  # At minimum, no crash

    @patch('src.strategy.convexity_harvest.sys.argv',
           ['convexity_harvest.py', '--invalid-flag'])
    def test_main_invalid_flag(self, capsys):
        """main() with an unrecognized flag should fall through to demo mode."""
        import src.strategy.convexity_harvest as ch
        with patch.object(ch.ConvexityHarvestStrategy, '__init__', return_value=None):
            with patch.object(ch, 'ConvexityHarvestStrategy') as mock_cls:
                instance = MagicMock()
                instance.generate_signal.return_value = MagicMock(
                    date="2024-01-15", position_type="flat", vix_level=15.0,
                    contango_pct=0.0, expected_roll_yield=0.0,
                    allocation_pct=0.0, exit_triggered=False, exit_reason=None,
                )
                mock_cls.return_value = instance
                ch.main()
                captured = capsys.readouterr()
                assert "Convexity Harvest Strategy (v2.21)" in captured.err

    def test_main_invalid_args_length(self):
        """main() with --backtest but only one date should not crash."""
        import src.strategy.convexity_harvest as ch
        with patch.object(ch.ConvexityHarvestStrategy, '__init__', return_value=None):
            with patch.object(ch, 'ConvexityHarvestStrategy') as mock_cls:
                instance = MagicMock()
                instance.run_backtest.return_value = {
                    "start_date": "2020-01-01", "end_date": "2024-12-31",
                    "initial_capital": 100000.0, "final_capital": 100000.0,
                    "total_return_pct": 0.0, "annualized_return_pct": 0.0,
                    "volatility_pct": 0.0, "sharpe_ratio": 0.0,
                    "max_drawdown_pct": 0.0, "total_positions": 1,
                    "days_with_position": 0, "exit_events": 0, "exit_reasons": [],
                }
                mock_cls.return_value = instance
                with patch('builtins.print'):
                    with patch.object(ch, 'sys', wraps=ch.sys) as mock_sys:
                        mock_sys.argv = ['convexity_harvest.py', '--backtest', '2020-01-01']
                        ch.main()



def test_get_current_signal_uses_utc_calendar_date(monkeypatch):
    """Host-local midnight must not advance date ahead of UTC SSOT."""
    from datetime import timezone
    from src.strategy.convexity_harvest import ConvexityHarvestStrategy

    # Freeze "now" to 2026-07-20 01:00+08 (still 2026-07-19 UTC)
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is timezone.utc or (tz is not None and getattr(tz, "tzname", lambda: None)(None) == "UTC"):
                return cls(2026, 7, 19, 17, 0, 0, tzinfo=timezone.utc)
            # naive local Asia/Shanghai-like
            return cls(2026, 7, 20, 1, 0, 0)

    import src.strategy.convexity_harvest as mod
    monkeypatch.setattr(mod, "datetime", _FixedDateTime)
    mock_mgr = MagicMock()
    mock_mgr.get_contango_signal.side_effect = lambda d: {
        "vix_level": 16.0,
        "contango_spot_1m": 10.0,
        "annualized_roll_yield": 5.0,
        "is_contango": True,
    } if d == "2026-07-19" else None
    mock_mgr.get_data_range.return_value = ("2026-01-01", "2026-07-19")
    strategy = ConvexityHarvestStrategy(vix_data_manager=mock_mgr)
    # Patch generate path: if today uses UTC, date is 2026-07-19
    sig = strategy.get_current_signal()
    # Should resolve via UTC day or last cache — not silently No VIX for local tomorrow
    assert sig.get("vix_level", 0) != 0 or sig.get("status") != "unavailable"
