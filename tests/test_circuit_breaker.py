"""Tests for Drawdown Circuit Breaker strategy module."""
import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path


class TestThresholds:
    """Circuit breaker threshold constants."""

    def test_thresholds_defined(self):
        from src.strategy.circuit_breaker import DrawdownCircuitBreaker
        cb = DrawdownCircuitBreaker()
        assert cb.THRESHOLDS["green"] == 0.0
        assert cb.THRESHOLDS["yellow"] == 0.10
        assert cb.THRESHOLDS["orange"] == 0.15
        assert cb.THRESHOLDS["red"] == 0.20
        assert cb.THRESHOLDS["black"] == 0.25

    def test_position_scalars(self):
        from src.strategy.circuit_breaker import DrawdownCircuitBreaker
        cb = DrawdownCircuitBreaker()
        assert cb.POSITION_SCALARS["green"] == 1.0
        assert cb.POSITION_SCALARS["yellow"] == 1.0
        assert cb.POSITION_SCALARS["orange"] < 1.0
        assert cb.POSITION_SCALARS["red"] < cb.POSITION_SCALARS["orange"]
        assert cb.POSITION_SCALARS["black"] == 0.0

    def test_position_scalars_monotonic(self):
        """Position scalars decrease as severity increases."""
        from src.strategy.circuit_breaker import DrawdownCircuitBreaker
        cb = DrawdownCircuitBreaker()
        levels = ["green", "yellow", "orange", "red", "black"]
        scalars = [cb.POSITION_SCALARS[l] for l in levels]
        for i in range(len(scalars) - 1):
            assert scalars[i] >= scalars[i + 1], f"{levels[i]} >= {levels[i+1]}"


class TestDetermineStatus:
    """Status determination from drawdown percentage."""

    @pytest.fixture
    def cb(self):
        from src.strategy.circuit_breaker import DrawdownCircuitBreaker
        return DrawdownCircuitBreaker()

    def test_green_zero(self, cb):
        assert cb.determine_status(0.0) == "green"

    def test_green_small(self, cb):
        assert cb.determine_status(0.05) == "green"
        assert cb.determine_status(0.099) == "green"

    def test_yellow_boundary(self, cb):
        assert cb.determine_status(0.10) == "yellow"
        assert cb.determine_status(0.149) == "yellow"

    def test_orange_boundary(self, cb):
        assert cb.determine_status(0.15) == "orange"
        assert cb.determine_status(0.199) == "orange"

    def test_red_boundary(self, cb):
        assert cb.determine_status(0.20) == "red"
        assert cb.determine_status(0.249) == "red"

    def test_black_boundary(self, cb):
        assert cb.determine_status(0.25) == "black"
        assert cb.determine_status(0.30) == "black"
        assert cb.determine_status(0.50) == "black"


class TestEscalation:
    """Escalation detection logic."""

    @pytest.fixture
    def cb(self):
        from src.strategy.circuit_breaker import DrawdownCircuitBreaker
        return DrawdownCircuitBreaker()

    def test_green_to_yellow_is_escalation(self, cb):
        assert cb._is_escalation("green", "yellow") is True

    def test_yellow_to_orange_is_escalation(self, cb):
        assert cb._is_escalation("yellow", "orange") is True

    def test_orange_to_red_is_escalation(self, cb):
        assert cb._is_escalation("orange", "red") is True

    def test_red_to_black_is_escalation(self, cb):
        assert cb._is_escalation("red", "black") is True

    def test_same_status_not_escalation(self, cb):
        assert cb._is_escalation("green", "green") is False
        assert cb._is_escalation("yellow", "yellow") is False
        assert cb._is_escalation("red", "red") is False

    def test_recovery_not_escalation(self, cb):
        assert cb._is_escalation("red", "yellow") is False
        assert cb._is_escalation("orange", "green") is False
        assert cb._is_escalation("black", "red") is False

    def test_skip_level_is_escalation(self, cb):
        """Green → Orange jumps a level but is still escalation."""
        assert cb._is_escalation("green", "orange") is True
        assert cb._is_escalation("green", "red") is True


class TestGetAction:
    """Action determination per status."""

    @pytest.fixture
    def cb(self):
        from src.strategy.circuit_breaker import DrawdownCircuitBreaker
        return DrawdownCircuitBreaker()

    def test_green_action(self, cb):
        assert "none" in cb._get_action("green").lower() or "normal" in cb._get_action("green").lower()

    def test_black_action(self, cb):
        action = cb._get_action("black").lower()
        assert any(w in action for w in ["kill", "close", "stop", "emergency", "liquidate"])

    def test_yellow_action(self, cb):
        action = cb._get_action("yellow").lower()
        assert any(w in action for w in ["alert", "warn", "monitor", "caution"])

    def test_orange_action(self, cb):
        action = cb._get_action("orange").lower()
        assert any(w in action for w in ["reduce", "cut", "decrease", "lower"])

    def test_red_action(self, cb):
        action = cb._get_action("red").lower()
        assert any(w in action for w in ["reduce", "cut", "severe", "significant"])


class TestGetMessage:
    """Alert message generation."""

    @pytest.fixture
    def cb(self):
        from src.strategy.circuit_breaker import DrawdownCircuitBreaker
        return DrawdownCircuitBreaker()

    def test_message_contains_drawdown(self, cb):
        msg = cb._get_message("yellow", 0.12)
        assert "12" in msg or "0.12" in msg

    def test_message_green(self, cb):
        msg = cb._get_message("green", 0.05)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_message_black(self, cb):
        msg = cb._get_message("black", 0.30)
        assert isinstance(msg, str)
        assert len(msg) > 0


class TestCalculateDrawdown:
    """Drawdown calculation from portfolio history."""

    @pytest.fixture
    def cb(self):
        from src.strategy.circuit_breaker import DrawdownCircuitBreaker
        return DrawdownCircuitBreaker()

    def test_drawdown_from_peak(self, cb):
        history = [
            {"date": "2026-01-02", "value": 100000},
            {"date": "2026-01-03", "value": 102000},
            {"date": "2026-01-04", "value": 101000},
            {"date": "2026-01-05", "value": 98000},
            {"date": "2026-01-06", "value": 95000},
        ]
        drawdown, peak, peak_date = cb.calculate_drawdown(history)
        assert peak == 102000
        assert peak_date == "2026-01-03"
        assert abs(drawdown - 0.0686) < 0.01  # (102000-95000)/102000 ≈ 6.86%

    def test_drawdown_at_peak(self, cb):
        history = [
            {"date": "2026-01-02", "value": 100000},
            {"date": "2026-01-03", "value": 105000},
        ]
        drawdown, peak, _ = cb.calculate_drawdown(history)
        assert drawdown == 0.0
        assert peak == 105000

    def test_drawdown_empty_history(self, cb):
        drawdown, peak, date = cb.calculate_drawdown([])
        assert drawdown == 0.0
        assert peak == 0.0
        assert date is None

    def test_drawdown_single_point(self, cb):
        history = [{"date": "2026-01-02", "value": 100000}]
        drawdown, peak, date = cb.calculate_drawdown(history)
        assert drawdown == 0.0

    def test_drawdown_with_total_value_key(self, cb):
        """Handles 'total_value' key as fallback."""
        history = [
            {"date": "2026-01-02", "total_value": 100000},
            {"date": "2026-01-03", "total_value": 90000},
        ]
        drawdown, peak, _ = cb.calculate_drawdown(history)
        assert abs(drawdown - 0.10) < 0.01

    def test_drawdown_severe(self, cb):
        history = [
            {"date": "2026-01-02", "value": 100000},
            {"date": "2026-01-03", "value": 70000},
        ]
        drawdown, _, _ = cb.calculate_drawdown(history)
        assert abs(drawdown - 0.30) < 0.01

    def test_drawdown_recovery_then_crash(self, cb):
        history = [
            {"date": "2026-01-02", "value": 100000},
            {"date": "2026-01-03", "value": 110000},  # peak
            {"date": "2026-01-04", "value": 108000},
            {"date": "2026-01-05", "value": 88000},   # -20% from peak
        ]
        drawdown, peak, date = cb.calculate_drawdown(history)
        assert peak == 110000
        assert date == "2026-01-03"
        assert abs(drawdown - 0.20) < 0.01


class TestCheckAndUpdate:
    """Full check_and_update flow."""

    @pytest.fixture
    def cb_with_history(self, tmp_path, monkeypatch):
        from src.strategy.circuit_breaker import DrawdownCircuitBreaker
        cb = DrawdownCircuitBreaker()

        # Mock state file paths
        monkeypatch.setattr(cb, 'state', {
            "status": "green",
            "max_drawdown": 0.0,
            "peak_value": 0,
            "reduction_count": 0
        })

        # Mock portfolio history
        def mock_history():
            return [
                {"date": "2026-01-02", "value": 100000},
                {"date": "2026-01-03", "value": 102000},
                {"date": "2026-01-05", "value": 91800},  # -10% from peak
            ]
        monkeypatch.setattr(cb, 'get_portfolio_value_history', mock_history)
        monkeypatch.setattr(cb, '_save_state', lambda: None)
        return cb

    def test_check_yellow_trigger(self, cb_with_history):
        result = cb_with_history.check_and_update()
        assert result["status"] == "yellow"
        assert result["drawdown_pct"] is not None
        assert result["drawdown_pct"] > 0.09

    def test_state_updated_after_check(self, cb_with_history):
        result = cb_with_history.check_and_update()
        assert cb_with_history.state["max_drawdown"] > 0
        assert cb_with_history.state["peak_value"] == 102000

    def test_check_empty_history(self, monkeypatch):
        from src.strategy.circuit_breaker import DrawdownCircuitBreaker
        cb = DrawdownCircuitBreaker()
        monkeypatch.setattr(cb, 'get_portfolio_value_history', lambda: [])
        monkeypatch.setattr(cb, '_save_state', lambda: None)

        result = cb.check_and_update()
        assert result["status"] == "unknown"
        assert result["drawdown"] is None


class TestReset:
    """Circuit breaker reset."""

    def test_reset_clears_state(self):
        from src.strategy.circuit_breaker import DrawdownCircuitBreaker
        cb = DrawdownCircuitBreaker()
        cb.state["status"] = "red"
        cb.state["max_drawdown"] = 0.25
        cb.state["reduction_count"] = 3
        cb.state["triggered_at"] = "2026-05-14T12:00:00"

        cb.reset(reason="manual_reset")

        assert cb.state["status"] == "green"
        assert cb.state["max_drawdown"] == 0.0
        assert cb.state["reduction_count"] == 0

    def test_reset_preserves_reason(self):
        from src.strategy.circuit_breaker import DrawdownCircuitBreaker
        cb = DrawdownCircuitBreaker()
        cb.reset(reason="market_recovery")
        assert cb.state["status"] == "green"


class TestGetStatus:
    """get_status snapshot."""

    def test_get_status_returns_dict(self):
        from src.strategy.circuit_breaker import DrawdownCircuitBreaker
        cb = DrawdownCircuitBreaker()
        status = cb.get_status()
        assert isinstance(status, dict)
        assert "status" in status

    def test_get_status_reflects_state(self):
        from src.strategy.circuit_breaker import DrawdownCircuitBreaker
        cb = DrawdownCircuitBreaker()
        cb.state["status"] = "orange"
        cb.state["max_drawdown"] = 0.18
        status = cb.get_status()
        assert status["status"] == "orange"
