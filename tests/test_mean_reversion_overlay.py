"""
Tests for VIX-Gated Mean-Reversion Overlay Strategy (v4.81)
"""

import json
import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from src.strategy.mean_reversion_overlay import (
    MeanReversionOverlay,
    MeanReversionAllocation,
    get_mean_reversion_ensemble_signals,
    ENSEMBLE_WEIGHT,
    FUNDING_SOURCE,
)


class TestMeanReversionAllocation:
    """Test MeanReversionAllocation data class."""

    def test_allocation_fields(self):
        alloc = MeanReversionAllocation(
            timestamp="2026-05-16T12:00:00Z",
            active=True,
            allocation_pct=3.0,
            entry_price=500.0,
            hold_days=2,
            trade_return_pct=0.5,
            vix_level=35.0,
            vix_regime="mean_reversion",
            rationale="ENTRY: SPY oversold",
            fund_from="GLD",
            ensemble_signal_value=-0.5,
            ensemble_weight=0.05,
            spy_3d_return=-3.0,
            spy_above_200ma=True,
            vpin_ok=True,
            entry_conditions_met=True,
        )
        assert alloc.active
        assert alloc.allocation_pct == 3.0
        assert alloc.ensemble_signal_value == -0.5
        assert alloc.ensemble_weight == 0.05
        assert alloc.fund_from == "GLD"

    def test_inactive_allocation(self):
        alloc = MeanReversionAllocation(
            timestamp="2026-05-16T12:00:00Z",
            active=False,
            allocation_pct=0.0,
            entry_price=None,
            hold_days=0,
            trade_return_pct=0.0,
            vix_level=15.0,
            vix_regime="trend_follow",
            rationale="TREND MODE",
            fund_from="GLD",
            ensemble_signal_value=0.0,
            ensemble_weight=0.05,
            spy_3d_return=0.5,
            spy_above_200ma=True,
            vpin_ok=True,
            entry_conditions_met=False,
        )
        assert not alloc.active
        assert alloc.entry_price is None
        assert alloc.ensemble_signal_value == 0.0


class TestMeanReversionOverlay:
    """Test MeanReversionOverlay strategy."""

    @pytest.fixture
    def overlay(self, tmp_path):
        """Create overlay with temporary state file."""
        state_path = tmp_path / "mean_reversion_state.json"
        with patch.object(MeanReversionOverlay, "__init__", return_value=None):
            overlay = MeanReversionOverlay.__new__(MeanReversionOverlay)
        overlay.state_path = state_path
        return overlay

    @pytest.fixture
    def mock_calculator(self):
        """Create a mock calculator with controlled signal."""
        calc = Mock()
        calc.generate_signal.return_value = Mock(
            timestamp="2026-05-16T12:00:00Z",
            vix_level=35.0,
            vix_regime="mean_reversion",
            spy_price=500.0,
            spy_3d_return=-3.0,
            spy_above_200ma=True,
            vpin_level=0.3,
            vpin_ok=True,
            entry_triggered=True,
            entry_reason="SPY oversold during elevated VIX",
            trade_state="entering",
            trade_entry_price=500.0,
            trade_hold_days=0,
            trade_return_pct=0.0,
            recommended_allocation_pct=3.0,
            allocation_rationale="ENTRY: SPY oversold. Allocate 3% from GLD.",
            signal_value=-0.5,
            signal_strength=0.6,
        )
        calc.compute_trade_state.return_value = {
            "active": False,
            "entry_date": None,
            "entry_price": None,
            "entry_vix": None,
            "hold_days": 0,
            "allocation_pct": 0.0,
        }
        return calc

    def test_get_status_entry(self, overlay, mock_calculator):
        """Test get_status returns entry state."""
        overlay.calculator = mock_calculator
        status = overlay.get_status()
        assert status["active"] is not None
        assert status["vix_regime"] == "mean_reversion"
        assert status["entry_triggered"] is True
        assert status["recommended_allocation_pct"] > 0
        assert status["signal_value"] < 0
        assert "ensemble_weight" in status
        assert status["fund_from"] == "GLD"

    def test_get_status_idle(self, overlay):
        """Test get_status returns idle state."""
        mock_calc = Mock()
        mock_calc.generate_signal.return_value = Mock(
            timestamp="2026-05-16T12:00:00Z",
            vix_level=15.0,
            vix_regime="trend_follow",
            spy_price=520.0,
            spy_3d_return=0.5,
            spy_above_200ma=True,
            vpin_level=0.3,
            vpin_ok=True,
            entry_triggered=False,
            entry_reason="",
            trade_state="idle",
            trade_entry_price=None,
            trade_hold_days=0,
            trade_return_pct=0.0,
            recommended_allocation_pct=0.0,
            allocation_rationale="TREND MODE: VIX below 20.",
            signal_value=0.0,
            signal_strength=0.0,
        )
        overlay.calculator = mock_calc
        status = overlay.get_status()
        assert status["vix_regime"] == "trend_follow"
        assert not status["entry_triggered"]
        assert status["recommended_allocation_pct"] == 0.0

    def test_get_allocation(self, overlay, mock_calculator):
        """Test get_allocation returns proper Alloc dataclass."""
        overlay.calculator = mock_calculator
        alloc = overlay.get_allocation()
        assert isinstance(alloc, MeanReversionAllocation)
        assert alloc.active
        assert alloc.allocation_pct == 3.0
        assert alloc.entry_price == 500.0
        assert alloc.ensemble_signal_value == -0.5
        assert alloc.ensemble_weight == 0.05
        assert alloc.fund_from == "GLD"

    def test_get_allocation_idle(self, overlay):
        """Test get_allocation returns inactive when no signal."""
        mock_calc = Mock()
        mock_calc.generate_signal.return_value = Mock(
            timestamp="2026-05-16T12:00:00Z",
            vix_level=15.0,
            vix_regime="trend_follow",
            spy_price=520.0,
            spy_3d_return=0.5,
            spy_above_200ma=True,
            vpin_level=0.3,
            vpin_ok=True,
            entry_triggered=False,
            entry_reason="",
            trade_state="idle",
            trade_entry_price=None,
            trade_hold_days=0,
            trade_return_pct=0.0,
            recommended_allocation_pct=0.0,
            allocation_rationale="TREND MODE",
            signal_value=0.0,
            signal_strength=0.0,
        )
        overlay.calculator = mock_calc
        alloc = overlay.get_allocation()
        assert not alloc.active
        assert alloc.allocation_pct == 0.0

    def test_get_trade_history_empty(self, tmp_path):
        """Test trade history returns empty list when no file."""
        from src.strategy.mean_reversion_overlay import TRADES_PATH
        overlay = MeanReversionOverlay.__new__(MeanReversionOverlay)
        overlay.calculator = None
        with patch("src.strategy.mean_reversion_overlay.TRADES_PATH", tmp_path / "nonexistent_trades.json"):
            overlay.TRADES_PATH = tmp_path / "nonexistent_trades.json"
            history = overlay.get_trade_history()
            assert history == []

    def test_get_trade_history_with_data(self, tmp_path):
        """Test trade history loads from file."""
        trades_path = tmp_path / "test_trades.json"
        trades = [{"entry_date": "2026-05-01", "return_pct": 2.0, "exit_reason": "recovery"}]
        with open(trades_path, "w") as f:
            json.dump(trades, f)

        with patch("src.strategy.mean_reversion_overlay.TRADES_PATH", trades_path):
            overlay = MeanReversionOverlay.__new__(MeanReversionOverlay)
            overlay.calculator = None
            history = overlay.get_trade_history()
            assert len(history) == 1
            assert history[0]["return_pct"] == 2.0

    def test_get_trade_summary_no_trades(self, overlay):
        """Test trade summary returns empty when no trades."""
        with patch.object(overlay, "get_trade_history", return_value=[]):
            summary = overlay.get_trade_summary()
            assert summary["total_trades"] == 0

    def test_get_trade_summary_with_trades(self, overlay):
        """Test trade summary computes correct metrics."""
        trades = [
            {"return_pct": 3.0, "hold_days": 5, "exit_reason": "recovery"},
            {"return_pct": -2.0, "hold_days": 3, "exit_reason": "stop_loss"},
            {"return_pct": 1.5, "hold_days": 7, "exit_reason": "recovery"},
            {"return_pct": -1.0, "hold_days": 4, "exit_reason": "vix_drop"},
            {"return_pct": 4.0, "hold_days": 6, "exit_reason": "recovery"},
        ]
        with patch.object(overlay, "get_trade_history", return_value=trades):
            summary = overlay.get_trade_summary()
        assert summary["total_trades"] == 5
        assert summary["win_rate_pct"] == 60.0  # 3/5 wins
        assert summary["avg_hold_days"] > 0
        assert summary["best_trade_pct"] == 4.0
        assert summary["worst_trade_pct"] == -2.0
        assert "recovery" in summary["exit_reasons"]

    def test_reset_state_success(self, overlay, tmp_path):
        """Test reset_state clears active trade."""
        state_path = tmp_path / "mean_reversion_state.json"
        overlay.calculator = Mock()
        overlay.calculator.save_trade_state.return_value = None
        overlay.calculator.compute_trade_state.return_value = {"active": False}
        overlay.state_path = state_path
        with patch.object(overlay, "state_path", state_path):
            result = overlay.reset_state()
            assert result is True

    def test_reset_state_failure(self, overlay):
        """Test reset_state handles errors."""
        overlay.calculator = Mock()
        overlay.calculator.save_trade_state.side_effect = Exception("IO error")
        result = overlay.reset_state()
        assert result is False

    def test_ensemble_constants(self):
        """Test ensemble integration constants."""
        assert ENSEMBLE_WEIGHT == 0.05
        assert FUNDING_SOURCE == "GLD"

    def test_ensemble_signals_inactive(self, overlay):
        """Test ensemble signal returns inactive when no signal."""
        mock_calc = Mock()
        mock_calc.generate_signal.return_value = Mock(
            timestamp="2026-05-16T12:00:00Z",
            vix_level=15.0,
            vix_regime="trend_follow",
            spy_price=520.0,
            spy_3d_return=0.5,
            spy_above_200ma=True,
            vpin_level=0.3,
            vpin_ok=True,
            entry_triggered=False,
            entry_reason="",
            trade_state="idle",
            trade_entry_price=None,
            trade_hold_days=0,
            trade_return_pct=0.0,
            recommended_allocation_pct=0.0,
            allocation_rationale="TREND MODE",
            signal_value=0.0,
            signal_strength=0.0,
        )
        with patch("src.strategy.mean_reversion_overlay.MeanReversionOverlay") as MockOverlay:
            MockOverlay.return_value.get_allocation.return_value = MeanReversionAllocation(
                timestamp="2026-05-16T12:00:00Z",
                active=False,
                allocation_pct=0.0,
                entry_price=None,
                hold_days=0,
                trade_return_pct=0.0,
                vix_level=15.0,
                vix_regime="trend_follow",
                rationale="TREND MODE",
                fund_from="GLD",
                ensemble_signal_value=0.0,
                ensemble_weight=0.05,
                spy_3d_return=0.5,
                spy_above_200ma=True,
                vpin_ok=True,
                entry_conditions_met=False,
            )
            signals = get_mean_reversion_ensemble_signals()
            assert "mean_reversion" in signals
            assert signals["mean_reversion"]["signal_value"] == 0.0
            assert signals["mean_reversion"]["active"] is False
            assert signals["mean_reversion"]["weight"] == 0.05

    def test_ensemble_signals_active(self):
        """Test ensemble signal returns active signal."""
        with patch("src.strategy.mean_reversion_overlay.MeanReversionOverlay") as MockOverlay:
            MockOverlay.return_value.get_allocation.return_value = MeanReversionAllocation(
                timestamp="2026-05-16T12:00:00Z",
                active=True,
                allocation_pct=3.0,
                entry_price=500.0,
                hold_days=1,
                trade_return_pct=0.0,
                vix_level=35.0,
                vix_regime="mean_reversion",
                rationale="ENTRY: SPY oversold",
                fund_from="GLD",
                ensemble_signal_value=-0.5,
                ensemble_weight=0.05,
                spy_3d_return=-3.0,
                spy_above_200ma=True,
                vpin_ok=True,
                entry_conditions_met=True,
            )
            signals = get_mean_reversion_ensemble_signals()
            assert signals["mean_reversion"]["active"] is True
            assert signals["mean_reversion"]["signal_value"] == -0.5
            assert signals["mean_reversion"]["allocation_pct"] == 3.0
            assert signals["mean_reversion"]["vix_regime"] == "mean_reversion"
            assert signals["mean_reversion"]["vix_level"] == 35.0


class TestOverlayCLI:
    """Test CLI entry points."""

    def test_overlay_cli_importable(self):
        """Test the module can parse CLI args."""
        from src.strategy.mean_reversion_overlay import main
        assert main is not None
        assert callable(main)
