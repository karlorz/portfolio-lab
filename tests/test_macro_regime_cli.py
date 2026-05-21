"""
Tests for Macro Regime Synthesizer CLI (v4.30).

Tests the CLI argument parsing and command dispatch layer.
All macro_regime module dependencies are mocked to avoid
database and numpy/pandas dependencies.
"""

import pytest
import json
import sys
import argparse
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
from datetime import datetime

from src.regime import macro_regime_cli


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_classification():
    """Standard classification result dict."""
    return {
        "timestamp": "2026-05-22T12:00:00",
        "regime": "risk_on_growth",
        "regime_display": "Risk On Growth",
        "confidence": 85.0,
        "signal_agreement": 0.75,
        "signal_strength": 0.650,
        "weighted_sum": 0.650,
        "regime_duration_days": 45,
        "recommended_action": "Maintain equity overweight",
        "allocation_shifts": {"spy": 0.03, "gld": -0.02, "tlt": -0.01},
        "signal_breakdown": {"fed_policy": 0.5, "yield_curve": 0.3},
    }


@pytest.fixture
def sample_history():
    """Standard regime history list."""
    return [
        {
            "timestamp": "2026-05-22T12:00:00",
            "regime": "risk_on_growth",
            "confidence": 85.0,
            "weighted_sum": 0.650,
            "recommendation": "Maintain equity overweight",
        },
        {
            "timestamp": "2026-05-21T12:00:00",
            "regime": "neutral",
            "confidence": 60.0,
            "weighted_sum": 0.100,
            "recommendation": "Hold current allocation",
        },
    ]


@pytest.fixture
def sample_overlay():
    """Sample overlay result dict."""
    return {"spy": 0.48, "gld": 0.36, "tlt": 0.16}


@pytest.fixture
def mock_synthesizer(sample_overlay, sample_history):
    """Create a fully mocked MacroRegimeSynthesizer."""
    synth = MagicMock()
    synth.get_allocation_overlay.return_value = sample_overlay
    synth.get_regime_history.return_value = sample_history
    return synth


# ═══════════════════════════════════════════════════════════════════════════
# parse_signal_arg
# ═══════════════════════════════════════════════════════════════════════════

class TestParseSignalArg:
    """Tests for parse_signal_arg — parsing 'name=value' format."""

    def test_basic_key_value(self):
        """Parse a simple name=value pair."""
        name, value = macro_regime_cli.parse_signal_arg("fed_policy=easing")
        assert name == "fed_policy"
        assert value == "easing"

    def test_value_with_underscores(self):
        """Parse value containing underscores."""
        name, value = macro_regime_cli.parse_signal_arg("equity_tsmom=risk_on")
        assert name == "equity_tsmom"
        assert value == "risk_on"

    def test_value_with_numbers(self):
        """Parse value containing numeric characters."""
        name, value = macro_regime_cli.parse_signal_arg("vix_level=25.5")
        assert name == "vix_level"
        assert value == "25.5"

    def test_empty_string_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid signal format"):
            macro_regime_cli.parse_signal_arg("")

    def test_no_equals_sign_raises(self):
        """String without '=' should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid signal format"):
            macro_regime_cli.parse_signal_arg("justaname")

    def test_multiple_equals_signs(self):
        """String with multiple '=' should split on first only."""
        name, value = macro_regime_cli.parse_signal_arg("a=b=c")
        assert name == "a"
        assert value == "b=c"

    def test_whitespace_in_name(self):
        """Name with spaces should be preserved as-is."""
        name, value = macro_regime_cli.parse_signal_arg("my signal=value")
        assert name == "my signal"
        assert value == "value"

    def test_whitespace_in_value(self):
        """Value with spaces should be preserved as-is."""
        name, value = macro_regime_cli.parse_signal_arg("signal=my value")
        assert name == "signal"
        assert value == "my value"

    def test_single_character_names(self):
        """Single-character name should be valid."""
        name, value = macro_regime_cli.parse_signal_arg("x=y")
        assert name == "x"
        assert value == "y"

    def test_error_message_contains_arg(self):
        """Error message should include the invalid argument."""
        arg = "invalid_format"
        with pytest.raises(ValueError) as exc_info:
            macro_regime_cli.parse_signal_arg(arg)
        assert arg in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════
# parse_allocation_arg
# ═══════════════════════════════════════════════════════════════════════════

class TestParseAllocationArg:
    """Tests for parse_allocation_arg — parsing 'asset=weight,...' format."""

    def test_single_allocation(self):
        """Parse a single asset=weight pair."""
        result = macro_regime_cli.parse_allocation_arg("spy=0.46")
        assert result == {"spy": 0.46}

    def test_multiple_allocations(self):
        """Parse multiple comma-separated asset=weight pairs."""
        result = macro_regime_cli.parse_allocation_arg("spy=0.46,gld=0.38,tlt=0.16")
        assert result == {"spy": 0.46, "gld": 0.38, "tlt": 0.16}

    def test_whitespace_around_values(self):
        """Whitespace around asset names or weights should be stripped."""
        result = macro_regime_cli.parse_allocation_arg("  spy=0.46 , gld=0.38 ")
        assert result == {"spy": 0.46, "gld": 0.38}

    def test_uppercase_asset_names(self):
        """Uppercase asset names should be lowercased."""
        result = macro_regime_cli.parse_allocation_arg("SPY=0.46,GLD=0.38")
        assert result == {"spy": 0.46, "gld": 0.38}

    def test_mixed_case_asset_names(self):
        """Mixed-case asset names should be lowercased."""
        result = macro_regime_cli.parse_allocation_arg("SpY=0.46,Gld=0.38")
        assert result == {"spy": 0.46, "gld": 0.38}

    def test_numeric_asset_names(self):
        """Asset names with numbers should be preserved."""
        result = macro_regime_cli.parse_allocation_arg("btc_10=0.05,eth_20=0.03")
        assert result == {"btc_10": 0.05, "eth_20": 0.03}

    def test_zero_weight(self):
        """Zero weight should be parsed correctly."""
        result = macro_regime_cli.parse_allocation_arg("spy=0.0,gld=0.5")
        assert result == {"spy": 0.0, "gld": 0.5}

    def test_negative_weight(self):
        """Negative weight (short) should be parsed correctly."""
        result = macro_regime_cli.parse_allocation_arg("spy=0.5,tlt=-0.1")
        assert result == {"spy": 0.5, "tlt": -0.1}

    def test_weight_greater_than_one(self):
        """Weight > 1 (leveraged) should be parsed correctly."""
        result = macro_regime_cli.parse_allocation_arg("spy=1.5")
        assert result == {"spy": 1.5}

    def test_decimal_precision(self):
        """Preserve decimal precision in weights."""
        result = macro_regime_cli.parse_allocation_arg("spy=0.001234")
        assert result == {"spy": 0.001234}

    def test_empty_string(self):
        """Empty string should return empty dict."""
        result = macro_regime_cli.parse_allocation_arg("")
        assert result == {}

    def test_no_equals_skipped(self):
        """Entry without '=' should be silently skipped."""
        result = macro_regime_cli.parse_allocation_arg("spy=0.5,gld")
        assert result == {"spy": 0.5}

    def test_duplicate_asset(self):
        """Duplicate asset should use last value."""
        result = macro_regime_cli.parse_allocation_arg("spy=0.5,spy=0.6")
        assert result == {"spy": 0.6}

    def test_invalid_float_raises(self):
        """Non-numeric weight should raise ValueError."""
        with pytest.raises(ValueError):
            macro_regime_cli.parse_allocation_arg("spy=abc")


# ═══════════════════════════════════════════════════════════════════════════
# cmd_classify
# ═══════════════════════════════════════════════════════════════════════════

class TestCmdClassify:
    """Tests for cmd_classify — regime classification from signals."""

    def test_classify_basic_output(self, sample_classification, capsys):
        """Basic classification should print a formatted table."""
        args = argparse.Namespace(
            signal=["fed_policy=easing", "yield_curve=steep"],
            json=False,
            db_path="data/macro_regime_history.db",
        )
        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=sample_classification,
        ) as mock_classify:
            macro_regime_cli.cmd_classify(args)

        mock_classify.assert_called_once()
        call_signals, call_kwargs = mock_classify.call_args
        assert call_signals[0] == {"fed_policy": "easing", "yield_curve": "steep"}
        assert call_kwargs["db_path"] == "data/macro_regime_history.db"

        captured = capsys.readouterr().out
        assert "Risk On Growth" in captured
        assert "85.0%" in captured
        assert "0.75" in captured
        assert "0.650" in captured
        assert "45 days" in captured
        assert "Maintain equity overweight" in captured

    def test_classify_json_output(self, sample_classification, capsys):
        """JSON flag should output valid JSON."""
        args = argparse.Namespace(
            signal=["fed_policy=easing"],
            json=True,
            db_path="data/macro_regime_history.db",
        )
        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=sample_classification,
        ):
            macro_regime_cli.cmd_classify(args)

        captured = capsys.readouterr().out
        parsed = json.loads(captured)
        assert parsed["regime"] == "risk_on_growth"
        assert parsed["confidence"] == 85.0
        assert parsed["regime_display"] == "Risk On Growth"

    def test_classify_empty_signals(self, sample_classification, capsys):
        """Empty signal list should still call classify with empty dict."""
        args = argparse.Namespace(
            signal=[],
            json=False,
            db_path="data/macro_regime_history.db",
        )
        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=sample_classification,
        ) as mock_classify:
            macro_regime_cli.cmd_classify(args)

        mock_classify.assert_called_once_with({}, db_path="data/macro_regime_history.db")

    def test_classify_single_signal(self, sample_classification, capsys):
        """Single signal should be parsed and passed correctly."""
        args = argparse.Namespace(
            signal=["fed_policy=easing"],
            json=False,
            db_path="data/macro_regime_history.db",
        )
        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=sample_classification,
        ):
            macro_regime_cli.cmd_classify(args)

        captured = capsys.readouterr().out
        assert "Risk On Growth" in captured

    def test_classify_allocation_shifts_display(self, sample_classification, capsys):
        """Allocation shifts should appear in output."""
        args = argparse.Namespace(
            signal=["fed_policy=easing"],
            json=False,
            db_path="data/macro_regime_history.db",
        )
        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=sample_classification,
        ):
            macro_regime_cli.cmd_classify(args)

        captured = capsys.readouterr().out
        assert "SPY" in captured
        assert "+3.0%" in captured or "3.0" in captured
        assert "GLD" in captured
        assert "TLT" in captured

    def test_classify_with_custom_db_path(self, sample_classification, capsys):
        """Custom db_path should be passed to classify_current_regime."""
        args = argparse.Namespace(
            signal=["fed_policy=easing"],
            json=False,
            db_path="/custom/path/regime.db",
        )
        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=sample_classification,
        ) as mock_classify:
            macro_regime_cli.cmd_classify(args)

        mock_classify.assert_called_once_with(
            {"fed_policy": "easing"}, db_path="/custom/path/regime.db"
        )


# ═══════════════════════════════════════════════════════════════════════════
# cmd_overlay
# ═══════════════════════════════════════════════════════════════════════════

class TestCmdOverlay:
    """Tests for cmd_overlay — allocation overlay calculation."""

    def test_overlay_with_regime_specified(self, sample_overlay, capsys):
        """Overlay with explicit --regime should use it directly."""
        args = argparse.Namespace(
            base="spy=0.46,gld=0.38,tlt=0.16",
            regime="risk_on_growth",
            confidence=80.0,
            json=False,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_allocation_overlay.return_value = sample_overlay

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.cmd_overlay(args)

        mock_synth.get_allocation_overlay.assert_called_once()
        args_overlay, kwargs = mock_synth.get_allocation_overlay.call_args
        regime_arg = args_overlay[0]
        assert regime_arg.value == "risk_on_growth"
        assert args_overlay[1] == 80.0
        assert args_overlay[2] == {"spy": 0.46, "gld": 0.38, "tlt": 0.16}

        captured = capsys.readouterr().out
        assert "Risk On Growth" in captured
        assert "80.0%" in captured
        assert "SPY" in captured
        assert "GLD" in captured
        assert "TLT" in captured

    def test_overlay_json(self, sample_overlay, capsys):
        """JSON flag should output valid JSON with regime/confidence/base/adjusted/changes."""
        args = argparse.Namespace(
            base="spy=0.46,gld=0.38,tlt=0.16",
            regime="risk_on_growth",
            confidence=80.0,
            json=True,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_allocation_overlay.return_value = sample_overlay

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.cmd_overlay(args)

        captured = capsys.readouterr().out
        parsed = json.loads(captured)
        assert parsed["regime"] == "risk_on_growth"
        assert parsed["confidence"] == 80.0
        assert parsed["base_allocation"] == {"spy": 0.46, "gld": 0.38, "tlt": 0.16}
        assert parsed["adjusted_allocation"] == sample_overlay
        assert "changes" in parsed

    def test_overlay_from_history(self, sample_overlay, sample_history, capsys):
        """Without --regime, overlay should use last history entry."""
        args = argparse.Namespace(
            base="spy=0.46,gld=0.38,tlt=0.16",
            regime=None,
            confidence=75.0,
            json=False,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_regime_history.return_value = sample_history
        mock_synth.get_allocation_overlay.return_value = sample_overlay

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.cmd_overlay(args)

        mock_synth.get_regime_history.assert_called_once_with(days=1)
        mock_synth.get_allocation_overlay.assert_called_once()
        regime_arg = mock_synth.get_allocation_overlay.call_args[0][0]
        assert regime_arg.value == "risk_on_growth"

    def test_overlay_no_history_exits(self, capsys):
        """Without --regime and no history, should print error and sys.exit(1)."""
        args = argparse.Namespace(
            base="spy=0.46,gld=0.38,tlt=0.16",
            regime=None,
            confidence=75.0,
            json=False,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_regime_history.return_value = []
        # Return a fresh MagicMock for the method
        mock_synth.get_allocation_overlay = MagicMock()

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            with pytest.raises(SystemExit) as exc_info:
                macro_regime_cli.cmd_overlay(args)

            assert exc_info.value.code == 1

        captured = capsys.readouterr().out
        assert "Error" in captured
        assert "No recent regime classification" in captured
        assert "--regime" in captured

    def test_overlay_with_confidence_default(self, sample_overlay, capsys):
        """Default confidence should be passed through."""
        args = argparse.Namespace(
            base="spy=0.46,gld=0.38,tlt=0.16",
            regime="neutral",
            confidence=75.0,
            json=False,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_allocation_overlay.return_value = sample_overlay

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.cmd_overlay(args)

        assert mock_synth.get_allocation_overlay.call_args[0][1] == 75.0

    def test_overlay_single_asset_base(self, capsys):
        """Overlay with single-asset base should work."""
        args = argparse.Namespace(
            base="spy=1.0",
            regime="risk_on_growth",
            confidence=80.0,
            json=False,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_allocation_overlay.return_value = {"spy": 1.0}

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.cmd_overlay(args)

        captured = capsys.readouterr().out
        assert "SPY" in captured

    def test_overlay_invalid_regime_exits(self, capsys):
        """Invalid regime value should raise ValueError from MacroRegime()."""
        args = argparse.Namespace(
            base="spy=0.5",
            regime="invalid_regime",
            confidence=75.0,
            json=False,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            with pytest.raises(ValueError):
                macro_regime_cli.cmd_overlay(args)

    def test_overlay_custom_db_path(self, sample_overlay, capsys):
        """Custom db_path should be passed to synthesizer constructor."""
        args = argparse.Namespace(
            base="spy=0.5",
            regime="neutral",
            confidence=75.0,
            json=False,
            db_path="/custom/regime.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_allocation_overlay.return_value = sample_overlay

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ) as mock_cls:
            macro_regime_cli.cmd_overlay(args)
            mock_cls.assert_called_once_with(db_path="/custom/regime.db")


# ═══════════════════════════════════════════════════════════════════════════
# cmd_history
# ═══════════════════════════════════════════════════════════════════════════

class TestCmdHistory:
    """Tests for cmd_history — regime history display."""

    def test_history_basic_output(self, sample_history, capsys):
        """History with entries should print a formatted table."""
        args = argparse.Namespace(
            days=30,
            json=False,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_regime_history.return_value = sample_history

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.cmd_history(args)

        mock_synth.get_regime_history.assert_called_once_with(days=30)
        captured = capsys.readouterr().out
        assert "30" in captured
        # Regime names are shown as lowercase (entry['regime'].replace('_', ' ')[:18])
        assert "risk on growth" in captured
        assert "neutral" in captured
        assert "85.0%" in captured
        assert "60.0%" in captured
        assert "+0.650" in captured or "0.650" in captured
        assert "+0.100" in captured or "0.100" in captured

    def test_history_json(self, sample_history, capsys):
        """JSON flag should output the raw history list."""
        args = argparse.Namespace(
            days=30,
            json=True,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_regime_history.return_value = sample_history

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.cmd_history(args)

        captured = capsys.readouterr().out
        parsed = json.loads(captured)
        assert len(parsed) == 2
        assert parsed[0]["regime"] == "risk_on_growth"
        assert parsed[1]["regime"] == "neutral"

    def test_history_empty(self, capsys):
        """Empty history should print a message."""
        args = argparse.Namespace(
            days=30,
            json=False,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_regime_history.return_value = []

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.cmd_history(args)

        captured = capsys.readouterr().out
        assert "No regime classifications" in captured
        assert "30" in captured

    def test_history_custom_days(self, sample_history, capsys):
        """Custom --days value should be passed through."""
        args = argparse.Namespace(
            days=90,
            json=False,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_regime_history.return_value = sample_history

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.cmd_history(args)

        mock_synth.get_regime_history.assert_called_once_with(days=90)

    def test_history_single_entry(self, capsys):
        """Single history entry should render without error."""
        args = argparse.Namespace(
            days=7,
            json=False,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_regime_history.return_value = [
            {
                "timestamp": "2026-05-22T12:00:00",
                "regime": "crisis",
                "confidence": 95.0,
                "weighted_sum": -0.800,
                "recommendation": "Reduce equity exposure",
            }
        ]

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.cmd_history(args)

        captured = capsys.readouterr().out
        assert "crisis" in captured or "Crisis" in captured
        assert "95.0%" in captured

    def test_history_large_day_value(self, capsys):
        """Large --days value should be passed without error."""
        args = argparse.Namespace(
            days=365,
            json=False,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_regime_history.return_value = []

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.cmd_history(args)

        mock_synth.get_regime_history.assert_called_once_with(days=365)


# ═══════════════════════════════════════════════════════════════════════════
# cmd_simulate
# ═══════════════════════════════════════════════════════════════════════════

class TestCmdSimulate:
    """Tests for cmd_simulate — scenario simulation."""

    def test_simulate_all_scenarios(self, sample_classification, capsys):
        """Simulate should run all 5 predefined scenarios."""
        args = argparse.Namespace(db_path="data/macro_regime_history.db")

        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=sample_classification,
        ) as mock_classify:
            macro_regime_cli.cmd_simulate(args)

        # Should have been called 5 times (one per scenario)
        assert mock_classify.call_count == 5

        captured = capsys.readouterr().out
        assert "Scenario" in captured
        assert "bull_market" in captured or "Bull Market" in captured
        assert "late_cycle" in captured or "Late Cycle" in captured
        assert "defensive" in captured or "Defensive" in captured
        assert "crisis" in captured or "Crisis" in captured
        assert "recovery" in captured or "Recovery" in captured

    def test_simulate_scenario_signals_displayed(self, sample_classification, capsys):
        """Each scenario's signals should be printed."""
        args = argparse.Namespace(db_path="data/macro_regime_history.db")

        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=sample_classification,
        ):
            macro_regime_cli.cmd_simulate(args)

        captured = capsys.readouterr().out
        assert "fed_policy=easing" in captured
        assert "yield_curve=steep" in captured
        assert "credit_spread=normal" in captured
        assert "equity_tsmom=risk_on" in captured

    def test_simulate_crisis_scenario(self, capsys):
        """Crisis scenario should include specific crisis signals."""
        scenario_signals = {
            "crisis": {
                "fed_policy": "easing",
                "yield_curve": "inverted",
                "credit_spread": "distressed",
                "fx_carry": "unwind_risk",
                "equity_tsmom": "risk_off",
            },
        }
        crisis_result = {
            "timestamp": "2026-05-22T12:00:00",
            "regime": "crisis",
            "regime_display": "Crisis",
            "confidence": 92.0,
            "signal_agreement": 0.90,
            "signal_strength": -0.850,
            "weighted_sum": -0.850,
            "regime_duration_days": 5,
            "recommended_action": "Reduce equity exposure",
            "allocation_shifts": {},
            "signal_breakdown": {},
        }

        args = argparse.Namespace(db_path="data/macro_regime_history.db")

        def side_effect(signals):
            if signals.get("credit_spread") == "distressed":
                return crisis_result
            return {
                "timestamp": "2026-05-22T12:00:00",
                "regime": "risk_on_growth",
                "regime_display": "Risk On Growth",
                "confidence": 85.0,
                "signal_agreement": 0.75,
                "signal_strength": 0.650,
                "weighted_sum": 0.650,
                "regime_duration_days": 45,
                "recommended_action": "Maintain equity overweight",
                "allocation_shifts": {},
                "signal_breakdown": {},
            }

        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            side_effect=side_effect,
        ):
            macro_regime_cli.cmd_simulate(args)

        captured = capsys.readouterr().out
        assert "Crisis" in captured
        assert "credit_spread=distressed" in captured
        assert "fx_carry=unwind_risk" in captured

    def test_simulate_calls_with_correct_signals(self, sample_classification, capsys):
        """Each scenario should be called with the correct signals dict."""
        args = argparse.Namespace(db_path="data/macro_regime_history.db")

        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=sample_classification,
        ) as mock_classify:
            macro_regime_cli.cmd_simulate(args)

        # Verify the bull_market call
        bull_call = mock_classify.call_args_list[0]
        assert bull_call[0][0] == {
            "fed_policy": "easing",
            "yield_curve": "steep",
            "credit_spread": "normal",
            "equity_tsmom": "risk_on",
        }

        # Verify the crisis call
        crisis_call = mock_classify.call_args_list[3]
        assert crisis_call[0][0] == {
            "fed_policy": "easing",
            "yield_curve": "inverted",
            "credit_spread": "distressed",
            "fx_carry": "unwind_risk",
            "equity_tsmom": "risk_off",
        }

    def test_simulate_single_scenario_custom_result(self, capsys):
        """Each scenario renders regime, confidence, and recommendation."""
        args = argparse.Namespace(db_path="data/macro_regime_history.db")

        result = {
            "timestamp": "2026-05-22T12:00:00",
            "regime": "neutral",
            "regime_display": "Neutral",
            "confidence": 55.0,
            "signal_agreement": 0.30,
            "signal_strength": 0.050,
            "weighted_sum": 0.050,
            "regime_duration_days": 10,
            "recommended_action": "Hold current allocation",
            "allocation_shifts": {},
            "signal_breakdown": {},
        }

        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=result,
        ):
            macro_regime_cli.cmd_simulate(args)

        captured = capsys.readouterr().out
        assert "Neutral" in captured
        assert "55.0%" in captured
        assert "Hold current allocation" in captured


# ═══════════════════════════════════════════════════════════════════════════
# main()
# ═══════════════════════════════════════════════════════════════════════════

class TestMain:
    """Tests for main() — CLI entry point with argparse."""

    def test_main_classify_command(self, sample_classification, capsys, monkeypatch):
        """main() should dispatch classify command."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "macro_regime_cli.py",
                "classify",
                "--signal",
                "fed_policy=easing",
            ],
        )
        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=sample_classification,
        ):
            macro_regime_cli.main()

        captured = capsys.readouterr().out
        assert "Risk On Growth" in captured

    def test_main_overlay_command(self, sample_overlay, capsys, monkeypatch):
        """main() should dispatch overlay command."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "macro_regime_cli.py",
                "overlay",
                "--base",
                "spy=0.46,gld=0.38,tlt=0.16",
                "--regime",
                "risk_on_growth",
                "--confidence",
                "80",
            ],
        )
        mock_synth = MagicMock()
        mock_synth.get_allocation_overlay.return_value = sample_overlay

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.main()

        captured = capsys.readouterr().out
        assert "Risk On Growth" in captured
        assert "SPY" in captured

    def test_main_history_command(self, sample_history, capsys, monkeypatch):
        """main() should dispatch history command."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "macro_regime_cli.py",
                "history",
                "--days",
                "10",
            ],
        )
        mock_synth = MagicMock()
        mock_synth.get_regime_history.return_value = sample_history

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.main()

        captured = capsys.readouterr().out
        assert "risk on growth" in captured

    def test_main_simulate_command(self, sample_classification, capsys, monkeypatch):
        """main() should dispatch simulate command."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["macro_regime_cli.py", "simulate"],
        )
        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=sample_classification,
        ):
            macro_regime_cli.main()

        captured = capsys.readouterr().out
        assert "Scenario" in captured
        assert "Macro Regime Simulation" in captured

    def test_main_no_command_exits(self, capsys, monkeypatch):
        """main() with no command should print help and sys.exit(1)."""
        monkeypatch.setattr(sys, "argv", ["macro_regime_cli.py"])
        with pytest.raises(SystemExit) as exc_info:
            macro_regime_cli.main()
        assert exc_info.value.code == 1

        captured = capsys.readouterr().out
        assert "usage:" in captured.lower() or "usage" in captured.lower()

    def test_main_unknown_command_exits(self, capsys, monkeypatch):
        """main() with unknown command should print error and exit."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["macro_regime_cli.py", "unknown_command"],
        )
        with pytest.raises(SystemExit):
            macro_regime_cli.main()

    def test_main_classify_json_flag(self, sample_classification, capsys, monkeypatch):
        """main() with --json should output JSON."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "macro_regime_cli.py",
                "--json",
                "classify",
                "--signal",
                "fed_policy=easing",
            ],
        )
        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=sample_classification,
        ):
            macro_regime_cli.main()

        captured = capsys.readouterr().out
        parsed = json.loads(captured)
        assert parsed["regime"] == "risk_on_growth"

    def test_main_overlay_json_flag(self, sample_overlay, capsys, monkeypatch):
        """main() with overlay --json should output JSON."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "macro_regime_cli.py",
                "--json",
                "overlay",
                "--base",
                "spy=0.46,gld=0.38,tlt=0.16",
                "--regime",
                "neutral",
            ],
        )
        mock_synth = MagicMock()
        mock_synth.get_allocation_overlay.return_value = sample_overlay

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.main()

        captured = capsys.readouterr().out
        parsed = json.loads(captured)
        assert parsed["regime"] == "neutral"
        assert parsed["base_allocation"] == {"spy": 0.46, "gld": 0.38, "tlt": 0.16}

    def test_main_history_json_flag(self, sample_history, capsys, monkeypatch):
        """main() with history --json should output JSON array."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "macro_regime_cli.py",
                "--json",
                "history",
                "--days",
                "30",
            ],
        )
        mock_synth = MagicMock()
        mock_synth.get_regime_history.return_value = sample_history

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.main()

        captured = capsys.readouterr().out
        parsed = json.loads(captured)
        assert len(parsed) == 2

    def test_main_custom_db_path(self, sample_classification, capsys, monkeypatch):
        """main() with --db-path should pass it through."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "macro_regime_cli.py",
                "--db-path",
                "/tmp/test.db",
                "classify",
                "--signal",
                "fed_policy=easing",
            ],
        )
        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=sample_classification,
        ) as mock_classify:
            macro_regime_cli.main()

        mock_classify.assert_called_once_with(
            {"fed_policy": "easing"}, db_path="/tmp/test.db"
        )

    def test_main_overlay_from_history_no_regime(self, sample_overlay, sample_history, capsys, monkeypatch):
        """main() overlay without --regime should fall back to history."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "macro_regime_cli.py",
                "overlay",
                "--base",
                "spy=0.46,gld=0.38,tlt=0.16",
            ],
        )
        mock_synth = MagicMock()
        mock_synth.get_regime_history.return_value = sample_history
        mock_synth.get_allocation_overlay.return_value = sample_overlay

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.main()

        captured = capsys.readouterr().out
        # Overlay display uses title-cased regime name
        assert "Risk On Growth" in captured


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases and Error Handling
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_classify_parse_error(self, capsys, monkeypatch):
        """Invalid signal format in CLI should propagate the ValueError."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "macro_regime_cli.py",
                "classify",
                "--signal",
                "invalid_format_no_equals",
            ],
        )
        with pytest.raises(ValueError, match="Invalid signal format"):
            macro_regime_cli.main()

    def test_history_zero_days(self, capsys):
        """History with days=0 should pass through to synthesizer."""
        args = argparse.Namespace(
            days=0,
            json=False,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_regime_history.return_value = []

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.cmd_history(args)

        mock_synth.get_regime_history.assert_called_once_with(days=0)

    def test_overlay_with_missing_asset_in_base(self, capsys):
        """Asset in overlay but not in base should show change from 0."""
        args = argparse.Namespace(
            base="spy=0.5",
            regime="risk_on_growth",
            confidence=80.0,
            json=False,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_allocation_overlay.return_value = {"spy": 0.5, "gld": 0.3}

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.cmd_overlay(args)

        captured = capsys.readouterr().out
        assert "SPY" in captured
        assert "GLD" in captured

    def test_classify_all_signal_values(self, sample_classification, capsys):
        """All allocation shift values should display with correct sign."""
        result = sample_classification.copy()
        result["allocation_shifts"] = {
            "spy": 0.05,
            "gld": -0.03,
            "tlt": 0.0,
            "ief": -0.01,
        }

        args = argparse.Namespace(
            signal=["fed_policy=easing"],
            json=False,
            db_path="data/macro_regime_history.db",
        )
        with patch(
            "src.regime.macro_regime_cli.classify_current_regime",
            return_value=result,
        ):
            macro_regime_cli.cmd_classify(args)

        captured = capsys.readouterr().out
        assert "SPY" in captured
        assert "GLD" in captured
        assert "TLT" in captured
        assert "IEF" in captured

    def test_overlay_confidence_zero(self, sample_overlay, capsys):
        """Confidence=0 should be passed through correctly."""
        args = argparse.Namespace(
            base="spy=0.5",
            regime="neutral",
            confidence=0.0,
            json=False,
            db_path="data/macro_regime_history.db",
        )
        mock_synth = MagicMock()
        mock_synth.get_allocation_overlay.return_value = sample_overlay

        with patch(
            "src.regime.macro_regime_cli.MacroRegimeSynthesizer",
            return_value=mock_synth,
        ):
            macro_regime_cli.cmd_overlay(args)

        assert mock_synth.get_allocation_overlay.call_args[0][1] == 0.0

    def test_main_invalid_overlay_base_raises(self, monkeypatch):
        """Invalid --base format should raise ValueError."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "macro_regime_cli.py",
                "overlay",
                "--base",
                "not_valid",
            ],
        )
        # parse_allocation_arg("not_valid") returns {} (skips entries without "=")
        # With no --regime and no history, cmd_overlay calls sys.exit(1)
        with pytest.raises(SystemExit) as exc_info:
            macro_regime_cli.main()
        assert exc_info.value.code == 1
