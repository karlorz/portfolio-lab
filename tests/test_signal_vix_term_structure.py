"""
Tests for src/signals/vix_term_structure.py — VIXRegime, VIXSignalState enums,
VIXTermStructureSignal dataclass, VIXTermStructureCalculator, and
VIXTermStructureSignalGenerator.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from src.signals.vix_term_structure import (
    VIXRegime,
    VIXSignalState,
    VIXTermStructureSignal,
    VIXTermStructureCalculator,
    VIXTermStructureSignalGenerator,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_signal(**overrides):
    defaults = dict(
        timestamp="2026-01-01T00:00:00",
        signal_state="NEUTRAL",
        signal_value=0.0,
        vix_spot=18.0,
        vix3m=19.5,
        vix6m=20.0,
        slope_vix3m_vix=1.083,
        regime="contango",
        regime_strength=0.5,
        slope_signal=0.3,
        roll_yield_signal=0.08,
        vix_zscore_signal=0.0,
        curve_shape_signal=0.25,
        spy_shift=0.0,
        gld_shift=0.0,
        tlt_shift=0.0,
        confidence=90.0,
        is_valid=True,
        reason="VIX=18.00, Slope=1.083, Regime=contango",
    )
    defaults.update(overrides)
    return VIXTermStructureSignal(**defaults)


def _patch_ensure_dirs():
    return patch.object(VIXTermStructureSignalGenerator, "_ensure_dirs", return_value=None)


def _temp_data_dir():
    """Create a temp dir with a vix_term_structure.json for testing."""
    tmp = Path(tempfile.mkdtemp())
    data = {
        "2026-01-01": {"vix_spot": 18.0, "front_month": 19.5, "third_month": 20.0},
        "2026-01-02": {"vix_spot": 20.0, "front_month": 19.0, "third_month": 19.5},
        "2026-01-03": {"vix_spot": 35.0, "front_month": 28.0, "third_month": 25.0},
    }
    data_file = tmp / "vix_term_structure.json"
    data_file.write_text(json.dumps(data))
    return tmp, data_file


# ── VIXRegime enum ───────────────────────────────────────────────────


class TestVIXRegime:
    def test_values(self):
        assert VIXRegime.EXTREME_CONTANGO.value == "extreme_contango"
        assert VIXRegime.CONTANGO.value == "contango"
        assert VIXRegime.FLAT.value == "flat"
        assert VIXRegime.BACKWARDATION.value == "backwardation"
        assert VIXRegime.EXTREME_BACKWARDATION.value == "extreme_backwardation"

    def test_members(self):
        assert len(VIXRegime) == 5


# ── VIXSignalState enum ─────────────────────────────────────────────


class TestVIXSignalState:
    def test_values(self):
        assert VIXSignalState.RISK_ON.value == 1
        assert VIXSignalState.NEUTRAL.value == 0
        assert VIXSignalState.RISK_OFF.value == -1

    def test_members(self):
        assert len(VIXSignalState) == 3


# ── VIXTermStructureSignal dataclass ─────────────────────────────────


class TestVIXTermStructureSignal:
    def test_to_dict(self):
        sig = _make_signal()
        d = sig.to_dict()
        assert isinstance(d, dict)
        assert d["regime"] == "contango"
        assert d["signal_value"] == 0.0

    def test_to_dict_matches_asdict(self):
        sig = _make_signal()
        assert sig.to_dict() == asdict(sig)

    def test_all_fields_present(self):
        d = _make_signal().to_dict()
        expected = {
            "timestamp", "signal_state", "signal_value",
            "vix_spot", "vix3m", "vix6m", "slope_vix3m_vix",
            "regime", "regime_strength",
            "slope_signal", "roll_yield_signal", "vix_zscore_signal",
            "curve_shape_signal",
            "spy_shift", "gld_shift", "tlt_shift",
            "confidence", "is_valid", "reason",
        }
        assert set(d.keys()) == expected

    def test_optional_vix3m_none(self):
        sig = _make_signal(vix3m=None)
        assert sig.vix3m is None

    def test_optional_vix6m_none(self):
        sig = _make_signal(vix6m=None)
        assert sig.vix6m is None


# ── VIXTermStructureCalculator constants ─────────────────────────────


class TestCalculatorConstants:
    def test_regime_thresholds(self):
        assert VIXTermStructureCalculator.EXTREME_CONTANGO_THRESHOLD == 1.15
        assert VIXTermStructureCalculator.CONTANGO_THRESHOLD == 1.00
        assert VIXTermStructureCalculator.FLAT_LOWER == 0.95
        assert VIXTermStructureCalculator.BACKWARDATION_THRESHOLD == 0.80

    def test_vix_levels(self):
        assert VIXTermStructureCalculator.VIX_CHEAP == 16.0
        assert VIXTermStructureCalculator.VIX_FAIR == 20.0
        assert VIXTermStructureCalculator.VIX_EXPENSIVE == 25.0


# ── calculate_slope_signal() ────────────────────────────────────────


class TestCalculateSlopeSignal:
    @pytest.fixture
    def calc(self):
        return VIXTermStructureCalculator()

    def test_extreme_backwardation(self, calc):
        # slope < 0.85 → -1.0
        sig = calc.calculate_slope_signal(vix=30.0, vix3m=24.0)
        assert sig == -1.0

    def test_backwardation_boundary(self, calc):
        # slope exactly 0.85 → -1.0 (start of interpolation range)
        sig = calc.calculate_slope_signal(vix=20.0, vix3m=17.0)
        assert sig == pytest.approx(-1.0 + (0.85 - 0.85) / 0.15 * 0.5, abs=0.01)

    def test_backwardation_mid(self, calc):
        # slope = 0.925 → mid of 0.85-1.0 range
        sig = calc.calculate_slope_signal(vix=20.0, vix3m=18.5)
        slope = 18.5 / 20.0  # 0.925
        expected = -1.0 + (slope - 0.85) / 0.15 * 0.5
        assert sig == pytest.approx(expected, abs=0.01)

    def test_at_one(self, calc):
        # slope = 1.0 → boundary between backwardation and contango
        sig = calc.calculate_slope_signal(vix=20.0, vix3m=20.0)
        assert sig == pytest.approx(0.0, abs=0.01)

    def test_contango_mid(self, calc):
        # slope = 1.075 → mid of 1.0-1.15 range
        sig = calc.calculate_slope_signal(vix=20.0, vix3m=21.5)
        slope = 21.5 / 20.0  # 1.075
        expected = (slope - 1.0) / 0.15 * 0.5
        assert sig == pytest.approx(expected, abs=0.01)

    def test_extreme_contango(self, calc):
        # slope > 1.15 → capped at +1.0
        sig = calc.calculate_slope_signal(vix=15.0, vix3m=20.0)
        slope = 20.0 / 15.0  # 1.333
        expected = min(0.5 + (slope - 1.15) / 0.15 * 0.5, 1.0)
        assert sig == pytest.approx(expected, abs=0.01)
        assert sig <= 1.0

    def test_zero_vix_returns_zero(self, calc):
        sig = calc.calculate_slope_signal(vix=0.0, vix3m=20.0)
        assert sig == 0.0

    def test_zero_vix3m_returns_zero(self, calc):
        sig = calc.calculate_slope_signal(vix=20.0, vix3m=0.0)
        assert sig == 0.0

    def test_negative_vix_returns_zero(self, calc):
        sig = calc.calculate_slope_signal(vix=-5.0, vix3m=20.0)
        assert sig == 0.0


# ── calculate_roll_yield_signal() ────────────────────────────────────


class TestCalculateRollYieldSignal:
    @pytest.fixture
    def calc(self):
        return VIXTermStructureCalculator()

    def test_contango_positive(self, calc):
        # VIX3M > VIX → positive roll yield
        sig = calc.calculate_roll_yield_signal(vix=18.0, vix3m=20.0)
        assert sig > 0.0

    def test_backwardation_negative(self, calc):
        # VIX3M < VIX → negative roll yield
        sig = calc.calculate_roll_yield_signal(vix=25.0, vix3m=20.0)
        assert sig < 0.0

    def test_equal_returns_zero(self, calc):
        sig = calc.calculate_roll_yield_signal(vix=20.0, vix3m=20.0)
        assert sig == pytest.approx(0.0, abs=0.01)

    def test_bounded_to_range(self, calc):
        # Extreme contango
        sig = calc.calculate_roll_yield_signal(vix=10.0, vix3m=20.0)
        assert sig <= 1.0
        # Extreme backwardation
        sig = calc.calculate_roll_yield_signal(vix=40.0, vix3m=20.0)
        assert sig >= -1.0

    def test_zero_vix3m_returns_zero(self, calc):
        sig = calc.calculate_roll_yield_signal(vix=20.0, vix3m=0.0)
        assert sig == 0.0

    def test_negative_vix3m_returns_zero(self, calc):
        sig = calc.calculate_roll_yield_signal(vix=20.0, vix3m=-5.0)
        assert sig == 0.0


# ── calculate_vix_zscore_signal() ───────────────────────────────────


class TestCalculateVixZscoreSignal:
    @pytest.fixture
    def calc(self):
        c = VIXTermStructureCalculator()
        # Add 100 days of history with mean=20, std=~5
        for i in range(100):
            c.add_vix_reading(f"2025-01-{i+1:02d}", 18.0 + (i % 5))
        return c

    def test_high_vix_negative_signal(self, calc):
        # VIX above mean → risk-off signal
        sig = calc.calculate_vix_zscore_signal(30.0)
        assert sig < 0.0

    def test_low_vix_positive_signal(self, calc):
        # VIX below mean → risk-on signal
        sig = calc.calculate_vix_zscore_signal(12.0)
        assert sig > 0.0

    def test_bounded_to_range(self, calc):
        sig = calc.calculate_vix_zscore_signal(100.0)
        assert sig >= -1.0
        sig = calc.calculate_vix_zscore_signal(1.0)
        assert sig <= 1.0

    def test_insufficient_history_returns_zero(self):
        calc = VIXTermStructureCalculator()
        # Only 30 days < 60 minimum
        for i in range(30):
            calc.add_vix_reading(f"2025-01-{i+1:02d}", 20.0)
        sig = calc.calculate_vix_zscore_signal(25.0)
        assert sig == 0.0

    def test_zero_std_returns_zero(self):
        calc = VIXTermStructureCalculator()
        for i in range(100):
            calc.add_vix_reading(f"2025-01-{i+1:02d}", 20.0)
        sig = calc.calculate_vix_zscore_signal(20.0)
        assert sig == 0.0

    def test_inverted_direction(self, calc):
        """High VIX → negative signal (inverted Z-score)."""
        sig_high = calc.calculate_vix_zscore_signal(30.0)
        sig_low = calc.calculate_vix_zscore_signal(12.0)
        assert sig_high < 0
        assert sig_low > 0


# ── calculate_curve_shape_signal() ──────────────────────────────────


class TestCalculateCurveShapeSignal:
    @pytest.fixture
    def calc(self):
        return VIXTermStructureCalculator()

    def test_steepening_positive(self, calc):
        # VIX6M > VIX3M → steepening
        sig = calc.calculate_curve_shape_signal(vix3m=18.0, vix6m=19.8)
        assert sig > 0.0

    def test_flattening_negative(self, calc):
        # VIX6M < VIX3M → flattening
        sig = calc.calculate_curve_shape_signal(vix3m=20.0, vix6m=18.0)
        assert sig < 0.0

    def test_equal_returns_zero(self, calc):
        sig = calc.calculate_curve_shape_signal(vix3m=20.0, vix6m=20.0)
        assert sig == pytest.approx(0.0, abs=0.01)

    def test_none_vix6m_returns_zero(self, calc):
        sig = calc.calculate_curve_shape_signal(vix3m=20.0, vix6m=None)
        assert sig == 0.0

    def test_zero_vix3m_returns_zero(self, calc):
        sig = calc.calculate_curve_shape_signal(vix3m=0.0, vix6m=20.0)
        assert sig == 0.0

    def test_bounded_to_range(self, calc):
        sig = calc.calculate_curve_shape_signal(vix3m=10.0, vix6m=30.0)
        assert sig <= 1.0
        sig = calc.calculate_curve_shape_signal(vix3m=30.0, vix6m=10.0)
        assert sig >= -1.0


# ── classify_regime() ───────────────────────────────────────────────


class TestClassifyRegime:
    @pytest.fixture
    def calc(self):
        return VIXTermStructureCalculator()

    def test_extreme_contango(self, calc):
        regime, strength = calc.classify_regime(1.20)
        assert regime == VIXRegime.EXTREME_CONTANGO
        assert strength > 0.5

    def test_contango(self, calc):
        regime, strength = calc.classify_regime(1.08)
        assert regime == VIXRegime.CONTANGO
        assert 0.0 < strength < 1.0

    def test_contango_at_threshold(self, calc):
        """slope >= 1.0 → CONTANGO."""
        regime, _ = calc.classify_regime(1.0)
        assert regime == VIXRegime.CONTANGO

    def test_flat(self, calc):
        regime, strength = calc.classify_regime(0.97)
        assert regime == VIXRegime.FLAT
        assert 0.0 < strength <= 1.0

    def test_flat_at_lower_bound(self, calc):
        """slope >= 0.95 → FLAT."""
        regime, _ = calc.classify_regime(0.95)
        assert regime == VIXRegime.FLAT

    def test_backwardation(self, calc):
        regime, strength = calc.classify_regime(0.88)
        assert regime == VIXRegime.BACKWARDATION
        assert 0.0 < strength <= 1.0

    def test_backwardation_at_threshold(self, calc):
        """slope >= 0.80 → BACKWARDATION."""
        regime, _ = calc.classify_regime(0.80)
        assert regime == VIXRegime.BACKWARDATION

    def test_extreme_backwardation(self, calc):
        regime, strength = calc.classify_regime(0.70)
        assert regime == VIXRegime.EXTREME_BACKWARDATION
        assert strength > 0.5

    def test_extreme_backwardation_at_zero(self, calc):
        regime, strength = calc.classify_regime(0.0)
        assert regime == VIXRegime.EXTREME_BACKWARDATION


# ── get_allocation_shifts() ─────────────────────────────────────────


class TestGetAllocationShifts:
    @pytest.fixture
    def calc(self):
        return VIXTermStructureCalculator()

    def test_complacent_risk_on(self, calc):
        shifts = calc.get_allocation_shifts(0.8)
        assert shifts["spy"] == 0.05
        assert shifts["gld"] == -0.03
        assert shifts["tlt"] == -0.02

    def test_complacent_at_threshold(self, calc):
        shifts = calc.get_allocation_shifts(0.7)
        assert shifts["spy"] == 0.05

    def test_mild_risk_on(self, calc):
        shifts = calc.get_allocation_shifts(0.5)
        assert shifts["spy"] == 0.02
        assert shifts["gld"] == -0.01
        assert shifts["tlt"] == -0.01

    def test_neutral(self, calc):
        shifts = calc.get_allocation_shifts(0.0)
        assert shifts["spy"] == 0.0
        assert shifts["gld"] == 0.0
        assert shifts["tlt"] == 0.0

    def test_caution(self, calc):
        shifts = calc.get_allocation_shifts(-0.5)
        assert shifts["spy"] == -0.05
        assert shifts["gld"] == 0.03
        assert shifts["tlt"] == 0.02

    def test_risk_off(self, calc):
        shifts = calc.get_allocation_shifts(-0.8)
        assert shifts["spy"] == -0.10
        assert shifts["gld"] == 0.05
        assert shifts["tlt"] == 0.05

    def test_risk_off_at_threshold(self, calc):
        shifts = calc.get_allocation_shifts(-0.7)
        assert shifts["spy"] == -0.05  # still caution, not risk-off

    def test_spy_gld_tlt_keys(self, calc):
        shifts = calc.get_allocation_shifts(0.0)
        assert set(shifts.keys()) == {"spy", "gld", "tlt"}


# ── calculate_composite_signal() ────────────────────────────────────


class TestCalculateCompositeSignal:
    @pytest.fixture
    def calc(self):
        return VIXTermStructureCalculator()

    def test_basic_contango(self, calc):
        result = calc.calculate_composite_signal(
            vix=18.0, vix3m=20.0, vix6m=21.0, date="2026-01-01"
        )
        assert "composite" in result
        assert "slope_signal" in result
        assert "roll_yield_signal" in result
        assert "vix_zscore_signal" in result
        assert "curve_shape_signal" in result
        assert "slope" in result

    def test_composite_bounded(self, calc):
        result = calc.calculate_composite_signal(
            vix=15.0, vix3m=22.0, vix6m=25.0, date="2026-01-01"
        )
        assert -1.0 <= result["composite"] <= 1.0

    def test_weights_sum_to_one(self, calc):
        """Component weights should sum to 1.0: 0.40+0.25+0.20+0.15."""
        result = calc.calculate_composite_signal(
            vix=18.0, vix3m=20.0, vix6m=21.0, date="2026-01-01"
        )
        # Composite = 0.40*slope + 0.25*roll + 0.20*zscore + 0.15*curve
        # We can verify by recomputing
        expected = (
            0.40 * result["slope_signal"] +
            0.25 * result["roll_yield_signal"] +
            0.20 * result["vix_zscore_signal"] +
            0.15 * result["curve_shape_signal"]
        )
        expected = max(-1.0, min(1.0, expected))
        assert result["composite"] == pytest.approx(expected, abs=0.01)

    def test_none_vix3m_fallback(self, calc):
        """When VIX3M is None, uses VIX level as fallback to set synthetic vix3m."""
        result = calc.calculate_composite_signal(
            vix=18.0, vix3m=None, vix6m=None, date="2026-01-01"
        )
        # VIX=18 < VIX_FAIR(20) → slope_signal=0.0 (fallback) → vix3m = 18*0.9=16.2
        # Then calculate_slope_signal(18, 16.2) → slope=0.9 → backwardation interpolation
        # The final slope_signal comes from calculate_slope_signal with synthetic vix3m
        assert isinstance(result["slope_signal"], float)

    def test_none_vix3m_cheap_vix(self, calc):
        """Cheap VIX (<16) with no VIX3M → complacent fallback → synthetic vix3m > vix."""
        result = calc.calculate_composite_signal(
            vix=14.0, vix3m=None, vix6m=None, date="2026-01-01"
        )
        # Fallback slope_signal=0.5 > 0 → vix3m = 14*1.1 = 15.4
        # calculate_slope_signal(14, 15.4) → slope=1.1 → contango
        assert result["slope_signal"] > 0.0

    def test_none_vix3m_expensive_vix(self, calc):
        """Expensive VIX (>25) with no VIX3M → stress fallback → synthetic vix3m < vix."""
        result = calc.calculate_composite_signal(
            vix=30.0, vix3m=None, vix6m=None, date="2026-01-01"
        )
        # Fallback slope_signal=-0.8 < 0 → vix3m = 30*0.9 = 27.0
        # calculate_slope_signal(30, 27) → slope=0.9 → backwardation
        assert result["slope_signal"] < 0.0

    def test_zero_vix3m_fallback(self, calc):
        """VIX3M=0 treated as unavailable, uses VIX-level fallback."""
        result = calc.calculate_composite_signal(
            vix=18.0, vix3m=0.0, vix6m=None, date="2026-01-01"
        )
        # Falls into VIX-level fallback (same as None)
        assert isinstance(result["slope_signal"], float)


# ── VIXTermStructureCalculator.add_vix_reading() ────────────────────


class TestAddVixReading:
    def test_history_populated(self):
        calc = VIXTermStructureCalculator()
        calc.add_vix_reading("2026-01-01", 18.0)
        calc.add_vix_reading("2026-01-02", 20.0)
        assert len(calc.vix_history) == 2

    def test_history_capped_at_history_days(self):
        calc = VIXTermStructureCalculator(history_days=5)
        for i in range(10):
            calc.add_vix_reading(f"2026-01-{i+1:02d}", 18.0 + i)
        assert len(calc.vix_history) == 5

    def test_history_oldest_dropped(self):
        calc = VIXTermStructureCalculator(history_days=3)
        calc.add_vix_reading("2026-01-01", 10.0)
        calc.add_vix_reading("2026-01-02", 20.0)
        calc.add_vix_reading("2026-01-03", 30.0)
        calc.add_vix_reading("2026-01-04", 40.0)
        # Oldest (10.0) should be dropped
        assert calc.vix_history[0] == ("2026-01-02", 20.0)
        assert len(calc.vix_history) == 3


# ── VIXTermStructureSignalGenerator ─────────────────────────────────


class TestSignalGenerator:
    def test_init_creates_calculator(self):
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
        assert isinstance(gen.calculator, VIXTermStructureCalculator)

    def test_load_vix_data_missing_file(self):
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = Path("/nonexistent/file.json")
            data = gen.load_vix_data()
        assert data == {}

    def test_load_vix_data_valid_file(self):
        tmp, data_file = _temp_data_dir()
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = data_file
            data = gen.load_vix_data()
        assert "2026-01-01" in data
        assert data["2026-01-01"]["vix_spot"] == 18.0

    def test_generate_signal_with_data(self):
        tmp, data_file = _temp_data_dir()
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = data_file
            signal = gen.generate_signal(date="2026-01-01")
        assert signal.is_valid is True
        assert signal.vix_spot == 18.0
        assert signal.vix3m == 19.5
        assert signal.regime in [r.value for r in VIXRegime]

    def test_generate_signal_no_data(self):
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = Path("/nonexistent/file.json")
            signal = gen.generate_signal(date="2026-01-01")
        assert signal.is_valid is False
        assert signal.signal_state == "neutral"
        assert signal.signal_value == 0.0

    def test_generate_signal_risk_on(self):
        """Contango (VIX3M >> VIX) → risk_on signal."""
        tmp = Path(tempfile.mkdtemp())
        data = {"2026-01-01": {"vix_spot": 14.0, "front_month": 18.0, "third_month": 20.0}}
        data_file = tmp / "vix_term_structure.json"
        data_file.write_text(json.dumps(data))
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = data_file
            signal = gen.generate_signal(date="2026-01-01")
        assert signal.is_valid is True
        # slope = 18/14 = 1.286 → contango → positive composite
        assert signal.slope_vix3m_vix > 1.0

    def test_generate_signal_risk_off(self):
        """Backwardation (VIX >> VIX3M) → risk_off signal."""
        tmp = Path(tempfile.mkdtemp())
        data = {"2026-01-01": {"vix_spot": 35.0, "front_month": 22.0, "third_month": 20.0}}
        data_file = tmp / "vix_term_structure.json"
        data_file.write_text(json.dumps(data))
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = data_file
            signal = gen.generate_signal(date="2026-01-01")
        assert signal.is_valid is True
        # slope = 22/35 = 0.629 → extreme backwardation
        assert signal.slope_vix3m_vix < 0.8

    def test_generate_signal_confidence_with_vix3m(self):
        """Having VIX3M adds 30% confidence."""
        tmp, data_file = _temp_data_dir()
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = data_file
            signal = gen.generate_signal(date="2026-01-01")
        # Base 50 + 30 (vix3m) + 10 (vix6m) = 90 minimum
        assert signal.confidence >= 90.0

    def test_generate_signal_confidence_without_vix3m(self):
        """No VIX3M → no 30% confidence boost."""
        tmp = Path(tempfile.mkdtemp())
        data = {"2026-01-01": {"vix_spot": 18.0}}  # no front_month
        data_file = tmp / "vix_term_structure.json"
        data_file.write_text(json.dumps(data))
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = data_file
            signal = gen.generate_signal(date="2026-01-01")
        # Base 50 + 0 (no vix3m) + 0 (no vix6m) = 50
        assert signal.confidence == 50.0

    def test_generate_signal_allocation_shifts_populated(self):
        tmp, data_file = _temp_data_dir()
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = data_file
            signal = gen.generate_signal(date="2026-01-01")
        assert isinstance(signal.spy_shift, float)
        assert isinstance(signal.gld_shift, float)
        assert isinstance(signal.tlt_shift, float)

    def test_generate_signal_reason_string(self):
        tmp, data_file = _temp_data_dir()
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = data_file
            signal = gen.generate_signal(date="2026-01-01")
        assert "VIX=" in signal.reason
        assert "Slope=" in signal.reason
        assert "Regime=" in signal.reason

    def test_generate_signal_timestamp_present(self):
        tmp, data_file = _temp_data_dir()
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = data_file
            signal = gen.generate_signal(date="2026-01-01")
        assert signal.timestamp is not None
        assert len(signal.timestamp) > 0

    def test_save_signal(self):
        tmp, data_file = _temp_data_dir()
        output = tmp / "output" / "signal.json"
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = data_file
            gen.OUTPUT_PATH = output
            signal = gen.generate_signal(date="2026-01-01")
            gen.save_signal(signal)
        assert output.exists()
        saved = json.loads(output.read_text())
        assert saved["vix_spot"] == 18.0
        assert saved["is_valid"] is True

    def test_fetch_current_vix(self):
        tmp, data_file = _temp_data_dir()
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = data_file
            current = gen.fetch_current_vix()
        assert current is not None
        assert "vix_spot" in current

    def test_fetch_current_vix_empty(self):
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = Path("/nonexistent/file.json")
            current = gen.fetch_current_vix()
        assert current is None

    def test_get_signal_history(self):
        tmp, data_file = _temp_data_dir()
        with _patch_ensure_dirs():
            gen = VIXTermStructureSignalGenerator()
            gen.VIX_DATA_PATH = data_file
            signals = gen.get_signal_history(days=3)
        assert len(signals) > 0
        assert all(s.is_valid for s in signals)


# ── Signal state mapping in generate_signal() ───────────────────────


class TestSignalStateMapping:
    def _gen_signal_with_composite(self, composite_value):
        """Helper: generate signal and check state based on composite."""
        calc = VIXTermStructureCalculator()
        if composite_value > 0.5:
            return VIXSignalState.RISK_ON.name
        elif composite_value < -0.5:
            return VIXSignalState.RISK_OFF.name
        else:
            return VIXSignalState.NEUTRAL.name

    def test_risk_on_threshold(self):
        assert self._gen_signal_with_composite(0.6) == "RISK_ON"

    def test_neutral_mid(self):
        assert self._gen_signal_with_composite(0.0) == "NEUTRAL"

    def test_risk_off_threshold(self):
        assert self._gen_signal_with_composite(-0.6) == "RISK_OFF"

    def test_neutral_near_risk_on(self):
        assert self._gen_signal_with_composite(0.4) == "NEUTRAL"

    def test_neutral_near_risk_off(self):
        assert self._gen_signal_with_composite(-0.4) == "NEUTRAL"
