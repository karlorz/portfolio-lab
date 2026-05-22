"""
Tests for src/signals/international_momentum.py — SignalType, ConfidenceLevel,
InternationalMomentumSignal dataclass, and InternationalMomentumGenerator
(with DB mocking).
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from src.signals.international_momentum import (
    SignalType,
    ConfidenceLevel,
    InternationalMomentumSignal,
    InternationalMomentumGenerator,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_signal(
    signal_type="neutral",
    confidence=0.0,
    confidence_level="low",
    efa_momentum_6m=0.0,
    eem_momentum_6m=0.0,
    spy_momentum_6m=0.0,
    efa_vs_spy=0.0,
    eem_vs_spy=0.0,
    spy_shift=0.0,
    efa_shift=0.0,
    eem_shift=0.0,
    max_allocation_efa=0.05,
    max_allocation_eem=0.03,
    holding_period_days=30,
    data_fresh=True,
    vix_filter_active=False,
    correlation_override=False,
    timestamp="2026-01-01T00:00:00",
):
    return InternationalMomentumSignal(
        timestamp=timestamp,
        signal_type=signal_type,
        confidence=confidence,
        confidence_level=confidence_level,
        efa_momentum_6m=efa_momentum_6m,
        eem_momentum_6m=eem_momentum_6m,
        spy_momentum_6m=spy_momentum_6m,
        efa_vs_spy=efa_vs_spy,
        eem_vs_spy=eem_vs_spy,
        spy_shift=spy_shift,
        efa_shift=efa_shift,
        eem_shift=eem_shift,
        max_allocation_efa=max_allocation_efa,
        max_allocation_eem=max_allocation_eem,
        holding_period_days=holding_period_days,
        data_fresh=data_fresh,
        vix_filter_active=vix_filter_active,
        correlation_override=correlation_override,
    )


def _patch_db_init():
    """Patch _init_signal_history to avoid real sqlite3 calls during __init__."""
    return patch.object(
        InternationalMomentumGenerator, "_init_signal_history", return_value=None
    )


def _patch_vix(vix_level=20.0):
    return patch.object(
        InternationalMomentumGenerator, "_get_vix_level", return_value=vix_level
    )


def _patch_correlation(corr=0.85):
    return patch.object(
        InternationalMomentumGenerator, "_get_correlation", return_value=corr
    )


def _patch_save_signal():
    return patch.object(
        InternationalMomentumGenerator, "_save_signal", return_value=None
    )


def _default_data(efa_vs_spy=0.0, eem_vs_spy=0.0, data_fresh=True):
    return {
        "timestamp": "2026-05-22T12:00:00",
        "data_fresh": data_fresh,
        "relative": {
            "efa_momentum_6m": 0.06,
            "eem_momentum_6m": 0.09,
            "spy_momentum_6m": 0.04,
            "efa_vs_spy": efa_vs_spy,
            "eem_vs_spy": eem_vs_spy,
        },
    }


# ── SignalType enum ──────────────────────────────────────────────────


class TestSignalType:
    def test_values(self):
        assert SignalType.NEUTRAL.value == "neutral"
        assert SignalType.EFA_LEAD.value == "efa_lead"
        assert SignalType.EEM_LEAD.value == "eem_lead"

    def test_members(self):
        assert len(SignalType) == 3


# ── ConfidenceLevel enum ─────────────────────────────────────────────


class TestConfidenceLevel:
    def test_values(self):
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.HIGH.value == "high"

    def test_members(self):
        assert len(ConfidenceLevel) == 3


# ── InternationalMomentumSignal dataclass ────────────────────────────


class TestInternationalMomentumSignal:
    def test_to_dict(self):
        sig = _make_signal()
        d = sig.to_dict()
        assert isinstance(d, dict)
        assert d["signal_type"] == "neutral"
        assert d["confidence"] == 0.0

    def test_to_dict_matches_asdict(self):
        sig = _make_signal(signal_type="efa_lead", confidence=0.6)
        assert sig.to_dict() == asdict(sig)

    def test_is_active_neutral_not_active(self):
        sig = _make_signal(signal_type="neutral")
        assert sig.is_active() is False

    def test_is_active_low_confidence_not_active(self):
        sig = _make_signal(signal_type="efa_lead", confidence=0.3)
        assert sig.is_active() is False

    def test_is_active_stale_data_not_active(self):
        sig = _make_signal(signal_type="efa_lead", confidence=0.6, data_fresh=False)
        assert sig.is_active() is False

    def test_is_active_vix_filter_not_active(self):
        sig = _make_signal(signal_type="efa_lead", confidence=0.6, vix_filter_active=True)
        assert sig.is_active() is False

    def test_is_active_correlation_override_not_active(self):
        sig = _make_signal(signal_type="efa_lead", confidence=0.6, correlation_override=True)
        assert sig.is_active() is False

    def test_is_active_all_conditions_met(self):
        sig = _make_signal(
            signal_type="efa_lead",
            confidence=0.6,
            data_fresh=True,
            vix_filter_active=False,
            correlation_override=False,
        )
        assert sig.is_active() is True

    def test_is_active_eem_lead(self):
        sig = _make_signal(
            signal_type="eem_lead",
            confidence=0.7,
            data_fresh=True,
        )
        assert sig.is_active() is True

    def test_is_active_confidence_boundary_050(self):
        sig = _make_signal(signal_type="efa_lead", confidence=0.5, data_fresh=True)
        assert sig.is_active() is True

    def test_is_active_confidence_just_below_050(self):
        sig = _make_signal(signal_type="efa_lead", confidence=0.49, data_fresh=True)
        assert sig.is_active() is False

    def test_get_allocation_delta_inactive(self):
        sig = _make_signal(signal_type="neutral")
        delta = sig.get_allocation_delta()
        assert delta == {"SPY": 0.0, "EFA": 0.0, "EEM": 0.0}

    def test_get_allocation_delta_active_efa(self):
        sig = _make_signal(
            signal_type="efa_lead",
            confidence=0.6,
            data_fresh=True,
            spy_shift=0.03,
            efa_shift=0.03,
            eem_shift=0.0,
        )
        delta = sig.get_allocation_delta()
        assert delta["SPY"] == -0.03
        assert delta["EFA"] == 0.03
        assert delta["EEM"] == 0.0

    def test_get_allocation_delta_active_eem(self):
        sig = _make_signal(
            signal_type="eem_lead",
            confidence=0.8,
            data_fresh=True,
            spy_shift=0.024,
            efa_shift=0.0,
            eem_shift=0.024,
        )
        delta = sig.get_allocation_delta()
        assert delta["SPY"] == -0.024
        assert delta["EFA"] == 0.0
        assert delta["EEM"] == 0.024

    def test_get_allocation_delta_vix_blocked(self):
        sig = _make_signal(
            signal_type="efa_lead",
            confidence=0.6,
            vix_filter_active=True,
            spy_shift=0.03,
            efa_shift=0.03,
        )
        delta = sig.get_allocation_delta()
        assert delta == {"SPY": 0.0, "EFA": 0.0, "EEM": 0.0}


# ── InternationalMomentumGenerator constants ─────────────────────────


class TestGeneratorConstants:
    def test_thresholds(self):
        assert InternationalMomentumGenerator.EFA_THRESHOLD == 0.05
        assert InternationalMomentumGenerator.EEM_THRESHOLD == 0.08

    def test_allocation_limits(self):
        assert InternationalMomentumGenerator.MAX_EFA_ALLOCATION == 0.05
        assert InternationalMomentumGenerator.MAX_EEM_ALLOCATION == 0.03

    def test_risk_filters(self):
        assert InternationalMomentumGenerator.VIX_CUTOFF == 30.0
        assert InternationalMomentumGenerator.CORRELATION_CUTOFF == 0.95


# ── _determine_signal_type() ─────────────────────────────────────────


class TestDetermineSignalType:
    @pytest.fixture
    def gen(self):
        with _patch_db_init():
            return InternationalMomentumGenerator()

    def test_neutral_below_thresholds(self, gen):
        sig_type, conf = gen._determine_signal_type(0.01, 0.02)
        assert sig_type == SignalType.NEUTRAL
        assert conf == 0.0

    def test_neutral_at_threshold_not_exceeded(self, gen):
        sig_type, conf = gen._determine_signal_type(0.05, 0.07)
        assert sig_type == SignalType.NEUTRAL
        assert conf == 0.0

    def test_efa_lead_just_above_threshold(self, gen):
        sig_type, conf = gen._determine_signal_type(0.06, 0.0)
        assert sig_type == SignalType.EFA_LEAD
        assert conf == pytest.approx(0.6, abs=0.01)

    def test_eem_lead_just_above_threshold(self, gen):
        sig_type, conf = gen._determine_signal_type(0.0, 0.09)
        assert sig_type == SignalType.EEM_LEAD
        assert conf == pytest.approx(0.6, abs=0.01)

    def test_efa_takes_priority_over_eem(self, gen):
        """EFA is checked first when both exceed thresholds."""
        sig_type, conf = gen._determine_signal_type(0.07, 0.12)
        assert sig_type == SignalType.EFA_LEAD

    def test_efa_confidence_scales(self, gen):
        _, conf = gen._determine_signal_type(0.051, 0.0)
        assert conf == pytest.approx(0.51, abs=0.01)
        _, conf = gen._determine_signal_type(0.10, 0.0)
        assert conf == pytest.approx(1.0, abs=0.01)

    def test_efa_confidence_capped_at_one(self, gen):
        _, conf = gen._determine_signal_type(0.15, 0.0)
        assert conf == 1.0

    def test_eem_confidence_scales(self, gen):
        _, conf = gen._determine_signal_type(0.0, 0.081)
        assert conf == pytest.approx(0.54, abs=0.01)

    def test_eem_confidence_capped_at_one(self, gen):
        _, conf = gen._determine_signal_type(0.0, 0.20)
        assert conf == 1.0


# ── _calculate_allocation_shifts() ───────────────────────────────────


class TestCalculateAllocationShifts:
    @pytest.fixture
    def gen(self):
        with _patch_db_init():
            return InternationalMomentumGenerator()

    def test_neutral_gives_zero_shifts(self, gen):
        spy, efa, eem = gen._calculate_allocation_shifts(SignalType.NEUTRAL, 0.5)
        assert spy == 0.0
        assert efa == 0.0
        assert eem == 0.0

    def test_efa_lead_full_confidence(self, gen):
        spy, efa, eem = gen._calculate_allocation_shifts(SignalType.EFA_LEAD, 1.0)
        assert spy == pytest.approx(0.05, abs=0.001)
        assert efa == pytest.approx(0.05, abs=0.001)
        assert eem == 0.0

    def test_efa_lead_half_confidence(self, gen):
        spy, efa, eem = gen._calculate_allocation_shifts(SignalType.EFA_LEAD, 0.5)
        assert spy == pytest.approx(0.025, abs=0.001)
        assert efa == pytest.approx(0.025, abs=0.001)
        assert eem == 0.0

    def test_eem_lead_full_confidence(self, gen):
        spy, efa, eem = gen._calculate_allocation_shifts(SignalType.EEM_LEAD, 1.0)
        assert spy == pytest.approx(0.03, abs=0.001)
        assert efa == 0.0
        assert eem == pytest.approx(0.03, abs=0.001)

    def test_eem_lead_half_confidence(self, gen):
        spy, efa, eem = gen._calculate_allocation_shifts(SignalType.EEM_LEAD, 0.5)
        assert spy == pytest.approx(0.015, abs=0.001)
        assert efa == 0.0
        assert eem == pytest.approx(0.015, abs=0.001)

    def test_zero_confidence_gives_zero_shifts(self, gen):
        spy, efa, eem = gen._calculate_allocation_shifts(SignalType.EFA_LEAD, 0.0)
        assert spy == 0.0
        assert efa == 0.0
        assert eem == 0.0

    def test_efa_spy_equals_efa_shift(self, gen):
        """SPY reduction equals EFA increase (paired shift)."""
        spy, efa, eem = gen._calculate_allocation_shifts(SignalType.EFA_LEAD, 0.8)
        assert spy == efa

    def test_eem_spy_equals_eem_shift(self, gen):
        """SPY reduction equals EEM increase (paired shift)."""
        spy, efa, eem = gen._calculate_allocation_shifts(SignalType.EEM_LEAD, 0.8)
        assert spy == eem


# ── generate_signal() ────────────────────────────────────────────────


class TestGenerateSignal:
    @pytest.fixture
    def gen(self):
        with _patch_db_init():
            return InternationalMomentumGenerator()

    def test_neutral_signal(self, gen):
        data = _default_data(efa_vs_spy=0.01, eem_vs_spy=0.02)
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.signal_type == "neutral"
        assert sig.confidence == 0.0

    def test_efa_lead_signal(self, gen):
        data = _default_data(efa_vs_spy=0.07, eem_vs_spy=0.0)
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.signal_type == "efa_lead"
        assert sig.confidence > 0.0
        assert sig.efa_shift > 0.0
        assert sig.spy_shift > 0.0

    def test_eem_lead_signal(self, gen):
        data = _default_data(efa_vs_spy=0.0, eem_vs_spy=0.10)
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.signal_type == "eem_lead"
        assert sig.eem_shift > 0.0
        assert sig.spy_shift > 0.0

    def test_vix_filter_activates_above_cutoff(self, gen):
        data = _default_data(efa_vs_spy=0.07)
        with _patch_vix(35.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.vix_filter_active is True
        assert sig.is_active() is False

    def test_vix_filter_not_active_below_cutoff(self, gen):
        data = _default_data(efa_vs_spy=0.07)
        with _patch_vix(25.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.vix_filter_active is False

    def test_vix_filter_at_cutoff(self, gen):
        data = _default_data(efa_vs_spy=0.07)
        with _patch_vix(30.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.vix_filter_active is False  # > 30, not >=

    def test_correlation_override_activates_above_cutoff(self, gen):
        data = _default_data(efa_vs_spy=0.07)
        with _patch_vix(20.0), _patch_correlation(0.96), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.correlation_override is True
        assert sig.is_active() is False

    def test_correlation_at_cutoff(self, gen):
        data = _default_data(efa_vs_spy=0.07)
        with _patch_vix(20.0), _patch_correlation(0.95), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.correlation_override is False  # > 0.95, not >=

    def test_confidence_level_low(self, gen):
        data = _default_data(efa_vs_spy=0.04)  # below threshold -> neutral, conf=0
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.confidence_level == "low"

    def test_confidence_level_medium(self, gen):
        data = _default_data(efa_vs_spy=0.06)  # just above threshold
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.confidence_level == "medium"

    def test_confidence_level_high(self, gen):
        data = _default_data(efa_vs_spy=0.10)  # at max confidence
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.confidence_level == "high"

    def test_timestamp_from_data(self, gen):
        data = _default_data()
        data["timestamp"] = "2026-03-15T10:30:00"
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.timestamp == "2026-03-15T10:30:00"

    def test_data_fresh_passed_through(self, gen):
        data = _default_data(efa_vs_spy=0.07, data_fresh=True)
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.data_fresh is True

    def test_data_not_fresh_inactive(self, gen):
        data = _default_data(efa_vs_spy=0.07, data_fresh=False)
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.data_fresh is False
        assert sig.is_active() is False

    def test_spy_shift_paired_with_efa(self, gen):
        data = _default_data(efa_vs_spy=0.07)
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.spy_shift == sig.efa_shift

    def test_max_allocation_constants_set(self, gen):
        data = _default_data(efa_vs_spy=0.10)
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.max_allocation_efa == 0.05
        assert sig.max_allocation_eem == 0.03
        assert sig.holding_period_days == 30

    def test_momentum_values_passed_through(self, gen):
        data = _default_data(efa_vs_spy=0.07)
        data["relative"]["efa_momentum_6m"] = 0.12
        data["relative"]["eem_momentum_6m"] = 0.15
        data["relative"]["spy_momentum_6m"] = 0.05
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.efa_momentum_6m == 0.12
        assert sig.eem_momentum_6m == 0.15
        assert sig.spy_momentum_6m == 0.05

    def test_save_signal_called(self, gen):
        data = _default_data(efa_vs_spy=0.07)
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal() as mock_save:
            sig = gen.generate_signal(data)
            mock_save.assert_called_once_with(sig)

    def test_default_timestamp_when_missing(self, gen):
        data = _default_data(efa_vs_spy=0.07)
        del data["timestamp"]
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.timestamp is not None
        assert len(sig.timestamp) > 0

    def test_missing_relative_defaults_to_zero(self, gen):
        data = {"timestamp": "2026-01-01T00:00:00", "data_fresh": True}
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.signal_type == "neutral"
        assert sig.efa_vs_spy == 0.0
        assert sig.eem_vs_spy == 0.0


# ── Confidence level boundary tests ──────────────────────────────────


class TestConfidenceLevelBoundaries:
    @pytest.fixture
    def gen(self):
        with _patch_db_init():
            return InternationalMomentumGenerator()

    def test_confidence_level_medium_boundary_050(self, gen):
        """confidence >= 0.5 and < 0.7 → medium."""
        data = _default_data(efa_vs_spy=0.05)
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        # efa_vs_spy=0.05 is NOT > 0.05 threshold, so neutral
        assert sig.confidence_level == "low"

    def test_confidence_level_medium_just_above_threshold(self, gen):
        """EFA just above threshold → low confidence → medium level."""
        data = _default_data(efa_vs_spy=0.051)
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        # 0.051/0.10 = 0.51 → medium
        assert sig.confidence_level == "medium"

    def test_confidence_level_high_at_070(self, gen):
        """confidence >= 0.7 → high."""
        data = _default_data(efa_vs_spy=0.07)
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        # 0.07/0.10 = 0.7 → high
        assert sig.confidence_level == "high"


# ── Integration: full active signal round-trip ───────────────────────


class TestActiveSignalRoundTrip:
    @pytest.fixture
    def gen(self):
        with _patch_db_init():
            return InternationalMomentumGenerator()

    def test_efa_signal_active_delta_nonzero(self, gen):
        data = _default_data(efa_vs_spy=0.08)
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.is_active()
        delta = sig.get_allocation_delta()
        assert delta["SPY"] < 0
        assert delta["EFA"] > 0
        assert delta["EEM"] == 0.0

    def test_eem_signal_active_delta_nonzero(self, gen):
        data = _default_data(eem_vs_spy=0.12)
        with _patch_vix(20.0), _patch_correlation(0.85), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert sig.is_active()
        delta = sig.get_allocation_delta()
        assert delta["SPY"] < 0
        assert delta["EFA"] == 0.0
        assert delta["EEM"] > 0.0

    def test_all_risk_filters_block(self, gen):
        """Even with strong momentum, all risk filters block activation."""
        data = _default_data(efa_vs_spy=0.10, eem_vs_spy=0.15)
        with _patch_vix(35.0), _patch_correlation(0.96), _patch_save_signal():
            sig = gen.generate_signal(data)
        assert not sig.is_active()
        assert sig.get_allocation_delta() == {"SPY": 0.0, "EFA": 0.0, "EEM": 0.0}
