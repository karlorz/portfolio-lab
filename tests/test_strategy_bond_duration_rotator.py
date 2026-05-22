"""
Tests for src/strategy/bond_duration_rotator.py — RotationStatus,
BondRotationDecision dataclass, and BondDurationRotator.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from src.strategy.bond_duration_rotator import (
    RotationStatus,
    BondRotationDecision,
    BondDurationRotator,
    calculate_bond_rotation,
    get_bond_duration_summary,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_decision(**overrides):
    defaults = dict(
        timestamp="2026-01-01T00:00:00",
        status="active",
        tlt_total=0.064,
        ief_total=0.064,
        shy_total=0.032,
        tlt_sleeve=0.40,
        ief_sleeve=0.40,
        shy_sleeve=0.20,
        spread_10y2y=0.50,
        curve_regime="normal",
        rate_direction="stable",
        real_rate=2.0,
        effective_duration=8.4,
        confidence=70.0,
        recommendation="Position=intermediate, TLT=40% IEF=40% SHY=20%",
        is_actionable=True,
    )
    defaults.update(overrides)
    return BondRotationDecision(**defaults)


def _patch_signal_gen_ensure_dirs():
    """Patch _ensure_dirs on the nested BondDurationSignalGenerator."""
    return patch(
        "src.signals.bond_duration_signal.BondDurationSignalGenerator._ensure_dirs",
        return_value=None,
    )


def _temp_state_file():
    """Return a temp file path for state isolation."""
    fd, path = tempfile.mkstemp(suffix=".json")
    Path(path).unlink()  # Delete so _load_state gets defaults
    return Path(path)


# ── RotationStatus enum ─────────────────────────────────────────────


class TestRotationStatus:
    def test_values(self):
        assert RotationStatus.ACTIVE.value == "active"
        assert RotationStatus.DEFENSIVE.value == "defensive"
        assert RotationStatus.DISABLED.value == "disabled"

    def test_members(self):
        assert len(RotationStatus) == 3


# ── BondRotationDecision dataclass ──────────────────────────────────


class TestBondRotationDecision:
    def test_to_dict(self):
        d = _make_decision().to_dict()
        assert isinstance(d, dict)
        assert d["status"] == "active"
        assert d["is_actionable"] is True

    def test_to_dict_matches_asdict(self):
        dec = _make_decision()
        assert dec.to_dict() == asdict(dec)

    def test_all_fields_present(self):
        d = _make_decision().to_dict()
        expected = {
            "timestamp", "status", "tlt_total", "ief_total", "shy_total",
            "tlt_sleeve", "ief_sleeve", "shy_sleeve", "spread_10y2y",
            "curve_regime", "rate_direction", "real_rate",
            "effective_duration", "confidence", "recommendation",
            "is_actionable",
        }
        assert set(d.keys()) == expected


# ── BondDurationRotator constants ───────────────────────────────────


class TestRotatorConstants:
    def test_bond_sleeve_weight(self):
        assert BondDurationRotator.BOND_SLEEVE_WEIGHT == 0.16

    def test_ensemble_weight(self):
        assert BondDurationRotator.ENSEMBLE_WEIGHT == 0.08


# ── BondDurationRotator state management ────────────────────────────


class TestRotatorState:
    def test_default_state(self):
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
        assert rot._state["status"] == "active"
        assert rot._state["tlt_weight"] == 0.0

    def test_load_existing_state(self):
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            sf.write_text(json.dumps({
                "status": "defensive",
                "last_rotation_date": "2026-01-01",
                "current_position": "short",
                "tlt_weight": 0.01,
                "ief_weight": 0.05,
                "shy_weight": 0.10,
            }))
            rot = BondDurationRotator(state_file=sf)
        assert rot._state["status"] == "defensive"
        assert rot._state["tlt_weight"] == 0.01

    def test_get_status(self):
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
        assert rot.get_status() == RotationStatus.ACTIVE


# ── BondDurationRotator.recommend() ─────────────────────────────────


class TestRecommend:
    def test_normal_curve_active(self):
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            dec = rot.recommend(yield_10y=4.5, yield_2y=4.0)
        assert dec.status == "active"
        assert dec.is_actionable is True

    def test_inverted_curve_defensive(self):
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            dec = rot.recommend(yield_10y=3.5, yield_2y=4.0)
        assert dec.status == "defensive"

    def test_sleeve_weights_scale(self):
        """Sleeve weights * BOND_SLEEVE_WEIGHT = total weights."""
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            dec = rot.recommend(yield_10y=4.5, yield_2y=4.0)
        assert abs(dec.tlt_total - dec.tlt_sleeve * 0.16) < 0.001
        assert abs(dec.ief_total - dec.ief_sleeve * 0.16) < 0.001
        assert abs(dec.shy_total - dec.shy_sleeve * 0.16) < 0.001

    def test_total_weights_sum_to_bond_sleeve(self):
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            dec = rot.recommend(yield_10y=4.5, yield_2y=4.0)
        assert abs(dec.tlt_total + dec.ief_total + dec.shy_total - 0.16) < 0.01

    def test_short_position_actionable_when_tlt_low(self):
        """Short position with very low TLT (<10%) IS actionable — strong defensive signal."""
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            dec = rot.recommend(yield_10y=3.5, yield_2y=4.0, rate_change_6m=0.50)
        # INVERTED + RISING → TLT=0.00, SHY=0.80
        assert dec.tlt_sleeve == 0.0
        assert dec.is_actionable is True  # Major TLT reduction IS actionable

    def test_short_position_not_actionable_moderate_tlt(self):
        """Short position with moderate TLT (>=10%) is NOT flagged as major reduction."""
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            # FLAT + STABLE → TLT=0.10, position=short
            dec = rot.recommend(yield_10y=4.10, yield_2y=3.95, rate_change_6m=0.0)
        assert dec.curve_regime == "flat"
        assert dec.status == "defensive"
        assert dec.tlt_sleeve == 0.10
        # tlt_weight=0.10 is NOT < 0.10 → not a major reduction → not actionable
        assert dec.is_actionable is False

    def test_short_position_actionable_when_tlt_moderate(self):
        """Short position with moderate TLT (>=10%) is actionable."""
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            dec = rot.recommend(yield_10y=3.7, yield_2y=4.0, rate_change_6m=-0.50)
        # INVERTED + FALLING → TLT=0.10
        assert dec.tlt_sleeve == 0.10
        assert dec.is_actionable is True

    def test_signal_fields_passed_through(self):
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            dec = rot.recommend(yield_10y=5.5, yield_2y=3.5, real_rate=3.0, rate_change_6m=-0.50)
        assert dec.curve_regime == "steep"
        assert dec.rate_direction == "falling"
        assert dec.real_rate == 3.0

    def test_recommendation_string_format(self):
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            dec = rot.recommend(yield_10y=4.5, yield_2y=4.0)
        assert "Position=" in dec.recommendation
        assert "TLT=" in dec.recommendation

    def test_state_saved_after_recommend(self):
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            dec = rot.recommend(yield_10y=4.5, yield_2y=4.0)
        # State file should exist now
        assert sf.exists()
        state = json.loads(sf.read_text())
        assert "status" in state
        assert "tlt_weight" in state


# ── get_allocation_shifts() ─────────────────────────────────────────


class TestGetAllocationShifts:
    def test_returns_shifts_dict(self):
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            shifts = rot.get_allocation_shifts()
        assert "tlt" in shifts
        assert "ief" in shifts
        assert "shy" in shifts
        assert "spy" in shifts
        assert "gld" in shifts

    def test_spy_gld_always_zero(self):
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            shifts = rot.get_allocation_shifts()
        assert shifts["spy"] == 0.0
        assert shifts["gld"] == 0.0

    def test_tlt_shift_negative_from_baseline(self):
        """When not fully in TLT, shift is negative from baseline=0.16."""
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            shifts = rot.get_allocation_shifts(baseline_tlt=0.16)
        # If tlt_total < 0.16, shift is negative
        if shifts["tlt"] != 0.0:
            assert shifts["tlt"] < 0  # Usually rotated away from TLT


# ── backtest() ───────────────────────────────────────────────────────


class TestBacktest:
    def test_insufficient_data(self):
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            result = rot.backtest(
                yield_history=[(4.5, 4.0, 2.0)],
                spy_returns=[0.01],
                tlt_returns=[0.005],
                ief_returns=[0.003],
                shy_returns=[0.001],
                gld_returns=[0.002],
                dates=["2026-01-01"],
            )
        assert "error" in result

    def test_basic_backtest(self):
        n = 30
        yields = [(4.5, 4.0, 2.0)] * n
        spy_r = [0.01] * n
        tlt_r = [0.005] * n
        ief_r = [0.003] * n
        shy_r = [0.001] * n
        gld_r = [0.002] * n
        dates = [f"2026-01-{i+1:02d}" for i in range(n)]

        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            result = rot.backtest(yields, spy_r, tlt_r, ief_r, shy_r, gld_r, dates)

        assert "dates" in result
        assert "baseline_returns" in result
        assert "rotated_returns" in result
        assert len(result["dates"]) == n

    def test_backtest_summary(self):
        n = 60
        yields = [(4.5, 4.0, 2.0)] * n
        spy_r = [0.001 * (i % 3 - 1) for i in range(n)]
        tlt_r = [0.0005] * n
        ief_r = [0.0003] * n
        shy_r = [0.0001] * n
        gld_r = [0.0002] * n
        dates = [f"2026-01-{i+1:02d}" for i in range(n)]

        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            rot = BondDurationRotator(state_file=sf)
            result = rot.backtest(yields, spy_r, tlt_r, ief_r, shy_r, gld_r, dates)

        assert "summary" in result
        assert "cagr_baseline" in result["summary"]
        assert "cagr_rotated" in result["summary"]
        assert "sharpe_baseline" in result["summary"]
        assert "sharpe_rotated" in result["summary"]


# ── Convenience functions ────────────────────────────────────────────


class TestConvenienceFunctions:
    def test_calculate_bond_rotation(self):
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            with patch.object(BondDurationRotator, "STATE_FILE", sf):
                dec = calculate_bond_rotation()
        assert isinstance(dec, BondRotationDecision)

    def test_get_bond_duration_summary(self):
        with _patch_signal_gen_ensure_dirs():
            sf = _temp_state_file()
            with patch.object(BondDurationRotator, "STATE_FILE", sf):
                summary = get_bond_duration_summary()
        assert "status" in summary
        assert "recommendation" in summary
        assert "effective_duration" in summary
