"""Tests for the dynamic hedge selector."""

from __future__ import annotations

import json

import pytest

from src.strategy.hedge_selector import HedgeSelector


def test_portfolio_regime_label_high_vol_maps_to_stress(tmp_path):
    selector = HedgeSelector({"state_file": str(tmp_path / "hedge_state.json")})

    rec = selector.select(vix_level=28.0, regime_confidence=0.8, regime_label="high_vol")

    assert rec.regime == "stress"
    assert rec.primary_hedge == "put_spread"
    assert rec.secondary_hedge == "vixy"


def test_min_hold_period_maintains_current_hedge(tmp_path):
    state_file = tmp_path / "hedge_state.json"
    state_file.write_text(
        json.dumps(
            {
                "timestamp": "2026-06-08T12:00:00",
                "current_hedge": "vixy",
                "current_size_pct": 3.0,
                "days_in_position": 2,
                "last_switch_date": "2026-06-06T12:00:00",
                "ytd_switches": 1,
                "ytd_cost_bps": 4.0,
            }
        )
    )
    selector = HedgeSelector({"state_file": str(state_file), "min_hold_days": 5})

    rec = selector.select(vix_level=45.0, regime_confidence=0.9, regime_label="crisis")

    assert rec.primary_hedge == "vixy"
    assert rec.primary_size_pct == 3.0
    assert rec.transition_cost_bps == 0.0


def test_normal_no_trade_band_overrides_min_hold_vixy_position(tmp_path):
    state_file = tmp_path / "hedge_state.json"
    state_file.write_text(
        json.dumps(
            {
                "timestamp": "2026-06-08T12:00:00",
                "current_hedge": "vixy",
                "current_size_pct": 3.0,
                "days_in_position": 2,
                "last_switch_date": "2026-06-06T12:00:00",
                "ytd_switches": 1,
                "ytd_cost_bps": 4.0,
            }
        )
    )
    selector = HedgeSelector({"state_file": str(state_file), "min_hold_days": 5})

    rec = selector.select(
        vix_level=16.0,
        regime_confidence=0.9,
        regime_label="normal",
        term_structure_signal=0.5,
    )

    assert rec.primary_hedge == "none"
    assert rec.primary_size_pct == 0.0
    assert rec.gate_reason == "normal_no_trade_band"


def test_confidence_scaling_reduces_or_zeros_size(tmp_path):
    selector = HedgeSelector({"state_file": str(tmp_path / "hedge_state.json")})

    high_conf = selector.select(vix_level=35.0, regime_confidence=0.9, regime_label="stress")
    medium_conf = selector.select(vix_level=35.0, regime_confidence=0.7, regime_label="stress")
    low_conf = selector.select(vix_level=35.0, regime_confidence=0.3, regime_label="stress")

    assert medium_conf.primary_size_pct == pytest.approx(high_conf.primary_size_pct * 0.5, abs=0.01)
    assert low_conf.primary_size_pct == 0.0


def test_cost_benefit_gate_can_block_expensive_non_crisis_hedge(tmp_path):
    selector = HedgeSelector(
        {
            "state_file": str(tmp_path / "hedge_state.json"),
            "put_spread_cost_bps": 10000,
        }
    )

    rec = selector.select(vix_level=35.0, regime_confidence=0.9, regime_label="stress")

    assert rec.primary_hedge == "put_spread"
    assert rec.net_benefit_bps < 0
    assert rec.cost_benefit_gate is False


def test_normal_regime_disables_vixy_and_discloses_canonical_controller(tmp_path):
    selector = HedgeSelector({"state_file": str(tmp_path / "hedge_state.json")})

    rec = selector.select(
        vix_level=16.0,
        regime_confidence=0.9,
        regime_label="normal",
        term_structure_signal=0.4,
    )

    assert rec.primary_hedge == "none"
    assert rec.primary_size_pct == 0.0
    assert rec.secondary_hedge is None
    assert rec.term_structure_gate is False
    assert rec.term_structure_multiplier == 0.0
    assert rec.canonical_controller == "hedge_selector"
    assert rec.vixy_role == "diagnostic_sizing_helper"
    assert rec.gate_reason == "normal_no_trade_band"


def test_elevated_vixy_requires_risk_off_term_structure_confirmation(tmp_path):
    selector = HedgeSelector({"state_file": str(tmp_path / "hedge_state.json")})

    rec = selector.select(
        vix_level=25.0,
        regime_confidence=0.9,
        regime_label="elevated",
        term_structure_signal=0.25,
    )

    assert rec.primary_hedge == "none"
    assert rec.primary_size_pct == 0.0
    assert rec.cost_benefit_gate is False
    assert rec.term_structure_gate is False
    assert rec.term_structure_multiplier == 0.0
    assert rec.gate_reason == "elevated_requires_risk_off_term_structure"


def test_elevated_confirmed_vixy_is_discounted_by_term_structure_strength(tmp_path):
    selector = HedgeSelector({"state_file": str(tmp_path / "hedge_state.json")})

    rec = selector.select(
        vix_level=25.0,
        regime_confidence=0.9,
        regime_label="elevated",
        term_structure_signal=-0.6,
    )

    assert rec.primary_hedge == "vixy"
    assert rec.primary_size_pct > 0.0
    assert rec.term_structure_gate is True
    assert 0.0 < rec.term_structure_multiplier < 1.0
    assert rec.term_structure_role == "gate_discount_multiplier"
    assert rec.gate_reason == "term_structure_confirmed"
