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
