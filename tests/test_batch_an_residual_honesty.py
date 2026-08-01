"""Batch AN residual honesty: health/rebalance/garch git sha + budget units."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_health_json_stamps_generator_git_sha_in_source():
    src = Path("src/dashboard/generator.py").read_text(encoding="utf-8")
    assert "health_data = _stamp_generator_git_sha(health_data)" in src


def test_rebalance_health_main_stamps_git_sha():
    src = Path("src/monitor/rebalance_health.py").read_text(encoding="utf-8")
    assert "_stamp_generator_git_sha" in src
    assert "rebalance_health.json" in src


def test_garch_public_payload_stamps_git_sha():
    src = Path("scripts/compute_garch_risk.py").read_text(encoding="utf-8")
    assert "_stamp_generator_git_sha" in src
    assert "garch_cvar.json" in src


def test_smart_rebalance_status_discloses_budget_units(tmp_path):
    from src.rebalancing.smart_rebalancer import SmartRebalancingController

    # Isolate from host DATA_DIR / smart_rebalance_state.json so YTD costs
    # from live lab state cannot zero out the fresh annual budget.
    ctrl = SmartRebalancingController(
        data_dir=tmp_path,
        state_path=tmp_path / "smart_rebalance_state.json",
        load_state=False,
    )
    status = ctrl.get_status()
    assert status["remaining_budget_pct_unit"] == "percent_of_portfolio"
    assert status["remaining_budget_ratio_unit"] == "portfolio_fraction"
    # Fresh tracker: full annual 0.5% budget remaining
    assert status["remaining_budget_pct"] == pytest.approx(0.5)
    assert status["remaining_budget_ratio"] == pytest.approx(0.005)


def test_signal_section_builder_emits_smart_rebalance_budget_unit_fields():
    src = Path("src/dashboard/signal_section_builder.py").read_text(encoding="utf-8")
    assert "'remaining_budget_pct_unit': 'percent_of_portfolio'" in src
    assert "'remaining_budget_ratio_unit': 'portfolio_fraction'" in src
    assert "'annual_cost_limit_pct': 0.5" in src
