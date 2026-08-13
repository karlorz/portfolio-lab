"""Batch AQ residual honesty: graduation raw Sharpe + IC half-life advisory table."""

from __future__ import annotations

from pathlib import Path


def test_advisory_factor_half_life_table_not_live_authoritative():
    from src.monitor.ic_decay_monitor import advisory_factor_half_life_table

    table = advisory_factor_half_life_table()
    assert table["live_authoritative"] is False
    assert table["role"] == "advisory"
    assert table["unit"] == "trading_days"
    sleeves = table["sleeves"]
    assert sleeves["momentum"]["suggested_rebalance_days"] == 63
    assert sleeves["value"]["suggested_rebalance_days"] == 84
    assert sleeves["strategic_spy_gld_tlt"]["suggested_rebalance_days"] == 252
    assert "do not override" in table["disclosure"].lower() or "not" in table["disclosure"].lower()


def test_compute_ic_decay_report_embeds_half_life_table(monkeypatch, tmp_path):
    from src.monitor import ic_decay_monitor as icm

    monkeypatch.setattr(icm, "IC_STATE_PATH", tmp_path / "ic_monitor_state.json")
    monkeypatch.setattr(icm, "_signal_prediction_backlog", lambda db_path=None: {
        "pending_rows": 0,
        "pending_dates": 0,
        "oldest_unresolved_date": None,
        "total_predictions": 0,
        "resolved_predictions": 0,
        "pending_semantics": "test",
    })
    report = icm.compute_ic_decay_report()
    assert "advisory_factor_half_life" in report
    assert report["advisory_factor_half_life"]["live_authoritative"] is False
    assert "momentum" in report["advisory_factor_half_life"]["sleeves"]


def test_generator_graduation_keeps_raw_implausible_sharpe_contract():
    """Static contract: test fixture must not expect coerced Sharpe 0.0."""
    src = Path("tests/test_generator_json_sections.py").read_text(encoding="utf-8")
    assert 'criteria["min_sharpe"]["value"] == 3.38' in src
    assert 'criteria["min_sharpe"]["value"] == 0.0' not in src
    checklist = Path("src/strategy/graduation_checklist.py").read_text(encoding="utf-8")
    assert "Never coerce to 0.0" in checklist
