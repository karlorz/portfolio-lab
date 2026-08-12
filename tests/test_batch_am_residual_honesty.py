"""Batch AM residual honesty: git sha on adaptive_sizing/risk_decomp/unified + UTC."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch



def test_adaptive_sizing_json_stamps_generator_git_sha(tmp_path, monkeypatch):
    from src.dashboard.generator import DashboardGenerator

    public = tmp_path / "public"
    public.mkdir()
    monkeypatch.setattr("src.dashboard.generator.PUBLIC_DIR", public)

    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen._build_advisory_allocation_artifact_role = lambda **k: {
        "role": "advisory",
        "live_authoritative": False,
    }
    gen._flatten_advisory_authority = lambda a: a

    class FakeDecision:
        base_allocation = {"SPY": 0.46}
        adjusted_allocation = {"SPY": 0.46}
        adjustments = {}
        regime_adjustment = 1.0
        volatility_adjustment = 1.0
        signal_adjustment = 1.0
        drawdown_adjustment = 1.0
        factors = {}

    class FakeSizer:
        def compute_allocation(self):
            return FakeDecision()

    with patch(
        "src.strategy.adaptive_sizing.AdaptiveSizer",
        FakeSizer,
    ), patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="adapt01sha",
    ):
        out = DashboardGenerator.generate_adaptive_sizing_json(gen)

    assert out is not None
    payload = json.loads(Path(out).read_text())
    assert payload["generator_git_sha"] == "adapt01sha"
    assert payload["generator_git_sha_status"] == "full_generate"


def test_risk_decomposition_json_stamps_and_utc(tmp_path, monkeypatch):
    from src.dashboard.generator import DashboardGenerator

    public = tmp_path / "public"
    public.mkdir()
    monkeypatch.setattr("src.dashboard.generator.PUBLIC_DIR", public)
    monkeypatch.setattr(
        "src.dashboard.generator.BASE_ALLOCATION",
        {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
    )

    class FakeResult:
        def to_dict(self):
            return {
                "timestamp": "2026-07-20T12:00:00",
                "total_portfolio_volatility": 0.12,
            }

    with patch(
        "src.monitor.risk_decomposition.decompose_portfolio",
        return_value=FakeResult(),
    ), patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="riskdec01",
    ):
        gen = DashboardGenerator.__new__(DashboardGenerator)
        out = DashboardGenerator.generate_risk_decomposition_json(gen)

    payload = json.loads(Path(out).read_text())
    assert payload["generator_git_sha"] == "riskdec01"
    gen_at = payload["generated_at"]
    assert "+00:00" in gen_at or gen_at.endswith("Z")
    dt = datetime.fromisoformat(gen_at.replace("Z", "+00:00"))
    assert dt.tzinfo is not None


def test_unified_dashboard_stamps_generator_git_sha(monkeypatch):
    from src.monitor import unified_dashboard as ud

    # Stub sections so generate is light
    empty = {"available": False}
    monkeypatch.setattr(ud, "_get_health_section", lambda: empty)
    monkeypatch.setattr(ud, "_get_portfolio_section", lambda: empty)
    monkeypatch.setattr(ud, "_get_risk_section", lambda: empty)
    monkeypatch.setattr(ud, "_get_risk_history_section", lambda: empty)
    monkeypatch.setattr(ud, "_get_tca_section", lambda: empty)
    monkeypatch.setattr(ud, "_get_overlays_section", lambda: {})
    monkeypatch.setattr(ud, "_get_regime_section", lambda: empty)
    monkeypatch.setattr(ud, "_get_attribution_section", lambda: empty)
    monkeypatch.setattr(ud, "_get_adaptive_weights_section", lambda: empty)
    monkeypatch.setattr(ud, "_get_cron_section", lambda: empty)

    with patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="unified01",
    ):
        payload = ud.generate_unified_dashboard()

    assert payload["generator_git_sha"] == "unified01"
    assert payload["generator_git_sha_status"] == "full_generate"


def test_risk_decomposition_module_timestamp_is_utc():
    src = Path("src/monitor/risk_decomposition.py").read_text(encoding="utf-8")
    assert "datetime.now(timezone.utc).isoformat()" in src
    assert "timestamp=datetime.now().isoformat()" not in src
