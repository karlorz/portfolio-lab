"""Batch AL residual honesty: generator_git_sha on analytics/graduation/overlay."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch



def test_stamp_generator_git_sha_helper():
    from src.dashboard.generator import _stamp_generator_git_sha

    with patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="abc123def",
    ):
        out = _stamp_generator_git_sha({"foo": 1})
    assert out["foo"] == 1
    assert out["generator_git_sha"] == "abc123def"
    assert out["generator_git_sha_status"] == "full_generate"


def test_analytics_json_stamps_generator_git_sha(tmp_path, monkeypatch):
    from src.dashboard.generator import DashboardGenerator

    public = tmp_path / "public"
    public.mkdir()
    monkeypatch.setattr("src.dashboard.generator.PUBLIC_DIR", public)
    monkeypatch.setattr("src.dashboard.generator.DATA_DIR", tmp_path / "data")
    (tmp_path / "data").mkdir()

    gen = DashboardGenerator.__new__(DashboardGenerator)
    fake_report = {"status": "success", "generated_at": "2026-07-20T12:00:00+00:00"}

    with patch("src.analytics.calculator.AnalyticsCalculator") as mock_calc, patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="cafe1234",
    ):
        mock_calc.return_value.generate_analytics_report.return_value = fake_report
        out = DashboardGenerator.generate_analytics_json(gen)

    payload = json.loads(Path(out).read_text())
    assert payload["generator_git_sha"] == "cafe1234"
    assert payload["generator_git_sha_status"] == "full_generate"


def test_graduation_json_stamps_generator_git_sha(tmp_path, monkeypatch):
    from src.dashboard.generator import DashboardGenerator
    from src.strategy.graduation_checklist import CheckResult

    public = tmp_path / "public"
    public.mkdir()
    monkeypatch.setattr("src.dashboard.generator.PUBLIC_DIR", public)
    monkeypatch.setattr("src.dashboard.generator.DATA_DIR", tmp_path / "data")
    (tmp_path / "data").mkdir()

    gen = DashboardGenerator.__new__(DashboardGenerator)
    gen._paper_trading_summary_for_dashboard = lambda *a, **k: {}
    gen._graduation_display_value = lambda x: str(x)

    class FakeChecklist:
        criteria = {"min_trading_days": {"value": 63}}

        def _load_state(self):
            return {}

        def check(self, state):
            return {
                "min_trading_days": CheckResult(
                    name="min_trading_days",
                    passed=True,
                    value=70,
                    required=63,
                    description="days",
                ),
                "manual_approval": CheckResult(
                    name="manual_approval",
                    passed=False,
                    value=0,
                    required=1,
                    description="manual",
                ),
            }

        def readiness_score(self, results):
            return 20.0

        def is_graduation_ready(self, results):
            return False

    with patch(
        "src.strategy.graduation_checklist.GraduationChecklist",
        FakeChecklist,
    ), patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="gradsha01",
    ):
        out = DashboardGenerator.generate_graduation_json(gen)

    assert out is not None
    payload = json.loads(Path(out).read_text())
    assert payload["generator_git_sha"] == "gradsha01"
    assert payload["generator_git_sha_status"] == "full_generate"


def test_overlay_save_stamps_generator_git_sha(tmp_path, monkeypatch):
    from src.dashboard.overlay_dashboard import (
        OverlayDashboardData,
        OverlayDashboardGenerator,
    )

    out = tmp_path / "overlay_dashboard.json"
    monkeypatch.setattr(OverlayDashboardGenerator, "OUTPUT_PATH", out)

    gen = OverlayDashboardGenerator(data_dir=tmp_path)
    dash = OverlayDashboardData(
        timestamp="2026-07-20T12:00:00+00:00",
        generated_at="2026-07-20T12:00:00+00:00",
        collar={"active": False},
        crypto={"active": False},
        bond_duration={"active": False},
        calendar={"active": False},
        kurtosis={"active": False},
        mean_reversion={"active": False},
        unified={"active": False},
        active_overlays=0,
        total_overlays=7,
        portfolio_risk="low",
        alerts=[],
    )
    with patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="abad1dea00",
    ):
        gen.save(dash)

    payload = json.loads(out.read_text())
    assert payload["generator_git_sha"] == "abad1dea00"
    assert payload["generator_git_sha_status"] == "full_generate"


def test_generate_overlay_json_stamps_public_payload(tmp_path, monkeypatch):
    from src.dashboard.generator import DashboardGenerator
    from src.dashboard.overlay_dashboard import OverlayDashboardData

    public = tmp_path / "public"
    public.mkdir()
    monkeypatch.setattr("src.dashboard.generator.PUBLIC_DIR", public)

    gen = DashboardGenerator.__new__(DashboardGenerator)
    dash = OverlayDashboardData(
        timestamp="2026-07-20T12:00:00+00:00",
        generated_at="2026-07-20T12:00:00+00:00",
        collar={"active": False},
        crypto={"active": False},
        bond_duration={"active": False},
        calendar={"active": False},
        kurtosis={"active": False},
        mean_reversion={"active": False},
        unified={"active": False},
        active_overlays=0,
        total_overlays=0,
        portfolio_risk="low",
        alerts=[],
    )

    class FakeOverlayGen:
        OUTPUT_PATH = tmp_path / "overlay_dashboard.json"

        def generate(self):
            return dash

        def save(self, dashboard):
            pass

    with patch(
        "src.dashboard.overlay_dashboard.OverlayDashboardGenerator",
        FakeOverlayGen,
    ), patch(
        "src.dashboard.generator._generator_git_sha_short",
        return_value="overlay99",
    ):
        out = DashboardGenerator.generate_overlay_json(gen)

    assert out is not None
    payload = json.loads(Path(out).read_text())
    assert payload["generator_git_sha"] == "overlay99"
    assert payload["generator_git_sha_status"] == "full_generate"
