#!/usr/bin/env python3
"""
Regression tests for the C7 alerts/incidents/promotion mixin extracted by
Item 22 (2026-08-12): ``src/dashboard/sections_alerts.py``
``_AlertsSectionsMixin`` (test file owed by the TEST-GAP coverage gap —
module has zero direct test references).

A1: getattr smoke — all 6 moved names resolve via BOTH ``DashboardGenerator``
    (MRO) and ``_AlertsSectionsMixin``.
A2: behavior-equality — canned fixtures for the pure statics
    (``_is_active_promote_candidacy``), tmp-file driven helpers
    (``_has_open_blocking_incident``, ``_stale_data_alerts_from_quality_report``),
    a faked ``GraduationChecklist`` for the gate/alert paths, and the
    FakeDateTime-deferred ``_empty_incident_summary``.
"""
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from src.dashboard.generator import DashboardGenerator
from src.dashboard.sections_alerts import _AlertsSectionsMixin

ALERTS_NAMES = (
    "_has_open_blocking_incident",
    "_promotion_gate_status",
    "_is_active_promote_candidacy",
    "_graduation_candidate_alert",
    "_stale_data_alerts_from_quality_report",
    "_empty_incident_summary",
)


def test_a1_getattr_resolution_via_both_surfaces():
    """All 6 C7 names resolve via DashboardGenerator MRO and the mixin."""
    for name in ALERTS_NAMES:
        assert hasattr(DashboardGenerator, name), name
        assert hasattr(_AlertsSectionsMixin, name), name


def test_a2_is_active_promote_candidacy_canned():
    """Live candidacy only; tombstones/unknown actions are not candidates."""
    for surface in (_AlertsSectionsMixin, DashboardGenerator):
        assert surface._is_active_promote_candidacy({}) is True  # legacy marker
        assert surface._is_active_promote_candidacy({"action": "promote_to_live"}) is True
        assert (
            surface._is_active_promote_candidacy(
                {"action": "promote_blocked_kill_switch"}
            )
            is False
        )
        assert (
            surface._is_active_promote_candidacy(
                {"action": "promote_blocked_graduation_checklist"}
            )
            is False
        )
        assert surface._is_active_promote_candidacy({"action": "random"}) is False
        assert surface._is_active_promote_candidacy({"action": 42}) is False


def _write(tmp_path, name, payload):
    (tmp_path / name).write_text(
        json.dumps(payload) if payload is not None else "[]", encoding="utf-8"
    )


def test_a2_has_open_blocking_incident_canned(tmp_path):
    """Blocking open incidents across both files; closed/non-blocking pass."""
    for surface in (_AlertsSectionsMixin, DashboardGenerator):
        assert surface._has_open_blocking_incident(tmp_path) is False  # no files

        _write(tmp_path, "incidents.json", {"incidents": [{"status": "open", "blocking": True}]})
        assert surface._has_open_blocking_incident(tmp_path) is True

        _write(tmp_path, "incidents.json", {"incidents": [{"status": "closed", "blocking": True}]})
        assert surface._has_open_blocking_incident(tmp_path) is False

        _write(tmp_path, "incidents.json", {"incidents": [{"status": "open", "blocks_promotion": True}]})
        assert surface._has_open_blocking_incident(tmp_path) is True

        _write(tmp_path, "incidents.json", {"incidents": [{"status": "open"}]})
        assert surface._has_open_blocking_incident(tmp_path) is False

        # incident_state.json open_incidents shape; incidents.json non-dict → skipped.
        (tmp_path / "incidents.json").unlink()
        _write(tmp_path, "incident_state.json", {"open_incidents": [{"status": "open", "blocking": True}]})
        assert surface._has_open_blocking_incident(tmp_path) is True
        (tmp_path / "incident_state.json").unlink()


class _PassingChecklist:
    def check(self):
        return {"manual_approval": SimpleNamespace(passed=True)}

    def is_graduation_ready(self, results):
        return True


class _FailingChecklist:
    def check(self):
        return {"manual_approval": None}

    def is_graduation_ready(self, results):
        return False


def test_a2_promotion_gate_status_canned(monkeypatch, tmp_path):
    """Kill/incident/manual/checklist blockers compose; clean gates pass."""
    monkeypatch.setattr(
        "src.strategy.graduation_checklist.GraduationChecklist", _PassingChecklist
    )
    for surface in (_AlertsSectionsMixin, DashboardGenerator):
        allowed, blockers = surface._promotion_gate_status(tmp_path)
        assert allowed is True
        assert blockers == []

        _write(tmp_path, "kill_switch.json", {"enabled": True, "level": "halt"})
        allowed, blockers = surface._promotion_gate_status(tmp_path)
        assert allowed is False
        assert blockers == ["kill_switch"]
        (tmp_path / "kill_switch.json").unlink()

        _write(tmp_path, "incidents.json", {"incidents": [{"status": "open", "blocking": True}]})
        allowed, blockers = surface._promotion_gate_status(tmp_path)
        assert allowed is False
        assert blockers == ["blocking_incident"]
        (tmp_path / "incidents.json").unlink()

    monkeypatch.setattr(
        "src.strategy.graduation_checklist.GraduationChecklist", _FailingChecklist
    )
    for surface in (_AlertsSectionsMixin, DashboardGenerator):
        allowed, blockers = surface._promotion_gate_status(tmp_path)
        assert allowed is False
        assert blockers == ["manual_approval", "graduation_checklist"]


def test_a2_graduation_candidate_alert_success(monkeypatch, tmp_path):
    """Passing gates → success alert with Sharpe from the marker."""
    monkeypatch.setattr(
        "src.strategy.graduation_checklist.GraduationChecklist", _PassingChecklist
    )
    _write(
        tmp_path,
        ".promote_to_live",
        {
            "action": "promote_to_live",
            "metrics": {"sharpe": 0.95},
            "timestamp": "2026-07-06T10:00:00Z",
        },
    )
    for surface in (_AlertsSectionsMixin, DashboardGenerator):
        alert = surface._graduation_candidate_alert(tmp_path)
        assert alert["level"] == "success"
        assert alert["type"] == "graduation_candidate"
        assert alert["title"] == "Paper Trading Graduation Ready"
        assert "Sharpe: 0.95" in alert["message"]
        assert alert["timestamp"] == "2026-07-06T10:00:00Z"
        assert alert["requires_action"] is True


def test_a2_graduation_candidate_alert_blocked(monkeypatch, tmp_path):
    """Kill switch on → warning alert naming the blocker."""
    monkeypatch.setattr(
        "src.strategy.graduation_checklist.GraduationChecklist", _PassingChecklist
    )
    _write(tmp_path, "kill_switch.json", {"enabled": True, "level": "halt"})
    _write(tmp_path, ".promote_to_live", {"action": "promote_to_live"})
    for surface in (_AlertsSectionsMixin, DashboardGenerator):
        alert = surface._graduation_candidate_alert(tmp_path)
        assert alert["level"] == "warning"
        assert alert["title"] == "Paper Trading Graduation Blocked"
        assert "kill_switch" in alert["message"]
        assert alert["requires_action"] is True


def test_a2_graduation_candidate_alert_tombstone_and_missing(monkeypatch, tmp_path):
    """Tombstone action or missing marker → no alert at all."""
    monkeypatch.setattr(
        "src.strategy.graduation_checklist.GraduationChecklist", _PassingChecklist
    )
    _write(tmp_path, ".promote_to_live", {"action": "promote_blocked_kill_switch"})
    for surface in (_AlertsSectionsMixin, DashboardGenerator):
        assert surface._graduation_candidate_alert(tmp_path) is None
    (tmp_path / ".promote_to_live").unlink()
    for surface in (_AlertsSectionsMixin, DashboardGenerator):
        assert surface._graduation_candidate_alert(tmp_path) is None


def test_a2_stale_data_alerts_from_quality_report_canned(tmp_path):
    """Stale rows alert with lag detail; padding covers count shortfall."""
    for surface in (_AlertsSectionsMixin, DashboardGenerator):
        assert surface._stale_data_alerts_from_quality_report(tmp_path) is None

        _write(tmp_path, "data_quality.json", {"issue_counts": {}})
        assert surface._stale_data_alerts_from_quality_report(tmp_path) == []
        (tmp_path / "data_quality.json").unlink()

        _write(
            tmp_path,
            "data_quality.json",
            {
                "issue_counts": {"stale_latest_dates": 2},
                "reference_date": "2026-07-06",
                "generated_at": "2026-07-06T08:00:00Z",
                "symbols": [
                    {
                        "symbol": "SPY",
                        "stale_latest_date": {
                            "latest_date": "2026-07-02",
                            "latest_lag_days": 2,
                        },
                    },
                    {"symbol": "GLD", "status": "fail"},
                    {"symbol": "TLT", "status": "ok"},
                ],
            },
        )
        alerts = surface._stale_data_alerts_from_quality_report(tmp_path)
        assert len(alerts) == 2
        assert alerts[0]["title"] == "Stale Data: SPY"
        assert alerts[0]["message"] == (
            "SPY latest date 2026-07-02 lags reference 2026-07-06 (2 trading day lag)"
        )
        assert alerts[0]["level"] == "warning"
        assert alerts[0]["type"] == "stale_data"
        assert alerts[0]["requires_action"] is False
        assert alerts[0]["timestamp"] == "2026-07-06T08:00:00Z"
        assert alerts[1]["title"] == "Stale Data: GLD"
        assert alerts[1]["message"] == "GLD latest date unknown lags reference 2026-07-06"
        (tmp_path / "data_quality.json").unlink()

        # Count shortfall → generic padding alerts until stale_count reached.
        _write(
            tmp_path,
            "data_quality.json",
            {
                "issue_counts": {"stale_latest_dates": 3},
                "generated_at": "2026-07-06T08:00:00Z",
                "symbols": [{"symbol": "SPY", "status": "failed"}],
            },
        )
        padded = surface._stale_data_alerts_from_quality_report(tmp_path)
        assert len(padded) == 3
        assert padded[1]["title"] == "Stale Data"
        assert padded[1]["message"] == (
            "data_quality.json reports 3 stale latest-date issue(s)"
        )
        (tmp_path / "data_quality.json").unlink()

        # Non-int stale_count guard → no alerts.
        _write(tmp_path, "data_quality.json", {"issue_counts": {"stale_latest_dates": "2"}})
        assert surface._stale_data_alerts_from_quality_report(tmp_path) == []
        (tmp_path / "data_quality.json").unlink()


class FakeDateTime(datetime):
    """Deterministic now(); mirrors the test_generator.py patch seam."""

    _value = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls._value.replace(tzinfo=None)
        return cls._value.astimezone(tz)


def test_a2_empty_incident_summary_deferred_timestamp():
    """Empty summary: FakeDateTime-deferred timestamp, zeroed metrics."""
    with patch("src.dashboard.generator.datetime", FakeDateTime):
        for surface in (_AlertsSectionsMixin, DashboardGenerator):
            result = surface._empty_incident_summary()
            assert result["generated_at"] == FakeDateTime.now(timezone.utc).isoformat()
            assert result["schema_version"] == "incident-lifecycle/v1"
            assert result["open_count"] == 0
            assert result["incidents"] == []
            assert result["metrics"] == {
                "incident_frequency": 0,
                "open_count": 0,
                "resolved_count": 0,
                "mean_mttr_seconds": None,
            }
