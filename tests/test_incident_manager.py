"""Tests for persisted alert incident lifecycle management."""

import json
from datetime import datetime, timezone

from src.monitor.incident_manager import IncidentManager, IncidentState


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_warn_alert_opens_p2_incident_and_summary(tmp_path):
    log_path = tmp_path / "incidents.jsonl"
    summary_path = tmp_path / "incidents.json"
    manager = IncidentManager(log_path=log_path, summary_path=summary_path)

    incident = manager.record_alert(
        channel="signal_staleness",
        level="warn",
        message="2 signals stale",
        details={"stale_signals": ["ensemble", "garch"]},
        now=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
    )

    assert incident is not None
    assert incident.state == IncidentState.FIRING
    assert incident.severity == "p2"
    assert incident.channel == "signal_staleness"

    events = _read_jsonl(log_path)
    assert [event["event"] for event in events] == ["opened"]
    assert events[0]["incident_id"] == incident.incident_id

    summary = json.loads(summary_path.read_text())
    assert summary["open_count"] == 1
    assert summary["metrics"]["incident_frequency"] == 1
    assert summary["incidents"][0]["message"] == "2 signals stale"


def test_pass_alert_resolves_open_incident_and_records_mttr(tmp_path):
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
    )

    opened = manager.record_alert(
        channel="portfolio_drift",
        level="halt",
        message="critical drift",
        now=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
    )
    resolved = manager.record_alert(
        channel="portfolio_drift",
        level="pass",
        message="drift within tolerance",
        now=datetime(2026, 7, 1, 0, 5, tzinfo=timezone.utc),
    )

    assert opened is not None
    assert resolved is not None
    assert resolved.incident_id == opened.incident_id
    assert resolved.state == IncidentState.RESOLVED
    assert resolved.severity == "p0"
    assert resolved.mttr_seconds == 300.0

    events = _read_jsonl(tmp_path / "incidents.jsonl")
    assert [event["event"] for event in events] == ["opened", "resolved"]

    summary = json.loads((tmp_path / "incidents.json").read_text())
    assert summary["open_count"] == 0
    assert summary["metrics"]["resolved_count"] == 1
    assert summary["metrics"]["mean_mttr_seconds"] == 300.0


def test_repeated_warn_updates_existing_open_incident_without_new_id(tmp_path):
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
    )

    first = manager.record_alert(
        channel="ic_decay",
        level="warn",
        message="signal warning",
        now=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
    )
    second = manager.record_alert(
        channel="ic_decay",
        level="halt",
        message="signal critical",
        now=datetime(2026, 7, 1, 0, 1, tzinfo=timezone.utc),
    )

    assert first is not None
    assert second is not None
    assert second.incident_id == first.incident_id
    assert second.severity == "p0"
    assert second.message == "signal critical"

    events = _read_jsonl(tmp_path / "incidents.jsonl")
    assert [event["event"] for event in events] == ["opened", "updated"]


def test_malformed_incident_event_is_skipped_when_replaying(tmp_path):
    log_path = tmp_path / "incidents.jsonl"
    log_path.write_text('{"event": "opened", "incident_id": "bad"}\n')
    manager = IncidentManager(
        log_path=log_path,
        summary_path=tmp_path / "incidents.json",
    )

    incident = manager.record_alert(
        channel="cron_failure",
        level="warn",
        message="cron failed",
        now=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
    )

    assert incident is not None
    assert incident.channel == "cron_failure"
    assert manager.open_incidents() == [incident]
