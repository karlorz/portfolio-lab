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


def test_unresolved_alerts_escalate_kill_switch_by_cycle_threshold(tmp_path):
    """HALT-severity incidents still escalate warning → restrict → halt by count."""
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=tmp_path / "kill_switch.json",
        escalation_cycles=2,
    )

    first = manager.record_alert(
        channel="signal_staleness",
        level="halt",
        message="signals stale",
        now=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
    )
    assert first is not None
    assert first.alert_count == 1
    assert first.severity == "p0"
    assert not (tmp_path / "kill_switch.json").exists()

    second = manager.record_alert(
        channel="signal_staleness",
        level="halt",
        message="signals still stale",
        now=datetime(2026, 7, 1, 0, 5, tzinfo=timezone.utc),
    )
    assert second is not None
    warning = json.loads((tmp_path / "kill_switch.json").read_text())
    assert warning["enabled"] is True
    assert warning["level"] == "warning"
    assert warning["position_reduction"] == 0.25
    assert warning["source"] == "incident_lifecycle"
    assert warning["incident_id"] == second.incident_id
    assert warning["incident_alert_count"] == 2

    fourth = None
    for minute in (10, 15):
        fourth = manager.record_alert(
            channel="signal_staleness",
            level="halt",
            message="signals still stale",
            now=datetime(2026, 7, 1, 0, minute, tzinfo=timezone.utc),
        )

    assert fourth is not None
    restrict = json.loads((tmp_path / "kill_switch.json").read_text())
    assert restrict["level"] == "restrict"
    assert restrict["position_reduction"] == 0.5
    assert restrict["incident_alert_count"] == 4

    sixth = None
    for minute in (20, 25):
        sixth = manager.record_alert(
            channel="signal_staleness",
            level="halt",
            message="all signals stale",
            now=datetime(2026, 7, 1, 0, minute, tzinfo=timezone.utc),
        )

    assert sixth is not None
    halt = json.loads((tmp_path / "kill_switch.json").read_text())
    assert halt["level"] == "halt"
    assert halt["position_reduction"] == 1.0
    assert halt["incident_alert_count"] == 6
    assert halt["reason"] == "unresolved_incident:signal_staleness"


def test_sustained_warn_never_escalates_kill_past_warning(tmp_path):
    """p2 WARN must not ratchet to restrict/halt solely via alert_count."""
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=tmp_path / "kill_switch.json",
        escalation_cycles=2,
    )

    last = None
    for minute in range(0, 40, 2):  # 20 updates → alert_count 20
        last = manager.record_alert(
            channel="signal_staleness",
            level="warn",
            message="optional signals unavailable",
            details={
                "stale_signals": [],
                "unavailable_signals": ["optional_a", "optional_b"],
                "policy": "unavailable_signals_nonempty_blocks_all_fresh_pass",
            },
            now=datetime(2026, 7, 1, 0, minute, tzinfo=timezone.utc),
        )

    assert last is not None
    assert last.severity == "p2"
    assert last.alert_count == 20
    kill = json.loads((tmp_path / "kill_switch.json").read_text())
    assert kill["level"] == "warning"
    assert kill["position_reduction"] == 0.25
    assert kill["incident_alert_count"] == 20
    assert kill["level"] not in {"restrict", "halt"}


def test_warn_then_halt_allows_full_kill_escalation(tmp_path):
    """Severity upgrade mid-incident unlocks restrict/halt from current count."""
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=tmp_path / "kill_switch.json",
        escalation_cycles=2,
    )

    for minute in (0, 2, 4, 6):
        manager.record_alert(
            channel="signal_staleness",
            level="warn",
            message="optional unavailable",
            now=datetime(2026, 7, 1, 0, minute, tzinfo=timezone.utc),
        )
    warn_kill = json.loads((tmp_path / "kill_switch.json").read_text())
    assert warn_kill["level"] == "warning"
    assert warn_kill["incident_alert_count"] == 4

    halt = manager.record_alert(
        channel="signal_staleness",
        level="halt",
        message="required signals stale",
        now=datetime(2026, 7, 1, 0, 8, tzinfo=timezone.utc),
    )
    assert halt is not None
    assert halt.severity == "p0"
    assert halt.alert_count == 5
    kill = json.loads((tmp_path / "kill_switch.json").read_text())
    # count 5 with cycles=2 → stage 2 (restrict) once severity is p0
    assert kill["level"] == "restrict"
    assert kill["incident_alert_count"] == 5


def test_incident_escalation_does_not_downgrade_existing_stronger_kill_switch(tmp_path):
    kill_switch_path = tmp_path / "kill_switch.json"
    kill_switch_path.write_text(json.dumps({
        "enabled": True,
        "level": "liquidate",
        "reason": "max_drawdown_-30.0%",
        "source": "risk_limits",
        "position_reduction": 1.0,
    }))
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=kill_switch_path,
        escalation_cycles=1,
    )

    manager.record_alert(
        channel="ic_decay",
        level="warn",
        message="IC warning",
        now=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
    )

    data = json.loads(kill_switch_path.read_text())
    assert data["level"] == "liquidate"
    assert data["reason"] == "max_drawdown_-30.0%"
    assert data["source"] == "risk_limits"


def test_incident_escalation_preserves_same_rank_non_incident_kill_switch(tmp_path):
    kill_switch_path = tmp_path / "kill_switch.json"
    kill_switch_path.write_text(json.dumps({
        "enabled": True,
        "level": "warning",
        "reason": "max_drawdown_-12.0%",
        "source": "risk_limits",
        "position_reduction": 0.25,
    }))
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=kill_switch_path,
        escalation_cycles=1,
    )

    manager.record_alert(
        channel="ic_decay",
        level="warn",
        message="IC warning",
        now=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
    )

    data = json.loads(kill_switch_path.read_text())
    assert data["level"] == "warning"
    assert data["reason"] == "max_drawdown_-12.0%"
    assert data["source"] == "risk_limits"


def test_pass_alert_clears_matching_incident_owned_kill_switch(tmp_path):
    kill_switch_path = tmp_path / "kill_switch.json"
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=kill_switch_path,
        escalation_cycles=1,
    )

    opened = manager.record_alert(
        channel="portfolio_drift",
        level="halt",
        message="critical drift",
        now=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
    )
    assert opened is not None
    assert kill_switch_path.exists()

    manager.record_alert(
        channel="portfolio_drift",
        level="pass",
        message="drift recovered",
        now=datetime(2026, 7, 1, 0, 5, tzinfo=timezone.utc),
    )

    assert not kill_switch_path.exists()


def test_pass_alert_restores_next_open_incident_owned_kill_switch(tmp_path):
    kill_switch_path = tmp_path / "kill_switch.json"
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=kill_switch_path,
        escalation_cycles=1,
    )

    critical = None
    for minute in (0, 1, 2):
        critical = manager.record_alert(
            channel="signal_staleness",
            level="halt",
            message="signals halted",
            now=datetime(2026, 7, 1, 0, minute, tzinfo=timezone.utc),
        )
    warning = manager.record_alert(
        channel="ic_decay",
        level="warn",
        message="IC warning",
        now=datetime(2026, 7, 1, 0, 3, tzinfo=timezone.utc),
    )
    assert critical is not None
    assert warning is not None
    assert json.loads(kill_switch_path.read_text())["incident_id"] == critical.incident_id

    manager.record_alert(
        channel="signal_staleness",
        level="pass",
        message="signals recovered",
        now=datetime(2026, 7, 1, 0, 5, tzinfo=timezone.utc),
    )

    restored = json.loads(kill_switch_path.read_text())
    assert restored["source"] == "incident_lifecycle"
    assert restored["incident_id"] == warning.incident_id
    assert restored["incident_channel"] == "ic_decay"
    assert restored["level"] == "warning"


def test_signal_staleness_recovery_clears_incident_owned_kill_switch(tmp_path):
    kill_switch_path = tmp_path / "kill_switch.json"
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=kill_switch_path,
        escalation_cycles=1,
    )

    opened = manager.record_alert(
        channel="signal_staleness",
        level="halt",
        message="all required signals stale",
        details={"stale_signals": ["ensemble_voting", "garch_cvar"]},
        now=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
    )
    assert opened is not None
    assert kill_switch_path.exists()
    assert json.loads(kill_switch_path.read_text())["reason"] == "unresolved_incident:signal_staleness"

    resolved = manager.record_alert(
        channel="signal_staleness",
        level="pass",
        message="required signals fresh and optional daily sections classified",
        details={"stale_signals": [], "unavailable_signals": ["collar"]},
        now=datetime(2026, 7, 1, 0, 5, tzinfo=timezone.utc),
    )

    assert resolved is not None
    assert resolved.incident_id == opened.incident_id
    assert resolved.state == IncidentState.RESOLVED
    assert not kill_switch_path.exists()


def test_pass_alert_does_not_clear_non_incident_kill_switch(tmp_path):
    kill_switch_path = tmp_path / "kill_switch.json"
    kill_switch_path.write_text(json.dumps({
        "enabled": True,
        "level": "halt",
        "reason": "max_drawdown_-22.0%",
        "source": "risk_limits",
        "position_reduction": 1.0,
    }))
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=kill_switch_path,
        escalation_cycles=1,
    )

    manager.record_alert(
        channel="cron_failure",
        level="halt",
        message="cron failed",
        now=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
    )
    manager.record_alert(
        channel="cron_failure",
        level="pass",
        message="cron recovered",
        now=datetime(2026, 7, 1, 0, 5, tzinfo=timezone.utc),
    )

    data = json.loads(kill_switch_path.read_text())
    assert data["source"] == "risk_limits"
    assert data["reason"] == "max_drawdown_-22.0%"


def test_disabled_incident_escalation_does_not_write_kill_switch(tmp_path):
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=tmp_path / "kill_switch.json",
        escalation_cycles=1,
        escalation_enabled=False,
    )

    incident = manager.record_alert(
        channel="ic_decay",
        level="halt",
        message="critical IC decay",
        now=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
    )

    assert incident is not None
    assert not (tmp_path / "kill_switch.json").exists()


def test_write_summary_dual_writes_public_incidents(tmp_path, monkeypatch):
    """PASS resolve must dual-write PUBLIC_DATA_DIR so operators never see split-brain."""
    from src.monitor import incident_manager as im

    public = tmp_path / "www"
    public.mkdir()
    monkeypatch.setattr(im, "PUBLIC_DATA_DIR", public)

    private = tmp_path / "private"
    private.mkdir()
    manager = im.IncidentManager(
        log_path=private / "incidents.jsonl",
        summary_path=private / "incidents.json",
        kill_switch_path=private / "kill_switch.json",
        escalation_enabled=False,
    )
    manager.record_alert(
        channel="signal_staleness",
        level="warn",
        message="13/23 signals unavailable",
        now=datetime(2026, 7, 20, 17, 40, tzinfo=timezone.utc),
    )
    assert (private / "incidents.json").exists()
    assert (public / "incidents.json").exists()
    assert json.loads((public / "incidents.json").read_text())["open_count"] == 1

    manager.record_alert(
        channel="signal_staleness",
        level="pass",
        message="All required signals fresh",
        now=datetime(2026, 7, 20, 17, 50, tzinfo=timezone.utc),
    )
    priv = json.loads((private / "incidents.json").read_text())
    pub = json.loads((public / "incidents.json").read_text())
    assert priv["open_count"] == 0
    assert pub["open_count"] == 0


# ── Task 2A: manual-review-required incidents are not auto-resolved ────

def test_manual_review_required_incident_survives_pass_alert(tmp_path):
    """A PASS on a manual-review-required channel must not resolve the incident."""
    from datetime import datetime

    from src.monitor.incident_manager import IncidentManager, IncidentState
    from src.monitor.alerting import AlertChannel, AlertLevel

    def _utc_iso(value: str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=tmp_path / "kill_switch.json",
        escalation_cycles=1,
    )
    manager.record_alert(
        channel=AlertChannel.IC_DECAY,
        level=AlertLevel.HALT,
        message="4 signal(s) with CRITICAL IC decay",
        now=_utc_iso("2026-08-01T00:00:00Z"),
    )
    manager.record_alert(
        channel=AlertChannel.IC_DECAY,
        level=AlertLevel.HALT,
        message="4 signal(s) with CRITICAL IC decay",
        now=_utc_iso("2026-08-02T00:00:00Z"),
    )
    manager.record_alert(
        channel=AlertChannel.IC_DECAY,
        level=AlertLevel.HALT,
        message="4 signal(s) with CRITICAL IC decay",
        now=_utc_iso("2026-08-03T00:00:00Z"),
    )
    incidents = manager.open_incidents()
    assert len(incidents) == 1
    assert incidents[0].manual_review_required is True
    assert incidents[0].state == IncidentState.FIRING

    # A PASS (e.g. warm-up or recovered IC) must NOT auto-resolve it.
    manager.record_alert(
        channel=AlertChannel.IC_DECAY,
        level=AlertLevel.PASS,
        message="IC monitor warming up: 1 signal(s)",
        now=_utc_iso("2026-08-09T00:00:00Z"),
    )
    incidents = manager.open_incidents()
    assert len(incidents) == 1
    assert incidents[0].state == IncidentState.FIRING
    # Kill switch stays armed.
    kill = json.loads((tmp_path / "kill_switch.json").read_text(encoding="utf-8"))
    assert kill["enabled"] is True
    assert kill["level"] == "halt"


def test_legacy_ic_decay_incident_without_flag_survives_pass_alert(tmp_path):
    """An ic_decay incident opened before the manual-review flag existed (e.g.
    live incident 8115a9c1) must still not be auto-resolved by a PASS — the
    channel policy holds regardless of the persisted flag."""
    from datetime import datetime, timezone

    from src.monitor.incident_manager import IncidentManager, IncidentState
    from src.monitor.alerting import AlertChannel, AlertLevel

    log_path = tmp_path / "incidents.jsonl"
    kill_switch_path = tmp_path / "kill_switch.json"
    # Seed the event log with a legacy "opened" event (pre-flag schema):
    # manual_review_required absent -> False, exactly like the live incident
    # opened before Task 2A shipped.
    legacy = {
        "event": "opened",
        "event_timestamp": "2026-08-01T00:00:00+00:00",
        "incident_id": "8115a9c1",
        "channel": "ic_decay",
        "severity": "p0",
        "state": "firing",
        "message": "4 signal(s) with CRITICAL IC decay",
        "details": {},
        "created_at": "2026-08-01T00:00:00+00:00",
        "updated_at": "2026-08-01T00:00:00+00:00",
        "alert_count": 429,
        "kill_switch_level": "halt",
    }
    log_path.write_text(json.dumps(legacy, sort_keys=True) + "\n", encoding="utf-8")
    kill_switch_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "level": "halt",
                "reason": "unresolved_incident:ic_decay",
                "mode": "paper",
                "timestamp": "2026-08-01T00:00:00+00:00",
                "position_reduction": 1.0,
                "source": "incident_lifecycle",
                "incident_id": "8115a9c1",
                "incident_channel": "ic_decay",
                "incident_severity": "p0",
                "incident_alert_count": 429,
                "message": "4 signal(s) with CRITICAL IC decay",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manager = IncidentManager(
        log_path=log_path,
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=kill_switch_path,
        escalation_cycles=1,
    )
    incidents = manager.open_incidents()
    assert len(incidents) == 1
    assert incidents[0].incident_id == "8115a9c1"
    assert incidents[0].manual_review_required is False  # legacy flag absent

    # A PASS (e.g. control-ineligible disclosure) must NOT auto-resolve the
    # legacy incident: the ic_decay channel policy holds unconditionally.
    manager.record_alert(
        channel=AlertChannel.IC_DECAY,
        level=AlertLevel.PASS,
        message="IC decay critical but control-ineligible; no escalation",
        now=datetime(2026, 8, 9, 0, 0, tzinfo=timezone.utc),
    )
    incidents = manager.open_incidents()
    assert len(incidents) == 1
    assert incidents[0].incident_id == "8115a9c1"
    assert incidents[0].state == IncidentState.FIRING
    assert incidents[0].manual_review_required is True  # flag backfilled on hold
    # Kill switch stays armed (no resolution event -> no clear path).
    kill = json.loads(kill_switch_path.read_text(encoding="utf-8"))
    assert kill["enabled"] is True
    assert kill["level"] == "halt"
    # Hold event recorded; no fabricated resolution event.
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert events[-1]["event"] == "pass_held_for_manual_review"
    assert events[-1]["manual_review_reason"] == "ic_evidence_correction"


# ── Item 16 s2: operator resolution CLI path ─────────────────────────

def test_operator_resolve_appends_event_clears_kill_and_reloads_resolved(tmp_path):
    """Explicit operator resolution: journal append + kill clear + summary;
    a fresh manager replaying the journal sees the incident resolved."""
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=tmp_path / "kill_switch.json",
        escalation_cycles=1,  # first alert arms the kill switch immediately
    )
    opened = manager.record_alert(
        channel="ic_decay",
        level="halt",
        message="signal critical",
        now=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
    )
    assert opened is not None
    kill = json.loads((tmp_path / "kill_switch.json").read_text())
    assert kill["incident_id"] == opened.incident_id
    assert kill["level"] == "warning"  # escalation_cycles=1 → stage 1

    resolved = manager.resolve_operator(
        opened.incident_id,
        "operator reviewed IC evidence; resolved",
        now=datetime(2026, 7, 1, 0, 5, tzinfo=timezone.utc),
    )
    assert resolved is not None
    assert resolved.state == IncidentState.RESOLVED
    assert resolved.resolution_notes == "operator reviewed IC evidence; resolved"
    assert resolved.mttr_seconds == 300.0

    events = _read_jsonl(tmp_path / "incidents.jsonl")
    assert events[-1]["event"] == "resolved"
    assert events[-1]["state"] == "resolved"
    # Kill switch cleared by the resolution (incident_lifecycle-owned).
    assert not (tmp_path / "kill_switch.json").exists()

    summary = json.loads((tmp_path / "incidents.json").read_text())
    assert summary["open_count"] == 0

    # Replay from the journal in a fresh manager: resolved, 0 open.
    reloaded = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
        kill_switch_path=tmp_path / "kill_switch.json",
    )
    assert reloaded.open_incidents() == []
    assert reloaded.incident_state(opened.incident_id) == "resolved"


def test_operator_resolve_idempotent_and_unknown(tmp_path):
    manager = IncidentManager(
        log_path=tmp_path / "incidents.jsonl",
        summary_path=tmp_path / "incidents.json",
    )
    opened = manager.record_alert(
        channel="cron_failure",
        level="warn",
        message="cron failed",
        now=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc),
    )
    assert opened is not None

    first = manager.resolve_operator(opened.incident_id, "fixed manually")
    assert first is not None

    # Already resolved: no-op (no duplicate event, still resolved).
    second = manager.resolve_operator(opened.incident_id, "again")
    assert second is None
    events = _read_jsonl(tmp_path / "incidents.jsonl")
    assert [event["event"] for event in events].count("resolved") == 1
    assert manager.incident_state(opened.incident_id) == "resolved"

    # Unknown id: None + incident_state None (CLI refuses with exit 1).
    assert manager.resolve_operator("00000000-unknown", "nope") is None
    assert manager.incident_state("00000000-unknown") is None
