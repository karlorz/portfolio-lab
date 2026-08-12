"""Batch IM/IN: kill write-through fan-out, PASS clear, advisory hysteresis, restamp.

Session A plan (IM DN–DP + IN DN3):
- DN: kill arm/clear must project mon/ops/public within the same call
- DO: live PASS (intentional gaps only) resolves open + clears kill surfaces
- DP: sole stale=alternative_data (advisory_shadow) must not WARN/escalate
- DN3: soft-mirror lag restamp must re-project kill/open from disk

Authority: never touches target_allocations / order_router.
"""

from __future__ import annotations

import json
from pathlib import Path



def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sticky_clear_monitor(ts: str = "2026-07-23T13:30:00+00:00") -> dict:
    return {
        "status": "ok",
        "timestamp": ts,
        "scope": "operational_readiness",
        "service": "portfolio-lab",
        "checks": {
            "circuit_breaker": {"status": "ok", "state": "closed"},
            "kill_switch": {"status": "ok", "enabled": False, "level": None},
            "open_incidents": {"status": "ok", "open_count": 0, "incidents": []},
            "data_freshness": {
                "prices": {"status": "ok", "age_hours": 0.1},
                "signals": {"status": "ok", "age_hours": 0.1},
            },
        },
    }


def _sticky_armed_monitor(
    *,
    incident_id: str = "inc-dn",
    ts: str = "2026-07-23T13:30:00+00:00",
) -> dict:
    return {
        "status": "warning",
        "timestamp": ts,
        "scope": "operational_readiness",
        "service": "portfolio-lab",
        "checks": {
            "circuit_breaker": {"status": "ok", "state": "closed"},
            "kill_switch": {
                "status": "warning",
                "enabled": True,
                "level": "warning",
                "reason": "unresolved_incident:signal_staleness",
                "source": "incident_lifecycle",
                "incident_id": incident_id,
                "message": "stale alternative_data",
            },
            "open_incidents": {
                "status": "warning",
                "open_count": 1,
                "incidents": [
                    {
                        "incident_id": incident_id,
                        "channel": "signal_staleness",
                        "severity": "p2",
                        "state": "firing",
                        "message": "stale alternative_data",
                        "kill_switch_level": "warning",
                    }
                ],
            },
        },
    }


def _sticky_public_dashboard(*, armed: bool, incident_id: str = "inc-dn") -> dict:
    if armed:
        return {
            "system_status": "warning",
            "generated_at": "2026-07-23T13:30:00+00:00",
            "kill_switch": {
                "status": "warning",
                "enabled": True,
                "level": "warning",
                "incident_id": incident_id,
                "source": "incident_lifecycle",
            },
            "open_incidents": {
                "status": "warning",
                "open_count": 1,
                "incidents": [
                    {
                        "incident_id": incident_id,
                        "channel": "signal_staleness",
                        "state": "firing",
                    }
                ],
            },
            "repo_public_mirror_lagging_count": 0,
            "repo_public_mirror_lag_status": "ok",
            "repo_public_mirror_lag": {"lagging_count": 0, "status": "ok"},
        }
    return {
        "system_status": "healthy",
        "generated_at": "2026-07-23T13:30:00+00:00",
        "kill_switch": {"status": "ok", "enabled": False, "level": None},
        "open_incidents": {"status": "ok", "open_count": 0, "incidents": []},
        "repo_public_mirror_lagging_count": 0,
        "repo_public_mirror_lag_status": "ok",
        "repo_public_mirror_lag": {"lagging_count": 0, "status": "ok"},
    }


def _seed_surfaces(
    data: Path,
    public: Path,
    *,
    mon: dict,
    ops: dict | None = None,
    pub: dict | None = None,
) -> None:
    _write_json(data / "health.json", mon)
    _write_json(data / "health_ops.json", ops if ops is not None else mon)
    _write_json(public / "health_ops.json", ops if ops is not None else mon)
    _write_json(
        public / "health.json",
        pub if pub is not None else _sticky_public_dashboard(armed=False),
    )


# ---------------------------------------------------------------------------
# DN — kill arm/clear write-through
# ---------------------------------------------------------------------------


def test_dn1_kill_arm_projects_mon_ops_public(tmp_path, monkeypatch) -> None:
    """DN1: IncidentManager escalate arm → mon/ops/pub kill.enabled within call."""
    from src.monitor.health_check import project_disk_kill_open_to_all_surfaces
    from src.monitor.incident_manager import IncidentManager

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    _seed_surfaces(
        data,
        public,
        mon=_sticky_clear_monitor(),
        pub=_sticky_public_dashboard(armed=False),
    )

    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.incident_manager.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.incident_manager.PUBLIC_DATA_DIR", public, raising=False)

    mgr = IncidentManager(
        log_path=data / "incidents.jsonl",
        summary_path=data / "incidents.json",
        kill_switch_path=data / "kill_switch.json",
        escalation_cycles=1,
        escalation_enabled=True,
    )
    # Force write-through after arm
    incident = mgr.record_alert(
        channel="signal_staleness",
        level="warn",
        message="1/23 signals stale: ensemble_voting",
        details={"stale_signals": ["ensemble_voting"]},
    )
    assert incident is not None
    assert (data / "kill_switch.json").exists()
    kill_disk = json.loads((data / "kill_switch.json").read_text(encoding="utf-8"))
    assert kill_disk.get("enabled") is True

    # Explicit projector (also invoked by write-through on manager)
    project_disk_kill_open_to_all_surfaces(data_dir=data, public_dir=public)

    mon = json.loads((data / "health.json").read_text(encoding="utf-8"))
    ops = json.loads((public / "health_ops.json").read_text(encoding="utf-8"))
    pub = json.loads((public / "health.json").read_text(encoding="utf-8"))
    assert mon["checks"]["kill_switch"].get("enabled") is True
    assert ops["checks"]["kill_switch"].get("enabled") is True
    assert pub["kill_switch"].get("enabled") is True
    assert mon["checks"]["open_incidents"].get("open_count") == 1
    assert ops["checks"]["open_incidents"].get("open_count") == 1
    assert pub["open_incidents"].get("open_count") == 1


def test_dn2_kill_clear_projects_all_surfaces(tmp_path, monkeypatch) -> None:
    """DN2: PASS resolve clears kill on mon/ops/pub."""
    from src.monitor.health_check import project_disk_kill_open_to_all_surfaces
    from src.monitor.incident_manager import IncidentManager

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()

    mgr = IncidentManager(
        log_path=data / "incidents.jsonl",
        summary_path=data / "incidents.json",
        kill_switch_path=data / "kill_switch.json",
        escalation_cycles=1,
        escalation_enabled=True,
    )
    opened = mgr.record_alert(
        channel="signal_staleness",
        level="warn",
        message="stale ensemble_voting",
        details={"stale_signals": ["ensemble_voting"]},
    )
    assert opened is not None
    incident_id = opened.incident_id

    _seed_surfaces(
        data,
        public,
        mon=_sticky_armed_monitor(incident_id=incident_id),
        pub=_sticky_public_dashboard(armed=True, incident_id=incident_id),
    )
    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.incident_manager.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.incident_manager.PUBLIC_DATA_DIR", public, raising=False)

    mgr.record_alert(
        channel="signal_staleness",
        level="pass",
        message="All required signals fresh",
        details={},
    )
    assert not (data / "kill_switch.json").exists()

    project_disk_kill_open_to_all_surfaces(data_dir=data, public_dir=public)

    mon = json.loads((data / "health.json").read_text(encoding="utf-8"))
    ops = json.loads((public / "health_ops.json").read_text(encoding="utf-8"))
    pub = json.loads((public / "health.json").read_text(encoding="utf-8"))
    assert mon["checks"]["kill_switch"].get("enabled") in (False, None, 0)
    assert ops["checks"]["kill_switch"].get("enabled") in (False, None, 0)
    assert pub["kill_switch"].get("enabled") in (False, None, 0)
    assert mon["checks"]["open_incidents"].get("open_count") == 0
    assert ops["checks"]["open_incidents"].get("open_count") == 0
    assert pub["open_incidents"].get("open_count") == 0


def test_dn_write_through_on_incident_manager(tmp_path, monkeypatch) -> None:
    """DN: IncidentManager.record_alert itself fans out without external call."""
    from src.monitor.incident_manager import IncidentManager

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    _seed_surfaces(
        data,
        public,
        mon=_sticky_clear_monitor(),
        pub=_sticky_public_dashboard(armed=False),
    )

    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.incident_manager.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.incident_manager.PUBLIC_DATA_DIR", public, raising=False)

    mgr = IncidentManager(
        log_path=data / "incidents.jsonl",
        summary_path=data / "incidents.json",
        kill_switch_path=data / "kill_switch.json",
        escalation_cycles=1,
        escalation_enabled=True,
    )
    mgr.record_alert(
        channel="signal_staleness",
        level="warn",
        message="stale ensemble_voting",
        details={"stale_signals": ["ensemble_voting"]},
    )

    mon = json.loads((data / "health.json").read_text(encoding="utf-8"))
    ops = json.loads((public / "health_ops.json").read_text(encoding="utf-8"))
    pub = json.loads((public / "health.json").read_text(encoding="utf-8"))
    assert mon["checks"]["kill_switch"].get("enabled") is True
    assert ops["checks"]["kill_switch"].get("enabled") is True
    assert pub["kill_switch"].get("enabled") is True


# ---------------------------------------------------------------------------
# DO — PASS clear mid-cycle
# ---------------------------------------------------------------------------


def test_do1_pass_clears_sticky_open_and_kill(tmp_path, monkeypatch) -> None:
    """DO1: check_staleness_and_alert PASS clears open + kill surfaces."""
    from src.monitor import alerting
    from src.monitor.health_check import project_disk_kill_open_to_all_surfaces
    from src.monitor.incident_manager import IncidentManager

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()

    mgr = IncidentManager(
        log_path=data / "incidents.jsonl",
        summary_path=data / "incidents.json",
        kill_switch_path=data / "kill_switch.json",
        escalation_cycles=1,
        escalation_enabled=True,
    )
    opened = mgr.record_alert(
        channel="signal_staleness",
        level="warn",
        message="stale ensemble_voting",
        details={"stale_signals": ["ensemble_voting"]},
    )
    assert opened is not None
    _seed_surfaces(
        data,
        public,
        mon=_sticky_armed_monitor(incident_id=opened.incident_id),
        pub=_sticky_public_dashboard(armed=True, incident_id=opened.incident_id),
    )

    monkeypatch.setattr(alerting, "_incident_manager", mgr)
    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr("src.monitor.incident_manager.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.incident_manager.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)

    # Live PASS: intentional FRED gaps only, no actionable stale
    alerting.check_staleness_and_alert(
        {
            "stale_signals": [],
            "unavailable_signals": [
                "two_stage_regime",
                "regime_transition",
                "fred_macro",
            ],
            "unavailable_ownership": [
                {
                    "signal": "two_stage_regime",
                    "intentional_when_fred_unconfigured": True,
                    "intentional_lab_gap": True,
                },
                {
                    "signal": "regime_transition",
                    "intentional_when_fred_unconfigured": True,
                    "intentional_lab_gap": True,
                },
                {
                    "signal": "fred_macro",
                    "intentional_when_fred_unconfigured": True,
                    "intentional_lab_gap": True,
                },
            ],
            "healthy_count": 20,
            "total_count": 23,
        }
    )

    assert not (data / "kill_switch.json").exists()
    incidents = json.loads((data / "incidents.json").read_text(encoding="utf-8"))
    assert incidents.get("open_count") == 0

    project_disk_kill_open_to_all_surfaces(data_dir=data, public_dir=public)
    mon = json.loads((data / "health.json").read_text(encoding="utf-8"))
    assert mon["checks"]["kill_switch"].get("enabled") in (False, None, 0)
    assert mon["checks"]["open_incidents"].get("open_count") == 0


# ---------------------------------------------------------------------------
# DP — advisory sole-stale must not WARN
# ---------------------------------------------------------------------------


def test_dp1_sole_advisory_alt_data_stale_is_pass() -> None:
    """DP1: sole stale=alternative_data (advisory_shadow) → PASS, not WARN."""
    from src.monitor.alerting import AlertLevel, classify_signal_staleness

    result = classify_signal_staleness(
        {
            "stale_signals": ["alternative_data"],
            "unavailable_signals": [],
            "healthy_count": 22,
            "total_count": 23,
            "signal_roles": {"alternative_data": "advisory_shadow"},
        }
    )
    assert result is not None
    level, message, details = result
    assert level == AlertLevel.PASS
    assert "advisory" in str(details.get("policy") or "").lower() or "alternative" in message.lower()


def test_dp2_required_stale_still_warns() -> None:
    """DP2: required missing/stale still WARNs and can escalate."""
    from src.monitor.alerting import AlertLevel, classify_signal_staleness

    result = classify_signal_staleness(
        {
            "stale_signals": ["ensemble_voting", "alternative_data"],
            "unavailable_signals": [],
            "healthy_count": 21,
            "total_count": 23,
            "signal_roles": {"alternative_data": "advisory_shadow"},
        }
    )
    assert result is not None
    level, message, details = result
    assert level == AlertLevel.WARN
    assert "ensemble_voting" in message
    # advisory may be filtered from message but required must remain
    assert "ensemble_voting" in (details.get("actionable_stale") or details.get("stale_signals") or [])


def test_dp1b_sole_alt_without_roles_key_still_advisory() -> None:
    """DP: alternative_data is advisory by default even without signal_roles."""
    from src.monitor.alerting import AlertLevel, classify_signal_staleness

    result = classify_signal_staleness(
        {
            "stale_signals": ["alternative_data"],
            "unavailable_signals": [
                "two_stage_regime",
            ],
            "unavailable_ownership": [
                {
                    "signal": "two_stage_regime",
                    "intentional_when_fred_unconfigured": True,
                    "intentional_lab_gap": True,
                }
            ],
            "healthy_count": 21,
            "total_count": 23,
        }
    )
    assert result is not None
    level, _message, details = result
    assert level == AlertLevel.PASS
    assert details.get("policy") in {
        "advisory_shadow_stale_only_pass",
        "intentional_lab_gaps_only_pass",
        "advisory_or_intentional_only_pass",
    }


# ---------------------------------------------------------------------------
# DN3 — restamp honesty
# ---------------------------------------------------------------------------


def test_dn3_restamp_reprojects_kill_from_disk(tmp_path, monkeypatch) -> None:
    """DN3: lag restamp of sticky-armed health must clear when disk kill gone."""
    from src.monitor.repo_public_mirror_lag import restamp_mirror_lag_on_health_documents

    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()

    # Disk SSOT clear
    _write_json(data / "incidents.json", {"open_count": 0, "incidents": []})
    # no kill_switch.json

    sticky = _sticky_armed_monitor(incident_id="sticky-old")
    sticky["repo_public_mirror_lagging_count"] = 1
    sticky["repo_public_mirror_lag_status"] = "warning"
    sticky["repo_public_mirror_lag"] = {
        "lagging_count": 1,
        "status": "warning",
        "paths": ["alerts.json"],
    }
    sticky["repo_public_mirror_source"] = str(public)
    sticky["repo_public_mirror_dest"] = str(tmp_path / "repo")
    _write_json(data / "health.json", sticky)
    _write_json(public / "health_ops.json", sticky)

    pub = _sticky_public_dashboard(armed=True, incident_id="sticky-old")
    pub["repo_public_mirror_lagging_count"] = 1
    pub["repo_public_mirror_lag_status"] = "warning"
    pub["repo_public_mirror_lag"] = {
        "lagging_count": 1,
        "status": "warning",
        "paths": ["alerts.json"],
    }
    _write_json(public / "health.json", pub)

    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", data, raising=False)
    monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False)
    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.DATA_DIR", data, raising=False
    )

    lag_summary = {
        "lagging_count": 0,
        "total": 36,
        "lagging_paths": [],
        "status": "ok",
        "badge": "ok",
        "source": str(public),
        "dest": str(tmp_path / "repo"),
        "policy": "test",
    }
    result = restamp_mirror_lag_on_health_documents(
        paths=[data / "health.json", public / "health_ops.json", public / "health.json"],
        lag_summary=lag_summary,
        source_root=public,
        dest_root=tmp_path / "repo",
    )
    assert result["restamped"] or result.get("errors") == []

    mon = json.loads((data / "health.json").read_text(encoding="utf-8"))
    ops = json.loads((public / "health_ops.json").read_text(encoding="utf-8"))
    pub_after = json.loads((public / "health.json").read_text(encoding="utf-8"))

    # Lag restamped AND kill/open match disk (clear)
    assert mon["checks"]["kill_switch"].get("enabled") in (False, None, 0)
    assert mon["checks"]["open_incidents"].get("open_count") == 0
    assert ops["checks"]["kill_switch"].get("enabled") in (False, None, 0)
    assert ops["checks"]["open_incidents"].get("open_count") == 0
    assert pub_after["kill_switch"].get("enabled") in (False, None, 0)
    assert pub_after["open_incidents"].get("open_count") == 0
    # lag fields still present
    assert "mirror_lag_restamped_at" in mon or mon.get("repo_public_mirror_lagging_count") == 0
