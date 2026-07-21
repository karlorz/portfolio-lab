"""Tests for portfolio-lab-prod-ideas: scan SSOT → channel delta + dual-mode wiring.

Drives the shipped scan/apply helpers with fixture SSOT dicts (not reimplementations).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.monitor import prod_ideas as pi


NOW = datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z") if dt.tzinfo else dt.isoformat() + "Z"


# ── scan_ssot intake filters ─────────────────────────────────────────────────


def test_kill_enabled_opens_kill_channel():
    snapshot = {
        "kill_switch": {
            "enabled": True,
            "level": "restrict",
            "reason": "manual_halt_test",
            "message": "operator armed kill",
            "timestamp": _iso(NOW),
            "incident_id": None,
            "channel": None,
        },
        "incidents": {"open_count": 0, "incidents": []},
        "cron_status": {"jobs": []},
        "health": {"system_status": "ok", "scheduler_status": {"status": "ok", "backends": {}}},
        "alerts": {"alerts": []},
        "tasker_state": [],
        "rebalance": {"hold_reason": "budget_exceeded"},
        "fred_readiness": {"reason": "missing_fred_api_key", "ready": False},
    }
    obs = pi.scan_ssot(snapshot)
    keys = {o["key"] for o in obs}
    assert "kill_switch" in keys
    kill = next(o for o in obs if o["key"] == "kill_switch")
    assert kill["channel"] == "kill_switch"
    assert kill["fingerprint"]
    assert kill["evidence"]["level"] == "restrict"
    assert "budget" not in keys
    assert not any(k.startswith("fred") for k in keys)


def test_same_fingerprint_twice_one_row_count_bump():
    snapshot = {
        "kill_switch": {
            "enabled": True,
            "level": "warning",
            "reason": "test",
            "message": "m",
            "timestamp": _iso(NOW),
        },
        "incidents": {"open_count": 0, "incidents": []},
        "cron_status": {"jobs": []},
        "health": {"system_status": "ok", "scheduler_status": {"status": "ok", "backends": {}}},
        "alerts": {"alerts": []},
        "tasker_state": [],
    }
    obs = pi.scan_ssot(snapshot)
    t0 = NOW
    state, events1 = pi.apply_channel_delta({}, obs, now=t0)
    assert len(state["channels"]) == 1
    assert state["channels"][0]["observation_count"] == 1
    assert any(e["type"] == "first_open" for e in events1)

    t1 = t0 + timedelta(hours=1)
    state2, events2 = pi.apply_channel_delta(state, obs, now=t1)
    assert len(state2["channels"]) == 1
    row = state2["channels"][0]
    assert row["observation_count"] == 2
    assert row["last_seen"] == pi._ts(t1)
    assert row["first_seen"] == pi._ts(t0)
    assert not any(e["type"] == "first_open" for e in events2)


def test_intentional_fred_only_unavail_no_channel():
    snapshot = {
        "kill_switch": {"enabled": False},
        "incidents": {"open_count": 0, "incidents": []},
        "cron_status": {"jobs": []},
        "health": {
            "system_status": "degraded",
            "scheduler_status": {"status": "ok", "backends": {"local": {"failed_jobs": 0}}},
            "fred_readiness": {
                "status": "warning",
                "reason": "missing_fred_api_key",
                "ready": False,
                "api_key_configured": False,
            },
            "signal_health": {
                "summary": {"degraded": 6, "unhealthy": 3},
            },
        },
        "alerts": {
            "alerts": [
                {
                    "level": "warning",
                    "type": "garch_tail",
                    "title": "GARCH tail warning",
                    "message": "tail risk elevated",
                }
            ]
        },
        "tasker_state": [],
        "rebalance": {"status": "hold", "hold_reason": "budget_overage", "budget_remaining": 0},
    }
    obs = pi.scan_ssot(snapshot)
    assert obs == []


def test_budget_only_rebalance_no_channel():
    snapshot = {
        "kill_switch": {"enabled": False},
        "incidents": {"open_count": 0, "incidents": []},
        "cron_status": {"jobs": []},
        "health": {"system_status": "ok", "scheduler_status": {"status": "ok", "backends": {}}},
        "alerts": {"alerts": []},
        "tasker_state": [],
        "rebalance": {
            "status": "hold",
            "hold_reason": "budget_exceeded",
            "budget_used_pct": 1.2,
        },
    }
    assert pi.scan_ssot(snapshot) == []


def test_cron_eval_error_opens_cron_failed_for_job():
    snapshot = {
        "kill_switch": {"enabled": False},
        "incidents": {"open_count": 0, "incidents": []},
        "cron_status": {
            "jobs": [
                {
                    "name": "portfolio-lab-eval",
                    "enabled": True,
                    "manual_only": False,
                    "status": "error",
                    "state": "scheduled",
                    "last_run": _iso(NOW),
                    "duration_seconds": 12.0,
                    "backend": "tasker",
                },
                {
                    "name": "portfolio-lab-build",
                    "enabled": False,
                    "manual_only": True,
                    "status": "disabled",
                    "state": "manual_only",
                },
            ]
        },
        "health": {"system_status": "ok", "scheduler_status": {"status": "ok", "backends": {}}},
        "alerts": {"alerts": []},
        "tasker_state": [],
    }
    obs = pi.scan_ssot(snapshot)
    keys = {o["key"] for o in obs}
    assert "cron_failed:portfolio-lab-eval" in keys
    assert "cron_failed:portfolio-lab-build" not in keys
    row = next(o for o in obs if o["key"] == "cron_failed:portfolio-lab-eval")
    assert row["evidence"]["job_name"] == "portfolio-lab-eval"
    assert row["evidence"]["status"] == "error"


def test_open_incident_opens_incident_channel():
    snapshot = {
        "kill_switch": {"enabled": False},
        "incidents": {
            "open_count": 1,
            "incidents": [
                {
                    "incident_id": "inc-1",
                    "channel": "evaluator_error",
                    "severity": "p1",
                    "state": "open",
                    "message": "eval failed 3x",
                    "kill_switch_level": "warning",
                }
            ],
        },
        "cron_status": {"jobs": []},
        "health": {"system_status": "ok", "scheduler_status": {"status": "ok", "backends": {}}},
        "alerts": {"alerts": []},
        "tasker_state": [],
    }
    obs = pi.scan_ssot(snapshot)
    keys = {o["key"] for o in obs}
    assert "incident:evaluator_error" in keys


def test_health_critical_and_scheduler_failed_jobs_open_channels():
    snapshot = {
        "kill_switch": {"enabled": False},
        "incidents": {"open_count": 0, "incidents": []},
        "cron_status": {"jobs": []},
        "health": {
            "system_status": "critical",
            "scheduler_status": {
                "status": "error",
                "backends": {
                    "local": {
                        "backend": "tasker",
                        "status": "error",
                        "failed_jobs": 2,
                        "source": "data/cron_status.json",
                    }
                },
            },
        },
        "alerts": {
            "alerts": [
                {
                    "level": "error",
                    "type": "health_slo",
                    "title": "Health Critical",
                    "message": "system_status=critical",
                    "scheduler_status": "error",
                }
            ]
        },
        "tasker_state": [],
    }
    obs = pi.scan_ssot(snapshot)
    keys = {o["key"] for o in obs}
    assert "health_critical" in keys
    assert "scheduler_failed" in keys


def test_tasker_consecutive_failures_open_channel():
    snapshot = {
        "kill_switch": {"enabled": False},
        "incidents": {"open_count": 0, "incidents": []},
        "cron_status": {"jobs": []},
        "health": {"system_status": "ok", "scheduler_status": {"status": "ok", "backends": {}}},
        "alerts": {"alerts": []},
        "tasker_state": [
            {
                "task_id": "portfolio-lab-data",
                "enabled": True,
                "consecutive_failures": 3,
                "last_status": "error",
                "last_exit_code": 1,
                "last_run_id": "run-abc",
                "log_path": "data/tasker_logs/run-abc.log",
            },
            {
                "task_id": "portfolio-lab-health",
                "enabled": True,
                "consecutive_failures": 0,
                "last_status": "success",
            },
        ],
    }
    obs = pi.scan_ssot(snapshot)
    keys = {o["key"] for o in obs}
    assert "tasker_failed:portfolio-lab-data" in keys
    assert "tasker_failed:portfolio-lab-health" not in keys
    row = next(o for o in obs if o["key"] == "tasker_failed:portfolio-lab-data")
    # relative path only — no absolute scrape
    assert row["evidence"]["log_path"] == "data/tasker_logs/run-abc.log"
    assert "log_tail" not in row["evidence"]


# ── lifecycle: clear + re-arm promote ────────────────────────────────────────


def test_clear_path_and_kill_rearm_sets_promote_candidate_without_planned_work():
    kill_on = {
        "kill_switch": {
            "enabled": True,
            "level": "restrict",
            "reason": "incident:x",
            "message": "armed",
            "timestamp": _iso(NOW),
        },
        "incidents": {"open_count": 0, "incidents": []},
        "cron_status": {"jobs": []},
        "health": {"system_status": "ok", "scheduler_status": {"status": "ok", "backends": {}}},
        "alerts": {"alerts": []},
        "tasker_state": [],
    }
    kill_off = {
        "kill_switch": {"enabled": False},
        "incidents": {"open_count": 0, "incidents": []},
        "cron_status": {"jobs": []},
        "health": {"system_status": "ok", "scheduler_status": {"status": "ok", "backends": {}}},
        "alerts": {"alerts": []},
        "tasker_state": [],
    }

    t0 = NOW
    state, _ = pi.apply_channel_delta({}, pi.scan_ssot(kill_on), now=t0)
    assert state["channels"][0]["status"] == "open"

    t1 = t0 + timedelta(hours=2)
    state, events = pi.apply_channel_delta(state, pi.scan_ssot(kill_off), now=t1)
    row = state["channels"][0]
    assert row["status"] == "cleared"
    assert any(e["type"] == "cleared" for e in events)

    # re-arm within 24h → promote_candidate badge only
    t2 = t1 + timedelta(hours=1)
    state, events = pi.apply_channel_delta(state, pi.scan_ssot(kill_on), now=t2)
    row = next(c for c in state["channels"] if c["key"] == "kill_switch")
    assert row["status"] == "open"
    assert row["promote_candidate"] is True
    assert "kill_rearm_within_24h" in row["promote_reasons"]
    assert not any("planned" in str(e).lower() for e in events)
    assert "work_item" not in json.dumps(state)


def test_fingerprint_change_updates_record_story():
    base = {
        "kill_switch": {
            "enabled": True,
            "level": "warning",
            "reason": "r1",
            "message": "m1",
            "timestamp": _iso(NOW),
        },
        "incidents": {"open_count": 0, "incidents": []},
        "cron_status": {"jobs": []},
        "health": {"system_status": "ok", "scheduler_status": {"status": "ok", "backends": {}}},
        "alerts": {"alerts": []},
        "tasker_state": [],
    }
    changed = dict(base)
    changed["kill_switch"] = {
        "enabled": True,
        "level": "halt",
        "reason": "r2",
        "message": "m2",
        "timestamp": _iso(NOW + timedelta(hours=1)),
    }
    state, _ = pi.apply_channel_delta({}, pi.scan_ssot(base), now=NOW)
    fp1 = state["channels"][0]["fingerprint"]
    state2, events = pi.apply_channel_delta(
        state, pi.scan_ssot(changed), now=NOW + timedelta(hours=1)
    )
    row = state2["channels"][0]
    assert row["fingerprint"] != fp1
    assert row["evidence"]["level"] == "halt"
    assert any(e["type"] == "fingerprint_change" for e in events)


def test_promote_candidate_after_open_ge_6h():
    snapshot = {
        "kill_switch": {
            "enabled": True,
            "level": "warning",
            "reason": "sticky",
            "message": "m",
            "timestamp": _iso(NOW),
        },
        "incidents": {"open_count": 0, "incidents": []},
        "cron_status": {"jobs": []},
        "health": {"system_status": "ok", "scheduler_status": {"status": "ok", "backends": {}}},
        "alerts": {"alerts": []},
        "tasker_state": [],
    }
    state, _ = pi.apply_channel_delta({}, pi.scan_ssot(snapshot), now=NOW)
    assert state["channels"][0]["promote_candidate"] is False

    later = NOW + timedelta(hours=7)
    state, events = pi.apply_channel_delta(state, pi.scan_ssot(snapshot), now=later)
    row = state["channels"][0]
    assert row["promote_candidate"] is True
    assert "open_ge_6h" in row["promote_reasons"]
    assert any(e["type"] == "promote_badge_flip" for e in events)


def test_promote_on_three_consecutive_failures():
    def snap(n: int) -> dict:
        return {
            "kill_switch": {"enabled": False},
            "incidents": {"open_count": 0, "incidents": []},
            "cron_status": {
                "jobs": [
                    {
                        "name": "portfolio-lab-eval",
                        "enabled": True,
                        "manual_only": False,
                        "status": "error",
                        "state": "scheduled",
                        "last_run": _iso(NOW),
                    }
                ]
            },
            "health": {
                "system_status": "ok",
                "scheduler_status": {"status": "ok", "backends": {}},
            },
            "alerts": {"alerts": []},
            "tasker_state": [
                {
                    "task_id": "portfolio-lab-eval",
                    "enabled": True,
                    "consecutive_failures": n,
                    "last_status": "error",
                    "last_exit_code": 1,
                    "log_path": "data/tasker_logs/run-x.log",
                }
            ],
        }

    state, _ = pi.apply_channel_delta({}, pi.scan_ssot(snap(1)), now=NOW)
    # cron channel may open; promote not yet from consecutive alone if <3
    cron_rows = [c for c in state["channels"] if c["key"].startswith("cron_failed:")]
    assert cron_rows
    assert cron_rows[0]["promote_candidate"] is False

    state2, _ = pi.apply_channel_delta(state, pi.scan_ssot(snap(3)), now=NOW + timedelta(hours=1))
    # tasker_failed channel should promote
    tasker = next(c for c in state2["channels"] if c["key"].startswith("tasker_failed:"))
    assert tasker["promote_candidate"] is True
    assert "consecutive_failures_ge_3" in tasker["promote_reasons"]


def test_payload_hygiene_no_log_bodies_or_prices():
    snapshot = {
        "kill_switch": {
            "enabled": True,
            "level": "halt",
            "reason": "manual",
            "message": "stop trading",
            "timestamp": _iso(NOW),
        },
        "incidents": {"open_count": 0, "incidents": []},
        "cron_status": {"jobs": []},
        "health": {
            "system_status": "critical",
            "scheduler_status": {
                "status": "error",
                "backends": {"local": {"failed_jobs": 1, "status": "error"}},
            },
            "unavailable_names": ["fred", "ml_regime"],
        },
        "alerts": {
            "alerts": [
                {"level": "error", "type": "health_slo", "title": "crit", "message": "x"}
            ]
        },
        "tasker_state": [],
    }
    obs = pi.scan_ssot(snapshot)
    blob = json.dumps(obs)
    assert "log_tail" not in blob
    assert "NAV" not in blob
    assert "price" not in blob.lower() or "prices" not in blob  # no price series
    # unavailable names allowed as sorted names only
    health = next(o for o in obs if o["key"] == "health_critical")
    assert health["evidence"].get("unavailable_names") == ["fred", "ml_regime"]


def test_apply_never_writes_planned_work_fields():
    snapshot = {
        "kill_switch": {
            "enabled": True,
            "level": "warning",
            "reason": "r",
            "message": "m",
            "timestamp": _iso(NOW),
        },
        "incidents": {"open_count": 0, "incidents": []},
        "cron_status": {"jobs": []},
        "health": {"system_status": "ok", "scheduler_status": {"status": "ok", "backends": {}}},
        "alerts": {"alerts": []},
        "tasker_state": [],
    }
    state, events = pi.apply_channel_delta({}, pi.scan_ssot(snapshot), now=NOW)
    dumped = json.dumps({"state": state, "events": events})
    assert "status: planned" not in dumped
    assert '"status": "planned"' not in dumped
    assert "work/" not in dumped
    assert "projects/portfolio-lab/work" not in dumped


# ── I/O loaders + writer (fixture paths) ─────────────────────────────────────


def test_load_snapshot_and_write_channels(tmp_path: Path):
    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    (data / "kill_switch.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "level": "restrict",
                "reason": "test",
                "message": "armed",
                "timestamp": _iso(NOW),
            }
        ),
        encoding="utf-8",
    )
    (data / "incidents.json").write_text(
        json.dumps({"open_count": 0, "incidents": []}), encoding="utf-8"
    )
    (data / "cron_status.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "portfolio-lab-eval",
                        "enabled": True,
                        "manual_only": False,
                        "status": "error",
                        "state": "scheduled",
                        "last_run": _iso(NOW),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (public / "health.json").write_text(
        json.dumps(
            {
                "system_status": "ok",
                "scheduler_status": {"status": "ok", "backends": {}},
            }
        ),
        encoding="utf-8",
    )
    (public / "alerts.json").write_text(json.dumps({"alerts": []}), encoding="utf-8")

    snap = pi.load_ssot_snapshot(
        data_dir=data,
        public_data_dir=public,
        tasker_db=None,
        now=NOW,
    )
    assert snap["kill_switch"]["enabled"] is True
    obs = pi.scan_ssot(snap)
    assert any(o["key"] == "kill_switch" for o in obs)
    assert any(o["key"] == "cron_failed:portfolio-lab-eval" for o in obs)

    out = data / "prod_idea_channels.json"
    state, events = pi.apply_channel_delta({}, obs, now=NOW)
    pi.write_channel_ssot(out, state, now=NOW)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == pi.SCHEMA_VERSION
    assert loaded["channels"]
    for ch in loaded["channels"]:
        assert "key" in ch
        assert "fingerprint" in ch
        assert "status" in ch
        assert "promote_candidate" in ch
        assert "evidence" in ch


def test_run_once_updates_machine_json_even_if_vault_unavailable(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    (data / "kill_switch.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "level": "warning",
                "reason": "r",
                "message": "m",
                "timestamp": _iso(NOW),
            }
        ),
        encoding="utf-8",
    )
    (data / "incidents.json").write_text(
        json.dumps({"open_count": 0, "incidents": []}), encoding="utf-8"
    )
    (data / "cron_status.json").write_text(json.dumps({"jobs": []}), encoding="utf-8")
    (public / "health.json").write_text(
        json.dumps(
            {"system_status": "ok", "scheduler_status": {"status": "ok", "backends": {}}}
        ),
        encoding="utf-8",
    )
    (public / "alerts.json").write_text(json.dumps({"alerts": []}), encoding="utf-8")

    def boom(*_a, **_k):
        raise OSError("vault fuse unavailable")

    monkeypatch.setattr(pi, "maybe_write_sparse_vault_note", boom)

    result = pi.run_once(
        data_dir=data,
        public_data_dir=public,
        tasker_db=None,
        vault_dir=tmp_path / "missing-vault",
        now=NOW,
    )
    assert result["ok"] is True
    assert (data / "prod_idea_channels.json").is_file()
    assert result["channels_open"] >= 1


def test_sparse_vault_note_only_on_first_open(tmp_path: Path):
    data = tmp_path / "data"
    public = tmp_path / "public"
    vault = tmp_path / "wiki"
    transcripts = vault / "raw" / "transcripts"
    transcripts.mkdir(parents=True)
    data.mkdir()
    public.mkdir()
    (data / "kill_switch.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "level": "warning",
                "reason": "r",
                "message": "m",
                "timestamp": _iso(NOW),
            }
        ),
        encoding="utf-8",
    )
    (data / "incidents.json").write_text(
        json.dumps({"open_count": 0, "incidents": []}), encoding="utf-8"
    )
    (data / "cron_status.json").write_text(json.dumps({"jobs": []}), encoding="utf-8")
    (public / "health.json").write_text(
        json.dumps(
            {"system_status": "ok", "scheduler_status": {"status": "ok", "backends": {}}}
        ),
        encoding="utf-8",
    )
    (public / "alerts.json").write_text(json.dumps({"alerts": []}), encoding="utf-8")

    r1 = pi.run_once(
        data_dir=data,
        public_data_dir=public,
        tasker_db=None,
        vault_dir=vault,
        now=NOW,
    )
    notes1 = list(transcripts.glob("*.md"))
    assert r1["vault_notes_written"] >= 1
    assert len(notes1) >= 1
    body = notes1[0].read_text(encoding="utf-8")
    assert "project: portfolio-lab" in body or "portfolio-lab" in body
    assert "log_tail" not in body

    # second hour same fingerprint → no new vault files
    r2 = pi.run_once(
        data_dir=data,
        public_data_dir=public,
        tasker_db=None,
        vault_dir=vault,
        now=NOW + timedelta(hours=1),
    )
    notes2 = list(transcripts.glob("*.md"))
    assert r2["vault_notes_written"] == 0
    assert len(notes2) == len(notes1)


def test_module_has_no_auto_planned_work_paths():
    """Static check: shipped module must not create planned work items."""
    src = Path(pi.__file__).read_text(encoding="utf-8")
    assert "status: planned" not in src
    assert 'status": "planned"' not in src
    assert "PROJECT_WORK_DIR" not in src
    assert "projects/portfolio-lab/work" not in src
    # no unattended prep promotion
    assert "dev-loop prep" not in src
    assert "auto_plan" not in src


def test_dual_mode_registration_in_repo():
    """Structural: job id registered in CRON_TARGETS, Makefile, crontab, tasker.yaml."""
    from src.cron_compat import CRON_TARGETS, CRON_EXPECTED_DURATIONS

    job = "portfolio-lab-prod-ideas"
    assert job in CRON_TARGETS
    assert job in CRON_EXPECTED_DURATIONS

    root = Path(__file__).resolve().parents[1]
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    assert "prod-ideas" in makefile
    assert job in makefile

    crontab = (root / "crontab").read_text(encoding="utf-8")
    assert "prod-ideas" in crontab

    tasker = (root / "config" / "tasker.yaml").read_text(encoding="utf-8")
    assert job in tasker
    assert "make prod-ideas" in tasker

    # cron_compat source contains duration entry
    compat = (root / "src" / "cron_compat.py").read_text(encoding="utf-8")
    assert job in compat
