"""Batch FX: soft-mirror restamps nested mirror-lag SLI; attach elevates status.

Deep-research (c445 sticky critical):
- Soft-mirror copies health_ops bytes including nested lag=11 critical while live
  probe is already 0/33 — false-critical holds until next health :30.
- attach_shared copies lag keys but leaves top-level status=ok under critical
  (elevate path dead for ops report).
- Consumer honesty: max(live, stamp) lagging_count prevents under-report (FV).
"""

from __future__ import annotations

import json

from src.dashboard.generator import project_repo_public_mirror_lag_onto_health
from src.monitor.health_check import attach_shared_freshness_slis_to_ops_report
from src.monitor.repo_public_mirror_lag import (
    resolve_mirror_lag_for_consumer,
    restamp_mirror_lag_on_health_documents,
)
from scripts.mirror_repo_public_data import mirror_repo_public_data


def test_project_elevates_status_ok_under_critical() -> None:
    health = project_repo_public_mirror_lag_onto_health(
        {"status": "ok"},
        {"lagging_count": 11, "total": 33, "lagging_paths": ["signals.json"]},
    )
    assert health["repo_public_mirror_lag_status"] == "critical"
    assert health["status"] == "warning"


def test_attach_shared_elevates_status_when_lag_critical(
    tmp_path, monkeypatch
) -> None:
    data = tmp_path / "data"
    public = tmp_path / "public"
    data.mkdir()
    public.mkdir()
    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", data, raising=False)
    monkeypatch.setattr(
        "src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False
    )

    import src.monitor.repo_public_mirror_lag as mlag

    monkeypatch.setattr(
        mlag,
        "summarize_repo_public_mirror_lag",
        lambda **k: {
            "lagging_count": 11,
            "total": 33,
            "lagging_paths": ["signals.json"],
            "source": "/var/www/x",
            "dest": str(public),
            "ok": True,
        },
    )

    report = attach_shared_freshness_slis_to_ops_report(
        {"status": "ok", "service": "portfolio-lab"},
        data_dir=data,
    )
    assert report["repo_public_mirror_lag_status"] == "critical"
    assert report["repo_public_mirror_lagging_count"] == 11
    # Batch FX / EP: soft elevate top-level status (ops hygiene, not halt)
    assert report["status"] == "warning"


def test_consumer_max_live_and_stamp_under_report() -> None:
    """EW: under-report stamp 0 while live 11 → consumer uses live."""
    resolved = resolve_mirror_lag_for_consumer(
        stamp={"lagging_count": 0, "total": 33, "status": "ok"},
        live={
            "lagging_count": 11,
            "total": 33,
            "lagging_paths": ["signals.json"],
            "ok": True,
        },
    )
    assert resolved["lagging_count"] == 11
    assert resolved["source_of_truth"] == "live"
    assert resolved["repo_public_mirror_lag_status"] == "critical"


def test_consumer_max_clears_when_both_zero() -> None:
    resolved = resolve_mirror_lag_for_consumer(
        stamp={"lagging_count": 0, "total": 33, "status": "ok"},
        live={"lagging_count": 0, "total": 33, "lagging_paths": [], "ok": True},
    )
    assert resolved["lagging_count"] == 0
    assert resolved["repo_public_mirror_lag_status"] == "ok"
    assert resolved["source_of_truth"] == "live"
    assert resolved["live_lagging_count"] == 0
    assert resolved["stamp_lagging_count"] == 0


def test_restamp_clears_sticky_honesty_meta(tmp_path) -> None:
    """Batch IF: restamp live=0 must rewrite mirror_lag_* honesty meta.

    Production residual after Session B multi-dest heal: repo_public_mirror
    lagging_count=0 while mirror_lag_stamp_lagging_count=1 and
    mirror_lag_source_of_truth=stamp (stale merge meta not restamped).
    """
    from src.monitor.repo_public_mirror_lag import apply_lag_summary_to_health_doc

    sticky = {
        "status": "ok",
        "repo_public_mirror_lagging_count": 0,
        "repo_public_mirror_total": 36,
        "repo_public_mirror_lag_status": "ok",
        "repo_public_mirror_lag_badge": "lagging=0/36",
        "repo_public_mirror_lag": {
            "lagging_count": 0,
            "total": 36,
            "status": "ok",
            "badge": "lagging=0/36",
        },
        # Sticky honesty meta from prior max(live,stamp) merge
        "mirror_lag_source_of_truth": "stamp",
        "mirror_lag_live_lagging_count": 0,
        "mirror_lag_stamp_lagging_count": 1,
    }
    live = {
        "lagging_count": 0,
        "total": 36,
        "lagging_paths": [],
        "source": "/var/www/portfolio-lab/data",
        "dest": "/root/projects/portfolio-lab/public/data",
        "ok": True,
    }
    out = apply_lag_summary_to_health_doc(sticky, live)
    assert out["repo_public_mirror_lagging_count"] == 0
    assert out["repo_public_mirror_lag_status"] == "ok"
    assert out["mirror_lag_source_of_truth"] == "live"
    assert out["mirror_lag_live_lagging_count"] == 0
    assert out["mirror_lag_stamp_lagging_count"] == 0


def test_restamp_document_honesty_meta_on_disk(tmp_path) -> None:
    """Batch IF: restamp_mirror_lag_on_health_documents rewrites honesty meta."""
    ops = tmp_path / "health_ops.json"
    sticky = {
        "status": "warning",
        "repo_public_mirror_lagging_count": 1,
        "repo_public_mirror_total": 36,
        "repo_public_mirror_lag_status": "lagging",
        "repo_public_mirror_lag": {
            "lagging_count": 1,
            "total": 36,
            "status": "lagging",
            "badge": "lagging=1/36",
        },
        "mirror_lag_source_of_truth": "stamp",
        "mirror_lag_live_lagging_count": 0,
        "mirror_lag_stamp_lagging_count": 1,
    }
    ops.write_text(json.dumps(sticky), encoding="utf-8")
    live = {
        "lagging_count": 0,
        "total": 36,
        "lagging_paths": [],
        "source": "/var/www/portfolio-lab/data",
        "dest": str(tmp_path),
        "ok": True,
    }
    result = restamp_mirror_lag_on_health_documents(paths=[ops], lag_summary=live)
    assert any("health_ops" in p for p in result["restamped"])
    out = json.loads(ops.read_text(encoding="utf-8"))
    assert out["repo_public_mirror_lagging_count"] == 0
    assert out["repo_public_mirror_lag_status"] == "ok"
    assert out["mirror_lag_source_of_truth"] == "live"
    assert out["mirror_lag_live_lagging_count"] == 0
    assert out["mirror_lag_stamp_lagging_count"] == 0


def test_restamp_rewrites_nested_lag_on_health_ops(tmp_path) -> None:
    """EY: restamp nested SLI when live probe disagrees with stamped critical."""
    ops = tmp_path / "health_ops.json"
    sticky = {
        "status": "ok",
        "timestamp": "2026-07-22T16:00:00+00:00",
        "repo_public_mirror_lagging_count": 11,
        "repo_public_mirror_lag_status": "critical",
        "repo_public_mirror_lag": {
            "lagging_count": 11,
            "total": 33,
            "status": "critical",
            "badge": "lagging=11/33",
        },
    }
    ops.write_text(json.dumps(sticky), encoding="utf-8")

    live = {
        "lagging_count": 0,
        "total": 33,
        "lagging_paths": [],
        "source": "/var/www/x",
        "dest": str(tmp_path),
        "ok": True,
    }
    result = restamp_mirror_lag_on_health_documents(
        paths=[ops],
        lag_summary=live,
    )
    assert result["restamped"] == ["health_ops.json"] or any(
        "health_ops" in p for p in result["restamped"]
    )
    out = json.loads(ops.read_text(encoding="utf-8"))
    assert out["repo_public_mirror_lagging_count"] == 0
    assert out["repo_public_mirror_lag_status"] == "ok"
    assert out["repo_public_mirror_lag"]["lagging_count"] == 0
    assert out["repo_public_mirror_lag"]["status"] == "ok"
    # Soft elevate only when lagging — cleared back toward prior ok
    assert out["status"] == "ok"


def test_soft_mirror_restamps_health_docs_after_copy(tmp_path, monkeypatch) -> None:
    """EN/EY: mirror_repo_public_data end-pipeline restamps nested lag on dest."""
    src = tmp_path / "live"
    dest = tmp_path / "repo"
    src.mkdir()
    dest.mkdir()

    sticky = {
        "status": "ok",
        "repo_public_mirror_lagging_count": 11,
        "repo_public_mirror_lag_status": "critical",
        "repo_public_mirror_total": 33,
        "repo_public_mirror_lag": {
            "lagging_count": 11,
            "total": 33,
            "status": "critical",
        },
        "payload": "v2",
    }
    (src / "health_ops.json").write_text(json.dumps(sticky), encoding="utf-8")
    (src / "health.json").write_text(json.dumps(sticky), encoding="utf-8")
    # Dest starts lagging (bytes unequal) then mirror equalizes
    old = dict(sticky)
    old["payload"] = "v1"
    (dest / "health_ops.json").write_text(json.dumps(old), encoding="utf-8")
    (dest / "health.json").write_text(json.dumps(old), encoding="utf-8")
    # Other catalog twin equal so live lag after mirror is only health docs
    # until restamp rewrites them (restamp changes dest → may re-introduce lag
    # vs source; restamp should also update source when requested).
    body = json.dumps({"generator_git_sha": "abc", "x": 1})
    (src / "signals.json").write_text(body, encoding="utf-8")
    (dest / "signals.json").write_text(body, encoding="utf-8")

    import src.monitor.repo_public_mirror_lag as mlag

    # After byte-equal copy, live probe returns 0 (mock honest post-heal)
    monkeypatch.setattr(
        mlag,
        "summarize_repo_public_mirror_lag",
        lambda **k: {
            "lagging_count": 0,
            "total": 33,
            "lagging_paths": [],
            "source": str(src),
            "dest": str(dest),
            "ok": True,
        },
    )

    report = mirror_repo_public_data(
        source_root=src,
        dest_root=dest,
        files=("health_ops.json", "health.json", "signals.json"),
        restamp_health_lag=True,
    )
    assert "health_ops.json" in report.copied
    dest_ops = json.loads((dest / "health_ops.json").read_text(encoding="utf-8"))
    assert dest_ops["repo_public_mirror_lagging_count"] == 0
    assert dest_ops["repo_public_mirror_lag_status"] == "ok"
    # Source sticky critical also cleared (end-pipeline finalize)
    src_ops = json.loads((src / "health_ops.json").read_text(encoding="utf-8"))
    assert src_ops["repo_public_mirror_lagging_count"] == 0
    assert src_ops["repo_public_mirror_lag_status"] == "ok"


def test_is_ephemeral_restamp_path_detects_pytest_trees() -> None:
    from src.monitor.repo_public_mirror_lag import is_ephemeral_restamp_path

    assert is_ephemeral_restamp_path("/tmp/pytest-of-root/pytest-1/health.json")
    assert is_ephemeral_restamp_path("/tmp/pytest-99/data/health.json")
    assert not is_ephemeral_restamp_path("/root/projects/portfolio-lab/data/health.json")
    assert not is_ephemeral_restamp_path("/var/www/portfolio-lab/data/health.json")


def test_restamp_skips_production_when_mixed_with_pytest(tmp_path, monkeypatch) -> None:
    """Batch HM DA: fixture restamp must not poison production private health SSOT."""
    from src.monitor.repo_public_mirror_lag import restamp_mirror_lag_on_health_documents

    fixture_ops = tmp_path / "health_ops.json"
    # Simulate production private path that must not be rewritten in mixed batch
    prod = tmp_path / "prod_data"
    prod.mkdir()
    prod_health = prod / "health.json"
    sticky_fixture = {
        "status": "ok",
        "repo_public_mirror_lagging_count": 11,
        "repo_public_mirror_lag_status": "critical",
        "repo_public_mirror_lag": {"lagging_count": 11, "total": 33, "status": "critical"},
    }
    clean_prod = {
        "status": "ok",
        "repo_public_mirror_lagging_count": 0,
        "repo_public_mirror_lag_status": "ok",
        "repo_public_mirror_lag": {"lagging_count": 0, "total": 33, "status": "ok"},
        "marker": "prod-ssot",
    }
    fixture_ops.write_text(json.dumps(sticky_fixture), encoding="utf-8")
    prod_health.write_text(json.dumps(clean_prod), encoding="utf-8")
    before = prod_health.read_text(encoding="utf-8")

    # Force fixture path classification via monkeypatch of is_ephemeral
    import src.monitor.repo_public_mirror_lag as mlag

    real_is_ephemeral = mlag.is_ephemeral_restamp_path

    def _classify(path):
        p = str(path)
        if "prod_data" in p:
            return False
        if str(tmp_path) in p:
            return True
        return real_is_ephemeral(path)

    monkeypatch.setattr(mlag, "is_ephemeral_restamp_path", _classify)

    live = {
        "lagging_count": 0,
        "total": 33,
        "lagging_paths": [],
        "source": "/var/www/x",
        "dest": str(tmp_path),
        "ok": True,
    }
    result = restamp_mirror_lag_on_health_documents(
        paths=[fixture_ops, prod_health],
        lag_summary=live,
    )
    assert any("health_ops" in p for p in result["restamped"])
    assert any("path-guard" in s or "prod" in s for s in result["skipped"]) or (
        prod_health.read_text(encoding="utf-8") == before
    )
    # Production SSOT must remain untouched
    assert prod_health.read_text(encoding="utf-8") == before
    assert json.loads(before)["marker"] == "prod-ssot"


def test_apply_lag_summary_demotes_lag_only_warning_when_lag_heals() -> None:
    """Lag-only sticky warning must demote when live lag is 0."""
    from src.monitor.repo_public_mirror_lag import apply_lag_summary_to_health_doc

    sticky = {
        "status": "warning",
        "ops_health_status": "warning",
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
        "scheduler_status": {"status": "ok"},
        "data_pipeline_slo": {"status": "ok"},
        "repo_public_mirror_lagging_count": 2,
        "repo_public_mirror_total": 35,
        "repo_public_mirror_lag_status": "lagging",
        "repo_public_mirror_lagging_paths": ["index.json", "incidents.json"],
        "signal_health": {
            "status": "degraded",
            "summary": {"healthy": 1, "total_tracked": 9, "quality_badge": "1/9"},
        },
    }
    live = {
        "lagging_count": 0,
        "total": 35,
        "lagging_paths": [],
        "source": "/var/www/portfolio-lab/data",
        "dest": "/root/projects/portfolio-lab/public/data",
        "ok": True,
    }
    out = apply_lag_summary_to_health_doc(sticky, live)
    assert out["repo_public_mirror_lagging_count"] == 0
    assert out["repo_public_mirror_lag_status"] == "ok"
    assert out["status"] == "ok"
    assert out.get("ops_health_status") == "ok"
    # Quality plane remains disclosed, not folded into ops status
    assert out["signal_health"]["summary"]["healthy"] == 1


def test_apply_lag_summary_does_not_demote_when_kill_enabled() -> None:
    from src.monitor.repo_public_mirror_lag import apply_lag_summary_to_health_doc

    sticky = {
        "status": "warning",
        "kill_switch": {"enabled": True, "level": "halt"},
        "open_incidents": {"open_count": 0},
        "scheduler_status": {"status": "ok"},
        "data_pipeline_slo": {"status": "ok"},
        "repo_public_mirror_lagging_count": 2,
        "repo_public_mirror_total": 35,
        "repo_public_mirror_lag_status": "lagging",
    }
    live = {
        "lagging_count": 0,
        "total": 35,
        "lagging_paths": [],
        "ok": True,
    }
    out = apply_lag_summary_to_health_doc(sticky, live)
    assert out["repo_public_mirror_lagging_count"] == 0
    assert out["status"] == "warning"  # real ops failure preserved


def test_apply_lag_summary_does_not_demote_when_open_incidents() -> None:
    from src.monitor.repo_public_mirror_lag import apply_lag_summary_to_health_doc

    sticky = {
        "status": "warning",
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 1},
        "scheduler_status": {"status": "ok"},
        "data_pipeline_slo": {"status": "ok"},
        "repo_public_mirror_lagging_count": 1,
        "repo_public_mirror_total": 35,
        "repo_public_mirror_lag_status": "lagging",
    }
    live = {"lagging_count": 0, "total": 35, "lagging_paths": [], "ok": True}
    out = apply_lag_summary_to_health_doc(sticky, live)
    assert out["status"] == "warning"


def test_apply_lag_summary_does_not_demote_when_scheduler_degraded() -> None:
    from src.monitor.repo_public_mirror_lag import apply_lag_summary_to_health_doc

    sticky = {
        "status": "warning",
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
        "scheduler_status": {"status": "degraded"},
        "data_pipeline_slo": {"status": "ok"},
        "repo_public_mirror_lagging_count": 1,
        "repo_public_mirror_total": 35,
        "repo_public_mirror_lag_status": "lagging",
    }
    live = {"lagging_count": 0, "total": 35, "lagging_paths": [], "ok": True}
    out = apply_lag_summary_to_health_doc(sticky, live)
    assert out["status"] == "warning"


def test_apply_lag_summary_does_not_demote_when_slo_warning() -> None:
    from src.monitor.repo_public_mirror_lag import apply_lag_summary_to_health_doc

    sticky = {
        "status": "warning",
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
        "scheduler_status": {"status": "ok"},
        "data_pipeline_slo": {"status": "warning"},
        "repo_public_mirror_lagging_count": 1,
        "repo_public_mirror_total": 35,
        "repo_public_mirror_lag_status": "lagging",
    }
    live = {"lagging_count": 0, "total": 35, "lagging_paths": [], "ok": True}
    out = apply_lag_summary_to_health_doc(sticky, live)
    assert out["status"] == "warning"


def test_apply_lag_summary_still_elevates_when_live_lag_present() -> None:
    from src.monitor.repo_public_mirror_lag import apply_lag_summary_to_health_doc

    base = {
        "status": "ok",
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
        "scheduler_status": {"status": "ok"},
        "data_pipeline_slo": {"status": "ok"},
        "repo_public_mirror_lagging_count": 0,
        "repo_public_mirror_total": 35,
        "repo_public_mirror_lag_status": "ok",
    }
    live = {
        "lagging_count": 2,
        "total": 35,
        "lagging_paths": ["prices.json", "signals.json"],
        "ok": True,
    }
    out = apply_lag_summary_to_health_doc(base, live)
    assert out["repo_public_mirror_lagging_count"] == 2
    assert out["status"] == "warning"


def test_dashboard_schema_demotes_system_status_on_lag_heal() -> None:
    from src.monitor.repo_public_mirror_lag import apply_lag_summary_to_health_doc

    sticky = {
        "system_status": "warning",
        "ops_health_status": "warning",
        "status": "warning",
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
        "scheduler_status": {"status": "ok"},
        "data_pipeline_slo": {"status": "ok"},
        "failed_cron_jobs": 0,
        "repo_public_mirror_lagging_count": 2,
        "repo_public_mirror_total": 35,
        "repo_public_mirror_lag_status": "lagging",
    }
    live = {"lagging_count": 0, "total": 35, "lagging_paths": [], "ok": True}
    out = apply_lag_summary_to_health_doc(sticky, live)
    assert out["repo_public_mirror_lagging_count"] == 0
    assert out["system_status"] in {"healthy", "ok"}
    assert out.get("ops_health_status") in {"ok", "healthy"}


def test_restamp_on_disk_demotes_lag_only_warning(tmp_path, monkeypatch) -> None:
    from src.monitor.repo_public_mirror_lag import restamp_mirror_lag_on_health_documents

    data = tmp_path / "data"
    data.mkdir()
    # Restamp re-projects disk kill/open via health_kill_surfaces'
    # _disk_kill_and_open_incidents() (post HEALTH-CHECK-SPLIT binding);
    # the hub health_check.DATA_DIR binding is not read on this path.
    monkeypatch.setattr("src.monitor.health_kill_surfaces.DATA_DIR", data)

    ops = tmp_path / "health_ops.json"
    sticky = {
        "status": "warning",
        "ops_health_status": "warning",
        "checks": {"kill_switch": {"enabled": False}},
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
        "scheduler_status": {"status": "ok"},
        "data_pipeline_slo": {"status": "ok"},
        "repo_public_mirror_lagging_count": 2,
        "repo_public_mirror_total": 35,
        "repo_public_mirror_lag_status": "lagging",
        "repo_public_mirror_lag": {
            "lagging_count": 2,
            "total": 35,
            "status": "lagging",
        },
    }
    ops.write_text(json.dumps(sticky), encoding="utf-8")
    live = {
        "lagging_count": 0,
        "total": 35,
        "lagging_paths": [],
        "source": str(tmp_path),
        "dest": str(tmp_path),
        "ok": True,
    }
    result = restamp_mirror_lag_on_health_documents(paths=[ops], lag_summary=live)
    assert any("health_ops" in p for p in result["restamped"])
    out = json.loads(ops.read_text(encoding="utf-8"))
    assert out["repo_public_mirror_lagging_count"] == 0
    assert out["status"] == "ok"


def test_lag_heal_does_not_let_signal_quality_block_ops_demotion() -> None:
    """Thin SH must not keep ops warning after lag-only heal."""
    from src.dashboard.health_slo_alerts import build_health_slo_alerts
    from src.monitor.repo_public_mirror_lag import apply_lag_summary_to_health_doc

    sticky = {
        "status": "warning",
        "system_status": "warning",
        "ops_health_status": "warning",
        "kill_switch": {"enabled": False},
        "open_incidents": {"open_count": 0},
        "scheduler_status": {"status": "ok"},
        "data_pipeline_slo": {"status": "ok"},
        "repo_public_mirror_lagging_count": 2,
        "repo_public_mirror_total": 35,
        "repo_public_mirror_lag_status": "lagging",
        "signal_health": {
            "status": "degraded",
            "summary": {
                "healthy": 1,
                "total_tracked": 9,
                "quality_badge": "1/9 healthy",
            },
        },
    }
    live = {"lagging_count": 0, "total": 35, "lagging_paths": [], "ok": True}
    out = apply_lag_summary_to_health_doc(sticky, live)
    assert out["system_status"] in {"healthy", "ok"}
    assert out["ops_health_status"] in {"ok", "healthy", "green", "success"}
    alerts = build_health_slo_alerts(out)
    ops_warn = [
        a
        for a in alerts
        if a.get("reason") == "system_status_warning"
        or (
            a.get("type") == "health_slo"
            and "ops" in str(a.get("title", "")).lower()
        )
    ]
    assert ops_warn == []
