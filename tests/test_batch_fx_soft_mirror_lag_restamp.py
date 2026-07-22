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
from pathlib import Path

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
