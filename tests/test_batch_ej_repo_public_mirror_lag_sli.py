"""Batch EJ: compact health SLI for repo public/data mirror lag.

Deep-research: expose mirror_lagging_count as freshness gauge (SoT =
operator PUBLIC_DATA_DIR; dest = repo public/data). Historical ops: 28–32/32
lag while cron green.
"""

from __future__ import annotations

import json

from src.dashboard.generator import project_repo_public_mirror_lag_onto_health
from src.monitor.health_check import refresh_signals_health_kill_fields
from src.monitor.repo_public_mirror_lag import summarize_repo_public_mirror_lag
from scripts.mirror_repo_public_data import (
    DEFAULT_FILE_GLOBS,
    lag_report,
    mirror_repo_public_data,
)


def test_project_mirror_lag_ok_when_zero() -> None:
    health = project_repo_public_mirror_lag_onto_health(
        {"status": "ok"},
        {
            "lagging_count": 0,
            "total": 33,
            "lagging_paths": [],
            "source": "/var/www/portfolio-lab/data",
            "dest": "/repo/public/data",
        },
    )
    assert health["repo_public_mirror_lag_status"] == "ok"
    assert health["repo_public_mirror_lagging_count"] == 0
    assert health["repo_public_mirror_total"] == 33
    assert health["status"] == "ok"
    assert "lagging=0/33" in (health.get("repo_public_mirror_lag_badge") or "")


def test_project_mirror_lag_warns_when_lagging() -> None:
    health = project_repo_public_mirror_lag_onto_health(
        {"status": "ok"},
        {
            "lagging_count": 11,
            "total": 33,
            "lagging_paths": ["signals.json", "health.json"],
            "source": "/var/www/x",
            "dest": "/repo/public/data",
        },
    )
    assert health["repo_public_mirror_lag_status"] == "critical"  # >=10
    assert health["repo_public_mirror_lagging_count"] == 11
    assert health["status"] == "warning"
    assert "signals.json" in health["repo_public_mirror_lagging_paths"]


def test_project_mirror_lag_status_lagging_below_critical() -> None:
    health = project_repo_public_mirror_lag_onto_health(
        {"status": "ok"},
        {"lagging_count": 3, "total": 33, "lagging_paths": ["a.json"]},
    )
    assert health["repo_public_mirror_lag_status"] == "lagging"
    assert health["status"] == "warning"


def test_summarize_detects_lag(tmp_path) -> None:
    src = tmp_path / "live"
    dest = tmp_path / "repo"
    src.mkdir()
    dest.mkdir()
    # Only files that appear in DEFAULT_FILE_GLOBS matter
    (src / "signals.json").write_text(
        json.dumps({"generator_git_sha": "aaa", "x": 1}), encoding="utf-8"
    )
    (dest / "signals.json").write_text(
        json.dumps({"generator_git_sha": "bbb", "x": 0}), encoding="utf-8"
    )
    # Matching twin should not lag
    body = json.dumps({"generator_git_sha": "ccc"})
    (src / "alerts.json").write_text(body, encoding="utf-8")
    (dest / "alerts.json").write_text(body, encoding="utf-8")

    summary = summarize_repo_public_mirror_lag(
        source_root=src, dest_root=dest
    )
    assert summary["ok"] is True
    assert summary["lagging_count"] >= 1
    assert "signals.json" in summary["lagging_paths"]
    assert summary["total"] >= 1


def test_health_ops_self_output_is_not_a_qualifying_lag(tmp_path) -> None:
    src = tmp_path / "live"
    dest = tmp_path / "repo"
    src.mkdir()
    dest.mkdir()

    (src / "health_ops.json").write_text('{"run": 2}', encoding="utf-8")
    (dest / "health_ops.json").write_text('{"run": 1}', encoding="utf-8")
    matching = '{"generator_git_sha": "same"}'
    (src / "alerts.json").write_text(matching, encoding="utf-8")
    (dest / "alerts.json").write_text(matching, encoding="utf-8")

    raw_rows = lag_report(src, dest)
    assert any(
        row["path"] == "health_ops.json" and row["lagging"]
        for row in raw_rows
    )

    summary = summarize_repo_public_mirror_lag(
        source_root=src, dest_root=dest
    )
    assert summary["ok"] is True
    assert summary["lagging_count"] == 0
    assert summary["lagging_paths"] == []
    assert summary["total"] == 1

    projected = project_repo_public_mirror_lag_onto_health(
        {"status": "ok"}, summary
    )
    assert projected["repo_public_mirror_lag_status"] == "ok"
    assert projected["repo_public_mirror_lag_badge"] == "lagging=0/1"
    assert projected["status"] == "ok"


def test_non_exempt_lag_remains_visible_and_uses_qualifying_total(tmp_path) -> None:
    src = tmp_path / "live"
    dest = tmp_path / "repo"
    src.mkdir()
    dest.mkdir()

    (src / "health_ops.json").write_text('{"run": 2}', encoding="utf-8")
    (dest / "health_ops.json").write_text('{"run": 1}', encoding="utf-8")
    (src / "signals.json").write_text('{"signal": 2}', encoding="utf-8")
    (dest / "signals.json").write_text('{"signal": 1}', encoding="utf-8")
    matching = '{"generator_git_sha": "same"}'
    (src / "alerts.json").write_text(matching, encoding="utf-8")
    (dest / "alerts.json").write_text(matching, encoding="utf-8")

    summary = summarize_repo_public_mirror_lag(
        source_root=src, dest_root=dest
    )
    assert summary["lagging_count"] == 1
    assert summary["lagging_paths"] == ["signals.json"]
    assert summary["total"] == 2

    projected = project_repo_public_mirror_lag_onto_health(
        {"status": "ok"}, summary
    )
    assert projected["repo_public_mirror_lag_status"] == "lagging"
    assert projected["repo_public_mirror_lag_badge"] == "lagging=1/2"
    assert projected["status"] == "warning"


def test_health_ops_self_output_remains_in_copy_catalog(tmp_path) -> None:
    src = tmp_path / "live"
    dest = tmp_path / "repo"
    src.mkdir()
    dest.mkdir()
    expected = b'{"run": 2}'
    (src / "health_ops.json").write_bytes(expected)
    assert "health_ops.json" in DEFAULT_FILE_GLOBS

    report = mirror_repo_public_data(
        source_root=src,
        dest_root=dest,
        restamp_health_lag=False,
    )

    assert report.copied == ["health_ops.json"]
    assert (dest / "health_ops.json").read_bytes() == expected


def test_partial_health_patch_projects_mirror_lag(tmp_path, monkeypatch) -> None:
    public = tmp_path / "public"
    private = tmp_path / "private"
    live = tmp_path / "live_public"
    repo_mirror = tmp_path / "repo_public"
    public.mkdir()
    private.mkdir()
    live.mkdir()
    repo_mirror.mkdir()

    (live / "signals.json").write_text(
        json.dumps({"generator_git_sha": "live1", "v": 2}), encoding="utf-8"
    )
    (repo_mirror / "signals.json").write_text(
        json.dumps({"generator_git_sha": "old", "v": 1}), encoding="utf-8"
    )

    signals = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "health": {"status": "ok"},
    }
    (public / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    (private / "signals.json").write_text(json.dumps(signals), encoding="utf-8")
    (private / "kill_switch.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )
    (private / "incidents.json").write_text(
        json.dumps({"open_count": 0, "status": "ok"}), encoding="utf-8"
    )

    monkeypatch.setattr(
        "src.monitor.health_check.PUBLIC_DATA_DIR", public, raising=False
    )
    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", private, raising=False)

    def _fake_summary(**kwargs):
        return {
            "lagging_count": 2,
            "total": 10,
            "lagging_paths": ["signals.json", "health.json"],
            "source": str(live),
            "dest": str(repo_mirror),
            "ok": True,
        }

    monkeypatch.setattr(
        "src.monitor.repo_public_mirror_lag.summarize_repo_public_mirror_lag",
        _fake_summary,
    )
    # Patch where health_check imports at call time — project path uses package
    import src.monitor.repo_public_mirror_lag as mlag

    monkeypatch.setattr(mlag, "summarize_repo_public_mirror_lag", _fake_summary)

    refresh_signals_health_kill_fields(
        {"status": "ok", "kill_switch": {"enabled": False}},
        public_dir=public,
        data_dir=private,
    )
    out = json.loads((public / "signals.json").read_text(encoding="utf-8"))
    h = out.get("health") or {}
    assert h.get("repo_public_mirror_lagging_count") == 2
    assert h.get("repo_public_mirror_lag_status") == "lagging"
    assert h.get("status") == "warning"
