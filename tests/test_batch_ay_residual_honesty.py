"""Batch AY residual honesty: attribution dual-write provenance + H18 live WWW guard."""

from __future__ import annotations

import json
from pathlib import Path


def test_attribution_save_report_dual_write_provenance(tmp_path, monkeypatch):
    from src.monitor import performance_attribution as pa
    from src.monitor.performance_attribution import PerformanceAttribution

    private = tmp_path / "data"
    public = tmp_path / "public"
    private.mkdir()
    public.mkdir()

    monkeypatch.setattr(pa, "PUBLIC_DATA_DIR", public)
    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "attribsha1234",
    )

    attr = PerformanceAttribution(data_dir=private)

    class FakeReport:
        timestamp = "2026-07-21T12:00:00+00:00"

        def to_dict(self):
            return {
                "timestamp": self.timestamp,
                "status": "no_data",
                "sources": {},
            }

    path = attr.save_report(FakeReport())
    assert path.exists()

    priv = json.loads(path.read_text())
    pub = json.loads((public / "attribution" / "latest.json").read_text())
    for body in (priv, pub):
        assert body.get("generator_git_sha") == "attribsha1234"
        pc = body["provenance_completeness"]
        assert pc["dual_write_attempted"] is True
        assert pc["dual_write_ok"] is True
        assert pc["paths_identical"] is False
        assert "attribution" in (pc.get("note") or "")


def test_attribution_history_reconciles_public_business_drift(tmp_path, monkeypatch):
    from src.monitor import performance_attribution as pa
    from src.monitor.performance_attribution import PerformanceAttribution
    from src.dashboard.public_projection import (
        find_public_internal_paths,
        public_business_values_equal,
    )

    private = tmp_path / "data"
    public = tmp_path / "public"
    private_attr = private / "attribution"
    public_attr = public / "attribution"
    private_attr.mkdir(parents=True)
    public_attr.mkdir(parents=True)
    monkeypatch.setattr(pa, "PUBLIC_DATA_DIR", public)
    monkeypatch.setenv("PUBLIC_DATA_DIR", str(public))
    monkeypatch.setenv("PORTFOLIO_LAB_FORCE_PUBLIC_PROJECTION", "1")

    private_path = private_attr / "attribution_2026-07-24.json"
    public_path = public_attr / private_path.name
    private_payload = {
        "timestamp": "2026-07-24T23:47:09.459741",
        "status": "ok",
        "avg_hit_rate": 0.3746,
        "private_diagnostic_path": str(private_path),
        "sources": {"signal": {"hit_rate": 0.61}},
    }
    public_payload = {
        "timestamp": "2026-07-24T18:47:06.754824",
        "status": "ok",
        "avg_hit_rate": 0.3743,
        "private_diagnostic_path": "data/attribution/attribution_2026-07-24.json",
        "sources": {"signal": {"hit_rate": 0.59}},
    }
    private_path.write_text(json.dumps(private_payload), encoding="utf-8")
    public_path.write_text(json.dumps(public_payload), encoding="utf-8")

    attr = PerformanceAttribution(data_dir=private)
    reconciled = attr.reconcile_public_history(public_root=public)

    assert reconciled == [public_path]
    private_after = json.loads(private_path.read_text(encoding="utf-8"))
    public_after = json.loads(public_path.read_text(encoding="utf-8"))
    assert public_after["avg_hit_rate"] == private_after["avg_hit_rate"] == 0.3746
    assert public_after["sources"] == private_after["sources"]
    assert private_after["private_diagnostic_path"] == str(private_path)
    assert find_public_internal_paths(public_after) == []
    assert public_business_values_equal(private_after, public_after)


def test_dual_write_canary_includes_attribution_latest():
    from scripts.check_public_data_consistency import DUAL_WRITE_PROVENANCE_FILES

    assert "attribution/latest.json" in DUAL_WRITE_PROVENANCE_FILES


def test_batch_ay_source_contracts():
    src = Path("src/monitor/performance_attribution.py").read_text(encoding="utf-8")
    assert "_attach_dual_write_provenance" in src
    assert "_stamp_generator_git_sha" in src

    conf = Path("tests/conftest.py").read_text(encoding="utf-8")
    assert "_guard_live_public_ssot_pollution" in conf
    assert "_FIXTURE_SHA_DENYLIST" in conf
    assert "PORTFOLIO_LAB_STRICT_LIVE_PUBLIC_GUARD" in conf
    assert "abad1dea00" in conf


def test_h18_guard_skips_when_live_public_absent(monkeypatch, tmp_path):
    """Guard is a no-op when live WWW tree is missing."""
    monkeypatch.setenv(
        "PORTFOLIO_LAB_LIVE_PUBLIC_DATA_DIR",
        str(tmp_path / "no-such-www"),
    )
    monkeypatch.delenv("PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC", raising=False)
    monkeypatch.delenv("PORTFOLIO_LAB_ALLOW_LIVE_WWW_MUTATION", raising=False)
    from tests.conftest import _resolve_live_public_root

    assert _resolve_live_public_root() is None


def test_h18_fixture_sha_denylist_detects_pollution(tmp_path, monkeypatch):
    live = tmp_path / "www"
    live.mkdir()
    (live / "overlay_dashboard.json").write_text(
        json.dumps({"generator_git_sha": "abad1dea00dead"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PORTFOLIO_LAB_LIVE_PUBLIC_DATA_DIR", str(live))
    monkeypatch.setenv("PUBLIC_DATA_DIR", str(tmp_path / "isolated"))
    monkeypatch.delenv("PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC", raising=False)
    monkeypatch.delenv("PORTFOLIO_LAB_ALLOW_LIVE_WWW_MUTATION", raising=False)

    from tests.conftest import _FIXTURE_SHA_DENYLIST, _LIVE_PUBLIC_WATCHLIST

    polluted = []
    for name in _LIVE_PUBLIC_WATCHLIST:
        p = live / name
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        for token in _FIXTURE_SHA_DENYLIST:
            if token in text:
                polluted.append(f"{name}:{token}")
    assert polluted, "denylist should catch abad1dea00 in overlay"
