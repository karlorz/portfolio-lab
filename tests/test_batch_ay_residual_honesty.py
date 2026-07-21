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
