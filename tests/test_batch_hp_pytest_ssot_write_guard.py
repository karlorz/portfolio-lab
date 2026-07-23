"""Batch HP: pytest must not rewrite production private/public SSOT.

Residuals after HM/HN/HO:
- make test isolates PUBLIC_DATA_DIR to /tmp/plab-pytest-public.* but leaves
  DATA_DIR and repo public/data on production. Multi-dest fan-out that
  resolves private → data/alerts.json or soft-mirror → public/data/* can
  poison live operator JSON while the suite is green.
- restamp lag_summary.source/dest from plab-pytest isolation can stamp
  fixture lag (0/1 false green) onto production health even when restamp
  *paths* are non-ephemeral.

Authority: never touches target_allocations / order_router.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def test_is_ephemeral_write_path_detects_plab_and_pytest() -> None:
    """Case DD: plab-pytest isolation + classic pytest tmp trees are ephemeral."""
    from src.monitor.signal_authority import is_ephemeral_write_path

    assert is_ephemeral_write_path("/tmp/plab-pytest-public.abc123/data/signals.json")
    assert is_ephemeral_write_path("/tmp/pytest-of-root/pytest-1/data/alerts.json")
    assert is_ephemeral_write_path("/tmp/pytest-99/health.json")
    assert not is_ephemeral_write_path("/root/projects/portfolio-lab/data/alerts.json")
    assert not is_ephemeral_write_path("/var/www/portfolio-lab/data/signals.json")
    assert not is_ephemeral_write_path(None)


def test_is_production_ssot_path_detects_live_trees() -> None:
    """Case DE: live private/public/repo soft-mirror paths are production SSOT."""
    from src.monitor.signal_authority import is_production_ssot_path
    from src.paths import PROJECT_ROOT

    project = Path(PROJECT_ROOT).resolve()
    assert is_production_ssot_path(project / "data" / "alerts.json")
    assert is_production_ssot_path(project / "public" / "data" / "alerts.json")
    assert is_production_ssot_path("/var/www/portfolio-lab/data/signals.json")
    assert not is_production_ssot_path("/tmp/plab-pytest-public.x/data/alerts.json")
    assert not is_production_ssot_path("/tmp/pytest-of-root/pytest-1/alerts.json")
    assert not is_production_ssot_path(None)


def test_write_json_multi_dest_skips_production_private_under_pytest(
    tmp_path, monkeypatch
) -> None:
    """Case DF: under pytest, production private dest is skipped; fixture dest writes."""
    from src.monitor import signal_authority as auth
    from src.paths import PROJECT_ROOT

    # Simulate pytest (already set in suite; force for clarity).
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_batch_hp.py::test_df")
    monkeypatch.delenv("PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC", raising=False)

    fixture_pub = tmp_path / "plab-pytest-public.xyz" / "data" / "alerts.json"
    fixture_pub.parent.mkdir(parents=True)

    # Point "production" private at a real-looking path under PROJECT_ROOT/data
    # but use a unique filename so we never clobber live alerts if the guard
    # regresses (test still asserts non-write via sentinel content).
    prod_priv = Path(PROJECT_ROOT) / "data" / ".batch_hp_ssot_guard_sentinel.json"
    if prod_priv.exists():
        prod_priv.unlink()
    sentinel_before = b"MUST_NOT_CHANGE"
    # Create a non-production sibling so we can prove only prod is skipped.
    other = tmp_path / "other" / "alerts.json"
    other.parent.mkdir(parents=True)

    payload = {"alerts": [], "count": 0, "batch": "hp-df"}
    result = auth.write_json_multi_dest(
        payload,
        public_path=fixture_pub,
        private_path=prod_priv,
        repo_path=other,
        soft_mirror_repo=True,
    )
    assert result.wrote_public is True
    assert result.wrote_private is False
    assert result.wrote_repo is True
    assert result.skipped_reason is not None
    assert "private:pytest-ssot-guard" in result.skipped_reason
    assert not prod_priv.exists()
    assert fixture_pub.is_file()
    assert other.is_file()
    assert json.loads(fixture_pub.read_text(encoding="utf-8"))["batch"] == "hp-df"


def test_write_json_multi_dest_writes_production_when_allow_live(
    tmp_path, monkeypatch
) -> None:
    """Case DG: PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC=1 opts out of the pytest SSOT guard."""
    from src.monitor import signal_authority as auth

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_batch_hp.py::test_dg")
    monkeypatch.setenv("PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC", "1")

    # Use tmp paths that *look* like production prefixes via monkeypatch of
    # is_production_ssot_path so we never touch live disks even with allow flag.
    pub = tmp_path / "www" / "alerts.json"
    priv = tmp_path / "data" / "alerts.json"
    pub.parent.mkdir(parents=True)
    priv.parent.mkdir(parents=True)

    monkeypatch.setattr(auth, "is_production_ssot_path", lambda p: True)
    result = auth.write_json_multi_dest(
        {"alerts": [], "count": 0, "batch": "hp-dg"},
        public_path=pub,
        private_path=priv,
        soft_mirror_repo=False,
    )
    assert result.wrote_public is True
    assert result.wrote_private is True
    assert pub.is_file() and priv.is_file()
    assert (pub.stat().st_mode & 0o777) == 0o644


def test_write_signals_multi_dest_skips_production_under_pytest(
    tmp_path, monkeypatch
) -> None:
    """Case DH: signals multi-dest refuses production paths under pytest."""
    from src.monitor import signal_authority as auth
    from src.paths import PROJECT_ROOT

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_batch_hp.py::test_dh")
    monkeypatch.delenv("PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC", raising=False)

    fixture_pub = tmp_path / "plab-pytest-public.dh" / "data" / "signals.json"
    fixture_pub.parent.mkdir(parents=True)
    prod_priv = Path(PROJECT_ROOT) / "data" / ".batch_hp_signals_ssot_sentinel.json"
    if prod_priv.exists():
        prod_priv.unlink()

    payload = {
        "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        "regime": {"regime": "normal", "vix": 16.0},
        "generated_at": "2026-07-23T00:00:00+00:00",
    }
    # Authority gate may require more keys — use try_ variant if hollow.
    # Prefer full validate=False only if needed; attempt real path first.
    try:
        result = auth.write_signals_multi_dest(
            payload,
            public_path=fixture_pub,
            private_path=prod_priv,
            soft_mirror_repo=False,
            validate=False,
        )
    except auth.AuthorityValidationError:
        result = auth.try_write_signals_multi_dest(
            payload,
            public_path=fixture_pub,
            private_path=prod_priv,
            soft_mirror_repo=False,
            validate=False,
        )

    assert result.wrote_public is True
    assert result.wrote_private is False
    assert result.skipped_reason is not None
    assert "private:pytest-ssot-guard" in (result.skipped_reason or "")
    assert not prod_priv.exists()
    assert fixture_pub.is_file()
    on_disk = json.loads(fixture_pub.read_text(encoding="utf-8"))
    assert on_disk["target_allocations"] == {
        "SPY": 0.46,
        "GLD": 0.38,
        "TLT": 0.16,
    }


def test_restamp_skips_production_when_lag_probe_is_plab_pytest(
    tmp_path, monkeypatch
) -> None:
    """Case DI: ephemeral lag source/dest must not restamp production health."""
    from src.monitor import repo_public_mirror_lag as mlag

    # Production-like health path under PROJECT_ROOT/data with unique name.
    from src.paths import PROJECT_ROOT

    prod_health = Path(PROJECT_ROOT) / "data" / ".batch_hp_health_ssot_sentinel.json"
    original = {
        "status": "warning",
        "timestamp": "2026-07-23T01:00:00+00:00",
        "repo_public_mirror_lagging_count": 10,
        "repo_public_mirror_lag": {
            "lagging_count": 10,
            "total": 33,
            "status": "critical",
            "source": "/var/www/portfolio-lab/data",
            "dest": str(Path(PROJECT_ROOT) / "public" / "data"),
        },
    }
    prod_health.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")
    before = prod_health.read_bytes()

    # Outside pytest path-guard so only lag-probe guard fires.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    lag_summary = {
        "lagging_count": 0,
        "total": 33,
        "status": "ok",
        "badge": "lagging=0/33",
        "source": "/tmp/plab-pytest-public.abc/data",
        "dest": str(tmp_path / "public" / "data"),
        "paths": [],
    }
    result = mlag.restamp_mirror_lag_on_health_documents(
        paths=[prod_health],
        lag_summary=lag_summary,
    )
    assert any("ephemeral-lag-source-guard" in s for s in result["skipped"])
    assert prod_health.read_bytes() == before
    # cleanup sentinel
    prod_health.unlink(missing_ok=True)


def test_is_ephemeral_restamp_path_detects_plab_pytest() -> None:
    """Case DJ: restamp classifier includes plab-pytest isolation roots."""
    from src.monitor.repo_public_mirror_lag import is_ephemeral_restamp_path

    assert is_ephemeral_restamp_path("/tmp/plab-pytest-public.xyz/data/health.json")
    assert is_ephemeral_restamp_path("/tmp/plab-pytest-public.xyz")
    assert not is_ephemeral_restamp_path("/root/projects/portfolio-lab/data/health.json")
