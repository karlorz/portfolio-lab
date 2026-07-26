"""Batch ID: private data/health_ops.json included in soft-mirror lag restamp.

Session A residual after IC:
- publish_ops multi-dest lands health_ops triple EQ @ 0o644.
- Soft-mirror end-pipeline restamp only appended private health.json +
  signals.json — not private data/health_ops.json.
- Live: public/repo gain mirror_lag_restamped_at while private twin lags
  (sha split; keys differ by restamp metadata only).

Authority: never touches signals.json.target_allocations / order_router.
"""

from __future__ import annotations

import json
from pathlib import Path


def _health_ops_doc(*, lagging: int = 0) -> dict:
    return {
        "status": "ok",
        "scope": "operational_readiness",
        "timestamp": "2026-07-23T00:00:00+00:00",
        "checks": {},
        "repo_public_mirror_lagging_count": lagging,
        "repo_public_mirror_total": 36,
        "repo_public_mirror_lag_status": "ok" if lagging == 0 else "lagging",
        "repo_public_mirror_lag_badge": f"lagging={lagging}/36",
        "repo_public_mirror_lagging_paths": [] if lagging == 0 else ["x.json"],
        "repo_public_mirror_source": "/var/www/portfolio-lab/data",
        "repo_public_mirror_dest": "/root/projects/portfolio-lab/public/data",
        "repo_public_mirror_lag": {
            "lagging_count": lagging,
            "total": 36,
            "status": "ok" if lagging == 0 else "lagging",
            "badge": f"lagging={lagging}/36",
            "paths": [] if lagging == 0 else ["x.json"],
        },
    }


def test_soft_mirror_restamp_path_list_includes_private_health_ops(
    tmp_path, monkeypatch
) -> None:
    """Case DU: soft-mirror private append list includes health_ops.json.

    Under pytest, real restamp skips non-ephemeral production paths (HM guard).
    Assert path *selection* includes private DATA_DIR/health_ops.json when
    roots are non-ephemeral (production soft-mirror shape).
    """
    from scripts.mirror_repo_public_data import mirror_repo_public_data
    import src.monitor.repo_public_mirror_lag as mlag

    public = tmp_path / "www"
    repo = tmp_path / "repo_public"
    private = tmp_path / "data"
    public.mkdir()
    repo.mkdir()
    private.mkdir()

    body = json.dumps(_health_ops_doc(lagging=0), indent=2) + "\n"
    (public / "health_ops.json").write_text(body, encoding="utf-8")
    (private / "health_ops.json").write_text(body, encoding="utf-8")
    (private / "health.json").write_text(
        json.dumps(_health_ops_doc()), encoding="utf-8"
    )
    (private / "signals.json").write_text(
        json.dumps(
            {
                "target_allocations": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
                "health": {
                    "repo_public_mirror_lagging_count": 0,
                    "repo_public_mirror_lag_status": "ok",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.paths.DATA_DIR", private)

    # Treat fixture trees as production-like so private append runs
    monkeypatch.setattr(mlag, "is_ephemeral_restamp_path", lambda path: False)

    captured: dict[str, list[Path]] = {}

    def _capture_restamp(*, paths=None, lag_summary=None, **kwargs):
        captured["paths"] = [Path(p) for p in (paths or [])]
        return {
            "restamped": [p.name for p in captured["paths"]],
            "skipped": [],
            "errors": [],
            "lag_summary": lag_summary or {},
        }

    monkeypatch.setattr(mlag, "restamp_mirror_lag_on_health_documents", _capture_restamp)
    monkeypatch.setattr(
        mlag,
        "summarize_repo_public_mirror_lag",
        lambda **k: {
            "lagging_count": 0,
            "total": 36,
            "lagging_paths": [],
            "source": str(public),
            "dest": str(repo),
            "ok": True,
        },
    )

    report = mirror_repo_public_data(
        source_root=public,
        dest_root=repo,
        files=("health_ops.json",),
        restamp_health_lag=True,
    )
    assert not report.errors, report.errors
    names = {p.name for p in captured.get("paths", [])}
    resolved = {str(p.resolve()) for p in captured.get("paths", [])}
    assert "health_ops.json" in names
    assert str((private / "health_ops.json").resolve()) in resolved
    # Hold prior private basenames
    assert str((private / "health.json").resolve()) in resolved
    assert str((private / "signals.json").resolve()) in resolved


def test_soft_mirror_restamp_skips_private_health_ops_when_ephemeral(
    tmp_path, monkeypatch
) -> None:
    """Case DV: ephemeral source/dest must not restamp production private ops."""
    from scripts.mirror_repo_public_data import mirror_repo_public_data
    import src.monitor.repo_public_mirror_lag as mlag

    public = tmp_path / "plab-pytest-public" / "data"
    repo = tmp_path / "plab-pytest-repo" / "data"
    private = tmp_path / "prod_data"
    public.mkdir(parents=True)
    repo.mkdir(parents=True)
    private.mkdir()

    body = json.dumps(_health_ops_doc(lagging=1), indent=2) + "\n"
    (public / "health_ops.json").write_text(body, encoding="utf-8")
    before = body.encode("utf-8")
    (private / "health_ops.json").write_bytes(before)

    monkeypatch.setattr("src.paths.DATA_DIR", private)

    # Real classifier: plab-pytest roots ephemeral; private prod_data not
    real = mlag.is_ephemeral_restamp_path

    def _classify(path):
        text = str(path or "")
        if "plab-pytest" in text:
            return True
        if "prod_data" in text:
            return False
        return real(path)

    monkeypatch.setattr(mlag, "is_ephemeral_restamp_path", _classify)

    captured: dict[str, list[Path]] = {}

    def _capture_restamp(*, paths=None, lag_summary=None, **kwargs):
        captured["paths"] = [Path(p) for p in (paths or [])]
        return {
            "restamped": [],
            "skipped": [],
            "errors": [],
            "lag_summary": lag_summary or {},
        }

    monkeypatch.setattr(mlag, "restamp_mirror_lag_on_health_documents", _capture_restamp)
    monkeypatch.setattr(
        mlag,
        "summarize_repo_public_mirror_lag",
        lambda **k: {
            "lagging_count": 0,
            "total": 1,
            "lagging_paths": [],
            "source": str(public),
            "dest": str(repo),
            "ok": True,
        },
    )

    mirror_repo_public_data(
        source_root=public,
        dest_root=repo,
        files=("health_ops.json",),
        restamp_health_lag=True,
    )
    # Private production twin must not appear when roots are ephemeral
    for p in captured.get("paths", []):
        assert "prod_data" not in str(p)
    assert (private / "health_ops.json").read_bytes() == before


def test_private_health_ops_restamp_preserves_target_allocations(
    tmp_path, monkeypatch
) -> None:
    """Case DW: restamping private health_ops does not rewrite signals authority."""
    from src.monitor.repo_public_mirror_lag import restamp_mirror_lag_on_health_documents

    private = tmp_path / "data"
    private.mkdir()
    champion = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
    signals = {
        "target_allocations": champion,
        "health": {
            "status": "ok",
            "repo_public_mirror_lagging_count": 2,
            "repo_public_mirror_total": 36,
            "repo_public_mirror_lag_status": "lagging",
        },
    }
    (private / "signals.json").write_text(
        json.dumps(signals, indent=2) + "\n", encoding="utf-8"
    )
    (private / "health_ops.json").write_text(
        json.dumps(_health_ops_doc(lagging=2), indent=2) + "\n", encoding="utf-8"
    )

    lag = {
        "lagging_count": 0,
        "total": 36,
        "lagging_paths": [],
        "source": str(private),
        "dest": str(private),
        "ok": True,
    }
    # Paths under tmp are ephemeral → restamp allowed under pytest
    result = restamp_mirror_lag_on_health_documents(
        paths=[private / "health_ops.json", private / "signals.json"],
        lag_summary=lag,
    )
    restamped_names = {Path(x).name for x in result["restamped"]}
    assert "health_ops.json" in restamped_names
    assert "signals.json" in restamped_names

    after_sig = json.loads((private / "signals.json").read_text(encoding="utf-8"))
    assert after_sig["target_allocations"] == champion
    after_ops = json.loads((private / "health_ops.json").read_text(encoding="utf-8"))
    assert after_ops["repo_public_mirror_lagging_count"] == 0
    assert "mirror_lag_restamped_at" in after_ops
    assert (private / "health_ops.json").stat().st_mode & 0o777 == 0o644
