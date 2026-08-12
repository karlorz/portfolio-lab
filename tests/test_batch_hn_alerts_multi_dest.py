"""Batch HN: alerts.json serialize-once multi-dest (public + private + repo).

Live residual: priv/www alerts equal while repo public/data/alerts.json lagged
(0/1/1 style). Health dual path.write_text never soft-mirrored; sticky 0600
risk without atomic 0o644. Does not touch order_router authority — only
operator JSON fan-out. Live TA remains signals.json.target_allocations.
"""

from __future__ import annotations

import json


def test_write_json_multi_dest_same_bytes_mode_644(tmp_path):
    """Case DA: non-authority multi-dest fan-out is equal bytes + 0o644."""
    from src.monitor.signal_authority import write_json_multi_dest

    public = tmp_path / "www" / "alerts.json"
    private = tmp_path / "data" / "alerts.json"
    repo = tmp_path / "repo" / "public" / "data" / "alerts.json"
    public.parent.mkdir(parents=True)
    private.parent.mkdir(parents=True)
    repo.parent.mkdir(parents=True)

    payload = {
        "alerts": [{"type": "info", "message": "batch-hn"}],
        "count": 1,
        "source": "test",
        "generated_at": "2026-07-23T00:00:00+00:00",
    }
    result = write_json_multi_dest(
        payload,
        public_path=public,
        private_path=private,
        repo_path=repo,
    )
    assert result.wrote_public is True
    assert result.wrote_private is True
    assert result.wrote_repo is True
    body = public.read_bytes()
    assert private.read_bytes() == body
    assert repo.read_bytes() == body
    modes = [(p.stat().st_mode & 0o777) for p in (public, private, repo)]
    assert modes == [0o644, 0o644, 0o644], list(map(oct, modes))
    on_disk = json.loads(body)
    assert on_disk["count"] == 1
    assert on_disk["alerts"][0]["message"] == "batch-hn"


def test_write_json_multi_dest_skips_auto_repo_under_pytest(tmp_path, monkeypatch):
    """Case DB: under pytest, auto soft-mirror must not clobber real checkout."""
    from src.monitor import signal_authority as auth

    public = tmp_path / "www" / "alerts.json"
    private = tmp_path / "data" / "alerts.json"
    public.parent.mkdir(parents=True)
    private.parent.mkdir(parents=True)

    sentinel = tmp_path / "MUST_NOT_WRITE.json"
    monkeypatch.setattr(
        auth,
        "default_repo_public_data_path",
        lambda name: sentinel,
    )
    # PYTEST_CURRENT_TEST is set by pytest; auto_repo path must skip.
    result = auth.write_json_multi_dest(
        {"alerts": [], "count": 0},
        public_path=public,
        private_path=private,
        soft_mirror_repo=True,
        repo_filename="alerts.json",
    )
    assert result.wrote_public is True
    assert result.wrote_private is True
    assert result.wrote_repo is False
    assert not sentinel.exists()


def test_publish_health_alerts_json_three_dest_equal(tmp_path, monkeypatch):
    """Case DC: health job multi-dests alerts to public + private + repo soft-mirror."""
    from src.monitor.health_check import publish_health_alerts_json

    public = tmp_path / "public"
    data = tmp_path / "data"
    repo = tmp_path / "repo" / "public" / "data"
    public.mkdir()
    data.mkdir()
    repo.mkdir(parents=True)

    monkeypatch.setattr("src.monitor.health_check.PUBLIC_DATA_DIR", public)
    monkeypatch.setattr("src.monitor.health_check.DATA_DIR", data)
    monkeypatch.setattr(
        "src.monitor.health_check.HEALTH_PATH", data / "health.json"
    )
    monkeypatch.setattr(
        "src.monitor.signal_authority.default_repo_public_data_path",
        lambda name: repo / name,
    )
    # Auto soft-mirror is gated off when PYTEST_CURRENT_TEST is set (Case DB).
    # Clear it so this integration case exercises the production three-dest path
    # while default_repo_public_data_path is redirected into tmp_path.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    report = {
        "status": "ok",
        "generated_at": "2026-07-23T12:00:00+00:00",
        "timestamp": "2026-07-23T12:00:00+00:00",
        "checks": {},
    }
    out = publish_health_alerts_json(report)
    assert out is not None
    assert out == public / "alerts.json"
    pub_body = (public / "alerts.json").read_bytes()
    priv_body = (data / "alerts.json").read_bytes()
    repo_body = (repo / "alerts.json").read_bytes()
    assert pub_body == priv_body == repo_body
    modes = [
        ((public / "alerts.json").stat().st_mode & 0o777),
        ((data / "alerts.json").stat().st_mode & 0o777),
        ((repo / "alerts.json").stat().st_mode & 0o777),
    ]
    assert modes == [0o644, 0o644, 0o644], list(map(oct, modes))
    payload = json.loads(pub_body)
    assert payload.get("source") == "health_check_job"
    assert payload.get("generated_at") == "2026-07-23T12:00:00+00:00"
    assert "alerts" in payload
    # Item 35: webhook disclosure present in BOTH writers' artifact
    assert payload.get("alerting") == {
        "webhook_configured": False,
        "webhook_source": "none",
    }
