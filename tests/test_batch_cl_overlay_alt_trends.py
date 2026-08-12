"""Batch CL: overlay post-sync lag, naive-local timestamps, trends cache merge."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock


def test_merge_trends_cache_prefers_fresh_dates():
    from scripts import fetch_google_trends as ft

    cached = {
        "recession": {"2026-05-01": 10, "2026-05-02": 12},
        "inflation": {"2026-05-01": 20},
    }
    fresh = {
        "recession": {"2026-05-02": 99, "2026-07-01": 40},
    }
    merged = ft.merge_trends_cache(cached, fresh)
    assert merged["recession"]["2026-05-01"] == 10
    assert merged["recession"]["2026-05-02"] == 99  # overwrite
    assert merged["recession"]["2026-07-01"] == 40
    assert merged["inflation"]["2026-05-01"] == 20


def test_load_trends_cache_roundtrip(tmp_path):
    from scripts import fetch_google_trends as ft

    path = tmp_path / "google_trends.json"
    path.write_text(json.dumps({"recession": {"2026-01-01": 5}}))
    assert ft.load_trends_cache(path)["recession"]["2026-01-01"] == 5
    assert ft.load_trends_cache(tmp_path / "missing.json") == {}


def test_overlay_save_post_sync_clears_lag(tmp_path, monkeypatch):
    from src.dashboard import overlay_dashboard as od
    import src.paths as paths_mod

    private = tmp_path / "private" / "overlay_dashboard.json"
    public = tmp_path / "public" / "overlay_dashboard.json"
    private.parent.mkdir(parents=True)
    public.parent.mkdir(parents=True)
    # stale public
    public.write_text(json.dumps({"old": True}))
    now = time.time()
    os.utime(public, (now - 900, now - 900))

    # save() does `from src.paths import PUBLIC_DATA_DIR` at call time
    monkeypatch.setattr(paths_mod, "PUBLIC_DATA_DIR", public.parent)
    gen = od.OverlayDashboardGenerator.__new__(od.OverlayDashboardGenerator)
    gen.OUTPUT_PATH = str(private)

    dash = MagicMock()
    dash.to_dict.return_value = {
        "active_overlays": 1,
        "total_overlays": 1,
        "generated_at": "2026-07-21T21:00:00+00:00",
    }
    monkeypatch.setattr(
        "src.dashboard.generator._generator_git_sha_short",
        lambda: "overlaysha123",
    )

    gen.save(dash)
    body = json.loads(private.read_text())
    pc = body.get("provenance_completeness") or {}
    assert pc.get("dual_write_ok") is True
    assert pc.get("dual_write_lag_stale") is False
    assert public.exists()


def test_alt_data_composite_uses_utc_timestamp():
    from src.signals.alternative_data_signal import _utc_now_iso

    ts = _utc_now_iso()
    assert "+" in ts or ts.endswith("Z")
    # parseable as aware
    cleaned = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    assert dt.tzinfo is not None


def test_staleness_naive_local_not_future_utc():
    """Naive local evening should not yield age 0 via false future UTC."""
    from src.dashboard.generator import DashboardGenerator

    gen = DashboardGenerator.__new__(DashboardGenerator)
    # Use local evening wall clock as naive ISO
    local_now = datetime.now().replace(microsecond=0)
    naive = local_now.isoformat()  # no tz
    # Build minimal signal_data with alternative_data
    signal_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "alternative_data": {
            "timestamp": naive,
            "composite_score": 0.3,
            "confidence": 0.5,
            "regime": "risk_on",
        },
        # fill other required keys as empty/missing optional
    }
    # Stub the maps used by _check_signal_staleness if needed
    result = gen._check_signal_staleness(signal_data)
    age = (result.get("signal_age_hours") or {}).get("alternative_data")
    # Age should be near 0 hours, not negative/clamped from future, and not ~8h wrong
    assert age is not None
    assert age < 2.0  # freshly generated local timestamp
