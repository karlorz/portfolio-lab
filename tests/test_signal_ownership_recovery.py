"""Signal ownership map + sustained unavailability recovery alerts."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.monitor.signal_ownership import (
    SIGNAL_OWNERSHIP,
    annotate_unavailable_signals,
    recovery_summary,
)
from src.monitor.alerting import (
    AlertChannel,
    AlertLevel,
    check_sustained_unavailability_and_alert,
)


def test_ownership_covers_common_unavailable_keys() -> None:
    for key in (
        "behavioral_sentiment",
        "calendar_seasonality",
        "crypto_allocation",
        "collar",
        "bond_momentum",
        "kurtosis_regime",
        "alternative_data",
        "stacking_ensemble",
    ):
        assert key in SIGNAL_OWNERSHIP
        assert SIGNAL_OWNERSHIP[key]["job"].startswith("portfolio-lab-")


def test_annotate_marks_ml_off_intentional() -> None:
    rows = annotate_unavailable_signals(
        ["behavioral_sentiment", "collar"],
        ml_enabled=False,
    )
    by_name = {r["signal"]: r for r in rows}
    assert by_name["behavioral_sentiment"]["intentional_when_ml_off"] is True
    assert by_name["collar"]["intentional_when_ml_off"] is False
    assert "overlay-signals" in by_name["collar"]["recovery"]


def test_recovery_summary_excludes_intentional_ml_off() -> None:
    rows = annotate_unavailable_signals(
        ["behavioral_sentiment", "collar", "kurtosis_regime"],
        ml_enabled=False,
    )
    summary = recovery_summary(rows)
    assert summary["intentional_ml_off_count"] == 1
    assert summary["actionable_unavailable_count"] == 2
    assert "portfolio-lab-overlay-signals" in summary["jobs_to_rerun"]
    assert "overlay-signals" in summary["make_targets"]


def test_sustained_unavailability_fires_under_old_kill(tmp_path, monkeypatch) -> None:
    from src.monitor import alerting as al

    old = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    (tmp_path / "kill_switch.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "level": "halt",
                "reason": "unresolved_incident:signal_staleness",
                "timestamp": old,
            }
        )
    )
    sent = []

    def _capture(channel, level, message, details=None):
        sent.append((channel, level, message, details))
        return True

    monkeypatch.setattr(al, "send_alert", _capture)
    staleness = {
        "unavailable_signals": [
            "collar",
            "bond_momentum",
            "kurtosis_regime",
            "calendar_seasonality",
            "crypto_allocation",
            "factor_rotation",
        ],
        "stale_signals": [],
        "total_count": 23,
        "healthy_count": 10,
    }
    fired = check_sustained_unavailability_and_alert(
        staleness,
        data_dir=tmp_path,
        min_unavailable=5,
        min_hours=2.0,
    )
    assert fired is True
    assert sent
    channel, level, message, details = sent[0]
    assert channel == AlertChannel.SIGNAL_RECOVERY
    assert level == AlertLevel.WARN
    assert "Sustained overlay unavailability" in message
    assert details["policy"] == "recovery_advisory_only_no_kill_clear"
    assert details["actionable_unavailable_count"] >= 5


def test_sustained_unavailability_skips_without_kill(tmp_path, monkeypatch) -> None:
    from src.monitor import alerting as al

    sent = []
    monkeypatch.setattr(al, "send_alert", lambda *a, **k: sent.append(a) or True)
    fired = check_sustained_unavailability_and_alert(
        {
            "unavailable_signals": ["collar"] * 6,
            "total_count": 23,
            "healthy_count": 10,
        },
        data_dir=tmp_path,
        min_unavailable=5,
        min_hours=0,
    )
    assert fired is False
    assert sent == []


def test_public_data_prices_not_git_tracked() -> None:
    """Tracked-prices sticky: public/data runtime artifacts must stay untracked."""
    import subprocess

    out = subprocess.run(
        ["git", "ls-files", "public/data/prices.json", "public/data/historical.json"],
        capture_output=True,
        text=True,
        check=False,
    )
    tracked = [line for line in out.stdout.splitlines() if line.strip()]
    assert tracked == [], f"unexpected tracked public data files: {tracked}"
