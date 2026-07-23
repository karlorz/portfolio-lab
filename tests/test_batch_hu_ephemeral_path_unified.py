"""Batch HU: single ephemeral-path classifier for write + restamp guards.

Residuals after HP: is_ephemeral_write_path and is_ephemeral_restamp_path
duplicated the same plab/pytest rules and could drift (false green lag stamp
vs multi-dest SSOT pollution). Authority: never touches target_allocations.
"""

from __future__ import annotations

import pytest


SAMPLES = [
    ("/tmp/plab-pytest-public.abc/data/signals.json", True),
    ("/tmp/plab-pytest-public.xyz", True),
    ("/tmp/pytest-of-root/pytest-1/data/health.json", True),
    ("/tmp/pytest-99/alerts.json", True),
    ("/var/folders/xx/pytest/T/health.json", True),
    ("/root/projects/portfolio-lab/data/health.json", False),
    ("/var/www/portfolio-lab/data/signals.json", False),
    ("/root/projects/portfolio-lab/public/data/alerts.json", False),
    (None, False),
    ("", False),
]


@pytest.mark.parametrize("path,expected", SAMPLES)
def test_write_and_restamp_ephemeral_classifiers_agree(path, expected) -> None:
    from src.monitor.signal_authority import is_ephemeral_write_path
    from src.monitor.repo_public_mirror_lag import is_ephemeral_restamp_path

    assert is_ephemeral_write_path(path) is expected
    assert is_ephemeral_restamp_path(path) is expected
    assert is_ephemeral_write_path(path) == is_ephemeral_restamp_path(path)


def test_restamp_delegates_to_write_classifier(monkeypatch) -> None:
    """Case DO: restamp path must call the shared write classifier (no drift)."""
    from src.monitor import repo_public_mirror_lag as mlag
    from src.monitor import signal_authority as auth

    calls: list[str] = []

    def _probe(path):
        calls.append(str(path))
        return "sentinel" in str(path)

    monkeypatch.setattr(auth, "is_ephemeral_write_path", _probe)
    assert mlag.is_ephemeral_restamp_path("/tmp/sentinel-path/x.json") is True
    assert mlag.is_ephemeral_restamp_path("/tmp/other/x.json") is False
    assert calls == ["/tmp/sentinel-path/x.json", "/tmp/other/x.json"]
