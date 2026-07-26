"""Batch HQ: satellite jobs soft-mirror live WWW → repo public/data.

Residuals after HP:
- Live probe lag was 10–14 while health stamps lagged; soft-mirror exists on
  data/dashboard/ops-regen only. Satellites (health, overlay-dashboard,
  unified-dashboard, rebalance-health, attribution, overlay-signals) rewrite
  PUBLIC_DATA_DIR without refreshing the gitignored checkout mirror → lag burn.
- Mirror copy used bare write_bytes + .tmp replace (mode/umask risk vs multi-dest
  0o644 contract).

Authority: never touches target_allocations / order_router.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.makefile_helpers import makefile_recipe


def test_satellite_targets_include_mirror_soft_gate() -> None:
    """Case DK: health + satellite writers soft-mirror on success (|| non-blocking)."""
    makefile = Path("Makefile").read_text(encoding="utf-8")
    targets = (
        "health",
        "overlay-dashboard",
        "unified-dashboard",
        "rebalance-health",
        "attribution",
        "overlay-signals",
    )
    for name in targets:
        body = makefile_recipe(makefile, name)
        assert "mirror-repo-public-data" in body, f"{name} missing mirror soft-gate"
        assert "||" in body, f"{name} mirror must soft-fail with ||"
        assert "non-blocking" in body or "soft-failed" in body, (
            f"{name} needs soft-fail message"
        )
        # Only on success
        assert "if [ $$EXIT -eq 0 ]" in body or "if [ $$EXIT -eq 0 ] &&" in body, (
            f"{name} should gate mirror on success EXIT"
        )


def test_data_dashboard_still_have_mirror_soft_gate() -> None:
    """Regression: Batch CA data/dashboard gates remain."""
    makefile = Path("Makefile").read_text(encoding="utf-8")
    for name in ("data", "dashboard"):
        body = makefile_recipe(makefile, name)
        assert "mirror-repo-public-data" in body
        assert "||" in body


def test_atomic_write_bytes_mode_644(tmp_path: Path) -> None:
    """Case DL: soft-mirror dest lands world-readable 0o644 (not mkstemp 0600)."""
    from scripts.mirror_repo_public_data import _atomic_write_bytes

    dest = tmp_path / "public" / "data" / "widget.json"
    payload = b'{"generator_git_sha": "hqtest", "v": 1}\n'
    _atomic_write_bytes(dest, payload, mode=0o644)
    assert dest.is_file()
    assert dest.read_bytes() == payload
    assert (dest.stat().st_mode & 0o777) == 0o644


def test_mirror_repo_public_data_writes_mode_644(tmp_path: Path) -> None:
    """Case DM: full mirror path leaves dest 0o644 and equal bytes."""
    from scripts.mirror_repo_public_data import mirror_repo_public_data

    src = tmp_path / "live"
    dst = tmp_path / "repo"
    src.mkdir()
    body = {"generator_git_sha": "hq-dm", "status": "ok", "n": 2}
    (src / "health.json").write_text(json.dumps(body), encoding="utf-8")
    report = mirror_repo_public_data(
        source_root=src,
        dest_root=dst,
        files=("health.json",),
        restamp_health_lag=False,
    )
    assert report.copied == ["health.json"]
    out = dst / "health.json"
    assert out.is_file()
    assert (out.stat().st_mode & 0o777) == 0o644
    assert json.loads(out.read_text(encoding="utf-8"))["generator_git_sha"] == "hq-dm"
