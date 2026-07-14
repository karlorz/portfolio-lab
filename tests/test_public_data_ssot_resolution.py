"""PUBLIC_DATA_DIR SSOT: ops auditors must not default to stale repo public/data."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.paths import PROJECT_ROOT, resolve_ops_public_data_dir
from scripts.check_public_data_consistency import check_public_data_consistency, main as consistency_main


def test_resolve_prefers_explicit_public_dir(tmp_path: Path) -> None:
    explicit = tmp_path / "www" / "data"
    explicit.mkdir(parents=True)
    got = resolve_ops_public_data_dir(
        PROJECT_ROOT,
        explicit,
        env={},
        live_public_data_dir=tmp_path / "other",
    )
    assert got == explicit.resolve()


def test_resolve_prefers_public_data_dir_env(tmp_path: Path) -> None:
    live = tmp_path / "live"
    live.mkdir()
    got = resolve_ops_public_data_dir(
        PROJECT_ROOT,
        None,
        env={"PUBLIC_DATA_DIR": str(live)},
        live_public_data_dir=live,
    )
    assert got == live.resolve()


def test_resolve_allows_repo_when_no_live_tree(tmp_path: Path) -> None:
    missing_live = tmp_path / "no-such-www"
    got = resolve_ops_public_data_dir(
        PROJECT_ROOT,
        None,
        env={},
        live_public_data_dir=missing_live,
    )
    assert got == (PROJECT_ROOT / "public" / "data").resolve()


def test_resolve_fails_closed_for_this_checkout_when_live_exists(tmp_path: Path) -> None:
    live = tmp_path / "www-data"
    live.mkdir()
    with pytest.raises(ValueError, match="PUBLIC_DATA_DIR is unset"):
        resolve_ops_public_data_dir(
            PROJECT_ROOT,
            None,
            env={},
            live_public_data_dir=live,
            allow_repo_public_data=False,
        )


def test_resolve_allow_repo_override(tmp_path: Path) -> None:
    live = tmp_path / "www-data"
    live.mkdir()
    got = resolve_ops_public_data_dir(
        PROJECT_ROOT,
        None,
        env={},
        live_public_data_dir=live,
        allow_repo_public_data=True,
    )
    assert got == (PROJECT_ROOT / "public" / "data").resolve()


def test_resolve_allows_fixture_app_dir_even_if_live_exists(tmp_path: Path) -> None:
    """tmp_path checkouts used by tests/deploy fixtures are not this repo root."""
    live = tmp_path / "www-data"
    live.mkdir()
    app = tmp_path / "fixture-app"
    (app / "public" / "data").mkdir(parents=True)
    got = resolve_ops_public_data_dir(
        app,
        None,
        env={},
        live_public_data_dir=live,
        allow_repo_public_data=False,
    )
    assert got == (app / "public" / "data").resolve()


def test_consistency_api_fails_closed_on_live_ssot_without_env(tmp_path: Path) -> None:
    live = tmp_path / "live-public"
    live.mkdir()
    result = check_public_data_consistency(
        PROJECT_ROOT,
        env={},
        live_public_data_dir=live,
        allow_repo_public_data=False,
    )
    assert result.ok is False
    assert any("PUBLIC_DATA_DIR is unset" in e for e in result.errors)


def test_consistency_cli_exits_1_when_live_ssot_unspecified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    live = tmp_path / "live-public"
    live.mkdir()
    # Clear env PUBLIC_DATA_DIR if present
    monkeypatch.delenv("PUBLIC_DATA_DIR", raising=False)
    monkeypatch.setenv("PORTFOLIO_LAB_LIVE_PUBLIC_DATA_DIR", str(live))
    # Force cwd-based default app-dir to this project
    code = consistency_main(["--app-dir", str(PROJECT_ROOT)])
    assert code == 1
    err = capsys.readouterr().err
    assert "PUBLIC_DATA_DIR is unset" in err


def test_consistency_cli_allow_repo_public_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = tmp_path / "live-public"
    live.mkdir()
    monkeypatch.delenv("PUBLIC_DATA_DIR", raising=False)
    monkeypatch.setenv("PORTFOLIO_LAB_LIVE_PUBLIC_DATA_DIR", str(live))
    # allow flag should not raise; may fail on missing artifacts but not SSOT gate
    code = consistency_main(
        ["--app-dir", str(PROJECT_ROOT), "--allow-repo-public-data"]
    )
    # Real checkout public/data may or may not be consistent — only assert not SSOT error via exit
    # Exit can be 0 or 1 from real checks; re-run API with allow to ensure no ValueError path
    result = check_public_data_consistency(
        PROJECT_ROOT,
        allow_repo_public_data=True,
        env={},
        live_public_data_dir=live,
    )
    assert not any("PUBLIC_DATA_DIR is unset" in e for e in result.errors)
    assert code in (0, 1)
