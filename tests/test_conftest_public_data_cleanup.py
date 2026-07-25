"""Regression coverage for bare-pytest PUBLIC_DATA_DIR cleanup."""

from __future__ import annotations

from pathlib import Path


def _project_conftest(pytestconfig):
    for _, plugin in pytestconfig.pluginmanager.list_name_plugin():
        plugin_file = getattr(plugin, "__file__", "")
        if plugin_file and Path(plugin_file).resolve() == Path(__file__).with_name(
            "conftest.py"
        ).resolve():
            return plugin
    raise AssertionError("project tests/conftest.py plugin was not loaded")


def test_owned_public_data_root_is_removed_at_session_finish(
    tmp_path: Path, monkeypatch, pytestconfig,
) -> None:
    conftest = _project_conftest(pytestconfig)

    owned_root = tmp_path / "plab-pytest-public-owned"
    public = owned_root / "data"
    public.mkdir(parents=True)
    (public / "prices.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(conftest, "_ISOLATED_PUBLIC_DATA_ROOT", owned_root)
    # Never let this mid-suite regression test touch the session-owned DB root.
    monkeypatch.setattr(conftest, "_ISOLATED_MARKET_DB_ROOT", None)
    conftest.pytest_sessionfinish(session=None, exitstatus=0)

    assert not owned_root.exists()


def test_caller_provided_public_data_dir_is_not_removed(
    tmp_path: Path, monkeypatch, pytestconfig,
) -> None:
    conftest = _project_conftest(pytestconfig)

    caller_root = tmp_path / "caller-owned"
    caller_root.mkdir()
    monkeypatch.setattr(conftest, "_ISOLATED_PUBLIC_DATA_DIR", caller_root)
    monkeypatch.setattr(conftest, "_ISOLATED_PUBLIC_DATA_ROOT", None)
    monkeypatch.setattr(conftest, "_ISOLATED_MARKET_DB_ROOT", None)

    conftest.pytest_sessionfinish(session=None, exitstatus=0)

    assert caller_root.is_dir()
