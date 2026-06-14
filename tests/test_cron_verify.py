"""Tests for cron status verification by scheduler backend."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRON_VERIFY_SCRIPT = PROJECT_ROOT / "scripts" / "cron_verify.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cron_verify", CRON_VERIFY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tasker_backend_uses_tasker_registry_expected_jobs(tmp_path, capsys) -> None:
    module = _load_module()
    expected = sorted(module.expected_jobs_for_backend("tasker"))
    status_file = tmp_path / "cron_status.json"
    status_file.write_text(
        json.dumps({"backend": "tasker", "jobs": [{"name": name} for name in expected]}),
        encoding="utf-8",
    )

    assert module.verify_status(status_file) == 0
    assert "expected tasker targets present" in capsys.readouterr().out


def test_legacy_backend_uses_cron_targets(tmp_path, capsys) -> None:
    module = _load_module()
    expected = sorted(module.CRON_TARGETS)
    status_file = tmp_path / "cron_status.json"
    status_file.write_text(
        json.dumps({"backend": "hermes", "jobs": [{"name": name} for name in expected]}),
        encoding="utf-8",
    )

    assert module.verify_status(status_file) == 0
    assert "expected hermes targets present" in capsys.readouterr().out


def test_verify_fails_when_expected_tasker_job_missing(tmp_path, capsys) -> None:
    module = _load_module()
    expected = sorted(module.expected_jobs_for_backend("tasker"))
    status_file = tmp_path / "cron_status.json"
    status_file.write_text(
        json.dumps({"backend": "tasker", "jobs": [{"name": name} for name in expected[1:]]}),
        encoding="utf-8",
    )

    assert module.verify_status(status_file) == 1
    assert "FAIL: Missing jobs" in capsys.readouterr().out


def test_crontab_make_targets_are_derived_from_cron_targets() -> None:
    module = _load_module()

    assert module.expected_crontab_make_targets(
        [
            "portfolio-lab-data",
            "portfolio-lab-overlay-signals",
            "portfolio-lab-position-sync",
            "portfolio-lab-mark-to-market",
        ]
    ) == {"data", "overlay-signals", "sync"}


def test_verify_crontab_targets_fails_when_cron_target_missing(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "CRON_TARGETS",
        ["portfolio-lab-data", "portfolio-lab-overlay-signals"],
    )
    crontab_file = tmp_path / "crontab"
    crontab_file.write_text(
        "5 * * * * CRON_BACKEND=crontab make -C /repo data\n",
        encoding="utf-8",
    )

    assert module.verify_crontab_targets(crontab_file) == 1
    output = capsys.readouterr().out
    assert "overlay-signals" in output
    assert "build" not in output


def test_verify_crontab_targets_accepts_targets_from_cron_targets(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "CRON_TARGETS",
        ["portfolio-lab-data", "portfolio-lab-overlay-signals"],
    )
    crontab_file = tmp_path / "crontab"
    crontab_file.write_text(
        "\n".join(
            [
                "5 * * * * CRON_BACKEND=crontab make -C /repo data",
                "3,23,43 * * * * CRON_BACKEND=crontab make -C /repo overlay-signals",
            ]
        ),
        encoding="utf-8",
    )

    assert module.verify_crontab_targets(crontab_file) == 0
    assert "expected crontab targets present" in capsys.readouterr().out
