"""Regression tests for cron status verification scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRON_VERIFY = PROJECT_ROOT / "scripts" / "cron_verify.py"


def test_cron_verify_runs_directly_without_makefile_pythonpath(tmp_path: Path) -> None:
    """Direct script execution should bootstrap imports outside Makefile/uv."""
    from src.cron_compat import CRON_TARGETS

    status_file = tmp_path / "cron_status.json"
    status_file.write_text(
        '{"jobs": [%s]}'
        % ",".join(f'{{"name": "{name}"}}' for name in CRON_TARGETS)
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    env["CRON_STATUS_FILE"] = str(status_file)

    result = subprocess.run(
        [sys.executable, str(CRON_VERIFY)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout
