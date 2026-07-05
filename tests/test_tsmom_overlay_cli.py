"""CLI output visibility tests for ``python -m src.signals.tsmom_overlay``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PORTFOLIO_LAB_ENABLE_ML": "0",
    }
    return subprocess.run(
        [sys.executable, "-m", "src.signals.tsmom_overlay", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )


def test_status_command_emits_visible_operator_output():
    result = _run_module("status")
    combined = result.stdout + result.stderr

    assert result.returncode == 0
    assert "TSMOM Overlay v2.52 - Status" in combined
    assert "Lookback:" in combined
    assert "Prices exist:" in combined


def test_live_command_emits_visible_recommendation_output():
    result = _run_module("live")
    combined = result.stdout + result.stderr

    assert result.returncode == 0
    assert combined.strip()
    assert "TSMOM Overlay v2.52" in combined
    assert "tsmom_allocation" in combined
    assert "signals" in combined
