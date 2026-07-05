"""CLI output visibility tests for ``python -m src.signals.stacking_feature_engine``."""

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
        [sys.executable, "-m", "src.signals.stacking_feature_engine", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )


def test_names_command_emits_visible_feature_names():
    result = _run_module("--names")
    combined = result.stdout + result.stderr

    assert result.returncode == 0
    assert "base_multi_speed_momentum" in combined
    assert "acc90d_unified_overlay" in combined


def test_test_command_emits_visible_demo_output():
    result = _run_module("--test")
    combined = result.stdout + result.stderr

    assert result.returncode == 0
    assert "Stacking Feature Engine Demo" in combined
    assert "Feature vector created" in combined
    assert "Shape: (128," in combined
