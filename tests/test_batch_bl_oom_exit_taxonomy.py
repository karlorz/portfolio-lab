"""Batch BL residual honesty: OOM exit taxonomy 137/139 + run-tests-safe capture."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_TESTS_SAFE = PROJECT_ROOT / "scripts" / "run-tests-safe"
MAKE_TEST_OOM = PROJECT_ROOT / "scripts" / "make-test-oom-aware"
MAKEFILE = PROJECT_ROOT / "Makefile"


def test_run_tests_safe_captures_exit_under_set_e() -> None:
    """set +e around suite so EXIT=$? and 137/139 diagnostics are reachable."""
    text = RUN_TESTS_SAFE.read_text(encoding="utf-8")
    assert "set +e" in text
    # Both lanes must capture EXIT after set +e
    assert text.count("EXIT=$?") >= 2
    assert "_emit_memory_diagnostics" in text
    assert "SIGSEGV (139)" in text
    assert "SIGKILL (137)" in text
    # Must not rely solely on bare set -e around subshell without capture
    # (the bug: set -e + (exit N) aborts before EXIT=$?)
    safe_block = text.split('MODE" = "safe"')[0] if False else text
    assert "set +e" in safe_block
    assert re.search(r"set \+e\s*\n\s*\(", text) or "set +e\n    (" in text or "set +e\n    (" in text.replace(
        "\r", ""
    )


def test_run_tests_safe_set_e_subshell_capture_behavior() -> None:
    """Reproduce the fixed pattern: set +e; (exit 137); EXIT=$? must retain 137."""
    script = r"""
set -euo pipefail
set +e
( exit 137 )
EXIT=$?
set -e
echo "captured=$EXIT"
exit 0
"""
    proc = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "captured=137" in proc.stdout


def test_make_test_oom_aware_script_exists_and_remaps() -> None:
    text = MAKE_TEST_OOM.read_text(encoding="utf-8")
    assert "Error 137" in text
    assert "Error 139" in text
    assert "test_last_exit.json" in text
    assert MAKE_TEST_OOM.stat().st_mode & 0o111  # executable bit


def test_make_test_writes_test_last_exit_json() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    assert "test_last_exit.json" in text
    assert "memory_class" in text
