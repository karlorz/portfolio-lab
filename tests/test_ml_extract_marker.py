import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_pytest(*args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PORTFOLIO_LAB_ENABLE_ML": "0",
    }
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _makefile_target_block(makefile: str, target: str) -> str:
    match = re.search(rf"^{re.escape(target)}:\n(?P<body>(?:\t.*\n|[ \t]*\n)+)", makefile, re.MULTILINE)
    assert match, f"Makefile target not found: {target}"
    return match.group("body")


def test_pytest_help_registers_ml_extract_option_and_marker() -> None:
    result = _run_pytest("--help")

    assert result.returncode == 0, result.stderr
    assert "--include-ml-extract" in result.stdout
    assert "ml_extract" in result.stdout


def test_ml_extract_selection_runs_without_heavy_or_ml_enabled() -> None:
    probe = REPO_ROOT / "tests" / "_ml_extract_marker_probe.py"
    probe.write_text(
        "\n".join(
            [
                "import pytest",
                "",
                "@pytest.mark.ml_extract",
                "def test_extracted_kernel_probe_stays_safe():",
                "    with pytest.raises(ImportError, match='PORTFOLIO_LAB_ENABLE_ML=0'):",
                "        __import__('xgboost')",
                "",
                "@pytest.mark.heavy",
                "def test_heavy_probe_must_stay_skipped():",
                "    raise AssertionError('heavy test ran during ml_extract lane')",
                "",
            ]
        )
    )
    try:
        result = _run_pytest(str(probe.relative_to(REPO_ROOT)), "-q", "--include-ml-extract", "--tb=short")
    finally:
        probe.unlink(missing_ok=True)

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "heavy test ran during ml_extract lane" not in output
    assert "1 passed" in output
    assert "1 skipped" in output


def test_run_tests_safe_exposes_safe_ml_extract_lane() -> None:
    runner = (REPO_ROOT / "scripts" / "run-tests-safe").read_text()

    assert "--ml-extract" in runner
    assert "--include-ml-extract" in runner
    assert "PORTFOLIO_LAB_ENABLE_ML=0" in runner


def test_makefile_exposes_safe_ml_extract_target() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text()
    target = _makefile_target_block(makefile, "test-ml-extract")

    assert "make test-ml-extract" in makefile
    assert "PORTFOLIO_LAB_ENABLE_ML=0" in target
    assert "--ml-extract" in target
    assert "--include-heavy" not in target
