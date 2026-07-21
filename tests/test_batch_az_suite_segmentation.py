"""Batch AZ / S18: suite segmentation targets (unit / generator / integration)."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = PROJECT_ROOT / "Makefile"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"

# Keep in sync with Makefile TEST_INTEGRATION_FILES
EXPECTED_INTEGRATION_FILES = (
    "tests/test_collect_signals_integration.py",
    "tests/test_e2e_overlay_pipeline.py",
    "tests/test_integration.py",
    "tests/test_rebalancing_integration.py",
    "tests/test_rebalancing_integration_cli.py",
    "tests/test_regime_bandit_integration.py",
    "tests/test_signal_backtest_integration.py",
    "tests/test_signal_tsmom_integration.py",
    "tests/test_tsmom_integration.py",
    "tests/test_vix_vol_targeting_integration.py",
)


def _makefile_text() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def test_makefile_defines_s18_segment_targets():
    mk = _makefile_text()
    for target in ("test-unit:", "test-generator:", "test-integration:"):
        assert target in mk, f"missing {target}"
    assert "TEST_GENERATOR_FILE" in mk
    assert "TEST_INTEGRATION_FILES" in mk
    assert "tests/test_generator.py" in mk


def test_makefile_help_discloses_segments_and_test_fast_subset():
    mk = _makefile_text()
    help_m = re.search(r"^help:\n(?P<body>(?:\t.*\n)+)", mk, re.M)
    assert help_m, "help target not found"
    body = help_m.group("body")
    assert "test-unit" in body
    assert "test-generator" in body
    assert "test-integration" in body
    assert "test-fast" in body
    # help discloses test-fast is ensemble subset, not full unit
    assert "subset" in body.lower() or "not full unit" in body.lower()


def test_test_unit_ignores_generator_and_integration_paths():
    mk = _makefile_text()
    # Extract test-unit recipe
    m = re.search(r"^test-unit:\n(?P<body>(?:\t.*\n)+)", mk, re.M)
    assert m, "test-unit target missing"
    body = m.group("body")
    assert "--ignore=" in body or "IGNORE_ARGS" in body
    assert "test_generator" in body or "TEST_GENERATOR_FILE" in mk
    for path in EXPECTED_INTEGRATION_FILES:
        assert path in mk, f"integration path missing from Makefile: {path}"
        assert Path(PROJECT_ROOT / path).is_file(), f"missing on disk: {path}"


def test_test_generator_runs_only_generator_file():
    mk = _makefile_text()
    m = re.search(r"^test-generator:\n(?P<body>(?:\t.*\n)+)", mk, re.M)
    assert m
    body = m.group("body")
    assert "TEST_GENERATOR_FILE" in body or "test_generator.py" in body
    assert "tests/" not in body or "test_generator" in body


def test_safe_prelude_isolates_public_and_memory_cap():
    mk = _makefile_text()
    # Segments should inherit isolation (mktemp PUBLIC + 6GB ulimit)
    for target in ("test-unit:", "test-generator:", "test-integration:"):
        m = re.search(rf"^{re.escape(target)}\n(?P<body>(?:\t.*\n)+)", mk, re.M)
        assert m, target
        body = m.group("body")
        assert "PUBLIC_DATA_DIR" in body
        assert "6291456" in body or "ulimit -v" in body
        assert "plab-pytest-public" in body


def test_pyproject_registers_unit_integration_markers():
    text = PYPROJECT.read_text(encoding="utf-8")
    assert '"unit:' in text or "unit:" in text
    assert '"integration:' in text or "integration:" in text


def test_conftest_registers_unit_integration_markers():
    conf = (PROJECT_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "unit:" in conf
    assert "integration:" in conf


def test_full_test_target_still_present_as_gate():
    mk = _makefile_text()
    assert re.search(r"^test:\n", mk, re.M)
    m = re.search(r"^test:\n(?P<body>(?:\t.*\n)+)", mk, re.M)
    assert m
    body = m.group("body")
    assert "3600" in body
    assert "6291456" in body
