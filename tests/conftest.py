"""
pytest configuration: ML feature gating.

ML features (torch/xgboost) are DISABLED by default via env var to prevent
OOM/CPU stalls. torch (~63MB) + xgboost (~78MB) accumulate in single-process
test runs, causing SIGKILL at ~47% through the 2853-test suite.

Default (safe, ~2700 tests):
  pytest tests/

Include ML tests:
  PORTFOLIO_LAB_ENABLE_ML=1 pytest tests/ -m heavy

All tests including ML:
  PORTFOLIO_LAB_ENABLE_ML=1 pytest tests/
"""

import os
import pytest

# Set before ANY test module imports to prevent torch/xgboost loading.
# Individual tests can override: os.environ["PORTFOLIO_LAB_ENABLE_ML"] = "1"
if "PORTFOLIO_LAB_ENABLE_ML" not in os.environ:
    os.environ["PORTFOLIO_LAB_ENABLE_ML"] = "0"


def pytest_addoption(parser):
    parser.addoption(
        "--include-heavy",
        action="store_true",
        default=False,
        help="Run tests marked heavy (torch/xgboost). Requires PORTFOLIO_LAB_ENABLE_ML=1.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "heavy: tests requiring heavy ML libraries (torch, xgboost)")


def pytest_collection_modifyitems(config, items):
    # Skip heavy tests unless --include-heavy AND env var is set
    ml_enabled = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"

    if not ml_enabled or not config.getoption("--include-heavy"):
        skip_heavy = pytest.mark.skip(
            reason="heavy ML tests skipped (set PORTFOLIO_LAB_ENABLE_ML=1 and use --include-heavy)"
        )
        count = 0
        for item in items:
            if "heavy" in item.keywords:
                item.add_marker(skip_heavy)
                count += 1
        if count > 0:
            print(f"\n[Skipped {count} heavy ML tests. Use PORTFOLIO_LAB_ENABLE_ML=1 --include-heavy to run.]")
