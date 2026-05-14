"""
pytest configuration: heavy-test markers, experimental gating.

Default: heavy tests (torch/xgboost) are SKIPPED to avoid OOM/CPU stalls.
The full 2853-test suite exhausts ~6.6GB RAM in a single process because
torch (63MB) + xgboost (78MB) accumulate alongside test data fixtures.

Use --include-heavy to run heavy ML tests alongside lightweight ones:
  pytest tests/ --include-heavy

Or run heavy tests separately:
  pytest tests/ -m heavy --include-heavy

Lightweight tests only (default, safe):
  pytest tests/
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--include-heavy",
        action="store_true",
        default=False,
        help="Include tests that require heavy ML libraries (torch, xgboost). Disabled by default.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "heavy: tests requiring heavy ML libraries (torch, xgboost)")


def pytest_collection_modifyitems(config, items):
    # Default: skip heavy tests unless --include-heavy is set
    if not config.getoption("--include-heavy"):
        skip_heavy = pytest.mark.skip(reason="heavy tests disabled (use --include-heavy to enable)")
        count = 0
        for item in items:
            if "heavy" in item.keywords:
                item.add_marker(skip_heavy)
                count += 1
        if count > 0:
            print(f"\n[Skipped {count} heavy tests (torch/xgboost). Use --include-heavy to run them.]")
