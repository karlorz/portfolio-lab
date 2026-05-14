"""
pytest configuration: heavy-test markers, --skip-heavy flag.
Adds @pytest.mark.heavy automatically to test files that import torch or xgboost.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-heavy",
        action="store_true",
        default=False,
        help="Skip tests that require heavy ML libraries (torch, xgboost)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "heavy: tests requiring heavy ML libraries (torch, xgboost)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-heavy"):
        skip_heavy = pytest.mark.skip(reason="--skip-heavy flag set")
        for item in items:
            if "heavy" in item.keywords:
                item.add_marker(skip_heavy)
