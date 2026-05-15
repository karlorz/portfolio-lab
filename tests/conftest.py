"""
pytest configuration: ML feature gating.

ML features (torch/xgboost/sklearn/hmmlearn) are DISABLED by default via
a 4-layer defense to prevent OOM/CPU stalls. torch (~63MB) + sklearn
(~78MB) + hmmlearn (~23MB) accumulate in single-process test runs, causing
SIGKILL at ~47% through the 3900-test suite on low-resource hosts.

Layered defense (each layer independently prevents host CPU exhaust):
  0. collect_ignore — pytest never opens known heavy test files (0 CPU)
  1. Env var gate — PORTFOLIO_LAB_ENABLE_ML=0 set before any import
  2. builtins.__import__ hook — blocks ML imports at interpreter level
  3. Post-collection leak check — warns if real ML libs evaded all guards
  4. ulimit -v (Makefile) — OS kernel enforces 1GB virtual memory cap

Default (safe, ~3700 non-ML tests, zero ML libs loaded):
  make test
  pytest tests/

Include ML tests:
  PORTFOLIO_LAB_ENABLE_ML=1 pytest tests/ --include-heavy

All tests including ML:
  PORTFOLIO_LAB_ENABLE_ML=1 pytest tests/
"""

import os
import sys
import builtins
import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Layer 0: collect_ignore — prevent pytest from OPENING heavy test files
# ═══════════════════════════════════════════════════════════════════════════
# When ML is disabled, these files are never read, imported, or parsed.
# This is the strongest guard — pytest skips them during directory listing,
# before any import machinery runs. Zero CPU cost, zero memory cost.
#
# New heavy test files MUST be added here to maintain the guarantee.
# The import hook (Layer 2) catches any that are missed, but files listed
# here are NEVER opened.

_HEAVY_TEST_FILES = [
    "test_execution_agent.py",
    "test_marl_trainer.py",
    "test_risk_agent_hmm.py",
    "test_stacking_trainer.py",
    "test_regime_ml.py",
]

if os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") != "1":
    collect_ignore = list(_HEAVY_TEST_FILES)
else:
    collect_ignore = []


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: Env var gate (set before ANY test module import)
# ═══════════════════════════════════════════════════════════════════════════

if "PORTFOLIO_LAB_ENABLE_ML" not in os.environ:
    os.environ["PORTFOLIO_LAB_ENABLE_ML"] = "0"

# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: builtins.__import__ hook — blocks ML libs at interpreter level
# ═══════════════════════════════════════════════════════════════════════════
# Catches imports that evaded collect_ignore (e.g. a non-heavy test file
# that transitively imports an ML library through a src module chain).
#
# sys.modules is checked by CPython BEFORE __import__ is called, so stub
# entries registered by base_agent.py (torch, torch.nn) short-circuit this
# hook — only real imports that would actually load the package reach here.

_ML_BLOCKED = frozenset({"torch", "sklearn", "xgboost", "hmmlearn"})
_original_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Block real ML library imports when PORTFOLIO_LAB_ENABLE_ML=0."""
    ml_enabled = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"
    if not ml_enabled:
        top_level = name.split(".")[0]
        if top_level in _ML_BLOCKED:
            raise ImportError(
                f"ML library '{name}' blocked: PORTFOLIO_LAB_ENABLE_ML=0. "
                f"Set PORTFOLIO_LAB_ENABLE_ML=1 to enable ML features."
            )
    return _original_import(name, globals, locals, fromlist, level)


builtins.__import__ = _guarded_import


# ═══════════════════════════════════════════════════════════════════════════
# pytest hooks
# ═══════════════════════════════════════════════════════════════════════════

def pytest_addoption(parser):
    parser.addoption(
        "--include-heavy",
        action="store_true",
        default=False,
        help="Run tests marked heavy (torch/xgboost/sklearn/hmmlearn). "
             "Requires PORTFOLIO_LAB_ENABLE_ML=1.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "heavy: tests requiring heavy ML libraries (torch, xgboost, sklearn, hmmlearn)",
    )


def pytest_collection_modifyitems(config, items):
    ml_enabled = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"

    # ═══════════════════════════════════════════════════════════════════════
    # Layer 3: Post-collection leak check
    # ═══════════════════════════════════════════════════════════════════════
    # Detects real ML libraries that evaded all prior layers (e.g. installed
    # via sys.modules manipulation without using __import__). This fires
    # AFTER all test files are collected — if a real ML lib is present,
    # something bypassed the import hook.
    if not ml_enabled:
        leaked = []
        for lib in ("torch", "xgboost", "sklearn", "hmmlearn"):
            mod = sys.modules.get(lib)
            if mod is not None:
                has_file = hasattr(mod, "__file__") and mod.__file__ is not None
                has_version = hasattr(mod, "__version__") and mod.__version__ is not None
                if has_file or has_version:
                    leaked.append(lib)
        if leaked:
            import warnings
            warnings.warn(
                f"ML library(ies) {leaked} loaded during test collection despite "
                f"PORTFOLIO_LAB_ENABLE_ML=0. The import hook may have been "
                f"bypassed — check for sys.modules injections."
            )

    # Skip heavy tests unless both env var AND CLI flag are set.
    # Even though collect_ignore (Layer 0) already prevents heavy files from
    # being opened, this handles the case where --include-heavy is passed but
    # PORTFOLIO_LAB_ENABLE_ML=0 (import hook still blocks ML libs).
    if not ml_enabled or not config.getoption("--include-heavy"):
        skip_heavy = pytest.mark.skip(
            reason="heavy ML tests skipped (set PORTFOLIO_LAB_ENABLE_ML=1 "
                   "and use --include-heavy)"
        )
        count = 0
        for item in items:
            if "heavy" in item.keywords:
                item.add_marker(skip_heavy)
                count += 1
        if count > 0:
            print(
                f"\n[Skipped {count} heavy ML tests. "
                f"Use PORTFOLIO_LAB_ENABLE_ML=1 --include-heavy to run.]"
            )
