"""
pytest configuration: ML feature gating.

ML features (torch/xgboost/sklearn/hmmlearn) are DISABLED by default via
env var + import hook to prevent OOM/CPU stalls. torch (~63MB) + sklearn
(~78MB) + hmmlearn (~23MB) accumulate in single-process test runs, causing
SIGKILL at ~47% through the 3900-test suite.

Layered defense:
  1. src/agents/*.py use conditional imports (PORTFOLIO_LAB_ENABLE_ML guard)
  2. builtins.__import__ hook blocks any ML import at the interpreter level
  3. Post-collection leak check warns if real ML libs snuck through

Default (safe, ~3700 non-ML tests):
  pytest tests/

Include ML tests:
  PORTFOLIO_LAB_ENABLE_ML=1 pytest tests/ -m heavy

All tests including ML:
  PORTFOLIO_LAB_ENABLE_ML=1 pytest tests/
"""

import os
import sys
import builtins
import pytest

# ── Layer 1: Env var gate (set before ANY import) ─────────────────────────
if "PORTFOLIO_LAB_ENABLE_ML" not in os.environ:
    os.environ["PORTFOLIO_LAB_ENABLE_ML"] = "0"

# ── Layer 2: Import hook — blocks torch/sklearn/xgboost/hmmlearn at the
#    interpreter level when ML is disabled. Catches unconditional imports in
#    test files AND src modules, including transitively-loaded dependencies.
#    Installed in builtins so it intercepts ALL import statements.
_ML_BLOCKED = frozenset({"torch", "sklearn", "xgboost", "hmmlearn"})
_original_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Block real ML library imports when PORTFOLIO_LAB_ENABLE_ML=0.

    sys.modules is checked by CPython BEFORE __import__ is called, so stub
    entries registered by base_agent.py (torch, torch.nn) short-circuit this
    hook — only real imports that would actually load the package reach here.
    """
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

    # ── Layer 3: Post-collection leak check ─────────────────────────────
    # Detects real ML libraries that evaded the import hook (e.g. installed
    # via sys.modules manipulation without using __import__).
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

    # Skip heavy tests unless both env var AND CLI flag are set
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
