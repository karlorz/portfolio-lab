"""
pytest configuration: ML feature gating.

ML features (torch/xgboost/sklearn/hmmlearn) are DISABLED by default via
a 4-layer defense to prevent OOM/CPU stalls. torch (~63MB) + sklearn
(~78MB) + hmmlearn (~23MB) accumulate in single-process test runs, causing
SIGKILL partway through safe-suite runs on low-resource hosts.

Layered defense (each layer independently prevents host CPU exhaust):
  0. collect_ignore — pytest never opens known heavy test files (0 CPU)
  1. Env var gate — PORTFOLIO_LAB_ENABLE_ML=0 set before any import
  2. builtins.__import__ hook — blocks ML imports at interpreter level
  3. Post-collection leak check — warns if real ML libs evaded all guards
  4. ulimit -v (Makefile) — OS kernel enforces 3GB virtual memory cap

Default (safe, ML-disabled lane; exact count from pytest output):
  make test
  pytest tests/

Include ML tests:
  PORTFOLIO_LAB_ENABLE_ML=1 pytest tests/ --include-heavy

Include extracted pure ML-adjacent kernel tests without heavy ML imports:
  PORTFOLIO_LAB_ENABLE_ML=0 pytest tests/ --include-ml-extract

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
    "test_black_litterman_mapper.py",
    "test_execution_agent.py",
    "test_marl_trainer.py",
    "test_risk_agent_hmm.py",
    "test_stacking_trainer.py",
    "test_transformer_regime.py",
    "test_base_agent.py",
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

# sklearn submodules that are data utilities (not ML models) — safe to import.
# When any of these is requested, all internal sklearn imports triggered by
# sklearn's __init__.py are allowed through (they're just plumbing, not models).
_SKLEARN_SAFE_PREFIXES = (
    "sklearn.model_selection",  # TimeSeriesSplit, KFold, etc. (data splitting only)
    "sklearn.utils",            # Validation, math helpers
    "sklearn.covariance",       # Ledoit-Wolf shrinkage for pypfopt CovarianceShrinkage
)
_original_import = builtins.__import__


# Track whether a safe sklearn import has been initiated — if so, allow
# internal sklearn plumbing imports (sklearn._config, sklearn.utils._typedef, etc.)
# These are triggered by sklearn's __init__.py and are not ML models.
_sklearn_safe_active = False


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Block real ML library imports when PORTFOLIO_LAB_ENABLE_ML=0."""
    global _sklearn_safe_active
    ml_enabled = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"
    if not ml_enabled:
        top_level = name.split(".")[0]
        if top_level in _ML_BLOCKED:
            # Allow safe sklearn submodules and their internal plumbing
            if top_level == "sklearn":
                if any(name.startswith(prefix) for prefix in _SKLEARN_SAFE_PREFIXES):
                    _sklearn_safe_active = True
                # Also check fromlist items: `from sklearn.covariance import X`
                # triggers __import__('sklearn', ..., fromlist=('covariance',))
                if fromlist and not _sklearn_safe_active:
                    for item in fromlist:
                        qualified = f"{name}.{item}"
                        if any(qualified.startswith(prefix) for prefix in _SKLEARN_SAFE_PREFIXES):
                            _sklearn_safe_active = True
                            break
                if _sklearn_safe_active:
                    pass  # Allow sklearn internals while safe import is active
                else:
                    raise ImportError(
                        f"ML library '{name}' blocked: PORTFOLIO_LAB_ENABLE_ML=0. "
                        f"Set PORTFOLIO_LAB_ENABLE_ML=1 to enable ML features."
                    )
            else:
                raise ImportError(
                    f"ML library '{name}' blocked: PORTFOLIO_LAB_ENABLE_ML=0. "
                    f"Set PORTFOLIO_LAB_ENABLE_ML=1 to enable ML features."
                )
    return _original_import(name, globals, locals, fromlist, level)


builtins.__import__ = _guarded_import


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures — test pollution guards
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _reset_sklearn_safe_flag():
    """Reset _sklearn_safe_active between tests to prevent leak-through."""
    global _sklearn_safe_active
    yield
    _sklearn_safe_active = False


@pytest.fixture(autouse=True)
def _clear_price_cache():
    """Replace TTL price caches with fresh instances between tests.

    Uses instance replacement rather than pop/clear because some tests
    replace the module-level _PRICE_CACHE / _DF_CACHE attributes directly
    (e.g., test_price_cache.py swaps them for short-TTL instances). When
    those tests restore the "old" cache in a finally block, they may
    resurrect a cache that was populated with mock data from an earlier
    test. Fresh instances guarantee zero cross-test contamination.
    """
    from cachetools import TTLCache
    import src.data.price_cache as pc_mod

    # Replace with fresh empty caches
    pc_mod._PRICE_CACHE = TTLCache(maxsize=1, ttl=pc_mod._PRICE_CACHE_TTL)
    pc_mod._DF_CACHE = TTLCache(maxsize=4, ttl=pc_mod._PRICE_CACHE_TTL)

    # Clear vpin_bvc bar cache
    from src.signals.vpin_bvc import _BARS_CACHE
    _BARS_CACHE.clear()

    yield

    # Teardown: clear the vpin cache again
    _BARS_CACHE.clear()
    # Note: do NOT restore the old caches — that would resurrect stale data
    # from before this fixture ran. Each test gets a clean slate.


# ═══════════════════════════════════════════════════════════════════════════
# Live performance.jsonl hermeticity (paper-trading host safety)
# ═══════════════════════════════════════════════════════════════════════════
# Evaluator PERFORMANCE_LOG / ORDERS_LOG resolve from evaluator.DATA_DIR at use
# time. Tests that forget to patch DATA_DIR would still append phantom cash
# rows to the live paper journal on this host. Isolate by default + guard hash.

from pathlib import Path
import hashlib

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LIVE_PERFORMANCE_JSONL = _PROJECT_ROOT / "data" / "performance.jsonl"


def _fingerprint_file(path: Path):
    """Return (sha256, size, mtime_ns) or None if missing.

    Hash only the first + last 64 KiB so session teardown stays cheap under the
    make-test 3GB virtual-memory cap (full-file hashing after a long suite can
    MemoryError when capture/logging has already exhausted the budget).
    """
    if not path.exists():
        return None
    stat = path.stat()
    size = stat.st_size
    digest = hashlib.sha256()
    sample = 64 * 1024
    try:
        with path.open("rb") as handle:
            head = handle.read(sample)
            digest.update(head)
            if size > sample * 2:
                handle.seek(max(0, size - sample))
                digest.update(handle.read(sample))
            elif size > sample:
                digest.update(handle.read())
    except MemoryError:
        # Fall back to metadata-only identity under extreme memory pressure.
        return ("", size, stat.st_mtime_ns)
    return (digest.hexdigest(), size, stat.st_mtime_ns)


@pytest.fixture(scope="session", autouse=True)
def _guard_live_performance_jsonl():
    """Fail the session if live data/performance.jsonl grows/shrinks during tests.

    Host tasker/cron may rewrite the same-size journal while pytest runs on a
    live lab host; those external rewrites must not fail the suite. Tests that
    leak appends change file size — that is what we gate on.
    """
    if os.environ.get("PORTFOLIO_LAB_ALLOW_LIVE_PERF_WRITES", "0") == "1":
        yield
        return
    before = _fingerprint_file(_LIVE_PERFORMANCE_JSONL)
    yield
    after = _fingerprint_file(_LIVE_PERFORMANCE_JSONL)
    if before is None and after is None:
        return
    before_size = None if before is None else before[1]
    after_size = None if after is None else after[1]
    if before_size != after_size:
        pytest.fail(
            f"Live {_LIVE_PERFORMANCE_JSONL} size changed during pytest "
            f"(before={before}, after={after}). Isolate "
            "src.strategy.evaluator.DATA_DIR (autouse) or mark tests "
            "allow_live_data only when intentional. Set "
            "PORTFOLIO_LAB_ALLOW_LIVE_PERF_WRITES=1 to bypass."
        )


@pytest.fixture(autouse=True)
def _isolate_evaluator_data_dir(request, tmp_path, monkeypatch):
    """Point evaluator DATA_DIR at tmp so PERFORMANCE_LOG/ORDERS_LOG stay hermetic.

    Opt out with @pytest.mark.allow_live_data when a test must touch live paths.
    """
    if request.node.get_closest_marker("allow_live_data"):
        yield
        return
    if os.environ.get("PORTFOLIO_LAB_ALLOW_LIVE_PERF_WRITES", "0") == "1":
        yield
        return
    try:
        import src.strategy.evaluator as ev
    except Exception:
        yield
        return
    monkeypatch.setattr(ev, "DATA_DIR", tmp_path, raising=False)
    yield


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
    parser.addoption(
        "--include-ml-extract",
        action="store_true",
        default=False,
        help="Run tests marked ml_extract for extracted pure ML-adjacent kernels. "
             "Keeps PORTFOLIO_LAB_ENABLE_ML=0 and does not include heavy tests.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "heavy: tests requiring heavy ML libraries (torch, xgboost, sklearn, hmmlearn)",
    )
    config.addinivalue_line(
        "markers",
        "ml_extract: safe-mode tests for extracted pure ML-adjacent kernels; "
        "must run with PORTFOLIO_LAB_ENABLE_ML=0 and without heavy ML imports",
    )
    config.addinivalue_line(
        "markers",
        "allow_live_data: allow test to use live repo data/ paths "
        "(disables evaluator DATA_DIR isolation)",
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

    # Skip extracted ML-kernel tests unless explicitly requested. These tests
    # are safe-mode only and remain independent from the heavy ML lane.
    if not config.getoption("--include-ml-extract"):
        skip_ml_extract = pytest.mark.skip(
            reason="ml_extract tests skipped (use --include-ml-extract with PORTFOLIO_LAB_ENABLE_ML=0)"
        )
        count = 0
        for item in items:
            if "ml_extract" in item.keywords:
                item.add_marker(skip_ml_extract)
                count += 1
        if count > 0:
            print(
                f"\n[Skipped {count} ml_extract tests. "
                f"Use PORTFOLIO_LAB_ENABLE_ML=0 --include-ml-extract to run.]"
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
