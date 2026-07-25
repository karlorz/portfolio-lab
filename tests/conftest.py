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
  4. ulimit -v (Makefile) — OS kernel enforces 6GB virtual memory cap
  5. PUBLIC_DATA_DIR isolation — dual-writes never hit live WWW / repo public

Default (safe, ML-disabled lane; exact count from pytest output):
  make test
  pytest tests/

Include ML tests:
  PORTFOLIO_LAB_ENABLE_ML=1 pytest tests/ --include-heavy

Include extracted pure ML-adjacent kernel tests without heavy ML imports:
  PORTFOLIO_LAB_ENABLE_ML=0 pytest tests/ --include-ml-extract

All tests including ML:
  PORTFOLIO_LAB_ENABLE_ML=1 pytest tests/

Allow live operator PUBLIC dual-writes (dangerous on lab hosts):
  PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC=1 pytest …
  or mark individual tests with @pytest.mark.allow_live_public_data
"""

import os
import sys
import builtins
import shutil
import sqlite3
import tempfile
from pathlib import Path
import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Layer 5 (early): PUBLIC_DATA_DIR isolation BEFORE any src.* import
# ═══════════════════════════════════════════════════════════════════════════
# Dual-write producers bind PUBLIC_DATA_DIR at import time. On lab hosts
# resolve_runtime_public_data_dir prefers /var/www/portfolio-lab/data when
# present — full pytest then wipes operator SSOT (investigate c307–c308).
# Pin env to a process-private temp dir unless the operator opts into live.
#
# Makefile / run-tests-safe also export PUBLIC_DATA_DIR; this bootstrap
# covers bare `pytest` / `uv run pytest` invocations.

_ISOLATED_PUBLIC_DATA_DIR: Path | None = None
_ISOLATED_PUBLIC_DATA_ROOT: Path | None = None
_ISOLATED_MARKET_DB: Path | None = None
_ISOLATED_MARKET_DB_ROOT: Path | None = None


def _seed_isolated_public_fixtures(public: Path) -> None:
    """Copy minimal price/public fixtures into hermetic PUBLIC_DATA_DIR.

    Many integration/backtest tests resolve ``PRICES_JSON`` via PUBLIC_DATA_DIR
    (H16 isolation). An empty mktemp tree yields ~100 FileNotFoundError fails
    for prices.json even when market.db is available. Seed from repo fixtures
    when present; never touch live WWW.
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "public" / "data" / "prices.json",
        repo_root / "data" / "prices.json",
        repo_root / "tests" / "fixtures" / "prices.json",
    ]
    dest = public / "prices.json"
    if dest.exists():
        return
    for src in candidates:
        if src.is_file() and src.stat().st_size > 0:
            try:
                shutil.copy2(src, dest)
            except OSError:
                continue
            break


def _bootstrap_public_data_dir_isolation() -> Path | None:
    """Ensure PUBLIC_DATA_DIR points at a hermetic temp tree for the session."""
    global _ISOLATED_PUBLIC_DATA_DIR, _ISOLATED_PUBLIC_DATA_ROOT
    if os.environ.get("PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC", "0") == "1":
        return None
    existing = os.environ.get("PUBLIC_DATA_DIR", "").strip()
    if existing:
        # Caller (Makefile) already isolated — remember path for rebind fixtures
        _ISOLATED_PUBLIC_DATA_DIR = Path(existing).expanduser()
        _ISOLATED_PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        _seed_isolated_public_fixtures(_ISOLATED_PUBLIC_DATA_DIR)
        return _ISOLATED_PUBLIC_DATA_DIR
    base = Path(tempfile.mkdtemp(prefix="plab-pytest-public-"))
    _ISOLATED_PUBLIC_DATA_ROOT = base
    public = base / "data"
    public.mkdir(parents=True, exist_ok=True)
    os.environ["PUBLIC_DATA_DIR"] = str(public)
    # Prevent resolve_* fallbacks from preferring live WWW if env is cleared
    os.environ.setdefault("PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA", "1")
    _seed_isolated_public_fixtures(public)
    _ISOLATED_PUBLIC_DATA_DIR = public
    return public


_bootstrap_public_data_dir_isolation()


def _bootstrap_market_db_isolation() -> Path | None:
    """Redirect default MARKET_DB reads/writes to a session-private backup.

    Ensemble tests exercise real collection paths that log health predictions.
    Without an import-time override, those synthetic rows land in the operator
    market.db and later receive real SPY labels. Seed the private database with
    the live schema/data so read-oriented tests retain realistic fixtures while
    every mutation remains disposable.
    """
    global _ISOLATED_MARKET_DB, _ISOLATED_MARKET_DB_ROOT
    if os.environ.get("PORTFOLIO_LAB_ALLOW_LIVE_MARKET_DB", "0") == "1":
        return None

    existing = os.environ.get("PORTFOLIO_LAB_MARKET_DB", "").strip()
    if existing:
        _ISOLATED_MARKET_DB = Path(existing).expanduser()
        _ISOLATED_MARKET_DB.parent.mkdir(parents=True, exist_ok=True)
        return _ISOLATED_MARKET_DB

    repo_root = Path(__file__).resolve().parent.parent
    source = repo_root / "data" / "market.db"
    root = Path(tempfile.mkdtemp(prefix="plab-pytest-market-db-"))
    isolated = root / "market.db"
    if source.is_file():
        # sqlite backup is consistent even while tasker has the live DB open.
        with sqlite3.connect(source) as source_conn, sqlite3.connect(isolated) as dest_conn:
            source_conn.backup(dest_conn)

    os.environ["PORTFOLIO_LAB_MARKET_DB"] = str(isolated)
    _ISOLATED_MARKET_DB_ROOT = root
    _ISOLATED_MARKET_DB = isolated
    return isolated


_bootstrap_market_db_isolation()


def pytest_sessionfinish(session, exitstatus) -> None:
    """Remove only isolation trees created by this pytest process."""
    del session, exitstatus
    global _ISOLATED_PUBLIC_DATA_ROOT, _ISOLATED_MARKET_DB_ROOT
    if _ISOLATED_PUBLIC_DATA_ROOT is not None:
        shutil.rmtree(_ISOLATED_PUBLIC_DATA_ROOT, ignore_errors=True)
    _ISOLATED_PUBLIC_DATA_ROOT = None
    if _ISOLATED_MARKET_DB_ROOT is not None:
        shutil.rmtree(_ISOLATED_MARKET_DB_ROOT, ignore_errors=True)
    _ISOLATED_MARKET_DB_ROOT = None


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
    make-test 6GB virtual-memory cap (full-file hashing after a long suite can
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


# Production journal writers (cron/ops). Size growth whose new tail lines look
# like these is external concurrent append, not a pytest leak (Batch BM).
_PRODUCTION_PERF_MARKERS = (
    b'"source": "capture_daily_pnl"',
    b'"source":"capture_daily_pnl"',
    b'"source": "evaluator"',
    b'"source":"evaluator"',
    b'"source": "paper_trading"',
    b'"source":"paper_trading"',
)
_TEST_LEAK_MARKERS = (
    b"pytest",
    b"PYTEST",
    b"tmp_path",
    b"/tmp/plab-pytest",
    b"conftest",
    b"unittest",
)


def _classify_performance_jsonl_growth(
    path: Path,
    before_size: int,
    after_size: int,
) -> str:
    """Classify live journal size change.

    Returns:
      ``external_append`` — growth looks like cron/ops production rows
      ``test_leak`` — growth looks like suite pollution or truncate/wipe
      ``unknown_growth`` — growth without clear markers (fail closed)
    """
    if after_size < before_size:
        return "test_leak"  # truncate / rewrite-smaller is never external-safe
    if after_size == before_size:
        return "external_append"  # same-size rewrite already allowed by size-only
    # Read only the new tail bytes
    try:
        with path.open("rb") as handle:
            handle.seek(before_size)
            new_bytes = handle.read(after_size - before_size)
    except OSError:
        return "unknown_growth"
    if not new_bytes.strip():
        return "external_append"
    if any(m in new_bytes for m in _TEST_LEAK_MARKERS):
        return "test_leak"
    # JSONL lines: require every non-empty new line to look production-like
    lines = [ln.strip() for ln in new_bytes.splitlines() if ln.strip()]
    if not lines:
        return "external_append"
    prod_hits = 0
    for ln in lines:
        if any(m in ln for m in _PRODUCTION_PERF_MARKERS):
            prod_hits += 1
            continue
        # Paper journal rows often omit source but carry mode=paper + total_value
        if b'"mode": "paper"' in ln or b'"mode":"paper"' in ln:
            if b"total_value" in ln or b"daily_return" in ln:
                prod_hits += 1
                continue
        # Unclassified line → not a clean external-only append
        return "unknown_growth"
    if prod_hits == len(lines):
        return "external_append"
    return "unknown_growth"


@pytest.fixture(scope="session", autouse=True)
def _guard_live_performance_jsonl():
    """Fail the session if live data/performance.jsonl is polluted by tests.

    Host tasker/cron may rewrite the same-size journal or *append* real
    capture_daily_pnl rows while a long suite runs. Same-size rewrites are
    ignored; growth is classified — production-marker tails warn+pass;
    truncate or test-looking growth fails (Batch BM).
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
    if before_size == after_size:
        return
    # File created during suite
    if before_size is None and after_size is not None:
        kind = _classify_performance_jsonl_growth(
            _LIVE_PERFORMANCE_JSONL, 0, after_size
        )
    elif before_size is not None and after_size is None:
        kind = "test_leak"
    else:
        kind = _classify_performance_jsonl_growth(
            _LIVE_PERFORMANCE_JSONL, int(before_size), int(after_size)
        )
    if kind == "external_append":
        import warnings

        warnings.warn(
            f"Live {_LIVE_PERFORMANCE_JSONL} size changed during pytest "
            f"(before_size={before_size}, after_size={after_size}) but new tail "
            "matches production journal markers — treating as concurrent cron "
            "append (Batch BM). Isolate DATA_DIR if tests should never see live.",
            UserWarning,
            stacklevel=1,
        )
        return
    pytest.fail(
        f"Live {_LIVE_PERFORMANCE_JSONL} size changed during pytest "
        f"(before={before}, after={after}, class={kind}). Isolate "
        "src.strategy.evaluator.DATA_DIR (autouse) or mark tests "
        "allow_live_data only when intentional. Set "
        "PORTFOLIO_LAB_ALLOW_LIVE_PERF_WRITES=1 to bypass."
    )


# H18: live operator PUBLIC SSOT pollution guard (investigate c307–c308)
# Deny-listed fixture SHAs that must never appear in live WWW after a suite.
_FIXTURE_SHA_DENYLIST = (
    "abad1dea00",
    "deadbeef12",
    "deadbeefcafe",
    "unifysha1234",
    "rebalsha12345",
    "overlaysha123",
    "samepathsha12",
    "incsha123456",
)
_LIVE_PUBLIC_WATCHLIST = (
    "overlay_dashboard.json",
    "health_ops.json",
    "health.json",
    "adaptive_sizing.json",
    "signals.json",
    "rebalance_health.json",
    "unified_dashboard.json",
    "incidents.json",
    "garch_cvar.json",
)


def _resolve_live_public_root() -> Path | None:
    """Return live WWW public tree if it exists and is distinct from isolation dir."""
    live = Path(
        os.environ.get(
            "PORTFOLIO_LAB_LIVE_PUBLIC_DATA_DIR",
            "/var/www/portfolio-lab/data",
        )
    ).expanduser()
    try:
        if not live.is_dir():
            return None
    except OSError:
        return None
    isolated = os.environ.get("PUBLIC_DATA_DIR", "").strip()
    if isolated:
        try:
            if Path(isolated).resolve() == live.resolve():
                return None  # intentionally testing live tree
        except OSError:
            pass
    return live


@pytest.fixture(scope="session", autouse=True)
def _guard_live_public_ssot_pollution():
    """Fail session if live WWW operator JSON gains fixture SHAs after pytest.

    Complements H16 isolation. Size-only guards false-fail under concurrent
    cron regen; fixture-SHA denylist catches dual-write pollution from tests
    (investigate: abad1dea00 overlay leak). Opt out:
      PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC=1
      PORTFOLIO_LAB_ALLOW_LIVE_WWW_MUTATION=1
    Strict size inventory (optional, noisy with cron):
      PORTFOLIO_LAB_STRICT_LIVE_PUBLIC_GUARD=1
    """
    if os.environ.get("PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC", "0") == "1":
        yield
        return
    if os.environ.get("PORTFOLIO_LAB_ALLOW_LIVE_WWW_MUTATION", "0") == "1":
        yield
        return
    live = _resolve_live_public_root()
    if live is None:
        yield
        return

    strict = os.environ.get("PORTFOLIO_LAB_STRICT_LIVE_PUBLIC_GUARD", "0") == "1"
    before_sizes: dict[str, int | None] = {}
    if strict:
        for name in _LIVE_PUBLIC_WATCHLIST:
            p = live / name
            try:
                before_sizes[name] = p.stat().st_size if p.is_file() else None
            except OSError:
                before_sizes[name] = None

    yield

    # Fixture SHA denylist — always when isolation is active
    polluted: list[str] = []
    for name in _LIVE_PUBLIC_WATCHLIST:
        p = live / name
        try:
            if not p.is_file() or p.stat().st_size > 2_000_000:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for token in _FIXTURE_SHA_DENYLIST:
            if token in text:
                polluted.append(f"{name} contains fixture token {token!r}")
                break
    if polluted:
        pytest.fail(
            "Live operator PUBLIC SSOT polluted during pytest (H18):\n  - "
            + "\n  - ".join(polluted)
            + f"\nLive root: {live}\n"
            "Ensure PUBLIC_DATA_DIR isolation (H16) and dual-write monkeypatches. "
            "Set PORTFOLIO_LAB_ALLOW_LIVE_WWW_MUTATION=1 to bypass."
        )

    if strict:
        size_changes: list[str] = []
        for name, before in before_sizes.items():
            p = live / name
            try:
                after = p.stat().st_size if p.is_file() else None
            except OSError:
                after = None
            if before != after:
                size_changes.append(f"{name}: {before} → {after}")
        if size_changes:
            pytest.fail(
                "Live PUBLIC watchlist size changed during pytest "
                f"(PORTFOLIO_LAB_STRICT_LIVE_PUBLIC_GUARD=1):\n  - "
                + "\n  - ".join(size_changes)
                + f"\nLive root: {live}"
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


@pytest.fixture(autouse=True)
def _isolate_generator_data_dir(request, tmp_path, monkeypatch):
    """Point generator DATA_DIR at tmp so host state files stay hermetic.

    Retro (P1): missing/ignored generator fixture inventory caused host-only
    noise. ``_isolate_live_ensemble_and_ic_health`` (in test_generator.py) stubs
    IC/vote compute but never rebinds ``src.dashboard.generator.DATA_DIR``, so
    tests that forgot the explicit ``patch("...DATA_DIR", tmp_path)`` read live
    host state (``data/.health_report.json``, ``performance.jsonl``,
    ``ensemble_weights.json``, etc).

    This mirrors ``_isolate_evaluator_data_dir``. Opt out with
    @pytest.mark.allow_live_data when a test must touch live paths.
    """
    if request.node.get_closest_marker("allow_live_data"):
        yield
        return
    if os.environ.get("PORTFOLIO_LAB_ALLOW_LIVE_PERF_WRITES", "0") == "1":
        yield
        return
    try:
        import src.dashboard.generator as gen_mod
    except Exception:
        yield
        return
    monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path, raising=False)
    yield


@pytest.fixture(scope="session")
def _incidents_isolate_root(tmp_path_factory):
    """One hermetic incidents tree for the whole suite (inode-friendly).

    Historical bug: function-scoped ``tmp_path_factory.mktemp("incidents-isolate")``
    created ~1 directory per test (~7k–15k inodes per ``make test``). That pushed
    ``/tmp/pytest-of-root`` over the hermes pytest-watchdog 50k-entry cleanup
    threshold every run and burned FS/CPU on mkdir churn.
    """
    return tmp_path_factory.mktemp("incidents-isolate")


def _clear_incidents_isolate_root(root: Path) -> None:
    """Drop leftover incident/kill files so tests do not inherit prior state."""
    try:
        children = list(root.iterdir())
    except OSError:
        return
    for child in children:
        try:
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


@pytest.fixture(autouse=True)
def _isolate_incident_manager_singleton(request, _incidents_isolate_root, monkeypatch):
    """Batch JG TI1: rebind default IncidentManager away from live DATA_DIR.

    Complements refuse-on-write guards in incident_manager. Opt out:
      @pytest.mark.allow_live_incidents
      PORTFOLIO_LAB_ALLOW_LIVE_INCIDENTS=1

    Uses a session-scoped temp root (cleared per test) instead of mktemp-per-test
    so full-suite runs stay under the pytest-watchdog inode threshold.
    """
    if request.node.get_closest_marker("allow_live_incidents"):
        yield
        return
    if os.environ.get("PORTFOLIO_LAB_ALLOW_LIVE_INCIDENTS", "0") == "1":
        yield
        return
    root = _incidents_isolate_root
    _clear_incidents_isolate_root(root)
    try:
        import src.monitor.alerting as alerting
    except Exception:
        yield
        return
    # Drop any process-local manager so get_incident_manager() rebuilds hermetic.
    monkeypatch.setattr(alerting, "_incident_manager", None, raising=False)
    try:
        from src.monitor.incident_manager import IncidentManager

        hermetic = IncidentManager(
            log_path=root / "incidents.jsonl",
            summary_path=root / "incidents.json",
            kill_switch_path=root / "kill_switch.json",
            escalation_enabled=False,
        )
        monkeypatch.setattr(alerting, "_incident_manager", hermetic, raising=False)
    except Exception:
        pass
    yield
    # Ensure next test does not inherit hermetic manager identity
    try:
        import src.monitor.alerting as alerting

        monkeypatch.setattr(alerting, "_incident_manager", None, raising=False)
    except Exception:
        pass


# Dual-write modules that import PUBLIC_DATA_DIR at module level — rebind each
# test so late imports and already-loaded producers never write live WWW/repo.
_PUBLIC_DUAL_WRITE_MODULES = (
    "src.paths",
    "src.dashboard.generator",
    "src.dashboard.overlay_dashboard",
    "src.dashboard.public_data_index",
    "src.dashboard.cron_scheduler_section",
    "src.monitor.health_check",
    "src.monitor.incident_manager",
    "src.monitor.rebalance_health",
    "src.monitor.unified_dashboard",
    "src.monitor.performance_attribution",
    "src.monitor.decision_registry",
    "src.monitor.daily_brief",
    "src.strategy.adaptive_sizing",
    "src.tasker.store",
    "src.research.labs_validation_report",
    "src.research.experiment_scorecard",
    "src.research.experiment_registry",
    "src.research.artifact_retention",
    "src.signals.cross_asset_regime_arb",
)


@pytest.fixture(autouse=True)
def _isolate_public_data_dir_modules(request, monkeypatch):
    """Rebind PUBLIC_DATA_DIR on dual-write modules to the hermetic session dir.

    Complements early env bootstrap (Layer 5). Opt out:
      @pytest.mark.allow_live_public_data
      PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC=1
    """
    if request.node.get_closest_marker("allow_live_public_data"):
        yield
        return
    if os.environ.get("PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC", "0") == "1":
        yield
        return
    target = _ISOLATED_PUBLIC_DATA_DIR
    if target is None:
        env_p = os.environ.get("PUBLIC_DATA_DIR", "").strip()
        if not env_p:
            yield
            return
        target = Path(env_p)
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PUBLIC_DATA_DIR", str(target))
    # Rebind module-level aliases already imported
    for mod_name in _PUBLIC_DUAL_WRITE_MODULES:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        if hasattr(mod, "PUBLIC_DATA_DIR"):
            monkeypatch.setattr(mod, "PUBLIC_DATA_DIR", target, raising=False)
        # paths.py also exposes derived JSON paths
        if mod_name == "src.paths":
            monkeypatch.setattr(mod, "PRICES_JSON", target / "prices.json", raising=False)
            monkeypatch.setattr(mod, "SIGNALS_JSON", target / "signals.json", raising=False)
            monkeypatch.setattr(
                mod, "HISTORICAL_JSON", target / "historical.json", raising=False
            )
            monkeypatch.setattr(mod, "YIELDS_JSON", target / "yields.json", raising=False)
            monkeypatch.setattr(
                mod,
                "PUBLIC_TASKER_STATUS_JSON",
                target / "tasker_status.json",
                raising=False,
            )
    yield


# ═══════════════════════════════════════════════════════════════════════════
# Mid-suite memory hygiene (S17) — insurance after 3GB→6GB MemoryError cascade
# ═══════════════════════════════════════════════════════════════════════════
# Large single-process suites retain fixture/report objects; periodic gc helps
# under ulimit -v. Default: light GC every N tests. Opt-in CHECK_LEAKS=1 enables
# tracemalloc per-test growth logging (advisory, never fails tests by default).
#
# Research: PythonSpeed leak fixture + yield teardown + periodic gc.collect;
# process isolation (pytest-forked) is heavier and deferred to suite segmentation.

_MID_SUITE_GC_EVERY = int(os.environ.get("PORTFOLIO_LAB_MID_SUITE_GC_EVERY", "200"))
_CHECK_LEAKS = os.environ.get("CHECK_LEAKS", "0") == "1"
_mid_suite_gc_counter = {"n": 0}


@pytest.fixture(autouse=True)
def _mid_suite_gc_hygiene():
    """Periodic garbage collection to reduce late-suite VSZ growth under ulimit.

    Frequency: PORTFOLIO_LAB_MID_SUITE_GC_EVERY (default 200). Set to 0 to disable.
    When CHECK_LEAKS=1, also run a full gen-2 collect after each test (slower).
    """
    yield
    if _CHECK_LEAKS:
        import gc

        gc.collect(2)
        return
    if _MID_SUITE_GC_EVERY <= 0:
        return
    _mid_suite_gc_counter["n"] += 1
    if _mid_suite_gc_counter["n"] % _MID_SUITE_GC_EVERY == 0:
        import gc

        gc.collect()


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
        "allow_live_incidents: allow IncidentManager writes to live DATA_DIR "
        "incident/kill SSOT under pytest (Batch JG TI1 opt-out)",
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
    config.addinivalue_line(
        "markers",
        "allow_live_public_data: allow test to use live PUBLIC_DATA_DIR "
        "(disables dual-write isolation; dangerous on lab hosts)",
    )
    config.addinivalue_line(
        "markers",
        "unit: fast hermetic unit tests (optional; path-based make test-unit is S18 primary)",
    )
    config.addinivalue_line(
        "markers",
        "integration: multi-module / host-touching flows (see Makefile TEST_INTEGRATION_FILES)",
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
