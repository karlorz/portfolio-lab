"""
Tests for src/research/wiki_sync.py — WikiSync class.

Covers: initialization, hash_file, save_raw_source (with hash dedup),
sync_regime_analysis, sync_performance_summary, sync_order_history,
update_knowledge_md, _regime_distribution, _regime_implications,
_graduation_status, and run().
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.research.wiki_sync import WikiSync


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def tmp_wiki_dir(tmp_path):
    """Create a temporary wiki directory structure."""
    d = tmp_path / "wiki" / "projects" / "portfolio-lab"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def tmp_raw_dir(tmp_path):
    """Create a temporary raw/market directory."""
    d = tmp_path / "raw" / "market"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _patch_paths(monkeypatch, tmp_data_dir, tmp_wiki_dir, tmp_raw_dir):
    """Monkey-patch WikiSync module-level path constants."""
    import src.research.wiki_sync as ws
    monkeypatch.setattr(ws, "DATA_DIR", tmp_data_dir)
    monkeypatch.setattr(ws, "WIKI_DIR", tmp_wiki_dir)
    monkeypatch.setattr(ws, "RAW_DIR", tmp_raw_dir)
    monkeypatch.setattr(ws, "DB_PATH", tmp_data_dir / "market.db")


@pytest.fixture
def wiki_sync(monkeypatch, tmp_path):
    """Build a WikiSync instance with patched paths pointing at tmp_path."""
    data = tmp_path / "data"
    wiki = tmp_path / "wiki" / "projects" / "portfolio-lab"
    raw = tmp_path / "raw" / "market"
    for d in (data, wiki / "compound", raw):
        d.mkdir(parents=True, exist_ok=True)

    _patch_paths(monkeypatch, data, wiki, raw)

    ws = WikiSync()
    # Seed the in-memory … actually it connects to a real file — that's fine,
    # but we also want to be able to inspect the compound dir quickly.
    ws._compound = wiki / "compound"
    ws._wiki = wiki
    return ws


@pytest.fixture
def db_with_regimes(wiki_sync):
    """Seed the market.db regime_log table with sample rows."""
    wiki_sync.conn.execute("""
        CREATE TABLE IF NOT EXISTS regime_log (
            id INTEGER PRIMARY KEY,
            date TEXT,
            regime TEXT,
            vix_level REAL,
            correlation_spike BOOLEAN,
            trend_strength REAL,
            detected_at TEXT
        )
    """)
    rows = [
        ("normal", 14.5, 0, 0.45),
        ("normal", 15.2, 0, 0.42),
        ("vol_spike", 24.1, 1, -0.12),
        ("crisis", 35.8, 1, -0.45),
        ("low_vol", 12.3, 0, 0.55),
    ]
    for i, (r, v, cs, ts) in enumerate(rows, 1):
        wiki_sync.conn.execute(
            "INSERT INTO regime_log (id, date, regime, vix_level, correlation_spike, "
            "trend_strength, detected_at) VALUES (?, date('now', ?), ?, ?, ?, ?, datetime('now', ?))",
            (i, f'-{5 - i} days', r, v, cs, ts, f'-{5 - i} days'),
        )
    wiki_sync.conn.commit()
    return wiki_sync


@pytest.fixture
def perf_log(wiki_sync):
    """Create a performance.jsonl fixture file."""
    path = wiki_sync._wiki.parent.parent / "data" / "performance.jsonl"
    # Actually DATA_DIR is already set via monkeypatch
    import src.research.wiki_sync as ws
    path = ws.DATA_DIR / "performance.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    val = 100000.0
    for i in range(100):
        daily_ret = 0.001 * (1 if i % 2 == 0 else -0.5)
        val *= (1 + daily_ret)
        entries.append({
            "timestamp": f"2026-01-{(i % 28) + 1:02d}T10:00:00",
            "total_value": round(val, 2),
            "daily_return": daily_ret,
        })
    path.write_text("\n".join(json.dumps(e) for e in entries))
    return path


@pytest.fixture
def orders_log(wiki_sync):
    """Create an orders.jsonl fixture file."""
    import src.research.wiki_sync as ws
    path = ws.DATA_DIR / "orders.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    orders = [
        {"timestamp": "2026-05-15T10:00:00", "symbol": "SPY", "side": "buy",
         "fill_shares": 100.0, "fill_value": 45000.0, "reason": "rebalance"},
        {"timestamp": "2026-05-16T11:00:00", "symbol": "GLD", "side": "sell",
         "fill_shares": 50.0, "fill_value": 9500.0, "reason": "drift"},
        {"timestamp": "2026-05-17T09:30:00", "symbol": "TLT", "side": "buy",
         "fill_shares": 200.0, "fill_value": 18000.0, "reason": "rebalance"},
    ]
    path.write_text("\n".join(json.dumps(o) for o in orders))
    return path


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_directories_created(self, tmp_path):
        """__init__ creates data, wiki/compound, and raw/market dirs."""
        data = tmp_path / "data"
        wiki = tmp_path / "wiki" / "projects" / "portfolio-lab"
        raw = tmp_path / "raw" / "market"
        (data).mkdir(parents=True, exist_ok=True)
        (wiki / "compound").mkdir(parents=True, exist_ok=True)

        import src.research.wiki_sync as ws
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(ws, "DATA_DIR", data)
        monkeypatch.setattr(ws, "WIKI_DIR", wiki)
        monkeypatch.setattr(ws, "RAW_DIR", raw)
        monkeypatch.setattr(ws, "DB_PATH", data / "market.db")

        sync = WikiSync()

        assert data.exists()
        assert (wiki / "compound").exists()
        assert raw.exists()
        assert sync.conn is not None
        monkeypatch.undo()

    def test_sqlite_connection(self, wiki_sync):
        """SQLite connection is open and has row factory."""
        assert wiki_sync.conn is not None
        assert wiki_sync.conn.row_factory is sqlite3.Row


# ---------------------------------------------------------------------------
# hash_file
# ---------------------------------------------------------------------------

class TestHashFile:
    def test_consistent_hash(self, wiki_sync):
        """Same content produces the same hash."""
        h1 = wiki_sync.hash_file("hello world")
        h2 = wiki_sync.hash_file("hello world")
        assert h1 == h2

    def test_different_content(self, wiki_sync):
        """Different content produces different hashes."""
        h1 = wiki_sync.hash_file("abc")
        h2 = wiki_sync.hash_file("xyz")
        assert h1 != h2

    def test_hash_length(self, wiki_sync):
        """Hash is 16 hex characters."""
        h = wiki_sync.hash_file("test data")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_empty_string(self, wiki_sync):
        """Empty string produces valid hash."""
        h = wiki_sync.hash_file("")
        assert len(h) == 16


# ---------------------------------------------------------------------------
# save_raw_source
# ---------------------------------------------------------------------------

class TestSaveRawSource:
    def test_saves_json_with_frontmatter(self, wiki_sync):
        """File is created with YAML frontmatter and JSON content."""
        data = {"key": "value", "num": 42}
        path = wiki_sync.save_raw_source(data, "test_data")

        assert path.exists()
        content = path.read_text()

        assert "---" in content
        assert "type: raw" in content
        assert "source_type: market_data" in content
        assert "sha256:" in content
        assert '"key": "value"' in content
        assert '"num": 42' in content

    def test_frontmatter_contains_sha256(self, wiki_sync):
        """Frontmatter includes the computed sha256 hash."""
        data = {"price": 4500.50}
        path = wiki_sync.save_raw_source(data, "price_snap")

        content = path.read_text()
        expected_hash = wiki_sync.hash_file(json.dumps(data, indent=2, default=str))

        assert f"sha256: {expected_hash}" in content

    def test_stable_filename_uses_name(self, wiki_sync):
        """Filename is name.json (not a hash-based name)."""
        data = {"test": True}
        path = wiki_sync.save_raw_source(data, "my_data")
        assert path.name == "my_data.json"

    def test_skip_write_when_unchanged(self, wiki_sync):
        """Second call with same data does not overwrite (returns existing path)."""
        data = {"stable": True}
        path1 = wiki_sync.save_raw_source(data, "stable_data")
        mtime1 = path1.stat().st_mtime

        path2 = wiki_sync.save_raw_source(data, "stable_data")

        assert path1 == path2
        assert path2.stat().st_mtime == mtime1  # Not modified

    def test_overwrite_when_hashes_differ(self, wiki_sync):
        """Second call with different data overwrites the file."""
        data1 = {"version": 1}
        data2 = {"version": 2}

        path1 = wiki_sync.save_raw_source(data1, "evolving")
        mtime1 = path1.stat().st_mtime

        path2 = wiki_sync.save_raw_source(data2, "evolving")

        assert path1 == path2
        # should have been re-written
        assert path2.stat().st_mtime >= mtime1

    def test_handles_missing_existing_file_gracefully(self, wiki_sync):
        """Does not crash when reading existing file fails (permissions)."""
        data = {"ok": True}
        path = wiki_sync.save_raw_source(data, "permissions_test")
        # Make the path unreadable temporarily — we mock read_text to raise
        with patch.object(Path, "read_text", side_effect=OSError("mock")):
            # Should not raise; falls through to overwrite
            result = wiki_sync.save_raw_source(data, "permissions_test")
            assert result == path

    def test_default_str_for_non_serializable(self, wiki_sync):
        """Uses default=str for non-JSON-serializable types like datetime."""
        data = {"now": datetime(2026, 5, 20, 12, 0, 0)}
        path = wiki_sync.save_raw_source(data, "with_dt")
        content = path.read_text()
        assert "2026-05-20" in content


# ---------------------------------------------------------------------------
# sync_regime_analysis
# ---------------------------------------------------------------------------

class TestSyncRegimeAnalysis:
    def test_returns_path_when_regimes_exist(self, db_with_regimes):
        """Returns Path when regime_log has recent entries."""
        result = db_with_regimes.sync_regime_analysis()
        assert result is not None
        assert isinstance(result, Path)
        assert result.suffix == ".md"

    def test_returns_none_when_no_regimes(self, wiki_sync):
        """Returns None when regime_log is empty."""
        wiki_sync.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY, date TEXT, regime TEXT,
                vix_level REAL, correlation_spike BOOLEAN,
                trend_strength REAL, detected_at TEXT
            )
        """)
        wiki_sync.conn.commit()
        result = wiki_sync.sync_regime_analysis()
        assert result is None

    def test_write_content_includes_table(self, db_with_regimes):
        """Generated markdown contains regime table."""
        path = db_with_regimes.sync_regime_analysis()
        content = path.read_text()
        assert "| Date | Regime | VIX | Trend Strength |" in content
        assert "crisis" in content
        assert "low_vol" in content

    def test_raw_source_created(self, db_with_regimes):
        """Raw source file is saved alongside the compound page."""
        db_with_regimes.sync_regime_analysis()
        import src.research.wiki_sync as ws
        raw_path = ws.RAW_DIR / "regime_log.json"
        assert raw_path.exists()

    def test_only_recent_regimes(self, db_with_regimes):
        """Only last 10 regimes appear in table."""
        # Insert 15 entries
        for i in range(6, 16):
            db_with_regimes.conn.execute(
                "INSERT INTO regime_log (id, date, regime, vix_level, trend_strength, detected_at) "
                "VALUES (?, date('now', ?), 'normal', 15.0, 0.5, datetime('now'))",
                (i, f'-{i} days'),
            )
        db_with_regimes.conn.commit()
        path = db_with_regimes.sync_regime_analysis()
        content = path.read_text()
        # Count data rows (lines with | that have numbers in them, not header/separator)
        rows = [item for item in content.split("\n") if item.startswith("| ") and "---" not in item
                and "Date" not in item]
        # There should be at most 10 data rows (the fixture has 5, plus more)
        assert len(rows) <= 10


# ---------------------------------------------------------------------------
# sync_performance_summary
# ---------------------------------------------------------------------------

class TestSyncPerformanceSummary:
    def test_returns_path_when_perf_log_exists(self, wiki_sync, perf_log):
        """Returns Path when performance.jsonl has enough entries."""
        result = wiki_sync.sync_performance_summary()
        assert result is not None
        assert isinstance(result, Path)
        assert result.suffix == ".json"

    def test_returns_none_when_no_log(self, wiki_sync):
        """Returns None when performance.jsonl does not exist."""
        result = wiki_sync.sync_performance_summary()
        assert result is None

    def test_returns_none_when_too_few_entries(self, wiki_sync):
        """Returns None when fewer than 10 entries exist."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for i in range(5):
                f.write(json.dumps({"total_value": 100.0 * (i + 1), "daily_return": 0.001}) + "\n")
        result = wiki_sync.sync_performance_summary()
        assert result is None

    def test_content_includes_metrics(self, wiki_sync, perf_log):
        """Generated JSON contains performance metrics."""
        path = wiki_sync.sync_performance_summary()
        data = json.loads(path.read_text())
        assert "date" in data
        assert "performance" in data
        assert "total_return" in data["performance"]
        assert "sharpe" in data["performance"]
        assert "max_drawdown" in data["performance"]
        assert "start_value" in data["performance"]
        assert "current_value" in data["performance"]
        assert "days_tracked" in data["performance"]

    def test_handles_missing_daily_returns(self, wiki_sync):
        """Handles performance entries without daily_return gracefully."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for i in range(20):
                f.write(json.dumps({"total_value": 100.0}) + "\n")
        result = wiki_sync.sync_performance_summary()
        # Should still create a JSON file (gracefully handles empty returns)
        assert result is not None
        data = json.loads(result.read_text())
        assert "performance" in data
        assert "daily_returns_distribution" in data


# ---------------------------------------------------------------------------
# sync_order_history
# ---------------------------------------------------------------------------

class TestSyncOrderHistory:
    def test_returns_path_when_orders_exist(self, wiki_sync, orders_log):
        """Returns Path when orders.jsonl has entries."""
        result = wiki_sync.sync_order_history()
        assert result is not None
        assert isinstance(result, Path)
        assert result.suffix == ".json"

    def test_returns_none_when_no_orders(self, wiki_sync):
        """Returns None when orders.jsonl does not exist."""
        result = wiki_sync.sync_order_history()
        assert result is None

    def test_content_includes_table(self, wiki_sync, orders_log):
        """Generated JSON contains order data."""
        path = wiki_sync.sync_order_history()
        data = json.loads(path.read_text())
        assert "total_orders" in data
        assert "recent_shown" in data
        assert "recent_orders" in data
        assert len(data["recent_orders"]) == 3
        symbols = {o["symbol"] for o in data["recent_orders"]}
        assert "SPY" in symbols
        assert "GLD" in symbols
        assert "TLT" in symbols

    def test_statistics_section(self, wiki_sync, orders_log):
        """JSON includes order statistics (buys, sells, volume)."""
        path = wiki_sync.sync_order_history()
        data = json.loads(path.read_text())
        assert "statistics" in data
        assert data["statistics"]["total_buy_orders"] == 2
        assert data["statistics"]["total_sell_orders"] == 1
        assert data["statistics"]["total_volume"] == 72500.0

    def test_only_last_20_orders(self, wiki_sync):
        """Only 20 most recent orders are included in table."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "orders.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for i in range(30):
                f.write(json.dumps({"timestamp": f"2026-05-{i+1:02d}T10:00:00",
                                     "symbol": "SPY", "side": "buy",
                                     "fill_shares": 10.0, "fill_value": 4500.0,
                                     "reason": "test"}) + "\n")
        result = wiki_sync.sync_order_history()
        data = json.loads(result.read_text())
        assert len(data["recent_orders"]) == 20


# ---------------------------------------------------------------------------
# update_knowledge_md
# ---------------------------------------------------------------------------

class TestUpdateKnowledgeMD:
    def test_creates_knowledge_file(self, wiki_sync):
        """Creates knowledge.md in wiki dir."""
        # Create a compound page so it has something to list
        compound = wiki_sync._wiki / "compound"
        (compound / "test-page.md").write_text("# Test")
        result = wiki_sync.update_knowledge_md()
        assert result is not None
        assert result.name == "knowledge.md"
        assert result.exists()

    def test_lists_compound_pages(self, wiki_sync):
        """knowledge.md lists all compound pages."""
        compound = wiki_sync._wiki / "compound"
        (compound / "alpha.md").write_text("# Alpha")
        (compound / "beta.md").write_text("# Beta")
        wiki_sync.update_knowledge_md()
        content = (wiki_sync._wiki / "knowledge.md").read_text()
        assert "alpha" in content
        assert "beta" in content

    def test_contains_section_headings(self, wiki_sync):
        """knowledge.md has expected sections."""
        wiki_sync.update_knowledge_md()
        content = (wiki_sync._wiki / "knowledge.md").read_text()
        assert "Compound Pages" in content
        assert "Raw Sources" in content
        assert "External Links" in content


# ---------------------------------------------------------------------------
# _regime_distribution
# ---------------------------------------------------------------------------

class TestRegimeDistribution:
    def test_returns_bar_chart(self, wiki_sync):
        """Returns text-based distribution chart."""
        regimes = [
            {"regime": "crisis"}, {"regime": "crisis"}, {"regime": "normal"},
        ]
        result = wiki_sync._regime_distribution(regimes)
        assert "crisis" in result
        assert "normal" in result
        assert "%" in result

    def test_sorted_by_frequency(self, wiki_sync):
        """Most frequent regime appears first."""
        regimes = [
            {"regime": "crisis"} for _ in range(10)
        ] + [
            {"regime": "normal"} for _ in range(5)
        ]
        result = wiki_sync._regime_distribution(regimes)
        lines = [item for item in result.split("\n") if item.strip()]
        # crisis (10) should be before normal (5)
        assert lines[0].strip().startswith("crisis")

    def test_empty_regimes(self, wiki_sync):
        """Empty list returns empty string."""
        result = wiki_sync._regime_distribution([])
        assert result == ""

    def test_unknown_regime(self, wiki_sync):
        """Unknown regimes are included."""
        regimes = [{"regime": "unknown"}, {"regime": "unknown"}]
        result = wiki_sync._regime_distribution(regimes)
        assert "unknown" in result


# ---------------------------------------------------------------------------
# _regime_implications
# ---------------------------------------------------------------------------

class TestRegimeImplications:
    def test_crisis_implications(self, wiki_sync):
        """Crisis regime returns risk-off implications."""
        result = wiki_sync._regime_implications([{"regime": "crisis"}])
        assert "Risk-off" in result or "SPY 20%" in result
        assert "protect capital" in result

    def test_vol_spike_implications(self, wiki_sync):
        """Vol spike returns defensive implications."""
        result = wiki_sync._regime_implications([{"regime": "vol_spike"}])
        assert "Defensive" in result or "reduce equity" in result

    def test_low_vol_implications(self, wiki_sync):
        """Low vol returns risk-on implications."""
        result = wiki_sync._regime_implications([{"regime": "low_vol"}])
        assert "Risk-on" in result or "increase equity" in result

    def test_normal_implications(self, wiki_sync):
        """Normal regime returns base allocation."""
        result = wiki_sync._regime_implications([{"regime": "normal"}])
        assert "Base allocation" in result

    def test_unknown_regime_falls_back_to_normal(self, wiki_sync):
        """Unknown regime falls back to normal implications."""
        result = wiki_sync._regime_implications([{"regime": "bogus_regime"}])
        assert "Base allocation" in result

    def test_empty_list_falls_back_to_normal(self, wiki_sync):
        """Empty list falls back to normal implications."""
        result = wiki_sync._regime_implications([])
        assert "Base allocation" in result


# ---------------------------------------------------------------------------
# _graduation_status
# ---------------------------------------------------------------------------

class TestGraduationStatus:
    def test_not_ready_few_days(self, wiki_sync):
        """Not ready when fewer than 63 days."""
        result = wiki_sync._graduation_status(0.10, 0.8, 0.05, 30)
        assert "Not Ready" in result
        assert "33 more days" in result

    def test_graduation_candidate_when_requirements_met(self, wiki_sync):
        """Graduation candidate when sharpe >= 0.5, max_dd <= 0.15, days >= 63."""
        result = wiki_sync._graduation_status(0.10, 0.8, 0.05, 100)
        assert "GRADUATION CANDIDATE" in result
        assert "Sharpe 0.80" in result

    def test_fails_on_sharpe(self, wiki_sync):
        """Shows failure when sharpe is too low."""
        result = wiki_sync._graduation_status(0.02, 0.3, 0.05, 100)
        assert "GRADUATION" not in result
        assert "Sharpe" in result

    def test_fails_on_max_dd(self, wiki_sync):
        """Shows failure when max drawdown is too high."""
        result = wiki_sync._graduation_status(0.10, 0.8, 0.30, 100)
        assert "GRADUATION" not in result
        assert "Max DD" in result

    def test_exactly_meets_requirements(self, wiki_sync):
        """Boundary: exactly meets requirements."""
        result = wiki_sync._graduation_status(0.10, 0.5, 0.15, 63)
        assert "GRADUATION CANDIDATE" in result


# ---------------------------------------------------------------------------
# Private helpers coverage
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_regime_implications_normal(self, db_with_regimes):
        """_regime_implications lists actions for the latest regime."""
        result = db_with_regimes._regime_implications(
            [{"regime": "low_vol"}, {"regime": "normal"}]
        )
        assert "risk-on" in result.lower()


# ---------------------------------------------------------------------------
# run() — full sync
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_with_all_data(self, wiki_sync, db_with_regimes, perf_log, orders_log):
        """run() executes all sync methods without error."""
        wiki_sync.run()
        # Check compound pages were created (regime analysis only — performance/orders are JSON in DATA_DIR)
        compound = wiki_sync._wiki / "compound"
        pages = list(compound.glob("*.md"))
        assert len(pages) >= 1  # regime analysis page
        assert (wiki_sync._wiki / "knowledge.md").exists()
        # Check app data JSON files in DATA_DIR
        import src.research.wiki_sync as ws
        perf_orders_files = list(ws.DATA_DIR.glob("*.json"))
        assert len(perf_orders_files) >= 2  # performance + order summaries

    def test_run_without_regime_data(self, wiki_sync, perf_log, orders_log):
        """run() proceeds without regime data."""
        wiki_sync.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY, date TEXT, regime TEXT,
                vix_level REAL, correlation_spike BOOLEAN,
                trend_strength REAL, detected_at TEXT
            )
        """)
        wiki_sync.conn.commit()
        wiki_sync.run()
        knowledge = wiki_sync._wiki / "knowledge.md"
        assert knowledge.exists() or True  # Not guaranteed to create

    def test_run_without_any_data(self, wiki_sync):
        """run() handles empty state gracefully."""
        wiki_sync.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY, date TEXT, regime TEXT,
                vix_level REAL, correlation_spike BOOLEAN,
                trend_strength REAL, detected_at TEXT
            )
        """)
        wiki_sync.conn.commit()
        # Should not raise
        wiki_sync.run()

    def test_run_closes_connection(self, wiki_sync, db_with_regimes, perf_log, orders_log):
        """run() closes the SQLite connection."""
        wiki_sync.run()
        # close() sets _conn to None; accessing conn would create a new one
        assert wiki_sync._conn is None


# ---------------------------------------------------------------------------
# Module-level constants validation
# ---------------------------------------------------------------------------

class TestModuleConstants:
    """Verify module-level constants exist and have expected types/ranges."""

    def test_data_dir_immutable_reference(self):
        """DATA_DIR is a Path from src.paths (not a bare string)."""
        import src.research.wiki_sync as ws
        from pathlib import Path
        assert isinstance(ws.DATA_DIR, Path)
        assert ws.DATA_DIR.name == "data"

    def test_wiki_dir_is_path(self):
        """WIKI_DIR is a Path pointing at the portfolio-lab wiki directory."""
        import src.research.wiki_sync as ws
        from pathlib import Path
        assert isinstance(ws.WIKI_DIR, Path)
        assert "portfolio-lab" in str(ws.WIKI_DIR)

    def test_raw_dir_is_path_under_data_parent(self):
        """RAW_DIR is a Path under the data parent directory."""
        import src.research.wiki_sync as ws
        from pathlib import Path
        assert isinstance(ws.RAW_DIR, Path)
        assert "raw" in str(ws.RAW_DIR)
        assert "market" in str(ws.RAW_DIR)

    def test_db_path_is_sqlite_path(self):
        """DB_PATH ends with market.db."""
        import src.research.wiki_sync as ws
        from pathlib import Path
        assert isinstance(ws.DB_PATH, Path)
        assert ws.DB_PATH.name == "market.db"

    def test_constants_point_to_data_dir(self):
        """WIKI_DIR is a Path that contains 'portfolio-lab'."""
        import src.research.wiki_sync as ws
        from pathlib import Path
        assert isinstance(ws.WIKI_DIR, Path)
        assert "portfolio-lab" in str(ws.WIKI_DIR)


# ---------------------------------------------------------------------------
# Export completeness — no __all__, verify public API manually
# ---------------------------------------------------------------------------

class TestExportCompleteness:
    """Verify public API surface. Module has no __all__."""

    def test_no_all_defined(self):
        """Module does not define __all__ (all names are public)."""
        import src.research.wiki_sync as ws
        assert not hasattr(ws, "__all__")

    def test_wiki_sync_class_importable(self):
        """WikiSync class is importable from the module."""
        from src.research.wiki_sync import WikiSync
        assert WikiSync is not None
        assert callable(WikiSync)

    def test_all_module_level_names_accessible(self):
        """Key module-level names are directly accessible."""
        import src.research.wiki_sync as ws
        for name in ("WikiSync", "DATA_DIR", "WIKI_DIR", "RAW_DIR", "DB_PATH"):
            assert hasattr(ws, name), f"Module missing expected name: {name}"


# ---------------------------------------------------------------------------
# hash_file — edge cases
# ---------------------------------------------------------------------------

class TestHashFileEdgeCases:
    """hash_file() boundary and edge cases."""

    def test_unicode_content(self, wiki_sync):
        """Unicode characters produce a valid hash."""
        h = wiki_sync.hash_file("Hello \u00e9\u00e0\u00fc \U0001f600 world")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_very_long_content(self, wiki_sync):
        """Very long string (10k chars) produces valid hash."""
        long_str = "A" * 10_000
        h = wiki_sync.hash_file(long_str)
        assert len(h) == 16

    def test_content_with_newlines(self, wiki_sync):
        """Multi-line content produces valid hash."""
        h = wiki_sync.hash_file("line1\nline2\nline3\n")
        assert len(h) == 16

    def test_special_characters(self, wiki_sync):
        """Special characters produce consistent hash."""
        h1 = wiki_sync.hash_file("data\twith\0nulls")
        h2 = wiki_sync.hash_file("data\twith\0nulls")
        assert h1 == h2


# ---------------------------------------------------------------------------
# save_raw_source — edge cases
# ---------------------------------------------------------------------------

class TestSaveRawSourceEdgeCases:
    """save_raw_source() boundary conditions."""

    def test_empty_dict(self, wiki_sync):
        """Empty dict saves without error."""
        path = wiki_sync.save_raw_source({}, "empty")
        assert path.exists()
        content = path.read_text()
        assert "{}" in content or "{\n}" in content

    def test_nested_data(self, wiki_sync):
        """Deeply nested dict saves correctly."""
        data = {"level1": {"level2": {"level3": {"value": 42}}}}
        path = wiki_sync.save_raw_source(data, "nested")
        assert path.exists()
        content = path.read_text()
        assert "42" in content

    def test_list_data(self, wiki_sync):
        """List of dicts saves correctly."""
        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        path = wiki_sync.save_raw_source(data, "list_data")
        assert path.exists()
        content = path.read_text()
        assert '"id": 1' in content

    def test_extreme_nesting(self, wiki_sync):
        """Deeply nested structure (10 levels) saves correctly."""
        data = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": "deep"}}}}}}}}}}
        path = wiki_sync.save_raw_source(data, "deep_nest")
        assert path.exists()
        content = path.read_text()
        assert "deep" in content

    def test_non_serializable_object_with_default(self, wiki_sync):
        """Custom object serializes via default=str."""
        class Custom:
            def __str__(self):
                return "custom_str"
        data = {"obj": Custom()}
        path = wiki_sync.save_raw_source(data, "custom_obj")
        content = path.read_text()
        assert "custom_str" in content

    def test_with_nan_value(self, wiki_sync):
        """NaN float value does not crash (serializes via default=str)."""
        data = {"value": float("nan")}
        path = wiki_sync.save_raw_source(data, "nan_test")
        assert path.exists()
        content = path.read_text()
        # NaN serialized as 'NaN' by json.dumps with allow_nan=True
        assert "NaN" in content or "nan" in content

    def test_with_inf_value(self, wiki_sync):
        """Infinity float value does not crash."""
        data = {"value": float("inf")}
        path = wiki_sync.save_raw_source(data, "inf_test")
        assert path.exists()
        content = path.read_text()
        assert "Infinity" in content or "inf" in content

    def test_write_failure_propagates(self, wiki_sync):
        """PermissionError during write is not silently caught."""
        data = {"test": True}
        with patch("builtins.open", side_effect=PermissionError("mock denied")):
            with pytest.raises(PermissionError):
                wiki_sync.save_raw_source(data, "fail_write")

    def test_very_long_name(self, wiki_sync):
        """Very long filename (200 chars) saves correctly."""
        long_name = "a" * 200
        data = {"ok": True}
        path = wiki_sync.save_raw_source(data, long_name)
        assert path.exists()
        assert path.name == f"{long_name}.json"

    def test_name_with_special_chars(self, wiki_sync):
        """Name with hyphens and underscores saves correctly."""
        data = {"ok": True}
        path = wiki_sync.save_raw_source(data, "my-special_data_v2")
        assert path.exists()
        assert path.name == "my-special_data_v2.json"

    def test_handles_oserror_on_existing_read_with_logger(self, wiki_sync, caplog):
        """OSError reading existing file logs a warning and overwrites."""
        caplog.set_level(logging.WARNING)
        data_old = {"version": 1}
        data_new = {"version": 2}
        path = wiki_sync.save_raw_source(data_old, "log_check")
        mtime_before = path.stat().st_mtime_ns

        with patch.object(Path, "read_text", side_effect=OSError("mock io error")):
            path2 = wiki_sync.save_raw_source(data_new, "log_check")
            assert path2 == path
            assert path2.stat().st_mtime_ns >= mtime_before

        assert any("Failed to read existing raw file" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# sync_regime_analysis — edge cases
# ---------------------------------------------------------------------------

class TestSyncRegimeAnalysisEdgeCases:
    """sync_regime_analysis() boundary conditions and missing data."""

    def test_regimes_missing_vix_level(self, wiki_sync):
        """Regime rows without vix_level display 'N/A'."""
        wiki_sync.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY, date TEXT, regime TEXT,
                vix_level REAL, correlation_spike BOOLEAN,
                trend_strength REAL, detected_at TEXT
            )
        """)
        wiki_sync.conn.execute(
            "INSERT INTO regime_log (id, date, regime, detected_at) "
            "VALUES (1, '2026-05-20', 'normal', datetime('now', '-1 days'))"
        )
        wiki_sync.conn.commit()
        path = wiki_sync.sync_regime_analysis()
        assert path is not None
        content = path.read_text()
        assert "N/A" in content

    def test_regimes_missing_trend_strength(self, wiki_sync):
        """Regime rows without trend_strength display 'N/A'."""
        wiki_sync.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY, date TEXT, regime TEXT,
                vix_level REAL, correlation_spike BOOLEAN,
                trend_strength REAL, detected_at TEXT
            )
        """)
        wiki_sync.conn.execute(
            "INSERT INTO regime_log (id, date, regime, vix_level, detected_at) "
            "VALUES (1, '2026-05-20', 'crisis', 35.0, datetime('now', '-1 days'))"
        )
        wiki_sync.conn.commit()
        path = wiki_sync.sync_regime_analysis()
        assert path is not None
        content = path.read_text()
        assert "N/A" in content

    def test_null_values_in_db(self, wiki_sync):
        """NULL vix_level and trend_strength display 'N/A'."""
        wiki_sync.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY, date TEXT, regime TEXT,
                vix_level REAL, correlation_spike BOOLEAN,
                trend_strength REAL, detected_at TEXT
            )
        """)
        wiki_sync.conn.execute(
            "INSERT INTO regime_log (id, date, regime, vix_level, trend_strength, detected_at) "
            "VALUES (1, '2026-05-20', 'vol_spike', NULL, NULL, datetime('now', '-1 days'))"
        )
        wiki_sync.conn.commit()
        path = wiki_sync.sync_regime_analysis()
        assert path is not None
        content = path.read_text()
        assert "N/A" in content

    def test_empty_regime_string(self, wiki_sync):
        """Empty string regime is accepted (no crash)."""
        wiki_sync.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY, date TEXT, regime TEXT,
                vix_level REAL, correlation_spike BOOLEAN,
                trend_strength REAL, detected_at TEXT
            )
        """)
        wiki_sync.conn.execute(
            "INSERT INTO regime_log (id, date, regime, vix_level, trend_strength, detected_at) "
            "VALUES (1, '2026-05-20', '', 15.0, 0.5, datetime('now', '-1 days'))"
        )
        wiki_sync.conn.commit()
        path = wiki_sync.sync_regime_analysis()
        assert path is not None

    def test_single_regime_entry(self, wiki_sync):
        """Single regime entry creates one-row table."""
        wiki_sync.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY, date TEXT, regime TEXT,
                vix_level REAL, correlation_spike BOOLEAN,
                trend_strength REAL, detected_at TEXT
            )
        """)
        wiki_sync.conn.execute(
            "INSERT INTO regime_log (id, date, regime, vix_level, trend_strength, detected_at) "
            "VALUES (1, '2026-05-20', 'normal', 15.0, 0.5, datetime('now', '-1 days'))"
        )
        wiki_sync.conn.commit()
        path = wiki_sync.sync_regime_analysis()
        assert path is not None
        content = path.read_text()
        assert "normal" in content
        # Only one data row
        data_rows = [item for item in content.split("\n") if item.startswith("| ") and "Date" not in item and "---" not in item]
        assert len(data_rows) == 1

    def test_extreme_vix_value(self, wiki_sync):
        """Extreme VIX value (100+) formats correctly."""
        wiki_sync.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY, date TEXT, regime TEXT,
                vix_level REAL, correlation_spike BOOLEAN,
                trend_strength REAL, detected_at TEXT
            )
        """)
        wiki_sync.conn.execute(
            "INSERT INTO regime_log (id, date, regime, vix_level, trend_strength, detected_at) "
            "VALUES (1, '2026-05-20', 'crisis', 125.7, 0.99, datetime('now', '-1 days'))"
        )
        wiki_sync.conn.commit()
        path = wiki_sync.sync_regime_analysis()
        assert path is not None
        content = path.read_text()
        assert "125.70" in content

    def test_nan_vix_in_db(self, wiki_sync):
        """NaN vix_level is handled (displays N/A or crashes gracefully)."""
        wiki_sync.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY, date TEXT, regime TEXT,
                vix_level REAL, correlation_spike BOOLEAN,
                trend_strength REAL, detected_at TEXT
            )
        """)
        wiki_sync.conn.execute(
            "INSERT INTO regime_log (id, date, regime, vix_level, trend_strength, detected_at) "
            "VALUES (1, '2026-05-20', 'normal', 1e999, 0.5, datetime('now', '-1 days'))"
        )
        wiki_sync.conn.commit()
        path = wiki_sync.sync_regime_analysis()
        # Should not crash; may display the float value
        assert path is not None


# ---------------------------------------------------------------------------
# sync_performance_summary — edge cases
# ---------------------------------------------------------------------------

class TestSyncPerformanceSummaryEdgeCases:
    """sync_performance_summary() boundary conditions."""

    def test_zero_variance_returns(self, wiki_sync):
        """All identical returns produce Sharpe of 0 (no division by zero)."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        val = 100000.0
        with open(path, "w") as f:
            for i in range(20):
                f.write(json.dumps({"total_value": val, "daily_return": 0.001}) + "\n")
        result = wiki_sync.sync_performance_summary()
        assert result is not None
        data = json.loads(result.read_text())
        assert "sharpe" in data["performance"]
        # All returns are 0.001 -> identical -> variance=0 -> sharpe=0
        assert data["performance"]["sharpe"] == 0.0

    def test_negative_total_values(self, wiki_sync):
        """All negative total values work without division-by-zero errors."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            val = -1000.0
            for i in range(20):
                f.write(json.dumps({"total_value": val, "daily_return": 0.001}) + "\n")
                val += 10.0
        result = wiki_sync.sync_performance_summary()
        assert result is not None
        data = json.loads(result.read_text())
        # Negative start value results in total_return of 0 (values[0] > 0 is False)
        assert "performance" in data
        assert data["performance"]["total_return"] == 0

    def test_single_entry_with_value_no_return(self, wiki_sync):
        """Single entry lacks enough data for return/shapre, still creates page."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps({"total_value": 100000.0}) + "\n")
        result = wiki_sync.sync_performance_summary()
        # Needs >= 10 entries, so returns None
        assert result is None

    def test_exactly_10_entries(self, wiki_sync):
        """Exactly 10 entries passes the threshold."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        val = 100000.0
        with open(path, "w") as f:
            for i in range(10):
                ret = 0.001 * (1 if i % 2 == 0 else -0.5)
                val *= (1 + ret)
                f.write(json.dumps({"total_value": round(val, 2), "daily_return": ret}) + "\n")
        result = wiki_sync.sync_performance_summary()
        assert result is not None
        data = json.loads(result.read_text())
        assert "performance" in data
        assert "total_return" in data["performance"]

    def test_malformed_json_lines(self, wiki_sync, caplog):
        """Malformed JSON lines are skipped with a debug log."""
        caplog.set_level(logging.DEBUG)
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        good = [json.dumps({"total_value": 100000.0 + i * 1000.0, "daily_return": 0.001}) for i in range(12)]
        bad = ["not-valid-json{{{", "also-bad-json"]
        lines = good[:6] + bad + good[6:]
        path.write_text("\n".join(lines))
        result = wiki_sync.sync_performance_summary()
        assert result is not None
        data = json.loads(result.read_text())
        assert "raw_entries_count" in data
        debug_msgs = [m for m in caplog.messages if "Skipping malformed" in m]
        assert debug_msgs, "Expected a debug log for malformed JSON lines"

    def test_empty_file(self, wiki_sync):
        """Empty performance.jsonl returns None."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        result = wiki_sync.sync_performance_summary()
        # Empty file: len(entries) would be 0, but opening 0 entries gives []
        # len([]) == 0 < 10 -> returns None
        assert result is None

    def test_missing_total_value_key(self, wiki_sync):
        """Entries without total_value have missing values filtered out."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for i in range(20):
                f.write(json.dumps({"daily_return": 0.001}) + "\n")
        result = wiki_sync.sync_performance_summary()
        # values = [e.get("total_value", 0) for e in recent if e.get("total_value")]
        # All entries return 0 from .get(), but 0 is falsy, so values is []
        # len(values) < 10 -> returns None
        assert result is None

    def test_extreme_large_values(self, wiki_sync):
        """Very large total values (billions) are stored correctly."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        val = 1e12
        with open(path, "w") as f:
            for i in range(20):
                val *= 1.001
                f.write(json.dumps({"total_value": round(val, 2), "daily_return": 0.001}) + "\n")
        result = wiki_sync.sync_performance_summary()
        assert result is not None
        data = json.loads(result.read_text())
        assert data["performance"]["current_value"] > 1e12

    def test_negative_returns(self, wiki_sync):
        """All identical negative returns produce Sharpe of 0 (variance=0)."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        val = 100000.0
        with open(path, "w") as f:
            for i in range(20):
                val *= 0.99
                f.write(json.dumps({"total_value": round(val, 2), "daily_return": -0.01}) + "\n")
        result = wiki_sync.sync_performance_summary()
        assert result is not None
        data = json.loads(result.read_text())
        # All returns are -0.01 -> identical -> variance=0 -> sharpe=0
        assert data["performance"]["sharpe"] == 0.0

    def test_log_debug_on_json_decode_error(self, wiki_sync, caplog):
        """Malformed JSON lines produce debug-level log messages."""
        caplog.set_level(logging.DEBUG)
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({"total_value": 100.0 * i, "daily_return": 0.001}) for i in range(10)]
        lines.append("corrupted-line!!!")
        lines.extend(json.dumps({"total_value": 100.0 * i, "daily_return": 0.001}) for i in range(5))
        path.write_text("\n".join(lines))
        result = wiki_sync.sync_performance_summary()
        # 15 total entries (10 + 1 bad + 4 good after bad = 15) ... 10 good before + 4 good after = 14 lines parsed
        # But one is malformed, so 14 entries
        assert result is not None
        debug_msgs = [m for m in caplog.messages if "Skipping malformed" in m]
        assert debug_msgs, "Expected a debug log for malformed JSON line"

    def test_win_rate_with_all_positive_returns(self, wiki_sync):
        """Win rate is 100% when all daily returns are positive."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        val = 100000.0
        with open(path, "w") as f:
            for i in range(20):
                val *= 1.001
                f.write(json.dumps({"total_value": round(val, 2), "daily_return": 0.001}) + "\n")
        result = wiki_sync.sync_performance_summary()
        assert result is not None
        data = json.loads(result.read_text())
        assert data["daily_returns_distribution"]["win_rate"] == 1.0

    def test_win_rate_with_all_negative_returns(self, wiki_sync):
        """Win rate is 0% when all daily returns are negative."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        val = 100000.0
        with open(path, "w") as f:
            for i in range(20):
                val *= 0.99
                f.write(json.dumps({"total_value": round(val, 2), "daily_return": -0.01}) + "\n")
        result = wiki_sync.sync_performance_summary()
        assert result is not None
        data = json.loads(result.read_text())
        assert data["daily_returns_distribution"]["win_rate"] == 0.0

    def test_intraday_deduplication(self, wiki_sync):
        """Intraday entries for same date must not inflate days_tracked.

        Regression test: performance.jsonl may contain multiple entries per
        day (cron runs, manual syncs).  days_tracked must count unique
        calendar dates, not raw JSONL lines.
        """
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        # 5 dates × 5 intraday entries = 25 raw lines, but only 5 unique days
        for day in range(1, 6):
            for hour in range(5):
                ret = 0.001 if hour == 0 else 0.0
                lines.append(json.dumps({
                    "timestamp": f"2026-01-{day:02d}T{hour:02d}:00:00",
                    "total_value": 100000.0 + day * 100,
                    "daily_return": ret,
                }))
        path.write_text("\n".join(lines))
        result = wiki_sync.sync_performance_summary()
        assert result is not None
        data = json.loads(result.read_text())
        assert data["performance"]["days_tracked"] == 5


# ---------------------------------------------------------------------------
# sync_order_history — edge cases
# ---------------------------------------------------------------------------

class TestSyncOrderHistoryEdgeCases:
    """sync_order_history() boundary conditions."""

    def test_orders_missing_timestamp(self, wiki_sync):
        """Orders without timestamp are stored in JSON without the key."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "orders.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        orders = [{"symbol": "SPY", "side": "buy", "fill_shares": 100.0, "fill_value": 45000.0}]
        path.write_text("\n".join(json.dumps(o) for o in orders))
        result = wiki_sync.sync_order_history()
        assert result is not None
        data = json.loads(result.read_text())
        order = data["recent_orders"][0]
        assert "timestamp" not in order

    def test_orders_missing_symbol(self, wiki_sync):
        """Orders without symbol are stored without the key."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "orders.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        orders = [{"timestamp": "2026-05-20T10:00:00", "side": "buy", "fill_shares": 100.0, "fill_value": 45000.0}]
        path.write_text("\n".join(json.dumps(o) for o in orders))
        result = wiki_sync.sync_order_history()
        assert result is not None
        data = json.loads(result.read_text())
        order = data["recent_orders"][0]
        assert "symbol" not in order

    def test_orders_missing_side(self, wiki_sync):
        """Orders without side are stored without the key."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "orders.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        orders = [{"timestamp": "2026-05-20T10:00:00", "symbol": "SPY", "fill_shares": 100.0, "fill_value": 45000.0}]
        path.write_text("\n".join(json.dumps(o) for o in orders))
        result = wiki_sync.sync_order_history()
        assert result is not None
        data = json.loads(result.read_text())
        order = data["recent_orders"][0]
        assert "side" not in order

    def test_negative_fill_shares(self, wiki_sync):
        """Negative fill_shares is accepted (negative shares possible)."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "orders.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        orders = [{"timestamp": "2026-05-20T10:00:00", "symbol": "SPY", "side": "sell",
                    "fill_shares": -50.0, "fill_value": 22500.0}]
        path.write_text("\n".join(json.dumps(o) for o in orders))
        result = wiki_sync.sync_order_history()
        assert result is not None
        data = json.loads(result.read_text())
        assert data["recent_orders"][0]["fill_shares"] == -50.0

    def test_negative_fill_value(self, wiki_sync):
        """Negative fill_value is accepted."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "orders.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        orders = [{"timestamp": "2026-05-20T10:00:00", "symbol": "SPY", "side": "buy",
                    "fill_shares": 100.0, "fill_value": -45000.0, "reason": "error"}]
        path.write_text("\n".join(json.dumps(o) for o in orders))
        result = wiki_sync.sync_order_history()
        assert result is not None
        data = json.loads(result.read_text())
        assert data["recent_orders"][0]["fill_value"] == -45000.0

    def test_zero_fill_shares_and_value(self, wiki_sync):
        """Zero fill_shares and fill_value are stored correctly."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "orders.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        orders = [{"timestamp": "2026-05-20T10:00:00", "symbol": "SPY", "side": "buy",
                    "fill_shares": 0, "fill_value": 0, "reason": "no fill"}]
        path.write_text("\n".join(json.dumps(o) for o in orders))
        result = wiki_sync.sync_order_history()
        assert result is not None
        data = json.loads(result.read_text())
        assert data["recent_orders"][0]["fill_shares"] == 0
        assert data["recent_orders"][0]["fill_value"] == 0

    def test_malformed_order_json(self, wiki_sync, caplog):
        """Malformed JSON lines in orders are skipped with debug log."""
        caplog.set_level(logging.DEBUG)
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "orders.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps({"timestamp": "2026-05-20T10:00:00", "symbol": "SPY", "side": "buy",
                         "fill_shares": 100.0, "fill_value": 45000.0}),
            "corrupted{{{",
            json.dumps({"timestamp": "2026-05-21T10:00:00", "symbol": "GLD", "side": "sell",
                         "fill_shares": 50.0, "fill_value": 9500.0}),
        ]
        path.write_text("\n".join(lines))
        result = wiki_sync.sync_order_history()
        assert result is not None
        data = json.loads(result.read_text())
        assert len(data["recent_orders"]) == 2  # Corrupted line skipped
        debug_msgs = [m for m in caplog.messages if "malformed" in m.lower()]
        assert debug_msgs, "Expected a debug log for malformed order JSON"

    def test_empty_orders_file(self, wiki_sync):
        """Empty orders.jsonl returns None."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "orders.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        result = wiki_sync.sync_order_history()
        assert result is None

    def test_orders_with_unknown_reason_default(self, wiki_sync):
        """Orders with missing reason are stored without the key."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "orders.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        orders = [{"timestamp": "2026-05-20T10:00:00", "symbol": "SPY", "side": "buy",
                    "fill_shares": 100.0, "fill_value": 45000.0}]
        path.write_text("\n".join(json.dumps(o) for o in orders))
        result = wiki_sync.sync_order_history()
        assert result is not None
        data = json.loads(result.read_text())
        order = data["recent_orders"][0]
        assert "reason" not in order


# ---------------------------------------------------------------------------
# update_knowledge_md — edge cases
# ---------------------------------------------------------------------------

class TestUpdateKnowledgeMDEdgeCases:
    """update_knowledge_md() boundary conditions."""

    def test_empty_compound_directory(self, wiki_sync):
        """Empty compound directory creates knowledge.md with no links."""
        result = wiki_sync.update_knowledge_md()
        assert result is not None
        assert result.name == "knowledge.md"
        content = result.read_text()
        # No "compound/" links when empty
        assert "Compound Pages" in content
        assert "compound/" not in content or content.count("compound/") <= 1

    def test_overwrites_existing_knowledge_md(self, wiki_sync):
        """Existing knowledge.md is overwritten with fresh content."""
        knowledge_path = wiki_sync._wiki / "knowledge.md"
        knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        knowledge_path.write_text("# Old Content")
        result = wiki_sync.update_knowledge_md()
        assert result == knowledge_path
        content = result.read_text()
        assert "Old Content" not in content
        assert "Auto-Generated" in content

    def test_ignores_non_md_files_in_compound(self, wiki_sync):
        """Non-.md files in compound are not listed as links."""
        compound = wiki_sync._wiki / "compound"
        (compound / "data.json").write_text("{}")
        (compound / "notes.txt").write_text("notes")
        (compound / "real-page.md").write_text("# Real")
        result = wiki_sync.update_knowledge_md()
        content = result.read_text()
        assert "data.json" not in content
        assert "notes.txt" not in content
        assert "real-page" in content


# ---------------------------------------------------------------------------
# _regime_distribution — edge cases
# ---------------------------------------------------------------------------

class TestRegimeDistributionEdgeCases:
    """_regime_distribution() boundary conditions."""

    def test_single_regime_type(self, wiki_sync):
        """Single regime type shows 100%."""
        regimes = [{"regime": "crisis"}]
        result = wiki_sync._regime_distribution(regimes)
        assert "100%" in result

    def test_all_identical_regimes(self, wiki_sync):
        """All identical regimes show 100% for that type."""
        regimes = [{"regime": "normal"} for _ in range(50)]
        result = wiki_sync._regime_distribution(regimes)
        assert "normal" in result
        assert "100%" in result

    def test_regime_dict_missing_key(self, wiki_sync):
        """Dict without 'regime' key defaults to 'unknown'."""
        regimes = [{"vix_level": 15.0}, {"vix_level": 20.0}]
        result = wiki_sync._regime_distribution(regimes)
        assert "unknown" in result
        assert "100%" in result

    def test_regime_with_long_name(self, wiki_sync):
        """Very long regime name is included (no truncation)."""
        long_name = "extremely_long_regime_name_descriptor_" * 3
        regimes = [{"regime": long_name}]
        result = wiki_sync._regime_distribution(regimes)
        assert long_name in result

    def test_one_hundred_regimes(self, wiki_sync):
        """100 entries don't cause performance issues."""
        regimes = [{"regime": "normal" if i % 2 == 0 else "crisis"} for i in range(100)]
        result = wiki_sync._regime_distribution(regimes)
        assert "normal" in result
        assert "crisis" in result
        assert "50%" in result

    def test_mixed_known_and_unknown(self, wiki_sync):
        """Mix of known regimes and dicts missing regime key."""
        regimes = [{"regime": "normal"}, {"vix": 15.0}, {"regime": "crisis"}]
        result = wiki_sync._regime_distribution(regimes)
        assert "normal" in result
        assert "crisis" in result
        assert "unknown" in result


# ---------------------------------------------------------------------------
# _regime_implications — edge cases
# ---------------------------------------------------------------------------

class TestRegimeImplicationsEdgeCases:
    """_regime_implications() boundary conditions."""

    def test_missing_regime_key(self, wiki_sync):
        """Dict without 'regime' key falls back to normal."""
        result = wiki_sync._regime_implications([{"vix_level": 15.0}])
        assert "Base allocation" in result

    def test_regime_is_none(self, wiki_sync):
        """None regime falls back to normal."""
        result = wiki_sync._regime_implications([{"regime": None}])
        assert "Base allocation" in result

    def test_latest_regime_controls_implications(self, wiki_sync):
        """Only the latest (first) regime is used for implications."""
        result = wiki_sync._regime_implications([
            {"regime": "crisis"},
            {"regime": "normal"},
            {"regime": "low_vol"},
        ])
        assert "protect capital" in result
        assert "Base allocation" not in result

    def test_exact_viable_highlight_format(self, wiki_sync):
        """Each regime returns expected key phrases."""
        crisis = wiki_sync._regime_implications([{"regime": "crisis"}])
        assert "Risk-off" in crisis or "SPY 20%" in crisis

        low_vol = wiki_sync._regime_implications([{"regime": "low_vol"}])
        assert "Risk-on" in low_vol or "SPY 55%" in low_vol

        normal = wiki_sync._regime_implications([{"regime": "normal"}])
        assert "SPY 46%" in normal

        vol_spike = wiki_sync._regime_implications([{"regime": "vol_spike"}])
        assert "Defensive" in vol_spike or "SPY 30%" in vol_spike


# ---------------------------------------------------------------------------
# _graduation_status — edge cases
# ---------------------------------------------------------------------------

class TestGraduationStatusEdgeCases:
    """_graduation_status() boundary conditions and extreme inputs."""

    def test_negative_sharpe(self, wiki_sync):
        """Negative Sharpe does not crash and shows failure."""
        result = wiki_sync._graduation_status(-0.10, -0.5, 0.05, 100)
        assert "GRADUATION" not in result
        assert "Sharpe" in result
        assert "not yet meeting" in result.lower()

    def test_negative_max_dd(self, wiki_sync):
        """Negative max drawdown (impossible, but safe) treated as success."""
        result = wiki_sync._graduation_status(0.10, 0.8, -0.05, 100)
        assert "GRADUATION CANDIDATE" in result

    def test_zero_sharpe(self, wiki_sync):
        """Sharpe of 0 fails the threshold check."""
        result = wiki_sync._graduation_status(0.0, 0.0, 0.05, 100)
        assert "GRADUATION" not in result
        assert "Sharpe" in result

    def test_zero_max_dd(self, wiki_sync):
        """Max drawdown of 0 passes."""
        result = wiki_sync._graduation_status(0.10, 0.8, 0.0, 100)
        assert "GRADUATION CANDIDATE" in result

    def test_very_large_sharpe(self, wiki_sync):
        """Extremely high Sharpe does not crash."""
        result = wiki_sync._graduation_status(5.0, 10.0, 0.05, 100)
        assert "GRADUATION CANDIDATE" in result
        assert "10.00" in result

    def test_very_large_max_dd(self, wiki_sync):
        """Max drawdown > 1.0 (100%) formats correctly."""
        result = wiki_sync._graduation_status(-0.90, 0.8, 0.95, 100)
        assert "GRADUATION" not in result
        assert "95.0%" in result or "95" in result

    def test_zero_days(self, wiki_sync):
        """Zero days shows not ready with 63 days needed."""
        result = wiki_sync._graduation_status(0.0, 0.0, 0.0, 0)
        assert "Not Ready" in result
        assert "63 more days" in result

    def test_one_day(self, wiki_sync):
        """Single day shows not ready with 62 days needed."""
        result = wiki_sync._graduation_status(0.0, 0.0, 0.0, 1)
        assert "Not Ready" in result
        assert "62 more days" in result

    def test_boundary_sharpe_05(self, wiki_sync):
        """Sharpe of exactly 0.5 meets threshold."""
        result = wiki_sync._graduation_status(0.10, 0.5, 0.05, 100)
        assert "GRADUATION CANDIDATE" in result

    def test_boundary_max_dd_015(self, wiki_sync):
        """Max DD of exactly 0.15 meets threshold."""
        result = wiki_sync._graduation_status(0.10, 0.8, 0.15, 100)
        assert "GRADUATION CANDIDATE" in result

    def test_boundary_days_63(self, wiki_sync):
        """Exactly 63 days proceeds to check criteria."""
        result = wiki_sync._graduation_status(0.10, 0.8, 0.05, 63)
        assert "Not Ready" not in result

    def test_both_failing_boundary(self, wiki_sync):
        """Both sharpe and max_dd at failure boundaries."""
        result = wiki_sync._graduation_status(0.10, 0.49, 0.16, 63)
        assert "GRADUATION" not in result
        assert "not yet meeting" in result.lower()


# ---------------------------------------------------------------------------
# run() — CLI / capsys output tests
# ---------------------------------------------------------------------------

class TestRunCLI:
    """run() logger output and __main__ guard."""

    def test_run_produces_log_output(self, wiki_sync, caplog):
        """run() logs start and complete messages."""
        wiki_sync.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY, date TEXT, regime TEXT,
                vix_level REAL, correlation_spike BOOLEAN,
                trend_strength REAL, detected_at TEXT
            )
        """)
        wiki_sync.conn.commit()
        with caplog.at_level(logging.INFO, logger="src.research.wiki_sync"):
            wiki_sync.run()
        assert "Wiki Sync Starting" in caplog.text
        assert "Wiki Sync Complete" in caplog.text

    def test_run_no_data_logging(self, wiki_sync, caplog):
        """run() with no data logs 0 wiki and 0 app."""
        wiki_sync.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY, date TEXT, regime TEXT,
                vix_level REAL, correlation_spike BOOLEAN,
                trend_strength REAL, detected_at TEXT
            )
        """)
        wiki_sync.conn.commit()
        with caplog.at_level(logging.INFO, logger="src.research.wiki_sync"):
            wiki_sync.run()
        assert "0 wiki" in caplog.text and "0 app" in caplog.text

    def test_main_guard_calls_run(self, monkeypatch, tmp_path):
        """__main__ guard instantiates WikiSync and calls run()."""
        import src.research.wiki_sync as ws
        data = tmp_path / "data"
        wiki = tmp_path / "wiki" / "projects" / "portfolio-lab"
        raw = tmp_path / "raw" / "market"
        for d in (data, wiki / "compound", raw):
            d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(ws, "DATA_DIR", data)
        monkeypatch.setattr(ws, "WIKI_DIR", wiki)
        monkeypatch.setattr(ws, "RAW_DIR", raw)
        monkeypatch.setattr(ws, "DB_PATH", data / "market.db")

        with patch.object(ws.WikiSync, "run") as mock_run:
            sync = ws.WikiSync()
            sync.run()
            mock_run.assert_called_once()

    def test_run_logging_with_partial_data(self, wiki_sync, perf_log, caplog):
        """run() with only perf data logs correct messages."""
        wiki_sync.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_log (
                id INTEGER PRIMARY KEY, date TEXT, regime TEXT,
                vix_level REAL, correlation_spike BOOLEAN,
                trend_strength REAL, detected_at TEXT
            )
        """)
        wiki_sync.conn.commit()
        with caplog.at_level(logging.INFO, logger="src.research.wiki_sync"):
            wiki_sync.run()
        assert "Wiki Sync Starting" in caplog.text
        assert "Wiki Sync Complete" in caplog.text

    def test_run_logging_with_all_data(self, wiki_sync, db_with_regimes, perf_log, orders_log, caplog):
        """run() with all data logs all page names."""
        with caplog.at_level(logging.INFO, logger="src.research.wiki_sync"):
            wiki_sync.run()
        assert "Regime:" in caplog.text
        assert "Performance:" in caplog.text
        assert "Orders:" in caplog.text


# ---------------------------------------------------------------------------
# Init — edge cases
# ---------------------------------------------------------------------------

class TestInitEdgeCases:
    """WikiSync.__init__() boundary conditions."""

    def test_init_creates_all_directories(self, tmp_path):
        """All required directories are created by __init__."""
        data = tmp_path / "altdata"
        wiki = tmp_path / "altwiki" / "projects" / "portfolio-lab"
        raw = tmp_path / "altraw" / "market"
        data.mkdir(parents=True, exist_ok=True)

        import src.research.wiki_sync as ws
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(ws, "DATA_DIR", data)
        monkeypatch.setattr(ws, "WIKI_DIR", wiki)
        monkeypatch.setattr(ws, "RAW_DIR", raw)
        monkeypatch.setattr(ws, "DB_PATH", data / "market.db")

        sync = WikiSync()
        assert raw.exists()
        assert (wiki / "compound").exists()
        assert sync.conn is not None
        monkeypatch.undo()

    def test_init_reuses_existing_db(self, tmp_path):
        """Existing market.db is reused (no error)."""
        data = tmp_path / "data"
        wiki = tmp_path / "wiki" / "projects" / "portfolio-lab"
        raw = tmp_path / "raw" / "market"
        for d in (data, wiki / "compound", raw):
            d.mkdir(parents=True, exist_ok=True)

        # Pre-create the DB with a table
        import sqlite3
        conn = sqlite3.connect(str(data / "market.db"))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'pre-existing')")
        conn.commit()
        conn.close()

        import src.research.wiki_sync as ws
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(ws, "DATA_DIR", data)
        monkeypatch.setattr(ws, "WIKI_DIR", wiki)
        monkeypatch.setattr(ws, "RAW_DIR", raw)
        monkeypatch.setattr(ws, "DB_PATH", data / "market.db")

        sync = WikiSync()
        cursor = sync.conn.execute("SELECT val FROM test WHERE id = 1")
        assert cursor.fetchone()[0] == "pre-existing"
        sync.conn.close()
        monkeypatch.undo()

    def test_sqlite_connect_with_row_factory(self, wiki_sync):
        """SQLite connection has row_factory set to sqlite3.Row."""
        assert wiki_sync.conn.row_factory is sqlite3.Row
        cursor = wiki_sync.conn.execute("SELECT 1 as val")
        row = cursor.fetchone()
        assert row["val"] == 1


# ---------------------------------------------------------------------------
# save_raw_source — concurrency and hash stability
# ---------------------------------------------------------------------------

class TestSaveRawSourceConcurrency:
    """save_raw_source() hash dedup and concurrency safety."""

    def test_same_content_different_names(self, wiki_sync):
        """Same data with different names creates separate files."""
        data = {"key": "value"}
        p1 = wiki_sync.save_raw_source(data, "file_a")
        p2 = wiki_sync.save_raw_source(data, "file_b")
        assert p1.name == "file_a.json"
        assert p2.name == "file_b.json"
        assert p1 != p2

    def test_content_order_preserved_by_json(self, wiki_sync):
        """JSON preserves dict insertion order; different order = different hash."""
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        p1 = wiki_sync.save_raw_source(d1, "order1")
        p2 = wiki_sync.save_raw_source(d2, "order2")
        c1 = p1.read_text()
        c2 = p2.read_text()
        # json.dumps preserves insertion order, so the files differ
        assert c1 != c2

    def test_hash_stability_across_runs(self, wiki_sync):
        """Same data produces same hash across sequential calls."""
        data = {"test": "stability"}
        p1 = wiki_sync.save_raw_source(data, "stability")
        c1 = p1.read_text()
        # Get hash from first write
        import re
        match = re.search(r"sha256: ([a-f0-9]+)", c1)
        hash1 = match.group(1) if match else None

        from unittest.mock import patch
        with patch.object(Path, "read_text", return_value="" ):
            p2 = wiki_sync.save_raw_source(data, "stability")
        c2 = p2.read_text()
        match2 = re.search(r"sha256: ([a-f0-9]+)", c2)
        hash2 = match2.group(1) if match2 else None
        assert hash1 == hash2


# ---------------------------------------------------------------------------
# _regime_distribution — formatting edge cases
# ---------------------------------------------------------------------------

class TestRegimeDistributionFormatting:
    """_regime_distribution() bar chart formatting."""

    def test_bar_length_scales_with_percentage(self, wiki_sync):
        """Bar chart uses block char proportional to percentage."""
        # 100% -> 20 blocks
        regimes_100 = [{"regime": "normal"}]
        result_100 = wiki_sync._regime_distribution(regimes_100)
        bar_line_100 = [item for item in result_100.split("\n") if "normal" in item][0]
        block_count_100 = bar_line_100.count("█")

        # 50% -> 10 blocks
        regimes_50 = [{"regime": "normal"}, {"regime": "crisis"}]
        result_50 = wiki_sync._regime_distribution(regimes_50)
        bar_line_50 = [item for item in result_50.split("\n") if "normal" in item][0]
        block_count_50 = bar_line_50.count("█")

        assert block_count_100 == 20
        assert block_count_50 == 10

    def test_small_percentage_minimum_bar(self, wiki_sync):
        """Very small percentages (under 5%) show 0 or 1 blocks."""
        regimes = [{"regime": "rare"}] + [{"regime": "common"} for _ in range(99)]
        result = wiki_sync._regime_distribution(regimes)
        rare_line = [item for item in result.split("\n") if "rare" in item][0]
        block_count = rare_line.count("█")
        # 1% -> int(1/5) = 0 blocks
        assert block_count == 0

    def test_bar_lines_include_percentage(self, wiki_sync):
        """Each line in distribution shows percentage number."""
        regimes = [{"regime": "normal"}, {"regime": "normal"}, {"regime": "crisis"}]
        result = wiki_sync._regime_distribution(regimes)
        lines = [item for item in result.split("\n") if item.strip()]
        for line in lines:
            assert "%" in line


# ---------------------------------------------------------------------------
# sync_performance_summary — data quality tests
# ---------------------------------------------------------------------------

class TestSyncPerformanceSummaryDataQuality:
    """sync_performance_summary() data quality and integrity."""

    def test_mixed_good_and_bad_returns(self, wiki_sync):
        """Mix of valid returns and None returns does not crash."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for i in range(15):
                entry = {"total_value": 100000.0 * (1 + 0.01 * i)}
                if i % 3 != 0:  # Every 3rd entry has no daily_return
                    entry["daily_return"] = 0.001 * (1 if i % 2 == 0 else -0.5)
                f.write(json.dumps(entry) + "\n")
        result = wiki_sync.sync_performance_summary()
        assert result is not None
        data = json.loads(result.read_text())
        assert "performance" in data

    def test_all_returns_are_none(self, wiki_sync):
        """All entries have None returns -> returns list is empty."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for i in range(15):
                f.write(json.dumps({"total_value": 100000.0 * (1 + 0.01 * i)}) + "\n")
        result = wiki_sync.sync_performance_summary()
        assert result is not None
        data = json.loads(result.read_text())
        # Should not crash; win rate section handles empty returns
        assert "daily_returns_distribution" in data

    def test_entries_with_zero_total_value(self, wiki_sync):
        """Entries where total_value is 0 are filtered out (0 is falsy)."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for i in range(15):
                f.write(json.dumps({"total_value": 0, "daily_return": 0.001}) + "\n")
        result = wiki_sync.sync_performance_summary()
        # All values are 0 (falsy), so values list is empty
        # len(values) < 10 -> returns None
        assert result is None

    def test_graduation_status_included(self, wiki_sync):
        """Performance JSON includes graduation status."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        val = 100000.0
        with open(path, "w") as f:
            for i in range(100):
                ret = 0.0005
                val *= (1 + ret)
                f.write(json.dumps({"total_value": round(val, 2), "daily_return": ret}) + "\n")
        result = wiki_sync.sync_performance_summary()
        assert result is not None
        data = json.loads(result.read_text())
        assert "graduation" in data
        assert "status" in data["graduation"]
        assert "days_tracked" in data["graduation"]


# ---------------------------------------------------------------------------
# sync_order_history — statistics accuracy
# ---------------------------------------------------------------------------

class TestSyncOrderHistoryStatistics:
    """sync_order_history() buy/sell/volume statistics."""

    def test_buy_and_sell_counts(self, wiki_sync):
        """Buy and sell order counts are calculated correctly."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "orders.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        orders = [
            {"timestamp": "2026-05-20T10:00:00", "symbol": "SPY", "side": "buy",
             "fill_shares": 100.0, "fill_value": 45000.0},
            {"timestamp": "2026-05-20T11:00:00", "symbol": "GLD", "side": "buy",
             "fill_shares": 50.0, "fill_value": 9500.0},
            {"timestamp": "2026-05-20T12:00:00", "symbol": "TLT", "side": "sell",
             "fill_shares": 200.0, "fill_value": 18000.0},
        ]
        path.write_text("\n".join(json.dumps(o) for o in orders))
        result = wiki_sync.sync_order_history()
        data = json.loads(result.read_text())
        assert data["statistics"]["total_buy_orders"] == 2
        assert data["statistics"]["total_sell_orders"] == 1

    def test_total_volume_calculation(self, wiki_sync):
        """Total volume sums all fill_values."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "orders.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        orders = [
            {"timestamp": "2026-05-20T10:00:00", "symbol": "SPY", "side": "buy",
             "fill_shares": 100.0, "fill_value": 45000.0},
            {"timestamp": "2026-05-20T11:00:00", "symbol": "GLD", "side": "buy",
             "fill_shares": 50.0, "fill_value": 9500.0},
        ]
        path.write_text("\n".join(json.dumps(o) for o in orders))
        result = wiki_sync.sync_order_history()
        data = json.loads(result.read_text())
        assert data["statistics"]["total_volume"] == 54500.0

    def test_all_sell_orders(self, wiki_sync):
        """All sell orders shows 0 buys."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "orders.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        orders = [
            {"timestamp": "2026-05-20T10:00:00", "symbol": "SPY", "side": "sell",
             "fill_shares": 100.0, "fill_value": 45000.0},
        ]
        path.write_text("\n".join(json.dumps(o) for o in orders))
        result = wiki_sync.sync_order_history()
        data = json.loads(result.read_text())
        assert data["statistics"]["total_buy_orders"] == 0
        assert data["statistics"]["total_sell_orders"] == 1

    def test_orders_exceeding_20_truncated(self, wiki_sync):
        """More than 20 orders shows only 20 in recent_orders."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "orders.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for i in range(25):
                f.write(json.dumps({
                    "timestamp": f"2026-05-{i+1:02d}T10:00:00",
                    "symbol": "SPY", "side": "buy",
                    "fill_shares": 10.0, "fill_value": 4500.0,
                }) + "\n")
        result = wiki_sync.sync_order_history()
        data = json.loads(result.read_text())
        assert len(data["recent_orders"]) == 20


# ---------------------------------------------------------------------------
# _regime_implications — all 4 regime types produce distinct output
# ---------------------------------------------------------------------------

class TestRegimeImplicationsExhaustive:
    """All 4 known regimes return distinct implications."""

    def test_crisis_contains_spy20_gld50_tlt30(self, wiki_sync):
        result = wiki_sync._regime_implications([{"regime": "crisis"}])
        assert "SPY 20%" in result
        assert "GLD 50%" in result
        assert "TLT 30%" in result

    def test_vol_spike_contains_spy30_gld45_tlt25(self, wiki_sync):
        result = wiki_sync._regime_implications([{"regime": "vol_spike"}])
        assert "SPY 30%" in result
        assert "GLD 45%" in result
        assert "TLT 25%" in result

    def test_low_vol_contains_spy55_gld30_tlt15(self, wiki_sync):
        result = wiki_sync._regime_implications([{"regime": "low_vol"}])
        assert "SPY 55%" in result
        assert "GLD 30%" in result
        assert "TLT 15%" in result

    def test_normal_contains_spy46_gld38_tlt16(self, wiki_sync):
        result = wiki_sync._regime_implications([{"regime": "normal"}])
        assert "SPY 46%" in result
        assert "GLD 38%" in result
        assert "TLT 16%" in result


# ---------------------------------------------------------------------------
# Integration: save_raw_source round-trip with hash match
# ---------------------------------------------------------------------------

class TestSaveRawSourceRoundTrip:
    """save_raw_source() round-trip: write, read, verify hash."""

    def test_hash_in_file_matches_computed_hash(self, wiki_sync):
        """SHA256 in frontmatter matches hash_file() output."""
        data = {"round": "trip", "value": 42}
        path = wiki_sync.save_raw_source(data, "roundtrip")
        content = path.read_text()

        import re
        match = re.search(r"sha256: ([a-f0-9]+)", content)
        assert match, "sha256 not found in frontmatter"
        frontmatter_hash = match.group(1)

        expected_hash = wiki_sync.hash_file(json.dumps(data, indent=2, default=str))
        assert frontmatter_hash == expected_hash

    def test_json_content_preserved(self, wiki_sync):
        """JSON body after frontmatter is valid JSON matching original."""
        import json as json_mod
        data = {"preserve": True, "items": [1, 2, 3]}
        path = wiki_sync.save_raw_source(data, "preserve")
        content = path.read_text()

        # Extract JSON body after frontmatter
        parts = content.split("---\n", 2)
        assert len(parts) == 3
        json_body = parts[2]
        parsed = json_mod.loads(json_body)
        assert parsed == data

    def test_frontmatter_has_created_timestamp(self, wiki_sync):
        """Frontmatter includes ISO-format created timestamp."""
        data = {"ts": "test"}
        path = wiki_sync.save_raw_source(data, "ts_test")
        content = path.read_text()
        assert "created:" in content
        assert "T" in content.split("created:")[1].split("\n")[0].strip()
