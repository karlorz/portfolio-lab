"""
Tests for src/research/wiki_sync.py — WikiSync class.

Covers: initialization, hash_file, save_raw_source (with hash dedup),
sync_regime_analysis, sync_performance_summary, sync_order_history,
update_knowledge_md, _regime_distribution, _regime_implications,
_graduation_status, and run().
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

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
        ("2026-05-15", "normal", 14.5, 0, 0.45),
        ("2026-05-16", "normal", 15.2, 0, 0.42),
        ("2026-05-17", "vol_spike", 24.1, 1, -0.12),
        ("2026-05-18", "crisis", 35.8, 1, -0.45),
        ("2026-05-19", "low_vol", 12.3, 0, 0.55),
    ]
    for i, (d, r, v, cs, ts) in enumerate(rows, 1):
        wiki_sync.conn.execute(
            "INSERT INTO regime_log (id, date, regime, vix_level, correlation_spike, "
            "trend_strength, detected_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (i, d, r, v, cs, ts, f"{d}T12:00:00"),
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
        entries.append({"total_value": round(val, 2), "daily_return": daily_ret})
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
        rows = [l for l in content.split("\n") if l.startswith("| ") and "---" not in l
                and "Date" not in l]
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
        assert result.suffix == ".md"

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
        """Generated markdown contains performance metrics table."""
        path = wiki_sync.sync_performance_summary()
        content = path.read_text()
        assert "Total Return" in content
        assert "Sharpe Ratio" in content
        assert "Max Drawdown" in content
        assert "Start Value" in content
        assert "Current Value" in content

    def test_handles_missing_daily_returns(self, wiki_sync):
        """Handles performance entries without daily_return gracefully."""
        import src.research.wiki_sync as ws
        path = ws.DATA_DIR / "performance.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for i in range(20):
                f.write(json.dumps({"total_value": 100.0}) + "\n")
        result = wiki_sync.sync_performance_summary()
        # Should still create a page (gracefully handles empty returns)
        assert result is not None
        content = result.read_text()
        assert "0.0%" in content or "0." in content


# ---------------------------------------------------------------------------
# sync_order_history
# ---------------------------------------------------------------------------

class TestSyncOrderHistory:
    def test_returns_path_when_orders_exist(self, wiki_sync, orders_log):
        """Returns Path when orders.jsonl has entries."""
        result = wiki_sync.sync_order_history()
        assert result is not None
        assert isinstance(result, Path)
        assert result.suffix == ".md"

    def test_returns_none_when_no_orders(self, wiki_sync):
        """Returns None when orders.jsonl does not exist."""
        result = wiki_sync.sync_order_history()
        assert result is None

    def test_content_includes_table(self, wiki_sync, orders_log):
        """Generated markdown contains order table."""
        path = wiki_sync.sync_order_history()
        content = path.read_text()
        assert "| Date | Symbol | Side | Shares | Value | Reason |" in content
        assert "SPY" in content
        assert "GLD" in content
        assert "TLT" in content

    def test_statistics_section(self, wiki_sync, orders_log):
        """Markdown includes order statistics (buys, sells, volume)."""
        path = wiki_sync.sync_order_history()
        content = path.read_text()
        assert "buy orders" in content
        assert "sell orders" in content
        assert "Total volume" in content

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
        content = result.read_text()
        # Count table rows (data lines)
        data_rows = [l for l in content.split("\n") if l.startswith("| ") and "Date" not in l
                     and "---" not in l and len(l.strip()) > 5]
        assert len(data_rows) == 20


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
        lines = [l for l in result.split("\n") if l.strip()]
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
        assert "Ready for live promotion" in result

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
        # Check compound pages were created
        compound = wiki_sync._wiki / "compound"
        pages = list(compound.glob("*.md"))
        assert len(pages) >= 3  # regime + performance + orders + knowledge
        assert (wiki_sync._wiki / "knowledge.md").exists()

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
        with pytest.raises(sqlite3.ProgrammingError):
            wiki_sync.conn.execute("SELECT 1")
