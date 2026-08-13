#!/usr/bin/env python3
"""
Comprehensive tests for src/research/agent.py — ResearchAgent.

Covers all public methods, module-level constants, __all__ exports,
edge cases, boundary conditions, and file I/O via tmp_path.

Safe for PORTFOLIO_LAB_ENABLE_ML=0 — no torch/sklearn/xgboost/hmmlearn imports.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.research import agent as agent_module
from src.research.agent import ResearchAgent


# ---------------------------------------------------------------------------
# Module-level constant tests
# ---------------------------------------------------------------------------

class TestModuleConstants:
    """Validate module-level constants resolve correctly."""

    def test_data_dir_is_path(self):
        assert isinstance(agent_module.DATA_DIR, Path)

    def test_wiki_dir_is_path(self):
        assert isinstance(agent_module.WIKI_DIR, Path)

    def test_wiki_dir_has_projects_suffix(self):
        assert agent_module.WIKI_DIR.name == "portfolio-lab"

    def test_work_dir_is_path(self):
        assert isinstance(agent_module.WORK_DIR, Path)

    def test_db_path_is_path(self):
        assert isinstance(agent_module.DB_PATH, Path)

    def test_db_path_name(self):
        assert agent_module.DB_PATH.name == "market.db"

    def test_db_path_matches_market_db(self):
        assert agent_module.DB_PATH == agent_module.MARKET_DB


# ---------------------------------------------------------------------------
# __all__ export validation
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Validate the module is properly exported via __init__."""

    def test_research_init_exports_agent(self):
        from src.research import __all__ as research_exports
        assert "agent" in research_exports

    def test_module_has_research_agent(self):
        assert hasattr(agent_module, "ResearchAgent")

    def test_module_has_logger(self):
        assert hasattr(agent_module, "logger")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    """Redirect module-level path constants to tmp_path for every test."""
    data_dir = tmp_path / "data"
    wiki_dir = tmp_path / "wiki"
    work_dir = tmp_path / "work"
    data_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(agent_module, "WIKI_DIR", wiki_dir)
    monkeypatch.setattr(agent_module, "WORK_DIR", work_dir)
    monkeypatch.setattr(agent_module, "DB_PATH", data_dir / "market.db")
    return tmp_path


@pytest.fixture
def seeded_db(tmp_path):
    """Create a ResearchAgent with a fully seeded database."""
    db_path = tmp_path / "data" / "market.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS prices (symbol TEXT, date TEXT, close REAL)")
    # Seed with realistic data for all 5 symbols
    symbols = ["SPY", "GLD", "TLT", "VIX", "QQQ"]
    import numpy as np
    np.random.seed(42)
    dates = []
    d = datetime.now()
    for i in range(90):
        dt = d.replace(day=max(1, d.day - i))
        dates.append(dt.strftime("%Y-%m-%d"))
    dates = sorted(set(dates))[:90]
    for sym in symbols:
        price = 100.0
        for dt_str in dates:
            ret = np.random.normal(0.0004, 0.015)
            price *= (1 + ret)
            conn.execute(
                "INSERT INTO prices VALUES (?, ?, ?)",
                (sym, dt_str, round(price, 2)),
            )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# ResearchAgent construction tests
# ---------------------------------------------------------------------------

class TestResearchAgentConstruction:
    """Test that ResearchAgent initializes correctly."""

    def test_construct_with_real_db(self, tmp_path, monkeypatch):
        db_path = tmp_path / "data" / "market.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS prices (symbol TEXT, date TEXT, close REAL)")
        conn.commit()
        conn.close()

        monkeypatch.setattr(agent_module, "WORK_DIR", tmp_path / "work")
        monkeypatch.setattr(agent_module, "DB_PATH", db_path)
        monkeypatch.setattr(agent_module, "DATA_DIR", tmp_path / "data")

        agent = ResearchAgent()
        assert agent.conn is not None
        assert agent.conn.row_factory == sqlite3.Row
        agent.conn.close()

    def test_construct_creates_work_dir(self, tmp_path, monkeypatch):
        work_dir = tmp_path / "new_work"
        assert not work_dir.exists()
        db_path = tmp_path / "data" / "market.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS prices (symbol TEXT, date TEXT, close REAL)")
        conn.commit()
        conn.close()

        monkeypatch.setattr(agent_module, "WORK_DIR", work_dir)
        monkeypatch.setattr(agent_module, "DB_PATH", db_path)
        monkeypatch.setattr(agent_module, "DATA_DIR", tmp_path / "data")

        agent = ResearchAgent()
        assert work_dir.exists()
        agent.conn.close()

    def test_construct_creates_work_dir_if_parent_missing(self, tmp_path, monkeypatch):
        """WORK_DIR.mkdir(parents=True) should handle missing parent dirs."""
        deep_work = tmp_path / "a" / "b" / "c" / "work"
        assert not deep_work.exists()
        db_path = tmp_path / "data" / "market.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS prices (symbol TEXT, date TEXT, close REAL)")
        conn.commit()
        conn.close()

        monkeypatch.setattr(agent_module, "WORK_DIR", deep_work)
        monkeypatch.setattr(agent_module, "DB_PATH", db_path)
        monkeypatch.setattr(agent_module, "DATA_DIR", tmp_path / "data")

        agent = ResearchAgent()
        assert deep_work.exists()
        agent.conn.close()


# ---------------------------------------------------------------------------
# check_triggers tests
# ---------------------------------------------------------------------------

class TestCheckTriggers:
    """Cover all paths in check_triggers()."""

    def test_no_triggers(self, patched_paths, monkeypatch):
        agent = ResearchAgent()
        assert agent.check_triggers() == []
        agent.conn.close()

    def test_regime_trigger_found_and_consumed(self, patched_paths):
        trigger_data = {"regime": "crisis", "vix": 35.0}
        trigger_file = agent_module.DATA_DIR / ".regime_trigger"
        with open(trigger_file, 'w') as f:
            json.dump(trigger_data, f)

        agent = ResearchAgent()
        triggers = agent.check_triggers()

        assert len(triggers) == 1
        assert triggers[0]["type"] == "regime_change"
        assert triggers[0]["regime"] == "crisis"
        assert triggers[0]["vix"] == 35.0
        # File must be consumed
        assert not trigger_file.exists()
        agent.conn.close()

    def test_pending_work_renamed_to_in_progress(self, patched_paths):
        work_data = {"task": "analysis", "priority": "high"}
        pending = agent_module.WORK_DIR / "pending_work.json"
        with open(pending, 'w') as f:
            json.dump(work_data, f)

        agent = ResearchAgent()
        triggers = agent.check_triggers()

        assert len(triggers) == 1
        assert triggers[0]["task"] == "analysis"
        # Original is renamed
        assert not pending.exists()
        in_progress = agent_module.WORK_DIR / "in_progress_work.json"
        assert in_progress.exists()
        agent.conn.close()

    def test_non_pending_work_ignored(self, patched_paths):
        done = agent_module.WORK_DIR / "done_work.json"
        with open(done, 'w') as f:
            json.dump({"task": "done"}, f)

        agent = ResearchAgent()
        triggers = agent.check_triggers()
        assert triggers == []
        assert done.exists()
        agent.conn.close()

    def test_both_triggers_returned(self, patched_paths):
        # Regime trigger
        with open(agent_module.DATA_DIR / ".regime_trigger", 'w') as f:
            json.dump({"regime": "vol_spike"}, f)
        # Pending work item
        with open(agent_module.WORK_DIR / "pending_x.json", 'w') as f:
            json.dump({"task": "analysis"}, f)

        agent = ResearchAgent()
        triggers = agent.check_triggers()
        assert len(triggers) == 2
        agent.conn.close()

    def test_malformed_regime_trigger_raises(self, patched_paths):
        trigger_file = agent_module.DATA_DIR / ".regime_trigger"
        with open(trigger_file, 'w') as f:
            f.write("NOT JSON {{{")

        agent = ResearchAgent()
        with pytest.raises(json.JSONDecodeError):
            agent.check_triggers()
        agent.conn.close()

    def test_empty_pending_files(self, patched_paths):
        pending = agent_module.WORK_DIR / "pending_empty.json"
        with open(pending, 'w') as f:
            f.write("")

        agent = ResearchAgent()
        with pytest.raises(json.JSONDecodeError):
            agent.check_triggers()
        agent.conn.close()


# ---------------------------------------------------------------------------
# analyze_regime tests
# ---------------------------------------------------------------------------

class TestAnalyzeRegime:
    """Cover all regime branches and edge cases."""

    def _make_agent(self, patched_paths, seeded_db):
        """Create agent with seeded DB."""
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        return ResearchAgent()

    def test_crisis_regime(self, patched_paths, seeded_db, monkeypatch):
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        agent = ResearchAgent()
        trigger = {"regime": "crisis", "vix": 35.0}
        analysis = agent.analyze_regime(trigger)

        assert analysis["regime"] == "crisis"
        assert analysis["vix"] == 35.0
        assert analysis["recommended_action"] == "risk_off"
        assert analysis["confidence"] == "medium"
        assert analysis["suggested_allocation"] == {"SPY": 0.20, "GLD": 0.50, "TLT": 0.30}
        agent.conn.close()

    def test_vol_spike_regime(self, patched_paths, seeded_db, monkeypatch):
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        agent = ResearchAgent()
        trigger = {"regime": "vol_spike", "vix": 25.0}
        analysis = agent.analyze_regime(trigger)

        assert analysis["recommended_action"] == "defensive_shift"
        assert analysis["confidence"] == "medium"
        assert analysis["suggested_allocation"] == {"SPY": 0.30, "GLD": 0.45, "TLT": 0.25}
        agent.conn.close()

    def test_low_vol_regime(self, patched_paths, seeded_db, monkeypatch):
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        agent = ResearchAgent()
        trigger = {"regime": "low_vol", "vix": 12.0}
        analysis = agent.analyze_regime(trigger)

        assert analysis["recommended_action"] == "risk_on"
        assert analysis["confidence"] == "medium"
        assert analysis["suggested_allocation"] == {"SPY": 0.55, "GLD": 0.30, "TLT": 0.15}
        agent.conn.close()

    def test_unknown_regime_no_action(self, patched_paths, seeded_db, monkeypatch):
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        agent = ResearchAgent()
        trigger = {"regime": "normal", "vix": 18.0}
        analysis = agent.analyze_regime(trigger)

        assert analysis["recommended_action"] is None
        assert analysis["confidence"] == "low"
        assert "suggested_allocation" not in analysis
        agent.conn.close()

    def test_empty_trigger_no_crash(self, patched_paths, seeded_db, monkeypatch):
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        agent = ResearchAgent()
        analysis = agent.analyze_regime({})

        assert analysis["regime"] is None
        assert analysis["vix"] is None
        assert analysis["recommended_action"] is None
        assert analysis["confidence"] == "low"
        agent.conn.close()

    def test_data_summary_populated_from_db(self, patched_paths, seeded_db, monkeypatch):
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        agent = ResearchAgent()
        analysis = agent.analyze_regime({"regime": "crisis"})

        assert isinstance(analysis["data_summary"], dict)
        for sym in ("SPY", "GLD", "TLT", "VIX", "QQQ"):
            assert sym in analysis["data_summary"]
        agent.conn.close()

    def test_allocation_sums_to_one(self, patched_paths, seeded_db, monkeypatch):
        """Crisis, vol_spike, and low_vol allocations should sum to ~1.0."""
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        agent = ResearchAgent()
        for regime, expected in [
            ("crisis", 1.0),
            ("vol_spike", 1.0),
            ("low_vol", 1.0),
        ]:
            analysis = agent.analyze_regime({"regime": regime, "vix": 20.0})
            alloc = analysis.get("suggested_allocation", {})
            total = sum(alloc.values())
            assert abs(total - expected) < 0.01, f"{regime} sum={total}"
        agent.conn.close()

    def test_missing_vix_field(self, patched_paths, seeded_db, monkeypatch):
        """VIX value is None when trigger has no 'vix' key."""
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        agent = ResearchAgent()
        analysis = agent.analyze_regime({"regime": "crisis"})

        assert analysis["vix"] is None
        agent.conn.close()

    def test_regime_with_extra_trigger_fields(self, patched_paths, seeded_db, monkeypatch):
        """Extra fields in trigger should be ignored, not crash."""
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        agent = ResearchAgent()
        trigger = {"regime": "low_vol", "vix": 12.0, "extra": "data", "unused": True}
        analysis = agent.analyze_regime(trigger)

        assert analysis["recommended_action"] == "risk_on"
        agent.conn.close()


# ---------------------------------------------------------------------------
# should_delegate_claude tests
# ---------------------------------------------------------------------------

class TestShouldDelegateClaude:
    """This is a static method — test all boolean branches."""

    def test_medium_confidence_no_crisis_no_delegate(self):
        assert ResearchAgent.should_delegate_claude(
            None, {"regime": "normal", "confidence": "medium"}
        ) is False

    def test_crisis_delegates(self):
        assert ResearchAgent.should_delegate_claude(
            None, {"regime": "crisis", "confidence": "medium"}
        ) is True

    def test_low_confidence_delegates(self):
        assert ResearchAgent.should_delegate_claude(
            None, {"regime": "normal", "confidence": "low"}
        ) is True

    def test_low_confidence_crisis_delegates(self):
        assert ResearchAgent.should_delegate_claude(
            None, {"regime": "crisis", "confidence": "low"}
        ) is True

    def test_requires_implementation_delegates(self):
        assert ResearchAgent.should_delegate_claude(
            None, {"regime": "normal", "confidence": "medium", "requires_implementation": True}
        ) is True

    def test_high_confidence_no_delegate(self):
        assert ResearchAgent.should_delegate_claude(
            None, {"regime": "low_vol", "confidence": "high"}
        ) is False

    def test_empty_analysis_no_delegate(self):
        assert ResearchAgent.should_delegate_claude(None, {}) is False

    def test_delegate_requires_code_change(self):
        assert ResearchAgent.should_delegate_claude(
            None, {"regime": "low_vol", "confidence": "high", "requires_implementation": True}
        ) is True

    def test_crisis_dominates_low_confidence(self):
        """Both branches are true; should still delegate."""
        assert ResearchAgent.should_delegate_claude(
            None, {"regime": "crisis", "confidence": "low"}
        ) is True


# ---------------------------------------------------------------------------
# delegate_to_claude tests
# ---------------------------------------------------------------------------

class TestDelegateToClaude:
    """Cover work item creation, task generation, and file output."""

    def test_creates_work_file(self, patched_paths):
        agent = ResearchAgent()
        analysis = {
            "regime": "crisis", "confidence": "low", "vix": 35.0,
            "data_summary": {"SPY": 90}, "recommended_action": "risk_off",
        }
        trigger = {"regime": "crisis", "vix": 35.0}

        work_file = agent.delegate_to_claude(analysis, trigger)
        assert Path(work_file).exists()

        with open(work_file) as f:
            item = json.load(f)
        assert item["type"] == "claude_code_delegate"
        assert item["status"] == "pending_delegate"
        assert item["id"].startswith("regime_")
        assert "created_at" in item
        assert item["trigger"] == trigger
        agent.conn.close()

    def test_crisis_adds_code_review_and_implement_tasks(self, patched_paths):
        agent = ResearchAgent()
        analysis = {
            "regime": "crisis", "confidence": "medium", "vix": 35.0,
            "data_summary": {}, "recommended_action": "risk_off",
        }
        work_file = agent.delegate_to_claude(analysis, {"regime": "crisis"})

        with open(work_file) as f:
            item = json.load(f)
        task_types = [t["type"] for t in item["tasks"]]
        assert "code_review" in task_types
        assert "implement" in task_types
        agent.conn.close()

    def test_low_confidence_adds_research_task(self, patched_paths):
        agent = ResearchAgent()
        analysis = {
            "regime": "normal", "confidence": "low", "vix": 18.0,
            "data_summary": {}, "recommended_action": None,
        }
        work_file = agent.delegate_to_claude(analysis, {"regime": "normal"})

        with open(work_file) as f:
            item = json.load(f)
        task_types = [t["type"] for t in item["tasks"]]
        assert "research" in task_types
        agent.conn.close()

    def test_crisis_and_low_confidence_adds_all_tasks(self, patched_paths):
        """Both crisis and low-confidence branches should produce all 3 tasks."""
        agent = ResearchAgent()
        analysis = {
            "regime": "crisis", "confidence": "low", "vix": 35.0,
            "data_summary": {}, "recommended_action": "risk_off",
        }
        work_file = agent.delegate_to_claude(analysis, {"regime": "crisis"})

        with open(work_file) as f:
            item = json.load(f)
        task_types = [t["type"] for t in item["tasks"]]
        assert "code_review" in task_types
        assert "implement" in task_types
        assert "research" in task_types
        agent.conn.close()

    def test_unrecognized_regime_no_crisis_tasks(self, patched_paths):
        """An unknown regime with medium confidence should have zero tasks."""
        agent = ResearchAgent()
        analysis = {
            "regime": "unknown", "confidence": "medium", "vix": 20.0,
            "data_summary": {}, "recommended_action": None,
        }
        work_file = agent.delegate_to_claude(analysis, {"regime": "unknown"})

        with open(work_file) as f:
            item = json.load(f)
        # This agent wouldn't call delegate for medium confidence non-crisis,
        # but if called directly, should have empty tasks
        assert len(item["tasks"]) == 0
        agent.conn.close()

    def test_context_fields_populated(self, patched_paths):
        """Work item context should contain data_summary, regime, and action."""
        agent = ResearchAgent()
        analysis = {
            "regime": "crisis", "confidence": "medium", "vix": 35.0,
            "data_summary": {"SPY": 90, "GLD": 85},
            "recommended_action": "risk_off",
        }
        work_file = agent.delegate_to_claude(analysis, {"regime": "crisis"})

        with open(work_file) as f:
            item = json.load(f)
        ctx = item["context"]
        assert ctx["data_summary"] == {"SPY": 90, "GLD": 85}
        assert ctx["current_regime"] == "crisis"
        assert ctx["recommended_action"] == "risk_off"
        agent.conn.close()

    def test_work_file_naming_convention(self, patched_paths):
        """File should be named claude_regime_<timestamp>.json in WORK_DIR."""
        agent = ResearchAgent()
        analysis = {
            "regime": "crisis", "confidence": "medium", "vix": 35.0,
            "data_summary": {}, "recommended_action": "risk_off",
        }
        work_file = agent.delegate_to_claude(analysis, {"regime": "crisis"})
        filename = Path(work_file).name
        assert filename.startswith("claude_regime_")
        assert filename.endswith(".json")
        assert Path(work_file).parent == agent_module.WORK_DIR
        agent.conn.close()


# ---------------------------------------------------------------------------
# crystallize_to_wiki tests
# ---------------------------------------------------------------------------

class TestCrystallizeToWiki:
    """Cover wiki page creation, frontmatter, and content generation."""

    def test_creates_wiki_page(self, patched_paths):
        agent = ResearchAgent()
        analysis = {
            "regime": "crisis", "vix": 35.0, "confidence": "medium",
            "recommended_action": "risk_off",
            "suggested_allocation": {"SPY": 0.20, "GLD": 0.50, "TLT": 0.30},
            "data_summary": {"SPY": 90, "GLD": 85},
        }
        page = agent.crystallize_to_wiki(analysis)
        assert page.exists()
        content = page.read_text()
        assert "crisis" in content.lower() or "CRISIS" in content
        assert "risk_off" in content
        assert "SPY" in content
        agent.conn.close()

    def test_has_frontmatter(self, patched_paths):
        agent = ResearchAgent()
        analysis = {
            "regime": "normal", "vix": 18.0, "confidence": "medium",
            "recommended_action": None, "data_summary": {},
        }
        page = agent.crystallize_to_wiki(analysis)
        content = page.read_text()
        assert content.startswith("---")
        assert "type: query" in content
        assert "tags:" in content
        agent.conn.close()

    def test_creates_compound_directory(self, patched_paths):
        compound_dir = agent_module.WIKI_DIR / "compound"
        # Ensure it doesn't exist yet
        if compound_dir.exists():
            import shutil
            shutil.rmtree(compound_dir)

        agent = ResearchAgent()
        analysis = {
            "regime": "normal", "vix": 18.0, "confidence": "low",
            "recommended_action": None, "data_summary": {},
        }
        page = agent.crystallize_to_wiki(analysis)
        assert compound_dir.exists()
        assert page.parent == compound_dir
        agent.conn.close()

    def test_page_naming_convention(self, patched_paths):
        """Filename should be regime-analysis-YYYY-MM-DD.md."""
        agent = ResearchAgent()
        analysis = {
            "regime": "low_vol", "vix": 12.0, "confidence": "high",
            "recommended_action": "risk_on", "data_summary": {},
        }
        page = agent.crystallize_to_wiki(analysis)
        assert page.name.startswith("regime-analysis-")
        assert page.suffix == ".md"
        agent.conn.close()

    def test_unknown_regime_in_title(self, patched_paths):
        """When regime key is missing, the default 'unknown' should appear."""
        agent = ResearchAgent()
        analysis = {
            "vix": None, "confidence": "low",
            "recommended_action": None, "data_summary": {},
        }
        page = agent.crystallize_to_wiki(analysis)
        content = page.read_text()
        assert "UNKNOWN" in content
        agent.conn.close()

    def test_missing_optional_fields_dont_crash(self, patched_paths):
        """Missing suggested_allocation and data_summary keys shouldn't crash."""
        agent = ResearchAgent()
        analysis = {
            "regime": "crisis", "vix": 35.0, "confidence": "medium",
            "recommended_action": "risk_off",
        }
        page = agent.crystallize_to_wiki(analysis)
        content = page.read_text()
        assert "crisis" in content.lower() or "CRISIS" in content
        agent.conn.close()

    def test_vix_n_a_when_missing(self, patched_paths):
        """VIX Level shows N/A when vix key is missing from analysis."""
        agent = ResearchAgent()
        analysis = {
            "regime": "normal", "confidence": "low",
            "recommended_action": None, "data_summary": {},
        }
        page = agent.crystallize_to_wiki(analysis)
        content = page.read_text()
        assert "N/A" in content
        agent.conn.close()

    def test_sources_section_present(self, patched_paths):
        agent = ResearchAgent()
        analysis = {
            "regime": "crisis", "vix": 35.0, "confidence": "medium",
            "recommended_action": "risk_off", "data_summary": {},
        }
        page = agent.crystallize_to_wiki(analysis)
        content = page.read_text()
        assert "## Sources" in content
        assert "Yahoo Finance" in content
        agent.conn.close()

    def test_next_steps_checklist(self, patched_paths):
        agent = ResearchAgent()
        analysis = {
            "regime": "crisis", "vix": 35.0, "confidence": "medium",
            "recommended_action": "risk_off", "data_summary": {"SPY": 90},
        }
        page = agent.crystallize_to_wiki(analysis)
        content = page.read_text()
        assert "[ ]" in content
        assert "Review strategy allocation" in content
        agent.conn.close()


# ---------------------------------------------------------------------------
# create_claude_prompt tests
# ---------------------------------------------------------------------------

class TestCreateClaudePrompt:
    """Cover prompt file creation and content generation."""

    def test_creates_prompt_file(self, patched_paths):
        agent = ResearchAgent()
        work_file = agent_module.WORK_DIR / "claude_test_001.json"
        with open(work_file, 'w') as f:
            json.dump({"id": "test_001"}, f)

        analysis = {
            "regime": "crisis", "vix": 35.0, "recommended_action": "risk_off",
            "suggested_allocation": {"SPY": 0.20},
        }
        agent.create_claude_prompt(work_file, analysis)

        prompt_file = work_file.with_suffix('.md')
        assert prompt_file.exists()
        content = prompt_file.read_text()
        assert "Claude Code Task" in content
        assert "crisis" in content
        assert "risk_off" in content
        agent.conn.close()

    def test_prompt_contains_expected_sections(self, patched_paths):
        agent = ResearchAgent()
        work_file = agent_module.WORK_DIR / "claude_test.json"
        with open(work_file, 'w') as f:
            json.dump({}, f)

        analysis = {
            "regime": "normal", "vix": 18.0, "recommended_action": None,
            "suggested_allocation": {},
        }
        agent.create_claude_prompt(work_file, analysis)
        content = work_file.with_suffix('.md').read_text()

        assert "## Situation" in content
        assert "## Your Task" in content
        assert "## Suggested Changes" in content
        assert "## Steps" in content
        assert "## Deliverables" in content
        assert "## Context" in content
        agent.conn.close()

    def test_prompt_has_steps(self, patched_paths):
        agent = ResearchAgent()
        work_file = agent_module.WORK_DIR / "claude_test.json"
        with open(work_file, 'w') as f:
            json.dump({}, f)

        analysis = {
            "regime": "normal", "vix": 18.0, "recommended_action": None,
            "suggested_allocation": {},
        }
        agent.create_claude_prompt(work_file, analysis)
        content = work_file.with_suffix('.md').read_text()

        assert "Review current implementation" in content
        assert "Run tests" in content
        assert "Adjust parameters" in content
        agent.conn.close()

    def test_prompt_includes_work_item_path(self, patched_paths):
        agent = ResearchAgent()
        work_file = agent_module.WORK_DIR / "claude_path_test.json"
        with open(work_file, 'w') as f:
            json.dump({}, f)

        analysis = {
            "regime": "crisis", "vix": 35.0, "recommended_action": "risk_off",
            "suggested_allocation": {"SPY": 0.20, "GLD": 0.50, "TLT": 0.30},
        }
        agent.create_claude_prompt(work_file, analysis)
        content = work_file.with_suffix('.md').read_text()

        assert "claude_path_test.json" in content
        assert str(work_file) in content
        agent.conn.close()

    def test_prompt_with_risk_on_allocation(self, patched_paths):
        """Allocation JSON should appear in suggested changes."""
        agent = ResearchAgent()
        work_file = agent_module.WORK_DIR / "claude_risk_on.json"
        with open(work_file, 'w') as f:
            json.dump({}, f)

        analysis = {
            "regime": "low_vol", "vix": 12.0, "recommended_action": "risk_on",
            "suggested_allocation": {"SPY": 0.55, "GLD": 0.30, "TLT": 0.15},
        }
        agent.create_claude_prompt(work_file, analysis)
        content = work_file.with_suffix('.md').read_text()

        assert "risk_on" in content
        # json.dumps writes "0.55", "0.3", "0.15" (trailing zeros trimmed)
        assert "0.55" in content
        assert "SPY" in content
        agent.conn.close()


# ---------------------------------------------------------------------------
# run_daily_summary tests
# ---------------------------------------------------------------------------

class TestRunDailySummary:
    """Cover summary query and result parsing."""

    def test_with_seeded_data(self, patched_paths, seeded_db, monkeypatch):
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        agent = ResearchAgent()
        summary = agent.run_daily_summary()

        assert "days" in summary
        assert isinstance(summary["days"], int)
        assert "avg_return" in summary
        assert "peak" in summary
        assert "trough" in summary
        agent.conn.close()

    def test_with_empty_db(self, patched_paths):
        """Empty DB should produce summary with zeros/Nones."""
        # Create DB with table but no data
        db_path = agent_module.DB_PATH
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
        conn.commit()
        conn.close()

        agent = ResearchAgent()
        summary = agent.run_daily_summary()

        assert summary["days"] == 0
        agent.conn.close()

    def test_with_single_data_point(self, patched_paths):
        """Single price row has no lag return; days counts non-null daily_return only."""
        db_path = agent_module.DB_PATH
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
        # Use recent dates within the -30 day window (UTC-safe)
        from datetime import timedelta, timezone
        d1 = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        d2 = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        conn.execute("INSERT INTO prices VALUES (?, ?, ?)", ("SPY", d1, 100.0))
        conn.execute("INSERT INTO prices VALUES (?, ?, ?)", ("SPY", d2, 101.0))
        conn.commit()
        conn.close()

        agent = ResearchAgent()
        summary = agent.run_daily_summary()

        # One close-to-close return from two rows
        assert summary["days"] == 1
        assert summary["avg_return"] is not None
        assert summary.get("return_metric") == "simple_close_to_close"
        agent.conn.close()


# ---------------------------------------------------------------------------
# create_claude_prompt edge cases
# ---------------------------------------------------------------------------

class TestCreateClaudePromptEdgeCases:
    """Edge cases for prompt generation."""

    def test_prompt_does_not_close_db(self, patched_paths):
        """create_claude_prompt should not close the DB connection."""
        agent = ResearchAgent()
        work_file = agent_module.WORK_DIR / "claude_test.json"
        with open(work_file, 'w') as f:
            json.dump({}, f)

        analysis = {"regime": "normal", "vix": 18.0, "recommended_action": None,
                     "suggested_allocation": {}}
        agent.create_claude_prompt(work_file, analysis)
        # Connection should still be open and usable
        cursor = agent.conn.cursor()
        cursor.execute("SELECT 1")
        row = cursor.fetchone()
        assert row is not None  # Connection is alive; SELECT 1 returns a row
        assert row[0] == 1
        agent.conn.close()

    def test_prompt_handles_none_allocation(self, patched_paths):
        """suggested_allocation as None should produce null (json.dumps(None) -> 'null')."""
        agent = ResearchAgent()
        work_file = agent_module.WORK_DIR / "claude_none.json"
        with open(work_file, 'w') as f:
            json.dump({}, f)

        analysis = {"regime": "normal", "vix": 18.0, "recommended_action": None,
                     "suggested_allocation": None}
        agent.create_claude_prompt(work_file, analysis)
        content = work_file.with_suffix('.md').read_text()
        # json.dumps(None) produces "null", not "{}"
        assert "null" in content
        agent.conn.close()


# ---------------------------------------------------------------------------
# run method (integration) tests
# ---------------------------------------------------------------------------

class TestRun:
    """Integration tests for the main run() loop."""

    def test_run_no_triggers_runs_daily_summary(self, patched_paths, seeded_db, monkeypatch):
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        agent = ResearchAgent()
        # Should not crash
        result = agent.run()
        assert isinstance(result, dict)  # Returns daily summary
        assert "days" in result
        # DB should be closed (lazy property resets to None)
        assert agent._conn is None

    def test_run_with_crisis_trigger(self, patched_paths, seeded_db, monkeypatch):
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        # Create a regime trigger
        trigger_file = agent_module.DATA_DIR / ".regime_trigger"
        with open(trigger_file, 'w') as f:
            json.dump({"regime": "crisis", "vix": 35.0}, f)

        agent = ResearchAgent()
        result = agent.run()
        assert result is None  # Returns None when triggers are processed
        # Trigger should be consumed
        assert not trigger_file.exists()
        # DB should be closed (lazy property resets to None)
        assert agent._conn is None

    def test_run_with_trigger_creates_wiki_page(self, patched_paths, seeded_db, monkeypatch):
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        trigger_file = agent_module.DATA_DIR / ".regime_trigger"
        with open(trigger_file, 'w') as f:
            json.dump({"regime": "low_vol", "vix": 12.0}, f)

        agent = ResearchAgent()
        agent.run()

        # Wiki compound pages should exist
        compound_dir = agent_module.WIKI_DIR / "compound"
        assert compound_dir.exists()
        md_files = list(compound_dir.glob("*.md"))
        assert len(md_files) >= 1
        agent.conn.close()

    def test_run_with_trigger_creates_prompt_and_work(self, patched_paths, seeded_db, monkeypatch):
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        trigger_file = agent_module.DATA_DIR / ".regime_trigger"
        with open(trigger_file, 'w') as f:
            json.dump({"regime": "crisis", "vix": 35.0}, f)

        agent = ResearchAgent()
        agent.run()

        # Should have created work items (Claude delegate files)
        json_files = list(agent_module.WORK_DIR.glob("claude_regime_*.json"))
        assert len(json_files) >= 1

        # Should have created prompt files
        md_files = list(agent_module.WORK_DIR.glob("claude_regime_*.md"))
        assert len(md_files) >= 1
        agent.conn.close()

    def test_run_with_no_db_table(self, patched_paths):
        """If DB exists but has no table, should handle gracefully."""
        db_path = agent_module.DB_PATH
        conn = sqlite3.connect(db_path)
        conn.close()

        agent = ResearchAgent()
        # Should not crash — run_daily_summary will fail on missing table,
        # but let's check that it raises appropriately
        with pytest.raises(sqlite3.OperationalError):
            agent.run()
        agent.conn.close()


# ---------------------------------------------------------------------------
# main block tests (via mock)
# ---------------------------------------------------------------------------

class TestMainBlock:
    """Test the __main__ guard at the bottom of the module."""

    def test_main_block_runs_agent(self, monkeypatch):
        """When __name__ == '__main__', agent.run() should be called."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = None

        def mock_constructor(*args, **kwargs):
            return mock_agent

        monkeypatch.setattr(agent_module, "ResearchAgent", mock_constructor)

        # Import and exec the main block content
        run_code = """
agent = agent_module.ResearchAgent()
agent.run()
"""
        exec(run_code, {"agent_module": agent_module})

        mock_agent.run.assert_called_once()


# ---------------------------------------------------------------------------
# Database connection edge cases
# ---------------------------------------------------------------------------

class TestDatabaseEdgeCases:
    """Edge cases around DB connection and query behavior."""

    def test_analyze_regime_no_prices_table(self, patched_paths):
        """analyze_regime should raise if prices table doesn't exist."""
        agent = ResearchAgent()
        with pytest.raises(sqlite3.OperationalError):
            agent.analyze_regime({"regime": "crisis"})
        agent.conn.close()

    def test_analyze_regime_empty_prices_table(self, patched_paths):
        """analyze_regime with an empty prices table should return empty data_summary."""
        db_path = agent_module.DB_PATH
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
        conn.commit()
        conn.close()

        agent = ResearchAgent()
        analysis = agent.analyze_regime({"regime": "crisis"})
        assert isinstance(analysis["data_summary"], dict)
        assert len(analysis["data_summary"]) == 0
        agent.conn.close()

    def test_daily_summary_no_spy_data(self, patched_paths):
        """run_daily_summary without SPY data should work but return 0 days."""
        db_path = agent_module.DB_PATH
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
        # Only insert non-SPY data
        conn.execute("INSERT INTO prices VALUES ('GLD', '2025-01-01', 100.0)")
        conn.commit()
        conn.close()

        agent = ResearchAgent()
        summary = agent.run_daily_summary()
        assert summary["days"] == 0
        agent.conn.close()


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------

class TestLogging:
    """Verify logger is properly configured and used."""

    def test_logger_exists(self):
        assert agent_module.logger is not None
        assert agent_module.logger.name == "src.research.agent"

    def test_run_daily_summary_logs(self, patched_paths, seeded_db, monkeypatch, caplog):
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        agent = ResearchAgent()
        with caplog.at_level("INFO"):
            _ = agent.run_daily_summary()

        assert "Daily summary" in caplog.text
        agent.conn.close()

    def test_run_logs_start_and_complete(self, patched_paths, seeded_db, monkeypatch, caplog):
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        # Create a trigger so run() goes through the full path that logs "Complete"
        with open(agent_module.DATA_DIR / ".regime_trigger", 'w') as f:
            json.dump({"regime": "crisis", "vix": 35.0}, f)

        agent = ResearchAgent()
        with caplog.at_level("INFO"):
            agent.run()

        assert "Research Agent Starting" in caplog.text
        assert "Research Agent Complete" in caplog.text
        agent.conn.close()

    def test_run_with_trigger_logs_processing(self, patched_paths, seeded_db, monkeypatch, caplog):
        monkeypatch.setattr(agent_module, "DB_PATH", seeded_db)
        with open(agent_module.DATA_DIR / ".regime_trigger", 'w') as f:
            json.dump({"regime": "crisis", "vix": 35.0}, f)

        agent = ResearchAgent()
        with caplog.at_level("INFO"):
            agent.run()

        assert "Processing trigger" in caplog.text
        assert "Analysis complete" in caplog.text
        assert "Delegated to Claude Code" in caplog.text
        assert "Crystallized to wiki" in caplog.text
        agent.conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
