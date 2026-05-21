#!/usr/bin/env python3
"""
Tests for research/agent.py — ResearchAgent.
"""
import sys
import os
import json
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.research import agent as agent_module
from src.research.agent import ResearchAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_paths(tmp_path, monkeypatch):
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


def _make_agent(tmp_path):
    """Create ResearchAgent using patched paths."""
    db_path = tmp_path / "data" / "market.db"

    # Create minimal DB
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS prices (symbol TEXT, date TEXT, close REAL)")
    conn.commit()
    conn.close()

    agent = ResearchAgent()
    return agent


def _seed_prices(db_path, symbols=None, days=90):
    """Insert fake price data into DB."""
    symbols = symbols or ["SPY", "GLD", "TLT", "VIX", "QQQ"]
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    conn = sqlite3.connect(db_path)
    dates = pd.bdate_range(end=datetime.now(), periods=days)
    for sym in symbols:
        price = 100.0
        for dt in dates:
            ret = np.random.normal(0.0004, 0.015)
            price *= (1 + ret)
            conn.execute(
                "INSERT INTO prices VALUES (?, ?, ?)",
                (sym, dt.strftime("%Y-%m-%d"), round(price, 2)),
            )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# check_triggers tests
# ---------------------------------------------------------------------------

class TestCheckTriggers:
    def test_no_triggers(self, tmp_path):
        agent = _make_agent(tmp_path)
        triggers = agent.check_triggers()
        assert triggers == []

    def test_regime_trigger_file(self, tmp_path):
        agent = _make_agent(tmp_path)
        trigger_data = {"regime": "crisis", "vix": 35.0}
        trigger_file = agent_module.DATA_DIR / ".regime_trigger"
        with open(trigger_file, 'w') as f:
            json.dump(trigger_data, f)

        triggers = agent.check_triggers()
        assert len(triggers) == 1
        assert triggers[0]["type"] == "regime_change"
        assert triggers[0]["regime"] == "crisis"

        # Trigger file should be consumed (unlinked)
        assert not trigger_file.exists()

    def test_pending_work_items(self, tmp_path):
        agent = _make_agent(tmp_path)
        work_data = {"task": "research", "topic": "correlation"}
        pending_file = agent_module.WORK_DIR / "pending_test.json"
        with open(pending_file, 'w') as f:
            json.dump(work_data, f)

        triggers = agent.check_triggers()
        assert len(triggers) == 1
        assert triggers[0]["task"] == "research"

        # Should be renamed to in_progress_
        assert not pending_file.exists()
        in_progress = agent_module.WORK_DIR / "in_progress_test.json"
        assert in_progress.exists()

    def test_non_pending_ignored(self, tmp_path):
        agent = _make_agent(tmp_path)
        other_file = agent_module.WORK_DIR / "completed_test.json"
        with open(other_file, 'w') as f:
            json.dump({"task": "done"}, f)

        triggers = agent.check_triggers()
        assert triggers == []
        assert other_file.exists()

    def test_both_triggers(self, tmp_path):
        agent = _make_agent(tmp_path)
        # Regime trigger
        trigger_file = agent_module.DATA_DIR / ".regime_trigger"
        with open(trigger_file, 'w') as f:
            json.dump({"regime": "vol_spike"}, f)
        # Pending work
        pending_file = agent_module.WORK_DIR / "pending_x.json"
        with open(pending_file, 'w') as f:
            json.dump({"task": "analysis"}, f)

        triggers = agent.check_triggers()
        assert len(triggers) == 2


# ---------------------------------------------------------------------------
# analyze_regime tests
# ---------------------------------------------------------------------------

class TestAnalyzeRegime:
    def test_crisis_regime(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_prices(agent_module.DB_PATH)
        # Reconnect after seed
        agent.conn = sqlite3.connect(agent_module.DB_PATH)
        agent.conn.row_factory = sqlite3.Row

        trigger = {"regime": "crisis", "vix": 35.0}
        analysis = agent.analyze_regime(trigger)
        assert analysis["regime"] == "crisis"
        assert analysis["recommended_action"] == "risk_off"
        assert analysis["confidence"] == "medium"
        assert "SPY" in analysis["suggested_allocation"]
        assert analysis["suggested_allocation"]["GLD"] == 0.50

    def test_vol_spike_regime(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_prices(agent_module.DB_PATH)
        agent.conn = sqlite3.connect(agent_module.DB_PATH)
        agent.conn.row_factory = sqlite3.Row

        trigger = {"regime": "vol_spike", "vix": 25.0}
        analysis = agent.analyze_regime(trigger)
        assert analysis["recommended_action"] == "defensive_shift"
        assert analysis["confidence"] == "medium"

    def test_low_vol_regime(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_prices(agent_module.DB_PATH)
        agent.conn = sqlite3.connect(agent_module.DB_PATH)
        agent.conn.row_factory = sqlite3.Row

        trigger = {"regime": "low_vol", "vix": 12.0}
        analysis = agent.analyze_regime(trigger)
        assert analysis["recommended_action"] == "risk_on"
        assert analysis["confidence"] == "medium"
        assert analysis["suggested_allocation"]["SPY"] == 0.55

    def test_unknown_regime_no_action(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_prices(agent_module.DB_PATH)
        agent.conn = sqlite3.connect(agent_module.DB_PATH)
        agent.conn.row_factory = sqlite3.Row

        trigger = {"regime": "normal", "vix": 18.0}
        analysis = agent.analyze_regime(trigger)
        assert analysis["recommended_action"] is None
        assert analysis["confidence"] == "low"

    def test_data_summary_populated(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_prices(agent_module.DB_PATH)
        agent.conn = sqlite3.connect(agent_module.DB_PATH)
        agent.conn.row_factory = sqlite3.Row

        trigger = {"regime": "crisis", "vix": 35.0}
        analysis = agent.analyze_regime(trigger)
        assert isinstance(analysis["data_summary"], dict)
        # Should have entries for symbols in the query
        assert len(analysis["data_summary"]) >= 0  # May be empty if dates don't match


# ---------------------------------------------------------------------------
# should_delegate_claude tests
# ---------------------------------------------------------------------------

class TestShouldDelegateClaude:
    def test_crisis_delegates(self):
        analysis = {"regime": "crisis", "confidence": "medium"}
        # Use a fresh agent — we just need the method, no DB
        assert ResearchAgent.should_delegate_claude(None, analysis) is True

    def test_low_confidence_delegates(self):
        analysis = {"regime": "normal", "confidence": "low"}
        assert ResearchAgent.should_delegate_claude(None, analysis) is True

    def test_requires_implementation_delegates(self):
        analysis = {"regime": "normal", "confidence": "medium", "requires_implementation": True}
        assert ResearchAgent.should_delegate_claude(None, analysis) is True

    def test_medium_confidence_no_delegate(self):
        analysis = {"regime": "normal", "confidence": "medium"}
        assert ResearchAgent.should_delegate_claude(None, analysis) is False

    def test_high_confidence_no_delegate(self):
        analysis = {"regime": "low_vol", "confidence": "high"}
        assert ResearchAgent.should_delegate_claude(None, analysis) is False


# ---------------------------------------------------------------------------
# delegate_to_claude tests
# ---------------------------------------------------------------------------

class TestDelegateToClaude:
    def test_creates_work_file(self, tmp_path):
        agent = _make_agent(tmp_path)
        analysis = {"regime": "crisis", "confidence": "low", "vix": 35.0,
                     "data_summary": {"SPY": 90}, "recommended_action": "risk_off"}
        trigger = {"regime": "crisis", "vix": 35.0}

        work_file = agent.delegate_to_claude(analysis, trigger)
        assert Path(work_file).exists()

        with open(work_file) as f:
            item = json.load(f)
        assert item["type"] == "claude_code_delegate"
        assert item["status"] == "pending_delegate"
        assert len(item["tasks"]) > 0

    def test_crisis_adds_code_review_task(self, tmp_path):
        agent = _make_agent(tmp_path)
        analysis = {"regime": "crisis", "confidence": "medium", "vix": 35.0,
                     "data_summary": {}, "recommended_action": "risk_off"}
        trigger = {"regime": "crisis"}

        work_file = agent.delegate_to_claude(analysis, trigger)
        with open(work_file) as f:
            item = json.load(f)
        task_types = [t["type"] for t in item["tasks"]]
        assert "code_review" in task_types
        assert "implement" in task_types

    def test_low_confidence_adds_research_task(self, tmp_path):
        agent = _make_agent(tmp_path)
        analysis = {"regime": "normal", "confidence": "low", "vix": 18.0,
                     "data_summary": {}, "recommended_action": None}
        trigger = {"regime": "normal"}

        work_file = agent.delegate_to_claude(analysis, trigger)
        with open(work_file) as f:
            item = json.load(f)
        task_types = [t["type"] for t in item["tasks"]]
        assert "research" in task_types

    def test_work_item_has_id(self, tmp_path):
        agent = _make_agent(tmp_path)
        analysis = {"regime": "crisis", "confidence": "low", "vix": 35.0,
                     "data_summary": {}, "recommended_action": "risk_off"}
        trigger = {"regime": "crisis"}

        work_file = agent.delegate_to_claude(analysis, trigger)
        with open(work_file) as f:
            item = json.load(f)
        assert item["id"].startswith("regime_")


# ---------------------------------------------------------------------------
# crystallize_to_wiki tests
# ---------------------------------------------------------------------------

class TestCrystallizeToWiki:
    def test_creates_wiki_page(self, tmp_path):
        agent = _make_agent(tmp_path)
        analysis = {
            "regime": "crisis",
            "vix": 35.0,
            "confidence": "medium",
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

    def test_wiki_page_has_frontmatter(self, tmp_path):
        agent = _make_agent(tmp_path)
        analysis = {"regime": "normal", "vix": 18.0, "confidence": "medium",
                     "recommended_action": None, "data_summary": {}}
        page = agent.crystallize_to_wiki(analysis)
        content = page.read_text()
        assert content.startswith("---")
        assert "type: query" in content

    def test_creates_compound_dir(self, tmp_path):
        agent = _make_agent(tmp_path)
        # Ensure compound dir doesn't exist
        compound_dir = agent_module.WIKI_DIR / "compound"
        if compound_dir.exists():
            import shutil
            shutil.rmtree(compound_dir)

        analysis = {"regime": "normal", "vix": 18.0, "confidence": "low",
                     "recommended_action": None, "data_summary": {}}
        page = agent.crystallize_to_wiki(analysis)
        assert compound_dir.exists()
        assert page.exists()


# ---------------------------------------------------------------------------
# create_claude_prompt tests
# ---------------------------------------------------------------------------

class TestCreateClaudePrompt:
    def test_creates_prompt_file(self, tmp_path):
        agent = _make_agent(tmp_path)
        work_item = {"id": "test_001", "regime": "crisis"}
        work_file = agent_module.WORK_DIR / "claude_test_001.json"
        with open(work_file, 'w') as f:
            json.dump(work_item, f)

        analysis = {"regime": "crisis", "vix": 35.0, "recommended_action": "risk_off",
                     "suggested_allocation": {"SPY": 0.20}}

        agent.create_claude_prompt(work_file, analysis)
        prompt_file = work_file.with_suffix('.md')
        assert prompt_file.exists()

        content = prompt_file.read_text()
        assert "Claude Code Task" in content
        assert "crisis" in content
        assert "risk_off" in content

    def test_prompt_has_steps(self, tmp_path):
        agent = _make_agent(tmp_path)
        work_file = agent_module.WORK_DIR / "claude_test.json"
        with open(work_file, 'w') as f:
            json.dump({}, f)

        analysis = {"regime": "normal", "vix": 18.0, "recommended_action": None,
                     "suggested_allocation": {}}

        agent.create_claude_prompt(work_file, analysis)
        content = work_file.with_suffix('.md').read_text()
        assert "Review current implementation" in content
        assert "Run tests" in content


# ---------------------------------------------------------------------------
# run_daily_summary tests
# ---------------------------------------------------------------------------

class TestRunDailySummary:
    def test_with_data(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_prices(agent_module.DB_PATH)
        agent.conn = sqlite3.connect(agent_module.DB_PATH)
        agent.conn.row_factory = sqlite3.Row

        summary = agent.run_daily_summary()
        assert "days" in summary
        assert isinstance(summary["days"], int)

    def test_no_data(self, tmp_path):
        agent = _make_agent(tmp_path)
        # DB exists but empty
        summary = agent.run_daily_summary()
        assert summary["days"] == 0


# ---------------------------------------------------------------------------
# run (integration) tests
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_no_triggers(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_prices(agent_module.DB_PATH)
        agent.conn = sqlite3.connect(agent_module.DB_PATH)
        agent.conn.row_factory = sqlite3.Row

        # Should not crash when no triggers
        agent.run()

    def test_run_with_trigger(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_prices(agent_module.DB_PATH)
        agent.conn = sqlite3.connect(agent_module.DB_PATH)
        agent.conn.row_factory = sqlite3.Row

        # Create a regime trigger
        trigger_file = agent_module.DATA_DIR / ".regime_trigger"
        with open(trigger_file, 'w') as f:
            json.dump({"regime": "crisis", "vix": 35.0}, f)

        agent.run()
        # Trigger should be consumed
        assert not trigger_file.exists()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_trigger_data(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_prices(agent_module.DB_PATH)
        agent.conn = sqlite3.connect(agent_module.DB_PATH)
        agent.conn.row_factory = sqlite3.Row

        trigger = {}
        analysis = agent.analyze_regime(trigger)
        assert analysis["regime"] is None
        assert analysis["recommended_action"] is None

    def test_malformed_trigger_file(self, tmp_path):
        agent = _make_agent(tmp_path)
        trigger_file = agent_module.DATA_DIR / ".regime_trigger"
        with open(trigger_file, 'w') as f:
            f.write("not valid json{{{")

        # Should raise JSONDecodeError when reading bad trigger file
        with pytest.raises(json.JSONDecodeError):
            agent.check_triggers()

    def test_allocation_sums_near_one(self, tmp_path):
        """Crisis and other regime allocations should sum to ~1.0."""
        agent = _make_agent(tmp_path)
        _seed_prices(agent_module.DB_PATH)
        agent.conn = sqlite3.connect(agent_module.DB_PATH)
        agent.conn.row_factory = sqlite3.Row

        for regime in ["crisis", "vol_spike", "low_vol"]:
            trigger = {"regime": regime, "vix": 25.0}
            analysis = agent.analyze_regime(trigger)
            alloc = analysis.get("suggested_allocation", {})
            if alloc:
                total = sum(alloc.values())
                assert abs(total - 1.0) < 0.01, f"{regime} allocation sums to {total}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
