#!/usr/bin/env python3
"""
Generator run tests — generator init, run(), and CLI classes
(TEST-GENERATOR-SPLIT s5, 2026-08-12).

Moved verbatim from tests/test_generator.py (TestGeneratorInit, TestRun,
TestRunEdgeCases, TestGeneratorInitEdgeCases, TestRunOverlay, TestCliMain) —
no tests renamed or weakened. Shared helpers live in tests/helpers.py (plain
module; the autouse fixture below is duplicated verbatim per split file —
never move it to conftest.py, it would pollute the full ~15k-test suite).
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.dashboard.generator import DashboardGenerator
from tests.helpers import _make_generator


@pytest.fixture(autouse=True)
def _isolate_live_ensemble_and_ic_health(request, monkeypatch):
    """Keep generator tests off live SignalHealthTracker.compute_ic / compute_vote.

    gen.run() and generate_health_json() otherwise call get_health_report() which
    runs hundreds of Spearman IC queries (~15–35s each on lab hosts). That was
    stalling make-test around the TestRun / health-json region (~44%).

    Opt out with @pytest.mark.allow_live_signal_health when a test intentionally
    exercises the real tracker (or already patches get_health_report itself).
    """
    if request.node.get_closest_marker("allow_live_signal_health"):
        yield
        return

    from src.strategy.ensemble_voter import EnsembleVote, Regime

    def _fake_vote(self, *args, **kwargs):
        return EnsembleVote(
            timestamp=datetime.now(timezone.utc).isoformat(),
            regime=Regime.NORMAL,
            regime_confidence=0.7,
            num_sources=1,
            weighted_consensus=0.1,
            agreement_ratio=0.5,
            equity_bias=0.1,
            duration_bias=0.0,
            gold_bias=0.0,
            action="neutral",
            confidence=0.5,
            reasoning="test-isolation",
            source_votes=[],
        )

    def _fake_bl_views(self, *args, **kwargs):
        from src.strategy.black_litterman_mapper import map_biases_to_views

        views = map_biases_to_views(
            0.1, 0.0, 0.0, health_scores=None, tau=0.15, prior="equal"
        )
        return {
            "views": views,
            "tau": 0.15,
            "prior": "equal",
            "health_scores_used": {},
            "equity_bias": 0.1,
            "duration_bias": 0.0,
            "gold_bias": 0.0,
        }

    def _fake_signal_health_section(**kwargs):
        return {
            "status": "ok",
            "sources": {},
            "summary": {"healthy": 0, "warning": 0, "critical": 0, "total": 0},
            "label_resolve": {"resolved": 0, "pending": 0, "skipped": True},
        }

    monkeypatch.setattr(
        "src.strategy.ensemble_voter.EnsembleVoter.compute_vote",
        _fake_vote,
        raising=False,
    )
    monkeypatch.setattr(
        "src.strategy.ensemble_voter.EnsembleVoter.get_bl_views",
        _fake_bl_views,
        raising=False,
    )
    monkeypatch.setattr(
        "src.dashboard.signal_health_section.build_signal_health_section",
        _fake_signal_health_section,
        raising=False,
    )
    monkeypatch.setattr(
        "src.dashboard.generator.build_signal_health_section",
        _fake_signal_health_section,
        raising=False,
    )
    yield

class TestGeneratorInit:
    """Test DashboardGenerator initialization."""

    def test_creates_with_db(self, tmp_path):
        """Generator connects to database."""
        gen, _ = _make_generator(tmp_path)
        assert gen.conn is not None
        gen.conn.close()

    def test_row_factory_set(self, tmp_path):
        """Row factory is set for dict-like access."""
        gen, _ = _make_generator(tmp_path)
        assert gen.conn.row_factory == sqlite3.Row
        gen.conn.close()

class TestRun:
    """Test run method."""

    def test_run_generates_all_files(self, tmp_path):
        """run() generates all dashboard files."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                gen.run()
        assert (tmp_path / "dashboard.json").exists()
        assert (tmp_path / "index.json").exists()
        # conn is closed by run()

class TestRunEdgeCases:
    """Test run() edge cases."""

    def test_run_closes_connection(self, tmp_path):
        """Connection is closed after run()."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                gen.run()
        # Connection should be None after close()
        assert gen.conn is None

    def test_run_creates_index_json(self, tmp_path):
        """run() creates index.json with files list."""
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                gen.run()
        with open(tmp_path / "index.json") as f:
            index = json.load(f)
        assert "files" in index
        assert len(index["files"]) >= 6  # At least 6 dashboard files
        assert "generated_at" in index
        # Connection is closed by run()

    def test_generated_at_populated_in_all_files(self, tmp_path):
        """All non-index JSON files have generated_at field."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps([{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    gen.run()
        json_files = ["dashboard.json", "stats.json",
                      "alerts.json", "health.json"]
        for name in json_files:
            fpath = tmp_path / name
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                assert "generated_at" in data, f"{name} missing generated_at"
        # signals.json uses "generated_at" consistent with other JSON outputs
        signals_path = tmp_path / "signals.json"
        if signals_path.exists():
            with open(signals_path) as f:
                signals_data = json.load(f)
            assert "generated_at" in signals_data
        # Connection already closed by run()

class TestGeneratorInitEdgeCases:
    """Additional DashboardGenerator initialization edge cases."""

    def test_public_dir_created(self, tmp_path):
        """PUBLIC_DIR is created during init."""
        new_public = tmp_path / "non_existent" / "data"
        assert not new_public.exists()
        # We can't easily test the constructor because it calls sqlite_connect
        # Instead verify that __init__ would create it
        gen, _ = _make_generator(tmp_path)
        with patch("src.dashboard.generator.PUBLIC_DIR", new_public):
            gen.__init__()
        assert new_public.exists()
        gen.conn.close()

class TestRunOverlay:
    """Test run() with overlay generation."""

    def test_run_with_overlay(self, tmp_path):
        """run() includes overlay path when overlay generates successfully."""
        gen, _ = _make_generator(tmp_path)
        yields_path = tmp_path / "yields.json"
        yields_path.write_text(json.dumps(
            [{"spread2s10s": 50, "dgs2": 4.0, "dgs10": 4.5} for _ in range(35)]
        ))
        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                with patch("src.dashboard.generator.YIELDS_JSON", yields_path):
                    gen.run()
        assert (tmp_path / "index.json").exists()
        # Verify signals.json was generated with regime data
        with open(tmp_path / "signals.json") as f:
            signals = json.load(f)
        assert "regime" in signals
        assert "generated_at" in signals
        # Connection already closed by run()

    def test_run_mirrors_required_public_data_contract_files_to_dist(self, tmp_path):
        """Dashboard generation keeps deploy-checked public/data and dist/data files in sync."""
        gen, _ = _make_generator(tmp_path)
        (tmp_path / "source_manifest.json").write_text(json.dumps({
            "schema_version": "market-data-source-manifest/v1",
            "generated_at": "2026-07-06T00:00:00+00:00",
            "artifacts": [],
        }))

        with patch("src.dashboard.generator.PUBLIC_DIR", tmp_path):
            with patch("src.dashboard.generator.DATA_DIR", tmp_path):
                gen.run()

        for filename in ("source_manifest.json", "index.json", "health.json"):
            public_file = tmp_path / filename
            dist_file = tmp_path.parent / "dist" / "data" / filename
            assert dist_file.exists(), f"{dist_file} missing"
            assert dist_file.read_bytes() == public_file.read_bytes()
        # Connection already closed by run()

class TestCliMain:
    """Test the __main__ CLI entry point."""

    def test_main_logic_runs_generator(self):
        """__main__ block's logic: DashboardGenerator().run()."""
        with patch.object(DashboardGenerator, "run") as mock_run:
            with patch.object(DashboardGenerator, "__init__", return_value=None):
                # This replicates the __main__ block: gen = DashboardGenerator(); gen.run()
                gen = DashboardGenerator()
                gen.run()
                mock_run.assert_called_once()

    def test_main_block_guard_reads_correctly(self):
        """The __main__ guard checks __name__ == '__main__'."""
        import ast
        import importlib.util
        source_path = importlib.util.find_spec("src.dashboard.generator").origin
        source = Path(source_path).read_text()
        tree = ast.parse(source)
        found_guard = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check if this is an if __name__ == "__main__" guard
                if (isinstance(node.test, ast.Compare)
                        and isinstance(node.test.left, ast.Name)
                        and node.test.left.id == "__name__"
                        and isinstance(node.test.comparators[0], ast.Constant)
                        and node.test.comparators[0].value == "__main__"):
                    found_guard = True
                    # Verify the body contains DashboardGenerator and run()
                    body_source = ast.unparse(node.body)
                    assert "DashboardGenerator" in body_source
                    assert ".run()" in body_source
                    break
        assert found_guard, "No __name__ == '__main__' guard found"

