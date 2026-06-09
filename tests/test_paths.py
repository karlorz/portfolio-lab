"""
Tests for centralized path resolution in src/paths.py.

Covers: all exported constants resolve to valid Path objects,
PROJECT_ROOT correctness, BASE_ALLOCATION structure.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from src import paths


class TestProjectRoot:
    def test_project_root_is_path(self):
        assert isinstance(paths.PROJECT_ROOT, Path)

    def test_project_root_exists(self):
        assert paths.PROJECT_ROOT.exists()

    def test_project_root_contains_src(self):
        assert (paths.PROJECT_ROOT / "src").is_dir()


class TestDataPaths:
    def test_data_dir_resolves(self):
        assert isinstance(paths.DATA_DIR, Path)
        assert paths.DATA_DIR.name == "data"

    def test_public_data_dir_resolves(self):
        assert isinstance(paths.PUBLIC_DATA_DIR, Path)
        assert paths.PUBLIC_DATA_DIR.name == "data"

    def test_market_db_path(self):
        assert paths.MARKET_DB.parent == paths.DATA_DIR
        assert paths.MARKET_DB.name == "market.db"


class TestDataFiles:
    def test_prices_json_path(self):
        assert paths.PRICES_JSON.name == "prices.json"

    def test_signals_json_path(self):
        assert paths.SIGNALS_JSON.name == "signals.json"

    def test_historical_json_path(self):
        assert paths.HISTORICAL_JSON.name == "historical.json"

    def test_yields_json_path(self):
        assert paths.YIELDS_JSON.name == "yields.json"


class TestSubdirectories:
    def test_signals_dir(self):
        assert isinstance(paths.SIGNALS_DIR, Path)
        assert paths.SIGNALS_DIR.name == "signals"

    def test_backtest_results_dir(self):
        assert paths.BACKTEST_RESULTS_DIR.name == "backtest_results"

    def test_factors_dir(self):
        assert paths.FACTORS_DIR.name == "factors"

    def test_cache_dir(self):
        assert paths.CACHE_DIR.name == "cache"

    def test_options_cache_dir(self):
        assert paths.OPTIONS_CACHE_DIR.name == "options"

    def test_llm_costs_dir(self):
        assert paths.LLM_COSTS_DIR.name == "llm_costs"

    def test_attribution_dir(self):
        assert paths.ATTRIBUTION_DIR.name == "attribution"


class TestBaseAllocation:
    def test_is_dict(self):
        assert isinstance(paths.BASE_ALLOCATION, dict)

    def test_has_required_keys(self):
        assert "SPY" in paths.BASE_ALLOCATION
        assert "GLD" in paths.BASE_ALLOCATION
        assert "TLT" in paths.BASE_ALLOCATION

    def test_sums_to_one(self):
        total = sum(paths.BASE_ALLOCATION.values())
        assert abs(total - 1.0) < 0.01

    def test_champion_weights(self):
        assert paths.BASE_ALLOCATION["SPY"] == 0.46
        assert paths.BASE_ALLOCATION["GLD"] == 0.38
        assert paths.BASE_ALLOCATION["TLT"] == 0.16


class TestHomeDirectories:
    def test_home_is_path(self):
        assert isinstance(paths.HOME, Path)

    def test_wiki_dir(self):
        assert paths.WIKI_DIR.name == "wiki"

    def test_work_dir(self):
        assert paths.WORK_DIR.name == "work"

    def test_project_wiki_dir(self):
        assert paths.PROJECT_WIKI_DIR == paths.WIKI_DIR / "projects" / "portfolio-lab"

    def test_work_dir_defaults_to_project_wiki_work_dir(self):
        assert paths.WORK_DIR == paths.PROJECT_WIKI_DIR / "work"


class TestSkillWikiVaultResolver:
    def test_import_without_configured_vault_still_exposes_non_wiki_paths(self, tmp_path):
        script = """
from src import paths
print(paths.DATA_DIR.name)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=paths.PROJECT_ROOT,
            env={
                "HOME": str(tmp_path),
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(paths.PROJECT_ROOT),
            },
            capture_output=True,
            text=True,
            timeout=5,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "data"

    def test_work_dir_override_allows_research_agent_import_without_vault(self, tmp_path):
        work_dir = tmp_path / "custom-work"
        script = """
from src.research.agent import ResearchAgent, WORK_DIR
ResearchAgent()
print(WORK_DIR.name)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=paths.PROJECT_ROOT,
            env={
                "HOME": str(tmp_path),
                "PATH": "/usr/bin:/bin",
                "PORTFOLIO_LAB_ENABLE_ML": "0",
                "PYTHONPATH": str(paths.PROJECT_ROOT),
                "WORK_DIR": str(work_dir),
            },
            capture_output=True,
            text=True,
            timeout=5,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "custom-work"
        assert work_dir.is_dir()

    def test_wiki_dir_env_override_wins(self, monkeypatch, tmp_path):
        override = tmp_path / "custom-wiki"
        monkeypatch.setenv("WIKI_DIR", str(override))

        assert paths.resolve_skillwiki_vault() == override

    def test_uses_skillwiki_path_when_env_absent(self, monkeypatch, tmp_path):
        resolved = tmp_path / "skillwiki-vault"
        monkeypatch.delenv("WIKI_DIR", raising=False)

        def fake_run(*args, **kwargs):
            assert args[0] == ["skillwiki", "path"]
            assert kwargs["timeout"] <= 2
            return SimpleNamespace(returncode=0, stdout=f"{resolved}\n", stderr="")

        monkeypatch.setattr(paths.subprocess, "run", fake_run)

        assert paths.resolve_skillwiki_vault() == resolved

    def test_validated_home_wiki_fallback_when_skillwiki_path_fails(self, monkeypatch, tmp_path):
        home_wiki = tmp_path / "wiki"
        (home_wiki / "projects").mkdir(parents=True)
        (home_wiki / "SCHEMA.md").write_text("# Vault Schema\n")
        monkeypatch.delenv("WIKI_DIR", raising=False)
        monkeypatch.setattr(paths, "HOME", tmp_path)

        def fail_run(*args, **kwargs):
            raise FileNotFoundError("skillwiki")

        monkeypatch.setattr(paths.subprocess, "run", fail_run)

        assert paths.resolve_skillwiki_vault() == home_wiki

    def test_rejects_unvalidated_home_wiki_fallback(self, monkeypatch, tmp_path):
        (tmp_path / "wiki").mkdir()
        monkeypatch.delenv("WIKI_DIR", raising=False)
        monkeypatch.setattr(paths, "HOME", tmp_path)

        def fail_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="skillwiki path", timeout=1)

        monkeypatch.setattr(paths.subprocess, "run", fail_run)

        with pytest.raises(RuntimeError, match="SkillWiki vault could not be resolved"):
            paths.resolve_skillwiki_vault()
