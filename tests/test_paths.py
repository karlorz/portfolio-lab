"""
Tests for centralized path resolution in src/paths.py.

Covers: all exported constants resolve to valid Path objects,
PROJECT_ROOT correctness, BASE_ALLOCATION structure.
"""

from pathlib import Path
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
